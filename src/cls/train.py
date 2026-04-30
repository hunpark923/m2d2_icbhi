"""
Training and final evaluation for the attention-MIL classifier.

The training procedure:

1. Build train/val cycles by patient-wise split.
2. Initialise the BEATs encoder from the SSL-pretrained checkpoint.
3. Train with focal loss + label smoothing + WeightedRandomSampler under
   gradient accumulation.
4. After every epoch, run TTA inference on the validation set, search the
   decision threshold, and update the best checkpoint by GS-then-PM.
5. After training, load the best checkpoint and evaluate on the test split
   using the *frozen* validation-tuned threshold.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.cls.config import ClsConfig
from src.cls.dataset import ICBHIClsDataset, cls_collate_fn
from src.cls.losses import focal_loss_with_smoothing
from src.cls.model import AttentionMILClassifier
from src.cls.threshold import tune_threshold_for_cls
from src.cls.tta import infer_probs_tta
from src.common.beats_loader import (
    create_beats_model,
    overwrite_beats_with_ssl_ckpt,
)
from src.common.icbhi_dataset import (
    load_or_build_cycles,
    split_train_val_by_patient,
)
from src.common.metrics import (
    apply_threshold_rule,
    icbhi_score_from_cm,
    patient_macro_icbhi,
)
from src.common.preprocessing import PreprocConfig
from src.common.seed import seed_everything


def _build_class_alpha(
    train_cycles: List[Dict],
    power: float,
    num_classes: int,
) -> torch.Tensor:
    """Compute per-class alpha weights for focal loss as
    ``alpha_c ~ 1 / count_c^power`` (normalised so that mean(alpha) == 1).
    """
    ys = [int(c["y"]) for c in train_cycles]
    counts = np.bincount(ys, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    alpha = 1.0 / np.power(counts, power)
    alpha = alpha / alpha.mean()
    return torch.tensor(alpha, dtype=torch.float32)


def _build_weighted_sampler(
    train_cycles: List[Dict],
    power: float,
    num_classes: int,
    generator: torch.Generator,
) -> WeightedRandomSampler:
    """Per-sample weights for WeightedRandomSampler, mirroring the alpha
    rule (weight ~= 1/count^power)."""
    ys = [int(c["y"]) for c in train_cycles]
    counts = np.bincount(ys, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    class_weight = 1.0 / np.power(counts, power)
    sample_weights = [float(class_weight[y]) for y in ys]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )


def _is_better(curr: Dict, best: Optional[Dict], metric: str) -> bool:
    """Lexicographic comparator over (metric, ...) for model selection."""
    if best is None:
        return True
    metric = metric.lower()
    if metric == "gs_then_pm":
        keys = ("global_score", "patient_macro", "specificity", "sensitivity")
    elif metric == "pm_then_gs":
        keys = ("patient_macro", "global_score", "specificity", "sensitivity")
    else:
        raise ValueError(f"Unknown selection_metric: {metric}")
    curr_t = tuple(float(curr[k]) for k in keys)
    best_t = tuple(float(best[k]) for k in keys)
    return curr_t > best_t


def train_cls(
    pre: PreprocConfig,
    ssl_ckpt: Optional[str],
    cfg: ClsConfig,
) -> Tuple[str, List[Dict]]:
    """Train the attention-MIL classifier and return the path to the best
    checkpoint together with the held-out test cycles.
    """
    generator = seed_everything(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    cfg.device = str(device)

    # ── Data ──────────────────────────────────────────────────────
    tr_cycles_all, te_cycles = load_or_build_cycles(pre)
    tr_cycles, va_cycles = split_train_val_by_patient(
        tr_cycles_all,
        val_ratio=cfg.val_ratio,
        seed=cfg.split_seed,
    )

    ds_tr = ICBHIClsDataset(tr_cycles, pre, is_train=True)
    ds_va = ICBHIClsDataset(va_cycles, pre, is_train=False)

    sampler = _build_weighted_sampler(
        tr_cycles, cfg.sampler_power, cfg.num_classes, generator
    )
    ld_tr = DataLoader(
        ds_tr,
        batch_size=cfg.batch_size,
        sampler=sampler,
        collate_fn=cls_collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        generator=generator,
    )

    # ── Model ─────────────────────────────────────────────────────
    encoder, dim, _ = create_beats_model(cfg.beats_checkpoint_path)
    if ssl_ckpt:
        overwrite_beats_with_ssl_ckpt(encoder, ssl_ckpt)
    model = AttentionMILClassifier(encoder, dim, cfg).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr
    )
    scaler = GradScaler(enabled=cfg.use_amp)

    alpha = _build_class_alpha(
        tr_cycles, cfg.sampler_power, cfg.num_classes
    ).to(device)

    # ── Output directory ──────────────────────────────────────────
    save_dir = os.path.join(
        cfg.save_root,
        f"{cfg.run_name}_seed{cfg.seed}_"
        f"{datetime.now().strftime('%m%d_%H%M')}",
    )
    os.makedirs(save_dir, exist_ok=True)
    best_ckpt_path = os.path.join(save_dir, "best_cls.pt")
    best_result: Optional[Dict] = None
    epochs_without_improvement = 0

    # ── Training loop ─────────────────────────────────────────────
    for ep in range(cfg.epochs):
        model.train()
        loss_sum = 0.0
        pbar = tqdm(ld_tr, desc=f"CLS ep {ep + 1}/{cfg.epochs}")
        optimizer.zero_grad(set_to_none=True)
        for step_idx, batch in enumerate(pbar):
            x = batch["instances"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            with autocast(enabled=cfg.use_amp):
                logits = model(x)
                loss = focal_loss_with_smoothing(
                    logits, y,
                    alpha=alpha,
                    gamma=cfg.focal_gamma,
                    label_smoothing=cfg.label_smoothing,
                ) / cfg.accumulation_steps

            scaler.scale(loss).backward()

            if (step_idx + 1) % cfg.accumulation_steps == 0:
                if cfg.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            loss_sum += loss.item() * cfg.accumulation_steps
            pbar.set_postfix(loss=f"{loss_sum / (step_idx + 1):.4f}")

        scheduler.step()

        # Validation: TTA inference + threshold tuning + model selection.
        probs_np, ys, patients, _ = infer_probs_tta(
            model=model,
            cycles=va_cycles,
            pre=pre,
            cfg=cfg,
            device=device,
            batch_size=cfg.batch_size,
            use_amp=cfg.use_amp,
        )
        result = tune_threshold_for_cls(
            probs_np=probs_np,
            ys=ys,
            patients=patients,
            threshold_steps=cfg.threshold_steps,
            threshold_min=cfg.threshold_min,
            threshold_max=cfg.threshold_max,
        )

        print(
            f"  Val: GS={result['global_score']:.4f}  "
            f"PM={result['patient_macro']:.4f}  "
            f"Sp={result['specificity']:.4f}  "
            f"Se={result['sensitivity']:.4f}  "
            f"th={result['threshold']:.3f}"
        )

        if _is_better(result, best_result, cfg.selection_metric):
            best_result = result
            torch.save(
                {
                    "model": model.state_dict(),
                    "threshold": float(result["threshold"]),
                    "val_global_score": float(result["global_score"]),
                    "val_patient_macro": float(result["patient_macro"]),
                    "epoch": ep + 1,
                    "config": cfg.__dict__,
                },
                best_ckpt_path,
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.patience:
                print(f"  Early stop at epoch {ep + 1}")
                break

    summary = {
        "best_val_global_score": float(best_result["global_score"]),
        "best_val_patient_macro": float(best_result["patient_macro"]),
        "best_threshold": float(best_result["threshold"]),
    }
    with open(os.path.join(save_dir, "best_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return best_ckpt_path, te_cycles


@torch.no_grad()
def evaluate_final(
    pre: PreprocConfig,
    cls_ckpt_path: str,
    te_cycles: List[Dict],
    cfg: ClsConfig,
) -> Dict:
    """Final evaluation on the held-out test split using the validation-tuned
    decision threshold.
    """
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(cls_ckpt_path, map_location="cpu")
    threshold = float(ckpt["threshold"])

    encoder, dim, _ = create_beats_model(cfg.beats_checkpoint_path)
    model = AttentionMILClassifier(encoder, dim, cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    probs_np, ys, patients, _ = infer_probs_tta(
        model=model,
        cycles=te_cycles,
        pre=pre,
        cfg=cfg,
        device=device,
        batch_size=cfg.batch_size,
        use_amp=cfg.use_amp,
    )
    preds = apply_threshold_rule(probs_np, threshold).tolist()

    metas = [{"patient": p, "y": int(y)} for p, y in zip(patients, ys)]
    pm, gs, cm = patient_macro_icbhi(metas, preds)
    _, se, sp = icbhi_score_from_cm(cm)

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS  (four-class strict sensitivity)")
    print("=" * 60)
    print(f"  ICBHI score      : {gs:.4f}")
    print(f"  Specificity      : {sp:.4f}")
    print(f"  Sensitivity (4c) : {se:.4f}")
    print(f"  Patient macro    : {pm:.4f}")
    print(f"  Threshold (frozen): {threshold:.3f}")
    print(f"\n  Confusion matrix:")
    print(cm)
    print("=" * 60)

    return {
        "global_score": float(gs),
        "specificity": float(sp),
        "sensitivity_4c": float(se),
        "patient_macro": float(pm),
        "threshold": float(threshold),
        "confusion_matrix": cm.tolist(),
    }
