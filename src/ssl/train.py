"""
SSL pretraining loop with periodic linear-probe monitoring and offline
deterministic re-rank for final checkpoint selection.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.cls.dataset import ICBHIClsDataset, cls_collate_fn
from src.common.beats_loader import (
    create_beats_model,
    overwrite_beats_with_ssl_ckpt,
)
from src.common.icbhi_dataset import (
    load_or_build_cycles,
    split_train_val_by_patient,
)
from src.common.metrics import patient_macro_icbhi
from src.common.preprocessing import PreprocConfig
from src.common.seed import (
    capture_rng_state,
    restore_rng_state,
)
from src.ssl.config import SSLConfig
from src.ssl.dataset import BackgroundNoisePool, SSLDataset
from src.ssl.m2d2 import M2D2Model


# ---------------------------------------------------------------------------
# Linear probe (used both for periodic monitoring and offline re-rank)
# ---------------------------------------------------------------------------
def linear_eval(
    ckpt_path: str,
    pre: PreprocConfig,
    cfg: SSLConfig,
    tr_cyc: List[Dict],
    va_cyc: List[Dict],
    lin_seed: int = 1042,
) -> float:
    """Train a single-layer linear classifier on frozen encoder features and
    return the patient-macro ICBHI score on the validation set.
    """
    import random

    random.seed(lin_seed)
    np.random.seed(lin_seed)
    torch.manual_seed(lin_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(lin_seed)
        torch.cuda.manual_seed_all(lin_seed)

    probe_gen = torch.Generator()
    probe_gen.manual_seed(lin_seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    enc, dim, _ = create_beats_model(cfg.beats_checkpoint_path)
    overwrite_beats_with_ssl_ckpt(enc, ckpt_path)
    enc = enc.to(device).eval()

    def _features(cycles):
        feats, ys, patients = [], [], []
        ds = ICBHIClsDataset(cycles, pre, is_train=False)
        ld = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=cls_collate_fn)
        with torch.no_grad():
            for b in ld:
                inst = b["instances"].to(device)
                bsz, n_inst, t_len = inst.shape
                f = enc.extract_features(inst.view(-1, t_len))[0]
                f = f.mean(dim=1).view(bsz, n_inst, -1).mean(dim=1)
                feats.append(f.cpu())
                ys.extend(b["y"].tolist())
                patients.extend(b["patient"])
        return torch.cat(feats), torch.tensor(ys), patients

    x_tr, y_tr, _ = _features(tr_cyc)
    x_va, y_va, p_va = _features(va_cyc)

    head = nn.Linear(dim, 4).to(device)
    opt = optim.Adam(head.parameters(), lr=cfg.lin_lr)
    ds_tr = TensorDataset(x_tr, y_tr)
    ld_tr = DataLoader(ds_tr, batch_size=256, shuffle=True, generator=probe_gen)

    for _ in range(cfg.lin_epochs):
        for bx, by in ld_tr:
            bx, by = bx.to(device), by.to(device)
            loss = nn.functional.cross_entropy(head(bx), by)
            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        preds = head(x_va.to(device)).cpu().argmax(dim=1).numpy()

    pm, _, _ = patient_macro_icbhi(
        [{"patient": p, "y": int(y)} for p, y in zip(p_va, y_va.tolist())],
        preds,
    )
    return pm


# ---------------------------------------------------------------------------
# Offline deterministic re-rank
# ---------------------------------------------------------------------------
def offline_rerank_checkpoints(
    save_dir: str,
    pre: PreprocConfig,
    cfg: SSLConfig,
    tr_cyc: List[Dict],
    va_cyc: List[Dict],
) -> Optional[str]:
    """Re-evaluate every saved encoder checkpoint with multiple deterministic
    linear-probe seeds and select the one with the highest mean score.

    The selected checkpoint is copied to ``best_encoder.pt`` for downstream
    classifier training.
    """
    candidates = sorted(
        os.path.join(save_dir, f) for f in os.listdir(save_dir)
        if f.startswith("encoder_e") and f.endswith(".pt")
    )
    if not candidates:
        print("No encoder_e*.pt files found for offline re-rank.")
        return None

    results = []
    best_avg, best_path = -1e9, None
    print("\n" + "=" * 70)
    print("OFFLINE DETERMINISTIC RE-RANK")
    print("=" * 70)

    for ckpt_path in candidates:
        seed_scores = []
        for lin_seed in cfg.rerank_lin_seeds:
            seed_scores.append(
                float(linear_eval(ckpt_path, pre, cfg, tr_cyc, va_cyc,
                                  lin_seed=int(lin_seed)))
            )
        avg, sd = float(np.mean(seed_scores)), float(np.std(seed_scores))
        results.append({
            "ckpt": os.path.basename(ckpt_path),
            "scores": seed_scores,
            "avg": avg,
            "std": sd,
        })
        print(f"  {os.path.basename(ckpt_path)}: scores={seed_scores}  "
              f"avg={avg:.4f}  std={sd:.4f}")
        if avg > best_avg:
            best_avg, best_path = avg, ckpt_path

    with open(os.path.join(save_dir, "offline_rerank_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    if best_path is None:
        return None

    final_path = os.path.join(save_dir, "best_encoder.pt")
    shutil.copy2(best_path, final_path)
    print(f"\nBest checkpoint: {os.path.basename(best_path)}  "
          f"(avg PM = {best_avg:.4f})")
    print(f"Saved as: {final_path}")
    return final_path


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train_ssl(pre: PreprocConfig, cfg: SSLConfig) -> Optional[str]:
    """Run the full M2D2 SSL pretraining and return the path to the
    deterministically-selected best encoder checkpoint.
    """
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    tr_cyc_all, _ = load_or_build_cycles(pre)
    tr_cyc, va_cyc = split_train_val_by_patient(
        tr_cyc_all, val_ratio=0.2, seed=cfg.seed
    )
    pool = BackgroundNoisePool(
        sample_rate=pre.sample_rate,
        noise_dir=cfg.bg_noise_dir,
        fallback_cycles=tr_cyc,
    )

    ds_tr = SSLDataset(tr_cyc, pre, cfg, pool, is_train=True)
    ds_va = SSLDataset(va_cyc, pre, cfg, pool, is_train=False)

    ld_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True,
                       num_workers=0)
    ld_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False,
                       num_workers=0)

    # Offline teacher: a frozen BEATs initialised from the public checkpoint.
    teacher, _, _ = create_beats_model(cfg.beats_checkpoint_path)
    model = M2D2Model(cfg, teacher).to(device)

    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.05)
    scaler = GradScaler()
    sched = optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=cfg.lr,
        total_steps=cfg.pretrain_epochs * len(ld_tr),
        pct_start=cfg.warmup_epochs / cfg.pretrain_epochs,
    )

    save_dir = os.path.join(
        cfg.save_root,
        f"{cfg.run_name}_{datetime.now().strftime('%m%d_%H%M')}",
    )
    os.makedirs(save_dir, exist_ok=True)
    best_score_online = -1e9

    for ep in range(cfg.pretrain_epochs):
        model.train()
        model.current_epoch = ep
        ep_t0 = time.time()
        loss_sum = 0.0

        pbar = tqdm(
            ld_tr,
            desc=f"SSL ep {ep + 1}/{cfg.pretrain_epochs} "
                 f"[t_max={model.get_current_tmax()}]",
        )
        for step_idx, (clean, mixed) in enumerate(pbar):
            clean, mixed = clean.to(device), mixed.to(device)
            with autocast():
                loss = model(clean, mixed) / cfg.accumulation_steps
            scaler.scale(loss).backward()

            if (ep * len(ld_tr) + step_idx + 1) % cfg.accumulation_steps == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                model.update_ema()
                sched.step()

            loss_sum += loss.item() * cfg.accumulation_steps
            pbar.set_postfix(
                loss=f"{loss_sum / (step_idx + 1):.4f}",
                lr=f"{opt.param_groups[0]['lr']:.2e}",
            )

        print(f"  Epoch {ep + 1} time: {time.time() - ep_t0:.1f}s")

        # Periodic checkpoint + linear-probe monitoring.
        if (ep + 1) % cfg.save_eval_every == 0 or (ep + 1) == cfg.pretrain_epochs:
            ckpt_path = os.path.join(save_dir, f"encoder_e{ep + 1:03d}.pt")
            torch.save({"model": model.online_encoder.state_dict()}, ckpt_path)

            # Isolate linear-probe RNG from the main training trajectory.
            rng_state = capture_rng_state()
            score = linear_eval(
                ckpt_path, pre, cfg, tr_cyc, va_cyc,
                lin_seed=int(cfg.seed) + 1000,
            )
            restore_rng_state(rng_state)
            print(f"  Online linear probe score (PM): {score:.4f}")

            if score > best_score_online:
                best_score_online = score
                torch.save(
                    {"model": model.online_encoder.state_dict(),
                     "score": score, "epoch": ep},
                    os.path.join(save_dir, "best_encoder_online.pt"),
                )

    final_best_path = offline_rerank_checkpoints(
        save_dir, pre, cfg, tr_cyc, va_cyc
    )
    return final_best_path
