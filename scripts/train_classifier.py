"""
Train the attention-MIL classifier on top of an SSL-pretrained encoder.

Usage
-----
    python -m scripts.train_classifier \
        --config configs/cls_finetune.yaml \
        --ssl-ckpt ./checkpoints/best_encoder.pt

    python -m scripts.train_classifier \
        --config configs/cls_finetune.yaml \
        --ssl-ckpt ./checkpoints/best_encoder.pt \
        --override seed=43

If ``--ssl-ckpt`` is omitted, the encoder is trained from the public BEATs
checkpoint specified in the config (this corresponds to the *w/o SSL*
ablation).
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._yaml_loader import (
    apply_overrides,
    build_arg_parser,
    fill_dataclass,
    load_yaml,
    resolve_config_path,
)
from src.cls.config import ClsConfig
from src.cls.train import evaluate_final, train_cls
from src.common.preprocessing import PreprocConfig


def main() -> None:
    parser = build_arg_parser("Train the attention-MIL classifier.")
    parser.add_argument(
        "--ssl-ckpt", type=str, default=None,
        help="Path to the SSL-pretrained encoder checkpoint. "
             "If omitted, training starts from the public BEATs checkpoint.",
    )
    parser.add_argument(
        "--no-eval", action="store_true",
        help="Skip the final test-set evaluation (training only).",
    )
    args = parser.parse_args()

    raw = load_yaml(resolve_config_path(args.config))

    preproc_cfg_dict = apply_overrides(
        raw.get("preproc", {}),
        [o for o in args.override if o.split("=", 1)[0] in
         {f.name for f in dataclasses.fields(PreprocConfig)}],
    )
    pre = fill_dataclass(PreprocConfig, preproc_cfg_dict)

    cls_cfg_dict = apply_overrides(
        raw.get("cls", {}),
        [o for o in args.override if o.split("=", 1)[0] in
         {f.name for f in dataclasses.fields(ClsConfig)}],
    )
    cls_cfg = fill_dataclass(ClsConfig, cls_cfg_dict)

    print("=" * 60)
    print("Attention-MIL classifier training")
    print(f"  SSL checkpoint  : {args.ssl_ckpt or '(none — w/o SSL ablation)'}")
    print(f"  output dir      : {cls_cfg.save_root}/{cls_cfg.run_name}_*")
    print(f"  epochs          : {cls_cfg.epochs}")
    print(f"  batch size      : {cls_cfg.batch_size} "
          f"(x{cls_cfg.accumulation_steps} accum)")
    print(f"  seed            : {cls_cfg.seed}")
    print("=" * 60)

    best_ckpt, te_cycles = train_cls(pre, args.ssl_ckpt, cls_cfg)
    print(f"\nBest classifier checkpoint: {best_ckpt}")

    if not args.no_eval:
        evaluate_final(pre, best_ckpt, te_cycles, cls_cfg)


if __name__ == "__main__":
    main()
