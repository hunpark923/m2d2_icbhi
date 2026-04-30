"""
Evaluate a trained classifier checkpoint on the official or corrected
ICBHI test split.

Usage
-----
    # Evaluate on the official 2,756-cycle test split.
    python -m scripts.evaluate \
        --config configs/cls_finetune.yaml \
        --cls-ckpt ./runs/cls/<run>/best_cls.pt

    # Evaluate on the corrected (leakage-free) 2,636-cycle test subset.
    python -m scripts.evaluate \
        --config configs/cls_finetune.yaml \
        --cls-ckpt ./runs/cls/<run>/best_cls.pt \
        --split-file ./splits/corrected_split.txt

The decision threshold stored in the checkpoint is reused without any
further tuning (i.e. the threshold is *frozen* at validation time).
"""
from __future__ import annotations

import dataclasses
import json
import os
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
from src.cls.train import evaluate_final
from src.common.icbhi_dataset import load_or_build_cycles
from src.common.preprocessing import PreprocConfig


def main() -> None:
    parser = build_arg_parser("Evaluate a trained classifier checkpoint.")
    parser.add_argument(
        "--cls-ckpt", type=str, required=True,
        help="Path to the classifier checkpoint (best_cls.pt).",
    )
    parser.add_argument(
        "--split-file", type=str, default=None,
        help="Path to a split file (defaults to the one in the config). "
             "Use ./splits/corrected_split.txt for the leakage-free subset.",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Optional path to write a JSON file with the test results.",
    )
    args = parser.parse_args()

    raw = load_yaml(resolve_config_path(args.config))

    preproc_dict = apply_overrides(
        raw.get("preproc", {}),
        [o for o in args.override if o.split("=", 1)[0] in
         {f.name for f in dataclasses.fields(PreprocConfig)}],
    )
    pre = fill_dataclass(PreprocConfig, preproc_dict)

    cls_dict = apply_overrides(
        raw.get("cls", {}),
        [o for o in args.override if o.split("=", 1)[0] in
         {f.name for f in dataclasses.fields(ClsConfig)}],
    )
    cls_cfg = fill_dataclass(ClsConfig, cls_dict)

    if args.split_file is not None:
        pre.official_split_file = args.split_file
        # Use a separate cache directory to avoid mixing official / corrected
        # cycles in the same .pt file.
        suffix = Path(args.split_file).stem
        pre.cache_dir = os.path.join(pre.cache_dir + "_" + suffix)

    _, te_cycles = load_or_build_cycles(pre)
    print(f"\nEvaluating on {len(te_cycles)} test cycles "
          f"(split file: {pre.official_split_file})")

    result = evaluate_final(pre, args.cls_ckpt, te_cycles, cls_cfg)

    if args.out is not None:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nWrote JSON results to: {args.out}")


if __name__ == "__main__":
    main()
