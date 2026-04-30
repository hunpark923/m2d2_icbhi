"""
Build (or refresh) the cycle cache used by both SSL pretraining and
classifier training.

This is optional: if the cache is missing, it will be built lazily on the
first training run. Pre-building can be useful when running on shared
filesystems where the first-run latency would otherwise affect the training
log.

Usage
-----
    python -m scripts.prepare_data --config configs/cls_finetune.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running this file directly from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._yaml_loader import (
    apply_overrides,
    build_arg_parser,
    fill_dataclass,
    load_yaml,
    resolve_config_path,
)
from src.common.icbhi_dataset import load_or_build_cycles
from src.common.preprocessing import PreprocConfig


def main() -> None:
    parser = build_arg_parser("Build (or refresh) the ICBHI cycle cache.")
    args = parser.parse_args()

    raw = load_yaml(resolve_config_path(args.config))
    pre_dict = apply_overrides(raw.get("preproc", {}), args.override)
    pre = fill_dataclass(PreprocConfig, pre_dict)

    train_cycles, test_cycles = load_or_build_cycles(pre)
    print("\n" + "=" * 60)
    print(f"Train cycles: {len(train_cycles)}")
    print(f"Test cycles : {len(test_cycles)}")
    print(f"Cache dir   : {pre.cache_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
