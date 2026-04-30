"""
Run M2D2 self-supervised pretraining.

Usage
-----
    python -m scripts.pretrain_ssl --config configs/ssl_pretrain.yaml
    python -m scripts.pretrain_ssl --config configs/ssl_pretrain.yaml \
        --override seed=43 pretrain_epochs=200
"""
from __future__ import annotations

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
from src.common.preprocessing import PreprocConfig
from src.ssl.config import SSLConfig
from src.ssl.train import train_ssl


def main() -> None:
    parser = build_arg_parser("Run M2D2 SSL pretraining.")
    args = parser.parse_args()

    raw = load_yaml(resolve_config_path(args.config))

    preproc_cfg_dict = apply_overrides(
        raw.get("preproc", {}),
        [o for o in args.override if o.split("=", 1)[0] in
         {f.name for f in __import__("dataclasses").fields(PreprocConfig)}],
    )
    pre = fill_dataclass(PreprocConfig, preproc_cfg_dict)

    ssl_cfg_dict = apply_overrides(
        raw.get("ssl", {}),
        [o for o in args.override if o.split("=", 1)[0] in
         {f.name for f in __import__("dataclasses").fields(SSLConfig)}],
    )
    ssl_cfg = fill_dataclass(SSLConfig, ssl_cfg_dict)

    print("=" * 60)
    print("M2D2 SSL pretraining")
    print(f"  preproc cache   : {pre.cache_dir}")
    print(f"  output dir      : {ssl_cfg.save_root}/{ssl_cfg.run_name}_*")
    print(f"  pretrain epochs : {ssl_cfg.pretrain_epochs}")
    print(f"  batch size      : {ssl_cfg.batch_size} "
          f"(x{ssl_cfg.accumulation_steps} accum)")
    print(f"  seed            : {ssl_cfg.seed}")
    print("=" * 60)

    best_path = train_ssl(pre, ssl_cfg)
    if best_path is None:
        print("\nNo best checkpoint produced.")
        return
    print(f"\nFinal best encoder checkpoint: {best_path}")


if __name__ == "__main__":
    main()
