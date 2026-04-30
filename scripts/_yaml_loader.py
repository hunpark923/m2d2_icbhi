"""
Helpers shared by the entry-point scripts to load YAML configs and apply
CLI overrides, returning fully-populated dataclasses.
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Any, Dict, Type

import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def fill_dataclass(cls: Type, mapping: Dict[str, Any]):
    """Construct a dataclass instance by selecting keys from ``mapping`` that
    match its fields. Unknown keys are ignored.
    """
    field_names = {f.name for f in dataclasses.fields(cls)}
    kwargs = {k: v for k, v in mapping.items() if k in field_names}
    # Convert lists that should be tuples (mostly for type hygiene).
    for f in dataclasses.fields(cls):
        if f.name in kwargs:
            t = f.type
            if str(t).startswith("Tuple") and isinstance(kwargs[f.name], list):
                kwargs[f.name] = tuple(kwargs[f.name])
    return cls(**kwargs)


def apply_overrides(d: Dict[str, Any], overrides: list[str]) -> Dict[str, Any]:
    """Apply ``key=value`` overrides to a dict (top-level keys only).

    Values are parsed as YAML scalars so that numbers and booleans are
    correctly typed (e.g. ``epochs=200`` -> int, ``use_amp=false`` -> bool).
    """
    for kv in overrides:
        if "=" not in kv:
            raise ValueError(f"Override must be of form key=value: {kv!r}")
        k, v = kv.split("=", 1)
        d[k.strip()] = yaml.safe_load(v)
    return d


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--config", type=str, required=True,
        help="Path to a YAML configuration file.",
    )
    p.add_argument(
        "--override", nargs="*", default=[],
        metavar="key=value",
        help="Override individual fields, e.g. --override seed=43 epochs=120.",
    )
    return p


def resolve_config_path(path: str) -> str:
    """Return an absolute path; useful when scripts are launched from
    arbitrary working directories."""
    return str(Path(path).expanduser().resolve())
