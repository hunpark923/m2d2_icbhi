"""
Loading of the BEATs encoder (Microsoft) and weight-overwriting from
M2D2-pretrained SSL checkpoints.

The BEATs reference implementation must be importable from PYTHONPATH; see
docs/DATA_SETUP.md for installation instructions. The path can also be
provided via the `BEATS_REPO_PATH` environment variable.
"""
from __future__ import annotations

import os
import sys
from typing import Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# BEATs import
# ---------------------------------------------------------------------------
_BEATS_REPO_ENV = os.environ.get("BEATS_REPO_PATH", "")
if _BEATS_REPO_ENV and os.path.isdir(_BEATS_REPO_ENV) and _BEATS_REPO_ENV not in sys.path:
    sys.path.insert(0, _BEATS_REPO_ENV)

try:
    from BEATs import BEATs, BEATsConfig  # type: ignore  # noqa: F401
    BEATS_AVAILABLE = True
except Exception as exc:  # pragma: no cover - environment dependent
    BEATS_AVAILABLE = False
    _BEATS_IMPORT_ERROR = exc


class _ConfigWrapper:
    """Provides attribute-style access to BEATs config dictionaries with
    sensible defaults for keys that may be absent in older checkpoints.
    """
    def __init__(self, config_dict: dict) -> None:
        self.__dict__.update(config_dict)

    def __getattr__(self, name: str):
        defaults = {
            "num_mel_bins": 128,
            "frame_length": 25.0,
            "frame_shift": 10.0,
            "sample_rate": 16000,
            "input_patch_size": (16, 16),
            "encoder_embed_dim": 768,
            "embed_dim": 512,
            "finetuned_model": False,
            "predictor_dropout": 0.0,
            "predictor_class": 527,
        }
        return self.__dict__.get(name, defaults.get(name))


def create_beats_model(
    beats_checkpoint_path: str,
) -> Tuple[nn.Module, int, Tuple[int, int]]:
    """Instantiate a BEATs encoder and load weights from a checkpoint.

    Parameters
    ----------
    beats_checkpoint_path
        Path to the pretrained BEATs checkpoint (e.g. ``BEATs_iter3_plus_AS2M.pt``).

    Returns
    -------
    model : nn.Module
        Loaded BEATs encoder.
    output_dim : int
        Encoder output (hidden) dimensionality.
    patch_size : tuple of int
        Patch size used by the encoder.
    """
    if not BEATS_AVAILABLE:
        raise RuntimeError(
            "BEATs library is not importable. Install it from "
            "https://github.com/microsoft/unilm/tree/master/beats and either "
            "place it on PYTHONPATH or set the BEATS_REPO_PATH environment "
            f"variable. Original error: {_BEATS_IMPORT_ERROR}"
        )
    if not os.path.exists(beats_checkpoint_path):
        raise FileNotFoundError(
            f"BEATs checkpoint not found: {beats_checkpoint_path}"
        )

    print(f"Loading BEATs from: {beats_checkpoint_path}")
    checkpoint = torch.load(beats_checkpoint_path, map_location="cpu")

    cfg_obj = None
    if "cfg" in checkpoint:
        cfg_data = checkpoint["cfg"]
        if isinstance(cfg_data, dict):
            cfg_obj = _ConfigWrapper(cfg_data.copy())
        else:
            cfg_obj = BEATsConfig(checkpoint["cfg"])
    if cfg_obj is None:
        cfg_obj = _ConfigWrapper({})

    model = BEATs(cfg_obj)

    try:
        model.load_state_dict(checkpoint["model"], strict=False)
    except Exception:
        # Fall back to a permissive load that drops mismatched tensors.
        model_dict = model.state_dict()
        pretrained_dict = {
            k: v for k, v in checkpoint["model"].items()
            if k in model_dict and v.size() == model_dict[k].size()
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)

    output_dim = getattr(cfg_obj, "encoder_embed_dim", 768)
    patch_size = getattr(cfg_obj, "input_patch_size", (16, 16))
    return model, output_dim, patch_size


def overwrite_beats_with_ssl_ckpt(
    beats_model: nn.Module,
    ssl_ckpt_path: str,
) -> None:
    """Overwrite a freshly-loaded BEATs encoder with weights from an
    M2D2 SSL checkpoint. Supports several common state-dict layouts.
    """
    if ssl_ckpt_path is None or str(ssl_ckpt_path).strip() == "":
        return
    if not os.path.exists(ssl_ckpt_path):
        raise FileNotFoundError(f"SSL checkpoint not found: {ssl_ckpt_path}")

    print(f"Loading SSL-pretrained weights from: {ssl_ckpt_path}")
    ckpt = torch.load(ssl_ckpt_path, map_location="cpu")

    if "model" in ckpt:
        sd = ckpt["model"]
    elif "online_encoder_state_dict" in ckpt:
        sd = ckpt["online_encoder_state_dict"]
    elif "target_encoder_state_dict" in ckpt:
        sd = ckpt["target_encoder_state_dict"]
    else:
        sd = ckpt

    # Strip "beats." prefix if present (legacy checkpoints).
    new_sd = {
        (k.replace("beats.", "") if k.startswith("beats.") else k): v
        for k, v in sd.items()
    }

    missing, unexpected = beats_model.load_state_dict(new_sd, strict=False)
    print(
        f"  Loaded SSL weights "
        f"(missing={len(missing)}, unexpected={len(unexpected)})"
    )
