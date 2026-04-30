"""
Reproducibility helpers.

Provides utilities to seed all relevant RNGs and to capture/restore RNG state,
which is used to isolate periodic linear-probe evaluation from the main
training trajectory during SSL pretraining.
"""
from __future__ import annotations

import random
from typing import Any, Dict

import numpy as np
import torch


def seed_everything(seed: int = 42) -> torch.Generator:
    """Seed Python's `random`, NumPy, and PyTorch RNGs and return a torch
    Generator initialised with the same seed (useful for DataLoaders).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    g = torch.Generator()
    g.manual_seed(seed)
    return g


def seed_worker(worker_id: int) -> None:
    """DataLoader worker initializer that derives its seed from the parent
    PyTorch RNG state, ensuring reproducible per-worker shuffles.
    """
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def capture_rng_state() -> Dict[str, Any]:
    """Snapshot the current state of all RNGs used by the training pipeline."""
    state = {
        "py": random.getstate(),
        "np": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    """Restore an RNG snapshot previously produced by `capture_rng_state`."""
    random.setstate(state["py"])
    np.random.set_state(state["np"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
