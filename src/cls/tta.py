"""
Test-time augmentation (TTA) for the classifier.

For each cycle, the bag is constructed at multiple temporal shifts of the
8-second waveform; predicted probabilities are averaged across shifts before
the threshold rule is applied.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from src.common.preprocessing import (
    PreprocConfig,
    cut_pad_to_length,
    robust_rms_normalize,
)
from src.cls.config import ClsConfig


def build_eval_instances_from_raw(
    wav_raw: torch.Tensor,
    pre: PreprocConfig,
    shift_sec: float = 0.0,
    eval_pad_sec: float = 0.0,
) -> torch.Tensor:
    """Build a (M, T) MIL bag for evaluation from a raw cycle waveform.

    Pipeline: optional zero padding -> cut/pad to 8 s -> shift the whole
    8-s bag -> RMS normalisation -> split into instances. Shifting at the
    8-s level (rather than per-instance) preserves overlap relationships
    between adjacent instances.
    """
    wav = wav_raw.clone().float().flatten()

    if float(eval_pad_sec) > 0:
        p = int(round(float(eval_pad_sec) * pre.sample_rate))
        if p > 0:
            wav = F.pad(wav, (p, p))

    wav8 = cut_pad_to_length(wav, pre)

    if abs(float(shift_sec)) > 1e-12:
        shift = int(round(float(shift_sec) * pre.sample_rate))
        if shift != 0:
            wav8 = torch.roll(wav8, shifts=shift, dims=0)

    wav8 = robust_rms_normalize(wav8, pre)

    seg_len = int(pre.seg_sec * pre.sample_rate)
    step = int(pre.instance_step_sec * pre.sample_rate)

    instances: List[torch.Tensor] = []
    for i in range(pre.n_instances):
        s = i * step
        e = s + seg_len
        seg = wav8[s:e]
        if seg.numel() < seg_len:
            seg = F.pad(seg, (0, seg_len - seg.numel()))
        instances.append(seg)
    return torch.stack(instances, dim=0)


@torch.no_grad()
def infer_probs_tta(
    model: nn.Module,
    cycles: List[Dict],
    pre: PreprocConfig,
    cfg: ClsConfig,
    device: torch.device,
    batch_size: int,
    use_amp: bool = True,
) -> Tuple[np.ndarray, List[int], List[str], List[str]]:
    """Infer per-cycle class probabilities using test-time augmentation.

    Returns
    -------
    probs : np.ndarray of shape (N, C)
        Probabilities averaged across all temporal shifts in
        ``cfg.eval_shift_secs``.
    ys : list of int
        Ground-truth labels (preserved order).
    patients : list of str
        Patient IDs (preserved order).
    uids : list of str
        Cycle unique IDs (preserved order).
    """
    model.eval()

    probs_all: List[torch.Tensor] = []
    ys: List[int] = []
    patients: List[str] = []
    uids: List[str] = []

    shift_secs = tuple(cfg.eval_shift_secs)

    for start in range(0, len(cycles), batch_size):
        batch_cycles = cycles[start:start + batch_size]
        ys.extend(int(c["y"]) for c in batch_cycles)
        patients.extend(str(c["patient"]) for c in batch_cycles)
        uids.extend(str(c["uid"]) for c in batch_cycles)

        tta_probs: List[torch.Tensor] = []
        for s in shift_secs:
            x = torch.stack(
                [
                    build_eval_instances_from_raw(
                        wav_raw=c["wav"],
                        pre=pre,
                        shift_sec=float(s),
                        eval_pad_sec=float(cfg.eval_pad_sec),
                    )
                    for c in batch_cycles
                ],
                dim=0,
            ).to(device, non_blocking=True)
            with autocast(enabled=use_amp and device.type == "cuda"):
                logits = model(x)
                probs = torch.softmax(logits.float(), dim=1)
            tta_probs.append(probs)

        avg = torch.stack(tta_probs, dim=0).mean(dim=0)
        probs_all.append(avg.cpu())

    probs_np = torch.cat(probs_all, dim=0).numpy()
    return probs_np, ys, patients, uids
