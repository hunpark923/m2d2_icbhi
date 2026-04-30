"""
Cycle-level dataset that produces 7-instance MIL bags for the classifier.
"""
from __future__ import annotations

import random
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.common.preprocessing import (
    PreprocConfig,
    cut_pad_to_length,
    robust_rms_normalize,
)


class ICBHIClsDataset(Dataset):
    """For each respiratory cycle, produce a (M=7) MIL bag of 2-second
    instances spaced by ``cfg.instance_step_sec``.

    During training the bag is randomly time-shifted; at evaluation time
    test-time augmentation (shifts) is applied via :func:`build_eval_instances_from_raw`
    in :mod:`src.cls.tta` rather than within this class.
    """

    def __init__(
        self,
        cycles: List[Dict],
        pre: PreprocConfig,
        is_train: bool,
        eval_pad_sec: float = 0.0,
    ) -> None:
        if cycles is None or len(cycles) == 0:
            raise RuntimeError(
                "ICBHIClsDataset received empty cycles. Check cache and split."
            )
        self.cycles = cycles
        self.pre = pre
        self.is_train = is_train
        self.eval_pad_sec = float(eval_pad_sec)

        self.seg_len = int(pre.seg_sec * pre.sample_rate)
        self.step = int(pre.instance_step_sec * pre.sample_rate)

    def _make_instances(self, wav_raw: torch.Tensor) -> torch.Tensor:
        wav = wav_raw.clone().float().flatten()
        if self.eval_pad_sec > 0:
            p = int(self.eval_pad_sec * self.pre.sample_rate)
            wav = F.pad(wav, (p, p))

        wav8 = cut_pad_to_length(wav, self.pre)

        if self.is_train and self.pre.train_time_shift_sec > 0:
            s = random.uniform(
                -self.pre.train_time_shift_sec,
                self.pre.train_time_shift_sec,
            )
            shift = int(s * self.pre.sample_rate)
            if shift != 0:
                wav8 = torch.roll(wav8, shifts=shift, dims=0)

        wav8 = robust_rms_normalize(wav8, self.pre)

        instances: List[torch.Tensor] = []
        for i in range(self.pre.n_instances):
            start = i * self.step
            end = start + self.seg_len
            seg = wav8[start:end]
            if seg.numel() < self.seg_len:
                seg = F.pad(seg, (0, self.seg_len - seg.numel()))
            instances.append(seg)
        return torch.stack(instances, dim=0)

    def __len__(self) -> int:
        return len(self.cycles)

    def __getitem__(self, idx: int) -> Dict:
        c = self.cycles[idx]
        instances = self._make_instances(c["wav"])
        return {
            "instances": instances,
            "y": int(c["y"]),
            "patient": c["patient"],
            "uid": c["uid"],
        }


def cls_collate_fn(batch: List[Dict]) -> Dict:
    """Default collate function for :class:`ICBHIClsDataset`."""
    return {
        "instances": torch.stack([b["instances"] for b in batch], dim=0),
        "y": torch.tensor([b["y"] for b in batch], dtype=torch.long),
        "patient": [b["patient"] for b in batch],
        "uid": [b["uid"] for b in batch],
    }
