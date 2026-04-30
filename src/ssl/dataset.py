"""
Datasets and noise sources for M2D2 self-supervised pretraining.
"""
from __future__ import annotations

import math
import os
import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.common.preprocessing import (
    PreprocConfig,
    apply_fade,
    cut_pad_to_length,
    read_wav_mono,
    robust_rms_normalize,
)
from src.ssl.config import SSLConfig


class BackgroundNoisePool:
    """Pool of background-noise WAV files used to mix into the SSL inputs.

    If no noise directory is configured (or it contains no files), the pool
    falls back to randomly drawn cycles from a provided list, ensuring that
    SSL pretraining can still proceed in the absence of FSD50K.
    """

    def __init__(
        self,
        sample_rate: int,
        noise_dir: str = "",
        fallback_cycles: Optional[List[Dict]] = None,
    ) -> None:
        self.sr = sample_rate
        self.paths: List[str] = []
        if noise_dir and os.path.isdir(noise_dir):
            for root, _, files in os.walk(noise_dir):
                for fn in files:
                    if fn.endswith(".wav"):
                        self.paths.append(os.path.join(root, fn))
        self.fallback = fallback_cycles or []
        print(
            f"BackgroundNoisePool: {len(self.paths)} files found "
            f"(+{len(self.fallback)} fallback cycles)."
        )

    def sample(self, target_len: int) -> Optional[torch.Tensor]:
        if not self.paths and not self.fallback:
            return None
        if self.paths:
            wav = read_wav_mono(random.choice(self.paths), self.sr)
        else:
            wav = random.choice(self.fallback)["wav"].float()

        if wav.numel() < target_len:
            wav = wav.repeat(math.ceil(target_len / max(1, wav.numel())))
        st = random.randint(0, max(0, wav.numel() - target_len))
        return wav[st:st + target_len]


class SSLDataset(Dataset):
    """Cycle-level dataset producing (clean, mixed) 2-second views for SSL.

    During training, cycles are time-shifted, RMS-normalised, randomly cropped
    to a 2-second window, and a noise-mixed view is constructed by linear
    blending with a randomly drawn FSD50K segment.
    """

    def __init__(
        self,
        cycles: List[Dict],
        pre: PreprocConfig,
        cfg: SSLConfig,
        noise_pool: BackgroundNoisePool,
        is_train: bool,
    ) -> None:
        if cycles is None or len(cycles) == 0:
            raise RuntimeError(
                "SSLDataset received empty cycles. Check cache and split."
            )
        self.cycles = cycles
        self.pre = pre
        self.cfg = cfg
        self.pool = noise_pool
        self.is_train = is_train
        self._size = 10000 if is_train else len(cycles)

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        c = self.cycles[idx % len(self.cycles)]
        wav = cut_pad_to_length(c["wav"], self.pre)

        # Random temporal shift during training.
        if self.is_train:
            shift = random.uniform(
                -self.pre.train_time_shift_sec,
                self.pre.train_time_shift_sec,
            )
            wav = torch.roll(wav, int(shift * self.pre.sample_rate), 0)

        # RMS normalisation.
        wav = robust_rms_normalize(wav, self.pre)

        # Random 2-second crop (deterministic centre crop during validation).
        step = int(self.pre.instance_step_sec * self.pre.sample_rate)
        seg_len = int(self.pre.seg_sec * self.pre.sample_rate)
        n_inst = self.pre.n_instances
        inst_idx = random.randint(0, n_inst - 1) if self.is_train else n_inst // 2
        s = inst_idx * step
        clean = wav[s:s + seg_len]
        if clean.numel() < seg_len:
            clean = F.pad(clean, (0, seg_len - clean.numel()))

        # Build the noise-mixed view.
        mixed = clean.clone()
        if self.is_train and random.random() < self.cfg.bg_prob:
            bg = self.pool.sample(seg_len)
            if bg is not None:
                bg = apply_fade(bg.float(), self.pre.sample_rate)
                bg_rms = torch.sqrt(torch.mean(bg ** 2) + 1e-12)
                cl_rms = torch.sqrt(torch.mean(clean ** 2) + 1e-12)
                if bg_rms > 1e-9:
                    bg = bg * (cl_rms / bg_rms)
                    mixed = (1 - self.cfg.bg_eta) * clean + self.cfg.bg_eta * bg
                    mixed = torch.clamp(mixed, -1, 1)

        clean = torch.nan_to_num(clean, nan=0.0, posinf=1.0, neginf=-1.0)
        mixed = torch.nan_to_num(mixed, nan=0.0, posinf=1.0, neginf=-1.0)
        return clean, mixed
