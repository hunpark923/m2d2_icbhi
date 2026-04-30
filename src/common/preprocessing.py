"""
Audio preprocessing primitives used by both SSL and classifier pipelines:
RMS normalization, fade ramps, length standardisation, and waveform I/O.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torchaudio


@dataclass
class PreprocConfig:
    """Configuration for cycle-level preprocessing.

    Attributes
    ----------
    data_dir
        Directory containing ICBHI .wav and .txt annotation files.
    official_split_file
        Path to the official ICBHI train/test split definition.
    sample_rate
        Target sample rate (Hz). All audio is resampled to this rate.
    desired_length_sec
        Cycle duration after preprocessing, in seconds.
    fade_samples_ratio
        Length of fade-in/out ramps as ``sample_rate / fade_samples_ratio``.
    pad_types
        ``"repeat"`` (cyclic repetition) or any other value for zero padding.
    train_time_shift_sec
        Maximum random temporal shift applied during training (seconds).
    target_rms
        Target RMS level for amplitude normalisation.
    norm_min_scale, norm_max_scale
        Bounds applied to the normalisation gain factor.
    clamp_amp
        Final amplitude clamp.
    seg_sec
        Length of each MIL instance, in seconds.
    n_instances
        Number of instances per bag (MIL).
    instance_step_sec
        Stride between successive instances within an 8-s cycle, in seconds.
    cache_dir
        Directory where pre-processed cycle caches are stored.
    use_cache
        If True, reuse cached cycle tensors when available.
    """
    data_dir: str = "./data/ICBHI_final_database"
    official_split_file: str = "./splits/official_split.txt"

    sample_rate: int = 16000
    desired_length_sec: float = 8.0
    fade_samples_ratio: int = 16
    pad_types: str = "repeat"

    train_time_shift_sec: float = 0.20
    target_rms: float = 0.05
    norm_min_scale: float = 0.2
    norm_max_scale: float = 5.0
    clamp_amp: float = 1.0

    seg_sec: float = 2.0
    n_instances: int = 7
    instance_step_sec: float = 1.0

    cache_dir: str = "./cache_cycles"
    use_cache: bool = True


def apply_fade(wav: torch.Tensor, sr: int, ratio: int = 16) -> torch.Tensor:
    """Apply a short Hann-shaped fade-in/out at the beginning and end of a
    waveform to suppress segmentation-induced artefacts at cycle boundaries.
    """
    wav = wav.flatten()
    if wav.numel() == 0:
        return wav
    n = int(sr / ratio)
    if n <= 1 or wav.numel() < 2 * n + 1:
        return wav
    fade_in = torch.linspace(0.0, 1.0, n, device=wav.device, dtype=wav.dtype)
    fade_out = torch.linspace(1.0, 0.0, n, device=wav.device, dtype=wav.dtype)
    wav = wav.clone()
    wav[:n] *= fade_in
    wav[-n:] *= fade_out
    return wav


def robust_rms_normalize(
    wav: torch.Tensor,
    cfg: PreprocConfig,
) -> torch.Tensor:
    """RMS-normalise a waveform with bounded gain and final amplitude clipping.
    """
    wav = wav.float()
    rms = torch.sqrt(torch.mean(wav ** 2) + 1e-12)
    if not torch.isfinite(rms) or rms.item() <= 0:
        return torch.zeros_like(wav)
    scale = cfg.target_rms / (rms + 1e-12)
    scale = torch.clamp(scale, cfg.norm_min_scale, cfg.norm_max_scale)
    wav = wav * scale
    wav = torch.clamp(wav, -cfg.clamp_amp, cfg.clamp_amp)
    return torch.nan_to_num(wav, nan=0.0)


def cut_pad_to_length(
    wav: torch.Tensor,
    cfg: PreprocConfig,
) -> torch.Tensor:
    """Standardise a waveform to ``cfg.desired_length_sec`` seconds: truncate
    if longer, otherwise extend by cyclic repetition (``pad_types='repeat'``)
    or zero padding.
    """
    desired_len = int(cfg.desired_length_sec * cfg.sample_rate)
    wav = wav.flatten()
    if wav.numel() > desired_len:
        return wav[:desired_len]
    if wav.numel() < desired_len:
        if cfg.pad_types == "repeat":
            n_rep = int(math.ceil(desired_len / max(1, wav.numel())))
            rep = wav.repeat(n_rep)[:desired_len]
            n = int(cfg.sample_rate / cfg.fade_samples_ratio)
            if n > 1 and rep.numel() >= 2 * n + 1:
                fade_out = torch.linspace(
                    1.0, 0.0, n, device=rep.device, dtype=rep.dtype
                )
                rep[-n:] *= fade_out
            return rep
        out = torch.zeros(desired_len, dtype=wav.dtype, device=wav.device)
        out[: wav.numel()] = wav
        out = apply_fade(out, cfg.sample_rate, cfg.fade_samples_ratio)
        return out
    return wav


def read_wav_mono(path: str, target_sr: int) -> torch.Tensor:
    """Load a .wav file, mix to mono, and resample to ``target_sr``."""
    wav, sr = torchaudio.load(path)
    if wav.ndim == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(
            wav.unsqueeze(0)
        ).squeeze(0)
    return wav.float()


def parse_annotation(txt_path: str) -> List[Tuple[float, float, int, int]]:
    """Parse an ICBHI annotation file into ``(start, end, crackle, wheeze)``
    tuples (one per respiratory cycle).
    """
    out: List[Tuple[float, float, int, int]] = []
    with open(txt_path, "r") as f:
        for line in f:
            sp = line.strip().split()
            if len(sp) < 4:
                continue
            st, ed = float(sp[0]), float(sp[1])
            c, w = int(sp[2]), int(sp[3])
            out.append((st, ed, c, w))
    return out


def label_from_flags(c: int, w: int) -> int:
    """Map crackle/wheeze binary indicators to a four-class label
    (0=Normal, 1=Crackle, 2=Wheeze, 3=Both).
    """
    if c == 0 and w == 0:
        return 0
    if c == 1 and w == 0:
        return 1
    if c == 0 and w == 1:
        return 2
    return 3
