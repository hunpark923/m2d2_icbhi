"""
Construction of cycle-level datasets from raw ICBHI recordings, and patient-
wise train/validation splitting.
"""
from __future__ import annotations

import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

from .preprocessing import (
    PreprocConfig,
    apply_fade,
    label_from_flags,
    parse_annotation,
    read_wav_mono,
)


def load_official_split(split_file: str) -> Tuple[List[str], List[str]]:
    """Load the ICBHI official train/test split file and return the lists of
    .wav filenames assigned to each partition.
    """
    train_files: List[str] = []
    test_files: List[str] = []
    with open(split_file, "r") as f:
        for line in f:
            sp = line.strip().split()
            if len(sp) < 2:
                continue
            fn, tag = sp[0], sp[1].lower()
            if not fn.endswith(".wav"):
                fn += ".wav"
            if "train" in tag:
                train_files.append(fn)
            elif "test" in tag:
                test_files.append(fn)
    return train_files, test_files


def build_cycles_from_files(
    file_list: List[str],
    cfg: PreprocConfig,
) -> List[Dict]:
    """Extract per-cycle waveforms from a list of recordings, attaching
    cycle-level metadata (uid, source file, patient ID, four-class label).
    """
    cycles: List[Dict] = []
    for wav_name in tqdm(file_list, desc="Building cycles"):
        wav_path = os.path.join(cfg.data_dir, wav_name)
        txt_path = os.path.splitext(wav_path)[0] + ".txt"
        if not (os.path.exists(wav_path) and os.path.exists(txt_path)):
            continue
        wav = read_wav_mono(wav_path, cfg.sample_rate)
        wav = apply_fade(wav, cfg.sample_rate, cfg.fade_samples_ratio)
        ann = parse_annotation(txt_path)
        for (st, ed, c, w) in ann:
            s_idx, e_idx = int(st * cfg.sample_rate), int(ed * cfg.sample_rate)
            if e_idx <= s_idx + 1:
                continue
            seg = wav[s_idx:e_idx]
            y = label_from_flags(c, w)
            uid = f"{os.path.splitext(wav_name)[0]}_{st:.3f}_{ed:.3f}"
            cycles.append({
                "uid": uid,
                "file": wav_name,
                "patient": wav_name.split("_")[0],
                "y": int(y),
                "wav": seg.cpu(),
            })
    return cycles


def load_or_build_cycles(
    cfg: PreprocConfig,
) -> Tuple[List[Dict], List[Dict]]:
    """Load cached train/test cycle lists if available, otherwise build them
    from the raw ICBHI database and cache the result.
    """
    os.makedirs(cfg.cache_dir, exist_ok=True)
    tr_cache = os.path.join(cfg.cache_dir, "cycles_train.pt")
    te_cache = os.path.join(cfg.cache_dir, "cycles_test.pt")
    if cfg.use_cache and os.path.exists(tr_cache) and os.path.exists(te_cache):
        print(f"Loaded cached cycles from {cfg.cache_dir}")
        return torch.load(tr_cache), torch.load(te_cache)
    tr_files, te_files = load_official_split(cfg.official_split_file)
    tr_cycles = build_cycles_from_files(tr_files, cfg)
    te_cycles = build_cycles_from_files(te_files, cfg)
    torch.save(tr_cycles, tr_cache)
    torch.save(te_cycles, te_cache)
    return tr_cycles, te_cycles


def split_train_val_by_patient(
    cycles: List[Dict],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Dict], List[Dict]]:
    """Split a list of cycles into training and validation subsets at the
    *patient* level, ensuring that no patient is shared between the two.

    Patients are shuffled with a fixed seed for reproducibility, and the
    first ``val_ratio`` fraction is assigned to the validation subset.
    """
    rng = random.Random(seed)
    by_patient: Dict[str, List[Dict]] = defaultdict(list)
    for c in cycles:
        by_patient[c["patient"]].append(c)
    patients = list(by_patient.keys())
    rng.shuffle(patients)

    n_val = max(1, int(len(patients) * val_ratio))
    val_set = set(patients[:n_val])

    train_cycles, val_cycles = [], []
    for c in cycles:
        (val_cycles if c["patient"] in val_set else train_cycles).append(c)
    return train_cycles, val_cycles
