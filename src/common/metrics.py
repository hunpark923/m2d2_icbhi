"""
Evaluation metrics for the ICBHI 2017 benchmark.

This module implements the **four-class (strict)** sensitivity definition used
throughout the paper: only exact-class matches are counted as correct. The
binary sensitivity variant — sometimes used in the literature — is provided as
a separate function for cross-study comparison.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np


def compute_cm(
    labels: List[int],
    preds: List[int],
    num_classes: int = 4,
) -> np.ndarray:
    """Compute a confusion matrix from class-index sequences.

    Rows correspond to true labels and columns to predicted labels.
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for y, p in zip(labels, preds):
        cm[int(y), int(p)] += 1
    return cm


def icbhi_score_from_cm(cm: np.ndarray) -> Tuple[float, float, float]:
    """Compute the ICBHI score, four-class sensitivity, and specificity from a
    confusion matrix.

    Class ordering: 0=Normal, 1=Crackle, 2=Wheeze, 3=Both.
    Four-class sensitivity counts only diagonal matches as correct.
    """
    n_normal = cm[0].sum()
    n_abnormal = cm[1:].sum()
    correct_normal = cm[0, 0]
    correct_abnormal = cm[1, 1] + cm[2, 2] + cm[3, 3]

    specificity = correct_normal / (n_normal + 1e-12)
    sensitivity = correct_abnormal / (n_abnormal + 1e-12)
    score = (specificity + sensitivity) / 2.0
    return float(score), float(sensitivity), float(specificity)


def binary_sensitivity_from_cm(cm: np.ndarray) -> float:
    """Binary (any-abnormal) sensitivity from a 4x4 confusion matrix.

    Any predicted abnormal class for a true abnormal cycle counts as correct.
    Provided to enable comparison with prior work that uses this convention.
    """
    n_abnormal = cm[1:].sum()
    correct_binary = cm[1:, 1:].sum()
    return float(correct_binary / (n_abnormal + 1e-12))


def patient_macro_icbhi(
    cycles_meta: List[Dict],
    preds: List[int],
) -> Tuple[float, float, np.ndarray]:
    """Patient-macro ICBHI score: average of per-patient ICBHI scores.

    Returns a tuple ``(patient_macro, global_score, global_cm)`` where
    `global_score` is the standard ICBHI score over the union of all cycles
    and `global_cm` is the corresponding confusion matrix.
    """
    by_p: Dict[str, List[int]] = defaultdict(list)
    for i, c in enumerate(cycles_meta):
        by_p[c["patient"]].append(i)

    per_patient_scores: List[float] = []
    for _, idxs in by_p.items():
        y = [int(cycles_meta[i]["y"]) for i in idxs]
        pr = [int(preds[i]) for i in idxs]
        cm = compute_cm(y, pr, num_classes=4)
        score, _, _ = icbhi_score_from_cm(cm)
        per_patient_scores.append(score)

    all_y = [int(c["y"]) for c in cycles_meta]
    all_pr = [int(p) for p in preds]
    global_cm = compute_cm(all_y, all_pr, num_classes=4)
    global_score, _, _ = icbhi_score_from_cm(global_cm)

    return float(np.mean(per_patient_scores)), float(global_score), global_cm


def apply_threshold_rule(probs: np.ndarray, th: float) -> np.ndarray:
    """Apply the Normal-probability threshold decision rule:
    ``Normal`` if ``p(Normal) >= th``, otherwise the argmax over the three
    abnormal classes.
    """
    preds = np.empty(len(probs), dtype=np.int64)
    for i, p in enumerate(probs):
        if p[0] >= th:
            preds[i] = 0
        else:
            preds[i] = 1 + int(np.argmax(p[1:]))
    return preds
