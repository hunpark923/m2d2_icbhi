"""
Threshold selection for the Normal-vs-abnormal decision rule.

The decision rule is::

    y_hat = Normal                                    if p(Normal) >= tau
            argmax_{c in {Crackle,Wheeze,Both}} p(c)  otherwise

The threshold ``tau`` is selected on the *validation* set by maximising the
ICBHI score, with secondary tie-breakers (patient-macro score, threshold
proximity to the median plateau threshold, specificity, sensitivity). The
selected threshold is then frozen for test-time evaluation.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from src.common.metrics import (
    apply_threshold_rule,
    icbhi_score_from_cm,
    patient_macro_icbhi,
)


def tune_threshold_for_cls(
    probs_np: np.ndarray,
    ys: List[int],
    patients: List[str],
    threshold_steps: int = 91,
    threshold_min: float = 0.05,
    threshold_max: float = 0.95,
    tie_eps: float = 1e-12,
) -> Dict:
    """Search the optimal Normal-probability threshold under the
    *GS-then-PM* tie-break order.

    Returns
    -------
    dict
        The winning row, with keys
        ``threshold, patient_macro, global_score, sensitivity,
        specificity, preds, cm``.
    """
    metas = [{"patient": p, "y": int(y)} for p, y in zip(patients, ys)]
    rows: List[Dict] = []

    for th in np.linspace(float(threshold_min),
                          float(threshold_max),
                          int(threshold_steps)):
        preds = apply_threshold_rule(probs_np, float(th)).tolist()
        pm, gs, cm = patient_macro_icbhi(metas, preds)
        _, se, sp = icbhi_score_from_cm(cm)
        rows.append({
            "threshold": float(th),
            "patient_macro": float(pm),
            "global_score": float(gs),
            "sensitivity": float(se),
            "specificity": float(sp),
            "preds": preds,
            "cm": cm,
        })

    # Tier 1: maximise GS.
    best_gs = max(r["global_score"] for r in rows)
    gs_cands = [r for r in rows
                if abs(r["global_score"] - best_gs) <= float(tie_eps)]

    # Tier 2: maximise PM among GS plateau.
    best_pm = max(r["patient_macro"] for r in gs_cands)
    pm_cands = [r for r in gs_cands
                if abs(r["patient_macro"] - best_pm) <= float(tie_eps)]

    # Tier 3: pick threshold closest to the plateau median; break further
    # ties by higher specificity, then higher sensitivity.
    median_th = float(np.median([r["threshold"] for r in pm_cands]))
    pm_cands.sort(
        key=lambda r: (
            abs(r["threshold"] - median_th),
            -r["specificity"],
            -r["sensitivity"],
        )
    )
    return pm_cands[0]
