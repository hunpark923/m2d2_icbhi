"""
Focal loss with optional label smoothing.

Used to mitigate the strong class imbalance and to discourage overconfident
predictions in the four-class respiratory-sound setting.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def focal_loss_with_smoothing(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Focal loss with class weighting and label smoothing.

    Parameters
    ----------
    logits
        Raw model outputs of shape (B, C).
    targets
        Class indices of shape (B,).
    alpha
        Optional per-class weight tensor of shape (C,). If provided, the
        loss for each sample is multiplied by ``alpha[target]``.
    gamma
        Focusing parameter; ``gamma = 0`` recovers (smoothed) cross-entropy.
    label_smoothing
        Smoothing strength; 0.0 disables smoothing.

    Returns
    -------
    Mean focal-loss value.
    """
    n_classes = logits.size(1)

    if label_smoothing > 0:
        with torch.no_grad():
            true_dist = torch.full_like(logits, label_smoothing / (n_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - label_smoothing)
        logp = F.log_softmax(logits, dim=1)
        p = torch.exp(logp)
        ce = -(true_dist * logp).sum(dim=1)
        pt = (true_dist * p).sum(dim=1).clamp_min(1e-8)
    else:
        logp = F.log_softmax(logits, dim=1)
        p = torch.exp(logp)
        ce = F.nll_loss(logp, targets, reduction="none")
        pt = p.gather(1, targets.unsqueeze(1)).squeeze(1).clamp_min(1e-8)

    fl = (1.0 - pt) ** gamma * ce
    if alpha is not None:
        a = alpha.gather(0, targets)
        fl = fl * a
    return fl.mean()
