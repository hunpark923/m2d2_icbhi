"""
Attention-based multiple instance learning (MIL) classifier.

A bag of seven 2-second instances is encoded by the (SSL-pretrained) BEATs
backbone, mean-pooled per instance, gated by a two-layer attention network,
and aggregated into a single bag embedding before a linear classification
head produces four-class logits.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.cls.config import ClsConfig


class AttentionMILClassifier(nn.Module):
    """Attention-based MIL classifier.

    Parameters
    ----------
    encoder
        BEATs encoder (typically initialised from an SSL checkpoint).
    encoder_dim
        Encoder output dimensionality.
    cfg
        Classifier configuration.
    """

    def __init__(
        self,
        encoder: nn.Module,
        encoder_dim: int,
        cfg: ClsConfig,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.cfg = cfg
        self.dim = encoder_dim

        self.inst_norm = nn.LayerNorm(encoder_dim)
        self.attention = nn.Sequential(
            nn.Linear(encoder_dim, 128),
            nn.Tanh(),
            nn.Dropout(cfg.dropout),
            nn.Linear(128, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Dropout(cfg.dropout),
            nn.Linear(encoder_dim, cfg.num_classes),
        )

    def extract_instance_features(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a (B, M, T) bag tensor into per-instance features (B, M, D).

        Forward pass is forced to FP32 to avoid numerical instability in
        the upstream BEATs implementation when using mixed precision.
        """
        b, m, t = x.shape
        x_flat = x.reshape(-1, t)
        with torch.cuda.amp.autocast(enabled=False):
            x_fp32 = x_flat.float()
            padding_mask = torch.zeros(
                b * m, t, dtype=torch.bool, device=x.device
            )
            feat, _ = self.encoder.extract_features(
                x_fp32, padding_mask=padding_mask
            )
        feat = feat.mean(dim=1).reshape(b, m, -1)
        feat = self.inst_norm(feat)
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning (B, num_classes) logits."""
        inst_feat = self.extract_instance_features(x)
        att = F.softmax(self.attention(inst_feat), dim=1)
        bag_feat = (inst_feat * att).sum(dim=1)
        return self.classifier(bag_feat)
