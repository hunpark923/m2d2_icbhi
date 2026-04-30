"""
Transformer predictor used by M2D2 to reconstruct masked-position
representations from a corrupted token sequence.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class TransformerPredictor(nn.Module):
    """Simple transformer predictor with a learnable mask token and dynamic
    sinusoidal position embeddings.

    Parameters
    ----------
    embed_dim
        Token dimensionality (matches the encoder output).
    depth
        Number of transformer encoder layers.
    num_heads
        Number of attention heads per layer.
    mlp_ratio
        Hidden dimension multiplier for the feed-forward block.
    """

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        # Position embedding cached lazily (rebuilt only when sequence
        # length changes).
        self._pos_embed: torch.Tensor | None = None

    def _build_pos_embed(self, n_tokens: int, dim: int, device) -> torch.Tensor:
        pos = torch.arange(n_tokens, dtype=torch.float32, device=device)
        omega = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device) / dim))
        pe = torch.zeros(1, n_tokens, dim, device=device)
        pe[0, :, 0::2] = torch.sin(pos.unsqueeze(1) * omega)
        pe[0, :, 1::2] = torch.cos(pos.unsqueeze(1) * omega)
        return pe

    def forward(
        self,
        x: torch.Tensor,
        vis_idx: List[torch.Tensor],
        mask_idx: List[torch.Tensor],
        n_tokens: int,
    ) -> torch.Tensor:
        """Reconstruct the full token sequence at the masked positions.

        Parameters
        ----------
        x
            Visible-token features from the online encoder, shape ``(B, N, D)``
            where ``N == n_tokens`` for the current batch.
        vis_idx, mask_idx
            Per-sample index tensors of visible and masked positions.
        n_tokens
            Total number of tokens per sample.
        """
        b, _, d = x.shape
        seq = torch.zeros(b, n_tokens, d, device=x.device)
        for i in range(b):
            if vis_idx[i].numel() > 0:
                seq[i, vis_idx[i]] = x[i, vis_idx[i]]
            if mask_idx[i].numel() > 0:
                seq[i, mask_idx[i]] = self.mask_token.squeeze(0)

        if self._pos_embed is None or self._pos_embed.shape[1] != n_tokens:
            self._pos_embed = self._build_pos_embed(n_tokens, d, x.device)
        seq = seq + self._pos_embed

        for blk in self.blocks:
            seq = blk(seq)
        return self.norm(seq)
