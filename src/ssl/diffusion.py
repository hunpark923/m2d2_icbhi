"""
Cosine-schedule diffusion process used to corrupt visible tokens during
M2D2 pretraining.
"""
from __future__ import annotations

import math
from typing import Union

import torch


class DiffusionSchedule:
    """Cosine variance schedule with helpers for forward (noising) diffusion.

    Parameters
    ----------
    timesteps
        Total number of timesteps in the schedule.
    """

    def __init__(self, timesteps: int = 1000) -> None:
        self.timesteps = timesteps
        self.betas = self._cosine_beta_schedule(timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1.0 - self.alphas_cumprod
        )

    @staticmethod
    def _cosine_beta_schedule(
        timesteps: int,
        s: float = 0.008,
    ) -> torch.Tensor:
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def add_noise(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Apply forward diffusion at timesteps ``t`` to features ``x``.

        Implements the standard formulation
            x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps,
        with eps ~ N(0, I).
        """
        t = t.clamp(0, self.timesteps - 1).to(x.device).long()
        noise = torch.randn_like(x)
        s_a = self.sqrt_alphas_cumprod[t].view(-1, 1)
        s_om_a = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        return s_a * x + s_om_a * noise

    def to(self, device: Union[str, torch.device]) -> "DiffusionSchedule":
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = (
            self.sqrt_one_minus_alphas_cumprod.to(device)
        )
        return self
