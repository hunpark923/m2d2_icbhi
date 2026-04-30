"""
M2D2 (Masked Modeling Duo with Diffusion) self-supervised learning model.

The model couples three encoder branches:

* *Online encoder*  — trainable BEATs encoder, processes the noise-mixed
  waveform under masking and diffusion corruption.
* *Target encoder*  — frozen copy of the online encoder, updated by EMA;
  produces clean targets at masked positions.
* *Offline teacher* — frozen BEATs encoder, retains original AudioSet
  knowledge as an auxiliary distillation target.

A transformer predictor reconstructs masked-position representations from the
visible (possibly diffusion-corrupted) tokens, supervised by an MSE loss
against both targets. The diffusion difficulty follows the *mountain
schedule* (warm-up / ramp-up / plateau / cool-down).
"""
from __future__ import annotations

import random
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.beats_loader import create_beats_model
from src.ssl.config import SSLConfig
from src.ssl.diffusion import DiffusionSchedule
from src.ssl.predictor import TransformerPredictor


def mountain_schedule_tmax(epoch: int, cfg: SSLConfig) -> int:
    """Maximum diffusion timestep ``t_max`` for the given epoch under the
    four-phase mountain-schedule curriculum.

    Phases (lengths controlled by ``cfg.mountain_*_frac``):
        warm-up    -> t_max = 0
        ramp-up    -> linear ramp 0 -> t_peak
        plateau    -> t_max = t_peak
        cool-down  -> linear ramp t_peak -> 0
    """
    total_e = cfg.ssl_noise_curriculum_epochs
    if epoch >= total_e:
        return 0

    t_peak = cfg.ssl_noise_t_end
    w_end = int(total_e * cfg.mountain_warmup_frac)
    r_end = w_end + int(total_e * cfg.mountain_ramp_frac)
    p_end = r_end + int(total_e * cfg.mountain_plateau_frac)

    if epoch < w_end:
        return 0
    if epoch < r_end:
        return int(t_peak * (epoch - w_end) / max(1, (r_end - w_end)))
    if epoch < p_end:
        return t_peak
    return int(t_peak * (1 - (epoch - p_end) / max(1, (total_e - p_end))))


class M2D2Model(nn.Module):
    """M2D2 self-supervised learning model.

    Parameters
    ----------
    cfg
        SSL configuration.
    offline_teacher
        Frozen BEATs encoder retaining the original AudioSet weights.
    """

    def __init__(self, cfg: SSLConfig, offline_teacher: nn.Module) -> None:
        super().__init__()
        self.cfg = cfg

        self.online_encoder, self.dim, _ = create_beats_model(
            cfg.beats_checkpoint_path
        )
        self.target_encoder, _, _ = create_beats_model(cfg.beats_checkpoint_path)
        self.target_encoder.load_state_dict(self.online_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.teacher = offline_teacher
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.predictor = TransformerPredictor(
            embed_dim=self.dim,
            depth=cfg.predictor_depth,
            num_heads=cfg.predictor_num_heads,
            mlp_ratio=cfg.predictor_mlp_ratio,
        )
        self.diffusion = DiffusionSchedule(cfg.diffusion_timesteps).to(cfg.device)
        self.current_epoch = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def get_current_tmax(self) -> int:
        """Current ``t_max`` according to the mountain-schedule curriculum."""
        return mountain_schedule_tmax(self.current_epoch, self.cfg)

    @staticmethod
    def _safe_extract(model: nn.Module, wav: torch.Tensor) -> torch.Tensor:
        """Forward through a BEATs encoder in FP32 with an explicit padding
        mask. Disabling autocast here is necessary because the upstream BEATs
        implementation contains ops that are unstable under FP16 with the
        very small batches used by SSL.
        """
        with torch.cuda.amp.autocast(enabled=False):
            wav_fp32 = wav.float()
            padding_mask = torch.zeros(
                wav_fp32.shape[0], wav_fp32.shape[1],
                dtype=torch.bool, device=wav.device,
            )
            feats, _ = model.extract_features(
                wav_fp32, padding_mask=padding_mask
            )
        return feats

    # ------------------------------------------------------------------
    # Forward / loss
    # ------------------------------------------------------------------
    def forward(
        self,
        clean_wav: torch.Tensor,
        mixed_wav: torch.Tensor,
    ) -> torch.Tensor:
        """One forward pass returning the M2D2 SSL loss.

        Parameters
        ----------
        clean_wav
            Clean reference waveform, fed to target encoder and teacher.
        mixed_wav
            Noise-mixed waveform, fed to the online encoder.
        """
        # 1. Online-encoder features on the (mixed) input.
        feat_online = self._safe_extract(self.online_encoder, mixed_wav)
        b, n, _ = feat_online.shape

        # 2. Random masking.
        m = int(n * self.cfg.mask_ratio)
        vis_list: List[torch.Tensor] = []
        mask_list: List[torch.Tensor] = []
        for _ in range(b):
            perm = torch.randperm(n, device=clean_wav.device)
            mask_list.append(perm[:m])
            vis_list.append(perm[m:])

        # 3. Diffusion corruption on a random subset of visible tokens
        #    (only when t_max > 0, i.e. outside the warm-up phase).
        t_max = self.get_current_tmax()
        feat_corrupted = feat_online.clone()
        if t_max > 0:
            for i in range(b):
                if random.random() > self.cfg.ssl_diff_prob:
                    continue
                vis_idx = vis_list[i]
                k = int(len(vis_idx) * self.cfg.ssl_diff_vis_ratio)
                if k > 0:
                    sub_idx = vis_idx[torch.randperm(len(vis_idx))[:k]]
                    t = torch.randint(
                        1, t_max + 1, (k,), device=clean_wav.device
                    )
                    feat_corrupted[i, sub_idx] = self.diffusion.add_noise(
                        feat_corrupted[i, sub_idx], t
                    )

        # 4. Predictor reconstructs the full sequence.
        pred = self.predictor(feat_corrupted, vis_list, mask_list, n)

        # 5. Targets from EMA target encoder and offline teacher (clean view).
        with torch.no_grad():
            tgt_ema = self._safe_extract(self.target_encoder, clean_wav)
            tgt_off = self._safe_extract(self.teacher, clean_wav)
            tgt_ema = (tgt_ema - tgt_ema.mean(dim=-1, keepdim=True)) / (
                tgt_ema.std(dim=-1, keepdim=True) + 1e-6
            )
            tgt_off = (tgt_off - tgt_off.mean(dim=-1, keepdim=True)) / (
                tgt_off.std(dim=-1, keepdim=True) + 1e-6
            )

        # 6. Sum of two MSE terms over masked positions.
        loss = torch.zeros((), device=clean_wav.device)
        cnt = 0
        for i in range(b):
            midx = mask_list[i]
            if midx.numel() == 0:
                continue
            p = pred[i, midx]
            t_e = tgt_ema[i, midx]
            t_o = tgt_off[i, midx]
            curr = F.mse_loss(p, t_e) + F.mse_loss(p, t_o)
            if torch.isfinite(curr):
                loss = loss + curr
                cnt += 1
        if cnt == 0:
            return torch.tensor(0.0, device=clean_wav.device, requires_grad=True)
        return loss / cnt

    @torch.no_grad()
    def update_ema(self, decay: float = 0.996) -> None:
        """EMA update of target-encoder parameters from the online encoder."""
        online_params = dict(self.online_encoder.named_parameters())
        for name, param in self.target_encoder.named_parameters():
            if name in online_params:
                param.data.mul_(decay).add_(
                    online_params[name].data, alpha=1 - decay
                )
