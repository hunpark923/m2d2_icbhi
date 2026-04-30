"""
Configuration dataclass for M2D2 self-supervised pretraining.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class SSLConfig:
    """Hyperparameters for M2D2 SSL pretraining.

    Group: model checkpoints
    --------------------------------------------------------------
    beats_checkpoint_path
        Path to the public BEATs checkpoint used to initialise the
        online encoder, target encoder, and offline teacher.

    Group: device / reproducibility
    --------------------------------------------------------------
    device
        ``"cuda"`` or ``"cpu"``.
    seed
        Master seed for all RNGs.

    Group: optimisation
    --------------------------------------------------------------
    lr, batch_size, accumulation_steps, pretrain_epochs, warmup_epochs
        Standard optimisation settings. The effective batch size is
        ``batch_size * accumulation_steps``.

    Group: SSL input
    --------------------------------------------------------------
    ssl_input_sec
        Length of each pretraining crop, in seconds.
    bg_noise_dir
        Directory containing FSD50K (or other) background-noise WAV files.
    bg_eta
        Linear-mix coefficient for noise; mixed = (1-eta)*clean + eta*noise.
    bg_prob
        Probability of applying noise mixing per training sample.

    Group: masking + diffusion
    --------------------------------------------------------------
    mask_ratio
        Fraction of tokens masked before passing through the online encoder.
    predictor_depth, predictor_num_heads, predictor_mlp_ratio
        Architecture of the transformer predictor.
    diffusion_timesteps
        Total number of timesteps in the cosine diffusion schedule.
    ssl_noise_t_end
        Peak diffusion timestep used during the plateau phase.
    ssl_noise_curriculum_epochs
        Total epochs over which the mountain-schedule curriculum is applied.

    Group: mountain-schedule fractions
    --------------------------------------------------------------
    mountain_warmup_frac, mountain_ramp_frac,
    mountain_plateau_frac, mountain_cooldown_frac
        Fractional lengths of the four phases. Should sum to 1.

    Group: corruption probabilities
    --------------------------------------------------------------
    ssl_diff_prob
        Per-sample probability of applying diffusion corruption.
    ssl_diff_vis_ratio
        Fraction of *visible* (unmasked) tokens to corrupt with diffusion.

    Group: bookkeeping
    --------------------------------------------------------------
    save_eval_every
        Save and linearly-evaluate the encoder every N epochs during training.
    save_root, run_name
        Output directory layout.
    lin_epochs, lin_lr
        Training settings for the linear-probe used during periodic
        monitoring.
    rerank_lin_seeds
        Multiple seeds used in the offline deterministic re-rank that
        selects the final best checkpoint.
    """

    # Model checkpoint
    beats_checkpoint_path: str = "./checkpoints/BEATs_iter3_plus_AS2M.pt"

    # Device / reproducibility
    device: str = "cuda"
    seed: int = 42

    # Optimisation
    lr: float = 2e-4
    batch_size: int = 64
    accumulation_steps: int = 2
    pretrain_epochs: int = 300
    warmup_epochs: int = 20

    # SSL input
    ssl_input_sec: float = 2.0
    bg_noise_dir: str = "./data/FSD50K"
    bg_eta: float = 0.3
    bg_prob: float = 1.0

    # Masking + predictor
    mask_ratio: float = 0.6
    predictor_depth: int = 8
    predictor_num_heads: int = 8
    predictor_mlp_ratio: float = 4.0

    # Diffusion
    diffusion_timesteps: int = 1000
    ssl_noise_t_end: int = 80
    ssl_noise_curriculum_epochs: int = 300

    # Mountain-schedule fractions
    mountain_warmup_frac: float = 0.10
    mountain_ramp_frac: float = 0.20
    mountain_plateau_frac: float = 0.50
    mountain_cooldown_frac: float = 0.20

    # Corruption probabilities
    ssl_diff_prob: float = 0.5
    ssl_diff_vis_ratio: float = 0.25

    # Bookkeeping
    save_eval_every: int = 25
    save_root: str = "./runs/ssl"
    run_name: str = "m2d2"

    # Periodic linear-probe (monitoring)
    lin_epochs: int = 100
    lin_lr: float = 1e-3

    # Final offline re-rank seeds
    rerank_lin_seeds: Tuple[int, int, int] = field(
        default_factory=lambda: (1042, 2042, 3042)
    )
