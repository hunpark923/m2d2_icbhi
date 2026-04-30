"""
Configuration dataclass for the attention-MIL classifier stage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class ClsConfig:
    """Hyperparameters for attention-MIL classifier training and evaluation.

    Group: model checkpoints
    --------------------------------------------------------------
    beats_checkpoint_path
        Path to the public BEATs checkpoint (used for architecture
        initialisation; weights are subsequently overwritten by the
        SSL-pretrained checkpoint specified at the command line).

    Group: device / reproducibility
    --------------------------------------------------------------
    device, seed, split_seed
        Master seeds. ``split_seed`` is used independently for the
        patient-wise train/val split so that the same partition can be
        reused across multiple runs.

    Group: data / batches
    --------------------------------------------------------------
    num_classes, sample_rate
        Task setup.
    batch_size, accumulation_steps
        Effective batch size = batch_size * accumulation_steps.

    Group: optimisation
    --------------------------------------------------------------
    epochs, lr, weight_decay, min_lr, use_amp, grad_clip
        Standard optimisation settings (AdamW + cosine annealing).
    patience
        Early-stopping patience on the validation ICBHI score.

    Group: imbalance handling
    --------------------------------------------------------------
    focal_gamma
        Focal-loss focusing parameter.
    label_smoothing
        Label-smoothing epsilon.
    sampler_power
        Class-frequency exponent for the WeightedRandomSampler;
        weight ~= 1 / count^sampler_power.

    Group: evaluation
    --------------------------------------------------------------
    val_ratio
        Fraction of patients held out from training as validation.
    eval_shift_secs
        Temporal shifts (seconds) used for test-time augmentation.
        Predicted probabilities are averaged over all shifts.
    eval_pad_sec
        Optional zero-padding (seconds) applied at both ends prior to
        cycle truncation; used to reduce boundary artefacts.

    Group: threshold tuning
    --------------------------------------------------------------
    threshold_steps, threshold_min, threshold_max
        Search grid for the Normal-probability decision threshold.
    selection_metric
        Tie-breaking order for threshold selection
        ("gs_then_pm" or "pm_then_gs").

    Group: output
    --------------------------------------------------------------
    save_root, run_name
        Output directory layout.
    """

    # Model checkpoint
    beats_checkpoint_path: str = "./checkpoints/BEATs_iter3_plus_AS2M.pt"

    # Device / reproducibility
    device: str = "cuda"
    seed: int = 42
    split_seed: int = 42

    # Data / batches
    num_classes: int = 4
    sample_rate: int = 16000
    batch_size: int = 8
    accumulation_steps: int = 4

    # Optimisation
    epochs: int = 100
    lr: float = 2e-5
    weight_decay: float = 0.01
    min_lr: float = 1e-6
    use_amp: bool = True
    grad_clip: float = 1.0
    patience: int = 10

    # Imbalance handling
    focal_gamma: float = 1.5
    label_smoothing: float = 0.05
    sampler_power: float = 0.5

    # Evaluation
    val_ratio: float = 0.20
    eval_shift_secs: Tuple[float, ...] = field(
        default_factory=lambda: (-0.35, -0.20, -0.10, 0.0, 0.10, 0.20, 0.35)
    )
    eval_pad_sec: float = 0.0

    # Threshold tuning
    threshold_steps: int = 91
    threshold_min: float = 0.05
    threshold_max: float = 0.95
    selection_metric: str = "gs_then_pm"

    # Output
    save_root: str = "./runs/cls"
    run_name: str = "attention_mil"

    # Misc
    dropout: float = 0.0
    num_workers: int = 0
    pin_memory: bool = True
