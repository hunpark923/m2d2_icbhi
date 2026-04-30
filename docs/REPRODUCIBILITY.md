# Reproducibility guide

This document explains how to reproduce the headline results reported in the
paper (ICBHI score 64.30 ± 0.73 % across five independent runs) and the
robustness analysis on the corrected split.

## 0. Prerequisites

Follow `docs/DATA_SETUP.md` to install dependencies, download the BEATs
checkpoint, the ICBHI dataset, and (optionally) the M2D2-pretrained
encoder checkpoint.

## 1. Reproduce results from the released encoder checkpoint

If you only want to reproduce the *classifier-stage* results without
redoing 300 epochs of SSL pretraining, place the released encoder
checkpoint at:

```
m2d2-icbhi/checkpoints/best_encoder.pt
```

Then run the classifier with the five reported seeds:

```bash
for seed in 42 43 44 45 46; do
  python -m scripts.train_classifier \
    --config configs/cls_finetune.yaml \
    --ssl-ckpt ./checkpoints/best_encoder.pt \
    --override seed=${seed} split_seed=${seed} \
                run_name=attention_mil_seed${seed}
done
```

Each run takes roughly 1.5–2 hours on a single RTX 4080 SUPER. The final
test-set evaluation is printed at the end of each run.

## 2. Reproduce results from scratch (SSL pretraining + classifier)

```bash
# Step 1: SSL pretraining (single seed, ~24 h on a single RTX 4080 SUPER).
python -m scripts.pretrain_ssl --config configs/ssl_pretrain.yaml

# The deterministically-selected best encoder is saved as
#   ./runs/ssl/<run>/best_encoder.pt

# Step 2: Train the classifier with five different seeds.
SSL_CKPT=./runs/ssl/<run>/best_encoder.pt
for seed in 42 43 44 45 46; do
  python -m scripts.train_classifier \
    --config configs/cls_finetune.yaml \
    --ssl-ckpt "${SSL_CKPT}" \
    --override seed=${seed} split_seed=${seed} \
                run_name=attention_mil_seed${seed}
done
```

## 3. Evaluate on the corrected (leakage-free) split

The corrected split removes the 12 test-side recordings of patients 156 and
218, whose other recordings appear in the official training partition.

After training, evaluate each saved classifier checkpoint on the corrected
split:

```bash
for seed in 42 43 44 45 46; do
  python -m scripts.evaluate \
    --config configs/cls_finetune.yaml \
    --cls-ckpt ./runs/cls/attention_mil_seed${seed}_*/best_cls.pt \
    --split-file ./splits/corrected_split.txt \
    --out ./runs/cls/attention_mil_seed${seed}_corrected.json
done
```

The decision threshold stored in each `best_cls.pt` is reused without
further tuning — the threshold is *frozen* at validation time, then
applied to both the official and the corrected test splits.

## 4. Ablation studies

Each ablation is implemented as a single-flag override on the same
configuration file:

| Ablation | Override |
|---|---|
| `w/o SSL`            | `--ssl-ckpt` not provided |
| `w/o focal loss`     | `--override focal_gamma=0.0` |
| `w/o label smoothing`| `--override label_smoothing=0.0` |
| `w/o MIL attention`  | (replaces attention pooling with mean pooling — see `src/cls/model.py`) |
| `w/o TTA`            | `--override eval_shift_secs="[0.0]"` |

For the MIL-attention ablation, swap `(inst_feat * att).sum(dim=1)` in
`AttentionMILClassifier.forward` for `inst_feat.mean(dim=1)`. A standalone
`MeanPoolingMILClassifier` is intentionally not provided in the public
release to keep the codebase focused on the published model; the
single-line modification is documented here for completeness.

## 5. Numerical determinism

The codebase sets `torch.backends.cudnn.deterministic = True` and disables
benchmark mode, but a few sources of non-determinism remain:

* Floating-point reduction order on GPU.
* The order in which PyTorch DataLoader workers process samples (we use
  `num_workers=0` by default for this reason).
* Mixed-precision (FP16) accumulation in the BEATs encoder.

In practice we observe seed-to-seed reproducibility within ~0.2 percentage
points on a single GPU, and slightly larger drift across different GPU
models. The reported 5-seed mean ± standard deviation captures this
remaining variability.
