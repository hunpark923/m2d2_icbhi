# Train/test split definitions

This directory contains two split files used by the codebase.

## `official_split.txt`

The original ICBHI 2017 train/test split as distributed with the dataset.
Each line contains a recording filename and a tag, e.g.

```
101_1b1_Al_sc_Meditron      train
102_1b1_Ar_sc_Meditron      test
...
```

The official split file ships with the ICBHI 2017 release (file
`ICBHI_challenge_train_test.txt`). Copy it here and rename it to
`official_split.txt` (or update `official_split_file` in your config).

## `corrected_split.txt`

A patient-level corrected version of the official split, introduced in this
work. The corrected split removes the 12 contaminated test recordings (120
respiratory cycles) belonging to patients **156** and **218**, whose other
recordings appear in the official **training** partition. The training
partition itself is left unchanged.

The resulting test subset contains 2,636 cycles (vs. 2,756 in the official
test split) and is used to verify that the reported performance is not
inflated by patient-level data leakage. See `docs/REPRODUCIBILITY.md` for the
exact list of removed recordings.

## Usage

```bash
# Standard evaluation on the official split (default).
python -m scripts.train_classifier --config configs/cls_finetune.yaml \
    --ssl-ckpt ./checkpoints/best_encoder.pt

# Final evaluation on the leakage-free corrected split.
python -m scripts.evaluate \
    --config configs/cls_finetune.yaml \
    --cls-ckpt ./runs/cls/<run>/best_cls.pt \
    --split-file ./splits/corrected_split.txt
```
