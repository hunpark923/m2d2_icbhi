# M2D2-ICBHI

**Diffusion-augmented self-supervised learning for respiratory sound
classification on the ICBHI 2017 benchmark.**

This repository contains the official implementation of the paper
*"Diffusion-augmented self-supervised learning for respiratory sound
classification on the ICBHI benchmark"* (under review). It provides:

* M2D2 (Masked Modeling Duo with Diffusion) self-supervised pretraining
  on the BEATs backbone.
* A four-phase *mountain-schedule* curriculum that progressively varies
  diffusion difficulty during pretraining.
* An attention-based multiple instance learning (MIL) classifier over
  seven overlapping 2-s instances per 8-s respiratory cycle.
* A *corrected* evaluation split that addresses a previously overlooked
  patient-level data-leakage issue in the official ICBHI partition.
* Released M2D2-pretrained encoder weights (link below) and the
  classifier-training pipeline needed to reproduce all reported results.

## Results

Five independent runs on the official ICBHI 2017 test split (2,756 cycles):

| Metric                      | Value (%)    |
|-----------------------------|--------------|
| ICBHI score                 | 64.30 ± 0.73 |
| Specificity (Normal recall) | 84.34 ± 1.81 |
| Four-class sensitivity      | 44.25 ± 0.73 |
| Binary sensitivity          | 56.23 ± 2.01 |

Robustness on the corrected (leakage-free) test split (2,636 cycles):

| Metric                      | Value (%)    |
|-----------------------------|--------------|
| ICBHI score                 | 64.28 ± 0.79 |
| Specificity                 | 84.47 ± 1.84 |
| Four-class sensitivity      | 44.09 ± 1.00 |

The near-identical numbers across the two splits indicate that the reported
performance is *not* artificially inflated by patient-level leakage.

## Repository layout

```
m2d2-icbhi/
├── README.md
├── LICENSE                       # MIT
├── requirements.txt
├── configs/                      # YAML hyperparameters
│   ├── ssl_pretrain.yaml
│   └── cls_finetune.yaml
├── splits/                       # Official + corrected split files
├── src/
│   ├── common/                   # BEATs loading, preprocessing, metrics
│   ├── ssl/                      # M2D2 model + SSL training loop
│   └── cls/                      # Attention-MIL classifier + TTA + evaluation
├── scripts/                      # CLI entry points
│   ├── prepare_data.py
│   ├── pretrain_ssl.py
│   ├── train_classifier.py
│   └── evaluate.py
└── docs/
    ├── DATA_SETUP.md
    └── REPRODUCIBILITY.md
```

## Quick start

### 1. Install

```bash
git clone https://github.com/<your-username>/m2d2-icbhi.git
cd m2d2-icbhi
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set up data and BEATs

See `docs/DATA_SETUP.md` for full instructions. In short:

```bash
# BEATs reference implementation (external dependency).
git clone https://github.com/microsoft/unilm.git
export BEATS_REPO_PATH=$(pwd)/unilm/beats

# Place the datasets and the BEATs checkpoint as follows:
#   ./checkpoints/BEATs_iter3_plus_AS2M.pt
#   ./data/ICBHI_final_database/*.{wav,txt}
#   ./data/FSD50K/dev_audio/*.wav
#   ./splits/official_split.txt        # (from ICBHI: ICBHI_challenge_train_test.txt)
```

### 3. Reproduce results from the released encoder checkpoint

If you only want to reproduce the classifier-stage results, download the
released M2D2-pretrained encoder checkpoint (link below), place it at
`./checkpoints/best_encoder.pt`, then:

```bash
python -m scripts.train_classifier \
    --config configs/cls_finetune.yaml \
    --ssl-ckpt ./checkpoints/best_encoder.pt
```

For the full five-seed reproduction sweep and the corrected-split
evaluation, see `docs/REPRODUCIBILITY.md`.

### 4. Reproduce from scratch

```bash
# (a) SSL pretraining (~24 h on a single RTX 4080 SUPER).
python -m scripts.pretrain_ssl --config configs/ssl_pretrain.yaml

# (b) Train the attention-MIL classifier on top of the SSL encoder.
python -m scripts.train_classifier \
    --config configs/cls_finetune.yaml \
    --ssl-ckpt ./runs/ssl/<run>/best_encoder.pt
```

## Released checkpoint

The M2D2-pretrained BEATs encoder checkpoint used to produce the results
in the paper is released as a separate file.

> **Download:** https://drive.google.com/file/d/10gjeopmA7VhP1f2bWhfbtNBlgp3cXvf8/view?usp=sharing
>
> **SHA-256:** *<fill in after upload>*
>
> **License:** Same MIT license as the code (see `LICENSE`).
> Note that the BEATs base weights themselves are governed by their
> upstream license; the released checkpoint contains BEATs weights
> **further pretrained** by us via M2D2 SSL.

After downloading, place the file at `./checkpoints/best_encoder.pt`
(or update `--ssl-ckpt` accordingly).

## Citing

If you use this code or the released checkpoint in your research, please
cite our paper:

```bibtex
@article{park2026m2d2,
  title   = {Diffusion-augmented self-supervised learning for respiratory
             sound classification on the ICBHI benchmark},
  author  = {Chan Hun Park and Jung Chan Lee},
  journal = {Biomedical Signal Processing and Control},
  year    = {2026},
  note    = {Under review}
}
```

We also build on:

* Niizumi et al., *Masked Modeling Duo: Towards a Universal Audio
  Pre-Training Framework*, IEEE/ACM TASLP 2024.
* Chen et al., *BEATs: Audio Pre-Training with Acoustic Tokenizers*, ICML
  2023.
* The ICBHI 2017 Respiratory Sound Database (Rocha et al.,
  *Physiol. Meas.* 2019).
* The FSD50K dataset (Fonseca et al., IEEE/ACM TASLP 2022).

Please cite these works when appropriate.

## License

Released under the [MIT License](LICENSE). The BEATs weights and the
ICBHI/FSD50K datasets are governed by their respective upstream licenses.

## Contact

Issues and pull requests are welcome. For questions about the paper or
the released checkpoint, please contact the corresponding author
listed in the paper.
