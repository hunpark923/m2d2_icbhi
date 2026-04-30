# Data and environment setup

This document explains how to obtain the datasets and external dependencies
required by this codebase.

## 1. Python environment

```bash
git clone https://github.com/<your-username>/m2d2-icbhi.git
cd m2d2-icbhi
python -m venv .venv
source .venv/bin/activate          # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

A CUDA-capable GPU is strongly recommended; CPU-only runs are possible but
extremely slow.

## 2. BEATs reference implementation

The codebase depends on the public BEATs implementation by Microsoft.
Clone it once and place it on `PYTHONPATH`, or set the environment variable
`BEATS_REPO_PATH`:

```bash
# Clone alongside this repository.
cd ..
git clone https://github.com/microsoft/unilm.git
export BEATS_REPO_PATH=$(pwd)/unilm/beats
cd m2d2-icbhi
```

Add the export to your shell profile (`~/.bashrc`, `~/.zshrc`, ...) to make it
persistent. Windows users should set `BEATS_REPO_PATH` via the System
Environment Variables panel.

## 3. BEATs pretrained checkpoint

Download `BEATs_iter3_plus_AS2M.pt` from the BEATs project page:

* https://github.com/microsoft/unilm/tree/master/beats

Place it at:

```
m2d2-icbhi/checkpoints/BEATs_iter3_plus_AS2M.pt
```

(Or update `beats_checkpoint_path` in the YAML configs if you prefer a
different location.)

## 4. ICBHI 2017 Respiratory Sound Database

Download the dataset from the official challenge page:

* https://bhichallenge.med.auth.gr/

Unpack such that the layout matches:

```
m2d2-icbhi/
└── data/
    └── ICBHI_final_database/
        ├── 101_1b1_Al_sc_Meditron.wav
        ├── 101_1b1_Al_sc_Meditron.txt
        ├── ...
```

Then copy the official `ICBHI_challenge_train_test.txt` into
`splits/official_split.txt`.

## 5. FSD50K (background-noise pool for SSL pretraining)

Download FSD50K from the official Zenodo release:

* https://zenodo.org/records/4060432

Only the `dev` (or `eval`) audio is required as a noise pool. Unpack such
that:

```
m2d2-icbhi/
└── data/
    └── FSD50K/
        ├── FSD50K.dev_audio/
        │   └── *.wav
        └── ...
```

The pool walks the directory recursively and uses every `.wav` it finds, so
any subset of FSD50K (or any other unlabelled audio collection) can be
substituted.

## 6. M2D2-pretrained encoder checkpoint (optional)

A pretrained checkpoint of the BEATs encoder *after* M2D2 pretraining is
released alongside this codebase (see the project README for a download
link). Saving it as

```
m2d2-icbhi/checkpoints/best_encoder.pt
```

allows the classifier stage to run without first re-doing 300 epochs of
SSL pretraining.

## 7. Verify the setup

```bash
# Pre-build the cycle cache (also confirms that the dataset is correctly
# placed and that the official split file is loadable).
python -m scripts.prepare_data --config configs/cls_finetune.yaml
```

Expected output (numbers may differ slightly across rebuilds, but the
totals should match the figures below):

```
Train cycles: 4142
Test cycles : 2756
```
