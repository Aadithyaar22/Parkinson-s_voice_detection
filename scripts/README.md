# Scripts

Two standalone utilities. Run them from the project root
(`cd /path/to/pva2 && python scripts/<name>.py ...`).

---

## `train_on_uci_original.py`

Retrain the pipeline on the clean 195-row original UCI Parkinson's CSV
(Little 2007), rather than the 588-row augmented version. No extra
dependencies needed.

```bash
# Make sure data/parkinsons_original.csv exists, then:
python -m scripts.train_on_uci_original
```

Outputs: `models_original/parkinsons_pipeline.joblib`,
`models_original/training_report.json`.

To deploy the original-CSV model instead of the augmented one, move or
copy the contents of `models_original/` into `models/` and restart
`app.py`.

---

## `extract_features_from_audio.py`

Walk a folder of audio files, extract the **full 56-feature set**
(UCI-22 + 34 extras: CPP, MFCC stats, formants F1–F3, spectral tilt),
and write one CSV row per recording. Run this locally on your MacBook
on datasets too large to upload (Italian Parkinson's Voice and Speech,
etc.).

### Install dependencies

```bash
pip install -r requirements.txt
# Also need ffmpeg for non-WAV formats:
brew install ffmpeg           # macOS
# or: sudo apt install ffmpeg
```

### Run on the Italian Parkinson's dataset

Assuming you've unzipped the dataset to `~/data/ItalianPVS/`, and the
folder structure looks like:

```
ItalianPVS/
├── 28 People with Parkinson's Disease/
│   ├── Subject01/<task>/*.wav
│   └── ...
├── 22 Young Healthy Control/
└── 15 Elderly Healthy Control/
```

Then:

```bash
cd /path/to/pva2
python scripts/extract_features_from_audio.py \
    --input  ~/data/ItalianPVS \
    --output italian_features.csv \
    --workers 4
```

On an M4 MacBook Air this processes roughly 1–3 files per second, so
~831 recordings takes **10–20 minutes**. Progress bar + ETA displayed.

### Label auto-detection

By default the script infers the label (0 = healthy, 1 = PD) and the
subject ID from the folder path. It handles keywords like `PD`,
`Parkinson`, `HC`, `Healthy`, `Control` in any part of the path. If
auto-detection fails for some files, they're skipped with a warning.

For anything auto-detection can't handle, provide a manual mapping:

```csv
filename,label,subject
path/to/file1.wav,1,Subject01
path/to/file2.wav,0,Subject02
```

Then pass `--labels-csv mapping.csv`.

### Smoke test before the full run

```bash
python scripts/extract_features_from_audio.py \
    --input ~/data/ItalianPVS --output test.csv --limit 5 --workers 1
```

This processes only 5 files. Check that `test.csv` has 60 columns
(`name, subject, status, 22 UCI features, 34 extras, error`) and the
values look reasonable.

### Output

A CSV with columns:

```
name          Relative path from --input root (unique identifier)
subject       Inferred subject ID (for subject-grouped CV)
status        0 = healthy, 1 = PD
[22 UCI features]
[34 extra features]
error         Error message if extraction failed (else empty)
```

## `extract_wav2vec2_embeddings.py`

Extract 1024-dim wav2vec2-XLS-R embeddings from a folder of audio.
Same structure as `extract_features_from_audio.py` but produces deep
embeddings instead of hand-crafted features. XLS-R was pretrained on
128 languages including Hindi, Tamil, Telugu, Bengali, Urdu, Marathi —
so the embeddings are expected to be more robust to cross-language /
cross-accent evaluation than MDVP features.

### Install extra deps (one-time, ~1.5GB)

```bash
pip install -r requirements_wav2vec2.txt
```

On Apple Silicon M1/M2/M3/M4, torch ships with MPS (Metal) support out
of the box. No other configuration needed.

### Run on the Italian training corpus

```bash
python scripts/extract_wav2vec2_embeddings.py \
    --input  ~/Downloads/ItalianPVS \
    --output italian_w2v2.csv \
    --labels auto
```

First run downloads the model weights (~1.2GB) from HuggingFace. After
that, everything's cached locally. On an M4 Air expect ~1-2 seconds per
recording, so the full Italian corpus takes 15-25 minutes.

The output CSV has columns `name, subject, status, emb_0000, ..., emb_1023,
error` — one row per file. Roughly 5-10 MB total.

### Run on YOUR OWN test recordings

For sanity-checking on non-Italian voices (you + friends saying "aaah"),
use `--labels unknown` so status is set to -1 for every file:

```bash
python scripts/extract_wav2vec2_embeddings.py \
    --input  ~/Desktop/my_recordings \
    --output my_test_w2v2.csv \
    --labels unknown
```

Upload both CSVs to Claude — one to train the classifier, one to predict
on your voice and tell you what the model thinks.

---
