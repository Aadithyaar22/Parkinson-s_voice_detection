# Parkinson's Voice Analyser

A voice-based Parkinson's disease screening system. Record a sustained
/a/ phonation, get a probability of Parkinson's. Two backends supported:

* **wav2vec2-XLS-R + LogReg** *(current default)* — multilingual
  self-supervised speech embeddings classified by logistic regression
* **Hand-crafted MDVP + tuned ensemble** — classical Praat-based
  acoustic features (22 UCI + 34 extensions) fed to a tuned voting
  classifier of XGBoost, LightGBM and RandomForest

Both are trained on the Italian Parkinson's Voice dataset (831
recordings, 61 subjects). All evaluation uses **subject-grouped
cross-validation** (no subject leakage between folds).

---

## Current deployment metrics

| metric | value |
|---|---|
| 5-fold subject-grouped CV AUC | **0.9725 ± 0.034** |
| 5-fold CV accuracy | **0.9423** |
| 5-fold CV F1 | 0.9445 |
| **Subject-level AUC** (averaged across subject's recordings) | **0.9955** |
| **Subject-level accuracy** | 0.9508 |
| **Subject-level F1** | 0.9388 |
| Decision threshold (tuned via Youden's J on OOF) | 0.380 |

Comparison with hand-crafted MDVP features (`models_italian_tuned/`):

| method | CV AUC | CV acc | CV F1 |
|---|---|---|---|
| Hand-crafted tuned voting | 0.9744 | 0.9292 | 0.9327 |
| **wav2vec2 + LogReg** | **0.9725** | **0.9423** | **0.9445** |

Essentially tied on AUC. wav2vec2 nudges ahead on accuracy/F1, and
critically, wav2vec2 embeddings are **language-agnostic** — the same
model trained on Italian correctly classified an Indian-English healthy
speaker as healthy (P(PD) = 0.094 averaged across 4 recordings).
Hand-crafted MDVP features don't generalize that way.

---

## What's different from v1 (the original repo)

The original project had silent feature-extraction bugs and used random
train/test splits despite multiple recordings per subject. This rebuild
fixes the entire pipeline:

| Issue in v1 | Fix in v2 |
|---|---|
| `MDVP:Jitter(%)`, `PPQ`, `RAP` all aliased to the same Praat value | Each from its own Praat call; Praat identities verified |
| `MDVP:Shimmer(dB)` set to `APQ3` value | Separate `Get shimmer (local_dB)` call |
| `Jitter:DDP` aliased to jitter_local (should be 3×RAP) | Correct call; `DDP ≈ 3 × RAP` verified in tests |
| `Shimmer:APQ5`, `DDA` hardcoded to `None`, median-imputed at inference | Computed via dedicated Praat calls |
| `RPDE`, `PPE`, `spread1`, `spread2`, `D2`, `DFA` hardcoded to `None` | Implemented in `src/nonlinear_features.py` |
| `DFA` relied on `nolds` which wasn't in `requirements.txt` | Added to requirements |
| Random train/test splits → subject leakage → inflated accuracy | `StratifiedGroupKFold` by subject throughout |
| Median-imputation at predict time, outside the model | `SimpleImputer` inside the sklearn Pipeline |
| No cross-dataset evaluation | Full cross-corpus + joint-training experiments |
| No deep-learning comparison | wav2vec2-XLS-R benchmark + deployment |

---

## Running it

```bash
# One-time setup
pip install -r requirements.txt

# For the wav2vec2 backend (current default), also install:
pip install -r requirements_wav2vec2.txt     # ~1.5GB

# ffmpeg for non-WAV uploads
brew install ffmpeg                          # macOS
# or: sudo apt install ffmpeg

# Launch the web app
python app.py
# -> http://127.0.0.1:5000
```

First `/predict` request downloads the wav2vec2 weights (~1.2GB from
HuggingFace). On an M4 Air with MPS, subsequent requests take ~1-2s.

The UI supports file upload and in-browser recording. The response
includes P(PD), decision label, and (for the hand-crafted backend) all
22 extracted feature values.

### Switching between backends

Model variants are saved in separate directories. To switch, copy the
chosen one into `models/` and restart the app:

```bash
# wav2vec2 (default, best generalization)
cp models_wav2vec2/* models/

# Hand-crafted tuned ensemble (doesn't need torch)
cp models_italian_tuned/* models/

# Joint UCI + Italian hand-crafted
cp models_joint/* models/

# UCI-only (clean 195-row baseline)
cp models_original/* models/

python app.py
```

The backend is auto-detected from the feature-list file — if the
feature names start with `emb_`, wav2vec2 mode is activated.

---

## Project layout

```
pva2/
├── README.md
├── FINAL_RESULTS.md                    end-to-end results summary
├── app.py                              Flask server, dual-backend
├── requirements.txt                    core deps (no torch)
├── requirements_wav2vec2.txt           torch + transformers
├── data/
│   ├── parkinsons_original.csv         UCI 195-row clean
│   ├── parkinsons.data                 UCI 588-row augmented (unused by default)
│   ├── italian_features.csv            Italian 831-row, hand-crafted feats
│   ├── italian_w2v2.csv                Italian 831-row, wav2vec2 embeddings
│   └── my_test_w2v2.csv                user's test recordings, wav2vec2 embeddings
├── src/
│   ├── audio_utils.py                  load/trim/normalise
│   ├── feature_extractor.py            22 UCI MDVP features via Praat
│   ├── extra_features.py               CPP + MFCC stats + formants + tilt
│   ├── nonlinear_features.py           RPDE, DFA, D2, PPE, spread1/2
│   ├── wav2vec2_inference.py           runtime w2v2 embeddings for Flask
│   ├── train.py                        baseline LR/RF/GBT
│   └── train_v2.py                     Optuna-tuned + stacking
├── scripts/
│   ├── README.md
│   ├── extract_features_from_audio.py  raw audio -> hand-crafted feats CSV
│   ├── extract_wav2vec2_embeddings.py  raw audio -> wav2vec2 embeddings CSV
│   ├── train_on_uci_original.py        UCI-only training
│   ├── train_joint_production.py       UCI + Italian joint
│   ├── train_italian_tuned.py          single-shot Italian tuned training
│   ├── tune_italian.py                 staged Italian tuning (xgb/lgbm/rf/finalize)
│   ├── wav2vec2_experiment.py          train+eval+predict on user w2v2 data
│   ├── crosscorpus_experiments.py      UCI vs Italian cross-corpus analysis
│   └── joint_training.py               3 joint-training strategies compared
├── models/                             current deployment (wav2vec2 by default)
├── models_wav2vec2/                    wav2vec2 backup
├── models_italian_tuned/               hand-crafted tuned voting
├── models_joint/                       UCI + Italian joint
├── models_original/                    UCI-only
├── reports/
│   ├── crosscorpus/                    SUMMARY.md + PNGs + results.json
│   ├── joint_training/                 results.json
│   ├── wav2vec2/                       results.json (final)
│   ├── roc_heldout.png
│   └── feature_importance.png
├── templates/index.html                single-page frontend
├── tests/test_extractor.py             15 sanity checks
└── uploads/                            runtime temp storage
```

---

## Experiments run (summary)

| # | Experiment | Location | Key finding |
|---|---|---|---|
| 1 | Fixed feature-extractor + train on clean UCI | `scripts/train_on_uci_original.py` | Held-out acc 0.857, AUC 0.869 |
| 2 | Extended to 56 features + Optuna tuning | `src/train_v2.py` | CV AUC 0.875, stable across folds |
| 3 | Added Italian corpus (4× data) | `scripts/wav2vec2_experiment.py` setup | Italian-only CV AUC 0.974 |
| 4 | Cross-corpus evaluation | `scripts/crosscorpus_experiments.py` | Italian→UCI transfer fails (AUC 0.31), UCI→Italian too (0.55) |
| 5 | Joint UCI + Italian training | `scripts/joint_training.py` | Lifts UCI transfer AUC from 0.31 → 0.76 |
| 6 | Aggressive tuning, Italian only | `scripts/tune_italian.py` | CV AUC plateau at 0.974 (ceiling hit) |
| 7 | wav2vec2 benchmark | `scripts/wav2vec2_experiment.py` | CV AUC 0.972, +1.3% accuracy, generalizes to Indian voice |
| 8 | Deployment | `app.py` (dual backend) | wav2vec2 as default, hand-crafted as fallback |

Full numerical results in `reports/*/results.json` and
`FINAL_RESULTS.md`.

---

## Honest caveats

These are project limitations worth disclosing in any writeup.

1. **Not a diagnostic device.** Research / educational only.
2. **Only validated on one Indian speaker (the developer).** The 4
   user-recording predictions (all HC, correctly) are a single-subject
   sanity check, not a proper cross-population study.
3. **No ground-truth validation on Indian PD patients.** We have not
   tested whether the model would correctly *flag* PD in an Indian
   speaker — only that it correctly marks a healthy Indian speaker as
   healthy.
4. **UCI within-corpus AUC is only 0.69** when trained alone. UCI has
   just 8 healthy subjects, so the model mostly learns about those 8
   people. Performance estimates on UCI are noisy.
5. **Recording conditions matter.** Italian is phone-recorded, UCI is
   lab-mic. Model performance on new audio depends partly on whether
   your recording environment matches one of these.
6. **Thresholds are tuned on OOF predictions from training**, not on a
   fully held-out set. They're sensibly calibrated, not unbiased.
7. **wav2vec2 embeddings are not interpretable.** If interpretability
   matters for your use case, use the hand-crafted backend.

---

## License

MIT (inherits from the upstream project).
