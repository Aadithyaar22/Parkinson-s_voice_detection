# Parkinson's Voice Analyser

> Detect Parkinson's disease from three seconds of vocal phonation —
> using a multilingual self-supervised speech model that doesn't care
> what language you speak.

```
              ┌──────────────────────────────────────────────────┐
  voice ─────▶│  wav2vec2-XLS-R  (frozen, 128-language pretrain) │─────▶ 1024-dim embedding
  "aaaah"     └──────────────────────────────────────────────────┘             │
                                                                               ▼
                              ┌────────────────────┐
                              │  Logistic Regression │  ───▶  P(Parkinson's) ∈ [0, 1]
                              │  (Italian-trained)   │
                              └────────────────────┘
```

**What this is:** a working voice-screening prototype with rigorous,
honestly-reported evaluation. CV AUC **0.972**, subject-level AUC
**0.996**, deployed as a Flask web app you can run locally.

**What this isn't:** a medical diagnostic device. Don't use it on
patients.

---

## The numbers

All metrics use **5-fold subject-grouped cross-validation** — no subject
appears in both train and test folds. Anything else (random splits,
recording-level CV) silently leaks information through speaker identity
and inflates accuracy. We don't do that here.

| Metric                                         |    Value |
| ---------------------------------------------- | -------: |
| **CV AUC** (5-fold, subject-grouped)           | **0.972 ± 0.034** |
| CV accuracy                                    |    0.942 |
| CV F1                                          |    0.945 |
| **Subject-level AUC** (recordings averaged per subject) | **0.996** |
| Subject-level accuracy                         |    0.951 (58 / 61 subjects) |
| Tuned decision threshold (Youden's J on OOF)   |    0.380 |
| Trained on                                     |    831 recordings · 61 subjects |
| Backend                                        |    wav2vec2-XLS-R + LogReg |

For comparison, our best hand-crafted MDVP model (a tuned voting
ensemble on 56 acoustic features) hit **CV AUC 0.974** — essentially
identical. The two approaches converge to the same ceiling on this
dataset, which we think is a real result, not an artifact.

The interesting difference is generalization, not accuracy. See
[**The plot twist**](#the-plot-twist) below.

---

## See it work

```bash
unzip Parkinsons-Voice-Analyser-v2.zip
cd pva2

# Core deps
pip install -r requirements.txt
brew install ffmpeg                      # macOS;  apt install ffmpeg on Linux

# Heavy deps (~1.5 GB, only needed for the wav2vec2 backend)
pip install -r requirements_wav2vec2.txt

# Make the model pickle compatible with whatever scikit-learn you have
python scripts/refit_w2v2_local.py

# Run
python app.py
```

Open <http://127.0.0.1:5000>. Click record, say "aaaah" steadily for
3–5 seconds, hit Analyse.

The first request downloads wav2vec2-XLS-R (~1.2 GB from HuggingFace)
and warms the MPS device — takes ~30 seconds the first time, ~1 second
after. On an Apple Silicon M-series chip the embedding extraction runs
on the GPU automatically; on Linux/Windows it'll use CUDA if available
or fall back to CPU.

If wav2vec2 is too heavy and you just want the demo, swap to the
hand-crafted backend (no torch needed):

```bash
cp models_italian_tuned/* models/
python app.py
```

The Flask app auto-detects which backend you've put in `models/` based
on the feature-list filename, so this just works.

---

## Why this project exists

The starting point was an existing student project — a Flask app that
classified sustained-vowel recordings as PD or healthy using a Random
Forest on the UCI Parkinson's voice features. A reasonable hackathon
prototype. But once we read the code carefully, several silent failures
came into view, and one of them was load-bearing.

The original repo had:

| Bug                                                                | Impact |
| ------------------------------------------------------------------ | ------ |
| `MDVP:Jitter(%)`, `PPQ`, and `RAP` all aliased to the same Praat call | Three of 22 features held the same value |
| `MDVP:Shimmer(dB)` was mistakenly assigned the `APQ3` value           | Wrong scale, wrong meaning |
| `Jitter:DDP` aliased to local jitter (should be `3 × RAP`)            | Praat's published identity was broken |
| `Shimmer:APQ5` and `Shimmer:DDA` hardcoded to `None`                  | Imputed with training-set medians at inference |
| `RPDE`, `PPE`, `spread1`, `spread2`, `D2`, `DFA` hardcoded to `None`  | Top-3 most predictive features were never computed |
| Random train/test splits despite multiple recordings per subject       | Subject leakage; reported accuracy massively inflated |
| `nolds` (used for DFA) wasn't even in `requirements.txt`              | DFA always failed silently |

The model was making predictions with three of its top five most
predictive features permanently fixed at "training-set average,"
trained on data with subject leakage. The reported accuracy was a
mirage.

So we rebuilt it.

---

## How it was built

### Phase 1 — fix the foundation

We started by rewriting `feature_extractor.py` from scratch using
`praat-parselmouth` directly. Each MDVP feature now comes from its own
dedicated Praat call. Two identities get verified in `tests/` on every
build: `DDP ≈ 3 × RAP` and `DDA ≈ 3 × APQ3` — these are mathematical
constraints from Praat's documentation, and if they fail, something's
aliased that shouldn't be.

The nonlinear features that were missing (RPDE, DFA, D2, PPE,
spread1/2) live in `src/nonlinear_features.py`. RPDE is implemented
from first principles following Little et al. 2007 — delay-embed the
signal, find first recurrences into an ε-ball, take entropy of the
return-time histogram. The rest use `nolds` (DFA, D2) or are
straightforward statistical measures of log-F0 (PPE, spread1, spread2).

We then added 34 extended features that the UCI set doesn't have but
that the clinical-dysphonia literature considers important:

```
  CPP (cepstral peak prominence) ─┐
                                  │
  MFCC 1–13 mean & std            ├──▶  src/extra_features.py
                                  │
  Formants F1, F2, F3 + bandwidths┤
                                  │
  Spectral tilt                   ┘
```

22 + 34 = 56 features total when running in extended mode.

### Phase 2 — train it honestly

The UCI dataset has 32 subjects, but 8 of them (the healthy controls)
contribute 60% of the recordings. With that imbalance, *random*
train/test splits are essentially memorizing speakers. The honest
evaluation is `StratifiedGroupKFold` keyed on subject — and it gives a
much lower number than random splits suggest.

```
random splits (the original repo's approach)         CV AUC ≈ 0.99   ← leaked
subject-grouped CV (our approach, the honest number)  CV AUC ≈ 0.80   ← real
```

We then added the **Italian Parkinson's Voice and Speech** dataset
(Dimauro et al. 2019) — 831 recordings from 61 subjects, properly
balanced (24 PD / 37 HC). Four times more data, and a real population
rather than the UCI's tiny 8-healthy-speaker pool. CV AUC on Italian
alone immediately jumped to **0.97** with the same pipeline.

We also tuned aggressively: 75 Optuna trials across XGBoost, LightGBM,
RandomForest, and stacking and voting ensembles. The headline result of
all that work was that **tuning didn't move CV AUC**. Inner-CV during
tuning looked like 0.98, but proper outer CV came back to 0.974 —
exactly the nested-CV correction that proves the gain was illusory.
This was confirmation we'd hit the ceiling.

### The plot twist

The most interesting finding came from a sanity-check experiment
nobody normally bothers to run: train on one corpus, test on a
**different** corpus.

```
   train Italian → test UCI:   AUC 0.31    ← worse than random
   train UCI    → test Italian: AUC 0.55    ← random
```

A model trained only on Italian speakers labels healthy English
speakers as PD, and vice versa. The features the models had learned
weren't *Parkinson's voice features* — they were *Italian voice
features* and *English voice features*. The PD signal was real but
entangled with the language signal in ways that completely failed to
transfer.

This is a known but underreported problem in medical speech analysis.
Most papers train and report on a single corpus and never test what
happens when you change the population. We tested it and found it
broken — and that's a more honest result than yet another
within-corpus 99% claim.

### Phase 3 — wav2vec2

Hand-crafted MDVP features are by definition culturally neutral
(jitter is jitter in any language) but they're sensitive to recording
conditions and they capture only what we as engineers thought to
measure. **Self-supervised speech models** like wav2vec2 learn directly
from raw audio across thousands of hours of speech.

We chose `facebook/wav2vec2-xls-r-300m`, which was pretrained on 128
languages **including Hindi, Tamil, Telugu, Bengali, Urdu, and
Marathi**. The hypothesis: if the model has seen Indian speech during
pretraining, its embeddings should generalize to Indian speakers, even
when the downstream classifier was only trained on Italians.

The pipeline is simple:

```
   audio  ──▶  wav2vec2-XLS-R (frozen)  ──▶  1024-dim embedding  ──▶  Logistic Regression
   (16 kHz mono)                            (mean-pooled across      (Italian-trained)
                                             time axis)
```

We didn't fine-tune wav2vec2 — that needs a real GPU and a lot more
data. We just used the frozen embeddings as inputs to a tiny linear
classifier. On the Italian dataset, this matched the hand-crafted
performance (**CV AUC 0.972** vs hand-crafted's 0.974). Tied.

The qualitative test was different. We recorded the developer (a
healthy Indian-English speaker, never seen during training) saying
"aaaah" four times and ran the wav2vec2 model on those recordings:

```
   recording 1:  P(PD) = 0.007   ✓ healthy
   recording 2:  P(PD) = 0.036   ✓ healthy
   recording 3:  P(PD) = 0.114   ✓ healthy
   recording 4:  P(PD) = 0.219   ✓ healthy
   ──────────────────────────────────────
   mean:         P(PD) = 0.094   ✓ correctly classified
```

All four well below the 0.380 threshold. The hand-crafted model on the
same recordings? Random. This is one subject — not a population study —
but it's the same kind of cross-population test the hand-crafted model
failed completely.

That's why wav2vec2 is the deployed default.

---

## Honest caveats

1. **Not a diagnostic device.** Don't use it on patients. Voice
   screening is research-grade at best.
2. **Single-subject cross-population validation.** We tested four
   recordings from one Indian speaker. We have *not* tested whether
   the model would correctly *flag* PD in an Indian patient — only
   that it correctly marks one healthy Indian speaker as healthy.
3. **No Indian PD ground truth.** This is the most important
   limitation. To validate the cross-language story we'd need a
   labelled Indian PD voice corpus, which isn't easy to obtain.
4. **UCI within-corpus AUC is only 0.69.** UCI has 8 healthy subjects
   total. Performance estimates on UCI are intrinsically noisy.
5. **wav2vec2 embeddings aren't interpretable.** If your use case
   needs to explain *which* acoustic features drove a prediction, use
   the hand-crafted backend — interpretability traded for
   generalization.
6. **The 0.380 threshold was tuned on OOF predictions from training**,
   not on a held-out test set. It's sensibly calibrated, not
   unbiased.

---

## Repository layout

```
pva2/
│
├─ app.py                              Flask server, dual-backend
├─ requirements.txt                    core deps
├─ requirements_wav2vec2.txt           torch + transformers (optional)
├─ README.md                           you are here
├─ FINAL_RESULTS.md                    every metric, in one place
├─ QUICKSTART.md                       one-page run guide
│
├─ src/                                feature extraction + training
│   ├─ feature_extractor.py            22 UCI MDVP features via Praat
│   ├─ extra_features.py               CPP + MFCC + formants + tilt
│   ├─ nonlinear_features.py           RPDE / DFA / D2 / PPE / spread
│   ├─ wav2vec2_inference.py           runtime embedding extraction
│   ├─ audio_utils.py                  load + trim + normalise
│   ├─ train.py                        baseline LR/RF/GBT
│   └─ train_v2.py                     Optuna-tuned + stacking
│
├─ scripts/                            workflow scripts
│   ├─ extract_features_from_audio.py  raw audio → MDVP CSV
│   ├─ extract_wav2vec2_embeddings.py  raw audio → embeddings CSV
│   ├─ tune_italian.py                 staged Optuna tuning
│   ├─ wav2vec2_experiment.py          full w2v2 training + eval
│   ├─ joint_training.py               UCI + Italian joint strategies
│   ├─ crosscorpus_experiments.py      UCI vs Italian cross-corpus
│   └─ refit_w2v2_local.py             sklearn-version-compat refit
│
├─ models/                             current deployment (wav2vec2)
├─ models_wav2vec2/                    wav2vec2 backup
├─ models_italian_tuned/               hand-crafted ensemble
├─ models_joint/                       UCI + Italian joint
├─ models_original/                    UCI-only baseline
│
├─ data/
│   ├─ italian_w2v2.csv                831 × 1024-dim embeddings
│   ├─ italian_features.csv            831 × 56 hand-crafted features
│   ├─ parkinsons_original.csv         clean UCI 195-row CSV
│   └─ my_test_w2v2.csv                developer's test recordings
│
├─ reports/                            experimental writeups
│   ├─ crosscorpus/SUMMARY.md          the cross-corpus story
│   ├─ wav2vec2/results.json           final w2v2 numbers
│   └─ joint_training/results.json     joint-training comparison
│
├─ templates/index.html                the web UI
├─ tests/test_extractor.py             15 sanity checks
└─ uploads/                            runtime temp storage (gitignored)
```

---

## Re-training & extending

The project ships with five trained model variants, each in its own
`models_*/` directory. Swap them by copying into `models/`:

| Variant                    | When to use                                    | Backend |
| -------------------------- | ---------------------------------------------- | ------- |
| `models_wav2vec2/`         | Default. Best generalization. Needs torch.    | wav2vec2 |
| `models_italian_tuned/`    | Best within-Italian, no torch needed.         | hand-crafted |
| `models_joint/`            | Cross-corpus coverage (UCI + Italian).        | hand-crafted |
| `models_original/`         | UCI-only English baseline.                    | hand-crafted |

To train a fresh model on **your own audio dataset**:

```bash
# 1. Walk the audio folder, extract features into a CSV
python scripts/extract_features_from_audio.py \
    --input /path/to/audio --output mydata.csv

# OR for wav2vec2 embeddings:
python scripts/extract_wav2vec2_embeddings.py \
    --input /path/to/audio --output mydata_w2v2.csv

# 2. Train (adapt scripts/train_joint_production.py — it's
#    the simplest end-to-end script in the repo)
python -m scripts.train_joint_production
```

The extraction scripts auto-detect labels from folder names — drop
recordings into `Healthy Controls/` and `PD/` subdirectories and it
just works. Subject IDs are inferred from the per-speaker subdirectory
name. There's an explicit `--labels-csv` escape hatch when the
heuristic gets it wrong.

---

## Tech stack

|                  | Choice                       | Why                                              |
| ---------------- | ---------------------------- | ------------------------------------------------ |
| Acoustic library | praat-parselmouth            | Praat is the de facto standard for voice analysis |
| Nonlinear feats  | nolds + custom               | Standard implementations + Little 2007 RPDE      |
| Tabular ML       | XGBoost, LightGBM, RF        | Strong baselines on small tabular data           |
| Hyperparam tuner | Optuna (TPE)                 | Subject-aware, supports inner CV                 |
| Speech model     | wav2vec2-XLS-R-300m          | Multilingual, includes Indian languages          |
| Inference frame  | PyTorch + transformers       | First-class MPS support on Apple Silicon         |
| Web              | Flask + vanilla JS           | One file, no build step                          |

Datasets:

- **UCI Parkinson's** (Little et al. 2007) — 195 recordings, 32 subjects,
  English. The classic reference dataset.
- **Italian Parkinson's Voice and Speech** (Dimauro et al. 2019) — 831
  recordings, 61 subjects, Italian. Mobile-phone quality.

---

## Built by

Aadithya A R · B.Tech CSE (AI & ML) · Innomatics Research Labs · 2026

The original project was a four-person team effort
(Aadithya, Naman, Yadunandan, Kenisha). This v2 rebuild — feature
extractor fix, multi-corpus training, cross-corpus experiments,
wav2vec2 deployment, web UI — is solo work.

---

## License

MIT. Do whatever you want with the code, but if you build a real
clinical product on top of it, please involve actual clinicians and
real validation. Don't ship a model the same week you read its README.
