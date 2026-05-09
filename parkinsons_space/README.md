---
title: Parkinson's Voice Detection
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
suggested_hardware: cpu-basic
suggested_storage: small
fullWidth: true
license: mit
short_description: Detect Parkinson's disease from voice using wav2vec2-XLS-R
tags:
  - speech
  - medical
  - parkinson
  - wav2vec2
  - voice-analysis
  - health
---

# 🧠 Parkinson's Voice Detection

> Detect Parkinson's disease from **three seconds of vocal phonation** using a
> multilingual self-supervised speech model.

## How to use

1. **Record** — click Record and say "aaaah" steadily for 3–5 seconds in a quiet room
2. **Or upload** — drop a WAV / MP3 / M4A file
3. **Analyse** — the model returns a probability of Parkinson's indicators

## How it works

Audio is passed through **wav2vec2-XLS-R** (Meta AI, pretrained on 128 languages
including Hindi, Tamil, Telugu, Bengali, Marathi) to produce a 1024-dimensional
speech embedding. A logistic regression classifier trained on the Italian
Parkinson's Voice and Speech dataset (Dimauro et al. 2019, 831 recordings,
61 subjects) then predicts the probability.

## Metrics

| Metric | Value |
|---|---|
| 5-fold subject-grouped CV AUC | **0.972 ± 0.034** |
| CV accuracy | 0.942 |
| Subject-level AUC | **0.996** |
| Subject-level accuracy | 0.951 (58 / 61 subjects) |

All evaluation uses **subject-grouped cross-validation** — no subject appears
in both train and test folds.

## ⚠️ Disclaimer

Research / educational prototype only. **Not a medical device.** Voice-based
screening has inherent limitations. Clinical decisions must be made by a
qualified physician or neurologist.

## Source

Full source code, training pipeline and methodology:
[github.com/Aadithyaar22/Parkinson-s_voice_detection](https://github.com/Aadithyaar22/Parkinson-s_voice_detection)
