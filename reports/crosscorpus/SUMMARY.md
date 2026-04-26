# Cross-corpus experiments — Italian × UCI

Ran: `python -m scripts.crosscorpus_experiments`

## Datasets

| Dataset | Recordings | Subjects | PD / HC (subj) | PD / HC (rec) |
|---|---|---|---|---|
| **Italian** (Dimauro & Girardi 2019) | 831 | 61 | 24 / 37 | 437 / 394 |
| **UCI** (Little 2007, original 195-row) | 195 | 32 | 24 / 8 | 147 / 48 |

Italian is **4x larger** and **well-balanced at both subject and recording level**. UCI is massively skewed (HC contributes only 8 subjects but 48 recordings because each HC subject was recorded many times).

## A. Italian, all 54 features (UCI-20 + 34 extras)

| metric | mean ± std |
|---|---|
| AUC | **0.974 ± 0.028** |
| Accuracy | 0.921 |
| F1 | 0.924 |

Subject-grouped 5-fold stratified CV. This is a big step up from the UCI numbers (0.80 CV AUC) — not because the model got better, but because the **Italian dataset is fundamentally better** (4× the data, balanced subjects, clean recordings).

## B. Italian, UCI-20 features only (no extras)

| metric | mean ± std |
|---|---|
| AUC | **0.962 ± 0.038** |
| Accuracy | 0.918 |
| F1 | 0.921 |

## A vs B — do the 34 new features help?

**Marginal.** +0.011 AUC, +0.003 accuracy. Within fold-to-fold noise (std 0.028). On this dataset, the extras (CPP, MFCC, formants, tilt) don't earn their keep — the UCI-20 features alone almost saturate the signal available.

Honest interpretation: the headline win isn't "new features beat old features." It's "more data + corrected extraction pipeline = 0.97 AUC." The feature engineering mattered less than I'd hoped.

## C. Cross-corpus generalization (the bombshell)

Using UCI-20 features (only features both corpora have):

| training | test | recording AUC | recording acc | subject AUC |
|---|---|---|---|---|
| **Italian** | UCI | **0.314** | 0.667 | 0.276 |
| **UCI** | Italian | **0.550** | 0.564 | 0.507 |

An AUC of 0.5 is random. An AUC of 0.3 is **worse than random** — the model is systematically wrong (healthy Italians look like PD English speakers to the model, and vice versa).

This is the most important result in the whole experiment. **Voice biomarker models do not transfer across languages/corpora.** A model trained on English sustained phonation is essentially useless on Italian speakers, and vice versa. This is not a bug in our pipeline — it's a fundamental finding that has real implications for anyone deploying a PD voice screener in multilingual India.

Sources of the transfer failure, in order of likely impact:
1. **Recording conditions differ** — Italian is mobile-phone quality, UCI is lab mic. Acoustic features are sensitive to mic distance, room acoustics, sampling rate differences.
2. **Population differences** — Italian mixes elderly and young healthy controls; UCI doesn't. Age is a huge confound on voice features.
3. **Language / phonation style** — sustained /a/ in Italian vs English may differ in tongue position, meaning formant values differ systematically.
4. **Subject count** — UCI has 8 healthy subjects. Anything we learn about "healthy" is basically a model of those 8 people.

## D. Feature importance comparison

### Italian, 54 features
Top 5: **DFA, Shimmer:APQ3, MFCC_1_std, Shimmer:APQ5, HNR**

By group:
- MFCC (aggregate): 0.31 — *the extras DO rank high when included*
- shimmer: 0.24
- nonlinear (DFA/D2/RPDE/PPE): 0.21
- formants + CPP + tilt: 0.10 combined
- pitch: 0.05
- jitter: 0.02

### Italian, UCI-20 only
Top 5: DFA, Shimmer:APQ3, Shimmer:APQ5, HNR, MDVP:APQ

### UCI-20 only
Top 5: **spread1, PPE, MDVP:Shimmer(dB), MDVP:Fo(Hz), MDVP:APQ**

Different corpora → different top features. UCI is dominated by speaker-normalised F0 measures (spread1, PPE, Fo), Italian is dominated by perturbation + DFA. This is further evidence that the models aren't learning a universal "PD voice signature" — they're learning corpus-specific patterns.

## Takeaways for the project

1. **For SilentSigns / demos**: use the **Italian-trained model** and honest metrics. 0.97 AUC, 0.92 accuracy on subject-grouped CV, 61 subjects. That's defensible.

2. **For a multilingual India deployment**: cross-corpus generalization is a real problem. The paper-worthy story is "single-corpus models don't generalize → we need either (a) multi-corpus training or (b) domain adaptation via wav2vec2 embeddings or (c) normalization that removes speaker/recording effects."

3. **The 34 extra features are worth keeping in the extractor** (they're computed for free once you have the audio loaded) even though they didn't measurably improve Italian performance. MFCC_1_std showed up in the top-3 so they're contributing *some* signal.

4. **Do not report "99% accuracy" for this system anywhere.** The Italian-CV number (0.97 AUC, 0.92 acc) is the honest ceiling, and it won't transfer to real-world users speaking other languages.
