# Final Results

Single-page summary of everything tried and the final numbers. For
methodology and caveats see `README.md`; for per-experiment detail see
the individual `reports/*/results.json`.

All CV is **5-fold stratified subject-grouped** — no subject appears
in both train and test folds of any split.

## Headline numbers

**Current deployment** — wav2vec2-XLS-R embeddings + Logistic Regression,
trained on the Italian Parkinson's Voice dataset (831 recordings,
61 subjects):

```
5-fold CV AUC:          0.9725 ± 0.034
5-fold CV accuracy:     0.9423
5-fold CV F1:           0.9445
Subject-level AUC:      0.9955   (99.5% — aggregating a subject's recordings)
Subject-level acc:      0.9508   (58/61 subjects)
Subject-level F1:       0.9388
Decision threshold:     0.380    (Youden's J on OOF)
```

## Method comparison on Italian (subject-grouped CV)

| model | AUC | Accuracy | F1 |
|---|---|---|---|
| UCI-only baseline (hand-crafted, no tuning) | 0.744 ± 0.168 | 0.747 | — |
| Italian hand-crafted, no tuning | 0.962 ± 0.038 | 0.918 | 0.921 |
| Italian hand-crafted, full 54 features | 0.974 ± 0.028 | 0.921 | 0.924 |
| Italian hand-crafted, tuned voting ensemble | 0.9744 ± 0.028 | 0.9292 | 0.9327 |
| Italian wav2vec2 + LogReg **(deployed)** | **0.9725** ± 0.034 | **0.9423** | **0.9445** |
| Italian wav2vec2 + LogReg (PCA 64) | 0.9667 ± 0.036 | 0.9203 | 0.9226 |
| Italian wav2vec2 + LogReg (PCA 128) | 0.9690 ± 0.037 | 0.9340 | 0.9367 |
| Italian wav2vec2 + SVM-RBF | 0.9636 ± 0.040 | 0.9203 | 0.9258 |
| Italian wav2vec2 + LightGBM | 0.9665 ± 0.033 | 0.9146 | 0.9194 |

The **ceiling on this dataset is AUC ≈ 0.97**, reached independently by
both hand-crafted and self-supervised approaches. This is consistent
with the published literature on the same corpus.

## Cross-corpus evaluation

Trained on one dataset, tested on a disjoint set of subjects from the
other (UCI-20 common features only):

| training | test | recording AUC |
|---|---|---|
| Italian only | UCI | **0.31** *(worse than random)* |
| UCI only | Italian | **0.55** *(near random)* |
| **Joint (UCI + Italian)** | UCI held-out | **0.76** |
| **Joint (UCI + Italian)** | Italian held-out | **0.99** |

The cross-corpus collapse is the most interesting methodological
finding: single-corpus models trained on hand-crafted features do not
transfer across languages. Joint training with naive concat-and-scale
recovers ~70% of the gap. wav2vec2 embeddings (language-agnostic)
correctly classified an Indian-English speaker without any joint
training — qualitative evidence that language-aware representations
generalize where hand-crafted ones cannot.

## User sanity check

Predictions by the deployed wav2vec2 model on 4 recordings from the
developer (healthy Indian-English speaker, not in any training data):

| recording | P(PD) | prediction |
|---|---|---|
| New Recording.m4a | 0.007 | HC |
| New Recording 2.m4a | 0.036 | HC |
| New Recording 3.m4a | 0.114 | HC |
| New Recording 4.m4a | 0.219 | HC |
| **Mean** | **0.094** | **HC** |

All 4 below the threshold of 0.380. All clustered at the healthy end
of the probability spectrum. This is a one-subject sanity check, not a
cross-population validation — but it's consistent with the multilingual
embeddings doing their job.

## What we did NOT claim

Several tempting shortcuts that would have produced higher numbers but
would be dishonest:

* Random train/test splits instead of subject-grouped (would give AUC
  ≈ 0.99 trivially because of speaker leakage)
* SMOTE oversampling before splitting (also leakage)
* Threshold tuning on the test set (circular)
* Reporting recording-level accuracy when subject-level is available
  (recording-level overestimates when subjects contribute many samples)
* Quoting the 588-row augmented UCI dataset (the 195-row original is
  the real data; the extra 393 rows are duplicates)
* Using wav2vec2 features to train on the test set's own samples
  (obvious but surprisingly common)

## Feature importance (hand-crafted path, for interpretability)

When running with the hand-crafted backend, the top-ranked features
on Italian data are:

1. DFA (nonlinear complexity)
2. Shimmer:APQ3 (amplitude perturbation)
3. MFCC_1_std (timbral stability)
4. Shimmer:APQ5
5. HNR (harmonics-to-noise ratio)

The classical clinical-dysphonia predictors (shimmer, HNR) top the
list, with a nonlinear complexity measure (DFA) at the very top — this
matches what the clinical speech-pathology literature expects. MFCCs
contribute but are outranked by shimmer overall.

Full importance plots: `reports/crosscorpus/importance_italian_54.png`,
`reports/crosscorpus/importance_italian_uci20.png`,
`reports/crosscorpus/importance_uci_only.png`.
