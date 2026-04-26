"""
Train on the clean, original UCI Parkinson's CSV (195 rows, 31 subjects).

This is different from `src/train_v2.py` only in the dataset it loads -
same tuned XGB/LGBM/RF stacking pipeline, subject-grouped CV, Youden
threshold tuning, held-out subject test set.

Run: python -m scripts.train_on_uci_original
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the v2 training module's building blocks
from src.train_v2 import (
    FEATURES, DROP_FEATURES, build_tuned_pipelines,
    tune_rf, tune_xgb, tune_lgbm,
    outer_cv, best_threshold_youden, subject_level_metrics,
)
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report,
)
from sklearn.model_selection import GroupShuffleSplit
import joblib


CSV_PATH = Path("data/parkinsons_original.csv")
OUT_DIR = Path("models_original")
REPORTS_DIR = Path("reports_original")


def load():
    df = pd.read_csv(CSV_PATH)
    df["subject"] = df["name"].str.extract(r"(S\d+)")[0]
    X = df[FEATURES].astype(float)
    y = df["status"].astype(int).values
    groups = df["subject"].values
    return X, y, groups


def main():
    OUT_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    X, y, groups = load()
    pos_weight = float(np.sum(y == 0) / max(np.sum(y == 1), 1))
    print(f"Dataset: {len(y)} recordings  {len(np.unique(groups))} subjects  "
          f"pos_weight={pos_weight:.3f}")
    print(f"Features: {len(FEATURES)} (dropped collinear: {DROP_FEATURES})")

    # Held-out subjects
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
    tr, te = next(gss.split(X, y, groups))
    X_tr, X_te = X.iloc[tr], X.iloc[te]
    y_tr, y_te = y[tr], y[te]
    g_tr, g_te = groups[tr], groups[te]
    print(f"Held-out: {sorted(set(g_te))}  ({len(te)} recordings)\n")

    # Tune
    print("Tuning (15 trials each)...")
    rf_p, rf_cv = tune_rf(X_tr, y_tr, g_tr, n_trials=15)
    print(f"  RF   inner-CV AUC {rf_cv:.3f}")
    xgb_p, xgb_cv = tune_xgb(X_tr, y_tr, g_tr, n_trials=15)
    print(f"  XGB  inner-CV AUC {xgb_cv:.3f}")
    lgbm_p, lgbm_cv = tune_lgbm(X_tr, y_tr, g_tr, n_trials=15)
    print(f"  LGBM inner-CV AUC {lgbm_cv:.3f}")

    # Outer CV
    base = build_tuned_pipelines(rf_p, xgb_p, lgbm_p, pos_weight)
    cands = dict(base)
    outer = {}
    oofs = {}
    print("\nOuter 4-fold subject-grouped CV:")
    for name, pipe in cands.items():
        m, oof = outer_cv(pipe, X_tr, y_tr, g_tr, n_splits=4)
        outer[name] = m; oofs[name] = oof
        print(f"  {name:8s}  acc={m['acc_mean']:.3f}+/-{m['acc_std']:.3f}  "
              f"auc={m['auc_mean']:.3f}+/-{m['auc_std']:.3f}  "
              f"f1={m['f1_mean']:.3f}+/-{m['f1_std']:.3f}")

    # Stacking
    stack = StackingClassifier(
        estimators=[(n, build_tuned_pipelines(rf_p, xgb_p, lgbm_p, pos_weight)[n])
                    for n in ["rf", "xgb", "lgbm", "logreg"]],
        final_estimator=LogisticRegression(max_iter=3000, random_state=0),
        cv=3, n_jobs=1,
    )
    m_s, oof_s = outer_cv(stack, X_tr, y_tr, g_tr, n_splits=4)
    outer["stack"] = m_s; oofs["stack"] = oof_s; cands["stack"] = stack
    print(f"  stack    acc={m_s['acc_mean']:.3f}+/-{m_s['acc_std']:.3f}  "
          f"auc={m_s['auc_mean']:.3f}+/-{m_s['auc_std']:.3f}  "
          f"f1={m_s['f1_mean']:.3f}+/-{m_s['f1_std']:.3f}")

    # Pick best
    best = max(outer, key=lambda k: outer[k]["auc_mean"])
    thr = best_threshold_youden(y_tr, oofs[best])
    print(f"\nBest: {best}  tuned_threshold={thr:.3f}")

    # Fit on training portion, test on held-out
    pipe = cands[best]
    pipe.fit(X_tr, y_tr)
    p_te = pipe.predict_proba(X_te)[:, 1]
    pred_t = (p_te >= thr).astype(int)
    pred05 = (p_te >= 0.5).astype(int)

    held = {
        "auc": float(roc_auc_score(y_te, p_te)),
        "threshold_used": thr,
        "at_tuned": {
            "acc": float(accuracy_score(y_te, pred_t)),
            "f1": float(f1_score(y_te, pred_t)),
            "confusion": confusion_matrix(y_te, pred_t).tolist(),
        },
        "at_0.5": {
            "acc": float(accuracy_score(y_te, pred05)),
            "f1": float(f1_score(y_te, pred05)),
        },
        "subject_level": subject_level_metrics(y_te, p_te, g_te, thr),
    }
    print("\nHeld-out:", json.dumps(held, indent=2))
    print(classification_report(y_te, pred_t, target_names=["healthy", "PD"]))

    # Fit best on ALL data for deployment
    final = build_tuned_pipelines(rf_p, xgb_p, lgbm_p, pos_weight)
    final["stack"] = StackingClassifier(
        estimators=[(n, build_tuned_pipelines(rf_p, xgb_p, lgbm_p, pos_weight)[n])
                    for n in ["rf", "xgb", "lgbm", "logreg"]],
        final_estimator=LogisticRegression(max_iter=3000, random_state=0),
        cv=3, n_jobs=1,
    )
    final_pipe = final[best]
    final_pipe.fit(X, y)
    joblib.dump(final_pipe, OUT_DIR / "parkinsons_pipeline.joblib")
    joblib.dump(FEATURES, OUT_DIR / "feature_names.pkl")

    rep = {
        "chosen_model": best,
        "tuned_threshold": thr,
        "features_used": FEATURES,
        "outer_cv_by_model": outer,
        "heldout_test": held,
        "dataset": {
            "source": "UCI Parkinson's (original, Little 2007)",
            "n_recordings": int(len(y)),
            "n_subjects": int(len(np.unique(groups))),
        },
    }
    with open(OUT_DIR / "training_report.json", "w") as f:
        json.dump(rep, f, indent=2)
    print(f"\nSaved: {OUT_DIR/'parkinsons_pipeline.joblib'}")


if __name__ == "__main__":
    main()
