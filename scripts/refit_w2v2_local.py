"""
Quick fix for the "LogisticRegression object has no attribute 'multi_class'"
error.

What it does: loads data/italian_w2v2.csv, retrains the exact same
LogisticRegression + StandardScaler pipeline on your machine with
whatever scikit-learn version you have installed, saves it to
models/parkinsons_pipeline.joblib. Takes ~5 seconds.

Run:  python scripts/refit_w2v2_local.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

CSV = Path("data/italian_w2v2.csv")
OUT_DIR = Path("models")

def main():
    print(f"Loading {CSV}...")
    df = pd.read_csv(CSV)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    X = df[emb_cols].astype(np.float32)
    y = df["status"].astype(int).values
    groups = df["subject"].values
    print(f"  {len(y)} recordings, {len(np.unique(groups))} subjects, "
          f"{len(emb_cols)}-dim embeddings")

    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0, max_iter=3000, class_weight="balanced",
            solver="lbfgs", random_state=0,
        )),
    ])

    # Quick subject-grouped CV for OOF threshold tuning
    print("Running 5-fold subject-grouped CV for metrics + threshold...")
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    oof = np.full(len(y), np.nan)
    aucs, accs, f1s = [], [], []
    for tr, te in cv.split(X, y, groups):
        pipe.fit(X.iloc[tr], y[tr])
        p = pipe.predict_proba(X.iloc[te])[:, 1]
        oof[te] = p
        aucs.append(roc_auc_score(y[te], p))
        accs.append(accuracy_score(y[te], (p >= 0.5).astype(int)))
        f1s.append(f1_score(y[te], (p >= 0.5).astype(int)))
    print(f"  CV AUC {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
    print(f"  CV acc {np.mean(accs):.4f}")
    print(f"  CV F1  {np.mean(f1s):.4f}")

    fpr, tpr, thr_grid = roc_curve(y, oof)
    thr = float(thr_grid[int(np.argmax(tpr - fpr))])
    print(f"  Tuned threshold (Youden): {thr:.3f}")

    # Refit on all data
    pipe.fit(X, y)

    # Subject-level metrics
    d = pd.DataFrame({"y": y, "p": oof, "g": groups})
    a = d.groupby("g").agg(y=("y", "first"), pm=("p", "mean"))
    subj_pred = (a["pm"] >= thr).astype(int)
    subj = {
        "n": int(len(a)),
        "auc": float(roc_auc_score(a["y"], a["pm"])),
        "acc": float(accuracy_score(a["y"], subj_pred)),
        "f1": float(f1_score(a["y"], subj_pred, zero_division=0)),
    }
    print(f"  Subject-level: n={subj['n']}  "
          f"AUC {subj['auc']:.4f}  acc {subj['acc']:.4f}  F1 {subj['f1']:.4f}")

    # Save
    OUT_DIR.mkdir(exist_ok=True)
    joblib.dump(pipe, OUT_DIR / "parkinsons_pipeline.joblib")
    joblib.dump(emb_cols, OUT_DIR / "feature_names.pkl")

    import sklearn
    report = {
        "best_classifier": "logreg",
        "chosen_model": "logreg",
        "tuned_threshold": thr,
        "sklearn_version_used_for_training": sklearn.__version__,
        "italian_cv": {
            "logreg": {
                "auc_mean": float(np.mean(aucs)),
                "auc_std": float(np.std(aucs)),
                "acc_mean": float(np.mean(accs)),
                "acc_std": float(np.std(accs)),
                "f1_mean": float(np.mean(f1s)),
                "f1_std": float(np.std(f1s)),
            }
        },
        "italian_subject_level": subj,
        "note": "Retrained locally via scripts/refit_w2v2_local.py to match "
                "your installed scikit-learn version.",
    }
    with open(OUT_DIR / "training_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved: {OUT_DIR/'parkinsons_pipeline.joblib'}")
    print(f"Saved: {OUT_DIR/'feature_names.pkl'}")
    print(f"Saved: {OUT_DIR/'training_report.json'}")
    print(f"\nRestart the Flask app: python app.py")


if __name__ == "__main__":
    main()
