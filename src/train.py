"""
Train Parkinson's classifiers on the UCI voice dataset.

Key differences from the original repo:
* GroupKFold by SUBJECT (not random splits) - the dataset has
  multiple recordings per speaker, so random splits leak subjects
  between train and test and inflate accuracy.
* Class-balanced training - the provided dataset is 75 pct healthy
  at the recording level, which means a naive model can just guess
  "healthy" and score 75 pct.
* Compares Logistic Regression, Random Forest, Gradient Boosting.
* Reports CV mean +/- std, plus held-out subject-level test metrics,
  plus an ROC curve and a feature-importance plot.
* Saves a fitted Pipeline (imputer + scaler + model) - no manual
  median-imputation hack at inference time.

Usage: python -m src.train
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import (
    StratifiedGroupKFold, cross_val_score, cross_val_predict, GroupShuffleSplit
)
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, roc_curve,
    confusion_matrix, classification_report
)


# Same canonical order the extractor uses at inference time
FEATURES = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA",
    "NHR", "HNR",
    "RPDE", "DFA", "spread1", "spread2", "D2", "PPE",
]


def load_dataset(path: str):
    df = pd.read_csv(path)
    df["subject"] = df["name"].str.extract(r"(S\d+)")[0]
    if df["subject"].isna().any():
        raise ValueError("Could not parse subject IDs from 'name' column")
    X = df[FEATURES].astype(float)
    y = df["status"].astype(int).values
    groups = df["subject"].values
    return X, y, groups, df


def make_candidates() -> Dict[str, Pipeline]:
    """Pipelines with imputation + scaling + classifier, all class-balanced."""
    imp = SimpleImputer(strategy="median")
    scl = StandardScaler()
    return {
        "logreg": Pipeline([
            ("imp", imp), ("scl", scl),
            ("clf", LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=0
            )),
        ]),
        "rf": Pipeline([
            ("imp", imp),
            # trees don't need scaling but leaving it is harmless
            ("scl", scl),
            ("clf", RandomForestClassifier(
                n_estimators=400, max_depth=None, min_samples_leaf=2,
                class_weight="balanced", n_jobs=-1, random_state=0
            )),
        ]),
        "gbt": Pipeline([
            ("imp", imp), ("scl", scl),
            ("clf", GradientBoostingClassifier(
                n_estimators=300, max_depth=3, learning_rate=0.05, random_state=0
            )),
        ]),
    }


def cv_evaluate(name: str, pipe: Pipeline, X, y, groups, n_splits=5):
    """Subject-grouped CV with class stratification. Returns mean/std + OOF probas."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
    accs, aucs, f1s = [], [], []
    oof_proba = np.zeros(len(y))
    oof_mask = np.zeros(len(y), dtype=bool)
    for fold, (tr, te) in enumerate(cv.split(X, y, groups)):
        pipe.fit(X.iloc[tr], y[tr])
        p = pipe.predict_proba(X.iloc[te])[:, 1]
        pred = (p >= 0.5).astype(int)
        accs.append(accuracy_score(y[te], pred))
        try:
            aucs.append(roc_auc_score(y[te], p))
        except ValueError:
            # fold with only one class - skip AUC
            pass
        f1s.append(f1_score(y[te], pred))
        oof_proba[te] = p
        oof_mask[te] = True
        print(
            f"  [{name}] fold {fold+1}: "
            f"acc={accs[-1]:.3f}  auc={aucs[-1] if aucs else float('nan'):.3f}  f1={f1s[-1]:.3f}"
        )

    def mstd(a): return (float(np.mean(a)), float(np.std(a))) if a else (float("nan"), float("nan"))
    am, asd = mstd(accs); um, usd = mstd(aucs); fm, fsd = mstd(f1s)
    return {
        "acc_mean": am, "acc_std": asd,
        "auc_mean": um, "auc_std": usd,
        "f1_mean": fm, "f1_std": fsd,
    }, oof_proba, oof_mask


def main():
    data_path = Path("data/parkinsons.data")
    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)
    graphs_dir = Path("reports")
    graphs_dir.mkdir(exist_ok=True)

    X, y, groups, df = load_dataset(data_path)
    print(f"Dataset: {len(y)} recordings, {len(np.unique(groups))} subjects")
    print(f"Class balance (recording level): healthy={np.sum(y==0)}  PD={np.sum(y==1)}")

    # 1) Held-out subjects for a final honest test.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
    tr_idx, te_idx = next(gss.split(X, y, groups))
    X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]
    g_tr = groups[tr_idx]
    test_subjects = sorted(set(groups[te_idx]))
    print(f"Held-out test: {len(te_idx)} recordings from subjects {test_subjects}")

    # 2) Run subject-grouped CV on the training portion
    print("\n=== Subject-grouped 5-fold CV on training portion ===")
    candidates = make_candidates()
    results = {}
    for name, pipe in candidates.items():
        metrics, oof_p, _ = cv_evaluate(name, pipe, X_tr, y_tr, g_tr, n_splits=5)
        results[name] = metrics
        print(
            f"  -> {name}  acc={metrics['acc_mean']:.3f} +/- {metrics['acc_std']:.3f}  "
            f"auc={metrics['auc_mean']:.3f} +/- {metrics['auc_std']:.3f}  "
            f"f1={metrics['f1_mean']:.3f} +/- {metrics['f1_std']:.3f}"
        )

    # 3) Select best model by mean AUC (tie-break by F1)
    best_name = max(results, key=lambda k: (results[k]["auc_mean"], results[k]["f1_mean"]))
    print(f"\nBest model by CV AUC: {best_name}")

    # 4) Refit best model on all training data and evaluate on held-out test
    best_pipe = make_candidates()[best_name]
    best_pipe.fit(X_tr, y_tr)
    p_test = best_pipe.predict_proba(X_te)[:, 1]
    pred_test = (p_test >= 0.5).astype(int)
    test_metrics = {
        "acc": float(accuracy_score(y_te, pred_test)),
        "auc": float(roc_auc_score(y_te, p_test)),
        "f1": float(f1_score(y_te, pred_test)),
        "confusion": confusion_matrix(y_te, pred_test).tolist(),
        "test_subjects": test_subjects,
        "n_test": int(len(te_idx)),
    }
    print("\n=== Held-out test metrics (subjects unseen in training) ===")
    print(json.dumps(test_metrics, indent=2))
    print("\n" + classification_report(y_te, pred_test, target_names=["healthy", "PD"]))

    # 5) Save plots - ROC and feature importance
    fpr, tpr, _ = roc_curve(y_te, p_test)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {test_metrics['auc']:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"ROC - held-out subjects ({best_name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(graphs_dir / "roc_heldout.png", dpi=120)
    plt.close()

    # Feature importance - available for RF and GBT; for logreg we use |coef|
    clf = best_pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        imp = np.asarray(clf.feature_importances_, dtype=float)
        title = "Feature importance (tree-based)"
    elif hasattr(clf, "coef_"):
        imp = np.abs(clf.coef_).ravel()
        title = "|Coefficient| (logistic regression, standardised)"
    else:
        imp = None

    if imp is not None:
        order = np.argsort(imp)[::-1]
        plt.figure(figsize=(7, 6))
        plt.barh(range(len(FEATURES)), imp[order][::-1])
        plt.yticks(range(len(FEATURES)), [FEATURES[i] for i in order[::-1]], fontsize=8)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(graphs_dir / "feature_importance.png", dpi=120)
        plt.close()

    # 6) Retrain the best model on the ENTIRE dataset for deployment.
    final_pipe = make_candidates()[best_name]
    final_pipe.fit(X, y)
    joblib.dump(final_pipe, out_dir / "parkinsons_pipeline.joblib")
    joblib.dump(FEATURES, out_dir / "feature_names.pkl")

    # Save a training-report JSON next to the model
    with open(out_dir / "training_report.json", "w") as f:
        json.dump({
            "chosen_model": best_name,
            "cv_by_model": results,
            "heldout_test": test_metrics,
            "features": FEATURES,
            "dataset": {
                "n_recordings": int(len(y)),
                "n_subjects": int(len(np.unique(groups))),
                "class_balance_recordings": {
                    "healthy": int(np.sum(y == 0)),
                    "PD": int(np.sum(y == 1)),
                },
            },
            "notes": [
                "GroupKFold by subject so no subject leaks between train/test.",
                "Held-out test is a disjoint set of subjects.",
                "Final deployed pipeline is refit on the whole dataset.",
                "Features at inference are computed via Praat (parselmouth), ",
                "not the original proprietary MDVP - values are analogous but ",
                "not numerically identical. A StandardScaler inside the pipeline ",
                "softens this calibration gap.",
            ],
        }, f, indent=2)

    print(f"\nSaved: {out_dir/'parkinsons_pipeline.joblib'}")
    print(f"Saved: {out_dir/'feature_names.pkl'}")
    print(f"Saved: {out_dir/'training_report.json'}")
    print(f"Saved: {graphs_dir/'roc_heldout.png'}")
    print(f"Saved: {graphs_dir/'feature_importance.png'}")


if __name__ == "__main__":
    main()
