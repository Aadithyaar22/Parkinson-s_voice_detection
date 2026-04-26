"""
Path B: productionize the joint UCI + Italian model.

Retrains the winning S1 strategy (naive concat + StandardScaler + XGBoost)
on ALL available data - no held-out set, full training for deployment -
then saves to models/ so app.py can load it.

Also tunes the decision threshold on subject-grouped OOF predictions from
CV on the training data, so the Flask app can classify at a sensible
threshold instead of the default 0.5.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from src.feature_extractor import FEATURE_NAMES as UCI_FULL

warnings.filterwarnings("ignore")

UCI_20 = [f for f in UCI_FULL if f not in ("Jitter:DDP", "Shimmer:DDA")]
MODELS_DIR = Path("models_joint")
MODELS_DIR.mkdir(exist_ok=True)


def load_combined() -> pd.DataFrame:
    """UCI + Italian with corpus-prefixed subject IDs."""
    uci = pd.read_csv("data/parkinsons_original.csv")
    uci["subject"] = "UCI_" + uci["name"].str.extract(r"(S\d+)")[0]
    uci["corpus"] = "UCI"

    it = pd.read_csv("data/italian_features.csv")
    it["subject"] = "IT_" + it["subject"].astype(str)
    it["corpus"] = "Italian"

    combined = pd.concat(
        [uci[["subject", "corpus", "status"] + UCI_20],
         it[["subject", "corpus", "status"] + UCI_20]],
        ignore_index=True,
    )
    return combined


def make_pipeline(pos_weight: float) -> Pipeline:
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("clf", xgb.XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.01, reg_lambda=1.0,
            scale_pos_weight=pos_weight,
            eval_metric="logloss", random_state=0, n_jobs=-1, verbosity=0,
        )),
    ])


def youden_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[int(np.argmax(tpr - fpr))])


def main():
    print("Loading combined corpus...")
    df = load_combined()
    X = df[UCI_20].astype(float)
    y = df["status"].astype(int).values
    groups = df["subject"].values
    corpora = df["corpus"].values

    pw = float(np.sum(y == 0) / max(np.sum(y == 1), 1))
    print(f"  {len(y)} recordings  |  {len(np.unique(groups))} subjects  "
          f"|  {np.sum(y==1)} PD, {np.sum(y==0)} HC  |  pos_weight={pw:.2f}")
    print(f"  UCI:     {(corpora=='UCI').sum()} recordings from "
          f"{len(set(df[df.corpus=='UCI'].subject))} subjects")
    print(f"  Italian: {(corpora=='Italian').sum()} recordings from "
          f"{len(set(df[df.corpus=='Italian'].subject))} subjects")

    # ---- Subject-grouped CV on combined data (to get OOF probs for threshold tuning)
    print("\nSubject-grouped 5-fold stratified CV on the combined dataset:")
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    oof = np.full(len(y), np.nan)
    fold_metrics = []
    fold_uci, fold_it = [], []
    for k, (tr, te) in enumerate(cv.split(X, y, groups)):
        pipe = make_pipeline(pw)
        pipe.fit(X.iloc[tr], y[tr])
        p = pipe.predict_proba(X.iloc[te])[:, 1]
        oof[te] = p
        pred = (p >= 0.5).astype(int)
        acc = accuracy_score(y[te], pred)
        auc = roc_auc_score(y[te], p)
        f1 = f1_score(y[te], pred)
        fold_metrics.append({"acc": acc, "auc": auc, "f1": f1})
        # Per-corpus AUC in each fold (for sanity)
        for c, store in [("UCI", fold_uci), ("Italian", fold_it)]:
            m = (corpora[te] == c)
            if m.sum() > 5 and len(set(y[te][m])) == 2:
                store.append(roc_auc_score(y[te][m], p[m]))
        print(f"  fold {k+1}: acc={acc:.3f}  auc={auc:.3f}  f1={f1:.3f}")

    summary = {
        "auc_mean": float(np.mean([m["auc"] for m in fold_metrics])),
        "auc_std": float(np.std([m["auc"] for m in fold_metrics])),
        "acc_mean": float(np.mean([m["acc"] for m in fold_metrics])),
        "acc_std": float(np.std([m["acc"] for m in fold_metrics])),
        "f1_mean": float(np.mean([m["f1"] for m in fold_metrics])),
        "per_corpus_fold_auc": {
            "UCI_mean": float(np.mean(fold_uci)) if fold_uci else None,
            "UCI_std": float(np.std(fold_uci)) if fold_uci else None,
            "Italian_mean": float(np.mean(fold_it)) if fold_it else None,
            "Italian_std": float(np.std(fold_it)) if fold_it else None,
        },
    }
    print(f"\nOverall CV:  AUC {summary['auc_mean']:.3f}+/-{summary['auc_std']:.3f}  "
          f"ACC {summary['acc_mean']:.3f}+/-{summary['acc_std']:.3f}")
    print(f"Per-corpus:  UCI AUC {summary['per_corpus_fold_auc']['UCI_mean']:.3f}  "
          f"Italian AUC {summary['per_corpus_fold_auc']['Italian_mean']:.3f}")

    # ---- Threshold on OOF
    thr = youden_threshold(y, oof)
    oof_pred = (oof >= thr).astype(int)
    acc_t = accuracy_score(y, oof_pred)
    f1_t = f1_score(y, oof_pred)
    print(f"\nTuned threshold (Youden on OOF): {thr:.3f}  "
          f"-> acc={acc_t:.3f}  f1={f1_t:.3f}")

    # ---- Fit deployment model on ALL data
    print("\nFitting deployment model on ALL data...")
    final_pipe = make_pipeline(pw)
    final_pipe.fit(X, y)

    joblib.dump(final_pipe, MODELS_DIR / "parkinsons_pipeline.joblib")
    joblib.dump(UCI_20, MODELS_DIR / "feature_names.pkl")

    report = {
        "chosen_model": "xgb_joint_uci_italian",
        "tuned_threshold": thr,
        "features_used": UCI_20,
        "dataset": {
            "n_recordings": int(len(y)),
            "n_subjects": int(len(np.unique(groups))),
            "uci_recordings": int((corpora == "UCI").sum()),
            "italian_recordings": int((corpora == "Italian").sum()),
        },
        "cv_summary": summary,
        "oof_on_full_training": {
            "auc": float(roc_auc_score(y, oof)),
            "acc_at_tuned_threshold": float(acc_t),
            "f1_at_tuned_threshold": float(f1_t),
            "confusion_matrix_at_tuned": confusion_matrix(y, oof_pred).tolist(),
        },
        "notes": [
            "Joint training on UCI (Little 2007) + Italian (Dimauro 2019).",
            "Subject-grouped stratified CV with corpus-prefixed subject IDs "
            "ensures no subject leakage.",
            "Held-out evaluation (separate script) showed this strategy lifts "
            "UCI cross-corpus AUC from 0.31 (Italian-only) to 0.76 (joint) "
            "while preserving Italian within-corpus AUC at 0.99.",
        ],
    }
    with open(MODELS_DIR / "training_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {MODELS_DIR/'parkinsons_pipeline.joblib'}")
    print(f"Saved: {MODELS_DIR/'training_report.json'}")


if __name__ == "__main__":
    main()
