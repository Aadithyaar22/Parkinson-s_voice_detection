"""
Cross-corpus experiments on Italian Parkinson's Voice + UCI (Little 2007).

Answers four questions:
  A. How well does our pipeline work on Italian data, subject-grouped?
  B. Do the 34 EXTRA features (CPP, MFCCs, formants, tilt) actually help
     on top of the UCI-22 set?
  C. Does a model trained on Italian generalise to English (UCI)?
     Does the reverse hold?
  D. Which features carry the most signal on each corpus?

Outputs a JSON report + feature importance plots.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import (
    GroupShuffleSplit, StratifiedGroupKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from src.feature_extractor import FEATURE_NAMES as UCI_FULL
from src.extra_features import EXTRA_FEATURE_NAMES

warnings.filterwarnings("ignore")

# The 20 features we actually train on (DDP and DDA dropped - collinear)
UCI_20 = [f for f in UCI_FULL if f not in ("Jitter:DDP", "Shimmer:DDA")]
ALL_56 = UCI_FULL + EXTRA_FEATURE_NAMES
ALL_54 = UCI_20 + EXTRA_FEATURE_NAMES  # 20 + 34

OUT = Path("reports/crosscorpus")
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def load_italian() -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    df = pd.read_csv("data/italian_features.csv")
    # Use ALL 54 training features (UCI-20 + 34 extras)
    X_full = df[ALL_54].astype(float)
    X_uci = df[UCI_20].astype(float)
    y = df["status"].astype(int).values
    g = df["subject"].values
    return X_full, X_uci, y, g


def load_uci() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = pd.read_csv("data/parkinsons_original.csv")
    df["subject"] = df["name"].str.extract(r"(S\d+)")[0]
    X = df[UCI_20].astype(float)
    y = df["status"].astype(int).values
    g = df["subject"].values
    return X, y, g


# ---------------------------------------------------------------------------
# Model factory - use a consistent tuned-ish XGBoost so comparisons are fair
# ---------------------------------------------------------------------------
def make_pipe(pos_weight: float = 1.0) -> Pipeline:
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


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def subject_grouped_cv(X, y, groups, pos_weight=1.0, n_splits=5, seed=0) -> Dict:
    """Subject-grouped stratified CV, returns mean +/- std of acc/auc/f1."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs, aucs, f1s = [], [], []
    for tr, te in cv.split(X, y, groups):
        pipe = make_pipe(pos_weight)
        pipe.fit(X.iloc[tr], y[tr])
        p = pipe.predict_proba(X.iloc[te])[:, 1]
        pred = (p >= 0.5).astype(int)
        accs.append(accuracy_score(y[te], pred))
        try:
            aucs.append(roc_auc_score(y[te], p))
        except ValueError:
            pass
        f1s.append(f1_score(y[te], pred))
    return {
        "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
        "auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
        "auc_std": float(np.std(aucs)) if aucs else float("nan"),
        "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
    }


def fit_and_score(X_tr, y_tr, X_te, y_te, pos_weight=1.0) -> Dict:
    pipe = make_pipe(pos_weight)
    pipe.fit(X_tr, y_tr)
    p = pipe.predict_proba(X_te)[:, 1]
    pred = (p >= 0.5).astype(int)
    return {
        "acc": float(accuracy_score(y_te, pred)),
        "auc": float(roc_auc_score(y_te, p)),
        "f1": float(f1_score(y_te, pred)),
        "confusion": confusion_matrix(y_te, pred).tolist(),
        "proba": p.tolist(),
    }


def subject_level_metrics(y_true, p, groups, thr=0.5) -> Dict:
    d = pd.DataFrame({"y": y_true, "p": p, "g": groups})
    agg = d.groupby("g").agg(y=("y", "first"), pm=("p", "mean"))
    pred = (agg["pm"] >= thr).astype(int)
    return {
        "n_subjects": int(len(agg)),
        "acc": float(accuracy_score(agg["y"], pred)),
        "auc": float(roc_auc_score(agg["y"], agg["pm"])),
        "f1": float(f1_score(agg["y"], pred)),
    }


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def exp_A_italian_full(X_full, y, g) -> Dict:
    """Italian trained on all 54 features (UCI-20 + 34 extras)."""
    print("\n=== A. Italian, all 54 features, subject-grouped 5-fold CV ===")
    pw = float(np.sum(y == 0) / max(np.sum(y == 1), 1))
    m = subject_grouped_cv(X_full, y, g, pos_weight=pw)
    print(f"  acc={m['acc_mean']:.3f}+/-{m['acc_std']:.3f}  "
          f"auc={m['auc_mean']:.3f}+/-{m['auc_std']:.3f}  "
          f"f1={m['f1_mean']:.3f}+/-{m['f1_std']:.3f}")
    return m


def exp_B_italian_uci_only(X_uci, y, g) -> Dict:
    """Italian with only the UCI-20 features - does the extra 34 help?"""
    print("\n=== B. Italian, UCI-20 features only, 5-fold CV ===")
    pw = float(np.sum(y == 0) / max(np.sum(y == 1), 1))
    m = subject_grouped_cv(X_uci, y, g, pos_weight=pw)
    print(f"  acc={m['acc_mean']:.3f}+/-{m['acc_std']:.3f}  "
          f"auc={m['auc_mean']:.3f}+/-{m['auc_std']:.3f}  "
          f"f1={m['f1_mean']:.3f}+/-{m['f1_std']:.3f}")
    return m


def exp_C_crosscorpus(X_it_uci, y_it, g_it, X_uci, y_uci, g_uci) -> Dict:
    """Train-on-A test-on-B, both directions, UCI-20 features only."""
    print("\n=== C. Cross-corpus (UCI-20 features only) ===")
    # Train Italian, test UCI
    pw_it = float(np.sum(y_it == 0) / max(np.sum(y_it == 1), 1))
    m_it2uci = fit_and_score(X_it_uci, y_it, X_uci, y_uci, pos_weight=pw_it)
    m_it2uci_subj = subject_level_metrics(
        y_uci, np.asarray(m_it2uci["proba"]), g_uci
    )
    print(f"  train=Italian -> test=UCI  "
          f"rec: acc={m_it2uci['acc']:.3f} auc={m_it2uci['auc']:.3f} f1={m_it2uci['f1']:.3f}  "
          f"|  subj: acc={m_it2uci_subj['acc']:.3f} auc={m_it2uci_subj['auc']:.3f}")

    # Train UCI, test Italian
    pw_uci = float(np.sum(y_uci == 0) / max(np.sum(y_uci == 1), 1))
    m_uci2it = fit_and_score(X_uci, y_uci, X_it_uci, y_it, pos_weight=pw_uci)
    m_uci2it_subj = subject_level_metrics(
        y_it, np.asarray(m_uci2it["proba"]), g_it
    )
    print(f"  train=UCI -> test=Italian  "
          f"rec: acc={m_uci2it['acc']:.3f} auc={m_uci2it['auc']:.3f} f1={m_uci2it['f1']:.3f}  "
          f"|  subj: acc={m_uci2it_subj['acc']:.3f} auc={m_uci2it_subj['auc']:.3f}")

    def strip_proba(d):
        d = dict(d); d.pop("proba", None); return d
    return {
        "italian_to_uci": {**strip_proba(m_it2uci), "subject_level": m_it2uci_subj},
        "uci_to_italian": {**strip_proba(m_uci2it), "subject_level": m_uci2it_subj},
    }


def exp_D_feature_importance(X_full, y, g, save_name: str) -> Dict:
    """Fit on the full Italian data, report XGB importance for each feature."""
    print(f"\n=== D. Feature importance ({save_name}) ===")
    pw = float(np.sum(y == 0) / max(np.sum(y == 1), 1))
    pipe = make_pipe(pw)
    pipe.fit(X_full, y)
    clf = pipe.named_steps["clf"]
    imp = np.asarray(clf.feature_importances_, dtype=float)

    cols = list(X_full.columns)
    imp_pairs = sorted(zip(cols, imp), key=lambda kv: -kv[1])

    # Tag each feature by group so we can sum by group too
    def group_of(name):
        if name in UCI_20 or name in UCI_FULL:
            if name.startswith("MDVP:F") and "Hz" in name:
                return "pitch"
            if "Jitter" in name:
                return "jitter"
            if "Shimmer" in name or "APQ" in name:
                return "shimmer"
            if name in ("HNR", "NHR"):
                return "noise"
            if name in ("RPDE", "DFA", "PPE", "D2", "spread1", "spread2"):
                return "nonlinear"
        if name == "CPP":
            return "extra:CPP"
        if name.startswith("MFCC"):
            return "extra:MFCC"
        if name.startswith("F") and ("_mean" in name or "_bw" in name):
            return "extra:formant"
        if name == "spectral_tilt":
            return "extra:tilt"
        return "other"

    per_group: Dict[str, float] = {}
    for name, v in zip(cols, imp):
        per_group[group_of(name)] = per_group.get(group_of(name), 0.0) + float(v)

    print("  Top 15 features:")
    for name, v in imp_pairs[:15]:
        print(f"    {name:22s}  {v:.4f}")
    print("\n  Importance by group:")
    for grp, tot in sorted(per_group.items(), key=lambda kv: -kv[1]):
        print(f"    {grp:20s}  {tot:.4f}")

    # Plot - top 20 features only
    top = imp_pairs[:20][::-1]
    plt.figure(figsize=(7, 6))
    plt.barh(range(len(top)), [v for _, v in top])
    plt.yticks(range(len(top)), [n for n, _ in top], fontsize=8)
    plt.title(f"Top 20 feature importances — {save_name}")
    plt.tight_layout()
    plt.savefig(OUT / f"importance_{save_name}.png", dpi=120)
    plt.close()
    return {"top20": imp_pairs[:20], "by_group": per_group}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading datasets...")
    X_it_full, X_it_uci, y_it, g_it = load_italian()
    X_uci, y_uci, g_uci = load_uci()
    print(f"  Italian: {len(y_it)} recs / {len(np.unique(g_it))} subjects "
          f"({(y_it==1).sum()} PD / {(y_it==0).sum()} HC)")
    print(f"  UCI:     {len(y_uci)} recs / {len(np.unique(g_uci))} subjects "
          f"({(y_uci==1).sum()} PD / {(y_uci==0).sum()} HC)")

    results = {}
    results["A_italian_54feats_cv"] = exp_A_italian_full(X_it_full, y_it, g_it)
    results["B_italian_uci20_cv"] = exp_B_italian_uci_only(X_it_uci, y_it, g_it)

    # Did the extras help?
    d_auc = results["A_italian_54feats_cv"]["auc_mean"] - results["B_italian_uci20_cv"]["auc_mean"]
    d_acc = results["A_italian_54feats_cv"]["acc_mean"] - results["B_italian_uci20_cv"]["acc_mean"]
    print(f"\n  Extras help by: AUC +{d_auc:.3f}  ACC +{d_acc:.3f}")

    results["C_crosscorpus"] = exp_C_crosscorpus(
        X_it_uci, y_it, g_it, X_uci, y_uci, g_uci
    )
    results["D_importance_italian_full"] = exp_D_feature_importance(
        X_it_full, y_it, g_it, "italian_54"
    )
    results["D_importance_italian_uci20"] = exp_D_feature_importance(
        X_it_uci, y_it, g_it, "italian_uci20"
    )
    results["D_importance_uci20_only"] = exp_D_feature_importance(
        X_uci, y_uci, g_uci, "uci_only"
    )

    # Strip numpy / tuple stuff for JSON
    def json_safe(o):
        if isinstance(o, dict):
            return {k: json_safe(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [json_safe(v) for v in o]
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        return o
    with open(OUT / "results.json", "w") as f:
        json.dump(json_safe(results), f, indent=2)
    print(f"\nSaved: {OUT/'results.json'}")
    print(f"Saved: {OUT}/importance_*.png")


if __name__ == "__main__":
    main()
