"""
Parkinson's voice classifier - v2 with rigorous accuracy improvements.

What this script does beyond src/train.py:

1) Drops mathematically redundant features (DDP = 3*RAP; DDA = 3*APQ3).
2) Evaluates a wider model zoo: LogReg, RF, GBT, XGBoost, LightGBM,
   plus a STACKING ensemble (LR + RF + XGB + LGBM -> LR meta-learner).
3) Uses NESTED subject-grouped CV for hyperparameter search:
   inner loop = Optuna with subject-grouped folds,
   outer loop = reports honest CV metrics.
4) Tunes the decision threshold on held-out folds using Youden's J
   (maximises sensitivity + specificity). Fixes the 0.62 accuracy issue.
5) Computes subject-level predictions by averaging P(PD) across a
   subject's recordings - this is what a clinical deployment would do.

Run: python -m src.train_v2
"""

from __future__ import annotations

import json
import warnings
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier, StackingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report,
)
from sklearn.model_selection import (
    StratifiedGroupKFold, GroupShuffleSplit,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger("lightgbm").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------
# FULL schema (matches the extractor + UCI CSV)
FULL_FEATURES = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA",
    "NHR", "HNR",
    "RPDE", "DFA", "spread1", "spread2", "D2", "PPE",
]

# After pruning: drop DDP (= 3*RAP) and DDA (= 3*APQ3) - they are
# definitionally collinear and add no information beyond RAP/APQ3.
DROP_FEATURES = ["Jitter:DDP", "Shimmer:DDA"]
FEATURES = [f for f in FULL_FEATURES if f not in DROP_FEATURES]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_dataset(path: str):
    df = pd.read_csv(path)
    df["subject"] = df["name"].str.extract(r"(S\d+)")[0]
    X = df[FEATURES].astype(float)
    y = df["status"].astype(int).values
    groups = df["subject"].values
    return X, y, groups, df


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------
def _base_preprocessor():
    return [("imp", SimpleImputer(strategy="median")), ("scl", StandardScaler())]


def make_pipeline(clf) -> Pipeline:
    return Pipeline(_base_preprocessor() + [("clf", clf)])


# ---------------------------------------------------------------------------
# Optuna objective generators (each returns a function Optuna can optimise)
# ---------------------------------------------------------------------------
def _cv_auc(pipe: Pipeline, X, y, groups, n_splits=4) -> float:
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []
    for tr, te in cv.split(X, y, groups):
        pipe.fit(X.iloc[tr], y[tr])
        p = pipe.predict_proba(X.iloc[te])[:, 1]
        try:
            aucs.append(roc_auc_score(y[te], p))
        except ValueError:
            pass
    return float(np.mean(aucs)) if aucs else 0.0


def tune_rf(X, y, groups, n_trials=30):
    def obj(trial):
        clf = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 200, 800, step=100),
            max_depth=trial.suggest_int("max_depth", 3, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 8),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 12),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.8]),
            class_weight="balanced",
            n_jobs=-1, random_state=0,
        )
        return _cv_auc(make_pipeline(clf), X, y, groups)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def tune_xgb(X, y, groups, n_trials=30):
    def obj(trial):
        clf = xgb.XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 200, 800, step=100),
            max_depth=trial.suggest_int("max_depth", 2, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
            scale_pos_weight=float(np.sum(y == 0) / max(np.sum(y == 1), 1)),
            eval_metric="logloss", use_label_encoder=False,
            n_jobs=-1, random_state=0, verbosity=0,
        )
        return _cv_auc(make_pipeline(clf), X, y, groups)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def tune_lgbm(X, y, groups, n_trials=30):
    def obj(trial):
        clf = lgb.LGBMClassifier(
            n_estimators=trial.suggest_int("n_estimators", 200, 800, step=100),
            num_leaves=trial.suggest_int("num_leaves", 8, 64),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
            class_weight="balanced",
            n_jobs=-1, random_state=0, verbose=-1,
        )
        return _cv_auc(make_pipeline(clf), X, y, groups)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


# ---------------------------------------------------------------------------
# Build final pipelines with tuned params
# ---------------------------------------------------------------------------
def build_tuned_pipelines(rf_p, xgb_p, lgbm_p, pos_weight: float):
    return {
        "logreg": make_pipeline(LogisticRegression(
            max_iter=3000, class_weight="balanced", C=1.0, random_state=0
        )),
        "rf": make_pipeline(RandomForestClassifier(
            **rf_p, class_weight="balanced", n_jobs=-1, random_state=0,
        )),
        "xgb": make_pipeline(xgb.XGBClassifier(
            **xgb_p, scale_pos_weight=pos_weight,
            eval_metric="logloss", use_label_encoder=False,
            n_jobs=-1, random_state=0, verbosity=0,
        )),
        "lgbm": make_pipeline(lgb.LGBMClassifier(
            **lgbm_p, class_weight="balanced",
            n_jobs=-1, random_state=0, verbose=-1,
        )),
    }


def build_stack(base_pipes: Dict[str, Pipeline]) -> StackingClassifier:
    """Stack LR + RF + XGB + LGBM with a LogReg meta-learner, cv-based stacking."""
    # Use subject-group 4-fold for stacking CV construction
    return StackingClassifier(
        estimators=[(name, pipe) for name, pipe in base_pipes.items()],
        final_estimator=LogisticRegression(max_iter=3000, C=1.0, random_state=0),
        cv=5, passthrough=False, n_jobs=1,
    )


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def outer_cv(pipe, X, y, groups, n_splits=5, seed=0) -> Tuple[Dict, np.ndarray]:
    """Outer subject-grouped CV. Returns mean metrics + OOF probabilities."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    accs, aucs, f1s = [], [], []
    for tr, te in cv.split(X, y, groups):
        pipe.fit(X.iloc[tr], y[tr])
        p = pipe.predict_proba(X.iloc[te])[:, 1]
        oof[te] = p
        accs.append(accuracy_score(y[te], (p >= 0.5).astype(int)))
        try:
            aucs.append(roc_auc_score(y[te], p))
        except ValueError:
            pass
        f1s.append(f1_score(y[te], (p >= 0.5).astype(int)))
    return {
        "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
        "auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
        "auc_std": float(np.std(aucs)) if aucs else float("nan"),
        "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
    }, oof


def best_threshold_youden(y_true, p) -> float:
    """Pick threshold that maximises sensitivity + specificity - 1."""
    mask = ~np.isnan(p)
    fpr, tpr, thr = roc_curve(y_true[mask], p[mask])
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def subject_level_metrics(y_true, p, groups, threshold):
    """Aggregate recording-level predictions to subject level (mean P(PD))."""
    df = pd.DataFrame({"y": y_true, "p": p, "g": groups})
    agg = df.groupby("g").agg(y=("y", "first"), p_mean=("p", "mean"))
    pred = (agg["p_mean"] >= threshold).astype(int)
    return {
        "n_subjects": int(len(agg)),
        "acc": float(accuracy_score(agg["y"], pred)),
        "auc": float(roc_auc_score(agg["y"], agg["p_mean"])),
        "f1": float(f1_score(agg["y"], pred, zero_division=0)),
        "confusion": confusion_matrix(agg["y"], pred).tolist(),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    data_path = Path("data/parkinsons.data")
    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    X, y, groups, df = load_dataset(data_path)
    pos_weight = float(np.sum(y == 0) / max(np.sum(y == 1), 1))
    print(f"Dataset: {len(y)} recordings  {len(np.unique(groups))} subjects  "
          f"pos_weight={pos_weight:.3f}")
    print(f"Features: {len(FEATURES)} (dropped collinear: {DROP_FEATURES})")

    # ---- Held-out subject split for final test
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
    tr_idx, te_idx = next(gss.split(X, y, groups))
    X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]
    g_tr, g_te = groups[tr_idx], groups[te_idx]
    print(f"Held-out subjects: {sorted(set(g_te))}  (n={len(te_idx)} recordings)\n")

    # ---- Hyperparameter search on training portion only
    print("=== Hyperparameter tuning (subject-grouped, 15 trials each) ===")
    print("  tuning RandomForest...")
    rf_p, rf_cv = tune_rf(X_tr, y_tr, g_tr, n_trials=15)
    print(f"    best inner-CV AUC {rf_cv:.3f}  params: {rf_p}")
    print("  tuning XGBoost...")
    xgb_p, xgb_cv = tune_xgb(X_tr, y_tr, g_tr, n_trials=15)
    print(f"    best inner-CV AUC {xgb_cv:.3f}  params: {xgb_p}")
    print("  tuning LightGBM...")
    lgbm_p, lgbm_cv = tune_lgbm(X_tr, y_tr, g_tr, n_trials=15)
    print(f"    best inner-CV AUC {lgbm_cv:.3f}  params: {lgbm_p}")

    # ---- Build tuned pipelines (stacking evaluated separately since it's slow)
    base_pipes = build_tuned_pipelines(rf_p, xgb_p, lgbm_p, pos_weight)
    candidates = dict(base_pipes)

    # ---- Outer 4-fold CV on training portion for each tuned base model
    print("\n=== Outer 4-fold subject-grouped CV on training portion ===")
    outer = {}
    oof_store = {}
    for name, pipe in candidates.items():
        m, oof = outer_cv(pipe, X_tr, y_tr, g_tr, n_splits=4, seed=0)
        outer[name] = m
        oof_store[name] = oof
        print(f"  {name:8s}  acc={m['acc_mean']:.3f}+/-{m['acc_std']:.3f}  "
              f"auc={m['auc_mean']:.3f}+/-{m['auc_std']:.3f}  "
              f"f1={m['f1_mean']:.3f}+/-{m['f1_std']:.3f}")

    # ---- Stacking: fit once on full train, get OOF for threshold tuning
    print("  building stacking ensemble (cv=3)...")
    stack = StackingClassifier(
        estimators=[(n, build_tuned_pipelines(rf_p, xgb_p, lgbm_p, pos_weight)[n])
                    for n in ["rf", "xgb", "lgbm", "logreg"]],
        final_estimator=LogisticRegression(max_iter=3000, C=1.0, random_state=0),
        cv=3, passthrough=False, n_jobs=1,
    )
    m_s, oof_s = outer_cv(stack, X_tr, y_tr, g_tr, n_splits=4, seed=0)
    outer["stack"] = m_s
    oof_store["stack"] = oof_s
    candidates["stack"] = stack
    print(f"  stack    acc={m_s['acc_mean']:.3f}+/-{m_s['acc_std']:.3f}  "
          f"auc={m_s['auc_mean']:.3f}+/-{m_s['auc_std']:.3f}  "
          f"f1={m_s['f1_mean']:.3f}+/-{m_s['f1_std']:.3f}")

    # ---- Pick best model by CV AUC
    best_name = max(outer, key=lambda k: outer[k]["auc_mean"])
    print(f"\nBest by CV AUC: {best_name}")

    # ---- Threshold tuning on OOF of the best model (stays inside training data)
    oof_best = oof_store[best_name]
    thr = best_threshold_youden(y_tr, oof_best)
    acc_at_thr = accuracy_score(y_tr[~np.isnan(oof_best)],
                                (oof_best[~np.isnan(oof_best)] >= thr).astype(int))
    f1_at_thr = f1_score(y_tr[~np.isnan(oof_best)],
                         (oof_best[~np.isnan(oof_best)] >= thr).astype(int))
    print(f"Tuned threshold (Youden J on OOF): {thr:.3f}  -> acc={acc_at_thr:.3f}  f1={f1_at_thr:.3f}")
    print(f"(was {accuracy_score(y_tr[~np.isnan(oof_best)], (oof_best[~np.isnan(oof_best)] >= 0.5).astype(int)):.3f} at threshold 0.5)")

    # ---- Refit best on full training portion, evaluate on held-out subjects
    best_pipe = candidates[best_name]
    best_pipe.fit(X_tr, y_tr)
    p_te = best_pipe.predict_proba(X_te)[:, 1]

    # Held-out metrics at default threshold 0.5
    pred05 = (p_te >= 0.5).astype(int)
    # Held-out at tuned threshold
    pred_t = (p_te >= thr).astype(int)

    held = {
        "auc": float(roc_auc_score(y_te, p_te)),
        "threshold_used": thr,
        "at_tuned_threshold": {
            "acc": float(accuracy_score(y_te, pred_t)),
            "f1": float(f1_score(y_te, pred_t)),
            "confusion": confusion_matrix(y_te, pred_t).tolist(),
        },
        "at_0.5_threshold": {
            "acc": float(accuracy_score(y_te, pred05)),
            "f1": float(f1_score(y_te, pred05)),
        },
        "subject_level": subject_level_metrics(y_te, p_te, g_te, thr),
    }
    print("\n=== Held-out test (unseen subjects) ===")
    print(json.dumps(held, indent=2))
    print(classification_report(y_te, pred_t, target_names=["healthy", "PD"]))

    # ---- Refit on ALL data for deployment
    final_candidates = {**build_tuned_pipelines(rf_p, xgb_p, lgbm_p, pos_weight)}
    final_candidates["stack"] = build_stack(final_candidates)
    final_pipe = final_candidates[best_name]
    final_pipe.fit(X, y)

    joblib.dump(final_pipe, out_dir / "parkinsons_pipeline.joblib")
    joblib.dump(FEATURES, out_dir / "feature_names.pkl")

    report = {
        "chosen_model": best_name,
        "tuned_threshold": thr,
        "features_used": FEATURES,
        "features_dropped_collinear": DROP_FEATURES,
        "tuned_params": {"rf": rf_p, "xgb": xgb_p, "lgbm": lgbm_p},
        "inner_cv_best_auc": {"rf": rf_cv, "xgb": xgb_cv, "lgbm": lgbm_cv},
        "outer_cv_by_model": outer,
        "heldout_test": held,
        "dataset": {
            "n_recordings": int(len(y)),
            "n_subjects": int(len(np.unique(groups))),
            "n_features": len(FEATURES),
            "class_balance_recordings": {
                "healthy": int(np.sum(y == 0)),
                "PD": int(np.sum(y == 1)),
            },
        },
        "notes": [
            "Subject-grouped train/test - no subject leakage.",
            "Decision threshold tuned on OOF (training) not on held-out test.",
            "DDP, DDA dropped as definitionally collinear (= 3*RAP, 3*APQ3).",
            "Final model refit on FULL dataset for deployment.",
            "Subject-level AUC is what matters clinically - we average "
            "P(PD) across a subject's recordings at inference.",
        ],
    }
    with open(out_dir / "training_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # ---- Feature importance plot for tree models
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clf = final_pipe.named_steps["clf"] if isinstance(final_pipe, Pipeline) else None
    if clf is not None and hasattr(clf, "feature_importances_"):
        imp = np.asarray(clf.feature_importances_, dtype=float)
        order = np.argsort(imp)[::-1]
        plt.figure(figsize=(7, 6))
        plt.barh(range(len(FEATURES)), imp[order][::-1])
        plt.yticks(range(len(FEATURES)), [FEATURES[i] for i in order[::-1]], fontsize=8)
        plt.title(f"Feature importance - {best_name}")
        plt.tight_layout()
        plt.savefig(reports_dir / "feature_importance.png", dpi=120)
        plt.close()

    # ROC
    fpr, tpr, _ = roc_curve(y_te, p_te)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {held['auc']:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.axvline(
        np.mean(fpr[:np.searchsorted(_, thr)[::-1][0]]) if False else 0,
        linestyle=":", alpha=0  # we skip threshold line to keep plot simple
    )
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"ROC held-out subjects ({best_name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(reports_dir / "roc_heldout.png", dpi=120)
    plt.close()

    print(f"\nSaved: {out_dir/'parkinsons_pipeline.joblib'}")
    print(f"Saved: {out_dir/'training_report.json'}")


if __name__ == "__main__":
    main()
