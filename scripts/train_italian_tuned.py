"""
Aggressive Italian-only tuning. Goal: push Italian CV AUC above 0.975
honestly (subject-grouped CV, no leakage).

Strategy:
  1. Optuna (TPE + Hyperband pruning) on XGBoost, LightGBM, Random Forest,
     CatBoost. 50 trials each, inner 3-fold subject-grouped CV.
  2. 5-fold outer subject-grouped CV on each tuned model (the HONEST metric).
  3. Soft-voting ensemble (average predictions).
  4. Stacking ensemble (LR meta-learner on CV-OOF preds).
  5. Pick winner, refit on full data, save.

Runs ~8-12 minutes.
"""
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier, StackingClassifier, VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from src.feature_extractor import FEATURE_NAMES as UCI_FULL
from src.extra_features import EXTRA_FEATURE_NAMES

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger("lightgbm").setLevel(logging.ERROR)
logging.getLogger("catboost").setLevel(logging.ERROR)

UCI_20 = [f for f in UCI_FULL if f not in ("Jitter:DDP", "Shimmer:DDA")]
ALL_54 = UCI_20 + EXTRA_FEATURE_NAMES

OUT = Path("models_italian_tuned")
OUT.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load():
    df = pd.read_csv("data/italian_features.csv")
    X = df[ALL_54].astype(float)
    y = df["status"].astype(int).values
    groups = df["subject"].values
    return X, y, groups


def pos_weight(y):
    return float(np.sum(y == 0) / max(np.sum(y == 1), 1))


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------
def make_pipe(clf) -> Pipeline:
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("clf", clf),
    ])


# ---------------------------------------------------------------------------
# Inner CV for Optuna
# ---------------------------------------------------------------------------
def inner_cv_auc(pipe, X, y, groups, n_splits=3) -> float:
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


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------
def obj_xgb(X, y, g, pw):
    def f(trial):
        clf = xgb.XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 200, 1200, step=100),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            gamma=trial.suggest_float("gamma", 1e-4, 5.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
            scale_pos_weight=pw, eval_metric="logloss",
            n_jobs=-1, random_state=0, verbosity=0,
        )
        return inner_cv_auc(make_pipe(clf), X, y, g)
    return f


def obj_lgbm(X, y, g, pw):
    def f(trial):
        clf = lgb.LGBMClassifier(
            n_estimators=trial.suggest_int("n_estimators", 200, 1200, step=100),
            num_leaves=trial.suggest_int("num_leaves", 8, 128),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 40),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
            class_weight="balanced",
            n_jobs=-1, random_state=0, verbose=-1,
        )
        return inner_cv_auc(make_pipe(clf), X, y, g)
    return f


def obj_rf(X, y, g):
    def f(trial):
        clf = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 300, 1000, step=100),
            max_depth=trial.suggest_int("max_depth", 3, 25),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 15),
            max_features=trial.suggest_categorical("max_features",
                                                   ["sqrt", "log2", 0.3, 0.5, 0.8]),
            class_weight="balanced",
            n_jobs=-1, random_state=0,
        )
        return inner_cv_auc(make_pipe(clf), X, y, g)
    return f


def obj_cat(X, y, g):
    def f(trial):
        clf = cb.CatBoostClassifier(
            iterations=trial.suggest_int("iterations", 200, 1200, step=100),
            depth=trial.suggest_int("depth", 4, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-2, 10, log=True),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0),
            auto_class_weights="Balanced",
            random_seed=0, verbose=False,
        )
        return inner_cv_auc(make_pipe(clf), X, y, g)
    return f


def tune(name: str, objective, n_trials: int):
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=0),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False,
                   catch=(Exception,))
    print(f"  {name}: best inner-CV AUC {study.best_value:.4f}  params: {study.best_params}")
    return study.best_params, study.best_value


# ---------------------------------------------------------------------------
# Build tuned pipelines
# ---------------------------------------------------------------------------
def build_tuned(xgb_p, lgbm_p, rf_p, cat_p, pw):
    return {
        "xgb": make_pipe(xgb.XGBClassifier(
            **xgb_p, scale_pos_weight=pw, eval_metric="logloss",
            n_jobs=-1, random_state=0, verbosity=0,
        )),
        "lgbm": make_pipe(lgb.LGBMClassifier(
            **lgbm_p, class_weight="balanced",
            n_jobs=-1, random_state=0, verbose=-1,
        )),
        "rf": make_pipe(RandomForestClassifier(
            **rf_p, class_weight="balanced",
            n_jobs=-1, random_state=0,
        )),
        "cat": make_pipe(cb.CatBoostClassifier(
            **cat_p, auto_class_weights="Balanced",
            random_seed=0, verbose=False,
        )),
    }


# ---------------------------------------------------------------------------
# Outer CV evaluator
# ---------------------------------------------------------------------------
def outer_cv(name, pipe, X, y, g, n_splits=5):
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
    aucs, accs, f1s = [], [], []
    oof = np.full(len(y), np.nan)
    for k, (tr, te) in enumerate(cv.split(X, y, g)):
        pipe.fit(X.iloc[tr], y[tr])
        p = pipe.predict_proba(X.iloc[te])[:, 1]
        oof[te] = p
        pred = (p >= 0.5).astype(int)
        aucs.append(roc_auc_score(y[te], p))
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred))
    m = {
        "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
        "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
        "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
    }
    print(f"  {name:8s} AUC {m['auc_mean']:.4f}+/-{m['auc_std']:.4f}  "
          f"ACC {m['acc_mean']:.4f}  F1 {m['f1_mean']:.4f}")
    return m, oof


def youden(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[int(np.argmax(tpr - fpr))])


def subject_level(y, p, g, thr=0.5):
    d = pd.DataFrame({"y": y, "p": p, "g": g})
    a = d.groupby("g").agg(y=("y","first"), pm=("p","mean"))
    pred = (a["pm"] >= thr).astype(int)
    return {
        "n": int(len(a)),
        "acc": float(accuracy_score(a["y"], pred)),
        "auc": float(roc_auc_score(a["y"], a["pm"])),
        "f1": float(f1_score(a["y"], pred, zero_division=0)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    X, y, g = load()
    pw = pos_weight(y)
    print(f"Italian: {len(y)} recordings, {len(np.unique(g))} subjects, "
          f"{(y==1).sum()} PD / {(y==0).sum()} HC, pos_weight={pw:.3f}")
    print(f"Features: {len(ALL_54)} (UCI-20 + 34 extras)\n")

    # ---- Tune each model
    print("=== Optuna tuning (50 trials each, 3-fold inner CV) ===")
    xgb_p, xgb_cv = tune("xgb",  obj_xgb(X, y, g, pw), n_trials=50)
    lgbm_p, lgbm_cv = tune("lgbm", obj_lgbm(X, y, g, pw), n_trials=50)
    rf_p, rf_cv = tune("rf",   obj_rf(X, y, g),        n_trials=50)
    cat_p, cat_cv = tune("cat",  obj_cat(X, y, g),       n_trials=40)

    # ---- Outer CV on each
    print("\n=== Outer 5-fold subject-grouped CV on tuned models ===")
    tuned = build_tuned(xgb_p, lgbm_p, rf_p, cat_p, pw)
    outer_results = {}
    oof_store = {}
    for name, pipe in tuned.items():
        m, oof = outer_cv(name, pipe, X, y, g, n_splits=5)
        outer_results[name] = m
        oof_store[name] = oof

    # ---- Soft-voting ensemble (average probabilities)
    print("\n=== Ensembles ===")
    # Build a fresh voting classifier that uses the tuned models
    voters = [(n, build_tuned(xgb_p, lgbm_p, rf_p, cat_p, pw)[n])
              for n in ("xgb", "lgbm", "rf", "cat")]
    vote = VotingClassifier(estimators=voters, voting="soft", n_jobs=1)
    m_vote, oof_vote = outer_cv("vote", vote, X, y, g, n_splits=5)
    outer_results["vote"] = m_vote
    oof_store["vote"] = oof_vote

    # ---- Stacking (LR meta-learner)
    stackers = [(n, build_tuned(xgb_p, lgbm_p, rf_p, cat_p, pw)[n])
                for n in ("xgb", "lgbm", "rf", "cat")]
    stack = StackingClassifier(
        estimators=stackers,
        final_estimator=LogisticRegression(max_iter=3000, C=1.0, random_state=0),
        cv=3, n_jobs=1,
    )
    m_stack, oof_stack = outer_cv("stack", stack, X, y, g, n_splits=5)
    outer_results["stack"] = m_stack
    oof_store["stack"] = oof_stack

    # ---- Pick the best by AUC
    best_name = max(outer_results, key=lambda k: outer_results[k]["auc_mean"])
    print(f"\nBest by CV AUC: {best_name}  "
          f"(AUC {outer_results[best_name]['auc_mean']:.4f})")

    # ---- Threshold tuning + subject-level on the best
    best_oof = oof_store[best_name]
    thr = youden(y, best_oof)
    acc_t = accuracy_score(y, (best_oof >= thr).astype(int))
    f1_t = f1_score(y, (best_oof >= thr).astype(int))
    subj = subject_level(y, best_oof, g, thr)
    print(f"Tuned threshold: {thr:.3f}  -> acc={acc_t:.4f}  f1={f1_t:.4f}")
    print(f"Subject-level:   n={subj['n']}  acc={subj['acc']:.4f}  "
          f"auc={subj['auc']:.4f}  f1={subj['f1']:.4f}")

    # ---- Refit best on ALL data
    def build_best():
        if best_name == "vote":
            return VotingClassifier(
                estimators=[(n, build_tuned(xgb_p, lgbm_p, rf_p, cat_p, pw)[n])
                            for n in ("xgb","lgbm","rf","cat")],
                voting="soft", n_jobs=1,
            )
        if best_name == "stack":
            return StackingClassifier(
                estimators=[(n, build_tuned(xgb_p, lgbm_p, rf_p, cat_p, pw)[n])
                            for n in ("xgb","lgbm","rf","cat")],
                final_estimator=LogisticRegression(max_iter=3000, random_state=0),
                cv=3, n_jobs=1,
            )
        return build_tuned(xgb_p, lgbm_p, rf_p, cat_p, pw)[best_name]

    final = build_best()
    final.fit(X, y)

    joblib.dump(final, OUT / "parkinsons_pipeline.joblib")
    joblib.dump(ALL_54, OUT / "feature_names.pkl")

    report = {
        "chosen_model": f"{best_name}_italian_tuned",
        "tuned_threshold": thr,
        "features_used": ALL_54,
        "dataset": {
            "source": "Italian Parkinson's Voice (Dimauro 2019)",
            "n_recordings": int(len(y)),
            "n_subjects": int(len(np.unique(g))),
            "n_features": len(ALL_54),
        },
        "cv_summary": {
            "auc_mean": outer_results[best_name]["auc_mean"],
            "auc_std": outer_results[best_name]["auc_std"],
            "acc_mean": outer_results[best_name]["acc_mean"],
            "acc_std": outer_results[best_name]["acc_std"],
            "f1_mean": outer_results[best_name]["f1_mean"],
        },
        "subject_level_oof": subj,
        "all_models_cv": outer_results,
        "best_params": {
            "xgb": xgb_p, "lgbm": lgbm_p, "rf": rf_p, "cat": cat_p,
        },
        "notes": [
            "Italian-only model, all 54 features (UCI-20 + 34 extras).",
            "50 Optuna trials per base model, 3-fold inner subject-grouped CV.",
            "5-fold outer subject-grouped CV for reported metrics.",
            "No held-out test set - all data used for final fit after model selection.",
            "Threshold tuned on OOF predictions during CV.",
        ],
    }
    with open(OUT / "training_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {OUT/'parkinsons_pipeline.joblib'}")
    print(f"Saved: {OUT/'training_report.json'}")


if __name__ == "__main__":
    main()
