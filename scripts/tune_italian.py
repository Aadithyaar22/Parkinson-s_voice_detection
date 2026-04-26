"""
Staged Italian-only tuning. Runs in chunks small enough to fit in
one command each. Saves params to disk so a rerun picks up where
it left off.

Usage:
  python -m scripts.tune_italian xgb        # ~3-4 min
  python -m scripts.tune_italian lgbm       # ~2-3 min
  python -m scripts.tune_italian rf         # ~2-3 min
  python -m scripts.tune_italian finalize   # outer CV + save model, ~3-5 min
"""
from __future__ import annotations

import json
import sys
import logging
import warnings
from pathlib import Path

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
    accuracy_score, f1_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

from src.feature_extractor import FEATURE_NAMES as UCI_FULL
from src.extra_features import EXTRA_FEATURE_NAMES

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger("lightgbm").setLevel(logging.ERROR)

UCI_20 = [f for f in UCI_FULL if f not in ("Jitter:DDP", "Shimmer:DDA")]
ALL_54 = UCI_20 + EXTRA_FEATURE_NAMES

OUT = Path("models_italian_tuned")
OUT.mkdir(exist_ok=True)
PARAMS = OUT / "params.json"


def load():
    df = pd.read_csv("data/italian_features.csv")
    return (df[ALL_54].astype(float),
            df["status"].astype(int).values,
            df["subject"].values)


def pw(y):
    return float(np.sum(y == 0) / max(np.sum(y == 1), 1))


def pipe(clf):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("clf", clf),
    ])


def inner_auc(p, X, y, g, n=3):
    cv = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=42)
    aucs = []
    for tr, te in cv.split(X, y, g):
        p.fit(X.iloc[tr], y[tr])
        aucs.append(roc_auc_score(y[te], p.predict_proba(X.iloc[te])[:, 1]))
    return float(np.mean(aucs))


def load_params():
    return json.loads(PARAMS.read_text()) if PARAMS.exists() else {}


def save_params(d):
    PARAMS.write_text(json.dumps(d, indent=2))


# ---------------------------------------------------------------------------
# Per-model tuners
# ---------------------------------------------------------------------------
def tune_xgb(X, y, g, n_trials=25):
    w = pw(y)
    def obj(t):
        clf = xgb.XGBClassifier(
            n_estimators=t.suggest_int("n_estimators", 200, 900, step=100),
            max_depth=t.suggest_int("max_depth", 3, 9),
            learning_rate=t.suggest_float("learning_rate", 0.01, 0.2, log=True),
            subsample=t.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=t.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_weight=t.suggest_int("min_child_weight", 1, 8),
            gamma=t.suggest_float("gamma", 1e-4, 2.0, log=True),
            reg_alpha=t.suggest_float("reg_alpha", 1e-4, 5, log=True),
            reg_lambda=t.suggest_float("reg_lambda", 1e-4, 5, log=True),
            scale_pos_weight=w, eval_metric="logloss",
            n_jobs=-1, random_state=0, verbosity=0,
        )
        return inner_auc(pipe(clf), X, y, g)
    s = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=0))
    s.optimize(obj, n_trials=n_trials, show_progress_bar=False,
               catch=(Exception,))
    return s.best_params, s.best_value


def tune_lgbm(X, y, g, n_trials=25):
    def obj(t):
        clf = lgb.LGBMClassifier(
            n_estimators=t.suggest_int("n_estimators", 200, 900, step=100),
            num_leaves=t.suggest_int("num_leaves", 8, 96),
            max_depth=t.suggest_int("max_depth", 3, 10),
            learning_rate=t.suggest_float("learning_rate", 0.01, 0.2, log=True),
            subsample=t.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=t.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_samples=t.suggest_int("min_child_samples", 5, 35),
            reg_alpha=t.suggest_float("reg_alpha", 1e-4, 5, log=True),
            reg_lambda=t.suggest_float("reg_lambda", 1e-4, 5, log=True),
            class_weight="balanced",
            n_jobs=-1, random_state=0, verbose=-1,
        )
        return inner_auc(pipe(clf), X, y, g)
    s = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=0))
    s.optimize(obj, n_trials=n_trials, show_progress_bar=False,
               catch=(Exception,))
    return s.best_params, s.best_value


def tune_rf(X, y, g, n_trials=25):
    def obj(t):
        clf = RandomForestClassifier(
            n_estimators=t.suggest_int("n_estimators", 300, 900, step=100),
            max_depth=t.suggest_int("max_depth", 3, 25),
            min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 10),
            min_samples_split=t.suggest_int("min_samples_split", 2, 15),
            max_features=t.suggest_categorical("max_features",
                                               ["sqrt", "log2", 0.3, 0.5, 0.8]),
            class_weight="balanced",
            n_jobs=-1, random_state=0,
        )
        return inner_auc(pipe(clf), X, y, g)
    s = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=0))
    s.optimize(obj, n_trials=n_trials, show_progress_bar=False,
               catch=(Exception,))
    return s.best_params, s.best_value


# ---------------------------------------------------------------------------
# Finalize: outer CV on all tuned models + ensembles, save winner
# ---------------------------------------------------------------------------
def build_tuned(P, w):
    return {
        "xgb": pipe(xgb.XGBClassifier(
            **P["xgb"], scale_pos_weight=w, eval_metric="logloss",
            n_jobs=-1, random_state=0, verbosity=0,
        )),
        "lgbm": pipe(lgb.LGBMClassifier(
            **P["lgbm"], class_weight="balanced",
            n_jobs=-1, random_state=0, verbose=-1,
        )),
        "rf": pipe(RandomForestClassifier(
            **P["rf"], class_weight="balanced",
            n_jobs=-1, random_state=0,
        )),
    }


def outer_cv(name, model, X, y, g, n=5):
    cv = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=0)
    aucs, accs, f1s = [], [], []
    oof = np.full(len(y), np.nan)
    for tr, te in cv.split(X, y, g):
        model.fit(X.iloc[tr], y[tr])
        p = model.predict_proba(X.iloc[te])[:, 1]
        oof[te] = p
        pred = (p >= 0.5).astype(int)
        aucs.append(roc_auc_score(y[te], p))
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred))
    m = {"auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
         "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
         "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s))}
    print(f"  {name:6s}  AUC {m['auc_mean']:.4f}+/-{m['auc_std']:.4f}  "
          f"ACC {m['acc_mean']:.4f}  F1 {m['f1_mean']:.4f}")
    return m, oof


def youden(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[int(np.argmax(tpr - fpr))])


def finalize():
    X, y, g = load()
    w = pw(y)
    P = load_params()
    missing = [m for m in ("xgb", "lgbm", "rf") if m not in P]
    if missing:
        sys.exit(f"Missing tuned params: {missing}. Run those first.")
    tuned = build_tuned(P, w)

    print("\nOuter 5-fold subject-grouped CV on tuned base models:")
    results = {}
    oofs = {}
    for n, p in tuned.items():
        m, oof = outer_cv(n, p, X, y, g)
        results[n] = m; oofs[n] = oof

    # Voting
    print("\nEnsembles:")
    vote = VotingClassifier(
        estimators=[(n, build_tuned(P, w)[n]) for n in ("xgb","lgbm","rf")],
        voting="soft", n_jobs=1)
    m_v, oof_v = outer_cv("vote", vote, X, y, g)
    results["vote"] = m_v; oofs["vote"] = oof_v

    stack = StackingClassifier(
        estimators=[(n, build_tuned(P, w)[n]) for n in ("xgb","lgbm","rf")],
        final_estimator=LogisticRegression(max_iter=3000, random_state=0),
        cv=3, n_jobs=1)
    m_s, oof_s = outer_cv("stack", stack, X, y, g)
    results["stack"] = m_s; oofs["stack"] = oof_s

    best = max(results, key=lambda k: results[k]["auc_mean"])
    print(f"\nBest: {best}  AUC {results[best]['auc_mean']:.4f}")

    thr = youden(y, oofs[best])
    acc_t = accuracy_score(y, (oofs[best] >= thr).astype(int))
    f1_t = f1_score(y, (oofs[best] >= thr).astype(int))
    print(f"Tuned threshold: {thr:.3f}  -> acc={acc_t:.4f}  f1={f1_t:.4f}")

    # Subject-level
    d = pd.DataFrame({"y": y, "p": oofs[best], "g": g})
    a = d.groupby("g").agg(y=("y","first"), pm=("p","mean"))
    subj_pred = (a["pm"] >= thr).astype(int)
    subj = {
        "n": int(len(a)),
        "auc": float(roc_auc_score(a["y"], a["pm"])),
        "acc": float(accuracy_score(a["y"], subj_pred)),
        "f1": float(f1_score(a["y"], subj_pred, zero_division=0)),
    }
    print(f"Subject-level: n={subj['n']}  auc={subj['auc']:.4f}  "
          f"acc={subj['acc']:.4f}  f1={subj['f1']:.4f}")

    # Refit winner on all data
    def rebuild():
        if best == "vote":
            return VotingClassifier(
                estimators=[(n, build_tuned(P, w)[n]) for n in ("xgb","lgbm","rf")],
                voting="soft", n_jobs=1)
        if best == "stack":
            return StackingClassifier(
                estimators=[(n, build_tuned(P, w)[n]) for n in ("xgb","lgbm","rf")],
                final_estimator=LogisticRegression(max_iter=3000, random_state=0),
                cv=3, n_jobs=1)
        return build_tuned(P, w)[best]

    final = rebuild()
    final.fit(X, y)
    joblib.dump(final, OUT / "parkinsons_pipeline.joblib")
    joblib.dump(ALL_54, OUT / "feature_names.pkl")

    report = {
        "chosen_model": f"{best}_italian_tuned",
        "tuned_threshold": thr,
        "features_used": ALL_54,
        "dataset": {
            "source": "Italian Parkinson's Voice (Dimauro 2019)",
            "n_recordings": int(len(y)),
            "n_subjects": int(len(np.unique(g))),
            "n_features": len(ALL_54),
        },
        "cv_summary": {k: results[best][k] for k in
                       ("auc_mean","auc_std","acc_mean","acc_std","f1_mean","f1_std")},
        "all_models_cv": results,
        "best_params": P,
        "subject_level_oof": subj,
    }
    with open(OUT / "training_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {OUT/'parkinsons_pipeline.joblib'}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        sys.exit("usage: tune_italian.py {xgb|lgbm|rf|finalize}")
    cmd = sys.argv[1]
    X, y, g = load()
    P = load_params()

    if cmd == "xgb":
        print("Tuning XGBoost (25 trials)...")
        best_p, best_v = tune_xgb(X, y, g)
        P["xgb"] = best_p; P["_xgb_inner_auc"] = best_v
        save_params(P)
        print(f"  inner-CV AUC: {best_v:.4f}")
        print(f"  saved to {PARAMS}")
    elif cmd == "lgbm":
        print("Tuning LightGBM (25 trials)...")
        best_p, best_v = tune_lgbm(X, y, g)
        P["lgbm"] = best_p; P["_lgbm_inner_auc"] = best_v
        save_params(P)
        print(f"  inner-CV AUC: {best_v:.4f}")
    elif cmd == "rf":
        print("Tuning RandomForest (25 trials)...")
        best_p, best_v = tune_rf(X, y, g)
        P["rf"] = best_p; P["_rf_inner_auc"] = best_v
        save_params(P)
        print(f"  inner-CV AUC: {best_v:.4f}")
    elif cmd == "finalize":
        finalize()
    else:
        sys.exit(f"unknown command: {cmd}")


if __name__ == "__main__":
    main()
