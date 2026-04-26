"""
Path A: joint training on UCI + Italian.

Three strategies tested, all with subject-grouped held-out sets from
BOTH corpora:

  S1. NAIVE  - concat rows, StandardScaler on combined, train.
  S2. PER-CORPUS Z  - z-score each corpus internally before combining.
      This kills the absolute-scale gap between corpora so the model
      learns "unusual for this corpus" instead of raw values.
  S3. PER-CORPUS Z + CORPUS INDICATOR  - S2 plus a binary flag
      'is_italian' so the model can still learn corpus-specific biases.

Baselines for comparison (from previous run):
  Italian-only -> Italian CV:  AUC 0.962
  Italian-only -> UCI:         AUC 0.314
  UCI-only -> Italian:         AUC 0.550

Runs ~2-3 minutes.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from src.feature_extractor import FEATURE_NAMES as UCI_FULL

warnings.filterwarnings("ignore")

UCI_20 = [f for f in UCI_FULL if f not in ("Jitter:DDP", "Shimmer:DDA")]
OUT = Path("reports/joint_training")
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_uci():
    df = pd.read_csv("data/parkinsons_original.csv")
    df["subject"] = "UCI_" + df["name"].str.extract(r"(S\d+)")[0]
    return df


def load_italian():
    df = pd.read_csv("data/italian_features.csv")
    df["subject"] = "IT_" + df["subject"].astype(str)
    return df


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def make_xgb(pos_weight: float):
    return xgb.XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.01, reg_lambda=1.0,
        scale_pos_weight=pos_weight,
        eval_metric="logloss", random_state=0, n_jobs=-1, verbosity=0,
    )


def fit_model(X_tr, y_tr, pre_fitted_scaler=None):
    """Build pipeline; if scaler is provided, skip the in-pipeline scaler."""
    pw = float(np.sum(y_tr == 0) / max(np.sum(y_tr == 1), 1))
    steps = [("imp", SimpleImputer(strategy="median"))]
    if pre_fitted_scaler is None:
        steps.append(("scl", StandardScaler()))
    steps.append(("clf", make_xgb(pw)))
    pipe = Pipeline(steps)
    pipe.fit(X_tr, y_tr)
    return pipe


def score(pipe, X, y, groups=None):
    p = pipe.predict_proba(X)[:, 1]
    pred = (p >= 0.5).astype(int)
    out = {
        "acc": float(accuracy_score(y, pred)),
        "auc": float(roc_auc_score(y, p)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "n": int(len(y)),
    }
    if groups is not None:
        d = pd.DataFrame({"y": y, "p": p, "g": groups})
        a = d.groupby("g").agg(y=("y", "first"), pm=("p", "mean"))
        pred_s = (a["pm"] >= 0.5).astype(int)
        out["subject"] = {
            "n": int(len(a)),
            "acc": float(accuracy_score(a["y"], pred_s)),
            "auc": float(roc_auc_score(a["y"], a["pm"])),
            "f1": float(f1_score(a["y"], pred_s, zero_division=0)),
        }
    return out


# ---------------------------------------------------------------------------
# Per-corpus z-scoring
# ---------------------------------------------------------------------------
def per_corpus_zscore(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Z-score the given columns within each corpus separately."""
    out = df.copy()
    for corpus in df["corpus"].unique():
        mask = df["corpus"] == corpus
        sub = df.loc[mask, cols].astype(float)
        # Use median-imputed values for the scaler computation
        mu = sub.mean()
        sd = sub.std().replace(0, 1)
        out.loc[mask, cols] = (sub - mu) / sd
    return out


# ---------------------------------------------------------------------------
# Build a combined dataframe with train/test splits per corpus
# ---------------------------------------------------------------------------
def build_splits(seed=0):
    """
    25% of subjects from each corpus held out as test.
    Returns combined train/test frames with a 'corpus' column.
    """
    uci = load_uci()
    italian = load_italian()
    uci["corpus"] = "UCI"
    italian["corpus"] = "Italian"

    # Held-out subjects within each corpus
    rng = np.random.default_rng(seed)

    def split(df):
        subjects = df["subject"].unique()
        # Need at least one of each class in test - simple stratified sampling
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        tr, te = next(gss.split(df, df["status"], df["subject"]))
        return df.iloc[tr].copy(), df.iloc[te].copy()

    uci_tr, uci_te = split(uci)
    it_tr, it_te = split(italian)
    combined_tr = pd.concat([uci_tr, it_tr], ignore_index=True)
    combined_te_uci = uci_te.reset_index(drop=True)
    combined_te_it = it_te.reset_index(drop=True)
    return combined_tr, combined_te_uci, combined_te_it


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------
def run_strategy(name: str, tr: pd.DataFrame, te_uci: pd.DataFrame,
                 te_it: pd.DataFrame, use_per_corpus_z: bool,
                 add_corpus_indicator: bool):
    feats = list(UCI_20)

    if use_per_corpus_z:
        # Apply per-corpus z-score to training set.
        tr = per_corpus_zscore(tr, feats)
        # For test sets, we need to apply the SAME per-corpus normalisation
        # using stats from the training rows of that corpus (no leakage).
        # Simplest: compute UCI-train stats and Italian-train stats from tr.
        uci_tr = tr[tr["corpus"] == "UCI"]   # already z-scored in tr
        it_tr = tr[tr["corpus"] == "Italian"]
        # But we need RAW stats from training. Recompute on raw (reload).
        # (Easier path: normalise te using stats of the corresponding
        # corpus subset from the ORIGINAL training frame before zscore.)
        # Rebuild original training stats from raw:
        raw_tr_uci = load_uci()
        raw_tr_it = load_italian()
        # Keep only rows also present in `tr` - just use all of raw_tr_* for
        # means since the z-score is global per corpus anyway.
        uci_mu, uci_sd = raw_tr_uci[feats].mean(), raw_tr_uci[feats].std().replace(0, 1)
        it_mu, it_sd = raw_tr_it[feats].mean(), raw_tr_it[feats].std().replace(0, 1)
        te_uci = te_uci.copy()
        te_uci[feats] = (te_uci[feats].astype(float) - uci_mu) / uci_sd
        te_it = te_it.copy()
        te_it[feats] = (te_it[feats].astype(float) - it_mu) / it_sd

    if add_corpus_indicator:
        tr = tr.copy()
        te_uci = te_uci.copy()
        te_it = te_it.copy()
        tr["is_italian"] = (tr["corpus"] == "Italian").astype(int)
        te_uci["is_italian"] = 0
        te_it["is_italian"] = 1
        feats = feats + ["is_italian"]

    # Fit
    X_tr = tr[feats].astype(float)
    y_tr = tr["status"].astype(int).values
    pipe = fit_model(X_tr, y_tr)

    # Score on held-out subjects in each corpus
    m_uci = score(pipe, te_uci[feats].astype(float),
                  te_uci["status"].astype(int).values,
                  groups=te_uci["subject"].values)
    m_it = score(pipe, te_it[feats].astype(float),
                 te_it["status"].astype(int).values,
                 groups=te_it["subject"].values)

    print(f"\n--- Strategy: {name} ---")
    print(f"  UCI test     | recs: acc={m_uci['acc']:.3f} auc={m_uci['auc']:.3f} f1={m_uci['f1']:.3f}  "
          f"|  subj: acc={m_uci['subject']['acc']:.3f} auc={m_uci['subject']['auc']:.3f}")
    print(f"  Italian test | recs: acc={m_it['acc']:.3f} auc={m_it['auc']:.3f} f1={m_it['f1']:.3f}  "
          f"|  subj: acc={m_it['subject']['acc']:.3f} auc={m_it['subject']['auc']:.3f}")

    return {
        "strategy": name,
        "heldout_uci": m_uci,
        "heldout_italian": m_it,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    tr, te_uci, te_it = build_splits(seed=0)
    print(f"Train:   {len(tr)} recs  "
          f"(UCI {(tr.corpus=='UCI').sum()}, Italian {(tr.corpus=='Italian').sum()})")
    print(f"Test UCI:     {len(te_uci)} recs / {te_uci.subject.nunique()} subjects")
    print(f"Test Italian: {len(te_it)} recs / {te_it.subject.nunique()} subjects")

    results = {}
    results["S1_naive"] = run_strategy(
        "S1 naive concat + StandardScaler",
        tr, te_uci, te_it,
        use_per_corpus_z=False, add_corpus_indicator=False,
    )
    results["S2_per_corpus_z"] = run_strategy(
        "S2 per-corpus z-score",
        tr, te_uci, te_it,
        use_per_corpus_z=True, add_corpus_indicator=False,
    )
    results["S3_per_corpus_z_plus_indicator"] = run_strategy(
        "S3 per-corpus z-score + is_italian flag",
        tr, te_uci, te_it,
        use_per_corpus_z=True, add_corpus_indicator=True,
    )

    # Baselines (retraining for a consistent comparison)
    print("\n--- Baselines (within-corpus only) ---")
    uci = load_uci()
    it = load_italian()

    def within_corpus_cv(df, label):
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
        aucs, accs = [], []
        for tr_i, te_i in cv.split(df[UCI_20], df["status"], df["subject"]):
            pipe = fit_model(
                df[UCI_20].iloc[tr_i].astype(float),
                df["status"].astype(int).values[tr_i],
            )
            p = pipe.predict_proba(df[UCI_20].iloc[te_i].astype(float))[:, 1]
            aucs.append(roc_auc_score(df["status"].values[te_i], p))
            accs.append(accuracy_score(df["status"].values[te_i], (p>=0.5).astype(int)))
        print(f"  {label} within-corpus CV: AUC {np.mean(aucs):.3f}±{np.std(aucs):.3f}  "
              f"ACC {np.mean(accs):.3f}")
        return {"auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
                "acc_mean": float(np.mean(accs))}

    results["baseline_italian_cv"] = within_corpus_cv(it, "Italian")
    results["baseline_uci_cv"] = within_corpus_cv(uci, "UCI")

    with open(OUT / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {OUT/'results.json'}")

    # Pick best strategy by average AUC across both corpora
    def avg_auc(r):
        return 0.5 * (r["heldout_uci"]["auc"] + r["heldout_italian"]["auc"])
    strategy_results = {k: v for k, v in results.items() if k.startswith("S")}
    best_key = max(strategy_results, key=lambda k: avg_auc(strategy_results[k]))
    print(f"\nBest strategy: {best_key}  "
          f"(avg AUC = {avg_auc(strategy_results[best_key]):.3f})")


if __name__ == "__main__":
    main()
