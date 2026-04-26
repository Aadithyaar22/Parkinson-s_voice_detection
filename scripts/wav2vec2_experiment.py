"""
Full wav2vec2 evaluation pipeline.

1. Load Italian wav2vec2 embeddings (1024-dim).
2. Train 3 classifiers (LogReg, LightGBM, SVM-RBF) with subject-grouped CV.
3. Compare results against hand-crafted features baseline.
4. Pick best classifier, refit on all Italian data.
5. Run inference on Aadithya's own recordings -> report what the model
   says about each.
6. Save deployment artefacts.

This is the honest end of the accuracy push. If w2v2 doesn't beat the
hand-crafted model here, we've shown the ceiling is real.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import lightgbm as lgb

warnings.filterwarnings("ignore")

OUT = Path("models_wav2vec2")
OUT.mkdir(exist_ok=True)
REPORT = Path("reports/wav2vec2")
REPORT.mkdir(parents=True, exist_ok=True)


def load_italian_emb():
    df = pd.read_csv("data/italian_w2v2.csv")
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    X = df[emb_cols].astype(np.float32)
    y = df["status"].astype(int).values
    g = df["subject"].values
    return X, y, g, emb_cols, df


def pw(y):
    return float(np.sum(y == 0) / max(np.sum(y == 1), 1))


def make_logreg():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0, max_iter=3000, class_weight="balanced",
            solver="lbfgs", random_state=0,
        )),
    ])


def make_logreg_pca(n_components=64):
    """Reduce 1024 -> 64 then LR. Often beats raw high-dim LR on small datasets."""
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=0)),
        ("clf", LogisticRegression(
            C=1.0, max_iter=3000, class_weight="balanced",
            solver="lbfgs", random_state=0,
        )),
    ])


def make_svm_rbf():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("clf", SVC(
            C=1.0, kernel="rbf", gamma="scale",
            probability=True, class_weight="balanced",
            random_state=0,
        )),
    ])


def make_lgbm(pos_weight):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        # No scaler - trees don't need it
        ("clf", lgb.LGBMClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.5,  # lower for 1024-dim
            reg_alpha=0.1, reg_lambda=0.1,
            class_weight="balanced",
            n_jobs=-1, random_state=0, verbose=-1,
        )),
    ])


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
    print(f"  {name:14s}  AUC {m['auc_mean']:.4f}+/-{m['auc_std']:.4f}  "
          f"ACC {m['acc_mean']:.4f}  F1 {m['f1_mean']:.4f}")
    return m, oof


def youden(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[int(np.argmax(tpr - fpr))])


def subject_level(y, p, g, thr=0.5):
    d = pd.DataFrame({"y": y, "p": p, "g": g})
    a = d.groupby("g").agg(y=("y", "first"), pm=("p", "mean"))
    pred = (a["pm"] >= thr).astype(int)
    return {
        "n": int(len(a)),
        "auc": float(roc_auc_score(a["y"], a["pm"])),
        "acc": float(accuracy_score(a["y"], pred)),
        "f1": float(f1_score(a["y"], pred, zero_division=0)),
    }


# ---------------------------------------------------------------------------
def main():
    print("Loading Italian wav2vec2 embeddings...")
    X, y, g, emb_cols, df = load_italian_emb()
    print(f"  {len(y)} recordings  {len(np.unique(g))} subjects  "
          f"{(y==1).sum()} PD / {(y==0).sum()} HC")
    print(f"  embedding dim: {len(emb_cols)}")

    # ---- Train multiple classifiers
    print("\n=== Subject-grouped 5-fold CV on wav2vec2 embeddings ===")
    w = pw(y)
    candidates = {
        "logreg": make_logreg(),
        "logreg_pca64": make_logreg_pca(n_components=64),
        "logreg_pca128": make_logreg_pca(n_components=128),
        "svm_rbf": make_svm_rbf(),
        "lgbm": make_lgbm(w),
    }
    results = {}
    oofs = {}
    for name, pipe in candidates.items():
        m, oof = outer_cv(name, pipe, X, y, g)
        results[name] = m
        oofs[name] = oof

    # ---- Pick best
    best = max(results, key=lambda k: results[k]["auc_mean"])
    print(f"\nBest: {best}  AUC {results[best]['auc_mean']:.4f}")

    # ---- Compare to hand-crafted baseline
    print("\n=== Comparison with hand-crafted features (from earlier) ===")
    print(f"  hand-crafted (tuned voting)  AUC 0.9744  ACC 0.9292  F1 0.9327")
    print(f"  wav2vec2 (best: {best})       "
          f"AUC {results[best]['auc_mean']:.4f}  "
          f"ACC {results[best]['acc_mean']:.4f}  "
          f"F1 {results[best]['f1_mean']:.4f}")
    delta_auc = results[best]["auc_mean"] - 0.9744
    delta_acc = results[best]["acc_mean"] - 0.9292
    print(f"  delta (wav2vec2 - handcrafted):  "
          f"AUC {delta_auc:+.4f}  ACC {delta_acc:+.4f}")

    # ---- Threshold + subject-level
    thr = youden(y, oofs[best])
    subj = subject_level(y, oofs[best], g, thr)
    print(f"\nTuned threshold: {thr:.3f}")
    print(f"Subject-level (n={subj['n']} subjects):  "
          f"AUC {subj['auc']:.4f}  ACC {subj['acc']:.4f}  F1 {subj['f1']:.4f}")

    # ---- Refit on ALL Italian data for deployment
    def rebuild_best():
        if best == "logreg": return make_logreg()
        if best == "logreg_pca64": return make_logreg_pca(64)
        if best == "logreg_pca128": return make_logreg_pca(128)
        if best == "svm_rbf": return make_svm_rbf()
        if best == "lgbm": return make_lgbm(w)

    final = rebuild_best()
    final.fit(X, y)
    joblib.dump(final, OUT / "parkinsons_pipeline.joblib")
    joblib.dump(emb_cols, OUT / "feature_names.pkl")

    # ---- Inference on user's own recordings
    print("\n=== Predicting on user recordings (my_test_w2v2.csv) ===")
    my = pd.read_csv("data/my_test_w2v2.csv")
    my_X = my[emb_cols].astype(np.float32)
    my_probs = final.predict_proba(my_X)[:, 1]
    for (_, row), p in zip(my.iterrows(), my_probs):
        pred = "PD" if p >= thr else "HC"
        label_line = f"  {row['name']:30s}  P(PD) = {p:.3f}  -> {pred}"
        print(label_line)

    # Subject-level aggregate across user's recordings
    user_mean_p = float(my_probs.mean())
    user_pred = "PD" if user_mean_p >= thr else "HC"
    print(f"\n  USER AVERAGE P(PD) = {user_mean_p:.3f}  ->  {user_pred}")
    print(f"  (threshold: {thr:.3f}; user_mean_p > thr means predicted PD)")

    # ---- Save report
    report = {
        "best_classifier": best,
        "tuned_threshold": thr,
        "italian_cv": results,
        "italian_subject_level": subj,
        "comparison_with_handcrafted": {
            "handcrafted_voting_auc": 0.9744,
            "handcrafted_voting_acc": 0.9292,
            "wav2vec2_auc": results[best]["auc_mean"],
            "wav2vec2_acc": results[best]["acc_mean"],
            "auc_delta": delta_auc,
            "acc_delta": delta_acc,
        },
        "user_predictions": [
            {"file": row["name"], "p_pd": float(p),
             "prediction": "PD" if p >= thr else "HC"}
            for (_, row), p in zip(my.iterrows(), my_probs)
        ],
        "user_mean_p_pd": user_mean_p,
        "user_aggregate_prediction": user_pred,
    }
    with open(OUT / "training_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    with open(REPORT / "results.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nSaved: {OUT/'parkinsons_pipeline.joblib'}")
    print(f"Saved: {OUT/'training_report.json'}")
    print(f"Saved: {REPORT/'results.json'}")


if __name__ == "__main__":
    main()
