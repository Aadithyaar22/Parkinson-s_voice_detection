"""
SHAP explainability analysis on the Italian hand-crafted tuned ensemble.
Produces:
  reports/shap/shap_summary.png       — beeswarm plot (paper Figure)
  reports/shap/shap_bar.png           — mean |SHAP| bar chart
  reports/shap/shap_values.csv        — raw SHAP values for all samples
  reports/shap/shap_report.json       — top features + clinical comparison

Run from pva2/:
  python scripts/shap_analysis.py
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import shap

warnings.filterwarnings("ignore")

OUT = Path("reports/shap")
OUT.mkdir(parents=True, exist_ok=True)

# ── Clinical literature reference (what neurologists say matters) ─────────────
CLINICAL_RANK = {
    "MDVP:Jitter(%)":   "High — pitch instability is a primary PD marker",
    "MDVP:Shimmer":     "High — amplitude instability, correlates with tremor",
    "HNR":              "High — reduced harmonicity indicates vocal fold dysfunction",
    "NHR":              "High — inverse of HNR; elevated in PD",
    "DFA":              "High — long-range vocal correlations disrupted in PD",
    "RPDE":             "Medium — nonlinear complexity, validated in Little 2007",
    "PPE":              "Medium — pitch entropy, validated in Little 2007",
    "MDVP:APQ":         "Medium — amplitude perturbation quotient",
    "Shimmer:APQ3":     "Medium — short-term amplitude variation",
    "spread1":          "Medium — nonlinear frequency spread",
    "D2":               "Lower — correlation dimension, computationally sensitive",
    "spread2":          "Lower — secondary nonlinear measure",
}


def load_model_and_data():
    # Load the hand-crafted tuned model (54 features)
    model_path = Path("models_italian_tuned/parkinsons_pipeline.joblib")
    feat_path  = Path("models_italian_tuned/feature_names.pkl")
    if not model_path.exists():
        # Fall back to models/
        model_path = Path("models/parkinsons_pipeline.joblib")
        feat_path  = Path("models/feature_names.pkl")

    pipe     = joblib.load(model_path)
    features = joblib.load(feat_path)

    # Skip wav2vec2 model — SHAP needs interpretable features
    if any(str(f).startswith("emb_") for f in features[:3]):
        print("Active model is wav2vec2 — loading models_italian_tuned/ instead")
        pipe     = joblib.load("models_italian_tuned/parkinsons_pipeline.joblib")
        features = joblib.load("models_italian_tuned/feature_names.pkl")

    df = pd.read_csv("data/italian_features.csv")
    X  = df[features].astype(float)
    y  = df["status"].astype(int).values
    return pipe, features, X, y


def get_xgb_model(pipe):
    """Extract the XGBoost sub-estimator from any pipeline structure."""
    import xgboost as xgb
    from sklearn.ensemble import VotingClassifier, StackingClassifier

    # Case 1: pipe is itself a VotingClassifier
    if isinstance(pipe, VotingClassifier):
        for name, est in zip(pipe.estimators, pipe.estimators_):
            inner = est.named_steps.get("clf") if hasattr(est, "named_steps") else est
            if isinstance(inner, xgb.XGBClassifier):
                return inner, est, name
        return None, None, None

    # Case 2: pipe is a Pipeline with clf step
    clf = pipe.named_steps.get("clf") if hasattr(pipe, "named_steps") else pipe

    # VotingClassifier as clf step
    if isinstance(clf, VotingClassifier):
        for name, est in zip(clf.estimators, clf.estimators_):
            inner = est.named_steps.get("clf") if hasattr(est, "named_steps") else est
            if isinstance(inner, xgb.XGBClassifier):
                return inner, est, name

    # Direct XGB
    if isinstance(clf, xgb.XGBClassifier):
        return clf, pipe, "xgb"

    return None, None, None


def main():
    print("Loading model and data...")
    pipe, features, X, y = load_model_and_data()
    print(f"  {len(X)} recordings, {len(features)} features")

    # ── Preprocess X through the pipeline's scaler/imputer ───────────────────
    # We need X in the transformed space for the XGB explainer
    xgb_clf, sub_pipe, model_name = get_xgb_model(pipe)

    if xgb_clf is None:
        print("Could not find XGBoost sub-model. Using TreeExplainer on full pipeline.")
        # Try direct approach
        clf = pipe.named_steps.get("clf")
        pre_steps = [s for s in list(pipe.named_steps.items())[:-1]]
        from sklearn.pipeline import Pipeline
        pre = Pipeline(pre_steps)
        X_pre = pre.fit_transform(X, y)
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_pre)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        X_display = pd.DataFrame(X_pre, columns=features)
    else:
        # Preprocess through the sub-pipeline steps before the classifier
        if hasattr(sub_pipe, "named_steps"):
            pre_steps = [(k, v) for k, v in sub_pipe.named_steps.items()
                         if k != "clf"]
            from sklearn.pipeline import Pipeline as _P
            pre = _P(pre_steps)
            X_pre = pre.fit_transform(X, y)
        elif hasattr(sub_pipe, "steps"):
            # Pipeline object
            pre_steps = sub_pipe.steps[:-1]
            from sklearn.pipeline import Pipeline as _P
            pre = _P(pre_steps)
            X_pre = pre.fit_transform(X, y)
        else:
            # Raw estimator — use SimpleImputer + StandardScaler manually
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import StandardScaler
            imp = SimpleImputer(strategy="median").fit(X)
            X_imp = imp.transform(X)
            scl = StandardScaler().fit(X_imp)
            X_pre = scl.transform(X_imp)
        print(f"  Using XGBoost sub-model ({model_name}) for SHAP")
        explainer   = shap.TreeExplainer(xgb_clf)
        shap_values = explainer.shap_values(X_pre)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        X_display = pd.DataFrame(X_pre, columns=features)

    print(f"  SHAP values shape: {shap_values.shape}")

    # ── Mean |SHAP| per feature ───────────────────────────────────────────────
    mean_abs = np.abs(shap_values).mean(axis=0)
    feat_importance = pd.DataFrame({
        "feature":   features,
        "mean_shap": mean_abs,
    }).sort_values("mean_shap", ascending=False).reset_index(drop=True)

    # ── Figure 1: Beeswarm summary plot ──────────────────────────────────────
    print("Generating beeswarm summary plot...")
    top_n = 20
    top_feats  = feat_importance["feature"].iloc[:top_n].tolist()
    top_idx    = [list(features).index(f) for f in top_feats]
    sv_top     = shap_values[:, top_idx]
    X_top      = X_display[top_feats]

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    shap.summary_plot(
        sv_top, X_top,
        feature_names=top_feats,
        plot_type="dot",
        show=False,
        color_bar=True,
        plot_size=None,
        max_display=top_n,
    )
    plt.title("SHAP Feature Importance — Italian Parkinson's Voice Dataset",
              fontsize=13, fontweight="bold", pad=16)
    plt.xlabel("SHAP value (impact on Parkinson's Probability)", fontsize=11)
    plt.tight_layout()
    fig_path = OUT / "shap_summary.png"
    plt.savefig(fig_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {fig_path}")

    # ── Figure 2: Clean bar chart ─────────────────────────────────────────────
    print("Generating bar chart...")
    top15 = feat_importance.head(15)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#1D4ED8" if i < 5 else "#60A5FA" if i < 10 else "#BFDBFE"
              for i in range(len(top15))]
    bars = ax.barh(range(len(top15)), top15["mean_shap"].values,
                   color=colors, edgecolor="white", height=0.65)
    ax.set_yticks(range(len(top15)))
    ax.set_yticklabels(top15["feature"].values, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value| (average impact on model output)", fontsize=10)
    ax.set_title("Top 15 Features — Mean Absolute SHAP Values\n"
                 "Italian Parkinson's Voice Dataset (n=831 recordings, 61 subjects)",
                 fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.25, linestyle="--")

    legend_patches = [
        mpatches.Patch(color="#1D4ED8", label="Top 5 features"),
        mpatches.Patch(color="#60A5FA", label="Features 6–10"),
        mpatches.Patch(color="#BFDBFE", label="Features 11–15"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)
    plt.tight_layout()
    bar_path = OUT / "shap_bar.png"
    plt.savefig(bar_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {bar_path}")

    # ── Save raw SHAP values ──────────────────────────────────────────────────
    shap_df = pd.DataFrame(shap_values, columns=features)
    shap_df["status"] = y
    shap_df.to_csv(OUT / "shap_values.csv", index=False)
    print(f"  Saved: {OUT/'shap_values.csv'}")

    # ── JSON report ───────────────────────────────────────────────────────────
    top10 = feat_importance.head(10)
    report = {
        "top_10_features": [
            {
                "rank": int(i + 1),
                "feature": row["feature"],
                "mean_abs_shap": round(float(row["mean_shap"]), 6),
                "clinical_significance": CLINICAL_RANK.get(row["feature"], "Not in clinical reference"),
            }
            for i, row in top10.iterrows()
        ],
        "n_recordings": len(X),
        "n_features_total": len(features),
        "model": "XGBoost (from tuned voting ensemble)",
        "dataset": "Italian Parkinson's Voice and Speech (Dimauro et al. 2019)",
        "notes": [
            "SHAP values computed using TreeExplainer on XGBoost sub-model.",
            "Positive SHAP = pushes prediction toward Parkinson's.",
            "Negative SHAP = pushes prediction toward Healthy.",
            "Features ordered by mean absolute SHAP across all 831 recordings.",
        ],
    }
    with open(OUT / "shap_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved: {OUT/'shap_report.json'}")

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n=== Top 10 Features by Mean |SHAP| ===")
    print(f"{'Rank':<5} {'Feature':<25} {'Mean |SHAP|':<14} Clinical significance")
    print("─" * 85)
    for i, row in feat_importance.head(10).iterrows():
        clin = CLINICAL_RANK.get(row["feature"], "—")
        print(f"{i+1:<5} {row['feature']:<25} {row['mean_shap']:.5f}       {clin}")

    print(f"\n✓ All SHAP outputs saved to: {OUT}/")
    print("  shap_summary.png — beeswarm plot (use as paper Figure)")
    print("  shap_bar.png     — bar chart (use as paper Figure)")
    print("  shap_values.csv  — raw values")
    print("  shap_report.json — structured results")


if __name__ == "__main__":
    main()
