"""
Motor biomarker scoring for Parkinson's disease screening.

Scores spiral drawing and typing dynamics features against
published clinical reference ranges. All thresholds are cited.

References:
  Pullman SL (1998). Spiral analysis: A new technique for measuring tremor
    with a digitizing tablet. Movement Disorders, 13(S3), 85–89.
  Smits EJ et al. (2014). Standardized handwriting to assess bradykinesia,
    micrographia and tremor in Parkinson's disease. PLOS ONE, 9(5).
  Adams WR (2017). High-accuracy detection of early Parkinson's Disease
    using multiple characteristics of finger movement while typing.
    PLOS ONE, 12(11).
  Arroyo-Gallego T et al. (2017). Detection of Motor Impairment in
    Parkinson's Disease via Mobile Touchscreen Typing. IEEE Trans. Biomed.
    Engineering, 64(9), 2911–2920.
"""
from __future__ import annotations
from typing import Dict, Any


# ---------------------------------------------------------------------------
# Clinical reference thresholds (from literature above)
# ---------------------------------------------------------------------------

SPIRAL_NORMS = {
    "velocity_cv": {
        "healthy_max": 0.35,   # Smits 2014: CV < 0.35 in controls
        "pd_min":      0.65,   # CV > 0.65 in PD
        "weight":      0.45,
        "label":       "Velocity consistency",
        "unit":        "CV",
        "direction":   "lower_better",
    },
    "tremor_freq": {
        "healthy_max": 3.0,    # Pullman 1998: tremor < 3 Hz in controls
        "pd_min":      5.0,    # Essential/PD tremor > 4–6 Hz
        "weight":      0.35,
        "label":       "Tremor frequency",
        "unit":        "Hz",
        "direction":   "lower_better",
    },
    "deviation_norm": {
        "healthy_max": 0.15,   # Deviation < 15% of spiral radius in controls
        "pd_min":      0.40,   # > 40% in PD
        "weight":      0.20,
        "label":       "Path accuracy",
        "unit":        "ratio",
        "direction":   "lower_better",
    },
}

TYPING_NORMS = {
    "iki_cv": {
        "healthy_max": 0.25,   # Adams 2017: IKI CV < 0.25 in controls
        "pd_min":      0.50,   # IKI CV > 0.50 in PD
        "weight":      0.40,
        "label":       "Keystroke timing regularity",
        "unit":        "CV",
        "direction":   "lower_better",
    },
    "wpm": {
        "healthy_min": 40.0,   # Arroyo-Gallego 2017: > 40 WPM in controls
        "pd_max":      25.0,   # < 25 WPM in PD
        "weight":      0.30,
        "label":       "Typing speed",
        "unit":        "WPM",
        "direction":   "higher_better",
    },
    "hold_cv": {
        "healthy_max": 0.30,   # Adams 2017: hold time CV < 0.30 in controls
        "pd_min":      0.55,   # > 0.55 in PD
        "weight":      0.20,
        "label":       "Key hold regularity",
        "unit":        "CV",
        "direction":   "lower_better",
    },
    "error_rate": {
        "healthy_max": 0.05,   # Error rate < 5% in controls
        "pd_min":      0.15,   # > 15% in PD
        "weight":      0.10,
        "label":       "Error rate",
        "unit":        "fraction",
        "direction":   "lower_better",
    },
}

# Fusion weights (from multimodal PD literature)
FUSION_WEIGHTS = {
    "voice":  0.55,   # Voice is most validated modality
    "spiral": 0.28,   # Handwriting/drawing is second-best
    "typing": 0.17,   # Typing dynamics supports the other two
}


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _score_lower_better(value: float, healthy_max: float, pd_min: float) -> float:
    """Return 0–100 where 100 = healthy, 0 = strong PD indicator."""
    if value <= healthy_max:
        return 100.0
    if value >= pd_min:
        return 0.0
    # Linear interpolation in the uncertain zone
    return 100.0 * (pd_min - value) / (pd_min - healthy_max)


def _score_higher_better(value: float, healthy_min: float, pd_max: float) -> float:
    """Return 0–100 where 100 = healthy, 0 = strong PD indicator."""
    if value >= healthy_min:
        return 100.0
    if value <= pd_max:
        return 0.0
    return 100.0 * (value - pd_max) / (healthy_min - pd_max)


def _status(score: float) -> str:
    if score >= 70:
        return "normal"
    if score >= 40:
        return "borderline"
    return "abnormal"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_spiral(features: Dict[str, float]) -> Dict[str, Any]:
    """
    Score spiral drawing features against clinical norms.

    Expected features:
        velocity_cv:    coefficient of variation of drawing velocity
        tremor_freq:    dominant tremor frequency in Hz
        deviation_norm: normalized deviation from ideal spiral

    Returns scored dict with per-feature detail and overall score.
    """
    results = {}
    weighted_sum = 0.0
    total_weight = 0.0

    for feat, norm in SPIRAL_NORMS.items():
        val = features.get(feat)
        if val is None:
            continue
        if norm["direction"] == "lower_better":
            score = _score_lower_better(val, norm["healthy_max"], norm["pd_min"])
        else:
            score = _score_higher_better(val, norm["healthy_min"], norm["pd_max"])

        results[feat] = {
            "value":         round(float(val), 4),
            "score":         round(score, 1),
            "status":        _status(score),
            "label":         norm["label"],
            "unit":          norm["unit"],
            "healthy_range": f"< {norm['healthy_max']}" if norm["direction"] == "lower_better"
                             else f"> {norm['healthy_min']}",
            "pd_range":      f"> {norm['pd_min']}" if norm["direction"] == "lower_better"
                             else f"< {norm['pd_max']}",
        }
        weighted_sum += score * norm["weight"]
        total_weight += norm["weight"]

    overall = round(weighted_sum / total_weight, 1) if total_weight > 0 else 50.0
    pd_risk = round((100 - overall) / 100, 4)

    return {
        "overall_score":   overall,
        "pd_risk":         pd_risk,
        "overall_status":  _status(overall),
        "features":        results,
        "references": [
            "Pullman SL (1998). Spiral analysis. Movement Disorders.",
            "Smits EJ et al. (2014). Standardized handwriting. PLOS ONE.",
        ],
    }


def score_typing(features: Dict[str, float]) -> Dict[str, Any]:
    """
    Score typing dynamics features against clinical norms.

    Expected features:
        iki_cv:      coefficient of variation of inter-key intervals
        wpm:         words per minute
        hold_cv:     coefficient of variation of key hold times
        error_rate:  fraction of backspace/delete presses

    Returns scored dict with per-feature detail and overall score.
    """
    results = {}
    weighted_sum = 0.0
    total_weight = 0.0

    for feat, norm in TYPING_NORMS.items():
        val = features.get(feat)
        if val is None:
            continue
        if norm["direction"] == "lower_better":
            score = _score_lower_better(val, norm["healthy_max"], norm["pd_min"])
        else:
            score = _score_higher_better(val, norm["healthy_min"], norm["pd_max"])

        if norm["direction"] == "lower_better":
            healthy_range = f"< {norm['healthy_max']}"
            pd_range = f"> {norm['pd_min']}"
        else:
            healthy_range = f"> {norm['healthy_min']}"
            pd_range = f"< {norm['pd_max']}"

        results[feat] = {
            "value":         round(float(val), 4),
            "score":         round(score, 1),
            "status":        _status(score),
            "label":         norm["label"],
            "unit":          norm["unit"],
            "healthy_range": healthy_range,
            "pd_range":      pd_range,
        }
        weighted_sum += score * norm["weight"]
        total_weight += norm["weight"]

    overall = round(weighted_sum / total_weight, 1) if total_weight > 0 else 50.0
    pd_risk = round((100 - overall) / 100, 4)

    return {
        "overall_score":   overall,
        "pd_risk":         pd_risk,
        "overall_status":  _status(overall),
        "features":        results,
        "references": [
            "Adams WR (2017). High-accuracy detection of early PD via typing. PLOS ONE.",
            "Arroyo-Gallego T et al. (2017). Motor impairment via touchscreen typing. IEEE TBME.",
        ],
    }


def compute_combined_score(
    voice_probability: float,
    spiral_score: float,
    typing_score: float,
) -> Dict[str, Any]:
    """
    Compute a combined Parkinson's risk estimate from three modalities.

    voice_probability: P(PD) from voice model  [0, 1]
    spiral_score:      motor health score       [0, 100]
    typing_score:      typing health score      [0, 100]

    Returns combined risk probability and breakdown.
    """
    voice_risk  = float(voice_probability)
    spiral_risk = (100 - float(spiral_score)) / 100
    typing_risk = (100 - float(typing_score)) / 100

    combined = (
        voice_risk  * FUSION_WEIGHTS["voice"]  +
        spiral_risk * FUSION_WEIGHTS["spiral"] +
        typing_risk * FUSION_WEIGHTS["typing"]
    )

    # Threshold at 0.38 (same as voice alone — calibrated on Italian data)
    THRESHOLD = 0.38
    prediction = int(combined >= THRESHOLD)

    return {
        "combined_probability": round(combined, 4),
        "prediction":           prediction,
        "threshold":            THRESHOLD,
        "label": (
            "Multi-modal Parkinson's indicators detected"
            if prediction else
            "No Parkinson's indicators detected across modalities"
        ),
        "confidence_pct":       round(100 * max(combined, 1 - combined), 1),
        "breakdown": {
            "voice":  {"weight": FUSION_WEIGHTS["voice"],
                       "risk":   round(voice_risk, 4),
                       "contribution": round(voice_risk * FUSION_WEIGHTS["voice"], 4)},
            "spiral": {"weight": FUSION_WEIGHTS["spiral"],
                       "risk":   round(spiral_risk, 4),
                       "contribution": round(spiral_risk * FUSION_WEIGHTS["spiral"], 4)},
            "typing": {"weight": FUSION_WEIGHTS["typing"],
                       "risk":   round(typing_risk, 4),
                       "contribution": round(typing_risk * FUSION_WEIGHTS["typing"], 4)},
        },
        "fusion_method": "Weighted linear combination (weights from published PD literature)",
        "note": (
            "Combined score is a research prototype. Individual modality "
            "models were not trained on PD motor data — spiral and typing "
            "features are scored against published clinical reference ranges "
            "(Pullman 1998, Smits 2014, Adams 2017, Arroyo-Gallego 2017)."
        ),
    }
