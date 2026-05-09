"""
Groq-powered natural language explanation for voice analysis results.
"""
from __future__ import annotations
import os
from typing import Generator

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL = "llama-3.3-70b-versatile"

FEATURE_CONTEXT = {
    "MDVP:Fo(Hz)": "average fundamental frequency (speaking pitch)",
    "MDVP:Jitter(%)": "cycle-to-cycle pitch variation (%)",
    "MDVP:Shimmer": "amplitude perturbation (volume variation)",
    "HNR": "harmonics-to-noise ratio (voice clarity, higher = cleaner)",
    "NHR": "noise-to-harmonics ratio (breathiness, lower = better)",
    "RPDE": "recurrence period density entropy (vocal complexity)",
    "DFA": "detrended fluctuation analysis (long-range vocal correlations)",
    "PPE": "pitch period entropy (irregularity of pitch control)",
}

def _build_prompt(probability, threshold, prediction, features, backend):
    pred_label = "Parkinson's indicators detected" if prediction == 1 else "No Parkinson's indicators detected"
    feature_summary = ""
    if backend == "handcrafted" and features:
        notable = []
        checks = {
            "MDVP:Jitter(%)": lambda v: ("elevated" if v > 0.008 else "normal", v, 0.008),
            "MDVP:Shimmer":   lambda v: ("elevated" if v > 0.04  else "normal", v, 0.04),
            "HNR":            lambda v: ("reduced"  if v < 16    else "normal", v, 16),
            "PPE":            lambda v: ("elevated" if v > 0.25  else "normal", v, 0.25),
            "DFA":            lambda v: ("abnormal" if v > 0.85  else "normal", v, 0.85),
        }
        for feat, check in checks.items():
            if feat in features and features[feat] is not None:
                status, value, ref = check(features[feat])
                label = FEATURE_CONTEXT.get(feat, feat)
                notable.append(f"  • {feat} ({label}): {value:.4f} — {status} (ref: {ref})")
        if notable:
            feature_summary = "\n\nKey acoustic measurements:\n" + "\n".join(notable)

    wav2vec_note = (
        "\n\nNote: This analysis used wav2vec2-XLS-R, a self-supervised speech model "
        "pretrained on 128 languages including Hindi, Tamil, Telugu, Bengali, and Marathi."
        if backend == "wav2vec2" else ""
    )

    return f"""You are a clinical speech-language pathologist assistant explaining a voice analysis result to a patient. Be clear, warm, and non-alarmist. Never make a definitive medical diagnosis.

Voice analysis result:
  Prediction: {pred_label}
  Probability of Parkinson's markers: {probability:.1%}
  Decision threshold: {threshold:.1%}
  Distance from threshold: {abs(probability - threshold):.1%} {'below' if probability < threshold else 'above'} the cutoff{feature_summary}{wav2vec_note}

Write a 3-paragraph explanation:

Paragraph 1 (2-3 sentences): What the result means in plain English. Use the probability and distance from threshold to convey confidence. Do NOT say "you have" or "you don't have" Parkinson's.

Paragraph 2 (2-3 sentences): What acoustic aspects of the voice drove this result. If wav2vec2 was used, explain what self-supervised speech models look for.

Paragraph 3 (2 sentences): What the person should do next. If markers were detected, recommend neurological consultation. If not, note this is a screening tool not a diagnosis.

Keep under 200 words. Plain paragraphs only, no bullet points or headers."""


def explain_stream(probability, threshold, prediction, features, backend):
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set.")
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    prompt = _build_prompt(probability, threshold, prediction, features, backend)
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.4,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta.encode("utf-8", errors="replace").decode("utf-8").replace(" ", " ").replace(" ", " ").replace("‰", " ")
