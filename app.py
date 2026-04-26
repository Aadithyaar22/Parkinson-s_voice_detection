"""
Parkinson's Voice Analyser - Flask web server.

Supports two backends, auto-detected from the saved feature list:

  * Hand-crafted:  22 UCI features or 22+34 extended features. Fast,
    no torch dependency. Training done via src/train_v2.py or
    scripts/tune_italian.py.

  * wav2vec2:      1024-dim self-supervised embeddings. Slower
    (model load ~1.2GB, first request warms MPS). Requires torch +
    transformers. Training done via scripts/wav2vec2_experiment.py.

The backend is determined by the contents of models/feature_names.pkl -
if it starts with 'emb_', we're in wav2vec2 mode; otherwise hand-crafted.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from src.feature_extractor import FEATURE_NAMES as FULL_FEATURE_NAMES, extract_features

PROJECT_ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = PROJECT_ROOT / "uploads"
MODEL_DIR = PROJECT_ROOT / "models"
UPLOAD_DIR.mkdir(exist_ok=True)
ALLOWED_EXT = {"wav", "mp3", "flac", "ogg", "webm", "m4a"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB

# ---------------------------------------------------------------------------
# Load model + feature list
# ---------------------------------------------------------------------------
PIPELINE_PATH = MODEL_DIR / "parkinsons_pipeline.joblib"
FEATURES_PATH = MODEL_DIR / "feature_names.pkl"
REPORT_PATH = MODEL_DIR / "training_report.json"

pipeline = None
MODEL_FEATURES = FULL_FEATURE_NAMES   # default hand-crafted
TUNED_THRESHOLD = 0.5
training_report = {}
BACKEND = "handcrafted"               # "handcrafted" | "wav2vec2"

if PIPELINE_PATH.exists():
    pipeline = joblib.load(PIPELINE_PATH)
    print(f"[app] loaded pipeline: {PIPELINE_PATH}")
if FEATURES_PATH.exists():
    MODEL_FEATURES = joblib.load(FEATURES_PATH)
    print(f"[app] model expects {len(MODEL_FEATURES)} features")
if REPORT_PATH.exists():
    training_report = json.loads(REPORT_PATH.read_text())
    if "tuned_threshold" in training_report:
        TUNED_THRESHOLD = float(training_report["tuned_threshold"])
        print(f"[app] using tuned threshold: {TUNED_THRESHOLD:.3f}")

# Detect backend
if (
    isinstance(MODEL_FEATURES, list)
    and len(MODEL_FEATURES) > 0
    and all(isinstance(f, str) and f.startswith("emb_") for f in MODEL_FEATURES[:3])
):
    BACKEND = "wav2vec2"
    print(f"[app] backend: wav2vec2 ({len(MODEL_FEATURES)}-dim embeddings)")
    from src import wav2vec2_inference
    if not wav2vec2_inference.is_available():
        print(f"[app] WARNING: wav2vec2 backend selected but not functional:")
        print(f"[app]   {wav2vec2_inference.load_error()}")
        print(f"[app] The /predict endpoint will return 503 until fixed.")
else:
    print(f"[app] backend: handcrafted ({len(MODEL_FEATURES)} features)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def _transcode_to_wav(src: str) -> str:
    """Use ffmpeg to produce a 16 kHz mono WAV."""
    dst_fd, dst = tempfile.mkstemp(suffix=".wav", prefix="conv_", dir=UPLOAD_DIR)
    os.close(dst_fd)
    cmd = ["ffmpeg", "-y", "-i", src, "-ar", "16000", "-ac", "1", dst]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        raise RuntimeError(f"ffmpeg transcode failed: {e}") from e
    return dst


def _jsonify_floats(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if v is None:
            out[k] = None
            continue
        try:
            f = float(v)
            out[k] = None if math.isnan(f) or math.isinf(f) else f
        except (TypeError, ValueError):
            out[k] = None
    return out


# ---------------------------------------------------------------------------
# Backend-specific inference
# ---------------------------------------------------------------------------
def _predict_handcrafted(wav_path: str) -> dict:
    need_extended = len(MODEL_FEATURES) > 22
    feats = extract_features(
        wav_path,
        compute_d2=True, compute_rpde=True,
        extended=need_extended,
    )
    feats_json = _jsonify_floats(feats)

    row = [feats.get(k, float("nan")) for k in MODEL_FEATURES]
    X_row = pd.DataFrame([row], columns=MODEL_FEATURES).astype(float)
    proba = float(pipeline.predict_proba(X_row)[0, 1])

    measured = [k for k in MODEL_FEATURES if feats_json.get(k) is not None]
    imputed = [k for k in MODEL_FEATURES if feats_json.get(k) is None]
    return {
        "probability_pd": proba,
        "features": feats_json,
        "feature_order": FULL_FEATURE_NAMES,
        "model_features": list(MODEL_FEATURES),
        "n_measured": len(measured),
        "n_imputed": len(imputed),
        "n_model_features": len(MODEL_FEATURES),
        "imputed_features": imputed,
    }


def _predict_wav2vec2(wav_path: str) -> dict:
    from src import wav2vec2_inference
    if not wav2vec2_inference.is_available():
        raise RuntimeError(
            f"wav2vec2 not available: {wav2vec2_inference.load_error()}"
        )
    emb = wav2vec2_inference.extract_embedding(wav_path)
    X_row = pd.DataFrame([emb.tolist()], columns=MODEL_FEATURES).astype(float)
    proba = float(pipeline.predict_proba(X_row)[0, 1])
    return {
        "probability_pd": proba,
        "features": {},
        "feature_order": [],
        "model_features": list(MODEL_FEATURES),
        "n_measured": len(MODEL_FEATURES),
        "n_imputed": 0,
        "n_model_features": len(MODEL_FEATURES),
        "imputed_features": [],
    }


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


@app.route("/")
def index():
    return render_template(
        "index.html",
        report=training_report,
        model_loaded=pipeline is not None,
        backend=BACKEND,
    )


@app.route("/static/<path:p>")
def _static(p):
    return send_from_directory("static", p)


@app.route("/health", methods=["GET"])
def health():
    chosen = (training_report.get("chosen_model")
              or training_report.get("best_classifier")
              or "unknown")
    h = {
        "ok": True,
        "model_loaded": pipeline is not None,
        "backend": BACKEND,
        "n_features": len(MODEL_FEATURES),
        "chosen_model": chosen,
        "threshold": TUNED_THRESHOLD,
    }
    if BACKEND == "wav2vec2":
        from src import wav2vec2_inference
        h["wav2vec2_available"] = wav2vec2_inference.is_available()
        if not wav2vec2_inference.is_available():
            h["wav2vec2_error"] = wav2vec2_inference.load_error()
    return jsonify(h)


@app.route("/predict", methods=["POST"])
def predict():
    if pipeline is None:
        return jsonify({
            "error": "model_not_loaded",
            "message": "No pipeline found. Train a model first."
        }), 503

    if BACKEND == "wav2vec2":
        from src import wav2vec2_inference
        if not wav2vec2_inference.is_available():
            return jsonify({
                "error": "wav2vec2_unavailable",
                "message": wav2vec2_inference.load_error(),
                "hint": "pip install -r requirements_wav2vec2.txt",
            }), 503

    if "audio" not in request.files:
        return jsonify({"error": "no_file", "message": "Missing 'audio' field"}), 400
    f = request.files["audio"]
    if not f.filename or not _allowed(f.filename):
        return jsonify({
            "error": "bad_filename",
            "message": f"Upload file with extension in {sorted(ALLOWED_EXT)}",
        }), 400

    fname = secure_filename(f.filename)
    saved = str(UPLOAD_DIR / fname)
    f.save(saved)
    wav_path = saved

    try:
        ext = fname.rsplit(".", 1)[1].lower()
        if ext != "wav":
            wav_path = _transcode_to_wav(saved)

        if BACKEND == "wav2vec2":
            result = _predict_wav2vec2(wav_path)
        else:
            result = _predict_handcrafted(wav_path)

        proba = result["probability_pd"]
        pred = int(proba >= TUNED_THRESHOLD)

        response = {
            "prediction": pred,
            "probability_pd": proba,
            "threshold_used": TUNED_THRESHOLD,
            "confidence_pct": round(100 * max(proba, 1 - proba), 1),
            "label": ("Parkinson's indicators detected" if pred
                      else "No Parkinson's indicators detected"),
            "backend": BACKEND,
            "model": (training_report.get("chosen_model")
                      or training_report.get("best_classifier")
                      or "unknown"),
            "disclaimer": (
                "Research/educational prototype only. NOT a diagnostic device. "
                "Voice screening has inherent limitations; any clinical decision "
                "must be made by a qualified physician."
            ),
            **result,
        }
        return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": "prediction_failed",
            "message": f"{type(e).__name__}: {e}",
        }), 500
    finally:
        for p in {saved, wav_path}:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
