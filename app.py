"""
Parkinson's Voice Analyser — Flask server.

Backends: wav2vec2-XLS-R (default) or hand-crafted MDVP features.
Auth:     JWT-based email + password (MongoDB Atlas).
Features: /predict, /explain (Groq), /api/register, /api/login,
          /api/readings (save + get), /api/stats, /dashboard.
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
from flask import (Flask, Response, jsonify, render_template, request,
                   send_from_directory, stream_with_context)
from werkzeug.utils import secure_filename

from src.feature_extractor import FEATURE_NAMES as FULL_FEATURE_NAMES, extract_features

PROJECT_ROOT = Path(__file__).resolve().parent
UPLOAD_DIR   = PROJECT_ROOT / "uploads"
MODEL_DIR    = PROJECT_ROOT / "models"
UPLOAD_DIR.mkdir(exist_ok=True)
ALLOWED_EXT = {"wav", "mp3", "flac", "ogg", "webm", "m4a"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
PIPELINE_PATH = MODEL_DIR / "parkinsons_pipeline.joblib"
FEATURES_PATH = MODEL_DIR / "feature_names.pkl"
REPORT_PATH   = MODEL_DIR / "training_report.json"

pipeline       = None
MODEL_FEATURES = FULL_FEATURE_NAMES
TUNED_THRESHOLD = 0.5
training_report = {}
BACKEND = "handcrafted"

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

if (isinstance(MODEL_FEATURES, list) and len(MODEL_FEATURES) > 0
        and all(isinstance(f, str) and f.startswith("emb_")
                for f in MODEL_FEATURES[:3])):
    BACKEND = "wav2vec2"
    print(f"[app] backend: wav2vec2 ({len(MODEL_FEATURES)}-dim embeddings)")
    from src import wav2vec2_inference
    if not wav2vec2_inference.is_available():
        print(f"[app] WARNING: wav2vec2 not yet loaded — will load on first request")
else:
    print(f"[app] backend: handcrafted ({len(MODEL_FEATURES)} features)")

GROQ_AVAILABLE = bool(os.environ.get("GROQ_API_KEY", ""))
print(f"[app] Groq explanations: {'enabled' if GROQ_AVAILABLE else 'disabled (set GROQ_API_KEY)'}")

MONGO_AVAILABLE = bool(os.environ.get("MONGODB_URI", ""))
print(f"[app] MongoDB: {'enabled' if MONGO_AVAILABLE else 'disabled (set MONGODB_URI)'}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def _transcode_to_wav(src: str) -> str:
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
# Inference
# ---------------------------------------------------------------------------
def _predict_handcrafted(wav_path: str) -> dict:
    need_extended = len(MODEL_FEATURES) > 22
    feats = extract_features(wav_path, compute_d2=True, compute_rpde=True,
                             extended=need_extended)
    feats_json = _jsonify_floats(feats)
    row = [feats.get(k, float("nan")) for k in MODEL_FEATURES]
    X_row = pd.DataFrame([row], columns=MODEL_FEATURES).astype(float)
    proba = float(pipeline.predict_proba(X_row)[0, 1])
    measured = [k for k in MODEL_FEATURES if feats_json.get(k) is not None]
    imputed  = [k for k in MODEL_FEATURES if feats_json.get(k) is None]
    return {"probability_pd": proba, "features": feats_json,
            "feature_order": FULL_FEATURE_NAMES, "model_features": list(MODEL_FEATURES),
            "n_measured": len(measured), "n_imputed": len(imputed),
            "n_model_features": len(MODEL_FEATURES), "imputed_features": imputed}


def _predict_wav2vec2(wav_path: str) -> dict:
    from src import wav2vec2_inference
    if not wav2vec2_inference.is_available():
        raise RuntimeError(f"wav2vec2 not available: {wav2vec2_inference.load_error()}")
    emb = wav2vec2_inference.extract_embedding(wav_path)
    X_row = pd.DataFrame([emb.tolist()], columns=MODEL_FEATURES).astype(float)
    proba = float(pipeline.predict_proba(X_row)[0, 1])
    return {"probability_pd": proba, "features": {}, "feature_order": [],
            "model_features": list(MODEL_FEATURES), "n_measured": len(MODEL_FEATURES),
            "n_imputed": 0, "n_model_features": len(MODEL_FEATURES), "imputed_features": []}


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


# ── Pages ──────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", report=training_report,
                           model_loaded=pipeline is not None, backend=BACKEND,
                           groq_available=GROQ_AVAILABLE,
                           mongo_available=MONGO_AVAILABLE)


@app.route("/login")
def login_page():
    return render_template("login.html")


@app.route("/dashboard")
def dashboard_page():
    return render_template("dashboard.html")


@app.route("/static/<path:p>")
def _static(p):
    return send_from_directory("static", p)


# ── Health ──────────────────────────────────────────────────────────────────
@app.route("/health")
def health():
    chosen = (training_report.get("chosen_model")
              or training_report.get("best_classifier") or "unknown")
    h = {"ok": True, "model_loaded": pipeline is not None, "backend": BACKEND,
         "n_features": len(MODEL_FEATURES), "chosen_model": chosen,
         "threshold": TUNED_THRESHOLD, "groq_available": GROQ_AVAILABLE,
         "mongo_available": MONGO_AVAILABLE}
    return jsonify(h)


# ── Auth API ────────────────────────────────────────────────────────────────
@app.route("/api/register", methods=["POST"])
def api_register():
    if not MONGO_AVAILABLE:
        return jsonify({"message": "Database not configured."}), 503
    body = request.get_json(silent=True) or {}
    name     = (body.get("name") or "").strip()
    email    = (body.get("email") or "").strip().lower()
    password = body.get("password") or ""
    if not name or not email or not password:
        return jsonify({"message": "Name, email and password are required."}), 400
    if len(password) < 8:
        return jsonify({"message": "Password must be at least 8 characters."}), 400
    try:
        from src.database import create_user, get_user_by_email
        from src.auth import hash_password, generate_token
        if get_user_by_email(email):
            return jsonify({"message": "An account with that email already exists."}), 409
        hashed = hash_password(password)
        user = create_user(email, hashed, name)
        token = generate_token(str(user["_id"]), email, name)
        return jsonify({"token": token, "name": name, "email": email})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"message": f"Registration failed: {e}"}), 500


@app.route("/api/login", methods=["POST"])
def api_login():
    if not MONGO_AVAILABLE:
        return jsonify({"message": "Database not configured."}), 503
    body = request.get_json(silent=True) or {}
    email    = (body.get("email") or "").strip().lower()
    password = body.get("password") or ""
    if not email or not password:
        return jsonify({"message": "Email and password are required."}), 400
    try:
        from src.database import get_user_by_email, update_last_login
        from src.auth import verify_password, generate_token
        user = get_user_by_email(email)
        if not user or not verify_password(password, user["password_hash"]):
            return jsonify({"message": "Invalid email or password."}), 401
        uid = str(user["_id"])
        update_last_login(uid)
        token = generate_token(uid, email, user["name"])
        return jsonify({"token": token, "name": user["name"], "email": email})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"message": f"Login failed: {e}"}), 500


@app.route("/api/me")
def api_me():
    from src.auth import require_auth
    payload = require_auth(request)
    if not payload:
        return jsonify({"message": "Unauthorised."}), 401
    try:
        from src.database import get_user_by_id
        user = get_user_by_id(payload["user_id"])
        if not user:
            return jsonify({"message": "User not found."}), 404
        return jsonify({
            "name": user["name"],
            "email": user["email"],
            "created_at": user["created_at"].isoformat(),
        })
    except Exception as e:
        return jsonify({"message": str(e)}), 500


# ── Readings API ─────────────────────────────────────────────────────────────
@app.route("/api/readings", methods=["POST"])
def api_save_reading():
    from src.auth import require_auth
    payload = require_auth(request)
    if not payload:
        return jsonify({"message": "Unauthorised."}), 401
    body = request.get_json(silent=True) or {}
    try:
        from src.database import save_reading
        reading = save_reading(
            user_id=payload["user_id"],
            probability_pd=float(body.get("probability_pd", 0)),
            prediction=int(body.get("prediction", 0)),
            confidence_pct=float(body.get("confidence_pct", 0)),
            audio_duration_s=float(body.get("audio_duration_s", 0)),
            backend=body.get("backend", BACKEND),
            model=body.get("model", "unknown"),
            notes=body.get("notes", ""),
        )
        return jsonify(reading)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"message": str(e)}), 500


@app.route("/api/readings", methods=["GET"])
def api_get_readings():
    from src.auth import require_auth
    payload = require_auth(request)
    if not payload:
        return jsonify({"message": "Unauthorised."}), 401
    try:
        from src.database import get_readings
        return jsonify(get_readings(payload["user_id"]))
    except Exception as e:
        return jsonify({"message": str(e)}), 500


@app.route("/api/stats")
def api_stats():
    from src.auth import require_auth
    payload = require_auth(request)
    if not payload:
        return jsonify({"message": "Unauthorised."}), 401
    try:
        from src.database import get_stats
        return jsonify(get_stats(payload["user_id"]))
    except Exception as e:
        return jsonify({"message": str(e)}), 500


# ── Predict ──────────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    if pipeline is None:
        return jsonify({"error": "model_not_loaded",
                        "message": "No pipeline found."}), 503
    if BACKEND == "wav2vec2":
        from src import wav2vec2_inference
        if not wav2vec2_inference.is_available():
            return jsonify({"error": "wav2vec2_unavailable",
                            "message": wav2vec2_inference.load_error()}), 503
    if "audio" not in request.files:
        return jsonify({"error": "no_file", "message": "Missing 'audio' field"}), 400
    f = request.files["audio"]
    if not f.filename or not _allowed(f.filename):
        return jsonify({"error": "bad_filename",
                        "message": f"Allowed: {sorted(ALLOWED_EXT)}"}), 400

    fname = secure_filename(f.filename)
    saved = str(UPLOAD_DIR / fname)
    f.save(saved)
    wav_path = saved
    try:
        ext = fname.rsplit(".", 1)[1].lower()
        if ext != "wav":
            wav_path = _transcode_to_wav(saved)
        result = _predict_wav2vec2(wav_path) if BACKEND == "wav2vec2" \
            else _predict_handcrafted(wav_path)

        # Recording quality assessment
        try:
            from src.quality_scorer import assess_quality
            quality = assess_quality(wav_path)
        except Exception:
            quality = {"overall_score": None, "overall_status": "warn",
                       "overall_message": "Quality assessment unavailable",
                       "metrics": {}, "recommendation": ""}
        proba = result["probability_pd"]
        pred  = int(proba >= TUNED_THRESHOLD)
        return jsonify({
            "prediction": pred,
            "probability_pd": proba,
            "threshold_used": TUNED_THRESHOLD,
            "confidence_pct": round(100 * max(proba, 1 - proba), 1),
            "label": ("Parkinson's indicators detected" if pred
                      else "No Parkinson's indicators detected"),
            "backend": BACKEND,
            "model": (training_report.get("chosen_model")
                      or training_report.get("best_classifier") or "unknown"),
            "groq_available": GROQ_AVAILABLE,
            "mongo_available": MONGO_AVAILABLE,
            "quality": quality,
            "disclaimer": ("Research/educational prototype only. NOT a diagnostic device. "
                           "Voice screening has inherent limitations; any clinical decision "
                           "must be made by a qualified physician."),
            **result,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "prediction_failed",
                        "message": f"{type(e).__name__}: {e}"}), 500
    finally:
        for p in {saved, wav_path}:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


# ── Explain ──────────────────────────────────────────────────────────────────
@app.route("/explain", methods=["POST"])
def explain():
    if not GROQ_AVAILABLE:
        return jsonify({"error": "groq_not_configured",
                        "message": "Set GROQ_API_KEY to enable explanations."}), 503
    body = request.get_json(silent=True) or {}
    try:
        from src.llm_explainer import explain_stream
        chunks = []
        for chunk in explain_stream(
            float(body.get("probability_pd", 0.5)),
            float(body.get("threshold_used", 0.5)),
            int(body.get("prediction", 0)),
            body.get("features", {}),
            body.get("backend", BACKEND),
        ):
            chunks.append(chunk)
        text = "".join(chunks)
        text = "".join(c if (32 <= ord(c) < 127 or c in " \n") else " " for c in text)
        text = " ".join(text.split())
        return jsonify({"explanation": text})
    except Exception as e:
        msg = "".join(c if ord(c) < 128 else "?" for c in str(e))
        return jsonify({"error": "explain_failed", "message": msg}), 500


# ── Run ──────────────────────────────────────────────────────────────────────
# ── Reading mutations ────────────────────────────────────────────────────────
@app.route("/api/readings/<reading_id>", methods=["DELETE"])
def api_delete_reading(reading_id):
    from src.auth import require_auth
    payload = require_auth(request)
    if not payload:
        return jsonify({"message": "Unauthorised."}), 401
    try:
        from src.database import delete_reading
        if not delete_reading(payload["user_id"], reading_id):
            return jsonify({"message": "Reading not found."}), 404
        return jsonify({"deleted": True})
    except Exception as e:
        return jsonify({"message": str(e)}), 500


@app.route("/api/readings/<reading_id>", methods=["PATCH"])
def api_edit_reading(reading_id):
    from src.auth import require_auth
    payload = require_auth(request)
    if not payload:
        return jsonify({"message": "Unauthorised."}), 401
    body = request.get_json(silent=True) or {}
    try:
        from src.database import update_reading_notes
        if not update_reading_notes(payload["user_id"], reading_id,
                                    body.get("notes", "")):
            return jsonify({"message": "Reading not found."}), 404
        return jsonify({"updated": True})
    except Exception as e:
        return jsonify({"message": str(e)}), 500


# ── Share links ───────────────────────────────────────────────────────────────
@app.route("/api/share", methods=["GET"])
def api_get_share():
    from src.auth import require_auth
    payload = require_auth(request)
    if not payload:
        return jsonify({"message": "Unauthorised."}), 401
    try:
        from src.database import get_share_for_user
        token = get_share_for_user(payload["user_id"])
        if not token:
            return jsonify({"token": None, "url": None})
        base = request.host_url.rstrip("/")
        return jsonify({"token": token, "url": f"{base}/report/{token}"})
    except Exception as e:
        return jsonify({"message": str(e)}), 500


@app.route("/api/share", methods=["POST"])
def api_create_share():
    from src.auth import require_auth
    payload = require_auth(request)
    if not payload:
        return jsonify({"message": "Unauthorised."}), 401
    try:
        from src.database import create_share_token
        token = create_share_token(payload["user_id"])
        base = request.host_url.rstrip("/")
        return jsonify({"token": token, "url": f"{base}/report/{token}"})
    except Exception as e:
        return jsonify({"message": str(e)}), 500


@app.route("/api/share", methods=["DELETE"])
def api_revoke_share():
    from src.auth import require_auth
    payload = require_auth(request)
    if not payload:
        return jsonify({"message": "Unauthorised."}), 401
    try:
        from src.database import revoke_share
        revoke_share(payload["user_id"])
        return jsonify({"revoked": True})
    except Exception as e:
        return jsonify({"message": str(e)}), 500


# ── Shared report (public, no auth) ───────────────────────────────────────────
@app.route("/report/<token>")
def shared_report(token):
    return render_template("report.html", token=token)


@app.route("/api/report/<token>")
def api_shared_report(token):
    try:
        from src.database import get_share_by_token, get_readings, get_stats, get_user_by_id
        from datetime import datetime as _dt
        share = get_share_by_token(token)
        if not share:
            return jsonify({"message": "Report not found or link has expired."}), 404
        uid = share["user_id"]
        user = get_user_by_id(uid)
        readings = get_readings(uid)
        stats = get_stats(uid)
        from src.database import get_motor_readings, get_motor_stats
        motor_readings = get_motor_readings(uid)
        motor_stats = get_motor_stats(uid)
        return jsonify({
            "patient_name": user["name"] if user else "Anonymous",
            "readings": readings,
            "stats": stats,
            "motor_readings": motor_readings,
            "motor_stats": motor_stats,
            "generated_at": _dt.utcnow().isoformat(),
        })
    except Exception as e:
        return jsonify({"message": str(e)}), 500


# ── Print-to-PDF export ───────────────────────────────────────────────────────
@app.route("/dashboard/export")
def export_pdf():
    from src.auth import verify_token
    token = request.args.get("token", "")
    if not token:
        return "Unauthorized", 401
    payload = verify_token(token)
    if not payload:
        return "Invalid or expired token", 401
    try:
        from src.database import get_readings, get_stats, get_user_by_id
        from datetime import datetime as _dt
        user = get_user_by_id(payload["user_id"])
        readings = get_readings(payload["user_id"])
        stats = get_stats(payload["user_id"])
        # Generate spectrogram from a placeholder (no audio stored)
        # In production this would come from a stored file
        spectrogram_b64 = None

        return render_template("export.html", user=user, readings=readings,
                               stats=stats, spectrogram_b64=spectrogram_b64,
                               generated_at=_dt.utcnow().strftime("%d %b %Y %H:%M UTC"))
    except Exception as e:
        return str(e), 500



@app.route("/api/motor/readings/<reading_id>", methods=["DELETE"])
def api_delete_motor(reading_id):
    from src.auth import require_auth
    payload = require_auth(request)
    if not payload:
        return jsonify({"message": "Unauthorised."}), 401
    try:
        from src.database import get_db
        from bson import ObjectId
        db = get_db()
        result = db.motor_readings.delete_one({
            "_id": ObjectId(reading_id),
            "user_id": ObjectId(payload["user_id"]),
        })
        if result.deleted_count == 0:
            return jsonify({"message": "Not found."}), 404
        return jsonify({"deleted": True})
    except Exception as e:
        return jsonify({"message": str(e)}), 500

# ── Motor save / read ─────────────────────────────────────────────────────────
@app.route("/api/motor/save", methods=["POST"])
def api_motor_save():
    from src.auth import require_auth
    payload = require_auth(request)
    if not payload:
        return jsonify({"message": "Unauthorised."}), 401
    body = request.get_json(silent=True) or {}
    try:
        from src.database import save_motor_reading
        doc = save_motor_reading(
            user_id=payload["user_id"],
            spiral_score=body.get("spiral_score", 50),
            spiral_pd_risk=body.get("spiral_pd_risk", 0.5),
            spiral_features=body.get("spiral_features", {}),
            typing_score=body.get("typing_score", 50),
            typing_pd_risk=body.get("typing_pd_risk", 0.5),
            typing_features=body.get("typing_features", {}),
            combined_probability=body.get("combined_probability", 0.5),
            combined_prediction=body.get("combined_prediction", 0),
            voice_probability=body.get("voice_probability"),
            notes=body.get("notes", ""),
        )
        return jsonify(doc)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"message": str(e)}), 500


@app.route("/api/motor/readings", methods=["GET"])
def api_motor_readings():
    from src.auth import require_auth
    payload = require_auth(request)
    if not payload:
        return jsonify({"message": "Unauthorised."}), 401
    try:
        from src.database import get_motor_readings
        return jsonify(get_motor_readings(payload["user_id"]))
    except Exception as e:
        return jsonify({"message": str(e)}), 500


@app.route("/api/motor/stats", methods=["GET"])
def api_motor_stats():
    from src.auth import require_auth
    payload = require_auth(request)
    if not payload:
        return jsonify({"message": "Unauthorised."}), 401
    try:
        from src.database import get_motor_stats
        return jsonify(get_motor_stats(payload["user_id"]))
    except Exception as e:
        return jsonify({"message": str(e)}), 500


# ── Motor assessment ──────────────────────────────────────────────────────────
@app.route("/motor")
def motor_page():
    return render_template("motor.html")


@app.route("/api/motor/spiral", methods=["POST"])
def api_motor_spiral():
    body = request.get_json(silent=True) or {}
    try:
        from src.motor_analyzer import score_spiral
        result = score_spiral({
            "velocity_cv":   body.get("velocity_cv"),
            "tremor_freq":   body.get("tremor_freq"),
            "deviation_norm": body.get("deviation_norm"),
        })
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/motor/typing", methods=["POST"])
def api_motor_typing():
    body = request.get_json(silent=True) or {}
    try:
        from src.motor_analyzer import score_typing
        result = score_typing({
            "iki_cv":     body.get("iki_cv"),
            "wpm":        body.get("wpm"),
            "hold_cv":    body.get("hold_cv"),
            "error_rate": body.get("error_rate"),
        })
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/motor/combined", methods=["POST"])
def api_motor_combined():
    body = request.get_json(silent=True) or {}
    try:
        from src.motor_analyzer import compute_combined_score
        voice_prob   = body.get("voice_probability")
        spiral_score = float(body.get("spiral_score", 50))
        typing_score = float(body.get("typing_score", 50))
        # If no voice reading, use neutral 0.5
        if voice_prob is None or float(voice_prob) < 0:
            voice_prob = 0.5
        result = compute_combined_score(
            float(voice_prob), spiral_score, typing_score
        )
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"[app] starting on {host}:{port}")
    app.run(host=host, port=port, debug=False)
