"""
Spiral drawing classifier for Parkinson's disease screening.

Model: HOG (Histogram of Oriented Gradients) + SVM-RBF
Trained on: Kaggle Parkinson's Drawings dataset (102 images, 51 PD / 51 Healthy)
LOO-AUC: 0.899  Test AUC: 0.889  Sensitivity: 0.867  Specificity: 0.800

Accepts two input types:
  1. Base64-encoded PNG from browser canvas (canvas.toDataURL())
  2. Uploaded image file (photo of paper spiral)

Reference:
  Chandra J et al. (2021). Screening of Parkinson's Disease Using Geometric
  Features Extracted from Spiral Drawings. Brain Sciences, 11(10), 1297.
"""
from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import joblib

try:
    from PIL import Image
    from skimage import feature, color
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

MODEL_DIR = Path(__file__).resolve().parent.parent / "models_motor"

_pipeline   = None
_hog_params = None
_img_size   = None


def _load_model():
    global _pipeline, _hog_params, _img_size
    if _pipeline is not None:
        return True
    try:
        _pipeline   = joblib.load(MODEL_DIR / "spiral_pipeline.joblib")
        _hog_params = joblib.load(MODEL_DIR / "spiral_hog_params.pkl")
        _img_size   = joblib.load(MODEL_DIR / "spiral_img_size.pkl")
        return True
    except Exception as e:
        print(f"[spiral] model load failed: {e}")
        return False


def is_available() -> bool:
    return _AVAILABLE and _load_model()


def _preprocess(img: Image.Image) -> np.ndarray:
    """Convert PIL image → HOG feature vector."""
    img = img.convert("RGB").resize(_img_size, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    gray = color.rgb2gray(arr)
    return feature.hog(gray, **_hog_params)


def _invert_if_dark(img: Image.Image) -> Image.Image:
    """
    If the image has a dark background (browser canvas default),
    invert it so spirals look like the training data (dark on white).
    Training data: dark spiral on white paper.
    Canvas: blue/black line on white — already correct.
    Photo: usually dark pen on white paper — also correct.
    Only invert if background is predominantly dark.
    """
    arr = np.array(img.convert("L"), dtype=np.float32)
    # Sample corners (likely background)
    h, w = arr.shape
    corners = [arr[:20,:20], arr[:20,-20:], arr[-20:,:20], arr[-20:,-20:]]
    bg_mean = np.mean([c.mean() for c in corners])
    if bg_mean < 100:  # dark background
        from PIL import ImageOps
        return ImageOps.invert(img.convert("RGB"))
    return img


def predict_from_base64(b64_str: str) -> Dict[str, Any]:
    """
    Classify a spiral from a base64-encoded PNG string (canvas.toDataURL()).

    Returns dict with probability_pd, prediction, confidence_pct, model_info.
    """
    if not is_available():
        return {"error": "spiral model not available"}
    try:
        # Strip data URL header if present
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_str)
        img = Image.open(io.BytesIO(img_bytes))
        img = _invert_if_dark(img)
        hog_feat = _preprocess(img)
        return _classify(hog_feat, source="canvas")
    except Exception as e:
        return {"error": f"canvas classification failed: {e}"}


def predict_from_file(file_path: str) -> Dict[str, Any]:
    """
    Classify a spiral from an uploaded image file.
    Accepts JPG, PNG, WebP — any format PIL can read.
    """
    if not is_available():
        return {"error": "spiral model not available"}
    try:
        img = Image.open(file_path)
        img = _invert_if_dark(img)
        hog_feat = _preprocess(img)
        return _classify(hog_feat, source="upload")
    except Exception as e:
        return {"error": f"file classification failed: {e}"}


def predict_from_bytes(file_bytes: bytes, filename: str = "") -> Dict[str, Any]:
    """Classify from raw bytes (uploaded via Flask request.files)."""
    if not is_available():
        return {"error": "spiral model not available"}
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img = _invert_if_dark(img)
        hog_feat = _preprocess(img)
        return _classify(hog_feat, source="upload")
    except Exception as e:
        return {"error": f"bytes classification failed: {e}"}


def _classify(hog_feat: np.ndarray, source: str) -> Dict[str, Any]:
    X = hog_feat.reshape(1, -1)
    proba = float(_pipeline.predict_proba(X)[0, 1])
    pred  = int(proba >= 0.5)
    conf  = round(100 * max(proba, 1 - proba), 1)

    return {
        "probability_pd": round(proba, 4),
        "prediction":     pred,
        "confidence_pct": conf,
        "label": (
            "Parkinson's spiral indicators detected"
            if pred else
            "No Parkinson's spiral indicators detected"
        ),
        "source":     source,
        "model":      "HOG + SVM-RBF",
        "loo_auc":    0.899,
        "test_auc":   0.889,
        "dataset":    "Kaggle Parkinson Drawings (102 images)",
        "disclaimer": (
            "Spiral classifier trained on clinical pen-on-paper drawings. "
            "Performance on browser canvas or phone photos may differ. "
            "Research prototype — not a diagnostic device."
        ),
    }


# Pre-load model at import time
_load_model()
