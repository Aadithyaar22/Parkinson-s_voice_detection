"""
Runtime wav2vec2 embedding extraction for the Flask app.

Separate from scripts/extract_wav2vec2_embeddings.py (which is a batch
offline script) because:
  - The Flask app needs a module-level loaded model (one load, many requests)
  - We want graceful degradation: if torch or transformers isn't
    installed, importing this module shouldn't crash the whole app -
    it should just mark w2v2 as unavailable.
"""
from __future__ import annotations

import os
import threading
from typing import Optional

import numpy as np

# Flag + lazy-loaded globals
_AVAILABLE = False
_LOAD_ERROR: Optional[str] = None
_model = None
_feature_extractor = None
_device = None
_load_lock = threading.Lock()

MODEL_NAME = os.environ.get(
    "PVA_W2V2_MODEL", "facebook/wav2vec2-xls-r-300m"
)
MAX_SECONDS = float(os.environ.get("PVA_W2V2_MAX_SECONDS", "12"))
TARGET_SR = 16000


def is_available() -> bool:
    """True if wav2vec2 inference is usable in this environment."""
    _ensure_loaded()
    return _AVAILABLE


def load_error() -> Optional[str]:
    return _LOAD_ERROR


def embedding_dim() -> Optional[int]:
    _ensure_loaded()
    if _model is None:
        return None
    return int(_model.config.hidden_size)


def _ensure_loaded() -> None:
    global _AVAILABLE, _LOAD_ERROR, _model, _feature_extractor, _device
    if _AVAILABLE or _LOAD_ERROR is not None:
        return
    with _load_lock:
        if _AVAILABLE or _LOAD_ERROR is not None:
            return
        try:
            import torch
            from transformers import AutoFeatureExtractor, AutoModel
        except ImportError as e:
            _LOAD_ERROR = (
                f"torch or transformers not installed: {e}. "
                f"Install with: pip install -r requirements_wav2vec2.txt"
            )
            return
        try:
            # Device: MPS on Apple Silicon, CUDA if available, else CPU
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                dev = torch.device("mps")
            elif torch.cuda.is_available():
                dev = torch.device("cuda")
            else:
                dev = torch.device("cpu")

            print(f"[w2v2] loading {MODEL_NAME} on {dev}... "
                  f"(first run downloads ~1.2GB)", flush=True)
            fe = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
            m = AutoModel.from_pretrained(MODEL_NAME)
            m.eval()
            m.to(dev)

            _feature_extractor = fe
            _model = m
            _device = dev
            _AVAILABLE = True
            print(f"[w2v2] loaded. embedding dim = {m.config.hidden_size}",
                  flush=True)
        except Exception as e:
            _LOAD_ERROR = f"failed to load model: {type(e).__name__}: {e}"
            print(f"[w2v2] {_LOAD_ERROR}", flush=True)


def extract_embedding(audio_path: str) -> np.ndarray:
    """
    Load audio at 16 kHz mono, trim silence, pump through wav2vec2,
    mean-pool across time, return a (hidden_size,) numpy array.

    Raises RuntimeError if wav2vec2 isn't available.
    """
    _ensure_loaded()
    if not _AVAILABLE:
        raise RuntimeError(f"wav2vec2 not available: {_LOAD_ERROR}")

    import librosa
    import torch

    y, _ = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    if y.size == 0:
        raise ValueError("empty audio")
    y, _ = librosa.effects.trim(y, top_db=25)
    if y.size < TARGET_SR // 4:
        y = np.pad(y, (0, TARGET_SR // 4 - y.size))
    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = y / peak
    y = y.astype(np.float32)

    # Clip to keep transformer memory sane (O(T^2))
    max_samples = int(MAX_SECONDS * TARGET_SR)
    if y.size > max_samples:
        y = y[:max_samples]

    inputs = _feature_extractor(
        y, sampling_rate=TARGET_SR, return_tensors="pt", padding=False,
    )
    input_values = inputs.input_values.to(_device)

    with torch.no_grad():
        out = _model(input_values)
    # out.last_hidden_state: (1, T, H) -> mean over T -> (H,)
    emb = out.last_hidden_state.mean(dim=1).squeeze(0)
    return emb.detach().cpu().numpy().astype(np.float32)
