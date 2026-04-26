"""
Audio I/O and pre-processing utilities.

Handles loading of arbitrary formats, resampling to a canonical 22050 Hz
mono, silence trimming, and peak normalisation. Everything downstream
expects the output of `load_and_clean`.
"""

from __future__ import annotations

import os
import tempfile
from typing import Tuple

import numpy as np
import librosa
import soundfile as sf

SR_TARGET = 22050  # Hz


def load_and_clean(
    path: str,
    sr: int = SR_TARGET,
    trim_db: float = 25.0,
    normalise: bool = True,
    min_len_sec: float = 0.5,
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file and return a clean mono waveform at `sr`.

    * Silence at both ends is trimmed at `trim_db` below peak.
    * Peak-normalised to [-1, 1] if `normalise` is True.
    * If the resulting clip is shorter than `min_len_sec`, raises
      ValueError so downstream feature extractors don't see garbage.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    y, sr_out = librosa.load(path, sr=sr, mono=True)
    if y.size == 0:
        raise ValueError(f"Empty audio file: {path}")

    y, _ = librosa.effects.trim(y, top_db=trim_db)
    if y.size < int(min_len_sec * sr_out):
        raise ValueError(
            f"Clip too short after silence trim: {y.size / sr_out:.2f}s "
            f"(need >= {min_len_sec}s of voice)"
        )

    if normalise:
        peak = float(np.max(np.abs(y)))
        if peak > 0:
            y = y / peak

    return y.astype(np.float64), sr_out


def write_temp_wav(y: np.ndarray, sr: int, prefix: str = "pva_") -> str:
    """Write a waveform to a temp WAV file and return the path."""
    fd, path = tempfile.mkstemp(suffix=".wav", prefix=prefix)
    os.close(fd)
    sf.write(path, y, sr)
    return path
