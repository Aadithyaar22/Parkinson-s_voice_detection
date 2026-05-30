"""
Standalone spectrogram generator.
Called by the Flask app at prediction time to produce a base64-encoded
spectrogram image that gets embedded in the PDF export.

Returns a base64 PNG string — no file saved to disk.
"""
from __future__ import annotations

import base64
import io
from typing import Optional

import numpy as np

try:
    import librosa
    import librosa.display
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


def generate_spectrogram_b64(
    wav_path: str,
    figsize: tuple = (8, 3),
    dpi: int = 120,
) -> Optional[str]:
    """
    Generate a mel-spectrogram for the given WAV file.
    Returns a base64-encoded PNG string, or None if unavailable.
    """
    if not _AVAILABLE:
        return None

    try:
        y, sr = librosa.load(wav_path, sr=None, mono=True)
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000

        # Trim silence
        y, _ = librosa.effects.trim(y, top_db=25)
        if len(y) < sr // 4:
            return None

        # Mel spectrogram
        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=80, fmax=8000,
            n_fft=1024, hop_length=160,
        )
        S_db = librosa.power_to_db(S, ref=np.max)

        fig, ax = plt.subplots(figsize=figsize)
        img = librosa.display.specshow(
            S_db, sr=sr, hop_length=160,
            x_axis="time", y_axis="mel",
            fmax=8000, ax=ax, cmap="Blues",
        )
        fig.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.01)
        ax.set_title("Mel Spectrogram", fontsize=10, fontweight="bold", pad=8)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel("Frequency (Hz)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout(pad=0.5)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    except Exception:
        return None
