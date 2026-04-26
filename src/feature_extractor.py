"""
MDVP-style feature extractor for sustained vowel phonation.

Produces the full 22-feature UCI Parkinson's vector:

    MDVP:Fo(Hz)        MDVP:Fhi(Hz)       MDVP:Flo(Hz)
    MDVP:Jitter(%)     MDVP:Jitter(Abs)   MDVP:RAP
    MDVP:PPQ           Jitter:DDP
    MDVP:Shimmer       MDVP:Shimmer(dB)   Shimmer:APQ3
    Shimmer:APQ5       MDVP:APQ           Shimmer:DDA
    NHR                HNR
    RPDE               DFA
    spread1            spread2            D2
    PPE

All jitter/shimmer variants come from distinct Praat PointProcess calls
(no aliasing). Nonlinear features (RPDE, DFA, PPE, D2, spread1/2) are
implemented in nonlinear_features.py.
"""

from __future__ import annotations

import os
import math
import warnings
from typing import Dict, Optional

import numpy as np
import parselmouth
from parselmouth.praat import call as praat_call
from scipy.signal import welch

from .audio_utils import load_and_clean, write_temp_wav, SR_TARGET
from . import nonlinear_features as nlf


# Canonical feature order matching the UCI Parkinson's CSV header
FEATURE_NAMES = [
    "MDVP:Fo(Hz)",
    "MDVP:Fhi(Hz)",
    "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)",
    "MDVP:RAP",
    "MDVP:PPQ",
    "Jitter:DDP",
    "MDVP:Shimmer",
    "MDVP:Shimmer(dB)",
    "Shimmer:APQ3",
    "Shimmer:APQ5",
    "MDVP:APQ",
    "Shimmer:DDA",
    "NHR",
    "HNR",
    "RPDE",
    "DFA",
    "spread1",
    "spread2",
    "D2",
    "PPE",
]


# ---------------------------------------------------------------------------
# Praat helpers
# ---------------------------------------------------------------------------
def _robust_pitch(sound: parselmouth.Sound) -> parselmouth.Pitch:
    """
    Run autocorrelation pitch extraction with the Praat default ranges
    tuned for sustained vowels. Falls back to a wider range if the first
    attempt produces too few voiced frames.
    """
    # Primary: Praat defaults suitable for adult sustained phonation
    pitch = sound.to_pitch_ac(
        time_step=0.01,
        pitch_floor=75.0,
        pitch_ceiling=600.0,
    )
    voiced_frac = _voiced_fraction(pitch)
    if voiced_frac >= 0.30:
        return pitch

    # Fallback for very low / very high voices
    pitch2 = sound.to_pitch_ac(
        time_step=0.01,
        pitch_floor=50.0,
        pitch_ceiling=800.0,
    )
    return pitch2 if _voiced_fraction(pitch2) > voiced_frac else pitch


def _voiced_fraction(pitch: parselmouth.Pitch) -> float:
    vals = pitch.selected_array["frequency"]
    voiced = np.count_nonzero(vals > 0)
    return voiced / max(vals.size, 1)


def _f0_contour(pitch: parselmouth.Pitch) -> np.ndarray:
    """Return only voiced F0 values in Hz."""
    vals = pitch.selected_array["frequency"]
    return vals[vals > 0]


def _point_process(sound: parselmouth.Sound, pitch: parselmouth.Pitch):
    """Glottal pulse PointProcess derived from sound + pitch."""
    return praat_call(
        [sound, pitch], "To PointProcess (cc)"
    )


# ---------------------------------------------------------------------------
# Jitter and shimmer - each variant from its OWN Praat call
# ---------------------------------------------------------------------------
# Standard Praat parameters (from the manual):
#   tmin=0, tmax=0 (whole object), period floor=0.0001s, period ceiling=0.02s,
#   max period factor=1.3 (shimmer also takes max amp factor=1.6)
_JITTER_ARGS = (0.0, 0.0, 0.0001, 0.02, 1.3)
_SHIMMER_ARGS = (0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)


def _jitter_measures(point_process) -> Dict[str, float]:
    j = {}
    j["local"] = praat_call(point_process, "Get jitter (local)", *_JITTER_ARGS)
    j["local_abs"] = praat_call(point_process, "Get jitter (local, absolute)", *_JITTER_ARGS)
    j["rap"] = praat_call(point_process, "Get jitter (rap)", *_JITTER_ARGS)
    j["ppq5"] = praat_call(point_process, "Get jitter (ppq5)", *_JITTER_ARGS)
    j["ddp"] = praat_call(point_process, "Get jitter (ddp)", *_JITTER_ARGS)
    return j


def _shimmer_measures(sound: parselmouth.Sound, point_process) -> Dict[str, float]:
    s = {}
    s["local"] = praat_call(
        [sound, point_process], "Get shimmer (local)", *_SHIMMER_ARGS
    )
    s["local_dB"] = praat_call(
        [sound, point_process], "Get shimmer (local_dB)", *_SHIMMER_ARGS
    )
    s["apq3"] = praat_call(
        [sound, point_process], "Get shimmer (apq3)", *_SHIMMER_ARGS
    )
    s["apq5"] = praat_call(
        [sound, point_process], "Get shimmer (apq5)", *_SHIMMER_ARGS
    )
    s["apq11"] = praat_call(
        [sound, point_process], "Get shimmer (apq11)", *_SHIMMER_ARGS
    )
    s["dda"] = praat_call(
        [sound, point_process], "Get shimmer (dda)", *_SHIMMER_ARGS
    )
    return s


# ---------------------------------------------------------------------------
# HNR and NHR
# ---------------------------------------------------------------------------
def _hnr(sound: parselmouth.Sound) -> float:
    """Harmonic-to-Noise ratio in dB via Praat's autocorrelation method."""
    harm = sound.to_harmonicity_cc(
        time_step=0.01, minimum_pitch=75.0, silence_threshold=0.1, periods_per_window=1.0
    )
    return float(praat_call(harm, "Get mean", 0.0, 0.0))


def _nhr_spectral(y: np.ndarray, sr: int, f0_mean: float) -> float:
    """
    MDVP-style NHR: ratio of inharmonic energy in 1500-4500 Hz to
    harmonic energy in 70-4500 Hz.

    We estimate the power spectral density with Welch's method, sum
    the power at narrow bands centred on each harmonic of F0 (this is
    the harmonic energy in 70-4500 Hz), and subtract those same bands
    from the 1500-4500 Hz region to get the inharmonic energy.
    """
    if f0_mean is None or not (50 < f0_mean < 800):
        return float("nan")

    nperseg = min(4096, y.size)
    f, Pxx = welch(y, fs=sr, nperseg=nperseg, noverlap=nperseg // 2)

    # Harmonic comb half-width: 20 Hz or 2 FFT bins, whichever is larger
    bin_hz = f[1] - f[0]
    half_w = max(20.0, 2 * bin_hz)

    harmonics = []
    k = 1
    while True:
        h = k * f0_mean
        if h > 4500:
            break
        if h >= 70:
            harmonics.append(h)
        k += 1
    if not harmonics:
        return float("nan")

    def _sum_bands(fmin: float, fmax: float, band_centres) -> float:
        mask = np.zeros_like(f, dtype=bool)
        for c in band_centres:
            if fmin <= c <= fmax:
                mask |= (f >= c - half_w) & (f <= c + half_w)
        return float(np.trapezoid(Pxx[mask], f[mask])) if mask.any() else 0.0

    # Harmonic energy: 70-4500 Hz at harmonic peaks
    harm_energy = _sum_bands(70.0, 4500.0, harmonics)
    # Noise energy: 1500-4500 Hz minus harmonic peaks
    noise_mask = (f >= 1500.0) & (f <= 4500.0)
    for c in harmonics:
        if 1500.0 <= c <= 4500.0:
            noise_mask &= ~((f >= c - half_w) & (f <= c + half_w))
    noise_energy = float(np.trapezoid(Pxx[noise_mask], f[noise_mask])) if noise_mask.any() else 0.0

    if harm_energy <= 0:
        return float("nan")
    return noise_energy / harm_energy


# ---------------------------------------------------------------------------
# Top-level extractor
# ---------------------------------------------------------------------------
def extract_features(
    path: str,
    compute_d2: bool = True,
    compute_rpde: bool = True,
    extended: bool = False,
) -> Dict[str, float]:
    """
    Extract acoustic features from a sustained vowel recording.

    Parameters
    ----------
    path        : audio file path (WAV/MP3/FLAC/OGG/etc. supported by librosa)
    compute_d2  : D2 (correlation dimension) is the slowest feature -
                  disable to speed up batch runs if D2 is not required.
    compute_rpde: same for RPDE (second slowest).
    extended    : if True, also compute the 34 extra features
                  (CPP, MFCC 1-13 mean+std, formants F1-F3 mean+bw,
                  spectral tilt). Default False for backward compatibility
                  with the UCI-22 schema.

    Returns
    -------
    dict {feature_name: float}. Missing/failed values are NaN. Use
    `FEATURE_NAMES` for the canonical UCI-22 column order.
    """
    y, sr = load_and_clean(path)

    # Praat operates on its own Sound object. Writing to a temp WAV is
    # the most reliable way to keep the exact samples we analysed.
    tmp_wav = write_temp_wav(y, sr)
    try:
        sound = parselmouth.Sound(tmp_wav)
        pitch = _robust_pitch(sound)

        f0 = _f0_contour(pitch)
        if f0.size < 5:
            raise RuntimeError(
                "Pitch extraction produced too few voiced frames. "
                "Is this a sustained vowel recording?"
            )
        mean_f0 = float(np.mean(f0))
        min_f0 = float(np.min(f0))
        max_f0 = float(np.max(f0))

        point_process = _point_process(sound, pitch)
        jit = _jitter_measures(point_process)
        shim = _shimmer_measures(sound, point_process)

        hnr_db = _hnr(sound)
        nhr_ratio = _nhr_spectral(y, sr, mean_f0)

        # Nonlinear measures
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rpde_val = nlf.rpde(y) if compute_rpde else float("nan")
            dfa_val = nlf.dfa(y)
            d2_val = nlf.correlation_dimension(y) if compute_d2 else float("nan")
            ppe_val = nlf.ppe(f0)
            spread1, spread2 = nlf.spread_measures(f0)

        feats: Dict[str, float] = {
            "MDVP:Fo(Hz)": mean_f0,
            "MDVP:Fhi(Hz)": max_f0,
            "MDVP:Flo(Hz)": min_f0,
            "MDVP:Jitter(%)": _safe(jit["local"]),
            "MDVP:Jitter(Abs)": _safe(jit["local_abs"]),
            "MDVP:RAP": _safe(jit["rap"]),
            "MDVP:PPQ": _safe(jit["ppq5"]),
            "Jitter:DDP": _safe(jit["ddp"]),
            "MDVP:Shimmer": _safe(shim["local"]),
            "MDVP:Shimmer(dB)": _safe(shim["local_dB"]),
            "Shimmer:APQ3": _safe(shim["apq3"]),
            "Shimmer:APQ5": _safe(shim["apq5"]),
            "MDVP:APQ": _safe(shim["apq11"]),
            "Shimmer:DDA": _safe(shim["dda"]),
            "NHR": _safe(nhr_ratio),
            "HNR": _safe(hnr_db),
            "RPDE": _safe(rpde_val),
            "DFA": _safe(dfa_val),
            "spread1": _safe(spread1),
            "spread2": _safe(spread2),
            "D2": _safe(d2_val),
            "PPE": _safe(ppe_val),
        }

        # Extended feature set (34 extra)
        if extended:
            from .extra_features import extract_extra_features
            feats.update(extract_extra_features(y, sr, sound))

        return feats
    finally:
        try:
            os.remove(tmp_wav)
        except OSError:
            pass


def _safe(x) -> float:
    try:
        f = float(x)
        if math.isnan(f) or math.isinf(f):
            return float("nan")
        return f
    except (TypeError, ValueError):
        return float("nan")
