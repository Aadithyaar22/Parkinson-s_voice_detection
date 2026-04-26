"""
Extended voice features beyond the UCI MDVP set.

All of these are standard in the dysphonia / PD-voice literature but are
not in the UCI Parkinson's CSV. They are computed on top of, not in place
of, the original 22 features.

Feature groups added:

  Cepstral Peak Prominence (CPP)
    A single scalar describing how "peaky" the cepstrum is around the
    pitch period. Healthy voices have a strong cepstral peak; dysphonic
    voices do not. This is one of the most robust perceptual-severity
    predictors in the clinical literature (Hillenbrand 1994).

  MFCC summary statistics
    Mean and std of the first 13 mel-frequency cepstral coefficients.
    Captures overall timbre/spectral shape in a compact form.
    Returns 26 features: MFCC_1_mean .. MFCC_13_std.

  Formants F1, F2, F3 (mean + bandwidth)
    Vocal tract resonance frequencies and their bandwidths, via Praat's
    Burg LPC method. PD affects articulation which shows up as formant
    centralisation and widened bandwidths.
    Returns 6 features: F1_mean, F1_bw, F2_mean, F2_bw, F3_mean, F3_bw.

  Spectral tilt (dB/octave)
    Slope of a log-log regression on the long-term average spectrum,
    roughly measuring how much high-frequency energy is present. PD
    typically shows steeper roll-off (more energy loss at high freqs).

Total new features: 1 + 26 + 6 + 1 = 34, giving 22 + 34 = 56 total.
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, List

import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call as praat_call
from scipy.signal import welch
from scipy.stats import linregress


# ---------------------------------------------------------------------------
# Feature name lists - used for CSV column ordering
# ---------------------------------------------------------------------------
MFCC_FEATURES = (
    [f"MFCC_{i+1}_mean" for i in range(13)] +
    [f"MFCC_{i+1}_std" for i in range(13)]
)
FORMANT_FEATURES = ["F1_mean", "F1_bw", "F2_mean", "F2_bw", "F3_mean", "F3_bw"]
EXTRA_FEATURE_NAMES: List[str] = (
    ["CPP"] + MFCC_FEATURES + FORMANT_FEATURES + ["spectral_tilt"]
)


# ---------------------------------------------------------------------------
# CPP - Cepstral Peak Prominence
# ---------------------------------------------------------------------------
def cpp(y: np.ndarray, sr: int, f0_range=(60, 400)) -> float:
    """
    Smoothed CPP following Hillenbrand 1994:
      1. Windowed real-cepstrum of the signal
      2. Average cepstrum across frames
      3. Fit a regression line across the quefrency axis (as baseline)
      4. Peak in the pitch-quefrency range, minus the regression value
         at the peak's quefrency.
    Returned in dB.
    """
    frame_len = int(0.04 * sr)  # 40 ms frames
    hop = int(0.01 * sr)        # 10 ms hop
    if y.size < frame_len:
        return float("nan")

    frames = librosa.util.frame(y, frame_length=frame_len, hop_length=hop).T
    win = np.hanning(frame_len)
    frames = frames * win

    # Real cepstrum per frame: IFFT of log|FFT|
    spec = np.fft.rfft(frames, axis=1)
    log_mag = np.log(np.abs(spec) + 1e-12)
    ceps = np.fft.irfft(log_mag, axis=1)
    # Square -> power cepstrum in dB
    power_ceps_db = 10 * np.log10(ceps ** 2 + 1e-20)

    # Average across frames
    avg_ceps = power_ceps_db.mean(axis=0)

    # Quefrencies in seconds -> index range for the pitch band
    q = np.arange(avg_ceps.size) / sr
    lo = max(1, int(sr / f0_range[1]))
    hi = min(avg_ceps.size - 1, int(sr / f0_range[0]))
    if hi <= lo:
        return float("nan")

    # Linear-regression baseline across full quefrency axis (skip 0)
    valid = slice(1, min(int(sr * 0.025), avg_ceps.size))  # up to 25 ms
    slope, intercept, *_ = linregress(q[valid], avg_ceps[valid])
    baseline = intercept + slope * q

    # Peak in the pitch band, measured as height above the baseline
    diff = avg_ceps[lo:hi] - baseline[lo:hi]
    peak_idx = int(np.argmax(diff))
    return float(diff[peak_idx])


# ---------------------------------------------------------------------------
# MFCC summary stats
# ---------------------------------------------------------------------------
def mfcc_stats(y: np.ndarray, sr: int, n_mfcc: int = 13) -> Dict[str, float]:
    """Mean + std of each of n_mfcc MFCCs. Returns 2*n_mfcc features."""
    if y.size < int(0.1 * sr):
        return {k: float("nan") for k in MFCC_FEATURES}
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)  # (n_mfcc, T)
    means = mfcc.mean(axis=1)
    stds = mfcc.std(axis=1)
    out = {}
    for i in range(n_mfcc):
        out[f"MFCC_{i+1}_mean"] = float(means[i])
        out[f"MFCC_{i+1}_std"] = float(stds[i])
    return out


# ---------------------------------------------------------------------------
# Formants via Praat Burg method
# ---------------------------------------------------------------------------
def formants(sound: parselmouth.Sound, max_formant: float = 5500.0) -> Dict[str, float]:
    """
    Mean F1/F2/F3 and their mean bandwidths (Hz) across voiced frames.

    Uses Praat's `To Formant (burg)` which is the standard method.
    max_formant=5500 Hz is Praat's recommendation for female voices;
    5000 is more typical for male voices but 5500 works for both and is
    what most published pipelines use.
    """
    out = {k: float("nan") for k in FORMANT_FEATURES}
    try:
        # Time step 0.01, max formants 5 (so we get F1-F5, use F1-F3),
        # max_formant=5500, window length 0.025s, pre-emphasis 50 Hz.
        form = praat_call(sound, "To Formant (burg)", 0.01, 5, max_formant, 0.025, 50.0)
    except Exception:
        return out

    times = np.arange(
        form.get_start_time(), form.get_end_time(), 0.01
    )
    for f_idx in (1, 2, 3):
        freqs, bws = [], []
        for t in times:
            try:
                fv = praat_call(form, "Get value at time", f_idx, float(t), "Hertz", "Linear")
                bv = praat_call(form, "Get bandwidth at time", f_idx, float(t), "Hertz", "Linear")
                if np.isfinite(fv) and np.isfinite(bv):
                    freqs.append(fv); bws.append(bv)
            except Exception:
                continue
        if freqs:
            out[f"F{f_idx}_mean"] = float(np.mean(freqs))
            out[f"F{f_idx}_bw"] = float(np.mean(bws))
    return out


# ---------------------------------------------------------------------------
# Spectral tilt (long-term average spectrum slope, dB/octave)
# ---------------------------------------------------------------------------
def spectral_tilt(y: np.ndarray, sr: int, fmin: float = 80.0, fmax: float = 5000.0) -> float:
    """
    Fit a line to log-magnitude vs log-frequency over [fmin, fmax] and
    return the slope in dB/octave.
    """
    nperseg = min(4096, y.size)
    if nperseg < 256:
        return float("nan")
    f, Pxx = welch(y, fs=sr, nperseg=nperseg, noverlap=nperseg // 2)
    mask = (f >= fmin) & (f <= fmax) & (Pxx > 0)
    if mask.sum() < 10:
        return float("nan")
    log_f = np.log2(f[mask])
    log_p_db = 10 * np.log10(Pxx[mask])
    slope, *_ = linregress(log_f, log_p_db)
    return float(slope)  # dB / octave (typically negative for speech)


# ---------------------------------------------------------------------------
# Top-level wrapper
# ---------------------------------------------------------------------------
def extract_extra_features(y: np.ndarray, sr: int, sound: parselmouth.Sound) -> Dict[str, float]:
    """
    Compute all 34 extra features on an already-loaded waveform.

    Designed to be called after the main MDVP extractor, which already
    has y, sr, and the Praat Sound object available.
    """
    feats: Dict[str, float] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            feats["CPP"] = cpp(y, sr)
        except Exception:
            feats["CPP"] = float("nan")

        feats.update(mfcc_stats(y, sr))
        feats.update(formants(sound))

        try:
            feats["spectral_tilt"] = spectral_tilt(y, sr)
        except Exception:
            feats["spectral_tilt"] = float("nan")

    # Sanitise infs / nans
    for k, v in list(feats.items()):
        try:
            f = float(v)
            if math.isnan(f) or math.isinf(f):
                feats[k] = float("nan")
            else:
                feats[k] = f
        except (TypeError, ValueError):
            feats[k] = float("nan")
    return feats
