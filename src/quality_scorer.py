"""
Recording quality assessment for the Parkinson's Voice Analyser.

Computes a quality score (0-100) and per-metric flags before running
the main model. Returns structured quality report so the Flask app
can warn users about poor recordings before showing the result.

Metrics assessed:
  1. Duration         — too short means unstable features
  2. SNR              — signal-to-noise ratio estimate
  3. Silence ratio    — too much silence means user stopped early
  4. Voiced fraction  — fraction of frames with detected voicing
  5. Clipping         — signal saturation (mic overload)
  6. F0 detectability — can a fundamental frequency be found?
"""
from __future__ import annotations
from typing import Dict, Any
import numpy as np

try:
    import librosa
    _LIBROSA = True
except ImportError:
    _LIBROSA = False

THRESHOLDS = {
    "duration":         {"min_good": 2.5, "min_warn": 1.5, "weight": 0.25},
    "snr":              {"min_good": 10.0, "min_warn": 4.0, "weight": 0.25},
    "silence_ratio":    {"max_good": 0.30, "max_warn": 0.50, "weight": 0.15},
    "voiced_fraction":  {"min_good": 0.50, "min_warn": 0.30, "weight": 0.20},
    "clipping":         {"max_good": 0.001, "max_warn": 0.01, "weight": 0.10},
    "f0_detectability": {"min_good": 0.40, "min_warn": 0.20, "weight": 0.05},
}

def _duration_score(y, sr):
    dur = len(y) / sr
    t = THRESHOLDS["duration"]
    if dur >= t["min_good"]:
        status, msg, score = "good", f"{dur:.1f}s — sufficient duration", 100
    elif dur >= t["min_warn"]:
        status, msg, score = "warn", f"{dur:.1f}s — slightly short, aim for 3–5s", 60
    else:
        status, msg, score = "poor", f"{dur:.1f}s — too short, hold the sound longer", 20
    return {"value": round(dur, 2), "unit": "seconds", "status": status, "message": msg, "score": score}

def _snr_score(y, sr):
    if not _LIBROSA:
        return {"value": None, "unit": "dB", "status": "warn", "message": "librosa not available", "score": 50}
    frame_len = int(sr * 0.025)
    hop = int(sr * 0.010)
    energy = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop)[0]
    noise_energy = float(np.mean(energy[energy < np.percentile(energy, 20)] ** 2)) + 1e-10
    signal_energy = float(np.mean(energy[energy > np.percentile(energy, 60)] ** 2)) + 1e-10
    snr = 10 * np.log10(signal_energy / noise_energy)
    t = THRESHOLDS["snr"]
    if snr >= t["min_good"]:
        status, msg, score = "good", f"{snr:.1f} dB — clear signal", 100
    elif snr >= t["min_warn"]:
        status, msg, score = "warn", f"{snr:.1f} dB — some background noise", 60
    else:
        status, msg, score = "poor", f"{snr:.1f} dB — high noise, find a quieter room", 20
    return {"value": round(snr, 1), "unit": "dB", "status": status, "message": msg, "score": score}

def _silence_ratio_score(y, sr):
    if not _LIBROSA:
        return {"value": None, "unit": "fraction", "status": "warn", "message": "librosa not available", "score": 50}
    intervals = librosa.effects.split(y, top_db=20)
    voiced = sum(e - s for s, e in intervals)
    ratio = 1 - voiced / max(len(y), 1)
    t = THRESHOLDS["silence_ratio"]
    if ratio <= t["max_good"]:
        status, msg, score = "good", f"{ratio*100:.0f}% silence — good continuous phonation", 100
    elif ratio <= t["max_warn"]:
        status, msg, score = "warn", f"{ratio*100:.0f}% silence — hold the sound more steadily", 60
    else:
        status, msg, score = "poor", f"{ratio*100:.0f}% silence — too many gaps, sustain the vowel", 20
    return {"value": round(ratio, 3), "unit": "fraction", "status": status, "message": msg, "score": score}

def _voiced_fraction_score(y, sr):
    if not _LIBROSA:
        return {"value": None, "unit": "fraction", "status": "warn", "message": "librosa not available", "score": 50}
    try:
        f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr,
            frame_length=int(sr * 0.04), hop_length=int(sr * 0.01))
        voiced_frac = float(np.mean(voiced_flag)) if voiced_flag is not None else 0.0
    except Exception:
        voiced_frac = 0.0
    t = THRESHOLDS["voiced_fraction"]
    if voiced_frac >= t["min_good"]:
        status, msg, score = "good", f"{voiced_frac*100:.0f}% voiced — clear vowel phonation", 100
    elif voiced_frac >= t["min_warn"]:
        status, msg, score = "warn", f"{voiced_frac*100:.0f}% voiced — sustain more consistently", 60
    else:
        status, msg, score = "poor", f"{voiced_frac*100:.0f}% voiced — say aaaah more clearly", 20
    return {"value": round(voiced_frac, 3), "unit": "fraction", "status": status, "message": msg, "score": score}

def _clipping_score(y):
    clipped = float(np.mean(np.abs(y) > 0.99))
    t = THRESHOLDS["clipping"]
    if clipped <= t["max_good"]:
        status, msg, score = "good", "No clipping detected", 100
    elif clipped <= t["max_warn"]:
        status, msg, score = "warn", f"{clipped*100:.2f}% clipping — move mic slightly further", 60
    else:
        status, msg, score = "poor", f"{clipped*100:.1f}% clipping — recording too loud", 20
    return {"value": round(clipped, 5), "unit": "fraction", "status": status, "message": msg, "score": score}

def _f0_detectability_score(y, sr):
    if not _LIBROSA:
        return {"value": None, "unit": "fraction", "status": "warn", "message": "librosa not available", "score": 50}
    try:
        f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr,
            frame_length=int(sr * 0.04), hop_length=int(sr * 0.01))
        if f0 is None or voiced_flag is None:
            return {"value": 0.0, "unit": "fraction", "status": "warn", "message": "F0 not detectable", "score": 30}
        voiced = voiced_flag.astype(bool)
        f0_det = float(np.mean(~np.isnan(f0[voiced]))) if voiced.any() else 0.0
        t = THRESHOLDS["f0_detectability"]
        if f0_det >= t["min_good"]:
            status, msg, score = "good", f"F0 detected in {f0_det*100:.0f}% of voiced frames", 100
        elif f0_det >= t["min_warn"]:
            status, msg, score = "warn", f"F0 in {f0_det*100:.0f}% of frames — irregular pitch", 60
        else:
            status, msg, score = "poor", "Fundamental frequency hard to detect", 20
        return {"value": round(f0_det, 3), "unit": "fraction", "status": status, "message": msg, "score": score}
    except Exception:
        return {"value": None, "unit": "fraction", "status": "warn", "message": "F0 analysis failed", "score": 50}

def assess_quality(wav_path: str) -> Dict[str, Any]:
    if not _LIBROSA:
        return {"overall_score": 50, "overall_status": "warn",
                "overall_message": "Quality assessment unavailable",
                "metrics": {}, "recommendation": "Install librosa for quality scoring."}
    try:
        y, sr = librosa.load(wav_path, sr=None, mono=True)
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000
        y = y.astype(np.float32)
        metrics = {
            "duration":         _duration_score(y, sr),
            "snr":              _snr_score(y, sr),
            "silence_ratio":    _silence_ratio_score(y, sr),
            "voiced_fraction":  _voiced_fraction_score(y, sr),
            "clipping":         _clipping_score(y),
            "f0_detectability": _f0_detectability_score(y, sr),
        }
        total_w = sum(THRESHOLDS[k]["weight"] for k in metrics)
        overall = sum(metrics[k]["score"] * THRESHOLDS[k]["weight"] for k in metrics) / total_w
        poor = sum(1 for m in metrics.values() if m["status"] == "poor")
        warn = sum(1 for m in metrics.values() if m["status"] == "warn")
        if poor >= 2 or overall < 40:
            st = "poor"
        elif poor >= 1 or warn >= 2 or overall < 70:
            st = "warn"
        else:
            st = "good"
        issues = [m["message"] for m in metrics.values() if m["status"] in ("poor", "warn")]
        if not issues:
            rec = "Recording quality is excellent. Results should be reliable."
        elif st == "poor":
            rec = "Recording quality is poor — results may be unreliable. Re-record in a quieter environment, sustain the aaaah sound for 3–5 seconds."
        else:
            rec = "Acceptable quality. For best results: " + "; ".join(issues[:2]) + "."
        msg = {"good": "✓ Good quality recording",
               "warn": "⚠ Fair quality — results may vary",
               "poor": "✗ Poor quality — please re-record"}[st]
        return {"overall_score": round(float(overall), 1), "overall_status": st,
                "overall_message": msg, "metrics": metrics, "recommendation": rec}
    except Exception as e:
        return {"overall_score": 50, "overall_status": "warn",
                "overall_message": f"Quality assessment failed: {e}",
                "metrics": {}, "recommendation": "Could not assess recording quality."}
