"""Smoke test: synthesise a vowel-like signal, extract features, sanity check."""
import os
import sys
import numpy as np
import soundfile as sf

sys.path.insert(0, "/home/claude/pva2")

from src.feature_extractor import extract_features, FEATURE_NAMES
from src.audio_utils import SR_TARGET


def synth_vowel(sr=SR_TARGET, dur=3.0, f0=150.0, jitter_frac=0.005, shimmer_frac=0.03, noise=0.01):
    """Build a rough vowel-ish signal with controllable perturbation."""
    rng = np.random.default_rng(42)
    n = int(sr * dur)
    t = np.arange(n) / sr

    # Add glottal jitter (small period-to-period F0 perturbation)
    f0_contour = f0 + f0 * jitter_frac * rng.standard_normal(n)
    phase = 2 * np.pi * np.cumsum(f0_contour) / sr

    # Sum of harmonics with amplitude shimmer
    y = np.zeros(n)
    for h, amp in [(1, 1.0), (2, 0.5), (3, 0.35), (4, 0.22), (5, 0.15)]:
        a = amp * (1.0 + shimmer_frac * rng.standard_normal(n))
        y += a * np.sin(h * phase)

    # Simple vowel-like formant shaping (very rough)
    y *= 0.5
    y += noise * rng.standard_normal(n)
    y /= np.max(np.abs(y)) + 1e-9
    return y, sr


def main():
    os.makedirs("/tmp/pva_tests", exist_ok=True)
    path = "/tmp/pva_tests/synth_vowel.wav"

    # Healthy-like parameters
    y, sr = synth_vowel(jitter_frac=0.003, shimmer_frac=0.02, noise=0.005)
    sf.write(path, y, sr)

    feats = extract_features(path, compute_d2=False, compute_rpde=True)

    print("=" * 60)
    print(f"{'feature':20s} {'value':>15s}")
    print("=" * 60)
    for name in FEATURE_NAMES:
        v = feats.get(name, float('nan'))
        print(f"{name:20s} {v:15.6f}")
    print("=" * 60)

    # Sanity checks
    checks = []
    def chk(cond, msg): checks.append((bool(cond), msg))

    chk(140 < feats["MDVP:Fo(Hz)"] < 160, "F0 near synthesised 150 Hz")
    chk(feats["MDVP:Flo(Hz)"] <= feats["MDVP:Fo(Hz)"] <= feats["MDVP:Fhi(Hz)"], "Flo <= Fo <= Fhi")
    chk(0 < feats["MDVP:Jitter(%)"] < 0.05, "jitter local in sane range")
    chk(0 < feats["MDVP:Shimmer"] < 0.2, "shimmer local in sane range")
    chk(feats["HNR"] > 10, "HNR high for clean synth")
    chk(0 <= feats["NHR"] <= 1, "NHR in [0,1] range")
    chk(np.isfinite(feats["RPDE"]) and 0 <= feats["RPDE"] <= 1, "RPDE in [0,1]")
    chk(np.isfinite(feats["DFA"]) and 0 < feats["DFA"] < 2, "DFA positive exponent")
    chk(np.isfinite(feats["PPE"]) and 0 <= feats["PPE"] <= 1, "PPE in [0,1]")
    chk(np.isfinite(feats["spread1"]), "spread1 computed")
    chk(np.isfinite(feats["spread2"]), "spread2 computed")

    # The original implementation had aliasing bugs - verify we don't
    chk(feats["Jitter:DDP"] != feats["MDVP:Jitter(%)"], "DDP distinct from Jitter(%)")
    chk(abs(feats["Jitter:DDP"] - 3 * feats["MDVP:RAP"]) / (feats["MDVP:RAP"] + 1e-9) < 0.05,
        "DDP ~= 3 * RAP (Praat identity)")
    chk(feats["Shimmer:APQ3"] != feats["Shimmer:APQ5"], "APQ3 distinct from APQ5")
    chk(abs(feats["Shimmer:DDA"] - 3 * feats["Shimmer:APQ3"]) / (feats["Shimmer:APQ3"] + 1e-9) < 0.05,
        "DDA ~= 3 * APQ3 (Praat identity)")

    passed = sum(1 for ok, _ in checks if ok)
    print(f"\n{passed}/{len(checks)} sanity checks passed")
    for ok, msg in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {msg}")

    return passed == len(checks)


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
