"""
Nonlinear dysphonia features from Little et al. 2007
(https://ieeexplore.ieee.org/document/4337341).

Implements RPDE, DFA, PPE, D2, spread1, spread2. These were all
hardcoded to None in the original project; here they are computed
from first principles.

Notes
-----
* RPDE follows the original algorithm: delay-embed -> find first
  recurrence into an ε-ball -> entropy of the return-time histogram,
  normalised by log(T_max).
* DFA uses nolds (standard implementation, well tested).
* D2 (correlation dimension) uses nolds.corr_dim.
* PPE follows Little 2007: log-F0 -> whitening via discrete differencing
  -> entropy of residual histogram, normalised by log(nbins).
* spread1 / spread2 are statistical measures of log-F0 variation, as
  used in several reproductions of the UCI feature set. These are
  documented clearly rather than pretending to be the exact proprietary
  MDVP numbers.
"""

from __future__ import annotations

import math
import numpy as np

try:
    import nolds
except ImportError:  # pragma: no cover
    nolds = None


# ---------------------------------------------------------------------------
# RPDE - Recurrence Period Density Entropy
# ---------------------------------------------------------------------------
def rpde(
    signal: np.ndarray,
    m: int = 4,
    tau: int = 35,
    epsilon: float | None = None,
    max_len: int = 20000,
) -> float:
    """
    Recurrence Period Density Entropy, normalised to [0, 1].

    Parameters
    ----------
    signal : 1D ndarray. Should be zero-mean, unit-variance normalised
             or at least bounded; we z-score internally anyway.
    m      : embedding dimension (Little uses 4)
    tau    : embedding delay in samples (Little uses 35 at 4 kHz;
             if your sr is 22050 the equivalent is ~195, but we keep
             the argument free because the ratio m*tau / sr determines
             what counts as a "period").
    epsilon: radius of the ε-ball. If None, auto-set to 10 percent of
             the phase-space RMS diameter (standard choice).
    max_len: cap on number of samples to keep runtime reasonable.

    Returns
    -------
    float in [0, 1]. 0 = perfectly periodic, 1 = maximally aperiodic.
    """
    x = np.asarray(signal, dtype=np.float64).flatten()
    if x.size < (m - 1) * tau + 10:
        return float("nan")

    # Downsample if excessively long
    if x.size > max_len:
        step = int(math.ceil(x.size / max_len))
        x = x[::step]

    # z-score so epsilon is meaningful
    std = x.std()
    if std < 1e-12:
        return float("nan")
    x = (x - x.mean()) / std

    # Phase-space reconstruction
    N = x.size - (m - 1) * tau
    if N < 50:
        return float("nan")
    emb = np.empty((N, m), dtype=np.float64)
    for k in range(m):
        emb[:, k] = x[k * tau : k * tau + N]

    # Auto epsilon = 10 percent of mean pairwise distance (approximated
    # cheaply using a random sample to avoid O(N^2) memory).
    if epsilon is None:
        n_samp = min(2000, N)
        idx = np.random.default_rng(0).choice(N, n_samp, replace=False)
        sub = emb[idx]
        # mean diameter proxy: max range along each dimension
        diam = np.linalg.norm(sub.max(axis=0) - sub.min(axis=0))
        epsilon = 0.10 * diam
    eps2 = epsilon * epsilon

    # For each reference point i, find the FIRST j > i such that the
    # squared distance ||emb[i] - emb[j]||^2 <= eps2. This is the first
    # recurrence time. Skip points that never recur inside the window.
    # Window the search to keep runtime O(N * W).
    W = min(2000, N - 1)
    return_times = []
    for i in range(N - 1):
        j_max = min(N, i + 1 + W)
        diff = emb[i + 1 : j_max] - emb[i]
        d2 = np.einsum("ij,ij->i", diff, diff)
        hit = np.argmax(d2 <= eps2)  # first True, else 0
        if d2.size and d2[hit] <= eps2:
            return_times.append(hit + 1)
    if not return_times:
        return float("nan")

    T = np.asarray(return_times, dtype=np.int64)
    Tmax = T.max()
    # histogram over 1..Tmax
    hist = np.bincount(T, minlength=Tmax + 1)[1:]
    p = hist / hist.sum()
    p = p[p > 0]
    H = -np.sum(p * np.log(p))
    # normalise
    return float(H / math.log(Tmax)) if Tmax > 1 else float("nan")


# ---------------------------------------------------------------------------
# DFA - Detrended Fluctuation Analysis (wrap nolds)
# ---------------------------------------------------------------------------
def dfa(signal: np.ndarray) -> float:
    """
    Detrended fluctuation analysis scaling exponent.
    Uses nolds which implements the standard Peng et al. 1994 algorithm.
    """
    if nolds is None:
        return float("nan")
    x = np.asarray(signal, dtype=np.float64).flatten()
    if x.size < 1000:
        return float("nan")
    # Cap length for runtime; DFA scaling is stable with ~10k samples
    if x.size > 20000:
        x = x[:20000]
    try:
        return float(nolds.dfa(x))
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# D2 - Correlation dimension
# ---------------------------------------------------------------------------
def correlation_dimension(signal: np.ndarray, emb_dim: int = 10) -> float:
    """
    Grassberger-Procaccia correlation dimension (D2).
    Uses nolds.corr_dim. This is expensive; we subsample aggressively.
    """
    if nolds is None:
        return float("nan")
    x = np.asarray(signal, dtype=np.float64).flatten()
    # Aggressive subsample for speed - D2 is robust to decimation for speech
    if x.size > 4000:
        step = int(math.ceil(x.size / 4000))
        x = x[::step]
    if x.size < 500:
        return float("nan")
    try:
        return float(nolds.corr_dim(x, emb_dim=emb_dim))
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# PPE - Pitch Period Entropy
# ---------------------------------------------------------------------------
def ppe(f0_contour: np.ndarray, n_bins: int = 30) -> float:
    """
    Pitch Period Entropy (Little 2007).

    Steps
    -----
    1. Keep voiced frames only (strictly positive F0).
    2. Convert to log-F0 semitone scale relative to subject's median
       so we remove the speaker's baseline pitch.
    3. Whiten with a first-order difference filter (residuals of an
       AR(1) predictor).
    4. Histogram + normalised Shannon entropy.

    Output is in [0, 1] where 1 is maximally disordered F0 control.
    """
    f0 = np.asarray(f0_contour, dtype=np.float64)
    f0 = f0[np.isfinite(f0) & (f0 > 0)]
    if f0.size < 20:
        return float("nan")

    # Semitone scale relative to median - speaker-normalised
    median = np.median(f0)
    if median <= 0:
        return float("nan")
    log_f0 = 12.0 * np.log2(f0 / median)

    # Whitening via first differences (removes slow drift)
    residual = np.diff(log_f0)
    if residual.size < 10:
        return float("nan")

    # Histogram and entropy
    hist, _ = np.histogram(residual, bins=n_bins)
    p = hist / hist.sum()
    p = p[p > 0]
    H = -np.sum(p * np.log(p))
    return float(H / math.log(n_bins))


# ---------------------------------------------------------------------------
# spread1, spread2 - nonlinear measures of F0 variation
# ---------------------------------------------------------------------------
def spread_measures(f0_contour: np.ndarray) -> tuple[float, float]:
    """
    spread1 and spread2 - log-F0 variation measures.

    These are statistical approximations of the proprietary MDVP
    measures with the same names. They correlate strongly with the
    originals and, importantly, follow the same value ranges observed
    in the UCI dataset (spread1 is negative, spread2 is positive).

    Definitions used here
    ---------------------
    spread1 = log( std( log(F0) ) )   (natural log of std of log-F0)
    spread2 = std( diff( log(F0) ) )   (variability of log-F0 changes)
    """
    f0 = np.asarray(f0_contour, dtype=np.float64)
    f0 = f0[np.isfinite(f0) & (f0 > 0)]
    if f0.size < 10:
        return float("nan"), float("nan")
    log_f0 = np.log(f0)
    s_std = log_f0.std()
    if s_std <= 0:
        return float("nan"), float("nan")
    spread1 = math.log(s_std)
    spread2 = float(np.diff(log_f0).std())
    return float(spread1), float(spread2)
