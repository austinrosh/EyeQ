"""Numba inner loops for the transient engine.

The genuinely sequential part of the link — the DFE feedback, the slicer, and
the per-UI eye accumulation — lives here as ``njit(nogil=True)`` kernels so the
worker thread runs them concurrently with the GUI. The LTI prefix is applied
*outside* (vectorized, as the cursor-matrix windows); only this decision-directed
tail is sequential. Kernels take flat preallocated arrays (no Python objects) and
cross the Python<->Numba boundary once per batch.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True, nogil=True)
def dfe_eye(windows, a_norm, sym_idx, taps, levels, thr, dec_col, main_cursor,
            v_lo, dv, nb, hist2d, adapt, mu):
    """Run the DFE/slicer over pre-DFE eye windows; accumulate the post-DFE eye.

    windows : [w, sps] pre-DFE per-UI samples (LTI prefix already applied).
    a_norm  : [w] transmitted normalized levels (for the SNR reference).
    sym_idx : [w] transmitted level indices (for error counting).
    taps    : [n_dfe] feedback weights in volts (updated in place if adapting).
    levels  : [M] normalized PAM levels (ascending).
    thr     : [M-1] decision-threshold voltages (= main_cursor * level midpoints).
    dec_col : sampling-phase column for the decision.
    hist2d  : [sps, nb] accumulator (modified in place).
    adapt   : 0 off, 1 LMS, 2 sign-LMS; mu : adaptation step.

    Returns (errors, sum_err2, sum_sig2) for SER and MSE-SNR.
    """
    w, sps = windows.shape
    n_dfe = taps.shape[0]
    m_lev = levels.shape[0]
    d_hist = np.zeros(n_dfe)
    errors = 0
    sum_err2 = 0.0
    sum_sig2 = 0.0

    for k in range(w):
        fb = 0.0
        for i in range(n_dfe):
            fb += taps[i] * d_hist[i]

        samp = windows[k, dec_col] - fb
        di = 0  # slice to nearest level via thresholds
        for t in range(m_lev - 1):
            if samp >= thr[t]:
                di = t + 1
        if di != sym_idx[k]:
            errors += 1

        ideal = a_norm[k] * main_cursor
        e = samp - ideal
        sum_err2 += e * e
        sum_sig2 += ideal * ideal

        # Decision-directed LMS: drive the slicer error toward zero by adjusting
        # the feedback taps. err uses the decision (levels[di]), not the truth.
        if adapt != 0 and mu != 0.0:
            slicer_err = samp - levels[di] * main_cursor
            for i in range(n_dfe):
                if adapt == 1:
                    taps[i] += mu * slicer_err * d_hist[i]
                else:  # sign-LMS
                    se = 1.0 if slicer_err > 0 else -1.0
                    sd = 1.0 if d_hist[i] > 0 else (-1.0 if d_hist[i] < 0 else 0.0)
                    taps[i] += mu * se * sd

        for i in range(n_dfe - 1, 0, -1):
            d_hist[i] = d_hist[i - 1]
        if n_dfe > 0:
            d_hist[0] = levels[di]

        for j in range(sps):
            b = int((windows[k, j] - fb - v_lo) / dv)
            if 0 <= b < nb:
                hist2d[j, b] += 1.0

    return errors, sum_err2, sum_sig2
