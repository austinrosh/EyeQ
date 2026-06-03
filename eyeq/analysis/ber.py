"""BER, bathtub curves, and COM from the statistical eye (Shakiba Part II, Sec. IV).

The statistical eye gives the *marginal* voltage PDF; BER needs the distribution
*conditioned on the transmitted level*. For each sampling phase we build the
residual PDF (ISI from the non-main cursors, plus amplitude noise and slope-
converted jitter), place a copy at each PAM level, and integrate the tails that
spill past the decision thresholds:

* horizontal bathtub = SER vs sampling phase  -> eye width at a target BER
* vertical bathtub   = SER vs decision level (worst eye) -> eye height at a target BER
* COM = 20*log10(A_signal / (Q * sigma_n))   (Part I, Eq. 1)

This resolves error rates far below what the Monte Carlo transient eye can reach
(its whole reason for existing), so it's the right tool for compliance numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import erfc, erfcinv

from ..core.pipeline import Pipeline

_FLOOR = 1e-18  # SER floor for log plotting


@dataclass(frozen=True)
class BerResult:
    ser: float                 # symbol error rate at the optimal sampling phase
    ber: float                 # ~ ser / bits_per_symbol (Gray-coded)
    com_db: float              # channel operating margin
    best_phase_ui: float
    target_ber: float
    t_axis: NDArray            # sampling phases [UI]
    h_bathtub: NDArray         # log10(SER) vs phase
    v_eye: NDArray             # decision-level axis for the worst eye [V]
    v_bathtub: NDArray         # log10(SER) vs decision level
    eye_height_v: float        # vertical opening at target_ber [V]
    eye_width_ui: float        # horizontal opening at target_ber [UI]
    # detector provenance (defaults keep the decision-point path's positional construction valid)
    detector: str = "decision"      # "decision" (slicer/DFE eye tail) | "mlsd" (sequence error)
    mlsd_dmin: float = float("nan")  # MLSD minimum distance [V]
    mlsd_truncated: bool = False     # the d_min search hit its node cap (estimate is approximate)


def assess(
    eng,
    pipe: Pipeline,
    sbr=None,
    *,
    target_ber: float = 1e-6,
    v_bins: int = 1024,
    phase_points: int = 65,
) -> BerResult:
    ctx = pipe.ctx
    sbr = sbr or eng.sbr(pipe)
    levels = ctx.levels
    m = levels.size
    level_step = float(levels[1] - levels[0])
    pulse, m0, sps = sbr.sbr, sbr.main_idx, ctx.sps
    m_range = sbr.cursor_k
    main_sel = m_range != 0  # ISI cursors (exclude the main)

    sigma_amp = eng._amplitude_sigma(pipe)
    sigma_t = eng._jitter_sigma_s(pipe, ctx)

    # Voltage grid (exact zero at center) wide enough for the deep tails.
    v_peak = 1.2 * (float(np.sum(np.abs(sbr.cursors))) * float(np.max(np.abs(levels))) + 6 * sigma_amp)
    v_peak = max(v_peak, 1e-6)
    dv = 2.0 * v_peak / v_bins
    v = (np.arange(v_bins) - v_bins // 2) * dv

    phase_offsets = np.round(np.linspace(-0.5, 0.5, phase_points, endpoint=False) * sps).astype(int)
    t_axis = phase_offsets / sps

    sers = np.empty(phase_points)
    residuals = []  # keep per-phase residual + main cursor for the best-phase vertical cut
    for pi, delta in enumerate(phase_offsets):
        center = m0 + delta
        cvals = eng._sample_cursors(pulse, center, m_range, sps)
        c0 = float(cvals[m_range == 0][0])
        residual = eng._convolve_cursor_pdfs(cvals[main_sel], levels, v, dv)
        slope = eng._local_slope(pulse, center, ctx.dt)
        sigma = float(np.hypot(sigma_amp, abs(slope) * sigma_t))
        if sigma > 0:
            residual = np.maximum(eng._gaussian_blur(residual, dv, sigma), 0.0)
        s = residual.sum()
        residual = residual / s if s > 0 else residual
        residuals.append((c0, residual))
        sers[pi] = _symbol_error_rate(residual, v, c0, levels)

    best_pi = int(np.argmin(sers))
    ser = float(max(sers[best_pi], _FLOOR))
    c0_best, res_best = residuals[best_pi]

    # Horizontal bathtub + eye width at target BER.
    h_bathtub = np.log10(np.maximum(sers, _FLOOR))
    eye_width_ui = _opening_width(t_axis, sers, target_ber)

    # Vertical bathtub over the worst eye at the best phase + eye height at target.
    v_eye, v_ser = _vertical_bathtub(res_best, v, c0_best, levels)
    v_bathtub = np.log10(np.maximum(v_ser, _FLOOR))
    eye_height_v = _opening_width(v_eye, v_ser, target_ber)

    com_db = _com(res_best, v, c0_best, level_step, target_ber)
    ber = ser / max(int(np.log2(m)), 1)
    return BerResult(
        ser, ber, com_db, float(t_axis[best_pi]), target_ber,
        t_axis, h_bathtub, v_eye, v_bathtub, eye_height_v, eye_width_ui,
    )


def assess_mlsd(
    eng,
    pipe: Pipeline,
    sbr=None,
    *,
    target_ber: float = 1e-12,
    mlsd_taps: int = 4,
    v_bins: int = 256,
    phase_points: int = 33,
) -> BerResult:
    """MLSD BER via the minimum-distance union bound (see :mod:`eyeq.analysis.mlsd`).

    Returns the *same* :class:`BerResult` as :func:`assess` so the report, bathtub,
    and FEC layers consume it identically — only the computation differs. The
    bathtub curves come from the sequence-error model, not the eye tail.
    """
    from . import mlsd as _mlsd

    ctx = pipe.ctx
    sbr = sbr or eng.sbr(pipe)
    levels = ctx.levels
    bits = max(int(np.log2(levels.size)), 1)
    pulse, m0, sps = sbr.sbr, sbr.main_idx, ctx.sps
    m_range = sbr.cursor_k                      # cursor positions (ascending: pre..main..post)
    sigma = eng._amplitude_sigma(pipe)          # front-end-referred noise std [V]
    L = int(min(max(int(mlsd_taps), 0), _mlsd.l_cap(levels.size)))

    phase_offsets = np.round(np.linspace(-0.5, 0.5, phase_points, endpoint=False) * sps).astype(int)
    t_axis = phase_offsets / sps

    sers = np.empty(phase_points)
    results = []
    for pi, delta in enumerate(phase_offsets):
        h = eng._sample_cursors(pulse, m0 + delta, m_range, sps)  # per-UI taps [V], in position order
        r = _mlsd.sequence_ber(h, levels, sigma, L=L)
        sers[pi] = max(r.ser, _FLOOR)
        results.append(r)

    best_pi = int(np.argmin(sers))
    rb = results[best_pi]
    ser = float(max(rb.ser, _FLOOR))
    ber = ser / bits
    h_bathtub = np.log10(np.maximum(sers, _FLOOR))           # log10(SER) vs phase (timing bathtub)
    eye_width_ui = _opening_width(t_axis, sers, target_ber)

    # Vertical analog: BER vs available margin voltage (no decision threshold under MLSD).
    margin = 0.5 * float(np.sqrt(max(rb.d2_min, 0.0)))       # d_min/2 — the MLSD effective half-opening
    v_eye = np.linspace(0.0, max(1.3 * margin, 1e-6), v_bins)
    if sigma > 0.0:
        v_ser = max(rb.n_events, 1) * 0.5 * erfc(v_eye / (sigma * np.sqrt(2.0)))
    else:
        v_ser = np.where(v_eye > 0.0, _FLOOR, 0.5)
    v_bathtub = np.log10(np.maximum(v_ser, _FLOOR))

    # COM analog: the min-distance margin vs the noise amplitude at the target.
    qinv = float(np.sqrt(2.0) * erfcinv(2.0 * target_ber))
    a_noise = sigma * qinv
    com_db = float(20.0 * np.log10(margin / a_noise)) if (a_noise > 0.0 and margin > 0.0) else 99.0

    return BerResult(
        ser, ber, com_db, float(t_axis[best_pi]), target_ber,
        t_axis, h_bathtub, v_eye, v_bathtub, margin, eye_width_ui,
        detector="mlsd", mlsd_dmin=2.0 * margin, mlsd_truncated=bool(rb.truncated),
    )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _symbol_error_rate(residual, v, c0, levels) -> float:
    """SER = mean over levels of P(residual crosses a decision threshold)."""
    thr = abs(c0) * 0.5 * (levels[:-1] + levels[1:])
    m = levels.size
    total = 0.0
    for i in range(m):
        center = abs(c0) * levels[i]
        pe = 0.0
        if i < m - 1:
            pe += float(residual[v > (thr[i] - center)].sum())       # spill up
        if i > 0:
            pe += float(residual[v < (thr[i - 1] - center)].sum())   # spill down
        total += pe
    return total / m


def _vertical_bathtub(residual, v, c0, levels):
    """SER vs decision level across the worst (smallest-margin) eye."""
    # all eyes share the same residual + margin here, so pick the inner pair.
    i = levels.size // 2 - 1
    lo, hi = abs(c0) * levels[i], abs(c0) * levels[i + 1]
    v_eye = np.linspace(lo, hi, 512)
    ser = np.array([
        0.5 * (float(residual[v > (vt - lo)].sum()) + float(residual[v < (vt - hi)].sum()))
        for vt in v_eye
    ])
    return v_eye, ser


def _opening_width(axis, ser, target):
    """Width of the contiguous region around the minimum where SER < target."""
    below = ser < target
    if not below.any():
        return 0.0
    j = int(np.argmin(ser))
    if not below[j]:
        return 0.0
    lo = j
    while lo > 0 and below[lo - 1]:
        lo -= 1
    hi = j
    while hi < axis.size - 1 and below[hi + 1]:
        hi += 1
    return float(axis[hi] - axis[lo])


def _com(residual, v, c0, level_step, target_ber) -> float:
    """Channel operating margin: 20*log10(A_signal / A_noise).

    A_noise is the noise amplitude at the target SER (Part I, Eq. 1). We use the
    *actual* residual tail quantile rather than the Gaussian Q*sigma_n, since the
    ISI-dominated residual is bounded, not Gaussian (Q*sigma_n is pessimistic and
    would report a negative margin for a wide-open error-free eye).
    """
    a_signal = abs(c0) * level_step / 2.0
    dv = v[1] - v[0]
    ccdf = np.cumsum(residual[::-1])[::-1]  # P(residual >= v)
    pos = np.where(v > 0)[0]
    target_side = target_ber / 2.0  # the inner eye has a tail on each side
    ok = pos[ccdf[pos] <= target_side]
    a_noise = float(v[ok[0]]) if ok.size else float(v[-1])
    return float(20.0 * np.log10(a_signal / max(a_noise, dv)))
