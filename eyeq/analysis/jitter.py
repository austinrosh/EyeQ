"""Jitter decomposition: RJ / DJ -> TJ(BER) via the dual-Dirac model.

A headless, pure-NumPy analysis layer (like :mod:`eyeq.analysis.fec` /
:mod:`eyeq.analysis.mlsd`). It reads the injected jitter from the ``txjitter``
block, estimates the data-dependent jitter (DDJ) from the channel's ISI, and
combines them into a total jitter at a target BER:

    DJ_pp = DCD_pp + SJ_pp + DDJ_pp
    TJ(BER) = DJ_pp + 2 * Q^-1(BER/2) * RJ_rms      (dual-Dirac)

DDJ is the horizontal analog of the peak-distortion eye-height bound: the
worst-case ISI voltage divided by the steepest edge slope, i.e. how far the
data-dependent zero crossings spread in time. It is 0 for a clean channel and
grows with loss. The *measured* timing margin (``eye_width_ui``, from
:func:`eyeq.analysis.ber.assess`) already folds in every component including
DDJ; the decomposition is the analytic budget breakdown beside it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import erfcinv

from ..core.pipeline import Pipeline


@dataclass(frozen=True)
class JitterResult:
    rj_rms_ui: float       # combined TX+RX random, in quadrature
    dcd_pp_ui: float       # TX duty-cycle distortion
    sj_pp_ui: float        # TX sinusoidal, *post-CDR-tracking* (effective, eye-closing)
    pj_pp_ui: float        # RX periodic, *post-CDR-tracking* (effective, eye-closing)
    ddj_pp_ui: float       # data-dependent (ISI), from the channel
    dj_pp_ui: float        # dcd + sj + pj + ddj (deterministic, bounded)
    tj_ui: float           # dual-Dirac total jitter at target_ber
    eye_width_ui: float    # measured horizontal opening from ber.assess
    rx_rj_rms_ui: float    # RX-clock random component alone (for the report)
    loop_bw_mhz: float     # CDR jitter-transfer corner (0 when not tracking)
    target_ber: float


def _q_inv(ber: float) -> float:
    """Gaussian Q^-1: number of sigmas whose one-sided tail probability is ``ber``."""
    return float(np.sqrt(2.0) * erfcinv(2.0 * max(float(ber), 1e-300)))


def estimate_ddj_ui(eng, pipe: Pipeline, sbr) -> float:
    """Data-dependent jitter [UI pp] = worst-case ISI swing / steepest edge slope.

    The horizontal analog of :meth:`StatisticalEngine._peak_distortion_eye_height`:
    ISI shifts the instantaneous voltage at a transition, and the crossing time moves
    by that voltage divided by the local edge slope. Capped at 1 UI."""
    ctx = pipe.ctx
    pulse = np.asarray(sbr.sbr, float)
    if pulse.size < 2:
        return 0.0
    edge_slope = float(np.abs(np.diff(pulse)).max()) / ctx.dt        # V/s, steepest flank
    if edge_slope <= 0.0:
        return 0.0
    max_level = float(np.max(np.abs(ctx.levels)))
    isi_pp_v = 2.0 * float(sbr.isi_sum) * max_level                  # worst-case ISI swing [V]
    return float(min((isi_pp_v / edge_slope) / ctx.ui, 1.0))


def decompose(eng, pipe: Pipeline, sbr, ber, *, target_ber: float = 1e-12) -> JitterResult:
    """Break the link's timing budget into RJ / DCD / SJ / PJ / DDJ and a dual-Dirac TJ.

    Combines TX (data-edge) and RX (sampling-clock) jitter: RJ adds in quadrature; the
    periodic components (TX SJ, RX PJ) are scaled by the CDR error response ``|1-H(f)|``
    at their frequencies (low-frequency periodic jitter is tracked out — see
    ``cdr_slicer.error_response``). DDJ is estimated from the channel ISI."""
    def get(name, *params):
        try:
            b = pipe.by_name(name)
            return [float(b.get(p)) for p in params]
        except KeyError:
            return [0.0] * len(params)

    tx_rj, tx_dcd, tx_sj, tx_fsj = get("txjitter", "rj_mui", "dcd_mui", "sj_mui", "sj_freq_mhz")
    rx_rj, rx_pj, rx_fpj = get("rxjitter", "rj_mui", "pj_mui", "pj_freq_mhz")
    try:
        cdr = pipe.by_name("cdr_slicer")
        h_tx, h_rx = cdr.error_response(tx_fsj * 1e6), cdr.error_response(rx_fpj * 1e6)
        loop_bw = cdr.get("loop_bw_mhz") if cdr.tracking() else 0.0
    except KeyError:
        h_tx = h_rx = 1.0
        loop_bw = 0.0

    rj_rms = float(np.hypot(tx_rj, rx_rj)) * 1e-3
    dcd_pp = tx_dcd * 1e-3
    sj_pp = 2.0 * tx_sj * h_tx * 1e-3                                # post-tracking, pp = 2*amp
    pj_pp = 2.0 * rx_pj * h_rx * 1e-3
    ddj_pp = estimate_ddj_ui(eng, pipe, sbr)
    dj_pp = dcd_pp + sj_pp + pj_pp + ddj_pp
    tj = dj_pp + 2.0 * _q_inv(target_ber / 2.0) * rj_rms
    eye_w = float(getattr(ber, "eye_width_ui", 0.0)) if ber is not None else 0.0
    return JitterResult(rj_rms, dcd_pp, sj_pp, pj_pp, ddj_pp, dj_pp, tj, eye_w,
                        rx_rj * 1e-3, float(loop_bw), float(target_ber))
