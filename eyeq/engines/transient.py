"""Transient engine — the GetWave-like Monte Carlo path (LTI-only in Phase 2b).

Pushes a symbol stream through the LTI link and folds the result into a 2-D
density eye (a phase x voltage histogram). The LTI response is the single-bit
response (SBR) from the statistical engine, so both engines share exactly the
same LTI path and cursor set — which is what makes the *density eye == statistical
eye* agreement a meaningful normalization check.

Each eye window is built directly from the cursors rather than by convolving the
whole waveform: with ``a`` the symbol stream and ``C[m, j] = SBR[main + m*sps + j]``
the cursor-vs-phase matrix,

    window[k, j] = sum_m a[k - m] * C[m, j]                 = (A @ C)[k, j]

a single matmul over the (~20) cursors — far faster than an N*sps-point FFT, and
using the exact cursor set the statistical eye uses. The nonlinear tail
(DFE/CDR/slicer) and its Numba inner loop arrive in Phase 3; the phase axis
matches the statistical eye ([-0.5, 0.5) UI) so the two can be overlaid directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.pipeline import Pipeline
from .statistical import SbrResult, StatisticalEngine


@dataclass(frozen=True)
class TransientResult:
    t_ui: NDArray            # sampling-phase axis across one UI [-0.5, 0.5)
    v: NDArray               # voltage bin centers [V]
    density: NDArray         # [phase, voltage] accumulated density (per-phase sum = 1)
    eye_height_v: float      # statistical inner-eye opening from the histogram [V]
    best_phase_ui: float
    mse_snr_db: float        # SNR at the decision point
    ser: float               # symbol error rate at the best sampling phase
    n_symbols: int


class TransientEngine:
    """Accumulates a Monte Carlo density eye for the LTI link."""

    def __init__(self, v_bins: int = 256):
        self.v_bins = v_bins
        self._stat = StatisticalEngine()

    def run_batch(
        self,
        pipe: Pipeline,
        *,
        n_symbols: int = 50_000,
        sbr: SbrResult | None = None,
        rng: np.random.Generator | None = None,
        v: NDArray | None = None,
        smooth_bins: float = 2.0,
    ) -> TransientResult:
        ctx = pipe.ctx
        sps = ctx.sps
        half = sps // 2
        sbr = sbr or self._stat.sbr(pipe)
        rng = rng or np.random.default_rng(ctx.rng_seed)

        a, sym_idx = pipe.by_name("source").generate(n_symbols, ctx, rng)

        # Cursor-vs-phase matrix C[m, j] over the SBR's cursor set (k from
        # -pre..+post) — a symbol |m| UI away contributes cursor m.
        m_range = sbr.cursor_k
        pre, post = int(-m_range.min()), int(m_range.max())
        offsets = np.arange(-half, sps - half)
        sbrv, L, main = sbr.sbr, sbr.sbr.size, sbr.main_idx
        gi = main + m_range[:, None] * sps + offsets[None, :]
        ok = (gi >= 0) & (gi < L)
        cmat = np.where(ok, sbrv[np.clip(gi, 0, L - 1)], 0.0)  # [n_cursors, sps]

        # window[k, j] = sum_m a[k-m] C[m, j] over valid symbols k in [post, N-pre);
        # k0 = post keeps a[k-m] in range for m up to +post.
        k0, w = post, n_symbols - post - pre
        a_win = np.stack([a[k0 - m : k0 - m + w] for m in m_range], axis=1)  # [w, n_cursors]
        windows = a_win @ cmat                                              # [w, sps]
        sidx = sym_idx[k0 : k0 + w]

        # Inject amplitude noise; jitter shifts each trace horizontally.
        sigma_v = self._stat._amplitude_sigma(pipe)  # front-end-referred RX noise
        if sigma_v > 0:
            windows = windows + rng.normal(0.0, sigma_v, windows.shape)
        rj_ui = self._jitter_ui(pipe)
        if rj_ui > 0:
            shift = np.round(rng.normal(0.0, rj_ui * sps, w)).astype(int)
            cols = (np.arange(sps)[None, :] - shift[:, None]) % sps
            windows = np.take_along_axis(windows, cols, axis=1)

        if v is None:
            v_peak = 1.1 * float(np.abs(windows).max())
            v = np.linspace(-v_peak, v_peak, self.v_bins)
        nb, dv = v.size, v[1] - v[0]
        main_cursor = sbr.main_cursor
        levels = ctx.levels
        dec_col = self._decision_col(pipe, sps, half)

        # Decision-directed tail (DFE) goes through the Numba kernel; the LTI-only
        # case stays fully vectorized.
        dfe = pipe.by_name("dfe") if "dfe" in pipe.names() else None
        if dfe is not None and dfe.is_active():
            density, mse_snr, ser = self._dfe_eye(
                windows, sidx, ctx, sbr, dfe.taps(), dec_col, v, dv, nb
            )
        else:
            density, mse_snr, ser = self._lti_eye(
                windows, sidx, levels, main_cursor, dec_col, v, dv, nb
            )

        if smooth_bins > 0:
            density = np.stack([self._stat._gaussian_blur(c, dv, smooth_bins * dv) for c in density])
        density = np.maximum(density, 0.0)  # FFT blur leaves tiny negative round-off
        density = density / np.maximum(density.sum(1, keepdims=True), 1e-30)

        eye_h, best_pi = self._stat._eye_height(density, v, main_cursor, levels)
        t_ui = offsets / sps
        return TransientResult(
            t_ui, v, density, eye_h, float(t_ui[best_pi]), float(mse_snr), ser, int(w)
        )

    # -- eye builders ---------------------------------------------------------
    @staticmethod
    def _lti_eye(windows, sidx, levels, main_cursor, dec_col, v, dv, nb):
        sps = windows.shape[1]
        edges = np.concatenate([v - 0.5 * dv, [v[-1] + 0.5 * dv]])
        bidx = np.clip(np.searchsorted(edges, windows, side="right") - 1, 0, nb - 1)
        phase = np.broadcast_to(np.arange(sps), bidx.shape)
        density = np.bincount((phase * nb + bidx).ravel(), minlength=sps * nb)
        density = density.reshape(sps, nb).astype(float)

        samp = windows[:, dec_col]
        ideal = levels[sidx] * main_cursor
        mse_snr = 10.0 * np.log10(np.mean(ideal**2) / max(np.mean((samp - ideal) ** 2), 1e-30))
        dec = np.argmin(np.abs(samp[:, None] - levels[None, :] * main_cursor), axis=1)
        return density, float(mse_snr), float(np.mean(dec != sidx))

    def _dfe_eye(self, windows, sidx, ctx, sbr, taps, dec_col, v, dv, nb):
        from ._kernels import dfe_eye  # lazy: numba only when the DFE runs

        sps = windows.shape[1]
        levels = ctx.levels
        thr = sbr.main_cursor * 0.5 * (levels[:-1] + levels[1:])
        hist2d = np.zeros((sps, nb))
        errors, sum_err2, sum_sig2 = dfe_eye(
            np.ascontiguousarray(windows), levels[sidx], sidx.astype(np.int64),
            np.ascontiguousarray(taps), levels, np.ascontiguousarray(thr),
            int(dec_col), float(sbr.main_cursor), float(v[0]), float(dv), int(nb), hist2d,
        )
        mse_snr = 10.0 * np.log10(sum_sig2 / max(sum_err2, 1e-30))
        return hist2d, float(mse_snr), float(errors) / windows.shape[0]

    @staticmethod
    def _decision_col(pipe, sps, half) -> int:
        ph = 0.0
        if "cdr_slicer" in pipe.names():
            ph = pipe.by_name("cdr_slicer").get("sample_phase_ui")
        return int(np.clip(half + round(ph * sps), 0, sps - 1))

    # -- helpers --------------------------------------------------------------
    @staticmethod
    def _jitter_ui(pipe: Pipeline) -> float:
        try:
            return pipe.by_name("txjitter").get("rj_mui") * 1e-3
        except KeyError:
            return 0.0
