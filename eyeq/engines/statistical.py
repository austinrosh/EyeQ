"""Statistical engine — the Init-like, fast, deterministic path.

Given a configured :class:`~eyeq.core.pipeline.Pipeline` it produces, on any
LTI/STRUCTURAL change:

1. :meth:`StatisticalEngine.cascade` — the three frequency-cascade traces
   (Channel, TX+Channel, TX+Channel+RX) by multiplying per-block transfers on
   the shared ``ctx.freq_grid()``.
2. :meth:`StatisticalEngine.sbr` — the single-bit/pulse response (volts) and the
   sampled cursors (main + pre/post), from the total LTI transfer.
3. :meth:`StatisticalEngine.stat_eye` — a peak-distortion-analysis statistical
   eye: at each sampling phase across one UI, the received-voltage PDF is the
   convolution of every cursor's symbol distribution (uniform over PAM levels),
   then blurred by amplitude noise and slope-converted timing jitter
   (Shakiba et al., Part II, Sec. IV).

Pure NumPy/SciPy; no rate branches (rate enters only through ``ctx`` sizes).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.pipeline import Pipeline
from ..spectral import to_db


@dataclass(frozen=True)
class CascadeResult:
    f: NDArray              # one-sided frequency grid [Hz]
    f_over_fnyq: NDArray
    H_channel: NDArray      # complex; channel (+ package)
    H_tx_chan: NDArray      # TX FFE x channel
    H_tx_chan_rx: NDArray   # TX FFE x channel x CTLE x RX FFE (to decision point)
    nyquist_loss_db: float  # channel insertion loss at Nyquist, relative to DC

    def mag_db(self, which: str = "H_tx_chan_rx") -> NDArray:
        return to_db(getattr(self, which))


@dataclass(frozen=True)
class SbrResult:
    t_ui: NDArray           # time axis [UI], 0 at the main cursor
    sbr: NDArray            # pulse response [V]
    cursors: NDArray        # cursor samples [V], ordered pre..main..post
    cursor_k: NDArray       # integer UI index of each cursor (0 = main)
    main_idx: int           # index of the main cursor within sbr
    sps: int

    @property
    def main_cursor(self) -> float:
        return float(self.cursors[np.where(self.cursor_k == 0)[0][0]])

    @property
    def isi_cursors(self) -> NDArray:
        return self.cursors[self.cursor_k != 0]

    @property
    def isi_sum(self) -> float:
        return float(np.sum(np.abs(self.isi_cursors)))


@dataclass(frozen=True)
class StatEyeResult:
    t_ui: NDArray           # sampling-phase axis across one UI [-0.5, 0.5)
    v: NDArray              # voltage bin centers [V]
    pdf: NDArray            # [phase, voltage] probability density
    eye_height_v: float     # statistical inner-eye opening at the best phase [V]
    pda_bound_v: float      # deterministic peak-distortion bound (pessimistic) [V]
    best_phase_ui: float    # sampling phase that maximizes the inner-eye opening


class StatisticalEngine:
    """Computes cascade, SBR, and the statistical eye for a pipeline."""

    # ---- frequency cascade --------------------------------------------------
    def cascade(self, pipe: Pipeline) -> CascadeResult:
        ctx = pipe.ctx
        f = ctx.freq_grid()
        one = np.ones(f.size, dtype=np.complex128)

        def H(name: str) -> NDArray:
            try:
                blk = pipe.by_name(name)
            except KeyError:
                return one
            return blk.transfer(ctx) if hasattr(blk, "transfer") else one

        tx, ch, ctle, rx = H("txffe"), H("channel"), H("ctle"), H("rxffe")
        H_channel = ch
        H_tx_chan = tx * ch
        H_tx_chan_rx = tx * ch * ctle * rx

        nyq = int(np.argmin(np.abs(f - ctx.f_nyq)))
        nyq_loss = float(-(to_db(ch[nyq]) - to_db(ch[0])))
        return CascadeResult(
            f, ctx.f_over_fnyq(f), H_channel, H_tx_chan, H_tx_chan_rx, nyq_loss
        )

    # ---- single-bit / pulse response ---------------------------------------
    def sbr(self, pipe: Pipeline, cascade: CascadeResult | None = None) -> SbrResult:
        ctx = pipe.ctx
        cascade = cascade or self.cascade(pipe)
        sps = ctx.sps

        # Impulse to the decision point, then a 1-UI hold -> pulse response.
        impulse = np.fft.irfft(cascade.H_tx_chan_rx, n=ctx.fft_len())
        pulse = np.convolve(impulse, np.ones(sps))

        # Scale to volts: the outermost PAM level launches +/- swing/2.
        try:
            swing = pipe.by_name("txffe").get("swing")
        except KeyError:
            swing = 1.0
        pulse = pulse * (swing / 2.0)

        main_idx = int(np.argmax(pulse))
        pre, post = ctx.cursor_span()
        ks = np.arange(-pre, post + 1)
        idx = main_idx + ks * sps
        valid = (idx >= 0) & (idx < pulse.size)
        cursors = np.where(valid, pulse[np.clip(idx, 0, pulse.size - 1)], 0.0)

        t_ui = (np.arange(pulse.size) - main_idx) / sps
        return SbrResult(t_ui, pulse, cursors, ks, main_idx, sps)

    # ---- statistical (PDA) eye ---------------------------------------------
    def stat_eye(
        self,
        pipe: Pipeline,
        sbr: SbrResult | None = None,
        *,
        v_bins: int = 256,
        phase_points: int = 32,
    ) -> StatEyeResult:
        ctx = pipe.ctx
        sbr = sbr or self.sbr(pipe)
        levels = ctx.levels
        pulse = sbr.sbr
        sps = sbr.sps
        m0 = sbr.main_idx

        # Voltage grid sized to the peak-distortion bound (all symbols aligned).
        v_peak = 1.15 * float(np.sum(np.abs(sbr.cursors))) * float(np.max(np.abs(levels)))
        v_peak = max(v_peak, 1e-6)
        v = np.linspace(-v_peak, v_peak, v_bins)
        dv = v[1] - v[0]

        # Per-phase noise/jitter sigmas (volts).
        sigma_amp = self._amplitude_sigma(pipe)
        sigma_t = self._jitter_sigma_s(pipe, ctx)

        pre = int(-sbr.cursor_k.min())
        post = int(sbr.cursor_k.max())
        m_range = np.arange(-post, pre + 1)  # cursor offsets (samples = m*sps)

        # Integer sample offsets for each sampling phase across one UI.
        phase_offsets = np.round(np.linspace(-0.5, 0.5, phase_points, endpoint=False) * sps).astype(int)
        t_axis = phase_offsets / sps

        pdf = np.zeros((phase_points, v_bins))
        for pi, delta in enumerate(phase_offsets):
            cvals = self._sample_cursors(pulse, m0 + delta, m_range, sps)
            col = self._convolve_cursor_pdfs(cvals, levels, v, dv)
            # local slope (V per second) at this phase -> jitter as amplitude noise
            slope = self._local_slope(pulse, m0 + delta, ctx.dt)
            sigma = np.hypot(sigma_amp, abs(slope) * sigma_t)
            if sigma > 0:
                col = self._gaussian_blur(col, dv, sigma)
            s = col.sum()
            pdf[pi] = col / s if s > 0 else col

        eye_h, best_pi = self._statistical_eye_height(pdf, dv)
        pda = self._peak_distortion_eye_height(sbr, levels)
        return StatEyeResult(t_axis, v, pdf, eye_h, pda, float(t_axis[best_pi]))

    def compute(self, pipe: Pipeline, **eye_kw):
        cascade = self.cascade(pipe)
        sbr = self.sbr(pipe, cascade)
        eye = self.stat_eye(pipe, sbr, **eye_kw)
        return cascade, sbr, eye

    # ---- helpers ------------------------------------------------------------
    @staticmethod
    def _sample_cursors(pulse, center, m_range, sps) -> NDArray:
        idx = center + m_range * sps
        valid = (idx >= 0) & (idx < pulse.size)
        return np.where(valid, pulse[np.clip(idx, 0, pulse.size - 1)], 0.0)

    @staticmethod
    def _cursor_pdf(cval, levels, v, dv) -> NDArray:
        """Distribution of cval*L for L uniform over PAM levels, linearly binned."""
        pdf = np.zeros(v.size)
        w = 1.0 / levels.size
        for L in levels:
            pos = (cval * L - v[0]) / dv
            i = int(np.floor(pos))
            frac = pos - i
            if 0 <= i < v.size:
                pdf[i] += w * (1.0 - frac)
            if 0 <= i + 1 < v.size:
                pdf[i + 1] += w * frac
        return pdf

    def _convolve_cursor_pdfs(self, cvals, levels, v, dv) -> NDArray:
        """Convolve every cursor's symbol distribution on the centered grid."""
        n = v.size
        acc = np.ones(n, dtype=complex)
        for c in cvals:
            p = self._cursor_pdf(c, levels, v, dv)
            acc *= np.fft.fft(np.fft.ifftshift(p))
        out = np.fft.fftshift(np.fft.ifft(acc).real)
        return np.maximum(out, 0.0)

    @staticmethod
    def _gaussian_blur(col, dv, sigma_v) -> NDArray:
        n = col.size
        half = n // 2
        x = (np.arange(n) - half) * dv
        k = np.exp(-0.5 * (x / sigma_v) ** 2)
        k /= k.sum()
        K = np.fft.fft(np.fft.ifftshift(k))
        return np.fft.fftshift(np.fft.ifft(np.fft.fft(col) * K).real)

    @staticmethod
    def _local_slope(pulse, center, dt) -> float:
        a = pulse[center - 1] if center - 1 >= 0 else pulse[center]
        b = pulse[center + 1] if center + 1 < pulse.size else pulse[center]
        return float((b - a) / (2.0 * dt))

    @staticmethod
    def _amplitude_sigma(pipe: Pipeline) -> float:
        try:
            return pipe.by_name("noise").get("sigma_mvrms") * 1e-3
        except KeyError:
            return 0.0

    @staticmethod
    def _jitter_sigma_s(pipe: Pipeline, ctx) -> float:
        rj_mui = 0.0
        for name in ("txjitter",):
            try:
                rj_mui = max(rj_mui, pipe.by_name(name).get("rj_mui"))
            except KeyError:
                pass
        return rj_mui * 1e-3 * ctx.ui

    @staticmethod
    def _peak_distortion_eye_height(sbr: SbrResult, levels) -> float:
        """Deterministic worst-case inner-eye opening [V] (level step x main - all ISI).

        Pessimistic: assumes every interfering symbol takes its worst value at
        once. Goes to zero well before the *statistical* eye closes.
        """
        level_step = float(levels[1] - levels[0]) if levels.size > 1 else 2.0
        opening = abs(sbr.main_cursor) * level_step - 2.0 * sbr.isi_sum
        return max(0.0, opening)

    @staticmethod
    def _eye_openings(col: NDArray, dv: float, frac: float = 1e-3) -> list[float]:
        """Interior open-gap widths [V] between density clusters in one column."""
        thr = col.max() * frac
        idx = np.where(col >= thr)[0]  # "closed" (cluster) bins
        if idx.size < 2:
            return []
        groups = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        return [(b[0] - a[-1] - 1) * dv for a, b in zip(groups[:-1], groups[1:])]

    def _statistical_eye_height(self, pdf: NDArray, dv: float) -> tuple[float, int]:
        """Worst (inner) eye opening at the phase that maximizes it.

        At each sampling phase the PDF separates into PAM-level clusters; the
        inner eye is the smallest gap between adjacent clusters. The eye height
        is that inner gap at the best sampling phase. Unlike the peak-distortion
        bound, this reflects the *probabilistic* opening an eye diagram shows.
        """
        best_h, best_pi = 0.0, 0
        for pi in range(pdf.shape[0]):
            gaps = self._eye_openings(pdf[pi], dv)
            inner = min(gaps) if gaps else 0.0
            if inner > best_h:
                best_h, best_pi = inner, pi
        return best_h, best_pi
