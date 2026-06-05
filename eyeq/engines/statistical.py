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

        # Voltage grid sized to the peak-distortion bound (all symbols aligned),
        # with an EXACT zero at index v_bins//2 so the per-cursor ifftshift in the
        # PDF convolution is not off by half a bin (that error accumulates over
        # the ~20 cursors and biases the mean / distorts the spread).
        v_peak = 1.15 * float(np.sum(np.abs(sbr.cursors))) * float(np.max(np.abs(levels)))
        v_peak = max(v_peak, 1e-6)
        dv = 2.0 * v_peak / v_bins
        v = (np.arange(v_bins) - v_bins // 2) * dv

        # Per-phase noise sigma (volts) + timing-jitter components (seconds).
        sigma_amp = self._amplitude_sigma(pipe)
        jit = self._jitter_params(pipe, ctx)

        # Cursor offsets m: a symbol that is |m| UI away contributes cursor m.
        # Must match the SBR's cursor convention (k from -pre..+post) exactly, or
        # the postcursor ISI is mis-counted.
        m_range = sbr.cursor_k

        # Integer sample offsets for each sampling phase across one UI.
        phase_offsets = np.round(np.linspace(-0.5, 0.5, phase_points, endpoint=False) * sps).astype(int)
        t_axis = phase_offsets / sps

        pdf = np.zeros((phase_points, v_bins))
        for pi, delta in enumerate(phase_offsets):
            cvals = self._sample_cursors(pulse, m0 + delta, m_range, sps)
            col = self._convolve_cursor_pdfs(cvals, levels, v, dv)
            # local slope (V per second) at this phase -> timing jitter as amplitude noise
            slope = self._local_slope(pulse, m0 + delta, ctx.dt)
            pdf[pi] = self._blur_jitter(col, dv, slope, sigma_amp, jit)

        eye_h, best_pi = self._eye_height(pdf, v, sbr.main_cursor, levels)
        pda = self._peak_distortion_eye_height(sbr, levels)
        return StatEyeResult(t_axis, v, pdf, eye_h, pda, float(t_axis[best_pi]))

    def compute(self, pipe: Pipeline, **eye_kw):
        cascade = self.cascade(pipe)
        sbr = self.sbr(pipe, cascade)
        eye = self.stat_eye(pipe, sbr, **eye_kw)
        return cascade, sbr, eye

    def decision_snr_db(self, pipe: Pipeline, sbr: SbrResult | None = None) -> float:
        """Analytic decision-point SNR (no DFE): main^2 / (residual ISI + noise).

        signal = main_cursor^2 * E[a^2];  distortion = sum(isi^2) * E[a^2] + sigma^2.
        The transient engine's measured MSE-SNR converges to this — a clean
        cross-engine scaling check.
        """
        ctx = pipe.ctx
        sbr = sbr or self.sbr(pipe)
        ea2 = float(np.mean(ctx.levels**2))
        sigma_v = self._amplitude_sigma(pipe)  # front-end-referred RX noise
        signal = sbr.main_cursor**2 * ea2
        distortion = float(np.sum(sbr.isi_cursors**2)) * ea2 + sigma_v**2
        return 10.0 * np.log10(signal / max(distortion, 1e-30))

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
        # Circular convolution with a zero-centered Gaussian kernel. The kernel is
        # built centered at index n//2 and ifftshift'd so its peak sits at index 0;
        # the result needs NO final fftshift (a spurious one shifts the whole
        # distribution by n//2 and collapses the apparent variance).
        n = col.size
        x = (np.arange(n) - n // 2) * dv
        k = np.exp(-0.5 * (x / sigma_v) ** 2)
        k /= k.sum()
        K = np.fft.fft(np.fft.ifftshift(k))
        return np.fft.ifft(np.fft.fft(col) * K).real

    @staticmethod
    def _local_slope(pulse, center, dt) -> float:
        a = pulse[center - 1] if center - 1 >= 0 else pulse[center]
        b = pulse[center + 1] if center + 1 < pulse.size else pulse[center]
        return float((b - a) / (2.0 * dt))

    @staticmethod
    def _amplitude_sigma(pipe: Pipeline) -> float:
        """Decision-point noise std: RX-input-referred noise x front-end gain.

        RX noise enters at the receiver input and is amplified by the front-end
        (CTLE x RX FFE), so an equalizing RX FFE pays a noise penalty — whereas a
        TX FFE operates on clean symbols and does not. Modeling this is what lets
        the TX/RX equalization split be optimized meaningfully.
        """
        try:
            sigma = pipe.by_name("noise").get("sigma_mvrms") * 1e-3
        except KeyError:
            return 0.0
        if sigma <= 0.0:
            return 0.0
        ctx = pipe.ctx
        f = ctx.freq_grid()
        band = f <= ctx.f_nyq
        h = np.ones(f.size, dtype=np.complex128)
        for name in ("ctle", "rxffe"):
            try:
                h = h * pipe.by_name(name).transfer(ctx)
            except KeyError:
                pass
        gain = float(np.sqrt(np.mean(np.abs(h[band]) ** 2)))
        return sigma * gain

    @staticmethod
    def _jitter_params(pipe: Pipeline, ctx):
        """Combined TX + RX timing jitter (seconds): ``(rj_rms, dcd_pp, [periodic_amps])``.

        TX (data-edge) and RX (sampling-clock) jitter combine: **RJ** in quadrature
        (broadband, fully eye-closing); **DCD** from TX; the **periodic** components
        (TX SJ at f_sj, RX PJ at f_pj) each scaled by the CDR error response
        ``|1-H(f)|`` (``cdr_slicer.error_response``) — low-frequency periodic jitter is
        tracked out, so only the part above the loop bandwidth survives. In static CDR
        mode |1-H|=1 and this reduces to the TX-only RJ/DCD/SJ of before."""
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
        except KeyError:
            h_tx = h_rx = 1.0

        s = 1e-3 * ctx.ui
        rj_s = float(np.hypot(tx_rj, rx_rj)) * s
        dcd_s = tx_dcd * s
        periodic = [a * s for a in (tx_sj * h_tx, rx_pj * h_rx) if a > 0.0]
        return rj_s, dcd_s, periodic

    def _blur_jitter(self, col, dv, slope, sigma_amp, jit) -> NDArray:
        """Blur an amplitude PDF by amplitude noise + RJ (Gaussian, in quadrature),
        then convolve the bounded DCD (2-Dirac) and each periodic (arcsine) timing
        jitter component, slope-converted to volts. Returns a normalized column. With
        no DCD and one full-amplitude SJ this is exactly the old RJ/DCD/SJ math."""
        rj_s, dcd_s, periodic_s = jit
        out = col
        sigma_g = float(np.hypot(sigma_amp, abs(slope) * rj_s))
        if sigma_g > 0.0:
            out = np.maximum(self._gaussian_blur(out, dv, sigma_g), 0.0)
        sl = abs(float(slope))
        if dcd_s > 0.0 and sl * dcd_s > dv:                 # DCD: ± half-pp in volts
            out = self._convolve_kernel(out, self._dcd_kernel(out.size, dv, sl * dcd_s))
        for amp_s in periodic_s:                            # SJ / PJ: arcsine of amplitude in volts
            if sl * amp_s > dv:
                out = self._convolve_kernel(out, self._arcsine_kernel(out.size, dv, sl * amp_s))
        s = out.sum()
        return out / s if s > 0 else out

    @staticmethod
    def _convolve_kernel(col, kernel) -> NDArray:
        """Circular convolution with a zero-centered kernel (peak at index n//2),
        ifftshift'd like :meth:`_gaussian_blur` so there is no net shift."""
        K = np.fft.fft(np.fft.ifftshift(kernel))
        return np.maximum(np.fft.ifft(np.fft.fft(col) * K).real, 0.0)

    @staticmethod
    def _dcd_kernel(n, dv, pp_v) -> NDArray:
        """Zero-centered 2-Dirac at ±pp_v/2 (duty-cycle distortion), grid-interpolated."""
        k = np.zeros(n)
        for sign in (-1.0, 1.0):
            pos = sign * 0.5 * pp_v / dv + n // 2
            i = int(np.floor(pos))
            frac = pos - i
            if 0 <= i < n:
                k[i] += 0.5 * (1.0 - frac)
            if 0 <= i + 1 < n:
                k[i + 1] += 0.5 * frac
        return k

    @staticmethod
    def _arcsine_kernel(n, dv, amp_v) -> NDArray:
        """Zero-centered arcsine PDF over [-amp_v, amp_v] (sinusoidal jitter), built by
        binning the arcsine CDF F(u)=1/2+asin(u)/pi so the endpoint mass is captured."""
        x = (np.arange(n) - n // 2) * dv
        lo = np.clip((x - 0.5 * dv) / amp_v, -1.0, 1.0)
        hi = np.clip((x + 0.5 * dv) / amp_v, -1.0, 1.0)
        k = (np.arcsin(hi) - np.arcsin(lo)) / np.pi
        s = k.sum()
        return k / s if s > 0 else k

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
    def _opening_at(col: NDArray, v: NDArray, v_thr: float, lim: float) -> float:
        """Width [V] of the contiguous open (col < lim) interval around v_thr."""
        dv = v[1] - v[0]
        j = int(np.clip(round((v_thr - v[0]) / dv), 0, v.size - 1))
        if col[j] >= lim:
            return 0.0
        lo = j
        while lo > 0 and col[lo - 1] < lim:
            lo -= 1
        hi = j
        while hi < v.size - 1 and col[hi + 1] < lim:
            hi += 1
        return (hi - lo) * dv

    def _eye_opening_in_col(
        self, col: NDArray, v: NDArray, main_cursor: float, levels: NDArray, frac: float = 1e-3
    ) -> float:
        """Worst inner-eye opening over the M-1 thresholds for a single phase column.

        The thresholds sit at the level midpoints scaled by the main cursor
        (``main_cursor * (L_i + L_{i+1}) / 2``); measuring the open gap *around a
        known threshold* is robust to the ragged ISI comb a Monte Carlo histogram
        shows (unlike cluster-gap detection), so the statistical PDF and the
        transient histogram score consistently.
        """
        thr_v = abs(main_cursor) * 0.5 * (levels[:-1] + levels[1:])
        lim = col.max() * frac
        return min((self._opening_at(col, v, tv, lim) for tv in thr_v), default=0.0)

    def _eye_height(
        self, pdf: NDArray, v: NDArray, main_cursor: float, levels: NDArray, frac: float = 1e-3
    ) -> tuple[float, int]:
        """Best inner-eye opening across all sampling phases (height, best column)."""
        best_h, best_pi = 0.0, 0
        for pi in range(pdf.shape[0]):
            inner = self._eye_opening_in_col(pdf[pi], v, main_cursor, levels, frac)
            if inner > best_h:
                best_h, best_pi = inner, pi
        return best_h, best_pi

    def _eye_height_at_col(
        self, pdf: NDArray, v: NDArray, main_cursor: float, levels: NDArray, col: int,
        frac: float = 1e-3,
    ) -> float:
        """Inner-eye opening at a *specific* sampling phase column (e.g. the CDR phase)."""
        col = int(np.clip(col, 0, pdf.shape[0] - 1))
        return self._eye_opening_in_col(pdf[col], v, main_cursor, levels, frac)
