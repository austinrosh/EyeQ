"""Closed-form MMSE auto-EQ (Shakiba et al., Part II).

One-shot "solve optimal taps" from the SBR cursors, following the decoupled
procedure: optimize the RX FFE for residual ISI, then let the DFE cancel the
remaining post-cursors.

* :func:`mmse_ffe` — MMSE FIR design with a noise-autocorrelation term and a
  main-tap-position search (Eqs. 6-7): ``w = y* X^T (sig_var * X X^T + R_nn)^-1``.
* :func:`solve_dfe` — DFE taps = the post-cursor amplitudes to cancel (volts).
* :func:`solve_tx_ffe` — TX FFE that removes pre-cursors (Eqs. 2-3).
* :func:`optimize_link` — sweeps CTLE peaking x TX FFE strength, solving RX FFE +
  DFE for each, and applies the best-SNR combination to a pipeline in place.

The CTLE/TX FFE/RX FFE are all LTI, so setting their taps/poles reshapes the SBR;
the DFE taps are then read from the post-front-end post-cursors. The CTLE does the
bulk of the high-loss equalization before the noise-amplifying RX FFE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.pipeline import Pipeline
from ..engines.statistical import StatisticalEngine


@dataclass(frozen=True)
class EqResult:
    txffe_taps: NDArray
    txffe_main: int
    rxffe_taps: NDArray
    rxffe_main: int
    dfe_taps: NDArray
    mmse: float


def _conv_matrix(x: NDArray, n_taps: int) -> NDArray:
    """Convolution (Toeplitz) matrix X [n_taps, Lx+n_taps-1]: row i is x at offset i."""
    lx = x.size
    lout = lx + n_taps - 1
    mat = np.zeros((n_taps, lout))
    for i in range(n_taps):
        mat[i, i : i + lx] = x
    return mat


def mmse_ffe(
    x: NDArray, n_taps: int, *, sig_var: float, noise_var: float
) -> tuple[NDArray, int, float]:
    """MMSE FIR taps + main-tap position minimizing residual ISI + noise.

    x: pre-FFE pulse cursors (1D, main at argmax). Returns (taps, main_pos, mmse).
    """
    x = np.asarray(x, dtype=float)
    n_taps = max(1, int(n_taps))
    X = _conv_matrix(x, n_taps)
    lout = X.shape[1]
    # Regularized noise autocorrelation (white): keeps the inverse well-posed.
    reg = max(noise_var, 1e-9 * sig_var * float(np.trace(X @ X.T)) / n_taps)
    A_inv = np.linalg.inv(sig_var * (X @ X.T) + reg * np.eye(n_taps))
    x_main = int(np.argmax(np.abs(x)))
    # Target the input's own main-cursor level (not unit), so the FFE flattens
    # ISI without applying an arbitrary ~1/main gain that inflates the signal.
    main_level = float(x[x_main])

    best_w, best_main, best_mmse = None, 0, np.inf
    for ffe_main in range(n_taps):
        out_main = ffe_main + x_main
        if out_main >= lout:
            continue
        y = np.zeros(lout)
        y[out_main] = main_level
        w = y @ X.T @ A_inv
        err = w @ X - y
        mmse = float(sig_var * (err @ err) + reg * (w @ w))
        if mmse < best_mmse:
            best_w, best_main, best_mmse = w, ffe_main, mmse
    return best_w, best_main, best_mmse


def solve_tx_ffe(pre_main: NDArray, n_taps: int) -> tuple[NDArray, int]:
    """MMSE TX FFE that removes pre-cursors (Shakiba et al., Part II, Eqs. 2-3).

    ``pre_main`` is the [pre-cursors .. main] row of the pulse at the CTLE output
    (TX FFE neutralized). Solves ``v = g* H^T (H H^T)^-1`` toward a main-only
    target, normalized by sum(|v|) to preserve the launch swing. Returns
    (taps, main_pos). Post-cursors are deliberately left for the RX FFE/DFE.
    """
    h = np.asarray(pre_main, dtype=float)
    n_taps = max(1, int(n_taps))
    n_h = h.size
    lout = n_taps + n_h - 1
    H = np.zeros((n_taps, lout))
    for i in range(n_taps):
        H[i, i : i + n_h] = h
    main_in = int(np.argmax(np.abs(h)))
    main_out = n_taps - 1 + main_in
    g = np.zeros(lout)
    g[main_out] = h[main_in]
    v = g @ H.T @ np.linalg.inv(H @ H.T)
    v = v / max(float(np.sum(np.abs(v))), 1e-30)  # preserve peak swing
    return v, int(np.argmax(np.abs(v)))


def solve_dfe(post_cursors: NDArray, n_dfe: int) -> NDArray:
    """DFE taps (volts) = the first n_dfe post-cursor amplitudes to cancel."""
    taps = np.zeros(int(n_dfe))
    m = min(int(n_dfe), post_cursors.size)
    taps[:m] = post_cursors[:m]
    return taps


# CTLE peaking is monotonic in a single "aggressiveness" knob alpha in [0,1]: the
# zero sweeps from a high frequency (little boost) down toward DC while the pole
# sweeps up, so more alpha = more Nyquist peaking. A bisection on alpha therefore
# hits any achievable target. The endpoints stay inside the (widened) CTLE bounds.
_FZ_HI, _FZ_LO = 2.0, 0.05
_FP_LO, _FP_HI = 0.5, 4.0


def _ctle_set_alpha(ctle, alpha: float) -> None:
    a = float(np.clip(alpha, 0.0, 1.0))
    ctle.set_params(fz=_FZ_HI - a * (_FZ_HI - _FZ_LO), fp=_FP_LO + a * (_FP_HI - _FP_LO))


def _ctle_max_peaking(ctle, ctx) -> float:
    _ctle_set_alpha(ctle, 1.0)
    return float(ctle.peaking_db(ctx))


def _ctle_set_peaking(ctle, ctx, target_db: float) -> float:
    """Set fz/fp for ~``target_db`` Nyquist peaking by bisection; returns achieved dB."""
    lo, hi = 0.0, 1.0
    for _ in range(28):
        mid = 0.5 * (lo + hi)
        _ctle_set_alpha(ctle, mid)
        if float(ctle.peaking_db(ctx)) < target_db:
            lo = mid
        else:
            hi = mid
    _ctle_set_alpha(ctle, 0.5 * (lo + hi))
    return float(ctle.peaking_db(ctx))


def _ctle_targets(stat: StatisticalEngine, pipe: Pipeline, ctle, ctx) -> list[float]:
    """Candidate Nyquist-peaking levels: none, plus fractions of the channel loss.

    The CTLE flattens the channel, so the right peaking tracks the channel's
    Nyquist insertion loss; we offer a few levels (capped at what the CTLE can
    deliver) and let the SNR sweep pick. 0 dB is always a candidate so a low-loss
    link is never force-peaked into the noise."""
    nyq_loss = max(0.0, float(stat.cascade(pipe).nyquist_loss_db))
    pk_max = _ctle_max_peaking(ctle, ctx)
    levels = {0.0, *(min(f * nyq_loss, pk_max) for f in (0.5, 0.8, 1.0))}
    return sorted({round(v, 3) for v in levels if v >= 0.0})


def _link_snr_db(stat: StatisticalEngine, pipe: Pipeline, n_dfe: int) -> float:
    """Post-DFE decision SNR (analytic, resolution-free) for the TX/RX/DFE sweep.

    Signal = main^2 E[a^2]; distortion = (pre-cursor + uncancelled post-cursor ISI)
    E[a^2] + front-end-referred noise^2. Captures the TX/RX split tradeoff: TX FFE
    cuts pre-cursors and the RX FFE noise gain but loses swing.
    """
    ctx = pipe.ctx
    sbr = stat.sbr(pipe)
    ea2 = float(np.mean(ctx.levels**2))
    pre = sbr.cursors[sbr.cursor_k < 0]
    post = sbr.cursors[sbr.cursor_k > 0]
    uncancelled = post[n_dfe:] if n_dfe < post.size else post[:0]
    sigma = stat._amplitude_sigma(pipe)
    isi = (float(np.sum(pre**2)) + float(np.sum(uncancelled**2))) * ea2
    return 10.0 * np.log10(sbr.main_cursor**2 * ea2 / max(isi + sigma**2, 1e-30))


def optimize_link(
    pipe: Pipeline,
    *,
    n_rxffe: int | None = None,
    n_dfe: int | None = None,
    tx_ffe: bool = True,
) -> EqResult:
    """Co-optimize CTLE -> TX FFE -> RX FFE -> DFE and apply the result in place.

    The CTLE flattens the channel with analogue peaking *before* the noise-amplifying
    RX FFE (so it carries the bulk of the high-loss equalization cheaply); the TX FFE
    removes pre-cursors noise-free (at the cost of swing); the MMSE RX FFE mops up
    residual ISI but amplifies front-end noise; the DFE cancels the remaining
    post-cursors. There is an optimum split (Part II), so we sweep a few CTLE peaking
    levels x TX FFE strengths, solve RX FFE + DFE for each, and keep the combination
    with the best resolution-free analytic post-DFE SNR — auto-EQ never makes the link
    worse by over-peaking (which would just amplify noise) or over-de-emphasizing.
    """
    stat = StatisticalEngine()
    ctx = pipe.ctx
    tx, rx, dfe = pipe.by_name("txffe"), pipe.by_name("rxffe"), pipe.by_name("dfe")
    try:
        ctle = pipe.by_name("ctle")
    except KeyError:
        ctle = None
    n_rxffe = int(n_rxffe or ctx.default_rxffe_taps())
    n_dfe = int(ctx.default_dfe_taps() if n_dfe is None else n_dfe)
    sig_var = float(np.mean(ctx.levels**2))

    def apply_tx(n_tx_pre: int):
        # Always solve the TX FFE on the channel+CTLE pulse with the RX FFE
        # neutral, so the result is independent of any prior candidate's state.
        tx.reset_taps()
        tx.set_params(pre=0.0, post=0.0)
        rx.reset_taps()
        rx.set_params(n_taps=1)
        if n_tx_pre > 0:
            s0 = stat.sbr(pipe)
            v_tx, tx_main = solve_tx_ffe(s0.cursors[s0.cursor_k <= 0], n_tx_pre + 1)
            tx.set_taps(v_tx, tx_main)

    def solve_rx_dfe():
        rx.reset_taps()
        rx.set_params(n_taps=1)
        # Front-end-referred noise: the FFE sees the RX noise *after* the CTLE has
        # shaped (and, when peaking, amplified) it. Regularizing MMSE with this
        # decision-point sigma — not the raw input sigma — stops the FFE from
        # over-equalizing on top of an already-peaked, already-noisy front end.
        nv = float(stat._amplitude_sigma(pipe)) ** 2
        w, rx_main, mmse = mmse_ffe(stat.sbr(pipe).cursors, n_rxffe,
                                    sig_var=sig_var, noise_var=nv)
        rx.set_taps(w, rx_main)
        s1 = stat.sbr(pipe)
        dfe_taps = solve_dfe(s1.cursors[s1.cursor_k > 0], n_dfe)
        dfe.set_taps(dfe_taps)
        return w, rx_main, dfe_taps, mmse

    # CTLE peaking candidates (None = leave the CTLE untouched, if there is none).
    if ctle is not None:
        ctle0 = {"enabled": ctle.get("enabled"), "fz": ctle.get("fz"), "fp": ctle.get("fp")}
        ctle_targets = _ctle_targets(stat, pipe, ctle, ctx)
    else:
        ctle0, ctle_targets = None, [None]

    tx_candidates = [0, 1, 2] if tx_ffe else [0]
    best = None
    for pk in ctle_targets:
        ctle_state = None
        if ctle is not None:
            ctle.set_params(enabled="on")
            _ctle_set_peaking(ctle, ctx, pk)
            ctle_state = {"enabled": "on", "fz": ctle.get("fz"), "fp": ctle.get("fp")}
        for n_tx_pre in tx_candidates:
            apply_tx(n_tx_pre)
            w, rx_main, dfe_taps, mmse = solve_rx_dfe()
            snr = _link_snr_db(stat, pipe, n_dfe)
            cand = (snr, ctle_state, n_tx_pre, w, rx_main, dfe_taps, mmse)
            if best is None or snr > best[0]:
                best = cand

    _, ctle_state, n_tx_pre, w, rx_main, dfe_taps, mmse = best
    if ctle is not None:
        ctle.set_params(**(ctle_state or ctle0))  # restore the winning CTLE
    apply_tx(n_tx_pre)  # re-solve the winning TX FFE on it, then re-apply RX/DFE taps
    rx.set_taps(w, rx_main)
    dfe.set_taps(dfe_taps)
    return EqResult(tx.taps(), tx.main_pos(), w, rx_main, dfe_taps, mmse)
