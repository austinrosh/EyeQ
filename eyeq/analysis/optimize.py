"""Closed-form MMSE auto-EQ (Shakiba et al., Part II).

One-shot "solve optimal taps" from the SBR cursors, following the decoupled
procedure: optimize the RX FFE for residual ISI, then let the DFE cancel the
remaining post-cursors.

* :func:`mmse_ffe` — MMSE FIR design with a noise-autocorrelation term and a
  main-tap-position search (Eqs. 6-7): ``w = y* X^T (sig_var * X X^T + R_nn)^-1``.
* :func:`solve_dfe` — DFE taps = the post-cursor amplitudes to cancel (volts).
* :func:`optimize_link` — applies RX FFE then DFE to a pipeline in place.

The RX FFE is LTI, so setting its taps reshapes the SBR; the DFE taps are then
read from the post-RX-FFE post-cursors. (TX FFE precursor optimization, Eqs. 2-3,
is a follow-on.)
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


def _noise_var(pipe: Pipeline) -> float:
    try:
        return (pipe.by_name("noise").get("sigma_mvrms") * 1e-3) ** 2
    except KeyError:
        return 0.0


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
    """Co-optimize TX FFE -> RX FFE -> DFE and apply the taps to ``pipe`` in place.

    The TX FFE removes pre-cursors noise-free (at the cost of swing); the MMSE RX
    FFE handles residual ISI but amplifies front-end noise; the DFE cancels the
    remaining post-cursors. There is an optimum TX/RX split (Part II), so we sweep
    a few TX FFE strengths, solve RX FFE + DFE for each, and keep the one with the
    best (resolution-free analytic) post-DFE SNR — auto-EQ never makes the link
    worse by over-de-emphasizing.
    """
    stat = StatisticalEngine()
    ctx = pipe.ctx
    tx, rx, dfe = pipe.by_name("txffe"), pipe.by_name("rxffe"), pipe.by_name("dfe")
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
        w, rx_main, mmse = mmse_ffe(stat.sbr(pipe).cursors, n_rxffe,
                                    sig_var=sig_var, noise_var=_noise_var(pipe))
        rx.set_taps(w, rx_main)
        s1 = stat.sbr(pipe)
        dfe_taps = solve_dfe(s1.cursors[s1.cursor_k > 0], n_dfe)
        dfe.set_taps(dfe_taps)
        return w, rx_main, dfe_taps, mmse

    candidates = [0, 1, 2] if tx_ffe else [0]
    best = None
    for n_tx_pre in candidates:
        apply_tx(n_tx_pre)
        w, rx_main, dfe_taps, mmse = solve_rx_dfe()
        snr = _link_snr_db(stat, pipe, n_dfe)
        cand = (snr, n_tx_pre, tx.taps().copy(), tx.main_pos(), w, rx_main, dfe_taps, mmse)
        if best is None or snr > best[0]:
            best = cand

    _, n_tx_pre, tx_taps, tx_main, w, rx_main, dfe_taps, mmse = best
    apply_tx(n_tx_pre)  # restore the winning TX FFE, then re-apply RX/DFE taps
    rx.set_taps(w, rx_main)
    dfe.set_taps(dfe_taps)
    return EqResult(tx.taps(), tx.main_pos(), w, rx_main, dfe_taps, mmse)
