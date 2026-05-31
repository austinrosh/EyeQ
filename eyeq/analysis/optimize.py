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


def optimize_link(
    pipe: Pipeline, *, n_rxffe: int | None = None, n_dfe: int | None = None
) -> EqResult:
    """Solve and apply MMSE RX FFE + DFE taps to ``pipe`` in place."""
    stat = StatisticalEngine()
    ctx = pipe.ctx
    rx = pipe.by_name("rxffe")
    dfe = pipe.by_name("dfe")
    n_rxffe = int(n_rxffe or ctx.default_rxffe_taps())
    n_dfe = int(ctx.default_dfe_taps() if n_dfe is None else n_dfe)

    # 1) RX FFE input pulse (RX FFE as identity), then MMSE-solve the taps.
    rx.reset_taps()
    rx.set_params(n_taps=1)
    x = stat.sbr(pipe).cursors
    sig_var = float(np.mean(ctx.levels**2))
    w, main_pos, mmse = mmse_ffe(x, n_rxffe, sig_var=sig_var, noise_var=_noise_var(pipe))
    rx.set_taps(w, main_pos)

    # 2) DFE cancels the post-cursors of the equalized (post-RX-FFE) pulse.
    sbr1 = stat.sbr(pipe)
    post = sbr1.cursors[sbr1.cursor_k > 0]
    dfe.set_taps(solve_dfe(post, n_dfe))
    return EqResult(w, main_pos, dfe.taps(), mmse)
