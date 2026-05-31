"""Touchstone (.s4p) import -> differential SDD21 -> simulation-grid transfer.

Pipeline:  .s4p -> skrf Network -> mixed-mode SDD21 -> resample/extrapolate onto
``ctx.freq_grid()`` -> (optionally) impulse.

The mixed-mode reduction uses the serdespy-compatible convention with single-
ended ports ordered (1,2) = differential pair A (+,-) and (3,4) = pair B (+,-):

    M = (1/sqrt2) * [[1,-1, 0, 0],
                     [0, 0, 1,-1],
                     [1, 1, 0, 0],
                     [0, 0, 1, 1]]          # rows = modes [dA, dB, cA, cB]
    S_mm = M @ S_se @ M.T,   SDD21 = S_mm[1, 0]

Beyond the file's frequency range the transfer is tapered to zero (a real channel
has no content past its measured band); magnitude/phase are interpolated on the
grid and clipped to passivity (|S| <= 1). Causal phase comes from the measured
data. Requires scikit-rf (the ``sim`` extra).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core.context import SimContext

# Orthonormal mixed-mode transform (serdespy port convention: 1,2 = pair A).
_M = (1.0 / np.sqrt(2.0)) * np.array(
    [[1, -1, 0, 0], [0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 1, 1]], dtype=float
)


def load_network(path: str):
    import skrf

    return skrf.Network(str(path))


def sdd21_from_se(s_se: NDArray) -> NDArray[np.complex128]:
    """Differential SDD21 from a single-ended 4-port S-matrix [nfreq,4,4]."""
    s_mm = _M @ s_se @ _M.T  # broadcast over the leading frequency axis
    return s_mm[:, 1, 0]


def load_sdd21(path: str) -> tuple[NDArray, NDArray[np.complex128]]:
    """Return (f [Hz], SDD21 complex) from a .s4p file."""
    net = load_network(path)
    f = np.asarray(net.f, dtype=float)
    return f, sdd21_from_se(np.asarray(net.s))


def _resample(f_file: NDArray, h_file: NDArray, f_grid: NDArray) -> NDArray[np.complex128]:
    """Interpolate complex SDD21 onto f_grid; taper to 0 above the file band."""
    mag = np.abs(h_file)
    phase = np.unwrap(np.angle(h_file))
    mag_g = np.interp(f_grid, f_file, mag, left=mag[0], right=0.0)
    phase_g = np.interp(f_grid, f_file, phase, left=phase[0], right=phase[-1])
    h = mag_g * np.exp(1j * phase_g)
    over = np.abs(h) > 1.0
    h[over] /= np.abs(h[over])  # enforce passivity
    return h


def s4p_to_transfer(path: str, ctx: SimContext) -> NDArray[np.complex128]:
    """Differential channel transfer on ``ctx.freq_grid()`` from a .s4p file."""
    f_file, h = load_sdd21(path)
    return _resample(f_file, h, ctx.freq_grid())


def s4p_to_impulse(path: str, ctx: SimContext) -> NDArray[np.float64]:
    """Causal impulse response (length ``ctx.fft_len()``) from a .s4p file."""
    return np.fft.irfft(s4p_to_transfer(path, ctx), n=ctx.fft_len())
