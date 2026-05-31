"""Shared spectral / DSP helpers (NumPy only).

Conventions used throughout EyeQ:

* Transfers are **one-sided** complex arrays on ``ctx.freq_grid()`` (0 .. fs/2,
  length ``n//2 + 1`` for an even FFT length ``n``).
* Time-domain responses are recovered with :func:`transfer_to_impulse`
  (``np.fft.irfft``), matching serdespy's Hermitian-symmetry ``freq2impulse``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_EPS = 1e-300


def to_db(mag: NDArray | complex) -> NDArray:
    """20*log10|x| with a floor to avoid -inf."""
    return 20.0 * np.log10(np.maximum(np.abs(mag), _EPS))


def from_db(db: NDArray | float) -> NDArray:
    return np.asarray(10.0 ** (np.asarray(db, dtype=float) / 20.0))


def transfer_to_impulse(h_onesided: NDArray, n: int) -> NDArray[np.float64]:
    """Inverse real FFT of a one-sided transfer to a length-``n`` real impulse."""
    return np.fft.irfft(h_onesided, n=n)


def minimum_phase_spectrum(mag_onesided: NDArray, n: int) -> NDArray[np.complex128]:
    """Minimum-phase complex spectrum with the given one-sided magnitude.

    Standard homomorphic (cepstral) reconstruction: build the even full-band
    magnitude, take the real cepstrum, fold it with the causal window, and
    exponentiate. The result has exactly the input magnitude and the unique
    minimum-phase (causal, stable) phase — the physically sensible choice for a
    smooth loss model that only specifies |H(f)|.
    """
    m = np.asarray(mag_onesided, dtype=float)
    # Even-symmetric full-band magnitude of length n (indices 0..n/2 then mirror).
    full = np.concatenate([m, m[-2:0:-1]])
    log_mag = np.log(np.maximum(full, _EPS))
    cepstrum = np.fft.ifft(log_mag).real
    window = np.zeros(n)
    window[0] = 1.0
    if n % 2 == 0:
        window[1 : n // 2] = 2.0
        window[n // 2] = 1.0
    else:  # pragma: no cover - EyeQ always uses power-of-two (even) n
        window[1 : (n + 1) // 2] = 2.0
    h_full = np.exp(np.fft.fft(cepstrum * window))
    return h_full[: n // 2 + 1]


def next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p
