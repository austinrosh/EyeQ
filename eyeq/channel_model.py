"""Analytical channel transfer math, shared by the Channel block and the
synthetic Touchstone generator so they agree exactly.

Magnitude is the reach-class loss budget with a skin/dielectric split, anchored
to a *reference* Nyquist (so NRZ and PAM4 sample the same physical curve at
different frequencies). Two phase models:

* ``simple_transfer`` — minimum-phase reconstruction from the magnitude.
* ``tl_transfer``     — transmission-line phase from the physical skin (1+j)sqrt(f)
  and dielectric Kramers-Kronig terms, plus a bulk transport delay.

``reflection_comb`` adds the periodic notches of an impedance discontinuity
(main path + a delayed echo) — the feature MR/LR channels have that the smooth
analytical magnitude cannot reproduce.

Frequency arrays are one-sided and uniform from 0 (so ``minimum_phase_spectrum``
sees a valid even spectrum of length ``2*(len(f)-1)``).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .spectral import from_db, minimum_phase_spectrum

K_SKIN = 0.6  # skin (sqrt f) fraction of the loss budget
K_DIEL = 0.4  # dielectric (f) fraction
_NEPER_PER_DB = 1.0 / 8.685889638


def loss_components_db(f: NDArray, ref_nyq: float, loss_db: float) -> tuple[NDArray, NDArray]:
    x = np.asarray(f, dtype=float) / ref_nyq
    return loss_db * K_SKIN * np.sqrt(x), loss_db * K_DIEL * x


def insertion_loss_db(f: NDArray, ref_nyq: float, loss_db: float) -> NDArray:
    skin, diel = loss_components_db(f, ref_nyq, loss_db)
    return skin + diel


def _full_len(f: NDArray) -> int:
    return 2 * (len(f) - 1)


def simple_transfer(f: NDArray, ref_nyq: float, loss_db: float) -> NDArray[np.complex128]:
    """Minimum-phase transfer with the loss-budget magnitude."""
    mag = from_db(-insertion_loss_db(f, ref_nyq, loss_db))
    return minimum_phase_spectrum(mag, _full_len(f))


def tl_transfer(
    f: NDArray, ref_nyq: float, loss_db: float, delay_s: float
) -> NDArray[np.complex128]:
    """Transmission-line transfer: loss-budget magnitude + physical phase."""
    f = np.asarray(f, dtype=float)
    skin_db, diel_db = loss_components_db(f, ref_nyq, loss_db)
    alpha = (skin_db + diel_db) * _NEPER_PER_DB           # attenuation [nepers]
    a_skin = skin_db * _NEPER_PER_DB
    a_diel = diel_db * _NEPER_PER_DB
    with np.errstate(divide="ignore", invalid="ignore"):
        beta_diel = a_diel * (2.0 / np.pi) * np.log(np.where(f > 0, f / ref_nyq, 1.0))
    beta = a_skin + beta_diel + 2.0 * np.pi * f * delay_s
    return np.exp(-alpha) * np.exp(-1j * beta)


def reflection_comb(f: NDArray, depth: float, spacing_hz: float) -> NDArray[np.complex128]:
    """A delayed echo -> periodic notches spaced ``spacing_hz`` (depth in [0,1))."""
    tau = 1.0 / spacing_hz
    return 1.0 - depth * np.exp(-1j * 2.0 * np.pi * np.asarray(f, dtype=float) * tau)
