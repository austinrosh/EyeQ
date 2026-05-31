"""Synthesize realistic reference .s4p channels, one per reach class.

Each file is a valid differential 4-port whose SDD21 follows the reach-class loss
budget (via the shared :mod:`eyeq.channel_model`), so importing a smooth one
reproduces the analytical channel. MR/LR additionally carry reflection notches
(impedance-discontinuity echoes) that the smooth analytical magnitude cannot
reproduce — demonstrating why measured/Touchstone data matters there.

The single-ended 4-port is built from mixed-mode blocks (differential SDD, common
SCC, no mode conversion) using the inverse of the importer's transform, with
ports ordered (1,2) = pair A (+,-), (3,4) = pair B (+,-):

    S_se = M.T @ S_mm @ M          (M is the importer's orthonormal transform)

Users can replace any of these with their own measured .s4p — the importer reads
any 4-port with the same port convention.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .. import channel_model as cm
from ..core.context import REACH_PRESETS

# Reflection notches per (generation, reach): list of (depth, spacing_hz).
# Smooth classes have none; MR/LR get impedance-discontinuity echoes.
_NOTCHES: dict[tuple[str, str], list[tuple[float, float]]] = {
    ("112G", "MR"): [(0.35, 40e9)],
    ("112G", "LR"): [(0.50, 38e9), (0.30, 22e9)],
}

_M = (1.0 / np.sqrt(2.0)) * np.array(
    [[1, -1, 0, 0], [0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 1, 1]], dtype=float
)


def differential_transfer(f: NDArray, generation: str, reach: str) -> NDArray[np.complex128]:
    """SDD21 of the synthetic channel: loss-budget TL magnitude + MR/LR notches."""
    rc = REACH_PRESETS[(generation, reach)]
    delay_s = 4.0 / rc.ref_nyquist_hz  # ~8 UI bulk transport delay
    H = cm.tl_transfer(f, rc.ref_nyquist_hz, rc.loss_db_nyq, delay_s)
    for depth, spacing in _NOTCHES.get((generation, reach), []):
        H = H * cm.reflection_comb(f, depth, spacing)
    return H


def _se_from_mixed_mode(sdd: NDArray, scc: NDArray) -> NDArray[np.complex128]:
    nf = sdd.shape[0]
    s_mm = np.zeros((nf, 4, 4), dtype=complex)
    s_mm[:, 0:2, 0:2] = sdd
    s_mm[:, 2:4, 2:4] = scc
    return _M.T @ s_mm @ _M  # inverse of the importer (M orthonormal)


def build_network(
    generation: str, reach: str, *, fmax: float = 120e9, n: int = 1201, z0: float = 50.0
):
    """Build a scikit-rf 4-port Network for the (generation, reach) channel."""
    import skrf

    rc = REACH_PRESETS[(generation, reach)]
    f = np.linspace(0.0, fmax, n)
    h = differential_transfer(f, generation, reach)

    sdd = np.zeros((n, 2, 2), dtype=complex)
    sdd[:, 0, 1] = sdd[:, 1, 0] = h
    # Differential return loss, kept passive (|S11|^2 + |S21|^2 <= 1).
    rl = np.minimum(0.05, 0.7 * np.sqrt(np.maximum(0.0, 1.0 - np.abs(h) ** 2)))
    sdd[:, 0, 0] = sdd[:, 1, 1] = rl

    # Benign common-mode path (not used by SDD21 extraction).
    hc = cm.tl_transfer(f, rc.ref_nyquist_hz, 6.0, 4.0 / rc.ref_nyquist_hz)
    scc = np.zeros((n, 2, 2), dtype=complex)
    scc[:, 0, 1] = scc[:, 1, 0] = hc

    s_se = _se_from_mixed_mode(sdd, scc)
    net = skrf.Network(s=s_se, frequency=skrf.Frequency.from_f(f, unit="hz"), z0=z0)
    net.name = f"{generation}_{reach}"
    return net


def _safe_name(generation: str, reach: str) -> str:
    return f"{generation}_{reach.replace('+', 'plus')}"


def write_reference_s4p(directory: str | Path, generation: str, reach: str, **kw) -> Path:
    """Write one reference .s4p; returns the written path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    net = build_network(generation, reach, **kw)
    base = directory / _safe_name(generation, reach)
    net.write_touchstone(str(base))  # skrf appends .s4p
    return base.with_suffix(".s4p")


def generate_all(directory: str | Path, generation: str = "112G") -> list[Path]:
    """Write the full reach-class set (XSR..LR) for a generation."""
    return [
        write_reference_s4p(directory, generation, reach)
        for reach in ("XSR", "XSR+", "VSR", "MR", "LR")
    ]
