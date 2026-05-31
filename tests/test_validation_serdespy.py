"""Cross-validation against the serdespy reference library.

Phase 1 validates the frequency->impulse convention, which the Phase 2 Touchstone
importer depends on: EyeQ's ``np.fft.irfft`` of a one-sided transfer must match
serdespy's Hermitian-symmetry ``freq2impulse``. Richer s4p/cursor cross-checks
land in Phase 2 alongside the Touchstone importer.
"""

import numpy as np
import pytest

serdespy = pytest.importorskip("serdespy")

from eyeq.core.context import REACH_PRESETS, Modulation, SimContext
from eyeq.io import build_pipeline, default_link_config
from eyeq.spectral import transfer_to_impulse


def _hermitian_onesided(n_half: int, seed: int = 0) -> np.ndarray:
    """A one-sided transfer with real DC and Nyquist bins (exactly invertible)."""
    rng = np.random.default_rng(seed)
    H = rng.standard_normal(n_half) + 1j * rng.standard_normal(n_half)
    H[0] = H[0].real
    H[-1] = H[-1].real
    return H


def test_irfft_matches_serdespy_freq2impulse():
    n_half = 513  # -> full length 1024
    H = _hermitian_onesided(n_half)
    f = np.linspace(0.0, 50e9, n_half)

    ours = transfer_to_impulse(H, n=2 * (n_half - 1))
    theirs, _ = serdespy.freq2impulse(H, f)

    assert ours.shape == theirs.shape
    assert np.allclose(ours, theirs, atol=1e-12)


def test_channel_impulse_uses_serdespy_convention():
    c = SimContext.from_data_rate(
        112.0, Modulation.PAM4, reach=REACH_PRESETS[("112G", "LR")], sps=32
    )
    ch = build_pipeline(default_link_config(reach_class="LR")).by_name("channel")
    H = ch.transfer(c)

    ours = ch.impulse_response(c)
    theirs, _ = serdespy.freq2impulse(H, c.freq_grid())
    assert np.allclose(ours, theirs, atol=1e-12)
