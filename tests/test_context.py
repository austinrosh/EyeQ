"""SimContext: basic rate math + the loss-driven, rate-agnostic derived sizes."""

import numpy as np
import pytest

from eyeq.core import Modulation, REACH_PRESETS, SimContext

VSR = REACH_PRESETS[("112G", "VSR")]


def test_basic_rate_math_pam4():
    ctx = SimContext.from_data_rate(112.0, Modulation.PAM4, reach=VSR, sps=32)
    assert ctx.fb == pytest.approx(56e9)
    assert ctx.f_nyq == pytest.approx(28e9)
    assert ctx.data_rate == pytest.approx(112e9)
    assert ctx.fs == pytest.approx(56e9 * 32)
    assert ctx.ui == pytest.approx(1 / 56e9)


def test_basic_rate_math_nrz():
    ctx = SimContext.from_data_rate(112.0, Modulation.NRZ, reach=VSR, sps=32)
    assert ctx.fb == pytest.approx(112e9)
    assert ctx.f_nyq == pytest.approx(56e9)
    assert ctx.data_rate == pytest.approx(112e9)


def test_levels():
    assert np.allclose(
        SimContext.from_data_rate(112, Modulation.NRZ, reach=VSR).levels, [-1, 1]
    )
    assert np.allclose(
        SimContext.from_data_rate(112, Modulation.PAM4, reach=VSR).levels,
        [-1, -1 / 3, 1 / 3, 1],
    )


# Golden derived sizes for 112G VSR (loss budget L = 16 dB, sps = 32).
# These are the §8 formula outputs; tune the formulas -> update these.
EXPECTED_VSR_SIZES = {
    "sbr_len_ui": 27,
    "sbr_len_samples": 27 * 32,
    "cursor_span": (4, 12),
    "txffe_taps": (1, 2),
    "rxffe_taps": 9,
    "dfe_taps": 10,
    "fft_len": 4096,
}


@pytest.mark.parametrize("mod", [Modulation.NRZ, Modulation.PAM4])
def test_derived_sizes_match_golden(mod):
    ctx = SimContext.from_data_rate(112.0, mod, reach=VSR, sps=32)
    assert ctx.sbr_len_ui() == EXPECTED_VSR_SIZES["sbr_len_ui"]
    assert ctx.sbr_len_samples() == EXPECTED_VSR_SIZES["sbr_len_samples"]
    assert ctx.cursor_span() == EXPECTED_VSR_SIZES["cursor_span"]
    assert ctx.default_txffe_taps() == EXPECTED_VSR_SIZES["txffe_taps"]
    assert ctx.default_rxffe_taps() == EXPECTED_VSR_SIZES["rxffe_taps"]
    assert ctx.default_dfe_taps() == EXPECTED_VSR_SIZES["dfe_taps"]
    assert ctx.fft_len() == EXPECTED_VSR_SIZES["fft_len"]


def test_sizes_are_loss_driven_not_rate_driven():
    """NRZ and PAM4 at the same reach share derived sizes but differ in Nyquist."""
    nrz = SimContext.from_data_rate(112.0, Modulation.NRZ, reach=VSR, sps=32)
    pam4 = SimContext.from_data_rate(112.0, Modulation.PAM4, reach=VSR, sps=32)
    assert nrz.sbr_len_samples() == pam4.sbr_len_samples()
    assert nrz.cursor_span() == pam4.cursor_span()
    assert nrz.default_dfe_taps() == pam4.default_dfe_taps()
    assert nrz.f_nyq != pam4.f_nyq  # NRZ Nyquist is 2x PAM4 at the same data rate


def test_higher_loss_grows_filters():
    xsr = REACH_PRESETS[("112G", "XSR")]  # 8 dB
    lr = REACH_PRESETS[("112G", "LR")]  # 28 dB
    c_xsr = SimContext.from_data_rate(112, Modulation.PAM4, reach=xsr)
    c_lr = SimContext.from_data_rate(112, Modulation.PAM4, reach=lr)
    assert c_lr.default_dfe_taps() > c_xsr.default_dfe_taps()
    assert c_lr.sbr_len_ui() > c_xsr.sbr_len_ui()
    assert c_lr.cursor_span()[1] > c_xsr.cursor_span()[1]


def test_context_is_immutable():
    ctx = SimContext.from_data_rate(112, Modulation.PAM4, reach=VSR)
    with pytest.raises(Exception):
        ctx.sps = 16  # frozen dataclass
    assert ctx.with_(sps=16).sps == 16 and ctx.sps == 32


def test_freq_grid_shape():
    ctx = SimContext.from_data_rate(112, Modulation.PAM4, reach=VSR, sps=32)
    f = ctx.freq_grid()
    assert f[0] == 0.0
    assert f[-1] == pytest.approx(ctx.fs / 2)
    assert len(f) == ctx.fft_len() // 2 + 1
