"""LTI block transfers: TX FFE + driver, channel models, CTLE, RX FFE."""

import numpy as np
import pytest

from eyeq.core.context import REACH_PRESETS, Modulation, SimContext
from eyeq.io import build_pipeline, default_link_config


def ctx(mod=Modulation.PAM4, reach="VSR"):
    return SimContext.from_data_rate(
        112.0, mod, reach=REACH_PRESETS[("112G", reach)], sps=32
    )


# --------------------------------------------------------------------------- #
# TX FFE + driver
# --------------------------------------------------------------------------- #
def test_txffe_main_tap_formula():
    tx = build_pipeline(default_link_config()).by_name("txffe")
    tx.set_params(pre=0.0, post=0.0)
    assert tx.main_tap() == pytest.approx(1.0)
    tx.set_params(pre=-0.2, post=-0.1)
    assert tx.main_tap() == pytest.approx(0.7)  # 1 - 0.3
    tx.set_params(pre=-0.4, post=-0.4)
    assert tx.main_tap() == pytest.approx(0.6)  # floored


def test_txffe_dc_gain_is_tap_sum():
    c = ctx()
    tx = build_pipeline(default_link_config()).by_name("txffe")
    tx.set_params(pre=-0.1, post=-0.15)
    H = tx.transfer(c)
    assert abs(H[0]) == pytest.approx(tx.taps().sum(), rel=1e-9)


def test_driver_is_lowpass():
    c = ctx()
    tx = build_pipeline(default_link_config()).by_name("txffe")
    d = np.abs(tx.driver_transfer(c))
    assert d[0] == pytest.approx(1.0)
    assert d[-1] < d[0]
    assert np.all(np.diff(d) <= 1e-12)  # monotonically non-increasing


# --------------------------------------------------------------------------- #
# Channel
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("reach", ["XSR", "XSR+", "VSR", "MR", "LR"])
def test_channel_hits_loss_budget_at_ref_nyquist(reach):
    c = ctx(reach=reach)
    ch = build_pipeline(default_link_config(reach_class=reach)).by_name("channel")
    assert ch.loss_at_ref_nyquist_db(c) == pytest.approx(c.reach.loss_db_nyq, abs=0.05)


def test_loss_scale_scales_budget():
    c = ctx()
    ch = build_pipeline(default_link_config()).by_name("channel")
    ch.set_params(loss_scale=0.5)
    assert ch.loss_at_ref_nyquist_db(c) == pytest.approx(0.5 * c.reach.loss_db_nyq, abs=0.05)


def test_nrz_sees_more_loss_than_pam4_same_channel():
    # Same reach/trace; NRZ Nyquist (56 GHz) sits higher on the loss curve.
    ch = build_pipeline(default_link_config()).by_name("channel")
    nyq_loss_pam4 = ch.insertion_loss_db(ctx(Modulation.PAM4))
    c_nrz = ctx(Modulation.NRZ)
    il_nrz = ch.insertion_loss_db(c_nrz)
    i_nrz = int(np.argmin(np.abs(c_nrz.freq_grid() - c_nrz.f_nyq)))
    c_pam4 = ctx(Modulation.PAM4)
    i_pam4 = int(np.argmin(np.abs(c_pam4.freq_grid() - c_pam4.f_nyq)))
    assert il_nrz[i_nrz] > 1.5 * nyq_loss_pam4[i_pam4]


def test_tl_and_simple_share_magnitude_differ_in_phase():
    c = ctx(reach="LR")
    ch = build_pipeline(default_link_config(reach_class="LR")).by_name("channel")
    ch.set_params(model="simple")
    Hs = ch.transfer(c)
    ch.set_params(model="tl")
    Ht = ch.transfer(c)
    assert np.allclose(np.abs(Hs), np.abs(Ht), atol=1e-9)  # identical magnitude
    assert not np.allclose(np.angle(Hs), np.angle(Ht))      # different phase


def test_package_stage_adds_loss():
    c = ctx(reach="LR")
    base = build_pipeline(default_link_config(reach_class="LR")).by_name("channel")
    pkg = build_pipeline(default_link_config(reach_class="LR")).by_name("channel")
    pkg.set_params(package="on")
    nyq = int(np.argmin(np.abs(c.freq_grid() - c.reach.ref_nyquist_hz)))
    extra = -20 * np.log10(np.abs(pkg.transfer(c)[nyq]) / np.abs(base.transfer(c)[nyq]))
    assert extra == pytest.approx(c.reach.pkg_db_nyq, abs=0.1)


def test_touchstone_not_yet_implemented():
    c = ctx()
    ch = build_pipeline(default_link_config()).by_name("channel")
    ch.set_params(model="touchstone")
    with pytest.raises(NotImplementedError):
        ch.transfer(c)


# --------------------------------------------------------------------------- #
# CTLE
# --------------------------------------------------------------------------- #
def test_ctle_peaking_matches_hand_computed():
    # fz=0.5, fp=1.0 (real pair), fpp=3.0/zeta=0.7 (near-flat resonant), dc=0.
    # real pair @Nyq: |1+2j|/|1+1j| = sqrt5/sqrt2 -> 3.9794 dB
    # resonant @Nyq:  9/(8+4.2j)            -> -0.0342 dB
    # total peaking = 3.9452 dB
    c = ctx()
    ctle = build_pipeline(default_link_config()).by_name("ctle")
    ctle.set_params(fz=0.5, fp=1.0, fpp=3.0, zeta_pp=0.7, dc_gain=0.0)
    assert ctle.peaking_db(c) == pytest.approx(3.945, abs=0.05)


def test_ctle_dc_gain_applies_at_dc():
    c = ctx()
    ctle = build_pipeline(default_link_config()).by_name("ctle")
    ctle.set_params(dc_gain=-6.0)
    assert abs(ctle.transfer(c)[0]) == pytest.approx(10 ** (-6.0 / 20), rel=1e-6)


def test_ctle_more_boost_with_lower_zero():
    c = ctx()
    ctle = build_pipeline(default_link_config()).by_name("ctle")
    ctle.set_params(fz=0.8, fp=1.0, fpp=3.0)
    low = ctle.peaking_db(c)
    ctle.set_params(fz=0.3)
    assert ctle.peaking_db(c) > low


# --------------------------------------------------------------------------- #
# RX FFE
# --------------------------------------------------------------------------- #
def test_rxffe_identity_in_phase1():
    c = ctx()
    rx = build_pipeline(default_link_config()).by_name("rxffe")
    assert np.allclose(np.abs(rx.transfer(c)), 1.0)
