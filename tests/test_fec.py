"""FEC modeling — RS waterfall math, assessment, and config persistence.

The headless math is validated against the canonical KP4 RS(544,514) anchor
(pre-FEC threshold ~2.4e-4, ~6.9 dB coding gain) and against a direct
(non-log-domain) binomial sum on a small code.
"""

from math import comb

import numpy as np
import pytest

from eyeq.analysis import fec, report
from eyeq.analysis.ber import assess
from eyeq.engines import StatisticalEngine
from eyeq.io import build_pipeline, default_link_config
from eyeq.io.config import default_fec, from_dict, load, save, to_dict

STAT = StatisticalEngine()


def _ber_ctx(modulation="PAM4", reach="VSR"):
    p = build_pipeline(default_link_config(modulation=modulation, reach_class=reach))
    return assess(STAT, p, target_ber=1e-12), p.ctx, p


def _cfg(**ov):
    return {**default_fec(), **ov}


# --------------------------------------------------------------------------- #
# RS waterfall math
# --------------------------------------------------------------------------- #
def test_kp4_pre_fec_threshold_matches_standard():
    # The canonical KP4 pre-FEC BER threshold for ~1e-15 post-FEC is ~2.4e-4.
    thr = fec.pre_fec_threshold(fec.SCHEMES["kp4"], 1e-15)
    assert 1.8e-4 < thr < 2.8e-4
    # at the textbook 2.4e-4 input, post-FEC is in the ~1e-14..1e-15 range
    assert fec.post_ber(2.4e-4, fec.SCHEMES["kp4"]) < 1e-13


def test_rs_output_matches_direct_binomial():
    # RS(15,11), t=2: log-domain sum equals the direct binomial sum.
    def direct(p_s, n, t):
        return sum(i * comb(n, i) * p_s ** i * (1 - p_s) ** (n - i)
                   for i in range(t + 1, n + 1)) / n
    for p_s in (1e-3, 1e-2, 5e-2, 0.2):
        assert fec._rs_output_symbol_error(p_s, 15, 2) == pytest.approx(direct(p_s, 15, 2), rel=1e-9)


def test_sym_err_prob_small_ber():
    # p_s ≈ m * BER for small BER (i.i.d. bits)
    assert fec.sym_err_prob(1e-6, 10) == pytest.approx(10e-6, rel=1e-3)
    assert fec.sym_err_prob(0.0, 10) == 0.0


def test_post_ber_monotone_and_improves_below_threshold():
    kp4 = fec.SCHEMES["kp4"]
    thr = fec.pre_fec_threshold(kp4, 1e-15)
    bers = np.array([thr / 5, thr / 2, thr, thr * 2, thr * 4])
    post = fec.post_ber(bers, kp4)
    assert np.all(np.diff(post) >= 0)                 # monotone non-decreasing
    assert fec.post_ber(thr / 4, kp4) < thr / 4        # FEC improves a clean-enough link
    assert post[-1] <= bers[-1] + 1e-30               # never worse than the input


def test_none_scheme_is_passthrough():
    none = fec.SCHEMES["none"]
    assert fec.post_ber(1e-4, none) == 1e-4
    assert fec.pre_fec_threshold(none, 1e-15) == 1e-15
    assert fec.coding_gain_db(1e-15, 1e-15) == 0.0


def test_kr4_weaker_than_kp4():
    thr_kp4 = fec.pre_fec_threshold(fec.SCHEMES["kp4"], 1e-15)
    thr_kr4 = fec.pre_fec_threshold(fec.SCHEMES["kr4"], 1e-15)
    assert thr_kr4 < thr_kp4  # t=7 needs a cleaner input than t=15
    assert fec.SCHEMES["kr4"].t == 7 and fec.SCHEMES["kp4"].t == 15


def test_coding_gain_kp4_is_about_7_db():
    thr = fec.pre_fec_threshold(fec.SCHEMES["kp4"], 1e-15)
    assert 6.0 < fec.coding_gain_db(thr, 1e-15) < 8.0  # textbook ~6.9 dB


def test_custom_scheme_derives_t():
    s = fec.scheme_from_cfg(_cfg(scheme="custom", custom_n=255, custom_k=239, custom_m=8))
    assert s.key == "custom" and s.n == 255 and s.k == 239 and s.t == 8


def test_bursty_model_degrades_post_ber():
    kp4 = fec.SCHEMES["kp4"]
    clean = fec.post_ber(2e-4, kp4)
    bursty = fec.post_ber(2e-4, kp4, error_model="bursty", burst_len_bits=400, interleave_depth=1)
    assert bursty > clean  # bursts waste correction capacity
    # enough interleaving recovers the random result
    spread = fec.post_ber(2e-4, kp4, error_model="bursty", burst_len_bits=10, interleave_depth=1)
    assert spread == pytest.approx(clean, rel=1e-9)


# --------------------------------------------------------------------------- #
# assess_fec (the GUI-facing entry point)
# --------------------------------------------------------------------------- #
def test_assess_fec_disabled_is_raw():
    ber, ctx, _ = _ber_ctx()
    r = fec.assess_fec(ber, ctx, _cfg(enabled=False, scheme="kp4"))
    assert not r.enabled and r.post_ber == ber.ber
    assert np.allclose(r.post_h_ber, r.pre_h_ber)  # no overlay shift when off


def test_assess_fec_enabled_builds_curves():
    ber, ctx, _ = _ber_ctx()
    r = fec.assess_fec(ber, ctx, _cfg(enabled=True, scheme="kp4"))
    assert r.enabled and r.n == 544 and r.t == 15
    assert r.post_h_ber.shape == ber.h_bathtub.shape
    assert r.post_v_ber.shape == ber.v_bathtub.shape
    # pre-FEC curve is the SER bathtub shifted to BER by bits-per-symbol
    assert np.allclose(r.pre_v_ber, ber.v_bathtub - np.log10(ctx.mod.bits_per_symbol))


def test_assess_fec_soft_pairing_note():
    ber, ctx, _ = _ber_ctx(modulation="PAM4")
    on_pairing = fec.assess_fec(ber, ctx, _cfg(enabled=True, scheme="kp4"))
    off_pairing = fec.assess_fec(ber, ctx, _cfg(enabled=True, scheme="kr4"))
    assert on_pairing.applicable and on_pairing.pairing_note == ""
    assert not off_pairing.applicable and "NRZ" in off_pairing.pairing_note  # still computed


# --------------------------------------------------------------------------- #
# report integration
# --------------------------------------------------------------------------- #
def test_report_gains_fec_rows():
    ber, ctx, pipe = _ber_ctx()
    r = fec.assess_fec(ber, ctx, _cfg(enabled=True, scheme="kp4"))
    rows = {x.key: x for x in report.evaluate(report.ReportContext(ber, {}, pipe, fec=r))}
    assert rows["fec_scheme"].value.startswith("KP4")
    assert rows["post_fec_ber"].raw is not None and rows["post_fec_ber"].model_limited
    assert rows["coding_gain"].raw is not None and rows["pre_fec_threshold"].raw is not None
    # disabled -> 'off' and dashes (not "not modeled")
    r0 = fec.assess_fec(ber, ctx, _cfg(enabled=False))
    rows0 = {x.key: x for x in report.evaluate(report.ReportContext(ber, {}, pipe, fec=r0))}
    assert rows0["fec_scheme"].value == "off" and rows0["post_fec_ber"].value == "—"


# --------------------------------------------------------------------------- #
# config persistence
# --------------------------------------------------------------------------- #
def test_fec_config_round_trip(tmp_path):
    cfg = default_link_config()
    cfg.fec.update({"enabled": True, "scheme": "kr4", "target_post_ber": 1e-12})
    path = tmp_path / "link.yaml"
    save(cfg, path)
    r = load(path)
    assert r.fec["enabled"] is True and r.fec["scheme"] == "kr4"
    assert r.fec["target_post_ber"] == 1e-12


def test_old_config_without_fec_loads_disabled():
    d = to_dict(default_link_config())
    d.pop("fec")                      # an old config predating the FEC field
    cfg = from_dict(d)
    assert cfg.fec == {}              # defaults to empty -> treated as disabled
