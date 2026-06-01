"""MLSD sequence detection — minimum-distance math, assessment, integration.

The core min-distance search is validated on a short-ISI channel whose minimum
distance can be checked by hand against the matched-filter bound.
"""

import numpy as np
import pytest

from eyeq.analysis import ber as B
from eyeq.analysis import fec, mlsd
from eyeq.analysis.optimize import optimize_link
from eyeq.engines import StatisticalEngine
from eyeq.io import build_pipeline, default_link_config
from eyeq.io.config import default_detector, from_dict, load, save, to_dict

STAT = StatisticalEngine()
NRZ = np.array([-1.0, 1.0])
PAM4 = np.linspace(-1.0, 1.0, 4)


# --------------------------------------------------------------------------- #
# minimum-distance math (hand-checkable)
# --------------------------------------------------------------------------- #
def test_error_symbols():
    assert np.allclose(mlsd.error_symbols(NRZ), [2.0])
    assert np.allclose(mlsd.error_symbols(PAM4), [2 / 3, 4 / 3, 2.0])


def test_min_distance_matches_matched_filter_bound():
    # NRZ h=[1, 0.5]: single-error event [2] -> filtered [2, 1] -> d^2 = 5,
    # which equals the matched-filter bound (step*||h||)^2 = (2*sqrt(1.25))^2.
    d2, n, trunc = mlsd.min_distance([1.0, 0.5], np.array([2.0]), max_len=8)
    assert d2 == pytest.approx(5.0)
    assert d2 == pytest.approx((2.0 * np.hypot(1.0, 0.5)) ** 2)
    assert n == 1 and not trunc


def test_min_distance_dicode():
    # h=[1, -1]: many equal-distance events, all d^2 = 8 = (2*sqrt(2))^2.
    d2, n, _ = mlsd.min_distance([1.0, -1.0], np.array([2.0]), max_len=8)
    assert d2 == pytest.approx(8.0)
    assert n >= 1


def test_union_bound_ber_behaviour():
    # monotone in sigma; noiseless -> 0; no signal -> random.
    b1 = mlsd.union_bound_ber(5.0, 1, 0.05, 1)[1]
    b2 = mlsd.union_bound_ber(5.0, 1, 0.20, 1)[1]
    assert b2 > b1
    assert mlsd.union_bound_ber(5.0, 1, 0.0, 1) == (0.0, 0.0)
    assert mlsd.union_bound_ber(0.0, 1, 0.1, 2)[0] == 0.5


def test_pam4_penalty_vs_nrz():
    # same channel + noise, but PAM-4's smaller min step -> higher BER than NRZ.
    h, sig = [1.0, 0.3], 0.15
    nrz = mlsd.sequence_ber(h, NRZ, sig, L=4).ber
    pam4 = mlsd.sequence_ber(h, PAM4, sig, L=4).ber
    assert pam4 > nrz


def test_l_caps():
    assert mlsd.l_cap(2) == 8 and mlsd.l_cap(4) == 5


def test_search_guard_terminates(monkeypatch):
    # A long PAM-4 channel at the cap must return (possibly truncated) without hanging.
    h = [1.0, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25]
    r = mlsd.sequence_ber(h, PAM4, 0.1, L=5, node_cap=20_000)
    assert np.isfinite(r.d2_min) and r.d2_min > 0
    # tiny node cap forces truncation, still returns
    r2 = mlsd.sequence_ber(h, PAM4, 0.1, L=5, node_cap=50)
    assert r2.truncated


# --------------------------------------------------------------------------- #
# assess_mlsd (BerResult-shaped, consumed uniformly)
# --------------------------------------------------------------------------- #
def _pipe(reach="VSR", mod="PAM4", noise=8.0):
    p = build_pipeline(default_link_config(modulation=mod, reach_class=reach))
    p.apply_params({"noise": {"sigma_mvrms": noise}})
    return p


def test_assess_mlsd_returns_berresult():
    p = _pipe()
    r = B.assess_mlsd(STAT, p, target_ber=1e-12, mlsd_taps=4)
    assert r.detector == "mlsd"
    assert np.isfinite(r.mlsd_dmin) and r.mlsd_dmin > 0
    assert r.eye_height_v == pytest.approx(r.mlsd_dmin / 2.0)
    assert np.all(np.isfinite(r.h_bathtub)) and np.all(np.isfinite(r.v_bathtub))
    assert 0.0 <= r.ber <= 0.5


def test_mlsd_taps_clamped_to_cap():
    p = _pipe(mod="PAM4")
    r = B.assess_mlsd(STAT, p, mlsd_taps=99)  # absurd L is clamped, no error/hang
    assert np.isfinite(r.ber)


def test_mlsd_beats_slicer_on_isi_link():
    # On an ISI-limited link the sequence detector is no worse than the slicer.
    p = _pipe("LR", "PAM4", noise=8.0)
    optimize_link(p)
    dec = B.assess(STAT, p, target_ber=1e-12)
    ml = B.assess_mlsd(STAT, p, target_ber=1e-12, mlsd_taps=4)
    assert ml.ber <= dec.ber + 1e-12


def test_fec_runs_on_mlsd_output():
    from eyeq.io.config import default_fec
    p = _pipe()
    ml = B.assess_mlsd(STAT, p, target_ber=1e-12)
    r = fec.assess_fec(ml, p.ctx, {**default_fec(), "enabled": True, "scheme": "kp4"})
    assert r.enabled and np.isfinite(r.post_ber) and r.post_ber <= ml.ber + 1e-30


# --------------------------------------------------------------------------- #
# report + config integration
# --------------------------------------------------------------------------- #
def test_report_detector_rows():
    from eyeq.analysis import report
    p = _pipe()
    ml = B.assess_mlsd(STAT, p, target_ber=1e-12, mlsd_taps=5)
    rc = report.ReportContext(ml, {}, p, detector={"mode": "mlsd", "mlsd_taps": 5})
    rows = {x.key: x for x in report.evaluate(rc)}
    assert rows["detector"].value == "MLSD"
    assert "union bound" in rows["ber_method"].value
    assert rows["trellis_l"].value == "5"
    assert rows["mlsd_margin"].raw is not None
    # decision detector hides the MLSD-only rows
    dec = B.assess(STAT, p, target_ber=1e-12)
    rcd = report.ReportContext(dec, {}, p, detector={"mode": "dfe"})
    rowsd = {x.key: x for x in report.evaluate(rcd)}
    assert rowsd["detector"].value == "DFE"
    assert rowsd["ber_method"].value == "eye-tail (decision point)"
    assert rowsd["trellis_l"].value == "—" and rowsd["mlsd_margin"].value == "—"


def test_detector_config_round_trip(tmp_path):
    cfg = default_link_config()
    cfg.detector.update({"mode": "mlsd", "mlsd_taps": 6})
    path = tmp_path / "link.yaml"
    save(cfg, path)
    r = load(path)
    assert r.detector["mode"] == "mlsd" and r.detector["mlsd_taps"] == 6


def test_old_config_without_detector_loads():
    d = to_dict(default_link_config())
    d.pop("detector")
    cfg = from_dict(d)
    assert cfg.detector == {}  # -> controller falls back to default_detector()
