"""RX (sampling-clock) jitter + the CDR jitter transfer (1-H) that tracks low-freq jitter."""

import numpy as np
import pytest

from eyeq.analysis import ber as B
from eyeq.analysis import jitter as J
from eyeq.analysis import report
from eyeq.analysis.optimize import optimize_link
from eyeq.core.pipeline import CANONICAL_ORDER
from eyeq.engines import StatisticalEngine, TransientEngine
from eyeq.io import build_pipeline, default_link_config, load, save
from eyeq.io.config import _NAME_TO_TYPE

STAT, TRAN = StatisticalEngine(), TransientEngine()


def _pipe(reach="VSR", eq=True):
    p = build_pipeline(default_link_config(modulation="PAM4", reach_class=reach))
    if eq:
        optimize_link(p)
    return p


def _set(p, block, **params):
    p.apply_params({block: params})


# --------------------------------------------------------------------------- #
# block registration
# --------------------------------------------------------------------------- #
def test_rxjitter_block_registered_and_roundtrips(tmp_path):
    assert "rxjitter" in CANONICAL_ORDER
    assert _NAME_TO_TYPE["rxjitter"] == "RXJitter"
    # rxjitter is on the RX side, before the slicer
    assert CANONICAL_ORDER.index("rxffe") < CANONICAL_ORDER.index("rxjitter") < \
        CANONICAL_ORDER.index("cdr_slicer")
    assert _pipe(eq=False).by_name("rxjitter").get("rj_mui") == 0.0  # default pipe has the block

    # a config with an rxjitter param override survives save -> load -> build
    cfg = default_link_config(modulation="PAM4", reach_class="VSR")
    rxj = next(b for b in cfg.blocks if b.type == "RXJitter")
    rxj.params = {"rj_mui": 7.0, "pj_mui": 12.0, "pj_freq_mhz": 250.0}
    out = tmp_path / "link.yaml"
    save(cfg, out)
    reloaded = next(b for b in load(out).blocks if b.type == "RXJitter")
    assert reloaded.params["rj_mui"] == 7.0 and reloaded.params["pj_mui"] == 12.0
    assert build_pipeline(load(out)).by_name("rxjitter").get("pj_freq_mhz") == 250.0


# --------------------------------------------------------------------------- #
# CDR jitter transfer (1-H)
# --------------------------------------------------------------------------- #
def test_error_response_shape():
    p = _pipe(eq=False)
    cdr = p.by_name("cdr_slicer")
    assert cdr.error_response(1e6) == 1.0                       # static -> no tracking
    _set(p, "cdr_slicer", cdr_mode="bang-bang", loop_bw_mhz=0.0)
    assert cdr.error_response(1e6) == 1.0                       # loop_bw 0 -> no tracking
    _set(p, "cdr_slicer", cdr_mode="bang-bang", loop_bw_mhz=10.0)
    assert cdr.error_response(1e3) < 0.01                       # f << fc -> tracked out
    assert cdr.error_response(10e6) == pytest.approx(1 / np.sqrt(2), rel=1e-6)  # at the corner
    assert cdr.error_response(1e9) == pytest.approx(1.0, abs=1e-3)              # f >> fc -> passes


def test_cdr_tracks_low_frequency_periodic_jitter():
    p = _pipe("VSR")
    _set(p, "cdr_slicer", cdr_mode="bang-bang", loop_bw_mhz=10.0)
    sbr = STAT.sbr(p)

    def ew(**jp):
        p.apply_params({"txjitter": {"rj_mui": 0.0, "dcd_mui": 0.0, "sj_mui": 0.0, **jp}})
        return B.assess(STAT, p, sbr, target_ber=1e-6, phase_points=129, v_bins=512).eye_width_ui

    base = ew()
    tracked = ew(sj_mui=80.0, sj_freq_mhz=1.0)      # f << 10 MHz -> tracked out
    passes = ew(sj_mui=80.0, sj_freq_mhz=200.0)     # f >> 10 MHz -> closes the eye
    assert tracked == pytest.approx(base, abs=1.0 / 129)        # low-freq SJ barely closes the eye
    assert passes < base - 1.0 / 129                            # high-freq SJ does close it


def test_static_mode_passes_all_jitter_regression():
    # default static CDR -> |1-H|=1 -> a periodic component closes the eye fully (old behavior)
    p = _pipe("VSR")
    assert p.by_name("cdr_slicer").get("cdr_mode") == "static"
    sbr = STAT.sbr(p)
    p.apply_params({"txjitter": {"sj_mui": 20.0, "sj_freq_mhz": 1.0}})  # very low freq
    jr = J.decompose(STAT, p, sbr, B.assess(STAT, p, sbr, target_ber=1e-6))
    assert jr.loop_bw_mhz == 0.0
    assert jr.sj_pp_ui == pytest.approx(2 * 20e-3)             # NOT tracked (static): full pp


# --------------------------------------------------------------------------- #
# RX clock jitter sources
# --------------------------------------------------------------------------- #
def test_rx_rj_adds_in_quadrature():
    p = _pipe("VSR")
    p.apply_params({"txjitter": {"rj_mui": 6.0}, "rxjitter": {"rj_mui": 8.0}})
    jr = J.decompose(STAT, p, STAT.sbr(p), None)
    assert jr.rj_rms_ui == pytest.approx(np.hypot(6e-3, 8e-3))   # sqrt(6²+8²)=10 mUI
    assert jr.rx_rj_rms_ui == pytest.approx(8e-3)


def test_rx_pj_closes_eye_and_appears_in_decomposition():
    p = _pipe("VSR")
    _set(p, "cdr_slicer", cdr_mode="bang-bang", loop_bw_mhz=10.0)
    sbr = STAT.sbr(p)
    base = B.assess(STAT, p, sbr, target_ber=1e-6, phase_points=129).eye_width_ui
    p.apply_params({"rxjitter": {"pj_mui": 60.0, "pj_freq_mhz": 300.0}})  # well above corner
    r = B.assess(STAT, p, sbr, target_ber=1e-6, phase_points=129)
    assert r.eye_width_ui < base                                # RX PJ closes the eye
    jr = J.decompose(STAT, p, sbr, r)
    assert jr.pj_pp_ui > 0.0
    assert jr.dj_pp_ui == pytest.approx(jr.dcd_pp_ui + jr.sj_pp_ui + jr.pj_pp_ui + jr.ddj_pp_ui)


def test_transient_jitter_shift_includes_rx():
    p = _pipe("VSR", eq=False)
    sps, ctx, rng = p.ctx.sps, p.ctx, np.random.default_rng(0)
    sidx = np.arange(40_000)
    assert TRAN._jitter_shift(p, sidx, sps, ctx, rng) is None   # no jitter -> None
    # RX RJ alone shifts; combined with TX RJ in quadrature
    p.apply_params({"txjitter": {"rj_mui": 0.0}, "rxjitter": {"rj_mui": 30.0}})
    s = TRAN._jitter_shift(p, sidx, sps, ctx, rng)
    assert s is not None and abs(np.std(s) - 0.030 * sps) < 0.15 * sps


def test_report_exposes_rx_jitter_rows():
    p = _pipe("VSR")
    _set(p, "cdr_slicer", cdr_mode="bang-bang", loop_bw_mhz=12.0)
    p.apply_params({"rxjitter": {"rj_mui": 5.0, "pj_mui": 20.0, "pj_freq_mhz": 100.0}})
    sbr = STAT.sbr(p)
    ber = B.assess(STAT, p, sbr, target_ber=1e-6)
    jr = J.decompose(STAT, p, sbr, ber)
    rows = {r.key: r for r in report.evaluate(report.ReportContext(ber=ber, stats={}, pipe=p, jitter=jr))}
    for key in ("rx_rj", "pj", "cdr_bw"):
        assert key in rows and rows[key].raw is not None
    assert rows["rx_rj"].raw == pytest.approx(5.0)
    assert rows["cdr_bw"].raw == pytest.approx(12.0)
