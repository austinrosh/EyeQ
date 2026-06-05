"""Jitter: DCD/SJ injection, the combined jitter PDF, and the RJ/DJ/TJ decomposition."""

import numpy as np
import pytest

from eyeq.analysis import ber as B
from eyeq.analysis import jitter as J
from eyeq.analysis import report
from eyeq.analysis.optimize import optimize_link
from eyeq.engines import StatisticalEngine, TransientEngine
from eyeq.io import build_pipeline, default_link_config

STAT, TRAN = StatisticalEngine(), TransientEngine()


def _pipe(reach="VSR", eq=True):
    p = build_pipeline(default_link_config(modulation="PAM4", reach_class=reach))
    if eq:
        optimize_link(p)
    return p


# --------------------------------------------------------------------------- #
# bounded-jitter kernels (DCD 2-Dirac, SJ arcsine)
# --------------------------------------------------------------------------- #
def test_dcd_kernel_is_two_dirac():
    n, dv, pp = 2048, 1e-3, 0.1
    k = StatisticalEngine._dcd_kernel(n, dv, pp)
    x = (np.arange(n) - n // 2) * dv
    assert k.sum() == pytest.approx(1.0, abs=1e-9)
    assert float(np.sum(k * x)) == pytest.approx(0.0, abs=1e-6)          # zero mean
    std = float(np.sqrt(np.sum(k * x**2)))
    assert 2.0 * std == pytest.approx(pp, rel=1e-3)                      # two masses at ±pp/2


def test_arcsine_kernel_variance():
    n, dv, amp = 4096, 5e-4, 0.2
    k = StatisticalEngine._arcsine_kernel(n, dv, amp)
    x = (np.arange(n) - n // 2) * dv
    assert k.sum() == pytest.approx(1.0, abs=1e-9)
    assert float(np.sum(k * x)) == pytest.approx(0.0, abs=1e-6)          # zero mean
    var = float(np.sum(k * x**2))
    assert var == pytest.approx(amp**2 / 2.0, rel=2e-2)                  # arcsine variance = A²/2


# --------------------------------------------------------------------------- #
# the jitter PDF closes the eye
# --------------------------------------------------------------------------- #
def _set_jitter(p, **j):
    """Set txjitter, explicitly zeroing the components not named (apply_params merges)."""
    p.apply_params({"txjitter": {"rj_mui": 0.0, "dcd_mui": 0.0, "sj_mui": 0.0, **j}})


def test_eye_width_shrinks_with_each_jitter_component():
    p = _pipe("VSR")
    sbr = STAT.sbr(p)  # jitter is a PDF blur — the pulse/SBR is unchanged

    def ew(**j):
        _set_jitter(p, **j)
        return B.assess(STAT, p, sbr, target_ber=1e-6, phase_points=129, v_bins=512).eye_width_ui

    base = ew()
    assert ew(rj_mui=40) < base
    assert ew(dcd_mui=50) < base
    assert ew(sj_mui=80) < base
    # monotone non-increasing as RJ grows (allow one phase-grid step of quantization)
    tol = 1.0 / 129
    widths = [ew(rj_mui=r) for r in (0, 10, 25, 40)]
    assert all(b <= a + tol for a, b in zip(widths, widths[1:]))
    assert widths[-1] < widths[0]


def test_zero_jitter_is_unchanged():
    # the DCD/SJ extension must reduce exactly to the old RJ-only math when both are 0
    p = _pipe("MR")
    sbr = STAT.sbr(p)
    p.apply_params({"txjitter": {"rj_mui": 8.0, "dcd_mui": 0.0, "sj_mui": 0.0}})
    r = B.assess(STAT, p, sbr, target_ber=1e-6)
    col = STAT._convolve_cursor_pdfs(STAT._sample_cursors(sbr.sbr, sbr.main_idx, sbr.cursor_k, p.ctx.sps),
                                     p.ctx.levels, np.linspace(-1, 1, 512), 2 / 512)
    # _jitter_params returns (rj_s, dcd_s, periodic_list); rj only -> no dcd, no periodic
    rj_s, dcd_s, periodic_s = STAT._jitter_params(p, p.ctx)
    assert dcd_s == 0.0 and periodic_s == [] and rj_s > 0.0
    assert np.all(np.isfinite(r.h_bathtub))


# --------------------------------------------------------------------------- #
# RJ / DCD / SJ / DDJ -> TJ decomposition
# --------------------------------------------------------------------------- #
def test_decompose_relationships():
    p = _pipe("VSR")
    p.apply_params({"txjitter": {"rj_mui": 5.0, "dcd_mui": 10.0, "sj_mui": 8.0, "sj_freq_mhz": 100.0}})
    sbr = STAT.sbr(p)
    ber = B.assess(STAT, p, sbr, target_ber=1e-6)
    jr = J.decompose(STAT, p, sbr, ber, target_ber=1e-12)
    assert jr.rj_rms_ui == pytest.approx(5e-3)
    assert jr.dcd_pp_ui == pytest.approx(10e-3)
    assert jr.sj_pp_ui == pytest.approx(16e-3)                           # pp = 2 × amplitude
    assert jr.dj_pp_ui == pytest.approx(jr.dcd_pp_ui + jr.sj_pp_ui + jr.ddj_pp_ui)
    assert jr.tj_ui == pytest.approx(jr.dj_pp_ui + 2 * J._q_inv(1e-12 / 2) * jr.rj_rms_ui)
    assert jr.eye_width_ui == pytest.approx(ber.eye_width_ui)


def test_ddj_tracks_isi():
    # DDJ is a UI-bounded quantity; it grows with channel loss and shrinks once the
    # equalizer cleans up the ISI.
    ddj_raw = {r: J.estimate_ddj_ui(STAT, _pipe(r, eq=False), STAT.sbr(_pipe(r, eq=False)))
               for r in ("XSR", "LR")}
    assert all(0.0 <= d <= 1.0 for d in ddj_raw.values())
    assert ddj_raw["XSR"] < ddj_raw["LR"]                               # more loss -> more DDJ
    p_eq = _pipe("VSR", eq=True)
    p_raw = _pipe("VSR", eq=False)
    assert J.estimate_ddj_ui(STAT, p_eq, STAT.sbr(p_eq)) < \
        J.estimate_ddj_ui(STAT, p_raw, STAT.sbr(p_raw))                 # EQ reduces DDJ


# --------------------------------------------------------------------------- #
# transient injection
# --------------------------------------------------------------------------- #
def test_jitter_shift_statistics():
    p = _pipe("VSR", eq=False)
    sps, ctx, rng = p.ctx.sps, p.ctx, np.random.default_rng(0)
    sidx = np.arange(40_000)

    assert TRAN._jitter_shift(p, sidx, sps, ctx, rng) is None           # no jitter -> None

    p.apply_params({"txjitter": {"rj_mui": 30.0}})
    s = TRAN._jitter_shift(p, sidx, sps, ctx, rng)
    assert abs(np.std(s) - 0.030 * sps) < 0.15 * sps                    # RJ std ≈ rj·sps

    p.apply_params({"txjitter": {"rj_mui": 0.0, "dcd_mui": 40.0}})
    s = TRAN._jitter_shift(p, sidx, sps, ctx, rng)
    assert set(np.unique(s)) <= {round(0.020 * sps), -round(0.020 * sps)}  # ±pp/2 by parity

    p.apply_params({"txjitter": {"dcd_mui": 0.0, "sj_mui": 60.0, "sj_freq_mhz": 200.0}})
    s = TRAN._jitter_shift(p, sidx, sps, ctx, rng)
    assert s.max() <= round(0.060 * sps) and s.min() >= -round(0.060 * sps)  # bounded by amplitude


def test_dcd_closes_the_transient_eye():
    p = _pipe("VSR")
    sbr = STAT.sbr(p)
    _, _, eye = STAT.compute(p)

    def eh(**j):
        _set_jitter(p, **j)
        return TRAN.run_batch(p, n_symbols=60_000, sbr=sbr, v=eye.v,
                              rng=np.random.default_rng(0)).eye_height_v

    assert eh(dcd_mui=50.0) < eh()                                      # DCD offsets close the eye


# --------------------------------------------------------------------------- #
# report surfacing
# --------------------------------------------------------------------------- #
def test_report_exposes_jitter_rows():
    p = _pipe("VSR")
    p.apply_params({"txjitter": {"rj_mui": 6.0, "dcd_mui": 12.0, "sj_mui": 5.0}})
    sbr = STAT.sbr(p)
    ber = B.assess(STAT, p, sbr, target_ber=1e-6)
    jr = J.decompose(STAT, p, sbr, ber, target_ber=1e-12)
    rc = report.ReportContext(ber=ber, stats={}, pipe=p, jitter=jr)
    rows = {r.key: r for r in report.evaluate(rc)}
    for key in ("rj", "ddj", "dj", "tj", "jtol"):
        assert key in rows and rows[key].raw is not None
    assert rows["rj"].raw == pytest.approx(6.0)                         # mUI
    assert rows["jtol"].raw == pytest.approx(ber.eye_width_ui)         # timing margin = measured opening
