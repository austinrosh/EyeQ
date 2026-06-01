"""The link-report metric registry (UX-polish task, item 4).

The report is an *extensible registry*: each metric is a descriptor with a getter
that reads the live data sources. Adding a metric — including the deferred
compliance metrics — must not require any change at the call site.
"""

from types import SimpleNamespace

import pytest

from eyeq.analysis import report
from eyeq.analysis.ber import assess
from eyeq.engines import StatisticalEngine
from eyeq.io import build_pipeline, default_link_config

STAT = StatisticalEngine()


def _context():
    p = build_pipeline(default_link_config(modulation="PAM4", reach_class="VSR"))
    ber = assess(STAT, p, target_ber=1e-12)
    stats = {
        "mse_snr_db": 18.3,
        "ser": 1e-7,
        "eye_height_at_phase_v": 0.041,
        "recovered_phase_ui": 0.02,
    }
    return report.ReportContext(ber=ber, stats=stats, pipe=p)


def test_evaluate_returns_one_row_per_metric():
    rows = report.evaluate(_context())
    assert len(rows) == len(report.METRICS)
    assert [r.key for r in rows] == [m.key for m in report.METRICS]


def test_getters_pull_from_the_context():
    rows = {r.key: r for r in report.evaluate(_context())}
    assert rows["snr"].value == "18.3"          # mse_snr_db from stats, dB
    assert rows["snr"].unit == "dB"
    assert rows["eye_height"].value == "41.0"   # 0.041 V -> mV
    assert rows["target_ber"].value == "1e-12"  # from the BER result
    assert rows["sample_phase"].value == "+0.020"


def test_active_eq_reflects_bypass_flags():
    rc = _context()
    rc.pipe.by_name("ctle").set_params(enabled="off")
    rc.pipe.by_name("txffe").set_params(driver_enabled="off")
    val = {r.key: r.value for r in report.evaluate(rc)}["eq_state"]
    assert "CTLE✗" in val and "TX-drv✗" in val and "RX-FFE✓" in val


def test_deferred_metrics_render_not_modeled():
    rows = {r.key: r for r in report.evaluate(_context())}
    for key in ("sndr", "rlm", "erl", "jtol"):
        assert rows[key].value == "— (not modeled)"
        assert rows[key].model_limited


def test_model_limited_metric_with_data_is_not_marked_not_modeled():
    # BER is model-limited but available -> shows a value, never "(not modeled)".
    ber_row = {r.key: r for r in report.evaluate(_context())}["ber"]
    assert ber_row.value != "— (not modeled)"
    assert ber_row.model_limited


def test_missing_data_renders_dash_not_none():
    rc = report.ReportContext(ber=None, stats={}, pipe=_context().pipe)
    rows = {r.key: r for r in report.evaluate(rc)}
    assert rows["com"].value == "—"            # no BER yet, not deferred
    assert rows["eye_height"].value == "—"     # no transient stats yet


def test_registry_is_extensible_without_call_site_changes():
    before = len(report.evaluate(_context()))
    extra = report.Metric("dummy", "Dummy", "x", "test-only metric",
                           lambda rc: 1.23, fmt="{:.2f}")
    report.METRICS.append(extra)
    try:
        rows = {r.key: r for r in report.evaluate(_context())}
        assert len(rows) == before + 1
        assert rows["dummy"].value == "1.23"   # new row, no change to evaluate()
    finally:
        report.METRICS.remove(extra)
    assert len(report.evaluate(_context())) == before
