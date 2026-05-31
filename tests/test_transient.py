"""Transient engine + worker (Phase 2b), including the keystone agreement test."""

import time

import numpy as np
import pytest

from eyeq.engines import StatisticalEngine, ThreadWorker, TransientEngine
from eyeq.io import build_pipeline, default_link_config

STAT, TRAN = StatisticalEngine(), TransientEngine()


def pipe(mod="PAM4", reach="XSR", **overrides):
    p = build_pipeline(default_link_config(modulation=mod, reach_class=reach))
    p.apply_params({
        "ctle": {"fz": 0.35, "fp": 1.0, "dc_gain": -2.0},
        "txffe": {"pre": -0.08, "post": -0.12},
    })
    if overrides:
        p.apply_params(overrides)
    return p


def _center(res_or_eye):
    return int(np.argmin(np.abs(res_or_eye.t_ui)))


def _moments(col, v):
    m = float((col * v).sum())
    return m, float(np.sqrt((col * (v - m) ** 2).sum()))


# --------------------------------------------------------------------------- #
# basic engine behavior
# --------------------------------------------------------------------------- #
def test_density_is_a_distribution():
    res = TRAN.run_batch(pipe(), n_symbols=20_000, rng=np.random.default_rng(0))
    assert res.density.shape[0] == pipe().ctx.sps
    assert np.allclose(res.density.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(res.density >= 0)


def test_metrics_are_finite():
    res = TRAN.run_batch(pipe(reach="XSR"), n_symbols=50_000, rng=np.random.default_rng(0))
    assert np.isfinite(res.mse_snr_db) and res.mse_snr_db > 0
    assert 0.0 <= res.ser <= 1.0


# --------------------------------------------------------------------------- #
# KEYSTONE: density eye == statistical eye for an LTI-only link
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mod,reach", [
    ("NRZ", "XSR"), ("PAM4", "XSR"), ("NRZ", "XSR+"), ("PAM4", "VSR"),
])
def test_density_distribution_matches_statistical(mod, reach):
    """The decision-phase voltage distribution must match between engines.

    This is the single most valuable normalization check: the transient
    histogram and the statistical PDF describe the same random process, so their
    mean (~0) and std agree to Monte Carlo precision regardless of how open the
    eye is.
    """
    p = pipe(mod, reach)
    _, sbr, eye = STAT.compute(p)
    res = TRAN.run_batch(p, n_symbols=200_000, sbr=sbr, v=eye.v, rng=np.random.default_rng(7))
    sm, ss = _moments(eye.pdf[_center(eye)], eye.v)
    tm, ts = _moments(res.density[_center(res)], res.v)
    assert abs(tm) < 0.02 * ss + 1e-4           # ~zero mean
    assert ts == pytest.approx(ss, rel=0.03)    # std agrees to <3%


@pytest.mark.parametrize("mod,reach", [("NRZ", "XSR"), ("PAM4", "XSR"), ("NRZ", "VSR")])
def test_transient_snr_matches_analytic(mod, reach):
    p = pipe(mod, reach)
    _, sbr, _ = STAT.compute(p)
    res = TRAN.run_batch(p, n_symbols=200_000, sbr=sbr, rng=np.random.default_rng(7))
    assert res.mse_snr_db == pytest.approx(STAT.decision_snr_db(p, sbr), abs=0.3)


def test_open_eye_heights_agree():
    # For a comfortably open eye the (tail-sensitive) eye-height scalars also agree.
    p = pipe("NRZ", "XSR")
    _, sbr, eye = STAT.compute(p)
    res = TRAN.run_batch(p, n_symbols=200_000, sbr=sbr, v=eye.v, rng=np.random.default_rng(7))
    assert res.eye_height_v == pytest.approx(eye.eye_height_v, rel=0.20)


# --------------------------------------------------------------------------- #
# impairments widen the eye
# --------------------------------------------------------------------------- #
def test_noise_widens_distribution():
    p0 = pipe("NRZ", "XSR")
    p1 = pipe("NRZ", "XSR", noise={"sigma_mvrms": 20.0})
    _, sbr0, eye0 = STAT.compute(p0)
    r0 = TRAN.run_batch(p0, n_symbols=100_000, sbr=sbr0, v=eye0.v, rng=np.random.default_rng(1))
    r1 = TRAN.run_batch(p1, n_symbols=100_000, sbr=sbr0, v=eye0.v, rng=np.random.default_rng(1))
    _, s0 = _moments(r0.density[_center(r0)], r0.v)
    _, s1 = _moments(r1.density[_center(r1)], r1.v)
    assert s1 > s0
    assert r1.mse_snr_db < r0.mse_snr_db


# --------------------------------------------------------------------------- #
# performance
# --------------------------------------------------------------------------- #
def test_throughput_meets_target():
    p = pipe("PAM4", "VSR")
    _, sbr, eye = STAT.compute(p)
    TRAN.run_batch(p, n_symbols=20_000, sbr=sbr, v=eye.v)  # warm
    t0 = time.perf_counter()
    res = TRAN.run_batch(p, n_symbols=200_000, sbr=sbr, v=eye.v, rng=np.random.default_rng(0))
    ui_per_s = res.n_symbols / (time.perf_counter() - t0)
    print(f"\ntransient throughput: {ui_per_s/1e6:.2f} M UI/s")
    assert ui_per_s >= 1.5e6


# --------------------------------------------------------------------------- #
# threaded worker
# --------------------------------------------------------------------------- #
def _spin_until(predicate, timeout=5.0):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_worker_runs_and_publishes_snapshots():
    w = ThreadWorker(pipe("PAM4", "XSR"), batch_symbols=10_000)
    with w:
        assert _spin_until(lambda: w.latest() is not None)
        s1 = w.latest()
        assert _spin_until(lambda: (w.latest().seq > s1.seq))
        snap = w.latest()
        assert snap.image.shape[0] == pipe().ctx.sps
        assert np.all(np.isfinite(snap.image))
        assert "mse_snr_db" in snap.stats


def test_worker_applies_coalesced_params():
    w = ThreadWorker(pipe("NRZ", "XSR"), batch_symbols=10_000)
    with w:
        assert _spin_until(lambda: w.latest() is not None)
        snr_before = w.latest().stats["mse_snr_db"]
        # Heavier loss via loss_scale (LTI) should drop the SNR.
        w.push_params({"channel": {"loss_scale": 2.0}})
        assert _spin_until(
            lambda: w.latest().stats["mse_snr_db"] < snr_before - 1.0, timeout=5.0
        )


def test_worker_stop_is_clean():
    w = ThreadWorker(pipe(), batch_symbols=5_000)
    w.start()
    assert _spin_until(lambda: w.latest() is not None)
    w.stop()
    assert w._thread is None
