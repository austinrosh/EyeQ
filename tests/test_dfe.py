"""DFE block + Numba kernel + transient nonlinear-tail behavior (Phase 3)."""

import time

import numpy as np
import pytest

from eyeq.engines import StatisticalEngine, TransientEngine
from eyeq.io import build_pipeline, default_link_config

STAT, TRAN = StatisticalEngine(), TransientEngine()


def pipe(mod="PAM4", reach="VSR", **overrides):
    p = build_pipeline(default_link_config(modulation=mod, reach_class=reach))
    p.apply_params({"ctle": {"fz": 0.35, "fp": 1.0, "dc_gain": -2.0}})
    if overrides:
        p.apply_params(overrides)
    return p


# --------------------------------------------------------------------------- #
# DFE block
# --------------------------------------------------------------------------- #
def test_dfe_taps_and_h1_sync():
    dfe = pipe().by_name("dfe")
    dfe.set_taps(np.array([0.02, 0.01, -0.005]))
    assert dfe.get("n_taps") == 3
    assert dfe.get("h1") == pytest.approx(20.0)  # 0.02 V -> 20 mV
    assert np.allclose(dfe.taps(), [0.02, 0.01, -0.005])
    assert dfe.is_active()


def test_dfe_inactive_when_zero():
    dfe = pipe().by_name("dfe")
    assert not dfe.is_active()  # default h1=0


# --------------------------------------------------------------------------- #
# DFE closes a known-ISI eye
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mod", ["NRZ", "PAM4"])
def test_dfe_closes_known_isi_eye(mod):
    p = pipe(mod, "VSR")
    _, sbr, eye = STAT.compute(p)
    before = TRAN.run_batch(p, n_symbols=80_000, sbr=sbr, v=eye.v, rng=np.random.default_rng(0))
    post = sbr.cursors[sbr.cursor_k > 0]
    p.by_name("dfe").set_taps(post)  # cancel all post-cursors
    after = TRAN.run_batch(p, n_symbols=80_000, sbr=sbr, v=eye.v, rng=np.random.default_rng(0))
    # SNR/SER are the robust evidence; the eye-height scalar is tail-sensitive on
    # marginal PAM4 eyes, so only require it not to regress.
    assert after.mse_snr_db > before.mse_snr_db + 3.0
    assert after.ser < before.ser
    assert after.eye_height_v >= before.eye_height_v


def test_dfe_snr_matches_analytic_postcursor_cancellation():
    # With every post-cursor cancelled, residual = pre-cursor ISI only.
    p = pipe("NRZ", "VSR")
    _, sbr, _ = STAT.compute(p)
    post = sbr.cursors[sbr.cursor_k > 0]
    p.by_name("dfe").set_taps(post)
    res = TRAN.run_batch(p, n_symbols=200_000, sbr=sbr, rng=np.random.default_rng(3))
    ea2 = float(np.mean(p.ctx.levels**2))
    pre = sbr.cursors[sbr.cursor_k < 0]
    analytic = 10 * np.log10(sbr.main_cursor**2 * ea2 / (np.sum(pre**2) * ea2))
    assert res.mse_snr_db == pytest.approx(analytic, abs=0.5)


# --------------------------------------------------------------------------- #
# CDR sample phase / jitter
# --------------------------------------------------------------------------- #
def test_sample_phase_moves_optimal_point():
    p = pipe("NRZ", "XSR", txffe={"pre": -0.08, "post": -0.12})
    _, sbr, eye = STAT.compute(p)

    def snr_at(ph):
        p.apply_params({"cdr_slicer": {"sample_phase_ui": ph}})
        return TRAN.run_batch(p, n_symbols=60_000, sbr=sbr, v=eye.v, rng=np.random.default_rng(0)).mse_snr_db

    assert snr_at(0.0) > snr_at(0.4) + 3.0
    assert snr_at(0.0) > snr_at(-0.4) + 3.0


# --------------------------------------------------------------------------- #
# online LMS / sign-LMS adaptation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", ["lms", "sign-lms"])
def test_lms_converges_dfe_taps_from_zero(mode):
    p = pipe("NRZ", "VSR")
    _, sbr, eye = STAT.compute(p)
    ideal = sbr.cursors[sbr.cursor_k > 0]
    n = int(min(6, ideal.size))
    mu = 0.02 if mode == "lms" else 2e-4
    p.apply_params({"dfe": {"n_taps": n, "h1": 0.0, "adapt": mode, "mu": mu}})

    rng = np.random.default_rng(0)
    for _ in range(15):  # adapt from zero over several batches
        res = TRAN.run_batch(p, n_symbols=50_000, sbr=sbr, v=eye.v, rng=rng)
    taps = p.by_name("dfe").taps()
    assert taps[0] == pytest.approx(ideal[0], rel=0.25)  # first tap finds the post-cursor
    assert res.ser < 1e-3                                # eye opened


def test_no_adaptation_leaves_taps_fixed():
    p = pipe("NRZ", "VSR")
    _, sbr, eye = STAT.compute(p)
    p.apply_params({"dfe": {"n_taps": 4, "h1": 20.0, "adapt": "off", "mu": 0.05}})
    before = p.by_name("dfe").taps().copy()
    TRAN.run_batch(p, n_symbols=50_000, sbr=sbr, v=eye.v, rng=np.random.default_rng(0))
    assert np.array_equal(p.by_name("dfe").taps(), before)  # adapt off -> no change


# --------------------------------------------------------------------------- #
# CDR phase recovery
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", ["mueller-muller", "bang-bang"])
@pytest.mark.parametrize("init", [0.25, -0.25])
def test_cdr_recovers_optimal_phase(mode, init):
    p = pipe("NRZ", "XSR", txffe={"pre": -0.08, "post": -0.12})
    _, sbr, eye = STAT.compute(p)
    p.apply_params({"cdr_slicer": {"sample_phase_ui": 0.0, "cdr_mode": "static"}})
    opt = TRAN.run_batch(p, n_symbols=80_000, sbr=sbr, v=eye.v, rng=np.random.default_rng(0)).mse_snr_db

    p.apply_params({"cdr_slicer": {"sample_phase_ui": init, "cdr_mode": mode, "kp": 0.04, "ki": 0.002}})
    rng = np.random.default_rng(0)
    for _ in range(6):  # let the loop lock from the bad initial phase
        r = TRAN.run_batch(p, n_symbols=60_000, sbr=sbr, v=eye.v, rng=rng)
    assert abs(r.recovered_phase_ui) < 0.1        # locked near the eye center
    assert r.mse_snr_db > opt - 1.5               # recovered SNR ~ the optimum


def test_static_cdr_uses_the_sample_phase_slider():
    p = pipe("NRZ", "XSR")
    _, sbr, eye = STAT.compute(p)
    p.apply_params({"cdr_slicer": {"sample_phase_ui": 0.25, "cdr_mode": "static"}})
    r = TRAN.run_batch(p, n_symbols=40_000, sbr=sbr, v=eye.v, rng=np.random.default_rng(0))
    assert r.recovered_phase_ui == pytest.approx(0.25, abs=1.0 / p.ctx.sps)


def test_jitter_shrinks_the_eye():
    p = pipe("NRZ", "XSR", txffe={"pre": -0.08, "post": -0.12})
    _, sbr, eye = STAT.compute(p)
    clean = TRAN.run_batch(p, n_symbols=80_000, sbr=sbr, v=eye.v, rng=np.random.default_rng(0))
    p.apply_params({"txjitter": {"rj_mui": 60.0}})
    jit = TRAN.run_batch(p, n_symbols=80_000, sbr=sbr, v=eye.v, rng=np.random.default_rng(0))
    assert jit.eye_height_v < clean.eye_height_v


# --------------------------------------------------------------------------- #
# performance: Numba DFE loop holds the throughput target at 448G
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rate", [112, 448])
def test_dfe_throughput(rate):
    p = build_pipeline(default_link_config(data_rate_gbps=rate, modulation="PAM4", reach_class="LR"))
    p.apply_params({"ctle": {"fz": 0.3, "fp": 1.0, "dc_gain": -3.0}})
    _, sbr, eye = STAT.compute(p)
    post = sbr.cursors[sbr.cursor_k > 0]
    p.by_name("dfe").set_taps(post[: p.ctx.default_dfe_taps()])
    TRAN.run_batch(p, n_symbols=20_000, sbr=sbr, v=eye.v)  # warm/compile
    t0 = time.perf_counter()
    res = TRAN.run_batch(p, n_symbols=200_000, sbr=sbr, v=eye.v, rng=np.random.default_rng(0))
    ui_per_s = res.n_symbols / (time.perf_counter() - t0)
    print(f"\n{rate}G DFE throughput: {ui_per_s/1e6:.2f} M UI/s ({p.by_name('dfe').get('n_taps'):.0f} taps)")
    assert ui_per_s >= 1.5e6
