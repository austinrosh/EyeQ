"""Statistical engine: cascade, SBR, and the PDA statistical eye."""

import time

import numpy as np
import pytest

from eyeq.core.context import Modulation
from eyeq.engines import StatisticalEngine
from eyeq.io import build_pipeline, default_link_config

ENG = StatisticalEngine()


def pipe(mod="PAM4", reach="VSR", **overrides):
    p = build_pipeline(default_link_config(modulation=mod, reach_class=reach))
    if overrides:
        p.apply_params(overrides)
    return p


def _neutralize_eq(p):
    """Flatten CTLE + FFE so the link is ~all-pass (for ideal-channel checks)."""
    p.apply_params({
        "ctle": {"fz": 1.0, "fp": 1.0, "fpp": 3.0, "zeta_pp": 0.7, "dc_gain": 0.0},
        "txffe": {"pre": 0.0, "post": 0.0, "tr_ui": 0.1},
        "channel": {"loss_scale": 0.0},
    })
    return p


# --------------------------------------------------------------------------- #
# cascade
# --------------------------------------------------------------------------- #
def test_cascade_shapes_and_traces():
    p = pipe()
    c = ENG.cascade(p)
    n = p.ctx.fft_len() // 2 + 1
    assert c.H_channel.shape == c.H_tx_chan.shape == c.H_tx_chan_rx.shape == (n,)
    # TX+Channel+RX includes the channel magnitude as a factor.
    assert np.all(np.abs(c.H_tx_chan_rx) <= np.abs(c.H_tx_chan) * 1e3)


def test_cascade_matches_hand_computed_channel():
    # Channel-only |H| at the reference Nyquist must equal 10^(-loss/20).
    p = pipe(reach="VSR")
    c = ENG.cascade(p)
    i = int(np.argmin(np.abs(c.f - p.ctx.reach.ref_nyquist_hz)))
    assert abs(c.H_channel[i]) == pytest.approx(10 ** (-16.0 / 20), rel=0.01)
    assert abs(c.H_channel[0]) == pytest.approx(1.0, abs=1e-6)  # ~0 dB at DC


def test_cascade_nyquist_loss_reported():
    assert ENG.cascade(pipe(reach="VSR")).nyquist_loss_db == pytest.approx(16.0, abs=0.1)
    # NRZ samples the same channel at 2x Nyquist -> more loss.
    assert ENG.cascade(pipe(mod="NRZ", reach="VSR")).nyquist_loss_db > 24.0


# --------------------------------------------------------------------------- #
# SBR
# --------------------------------------------------------------------------- #
def test_sbr_cursor_layout():
    p = pipe()
    s = ENG.sbr(p)
    pre, post = p.ctx.cursor_span()
    assert s.cursors.size == pre + post + 1
    assert s.cursor_k[0] == -pre and s.cursor_k[-1] == post
    assert 0 in s.cursor_k


def test_sbr_main_cursor_is_peak_and_positive():
    s = ENG.sbr(pipe())
    assert s.main_cursor > 0
    assert s.main_cursor == pytest.approx(np.max(s.sbr), rel=1e-9)


def test_ideal_link_has_negligible_isi_and_full_main():
    p = _neutralize_eq(pipe())
    s = ENG.sbr(p)
    swing = p.by_name("txffe").get("swing")
    assert s.main_cursor == pytest.approx(swing / 2, rel=0.05)  # ~full launch
    assert s.isi_sum / s.main_cursor < 0.05                     # no fabricated ISI


def test_more_loss_reduces_main_cursor():
    lo = ENG.sbr(pipe(reach="XSR")).main_cursor
    hi = ENG.sbr(pipe(reach="LR")).main_cursor
    assert hi < lo


# --------------------------------------------------------------------------- #
# statistical eye
# --------------------------------------------------------------------------- #
def test_eye_columns_are_distributions():
    eye = ENG.stat_eye(pipe())
    assert eye.pdf.shape == (32, 256)
    assert np.allclose(eye.pdf.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(eye.pdf >= 0)


@pytest.mark.parametrize("mod,n_clusters", [("NRZ", 2), ("PAM4", 4)])
def test_ideal_eye_concentrates_at_levels(mod, n_clusters):
    p = _neutralize_eq(pipe(mod=mod))
    eye = ENG.stat_eye(p)
    center = eye.pdf[np.argmin(np.abs(eye.t_ui))]  # PDF at best sampling phase
    # Count well-separated probability clusters -> number of PAM levels.
    occupied = center > (center.max() * 0.05)
    clusters = np.sum(np.diff(occupied.astype(int)) == 1) + int(occupied[0])
    assert clusters == n_clusters


def test_eye_height_shrinks_with_loss():
    open_eye = ENG.stat_eye(pipe(mod="NRZ", reach="XSR")).eye_height_v
    closed = ENG.stat_eye(pipe(mod="NRZ", reach="LR")).eye_height_v
    assert open_eye > closed


def test_noise_blurs_the_eye():
    sharp = ENG.stat_eye(_neutralize_eq(pipe(mod="NRZ")))
    p = _neutralize_eq(pipe(mod="NRZ"))
    p.apply_params({"noise": {"sigma_mvrms": 20.0}})
    blurred = ENG.stat_eye(p)
    # Peak density drops when the same mass is spread by noise.
    assert blurred.pdf.max() < sharp.pdf.max()


# --------------------------------------------------------------------------- #
# performance
# --------------------------------------------------------------------------- #
def test_lti_recompute_is_fast():
    p = pipe()
    ENG.cascade(p)  # warm
    t0 = time.perf_counter()
    for _ in range(20):
        ENG.sbr(p)  # the cascade+SBR recompute a slider triggers
    dt_ms = (time.perf_counter() - t0) / 20 * 1e3
    print(f"\ncascade+SBR recompute: {dt_ms:.3f} ms")
    assert dt_ms < 10.0  # generous; typically well under 1 ms


@pytest.mark.parametrize("rate,mod", [(112, "PAM4"), (224, "PAM4"), (448, "PAM4"), (112, "NRZ")])
def test_engine_runs_at_all_rates(rate, mod):
    p = build_pipeline(default_link_config(data_rate_gbps=rate, modulation=mod))
    c, s, e = ENG.compute(p)
    assert np.isfinite(s.main_cursor)
    assert np.allclose(e.pdf.sum(axis=1), 1.0, atol=1e-6)
