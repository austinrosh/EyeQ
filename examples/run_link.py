#!/usr/bin/env python
"""Headless example: build a link and run the statistical engine.

Doubles as a living integration test — the engine is fully usable without the
GUI. Prints the frequency-cascade loss, CTLE peaking, the SBR cursors, and an
ASCII render of the PDA statistical eye, then sweeps equalization to show the
eye opening.

Run:  python examples/run_link.py
It also (re)writes the example YAML configs next to this file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from eyeq.engines import StatisticalEngine
from eyeq.io import build_pipeline, default_link_config, load, save

HERE = Path(__file__).resolve().parent
ENG = StatisticalEngine()
_RAMP = " .:-=+*#%@"


def ascii_eye(eye, rows: int = 17, cols: int = 48) -> str:
    """Render the [phase, voltage] PDF as a compact ASCII eye (two UI wide)."""
    pdf = eye.pdf
    # tile two UI horizontally and orient voltage as rows (high at top)
    two = np.vstack([pdf, pdf])  # [2*phase, v]
    img = two.T[::-1]            # [v, 2*phase]
    vi = np.linspace(0, img.shape[0] - 1, rows).astype(int)
    ci = np.linspace(0, img.shape[1] - 1, cols).astype(int)
    g = img[np.ix_(vi, ci)]
    g = np.sqrt(g / (g.max() + 1e-30))  # sqrt for visibility
    out = []
    for r in g:
        out.append("".join(_RAMP[min(len(_RAMP) - 1, int(v * (len(_RAMP) - 1)))] for v in r))
    return "\n".join(out)


def report(scenario: str, modulation: str, reach: str, note: str = "") -> None:
    cfg = default_link_config(modulation=modulation, reach_class=reach)
    pipe = build_pipeline(cfg)
    # A modest TX de-emphasis + CTLE peaking (no DFE yet — that is Phase 3).
    pipe.apply_params({
        "txffe": {"pre": -0.08, "post": -0.12, "swing": 0.8},
        "ctle": {"fz": 0.35, "fp": 1.0, "fpp": 1.5, "zeta_pp": 0.7, "dc_gain": -2.0},
    })
    casc, sbr, eye = ENG.compute(pipe)

    print(f"\n{'='*60}\n{scenario}  ({modulation}, {reach}, "
          f"{pipe.ctx.data_rate/1e9:.0f} Gb/s, fnyq={pipe.ctx.f_nyq/1e9:.0f} GHz)"
          f"{'  — ' + note if note else ''}\n{'='*60}")
    print(f"channel loss @signal-Nyquist : {casc.nyquist_loss_db:6.2f} dB")
    print(f"CTLE peaking @Nyquist        : {pipe.by_name('ctle').peaking_db(pipe.ctx):6.2f} dB")
    print(f"SBR main cursor              : {sbr.main_cursor*1e3:6.1f} mV")
    print(f"SBR residual ISI (sum)       : {sbr.isi_sum*1e3:6.1f} mV  "
          f"(ISI/main = {sbr.isi_sum/abs(sbr.main_cursor):.2f})")
    print(f"statistical eye height       : {eye.eye_height_v*1e3:6.1f} mV "
          f"(best phase {eye.best_phase_ui:+.2f} UI)")
    print(f"peak-distortion bound        : {eye.pda_bound_v*1e3:6.1f} mV")
    print("statistical eye (2 UI):")
    print(ascii_eye(eye))

    path = HERE / f"{scenario}.yaml"
    save(cfg, path)
    assert load(path) == cfg, "config did not round-trip"


def eq_sweep() -> None:
    """Show the eye height improving as CTLE peaking is added (112G PAM4 XSR)."""
    print(f"\n{'='*60}\nEQ sweep — 112G PAM4 XSR: CTLE zero vs statistical eye\n{'='*60}")
    print(f"{'fz (xfnyq)':>12} {'peaking(dB)':>12} {'eye height(mV)':>15}")
    for fz in (0.9, 0.7, 0.5, 0.35, 0.25):
        pipe = build_pipeline(default_link_config(modulation="PAM4", reach_class="XSR"))
        pipe.apply_params({"ctle": {"fz": fz, "fp": 1.0, "fpp": 1.5, "dc_gain": -2.0},
                           "txffe": {"pre": -0.08, "post": -0.12}})
        _, _, eye = ENG.compute(pipe)
        pk = pipe.by_name("ctle").peaking_db(pipe.ctx)
        print(f"{fz:>12.2f} {pk:>12.2f} {eye.eye_height_v*1e3:>15.1f}")


def main() -> None:
    # XSR closes a clean LTI-only eye; VSR/PAM4 is marginal without a DFE (Phase 3).
    report("112g_pam4", "PAM4", "XSR", note="open with CTLE alone")
    report("112g_pam4_vsr", "PAM4", "VSR", note="marginal — needs DFE (Phase 3)")
    report("112g_nrz", "NRZ", "XSR", note="wide open")
    eq_sweep()


if __name__ == "__main__":
    main()
