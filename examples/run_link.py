#!/usr/bin/env python
"""Headless smoke example: build link pipelines and print their summaries.

This is the kind of script that doubles as a living integration test — the
engine is fully usable without the GUI. Phase 0 only exercises construction,
the rate-aware context, and config round-tripping; later phases will add the
statistical/transient engines here.

Run:  python examples/run_link.py
It also (re)writes the example YAML configs next to this file.
"""

from __future__ import annotations

import json
from pathlib import Path

from eyeq.core import Kind
from eyeq.io import build_pipeline, default_link_config, load, save

HERE = Path(__file__).resolve().parent


def show(scenario: str, modulation: str, reach: str) -> None:
    cfg = default_link_config(modulation=modulation, reach_class=reach)
    pipe = build_pipeline(cfg)
    ctx = pipe.ctx

    print(f"\n=== {scenario} ===")
    print(json.dumps(ctx.summary(), indent=2))
    print("pipeline:", " -> ".join(pipe.names()))
    print("LTI prefix:", [b.name for b in pipe.lti_prefix()])
    print("nonlinear tail:", [b.name for b in pipe.nonlinear_tail()])

    # Demonstrate update routing: a CTLE peak change is LTI; a DFE tap is NONLINEAR.
    changed = pipe.apply_params({"ctle": {"fz": 0.6}, "dfe": {"h1": 12.0}})
    print("changed kinds from {ctle.fz, dfe.h1}:", sorted(k.name for k in changed))

    # Round-trip the config through YAML.
    path = HERE / f"{scenario}.yaml"
    save(cfg, path)
    reloaded = load(path)
    assert reloaded == cfg, "config did not round-trip"
    print(f"config round-trip OK -> {path.name}")


def main() -> None:
    show("112g_pam4", "PAM4", "VSR")
    show("112g_nrz", "NRZ", "VSR")
    # Same reach, different modulation -> identical derived sizes, different fb.
    pam4 = build_pipeline(default_link_config(modulation="PAM4", reach_class="VSR")).ctx
    nrz = build_pipeline(default_link_config(modulation="NRZ", reach_class="VSR")).ctx
    assert pam4.sbr_len_samples() == nrz.sbr_len_samples()
    assert pam4.f_nyq != nrz.f_nyq
    print(
        f"\nrate-agnostic check: PAM4 f_nyq={pam4.f_nyq/1e9:.1f} GHz, "
        f"NRZ f_nyq={nrz.f_nyq/1e9:.1f} GHz, "
        f"shared sbr_len_samples={pam4.sbr_len_samples()}"
    )


if __name__ == "__main__":
    main()
