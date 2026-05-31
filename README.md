# EyeQ

An interactive SerDes link-modeling tool: a live dashboard that runs a
continuously-updating transient simulation of an end-to-end serial link and
shows the RX eye (NRZ/PAM4) responding in real time to TX/RX FFE, CTLE, DFE,
CDR sampling phase, channel loss, and jitter — a Python analogue of MATLAB
SerDes Toolbox with full control over every block.

Supports 112 / 224 / 448 Gb/s from one codebase. Everything is normalized to
UI and f/f_nyq, so the link rate is metadata; only filter lengths scale.

## Architecture

The link splits into an **LTI cascade** and a **nonlinear/time-varying tail**
(mirroring the IBIS-AMI Init/GetWave split). Two engines operate on the same
configured pipeline:

- **Statistical engine** — fast, deterministic: frequency cascade, single-bit
  response (SBR), and a peak-distortion-analysis statistical eye.
- **Transient engine** — continuous, batched: a Monte Carlo loop through the
  nonlinear tail accumulating a decaying density eye (Numba inner loop).

The engine is headless, scriptable, and testable; the GUI is a thin, swappable
client. See `eyeq/core/` for the contracts (`SimContext`, `Param`/`Kind`,
`Block`, `Pipeline`).

Built on the modeling/optimization framework of Shakiba, Tonietto &
Sheikholeslami, *"High-Speed Wireline Links — Parts I & II"*, IEEE OJSSCS 2024.

## Status

Phase 0 (skeleton): core contracts, stubbed blocks, config save/load, tests.
See the build plan for the phased roadmap.

## Layout

```
eyeq/
  core/      SimContext, param schema, Block protocol, Pipeline, registry
  blocks/    source, txffe, txjitter, channel, noise, ctle, rxffe, dfe, cdr_slicer
  engines/   statistical, transient, worker
  analysis/  eye, ber, snr, jitter, optimize (MMSE auto-EQ)
  io/        touchstone, config
  gui/       dashboard, binding, plots
  validation/ golden tests
examples/    headless scripted setups
tests/       unit + integration tests
```

## Develop

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # core + tests
pip install -e ".[dev,sim,gui]"  # everything (later phases)

pytest                           # run the test suite
python examples/run_link.py      # headless smoke example
```
