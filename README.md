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

- **Phase 0** (skeleton): core contracts, stubbed blocks, config save/load, tests.
- **Phase 1** (statistical core): TX FFE + Gaussian driver, two analytical channel
  models (`simple` minimum-phase + `tl` transmission-line) with a composable
  package stage, the Eq.-22 CTLE, RX FFE; the statistical engine producing the
  frequency cascade, the SBR + cursors, and a PDA statistical eye (NRZ + PAM4).
  Cross-validated against serdespy; sub-ms recompute.

- **Phase 2a** (Touchstone): synthetic reference `.s4p` generator (one per reach
  class; MR/LR carry reflection notches) and the importer (`.s4p` -> mixed-mode
  SDD21 -> simulation-grid transfer), wired into the Channel `touchstone` model.
  Drop your own measured `.s4p` in `examples/data/` and point a config's
  `channel_s4p` at it.
- **Phase 2b** (transient engine): vectorized LTI-only Monte Carlo density eye,
  MSE-SNR + SER, and a threaded worker with a double-buffered snapshot and
  coalesced parameter updates. Keystone validated: the density eye's distribution
  matches the statistical eye (std to <3%, decision SNR to <0.3 dB);
  >= 1.5M UI/s throughput.

- **Phase 3** (nonlinear tail): DFE with a Numba `njit` feedback kernel, CDR/slicer
  with a controllable sample phase, TX/RX jitter + noise injected in the loop, and
  one-click closed-form **MMSE auto-EQ** (RX FFE Eqs. 6-7 + DFE post-cursor
  cancellation). The DFE closes a known-ISI eye; auto-EQ opens a fully-closed LR
  eye (SER 0.5 -> 0); >= 6.5M UI/s at 112G and 448G with a long DFE.
- **Phase 4** (live dashboard): a PyQtGraph + PySide6 dashboard — the density eye,
  frequency cascade, SBR, and amplitude histogram, with a slider panel
  auto-generated from each block's parameter schema, an Auto-EQ button, modulation
  /rate selectors, start/stop, and config load/save. The statistical engine
  recomputes on the GUI thread (LTI sliders); the transient engine runs on a worker
  thread pulled at ~30 FPS, so dragging never blocks.

## Run the dashboard

```bash
pip install -e ".[dev,sim,gui]"
python -m eyeq.gui.dashboard                      # boots a 112G PAM4 VSR link
python -m eyeq.gui.dashboard --config examples/112g_pam4.yaml
```

Click **Start**, then **Auto-EQ** to watch a closed eye open.

- **Phase 5a** (performance assessment): BER, horizontal/vertical **bathtub curves**,
  and **COM** (channel operating margin) from the statistical eye — error rates far
  below what the Monte Carlo eye can reach. The dashboard's voltage panel shows the
  amplitude histogram plus the vertical bathtub, with a live BER / COM readout.

See the build plan for the remaining Phase 5 work (TX FFE auto-EQ + online LMS,
a real CDR loop; crosstalk / package-`.s4p` / jitter decomposition optional).

To (re)generate the reference channels: `python examples/generate_reference_channels.py`.

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
