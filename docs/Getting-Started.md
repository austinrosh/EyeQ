# EyeQ — Getting Started & Usage Guide

This guide takes you from a fresh checkout to confidently driving the dashboard and scripting the
engine. For the underlying theory and equations, see the
**[Technical Reference](EyeQ-Technical-Reference.md)**; for a high-level overview and motivation, see the
**[README](../README.md)**.

**Contents**

1. [What EyeQ is EyeQ](#1-what-eyeq-is-in-one-minute)
2. [Installation](#2-installation)
3. [Launching the dashboard](#3-launching-the-dashboard)
4. [A tour of the dashboard](#4-a-tour-of-the-dashboard)
5. [Core workflows](#5-core-workflows)
6. [Switching scenarios & saving configs](#6-switching-scenarios--saving-configs)
7. [Headless scripting recipes](#7-headless-scripting-recipes)
8. [Understanding the numbers (and their limits)](#8-understanding-the-numbers-and-their-limits)
9. [Troubleshooting & FAQ](#9-troubleshooting--faq)

---

## 1. What EyeQ is 
EyeQ models a high-speed serial link **behaviorally**: each block is represented by its effect (a
transfer function, a noise/jitter PDF, or a decision rule) rather than its transistors. It then computes
the receiver eye and the bit error rate two ways — a fast **statistical** method that reaches BER ~10⁻¹⁸,
and a **transient** Monte-Carlo method that shows the live density eye. Because this is far faster than
circuit simulation, the dashboard updates as you turn knobs, making it a tool for *exploring* link and
equalizer trade-offs. See the [README motivation](../README.md#why-eyeq) for the full rationale.

---

## 2. Installation

### Requirements

- **Python 3.11** (the DFE/CDR hot loop uses Numba; 3.11 has the most battle-tested wheels).
- macOS / Linux. The dashboard needs a working Qt platform (PySide6).

### Steps

```bash
# from the repository root
python3.11 -m venv ~/eyeq-venv
~/eyeq-venv/bin/pip install -e ".[dev,sim,gui]"
```

The optional extras are:

| Extra | Pulls in | Needed for |
|---|---|---|
| `sim` | numpy, scipy, numba, scikit-rf | the engine (statistical + transient + Touchstone) |
| `gui` | pyqtgraph, PySide6 | the dashboard |
| `dev` | pytest | running the test suite |

The headless engine works with just `sim`; install `gui` only if you want the dashboard.

### macOS: Qt platform-plugin errors

If the dashboard fails to start with `Could not find the Qt platform plugin "cocoa"`, the Qt plugin
files have likely been marked with the macOS *hidden* flag. `run_dashboard.sh` clears it and pins the
plugin path automatically on every launch; if you start the module directly, clear it yourself:

```bash
chflags -R nohidden "$(~/eyeq-venv/bin/python -c 'import os,PySide6; print(os.path.dirname(PySide6.__file__))')"
```

---

## 3. Launching the dashboard

```bash
./run_dashboard.sh                                  # 112G PAM-4 VSR (default)
./run_dashboard.sh --config examples/112g_pam4.yaml # a saved scenario
./run_dashboard.sh --config examples/112g_nrz.yaml  # NRZ
```

`run_dashboard.sh` uses `~/eyeq-venv` (falling back to an in-project `.venv`), pins the Qt plugin path,
self-heals the hidden flag, and runs `python -m eyeq.gui.dashboard`.

**First run:** click **Start** (top-left) to begin the transient engine, then **Auto-EQ**. On a
loss-limited reach (VSR/MR/LR) the eye starts closed and visibly opens. To quit, close the window (red
button / Cmd-Q) or press Ctrl-C in the launching terminal.

---

## 4. A tour of the dashboard

![EyeQ dashboard — 112G PAM-4](img/dashboard_pam4.png)

### Panels

- **Eye** — the live RX density eye (a phase × voltage histogram, **Turbo** rainbow colormap by default).
  The red dashed line is the CDR-recovered sampling phase; the top-left annotation gives **eye height**
  (at that phase), **eye width** (timing margin), and the sampling point. The title shows the live MSE-SNR
  and SER. **Hover** anywhere on the eye for a crosshair with a live UI / mV readout. The amplitude axis is
  swing-tied so the eye *breathes* with loss, and expands as needed so an equalized eye is never clipped
  top/bottom. Colormap, light/dark theme, log/linear density, amplitude axis, and **eye liveliness (avg
  factor)** are all in the **View** menu.
- **Histogram** — the amplitude distribution at the decision phase (the eye's vertical slice). PAM-4
  shows four lobes; NRZ shows two.
- **Frequency cascade** — the magnitude (dB) vs `f/f_nyq` of three cumulative transfers: Channel,
  TX+Channel, and TX+Channel+RX. The dashed vertical line is Nyquist; the title reports Nyquist loss.
- **Pulse response (SBR)** — the single-bit response in volts, with the sampled cursors (main + pre/post)
  marked. This is the ISI fingerprint the equalizers fight.
- **Controls** — an auto-generated panel, one collapsible group per block (source, txffe, channel,
  noise, ctle, rxffe, dfe, cdr_slicer), with a slider or combo per parameter. Dragging a slider updates
  the relevant plots live.

### Toolbar (left → right)

| Control | What it does |
|---|---|
| **Start / Stop** | Run / pause the transient (Monte-Carlo) engine. |
| **Auto-EQ** | One-click closed-form MMSE: co-optimizes CTLE peaking → TX FFE → RX FFE → DFE. |
| **EQ** checkboxes | Per-stage *true bypass*: CTLE · TX-FFE · TX-drv · RX-FFE · DFE. |
| **Detector** combo | Receiver architecture: **Slicer** / **DFE** / **MLSD**. |
| **Detector…** | Opens the detector settings (MLSD trellis memory `L`). |
| **Bathtub** | Opens the bathtub-curves window. |
| **Report** | Opens the link performance report. |
| **FEC** / **FEC…** | Master FEC on/off, and the FEC settings window. |
| **Mod / Rate** | Modulation (NRZ/PAM4) and data rate (112/224/448 Gb/s). |
| **Load / Save** | Read/write a link configuration (YAML/JSON). |

### Plot interaction (all plots)

- **Scroll** to zoom, **drag** to pan.
- **Double-click** (or right-click → *Reset view*) to fit the plot to its natural extents — handy when a
  plot gets stuck at an awkward zoom.
- **Right-click → Export PNG… / Export CSV…** — PNG on every plot; CSV wherever the plot is data-backed
  (the eye density matrix, the cascade, the SBR, the bathtub curves, the report table). Filenames default
  to `<plot>_<timestamp>` in your last config directory.

---

## 5. Core workflows

### 5.1 Open a closed eye with Auto-EQ

Start on a loss-limited reach (the default VSR, or pick MR/LR/LR-noise). The raw eye is closed
(SER ≈ 0.5). Click **Auto-EQ**: EyeQ sweeps the CTLE peaking level × TX/RX split, solving the MMSE RX FFE
and DFE taps for each, and keeps the best analytic post-DFE SNR. The CTLE's analogue peaking does the
bulk of the high-loss equalization *before* the noise-amplifying RX FFE — which is what lets the hardest
reaches (LR, 28 dB) open under noise. Watch the cascade gain a high-frequency boost, the SBR cursors
tighten, and the eye open. The control sliders update to the solved CTLE/tap values.

The RX FFE and DFE are tap-adapted (Auto-EQ / LMS), so there are no manual per-tap sliders — set the
number of taps and let Auto-EQ (or the DFE `adapt` toggle) solve them, matching how a real RX adapts.

### 5.2 Isolate an equalizer stage

The **EQ** checkboxes are *true bypasses* — disabling a stage passes the signal unmodified (not just
zeroed coefficients), and is independent of Auto-EQ (your solved taps are preserved). Use this to answer
"how much is the CTLE actually buying me here?": with the eye open, uncheck **CTLE** and watch it close;
re-check it to restore. Each toggle updates the eye, bathtub, and report live.

### 5.3 Read the bathtub curves

Click **Bathtub**:

![Bathtub curves](img/bathtub.png)

- The **horizontal bathtub** (bottom) is log₁₀(error rate) vs sampling phase — the **timing margin**.
  The shaded region is the eye opening at the target BER.
- The **vertical bathtub** (top) is log₁₀(error rate) vs decision level — the **voltage margin**.
- These reach far below the Monte-Carlo floor (the statistical engine integrates the distribution tails),
  which is the whole point of having a statistical engine.

When **FEC** is on (as shown), the post-FEC curve (green) is overlaid and the **pre-FEC threshold** is
marked — the raw BER must stay left/below that line for the code to deliver the target. This turns the
bathtub into a pass/fail picture.

### 5.4 The performance report

Click **Report**:

![Report](img/report.png)

Each row gives a metric, its unit, a one-line definition, and the live value: pre-FEC BER, SER, COM, eye
height/width, SNR, target BER, sampling phase, the active EQ/CDR/detector state, and (when enabled) the
FEC rows (post-FEC BER, scheme, coding gain, pre-FEC threshold). **Greyed rows are model-limited or not
yet modeled** — EyeQ does not fabricate precision (the deferred SNDR/RLM/ERL/jitter-tolerance metrics
appear here as "— (not modeled)").

**Capture & compare:** click **Capture config** to snapshot the current numbers into a new column; then
change something (toggle an EQ stage, switch the detector, enable FEC) and the report shows the new live
values beside the captured ones with deltas. This is the cleanest way to quantify "DFE vs MLSD at matched
settings" or "what did that CTLE setting buy."

### 5.5 FEC: pre- vs post-FEC

Open **FEC…**:

![FEC settings](img/fec.png)

Pick a scheme (**KP4** RS(544,514), **KR4** RS(528,514), a **custom** RS(n,k), or none), a target
post-FEC BER (e.g. 1e-15), and the error model (random by default; an optional coarse bursty knob). Tick
the **FEC** toolbar box to enable it. The report gains the post-FEC BER and coding gain; the bathtub
overlays the post-FEC curve and the pre-FEC threshold. Post-FEC numbers are **model-based** (hard-decision
RS assuming i.i.d. symbol errors) and are labeled so.

### 5.6 MLSD (sequence detection)

Set the **Detector** combo to **MLSD**. Instead of slicing each symbol, MLSD finds the most-likely
transmitted *sequence* over a trellis built from the channel ISI — it *uses* the ISI rather than
cancelling it. Its BER is governed by the minimum sequence distance, so EyeQ switches the BER computation
to a **minimum-distance union bound** (the report's "BER method" row says so) and the bathtub is labeled
"sequence-error model." The **Detector…** window sets the trellis memory `L` (capped per modulation so a
PAM-4 × large-L choice can't hang the tool). The eye annotation reminds you that *eye opening is not the
MLSD BER predictor*. FEC still runs downstream of either detector — use Capture & compare to put MLSD next
to the DFE.

### 5.7 Export

Right-click any plot → **Export PNG…** (always) or **Export CSV…** (eye matrix, cascade, SBR, bathtub) —
or the report's **Export CSV… / PNG…** buttons — to take results out for a paper or a notebook.

---

## 6. Switching scenarios & saving configs

- **Mod / Rate** selectors switch modulation (NRZ/PAM4) and data rate (112/224/448 Gb/s). Because EyeQ is
  rate-agnostic, only the buffer sizes change; the physical channel is the same — which is why NRZ shows
  ~2× the Nyquist loss of PAM-4 over the same trace.
- **Channel → reach** (in the Controls panel) picks the reach class (XSR / XSR+ / VSR / MR / LR), which
  sets the loss budget and target BER. MR/LR carry reflection notches (they switch the channel toward the
  Touchstone model).
- **Load / Save** round-trip the full setup — scenario, every block's parameters, FEC and detector
  settings — to YAML or JSON. Example configs live in `examples/` (`112g_pam4.yaml`, `112g_pam4_vsr.yaml`,
  `112g_nrz.yaml`). Provide your own measured channel by pointing `channel_s4p` at an `.s4p` file (drop it
  in `examples/data/`).

---

## 7. Headless scripting recipes

The engine is fully usable without Qt. Use `~/eyeq-venv/bin/python`.

**Assess a link and its margins:**

```python
from eyeq.io import build_pipeline, default_link_config
from eyeq.engines import StatisticalEngine
from eyeq.analysis import ber
from eyeq.analysis.optimize import optimize_link

pipe = build_pipeline(default_link_config(modulation="PAM4", reach_class="VSR"))
pipe.apply_params({"noise": {"sigma_mvrms": 5.0}, "txjitter": {"rj_mui": 2.0}})
optimize_link(pipe)

stat = StatisticalEngine()
_, sbr, _ = stat.compute(pipe)
r = ber.assess(stat, pipe, sbr, target_ber=pipe.ctx.reach.target_ber)
print(f"BER={r.ber:.2e}  COM={r.com_db:+.1f} dB  "
      f"eye={r.eye_height_v*1e3:.1f} mV x {r.eye_width_ui:.3f} UI")
```

**Add FEC and compare detectors:**

```python
from eyeq.analysis import fec

dfe = ber.assess(stat, pipe, sbr, target_ber=1e-12)
mlsd = ber.assess_mlsd(stat, pipe, target_ber=1e-12, mlsd_taps=4)
print(f"DFE BER={dfe.ber:.2e}   MLSD BER={mlsd.ber:.2e}  (d_min/2={mlsd.mlsd_dmin/2*1e3:.1f} mV)")

f = fec.assess_fec(dfe, pipe.ctx, {"enabled": True, "scheme": "kp4", "target_post_ber": 1e-15})
print(f"KP4 post-FEC BER={f.post_ber:.2e}  threshold={f.pre_threshold_ber:.2e}  gain={f.coding_gain_db:+.1f} dB")
```

**Sweep a parameter (e.g. CTLE peaking):**

```python
import numpy as np
for fz in np.linspace(0.3, 1.0, 8):
    pipe.apply_params({"ctle": {"fz": float(fz)}})
    _, sbr, _ = stat.compute(pipe)
    r = ber.assess(stat, pipe, sbr, target_ber=1e-6)
    print(f"fz={fz:.2f}  BER={r.ber:.2e}  COM={r.com_db:+.1f} dB")
```

**Run the live transient engine headlessly** (a few batches):

```python
from eyeq.engines import TransientEngine
import numpy as np
eng = TransientEngine()
res = eng.run_batch(pipe, n_symbols=200_000, sbr=sbr, rng=np.random.default_rng(0))
print(f"measured MSE-SNR={res.mse_snr_db:.1f} dB  SER={res.ser:.2e}")
```

`examples/run_link.py` is a runnable end-to-end example; the `tests/` directory has many more usage
patterns.

---

## 8. Understanding the numbers (and their limits)

| Reading | What it means | Caveat |
|---|---|---|
| **Eye height / width** | Vertical (mV) and timing (UI) opening at the target BER | Height shown at the CDR-recovered phase |
| **BER (pre-FEC)** | Raw bit error rate at the slicer (Gray-coded) | Reflects the *linear-equalized* eye; DFE cancellation appears in the transient SER, not this number |
| **COM** | Channel operating margin (signal/noise at the target BER, dB) | Uses the actual residual-tail quantile, not Gaussian Q·σ |
| **MSE-SNR** | Mean-square SNR at the decision point (live, transient) | Monte-Carlo; floor ~10⁻⁵ for SER |
| **Post-FEC BER / coding gain** | RS waterfall estimate | Model-based: assumes i.i.d. (random) symbol errors; Gaussian-approx gain |
| **MLSD BER / margin** | Minimum-distance union bound | Optimistic at low SNR; whitened-matched-filter assumption |

EyeQ's guiding principle: a number is shown with its method and its limits. Greyed/"not modeled" entries
(crosstalk, DJ/SJ, SNDR/RLM/ERL/jitter-tolerance) are genuinely not yet modeled — they are placeholders,
not zeros. The full derivations and assumptions are in the
[Technical Reference §16 (Fidelity)](EyeQ-Technical-Reference.md#16-fidelity-assumptions--limitations).

---

## 9. Troubleshooting & FAQ

**`Could not find the Qt platform plugin "cocoa"` (macOS).** The Qt plugin files have the macOS hidden
flag set. Re-run via `./run_dashboard.sh` (it clears the flag automatically), or do it manually:
`chflags -R nohidden "$(~/eyeq-venv/bin/python -c 'import os,PySide6;print(os.path.dirname(PySide6.__file__))')"`.

**The dashboard launches but the eye is empty.** Click **Start** to run the transient engine; the density
eye accumulates over a few batches.

**The eye looks too static / too noisy.** The **View → Eye liveliness (avg factor)** control sets how many
batches the eye averages: a low factor gives a live, shimmering eye that snaps to tap changes; a high
factor gives a smooth, persistent eye. (Tap and detector edits also reset the accumulation so the change
shows at once.)

**The eye looks stuck at a weird zoom.** Double-click it (or right-click → *Reset view*) to fit it to its
data-driven extents.

**Auto-EQ didn't fully open the eye.** Auto-EQ now peaks the CTLE automatically (toward SOTA boost on the
hardest reaches), but on LR (28 dB) with significant noise the link can still be margin-limited; increase
the DFE tap count (`dfe → n_taps`) or reduce `noise → sigma_mvrms`, then Auto-EQ again. FEC may still
carry it — enable KP4 and check the post-FEC BER.

**"DFE" and "Slicer" detector modes show the same BER.** Expected for the *analytic* BER: it reflects the
linear-equalized eye and does not (yet) model DFE postcursor cancellation analytically (that shows in the
transient SER). The honest analytic comparison is MLSD vs linear-decision.

**Numba is recompiling / the first transient batch is slow.** The kernel compiles once (cached); the
first run after an install or a structural change pays a one-time compile cost.

**Headless on a server with no display.** Don't install/launch the GUI; use the engine API (§7). For
offscreen rendering set `QT_QPA_PLATFORM=offscreen`.

---

*Next: the [Technical Reference](EyeQ-Technical-Reference.md) for the full theory, equations, and module
detail.*
