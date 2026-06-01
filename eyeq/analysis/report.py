"""Link performance report — an extensible registry of assessment metrics.

The report is a *registry*, not a hardcoded list: each metric is a :class:`Metric`
descriptor (label, unit, one-line definition, and a ``getter`` that pulls the
value from the live data sources). :func:`evaluate` turns the registry into rows
a GUI table can render directly. Adding a metric — including the still-deferred
SNDR / RLM / ERL / jitter-tolerance — is a one-line append here; nothing at the
call site changes.

Everything is headless (no Qt) so the registry is unit-testable. The three live
data sources are bundled in :class:`ReportContext`:

* ``ber``   — a :class:`~eyeq.analysis.ber.BerResult` (BER, COM, eye opening,
  bathtubs, target BER) from the statistical engine.
* ``stats`` — the live transient snapshot stats dict (MSE-SNR at the slicer,
  SER, eye height at the recovered phase, recovered sampling phase).
* ``pipe``  — the configured :class:`~eyeq.core.pipeline.Pipeline`, for the
  active EQ / CDR state.

A metric whose ``getter`` returns ``None`` is not yet available (the engine does
not compute it, or the transient has not produced a snapshot); it renders as
"—". Metrics whose precision is bounded by the model (BER/COM by the noise model;
the deferred compliance metrics) carry ``model_limited=True`` so the GUI can flag
them honestly rather than imply false precision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class Metric:
    """One report metric: how to label, define, fetch, and format a value."""

    key: str
    label: str
    unit: str
    definition: str
    getter: Callable[["ReportContext"], Any]
    fmt: str = "{:.3g}"        # applied to numeric values; ignored for strings
    model_limited: bool = False  # value precision is bounded by the model (flag, don't hide)
    deferred: bool = False       # not computed by any engine yet (renders "not modeled")


@dataclass
class ReportContext:
    """The live data sources a metric getter may read from."""

    ber: Any                       # eyeq.analysis.ber.BerResult (or None before first assess)
    stats: dict                    # live transient snapshot stats (may be empty)
    pipe: Any                      # eyeq.core.pipeline.Pipeline
    fec: Any = None                # eyeq.analysis.fec.FecResult (or None when FEC is off)
    detector: Any = None           # the controller's detector config dict (mode, mlsd_taps)


@dataclass(frozen=True)
class MetricRow:
    """An evaluated metric ready for display."""

    key: str
    label: str
    value: str                     # formatted text, or "—" when unavailable
    unit: str
    definition: str
    model_limited: bool
    raw: Any                       # unformatted value (for capture/compare deltas); None if N/A


# --------------------------------------------------------------------------- #
# getters (kept small; each reads only what it needs from the ReportContext)
# --------------------------------------------------------------------------- #
def _ber(rc):
    return None if rc.ber is None else rc.ber.ber


def _ser(rc):
    return None if rc.ber is None else rc.ber.ser


def _com(rc):
    return None if rc.ber is None else rc.ber.com_db


def _eye_height_mv(rc):
    v = rc.stats.get("eye_height_at_phase_v")
    return None if v is None else v * 1e3


def _eye_width_ui(rc):
    return None if rc.ber is None else rc.ber.eye_width_ui


def _snr_db(rc):
    return rc.stats.get("mse_snr_db")


def _target_ber(rc):
    return None if rc.ber is None else rc.ber.target_ber


def _sample_phase_ui(rc):
    return rc.stats.get("recovered_phase_ui")


def _eq_state(rc) -> str:
    """Compact on/off summary of every equalizer stage (the bypass toggles)."""
    def on(name: str, param: str = "enabled") -> Optional[bool]:
        try:
            return rc.pipe.by_name(name).get(param) == "on"
        except KeyError:
            return None

    stages = [
        ("CTLE", on("ctle")),
        ("TX-FFE", on("txffe", "ffe_enabled")),
        ("TX-drv", on("txffe", "driver_enabled")),
        ("RX-FFE", on("rxffe")),
        ("DFE", on("dfe")),
    ]
    mark = {True: "✓", False: "✗", None: "-"}
    return "  ".join(f"{n}{mark[s]}" for n, s in stages)


def _cdr_state(rc) -> str:
    try:
        mode = rc.pipe.by_name("cdr_slicer").get("cdr_mode")
    except KeyError:
        return "n/a"
    ph = rc.stats.get("recovered_phase_ui")
    return f"{mode} @ {ph:+.3f} UI" if ph is not None else str(mode)


_MODE_LABEL = {"slicer": "Slicer", "dfe": "DFE", "mlsd": "MLSD"}


def _detector_mode(rc) -> str:
    m = (rc.detector or {}).get("mode", "dfe")
    return _MODE_LABEL.get(m, str(m))


def _ber_method(rc):
    if rc.ber is None:
        return None
    if getattr(rc.ber, "detector", "decision") == "mlsd":
        s = "min-distance union bound (MLSD)"
        return s + " — search truncated" if getattr(rc.ber, "mlsd_truncated", False) else s
    return "eye-tail (decision point)"


def _trellis_l(rc):
    d = rc.detector or {}
    return d.get("mlsd_taps") if d.get("mode") == "mlsd" else None


def _mlsd_margin_mv(rc):
    if rc.ber is None or getattr(rc.ber, "detector", "decision") != "mlsd":
        return None
    dmin = getattr(rc.ber, "mlsd_dmin", float("nan"))
    return (dmin / 2.0) * 1e3 if math.isfinite(dmin) else None


def _fec_on(rc) -> bool:
    return rc.fec is not None and getattr(rc.fec, "enabled", False)


def _post_fec_ber(rc):
    return rc.fec.post_ber if _fec_on(rc) else None


def _fec_scheme(rc) -> str:
    return rc.fec.scheme_label if _fec_on(rc) else "off"


def _coding_gain(rc):
    return rc.fec.coding_gain_db if _fec_on(rc) else None


def _pre_fec_threshold(rc):
    return rc.fec.pre_threshold_ber if _fec_on(rc) else None


def _post_fec_target(rc):
    return rc.fec.target_post_ber if _fec_on(rc) else None


# --------------------------------------------------------------------------- #
# the registry — append a Metric to add a row; no call-site change needed
# --------------------------------------------------------------------------- #
METRICS: list[Metric] = [
    Metric("ber", "BER (pre-FEC)", "", "Raw bit error rate at the slicer (from the statistical "
           "eye, Gray-coded), before any FEC. Bounded below by the front-end noise model.",
           _ber, fmt="{:.2e}", model_limited=True),
    Metric("ser", "SER", "", "Symbol error rate at the optimal sampling phase.",
           _ser, fmt="{:.2e}", model_limited=True),
    Metric("com", "COM", "dB", "Channel operating margin: 20·log10(signal / noise) "
           "at the target BER (Shakiba Part I, Eq. 1).", _com, fmt="{:+.1f}"),
    Metric("eye_height", "Eye height", "mV", "Vertical inner-eye opening measured at the "
           "CDR-recovered sampling phase.", _eye_height_mv, fmt="{:.1f}"),
    Metric("eye_width", "Eye width", "UI", "Horizontal (timing) opening at the target BER, "
           "from the horizontal bathtub.", _eye_width_ui, fmt="{:.3f}"),
    Metric("snr", "SNR", "dB", "MSE-SNR measured at the slicer decision point "
           "(post-DFE, with the live CDR).", _snr_db, fmt="{:.1f}"),
    Metric("target_ber", "Target BER", "", "Spec BER the openings / COM are evaluated at "
           "(the reach class's target).", _target_ber, fmt="{:.0e}"),
    Metric("sample_phase", "Sampling phase", "UI", "CDR-recovered decision phase within the UI.",
           _sample_phase_ui, fmt="{:+.3f}"),
    Metric("eq_state", "Active EQ", "", "On/off state of each equalizer stage.", _eq_state),
    Metric("cdr_state", "CDR", "", "CDR mode and recovered phase.", _cdr_state),
    # ---- detector (receiver architecture) ----
    Metric("detector", "Detector", "", "Active receiver detection mode (Slicer / DFE / MLSD).",
           _detector_mode),
    Metric("ber_method", "BER method", "", "How the displayed BER was computed — eye-tail for the "
           "decision-point detectors, minimum-distance union bound for MLSD.", _ber_method),
    Metric("trellis_l", "Trellis memory", "taps", "MLSD channel-memory length L (— off MLSD).",
           _trellis_l, fmt="{:.0f}"),
    Metric("mlsd_margin", "MLSD margin", "mV", "Half the minimum sequence distance (d_min/2) — the "
           "MLSD analog of the eye half-opening. Model-based (union-bound estimate).",
           _mlsd_margin_mv, fmt="{:.1f}", model_limited=True),
    # ---- FEC (analysis-layer; shown when FEC is enabled) ----
    Metric("post_fec_ber", "Post-FEC BER", "", "Estimated bit error rate after FEC decoding. "
           "Model-based: assumes i.i.d. (random) symbol errors — EyeQ's noise model does not "
           "generate the real bursts a hard-decision RS waterfall would see.",
           _post_fec_ber, fmt="{:.2e}", model_limited=True),
    Metric("fec_scheme", "FEC scheme", "", "Active forward-error-correction code ('off' when "
           "disabled).", _fec_scheme),
    Metric("coding_gain", "Coding gain", "dB", "Gaussian-approx SNR gain vs the no-FEC case at the "
           "target BER: 20·log10(Qinv(target)/Qinv(pre-FEC threshold)).",
           _coding_gain, fmt="{:+.1f}", model_limited=True),
    Metric("pre_fec_threshold", "Pre-FEC threshold", "", "Raw (pre-FEC) BER must stay below this for "
           "the code to deliver the post-FEC target.", _pre_fec_threshold, fmt="{:.2e}"),
    Metric("post_fec_target", "Post-FEC target", "", "Target BER the FEC is designed to deliver "
           "(e.g. 1e-15).", _post_fec_target, fmt="{:.0e}"),
    # ---- deferred compliance metrics: descriptors present, values not yet modeled ----
    Metric("sndr", "SNDR", "dB", "Signal-to-noise-and-distortion ratio (deferred).",
           lambda rc: None, model_limited=True, deferred=True),
    Metric("rlm", "RLM", "", "Ratio of level mismatch / PAM4 linearity (deferred).",
           lambda rc: None, model_limited=True, deferred=True),
    Metric("erl", "ERL", "dB", "Effective return loss (deferred; needs reflection model).",
           lambda rc: None, model_limited=True, deferred=True),
    Metric("jtol", "Jitter tolerance", "UI", "Jitter-tolerance margin (deferred; needs DJ/SJ).",
           lambda rc: None, model_limited=True, deferred=True),
]


def evaluate(rc: ReportContext) -> list[MetricRow]:
    """Evaluate every registered metric against the live context into display rows."""
    rows: list[MetricRow] = []
    for m in METRICS:
        try:
            val = m.getter(rc)
        except Exception:
            val = None
        if val is None:
            disp = "— (not modeled)" if m.deferred else "—"
        elif isinstance(val, str):
            disp = val
        else:
            disp = m.fmt.format(val)
        rows.append(MetricRow(m.key, m.label, disp, m.unit, m.definition, m.model_limited, val))
    return rows
