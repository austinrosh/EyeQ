"""SimContext — the immutable rate spine of EyeQ.

Everything that scales with link rate or channel loss is derived *here* so that
no block, engine, or GUI widget ever branches on rate. ``SimContext`` is a
frozen value object: changing rate or modulation means constructing a new one.

Key design point (NRZ vs PAM4 at a fixed data rate): the *physical* channel is
the channel; modulation only decides where Nyquist lands. NRZ at 112 Gb/s has a
56 GHz Nyquist while PAM4 at 112 Gb/s has a 28 GHz Nyquist, so the same physical
trace shows roughly twice the loss-at-Nyquist for NRZ. Reach-class presets
therefore describe the physical channel at a *reference* Nyquist; the effective
loss is evaluated at ``f_nyq`` at runtime.

Rate-aware sizes (SBR window, cursor span, default tap counts) are driven by the
*loss budget*, not the rate: the rate only sets the time scale, which ``sps``
absorbs. The numeric constants below are starting values calibrated to the 112G
reach table (Shakiba et al., Part I, Table 1) and are expected to be tuned
against the golden tests in ``eyeq/validation``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from math import log2

import numpy as np
from numpy.typing import NDArray

DEFAULT_SPS = 32  # samples per UI; 16 is the "fast/live" setting.


class Modulation(Enum):
    """Pulse-amplitude modulation order. The enum *value* is the level count."""

    NRZ = 2  # also called PAM2
    PAM4 = 4

    @property
    def n_levels(self) -> int:
        return int(self.value)

    @property
    def bits_per_symbol(self) -> int:
        return int(log2(self.value))


@dataclass(frozen=True)
class ReachClass:
    """An OIF-CEI-style reach class, keyed elsewhere to a (generation, name).

    The loss budget is defined at ``ref_nyquist_hz`` (the generation's reference
    Nyquist). ``pkg_db_nyq`` is the *composable* package adder (the bump-to-bump
    contribution), modeled as a separate LTI stage rather than baked into the
    trace, so package co-design studies can include or exclude it.

    ``models_reflections`` declares the fidelity regime: the smooth analytical
    channel models loss budget + slope only (fine for XSR/XSR+/VSR). MR/LR need
    measured/Touchstone S-parameters to reproduce reflection notches; a smooth
    analytical curve there is a model-fidelity limit, not a bug.
    """

    name: str
    ref_nyquist_hz: float
    loss_db_nyq: float
    pkg_db_nyq: float
    target_ber: float
    models_reflections: bool


# Reach presets keyed to (generation, reach class). 112G/4-PAM budgets are from
# Shakiba et al., "High-Speed Wireline Links - Part I", Table 1 (Nyquist 28 GHz).
# Package adders are approximate round-trip bump-to-bump deltas; tune as needed.
REACH_PRESETS: dict[tuple[str, str], ReachClass] = {
    ("112G", "XSR"): ReachClass("XSR", 28e9, 8.0, 0.0, 1e-9, False),
    ("112G", "XSR+"): ReachClass("XSR+", 28e9, 13.0, 0.0, 1e-6, False),
    ("112G", "VSR"): ReachClass("VSR", 28e9, 16.0, 6.0, 1e-6, False),
    ("112G", "MR"): ReachClass("MR", 28e9, 20.0, 6.0, 1e-6, True),
    ("112G", "LR"): ReachClass("LR", 28e9, 28.0, 6.0, 1e-4, True),
}

DEFAULT_REACH = REACH_PRESETS[("112G", "VSR")]


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


@dataclass(frozen=True)
class SimContext:
    """Immutable per-configuration rate metadata + derived, loss-aware sizes."""

    fb: float  # symbol (baud) rate [sym/s]
    sps: int = DEFAULT_SPS
    mod: Modulation = Modulation.PAM4
    reach: ReachClass = DEFAULT_REACH
    rng_seed: int = 0

    # -- alternate constructor -------------------------------------------------
    @classmethod
    def from_data_rate(
        cls,
        data_rate_gbps: float,
        mod: Modulation = Modulation.PAM4,
        *,
        reach: ReachClass = DEFAULT_REACH,
        sps: int = DEFAULT_SPS,
        rng_seed: int = 0,
    ) -> "SimContext":
        """Build from a *data* rate (Gb/s); the symbol rate is derived per mod."""
        fb = data_rate_gbps * 1e9 / mod.bits_per_symbol
        return cls(fb=fb, sps=sps, mod=mod, reach=reach, rng_seed=rng_seed)

    def with_(self, **changes) -> "SimContext":
        """Return a copy with fields replaced (frozen-dataclass convenience)."""
        return replace(self, **changes)

    # -- basic derived quantities ---------------------------------------------
    @property
    def fs(self) -> float:
        """Sample rate [Hz]."""
        return self.fb * self.sps

    @property
    def f_nyq(self) -> float:
        """Symbol-rate Nyquist frequency [Hz] (= fb / 2)."""
        return self.fb / 2.0

    @property
    def ui(self) -> float:
        """Unit interval [s]."""
        return 1.0 / self.fb

    @property
    def dt(self) -> float:
        """Sample period [s]."""
        return 1.0 / self.fs

    @property
    def data_rate(self) -> float:
        """Data rate [b/s] = fb * bits_per_symbol."""
        return self.fb * self.mod.bits_per_symbol

    @property
    def levels(self) -> NDArray[np.float64]:
        """Normalized PAM levels in [-1, 1] (NRZ: [-1, 1]; PAM4: [-1,-1/3,1/3,1])."""
        return np.linspace(-1.0, 1.0, self.mod.n_levels)

    # -- loss budget ----------------------------------------------------------
    @property
    def loss_budget_db(self) -> float:
        """Trace-only loss-at-(reference-)Nyquist used to size filters/windows."""
        return self.reach.loss_db_nyq

    # -- rate + LOSS-aware sizes (the heart of "rate is metadata") -------------
    # All are functions of the loss budget L, not the rate. The rate sets only
    # the time scale, which `sps` absorbs -> NRZ and PAM4 at the same reach give
    # identical sizes while fb/fs/f_nyq differ.
    def sbr_len_ui(self) -> int:
        return int(_clamp(round(8 + 1.2 * self.loss_budget_db), 16, 128))

    def sbr_len_samples(self) -> int:
        return self.sbr_len_ui() * self.sps

    def cursor_span(self) -> tuple[int, int]:
        """(pre, post) cursor counts around the main cursor."""
        pre = int(_clamp(round(2 + 0.10 * self.loss_budget_db), 2, 8))
        post = int(_clamp(round(4 + 0.50 * self.loss_budget_db), 6, 64))
        return pre, post

    def default_txffe_taps(self) -> tuple[int, int]:
        """Default (pre, post) TX FFE tap counts (main tap implied)."""
        post = int(_clamp(round(0.10 * self.loss_budget_db), 1, 3))
        return 1, post

    def default_rxffe_taps(self) -> int:
        """Default RX FFE tap count (forced odd so a main tap can sit centered)."""
        n = int(_clamp(round(4 + 0.30 * self.loss_budget_db), 5, 31))
        return n if n % 2 == 1 else n + 1

    def default_dfe_taps(self) -> int:
        return int(_clamp(round(0.6 * self.loss_budget_db), 1, 32))

    def fft_len(self) -> int:
        """Power-of-two FFT length for the frequency grid / SBR transforms."""
        return _next_pow2(max(4096, 4 * self.sbr_len_samples()))

    # -- frequency grids ------------------------------------------------------
    def freq_grid(self, n: int | None = None) -> NDArray[np.float64]:
        """One-sided frequency grid [Hz] from 0 to fs/2 with ``n//2 + 1`` points."""
        n = n if n is not None else self.fft_len()
        return np.linspace(0.0, self.fs / 2.0, n // 2 + 1)

    def f_over_fnyq(self, f: NDArray[np.float64]) -> NDArray[np.float64]:
        """Normalize a frequency axis to f / f_nyq."""
        return np.asarray(f, dtype=float) / self.f_nyq

    def summary(self) -> dict:
        """Human-readable snapshot of the context (for logging / examples)."""
        pre, post = self.cursor_span()
        return {
            "data_rate_Gbps": round(self.data_rate / 1e9, 3),
            "fb_Gbaud": round(self.fb / 1e9, 3),
            "modulation": self.mod.name,
            "sps": self.sps,
            "f_nyq_GHz": round(self.f_nyq / 1e9, 3),
            "reach": self.reach.name,
            "loss_budget_dB": self.loss_budget_db,
            "sbr_len_ui": self.sbr_len_ui(),
            "sbr_len_samples": self.sbr_len_samples(),
            "cursor_span(pre,post)": (pre, post),
            "txffe_taps(pre,post)": self.default_txffe_taps(),
            "rxffe_taps": self.default_rxffe_taps(),
            "dfe_taps": self.default_dfe_taps(),
            "fft_len": self.fft_len(),
        }
