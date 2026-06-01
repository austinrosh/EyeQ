"""DFE — decision-feedback equalizer.

The first nonlinear-tail block (``is_tail=True``): no concatenable LTI impulse,
because the feedback is decision-directed. The inner feedback loop runs in the
transient engine's Numba kernel.

Tap weights are in **volts** (a tap cancels a postcursor of that amplitude): the
feedback subtracted at the slicer is ``sum_i taps[i] * decided_level[k-i]`` with
the decided level normalized to the PAM alphabet. ``h1`` (mV) is the first tap
exposed as a GUI slider; ``set_taps`` sets the whole vector (auto-EQ / LMS).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core.block import BlockBase
from ..core.registry import register
from ..core.schema import Kind, Param


@register("DFE")
class DFE(BlockBase):
    name = "dfe"
    is_lti = False
    is_tail = True  # marks the start of the nonlinear tail
    PARAMS = [
        Param("enabled", 0, 0, "on", kind=Kind.NONLINEAR, choices=("off", "on"), hidden=True),
        Param("n_taps", 0, 32, 1, kind=Kind.STRUCTURAL, step=1),
        Param("h1", -100.0, 100.0, 0.0, unit="mV", kind=Kind.NONLINEAR),
        Param("adapt", 0, 0, "off", kind=Kind.NONLINEAR,
              choices=("off", "lms", "sign-lms")),
        Param("mu", 0.0, 0.1, 0.0, kind=Kind.NONLINEAR),
    ]

    def __init__(self, **overrides):
        super().__init__(**overrides)
        self._extra_taps = np.zeros(0)  # taps[1:] in volts (set by auto-EQ/LMS)

    def taps(self) -> NDArray[np.float64]:
        """Tap weights in volts: taps[0] = h1, taps[1:] from auto-EQ/LMS."""
        n = int(self.get("n_taps"))
        t = np.zeros(n)
        if n > 0:
            t[0] = self.get("h1") * 1e-3  # h1 [mV] -> V
            m = min(n - 1, self._extra_taps.size)
            if m > 0:
                t[1 : 1 + m] = self._extra_taps[:m]
        return t

    def set_taps(self, volts: NDArray) -> None:
        """Set the full tap vector (volts); syncs h1 and n_taps."""
        volts = np.asarray(volts, dtype=float)
        self.set_params(n_taps=int(volts.size))
        if volts.size > 0:
            self.set_params(h1=float(volts[0]) * 1e3)
            self._extra_taps = volts[1:].copy()
        else:
            self._extra_taps = np.zeros(0)

    def is_active(self) -> bool:
        return int(self.get("n_taps")) > 0 and bool(np.any(self.taps() != 0.0))
