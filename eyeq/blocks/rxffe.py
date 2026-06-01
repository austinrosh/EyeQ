"""RX FFE — receive feed-forward equalizer.

A symbol-spaced FIR in the LTI chain (its transfer reshapes the SBR). Tap values
default to identity (a single unit main tap) and are set by the closed-form MMSE
auto-EQ (Shakiba et al., Part II, Eqs. 6-7) or online LMS. The main-tap position
is tracked so the filter introduces no net bulk delay.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core.block import LTIBlock
from ..core.context import SimContext
from ..core.registry import register
from ..core.schema import Kind, Param


@register("RXFFE")
class RXFFE(LTIBlock):
    name = "rxffe"
    PARAMS = [
        Param("enabled", 0, 0, "on", kind=Kind.LTI, choices=("off", "on"), hidden=True),
        Param("n_taps", 1, 31, 1, kind=Kind.STRUCTURAL, step=1),
        Param("adapt", 0, 0, "off", kind=Kind.NONLINEAR,
              choices=("off", "lms", "sign-lms")),
        Param("mu", 0.0, 0.1, 0.0, kind=Kind.NONLINEAR),
    ]

    def __init__(self, **overrides):
        super().__init__(**overrides)
        self._taps: NDArray | None = None  # explicit taps (auto-EQ / LMS)
        self._main_pos: int | None = None

    def set_taps(self, taps: NDArray, main_pos: int) -> None:
        self._taps = np.asarray(taps, dtype=float)
        self._main_pos = int(main_pos)
        self.set_params(n_taps=int(self._taps.size))

    def reset_taps(self) -> None:
        self._taps = None
        self._main_pos = None

    def main_pos(self) -> int:
        if self._main_pos is not None:
            return self._main_pos
        return int(self.get("n_taps")) // 2

    def taps(self) -> NDArray[np.float64]:
        if self._taps is not None:
            return self._taps
        n = int(self.get("n_taps"))
        t = np.zeros(n)
        t[n // 2] = 1.0  # identity
        return t

    def transfer(self, ctx: SimContext) -> NDArray[np.complex128]:
        f = ctx.freq_grid()
        if self.get("enabled") == "off":  # true bypass (identity), preserving solved taps
            return np.ones(f.size, dtype=np.complex128)
        taps = self.taps()
        k = np.arange(taps.size) - self.main_pos()  # centered: no net bulk delay
        return (taps[None, :] * np.exp(-1j * 2 * np.pi * np.outer(f, k) * ctx.ui)).sum(1)
