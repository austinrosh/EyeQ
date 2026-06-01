"""TX FFE + driver.

A 3-tap discrete-time feed-forward equalizer (pre/main/post) followed by the
driver rise-time filter, together forming the transmitter's LTI transfer. The
main tap is derived as ``c0 = max(0.6, 1 - (|pre| + |post|))`` (Shakiba et al.,
Part I, Table 2) so the launch keeps unit peak headroom. The driver is a Gaussian
filter ``H(f) = exp(-(pi f / a)^2)`` with ``a = 0.8 / tr`` (Eqs. 18-19), where
``tr`` is the 20-80% rise time in UI.

The transfer is dimensionless (a shape); the launch voltage ``swing`` is applied
by the engine when scaling the SBR/eye to volts, so the cascade magnitude plot
stays normalized. ``norm`` selects how the FFE is normalized (peak: main tap is
the reference; energy: reserved for a later phase). Phase 3 adds MMSE auto-EQ.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core.block import LTIBlock
from ..core.context import SimContext
from ..core.registry import register
from ..core.schema import Kind, Param


@register("TXFFE")
class TXFFE(LTIBlock):
    name = "txffe"
    PARAMS = [
        Param("ffe_enabled", 0, 0, "on", kind=Kind.LTI, choices=("off", "on"), hidden=True),
        Param("driver_enabled", 0, 0, "on", kind=Kind.LTI, choices=("off", "on"), hidden=True),
        Param("swing", 0.0, 1.2, 0.8, unit="V", kind=Kind.LTI),
        Param("pre", -0.5, 0.5, 0.0, kind=Kind.LTI),
        Param("post", -0.5, 0.5, 0.0, kind=Kind.LTI),
        Param("tr_ui", 0.1, 0.8, 0.4, unit="UI", kind=Kind.LTI),
        Param("norm", 0, 0, "peak", kind=Kind.STRUCTURAL, choices=("peak", "energy")),
    ]

    def __init__(self, **overrides):
        super().__init__(**overrides)
        self._taps: NDArray | None = None  # explicit taps (auto-EQ / LMS)
        self._main_pos: int | None = None

    def set_params(self, **values) -> None:
        # Dragging a manual tap returns to manual mode (drops any auto-EQ override).
        if {"pre", "post", "swing"} & set(values):
            self._taps = None
            self._main_pos = None
        super().set_params(**values)

    def set_taps(self, taps: NDArray, main_pos: int) -> None:
        self._taps = np.asarray(taps, dtype=float)
        self._main_pos = int(main_pos)

    def reset_taps(self) -> None:
        self._taps = None
        self._main_pos = None

    def main_tap(self) -> float:
        """Derived main cursor tap c0 = max(0.6, 1 - sum of |sub-taps|)."""
        return max(0.6, 1.0 - (abs(self.get("pre")) + abs(self.get("post"))))

    def taps(self) -> NDArray[np.float64]:
        """FFE taps: explicit (auto-EQ) if set, else [pre, main, post]."""
        if self._taps is not None:
            return self._taps
        return np.array([self.get("pre"), self.main_tap(), self.get("post")], float)

    def main_pos(self) -> int:
        return self._main_pos if self._main_pos is not None else 1  # [pre, main, post]

    def ffe_transfer(self, ctx: SimContext) -> NDArray[np.complex128]:
        """FIR de-emphasis response: sum_k c_k exp(-j 2 pi f (k - main) UI)."""
        f = ctx.freq_grid()
        taps = self.taps()
        k = np.arange(taps.size) - self.main_pos()  # centered: no net bulk delay
        return (taps[None, :] * np.exp(-1j * 2 * np.pi * np.outer(f, k) * ctx.ui)).sum(1)

    def driver_transfer(self, ctx: SimContext) -> NDArray[np.complex128]:
        """Gaussian driver low-pass H(f) = exp(-(pi f / a)^2), a = 0.8 / tr."""
        f = ctx.freq_grid()
        tr = self.get("tr_ui") * ctx.ui
        a = 0.8 / tr
        return np.exp(-((np.pi * f / a) ** 2)).astype(np.complex128)

    def transfer(self, ctx: SimContext) -> NDArray[np.complex128]:
        # The FFE de-emphasis and the analog driver bandwidth are independently
        # bypassable: disabling the FFE isolates the equalizer's contribution,
        # disabling the driver gives an ideal full-bandwidth launch.
        ones = np.ones(ctx.freq_grid().size, dtype=np.complex128)
        ffe = self.ffe_transfer(ctx) if self.get("ffe_enabled") == "on" else ones
        drv = self.driver_transfer(ctx) if self.get("driver_enabled") == "on" else ones
        return ffe * drv
