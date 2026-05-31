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
        Param("swing", 0.0, 1.2, 0.8, unit="V", kind=Kind.LTI),
        Param("pre", -0.5, 0.5, 0.0, kind=Kind.LTI),
        Param("post", -0.5, 0.5, 0.0, kind=Kind.LTI),
        Param("tr_ui", 0.1, 0.8, 0.4, unit="UI", kind=Kind.LTI),
        Param("norm", 0, 0, "peak", kind=Kind.STRUCTURAL, choices=("peak", "energy")),
    ]

    def main_tap(self) -> float:
        """Derived main cursor tap c0 = max(0.6, 1 - sum of |sub-taps|)."""
        return max(0.6, 1.0 - (abs(self.get("pre")) + abs(self.get("post"))))

    def taps(self) -> NDArray[np.float64]:
        """FFE taps in time order [pre, main, post] (symbol-spaced)."""
        return np.array([self.get("pre"), self.main_tap(), self.get("post")], float)

    def ffe_transfer(self, ctx: SimContext) -> NDArray[np.complex128]:
        """FIR de-emphasis response: sum_k c_k exp(-j 2 pi f k UI)."""
        f = ctx.freq_grid()
        taps = self.taps()
        k = np.arange(taps.size)  # delays 0,1,2 in UI (bulk 1-UI delay is harmless)
        return (taps[None, :] * np.exp(-1j * 2 * np.pi * np.outer(f, k) * ctx.ui)).sum(1)

    def driver_transfer(self, ctx: SimContext) -> NDArray[np.complex128]:
        """Gaussian driver low-pass H(f) = exp(-(pi f / a)^2), a = 0.8 / tr."""
        f = ctx.freq_grid()
        tr = self.get("tr_ui") * ctx.ui
        a = 0.8 / tr
        return np.exp(-((np.pi * f / a) ** 2)).astype(np.complex128)

    def transfer(self, ctx: SimContext) -> NDArray[np.complex128]:
        return self.ffe_transfer(ctx) * self.driver_transfer(ctx)
