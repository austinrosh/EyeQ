"""RX FFE — receive feed-forward equalizer.

A symbol-spaced FIR. The tap count is STRUCTURAL (resizes state); tap values are
LTI; adaptation mode/rate are NONLINEAR. Phase 1 ships it as identity (a single
main tap = 1) so the cascade's "TX+Channel+RX" trace is well-defined; the tap
*values* and closed-form MMSE auto-EQ (main-tap-position search, Shakiba et al.,
Part II, Eqs. 6-7) plus online LMS/sign-LMS arrive in Phase 3.
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
        Param("n_taps", 1, 31, 1, kind=Kind.STRUCTURAL, step=1),
        Param("adapt", 0, 0, "off", kind=Kind.NONLINEAR,
              choices=("off", "lms", "sign-lms")),
        Param("mu", 0.0, 0.1, 0.0, kind=Kind.NONLINEAR),
    ]

    def taps(self, ctx: SimContext) -> NDArray[np.float64]:
        """Phase 1: a single unit main tap (identity)."""
        n = int(self.get("n_taps"))
        taps = np.zeros(n)
        taps[n // 2] = 1.0
        return taps

    def transfer(self, ctx: SimContext) -> NDArray[np.complex128]:
        f = ctx.freq_grid()
        taps = self.taps(ctx)
        k = np.arange(taps.size)
        return (taps[None, :] * np.exp(-1j * 2 * np.pi * np.outer(f, k) * ctx.ui)).sum(1)
