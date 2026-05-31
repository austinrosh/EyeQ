"""Source — symbol generator.

Generates the transmitted symbol stream mapped to normalized PAM levels. Phase 2b
ships an i.i.d.-uniform generator (the assumption the statistical eye is built on,
so the two engines agree); a deterministic PRBS/PRQS LFSR with Gray mapping is a
later refinement. From the LTI engine's point of view the source is stochastic
(no concatenable impulse).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core.block import BlockBase
from ..core.context import SimContext
from ..core.registry import register
from ..core.schema import Kind, Param

_PATTERNS = ("PRBS7", "PRBS9", "PRBS13", "PRBS15", "PRBS23", "PRBS31", "PRQS10", "PRQS12")


@register("Source")
class Source(BlockBase):
    name = "source"
    is_lti = False  # stochastic: no concatenable impulse
    PARAMS = [
        Param("pattern", 0, 0, "PRBS13", kind=Kind.STRUCTURAL, choices=_PATTERNS),
    ]

    def generate(
        self, n_symbols: int, ctx: SimContext, rng: np.random.Generator
    ) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
        """Return (symbol voltages in normalized levels, symbol indices 0..M-1)."""
        idx = rng.integers(0, ctx.mod.n_levels, n_symbols)
        return ctx.levels[idx], idx
