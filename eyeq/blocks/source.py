"""Source — PRBS/PRQS symbol generator (Phase 0 stub).

Generates the transmitted symbol stream: a PRBS/PRQS pattern mapped to NRZ or
PAM4 levels with Gray coding, upsampled to ``ctx.sps`` (zero-order hold). It is
stochastic from the LTI engine's point of view (no concatenable impulse).

Phase 1 fills in: LFSR PRBS7..31 / PRQS, Gray mapping, ZOH upsample.
"""

from __future__ import annotations

from ..core.block import BlockBase
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
