"""CDR / slicer (Phase 0 stub).

v1 exposes a static (or swept) sampling phase within the UI as a control; the
slicer makes the symbol decision at that phase. A clean hook is left for a
bang-bang / Mueller-Muller CDR with a loop filter (``kp``/``ki``) in a later
phase. Part of the nonlinear tail, but not the tail boundary (the DFE is).

Phase 3 fills in the slicer + controllable sample phase; Phase 5 adds CDR modes.
"""

from __future__ import annotations

from ..core.block import BlockBase
from ..core.registry import register
from ..core.schema import Kind, Param


@register("CDRSlicer")
class CDRSlicer(BlockBase):
    name = "cdr_slicer"
    is_lti = False
    PARAMS = [
        Param("sample_phase_ui", -0.5, 0.5, 0.0, unit="UI", kind=Kind.NONLINEAR),
    ]
