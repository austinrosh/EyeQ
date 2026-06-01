"""CDR / slicer.

The slicer samples at a phase within the UI. ``cdr_mode`` selects how that phase
is set: ``static`` uses the ``sample_phase_ui`` slider as a fixed offset, while
``bang-bang`` (Alexander) and ``mueller-muller`` recover the phase from the data
with a PI loop filter (``kp``/``ki``) — ``sample_phase_ui`` is then the initial
condition the loop corrects from, and the phase tracks jitter. Part of the
nonlinear tail (runs in the transient kernel), but not the tail boundary.
"""

from __future__ import annotations

from ..core.block import BlockBase
from ..core.registry import register
from ..core.schema import Kind, Param

_CDR_MODES = ("static", "bang-bang", "mueller-muller")


@register("CDRSlicer")
class CDRSlicer(BlockBase):
    name = "cdr_slicer"
    is_lti = False
    PARAMS = [
        Param("sample_phase_ui", -0.5, 0.5, 0.0, unit="UI", kind=Kind.NONLINEAR),
        Param("cdr_mode", 0, 0, "static", kind=Kind.NONLINEAR, choices=_CDR_MODES),
        Param("kp", 0.0, 0.5, 0.05, kind=Kind.NONLINEAR),
        Param("ki", 0.0, 0.05, 0.001, kind=Kind.NONLINEAR),
    ]

    def cdr_mode_int(self) -> int:
        return _CDR_MODES.index(self.get("cdr_mode"))
