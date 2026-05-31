"""DFE — decision-feedback equalizer (Phase 0 stub).

The first nonlinear-tail block (``is_tail=True``): it has no concatenable LTI
impulse because its feedback is decision-directed. The inner feedback loop is the
performance-critical path and will be a Numba ``njit(nogil=True)`` kernel in the
transient engine (Phase 3); the decision is made at the symbol center and the
post-cursor feedback is applied for the following UI.

Parameters: ``n_taps`` (STRUCTURAL, resizes state), the first-tap weight ``h1``
and its delay ``h1_td``, and adaptation mode/rate (NONLINEAR). Phase 3 fills in
the loop + LMS/sign-LMS adaptation.
"""

from __future__ import annotations

from ..core.block import BlockBase
from ..core.registry import register
from ..core.schema import Kind, Param


@register("DFE")
class DFE(BlockBase):
    name = "dfe"
    is_lti = False
    is_tail = True  # marks the start of the nonlinear tail
    PARAMS = [
        Param("n_taps", 0, 32, 1, kind=Kind.STRUCTURAL, step=1),
        Param("h1", -100.0, 100.0, 0.0, unit="mV", kind=Kind.NONLINEAR),
        Param("h1_td", 0.0, 1.0, 0.5, unit="UI", kind=Kind.NONLINEAR),
        Param("adapt", 0, 0, "off", kind=Kind.NONLINEAR,
              choices=("off", "lms", "sign-lms")),
        Param("mu", 0.0, 0.1, 0.0, kind=Kind.NONLINEAR),
    ]
