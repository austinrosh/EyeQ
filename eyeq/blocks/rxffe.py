"""RX FFE — receive feed-forward equalizer (Phase 0 stub).

A symbol- or fractionally-spaced FIR with an optional adaptation mode. The tap
count is STRUCTURAL (resizes state); tap values are LTI (linear filtering);
adaptation mode/rate are NONLINEAR (time-varying). Phase 3 adds LMS/sign-LMS
online adaptation and closed-form MMSE auto-EQ with a main-tap-position search
(Shakiba et al., Part II, Eqs. 6-7).
"""

from __future__ import annotations

from ..core.block import BlockBase
from ..core.registry import register
from ..core.schema import Kind, Param


@register("RXFFE")
class RXFFE(BlockBase):
    name = "rxffe"
    is_lti = True
    PARAMS = [
        Param("n_taps", 1, 31, 1, kind=Kind.STRUCTURAL, step=1),
        Param("adapt", 0, 0, "off", kind=Kind.NONLINEAR,
              choices=("off", "lms", "sign-lms")),
        Param("mu", 0.0, 0.1, 0.0, kind=Kind.NONLINEAR),
    ]
