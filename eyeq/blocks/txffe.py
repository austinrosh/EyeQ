"""TX FFE + driver (Phase 0 stub).

A discrete-time feed-forward equalizer (pre/main/post taps) followed by the
driver rise-time filter, modeled together as the transmitter's LTI transfer. The
main tap is derived as ``c0 = max(0.6, 1 - sum(|ci|))`` (Shakiba et al., Part I,
Table 2); the driver is a Gaussian filter ``H(f) = exp(-(pi f / a)^2)`` with
``a = 0.8 / tr`` (Eq. 18-19), where ``tr`` is the 20-80% rise time.

Phase 1 fills in: FIR transfer, swing normalization (peak/energy), Gaussian
driver, ``impulse_response``. Phase 3 adds MMSE auto-EQ (Part II, Eqs. 2-3).
"""

from __future__ import annotations

from ..core.block import BlockBase
from ..core.registry import register
from ..core.schema import Kind, Param, Scale


@register("TXFFE")
class TXFFE(BlockBase):
    name = "txffe"
    is_lti = True
    PARAMS = [
        Param("swing", 0.0, 1.2, 0.8, unit="V", kind=Kind.LTI),
        Param("pre", -0.5, 0.5, 0.0, kind=Kind.LTI),
        Param("post", -0.5, 0.5, 0.0, kind=Kind.LTI),
        Param("tr_ui", 0.1, 0.8, 0.4, unit="UI", kind=Kind.LTI),
        Param("norm", 0, 0, "peak", kind=Kind.STRUCTURAL, choices=("peak", "energy")),
    ]
