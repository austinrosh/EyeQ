"""Channel / RX noise (Phase 0 stub).

Additive Gaussian noise specified in mVrms. Dual-nature like jitter: injected by
the transient engine and consumed as a Gaussian PDF by the statistical engine
(hence ``also_statistical=True``). A hook for crosstalk (FEXT/NEXT) aggressors
is added in Phase 5.
"""

from __future__ import annotations

from ..core.block import BlockBase
from ..core.registry import register
from ..core.schema import Kind, Param


@register("Noise")
class Noise(BlockBase):
    name = "noise"
    is_lti = False  # stochastic
    PARAMS = [
        Param("sigma_mvrms", 0.0, 50.0, 0.0, unit="mVrms", kind=Kind.NONLINEAR,
              also_statistical=True),
    ]
