"""TX jitter (Phase 0 stub).

Injects transmit timing jitter. v1 models random jitter (RJ) as Gaussian phase
noise in mUI. It is a dual-nature block: the transient engine injects it as a
sampling-time perturbation, and the statistical engine consumes its RMS as a
jitter PDF (jitter -> voltage noise via the local slope at the sampling point),
hence ``also_statistical=True``.

Phase 3 fills in transient injection; Phase 5 adds DJ/SJ/PJ hooks.
"""

from __future__ import annotations

from ..core.block import BlockBase
from ..core.registry import register
from ..core.schema import Kind, Param


@register("TXJitter")
class TXJitter(BlockBase):
    name = "txjitter"
    is_lti = False  # stochastic
    PARAMS = [
        Param("rj_mui", 0.0, 50.0, 0.0, unit="mUI", kind=Kind.NONLINEAR,
              also_statistical=True),
    ]
