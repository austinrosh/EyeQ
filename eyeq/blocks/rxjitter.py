"""RX jitter — the recovered sampling clock's timing error.

The receiver's CDR-recovered sampling clock has its own jitter, uncorrelated with
the data: oscillator/PLL phase noise (random, **RJ**) and reference/supply spurs
(periodic, **PJ**). Unlike TX/data jitter, this is the loop's *own* noise — but both
data jitter and clock jitter are shaped by the CDR error response :math:`(1-H)`
(``cdr_slicer.error_response``): only the part above the loop bandwidth closes the
eye. RJ is broadband (closes fully, adds in quadrature with TX RJ); PJ at its
frequency closes the eye scaled by :math:`|1-H(f_{pj})|`.

Dual-nature like ``txjitter``: the statistical engine convolves the slope-converted
jitter PDF into the eye, and the transient engine injects a per-symbol sample shift.
The PJ *frequency* only sets the transient sinusoid rate + the CDR-transfer scaling.
"""

from __future__ import annotations

from ..core.block import BlockBase
from ..core.registry import register
from ..core.schema import Kind, Param


@register("RXJitter")
class RXJitter(BlockBase):
    name = "rxjitter"
    is_lti = False  # stochastic
    PARAMS = [
        Param("rj_mui", 0.0, 50.0, 0.0, unit="mUI", kind=Kind.NONLINEAR,
              also_statistical=True),
        Param("pj_mui", 0.0, 100.0, 0.0, unit="mUI", kind=Kind.NONLINEAR,
              also_statistical=True),
        # the frequency shapes the statistical eye through the CDR transfer |1-H(f)|,
        # so it is also_statistical (recomputes the eye), not transient-only.
        Param("pj_freq_mhz", 0.0, 1000.0, 100.0, unit="MHz", kind=Kind.NONLINEAR,
              also_statistical=True),
    ]
