"""TX jitter.

Injects transmit timing jitter with the standard random/deterministic split:

* ``rj_mui``      — random jitter (RJ), Gaussian, specified as an RMS.
* ``dcd_mui``     — duty-cycle distortion (DCD), a bounded even/odd-UI offset (a
  2-Dirac jitter), specified peak-to-peak.
* ``sj_mui`` / ``sj_freq_mhz`` — sinusoidal/periodic jitter (SJ/PJ), an arcsine
  distribution of peak amplitude ``sj_mui`` at ``sj_freq_mhz``.

It is a dual-nature block: the transient engine injects each component as a
sampling-time perturbation, and the statistical engine convolves the combined
jitter PDF into the eye (jitter -> voltage noise via the local slope), hence
``also_statistical=True``. The SJ *frequency* only sets the transient sinusoid's
rate — the marginal eye sees the arcsine regardless — so it is transient-only.
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
        Param("dcd_mui", 0.0, 50.0, 0.0, unit="mUI", kind=Kind.NONLINEAR,
              also_statistical=True),
        Param("sj_mui", 0.0, 100.0, 0.0, unit="mUI", kind=Kind.NONLINEAR,
              also_statistical=True),
        # frequency shapes the statistical eye via the CDR jitter transfer |1-H(f)|
        # (in the CDR tracking modes), so it must recompute the eye -> also_statistical.
        Param("sj_freq_mhz", 0.0, 1000.0, 100.0, unit="MHz", kind=Kind.NONLINEAR,
              also_statistical=True),
    ]
