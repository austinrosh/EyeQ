"""CTLE — continuous-time linear equalizer.

A zero-pole-gain rational that expresses both a first-order CTLE and a resonant
(Q-shaping) section, parameterized as in the spec with frequencies normalized to
f/f_nyq so the control is rate-agnostic:

    H(s) = dc * (1 + s/wz) / (1 + s/wp) * wpp^2 / (s^2 + 2*zeta*wpp*s + wpp^2)

* ``dc_gain`` (dB)  -> dc = 10^(dc_gain/20). At s=0, H = dc.
* ``fz``, ``fp``    -> a real zero/pole pair (the main high-frequency peaking);
  fz < fp boosts toward Nyquist.
* ``fpp``, ``zeta_pp`` -> a complex pole pair at natural frequency fpp with
  damping zeta_pp (Q = 1/(2*zeta_pp)), the resonant section.

This is the general form of the reference CTLE (Shakiba et al., Part I, Eq. 22),
whose complex pole/zero pairs are expressed as (omega0, Q). Built by direct
evaluation of H(j*2*pi*f) on the frequency grid (equivalent to scipy zpk).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core.block import LTIBlock
from ..core.context import SimContext
from ..core.registry import register
from ..core.schema import Kind, Param
from ..spectral import from_db


@register("CTLE")
class CTLE(LTIBlock):
    name = "ctle"
    PARAMS = [
        Param("dc_gain", -20.0, 0.0, 0.0, unit="dB", kind=Kind.LTI),
        Param("fz", 0.1, 2.0, 0.5, unit="xfnyq", kind=Kind.LTI),
        Param("fp", 0.5, 3.0, 1.0, unit="xfnyq", kind=Kind.LTI),
        Param("fpp", 0.5, 3.0, 1.5, unit="xfnyq", kind=Kind.LTI),
        Param("zeta_pp", 0.3, 2.0, 0.7, kind=Kind.LTI),
    ]

    def transfer(self, ctx: SimContext) -> NDArray[np.complex128]:
        f = ctx.freq_grid()
        s = 1j * 2.0 * np.pi * f
        wz = 2.0 * np.pi * self.get("fz") * ctx.f_nyq
        wp = 2.0 * np.pi * self.get("fp") * ctx.f_nyq
        wpp = 2.0 * np.pi * self.get("fpp") * ctx.f_nyq
        zeta = self.get("zeta_pp")
        dc = float(from_db(self.get("dc_gain")))

        real_pair = (1.0 + s / wz) / (1.0 + s / wp)
        resonant = wpp**2 / (s**2 + 2.0 * zeta * wpp * s + wpp**2)
        return dc * real_pair * resonant

    def peaking_db(self, ctx: SimContext) -> float:
        """Magnitude at Nyquist relative to DC, in dB (the 'CTLE boost')."""
        H = self.transfer(ctx)
        nyq = int(np.argmin(np.abs(ctx.freq_grid() - ctx.f_nyq)))
        return float(20.0 * np.log10(np.abs(H[nyq]) / np.abs(H[0])))
