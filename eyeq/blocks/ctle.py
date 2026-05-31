"""CTLE — continuous-time linear equalizer (Phase 0 stub).

A general zero-pole-gain rational that expresses both a first-order CTLE and a
resonant/Q-shaping section. Parameters follow the spec (``fz``, ``fp``, ``fpp``,
``zeta_pp``, ``dc_gain``); frequencies are normalized to f/f_nyq so the control
is rate-agnostic. This maps onto the reference CTLE transfer function (Shakiba
et al., Part I, Eq. 22): a low/Nyquist pole-zero pair plus a roll-off pole, with
complex pole pairs expressed as (omega0, Q) <-> (fpp, zeta_pp).

Phase 1: build via scipy.signal zpk -> frequency response -> impulse response.
"""

from __future__ import annotations

from ..core.block import BlockBase
from ..core.registry import register
from ..core.schema import Kind, Param


@register("CTLE")
class CTLE(BlockBase):
    name = "ctle"
    is_lti = True
    PARAMS = [
        Param("dc_gain", -20.0, 0.0, 0.0, unit="dB", kind=Kind.LTI),
        Param("fz", 0.1, 2.0, 0.5, unit="xfnyq", kind=Kind.LTI),
        Param("fp", 0.5, 3.0, 1.0, unit="xfnyq", kind=Kind.LTI),
        Param("fpp", 0.5, 3.0, 1.5, unit="xfnyq", kind=Kind.LTI),
        Param("zeta_pp", 0.3, 2.0, 0.7, kind=Kind.LTI),
    ]
