"""Channel (Phase 0 stub).

Three composable layers behind one interface, selected by ``model``:

* ``simple``     — IL(f) ~ a*sqrt(f) [skin] + b*f [dielectric], fit to the reach
  class loss-at-Nyquist. Smooth, monotonic; good for XSR/XSR+/VSR. Fast default.
* ``tl``         — the physical transmission-line model (Shakiba et al., Part I):
  gamma(f) = g0 + g1*sqrt(f) + g2*f, S-parameter cascade, termination H21.
* ``touchstone`` — an imported .s4p -> mixed-mode SDD21 -> causal/passive ->
  impulse (needed for MR/LR reflection notches).

Fidelity boundary: the analytical models capture loss budget + slope only; they
do not reproduce MR/LR reflection notches. The reach class declares its regime
via ``ReachClass.models_reflections``.

The bump-to-bump package contribution is a *separate* composable stage (toggled
by ``package``) rather than baked into the trace, enabling package co-design.

Phase 1: simple + tl analytical transfers + package stage. Phase 2: Touchstone.
"""

from __future__ import annotations

from ..core.block import BlockBase
from ..core.registry import register
from ..core.schema import Kind, Param

_MODELS = ("simple", "tl", "touchstone")
_REACH = ("XSR", "XSR+", "VSR", "MR", "LR")


@register("Channel")
class Channel(BlockBase):
    name = "channel"
    is_lti = True
    PARAMS = [
        Param("model", 0, 0, "simple", kind=Kind.STRUCTURAL, choices=_MODELS),
        Param("reach", 0, 0, "VSR", kind=Kind.STRUCTURAL, choices=_REACH),
        Param("loss_scale", 0.0, 2.0, 1.0, kind=Kind.LTI),
        Param("package", 0, 0, "off", kind=Kind.STRUCTURAL, choices=("off", "on")),
    ]
