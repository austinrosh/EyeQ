"""Channel — analytical loss models + composable package stage + Touchstone.

Selected by ``model``:

* ``simple``     — minimum-phase transfer with the reach-class loss-budget
  magnitude (skin + dielectric). Smooth; good for XSR/XSR+/VSR.
* ``tl``         — transmission-line phase from the physical skin/dielectric
  terms plus a bulk delay (same magnitude as ``simple``).
* ``touchstone`` — an imported .s4p -> mixed-mode SDD21 -> impulse, resampled
  onto the simulation grid (needed for MR/LR reflection notches; lets the user
  drop in measured channels).

The analytical magnitude is anchored to the reach class's *reference* Nyquist, so
NRZ@112G (56 GHz Nyquist) sees ~2x the loss of PAM4@112G (28 GHz) over the same
trace. The bump-to-bump package contribution is a separate composable stage
(toggled by ``package``). Fidelity boundary: neither analytical model reproduces
MR/LR reflection notches (``ReachClass.models_reflections`` declares this).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .. import channel_model as cm
from ..core.block import LTIBlock
from ..core.context import SimContext
from ..core.registry import register
from ..core.schema import Kind, Param

_MODELS = ("simple", "tl", "touchstone")
_REACH = ("XSR", "XSR+", "VSR", "MR", "LR")
_TL_DELAY_UI = 4.0  # bulk transport delay for the tl model (positions the SBR)


@register("Channel")
class Channel(LTIBlock):
    name = "channel"
    PARAMS = [
        Param("model", 0, 0, "simple", kind=Kind.STRUCTURAL, choices=_MODELS),
        Param("reach", 0, 0, "VSR", kind=Kind.STRUCTURAL, choices=_REACH),
        Param("loss_scale", 0.0, 2.0, 1.0, kind=Kind.LTI),
        Param("package", 0, 0, "off", kind=Kind.STRUCTURAL, choices=("off", "on")),
    ]

    def __init__(self, **overrides):
        super().__init__(**overrides)
        self._touchstone_path: str | None = None

    # -- Touchstone source ----------------------------------------------------
    def set_touchstone(self, path: str | None) -> None:
        self._touchstone_path = path

    @property
    def touchstone_path(self) -> str | None:
        return self._touchstone_path

    # -- magnitude / loss -----------------------------------------------------
    def loss_db_nyq(self, ctx: SimContext) -> float:
        """Effective trace loss-at-(reference-)Nyquist (budget x loss_scale)."""
        return ctx.reach.loss_db_nyq * self.get("loss_scale")

    def insertion_loss_db(self, ctx: SimContext) -> NDArray[np.float64]:
        return cm.insertion_loss_db(ctx.freq_grid(), ctx.reach.ref_nyquist_hz, self.loss_db_nyq(ctx))

    def loss_at_ref_nyquist_db(self, ctx: SimContext) -> float:
        """Insertion loss at the reach reference Nyquist (== the budget)."""
        il = self.insertion_loss_db(ctx)
        nyq = int(np.argmin(np.abs(ctx.freq_grid() - ctx.reach.ref_nyquist_hz)))
        return float(il[nyq])

    # -- model transfers ------------------------------------------------------
    def _simple_transfer(self, ctx: SimContext) -> NDArray[np.complex128]:
        return cm.simple_transfer(ctx.freq_grid(), ctx.reach.ref_nyquist_hz, self.loss_db_nyq(ctx))

    def _tl_transfer(self, ctx: SimContext) -> NDArray[np.complex128]:
        return cm.tl_transfer(
            ctx.freq_grid(), ctx.reach.ref_nyquist_hz, self.loss_db_nyq(ctx),
            _TL_DELAY_UI * ctx.ui,
        )

    def _touchstone_transfer(self, ctx: SimContext) -> NDArray[np.complex128]:
        if not self._touchstone_path:
            raise ValueError("channel model is 'touchstone' but no .s4p path is set")
        from ..io.touchstone import s4p_to_transfer  # lazy: skrf only when used

        return s4p_to_transfer(self._touchstone_path, ctx)

    def _package_transfer(self, ctx: SimContext) -> NDArray[np.complex128]:
        """Composable bump-to-bump package stage (reach package adder)."""
        pkg_db = ctx.reach.pkg_db_nyq
        if pkg_db <= 0.0:
            return np.ones(ctx.freq_grid().size, dtype=np.complex128)
        return cm.simple_transfer(ctx.freq_grid(), ctx.reach.ref_nyquist_hz, pkg_db)

    # -- Block API ------------------------------------------------------------
    def transfer(self, ctx: SimContext) -> NDArray[np.complex128]:
        model = self.get("model")
        if model == "touchstone":
            H = self._touchstone_transfer(ctx)
        elif model == "tl":
            H = self._tl_transfer(ctx)
        else:
            H = self._simple_transfer(ctx)
        if self.get("package") == "on":
            H = H * self._package_transfer(ctx)
        return H
