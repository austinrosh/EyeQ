"""Channel — analytical loss models + composable package stage.

Magnitude (shared by both analytical models) is set by the reach-class loss
budget with a skin/dielectric split, hitting the target loss-at-Nyquist exactly:

    IL_dB(f) = L * (k_skin * sqrt(f/f_nyq) + k_diel * (f/f_nyq)),   k_skin+k_diel=1
    => IL_dB(f_nyq) = L     (the reach class loss budget, scaled by loss_scale)

The two analytical models differ only in **phase** (the magnitude, hence the
loss budget, is identical):

* ``simple`` — minimum-phase reconstruction from the magnitude. Smooth, causal.
* ``tl``     — transmission-line phase from the physical skin/dielectric terms
  (Shakiba et al., Part I): the (1+j)*sqrt(f) skin term and the dielectric
  Kramers-Kronig excess phase ~ (2/pi)*ln(f/f_ref), plus a bulk transport delay.

Fidelity boundary: neither analytical model reproduces MR/LR reflection notches
(``ReachClass.models_reflections`` declares this). The bump-to-bump package
contribution is a separate composable stage (toggled by ``package``), modeled
with the same magnitude form using the reach class's package adder.

``touchstone`` import lands in Phase 2.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core.block import LTIBlock
from ..core.context import SimContext
from ..core.registry import register
from ..core.schema import Kind, Param
from ..spectral import from_db, minimum_phase_spectrum

_MODELS = ("simple", "tl", "touchstone")
_REACH = ("XSR", "XSR+", "VSR", "MR", "LR")

_K_SKIN = 0.6  # skin (sqrt f) fraction of the loss budget
_K_DIEL = 0.4  # dielectric (f) fraction
_NEPER_PER_DB = 1.0 / 8.685889638  # dB -> nepers
_TL_DELAY_UI = 4.0  # bulk transport delay for the tl model (positions the SBR)


def _loss_shape(ctx: SimContext, loss_db: float) -> tuple[NDArray, NDArray]:
    """Return (skin_dB, diel_dB) loss components on the freq grid.

    The budget is anchored to the reach class's *reference* Nyquist (the
    generation's symbol Nyquist, e.g. 28 GHz for 112G/PAM4), NOT to ctx.f_nyq.
    This is what makes a channel physical across modulations: NRZ@112G samples
    the same loss curve at 56 GHz and so sees ~2x the loss of PAM4@112G.
    """
    x = ctx.freq_grid() / ctx.reach.ref_nyquist_hz
    return loss_db * _K_SKIN * np.sqrt(x), loss_db * _K_DIEL * x


@register("Channel")
class Channel(LTIBlock):
    name = "channel"
    PARAMS = [
        Param("model", 0, 0, "simple", kind=Kind.STRUCTURAL, choices=_MODELS),
        Param("reach", 0, 0, "VSR", kind=Kind.STRUCTURAL, choices=_REACH),
        Param("loss_scale", 0.0, 2.0, 1.0, kind=Kind.LTI),
        Param("package", 0, 0, "off", kind=Kind.STRUCTURAL, choices=("off", "on")),
    ]

    # -- magnitude / loss -----------------------------------------------------
    def loss_db_nyq(self, ctx: SimContext) -> float:
        """Effective trace loss-at-Nyquist (reach budget x loss_scale)."""
        return ctx.reach.loss_db_nyq * self.get("loss_scale")

    def insertion_loss_db(self, ctx: SimContext) -> NDArray[np.float64]:
        skin, diel = _loss_shape(ctx, self.loss_db_nyq(ctx))
        return skin + diel

    def loss_at_ref_nyquist_db(self, ctx: SimContext) -> float:
        """Insertion loss at the reach reference Nyquist (== the budget)."""
        il = self.insertion_loss_db(ctx)
        nyq = int(np.argmin(np.abs(ctx.freq_grid() - ctx.reach.ref_nyquist_hz)))
        return float(il[nyq])

    # -- model transfers ------------------------------------------------------
    def _simple_transfer(self, ctx: SimContext) -> NDArray[np.complex128]:
        mag = from_db(-self.insertion_loss_db(ctx))
        return minimum_phase_spectrum(mag, ctx.fft_len())

    def _tl_transfer(self, ctx: SimContext) -> NDArray[np.complex128]:
        f = ctx.freq_grid()
        skin_db, diel_db = _loss_shape(ctx, self.loss_db_nyq(ctx))
        alpha = (skin_db + diel_db) * _NEPER_PER_DB  # attenuation [nepers]
        # Phase: skin (1+j)sqrt(f) -> beta_skin == alpha_skin; dielectric excess
        # phase ~ (2/pi) ln(f/f_ref) * alpha_diel (Kramers-Kronig); + bulk delay.
        a_skin = skin_db * _NEPER_PER_DB
        a_diel = diel_db * _NEPER_PER_DB
        fref = ctx.f_nyq
        with np.errstate(divide="ignore", invalid="ignore"):
            beta_diel = a_diel * (2.0 / np.pi) * np.log(np.where(f > 0, f / fref, 1.0))
        beta_delay = 2.0 * np.pi * f * (_TL_DELAY_UI * ctx.ui)
        beta = a_skin + beta_diel + beta_delay
        return np.exp(-alpha) * np.exp(-1j * beta)

    def _package_transfer(self, ctx: SimContext) -> NDArray[np.complex128]:
        """Composable bump-to-bump package stage (reach package adder)."""
        pkg_db = ctx.reach.pkg_db_nyq
        if pkg_db <= 0.0:
            return np.ones(ctx.freq_grid().size, dtype=np.complex128)
        skin, diel = _loss_shape(ctx, pkg_db)
        return minimum_phase_spectrum(from_db(-(skin + diel)), ctx.fft_len())

    # -- Block API ------------------------------------------------------------
    def transfer(self, ctx: SimContext) -> NDArray[np.complex128]:
        model = self.get("model")
        if model == "touchstone":
            raise NotImplementedError("Touchstone import arrives in Phase 2")
        H = self._tl_transfer(ctx) if model == "tl" else self._simple_transfer(ctx)
        if self.get("package") == "on":
            H = H * self._package_transfer(ctx)
        return H
