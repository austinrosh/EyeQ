"""The Block protocol and a convenience base class.

A block is the unit of the link pipeline. The engines bind against the
structural :class:`Block` protocol (duck-typed, no inheritance required); most
concrete blocks subclass :class:`BlockBase` for free parameter storage.

Two engine-facing methods define the LTI/nonlinear split:

* ``impulse_response(ctx)`` returns the block's LTI contribution at ``ctx.fs``,
  or ``None`` if the block has no concatenable impulse (it is stochastic, e.g.
  a noise source, or genuinely nonlinear, e.g. a DFE).
* ``process(x, state, ctx)`` is the sample-domain path used by the transient
  engine; ``state`` carries DFE taps, CDR phase, adaptation accumulators, RNG.

Because a stochastic block (source/jitter/noise) and a nonlinear block (DFE/CDR)
both return ``None`` from ``impulse_response``, the *boundary* between the LTI
prefix and the nonlinear tail is marked structurally by the ``is_tail`` flag
rather than inferred from ``None``. The statistical engine concatenates the LTI
prefix and models the stochastic blocks' PDFs analytically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

from numpy.typing import NDArray

from .context import SimContext
from .schema import Param


@dataclass
class BlockState:
    """Opaque, mutable per-block state for the transient engine.

    Held as a flat dict of (mostly) preallocated numpy arrays / scalars so the
    transient worker can reuse buffers across batches and the Numba kernels can
    take plain arrays.
    """

    data: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Block(Protocol):
    """Structural contract every pipeline block satisfies."""

    name: str

    @property
    def params(self) -> list[Param]: ...

    def set_params(self, **values) -> None: ...

    def get_params(self) -> dict[str, Any]: ...

    def impulse_response(self, ctx: SimContext) -> Optional[NDArray]: ...

    def init_state(self, ctx: SimContext) -> BlockState: ...

    def process(
        self, x: NDArray, state: BlockState, ctx: SimContext
    ) -> tuple[NDArray, BlockState]: ...


class BlockBase:
    """Default implementation of the :class:`Block` protocol.

    Subclasses declare:

    * ``name``     — canonical lowercase id used for ordering / lookup.
    * ``PARAMS``   — the declarative schema (a list of :class:`Param`).
    * ``is_lti``   — whether the block contributes a concatenable transfer.
    * ``is_tail``  — whether this block begins the nonlinear tail.

    and override ``impulse_response`` / ``process`` as needed. Phase 0 ships
    every block as a passthrough stub; the DSP arrives in later phases.
    """

    name: str = "block"
    PARAMS: list[Param] = []
    is_lti: bool = True
    is_tail: bool = False

    def __init__(self, **overrides):
        self._values: dict[str, Any] = {p.name: p.default for p in self.PARAMS}
        if overrides:
            self.set_params(**overrides)

    # -- parameters -----------------------------------------------------------
    @property
    def params(self) -> list[Param]:
        return list(self.PARAMS)

    def _param(self, name: str) -> Param:
        for p in self.PARAMS:
            if p.name == name:
                return p
        raise KeyError(f"{self.name}: unknown parameter {name!r}")

    def set_params(self, **values) -> None:
        for k, v in values.items():
            self._values[k] = self._param(k).clamp(v)

    def get_params(self) -> dict[str, Any]:
        return dict(self._values)

    def get(self, name: str):
        return self._values[name]

    # -- engine hooks (passthrough defaults / Phase 0 stubs) ------------------
    def impulse_response(self, ctx: SimContext) -> Optional[NDArray]:
        return None

    def init_state(self, ctx: SimContext) -> BlockState:
        return BlockState()

    def process(
        self, x: NDArray, state: BlockState, ctx: SimContext
    ) -> tuple[NDArray, BlockState]:
        return x, state

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"{type(self).__name__}(name={self.name!r}, params={self._values})"


class LTIBlock(BlockBase):
    """Base for blocks defined by a frequency-domain transfer function.

    Subclasses implement :meth:`transfer` (a one-sided complex transfer on
    ``ctx.freq_grid()``); the impulse response is derived from it. The
    statistical engine multiplies the per-block transfers directly.
    """

    is_lti = True

    def transfer(self, ctx: SimContext) -> NDArray:  # pragma: no cover - abstract
        raise NotImplementedError(f"{self.name}: transfer() not implemented")

    def impulse_response(self, ctx: SimContext) -> NDArray:
        from ..spectral import transfer_to_impulse

        return transfer_to_impulse(self.transfer(ctx), ctx.fft_len())
