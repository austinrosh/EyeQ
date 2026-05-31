"""Declarative parameter schema.

Every block exposes a list of :class:`Param`. This single source of truth drives
three things with no per-block GUI code:

1. **GUI auto-binding** — a slider/spinbox is generated from each Param's
   ``min``/``max``/``scale``; adding a block automatically surfaces its controls.
2. **Validation** — :meth:`Param.clamp` keeps values in range.
3. **Update routing** — :class:`Kind` tells the controller where a change goes:

   * ``LTI``        -> recompute the (fast, event-driven) statistical engine.
   * ``NONLINEAR``  -> push to the transient worker, applied at a batch boundary.
   * ``STRUCTURAL`` -> rebuild the pipeline / re-derive sizes (tap counts, rate,
     modulation): these resize buffers the transient thread owns and so are
     neither a cheap recompute nor a hot-path push.

   Dual-nature parameters (TX/RX jitter, channel noise) are ``NONLINEAR`` *and*
   carry ``also_statistical=True``: the transient engine injects them while the
   statistical engine consumes their RMS as a PDF parameter.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Sequence


class Scale(Enum):
    """How a control maps its slider position to a value."""

    LINEAR = auto()
    LOG = auto()


class Kind(Enum):
    """Routing class for a parameter change."""

    LTI = auto()
    NONLINEAR = auto()
    STRUCTURAL = auto()


@dataclass(frozen=True)
class Param:
    """One controllable parameter of a block.

    ``choices`` makes the Param an enumeration (a dropdown, not a slider); its
    value is one of ``choices`` and is not numerically clamped. Otherwise the
    value is a float clamped to ``[min, max]``.
    """

    name: str
    min: float
    max: float
    default: float
    unit: str = ""
    scale: Scale = Scale.LINEAR
    kind: Kind = Kind.LTI
    step: Optional[float] = None
    choices: Optional[Sequence] = None
    also_statistical: bool = False

    @property
    def is_choice(self) -> bool:
        return self.choices is not None

    def clamp(self, v):
        """Clamp a numeric value; pass choice/enumerated values through unchanged."""
        if self.is_choice:
            if v not in self.choices:
                raise ValueError(
                    f"{self.name}: {v!r} not in choices {tuple(self.choices)!r}"
                )
            return v
        return max(self.min, min(self.max, float(v)))

    def to_slider(self) -> dict:
        """Widget config for the GUI auto-binder (Phase 4 consumes this)."""
        return {
            "name": self.name,
            "min": self.min,
            "max": self.max,
            "default": self.default,
            "unit": self.unit,
            "scale": self.scale.name.lower(),
            "kind": self.kind.name.lower(),
            "step": self.step,
            "choices": list(self.choices) if self.choices is not None else None,
        }
