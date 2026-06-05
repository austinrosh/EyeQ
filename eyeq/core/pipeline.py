"""The ordered link pipeline and its LTI-prefix / nonlinear-tail split.

The pipeline is a list of blocks in canonical order. The statistical engine
consumes the LTI prefix (concatenating impulse responses); the transient engine
runs the nonlinear tail sample-by-sample. The boundary is the first block with
``is_tail=True`` (the DFE).

Threading rule: the live ``Pipeline`` lives on the controller thread. The
transient worker receives an *immutable snapshot* of the tail parameters plus a
``SimContext`` — never the mutable block objects. "Same pipeline" means same
*configuration*, not shared mutable state. :meth:`Pipeline.snapshot_params`
produces that copy; :meth:`Pipeline.apply_params` routes incoming updates and
reports which :class:`Kind`s changed so the controller knows what to recompute.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from numpy.typing import NDArray

from .block import Block, BlockState
from .context import SimContext
from .schema import Kind

# Canonical block order. Analysis is engine-side, not a pipeline block.
CANONICAL_ORDER: list[str] = [
    "source",
    "txffe",
    "txjitter",
    "channel",
    "noise",
    "ctle",
    "rxffe",
    "rxjitter",
    "dfe",
    "cdr_slicer",
]


@dataclass
class Pipeline:
    """An ordered list of blocks bound to a :class:`SimContext`."""

    blocks: list[Block] = field(default_factory=list)
    ctx: Optional[SimContext] = None

    # -- lookup ---------------------------------------------------------------
    def by_name(self, name: str) -> Block:
        for b in self.blocks:
            if b.name == name:
                return b
        raise KeyError(f"no block named {name!r} in pipeline")

    def names(self) -> list[str]:
        return [b.name for b in self.blocks]

    # -- LTI / nonlinear split ------------------------------------------------
    def _tail_index(self) -> int:
        """Index of the first nonlinear-tail block, or len(blocks) if none."""
        for i, b in enumerate(self.blocks):
            if getattr(b, "is_tail", False):
                return i
        return len(self.blocks)

    def lti_prefix(self) -> list[Block]:
        """Blocks up to (not including) the first nonlinear-tail block."""
        return self.blocks[: self._tail_index()]

    def nonlinear_tail(self) -> list[Block]:
        """The nonlinear tail (DFE onward)."""
        return self.blocks[self._tail_index() :]

    def collect_impulses(self, ctx: Optional[SimContext] = None) -> list[NDArray]:
        """Impulse responses of the LTI prefix, skipping stochastic (None) blocks.

        The statistical engine concatenates these; stochastic blocks (source,
        jitter, noise) contribute None and are modeled as PDFs instead.
        """
        ctx = ctx or self.ctx
        out: list[NDArray] = []
        for b in self.lti_prefix():
            h = b.impulse_response(ctx)
            if h is not None:
                out.append(h)
        return out

    def init_states(self, ctx: Optional[SimContext] = None) -> dict[str, BlockState]:
        ctx = ctx or self.ctx
        return {b.name: b.init_state(ctx) for b in self.blocks}

    # -- config / cross-thread copy -------------------------------------------
    def snapshot_params(self) -> dict[str, dict[str, Any]]:
        """An immutable-by-copy view of every block's parameters (block -> params)."""
        return {b.name: b.get_params() for b in self.blocks}

    def apply_params(self, updates: dict[str, dict[str, Any]]) -> set[Kind]:
        """Apply ``{block_name: {param: value}}`` updates; return changed Kinds.

        The returned set is the routing primitive: the controller inspects it to
        decide statistical recompute (LTI), transient push (NONLINEAR), and/or
        full rebuild (STRUCTURAL).
        """
        changed: set[Kind] = set()
        for block_name, param_updates in updates.items():
            block = self.by_name(block_name)
            for pname, value in param_updates.items():
                p = block._param(pname)
                block.set_params(**{pname: value})
                changed.add(p.kind)
                if p.also_statistical:
                    changed.add(Kind.LTI)
        return changed
