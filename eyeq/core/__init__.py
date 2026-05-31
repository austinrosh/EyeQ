"""Core contracts shared by every block and both engines.

Nothing in this package depends on the GUI, on SciPy/Numba, or on any specific
block implementation. These are the load-bearing interfaces:

* :mod:`eyeq.core.context`  — :class:`SimContext`, the immutable rate spine that
  derives every rate- and loss-aware size so no other module branches on rate.
* :mod:`eyeq.core.schema`   — :class:`Param` / :class:`Kind`, the declarative
  parameter schema that drives the GUI, validation, and update routing.
* :mod:`eyeq.core.block`    — the :class:`Block` protocol and :class:`BlockBase`.
* :mod:`eyeq.core.pipeline` — the ordered :class:`Pipeline` and LTI/tail split.
* :mod:`eyeq.core.registry` — a name -> block-class registry for config-driven
  construction.
"""

from .context import Modulation, ReachClass, SimContext, REACH_PRESETS
from .schema import Kind, Param, Scale
from .block import Block, BlockBase, BlockState, LTIBlock
from .pipeline import CANONICAL_ORDER, Pipeline
from . import registry

__all__ = [
    "Modulation",
    "ReachClass",
    "SimContext",
    "REACH_PRESETS",
    "Kind",
    "Param",
    "Scale",
    "Block",
    "BlockBase",
    "BlockState",
    "LTIBlock",
    "CANONICAL_ORDER",
    "Pipeline",
    "registry",
]
