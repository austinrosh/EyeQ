"""Built-in pipeline blocks.

Importing this package registers every built-in block in
:mod:`eyeq.core.registry`. Phase 0 ships each block as a passthrough stub with a
complete parameter schema (so the GUI auto-binder and config system are exercised
now); the DSP is filled in per the phased plan.
"""

from .source import Source
from .txffe import TXFFE
from .txjitter import TXJitter
from .channel import Channel
from .noise import Noise
from .ctle import CTLE
from .rxffe import RXFFE
from .dfe import DFE
from .rxjitter import RXJitter
from .cdr_slicer import CDRSlicer

__all__ = [
    "Source",
    "TXFFE",
    "TXJitter",
    "Channel",
    "Noise",
    "CTLE",
    "RXFFE",
    "DFE",
    "RXJitter",
    "CDRSlicer",
]
