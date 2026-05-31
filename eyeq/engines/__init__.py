"""Simulation engines.

* :mod:`eyeq.engines.statistical` — the Init-like engine: frequency cascade,
  single-bit response (SBR), and a peak-distortion-analysis statistical eye.
  Pure NumPy/SciPy, event-driven on LTI/STRUCTURAL changes. (Phase 1)
* :mod:`eyeq.engines.transient`  — the GetWave-like engine: a batched Monte
  Carlo loop through the nonlinear tail accumulating a decaying density eye.
  Sequential DFE/CDR inner loop in Numba. (Phase 2b/3)
* :mod:`eyeq.engines.worker`     — the threaded worker + double-buffered density
  snapshot + coalesced parameter mailbox that hosts the transient engine. (2b)
"""

from .statistical import (
    CascadeResult,
    SbrResult,
    StatEyeResult,
    StatisticalEngine,
)
from .transient import TransientEngine, TransientResult
from .worker import DensitySnapshot, ThreadWorker

__all__ = [
    "StatisticalEngine",
    "CascadeResult",
    "SbrResult",
    "StatEyeResult",
    "TransientEngine",
    "TransientResult",
    "ThreadWorker",
    "DensitySnapshot",
]
