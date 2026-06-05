"""CDR / slicer.

The slicer samples at a phase within the UI. ``cdr_mode`` selects how that phase
is set: ``static`` uses the ``sample_phase_ui`` slider as a fixed offset, while
``bang-bang`` (Alexander) and ``mueller-muller`` recover the phase from the data
with a PI loop filter (``kp``/``ki``) — ``sample_phase_ui`` is then the initial
condition the loop corrects from, and the phase tracks jitter. Part of the
nonlinear tail (runs in the transient kernel), but not the tail boundary.

**Jitter transfer.** A recovering CDR tracks low-frequency jitter (the clock
follows the data), so the eye-closing timing error is the input jitter shaped by
the high-pass error response :math:`1-H(f)`. We model a 1st-order jitter transfer
:math:`H(f)=1/(1+jf/f_c)` with corner ``loop_bw_mhz``, so
:math:`|1-H(f)| = (f/f_c)/\\sqrt{1+(f/f_c)^2}` — 0 at DC (fully tracked), 1 well
above :math:`f_c` (fully eye-closing), :math:`1/\\sqrt2` at the corner. Only the
tracking modes recover a clock, so ``static`` (and ``loop_bw_mhz=0``) returns 1 —
no tracking, every jitter component closes the eye. :func:`error_response` is what
the statistical engine and the jitter decomposition use to scale periodic jitter.
"""

from __future__ import annotations

import numpy as np

from ..core.block import BlockBase
from ..core.registry import register
from ..core.schema import Kind, Param

_CDR_MODES = ("static", "bang-bang", "mueller-muller")


@register("CDRSlicer")
class CDRSlicer(BlockBase):
    name = "cdr_slicer"
    is_lti = False
    PARAMS = [
        Param("sample_phase_ui", -0.5, 0.5, 0.0, unit="UI", kind=Kind.NONLINEAR),
        Param("cdr_mode", 0, 0, "static", kind=Kind.NONLINEAR, choices=_CDR_MODES),
        Param("kp", 0.0, 0.5, 0.05, kind=Kind.NONLINEAR),
        Param("ki", 0.0, 0.05, 0.001, kind=Kind.NONLINEAR),
        Param("loop_bw_mhz", 0.0, 200.0, 10.0, unit="MHz", kind=Kind.NONLINEAR,
              also_statistical=True),
    ]

    def cdr_mode_int(self) -> int:
        return _CDR_MODES.index(self.get("cdr_mode"))

    def tracking(self) -> bool:
        """True when a clock is being recovered (so low-freq jitter is tracked out)."""
        return self.get("cdr_mode") != "static" and self.get("loop_bw_mhz") > 0.0

    def error_response(self, f_hz: float) -> float:
        """CDR high-pass error response :math:`|1-H(f)|` in [0, 1] (1 = no tracking)."""
        if not self.tracking():
            return 1.0
        r = float(f_hz) / (self.get("loop_bw_mhz") * 1e6)
        return float(abs(r) / np.sqrt(1.0 + r * r))
