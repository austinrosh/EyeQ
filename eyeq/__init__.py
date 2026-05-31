"""EyeQ — an interactive SerDes link-modeling tool.

The link is split into an LTI cascade and a nonlinear/time-varying tail
(mirroring the IBIS-AMI Init/GetWave split). Two engines operate on the same
configured pipeline:

* a *statistical* engine (fast, deterministic) that builds the frequency
  cascade, single-bit response (SBR), and a peak-distortion-analysis eye; and
* a *transient* engine (continuous, batched) that accumulates a density eye.

The engine is headless, scriptable, and testable; the GUI is a thin, swappable
client. See ``eyeq/core`` for the contracts every block and engine binds to.
"""

__version__ = "0.0.0"
