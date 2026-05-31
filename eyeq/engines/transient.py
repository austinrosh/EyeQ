"""Transient engine (Phase 2b/3 — not yet implemented).

Pushes a symbol stream through the nonlinear tail in batches, accumulating a
decaying 2-D density eye (running histogram with exponential decay = averaging
factor). The LTI prefix is applied outside the hot loop with
``scipy.signal.oaconvolve`` (recomputed only on LTI change); only the genuinely
sequential DFE/CDR/slicer tail goes through a single Numba
``njit(cache=True, fastmath=True, nogil=True)`` kernel, crossing the Python<->
Numba boundary once per batch (never per sample). State is flat preallocated
arrays reused across batches (zero per-batch allocation).

Throughput target: >= 1.5M UI/s sustained at 112/224/448G, including 448G with a
long DFE.
"""
