"""Transient worker + double-buffered snapshot (Phase 2b — not yet implemented).

Hosts the transient engine off the GUI thread. A thread (running the
``nogil=True`` Numba kernel, which releases the GIL) is the default, behind a
``TransientWorker`` protocol so a ``multiprocessing`` implementation is a later
drop-in.

Planned interface::

    @dataclass
    class DensitySnapshot: image; levels; seq; stats

    class TransientWorker(Protocol):
        def start(self) -> None: ...
        def stop(self) -> None: ...
        def push_params(self, updates: dict) -> None: ...    # coalesced, thread-safe
        def push_context(self, ctx: SimContext) -> None: ... # STRUCTURAL -> rebuild buffers
        def latest(self) -> DensitySnapshot: ...             # lock-free read of front buffer

Concurrency primitives: parameter updates land in a lock-guarded pending-dict
drained at batch boundaries (coalesced); the density eye is published via a
double buffer + atomic index swap and a monotonic ``seq`` counter. A warm-up
compile runs at startup so the first live frame is hot.
"""
