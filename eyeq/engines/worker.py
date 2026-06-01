"""Transient worker — runs the transient engine off the GUI thread.

A ``ThreadWorker`` runs batches continuously on a background thread, accumulating
the density eye with exponential decay and publishing a double-buffered snapshot
the GUI can read lock-free-ish via :meth:`latest`. Parameter updates land in a
coalesced pending-dict and are applied at batch boundaries (a fast slider drag
collapses to one apply); an LTI/STRUCTURAL change recomputes the SBR and clears
the accumulation so the eye refreshes.

Phase 2b uses a thread with the vectorized LTI-only engine. The interface is the
seam where a process-backed worker (for the Numba tail) drops in later; the
density snapshot and coalescing contract stay identical.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..core.pipeline import Pipeline
from ..core.schema import Kind
from .statistical import StatisticalEngine
from .transient import TransientEngine


@dataclass(frozen=True)
class DensitySnapshot:
    t_ui: NDArray
    v: NDArray
    image: NDArray                  # [phase, voltage] density, per-phase normalized
    levels: tuple[float, float]     # fixed (min, max) voltage for a stable LUT
    seq: int                        # monotonic; GUI compares to detect new frames
    stats: dict[str, Any] = field(default_factory=dict)


class ThreadWorker:
    def __init__(
        self,
        pipe: Pipeline,
        *,
        engine: TransientEngine | None = None,
        batch_symbols: int = 20_000,
        decay: float = 0.9,
    ):
        self.pipe = pipe
        self.engine = engine or TransientEngine()
        self.batch_symbols = batch_symbols
        self.decay = decay

        self._lock = threading.Lock()
        self._pending: dict[str, dict[str, Any]] = {}
        self._pending_ctx = None
        self._dirty = False
        self._running = False
        self._thread: threading.Thread | None = None
        self._front: DensitySnapshot | None = None
        self._seq = 0
        self._rng = np.random.default_rng(pipe.ctx.rng_seed)
        self._stat = StatisticalEngine()

        self._sbr = None
        self._v: NDArray | None = None
        self._lti_dirty = False
        self._accum: NDArray | None = None
        self._n_batches = 0
        self._refresh()  # initial SBR + voltage grid

    def _refresh(self) -> None:
        """Recompute the SBR and a voltage grid sized to it; reset accumulation."""
        self._sbr = self._stat.sbr(self.pipe)
        v_peak = max(1.15 * float(np.sum(np.abs(self._sbr.cursors))), 1e-6)
        self._v = np.linspace(-v_peak, v_peak, self.engine.v_bins)
        self._accum = None

    # -- lifecycle ------------------------------------------------------------
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="eyeq-transient", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    # -- external API (thread-safe) ------------------------------------------
    def push_params(self, updates: dict[str, dict[str, Any]]) -> None:
        with self._lock:
            for block, params in updates.items():
                self._pending.setdefault(block, {}).update(params)

    def push_context(self, ctx) -> None:
        with self._lock:
            self._pending_ctx = ctx

    def mark_dirty(self) -> None:
        """Signal an LTI change made directly on the shared pipeline (GUI path)."""
        with self._lock:
            self._dirty = True

    def latest(self) -> DensitySnapshot | None:
        with self._lock:
            return self._front

    # -- worker loop ----------------------------------------------------------
    def _drain(self) -> None:
        with self._lock:
            updates, self._pending = self._pending, {}
            ctx, self._pending_ctx = self._pending_ctx, None
            ext_dirty, self._dirty = self._dirty, False
        if ctx is not None:
            self.pipe.ctx = ctx
            self._lti_dirty = True
        if updates:
            kinds = self.pipe.apply_params(updates)
            if Kind.LTI in kinds or Kind.STRUCTURAL in kinds:
                self._lti_dirty = True
        if ext_dirty:
            self._lti_dirty = True

    def _loop(self) -> None:
        while self._running:
            self._drain()
            if self._lti_dirty:
                self._refresh()  # new SBR + voltage grid; clears accumulation
                self._lti_dirty = False
            res = self.engine.run_batch(
                self.pipe, n_symbols=self.batch_symbols, sbr=self._sbr, rng=self._rng, v=self._v
            )
            if self._accum is None:
                self._accum = res.density.copy()
            else:
                self._accum = self.decay * self._accum + res.density
            self._n_batches += 1

            image = self._accum / np.maximum(self._accum.sum(1, keepdims=True), 1e-30)
            snap = DensitySnapshot(
                t_ui=res.t_ui,
                v=self._v,
                image=image,
                levels=(float(self._v[0]), float(self._v[-1])),
                seq=self._seq + 1,
                stats={
                    "mse_snr_db": res.mse_snr_db,
                    "ser": res.ser,
                    "eye_height_v": res.eye_height_v,
                    "recovered_phase_ui": res.recovered_phase_ui,
                    "n_batches": self._n_batches,
                    "ui_per_batch": res.n_symbols,
                },
            )
            with self._lock:
                self._seq += 1
                self._front = snap
