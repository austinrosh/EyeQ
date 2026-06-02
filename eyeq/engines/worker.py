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


# Eye liveliness. The accumulator is x[n] = decay*x[n-1] + batch, so it averages
# ~1/(1-decay) batches: small decay = a live, shimmering eye that snaps to changes;
# large decay = a smooth, persistent eye. Mapping the user-facing "avg factor" N to
# decay = 1 - 1/N makes N the effective number of batches averaged. Clamped so N=1
# still shows a little persistence and large N never fully freezes.
_DECAY_MIN, _DECAY_MAX = 0.5, 0.97


def decay_for(avg_factor: float) -> float:
    """Map an eye-averaging factor (>=1) to an exponential-decay coefficient."""
    n = max(1.0, float(avg_factor))
    return float(min(_DECAY_MAX, max(_DECAY_MIN, 1.0 - 1.0 / n)))


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
        self._clear = False
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
        # Size the grid to the worst-case excursion (sum of |cursors|) with generous
        # headroom: an equalizing RX FFE overshoots past that sum (up to ~1.4x on the
        # hardest reaches), and samples beyond the grid clamp into the edge bins — a
        # bright "clipped" band at the top/bottom of the eye. 1.5x keeps the eye clear.
        v_peak = max(1.5 * float(np.sum(np.abs(self._sbr.cursors))), 1e-6)
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

    def clear_accumulation(self) -> None:
        """Drop the decayed eye so the next batch shows the current pipeline at once.

        Used for NONLINEAR edits (DFE tap, CDR, adapt) that the worker already picks
        up from the shared pipeline but which would otherwise fade in slowly under
        the exponential decay — clearing makes the eye visibly jump to the change."""
        with self._lock:
            self._clear = True

    def set_decay(self, decay: float) -> None:
        """Set the eye-accumulation decay (eye liveliness) live; see :func:`decay_for`."""
        self.decay = float(decay)

    def latest(self) -> DensitySnapshot | None:
        with self._lock:
            return self._front

    # -- worker loop ----------------------------------------------------------
    def _drain(self) -> None:
        with self._lock:
            updates, self._pending = self._pending, {}
            ctx, self._pending_ctx = self._pending_ctx, None
            ext_dirty, self._dirty = self._dirty, False
            clear, self._clear = self._clear, False
        if ctx is not None:
            self.pipe.ctx = ctx
            self._lti_dirty = True
        if updates:
            kinds = self.pipe.apply_params(updates)
            if Kind.LTI in kinds or Kind.STRUCTURAL in kinds:
                self._lti_dirty = True
        if ext_dirty:
            self._lti_dirty = True
        if clear:  # NONLINEAR change: keep the SBR/grid, just reset the decayed eye
            self._accum = None

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

            # Eye height at the recovered sampling phase (item 2): measured on the
            # smooth accumulated image rather than the noisy single batch, at the
            # column the CDR actually samples (the eye's dashed marker).
            ctx = self.pipe.ctx
            main_cursor = self._sbr.main_cursor
            col = int(round(res.recovered_phase_ui * ctx.sps + ctx.sps // 2))
            eye_h_phase = self._stat._eye_height_at_col(image, self._v, main_cursor, ctx.levels, col)

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
                    "eye_height_at_phase_v": float(eye_h_phase),
                    "recovered_phase_ui": res.recovered_phase_ui,
                    "n_batches": self._n_batches,
                    "ui_per_batch": res.n_symbols,
                },
            )
            with self._lock:
                self._seq += 1
                self._front = snap
