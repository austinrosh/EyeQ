"""MLSD (maximum-likelihood sequence detection) BER modeling.

EyeQ's decision-point BER (``analysis/ber.py``) integrates the amplitude tail at
a single sampling instant — valid for a slicer/DFE, but **not** for MLSD. An MLSD
(Viterbi) receiver picks the most-likely transmitted *sequence* over a trellis
built from the channel ISI, so its error rate is governed by the **minimum
Euclidean distance** between distinct transmittable sequences relative to the
noise, not by a single-sample margin.

We estimate the MLSD error rate with the standard **minimum-distance union
bound** (a matched-filter-bound-style high-SNR estimate):

    SER ≈ N_dmin · Q(d_min / (2·σ)),     d_min² = min over nonzero error events e of ‖e ⊛ h‖²

where ``h`` is the sampled channel pulse response (the SBR cursors, in volts),
``e`` is a difference of two valid symbol sequences (symbol-difference alphabet),
and ``σ`` is the front-end-referred noise std (volts). ``d_min`` is found by a
bounded depth-first search over error events with branch-and-bound pruning
(``‖closed outputs‖²`` is a valid lower bound on any extension, since later error
symbols cannot change already-emitted output samples).

**Model limits (label these):** the union bound is optimistic at low SNR (it
counts only the minimum-distance events and assumes a whitened-matched-filter
front end), and it inherits the same noise/jitter limits as the rest of the tool.
It is an estimate of the MLSD *gain*, not a measured sequence-error rate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import erfc

# Per-modulation trellis-memory caps. State count grows as alphabet^L, and the
# error-event search branches with the symbol-difference alphabet, so PAM-4 is
# capped tighter. These bound compute; the DFS also has a hard node cap.
L_CAP = {2: 8, 4: 5}
_NODE_CAP = 200_000


@dataclass(frozen=True)
class SeqResult:
    ber: float
    ser: float
    d2_min: float        # minimum squared distance [V^2]
    n_events: int        # multiplicity of minimum-distance error events (canonical count)
    truncated: bool      # the d_min search hit the node cap (label as approximate)


def l_cap(n_levels: int) -> int:
    return L_CAP.get(int(n_levels), 5)


def error_symbols(levels: NDArray) -> NDArray[np.float64]:
    """Distinct positive symbol-difference magnitudes (the error alphabet).

    NRZ levels [-1, 1] -> [2]; PAM-4 [-1, -1/3, 1/3, 1] -> [2/3, 4/3, 2].
    Positive-only: an error event and its negation have equal distance, so the
    first error symbol is taken positive WLOG.
    """
    levels = np.asarray(levels, float)
    diffs = np.abs(levels[:, None] - levels[None, :])
    vals = np.unique(np.round(diffs[diffs > 0], 12))
    return vals


def min_distance(h, err_syms, *, max_len: int, node_cap: int = _NODE_CAP):
    """Minimum squared distance of a channel-filtered nonzero error event.

    Returns ``(d2_min, n_events, truncated)``. ``h`` is the channel pulse
    (``h[0]`` the main cursor); ``err_syms`` the positive error alphabet.
    """
    h = np.ascontiguousarray(h, dtype=float)
    lh = h.size
    first = np.asarray(err_syms, float)                       # event start: positive only
    rest = np.concatenate(([0.0], first, -first))             # then 0 or ±
    out_len = max_len + lh - 1

    best = math.inf
    n_events = 0
    nodes = 0
    truncated = False
    tol = 1e-12

    def dfs(out: np.ndarray, t: int):
        nonlocal best, n_events, nodes, truncated
        nodes += 1
        if nodes > node_cap:
            truncated = True
            return
        if t == max_len:
            return
        closed = float(out[:t] @ out[:t]) if t else 0.0       # lower bound on any extension
        if t and closed >= best - tol:
            return
        for s in (first if t == 0 else rest):
            nxt = out.copy()
            if s != 0.0:
                nxt[t:t + lh] += s * h
                d2 = float(nxt @ nxt)                          # event ending at this nonzero symbol
                if d2 < best - tol:
                    best, n_events = d2, 1
                elif abs(d2 - best) <= tol:
                    n_events += 1
            dfs(nxt, t + 1)                                    # extend (zeros allowed mid-event)

    dfs(np.zeros(out_len), 0)
    if not math.isfinite(best):
        best = 0.0
    return best, n_events, truncated


def _q(x: float) -> float:
    """Gaussian Q-function Q(x) = 0.5*erfc(x/sqrt(2))."""
    return 0.5 * float(erfc(x / math.sqrt(2.0)))


def union_bound_ber(d2_min: float, n_events: int, sigma: float, bits_per_symbol: int) -> tuple[float, float]:
    """(ser, ber) from the minimum-distance union bound. σ in volts (same as d_min)."""
    if d2_min <= 0.0:
        return 0.5, 0.5 / max(bits_per_symbol, 1)             # no usable signal -> random
    if sigma <= 0.0:
        return 0.0, 0.0                                       # noiseless -> error-free
    ser = min(0.5, max(n_events, 1) * _q(math.sqrt(d2_min) / (2.0 * sigma)))
    return ser, ser / max(bits_per_symbol, 1)


def sequence_ber(h, levels, sigma, *, L, node_cap: int = _NODE_CAP) -> SeqResult:
    """End-to-end MLSD BER estimate for a sampled channel pulse ``h`` (volts).

    ``h`` is trimmed to the ``L+1`` most significant taps before the search.
    """
    h = np.asarray(h, float)
    if h.size > L + 1:                                        # keep the L+1 strongest taps
        keep = np.argsort(np.abs(h))[::-1][: L + 1]
        h = h[np.sort(keep)]
    syms = error_symbols(levels)
    d2, n_ev, trunc = min_distance(h, syms, max_len=L + 6, node_cap=node_cap)
    bits = int(np.log2(len(levels))) or 1
    ser, ber = union_bound_ber(d2, n_ev, float(sigma), bits)
    return SeqResult(ber=ber, ser=ser, d2_min=d2, n_events=n_ev, truncated=trunc)
