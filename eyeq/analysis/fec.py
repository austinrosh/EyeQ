"""Forward Error Correction (FEC) modeling — a post-decode error-rate layer.

EyeQ's engines produce a *pre-FEC* error rate at the decision point. Real links
run FEC, so a raw BER of ~1e-4 (which looks like a failing link) is a passing
link after KP4. This module maps a pre-FEC bit error rate to an estimated
post-FEC BER for the Reed-Solomon codes used in current high-speed links, and
derives the code's pre-FEC BER *threshold* (the input BER below which the code
delivers a target post-FEC BER) and a coding-gain figure.

**Model and its limits.** We use the standard hard-decision, bounded-distance RS
decoder with **i.i.d. (random) symbol errors** — the model EyeQ's noise process
actually justifies (Gaussian noise + bounded ISI, no burst mechanism). For an
RS(n, k) code over GF(2^m) correcting ``t = (n-k)//2`` symbols, an m-bit symbol
errs with probability ``p_s = 1-(1-BER)^m`` and the post-decode output symbol
error rate is

    SER_out = (1/n) * sum_{i=t+1}^{n} i * C(n,i) * p_s^i * (1-p_s)^(n-i)

(a bounded-distance decoder leaves all i symbols of an uncorrectable codeword in
error), giving ``BER_out = SER_out * (2^(m-1)/(2^m-1))`` (the average fraction of
bits wrong in a wrong symbol). This is a **model-based estimate**: it does not
capture real burst/correlated errors, which the noise model does not yet
generate. An optional *bursty* knob (burst length vs interleave depth) is a
deliberately coarse approximation that shrinks the effective ``t``; it is labeled
as such and is not derived from a real burst-generating channel.

Verified anchors (IEEE 802.3): KP4 = RS(544,514)/GF(2^10), t=15, pre-FEC
threshold ~2.4e-4 for ~1e-15 post-FEC, ~6.9 dB coding gain; KR4 = RS(528,514),
t=7.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import erfcinv, gammaln

_BER_FLOOR = 1e-300  # keep log10 finite for vanishing post-FEC rates


# --------------------------------------------------------------------------- #
# scheme registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FecScheme:
    """A named FEC code. ``kind='rs'`` is a Reed-Solomon RS(n,k) over GF(2^m)."""

    key: str
    label: str
    kind: str          # "none" | "rs"  (room for "concatenated" later)
    n: int             # codeword length [symbols]
    k: int             # message length [symbols]
    m: int             # symbol size [bits] (GF(2^m))
    pairing: tuple     # modulations this is the *standard* pairing for (advisory only)
    note: str = ""

    @property
    def t(self) -> int:
        """Correctable symbols per codeword (bounded-distance: (n-k)//2)."""
        return (self.n - self.k) // 2

    @property
    def overhead_pct(self) -> float:
        return 100.0 * (self.n - self.k) / self.k if self.k else 0.0


SCHEMES: dict[str, FecScheme] = {
    "none": FecScheme("none", "None (raw / pre-FEC)", "none", 0, 0, 0, (),
                      "No coding — the raw pre-FEC baseline for comparison."),
    "kp4": FecScheme("kp4", "KP4 — RS(544,514)", "rs", 544, 514, 10, ("PAM4",),
                     "IEEE 802.3 KP4 RS(544,514)/GF(2^10), t=15 — the 100G/lane PAM4 workhorse."),
    "kr4": FecScheme("kr4", "KR4 — RS(528,514)", "rs", 528, 514, 10, ("NRZ",),
                     "IEEE 802.3 KR4/CR4 RS(528,514)/GF(2^10), t=7 — 25/50G NRZ backplane."),
}


def custom_scheme(n: int, k: int, m: int) -> FecScheme:
    return FecScheme("custom", f"Custom RS({int(n)},{int(k)})", "rs",
                     int(n), int(k), int(m), (), "User-defined Reed-Solomon code.")


def scheme_from_cfg(cfg: dict) -> FecScheme:
    """Resolve the scheme named (or 'custom'-built) by a FEC config dict."""
    key = cfg.get("scheme", "kp4")
    if key == "custom":
        return custom_scheme(cfg.get("custom_n", 544), cfg.get("custom_k", 514),
                             cfg.get("custom_m", 10))
    return SCHEMES.get(key, SCHEMES["none"])


# --------------------------------------------------------------------------- #
# RS post-decode error-rate math
# --------------------------------------------------------------------------- #
def sym_err_prob(ber: float, m: int) -> float:
    """Probability an m-bit FEC symbol is wrong, for i.i.d. bit errors at ``ber``."""
    ber = min(max(float(ber), 0.0), 1.0)
    if ber >= 1.0:
        return 1.0
    return float(-np.expm1(m * np.log1p(-ber)))


def _rs_output_symbol_error(p_s: float, n: int, t: int) -> float:
    """SER_out = (1/n) sum_{i=t+1}^n i*C(n,i)*p_s^i*(1-p_s)^(n-i), log-domain.

    ``t`` is clamped to ``[0, n]`` so a degenerate/transient code (e.g. k>n while
    custom parameters are mid-edit) corrects nothing rather than indexing
    ``log`` at non-positive symbol counts.
    """
    t = max(0, min(int(t), n))
    if p_s <= 0.0 or n <= 0 or t >= n:
        return 0.0
    if p_s >= 1.0:
        return 1.0
    i = np.arange(t + 1, n + 1)
    log_terms = (np.log(i) + gammaln(n + 1) - gammaln(i + 1) - gammaln(n - i + 1)
                 + i * np.log(p_s) + (n - i) * np.log1p(-p_s))
    peak = log_terms.max()
    total = np.exp(peak) * np.sum(np.exp(log_terms - peak))
    return float(min(total / n, 1.0))


def _rs_one(ber_in: float, n: int, k: int, m: int, t: int | None = None) -> float:
    t = (n - k) // 2 if t is None else t
    p_s = sym_err_prob(ber_in, m)
    ser_out = _rs_output_symbol_error(p_s, n, t)
    bit_frac = 2 ** (m - 1) / (2 ** m - 1)  # avg fraction of bits wrong in a wrong symbol (~0.5)
    return min(float(ber_in), ser_out * bit_frac)


def rs_output_ber(ber_in, n: int, k: int, m: int, t: int | None = None):
    """Post-FEC BER for RS(n,k)/GF(2^m); accepts a scalar or an ndarray of BERs."""
    if np.ndim(ber_in) == 0:
        return _rs_one(float(ber_in), n, k, m, t)
    return np.array([_rs_one(float(b), n, k, m, t) for b in np.asarray(ber_in, float)])


def post_ber(ber_in, scheme: FecScheme, *, error_model: str = "random",
             burst_len_bits: int = 1, interleave_depth: int = 1):
    """Map pre-FEC BER -> post-FEC BER for a scheme. ``none`` is a passthrough.

    ``error_model='bursty'`` is a coarse approximation: a burst of
    ``burst_len_bits`` spread over ``interleave_depth`` codewords corrupts
    ``g = ceil(L/(m*D))`` symbols at once, so the code's usable correction drops
    to ``t//g``. ``g=1`` (the random default) recovers the exact RS model.
    """
    if scheme.kind == "none":
        return ber_in
    t = scheme.t
    if error_model == "bursty":
        g = max(1, math.ceil(burst_len_bits / (scheme.m * max(1, interleave_depth))))
        t = max(0, t // g)
    return rs_output_ber(ber_in, scheme.n, scheme.k, scheme.m, t)


def pre_fec_threshold(scheme: FecScheme, target_post_ber: float, **kw) -> float:
    """Input BER at which the scheme delivers ``target_post_ber`` (bisection)."""
    if scheme.kind == "none":
        return float(target_post_ber)
    lo, hi = -12.0, math.log10(0.5)  # log10(BER_in) bracket; post_ber is monotone
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if post_ber(10.0 ** mid, scheme, **kw) < target_post_ber:
            lo = mid
        else:
            hi = mid
    return 10.0 ** (0.5 * (lo + hi))


def _qinv(p: float) -> float:
    """Inverse Gaussian Q-function: Q(qinv(p)) = p."""
    return float(np.sqrt(2.0) * erfcinv(2.0 * p))


def coding_gain_db(pre_threshold_ber: float, target_post_ber: float) -> float:
    """Gaussian-approx SNR coding gain: 20*log10(Qinv(target)/Qinv(threshold)).

    Relative to the no-FEC case at the target BER: an uncoded link must reach the
    target BER directly (SNR ~ Qinv(target)), while the coded link only needs its
    raw BER below the (larger) pre-FEC threshold (SNR ~ Qinv(threshold)), so the
    saved SNR is the ratio of the two. Embeds a Gaussian BER<->SNR assumption (so
    it is an approximation), but reproduces the textbook ~6.9 dB for KP4 — a
    useful sanity anchor.
    """
    if not (0.0 < pre_threshold_ber < 0.5) or not (0.0 < target_post_ber < 0.5):
        return 0.0
    return float(20.0 * np.log10(_qinv(target_post_ber) / _qinv(pre_threshold_ber)))


# --------------------------------------------------------------------------- #
# top-level assessment (consumed by the GUI controller)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FecResult:
    enabled: bool
    scheme_key: str
    scheme_label: str
    n: int
    k: int
    m: int
    t: int
    target_post_ber: float
    pre_threshold_ber: float       # raw BER must stay below this (NaN if disabled)
    coding_gain_db: float
    pre_ber: float                 # operating-point pre-FEC BER
    post_ber: float                # operating-point post-FEC BER (== pre when disabled)
    error_model: str
    applicable: bool               # is this the standard pairing for the current modulation?
    pairing_note: str
    # bathtub curves, log10(BER), on the existing phase/level axes:
    pre_h_ber: NDArray             # pre-FEC BER vs phase
    pre_v_ber: NDArray             # pre-FEC BER vs level
    post_h_ber: NDArray            # post-FEC BER vs phase
    post_v_ber: NDArray            # post-FEC BER vs level


def assess_fec(ber, ctx, cfg: dict) -> FecResult:
    """Build a :class:`FecResult` from a pre-FEC :class:`~eyeq.analysis.ber.BerResult`.

    ``cfg`` is the FEC settings dict (see ``io.config.default_fec``). The bathtub
    arrays in ``ber`` are ``log10(SER)``; pre-FEC BER = SER / bits_per_symbol.
    """
    scheme = scheme_from_cfg(cfg)
    enabled = bool(cfg.get("enabled", False)) and scheme.kind != "none"
    target = float(cfg.get("target_post_ber", 1e-15))
    em = cfg.get("error_model", "random")
    kw = dict(error_model=em,
              burst_len_bits=int(cfg.get("burst_len_bits", 1)),
              interleave_depth=int(cfg.get("interleave_depth", 1)))

    bits = ctx.mod.bits_per_symbol
    mod_name = ctx.mod.name
    applicable = (not scheme.pairing) or (mod_name in scheme.pairing)
    pairing_note = "" if (applicable or scheme.kind == "none") else (
        f"{scheme.label.split(' ')[0]} is the standard pairing for "
        f"{'/'.join(scheme.pairing)}; selectable here anyway.")

    log_bits = math.log10(bits)
    pre_h_ber = ber.h_bathtub - log_bits        # log10(SER) -> log10(BER)
    pre_v_ber = ber.v_bathtub - log_bits

    if enabled:
        thr = pre_fec_threshold(scheme, target, **kw)
        gain = coding_gain_db(thr, target)
        op_post = float(post_ber(ber.ber, scheme, **kw))
        post_h = np.log10(np.maximum(post_ber(10.0 ** ber.h_bathtub / bits, scheme, **kw), _BER_FLOOR))
        post_v = np.log10(np.maximum(post_ber(10.0 ** ber.v_bathtub / bits, scheme, **kw), _BER_FLOOR))
    else:
        thr, gain, op_post = float("nan"), 0.0, ber.ber
        post_h, post_v = pre_h_ber.copy(), pre_v_ber.copy()

    return FecResult(
        enabled=enabled, scheme_key=scheme.key, scheme_label=scheme.label,
        n=scheme.n, k=scheme.k, m=scheme.m, t=scheme.t,
        target_post_ber=target, pre_threshold_ber=thr, coding_gain_db=gain,
        pre_ber=ber.ber, post_ber=op_post, error_model=em,
        applicable=applicable, pairing_note=pairing_note,
        pre_h_ber=pre_h_ber, pre_v_ber=pre_v_ber, post_h_ber=post_h, post_v_ber=post_v,
    )
