"""Golden / cross-validation tests run headless from ``examples`` (Phases 1-5).

Targets:
1. Closed-form ISI for a known channel vs the analytic SBR.
2. serdespy cross-check on the same .s4p + block settings.
3. Frequency-domain sanity: each reach preset hits its loss-at-Nyquist; smooth
   analytical curves track XSR/VSR profiles. MR/LR notch divergence is a
   documented model-fidelity limit, not a bug.
4. CTLE peak matches its controls (Eq. 22).
5. Statistical-vs-transient eye agreement, strictly for an LTI-only link — the
   single most valuable test for scaling/normalization bugs.
"""
