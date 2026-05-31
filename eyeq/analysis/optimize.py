"""Closed-form MMSE auto-EQ (Phase 3 — placeholder).

One-shot "solve optimal taps" from the SBR cursors (Shakiba et al., Part II):

* TX FFE (zero-forcing/MMSE): pulse-response matrix H from the cursor row;
  ``v_opt = g* . H^T (H H^T)^-1``, normalized by sum(|v|) to preserve swing
  (Eqs. 2-3). Targets precursor removal.
* RX FFE (MMSE with noise autocorrelation R_NN):
  ``w_opt,i = y*_i . X^T (sigma_a^2 X X^T + R_NN)^-1``, searching the main-tap
  position i that minimizes MMSE (Eqs. 6-7); optional floating-tap groups.
* RX FFE/DFE co-optimization: target pulse with the first post-cursor (alpha)
  handled by the DFE.

Surfaced in the GUI as an "Auto-EQ" button; LMS/sign-LMS remain the online
variant in the transient loop.
"""
