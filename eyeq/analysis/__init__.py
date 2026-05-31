"""Analysis & metrics (filled in across Phases 2-5).

* :mod:`eyeq.analysis.eye`      — eye-density accumulation (running 2-D histogram
  with exponential decay) and amplitude histograms / eye margins.
* :mod:`eyeq.analysis.ber`      — BER by error counting and BER-from-statistical-
  eye SER contours / bathtub curves.
* :mod:`eyeq.analysis.snr`      — MSE-SNR and COM at the decision point.
* :mod:`eyeq.analysis.jitter`   — jitter decomposition (Phase 5).
* :mod:`eyeq.analysis.optimize` — closed-form MMSE auto-EQ for TX/RX FFE + DFE
  (Shakiba et al., Part II, Eqs. 2-8). (Phase 3)
"""
