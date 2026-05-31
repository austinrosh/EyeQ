"""Eye-density accumulation & amplitude histograms (Phase 2 — placeholder).

A running 2-D histogram [phase x voltage] updated each batch with exponential
decay (the "averaging factor"), plus marginal amplitude histograms for eye
margins. Written for the transient engine; cross-checked against the statistical
eye for LTI-only links.
"""
