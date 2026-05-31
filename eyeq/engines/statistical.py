"""Statistical engine (Phase 1 — not yet implemented).

Event-driven, deterministic, sub-millisecond. Given a configured pipeline it
produces, on any LTI/STRUCTURAL change:

1. **Frequency cascade** — multiply each LTI block's ``transfer(ctx)`` on the
   shared ``ctx.freq_grid()`` to get the three plotted traces:
   Channel, TX+Channel, TX+Channel+RX.
2. **SBR** — total LTI transfer x one-UI input spectrum -> irfft -> windowed
   single-bit/pulse response of length ``ctx.sbr_len_samples()``.
3. **Cursors** — sample the SBR at stride ``sps`` aligned to the main cursor,
   sliced to ``ctx.cursor_span()``.
4. **Statistical eye (PDA)** — per-cursor ISI PDFs convolved (FFT-domain on a
   fixed voltage grid) with TX-noise, RX-noise, crosstalk, and jitter PDFs;
   evaluated at ~32 phases across one UI -> 3-D PDF eye -> SER contours ->
   bathtub curves (Shakiba et al., Part II, Sec. IV; Eqs. 9-11).

Planned interface::

    @dataclass(frozen=True)
    class CascadeResult: f; H_channel; H_tx_chan; H_tx_chan_rx
    @dataclass(frozen=True)
    class SbrResult: sbr; t; cursors; main_idx
    @dataclass(frozen=True)
    class StatEyeResult: v_axis; t_axis; pdf; ser_contours; bathtub_h; bathtub_v

    class StatisticalEngine:
        def compute(self, pipe: Pipeline) -> tuple[CascadeResult, SbrResult, StatEyeResult]: ...
"""
