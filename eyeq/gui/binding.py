"""Auto-generate controls from each block's parameter schema.

Adding a block automatically surfaces its controls — there is no per-block GUI
code. A numeric :class:`~eyeq.core.schema.Param` becomes a slider (+ value
label); a choice param becomes a combo box. Each control emits
``changed(block_name, param_name, value)``; the controller routes by ``Kind``.
"""

from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtWidgets

from ..core.schema import Param, Scale

_STEPS = 1000  # slider integer resolution


class ParamControl(QtWidgets.QWidget):
    changed = QtCore.Signal(str, str, object)

    def __init__(self, block_name: str, p: Param):
        super().__init__()
        self.block_name = block_name
        self.p = p
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(2, 1, 2, 1)
        label = QtWidgets.QLabel(p.name)
        label.setMinimumWidth(95)
        lay.addWidget(label)

        if p.is_choice:
            self.combo = QtWidgets.QComboBox()
            self.combo.addItems([str(c) for c in p.choices])
            self.combo.setCurrentText(str(p.default))
            self.combo.currentTextChanged.connect(
                lambda t: self.changed.emit(self.block_name, p.name, t)
            )
            lay.addWidget(self.combo, 1)
            self.value_label = None
        else:
            self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.slider.setRange(0, _STEPS)
            self.slider.setValue(self._to_slider(p.default))
            self.slider.valueChanged.connect(self._on_slider)
            lay.addWidget(self.slider, 1)
            self.value_label = QtWidgets.QLabel()
            self.value_label.setMinimumWidth(78)
            self.value_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            lay.addWidget(self.value_label)
            self._set_label(p.default)

    # -- slider <-> value mapping --------------------------------------------
    def _to_slider(self, v: float) -> int:
        p = self.p
        if p.scale is Scale.LOG:
            lo, hi = np.log10(max(p.min, 1e-12)), np.log10(p.max)
            frac = (np.log10(max(float(v), 1e-12)) - lo) / (hi - lo)
        else:
            frac = (float(v) - p.min) / (p.max - p.min)
        return int(round(np.clip(frac, 0, 1) * _STEPS))

    def _from_slider(self, s: int) -> float:
        p = self.p
        frac = s / _STEPS
        if p.scale is Scale.LOG:
            lo, hi = np.log10(max(p.min, 1e-12)), np.log10(p.max)
            return float(10 ** (lo + frac * (hi - lo)))
        return float(p.min + frac * (p.max - p.min))

    def _on_slider(self, s: int) -> None:
        v = self._from_slider(s)
        self._set_label(v)
        self.changed.emit(self.block_name, self.p.name, v)

    def _set_label(self, v: float) -> None:
        if self.value_label is not None:
            unit = f" {self.p.unit}" if self.p.unit else ""
            self.value_label.setText(f"{v:.3g}{unit}")

    def set_value(self, v) -> None:
        """Update the control programmatically (auto-EQ / config load); no signal."""
        if self.p.is_choice:
            self.combo.blockSignals(True)
            self.combo.setCurrentText(str(v))
            self.combo.blockSignals(False)
        else:
            self.slider.blockSignals(True)
            self.slider.setValue(self._to_slider(v))
            self.slider.blockSignals(False)
            self._set_label(float(v))


class BlockPanel(QtWidgets.QGroupBox):
    def __init__(self, block, on_change):
        super().__init__(block.name)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(1)
        self.controls: dict[str, ParamControl] = {}
        for p in block.params:
            if p.hidden:  # surfaced elsewhere (e.g. the toolbar EQ-bypass toggles)
                continue
            c = ParamControl(block.name, p)
            c.changed.connect(on_change)
            lay.addWidget(c)
            self.controls[p.name] = c

    def sync(self, block) -> None:
        for name, ctrl in self.controls.items():
            ctrl.set_value(block.get(name))


def build_param_panel(pipe, on_change):
    """Return (scroll_area, {block_name: BlockPanel}) auto-built from the pipeline."""
    container = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(container)
    lay.setSpacing(4)
    panels: dict[str, BlockPanel] = {}
    for b in pipe.blocks:
        panel = BlockPanel(b, on_change)
        lay.addWidget(panel)
        panels[b.name] = panel
    lay.addStretch(1)
    scroll = QtWidgets.QScrollArea()
    scroll.setWidget(container)
    scroll.setWidgetResizable(True)
    scroll.setMinimumWidth(330)
    return scroll, panels
