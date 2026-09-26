"""Input panels: system parameters, applied force, and simulation controls."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from ..core import ChainSystem, ForceController, ForceKind, ModalResult
from ..core.model import DEFAULT_DAMPING, DEFAULT_MASS, DEFAULT_STIFFNESS
from .style import MASS_COLORS, MAX_DOF


def spin(lo: float, hi: float, value: float, decimals: int, suffix: str = "") -> QtWidgets.QDoubleSpinBox:
    box = QtWidgets.QDoubleSpinBox()
    box.setRange(lo, hi)
    box.setDecimals(decimals)
    box.setValue(value)
    box.setStepType(QtWidgets.QAbstractSpinBox.StepType.AdaptiveDecimalStepType)
    box.setKeyboardTracking(False)  # typed values apply on Enter / focus-out
    box.setAccelerated(True)
    if suffix:
        box.setSuffix(suffix)
    return box


def fill_mode_combo(combo: QtWidgets.QComboBox, result: ModalResult, placeholder: str) -> None:
    """List the modes in a one-shot "pick a mode" combo; item data is fn in Hz."""
    combo.clear()
    combo.addItem(placeholder)
    for mode in result.modes:
        if mode.fn_hz > 0:  # skip rigid-body modes
            combo.addItem(f"Mode {mode.index} ({mode.fn_hz:.3g} Hz)", mode.fn_hz)


class ParameterPanel(QtWidgets.QGroupBox):
    """Mass, spring and damper values for each element of the chain."""

    changed = QtCore.Signal(object)  # ChainSystem

    def __init__(self, system: ChainSystem, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("System parameters", parent)
        layout = QtWidgets.QVBoxLayout(self)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Number of masses:"))
        self.dof = QtWidgets.QSpinBox()
        self.dof.setRange(1, MAX_DOF)
        self.dof.setValue(system.n)
        self.dof.valueChanged.connect(self._on_dof)
        top.addWidget(self.dof)
        top.addStretch(1)
        reset = QtWidgets.QPushButton("Defaults")
        reset.setToolTip("Reset every mass, spring and damper to its default value")
        reset.clicked.connect(self._reset_defaults)
        top.addWidget(reset)
        uniform = QtWidgets.QPushButton("Copy row 1 → all")
        uniform.setToolTip("Make the chain uniform using the first row's values")
        uniform.clicked.connect(self._copy_first_row)
        top.addWidget(uniform)
        layout.addLayout(top)

        self.grid = QtWidgets.QGridLayout()
        for col, text in enumerate(["", "m [kg]", "k [N/m]", "c [N·s/m]"]):
            lbl = QtWidgets.QLabel(f"<b>{text}</b>")
            lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.grid.addWidget(lbl, 0, col)
        layout.addLayout(self.grid)
        note = QtWidgets.QLabel("kᵢ, cᵢ connect mᵢ₋₁ to mᵢ (k₁, c₁ connect m₁ to ground).")
        note.setWordWrap(True)
        note.setStyleSheet("color: #666;")
        layout.addWidget(note)

        self.rows: list[tuple[QtWidgets.QWidget, ...]] = []  # (label, m, k, c)
        self._build_rows(system)

    def _build_rows(self, system: ChainSystem) -> None:
        for row in self.rows:
            for w in row:
                self.grid.removeWidget(w)
                w.deleteLater()
        self.rows = []
        for i in range(system.n):
            label = QtWidgets.QLabel(f"<b style='color:{MASS_COLORS[i]}'>■</b> {i + 1}")
            m = spin(1e-3, 1e4, system.masses[i], 3)
            k = spin(0.0, 1e7, system.stiffness[i], 1)
            c = spin(0.0, 1e5, system.damping[i], 3)
            for col, w in enumerate((label, m, k, c)):
                self.grid.addWidget(w, i + 1, col)
            for box in (m, k, c):
                box.valueChanged.connect(self._emit)
            self.rows.append((label, m, k, c))

    def system(self) -> ChainSystem:
        return ChainSystem(
            [r[1].value() for r in self.rows],
            [r[2].value() for r in self.rows],
            [r[3].value() for r in self.rows],
        )

    def _emit(self) -> None:
        self.changed.emit(self.system())

    def _on_dof(self, n: int) -> None:
        self._build_rows(self.system().resized(n))
        self._emit()

    def _set_all(self, system: ChainSystem) -> None:
        for (_, m, k, c), mv, kv, cv in zip(self.rows, system.masses, system.stiffness, system.damping):
            for box, v in ((m, mv), (k, kv), (c, cv)):
                box.blockSignals(True)
                box.setValue(v)
                box.blockSignals(False)
        self._emit()

    def _reset_defaults(self) -> None:
        self._set_all(ChainSystem.uniform(len(self.rows), DEFAULT_MASS, DEFAULT_STIFFNESS, DEFAULT_DAMPING))

    def _copy_first_row(self) -> None:
        s = self.system()
        self._set_all(ChainSystem.uniform(s.n, s.masses[0], s.stiffness[0], s.damping[0]))


class ForcePanel(QtWidgets.QGroupBox):
    """Edits the ForceController's settings in place and switches it on/off."""

    settings_changed = QtCore.Signal()

    def __init__(self, force: ForceController, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("Applied force", parent)
        self.force = force
        s = force.settings
        form = QtWidgets.QFormLayout(self)

        self.target = QtWidgets.QComboBox()
        form.addRow("Apply to:", self.target)

        self.kind = QtWidgets.QComboBox()
        for k in ForceKind:
            self.kind.addItem(k.value, k)
        self.kind.setCurrentIndex(list(ForceKind).index(s.kind))
        form.addRow("Type:", self.kind)

        self.amplitude = spin(-1e5, 1e5, s.amplitude, 3, " N")
        form.addRow("Amplitude:", self.amplitude)

        freq_row = QtWidgets.QHBoxLayout()
        self.freq = spin(0.001, 1000.0, s.freq_hz, 3, " Hz")
        freq_row.addWidget(self.freq, 1)
        self.tune = QtWidgets.QComboBox()
        self.tune.setToolTip("Set the drive frequency to a natural frequency (resonance)")
        freq_row.addWidget(self.tune)
        self.freq_label = QtWidgets.QLabel("Frequency:")
        form.addRow(self.freq_label, freq_row)

        self.duration = spin(1e-4, 10.0, s.pulse_duration, 4, " s")
        self.duration_label = QtWidgets.QLabel("Pulse length:")
        form.addRow(self.duration_label, self.duration)

        self.button = QtWidgets.QPushButton()
        self.button.setMinimumHeight(40)
        self.button.setShortcut("Space")
        self.button.clicked.connect(self._on_button)
        form.addRow(self.button)

        self.target.currentIndexChanged.connect(self._apply)
        self.kind.currentIndexChanged.connect(self._apply)
        self.amplitude.valueChanged.connect(self._apply)
        self.freq.valueChanged.connect(self._apply)
        self.duration.valueChanged.connect(self._apply)
        self.tune.activated.connect(self._on_tune)
        self.refresh()

    def set_dof(self, n: int) -> None:
        self.target.blockSignals(True)
        self.target.clear()
        self.target.addItems([f"m{i + 1}" for i in range(n)])
        self.target.setCurrentIndex(min(self.force.settings.target, n - 1))
        self.target.blockSignals(False)
        self._apply()

    def set_modes(self, result: ModalResult) -> None:
        fill_mode_combo(self.tune, result, "Tune to…")

    def _on_tune(self, index: int) -> None:
        f = self.tune.itemData(index)
        if f:
            self.freq.setValue(f)
        self.tune.setCurrentIndex(0)

    def _apply(self) -> None:
        s = self.force.settings
        kind = self.kind.currentData()
        if kind is not s.kind:
            self.force.switch_off()
        s.target = max(0, self.target.currentIndex())
        s.kind = kind
        s.amplitude = self.amplitude.value()
        s.freq_hz = self.freq.value()
        s.pulse_duration = self.duration.value()
        self.refresh()
        self.settings_changed.emit()

    def _on_button(self) -> None:
        if self.force.settings.kind is ForceKind.PULSE:
            self.force.fire_pulse()
        elif self.force.on:
            self.force.switch_off()
        else:
            self.force.switch_on()
        self.refresh()
        self.settings_changed.emit()

    def refresh(self) -> None:
        kind = self.force.settings.kind
        harmonic = kind is ForceKind.HARMONIC
        pulse = kind is ForceKind.PULSE
        for w in (self.freq_label, self.freq, self.tune):
            w.setVisible(harmonic)
        self.duration_label.setVisible(pulse)
        self.duration.setVisible(pulse)
        if pulse:
            self.button.setText("Fire pulse  [Space]")
            self.button.setStyleSheet("")
        elif self.force.on:
            self.button.setText("Force ON — click to release  [Space]")
            self.button.setStyleSheet("background-color: #c1121f; color: white; font-weight: bold;")
        else:
            self.button.setText("Apply force  [Space]")
            self.button.setStyleSheet("")


class SimControls(QtWidgets.QGroupBox):
    run_toggled = QtCore.Signal(bool)
    reset_clicked = QtCore.Signal()

    SPEEDS = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("Simulation", parent)
        form = QtWidgets.QFormLayout(self)

        buttons = QtWidgets.QHBoxLayout()
        self.run = QtWidgets.QPushButton("Pause")
        self.run.setCheckable(True)
        self.run.setChecked(True)
        self.run.toggled.connect(self._on_run)
        self.reset = QtWidgets.QPushButton("Reset")
        self.reset.setToolTip("Return all masses to rest, switch the force off and clear the plots")
        self.reset.clicked.connect(self.reset_clicked)
        buttons.addWidget(self.run)
        buttons.addWidget(self.reset)
        form.addRow(buttons)

        self.speed = QtWidgets.QComboBox()
        for s in self.SPEEDS:
            self.speed.addItem(f"{s:g}× real time", s)
        self.speed.setCurrentIndex(self.SPEEDS.index(1.0))
        form.addRow("Speed:", self.speed)

        self.window = spin(0.05, 120.0, 10.0, 2, " s")
        form.addRow("Plot window:", self.window)

        fit_row = QtWidgets.QHBoxLayout()
        self.cycles = QtWidgets.QSpinBox()
        self.cycles.setRange(1, 200)
        self.cycles.setValue(10)
        self.cycles.setSuffix(" cycles")
        self.cycles.setKeyboardTracking(False)
        self.cycles.valueChanged.connect(self._apply_fit)
        fit_row.addWidget(self.cycles)
        fit_row.addWidget(QtWidgets.QLabel("of"))
        self.fit_window = QtWidgets.QComboBox()
        self.fit_window.activated.connect(self._on_fit_window)
        fit_row.addWidget(self.fit_window, 1)
        fit_tip = (
            "Set the plot window to show this many cycles of a mode (window = cycles / f\u2099).\n"
            "Changing the cycle count re-fits to the last mode picked."
        )
        for w in (self.cycles, self.fit_window):
            w.setToolTip(fit_tip)
        form.addRow("Fit window:", fit_row)
        self._fit_item: int | None = None  # combo index of the last mode picked

        self.auto_scale = QtWidgets.QCheckBox("Auto-scale animation and plots")
        self.auto_scale.setChecked(True)
        self.auto_scale.setToolTip("On: magnify the animation and fit the plot axes to the recent motion.\n"
            "Off: freeze the current magnification and plot ranges (drag an axis to adjust).")
        form.addRow(self.auto_scale)

        self.time_label = QtWidgets.QLabel()
        form.addRow("Sim time:", self.time_label)

    def set_modes(self, result: ModalResult) -> None:
        fill_mode_combo(self.fit_window, result, "mode…")

    def _on_fit_window(self, index: int) -> None:
        if self.fit_window.itemData(index):
            self._fit_item = index
            self._apply_fit()
        self.fit_window.setCurrentIndex(0)

    def _apply_fit(self) -> None:
        # Look the frequency up each time so it reflects the current parameters.
        f = self.fit_window.itemData(self._fit_item) if self._fit_item else None
        if f:
            self.window.setValue(self.cycles.value() / f)

    def _on_run(self, running: bool) -> None:
        self.run.setText("Pause" if running else "Run")
        self.run_toggled.emit(running)

    @property
    def speed_factor(self) -> float:
        return float(self.speed.currentData())
