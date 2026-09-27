"""Input panels: system parameters, applied force, and simulation controls."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from ..core import ChainSystem, ForceController, ForceKind
from ..core.model import DEFAULT_DAMPING, DEFAULT_MASS, DEFAULT_STIFFNESS
from .modes import ModeEntry
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


def fill_mode_combo(
    combo: QtWidgets.QComboBox, entries: list[ModeEntry], placeholder: str, data=lambda e: e.freq_hz
) -> None:
    """List the modes in a one-shot "pick a mode" combo; item data is data(entry)."""
    combo.clear()
    combo.addItem(placeholder)
    for e in entries:
        if e.listed:  # skip rigid / non-oscillatory modes and conjugate duplicates
            combo.addItem(f"{e.key} ({e.freq_hz:.3g} Hz)", data(e))


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


BASE_TIP = (
    "<p><b>Ground motion (base excitation):</b> instead of pushing a mass, the wall that "
    "k<sub>1</sub> and c<sub>1</sub> are fixed to moves, x<sub>g</sub>(t), like a machine on a "
    "shaking floor or a building in an earthquake. It reaches the chain only through element 1, "
    "as the force k<sub>1</sub>x<sub>g</sub> + c<sub>1</sub>ẋ<sub>g</sub> on m1.</p>"
    "<p>The plots show the absolute displacements x; the <i>Frequency response</i> tab shows "
    "the transmissibility |X<sub>i</sub>/X<sub>g</sub>|, which is 1 at low frequency (the chain "
    "moves with the ground) and falls away above the modes (isolation).</p>"
)
CHIRP_TIP = (
    "<p><b>Chirp:</b> a sine whose frequency sweeps from the start to the end frequency over "
    "the sweep time, then stops. As it passes each natural frequency the response swells, so "
    "a slow sweep traces out the frequency response in time. The dashed line on the "
    "<i>Frequency response</i> tab follows the frequency.</p>"
    "<p>Sweep too fast and a lightly damped mode has no time to build up: its peak comes "
    "late and low, and it rings on after the sweep has moved on.</p>"
    "<p><b>Log sweep:</b> equal time per octave rather than per hertz, so the low modes, "
    "which are closer together, get as long as the high ones.</p>"
)


class ForcePanel(QtWidgets.QGroupBox):
    """Edits the ForceController's settings in place and switches it on/off."""

    settings_changed = QtCore.Signal()

    def __init__(self, force: ForceController, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("Excitation", parent)
        self.force = force
        s = force.settings
        form = QtWidgets.QFormLayout(self)

        self.input = QtWidgets.QComboBox()
        self.input.addItem("Force on a mass", False)
        self.input.addItem("Ground motion (base)", True)
        self.input.setCurrentIndex(int(s.base))
        self.input.setToolTip(BASE_TIP)
        form.addRow("Input:", self.input)

        self.target = QtWidgets.QComboBox()
        self.target_label = QtWidgets.QLabel("Apply to:")
        form.addRow(self.target_label, self.target)

        self.kind = QtWidgets.QComboBox()
        for k in ForceKind:
            self.kind.addItem(k.value, k)
        self.kind.setCurrentIndex(list(ForceKind).index(s.kind))
        form.addRow("Type:", self.kind)

        amp_row = QtWidgets.QHBoxLayout()
        self.amplitude = spin(-1e5, 1e5, s.amplitude, 3, " N")
        self.base_amplitude = spin(-1e4, 1e4, s.base_amplitude * 1e3, 3, " mm")
        amp_row.addWidget(self.amplitude)
        amp_row.addWidget(self.base_amplitude)
        form.addRow("Amplitude:", amp_row)

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

        sweep_row = QtWidgets.QHBoxLayout()
        self.sweep_start = spin(0.001, 1000.0, s.sweep_start_hz, 3, " Hz")
        self.sweep_end = spin(0.001, 1000.0, s.sweep_end_hz, 3, " Hz")
        sweep_to = QtWidgets.QLabel("to")
        sweep_row.addWidget(self.sweep_start, 1)
        sweep_row.addWidget(sweep_to)
        sweep_row.addWidget(self.sweep_end, 1)
        self.sweep_label = QtWidgets.QLabel("Sweep:")
        form.addRow(self.sweep_label, sweep_row)
        time_row = QtWidgets.QHBoxLayout()
        self.sweep_time = spin(0.1, 3600.0, s.sweep_time, 1, " s")
        self.sweep_log = QtWidgets.QCheckBox("Log sweep")
        self.sweep_log.setChecked(s.sweep_log)
        time_row.addWidget(self.sweep_time, 1)
        time_row.addWidget(self.sweep_log)
        self.sweep_time_label = QtWidgets.QLabel("Sweep time:")
        form.addRow(self.sweep_time_label, time_row)
        self._sweep_widgets = (
            self.sweep_label, self.sweep_start, sweep_to, self.sweep_end,
            self.sweep_time_label, self.sweep_time, self.sweep_log,
        )
        for w in self._sweep_widgets:
            w.setToolTip(CHIRP_TIP)

        self.button = QtWidgets.QPushButton()
        self.button.setMinimumHeight(40)
        self.button.setShortcut("Space")
        self.button.clicked.connect(self._on_button)
        form.addRow(self.button)

        self.input.currentIndexChanged.connect(self._apply)
        self.target.currentIndexChanged.connect(self._apply)
        self.kind.currentIndexChanged.connect(self._apply)
        for box in (
            self.amplitude, self.base_amplitude, self.freq, self.duration,
            self.sweep_start, self.sweep_end, self.sweep_time,
        ):
            box.valueChanged.connect(self._apply)
        self.sweep_log.toggled.connect(self._apply)
        self.tune.activated.connect(self._on_tune)
        self.refresh()

    def set_dof(self, n: int) -> None:
        self.target.blockSignals(True)
        self.target.clear()
        self.target.addItems([f"m{i + 1}" for i in range(n)])
        self.target.setCurrentIndex(min(self.force.settings.target, n - 1))
        self.target.blockSignals(False)
        self._apply()

    def set_modes(self, entries: list[ModeEntry]) -> None:
        fill_mode_combo(self.tune, entries, "Tune to…")

    def _on_tune(self, index: int) -> None:
        f = self.tune.itemData(index)
        if f:
            self.freq.setValue(f)
        self.tune.setCurrentIndex(0)

    def _apply(self) -> None:
        s = self.force.settings
        kind = self.kind.currentData()
        base = bool(self.input.currentData())
        if kind is not s.kind or base != s.base:
            self.force.switch_off()
        s.base = base
        s.target = max(0, self.target.currentIndex())
        s.kind = kind
        s.amplitude = self.amplitude.value()
        s.base_amplitude = self.base_amplitude.value() * 1e-3
        s.freq_hz = self.freq.value()
        s.pulse_duration = self.duration.value()
        s.sweep_start_hz = self.sweep_start.value()
        s.sweep_end_hz = self.sweep_end.value()
        s.sweep_time = self.sweep_time.value()
        s.sweep_log = self.sweep_log.isChecked()
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
        s = self.force.settings
        harmonic = s.kind is ForceKind.HARMONIC
        pulse = s.kind is ForceKind.PULSE
        chirp = s.kind is ForceKind.CHIRP
        for w in (self.freq_label, self.freq, self.tune):
            w.setVisible(harmonic)
        self.duration_label.setVisible(pulse)
        self.duration.setVisible(pulse)
        for w in self._sweep_widgets:
            w.setVisible(chirp)
        self.target_label.setVisible(not s.base)
        self.target.setVisible(not s.base)
        self.amplitude.setVisible(not s.base)
        self.base_amplitude.setVisible(s.base)
        noun = "ground motion" if s.base else "force"
        if pulse:
            self.button.setText("Fire pulse  [Space]")
            self.button.setStyleSheet("")
        elif self.force.on:
            action = "sweeping" if chirp else "ON"
            self.button.setText(f"{noun.capitalize()} {action} — click to stop  [Space]")
            self.button.setStyleSheet("background-color: #c1121f; color: white; font-weight: bold;")
        else:
            verb = "Start sweep" if chirp else ("Move ground" if s.base else "Apply force")
            self.button.setText(f"{verb}  [Space]")
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
            "Set the plot window to show this many cycles of a mode (window = cycles / f\u2099;\n"
            "f_d for the state-space method).\n"
            "Changing the cycle count re-fits to the last mode picked.\n"
            "Releasing a mode also fits to it while auto-scale is on."
        )
        for w in (self.cycles, self.fit_window):
            w.setToolTip(fit_tip)
        form.addRow("Fit window:", fit_row)
        self._fit_mode: str | None = None  # key of the last mode fitted to
        self._mode_freqs: dict[str, float] = {}  # mode key -> frequency [Hz]

        self.auto_scale = QtWidgets.QCheckBox("Auto-scale animation and plots")
        self.auto_scale.setChecked(True)
        self.auto_scale.setToolTip("On: magnify the animation and fit the plot axes to the recent motion.\n"
            "Off: freeze the current magnification and plot ranges (drag an axis to adjust).")
        form.addRow(self.auto_scale)

        self.time_label = QtWidgets.QLabel()
        form.addRow("Sim time:", self.time_label)

    def set_modes(self, entries: list[ModeEntry]) -> None:
        fill_mode_combo(self.fit_window, entries, "mode\u2026", data=lambda e: e.key)
        self._mode_freqs = {e.key: e.freq_hz for e in entries if e.freq_hz > 0}

    def fit_to_mode(self, key: str) -> None:
        """Size the plot window to show `cycles` periods of the given mode."""
        self._fit_mode = key
        self._apply_fit()

    def _on_fit_window(self, index: int) -> None:
        key = self.fit_window.itemData(index)
        if key:
            self.fit_to_mode(key)
        self.fit_window.setCurrentIndex(0)

    def _apply_fit(self) -> None:
        # Look the frequency up each time so it reflects the current parameters.
        f = self._mode_freqs.get(self._fit_mode) if self._fit_mode else None
        if f:
            self.window.setValue(self.cycles.value() / f)

    def _on_run(self, running: bool) -> None:
        self.run.setText("Pause" if running else "Run")
        self.run_toggled.emit(running)

    @property
    def speed_factor(self) -> float:
        return float(self.speed.currentData())
