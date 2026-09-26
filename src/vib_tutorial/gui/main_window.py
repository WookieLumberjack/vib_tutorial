"""Main window: wires the panels and plots to the simulator on a real-time timer."""

from __future__ import annotations

import math

import numpy as np
from PySide6 import QtCore, QtWidgets

from ..core import ChainSystem, ForceController, ForceKind, ForceSettings, Simulator, modal_analysis
from .animation import ChainView
from .history import History
from .panels import ForcePanel, ParameterPanel, SimControls, spin
from .plots import FrfPlot, ModalTable, ModeShapePlot, TimeHistoryPlot

FRAME_MS = 16  # ~60 fps
MODE_ANIMATION_HZ = 0.5  # visual rate for the animated mode-shape plot


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Vibration Tutorial — lumped mass-spring-damper chain")

        system = ChainSystem.uniform(4)
        self.force = ForceController(ForceSettings(target=system.n - 1))
        self.sim = Simulator(system, self.force)
        self.history = History(system.n)
        self.modal = modal_analysis(system)
        self.running = True

        # --- left: inputs
        self.params = ParameterPanel(system)
        self.force_panel = ForcePanel(self.force)
        self.controls = SimControls()
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        for w in (self.params, self.force_panel, self.controls):
            lv.addWidget(w)
        lv.addStretch(1)
        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidget(left)
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(380)

        # --- center: animation + time histories
        self.chain = ChainView()
        self.time_plot = TimeHistoryPlot()
        center = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        center.addWidget(self.chain)
        center.addWidget(self.time_plot)
        center.setSizes([300, 450])

        # --- right: modal reference
        self.table = ModalTable()
        self.mode_plot = ModeShapePlot()
        self.modal_note = QtWidgets.QLabel()
        self.modal_note.setWordWrap(True)
        self.animate_modes = QtWidgets.QCheckBox("Animate mode shapes")
        self.release_amp = spin(0.001, 1000.0, 20.0, 3, " mm")
        amp_tip = (
            "<p>Starting displacement of the mass that moves most when a mode is released. "
            "The other masses are scaled by the mode shape.</p>"
            "<p>The system is linear, so this changes the size of the motion but not how fast "
            "it decays; the decay rate is set by the mode's damping ratio ζ. To watch a "
            "well-damped mode, slow the simulation down (Simulation → Speed).</p>"
        )
        self.release_amp.setToolTip(amp_tip)
        amp_label = QtWidgets.QLabel("Initial displacement:")
        amp_label.setToolTip(amp_tip)
        release = QtWidgets.QPushButton("Release selected mode")
        release.setToolTip(
            "<p><b>Free vibration from a mode shape.</b></p>"
            "<p>Switches off the applied force, moves the masses into the shape of the mode "
            "selected in the table above (mode 1 if none is selected), holds them at rest, "
            "then lets go.</p>"
            "<p>With <b>proportional</b> damping only that mode responds: every mass oscillates "
            "at the mode's damped frequency f_d, keeping the same shape while the motion "
            "decays at the rate set by ζ.</p>"
            "<p>With <b>non-proportional</b> damping the real (undamped) shape is not an exact "
            "mode, so other modes are excited too and the shape drifts as it decays.</p>"
            "<p>While <i>Auto-scale animation and plots</i> is on, the plot window is also "
            "fitted to this mode (Simulation \u2192 Fit window cycles).</p>"
        )
        release.clicked.connect(self._release_mode)
        modal_tab = QtWidgets.QWidget()
        mv = QtWidgets.QVBoxLayout(modal_tab)
        mv.addWidget(self.table)
        mv.addWidget(self.modal_note)
        mv.addWidget(self.mode_plot, 1)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.animate_modes)
        row.addStretch(1)
        row.addWidget(amp_label)
        row.addWidget(self.release_amp)
        row.addWidget(release)
        mv.addLayout(row)

        self.frf_plot = FrfPlot()
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(modal_tab, "Modal analysis")
        tabs.addTab(self.frf_plot, "Frequency response")

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(left_scroll)
        splitter.addWidget(center)
        splitter.addWidget(tabs)
        splitter.setSizes([400, 750, 550])
        self.setCentralWidget(splitter)
        self.resize(1700, 900)

        # --- wiring
        self.params.changed.connect(self._on_system_changed)
        self.force_panel.settings_changed.connect(self._on_force_changed)
        self.controls.run_toggled.connect(self._on_run_toggled)
        self.controls.reset_clicked.connect(self._reset)
        self.controls.auto_scale.toggled.connect(self._on_auto_scale)
        self.animate_modes.toggled.connect(lambda on: on or self.mode_plot.animate(1.0))
        self.table.itemSelectionChanged.connect(
            lambda: self.mode_plot.set_highlight(self.table.currentRow() if self.table.selectedItems() else None)
        )

        self._apply_dof(system.n)
        self._refresh_modal()

        self._clock = QtCore.QElapsedTimer()
        self._clock.start()
        self._last_wall = 0.0
        self._sim_target = 0.0
        # Single-shot timer re-armed after each frame: if a frame runs long, the
        # event loop still gets to handle input before the next one starts.
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._frame)
        self._timer.start(FRAME_MS)

    # --------------------------------------------------------------- events
    def _on_system_changed(self, system: ChainSystem) -> None:
        n_changed = system.n != self.sim.system.n
        self.sim.set_system(system)
        if n_changed:
            self._apply_dof(system.n)
        else:
            self.chain.set_masses(system.masses)
        self._refresh_modal()

    def _apply_dof(self, n: int) -> None:
        self.history.reset(n)
        self._sim_target = self.sim.t
        self.chain.set_masses(self.sim.system.masses)
        self.time_plot.set_dof(n)
        self.force_panel.set_dof(n)

    def _refresh_modal(self) -> None:
        self.modal = modal_analysis(self.sim.system)
        self.table.set_result(self.modal)
        self.mode_plot.set_result(self.modal)
        self.force_panel.set_modes(self.modal)
        self.controls.set_modes(self.modal)
        notes = []
        if self.modal.is_proportional:
            notes.append("Damping is <b>proportional</b>: modes are real and uncoupled, so ζ modal = ζ exact.")
        else:
            notes.append(
                f"Damping is <b>non-proportional</b> (coupling {self.modal.coupling:.2f}): exact modes are "
                "complex and ζ modal is an approximation."
            )
        if self.modal.overdamped_roots:
            roots = ", ".join(f"{r:.3g}" for r in self.modal.overdamped_roots)
            notes.append(f"Non-oscillatory (overdamped) roots λ = {roots} 1/s.")
        self.modal_note.setText(" ".join(notes))
        self._on_force_changed()

    def _on_force_changed(self) -> None:
        s = self.force.settings
        self.frf_plot.set_system(self.sim.system, self.modal, s.target)
        self.frf_plot.set_drive(s.freq_hz if s.kind is ForceKind.HARMONIC else None)

    def _on_auto_scale(self, on: bool) -> None:
        self.chain.auto_scale = on
        self.time_plot.set_auto_range(on)

    def _on_run_toggled(self, running: bool) -> None:
        self.running = running
        self._sim_target = self.sim.t

    def _reset(self) -> None:
        self.sim.reset()
        self.history.reset(self.sim.system.n)
        self._sim_target = 0.0
        self.force_panel.refresh()

    def _release_mode(self) -> None:
        r = self.table.currentRow()
        if not (0 <= r < len(self.modal.modes)):
            r = 0
            self.table.selectRow(0)
        self.force.switch_off()
        self.force_panel.refresh()
        mode = self.modal.modes[r]
        self.sim.set_displacement(mode.shape * self.release_amp.value() * 1e-3)
        if self.controls.auto_scale.isChecked():
            self.controls.fit_to_mode(mode.index)

    # ------------------------------------------------------------ main loop
    def _frame(self) -> None:
        start = self._clock.elapsed()
        try:
            self._tick()
        finally:
            spent = self._clock.elapsed() - start
            self._timer.start(max(1, FRAME_MS - spent))

    def _tick(self) -> None:
        now = self._clock.elapsed() / 1000.0
        dt = min(now - self._last_wall, 0.1)  # don't jump after a stall
        self._last_wall = now

        if self.running:
            speed = self.controls.speed_factor
            self._sim_target += dt * speed
            # If the simulator can't keep up (very stiff system), drop the backlog.
            self._sim_target = min(self._sim_target, self.sim.t + 0.25 * speed + 0.05)
            ts, xs, fs = self.sim.advance(self._sim_target - self.sim.t)
            self.history.extend(ts, xs, fs)

        window = self.controls.window.value()
        t, x, f = self.history.window(window)
        peak = float(np.abs(x[-min(len(x), 20_000) :]).max()) if x.size else 0.0
        s = self.force.settings
        self.chain.update_state(self.sim.displacement, peak, self.force.value(), s.target, abs(s.amplitude))
        self.time_plot.update_data(t, x, f, window)
        if self.animate_modes.isChecked():
            self.mode_plot.animate(math.cos(2 * math.pi * MODE_ANIMATION_HZ * now))
        self.controls.time_label.setText(f"{self.sim.t:8.3f} s   (step {self.sim.step_size() * 1e3:.3g} ms)")
