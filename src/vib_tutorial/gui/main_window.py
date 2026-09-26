"""Main window: wires the panels and plots to the simulator on a real-time timer."""

from __future__ import annotations

import math

import numpy as np
from PySide6 import QtCore, QtWidgets

from ..core import ChainSystem, ForceController, ForceKind, ForceSettings, Simulator, modal_analysis
from .animation import ChainView
from .background import COUPLING_TIP, make_background_view
from .history import History
from .modes import Method, mode_entries
from .panels import ForcePanel, ParameterPanel, SimControls, spin
from .plots import FrfPlot, ModalTable, ModeShapePlot, PhasorPanel, TimeHistoryPlot

FRAME_MS = 16  # ~60 fps
MODE_ANIMATION_HZ = 0.5  # visual rate for the animated mode-shape plot

METHOD_TIP = (
    "<p><b>Classical:</b> solve the undamped problem Kφ = ω²Mφ for N real mode shapes, "
    "then estimate each mode's damping from φᵀCφ. Exact only for proportional damping.</p>"
    "<p><b>State-space:</b> rewrite the N second-order equations as 2N first-order ones, "
    "ż = Az with z = [x, ẋ], and solve the full damped eigenproblem Aψ = λψ. This gives "
    "2N eigenvalues: complex-conjugate pairs λ, λ* for oscillatory modes (and real ones for "
    "overdamped motion), with complex mode shapes. Exact for any damping.</p>"
)
RELEASE_TIP_COMMON = (
    "<p><b>Free vibration from a mode shape.</b></p>"
    "<p>Switches off the applied force, puts the masses into the mode selected in the table "
    "above (the first row if none is selected), then lets go.</p>"
)
RELEASE_TIP = {
    Method.CLASSICAL: RELEASE_TIP_COMMON
    + "<p>The masses start at rest in the real (undamped) shape.</p>"
    "<p>With <b>proportional</b> damping only that mode responds: every mass oscillates "
    "at the mode's damped frequency f_d, keeping the same shape while the motion "
    "decays at the rate set by ζ.</p>"
    "<p>With <b>non-proportional</b> damping the real shape is not an exact "
    "mode, so other modes are excited too and the shape drifts as it decays.</p>",
    Method.STATE_SPACE: RELEASE_TIP_COMMON
    + "<p>The motion of one eigenvalue pair is x(t) = Re(ψ e<sup>λt</sup>), so the masses "
    "start at displacement Re(ψ) <i>with velocity</i> Re(λψ). That excites only this "
    "eigenvalue pair, even with non-proportional damping: the masses keep their relative "
    "amplitudes and phase lags while the motion decays.</p>"
    "<p>λ and its conjugate λ* give the same real motion, so releasing either one does "
    "the same thing. A real eigenvalue releases a non-oscillatory decay.</p>",
}
RELEASE_FIT_TIP = (
    "<p>While <i>Auto-scale animation and plots</i> is on, the plot window is also "
    "fitted to this mode (Simulation \u2192 Fit window cycles).</p>"
)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Vibration Tutorial — lumped mass-spring-damper chain")

        system = ChainSystem.uniform(4)
        self.force = ForceController(ForceSettings(target=system.n - 1))
        self.sim = Simulator(system, self.force)
        self.history = History(system.n)
        self.modal = modal_analysis(system)
        self.method = Method.CLASSICAL
        self.entries = mode_entries(self.modal, self.method)
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
        self.method_combo = QtWidgets.QComboBox()
        for m in Method:
            self.method_combo.addItem(m.value, m)
        self.method_combo.setToolTip(METHOD_TIP)
        method_label = QtWidgets.QLabel("Method:")
        method_label.setToolTip(METHOD_TIP)
        self.table = ModalTable()
        self.mode_plot = ModeShapePlot()
        self.phasor_plot = PhasorPanel()
        self.phasor_plot.setToolTip(
            "<p>The selected mode shape ψ in the complex plane: one arrow per mass "
            "(mass colors), length = amplitude, angle = phase.</p>"
            "<p>With <i>Animate mode shapes</i> on, the arrows rotate as ψe<sup>iω_d t</sup> "
            "(anticlockwise for Im λ &gt; 0, clockwise for its conjugate λ*). The circles on "
            "the real axis are the real parts, i.e. the masses' displacements.</p>"
            "<p>All arrows on one line (0° or 180° apart): a real mode, a standing wave. "
            "Arrows at other angles: the masses peak at different times.</p>"
        )
        self.phasor_plot.setMinimumWidth(180)
        self.modal_note = QtWidgets.QLabel()
        self.modal_note.setWordWrap(True)
        self.modal_note.setToolTip(COUPLING_TIP)
        self.modal_note.linkActivated.connect(lambda _: self.tabs.setCurrentWidget(self.background))
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
        self.release_button = release = QtWidgets.QPushButton("Release selected mode")
        release.clicked.connect(self._release_mode)
        modal_tab = QtWidgets.QWidget()
        mv = QtWidgets.QVBoxLayout(modal_tab)
        method_row = QtWidgets.QHBoxLayout()
        method_row.addWidget(method_label)
        method_row.addWidget(self.method_combo, 1)
        mv.addLayout(method_row)
        mv.addWidget(self.table)
        mv.addWidget(self.modal_note)
        plots_row = QtWidgets.QHBoxLayout()
        plots_row.addWidget(self.mode_plot, 3)
        plots_row.addWidget(self.phasor_plot, 2)
        mv.addLayout(plots_row, 1)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.animate_modes)
        row.addStretch(1)
        row.addWidget(amp_label)
        row.addWidget(self.release_amp)
        row.addWidget(release)
        mv.addLayout(row)

        self.frf_plot = FrfPlot()
        self.background = make_background_view()
        tabs = self.tabs = QtWidgets.QTabWidget()
        tabs.addTab(modal_tab, "Modal analysis")
        tabs.addTab(self.frf_plot, "Frequency response")
        tabs.addTab(self.background, "Background")

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
        self.animate_modes.toggled.connect(lambda on: on or self._animate_modes(0.0))
        self.table.itemSelectionChanged.connect(self._on_mode_selected)
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)

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
        self._refresh_modal_views()

    def _on_method_changed(self) -> None:
        self.method = self.method_combo.currentData()
        self._refresh_modal_views()

    def _refresh_modal_views(self) -> None:
        """Show the current modal result using the selected method."""
        self.entries = mode_entries(self.modal, self.method)
        state_space = self.method is Method.STATE_SPACE
        self.table.set_result(self.modal, self.method, self.entries)
        self.mode_plot.set_entries(self.entries, show_envelope=state_space)
        self.phasor_plot.setVisible(state_space)
        self.phasor_plot.set_entries(self.entries)
        self._on_mode_selected()
        self.force_panel.set_modes(self.entries)
        self.controls.set_modes(self.entries)
        self.release_button.setToolTip(RELEASE_TIP[self.method] + RELEASE_FIT_TIP)
        self.modal_note.setText(self._modal_note())
        self._on_force_changed()

    def _modal_note(self) -> str:
        notes = []
        n = self.sim.system.n
        if self.method is Method.STATE_SPACE:
            notes.append(
                f"State-space form: N = {n} second-order equations → 2N = {2 * n} first-order "
                "ones, so 2N eigenvalues. Oscillatory ones come in conjugate pairs λ, λ* that "
                "together make one real motion x(t) = 2 Re(ψ e<sup>λt</sup>)."
            )
            if self.modal.is_proportional:
                notes.append("Damping is <b>proportional</b>, so every ψ is real (phases 0° or 180°).")
            else:
                notes.append(
                    "Damping is <b>non-proportional</b>, so the ψ are complex: the masses peak "
                    "at different times (select a row to see the phases)."
                )
        elif self.modal.is_proportional:
            notes.append(
                "Damping is <b>proportional</b>: it does not couple the undamped modes, so the "
                "modes are real and ζ modal = ζ exact."
            )
        else:
            notes.append(
                "Damping is <b>non-proportional</b>: it couples the undamped modes "
                f"(coupling index {self.modal.coupling:.2f}; 0 = none, 1 = strong). The exact damped "
                "modes are complex (masses peak at different times) and ζ modal is an approximation."
            )
        notes.append("<a href='#background'>Why?</a>")
        if self.method is Method.CLASSICAL and self.modal.overdamped_roots:
            roots = ", ".join(f"{r:.3g}" for r in self.modal.overdamped_roots)
            notes.append(f"Non-oscillatory (overdamped) roots λ = {roots} 1/s.")
        return " ".join(notes)

    def _on_mode_selected(self) -> None:
        r = self.table.currentRow() if self.table.selectedItems() else None
        self.mode_plot.set_highlight(r)
        self.phasor_plot.set_highlight(r)

    def _animate_modes(self, theta: float) -> None:
        self.mode_plot.animate(theta)
        if self.phasor_plot.isVisible():
            self.phasor_plot.animate(theta)

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
        if not (0 <= r < len(self.entries)) or not self.table.selectedItems():
            r = 0
            self.table.selectRow(0)
        self.force.switch_off()
        self.force_panel.refresh()
        entry = self.entries[r]
        n = self.sim.system.n
        z0 = entry.state0 * self.release_amp.value() * 1e-3
        self.sim.set_state(z0[:n], z0[n:])
        if self.controls.auto_scale.isChecked() and entry.freq_hz > 0:
            self.controls.fit_to_mode(entry.key)

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
            self._animate_modes(2 * math.pi * MODE_ANIMATION_HZ * now)
        self.controls.time_label.setText(f"{self.sim.t:8.3f} s   (step {self.sim.step_size() * 1e3:.3g} ms)")
