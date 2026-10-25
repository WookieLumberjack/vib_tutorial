"""Duffing oscillator page: a mass on a hardening or softening spring, swept up and down through its resonance.

Left: the oscillator, the drive and the playback. Centre: the amplitude against the
frequency (harmonic balance's branches, backbone and folds, under each drive cycle's
measured first harmonic) and the displacement's strip chart. Right: the phase plane with
its Poincaré section, and the notes. The page has its own model (core/duffing.py), not
the chain.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.duffing import (
    PRESETS,
    Cycle,
    Duffing,
    DuffingSimulator,
    ResponseCurve,
    Sweep,
    amplitudes,
    linear_response,
    measured_jump,
    response_curve,
)
from . import frame_stats, settle_plots
from .layouts import ColumnScroll, FlowLayout, plot_height
from .panels import spin
from .style import colors
from .theming import add_legend, mute

FRAME_MS = 16
FRAME_BUDGET = 0.008  # s of computing per frame: past it the page runs slower than asked
SPEEDS = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
WINDOW = 20.0  # s of simulated time on the strip chart
PLOT_POINTS = 1200  # the strip chart's buckets (min and max of each)
CAPACITY = 1 << 18  # samples kept for the strip chart and the phase plane
PHASE_CYCLES = 3  # drive cycles of trajectory in the phase plane
SECTION_POINTS = 4000  # Poincaré dots kept
CUSTOM = "Custom"

TIPS = {
    "mass": "<p>The mass m.</p>",
    "stiffness": "<p>The spring's linear stiffness k: its stiffness at small amplitude.</p>",
    "damping": "<p>The viscous damping c.</p>",
    "cubic": "<p>The spring's cubic term k<sub>3</sub>: its force is k x + k<sub>3</sub> x³. Positive hardens "
             "the spring (a large swing is stiffer, so faster), negative softens it, down to nothing at the "
             "escape amplitude √(k / −k<sub>3</sub>), where the potential has a hilltop.</p>",
    "force": "<p>The drive's amplitude F: the force is F cos φ, with φ advancing at the drive's frequency.</p>",
    "freq": "<p>The drive's frequency. Held, it stays here; sweeping, this follows the sweep. Typing a frequency "
            "while sweeping stops the sweep there.</p>",
    "range": "<p>The sweep's lower and upper frequencies. A sweep up runs from the first to the second.</p>",
    "time": "<p>How long each leg of a sweep takes, in simulated time. Slower sweeps stay nearer the steady "
            "response and leave a branch nearer its fold.</p>",
    "branch": "<p>Put the mass on a steady response of harmonic balance at the held frequency, at the drive's "
              "phase now. Between the folds there are three: the lower and upper are stable, the middle one "
              "is not.</p>",
}
RESPONSE_TIP = (
    "<p>The amplitude of the displacement's first harmonic against the drive's frequency. Solid: harmonic "
    "balance's stable steady states; dashed: its unstable ones (the middle branch); dotted: the backbone, the "
    "free oscillation's frequency at each amplitude; grey: the spring without its cubic term. The arrows are "
    "the jumps from the folds, where a branch ends. The dots are each drive cycle's measured first harmonic, "
    "coloured by the sweep's direction.</p>")
PHASE_TIP = (
    "<p>The phase plane: velocity against displacement over the last few drive cycles. The dots are the "
    "Poincaré section, the state once per drive cycle (where the force peaks): one dot for a steady response "
    "at the drive's frequency, a few for a subharmonic, a cloud that never settles for chaos. Only while the "
    "frequency is held: a sweep's would be a different section each cycle.</p>")

NOTES_HTML = """
<h3>The Duffing oscillator</h3>
<p>A mass on a spring whose force is not proportional to its stretch:</p>
<p align="center">m x″ + c x′ + k x + k<sub>3</sub> x³ = F cos φ, &nbsp; φ′ = Ω</p>
<p>With k<sub>3</sub> &gt; 0 the spring <i>hardens</i>: the further it stretches, the stiffer it gets, so a big
swing is a fast one. With k<sub>3</sub> &lt; 0 it <i>softens</i>, and its force peaks at the escape amplitude
√(k / −k<sub>3</sub>): past that the spring pushes outward, and a mass carried over the potential's hilltop runs
away.</p>

<h3>Harmonic balance</h3>
<p>Guess that the mass moves at the drive's frequency, x = a cos(φ − θ), and keep only the first harmonic of
each force. Since cos³ = ¾ cos + ¼ cos 3φ, the spring acts on the first harmonic as one of stiffness</p>
<p align="center">κ(a) = k + ¾ k<sub>3</sub> a²,</p>
<p>stiffer the bigger the swing. Matching the cosine and sine parts:</p>
<p align="center">(κ(a) − mΩ²) a = F cos θ, &nbsp; cΩ a = F sin θ,</p>
<p>and squaring and adding, a²[(κ(a) − mΩ²)² + (cΩ)²] = F². That is a cubic in a², so at one frequency there
can be one steady amplitude or three. The curve leans along the <b>backbone</b>, mΩ² = κ(a): the frequency at
which the mass would swing freely at amplitude a. A hardening spring's leans to the right, a softening one's to
the left.</p>

<h3>The jump</h3>
<p>Where there are three amplitudes, the middle one is unstable (a saddle): nudged, the mass leaves it for one of
the other two, which are both stable. So the response depends on where it came from. Sweeping a hardening spring
up, the mass rides the upper branch past the linear resonance until the branch ends at a <b>fold</b>, and drops
to the lower one. Sweeping down, it follows the lower branch to its own fold, at a lower frequency, and jumps up.
Between the folds the two sweeps disagree: hysteresis. A softening spring does the same, mirrored.</p>
<p>Near a fold the steady state is barely stable: a disturbance there dies away ever more slowly (critical
slowing down), so a sweep overruns the fold before the mass leaves. The slower the sweep, the nearer the fold it
jumps. A jump up takes a few of the damping's time constants, 2m/c, to grow.</p>
<p>Hold the frequency inside the band and start the mass on each branch (<i>Start on a branch</i>): from the lower
or upper it stays; from the middle one it falls off to one or the other, depending on the slightest difference.
From rest it settles on the lower branch.</p>

<h3>How good is one harmonic?</h3>
<p>The cubic makes 3Ω too, which the balance leaves out. The measured dots are each drive cycle's first harmonic,
so they compare like with like, and they sit on the curve to within about 1 % even where the cubic doubles the
spring's stiffness. A strongly driven, lightly damped spring has more: <b>superharmonic</b> resonances, where 3Ω
(or 2Ω, with a static offset) meets the natural frequency, so near a third of it; <b>subharmonic</b> ones, at
three times it, where the mass swings at Ω/3; and <b>chaos</b>.</p>

<h3>Chaos</h3>
<p>Ueda's oscillator, x″ + 0.05 x′ + x³ = 7.5 cos t, has no linear spring, light damping and a big drive. Its
motion never repeats. The Poincaré section, the state once per drive cycle, shows it: one dot for a steady
response at the drive's frequency, three for a subharmonic at a third of it, and for chaos a cloud that fills a
folded, layered shape (a strange attractor) without ever coming back to the same point.</p>

<h3>The simulation</h3>
<p>The linear part, m x″ + c x′ + k x, is stepped exactly, as the chain's simulator steps it (a first-order hold
on the input). The cubic's force goes in as an input that changes linearly over each step, its end value found
by Newton from where the step ends. The step resolves the drive and the spring's stiffness at the largest
amplitude in reach. The measured amplitude is the displacement's first harmonic over each drive cycle.</p>
"""


def fmt_length(a: float) -> str:
    """A length in mm below a metre, else in m."""
    return f"{a * 1e3:.3g} mm" if abs(a) < 1.0 else f"{a:.3g} m"


class _Trace:
    """The last CAPACITY samples of t, x, v: a buffer twice as long, slid back when it fills."""

    def __init__(self) -> None:
        self.data = np.empty((3, 2 * CAPACITY))
        self.n = 0

    def clear(self) -> None:
        self.n = 0

    def add(self, t: np.ndarray, x: np.ndarray, v: np.ndarray) -> None:
        k = t.size
        if k >= CAPACITY:
            t, x, v, k = t[-CAPACITY:], x[-CAPACITY:], v[-CAPACITY:], CAPACITY
        if self.n + k > self.data.shape[1]:
            keep = CAPACITY - k
            self.data[:, :keep] = self.data[:, self.n - keep:self.n]
            self.n = keep
        self.data[:, self.n:self.n + k] = (t, x, v)
        self.n += k

    def since(self, t0: float) -> np.ndarray:
        """(3, n) of the samples from time t0 on."""
        t = self.data[0, :self.n]
        return self.data[:, int(np.searchsorted(t, t0)):self.n]


def min_max(t: np.ndarray, y: np.ndarray, buckets: int) -> tuple[np.ndarray, np.ndarray]:
    """Each bucket's min and max, in order, so the line covers what the whole record would."""
    if y.size <= 2 * buckets:
        return t, y
    size = -(-y.size // buckets)
    n = y.size // size * size
    yy = y[:n].reshape(-1, size)
    tt = t[:n:size]
    out = np.empty((2, yy.shape[0]))
    out[0], out[1] = yy.min(axis=1), yy.max(axis=1)
    return np.repeat(tt, 2), out.T.ravel()


def _plot(left: str, left_units: str | None, bottom: str, bottom_units: str | None) -> pg.PlotWidget:
    p = pg.PlotWidget()
    p.setMenuEnabled(False)
    p.hideButtons()
    p.showGrid(x=True, y=True, alpha=0.15)
    p.setLabel("left", left, units=left_units)
    p.setLabel("bottom", bottom, units=bottom_units)
    return p


class DuffingPage(QtWidgets.QWidget):
    """Controls on the left, the response and the strip chart in the centre, phase plane and notes on the right."""

    follows_chain = False  # MainWindow: this page has its own model, not the chain

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.sim = DuffingSimulator()
        self.curve: ResponseCurve | None = None
        self.running = True
        self.cycles: list[Cycle] = []  # measured since the last sweep started (or the parameters changed)
        self.section: list[tuple[float, float]] = []
        self.trace = _Trace()
        self._y_top = 0.0
        self._loading = False

        # --- left: the oscillator, the drive, the playback
        self.preset = QtWidgets.QComboBox()
        self.preset.addItems([p.name for p in PRESETS] + [CUSTOM])
        self.preset.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.preset_note = QtWidgets.QLabel()
        self.preset_note.setWordWrap(True)
        mute(self.preset_note)

        osc = QtWidgets.QGroupBox("Oscillator")
        form = QtWidgets.QFormLayout(osc)
        self.mass = spin(0.01, 1000.0, 1.0, 3, " kg")
        self.stiffness = spin(0.0, 1e7, 400.0, 1, " N/m")
        self.damping = spin(0.0, 1e4, 1.0, 3, " N·s/m")
        self.cubic = spin(-1e9, 1e9, 1e5, 1, " N/m³")
        self.force = spin(0.0, 1e4, 2.0, 3, " N")
        for text, box, key in (("Mass m:", self.mass, "mass"), ("Stiffness k:", self.stiffness, "stiffness"),
                               ("Damping c:", self.damping, "damping"), ("Cubic k₃:", self.cubic, "cubic"),
                               ("Drive F:", self.force, "force")):
            box.setToolTip(TIPS[key])
            box.valueChanged.connect(self._on_params)
            label = QtWidgets.QLabel(text)
            label.setToolTip(TIPS[key])
            form.addRow(label, box)
        self.osc_info = QtWidgets.QLabel()
        self.osc_info.setWordWrap(True)
        mute(self.osc_info)
        form.addRow(self.osc_info)

        drive = QtWidgets.QGroupBox("Drive")
        dv = QtWidgets.QFormLayout(drive)
        self.freq = spin(0.0, 1000.0, 3.0, 3, " Hz")
        self.start_hz = spin(0.0, 1000.0, 2.0, 3, " Hz")
        self.end_hz = spin(0.0, 1000.0, 6.0, 3, " Hz")
        self.sweep_time = spin(1.0, 1e5, 120.0, 0, " s")
        for text, box, key in (("Frequency:", self.freq, "freq"), ("Sweep from:", self.start_hz, "range"),
                               ("to:", self.end_hz, "range"), ("Each leg takes:", self.sweep_time, "time")):
            box.setToolTip(TIPS[key])
            label = QtWidgets.QLabel(text)
            label.setToolTip(TIPS[key])
            dv.addRow(label, box)
        self.freq.valueChanged.connect(self._on_freq)
        for box in (self.start_hz, self.end_hz, self.sweep_time):
            box.valueChanged.connect(self._on_sweep_range)
        self.sweep_buttons: dict[Sweep, QtWidgets.QPushButton] = {}
        row = FlowLayout()
        for mode, text in ((Sweep.UP, "Sweep up"), (Sweep.DOWN, "Sweep down"),
                           (Sweep.UP_DOWN, "Up, then down"), (Sweep.HOLD, "Hold")):
            b = self.sweep_buttons[mode] = QtWidgets.QPushButton(text)
            b.clicked.connect(lambda _=False, m=mode: self.start_sweep(m))
            row.add(b)
        dv.addRow(row)
        self.drive_state = QtWidgets.QLabel()
        mute(self.drive_state)
        dv.addRow(self.drive_state)

        branch = QtWidgets.QGroupBox("Start on a branch")
        branch.setToolTip(TIPS["branch"])
        bv = QtWidgets.QVBoxLayout(branch)
        row = FlowLayout()
        self.branch_buttons = []
        for i, text in enumerate(("Lower", "Middle", "Upper")):
            b = QtWidgets.QPushButton(text)
            b.setToolTip(TIPS["branch"])
            b.clicked.connect(lambda _=False, k=i: self.start_on_branch(k))
            self.branch_buttons.append(b)
            row.add(b)
        bv.addLayout(row)
        self.branch_note = QtWidgets.QLabel()
        self.branch_note.setWordWrap(True)
        mute(self.branch_note)
        bv.addWidget(self.branch_note)

        play = QtWidgets.QGroupBox("Simulation")
        pv = QtWidgets.QVBoxLayout(play)
        self.sim_speed = QtWidgets.QComboBox()
        for s in SPEEDS:
            self.sim_speed.addItem(f"{s:g}× real time", s)
        self.sim_speed.setCurrentIndex(SPEEDS.index(5.0))
        self.sim_speed.setToolTip("<p>Simulated seconds per second. A sweep of 120 s a leg takes 48 s at 5×.</p>")
        self.run_button = QtWidgets.QPushButton("Pause")
        self.run_button.clicked.connect(self._on_run)
        self.reset_button = QtWidgets.QPushButton("Reset")
        self.reset_button.setToolTip("<p>Back to rest at t = 0, the measured points cleared; a sweep starts "
                                     "again from its start.</p>")
        self.reset_button.clicked.connect(self.reset)
        self.clear_button = QtWidgets.QPushButton("Clear points")
        self.clear_button.setToolTip("<p>Forget the measured points (a new sweep does too).</p>")
        self.clear_button.clicked.connect(self.clear_points)
        row = FlowLayout()
        row.add(self.sim_speed, self.run_button, self.reset_button, self.clear_button)
        pv.addLayout(row)

        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.addWidget(self.preset)
        lv.addWidget(self.preset_note)
        for box in (osc, drive, branch, play):
            lv.addWidget(box)
        lv.addStretch(1)
        left_scroll = ColumnScroll(left)

        # --- centre: the response against frequency, the readout, the strip chart
        self.response = _plot("Amplitude (first harmonic)", "m", "Drive frequency", "Hz")
        self.response.setMouseEnabled(x=False, y=False)
        self.response.setToolTip(RESPONSE_TIP)
        self.legend = add_legend(self.response.getPlotItem(), offset=(10, 10), colCount=2)
        self.linear_item = pg.PlotDataItem(name="Linear spring (k₃ = 0)")
        self.backbone_item = pg.PlotDataItem(name="Backbone")
        self.stable_item = pg.PlotDataItem(name="Harmonic balance", connect="finite")
        self.unstable_item = pg.PlotDataItem(name="… unstable", connect="finite")
        self.jump_item = pg.PlotDataItem(connect="pairs")
        self.fold_item = pg.ScatterPlotItem(size=7)
        self.up_item = pg.ScatterPlotItem(size=4, name="Sweeping up")
        self.down_item = pg.ScatterPlotItem(size=4, name="Sweeping down")
        self.hold_item = pg.ScatterPlotItem(size=4, name="Held")
        self.now_item = pg.ScatterPlotItem(size=11)
        for item in (self.linear_item, self.backbone_item, self.unstable_item, self.stable_item, self.jump_item,
                     self.fold_item, self.up_item, self.down_item, self.hold_item, self.now_item):
            self.response.addItem(item)
        plot_height(self.response, 220)
        self.readout = QtWidgets.QLabel()
        self.readout.setWordWrap(True)
        self.readout.setTextFormat(QtCore.Qt.TextFormat.RichText)

        self.strip = _plot("Displacement x", "m", "Time", "s")
        self.strip.setMouseEnabled(x=False, y=False)
        self.strip.setToolTip("<p>The mass's displacement over the last 20 s of simulated time.</p>")
        self.x_item = pg.PlotDataItem()
        self.strip.addItem(self.x_item)
        plot_height(self.strip, 140)

        center = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(center)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.addWidget(self.response, 3)
        cv.addWidget(self.readout)
        cv.addWidget(self.strip, 2)

        # --- right: the phase plane and the notes
        self.phase = _plot("Velocity", "m/s", "Displacement", "m")
        self.phase.setToolTip(PHASE_TIP)
        self.orbit_item = pg.PlotDataItem()
        self.section_item = pg.ScatterPlotItem(size=3)
        self.phase.addItem(self.orbit_item)
        self.phase.addItem(self.section_item)
        self.phase_note = QtWidgets.QLabel(
            "The last few drive cycles; the dots are the Poincaré section, the state once per cycle (while the "
            "frequency is held).")
        self.phase_note.setWordWrap(True)
        mute(self.phase_note)
        phase_tab = QtWidgets.QWidget()
        ph = QtWidgets.QVBoxLayout(phase_tab)
        ph.addWidget(self.phase, 1)
        ph.addWidget(self.phase_note)
        self.notes = QtWidgets.QTextBrowser()
        self.notes.setHtml(NOTES_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(phase_tab, "Phase plane")
        self.tabs.addTab(self.notes, "Notes")

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(left_scroll)
        splitter.addWidget(center)
        splitter.addWidget(self.tabs)
        splitter.setSizes([330, 820, 550])
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        # --- loop
        self._clock = QtCore.QElapsedTimer()
        self._last = 0.0
        self._cost = 1e-6  # s of computing per simulated s (see step)
        self._target_t = 0.0
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._frame)

        self.preset.currentIndexChanged.connect(self._on_preset)
        self.apply_theme()
        self.load_preset(PRESETS[0].name)

    # ------------------------------------------------------------- inputs
    def oscillator(self) -> Duffing:
        """The oscillator the boxes describe (ValueError if they don't describe one)."""
        return Duffing(self.mass.value(), self.stiffness.value(), self.damping.value(), self.cubic.value(),
                       self.force.value())

    def load_preset(self, name: str) -> None:
        p = next(p for p in PRESETS if p.name == name)
        self._loading = True
        for box, value in ((self.mass, p.osc.mass), (self.stiffness, p.osc.stiffness),
                           (self.damping, p.osc.damping), (self.cubic, p.osc.cubic), (self.force, p.osc.force),
                           (self.freq, p.freq_hz), (self.start_hz, p.start_hz), (self.end_hz, p.end_hz),
                           (self.sweep_time, p.sweep_time)):
            box.setValue(value)
        self.preset.blockSignals(True)
        self.preset.setCurrentText(name)
        self.preset.blockSignals(False)
        self.preset_note.setText(p.note)
        self._loading = False
        d = self.sim.drive
        d.freq_hz, d.start_hz, d.end_hz, d.sweep_time = p.freq_hz, p.start_hz, p.end_hz, p.sweep_time
        self._apply_params()
        self._mode = p.mode
        self.reset()

    def frame_context(self) -> str:
        """For the frame log (frame_stats): the preset showing."""
        return self.preset.currentText()

    def _on_preset(self, index: int) -> None:
        if index < len(PRESETS):
            self.load_preset(PRESETS[index].name)

    def _on_params(self) -> None:
        if self._loading:
            return
        self._to_custom()
        self._apply_params()

    def _to_custom(self) -> None:
        self.preset.blockSignals(True)
        self.preset.setCurrentText(CUSTOM)
        self.preset.blockSignals(False)
        self.preset_note.setText("")

    def _apply_params(self) -> None:
        try:
            osc = self.oscillator()
        except ValueError as e:
            self.osc_info.setText(f"<b>{e}</b>: the page keeps the last oscillator.")
            return
        self.sim.set_system(osc)
        self.clear_points()
        self._draw_curve()
        self._show_osc_info()
        self._update_branch_buttons()
        self._show_readout()

    def _on_freq(self, value: float) -> None:
        if self._loading:
            return
        self.sim.drive.freq_hz = value
        if self.sim.drive.legs:  # typing a frequency stops the sweep there
            self.sim.drive.start(Sweep.HOLD)
            self._mode = Sweep.HOLD
        self._update_branch_buttons()
        self._show_drive_state()

    def _on_sweep_range(self) -> None:
        if self._loading:
            return
        d = self.sim.drive
        d.start_hz, d.end_hz, d.sweep_time = self.start_hz.value(), self.end_hz.value(), self.sweep_time.value()
        if d.legs:  # the sweep it was running no longer exists: hold where it is
            d.start(Sweep.HOLD)
            self._mode = Sweep.HOLD
        self._to_custom()
        self.sim.set_system(self.sim.osc)  # the step resolves the new range
        self._draw_curve()
        self._show_drive_state()

    def start_sweep(self, mode: Sweep) -> None:
        """Start a sweep from its start (clearing the measured points), or hold the frequency where it is."""
        self._mode = mode
        self.sim.drive.start(mode)
        if mode is not Sweep.HOLD:
            self.clear_points()
        self._sync_freq()
        self._update_branch_buttons()
        self._show_drive_state()
        if not self.running and not self.sim.escaped:
            self._on_run()

    def start_on_branch(self, k: int) -> None:
        """Put the mass on the k-th steady amplitude at the held frequency (0 lower, 1 middle, 2 upper)."""
        a, _ = amplitudes(self.sim.osc, self.sim.drive.freq_hz)
        if not a.size:
            return
        if self.sim.drive.legs:
            self.sim.drive.start(Sweep.HOLD)
            self._mode = Sweep.HOLD
        self.sim.start_on(float(a[k] if a.size == 3 else a[0]))
        self._target_t = self.sim.t
        self._show_drive_state()

    def _update_branch_buttons(self) -> None:
        a, _ = amplitudes(self.sim.osc, self.sim.drive.freq_hz)
        three = a.size == 3
        for b in self.branch_buttons[1:]:
            b.setEnabled(three)
        self.branch_buttons[0].setEnabled(a.size > 0)
        f = self.sim.drive.freq_hz
        if three:
            self.branch_note.setText(f"At {f:.3g} Hz: {', '.join(fmt_length(x) for x in a)} (the middle unstable).")
        elif a.size:
            self.branch_note.setText(f"At {f:.3g} Hz there is one steady amplitude, {fmt_length(a[0])}: "
                                     "hold inside the band for three.")
        else:
            self.branch_note.setText(f"At {f:.3g} Hz there is no steady response below the escape amplitude.")

    # ------------------------------------------------------------- run
    def _on_run(self) -> None:
        self.running = not self.running
        self.run_button.setText("Pause" if self.running else "Run")
        self._start_timer()

    def reset(self) -> None:
        """Back to rest at t = 0; a sweep starts again from its start."""
        self.sim.reset()
        self.sim.drive.start(self._mode)
        if self._mode is Sweep.HOLD:
            self.sim.drive.freq_hz = self.freq.value()
        self._target_t = 0.0
        self.trace.clear()
        self.clear_points()
        if not self.running:
            self._on_run()
        self._sync_freq()
        self._update_branch_buttons()
        self._show_drive_state()
        self._draw_live()

    def clear_points(self) -> None:
        self.cycles = []
        self.section = []
        self._draw_points()
        self._show_readout()

    def showEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().showEvent(event)
        self._start_timer()

    def hideEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().hideEvent(event)
        self._timer.stop()

    def _start_timer(self) -> None:
        if self.isVisible() and self.running and not self._timer.isActive():
            self._clock.restart()
            self._last = 0.0
            self._target_t = self.sim.t
            self._timer.start(FRAME_MS)

    def _frame(self) -> None:
        began = self._clock.elapsed()
        now = began / 1000.0
        dt = min(now - self._last, 0.1)
        self._last = now
        with frame_stats.frame("Duffing oscillator", lambda: self.sim.t, FRAME_MS):
            self.step(dt * float(self.sim_speed.currentData()), FRAME_BUDGET)
            settle_plots(self)
        if self.isVisible() and self.running:  # the next frame FRAME_MS after this one began, not after it ended
            self._timer.start(max(1, FRAME_MS - (self._clock.elapsed() - began)))

    def step(self, duration: float, budget: float | None = None) -> None:
        """Advance by `duration` s of simulated time and redraw.

        With a `budget` (s of computing), advance no further than it allows at the cost seen so far,
        so a fast playback falls behind rather than stalling the window.
        """
        self._target_t = min(self._target_t + duration, self.sim.t + 0.1 + duration)
        ahead = self._target_t - self.sim.t
        if budget is not None:
            ahead = min(ahead, budget / self._cost)
        if ahead > 0.5 * self.sim.step_size():
            start = time.perf_counter()
            s = self.sim.advance(ahead)
            if s.t.size:
                self._cost = 0.8 * self._cost + 0.2 * (time.perf_counter() - start) / (s.t[-1] - s.t[0] + 1e-9)
                self.trace.add(s.t, s.x, s.v)
            if s.cycles:
                self.cycles += s.cycles
                self.section += [(c.x, c.v) for c in s.cycles if c.leg == "hold"]  # at one frequency only
                del self.section[:-SECTION_POINTS]
                self._draw_points()
                self._show_readout()
                self._update_branch_buttons()
        if self.sim.escaped:
            self._target_t = self.sim.t
            if self.running:
                self._on_run()
            self._show_readout()
        self._sync_freq()
        self._show_drive_state()
        self._draw_live()

    def _sync_freq(self) -> None:
        f = round(self.sim.drive.freq_hz, 3)
        if f != self.freq.value():
            self.freq.blockSignals(True)
            self.freq.setValue(f)
            self.freq.blockSignals(False)

    # ------------------------------------------------------------- drawing
    def apply_theme(self) -> None:
        c = colors
        self.linear_item.setPen(pg.mkPen(c.grey, width=1, style=QtCore.Qt.PenStyle.DashLine))
        self.backbone_item.setPen(pg.mkPen(c.muted, width=1, style=QtCore.Qt.PenStyle.DotLine))
        self.stable_item.setPen(pg.mkPen(c.strong, width=2))
        self.unstable_item.setPen(pg.mkPen(c.strong, width=1.5, style=QtCore.Qt.PenStyle.DashLine))
        self.jump_item.setPen(pg.mkPen(c.force, width=1.5, style=QtCore.Qt.PenStyle.DashLine))
        self.fold_item.setPen(pg.mkPen(c.force, width=1.5))
        self.fold_item.setBrush(pg.mkBrush(c.background))
        for item, color in ((self.up_item, c.mass[0]), (self.down_item, c.mass[1]), (self.hold_item, c.mass[2])):
            item.setPen(None)
            item.setBrush(pg.mkBrush(color))
        self.now_item.setPen(pg.mkPen(c.force, width=2))
        self.now_item.setBrush(None)
        self.x_item.setPen(pg.mkPen(c.mass[0], width=1))
        self.orbit_item.setPen(pg.mkPen(c.mass[0], width=1))
        self.section_item.setPen(None)
        self.section_item.setBrush(pg.mkBrush(c.force))
        if self.curve is not None:
            self._draw_curve()
            self._draw_points()
            self._show_readout()

    def _x_range(self) -> tuple[float, float]:
        d = self.sim.drive
        lo = min(d.start_hz, d.end_hz, d.freq_hz)
        hi = max(d.start_hz, d.end_hz, d.freq_hz)
        pad = 0.1 * (hi - lo) if hi > lo else 0.5 * max(hi, 0.1)
        return max(lo - pad, 0.0), hi + pad

    def _draw_curve(self) -> None:
        osc = self.sim.osc
        lo, hi = self._x_range()
        self.curve = response_curve(osc, hi)
        top = 0.0
        stable, unstable = [], []
        for f, a, ok in self.curve.pieces:
            (stable if ok else unstable).append((f, a))
            inside = (f >= lo) & (f <= hi)
            if inside.any():
                top = max(top, float(a[inside].max()))
        top = top or 1e-3
        self._y_top = 1.15 * top

        def joined(pieces):
            if not pieces:
                return np.empty(0), np.empty(0)
            f = np.concatenate([np.append(p[0], np.nan) for p in pieces])
            return f, np.concatenate([np.append(p[1], np.nan) for p in pieces])

        self.stable_item.setData(*joined(stable))
        self.unstable_item.setData(*joined(unstable))
        f = np.linspace(lo, hi, 600)
        lin = linear_response(osc, f)
        self.linear_item.setData(f, np.where(lin <= 2 * self._y_top, lin, np.nan), connect="finite")
        a = np.linspace(0.0, 2 * self._y_top, 300)
        self.backbone_item.setData(osc.backbone_hz(a), a, connect="finite")
        jumps = [(fold.freq_hz, fold.amplitude, fold.freq_hz, fold.lands_at) for fold in self.curve.folds]
        self.jump_item.setData(np.ravel([(j[0], j[2]) for j in jumps]), np.ravel([(j[1], j[3]) for j in jumps]))
        self.fold_item.setData([j[0] for j in jumps], [j[1] for j in jumps])
        self.response.setXRange(lo, hi, padding=0)
        self.response.setYRange(0.0, self._y_top, padding=0)
        if osc.cubic < 0:  # in the corner the curve leans away from
            self.legend.anchor((1, 0), (1, 0), offset=(-10, 10))
        else:
            self.legend.anchor((0, 0), (0, 0), offset=(10, 10))

    def _draw_points(self) -> None:
        for item, leg in ((self.up_item, "up"), (self.down_item, "down"), (self.hold_item, "hold")):
            pts = [(c.freq_hz, c.amplitude) for c in self.cycles if c.leg == leg]
            if leg == "hold":
                pts = pts[-200:]
            item.setData(pos=np.array(pts).reshape(-1, 2))
        last = self.cycles[-1] if self.cycles else None
        self.now_item.setData(pos=np.array([[last.freq_hz, last.amplitude]]) if last else np.empty((0, 2)))
        if last is not None and last.amplitude > self._y_top and math.isfinite(last.amplitude):
            self._y_top = 1.15 * max(c.amplitude for c in self.cycles[-400:])
            self.response.setYRange(0.0, self._y_top, padding=0)
        self.section_item.setData(pos=np.array(self.section).reshape(-1, 2))

    def _draw_live(self) -> None:
        t, x, v = self.trace.since(self.sim.t - WINDOW)
        tt, xx = min_max(t, x, PLOT_POINTS)
        self.x_item.setData(tt, xx)
        if t.size:
            self.strip.setXRange(max(self.sim.t - WINDOW, 0.0), max(self.sim.t, WINDOW), padding=0)
        f = self.sim.drive.freq_hz
        span = PHASE_CYCLES / f if f > 0 else 2.0
        t, x, v = self.trace.since(self.sim.t - span)
        self.orbit_item.setData(x, v)

    def _show_osc_info(self) -> None:
        osc = self.sim.osc
        parts = []
        if osc.stiffness > 0:
            zeta = osc.damping / (2 * math.sqrt(osc.stiffness * osc.mass))
            parts.append(f"Small swings: f<sub>n</sub> = {osc.natural_hz:.3g} Hz, ζ = {100 * zeta:.2g} %.")
        else:
            parts.append("No linear spring: the frequency depends on the swing alone.")
        if osc.cubic < 0:
            parts.append(f"Escape amplitude {fmt_length(osc.escape)}.")
        elif osc.cubic > 0 and osc.stiffness > 0 and math.isfinite(self.curve.tip):
            ratio = 0.75 * osc.cubic * self.curve.tip**2 / osc.stiffness
            parts.append(f"At the peak ({fmt_length(self.curve.tip)}) the cubic adds {100 * ratio:.3g} % "
                         "to the spring's stiffness.")
        self.osc_info.setText(" ".join(parts))

    def _show_drive_state(self) -> None:
        d = self.sim.drive
        text = {"up": "Sweeping up", "down": "Sweeping down", "hold": "Holding"}[d.leg]
        self.drive_state.setText(f"{text} at {d.freq_hz:.3f} Hz, t = {self.sim.t:.1f} s")

    def _show_readout(self) -> None:
        if self.curve is None:
            return
        osc = self.sim.osc
        lines = []
        folds = sorted(self.curve.folds, key=lambda f: f.freq_hz)
        if len(folds) >= 2:
            lo, hi = folds[0], folds[-1]
            lines.append(f"Harmonic balance: two stable amplitudes between {lo.freq_hz:.3g} and {hi.freq_hz:.3g} Hz. "
                         f"Sweeping up, the {'upper' if hi.down else 'lower'} branch ends at {hi.freq_hz:.3g} Hz "
                         f"({fmt_length(hi.amplitude)}) and {'drops' if hi.down else 'jumps'} to "
                         f"{fmt_length(hi.lands_at)}; sweeping down, the {'upper' if lo.down else 'lower'} ends at "
                         f"{lo.freq_hz:.3g} Hz ({fmt_length(lo.amplitude)}) and "
                         f"{'drops' if lo.down else 'jumps'} to {fmt_length(lo.lands_at)}.")
            measured = []
            for leg, word in (("up", "going up"), ("down", "going down")):
                f = measured_jump(self.cycles, leg)
                if f is not None:
                    measured.append(f"{f:.3g} Hz {word}")
            if measured:
                lines.append("Measured: the jump at " + " and ".join(measured) + ".")
        elif folds:
            fold = folds[0]
            lines.append(f"Harmonic balance: a branch ends at {fold.freq_hz:.3g} Hz ({fmt_length(fold.amplitude)}) "
                         f"and {'drops' if fold.down else 'jumps'} to {fmt_length(fold.lands_at)}; the other runs "
                         "into the escape amplitude." if osc.cubic < 0 else
                         f"Harmonic balance: a branch ends at {fold.freq_hz:.3g} Hz; the other runs off the top "
                         "(no damping to bound it).")
        else:
            lines.append("Harmonic balance: one steady amplitude at each frequency, so no jump: sweeps up and down "
                         "agree.")
        if self.sim.escaped:
            lines.append(f"<b>The mass went over the potential's hilltop (±{fmt_length(osc.escape)}) and ran "
                         "away.</b> Reset to start again.")
        self.readout.setText("<br>".join(lines))
