"""Shock response spectrum page: a pulse on the base of a bank of single-degree-of-freedom oscillators.

Left: the pulse (a classical shape, or a measured record: the absolute acceleration of a mass on
the chain under a base half-sine, or a pyrotechnic-like transient, with noise and an offset), the
spectrum's Q and the oscillator looked at, and things to try. Centre: the pulse, its velocity, and
the SRS (maximax, primary and residual, positive and negative) with the chain's modes marked.
Right: the oscillator picked on the spectrum, its response against the pulse, and the notes.
Follows the chain from the Simulation page (friction left out: all of this is linear).
"""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.modal import TWO_PI, ModalResult, modal_analysis
from ..core.model import ChainSystem
from ..core.random_vibration import G, linear
from ..core.shock import (
    Pulse,
    Record,
    Shape,
    Spectrum,
    bank,
    oscillator,
    record,
    sample_rate,
    srs,
)
from .axes import log_axes
from .layouts import ColumnScroll, group, plot_height
from .panels import spin
from .style import MAX_DOF, colors
from .theming import add_legend, mute

RECOMPUTE_MS = 150  # the parameters settle this long before the spectrum is worked out again
DEFAULT_Q = 10.0
TAIL_PERIODS = 2.0  # of the picked oscillator, shown after the record

TIPS = {
    "shape": "<p>The base's acceleration. <b>Classical</b> pulses, as shock tests specify them: half-sine, "
             "haversine, the two sawtooths, a trapezoid, a square. <b>Measured at a mass</b>: the absolute "
             "acceleration of a mass on the chain, its base given a half-sine (the shock a component mounted "
             "there sees). <b>Measured transient</b>: damped sines at scattered frequencies, as a pyrotechnic "
             "shock is.</p>",
    "peak": "<p>The pulse's peak acceleration, in g (a measured transient is scaled to it; a measured mass's "
            "record comes from a base half-sine this high).</p>",
    "duration": "<p>The pulse's duration T (a measured transient's length; a measured mass's base half-sine). "
                "The spectrum's shape depends on f T.</p>",
    "mass": "<p>The mass whose acceleration is the measured record.</p>",
    "noise": "<p>White noise added to a measured record, its RMS a fraction of the peak, as an accelerometer's "
             "and its amplifier's.</p>",
    "offset": "<p>A constant added to a measured record, a fraction of the peak: an accelerometer's zero shift, "
              "common in pyrotechnic shocks. Integrated, it is a velocity change the shock didn't have.</p>",
    "again": "<p>The same record with other noise.</p>",
    "q": "<p>The oscillators' Q = 1/2ζ: the same for every one. Shock specifications use Q = 10 (ζ = 5 %).</p>",
    "split": "<p>Show the <b>primary</b> spectrum (the peaks while the pulse lasts) and the <b>residual</b> one "
             "(after it, the free decay), and the <b>positive</b> and <b>negative</b> peaks apart. The maximax "
             "is the largest of them. For a measured record, primary means within the record.</p>",
    "f": "<p>The oscillator shown on the right. Drag the line on the spectrum, or click it.</p>",
}
PULSE_TIP = "<p>The base's acceleration (for a measured mass, the base's half-sine is grey).</p>"
VELOCITY_TIP = ("<p>The base's velocity, ∫a dt. The change Δv over the pulse decides the spectrum's low end: a soft "
                "oscillator takes the pulse as an impulse.</p>")
SRS_TIP = (
    "<p>The shock response spectrum: each oscillator's largest absolute acceleration, against its natural "
    "frequency. Dashed: the pulse's peak (the stiff oscillators' asymptote) and 2π f Δv (the soft ones'). "
    "Dotted: the chain's modes.</p>")
OSC_TIP = "<p>The picked oscillator's absolute acceleration, the pulse behind it; the dot marks its largest value.</p>"
DISP_TIP = "<p>The oscillator's displacement relative to the base: its spring's stretch.</p>"

# Things to try: (name, what it sets, note). The duration is set from the chain (`duration_for`).
TRIALS = (
    ("Half-sine", {},
     "A 10 g half-sine. At Q = 10 the spectrum peaks at 1.65 × the pulse's peak, at f T = 0.83: an oscillator "
     "whose period is a little longer than the pulse. Stiff oscillators follow the pulse (1 ×); soft ones take it as "
     "an impulse and ring with ω Δv, the dashed line at the left."),
    ("Square", {"shape": Shape.SQUARE},
     "A square pulse: a step up and one down. Every stiff oscillator overshoots the step, to 1.85 × at Q = 10 "
     "(2 × undamped), so the spectrum stays high across the top. Its Δv is the largest of the pulses (the area "
     "A T), so it is highest at the left too."),
    ("Terminal-peak sawtooth", {"shape": Shape.TERMINAL_PEAK},
     "Rising slowly, dropping at once: the drop makes stiff oscillators ring the other way, so above f T ≈ 2 "
     "the negative spectrum is 0.85–0.86 × the positive one. Tests like it for that: a part is loaded both ways."),
    ("Haversine", {"shape": Shape.HAVERSINE},
     "A smooth pulse: no corner, so stiff oscillators don't ring and the negative spectrum at the top is "
     "nearly nothing. Its peak, 1.60 × at f T = 1.0, is close to the half-sine's."),
    ("Light damping, Q = 50", {"q": 50.0},
     "The half-sine at Q = 50: the spectrum hardly changes (1.74 × at the peak against 1.65 ×). A shock is over "
     "before damping has time to act on the primary peak, unlike a steady sine, whose response grows with Q."),
    ("A component on the last mass", {"shape": Shape.AT_A_MASS},
     "The last mass's acceleration with the base given the half-sine: it rings at the chain's modes, and its "
     "spectrum peaks there, 3.3 × the record's own peak at mode 1. A component tuned near the chain's mode "
     "would be hit far harder than the record's peak suggests."),
    ("Pyrotechnic transient", {"shape": Shape.TRANSIENT, "noise": 0.02},
     "A burst of damped sines with a little noise: the spectrum peaks among its frequencies, at 4.5 × the "
     "record's peak (f T ≈ 38). It ends at rest, with no net Δv, so the soft oscillators at the left feel only "
     "their dampers' push: at f T = 0.02 the spectrum is 10⁻⁴ of the peak."),
    ("Zero shift", {"shape": Shape.TRANSIENT, "noise": 0.02, "offset": 0.01},
     "The same record with a 1 % offset, an accelerometer's zero shift: integrated over the record it is a "
     "velocity change the shock didn't have, and the spectrum's low end climbs to the 2π f Δv line, ten times "
     "what it was at f T = 0.02. A measured spectrum whose low end follows that line is suspect."),
)

NOTES_HTML = """
<h3>The shock response spectrum</h3>
<p>A shock is a short, strong acceleration of a structure's base: a drop, a collision, a stage separation.
Its time history alone says little about what it does. What matters is what it does to the things mounted on
the structure, and they are, to a first approximation, single-degree-of-freedom oscillators: a mass on a
spring and damper, natural frequency f, damping ζ. The <b>shock response spectrum</b> (SRS) is the largest
absolute acceleration of such an oscillator on that base, against f, all at one Q = 1/2ζ (10 by custom).
A component whose natural frequency is f sees that peak; its spring carries that peak times its mass.</p>
<p>In the coordinate z relative to the base, z'' + 2ζω z' + ω² z = −a(t), and the oscillator's absolute
acceleration is −2ζω z' − ω² z. Each is stepped here by the same exact update as the chain (the input linear
between samples), its peaks refined between samples.</p>

<h3>Three regions</h3>
<ul>
<li><b>Stiff oscillators</b> (f T ≫ 1) follow the pulse: the spectrum tends to the pulse's peak. A pulse that
jumps (a square, an initial-peak sawtooth) makes them overshoot, to nearly twice it.</li>
<li><b>Near f T ≈ 1</b> the oscillator amplifies: 1.65 × at Q = 10 for a half-sine.</li>
<li><b>Soft oscillators</b> (f T ≪ 1) barely move while the pulse lasts. They take it as an impulse, a
change of velocity Δv, and ring with amplitude ω Δv: the spectrum rises as 2π f Δv. Only the pulse's area
matters there, not its shape. A transient with no net Δv is far lower there: only the damper's push, 2ζω
times the base's velocity, reaches a soft oscillator.</li>
</ul>

<h3>Primary, residual, positive, negative</h3>
<p>The <b>primary</b> spectrum is the largest response while the pulse lasts, the <b>residual</b> one the
free decay after it (soft oscillators peak there); each has a <b>positive</b> and a <b>negative</b> peak. The
<b>maximax</b> is the largest of the four. Pulses with the same maximax can load a part very differently one
way and the other: a terminal-peak sawtooth's drop rings stiff oscillators backwards, so its negative spectrum
is nearly its positive one, which is why tests use it.</p>

<h3>Damping</h3>
<p>The primary peak barely depends on Q: the shock is over before the damping has acted. That is unlike a
steady sine, whose resonant response is Q times larger. The residual decays with ζ, but its first peak, which
the spectrum counts, hardly does.</p>

<h3>Measured shocks</h3>
<p>A component mounted on a structure sees the structure's response, not the base's pulse: here a mass on the
chain, ringing at the chain's modes. Its spectrum, an in-structure shock spectrum, peaks at those modes. A
pyrotechnic shock is a burst of high-frequency ringing with almost no net velocity change, and its spectrum
falls steeply below its frequencies. An accelerometer's <b>zero shift</b>, a small offset after the shock,
integrates into a large false Δv and lifts the low end: a measured spectrum is checked for it by its slope
there.</p>
"""


def _plot(left: str, left_units: str | None, bottom: str, bottom_units: str | None, log: bool = False
          ) -> pg.PlotWidget:
    p = pg.PlotWidget(axisItems=log_axes()) if log else pg.PlotWidget()
    p.setMenuEnabled(False)
    p.hideButtons()
    p.showGrid(x=True, y=True, alpha=0.15)
    p.setLabel("left", left, units=left_units)
    p.setLabel("bottom", bottom, units=bottom_units)
    return p


def _form_row(form: QtWidgets.QFormLayout, text: str, box: QtWidgets.QWidget, tip: str) -> QtWidgets.QLabel:
    box.setToolTip(tip)
    label = QtWidgets.QLabel(text)
    label.setToolTip(tip)
    form.addRow(label, box)
    return label


def duration_for(result: ModalResult) -> float:
    """A pulse that spreads the chain's modes across the spectrum: 0.6 of its fastest mode's period, rounded to
    1, 2 or 5 ms times a power of ten."""
    top = max((m.fn_hz for m in result.modes if m.fn_hz > 0), default=6.0)
    return _nice(0.6 / top)


def _nice(x: float) -> float:
    e = 10.0 ** math.floor(math.log10(x))
    return min((m * e for m in (1, 2, 5, 10)), key=lambda v: abs(math.log(v / x)))


class ShockPage(QtWidgets.QWidget):
    """Controls on the left, the pulse and its spectrum in the centre, the picked oscillator and notes on the right."""

    dof_requested = QtCore.Signal(int)  # the user changed N here; the main window owns the system
    edit_parameters = QtCore.Signal()  # the user wants the Simulation page's parameter panel

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.system: ChainSystem | None = None
        self.result: ModalResult | None = None
        self.record: Record | None = None
        self.spectrum: Spectrum | None = None
        self.seed = 1
        self._loading = False
        self._dirty = False
        self._rerecord = True
        self._recompute = QtCore.QTimer(self)
        self._recompute.setSingleShot(True)
        self._recompute.setInterval(RECOMPUTE_MS)
        self._recompute.timeout.connect(self.recompute)

        # --- left: the pulse, the spectrum, things to try
        box = QtWidgets.QGroupBox("Pulse")
        form = QtWidgets.QFormLayout(box)
        self.dof = QtWidgets.QSpinBox()
        self.dof.setRange(1, MAX_DOF)
        self.dof.setToolTip("Same chain as the Simulation page; changing it here changes it there too.")
        self.dof.valueChanged.connect(self.dof_requested)
        form.addRow("Number of masses:", self.dof)
        self.shape = QtWidgets.QComboBox()
        for s in Shape:
            self.shape.addItem(s.value, s)
        self.shape.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        _form_row(form, "Shape:", self.shape, TIPS["shape"])
        self.peak = spin(0.001, 1e5, 10.0, 3, " g")
        _form_row(form, "Peak:", self.peak, TIPS["peak"])
        self.duration = spin(0.01, 1e5, 100.0, 2, " ms")
        _form_row(form, "Duration:", self.duration, TIPS["duration"])
        self.mass = QtWidgets.QComboBox()
        self.mass_label = _form_row(form, "Measured on:", self.mass, TIPS["mass"])
        self.noise = spin(0.0, 50.0, 0.0, 2, " %")
        self.noise_label = _form_row(form, "Noise:", self.noise, TIPS["noise"])
        self.offset = spin(-50.0, 50.0, 0.0, 2, " %")
        self.offset_label = _form_row(form, "Offset:", self.offset, TIPS["offset"])
        self.again = QtWidgets.QPushButton("New noise")
        self.again.setToolTip(TIPS["again"])
        form.addRow(self.again)
        self.pulse_info = QtWidgets.QLabel()
        self.pulse_info.setWordWrap(True)
        mute(self.pulse_info)
        form.addRow(self.pulse_info)
        link = QtWidgets.QLabel("Masses, springs and dampers are edited on the <a href='#sim'>Simulation page</a>.")
        mute(link)
        link.setWordWrap(True)
        link.linkActivated.connect(lambda _: self.edit_parameters.emit())
        form.addRow(link)

        spec_box = QtWidgets.QGroupBox("Spectrum")
        form = QtWidgets.QFormLayout(spec_box)
        self.q = spin(1.0, 500.0, DEFAULT_Q, 1, "")
        _form_row(form, "Q:", self.q, TIPS["q"])
        self.split_time = QtWidgets.QCheckBox("Primary and residual")
        self.split_time.setToolTip(TIPS["split"])
        self.split_sign = QtWidgets.QCheckBox("Positive and negative")
        self.split_sign.setToolTip(TIPS["split"])
        form.addRow(group(self.split_time))
        form.addRow(group(self.split_sign))
        self.f_pick = spin(1e-4, 1e7, 1.0, 3, " Hz")
        _form_row(form, "Oscillator:", self.f_pick, TIPS["f"])

        trials = QtWidgets.QGroupBox("Things to try")
        tv = QtWidgets.QVBoxLayout(trials)
        self.trial = QtWidgets.QComboBox()
        self.trial.addItems([name for name, _, _ in TRIALS])
        self.trial.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.trial.setToolTip("<p>Pulses that show something about shock spectra, the duration set from the "
                              "chain as it is now.</p>")
        tv.addWidget(self.trial)
        self.trial_note = QtWidgets.QLabel()
        self.trial_note.setWordWrap(True)
        mute(self.trial_note)
        tv.addWidget(self.trial_note)

        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        for b in (box, spec_box, trials):
            lv.addWidget(b)
        lv.addStretch(1)
        left_scroll = ColumnScroll(left)

        # --- centre: the pulse, its velocity, the spectrum
        self.pulse_plot = _plot("Base acceleration", None, "Time", "s")
        self.pulse_plot.setToolTip(PULSE_TIP)
        plot_height(self.pulse_plot, 110)
        self.velocity_plot = _plot("v (m/s)", None, "Time", "s")
        self.velocity_plot.setToolTip(VELOCITY_TIP)
        plot_height(self.velocity_plot, 90)
        self.velocity_plot.getAxis("left").enableAutoSIPrefix(False)
        self.velocity_plot.setXLink(self.pulse_plot)
        self.srs_plot = _plot("Peak acceleration (g)", None, "Natural frequency", "Hz", log=True)
        self.srs_plot.setLogMode(x=True, y=True)
        self.srs_plot.getAxis("left").enableAutoSIPrefix(False)
        self.srs_plot.setToolTip(SRS_TIP)
        plot_height(self.srs_plot, 220)
        self.srs_legend = add_legend(self.srs_plot.getPlotItem(), offset=(-10, -10))
        self.pick_line = pg.InfiniteLine(0.0, angle=90, movable=True)
        self.pick_line.setZValue(10)
        self.srs_plot.addItem(self.pick_line)
        self.readout = QtWidgets.QLabel()
        self.readout.setWordWrap(True)
        self.readout.setTextFormat(QtCore.Qt.TextFormat.RichText)
        center = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(center)
        cv.addWidget(self.pulse_plot, 2)
        cv.addWidget(self.velocity_plot, 1)
        cv.addWidget(self.srs_plot, 5)
        cv.addWidget(self.readout)

        # --- right: the oscillator, the notes
        self.osc_plot = _plot("Absolute acceleration (g)", None, "Time", "s")
        self.osc_plot.setToolTip(OSC_TIP)
        plot_height(self.osc_plot, 160)
        self.osc_legend = add_legend(self.osc_plot.getPlotItem(), offset=(-10, 10))
        self.disp_plot = _plot("Relative displacement", "m", "Time", "s")
        self.disp_plot.setToolTip(DISP_TIP)
        plot_height(self.disp_plot, 110)
        self.disp_plot.setXLink(self.osc_plot)
        self.osc_note = QtWidgets.QLabel()
        self.osc_note.setWordWrap(True)
        self.osc_note.setTextFormat(QtCore.Qt.TextFormat.RichText)
        osc_tab = QtWidgets.QWidget()
        ov = QtWidgets.QVBoxLayout(osc_tab)
        ov.addWidget(self.osc_plot, 3)
        ov.addWidget(self.disp_plot, 2)
        ov.addWidget(self.osc_note)
        self.osc_tab = osc_tab

        self.notes = QtWidgets.QTextBrowser()
        self.notes.setHtml(NOTES_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(osc_tab, "Oscillator")
        self.tabs.addTab(self.notes, "Notes")

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(left_scroll)
        splitter.addWidget(center)
        splitter.addWidget(self.tabs)
        splitter.setSizes([340, 810, 550])
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        # --- wiring
        self.shape.currentIndexChanged.connect(self._on_pulse)
        self.mass.currentIndexChanged.connect(self._on_pulse)
        for b in (self.peak, self.duration, self.noise, self.offset):
            b.valueChanged.connect(self._on_pulse)
        self.q.valueChanged.connect(self._on_spectrum)
        for c in (self.split_time, self.split_sign):
            c.toggled.connect(lambda _: self._draw_srs())
        self.again.clicked.connect(self._on_again)
        self.trial.currentIndexChanged.connect(lambda _: self.apply_trial())
        self.pick_line.sigPositionChanged.connect(self._on_line)
        self.f_pick.valueChanged.connect(self._on_pick)
        self.srs_plot.scene().sigMouseClicked.connect(self._on_click)

    # ------------------------------------------------------------- inputs
    def set_system(self, system: ChainSystem, result: ModalResult | None = None) -> None:
        """New chain parameters: the trial's settings follow its modes, and it's worked out again (now if shown)."""
        n_changed = self.system is None or system.n != self.system.n
        self.system = linear(system)
        self.result = modal_analysis(self.system) if result is None or system.friction.any() else result
        self._loading = True
        self.dof.blockSignals(True)
        self.dof.setValue(system.n)
        self.dof.blockSignals(False)
        if n_changed:
            mass = self.mass.currentIndex()
            self.mass.clear()
            self.mass.addItems([f"m{i + 1}" for i in range(system.n)])
            self.mass.setCurrentIndex(mass if 0 <= mass < system.n else system.n - 1)
        self._loading = False
        self.apply_trial()

    def apply_trial(self) -> None:
        """Set the pulse the chosen trial asks for, its duration from the chain as it is; the seed goes back to 1
        so the notes' numbers hold."""
        if self.result is None:
            return
        _, changes, note = TRIALS[self.trial.currentIndex()]
        self._loading = True
        self.shape.setCurrentIndex(self.shape.findData(changes.get("shape", Shape.HALF_SINE)))
        self.peak.setValue(10.0)
        self.duration.setValue(1e3 * duration_for(self.result))
        self.mass.setCurrentIndex(self.system.n - 1)
        self.noise.setValue(100.0 * changes.get("noise", 0.0))
        self.offset.setValue(100.0 * changes.get("offset", 0.0))
        self.q.setValue(changes.get("q", DEFAULT_Q))
        self.seed = 1
        self._loading = False
        self._pick_peak = True
        self._show_controls()
        self.trial_note.setText(note)
        self._schedule(rerecord=True)

    def pulse(self) -> Pulse:
        return Pulse(self.shape.currentData(), self.peak.value(), 1e-3 * self.duration.value(),
                     max(self.mass.currentIndex(), 0), 0.01 * self.noise.value(), 0.01 * self.offset.value(),
                     self.seed)

    @property
    def zeta(self) -> float:
        return 1.0 / (2.0 * self.q.value())

    def _show_controls(self) -> None:
        shape = self.shape.currentData()
        for w in (self.mass, self.mass_label):
            w.setVisible(shape is Shape.AT_A_MASS)
        for w in (self.noise, self.noise_label, self.offset, self.offset_label, self.again):
            w.setVisible(shape.measured)
        self.again.setEnabled(self.noise.value() > 0)

    def _on_pulse(self, *_args) -> None:
        if self._loading:
            return
        if self.sender() is self.shape:
            self._pick_peak = True
        self._show_controls()
        self._schedule(rerecord=True)

    def _on_spectrum(self, *_args) -> None:
        if not self._loading:
            self._schedule(rerecord=False)

    def _on_again(self) -> None:
        self.seed += 1
        self._schedule(rerecord=True, now=True)

    def _schedule(self, rerecord: bool, now: bool = False) -> None:
        self._rerecord |= rerecord
        if not self.isVisible():
            self._dirty = True
            return
        if now:
            self.recompute()
        else:
            self._recompute.start()

    def showEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().showEvent(event)
        if self._dirty:
            self.recompute()

    # the picked oscillator: the line, the spin box and a click on the spectrum
    def _on_line(self) -> None:
        f = 10.0 ** self.pick_line.value()
        if abs(f / self.f_pick.value() - 1.0) > 1e-6:
            self.f_pick.setValue(f)

    def _on_pick(self, f: float) -> None:
        if f > 0 and abs(10.0 ** self.pick_line.value() / f - 1.0) > 1e-6:
            self.pick_line.blockSignals(True)
            self.pick_line.setValue(math.log10(f))
            self.pick_line.blockSignals(False)
        self._draw_oscillator()

    def _on_click(self, event) -> None:
        vb = self.srs_plot.getPlotItem().getViewBox()
        if not vb.sceneBoundingRect().contains(event.scenePos()):
            return
        self.f_pick.setValue(10.0 ** vb.mapSceneToView(event.scenePos()).x())

    # ------------------------------------------------------------- computing
    def recompute(self) -> None:
        """Record the pulse again if asked for (or stale), then its spectrum, and draw."""
        self._recompute.stop()
        self._dirty = False
        if self.system is None:
            return
        pulse = self.pulse()
        self.freqs = bank(pulse, self.result)
        if self._rerecord or self.record is None:
            try:
                self.record = record(pulse, sample_rate(pulse, self.freqs[-1]), self.system, self.result)
            except ValueError as e:
                self.record = None
                self.readout.setText(f"<b>{e}</b>.")
                return
            self._rerecord = False
        self.spectrum = srs(self.record, self.freqs, self.zeta)
        if getattr(self, "_pick_peak", True):
            self._pick_peak = False
            if pulse.shape is Shape.AT_A_MASS:
                f = min(m.fn_hz for m in self.result.modes if m.fn_hz > 0)
            else:
                f = float(self.freqs[np.argmax(self.spectrum.maximax)])
            self.f_pick.blockSignals(True)
            self.f_pick.setValue(f)
            self.f_pick.blockSignals(False)
        self.redraw()

    # ------------------------------------------------------------- drawing
    def apply_theme(self) -> None:
        if self.spectrum is not None:
            self.redraw()

    def redraw(self) -> None:
        self._draw_pulse()
        self._draw_srs()
        self._draw_oscillator()

    def _draw_pulse(self) -> None:
        rec = self.record
        t = rec.t
        p = self.pulse_plot
        p.clear()
        p.setLabel("left", "Base acceleration (g)" if rec.pulse.shape is not Shape.AT_A_MASS
                   else f"m{rec.pulse.mass % self.system.n + 1}'s acceleration (g)")
        if rec.pulse.shape is Shape.AT_A_MASS:
            k = int(round(rec.pulse.duration * rec.fs))
            base = np.zeros(t.size)
            base[:k + 1] = rec.pulse.peak * np.sin(np.pi * np.arange(k + 1) / k)
            p.plot(t, base, pen=pg.mkPen(colors.grey, width=1))
        p.plot(np.concatenate([[t[0] - rec.h], t, [t[-1] + rec.h]]),
               np.concatenate([[0.0], rec.a / G, [0.0]]), pen=pg.mkPen(colors.force, width=2))
        p.addItem(pg.InfiniteLine(0.0, angle=0, pen=pg.mkPen(colors.faint, width=1)))
        v = rec.velocity()
        p = self.velocity_plot
        p.clear()
        p.setLabel("left", "v (m/s)")
        p.plot(t, v, pen=pg.mkPen(colors.mass[0], width=1.5))
        p.addItem(pg.InfiniteLine(0.0, angle=0, pen=pg.mkPen(colors.faint, width=1)))
        self.pulse_plot.setXRange(-0.02 * rec.duration, 1.02 * rec.duration, padding=0.0)
        self.pulse_info.setText(
            f"Δv = {rec.velocity_change:.4g} m/s over the record ({1e3 * rec.duration:.4g} ms); "
            f"{rec.a.size:,} samples at {rec.fs:.4g} Hz; {self.freqs.size} oscillators, {self.freqs[0]:.3g} to "
            f"{self.freqs[-1]:.4g} Hz.")

    def _draw_srs(self) -> None:
        sp, rec = self.spectrum, self.record
        if sp is None or rec is None:
            return
        f = sp.f
        p = self.srs_plot
        for item in list(p.getPlotItem().items):
            if item is not self.pick_line:
                p.removeItem(item)
        self.srs_legend.clear()
        floor = 1e-6 * float(np.max(sp.maximax))

        def curve(values, color, width=1.5, style=QtCore.Qt.PenStyle.SolidLine, name=None):
            item = p.plot(f, np.maximum(values / G, floor / G), pen=pg.mkPen(color, width=width, style=style))
            if name:
                self.srs_legend.addItem(item, name)

        dash, dot = QtCore.Qt.PenStyle.DashLine, QtCore.Qt.PenStyle.DotLine
        if self.split_time.isChecked() and self.split_sign.isChecked():
            curve(sp.primary[0], colors.mass[0], name="Primary +")
            curve(sp.primary[1], colors.mass[0], style=dash, name="Primary −")
            curve(sp.residual[0], colors.mass[1], name="Residual +")
            curve(sp.residual[1], colors.mass[1], style=dash, name="Residual −")
        elif self.split_time.isChecked():
            curve(sp.primary.max(axis=0), colors.mass[0], name="Primary")
            curve(sp.residual.max(axis=0), colors.mass[1], name="Residual")
        elif self.split_sign.isChecked():
            curve(sp.positive, colors.mass[0], name="Positive")
            curve(sp.negative, colors.mass[1], style=dash, name="Negative")
        curve(sp.maximax, colors.strong, width=2.5, name=f"Maximax, Q = {sp.q:.3g}")
        peak = float(np.max(np.abs(rec.a)))
        line = pg.mkPen(colors.muted, width=1, style=dash)
        p.plot(f[[0, -1]], np.full(2, peak / G), pen=line)
        dv = abs(rec.velocity_change)
        if dv > 0:
            low = TWO_PI * f * dv
            keep = low < 3.0 * float(np.max(sp.maximax))
            if keep.any():
                p.plot(f[keep], low[keep] / G, pen=line)
        for i, m in enumerate(self.result.modes):
            if m.fn_hz > 0:
                p.addItem(pg.InfiniteLine(math.log10(m.fn_hz), angle=90, pen=pg.mkPen(
                    colors.mode[i % len(colors.mode)], width=1, style=dot)))
        self.pick_line.setPen(pg.mkPen(colors.force, width=2))
        self.pick_line.blockSignals(True)
        self.pick_line.setValue(math.log10(self.f_pick.value()))
        self.pick_line.blockSignals(False)
        top = float(np.max(sp.maximax)) / G
        p.setXRange(math.log10(f[0]), math.log10(f[-1]), padding=0.0)
        bottom = min(max(float(np.min(sp.maximax)) / G / 1.5, top * 1e-5), top * 1e-3)
        p.setYRange(math.log10(bottom), math.log10(top) + 0.3, padding=0.0)

        k = int(np.argmax(sp.maximax))
        T = rec.pulse.duration
        modes = ", ".join(f"{m.fn_hz:.3g} Hz {sp.at(m.fn_hz) / G:.3g} g" for m in self.result.modes if m.fn_hz > 0)
        self.readout.setText(
            f"<b>Maximax</b> {sp.maximax[k] / G:.3g} g at {f[k]:.3g} Hz (f T = {f[k] * T:.2g}), "
            f"{sp.maximax[k] / peak:.3g} × the record's peak {peak / G:.3g} g. At the chain's modes: {modes}.")

    def _draw_oscillator(self) -> None:
        rec = self.record
        if rec is None or self.spectrum is None:
            return
        f = self.f_pick.value()
        tail = max(TAIL_PERIODS / f, 0.25 * rec.duration)
        osc = oscillator(rec, f, self.zeta, min(tail, 4.0 * rec.duration + TAIL_PERIODS / f))
        p = self.osc_plot
        p.clear()
        self.osc_legend.clear()
        a = np.zeros(osc.t.size)
        a[:rec.a.size] = rec.a
        base = p.plot(osc.t, a / G, pen=pg.mkPen(colors.grey, width=1))
        y = p.plot(osc.t, osc.acceleration / G, pen=pg.mkPen(colors.mass[0], width=2))
        self.osc_legend.addItem(base, "The pulse")
        self.osc_legend.addItem(y, f"The oscillator, {f:.3g} Hz")
        k = int(np.argmax(np.abs(osc.acceleration)))
        p.plot([osc.t[k]], [osc.acceleration[k] / G], pen=None, symbol="o", symbolSize=8,
               symbolBrush=colors.force, symbolPen=None)
        end = pg.mkPen(colors.faint, width=1, style=QtCore.Qt.PenStyle.DashLine)
        p.addItem(pg.InfiniteLine(osc.end, angle=90, pen=end))
        p.addItem(pg.InfiniteLine(0.0, angle=0, pen=pg.mkPen(colors.faint, width=1)))
        p.setXRange(0.0, osc.t[-1], padding=0.01)
        d = self.disp_plot
        d.clear()
        d.plot(osc.t, osc.displacement, pen=pg.mkPen(colors.mass[1], width=1.5))
        d.addItem(pg.InfiniteLine(osc.end, angle=90, pen=end))
        d.addItem(pg.InfiniteLine(0.0, angle=0, pen=pg.mkPen(colors.faint, width=1)))

        during = osc.t <= osc.end
        y = osc.acceleration / G
        when = "record" if rec.end is None else "pulse"
        self.osc_note.setText(
            f"f = {f:.4g} Hz, f T = {f * rec.pulse.duration:.3g}, Q = {self.q.value():.3g}. During the {when}: "
            f"+{_positive(y[during].max()):.3g} / −{_positive(-y[during].min()):.3g} g; after it (residual): "
            f"+{_positive(y[~during].max()):.3g} / −{_positive(-y[~during].min()):.3g} g. The dot marks the largest, "
            f"{abs(y[k]):.3g} g. Its spring stretches up to {_length(np.max(np.abs(osc.displacement)))}.")


def _length(x: float) -> str:
    return f"{x:.3g} m" if x >= 1.0 else f"{1e3 * x:.3g} mm"


def _positive(x: float) -> float:
    return max(float(x), 0.0) + 0.0  # no "−-0"
