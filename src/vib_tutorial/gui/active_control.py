"""Active control page: velocity feedback on the Simulation page's chain, collocated or not, against a TMD.

Left: the sensor and actuator masses, the gain and the actuator's lag, the disturbance and response masses, and a
tuned mass damper to compare. Centre: the receptance without control, with it, and with the TMD, and each mode's
damping. Right: the root locus as the gain rises, each mode's damping against the gain, skyhook isolation, and the
Theory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.active_control import (
    Controller,
    closed_loop,
    frf,
    max_real,
    modal_damping_estimate,
    oscillating_poles,
    skyhook,
    tmd_for,
    tracked_damping,
    undamped_modes,
)
from ..core.model import ChainSystem
from . import settle_plots
from .lesson import (
    Form,
    TrialBox,
    column,
    connect_all,
    mark,
    marker,
    note_label,
    plot,
    readout_label,
    set_quietly,
    stack,
    theory,
    three_panes,
    trials_html,
)
from .links import before, takeaways
from .style import MAX_DOF, colors
from .theming import add_legend, mute

TAB_LOCUS, TAB_GAIN, TAB_SKYHOOK = range(3)
GAINS = np.concatenate([[0.0], np.geomspace(0.05, 2000.0, 160)])  # N·s/m, for the locus and the damping plot


@dataclass(frozen=True)
class ControlTrial:
    name: str
    note: str
    sensor: int = 4  # 1-based, as shown
    actuator: int = 4
    gain: float = 20.0
    lag_ms: float = 0.0
    tmd: bool = False
    tab: int = TAB_GAIN
    zeta_passive: float = 0.1
    zeta_sky: float = 0.0


TRIALS = (
    ControlTrial(
        "Velocity feedback at the tip",
        "The uniform chain (four 1 kg masses, 400 N/m), a velocity sensor and a force actuator both on m4, "
        "u = −g ẋ₄ with g = 20 N·s/m: an active damper to the ground. Every mode gains damping, mode 1 most "
        "(ζ from 1.7 % to 72 %), and the tip's resonance peak all but disappears. For light gain each mode gains "
        "Δζ = gφ<sub>r</sub>(4)²/2ω<sub>r</sub>, so the modes that move the tip most are damped most.",
        tab=TAB_GAIN),
    ControlTrial(
        "Too much gain pins the mass",
        "Collocated feedback is stable at any gain, but more isn't better. Mode 1 turns overdamped at about "
        "26 N·s/m (its curve ends), and the other modes' damping peaks near there and falls again: the actuator "
        "holds m4 nearly still, and the chain vibrates around it as the chain with m4 fixed, three lightly "
        "damped modes (2.4, 4.5 and 5.9 Hz at 200 N·s/m). Each curve has an optimum, as a damper between two "
        "points does.",
        gain=200.0, tab=TAB_GAIN),
    ControlTrial(
        "Non-collocated: spillover",
        "Sensor on m1, actuator on m4, g = 20 N·s/m. In mode 1 the two move together, so mode 1 is damped; in "
        "mode 2 they move against each other, so the feedback pushes with the motion and mode 2's damping goes "
        "negative (estimate −0.17, exact −0.10): the chain is unstable from g = 6.1 N·s/m. Control designed for "
        "one mode spills over into the others.",
        sensor=1, actuator=4, gain=20.0, tab=TAB_LOCUS),
    ControlTrial(
        "Actuator lag",
        "Back to collocated, but the sensor filter and the actuator each lag by τ = 20 ms. At high frequency the "
        "two lags turn the force by 180°, and the feedback that damped now feeds the motion: the loop is unstable "
        "above 96 N·s/m (with 10 ms, 200). Every real loop has lag, so collocation alone doesn't guarantee "
        "stability.",
        lag_ms=20.0, gain=100.0, tab=TAB_LOCUS),
    ControlTrial(
        "Active against a tuned mass damper",
        "A Den Hartog tuned mass damper at m4 with 5 % of mode 1's modal mass there (116 g) cuts the tip's "
        "receptance peak from 0.26 to 0.051 m/N. Collocated velocity feedback does the same with 2.3 N·s/m, and "
        "10 times that does much better and damps every mode, not just the one it's tuned to; but it needs "
        "power, a sensor and a stable loop.",
        gain=2.3, tmd=True, tab=TAB_GAIN),
    ControlTrial(
        "Skyhook isolation",
        "A machine on a mount over a vibrating floor (Skyhook tab). A passive damper (ζ = 0.1) trades the "
        "resonance peak (5.1) against isolation above √2 f<sub>n</sub> (at 5 f<sub>n</sub>: 0.059). A "
        "skyhook damper, pushing against an inertial frame through an actuator that feeds back the mass's "
        "absolute velocity, has neither problem: ζ<sub>sky</sub> = 0.5 caps the peak at 1.15 and keeps the "
        "−40 dB/decade roll-off (0.041 at 5 f<sub>n</sub>).",
        tab=TAB_SKYHOOK, zeta_passive=0.1, zeta_sky=0.5),
)

TIPS = {
    "dof": "<p>The number of masses: the same chain as the Simulation page.</p>",
    "sensor": "<p>The mass whose velocity is measured.</p>",
    "actuator": "<p>The mass the actuator pushes on (against the ground).</p>",
    "gain": "<p>u = −g ẋ<sub>sensor</sub>: the gain g, in N·s/m (a damper's units).</p>",
    "lag": "<p>The time constant of each of two first-order lags in the loop (sensor filter and actuator). 0: "
           "ideal.</p>",
    "inp": "<p>Where the disturbing force acts, for the receptance plot.</p>",
    "out": "<p>Where the response is measured, for the receptance plot.</p>",
    "tmd": "<p>Hang a Den Hartog tuned mass damper on the actuator's mass, tuned to mode 1, instead of the "
           "control (dashed curve).</p>",
    "ratio": "<p>The absorber's mass, as a fraction of mode 1's modal mass at that mass.</p>",
    "zp": "<p>The passive mount's damping ratio (a damper to the floor).</p>",
    "zs": "<p>The skyhook damper's ratio (a force −c ẋ against an inertial frame).</p>",
}
FRF_TIP = "<p>The receptance from the disturbance to the response: open loop, closed loop, and with the TMD.</p>"
LOCUS_TIP = ("<p>The closed-loop poles as the gain rises from 0 (open dots) to 2000 N·s/m; the filled dots are at "
             "the gain set. Right of Re = 0 is unstable.</p>")
GAIN_TIP = "<p>Each mode's damping ratio against the gain (followed by continuity); the dotted line is the gain set.</p>"
SKY_TIP = "<p>Transmissibility from floor to machine: passive damper, skyhook, and both.</p>"

NOTES_HTML = """
<!--BEFORE-->
<h3><a name="feedback"></a>Feedback</h3>
<p>Active control measures the motion and pushes back through an actuator. The simplest law is <b>direct velocity
feedback</b>: a force u = −g ẋ<sub>s</sub> at mass a from the velocity at mass s. With s = a (<b>collocated</b>)
it is an ideal damper from that mass to the ground: it can only take energy out, so it is stable at any gain, and
for light gain each mode gains</p>
<p align="center">Δζ<sub>r</sub> = g φ<sub>r</sub>(a) φ<sub>r</sub>(s) / 2ω<sub>r</sub></p>
<p>(φ mass-normalized), the same formula as a damper's modal damping. Too much gain holds the mass still and the
rest of the structure vibrates around it, so there is an optimal gain, as there is for any one damper.</p>

<h3><a name="collocation"></a>Collocation and spillover</h3>
<p>When the sensor and actuator are at different places, φ<sub>r</sub>(a)φ<sub>r</sub>(s) can be negative for
some modes: the feedback then <b>adds energy</b> to them. A controller designed to damp one mode (with the
others ignored) <b>spills over</b> into the modes it wasn't designed for, and the unmodelled ones can go unstable.
Collocated, dual sensor–actuator pairs avoid this: their transfer functions have alternating poles and zeros, and
the phase stays between 0 and −180°.</p>

<h3><a name="lag"></a>Lag</h3>
<p>Real actuators, amplifiers, filters and digital controllers lag. A lag turns the feedback's phase, so at high
enough frequency the force that should oppose the velocity helps it. Two first-order lags of τ (as the page has)
reach 180° far above 1/τ, and a high-gain loop goes unstable there; a time delay does it sooner. This is why
practical active damping uses modest gains, roll-off filters, and collocated pairs.</p>

<h3><a name="tmd"></a>Active or passive?</h3>
<p>A tuned mass damper (the Simulation page's absorber) needs no power and cannot go unstable, but it damps one
mode, near one frequency, and needs mass (Den Hartog's optimum: tune to f<sub>1</sub>/(1 + μ) with
ζ = √(3μ/8(1 + μ)³)). Active control damps many modes at once with a small actuator, at the cost of power,
sensors and stability margins. Hybrid and semi-active devices (variable dampers, magnetorheological fluid) sit in
between.</p>

<h3><a name="skyhook"></a>Skyhook</h3>
<p>A passive isolator's damper connects the machine to the floor, so above √2 f<sub>n</sub> it passes the floor's
motion through: more damping, less resonance, worse isolation. A <b>skyhook</b> damper acts between the machine
and a fixed point in space: impossible passively, but an actuator fed the machine's absolute velocity does it.
Its transmissibility 1/(1 − r² + 2iζr) has a bounded peak and the full −40 dB/decade roll-off. Semi-active car
suspensions approximate it.</p>

<h3><a name="try-it"></a>Things to try</h3>
<p>Each of these loads the uniform chain into the Simulation page and sets up the loop (Things to try, top
left).</p>
<!--TRIALS-->
<!--TAKEAWAYS-->
"""
NOTES_HTML = NOTES_HTML.replace("<!--BEFORE-->", before(
    ("sim_page", "vibration-absorbers-and-tuned-mass-dampers", "Absorbers and tuned mass dampers"),
    ("isolation_page", "damping-trade-off", "Isolation's damping trade-off"),
    ("complex_page", "state-space", "The state-space form"),
)).replace("<!--TRIALS-->", trials_html(TRIALS)).replace("<!--TAKEAWAYS-->", takeaways(
    "Collocated velocity feedback is an active damper: stable at any gain (ideally), with an optimal gain.",
    "Each mode gains Δζ = gφ<sub>r</sub>(a)φ<sub>r</sub>(s)/2ω<sub>r</sub>; non-collocated, it can be negative: "
    "spillover.",
    "Lag in the loop limits the gain even when collocated.",
    "A TMD damps one mode without power; active control damps many, with power and stability to look after.",
    "Skyhook damping caps the resonance without giving up high-frequency isolation.",
))


class ActiveControlPage(QtWidgets.QWidget):
    """Controls on the left, the receptances in the centre, the locus and damping on the right."""

    dof_requested = QtCore.Signal(int)
    system_requested = QtCore.Signal(object)
    edit_parameters = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.system = ChainSystem.uniform(4)
        self._dirty = True

        self.trials = TrialBox([t.name for t in TRIALS],
                               "<p>Each loads the uniform chain into the Simulation page and sets up the loop to "
                               "show one idea from the Theory tab.</p>")
        self.trials.picked.connect(self.load_trial)

        ch = Form("Chain")
        self.dof = QtWidgets.QSpinBox()
        self.dof.setRange(1, MAX_DOF)
        self.dof.valueChanged.connect(self.dof_requested)
        ch.add("Number of masses:", self.dof, TIPS["dof"])
        link = QtWidgets.QLabel("Masses, springs and dampers are edited on the <a href='#sim'>Simulation page</a>.")
        link.setWordWrap(True)
        mute(link)
        link.linkActivated.connect(lambda _: self.edit_parameters.emit())
        ch.row(link)

        loop = Form("Feedback loop")
        self.sensor = loop.number("Sensor on mass:", 1, MAX_DOF, 4, 0, "", TIPS["sensor"])
        self.actuator = loop.number("Actuator on mass:", 1, MAX_DOF, 4, 0, "", TIPS["actuator"])
        self.gain = loop.number("Gain g:", 0.0, 5000.0, 20.0, 2, " N·s/m", TIPS["gain"])
        self.lag = loop.number("Lags τ:", 0.0, 500.0, 0.0, 1, " ms", TIPS["lag"])
        self.loop_info = note_label()
        loop.row(self.loop_info)

        plotting = Form("Receptance")
        self.inp = plotting.number("Force on mass:", 1, MAX_DOF, 4, 0, "", TIPS["inp"])
        self.out = plotting.number("Response of mass:", 1, MAX_DOF, 4, 0, "", TIPS["out"])
        self.tmd = QtWidgets.QCheckBox("Compare a tuned mass damper")
        self.tmd.setToolTip(TIPS["tmd"])
        plotting.row(self.tmd)
        self.ratio = plotting.number("Its mass ratio:", 0.005, 0.5, 0.05, 3, "", TIPS["ratio"])

        connect_all((self.sensor, self.actuator, self.gain, self.lag, self.inp, self.out, self.tmd, self.ratio),
                    self.redraw)

        # --- centre
        self.frf_plot = plot("|Receptance|", "m/N", "Frequency", "Hz", log_x=True, log_y=True, height=200)
        self.frf_plot.setToolTip(FRF_TIP)
        add_legend(self.frf_plot.getPlotItem(), offset=(-10, 10))
        self.open_item = pg.PlotDataItem(name="no control")
        self.closed_item = pg.PlotDataItem(name="velocity feedback")
        self.tmd_item = pg.PlotDataItem(name="tuned mass damper")
        for item in (self.open_item, self.tmd_item, self.closed_item):
            self.frf_plot.addItem(item)
        self.readout = readout_label()

        # --- right
        self.locus_plot = plot("Im λ / 2π", "Hz", "Re λ", "1/s", height=200)
        self.locus_plot.setToolTip(LOCUS_TIP)
        self.locus_items = marker(4)
        self.locus_open = marker(9)
        self.locus_now = marker(9)
        self.locus_axis = pg.InfiniteLine(angle=90, pos=0)
        for item in (self.locus_axis, self.locus_items, self.locus_open, self.locus_now):
            self.locus_plot.addItem(item)
        self.gain_plot = plot("Damping ratio ζ (%)", None, "Gain g (N·s/m)", None, log_x=True, height=200)
        self.gain_plot.setToolTip(GAIN_TIP)
        add_legend(self.gain_plot.getPlotItem(), offset=(-10, 10))
        self.gain_line = pg.InfiniteLine(angle=90)
        self.gain_zero = pg.InfiniteLine(angle=0, pos=0)
        self.gain_plot.addItem(self.gain_line)
        self.gain_plot.addItem(self.gain_zero)
        self.gain_curves: list[pg.PlotDataItem] = []
        sky = QtWidgets.QWidget()
        sv = QtWidgets.QVBoxLayout(sky)
        sf = Form("")
        self.zp = sf.number("Passive ζ:", 0.0, 2.0, 0.1, 3, "", TIPS["zp"])
        self.zs = sf.number("Skyhook ζ:", 0.0, 2.0, 0.5, 3, "", TIPS["zs"])
        connect_all((self.zp, self.zs), self._draw_tab)
        self.sky_plot = plot("Transmissibility |X/Y|", None, "Frequency ratio f/f<sub>n</sub>", None, log_x=True,
                             log_y=True, height=200)
        self.sky_plot.setToolTip(SKY_TIP)
        add_legend(self.sky_plot.getPlotItem(), offset=(-10, 10))
        self.sky_items = [pg.PlotDataItem(name="passive (damper to the floor)"),
                          pg.PlotDataItem(name="skyhook (damper to the sky)"), pg.PlotDataItem(name="both")]
        for item in self.sky_items:
            self.sky_plot.addItem(item)
        sv.addWidget(sf)
        sv.addWidget(self.sky_plot, 1)
        self.tab_note = note_label()

        self.notes = theory(NOTES_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(column(self.locus_plot, stretch=False), "Root locus")
        self.tabs.addTab(column(self.gain_plot, stretch=False), "Damping vs gain")
        self.tabs.addTab(sky, "Skyhook")
        self.tabs.addTab(self.notes, "Theory")
        self.tabs.currentChanged.connect(lambda _: self._draw_tab())
        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(self.tabs, 1)
        rv.addWidget(self.tab_note)

        three_panes(self, column(self.trials, ch, loop, plotting), stack((self.frf_plot, 1), (self.readout, 0)),
                    right)
        self.apply_theme()
        self.set_system(self.system)

    # ------------------------------------------------------------- inputs
    def set_system(self, system: ChainSystem, result=None) -> None:
        self.system = system
        n = system.n
        set_quietly(self.dof, n)
        for box in (self.sensor, self.actuator, self.inp, self.out):
            box.blockSignals(True)
            box.setMaximum(n)
            box.blockSignals(False)
        self._gain_built = -1
        self._dirty = True
        if self.isVisible():
            self.redraw()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._dirty:
            self.redraw()

    def load_trial(self, index: int) -> None:
        t = TRIALS[index]
        system = ChainSystem.uniform(4)
        self.system_requested.emit(system)
        s = self.system
        if (s.n != 4 or not np.allclose(s.masses, 1.0) or not np.allclose(s.stiffness, 400.0)
                or not np.allclose(s.damping, 2.0)):
            self.set_system(system)
        for box, value in ((self.sensor, t.sensor), (self.actuator, t.actuator), (self.gain, t.gain),
                           (self.lag, t.lag_ms), (self.inp, 4), (self.out, 4), (self.tmd, t.tmd),
                           (self.ratio, 0.05), (self.zp, t.zeta_passive), (self.zs, t.zeta_sky or 0.5)):
            set_quietly(box, value)
        self.trials.show_note(t.note)
        self.tabs.setCurrentIndex(t.tab)
        self.redraw()

    def controller(self, gain: float | None = None) -> Controller:
        n = self.system.n
        return Controller(min(int(self.sensor.value()), n) - 1, min(int(self.actuator.value()), n) - 1,
                          self.gain.value() if gain is None else gain, self.lag.value() / 1000.0)

    # ------------------------------------------------------------- drawing
    def apply_theme(self) -> None:
        c = colors
        self.open_item.setPen(pg.mkPen(c.grey, width=2))
        self.closed_item.setPen(pg.mkPen(c.mass[0], width=2.5))
        self.tmd_item.setPen(pg.mkPen(c.mass[1], width=2, style=QtCore.Qt.PenStyle.DashLine))
        mark(self.locus_items, c.mass[0], filled=True)
        mark(self.locus_open, c.grey)
        mark(self.locus_now, c.force, filled=True)
        self.locus_axis.setPen(pg.mkPen(c.strong, width=1, style=QtCore.Qt.PenStyle.DotLine))
        self.gain_line.setPen(pg.mkPen(c.grey, width=1.5, style=QtCore.Qt.PenStyle.DotLine))
        self.gain_zero.setPen(pg.mkPen(c.faint, width=1))
        for item, color, style in zip(self.sky_items, (c.grey, c.mass[0], c.mass[1]),
                                      (QtCore.Qt.PenStyle.SolidLine, QtCore.Qt.PenStyle.SolidLine,
                                       QtCore.Qt.PenStyle.DashLine)):
            item.setPen(pg.mkPen(color, width=2, style=style))
        self._gain_built = -1
        self._dirty = True
        if self.isVisible():
            self.redraw()

    def redraw(self) -> None:
        self._dirty = False
        s = self.system
        ctrl = self.controller()
        self.ratio.setEnabled(self.tmd.isChecked())
        w, _ = undamped_modes(s)
        top = 2.0 * float(w[-1]) / (2 * np.pi) if w[-1] > 0 else 10.0
        bottom = max(0.2 * float(w[w > 1e-6][0]) / (2 * np.pi), 1e-3) if np.any(w > 1e-6) else 0.05
        f = np.geomspace(bottom, top, 1500)
        n = s.n
        out, inp = min(int(self.out.value()), n) - 1, min(int(self.inp.value()), n) - 1
        A0, B0, C0 = closed_loop(s, None)
        A1, B1, C1 = closed_loop(s, ctrl)
        H0 = np.abs(frf(A0, B0, C0, out, inp, f))
        stable = max_real(A1) < 0
        H1 = np.abs(frf(A1, B1, C1, out, inp, f)) if stable else None
        self.open_item.setData(f, H0)
        self.closed_item.setData(f, H1) if H1 is not None else self.closed_item.setData([], [])
        tmd_text = ""
        if self.tmd.isChecked() and w[0] > 1e-6:
            t = tmd_for(s, ctrl.actuator, self.ratio.value())
            A2, B2, C2 = closed_loop(s, None, t)
            H2 = np.abs(frf(A2, B2, C2, out, inp, f))
            self.tmd_item.setData(f, H2)
            tmd_text = (f" TMD ({t[1] * 1e3:.3g} g on m{ctrl.actuator + 1}): peak {H2.max():.3g} m/N.")
        else:
            self.tmd_item.setData([], [])
        self.frf_plot.setXRange(np.log10(f[0]), np.log10(f[-1]), padding=0)

        open_poles = oscillating_poles(A0)
        closed = oscillating_poles(A1)
        est = modal_damping_estimate(s, ctrl)
        rows = "".join(
            f"<tr><td>mode {j + 1}</td><td align='right'>{p.freq_hz:.3g} Hz</td>"
            f"<td align='right'>{100 * p.zeta:.3g} %</td><td align='right'>{100 * (p.zeta + est[j]):.3g} %</td></tr>"
            for j, p in enumerate(open_poles))
        closed_rows = ", ".join(f"{p.freq_hz:.3g} Hz at {100 * p.zeta:.3g} %" for p in closed)
        self.readout.setText(
            ("" if stable else "<b>The loop is unstable</b> (a pole in the right half plane); no closed-loop "
             "receptance. ")
            + f"Peak without control {H0.max():.3g} m/N" + (f", with it {H1.max():.3g} m/N." if H1 is not None
                                                            else ".") + tmd_text
            + "<table cellspacing='0' cellpadding='2'><tr><th></th><th>f</th><th>ζ open</th>"
              "<th>ζ + gφ(a)φ(s)/2ω</th></tr>" + rows + "</table>"
            + f"Closed loop, exactly: {closed_rows}.")
        collocated = ctrl.sensor == ctrl.actuator
        self.loop_info.setText("Collocated: a damper to the ground (ideally stable at any gain)." if collocated else
                               "Not collocated: modes where the two masses move against each other are fed "
                               "energy.")
        self._draw_tab()
        settle_plots(self)

    def _draw_tab(self) -> None:
        i = self.tabs.currentIndex()
        s = self.system
        ctrl = self.controller()
        if i == TAB_LOCUS:
            pts = []
            for g in GAINS:
                lam = np.linalg.eigvals(closed_loop(s, self.controller(float(g)))[0])
                pts.extend(lam[lam.imag >= 0])
            pts = np.array(pts)
            self.locus_items.setData(pts.real, pts.imag / (2 * np.pi))
            lam0 = np.linalg.eigvals(closed_loop(s, None)[0])
            lam0 = lam0[lam0.imag >= 0]
            self.locus_open.setData(lam0.real, lam0.imag / (2 * np.pi))
            lam = np.linalg.eigvals(closed_loop(s, ctrl)[0])
            lam = lam[lam.imag >= 0]
            self.locus_now.setData(lam.real, lam.imag / (2 * np.pi))
            w = undamped_modes(s)[0]
            fmax = float(w[-1]) / (2 * np.pi) * 1.3
            reals = pts.real[np.abs(pts.real) < 5 * fmax * 2 * np.pi]
            self.locus_plot.setXRange(max(float(reals.min()), -4 * fmax * 2 * np.pi * 0.3),
                                      max(float(reals.max()), 1.0) * 1.1, padding=0.05)
            self.locus_plot.setYRange(0.0, fmax, padding=0.02)
            self.tab_note.setText("Open dots: the open-loop poles; small dots: g from 0 to 2000 N·s/m; red: the gain "
                                  "set. " + ("All stay left of Re λ = 0." if max(pts.real) < 0 else
                                             "Some cross into the right half plane: unstable."))
        elif i == TAB_GAIN:
            n = s.n
            c = colors
            if self._gain_built != n:
                for item in self.gain_curves:
                    self.gain_plot.removeItem(item)
                legend = self.gain_plot.getPlotItem().legend
                legend.clear()
                self.gain_curves = []
                for j in range(n):
                    item = pg.PlotDataItem(pen=pg.mkPen(c.mode[j % len(c.mode)], width=2))
                    self.gain_plot.addItem(item)
                    legend.addItem(item, f"mode {j + 1}")
                    self.gain_curves.append(item)
                self._gain_built = n
            g = GAINS[1:]
            z = tracked_damping(s, ctrl, g)
            for j, item in enumerate(self.gain_curves):
                item.setData(g, 100 * z[:, j], connect="finite")
            self.gain_line.setValue(np.log10(max(ctrl.gain, g[0])))
            self.gain_plot.setXRange(np.log10(g[0]), np.log10(g[-1]), padding=0)
            finite = z[np.isfinite(z)]
            self.gain_plot.setYRange(100 * min(-0.05, float(finite.min())), 100 * min(1.05, float(finite.max()) + 0.05),
                                     padding=0)
            self.tab_note.setText("A curve that ends has stopped oscillating: its pole has reached the real "
                                  "axis (overdamped).")
        elif i == TAB_SKYHOOK:
            r = np.geomspace(0.1, 10.0, 600)
            for item, y in zip(self.sky_items, skyhook(r, self.zp.value(), self.zs.value())):
                item.setData(r, y)
            self.sky_plot.setXRange(-1, 1, padding=0)
            p, sk, _ = skyhook(np.array([1.0, 5.0]), self.zp.value(), self.zs.value())
            self.tab_note.setText(f"At f<sub>n</sub>: passive {p[0]:.3g}, skyhook {sk[0]:.3g}. At 5 f<sub>n</sub>: "
                                  f"passive {p[1]:.3g}, skyhook {sk[1]:.3g}.")
        else:
            self.tab_note.setText("")
