"""Single oscillator page: one mass on a spring and a damper, free and forced, worked out in closed form.

Left: the oscillator, its start and its drive. Centre: the free decay (with its envelope and peaks, or
three dampings side by side) and the response to a harmonic force switched on from rest. Right: the
frequency response with its half-power band, the Nyquist plot, the phase plane, the poles (click to move
them) and the Theory. The page has its own model (core/sdof.py), not the chain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.sdof import (
    TWO_PI,
    Oscillator,
    free_response,
    frequency_grid,
    harmonic_from_rest,
    measured_decrement,
    peaks,
    receptance,
)
from . import settle_plots
from .layouts import ColumnScroll
from .lesson import (
    Form,
    TrialBox,
    fmt,
    note_label,
    plot,
    readout_label,
    set_quietly,
    trials_html,
)
from .links import course_map, takeaways
from .style import colors
from .theming import add_legend

POINTS = 3000  # samples on each time plot
COMPARE_ZETAS = (0.1, 1.0, 2.0)  # the damping ratios drawn side by side
FORCE, UNBALANCE = 0, 1  # the drive's kinds, as the combo lists them
TAB_FRF, TAB_NYQUIST, TAB_PHASE, TAB_POLES = range(4)


@dataclass(frozen=True)
class SdofTrial:
    name: str
    note: str
    mass: float = 1.0
    stiffness: float = 400.0
    damping: float = 2.0
    x0_mm: float = 10.0
    v0_mms: float = 0.0
    compare: bool = False
    kind: int = FORCE
    amplitude: float = 1.0  # N, or the unbalance m_e e in kg·m
    freq_hz: float = 3.0
    tune: bool = False  # drive at f_n
    tab: int = TAB_FRF


TRIALS = (
    SdofTrial(
        "Ringing down: the log decrement",
        "m = 1 kg, k = 400 N/m, c = 2 N·s/m: f<sub>n</sub> = 3.18 Hz and ζ = 0.05. Released from 10 mm, each "
        "peak is 1.37 times the next: the log decrement δ = ln 1.37 = 0.315, and ζ = δ/2π = 0.050. The readout "
        "under the decay measures it from the curve's peaks. The dashed envelope falls by e in the time "
        "constant 1/ζω<sub>n</sub> = 1 s.",
        tab=TAB_PHASE),
    SdofTrial(
        "Critical damping: the quickest return",
        "c = 40 N·s/m, the critical damping 2√(km), with ζ = 0.1, 1 and 2 drawn beside it. ζ = 0.1 overshoots "
        "and rings; ζ = 1 comes back without crossing zero, the fastest that does; ζ = 2 creeps back, held up "
        "by its slow root at −0.27ω<sub>n</sub> (a time constant of 0.19 s against critical's 0.05 s).",
        damping=40.0, compare=True, tab=TAB_POLES),
    SdofTrial(
        "Overdamped: two real poles",
        "c = 80 N·s/m, ζ = 2. The poles have left the imaginary axis for two points on the real one, at "
        "−5.4 and −74.6 s<sup>−1</sup>: two decays and no oscillation. Click in the Poles plot to put them "
        "anywhere; the page works out the k and c that put them there.",
        damping=80.0, tab=TAB_POLES),
    SdofTrial(
        "Resonance: a 90° lag",
        "A 1 N force at f<sub>n</sub> = 3.18 Hz. From rest the swing grows over a few time constants (1 s) to "
        "(F/k)/2ζ = 25 mm, ten times the static 2.5 mm: the quality factor Q = 1/2ζ = 10. The response lags "
        "the force by 90°, where the phase plot crosses −90°.",
        tune=True, tab=TAB_FRF),
    SdofTrial(
        "Half-power bandwidth",
        "The default oscillator's peak, with its half-power points marked: |H| falls to 1/√2 of the peak at "
        "3.01 and 3.33 Hz, 0.32 Hz apart, so ζ ≈ Δf/2f<sub>n</sub> = 0.050. Raise c and the band widens in "
        "proportion.",
        freq_hz=3.0, tab=TAB_FRF),
    SdofTrial(
        "Beats: undamped, just off resonance",
        "c = 0, a 1 N force at 3.0 Hz, 0.18 Hz below f<sub>n</sub>. The free vibration the start sets off never "
        "dies, and it drifts in and out of step with the forced one: the swing beats at |f − f<sub>n</sub>| = "
        "0.18 Hz, every 5.5 s, between zero and twice the steady 22 mm.",
        damping=0.0, freq_hz=3.0, tab=TAB_FRF),
    SdofTrial(
        "Exactly at resonance, undamped",
        "c = 0 and the force at f<sub>n</sub>. There is no steady state: the swing grows in a straight line, "
        "F/2mω<sub>n</sub> = 25 mm a second, for ever. Any damping at all bounds it.",
        damping=0.0, tune=True, tab=TAB_FRF),
    SdofTrial(
        "A rotating unbalance",
        "An unbalance m<sub>e</sub>e = 1×10<sup>−4</sup> kg·m (10 g at 10 mm) on a 1 kg machine. Its force "
        "grows as Ω², so the response starts at zero, peaks at resonance at (m<sub>e</sub>e/m)/2ζ = 1 mm, and "
        "levels off far above it at m<sub>e</sub>e/m = 0.1 mm, 180° out of phase: the machine turns round "
        "the shared centre of mass.",
        kind=UNBALANCE, amplitude=1e-4, tune=True, tab=TAB_FRF),
    SdofTrial(
        "The Nyquist circle",
        "The receptance drawn as Im H against Re H. Near resonance it traces nearly a circle of diameter "
        "1/cω<sub>n</sub> = 25 mm/N, crossing the imaginary axis at f<sub>n</sub>; the points (1 per "
        "0.02 Hz) are furthest apart there, which is how modal tests find f<sub>n</sub>.",
        freq_hz=3.0, tab=TAB_NYQUIST),
)

TIPS = {
    "mass": "<p>The mass m.</p>",
    "stiffness": "<p>The spring's stiffness k: the force per metre of stretch.</p>",
    "damping": "<p>The viscous damper's coefficient c: the force per m/s of velocity. Its critical value "
               "2√(km) is the least that stops the oscillator overshooting.</p>",
    "x0": "<p>Where the mass starts the free vibration (the driven response starts from rest).</p>",
    "v0": "<p>The velocity it starts with.</p>",
    "compare": "<p>Draw the free vibration for ζ = 0.1, 1 and 2 as well, with the same mass and spring "
               "(underdamped, critically damped, overdamped).</p>",
    "kind": "<p><b>Force:</b> F cos Ωt on the mass.<br><b>Rotating unbalance:</b> a mass m<sub>e</sub> turning "
            "at radius e inside the machine, whose force m<sub>e</sub>eΩ² cos Ωt grows with the speed.</p>",
    "amplitude": "<p>The force's amplitude F, or the unbalance m<sub>e</sub>e.</p>",
    "freq": "<p>The drive's frequency Ω/2π.</p>",
}
FRF_TIP = (
    "<p>The steady amplitude over the static one, |X|k/F (the dynamic amplification), or for the unbalance "
    "m|X|/m<sub>e</sub>e, against the drive's frequency, and the phase by which the response lags. The band "
    "is the half-power width, the dotted line f<sub>n</sub>, the circle the drive.</p>")
NYQUIST_TIP = (
    "<p>The receptance H = X/F in the complex plane, Im against Re, from low frequency (near 1/k on the real "
    "axis) round to the origin. The dots are 0.02 Hz apart; the circle is the drive.</p>")
PHASE_TIP = (
    "<p>The free vibration as a path in the phase plane: velocity over ω<sub>n</sub> against displacement. "
    "Undamped it is a circle; damping winds it into a spiral, a turn per cycle.</p>")
POLES_TIP = (
    "<p>The roots of m s² + c s + k in the complex plane: −ζω<sub>n</sub> ± iω<sub>d</sub>. Their distance "
    "from the origin is ω<sub>n</sub>, the angle from the imaginary axis is sin⁻¹ζ. The rays are ζ = 0.05, "
    "0.2, 0.5 and 0.8, the arc the current ω<sub>n</sub>. Click to put the upper pole there (the mass is "
    "kept, k and c follow).</p>")

NOTES_HTML = """
<h3><a name="course-map"></a>The pages, in order</h3>
<p>This is the first page. The pages are meant to be read in order, left to right, group by group: each one
uses only ideas from the pages before it, and its Theory tab (F1) starts by saying which. The <i>One mass</i>
pages come first, then the Simulation page builds a chain of these oscillators. The Theory window (its
button, or Ctrl+F1) lists them too, and keeps its place as you move between pages.</p>
<!--COURSE-->
<h3><a name="equation"></a>One mass on a spring</h3>
<p>Every vibration on these pages is built from one oscillator: a mass m on a spring k and a damper c,</p>
<p align="center"><b>m ẍ + c ẋ + k x = f(t).</b></p>
<p>The spring pulls back in proportion to the stretch, the damper in proportion to the velocity, and
whatever is left over accelerates the mass. Three numbers describe it, whatever m, k and c are:</p>
<ul>
<li>the <b>natural frequency</b> ω<sub>n</sub> = √(k/m), f<sub>n</sub> = ω<sub>n</sub>/2π: how fast it swings
when nothing damps it. A stiffer spring is faster, a heavier mass slower;</li>
<li>the <b>damping ratio</b> ζ = c/c<sub>c</sub>, the damper against the <b>critical damping</b>
c<sub>c</sub> = 2√(km) = 2mω<sub>n</sub>;</li>
<li>the <b>static compliance</b> 1/k, how far a steady force pushes it.</li>
</ul>
<p>Dividing by m gives the form every later page uses: ẍ + 2ζω<sub>n</sub>ẋ + ω<sub>n</sub>²x = f/m.</p>

<h3><a name="free-vibration"></a>Free vibration</h3>
<p>Pull the mass to x<sub>0</sub> and let go. Below critical damping (ζ &lt; 1) it swings at the <b>damped
natural frequency</b> ω<sub>d</sub> = ω<sub>n</sub>√(1 − ζ²), inside an envelope that shrinks
exponentially:</p>
<p align="center">x(t) = e<sup>−ζω<sub>n</sub>t</sup> (x<sub>0</sub> cos ω<sub>d</sub>t + (v<sub>0</sub> +
ζω<sub>n</sub>x<sub>0</sub>)/ω<sub>d</sub> sin ω<sub>d</sub>t).</p>
<p>The envelope falls by e in the <b>time constant</b> τ = 1/ζω<sub>n</sub>. For the 0.5 % to 5 % damping of
most structures ω<sub>d</sub> is within 0.1 % of ω<sub>n</sub>: damping shortens the ringing, hardly changes
its pitch.</p>

<h3><a name="damping-regimes"></a>Under, critical and over</h3>
<p>Tick <i>Compare ζ = 0.1, 1 and 2</i> to see the three cases together:</p>
<ul>
<li><b>Underdamped</b> (ζ &lt; 1): it overshoots and rings down.</li>
<li><b>Critically damped</b> (ζ = 1): x = (x<sub>0</sub> + (v<sub>0</sub> + ω<sub>n</sub>x<sub>0</sub>)t)
e<sup>−ω<sub>n</sub>t</sup>. Released from rest it returns without crossing zero, and faster than any
heavier damping does. Door closers and instrument needles are made near it.</li>
<li><b>Overdamped</b> (ζ &gt; 1): two plain decays, e<sup>s<sub>1</sub>t</sup> and e<sup>s<sub>2</sub>t</sup>.
The slow one, s<sub>1</sub> = −ω<sub>n</sub>(ζ − √(ζ² − 1)), gets <i>slower</i> as the damping grows: a
mass in treacle creeps back.</li>
</ul>

<h3><a name="log-decrement"></a>Measuring damping from a decay</h3>
<p>Successive peaks of a free decay, one damped period apart, keep the same ratio. Its logarithm is the
<b>logarithmic decrement</b></p>
<p align="center">δ = ln(x<sub>k</sub>/x<sub>k+1</sub>) = 2πζ/√(1 − ζ²) ≈ 2πζ.</p>
<p>So ζ ≈ δ/2π. Over n cycles, δ = (1/n) ln(x<sub>0</sub>/x<sub>n</sub>), which averages out the noise of a
real record. The readout under the decay does exactly this on the curve's peaks. The phase plane shows it
too: each turn of the spiral is e<sup>δ</sup> times smaller than the last.</p>

<h3><a name="poles"></a>The poles</h3>
<p>Try x = e<sup>st</sup> in the free equation: m s² + c s + k = 0, so</p>
<p align="center">s = −ζω<sub>n</sub> ± iω<sub>d</sub> (ζ &lt; 1), &nbsp; s = −ω<sub>n</sub>(ζ ∓ √(ζ² − 1)) (ζ &gt; 1).</p>
<p>These two <b>poles</b> say everything about the free motion: the real part is the decay rate, the
imaginary part the frequency. Below critical they sit on a circle of radius ω<sub>n</sub> at an angle
sin⁻¹ζ from the imaginary axis; at ζ = 1 they meet on the real axis and then split along it. Every mode of
every later page is such a pair (the state-space method's eigenvalues λ are these s).</p>

<h3><a name="forced-response"></a>Forced response</h3>
<p>Under f = F cos Ωt, once the start has died away, the mass swings at the drive's frequency with
x = Re(X e<sup>iΩt</sup>), and X = H(Ω)F with the <b>receptance</b></p>
<p align="center">H(Ω) = 1/(k − Ω²m + iΩc) = (1/k) / (1 − r² + 2iζr), &nbsp; r = Ω/ω<sub>n</sub>.</p>
<ul>
<li>Well below resonance the spring carries the force: X ≈ F/k, in phase.</li>
<li>Well above it the mass does: X ≈ −F/mΩ², 180° behind, smaller the faster.</li>
<li>At resonance (r = 1) the spring's force and the mass's cancel and only the damper resists: |X| =
(F/k)/2ζ, lagging by exactly 90°.</li>
</ul>
<p>The ratio |X|k/F is the <b>dynamic amplification</b>; at resonance it is the <b>quality factor</b>
Q = 1/2ζ. The displacement's own peak is a little lower, at r = √(1 − 2ζ²), and there is none at all for
ζ ≥ 1/√2.</p>

<h3><a name="half-power"></a>The half-power bandwidth</h3>
<p>The power a damper takes is proportional to |X|², so the points where |X| falls to 1/√2 of its peak are
the <b>half-power points</b>. They lie at r² = 1 − 2ζ² ± 2ζ√(1 − ζ²), about 2ζf<sub>n</sub> apart, so</p>
<p align="center">ζ ≈ (f<sub>2</sub> − f<sub>1</sub>)/2f<sub>n</sub> = 1/2Q.</p>
<p>A lightly damped peak is tall and narrow. This is how damping is read off a measured frequency response,
and why a test needs fine frequency lines to see it.</p>

<h3><a name="beats"></a>Switching on: transients and beats</h3>
<p>From rest, the steady response alone would start with the wrong position and velocity, so a free
vibration at ω<sub>d</sub> is added to make up the difference. It dies away in a few time constants and
leaves the steady state. Undamped, it never dies: driven near resonance the two swings drift in and out of
step, and the motion <b>beats</b> at the difference frequency |f − f<sub>n</sub>|. Exactly at resonance
there is no steady state: x = (F/2mω<sub>n</sub>) t sin ω<sub>n</sub>t grows without bound, and only damping
(or the spring breaking) stops it.</p>

<h3><a name="unbalance"></a>A rotating unbalance</h3>
<p>A machine of mass m whose rotor has an unbalance m<sub>e</sub> at radius e feels the force
m<sub>e</sub>eΩ² cos Ωt. So</p>
<p align="center">m|X|/m<sub>e</sub>e = r²/√((1 − r²)² + (2ζr)²).</p>
<p>It is 0 at rest, (1/2ζ) at resonance, and tends to 1 far above: the machine swings by m<sub>e</sub>e/m,
in anti-phase with the unbalance, so that the two turn round their shared centre of mass. Fans, pumps and
washing machines start up through this peak (the <a href="page:isolation_page#start-up">Vibration isolation
page</a> runs one).</p>

<h3><a name="nyquist"></a>The Nyquist plot</h3>
<p>Drawing H as a point in the complex plane at each frequency (Im H against Re H) turns the resonance into
nearly a circle of diameter 1/cω<sub>n</sub>, which crosses the imaginary axis at f<sub>n</sub>, where the
points race round fastest. (The mobility, iΩH, makes an exact circle.) Modal testing fits circles like this
to find each mode's frequency and damping.</p>

<h3><a name="phase-plane"></a>The phase plane</h3>
<p>The state of the oscillator is two numbers, position and velocity. Plotting ẋ/ω<sub>n</sub> against x
turns undamped motion into a circle (its energy, ½kx² + ½mẋ², is constant) and damped motion into a
spiral, losing a fixed fraction of its radius each turn. Overdamped, the path runs straight into the
origin along the slow root's direction.</p>

<h3><a name="try-it"></a>Things to try</h3>
<p>Each of these sets up the oscillator and the plots (Things to try, top left).</p>
<!--TRIALS-->
<!--TAKEAWAYS-->
<h3><a name="notation"></a>Notation</h3>
<table border="1" cellspacing="0" cellpadding="3">
<tr><th>Symbol</th><th>Meaning</th><th>Units</th></tr>
<tr><td>m, k, c</td><td>mass, stiffness, damping coefficient</td><td>kg, N/m, N·s/m</td></tr>
<tr><td>ω<sub>n</sub>, f<sub>n</sub></td><td>natural frequency √(k/m), and over 2π</td><td>rad/s, Hz</td></tr>
<tr><td>ω<sub>d</sub></td><td>damped natural frequency ω<sub>n</sub>√(1 − ζ²)</td><td>rad/s</td></tr>
<tr><td>ζ</td><td>damping ratio c/2√(km)</td><td>–</td></tr>
<tr><td>τ</td><td>time constant 1/ζω<sub>n</sub></td><td>s</td></tr>
<tr><td>δ</td><td>logarithmic decrement</td><td>–</td></tr>
<tr><td>Ω, r</td><td>drive frequency, and over ω<sub>n</sub></td><td>rad/s, –</td></tr>
<tr><td>H</td><td>receptance X/F</td><td>m/N</td></tr>
<tr><td>Q</td><td>quality factor 1/2ζ</td><td>–</td></tr>
<tr><td>s</td><td>pole, root of m s² + c s + k</td><td>1/s</td></tr>
</table>
"""
NOTES_HTML = NOTES_HTML.replace("<!--COURSE-->", course_map()).replace("<!--TRIALS-->", trials_html(TRIALS)).replace("<!--TAKEAWAYS-->", takeaways(
    "One mass on a spring and a damper is set by f<sub>n</sub> = √(k/m)/2π, ζ = c/2√(km) and 1/k.",
    "Underdamped, it rings at ω<sub>d</sub> ≈ ω<sub>n</sub> inside an envelope e<sup>−ζω<sub>n</sub>t</sup>; "
    "critically damped it returns fastest without overshoot; overdamped it creeps.",
    "The log decrement δ ≈ 2πζ and the half-power width Δf ≈ 2ζf<sub>n</sub> both measure damping.",
    "Driven, it follows the force below resonance, opposes it above, and lags by 90° at resonance, where "
    "only damping limits it to Q = 1/2ζ times the static response.",
    "A switched-on force adds a free vibration that dies away; undamped, it beats instead.",
))


class SdofPage(QtWidgets.QWidget):
    """Controls on the left, the free and forced responses in the centre, frequency views and theory on the right."""

    follows_chain = False  # MainWindow: this page has its own model, not the chain

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.osc = Oscillator()

        # --- left
        self.trials = TrialBox([t.name for t in TRIALS],
                               "<p>Each sets up the oscillator and the plots to show one idea from the Theory "
                               "tab.</p>")
        self.trials.picked.connect(self.load_trial)

        box = Form("Oscillator")
        self.mass = box.number("Mass m:", 0.01, 1e4, 1.0, 3, " kg", TIPS["mass"])
        self.stiffness = box.number("Stiffness k:", 0.01, 1e8, 400.0, 2, " N/m", TIPS["stiffness"])
        self.damping = box.number("Damping c:", 0.0, 1e6, 2.0, 3, " N·s/m", TIPS["damping"])
        self.info = note_label()
        box.row(self.info)
        box.changed.connect(self._on_oscillator)

        start = Form("Free vibration")
        self.x0 = start.number("Start x₀:", -1000.0, 1000.0, 10.0, 2, " mm", TIPS["x0"])
        self.v0 = start.number("Velocity v₀:", -1e5, 1e5, 0.0, 1, " mm/s", TIPS["v0"])
        self.compare = QtWidgets.QCheckBox("Compare ζ = 0.1, 1 and 2")
        self.compare.setToolTip(TIPS["compare"])
        start.row(self.compare)
        start.changed.connect(self.redraw)
        self.compare.toggled.connect(self.redraw)

        drive = Form("Drive")
        self.kind = QtWidgets.QComboBox()
        self.kind.addItems(["Force F cos Ωt", "Rotating unbalance"])
        drive.add("Input:", self.kind, TIPS["kind"])
        self.amplitude = drive.number("Force F:", 0.0, 1e6, 1.0, 4, " N", TIPS["amplitude"])
        self.amplitude_label = drive.form.labelForField(self.amplitude)
        self.freq = drive.number("Frequency:", 0.001, 1e4, 3.0, 3, " Hz", TIPS["freq"])
        self.tune = QtWidgets.QPushButton("Tune to fₙ")
        self.tune.setToolTip("<p>Set the drive's frequency to the natural frequency.</p>")
        self.tune.clicked.connect(self.tune_to_natural)
        drive.row(self.tune)
        drive.changed.connect(self.redraw)
        self.kind.currentIndexChanged.connect(self._on_kind)

        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        for w in (self.trials, box, start, drive):
            lv.addWidget(w)
        lv.addStretch(1)

        # --- centre
        self.free_plot = plot("Displacement x", "m", "Time", "s", height=170)
        self.free_plot.setToolTip("<p>The free vibration from x₀ and v₀, its envelope ±e<sup>−ζω<sub>n</sub>t</sup>"
                                  " (dashed) and its peaks.</p>")
        self.free_legend = add_legend(self.free_plot.getPlotItem(), offset=(-10, 10))
        self.free_item = pg.PlotDataItem(name="x")
        self.envelope_items = [pg.PlotDataItem() for _ in range(2)]
        self.peak_item = pg.ScatterPlotItem(size=6)
        self.compare_items = [pg.PlotDataItem(name=f"ζ = {z:g}") for z in COMPARE_ZETAS]
        for item in (*self.envelope_items, *self.compare_items, self.free_item, self.peak_item):
            self.free_plot.addItem(item)
        self.free_note = readout_label()

        self.forced_plot = plot("Displacement x", "m", "Time", "s", height=170)
        self.forced_plot.setToolTip("<p>The drive switched on at t = 0 with the mass at rest: the whole response, "
                                    "and dashed the steady response it settles into.</p>")
        add_legend(self.forced_plot.getPlotItem(), offset=(-10, 10))
        self.steady_item = pg.PlotDataItem(name="steady")
        self.forced_item = pg.PlotDataItem(name="from rest")
        self.forced_plot.addItem(self.steady_item)
        self.forced_plot.addItem(self.forced_item)
        self.forced_note = readout_label()

        center = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(center)
        cv.setContentsMargins(0, 0, 0, 0)
        title = QtWidgets.QLabel("<b>Free vibration</b>")
        cv.addWidget(title)
        cv.addWidget(self.free_plot, 1)
        cv.addWidget(self.free_note)
        cv.addWidget(QtWidgets.QLabel("<b>Driven from rest, switched on at t = 0</b>"))
        cv.addWidget(self.forced_plot, 1)
        cv.addWidget(self.forced_note)

        # --- right
        frf = QtWidgets.QWidget()
        fv = QtWidgets.QVBoxLayout(frf)
        self.mag_plot = plot("Amplification |X|k/F", None, "Frequency", "Hz", log_x=True, log_y=True, height=150)
        self.phase_plot = plot("Lag", "°", "Frequency", "Hz", log_x=True, height=110)
        self.phase_plot.setXLink(self.mag_plot)
        for p in (self.mag_plot, self.phase_plot):
            p.setToolTip(FRF_TIP)
            p.setMouseEnabled(x=False, y=False)
        self.band = pg.LinearRegionItem(movable=False)
        self.mag_item = pg.PlotDataItem()
        self.fn_lines = [pg.InfiniteLine(angle=90) for _ in range(2)]
        # PlotDataItems, not ScatterPlotItems: only they follow the plots' log axes
        self.half_item = pg.PlotDataItem(pen=None, symbol="o", symbolSize=7)
        self.drive_items = [pg.PlotDataItem(pen=None, symbol="o", symbolSize=11) for _ in range(2)]
        self.lag_item = pg.PlotDataItem()
        self.mag_plot.addItem(self.band)
        for item in (self.mag_item, self.fn_lines[0], self.half_item, self.drive_items[0]):
            self.mag_plot.addItem(item)
        for item in (self.lag_item, self.fn_lines[1], self.drive_items[1]):
            self.phase_plot.addItem(item)
        self.phase_plot.setYRange(0, 180, padding=0.05)
        self.phase_plot.getAxis("left").setTicks([[(v, f"{v}") for v in (0, 45, 90, 135, 180)]])
        fv.addWidget(self.mag_plot, 3)
        fv.addWidget(self.phase_plot, 2)
        self.frf_note = note_label()
        fv.addWidget(self.frf_note)

        nyq = QtWidgets.QWidget()
        nv = QtWidgets.QVBoxLayout(nyq)
        self.nyq_plot = plot("Im H", "m/N", "Re H", "m/N", height=200)
        self.nyq_plot.setAspectLocked(True)
        self.nyq_plot.setToolTip(NYQUIST_TIP)
        self.nyq_item = pg.PlotDataItem()
        self.nyq_dots = pg.ScatterPlotItem(size=6)
        self.nyq_fn = pg.ScatterPlotItem(size=9, symbol="d")
        self.nyq_drive = pg.ScatterPlotItem(size=11)
        for item in (self.nyq_item, self.nyq_dots, self.nyq_fn, self.nyq_drive):
            self.nyq_plot.addItem(item)
        nv.addWidget(self.nyq_plot, 1)
        self.nyq_note = note_label()
        nv.addWidget(self.nyq_note)

        phase = QtWidgets.QWidget()
        pv = QtWidgets.QVBoxLayout(phase)
        self.plane = plot("Velocity / ωₙ", "m", "Displacement", "m", height=200)
        self.plane.setAspectLocked(True)
        self.plane.setToolTip(PHASE_TIP)
        self.plane_item = pg.PlotDataItem()
        self.plane_start = pg.ScatterPlotItem(size=9)
        self.plane.addItem(self.plane_item)
        self.plane.addItem(self.plane_start)
        pv.addWidget(self.plane, 1)
        pv.addWidget(note_label("The free vibration's path: a circle undamped, a spiral damped. The dot is "
                                "the start."))

        poles = QtWidgets.QWidget()
        sv = QtWidgets.QVBoxLayout(poles)
        self.s_plot = plot("Imaginary part", "rad/s", "Real part", "1/s", height=200)
        self.s_plot.setAspectLocked(True)
        self.s_plot.setMouseEnabled(x=False, y=False)
        self.s_plot.setToolTip(POLES_TIP)
        self.rays = [pg.PlotDataItem() for _ in range(4)]
        self.arc = pg.PlotDataItem()
        self.pole_item = pg.ScatterPlotItem(size=12, symbol="x")
        for item in (*self.rays, self.arc, self.pole_item):
            self.s_plot.addItem(item)
        self.s_plot.scene().sigMouseClicked.connect(self._on_pole_click)
        sv.addWidget(self.s_plot, 1)
        self.pole_note = note_label()
        sv.addWidget(self.pole_note)

        self.notes = QtWidgets.QTextBrowser()
        self.notes.setHtml(NOTES_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(frf, "Frequency response")
        self.tabs.addTab(nyq, "Nyquist")
        self.tabs.addTab(phase, "Phase plane")
        self.tabs.addTab(poles, "Poles")
        self.tabs.addTab(self.notes, "Theory")

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(ColumnScroll(left))
        splitter.addWidget(center)
        splitter.addWidget(self.tabs)
        splitter.setSizes([320, 760, 560])
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        self.apply_theme()

    # ------------------------------------------------------------- inputs
    def load_trial(self, index: int) -> None:
        t = TRIALS[index]
        for box, value in ((self.mass, t.mass), (self.stiffness, t.stiffness), (self.damping, t.damping),
                           (self.x0, t.x0_mm), (self.v0, t.v0_mms), (self.compare, t.compare), (self.kind, t.kind),
                           (self.amplitude, t.amplitude), (self.freq, t.freq_hz)):
            set_quietly(box, value)
        self._on_kind(redraw=False)
        self.osc = Oscillator(t.mass, t.stiffness, t.damping)
        if t.tune:
            set_quietly(self.freq, round(self.osc.natural_hz, 6))
        self.trials.show_note(t.note)
        self.tabs.setCurrentIndex(t.tab)
        self.redraw()

    def _on_oscillator(self) -> None:
        self.osc = Oscillator(self.mass.value(), self.stiffness.value(), self.damping.value())
        self.redraw()

    def _on_kind(self, *_, redraw: bool = True) -> None:
        unbalance = self.kind.currentIndex() == UNBALANCE
        self.amplitude_label.setText("Unbalance mₑe:" if unbalance else "Force F:")
        self.amplitude.setSuffix(" kg·m" if unbalance else " N")
        self.amplitude.setDecimals(6 if unbalance else 4)
        if redraw:
            self.redraw()

    def tune_to_natural(self) -> None:
        self.freq.setValue(round(self.osc.natural_hz, 6))

    def _on_pole_click(self, event) -> None:
        vb = self.s_plot.getPlotItem().getViewBox()
        if not vb.sceneBoundingRect().contains(event.scenePos()):
            return
        pos = vb.mapSceneToView(event.scenePos())
        sigma, w = pos.x(), abs(pos.y())
        if sigma >= 0 or w <= 0:
            return
        wn = math.hypot(sigma, w)
        m = self.osc.mass
        set_quietly(self.stiffness, m * wn * wn)
        set_quietly(self.damping, -2.0 * sigma * m)
        self._on_oscillator()

    def force_amplitude(self, freq_hz: float | np.ndarray) -> float | np.ndarray:
        """The drive force's amplitude at a frequency: F, or m_e e Ω² for the unbalance."""
        a = self.amplitude.value()
        if self.kind.currentIndex() == UNBALANCE:
            return a * (TWO_PI * np.asarray(freq_hz)) ** 2
        return a + 0 * np.asarray(freq_hz)

    # ------------------------------------------------------------- drawing
    def apply_theme(self) -> None:
        c = colors
        self.free_item.setPen(pg.mkPen(c.mass[0], width=2))
        for item in self.envelope_items:
            item.setPen(pg.mkPen(c.grey, width=1, style=QtCore.Qt.PenStyle.DashLine))
        self.peak_item.setPen(None)
        self.peak_item.setBrush(pg.mkBrush(c.force))
        for item, color in zip(self.compare_items, (c.mode[0], c.mode[1], c.mode[2])):
            item.setPen(pg.mkPen(color, width=1.5))
        self.forced_item.setPen(pg.mkPen(c.mass[0], width=2))
        self.steady_item.setPen(pg.mkPen(c.grey, width=1.5, style=QtCore.Qt.PenStyle.DashLine))
        self.mag_item.setPen(pg.mkPen(c.strong, width=2))
        self.lag_item.setPen(pg.mkPen(c.strong, width=2))
        self.band.setBrush(pg.mkBrush(*pg.mkColor(c.fit).getRgb()[:3], 45))
        for line in self.band.lines:
            line.setPen(pg.mkPen(c.fit, width=1))
        for line in self.fn_lines:
            line.setPen(pg.mkPen(c.muted, width=1, style=QtCore.Qt.PenStyle.DotLine))
        self.half_item.setSymbolPen(pg.mkPen(c.fit, width=1.5))
        self.half_item.setSymbolBrush(pg.mkBrush(c.background))
        for item in self.drive_items:
            item.setSymbolPen(pg.mkPen(c.force, width=2))
            item.setSymbolBrush(None)
        self.nyq_drive.setPen(pg.mkPen(c.force, width=2))
        self.nyq_drive.setBrush(None)
        self.nyq_item.setPen(pg.mkPen(c.strong, width=1.5))
        self.nyq_dots.setPen(None)
        self.nyq_dots.setBrush(pg.mkBrush(c.mass[0]))
        self.nyq_fn.setPen(None)
        self.nyq_fn.setBrush(pg.mkBrush(c.fit))
        self.plane_item.setPen(pg.mkPen(c.mass[0], width=1.5))
        self.plane_start.setPen(None)
        self.plane_start.setBrush(pg.mkBrush(c.force))
        for ray in self.rays:
            ray.setPen(pg.mkPen(c.faint, width=1, style=QtCore.Qt.PenStyle.DashLine))
        self.arc.setPen(pg.mkPen(c.faint, width=1))
        self.pole_item.setPen(pg.mkPen(c.force, width=2.5))
        self.pole_item.setBrush(pg.mkBrush(c.force))
        self.redraw()

    def redraw(self) -> None:
        self._show_info()
        self._draw_free()
        self._draw_forced()
        self._draw_frf()
        self._draw_poles()
        settle_plots(self)

    def _show_info(self) -> None:
        o = self.osc
        z = o.zeta
        parts = [f"f<sub>n</sub> = {o.natural_hz:.4g} Hz, ζ = {z:.4g} (c<sub>c</sub> = {o.critical:.4g} N·s/m)."]
        if 0 < z < 1:
            parts.append(f"f<sub>d</sub> = {o.damped_hz:.4g} Hz, τ = {o.time_constant:.3g} s, δ = {o.log_decrement:.3g},"
                         f" Q = {1 / (2 * z):.3g}.")
        elif z == 0:
            parts.append("Undamped: it rings for ever.")
        elif z >= 1:
            what = "Critically damped" if abs(z - 1) < 1e-6 else "Overdamped"
            parts.append(f"{what}: no oscillation; slowest time constant {o.time_constant:.3g} s.")
        self.info.setText(" ".join(parts))

    def _free_span(self) -> float:
        o = self.osc
        if o.zeta < 1 and o.damping > 0:
            return min(max(5.0 * o.time_constant, 3.0 * o.period), 60.0 * o.period)
        if o.damping == 0:
            return 10.0 * o.period
        return min(6.0 * o.time_constant, 60.0 * o.period)

    def _draw_free(self) -> None:
        o = self.osc
        x0, v0 = 1e-3 * self.x0.value(), 1e-3 * self.v0.value()
        span = self._free_span()
        if self.compare.isChecked():
            # A span that shows the slowest of the compared ones settling too.
            span = max(span, min(6.0 * Oscillator(o.mass, o.stiffness, COMPARE_ZETAS[-1] * o.critical).time_constant,
                                 60.0 * o.period), 5.0 / (COMPARE_ZETAS[0] * o.omega_n))
        t = np.linspace(0.0, span, POINTS)
        x, v = free_response(o, x0, v0, t)
        self.free_item.setData(t, x)
        if 0 < o.zeta < 1:
            amp = math.hypot(x0, (v0 + o.zeta * o.omega_n * x0) / o.omega_d)
            env = amp * np.exp(-o.zeta * o.omega_n * t)
            self.envelope_items[0].setData(t, env)
            self.envelope_items[1].setData(t, -env)
        else:
            for item in self.envelope_items:
                item.setData([], [])
        pt, px = peaks(t, x)
        self.peak_item.setData(pt[:40], px[:40])
        self.free_legend.clear()
        if self.compare.isChecked():
            self.free_legend.addItem(self.free_item, f"ζ = {o.zeta:.3g}")
            for item, z in zip(self.compare_items, COMPARE_ZETAS):
                if abs(z - o.zeta) < 1e-3:  # it's the oscillator's own: drawn already
                    item.setData([], [])
                    continue
                xc, _ = free_response(Oscillator(o.mass, o.stiffness, z * o.critical), x0, v0, t)
                item.setData(t, xc)
                self.free_legend.addItem(item, f"ζ = {z:g}")
        else:
            for item in self.compare_items:
                item.setData([], [])
        self.free_plot.setXRange(0, span, padding=0)
        lines = []
        d = measured_decrement(t, x)
        if d is not None and 0 < o.zeta < 1:
            lines.append(f"Measured from {min(len(px), 40)} peaks: δ = {d:.4g}, so ζ = δ/√(4π² + δ²) = "
                         f"{d / math.sqrt(4 * math.pi**2 + d * d):.4g} (the oscillator's ζ is {o.zeta:.4g}). "
                         f"Each peak is {math.exp(d):.4g} times the next.")
        elif o.zeta == 0:
            lines.append(f"No damping: it swings at {o.natural_hz:.4g} Hz for ever, its energy "
                         f"½kx₀² + ½mv₀² = {fmt(0.5 * o.stiffness * x0**2 + 0.5 * o.mass * v0**2, 'J')} kept.")
        elif o.zeta >= 1:
            s = o.poles().real
            lines.append(f"No oscillation: the poles are real, s = {s[0]:.4g} and {s[1]:.4g} s⁻¹.")
        self.free_note.setText(" ".join(lines))
        # the phase plane
        self.plane_item.setData(x, v / o.omega_n)
        self.plane_start.setData([x0], [v0 / o.omega_n])

    def _draw_forced(self) -> None:
        o = self.osc
        f = self.freq.value()
        F = float(self.force_amplitude(f))
        x0 = v0 = 0.0  # from rest
        T = 1.0 / f
        if o.damping > 0:
            span = max(10.0 * T, min(6.0 * o.time_constant, 80.0 * T))
            span = max(span, 4.0 * o.period)
        else:
            beat = abs(f - o.natural_hz)
            span = 30.0 * T if beat < 1e-9 * f else min(max(2.2 / beat, 10.0 * T), 400.0 * T)
        span = min(span, 600.0)
        t = np.linspace(0.0, span, max(POINTS, int(40 * span * max(f, o.natural_hz))))
        t = t if t.size <= 40000 else np.linspace(0.0, span, 40000)
        x, steady = harmonic_from_rest(o, F, f, t, x0, v0)
        self.forced_item.setData(t, x)
        self.steady_item.setData(t, steady, connect="finite")
        self.forced_plot.setXRange(0, span, padding=0)
        h = receptance(o, np.array([f]))[0]
        r = f / o.natural_hz
        if np.isfinite(h):
            X = abs(F * h)
            lag = -math.degrees(math.atan2(h.imag, h.real))
            what = (f"Unbalance force m<sub>e</sub>eΩ² = {fmt(F, 'N')}. " if self.kind.currentIndex() == UNBALANCE
                    else "")
            text = (f"{what}Steady amplitude {fmt(X, 'm')} at r = f/f<sub>n</sub> = {r:.3g}, lagging the force by "
                    f"{lag:.1f}°: {X * o.stiffness / F:.3g} times the static {fmt(F / o.stiffness, 'm')}." if F > 0
                    else "No force.")
            if o.damping == 0 and F > 0:
                text += (f" Undamped, the start never dies away: it beats every {1 / abs(f - o.natural_hz):.3g} s, "
                         f"at |f − f<sub>n</sub>| = {abs(f - o.natural_hz):.3g} Hz.")
        else:
            text = (f"At resonance with no damping there is no steady state: the swing grows by "
                    f"F/2mω<sub>n</sub> = {fmt(F / (2 * o.mass * o.omega_n), 'm')} every second.")
        self.forced_note.setText(text)

    def _draw_frf(self) -> None:
        o = self.osc
        fr = frequency_grid(o)
        H = receptance(o, fr)
        unbalance = self.kind.currentIndex() == UNBALANCE
        # what the plot shows: |X|k/F, or m|X|/m_e e (= m Ω² |H|)
        norm = o.mass * (TWO_PI * fr) ** 2 if unbalance else o.stiffness + 0 * fr
        amp = np.abs(H) * norm
        self.mag_plot.setLabel("left", "m|X|/mₑe" if unbalance else "Amplification |X|k/F")
        self.mag_item.setData(fr, amp)
        lag = -np.degrees(np.angle(H))
        lag = np.where(lag < -1e-9, lag + 360.0, lag)
        self.lag_item.setData(fr, lag)
        lf = math.log10(o.natural_hz)
        for line in self.fn_lines:
            line.setPos(lf)
        top = float(np.nanmax(amp[np.isfinite(amp)])) if np.isfinite(amp).any() else 1.0
        bottom = float(np.nanmin(amp[amp > 0]))
        self.mag_plot.setXRange(math.log10(fr[0]), math.log10(fr[-1]), padding=0)
        self.mag_plot.setYRange(math.log10(max(bottom, top * 1e-6)), math.log10(top * 2.0), padding=0)
        half = o.half_power_hz()
        if half is not None and not unbalance:
            self.band.setRegion((math.log10(half[0]), math.log10(half[1])))
            self.band.setVisible(True)
            peak = abs(receptance(o, np.array([o.peak_hz()]))[0]) * o.stiffness
            self.half_item.setData([half[0], half[1]], [peak / math.sqrt(2)] * 2)
        else:
            self.band.setVisible(False)
            self.half_item.setData([], [])
        f = self.freq.value()
        hd = receptance(o, np.array([f]))[0]
        if np.isfinite(hd):
            nd = o.mass * (TWO_PI * f) ** 2 if unbalance else o.stiffness
            lg = -math.degrees(math.atan2(hd.imag, hd.real))
            self.drive_items[0].setData([f], [abs(hd) * nd])
            self.drive_items[1].setData([f], [lg + 360.0 if lg < -1e-9 else lg])
        else:
            for item in self.drive_items:
                item.setData([], [])
        if half is not None and not unbalance:
            self.frf_note.setText(
                f"Half-power points {half[0]:.4g} and {half[1]:.4g} Hz, {half[1] - half[0]:.3g} Hz apart: "
                f"Δf/2f<sub>n</sub> = {(half[1] - half[0]) / (2 * o.natural_hz):.4g} (ζ = {o.zeta:.4g}). "
                f"Peak {top:.4g} times static, at {o.peak_hz():.4g} Hz.")
        elif unbalance:
            self.frf_note.setText(f"Far above resonance m|X|/m<sub>e</sub>e → 1: the machine swings by "
                                  f"m<sub>e</sub>e/m = {fmt(self.amplitude.value() / o.mass, 'm')}.")
        elif o.damping == 0:
            self.frf_note.setText("No damping: the peak is infinite, and the phase jumps from 0° to 180° at "
                                  "f<sub>n</sub>.")
        else:
            self.frf_note.setText("ζ ≥ 1/√2: no resonant peak, so no half-power band.")
        # Nyquist: a 0.02 Hz grid for the dots, the fine grid for the line
        finite = np.isfinite(H)
        self.nyq_item.setData(H.real[finite], H.imag[finite])
        lo, hi = max(0.02, 0.3 * o.natural_hz), 2.0 * o.natural_hz
        fd = np.arange(math.ceil(lo / 0.02) * 0.02, hi, 0.02)
        Hd = receptance(o, fd)
        ok = np.isfinite(Hd)
        self.nyq_dots.setData(Hd.real[ok], Hd.imag[ok])
        hn = receptance(o, np.array([o.natural_hz]))[0]
        if np.isfinite(hn):
            self.nyq_fn.setData([hn.real], [hn.imag])
        else:
            self.nyq_fn.setData([], [])
        self.nyq_drive.setData([hd.real] if np.isfinite(hd) else [], [hd.imag] if np.isfinite(hd) else [])
        if o.damping > 0:
            d = 1.0 / (o.damping * o.omega_n)
            self.nyq_plot.setRange(xRange=(-0.65 * d, max(0.65 * d, 1.1 / o.stiffness)), yRange=(-1.1 * d, 0.1 * d),
                                   padding=0)
            self.nyq_note.setText(f"Diameter of the circle ≈ 1/cω<sub>n</sub> = {fmt(d, 'm/N')}; the diamond is "
                                  f"f<sub>n</sub>, where the dots (0.02 Hz apart) are furthest apart.")
        else:
            self.nyq_plot.autoRange()
            self.nyq_note.setText("No damping: H is real, running off to ±∞ at f<sub>n</sub>.")

    def _draw_poles(self) -> None:
        o = self.osc
        s = o.poles()
        self.pole_item.setData(s.real, s.imag)
        wn = o.omega_n
        reach = max(1.3 * wn, 1.1 * float(np.max(np.abs(s.real))))
        for ray, z in zip(self.rays, (0.05, 0.2, 0.5, 0.8)):
            a = math.asin(z)
            ray.setData([0.0, -reach * math.sin(a), 0.0, -reach * math.sin(a)],
                        [0.0, reach * math.cos(a), 0.0, -reach * math.cos(a)], connect="pairs")
        th = np.linspace(math.pi / 2, 3 * math.pi / 2, 200)
        self.arc.setData(wn * np.cos(th), wn * np.sin(th))
        self.s_plot.setRange(xRange=(-reach, 0.15 * reach), yRange=(-1.2 * wn, 1.2 * wn), padding=0.02)
        if o.zeta < 1:
            self.pole_note.setText(f"s = {s[0].real:.4g} ± {abs(s[0].imag):.4g}i s⁻¹: decay rate ζω<sub>n</sub> = "
                                   f"{-s[0].real:.4g} s⁻¹, frequency ω<sub>d</sub> = {abs(s[0].imag):.4g} rad/s. "
                                   f"|s| = ω<sub>n</sub> = {wn:.4g} rad/s.")
        else:
            self.pole_note.setText(f"Two real poles, {s[0].real:.4g} and {s[1].real:.4g} s⁻¹ (their product is "
                                   f"ω<sub>n</sub>² = {wn * wn:.4g}).")
