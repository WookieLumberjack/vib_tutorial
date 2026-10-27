"""Transient response page: an impulse, a step, a ramp or a pulse on one oscillator, and the convolution behind it.

Left: the oscillator and the input. Centre: the force, the response (exact) and, at a chosen instant, the
convolution (Duhamel's) integral drawn out: the force, the impulse response flipped and slid along, and
their product, whose area is the response. Right: the pulse spectrum (the largest response against the
pulse's length), the impulse response beside its Fourier transform, the step response's overshoot and
settling, and the Theory. The page has its own model (core/transient.py on core/sdof.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.sdof import Oscillator, impulse_response, receptance, step_response
from ..core.transient import (
    Pulse,
    PulseKind,
    convolution,
    impulse_spectrum,
    peak_ratio,
    pulse_response,
    step_metrics,
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
from .links import before, takeaways
from .style import colors
from .theming import add_legend

KINDS = list(PulseKind)
POINTS = 2500
RATIOS = np.geomspace(0.05, 4.0, 120)  # pulse length over the natural period, on the pulse spectrum
SPECTRUM_KINDS = (PulseKind.RECTANGLE, PulseKind.HALF_SINE, PulseKind.RAMP)
SLIDER_STEPS = 1000
TAB_SPECTRUM, TAB_FOURIER, TAB_STEP = range(3)


@dataclass(frozen=True)
class TransientTrial:
    name: str
    note: str
    kind: PulseKind
    amplitude: float = 1.0
    duration: float = 0.1
    damping: float = 2.0
    at: float = 0.3  # s, the convolution's instant
    tab: int = TAB_SPECTRUM


TRIALS = (
    TransientTrial(
        "The impulse response",
        "A blow of 0.1 N·s on the 1 kg oscillator (f<sub>n</sub> = 3.18 Hz, ζ = 0.05) starts it at "
        "0.1 m/s from where it is: the free vibration h(t) times 0.1, its first swing about I/mω<sub>n</sub> = "
        "5 mm. The h and H tab sets h(t) beside its Fourier transform, the receptance.",
        PulseKind.IMPULSE, amplitude=0.1, at=0.4, tab=TAB_FOURIER),
    TransientTrial(
        "A step: overshoot and settling",
        "1 N switched on and left. The mass overshoots its static 2.5 mm by e<sup>−ζπ/√(1 − ζ²)</sup> = 85 %, "
        "to 4.64 mm, at π/ω<sub>d</sub> = 0.157 s, and settles within 2 % after 3.8 s (the rule 4/ζω<sub>n</sub> "
        "says 4 s). The Step response tab draws it.",
        PulseKind.STEP, at=0.157, tab=TAB_STEP),
    TransientTrial(
        "A step, critically damped",
        "c = 40 N·s/m (ζ = 1): the step response rises to 2.5 mm without overshooting, within 2 % after "
        "0.29 s, far sooner than the lightly damped one settles.",
        PulseKind.STEP, damping=40.0, at=0.1, tab=TAB_STEP),
    TransientTrial(
        "Watch the convolution build",
        "A 0.2 s rectangular pulse of 1 N. The bottom plot is the integral at the chosen instant: the force, "
        "the impulse response flipped and slid to that instant, and their product, whose signed area is the "
        "response there. Drag the slider: the area follows the response curve.",
        PulseKind.RECTANGLE, duration=0.2, at=0.3, tab=TAB_SPECTRUM),
    TransientTrial(
        "A short pulse is an impulse",
        "10 N for 0.0157 s, a twentieth of the period. Only its area, 0.157 N·s, matters: the response is "
        "that impulse's, peaking at 2 sin(π/20) = 0.31 times the static 25 mm (undamped). On the pulse spectrum "
        "the point sits on the rising, impulse-like part, where the peak is proportional to the length.",
        PulseKind.RECTANGLE, amplitude=10.0, duration=0.0157, at=0.08, tab=TAB_SPECTRUM),
    TransientTrial(
        "Half a period or longer: twice the static",
        "A 1 N rectangle 0.16 s long, half the period. Undamped, a rectangle of half a period or more throws "
        "the mass to twice its static deflection, as a step would: the most a rectangle can do. Its "
        "pulse-spectrum curve rises to 2 at T = T<sub>n</sub>/2 and stays there.",
        PulseKind.RECTANGLE, duration=0.157, at=0.2, tab=TAB_SPECTRUM),
    TransientTrial(
        "A half-sine's worst length",
        "A 1 N half-sine 0.26 s long, about 0.8 of the period: undamped, its peak would be 1.77 times "
        "static, the half-sine curve's top; with ζ = 0.05, 1.65. Shorter, it is an impulse; much longer, "
        "the mass follows it statically.",
        PulseKind.HALF_SINE, duration=0.26, at=0.25, tab=TAB_SPECTRUM),
    TransientTrial(
        "A slow ramp is nearly static",
        "1 N reached over 0.87 s, 2.75 periods. The mass follows the ramp, lagging, and overshoots the end by "
        "only |sin πr|/πr = 8 % of the static 2.5 mm (undamped, r = T/T<sub>n</sub>; 6 % with ζ = 0.05). "
        "A ramp of a whole number of periods leaves no swing at all.",
        PulseKind.RAMP, duration=0.864, at=0.6, tab=TAB_SPECTRUM),
)

TIPS = {
    "kind": "<p><b>Impulse:</b> a blow of impulse I (N·s), over in an instant. <b>Step:</b> F switched on and "
            "left on. <b>Ramp, then hold:</b> F reached in a straight line over the time T. <b>Rectangular "
            "pulse:</b> F for T. <b>Half-sine pulse:</b> F sin(πt/T) for T.</p>",
    "amplitude": "<p>The force F, or the impulse I = ∫F dt for an impulse.</p>",
    "duration": "<p>The ramp's rise time, or the pulse's length T.</p>",
    "at": "<p>The instant t at which the convolution integral is drawn out (bottom plot).</p>",
}
SPECTRUM_TIP = (
    "<p>The largest |x| over all time, over the static F/k, against the pulse's length over the natural "
    "period, for this damping ratio. Short pulses are impulses (the peak grows with the area); long ones are "
    "static, a rectangle's at twice and a ramp's at once the static deflection. The circle is this page's "
    "pulse.</p>")
FOURIER_TIP = (
    "<p>Top: the impulse response h(t). Bottom: its Fourier transform by the FFT (dots) on the receptance "
    "|H(f)| = 1/|k − ω²m + iωc| (line). They are one function, seen in time and in frequency.</p>")
STEP_TIP = (
    "<p>The response to a unit step over its final value 1/k, with the ±2 % band. The marks are the overshoot "
    "and the settling time (the last time it is outside the band).</p>")
CONV_TIP = (
    "<p>The convolution x(t) = ∫₀ᵗ F(τ) h(t − τ) dτ at the chosen t: each slice F(τ)dτ of the force is a small "
    "impulse, which rings as h from then on; at t it has rung for t − τ. Their product (shaded) summed is x(t). "
    "F and h are scaled to the plot.</p>")

NOTES_HTML = """
<!--BEFORE-->
<h3><a name="impulse-response"></a>An impulse and the impulse response</h3>
<p>A hammer blow lasts far less than a period. All that it leaves behind is its <b>impulse</b>
I = ∫F dt: it changes the mass's momentum by I before the spring has time to stretch, so the mass starts
from where it was with velocity I/m and then vibrates freely. For a unit impulse (1 N·s) on a mass at rest
this free vibration is the <b>impulse response</b></p>
<p align="center">h(t) = e<sup>−ζω<sub>n</sub>t</sup> sin(ω<sub>d</sub>t) / mω<sub>d</sub> &nbsp; (t ≥ 0),</p>
<p>and 0 before the blow: nothing moves until it is struck (causality).</p>

<h3><a name="convolution"></a>Convolution: Duhamel's integral</h3>
<p>Any force is a string of small impulses: in the moment dτ at τ it gives the impulse F(τ)dτ. Each rings
as its own impulse response, which at the later time t has been ringing for t − τ. The system is linear,
so the responses add:</p>
<p align="center"><b>x(t) = ∫₀ᵗ F(τ) h(t − τ) dτ.</b></p>
<p>This is <b>Duhamel's integral</b>, the convolution of the force with h. The bottom plot draws it at one
instant: h is flipped (time runs backwards from t) and slid along, multiplied by the force, and the shaded
area is x(t). Recent force counts most; force from long ago has rung down. Drag the slider and the area
traces out the response.</p>
<p>So h says everything about the system: knowing it, every response follows. (This page steps the
response exactly instead, and sets the integral, worked out numerically, beside it: they agree to the
integral's accuracy.)</p>

<h3><a name="fourier-pair"></a>h and H: one function, two views</h3>
<p>Convolution in time is multiplication in frequency. Transform Duhamel's integral and it becomes
X(ω) = H(ω)F(ω): the <b>frequency response is the Fourier transform of the impulse response</b>,
H(ω) = ∫h(t)e<sup>−iωt</sup>dt. A long, lightly damped ring in time is a tall, narrow peak in frequency; a
quick decay is a broad, low one. The h and H tab transforms h by the FFT and lays it on the receptance from the
Single oscillator page. This is why a hammer test works: one blow excites every frequency at once, and the
ring that follows holds the whole H.</p>

<h3><a name="step-response"></a>The step response</h3>
<p>A force F switched on and left on is the integral of h:</p>
<p align="center">x(t) = (F/k)[1 − e<sup>−ζω<sub>n</sub>t</sup>(cos ω<sub>d</sub>t + (ζω<sub>n</sub>/ω<sub>d</sub>)
sin ω<sub>d</sub>t)].</p>
<p>It rises towards the static F/k but overshoots it, by the fraction</p>
<p align="center">M<sub>p</sub> = e<sup>−ζπ/√(1 − ζ²)</sup>,</p>
<p>at the <b>peak time</b> π/ω<sub>d</sub>, and then rings down about it. Undamped it swings to <b>twice</b> the
static deflection: suddenly applied, a load does double the damage it does when placed gently. It
<b>settles</b> within 2 % after about 4/ζω<sub>n</sub> (four time constants). Critically damped, it rises
without overshoot; overdamped, slowly.</p>

<h3><a name="laplace"></a>The Laplace transform and the transfer function</h3>
<p>The Laplace transform turns the equation of motion into algebra. With x(0) = x<sub>0</sub> and
ẋ(0) = v<sub>0</sub>,</p>
<p align="center">(ms² + cs + k)X(s) = F(s) + m(sx<sub>0</sub> + v<sub>0</sub>) + cx<sub>0</sub>,</p>
<p>so X(s) = H(s)F(s) plus a term from the start, with the <b>transfer function</b>
H(s) = 1/(ms² + cs + k). Its poles are the roots of ms² + cs + k, the s-plane points on the
<a href="page:sdof_page#poles">Single oscillator page</a>: they set the free response's decay and frequency.
On the imaginary axis, s = iω, H(s) is the frequency response; its inverse transform is h(t). A step is
F/s, so its response is F/s(ms² + cs + k), which partial fractions turn back into the formula above.</p>

<h3><a name="pulses"></a>Pulses: impulse, step, or in between</h3>
<p>What a pulse of length T does depends on T against the natural period T<sub>n</sub>:</p>
<ul>
<li><b>T ≪ T<sub>n</sub>:</b> it is an impulse. Only its area counts, and the peak grows with it: a rectangle
reaches 2 sin(πT/T<sub>n</sub>) times static, about 2πT/T<sub>n</sub>.</li>
<li><b>T ≫ T<sub>n</sub>:</b> the mass follows it statically: a ramp's peak tends to 1 (its overshoot is
|sin πr|/πr, r = T/T<sub>n</sub>, and none at all for a whole number of periods); a rectangle, which is
switched on and off suddenly, still throws it to 2.</li>
<li><b>In between:</b> the worst case. A half-sine is worst at T ≈ 0.8T<sub>n</sub>, 1.77 times static
undamped.</li>
</ul>
<p>The <i>Pulse spectrum</i> tab plots this peak ratio against T/T<sub>n</sub>: it is a shock response
spectrum. The <a href="page:shock_page#srs">Shock spectrum page</a> turns it round: one pulse, a bank of
oscillators of every frequency.</p>

<h3><a name="computing"></a>How this page computes it</h3>
<p>An impulse's response is the free vibration from v = I/m, a step's and a rectangle's are sums of step
responses, all in closed form. The ramp and the half-sine are stepped by the exact update the chain's
simulator uses, which is exact for a force linear between steps (the ramp) and accurate to a few millionths
for the half-sine on 200 steps a period. The convolution integral is worked out separately by the trapezium
rule, to show that it gives the same.</p>

<h3><a name="try-it"></a>Things to try</h3>
<p>Each of these sets up the oscillator, the input and the plots (Things to try, top left).</p>
<!--TRIALS-->
<!--TAKEAWAYS-->
"""
NOTES_HTML = NOTES_HTML.replace("<!--BEFORE-->", before(
    ("sdof_page", "free-vibration", "Free vibration"),
    ("sdof_page", "poles", "The poles"),
    ("sdof_page", "forced-response", "Forced response"),
)).replace("<!--TRIALS-->", trials_html(TRIALS)).replace("<!--TAKEAWAYS-->", takeaways(
    "An impulse I starts the mass at I/m; the response to a unit impulse, h(t), is a free vibration.",
    "Any response is the convolution of the force with h (Duhamel's integral): every slice of force rings "
    "as h from its own moment on.",
    "h(t) and H(ω) are a Fourier pair; with the Laplace transform, H(s) = 1/(ms² + cs + k) and its poles.",
    "A step overshoots its static value by e<sup>−ζπ/√(1 − ζ²)</sup> (twice the static deflection, "
    "undamped) and settles in about 4/ζω<sub>n</sub>.",
    "A pulse much shorter than the period acts by its area, one much longer statically; the worst lengths "
    "lie near half a period to a period.",
))


class TransientPage(QtWidgets.QWidget):
    """Controls on the left, force, response and convolution in the centre, spectra and theory on the right."""

    follows_chain = False  # MainWindow: this page has its own model, not the chain

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.osc = Oscillator()
        self.pulse = Pulse()
        self._spectrum_key = None  # (zeta) the pulse spectrum was drawn for
        self._span = 1.0

        self.trials = TrialBox([t.name for t in TRIALS],
                               "<p>Each sets up the oscillator and the input to show one idea from the Theory "
                               "tab.</p>")
        self.trials.picked.connect(self.load_trial)

        box = Form("Oscillator")
        self.mass = box.number("Mass m:", 0.01, 1e4, 1.0, 3, " kg", "<p>The mass m.</p>")
        self.stiffness = box.number("Stiffness k:", 0.01, 1e8, 400.0, 2, " N/m", "<p>The spring's stiffness k.</p>")
        self.damping = box.number("Damping c:", 0.0, 1e6, 2.0, 3, " N·s/m", "<p>The damper's coefficient c.</p>")
        self.info = note_label()
        box.row(self.info)
        box.changed.connect(self._on_inputs)

        inp = Form("Input")
        self.kind = QtWidgets.QComboBox()
        self.kind.addItems([k.value for k in KINDS])
        inp.add("Force:", self.kind, TIPS["kind"])
        self.amplitude = inp.number("Amplitude F:", 0.0, 1e6, 1.0, 4, " N", TIPS["amplitude"])
        self.amplitude_label = inp.form.labelForField(self.amplitude)
        self.duration = inp.number("Length T:", 1e-4, 100.0, 0.1, 4, " s", TIPS["duration"])
        self.duration_label = inp.form.labelForField(self.duration)
        self.ratio_note = note_label()
        inp.row(self.ratio_note)
        inp.changed.connect(self._on_inputs)
        self.kind.currentIndexChanged.connect(self._on_inputs)

        conv = Form("Convolution")
        self.at = conv.number("At t =", 0.0, 1e4, 0.3, 4, " s", TIPS["at"])
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, SLIDER_STEPS)
        self.slider.setToolTip(TIPS["at"])
        conv.row(self.slider)
        self.at.valueChanged.connect(self._on_at)
        self.slider.valueChanged.connect(self._on_slider)

        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        for w in (self.trials, box, inp, conv):
            lv.addWidget(w)
        lv.addStretch(1)

        # --- centre
        self.force_plot = plot("Force F", "N", "Time", "s", height=90)
        self.force_item = pg.PlotDataItem()
        self.impulse_item = pg.PlotDataItem()  # an arrow-like stem for an impulse
        self.force_plot.addItem(self.force_item)
        self.force_plot.addItem(self.impulse_item)
        self.x_plot = plot("Displacement x", "m", "Time", "s", height=140)
        self.x_plot.setXLink(self.force_plot)
        add_legend(self.x_plot.getPlotItem(), offset=(-10, 10))
        self.static_item = pg.PlotDataItem(name="static F/k")
        self.x_item = pg.PlotDataItem(name="x(t)")
        self.at_line = pg.InfiniteLine(angle=90)
        self.at_dot = pg.ScatterPlotItem(size=10)
        for item in (self.static_item, self.x_item, self.at_dot):
            self.x_plot.addItem(item)
        self.x_plot.addItem(self.at_line)
        self.force_plot.addItem(at_line2 := pg.InfiniteLine(angle=90))
        self.at_line2 = at_line2
        self.conv_plot = plot("Scaled", None, "τ", "s", height=140)
        self.conv_plot.setToolTip(CONV_TIP)
        add_legend(self.conv_plot.getPlotItem(), offset=(-10, 10))
        self.prod_zero = pg.PlotDataItem()
        self.prod_item = pg.PlotDataItem(name="F(τ) h(t − τ)")
        self.prod_fill = pg.FillBetweenItem(self.prod_item, self.prod_zero)
        self.cf_item = pg.PlotDataItem(name="F(τ)")
        self.ch_item = pg.PlotDataItem(name="h(t − τ)")
        for item in (self.prod_fill, self.prod_zero, self.prod_item, self.cf_item, self.ch_item):
            self.conv_plot.addItem(item)
        self.conv_note = readout_label()

        center = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(center)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.addWidget(self.force_plot, 2)
        cv.addWidget(self.x_plot, 3)
        cv.addWidget(QtWidgets.QLabel("<b>Duhamel's integral at the chosen t</b>"))
        cv.addWidget(self.conv_plot, 3)
        cv.addWidget(self.conv_note)

        # --- right
        spec = QtWidgets.QWidget()
        sv = QtWidgets.QVBoxLayout(spec)
        self.spec_plot = plot("Peak |x| k/F", None, "Pulse length / natural period", None, log_x=True, height=200)
        self.spec_plot.setToolTip(SPECTRUM_TIP)
        self.spec_plot.setMouseEnabled(x=False, y=False)
        add_legend(self.spec_plot.getPlotItem(), offset=(10, 10))
        self.spec_items = {k: pg.PlotDataItem(name=k.value.replace(" pulse", "").replace(", then hold", ""))
                           for k in SPECTRUM_KINDS}
        for item in self.spec_items.values():
            self.spec_plot.addItem(item)
        self.spec_now = pg.PlotDataItem(pen=None, symbol="o", symbolSize=12)  # a scatter item ignores log axes
        self.spec_plot.addItem(self.spec_now)
        sv.addWidget(self.spec_plot, 1)
        self.spec_note = note_label()
        sv.addWidget(self.spec_note)

        four = QtWidgets.QWidget()
        fv = QtWidgets.QVBoxLayout(four)
        self.h_plot = plot("h(t)", "m/(N·s)", "Time", "s", height=120)
        self.h_item = pg.PlotDataItem()
        self.h_plot.addItem(self.h_item)
        self.H_plot = plot("|H|", "m/N", "Frequency", "Hz", log_x=True, log_y=True, height=150)
        self.H_plot.setToolTip(FOURIER_TIP)
        self.h_plot.setToolTip(FOURIER_TIP)
        add_legend(self.H_plot.getPlotItem(), offset=(-10, 10))
        self.H_exact = pg.PlotDataItem(name="receptance")
        self.H_fft = pg.PlotDataItem(pen=None, symbol="o", symbolSize=4, name="FFT of h")
        self.H_plot.addItem(self.H_exact)
        self.H_plot.addItem(self.H_fft)
        fv.addWidget(self.h_plot, 2)
        fv.addWidget(self.H_plot, 3)
        self.fourier_note = note_label()
        fv.addWidget(self.fourier_note)

        step = QtWidgets.QWidget()
        stv = QtWidgets.QVBoxLayout(step)
        self.step_plot = plot("x k / F", None, "Time", "s", height=200)
        self.step_plot.setToolTip(STEP_TIP)
        self.band_lines = [pg.InfiniteLine(angle=0, pos=v) for v in (0.98, 1.02)]
        self.step_item = pg.PlotDataItem()
        self.step_marks = pg.ScatterPlotItem(size=9)
        self.settle_line = pg.InfiniteLine(angle=90)
        for item in (*self.band_lines, self.step_item, self.step_marks, self.settle_line):
            self.step_plot.addItem(item)
        stv.addWidget(self.step_plot, 1)
        self.step_note = note_label()
        stv.addWidget(self.step_note)

        self.notes = QtWidgets.QTextBrowser()
        self.notes.setHtml(NOTES_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(spec, "Pulse spectrum")
        self.tabs.addTab(four, "h and H")
        self.tabs.addTab(step, "Step response")
        self.tabs.addTab(self.notes, "Theory")
        self.tabs.currentChanged.connect(lambda _: self._draw_tab())

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
        for box, value in ((self.mass, 1.0), (self.stiffness, 400.0), (self.damping, t.damping),
                           (self.kind, KINDS.index(t.kind)), (self.amplitude, t.amplitude),
                           (self.duration, t.duration), (self.at, t.at)):
            set_quietly(box, value)
        self.trials.show_note(t.note)
        self.tabs.setCurrentIndex(t.tab)
        self._on_inputs()

    def _on_inputs(self) -> None:
        self.osc = Oscillator(self.mass.value(), self.stiffness.value(), self.damping.value())
        kind = KINDS[self.kind.currentIndex()]
        self.pulse = Pulse(kind, self.amplitude.value(), self.duration.value())
        impulse = kind is PulseKind.IMPULSE
        self.amplitude_label.setText("Impulse I:" if impulse else "Amplitude F:")
        self.amplitude.setSuffix(" N·s" if impulse else " N")
        self.duration_label.setText("Rise time T:" if kind is PulseKind.RAMP else "Length T:")
        self.duration.setEnabled(self.pulse.has_duration)
        self.duration_label.setEnabled(self.pulse.has_duration)
        self.redraw()

    def _on_at(self, value: float) -> None:
        set_quietly(self.slider, round(SLIDER_STEPS * min(value / self._span, 1.0)))
        self._draw_convolution()
        settle_plots(self)

    def _on_slider(self, value: int) -> None:
        set_quietly(self.at, round(self._span * value / SLIDER_STEPS, 4))
        self._draw_convolution()
        settle_plots(self)

    # ------------------------------------------------------------- drawing
    def apply_theme(self) -> None:
        c = colors
        self.force_item.setPen(pg.mkPen(c.force, width=2))
        self.impulse_item.setPen(pg.mkPen(c.force, width=3))
        self.x_item.setPen(pg.mkPen(c.mass[0], width=2))
        self.static_item.setPen(pg.mkPen(c.grey, width=1, style=QtCore.Qt.PenStyle.DashLine))
        for line in (self.at_line, self.at_line2):
            line.setPen(pg.mkPen(c.muted, width=1, style=QtCore.Qt.PenStyle.DotLine))
        self.at_dot.setPen(pg.mkPen(c.strong, width=2))
        self.at_dot.setBrush(None)
        self.cf_item.setPen(pg.mkPen(c.force, width=1.5))
        self.ch_item.setPen(pg.mkPen(c.mode[1], width=1.5))
        self.prod_item.setPen(pg.mkPen(c.mass[0], width=1.5))
        self.prod_zero.setPen(pg.mkPen(c.faint, width=1))
        self.prod_fill.setBrush(pg.mkBrush(*pg.mkColor(c.mass[0]).getRgb()[:3], 70))
        for kind, color in zip(SPECTRUM_KINDS, (c.mode[0], c.mode[1], c.mode[2])):
            self.spec_items[kind].setPen(pg.mkPen(color, width=2))
        self.spec_now.setSymbolPen(pg.mkPen(c.force, width=2))
        self.spec_now.setSymbolBrush(None)
        self.h_item.setPen(pg.mkPen(c.mass[0], width=1.5))
        self.H_exact.setPen(pg.mkPen(c.strong, width=2))
        self.H_fft.setSymbolPen(None)
        self.H_fft.setSymbolBrush(pg.mkBrush(c.mass[0]))
        for line in self.band_lines:
            line.setPen(pg.mkPen(c.faint, width=1, style=QtCore.Qt.PenStyle.DashLine))
        self.step_item.setPen(pg.mkPen(c.mass[0], width=2))
        self.step_marks.setPen(None)
        self.step_marks.setBrush(pg.mkBrush(c.force))
        self.settle_line.setPen(pg.mkPen(c.fit, width=1, style=QtCore.Qt.PenStyle.DashLine))
        self._spectrum_key = None
        self.redraw()

    def redraw(self) -> None:
        o, p = self.osc, self.pulse
        z = o.zeta
        info = f"f<sub>n</sub> = {o.natural_hz:.4g} Hz (T<sub>n</sub> = {o.period:.4g} s), ζ = {z:.4g}"
        info += f", τ = {o.time_constant:.3g} s." if o.damping > 0 else ", undamped."
        self.info.setText(info)
        self.ratio_note.setText(f"T/T<sub>n</sub> = {p.duration / o.period:.3g}." if p.has_duration else "")
        self._draw_response()
        self._draw_convolution()
        self._draw_tab()
        settle_plots(self)

    def _time_span(self) -> float:
        o, p = self.osc, self.pulse
        ring = min(5.0 * o.time_constant, 12.0 * o.period) if o.damping > 0 else 6.0 * o.period
        length = p.duration if p.has_duration else 0.0
        return max(length + min(ring, max(3.0 * o.period, 1.5 * length)), 3.0 * o.period)

    def _draw_response(self) -> None:
        o, p = self.osc, self.pulse
        self._span = span = self._time_span()
        t = np.linspace(0.0, span, POINTS)
        self._t = t
        self._x = x = pulse_response(o, p, t)
        tf = np.concatenate([[-0.03 * span, -1e-9 * span], t])
        self.force_item.setData(tf, p.force(tf))
        if p.kind is PulseKind.IMPULSE:
            self.impulse_item.setData([0.0, 0.0], [0.0, 1.0])
            self.force_plot.setLabel("left", "Impulse")
            self.force_plot.setYRange(0, 1.2, padding=0)
        else:
            self.impulse_item.setData([], [])
            self.force_plot.setLabel("left", "Force F", units="N")
            self.force_plot.enableAutoRange(axis="y")
        self.force_plot.setXRange(-0.03 * span, span, padding=0)
        self.x_item.setData(t, x)
        if p.kind is not PulseKind.IMPULSE:
            self.static_item.setData([0.0, span], [p.amplitude / o.stiffness] * 2)
        else:
            self.static_item.setData([], [])
        self.at.setMaximum(span)
        set_quietly(self.slider, round(SLIDER_STEPS * min(self.at.value() / span, 1.0)))

    def _draw_convolution(self) -> None:
        o, p = self.osc, self.pulse
        at = min(self.at.value(), self._span)
        tau, f, h, xc = convolution(o, p, at)
        x_exact = float(pulse_response(o, p, np.array([0.0, at]) if at > 0 else np.array([0.0]))[-1])
        for line in (self.at_line, self.at_line2):
            line.setPos(at)
        self.at_dot.setData([at], [x_exact])
        span = self._span
        if p.kind is PulseKind.IMPULSE:
            hh = impulse_response(o, at - np.linspace(0.0, span, 600))
            tt = np.linspace(0.0, span, 600)
            scale_h = np.max(np.abs(hh)) or 1.0
            self.ch_item.setData(tt, hh / scale_h)
            self.cf_item.setData([0.0, 0.0], [0.0, 1.0])
            self.prod_item.setData([0.0, 0.0], [0.0, 0.0])
            self.prod_zero.setData([0.0, 0.0], [0.0, 0.0])
            self.conv_note.setText(f"An impulse is one slice: x(t) = I h(t) = {fmt(xc, 'm')} at t = {at:.4g} s.")
        else:
            tt = np.linspace(0.0, span, 800)
            hh = impulse_response(o, at - tt)
            ff = p.force(tt)
            scale_f = np.max(np.abs(ff)) or 1.0
            scale_h = np.max(np.abs(impulse_response(o, np.linspace(0.0, span, 800)))) or 1.0
            self.cf_item.setData(tt, ff / scale_f)
            self.ch_item.setData(tt, hh / scale_h)
            y = f * h
            scale_y = max(np.max(np.abs(y)), 1e-300)
            self.prod_item.setData(tau, y / scale_y)
            self.prod_zero.setData(tau, np.zeros_like(tau))
            self.conv_note.setText(
                f"At t = {at:.4g} s the shaded area (∫F h dτ, by the trapezium rule) is {fmt(xc, 'm')}; "
                f"the exact response is {fmt(x_exact, 'm')}.")
        self.conv_plot.setXRange(0, span, padding=0)
        self.conv_plot.setYRange(-1.1, 1.1, padding=0)

    def _draw_tab(self) -> None:
        i = self.tabs.currentIndex()
        if i == TAB_SPECTRUM:
            self._draw_spectrum()
        elif i == TAB_FOURIER:
            self._draw_fourier()
        elif i == TAB_STEP:
            self._draw_step()

    def _draw_spectrum(self) -> None:
        o, p = self.osc, self.pulse
        z = round(o.zeta, 6)
        if self._spectrum_key != z:
            self._spectrum_key = z
            for kind, item in self.spec_items.items():
                item.setData(RATIOS, peak_ratio(min(z, 5.0), kind, RATIOS))
            self.spec_plot.setXRange(math.log10(RATIOS[0]), math.log10(RATIOS[-1]), padding=0)
            self.spec_plot.setYRange(0, 2.1, padding=0)
        if p.kind in SPECTRUM_KINDS:
            r = p.duration / o.period
            peak = float(np.max(np.abs(pulse_response(o, p, np.linspace(0, p.duration + 1.05 * o.period, 4000)))))
            ratio = peak * o.stiffness / p.amplitude if p.amplitude > 0 else 0.0
            if RATIOS[0] <= r <= RATIOS[-1]:
                self.spec_now.setData([r], [ratio])
            else:
                self.spec_now.setData([], [])
            self.spec_note.setText(f"This {p.kind.value.lower()}: T/T<sub>n</sub> = {r:.3g}, peak "
                                   f"{fmt(peak, 'm')}, {ratio:.3g} times the static {fmt(p.amplitude / o.stiffness, 'm')}"
                                   f" (ζ = {o.zeta:.3g}).")
        else:
            self.spec_now.setData([], [])
            self.spec_note.setText("Pick a ramp, a rectangular or a half-sine pulse to place it on the spectrum.")

    def _draw_fourier(self) -> None:
        o = self.osc
        span = min(12.0 * o.time_constant, 400.0 * o.period) if o.damping > 0 else 200.0 * o.period
        t = np.linspace(0.0, min(span, 8.0 * o.time_constant if o.damping > 0 else 10 * o.period), 1500)
        self.h_item.setData(t, impulse_response(o, t))
        n = 2 ** 16
        f, H = impulse_spectrum(o, span, n)
        keep = (f > 0.05 * o.natural_hz) & (f < 5.0 * o.natural_hz)
        f, H = f[keep], H[keep]
        stride = max(1, f.size // 300)
        self.H_fft.setData(f[::stride], np.abs(H[::stride]))
        fe = np.geomspace(0.05 * o.natural_hz, 5.0 * o.natural_hz, 800)
        He = receptance(o, fe)
        ok = np.isfinite(He)
        self.H_exact.setData(fe[ok], np.abs(He[ok]))
        if o.damping > 0:
            peak = 1.0 / (o.damping * o.omega_n)
            self.fourier_note.setText(f"A ring of time constant {o.time_constant:.3g} s is a peak "
                                      f"{2 * o.zeta * o.natural_hz:.3g} Hz wide (2ζf<sub>n</sub>), "
                                      f"{fmt(peak, 'm/N')} high (1/cω<sub>n</sub>).")
        else:
            self.fourier_note.setText("Undamped, h rings for ever: its transform over a finite record has a "
                                      "finite peak, of the record's length, where the receptance's is infinite.")

    def _draw_step(self) -> None:
        o = self.osc
        m = step_metrics(o)
        span = min(max(1.3 * m.settling if math.isfinite(m.settling) else 10 * o.period, 3 * o.period),
                   60.0 * o.period)
        t = np.linspace(0.0, span, 3000)
        y = step_response(o, t) * o.stiffness
        self.step_item.setData(t, y)
        marks_t, marks_y = [], []
        if math.isfinite(m.peak_time) and m.peak_time <= span:
            marks_t.append(m.peak_time)
            marks_y.append(1.0 + m.overshoot)
        self.step_marks.setData(marks_t, marks_y)
        if math.isfinite(m.settling):
            self.settle_line.setPos(m.settling)
            self.settle_line.setVisible(True)
        else:
            self.settle_line.setVisible(False)
        self.step_plot.setXRange(0, span, padding=0)
        if o.zeta < 1:
            text = (f"Overshoot e<sup>−ζπ/√(1 − ζ²)</sup> = {100 * m.overshoot:.3g} % at π/ω<sub>d</sub> = "
                    f"{m.peak_time:.3g} s. ")
        else:
            text = "No overshoot (ζ ≥ 1). "
        if math.isfinite(m.settling):
            text += (f"Within 2 % from {m.settling:.3g} s (dashed); the rule 4/ζω<sub>n</sub> says "
                     f"{m.settling_estimate:.3g} s.")
        else:
            text += "Undamped, it never settles: it swings between 0 and twice the static deflection."
        self.step_note.setText(text)
