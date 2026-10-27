"""Self-excited vibration page: van der Pol, galloping, vortex-induced vibration and flutter.

Left: the model and its parameters. Centre: its motion from a small start (time history and phase plane), or for
flutter the two modes' frequencies and damping against the speed. Right: the sweep (the steady amplitude against
the parameter that drives it, which shows the Hopf point and any hysteresis), and the Theory. The page has its own
models (core/self_excited.py).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.self_excited import (
    Flutter,
    Galloping,
    VanDerPol,
    Viv,
    dominant_hz,
    galloping_branches,
    settled_amplitude,
)
from . import settle_plots
from .lesson import (
    Form,
    TrialBox,
    column,
    combo,
    connect_all,
    mark,
    marker,
    note_label,
    plot,
    readout_label,
    relabel,
    set_quietly,
    stack,
    theory,
    three_panes,
    trials_html,
)
from .links import before, takeaways
from .style import colors
from .theming import add_legend


class Kind(enum.Enum):
    VDP = "Van der Pol (negative damping)"
    GALLOP = "Galloping (a square section in wind)"
    VIV = "Vortex-induced vibration"
    FLUTTER = "Flutter (a wing section)"


KINDS = tuple(Kind)
TAB_SWEEP = 0
VIV_SWEEP = np.linspace(2.0, 14.0, 41)


@dataclass(frozen=True)
class SelfTrial:
    name: str
    note: str
    kind: Kind
    values: dict = field(default_factory=dict)  # control name → value
    tab: int = TAB_SWEEP


TRIALS = (
    SelfTrial(
        "Negative damping grows to a limit cycle",
        "ẍ + ω²x = ω(ε − x²/x<sub>r</sub>²)ẋ with ε = 0.2: at small amplitude the damping is negative (ζ = −0.1), "
        "so a 0.5 mm start grows; at large amplitude it is positive. It settles where the cycle's energy in and out "
        "balance: 8.9 mm, 2x<sub>r</sub>√ε. Start it anywhere and it ends on the same closed curve in the phase "
        "plane: a limit cycle.",
        Kind.VDP, {"vdp_eps": 0.2}),
    SelfTrial(
        "The Hopf point",
        "ε = −0.1: positive damping everywhere near rest, and the start dies away. The Sweep tab shows the settled "
        "amplitude against ε: zero below ε = 0, then 2x<sub>r</sub>√ε, rising steeply from nothing. At ε = 0 the "
        "pair of poles crosses the imaginary axis: a (supercritical) Hopf bifurcation.",
        Kind.VDP, {"vdp_eps": -0.1}),
    SelfTrial(
        "Relaxation oscillations",
        "ε = 5: the nonlinearity dominates. The motion creeps, then snaps: a period of 1.84 s against 1 s for small "
        "ε, and a squared-off wave. A dripping tap, a squealing brake and a heartbeat oscillate this way.",
        Kind.VDP, {"vdp_eps": 5.0, "vdp_periods": 20.0}),
    SelfTrial(
        "Galloping: Den Hartog's criterion",
        "A 100 mm square section (5 kg/m, 2 Hz, ζ = 0.5 %) across a 4.5 m/s wind. Moving up, it sees the wind "
        "come from above; for a square, the force that angle makes pushes it further up (Den Hartog: dC<sub>L</sub>/dα "
        "+ C<sub>D</sub> &lt; 0). The wind's negative damping ½ρUDA₁ beats the structure's 2mζω above "
        "U<sub>g</sub> = 3.81 m/s, so the motion grows, slowly this close to the onset.",
        Kind.GALLOP, {"g_speed": 4.5, "g_start": 0.001, "g_periods": 300.0}),
    SelfTrial(
        "Galloping's hysteresis",
        "Between 4.7 and 7.0 m/s a square section has two stable motions, at 6 m/s 0.51 D and 1.28 D, with an "
        "unstable one between (Sweep tab: filled and open dots). From a small start it settles on the lower; set "
        "the start to 0.12 m and it settles on the upper. Raise the wind past 7.0 m/s and the lower one is gone: "
        "the motion jumps to the upper, and stays there until the wind drops below 4.7 m/s.",
        Kind.GALLOP, {"g_speed": 6.0, "g_start": 0.001, "g_periods": 600.0}),
    SelfTrial(
        "Galloping grows with the wind",
        "At 12 m/s, three times the onset speed, the section swings 2.7 D from side to side. Unlike flutter or "
        "resonance there is no upper limit: the amplitude keeps growing with the wind. Damping only moves the "
        "onset (U<sub>g</sub> ∝ mζ, the Scruton number).",
        Kind.GALLOP, {"g_speed": 12.0, "g_start": 0.001, "g_periods": 200.0}),
    SelfTrial(
        "Lock-in",
        "A 50 mm cylinder (20 kg/m, mass ratio 10) on a 1 Hz spring in a current at reduced velocity "
        "U/f<sub>n</sub>D = 5.5. Vortices would shed at St U/D = 1.1 f<sub>n</sub>, but the structure's motion "
        "pulls the wake to its own frequency: it swings 0.21 D at f<sub>n</sub>. The Sweep tab shows lock-in, the "
        "response within 4 % of f<sub>n</sub>, from U<sub>r</sub> = 4.5 to 6.25: a band of speeds, not a single "
        "resonance.",
        Kind.VIV, {"v_reduced": 5.5}),
    SelfTrial(
        "Out of lock-in",
        "U<sub>r</sub> = 9: the wake sheds near its own Strouhal frequency, 1.8 f<sub>n</sub>, far above the "
        "structure's (the response is at 1.7 f<sub>n</sub>), and it is a forced one: 0.019 D. Vortex-induced vibration is self-limited "
        "(about one diameter at most, in practice), unlike galloping.",
        Kind.VIV, {"v_reduced": 9.0}),
    SelfTrial(
        "Flutter: two modes coalesce",
        "A wing section with plunge at 3 Hz and pitch at 6 Hz, its centre of mass behind the elastic axis. As the "
        "speed rises the lift couples the two modes: their frequencies close in (3 and 6.7 Hz at rest, 3.4 and "
        "5.0 Hz at 19 m/s) and the pitch-like mode's damping falls to zero at 19.1 m/s, the flutter speed, well "
        "below the 39 m/s at which the pitch spring alone would diverge.",
        Kind.FLUTTER, {"f_speed": 15.0}),
    SelfTrial(
        "Mass balance",
        "Move the centre of mass ahead of the elastic axis, to x<sub>cg</sub> = −0.2 semichords: the inertia now "
        "couples plunge and pitch the other way, and the section no longer flutters at all; it only diverges, at "
        "39 m/s, where the pitch spring gives out. Mass-balancing control surfaces (lead in the leading edge) "
        "does this. The trend isn't monotonic in this quasi-steady model: with the centre of mass on the axis "
        "(x<sub>cg</sub> = 0) flutter comes earlier, at 17.7 m/s.",
        Kind.FLUTTER, {"f_cg": -0.2, "f_speed": 15.0}),
)

TIPS = {
    "kind": "<p>Which self-excited system.</p>",
    "vdp_freq": "<p>The natural frequency.</p>",
    "vdp_eps": "<p>ε: the damping at small amplitude is −ε/2 of critical (negative for ε &gt; 0).</p>",
    "vdp_ref": "<p>The amplitude scale: the damping changes sign at x = x<sub>r</sub>√ε.</p>",
    "start": "<p>The start: displaced this far, at rest.</p>",
    "periods": "<p>How long to run, in natural periods.</p>",
    "side": "<p>The square's side D.</p>",
    "mass": "<p>Mass per metre of length.</p>",
    "fn": "<p>The natural frequency.</p>",
    "zeta": "<p>The structure's damping ratio.</p>",
    "speed": "<p>The wind speed.</p>",
    "reduced": "<p>The current as a reduced velocity U/f<sub>n</sub>D.</p>",
    "plunge": "<p>The plunge (bending) mode's frequency, without wind.</p>",
    "pitch": "<p>The pitch (torsion) mode's frequency, without wind.</p>",
    "cg": "<p>The centre of mass behind the elastic axis, in semichords.</p>",
    "e": "<p>The elastic axis behind the quarter chord (where the lift acts), in semichords.</p>",
}
SWEEP_TIP = "<p>The settled motion against the parameter that drives it; the dotted line is the value set.</p>"

NOTES_HTML = """
<!--BEFORE-->
<h3><a name="self"></a>Motion that feeds itself</h3>
<p>Forced vibration needs a force at the frequency of the motion; free vibration dies away. A <b>self-excited</b>
vibration has neither: a steady source (a power supply, the wind, a current, a rotating shaft) feeds it at its own
frequency, because the force depends on the motion itself. Near rest the effect is a damping term, and when the
source's share makes the net damping negative, the rest state is unstable: the smallest disturbance grows.</p>

<h3><a name="hopf"></a>Hopf bifurcation and limit cycles</h3>
<p>As the parameter (ε, the wind speed) rises past its critical value, a pair of complex poles crosses into the
right half plane: a <b>Hopf bifurcation</b>. Linear theory says the motion then grows without end; in practice a
nonlinearity, damping that rises with amplitude, stops it on a <b>limit cycle</b>, a closed orbit that motions on
both sides approach. Averaging (the energy each cycle puts in against what it takes out) gives its size. If it
grows from zero at the onset the bifurcation is supercritical; if the curve folds back (as galloping's does) the
system has <b>hysteresis</b>: two stable motions at one speed, and jumps between them, as with the Duffing
oscillator's jump.</p>

<h3><a name="vdp"></a>Van der Pol</h3>
<p>ẍ + ω²x = ω(ε − x²/x<sub>r</sub>²)ẋ: negative damping ε at small amplitude, positive past x<sub>r</sub>√ε. For small
ε the motion is nearly a sine of amplitude 2x<sub>r</sub>√ε at ω; for large ε it becomes a relaxation oscillation,
slow creeps and fast snaps, its period growing like ε/ω. Van der Pol wrote it for a valve oscillator; it describes
any system whose energy source pushes harder at small motion than large.</p>

<h3><a name="galloping"></a>Galloping</h3>
<p>A bluff section moving across a wind U at ẏ sees the wind at an angle α = ẏ/U. If the force that angle makes
points the way it is moving, the wind does work on it every cycle. The quasi-steady force coefficient's slope at
α = 0 decides: <b>Den Hartog's criterion</b>, unstable when dC<sub>L</sub>/dα + C<sub>D</sub> &lt; 0. For a
square, Parkinson and Smith's fit is C<sub>Fy</sub> = A₁α − A₃α³ + A₅α⁵ − A₇α⁷ with A₁ = 2.69, so the wind adds damping
−½ρUDA₁ and galloping starts at</p>
<p align="center">U<sub>g</sub> = 4mζω / (ρDA₁),</p>
<p>proportional to the mass–damping (Scruton) number. Above it the amplitude grows roughly in proportion to U, with
no upper limit; the square's higher terms give two stable branches over a range of speeds. Iced power lines, tall
square towers and bridge decks gallop.</p>

<h3><a name="viv"></a>Vortex-induced vibration</h3>
<p>A cylinder in a flow sheds vortices alternately from each side at f<sub>s</sub> = St U/D (St ≈ 0.2), with a
fluctuating lift at that frequency. When f<sub>s</sub> nears f<sub>n</sub> (reduced velocity U/f<sub>n</sub>D ≈ 1/St
= 5) the structure's motion takes control of the shedding: over a band of speeds the vortices shed at
f<sub>n</sub> instead (<b>lock-in</b>). The page couples the structure to a van der Pol <b>wake oscillator</b>
(Facchinetti, de Langre and Biolley) driven by the structure's acceleration. The amplitude is self-limiting, about
a diameter at most, and the lock-in band widens as the mass ratio falls. Risers, chimneys and cables are protected
by helical strakes that break up the shedding.</p>

<h3><a name="flutter"></a>Flutter</h3>
<p>A wing section in plunge h and pitch θ: the lift, proportional to θ and to the dynamic pressure, couples the two
modes. As the speed rises their frequencies approach, and at the <b>flutter speed</b> the two modes merge into one
whose damping crosses zero: the wind feeds a combined bending–torsion motion with the right phase between them.
Moving the centre of mass forward onto the elastic axis (<b>mass balance</b>) removes the inertial coupling and
raises the flutter speed. Separately, the pitch spring alone is overcome at the <b>divergence</b> speed, a static
instability. The page uses quasi-steady lift (the lift follows the angle at once); real flutter analysis uses
unsteady aerodynamics (Theodorsen), which changes the numbers but not the coalescence.</p>

<h3><a name="try-it"></a>Things to try</h3>
<p>Each of these picks a model and sets it up (Things to try, top left).</p>
<!--TRIALS-->
<!--TAKEAWAYS-->
"""
NOTES_HTML = NOTES_HTML.replace("<!--BEFORE-->", before(
    ("sdof_page", "poles", "Poles"),
    ("duffing_page", "jump", "The jump"),
)).replace("<!--TRIALS-->", trials_html(TRIALS)).replace("<!--TAKEAWAYS-->", takeaways(
    "Self-excitation is negative damping drawn from a steady source; the rest state loses stability at a Hopf "
    "bifurcation.",
    "A nonlinearity sets the size of the limit cycle that follows; averaging the energy per cycle finds it.",
    "Galloping starts at U<sub>g</sub> ∝ mζ (Den Hartog: dC<sub>L</sub>/dα + C<sub>D</sub> &lt; 0) and grows with the "
    "wind; vortex-induced vibration locks in over a band of speeds and limits itself.",
    "Flutter is two modes coupled by the flow until one's damping crosses zero; mass balance raises its speed.",
))


class SelfExcitedPage(QtWidgets.QWidget):
    """Controls on the left, the motion in the centre, the sweep on the right."""

    follows_chain = False  # MainWindow: this page has its own models, not the chain

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._viv_cache: tuple | None = None
        self.trials = TrialBox([t.name for t in TRIALS],
                               "<p>Each picks a model and sets it up to show one idea from the Theory tab.</p>")
        self.trials.picked.connect(self.load_trial)

        top = Form("Model")
        self.kind = combo([k.value for k in KINDS], TIPS["kind"])
        top.add("System:", self.kind, TIPS["kind"])

        self.boxes: dict[str, QtWidgets.QDoubleSpinBox] = {}
        self.pages = QtWidgets.QStackedWidget()
        vdp = Form("Van der Pol")
        self._add(vdp, "vdp_freq", "Frequency:", 0.05, 100.0, 1.0, 3, " Hz", "vdp_freq")
        self._add(vdp, "vdp_eps", "ε:", -2.0, 20.0, 0.2, 3, "", "vdp_eps")
        self._add(vdp, "vdp_ref", "x_r:", 1e-4, 1.0, 0.01, 4, " m", "vdp_ref")
        self._add(vdp, "vdp_start", "Start:", 0.0, 1.0, 0.0005, 4, " m", "start")
        self._add(vdp, "vdp_periods", "Periods:", 2.0, 400.0, 60.0, 0, "", "periods")
        gal = Form("Galloping")
        self._add(gal, "g_side", "Side D:", 0.01, 2.0, 0.1, 3, " m", "side")
        self._add(gal, "g_mass", "Mass:", 0.1, 1000.0, 5.0, 2, " kg/m", "mass")
        self._add(gal, "g_fn", "f_n:", 0.05, 50.0, 2.0, 3, " Hz", "fn")
        self._add(gal, "g_zeta", "ζ:", 0.0, 0.2, 0.005, 4, "", "zeta")
        self._add(gal, "g_speed", "Wind U:", 0.0, 60.0, 6.0, 2, " m/s", "speed")
        self._add(gal, "g_start", "Start:", 0.0, 2.0, 0.001, 4, " m", "start")
        self._add(gal, "g_periods", "Periods:", 10.0, 1000.0, 300.0, 0, "", "periods")
        viv = Form("Vortex-induced vibration (in water)")
        self._add(viv, "v_d", "Diameter D:", 0.005, 2.0, 0.05, 3, " m", "side")
        self._add(viv, "v_mass", "Mass:", 0.1, 5000.0, 20.0, 2, " kg/m", "mass")
        self._add(viv, "v_fn", "f_n (in water):", 0.05, 50.0, 1.0, 3, " Hz", "fn")
        self._add(viv, "v_zeta", "ζ:", 0.0, 0.2, 0.005, 4, "", "zeta")
        self._add(viv, "v_reduced", "U/f_n D:", 1.0, 20.0, 5.5, 2, "", "reduced")
        fl = Form("Flutter")
        self._add(fl, "f_plunge", "Plunge:", 0.1, 50.0, 3.0, 2, " Hz", "plunge")
        self._add(fl, "f_pitch", "Pitch:", 0.1, 50.0, 6.0, 2, " Hz", "pitch")
        self._add(fl, "f_cg", "x_cg:", -0.5, 1.0, 0.2, 2, " b", "cg")
        self._add(fl, "f_e", "e:", -0.5, 1.0, 0.3, 2, " b", "e")
        self._add(fl, "f_zeta", "ζ:", 0.0, 0.2, 0.01, 3, "", "zeta")
        self._add(fl, "f_speed", "Speed U:", 0.0, 200.0, 15.0, 1, " m/s", "speed")
        for form in (vdp, gal, viv, fl):
            self.pages.addWidget(form)
        self.info = note_label()
        connect_all((self.kind, *self.boxes.values()), self._on_inputs)

        # --- centre
        self.top_plot = plot("x", "m", "Time", "s", height=150)
        self.bottom_plot = plot("Velocity", "m/s", "x", "m", height=150)
        add_legend(self.top_plot.getPlotItem(), offset=(-10, 10))
        add_legend(self.bottom_plot.getPlotItem(), offset=(-10, 10))
        self.top_items = [pg.PlotDataItem(), pg.PlotDataItem()]
        self.bottom_items = [pg.PlotDataItem(), pg.PlotDataItem()]
        self.top_line = pg.InfiniteLine(angle=90)
        self.bottom_line = pg.InfiniteLine(angle=90)
        self.bottom_zero = pg.InfiniteLine(angle=0, pos=0)
        for item in (self.top_line, *self.top_items):
            self.top_plot.addItem(item)
        for item in (self.bottom_zero, self.bottom_line, *self.bottom_items):
            self.bottom_plot.addItem(item)
        self.readout = readout_label()

        # --- right
        self.sweep_plot = plot("Amplitude", "m", "Parameter", None, height=200)
        self.sweep_plot.setToolTip(SWEEP_TIP)
        add_legend(self.sweep_plot.getPlotItem(), offset=(-10, 10))
        self.sweep2_plot = plot("Frequency / f_n", None, "Parameter", None, height=160)
        self.sweep2_plot.setXLink(self.sweep_plot)
        add_legend(self.sweep2_plot.getPlotItem(), offset=(-10, 10))
        self.sweep_curve = pg.PlotDataItem()
        self.sweep_theory = pg.PlotDataItem()
        self.sweep_stable = marker(7)
        self.sweep_unstable = marker(7)
        self.sweep_line = pg.InfiniteLine(angle=90)
        for item in (self.sweep_line, self.sweep_theory, self.sweep_curve, self.sweep_stable, self.sweep_unstable):
            self.sweep_plot.addItem(item)
        self.sweep2_curve = pg.PlotDataItem()
        self.sweep2_ref = pg.PlotDataItem()
        self.sweep2_line = pg.InfiniteLine(angle=90)
        for item in (self.sweep2_line, self.sweep2_ref, self.sweep2_curve):
            self.sweep2_plot.addItem(item)
        self.sweep_note = note_label()
        self.notes = theory(NOTES_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(column(self.sweep_plot, self.sweep2_plot, self.sweep_note, stretch=False), "Sweep")
        self.tabs.addTab(self.notes, "Theory")
        self.tabs.currentChanged.connect(lambda _: self._draw_sweep())

        three_panes(self, column(self.trials, top, self.pages, self.info),
                    stack((self.top_plot, 1), (self.bottom_plot, 1), (self.readout, 0)), self.tabs)
        self.apply_theme()
        self._on_inputs()

    def _add(self, form: Form, key: str, label: str, lo: float, hi: float, value: float, decimals: int,
             suffix: str, tip: str) -> None:
        self.boxes[key] = form.number(label, lo, hi, value, decimals, suffix, TIPS[tip])

    def v(self, key: str) -> float:
        return float(self.boxes[key].value())

    # ------------------------------------------------------------- inputs
    def load_trial(self, index: int) -> None:
        t = TRIALS[index]
        defaults = {"vdp_freq": 1.0, "vdp_eps": 0.2, "vdp_ref": 0.01, "vdp_start": 0.0005, "vdp_periods": 60.0,
                    "g_side": 0.1, "g_mass": 5.0, "g_fn": 2.0, "g_zeta": 0.005, "g_speed": 6.0, "g_start": 0.001,
                    "g_periods": 300.0, "v_d": 0.05, "v_mass": 20.0, "v_fn": 1.0, "v_zeta": 0.005,
                    "v_reduced": 5.5, "f_plunge": 3.0, "f_pitch": 6.0, "f_cg": 0.2, "f_e": 0.3, "f_zeta": 0.01,
                    "f_speed": 15.0}
        defaults.update(t.values)
        for key, value in defaults.items():
            set_quietly(self.boxes[key], value)
        set_quietly(self.kind, KINDS.index(t.kind))
        self.trials.show_note(t.note)
        self.tabs.setCurrentIndex(t.tab)
        self._on_inputs()

    def _on_inputs(self) -> None:
        self.current = KINDS[self.kind.currentIndex()]
        self.pages.setCurrentIndex(self.kind.currentIndex())
        self.redraw()

    def vdp(self) -> VanDerPol:
        return VanDerPol(self.v("vdp_freq"), self.v("vdp_eps"), self.v("vdp_ref"), self.v("vdp_start"))

    def galloping(self) -> Galloping:
        return Galloping(self.v("g_side"), self.v("g_mass"), self.v("g_fn"), self.v("g_zeta"), self.v("g_speed"),
                         start=self.v("g_start"))

    def viv(self) -> Viv:
        return Viv(self.v("v_d"), self.v("v_mass"), self.v("v_fn"), self.v("v_zeta"), self.v("v_reduced"))

    def flutter(self) -> Flutter:
        return Flutter(x_cg=self.v("f_cg"), e=self.v("f_e"), plunge_hz=self.v("f_plunge"),
                       pitch_hz=self.v("f_pitch"), zeta=self.v("f_zeta"))

    # ------------------------------------------------------------- drawing
    def apply_theme(self) -> None:
        c = colors
        for item, color in zip(self.top_items, (c.mass[0], c.force)):
            item.setPen(pg.mkPen(color, width=1.5))
        for item, color in zip(self.bottom_items, (c.mass[0], c.mass[1])):
            item.setPen(pg.mkPen(color, width=1.5))
        dot = QtCore.Qt.PenStyle.DotLine
        for line in (self.top_line, self.bottom_line, self.sweep_line, self.sweep2_line):
            line.setPen(pg.mkPen(c.grey, width=1.5, style=dot))
        self.bottom_zero.setPen(pg.mkPen(c.faint, width=1))
        self.sweep_curve.setPen(pg.mkPen(c.mass[0], width=2))
        self.sweep_curve.setSymbolBrush(pg.mkBrush(c.mass[0]))
        self.sweep_curve.setSymbolPen(None)
        self.sweep_theory.setPen(pg.mkPen(c.strong, width=1.5, style=QtCore.Qt.PenStyle.DashLine))
        mark(self.sweep_stable, c.mass[0], filled=True)
        mark(self.sweep_unstable, c.force)
        self.sweep2_curve.setPen(pg.mkPen(c.mass[0], width=2))
        self.sweep2_ref.setPen(pg.mkPen(c.strong, width=1.5, style=QtCore.Qt.PenStyle.DashLine))
        if hasattr(self, "current"):
            self.redraw()

    def redraw(self) -> None:
        for items in (self.top_items, self.bottom_items):
            for item in items:
                item.setData([], [])
        for p in (self.top_plot, self.bottom_plot):
            p.getPlotItem().legend.clear()
        self.top_line.setVisible(False)
        self.bottom_line.setVisible(False)
        self.bottom_zero.setVisible(False)
        {Kind.VDP: self._draw_vdp, Kind.GALLOP: self._draw_galloping, Kind.VIV: self._draw_viv,
         Kind.FLUTTER: self._draw_flutter}[self.current]()
        self._draw_sweep()
        settle_plots(self)

    def _time_and_phase(self, t: np.ndarray, x: np.ndarray, v: np.ndarray, scale: float, unit: str,
                        name: str) -> None:
        relabel(self.top_plot, "left", name, unit)
        relabel(self.top_plot, "bottom", "Time", "s")
        relabel(self.bottom_plot, "left", "Velocity", "m/s")
        relabel(self.bottom_plot, "bottom", name, unit)
        self.top_items[0].setData(t, x / scale)
        last = t >= t[-1] - 10.0 * (t[-1] / max(self._periods(), 1.0))  # the phase plane: the last ten periods
        self.bottom_items[0].setData(x[last] / scale, v[last])
        self.top_plot.setXRange(0.0, float(t[-1]), padding=0)
        self.top_plot.enableAutoRange(axis="y")
        self.bottom_plot.enableAutoRange()

    def _periods(self) -> float:
        return {Kind.VDP: self.v("vdp_periods"), Kind.GALLOP: self.v("g_periods"), Kind.VIV: 150.0}.get(
            self.current, 1.0)

    def _draw_vdp(self) -> None:
        m = self.vdp()
        t, x, v = m.simulate(self.v("vdp_periods"))
        self._time_and_phase(t, x, v, 1.0, "m", "x")
        amp = float(settled_amplitude(x))
        self.info.setText(f"Damping at rest ζ = {-m.eps / 2:.3g}; averaging's limit cycle: "
                          f"{float(m.theory_amplitude(m.eps)) * 1e3:.3g} mm.")
        period = 1.0 / float(dominant_hz(x, t[1] - t[0])) if amp > 1e-9 else float("nan")
        self.readout.setText(f"Settled amplitude {amp * 1e3:.3g} mm (2x<sub>r</sub>√ε = "
                             f"{float(m.theory_amplitude(m.eps)) * 1e3:.3g} mm); period {period:.3g} s against "
                             f"{1 / m.freq_hz:.3g} s at small ε." if m.eps > 0 else
                             f"ε ≤ 0: the start dies away (amplitude at the end {abs(x[-1]) * 1e3:.3g} mm).")

    def _draw_galloping(self) -> None:
        m = self.galloping()
        t, y, v = m.simulate(self.v("g_periods"))
        self._time_and_phase(t, y, v, m.side, None, "y / D")
        amp = float(settled_amplitude(y))
        self.info.setText(f"Onset U<sub>g</sub> = 4mζω/(ρDA₁) = {m.onset:.3g} m/s; U/U<sub>g</sub> = "
                          f"{m.speed / m.onset:.3g}.")
        self.readout.setText(f"At {m.speed:.3g} m/s from {m.start * 1e3:.3g} mm: the last quarter swings "
                             f"{amp / m.side:.3g} D ({amp * 1e3:.3g} mm).")

    def _draw_viv(self) -> None:
        m = self.viv()
        t, z = m.run(np.array([m.reduced]), periods=150.0)
        y, v, q = z[:, 0, 0], z[:, 1, 0], z[:, 2, 0]
        self._time_and_phase(t, y, v, m.diameter, None, "y / D")
        legend = self.top_plot.getPlotItem().legend
        scale = max(float(np.max(np.abs(y))) / m.diameter, 1e-6) / max(float(np.max(np.abs(q))), 1e-9)
        self.top_items[1].setData(t, q * scale)
        legend.addItem(self.top_items[0], "y / D")
        legend.addItem(self.top_items[1], "wake q (scaled)")
        amp = float(settled_amplitude(y)) / m.diameter
        f = float(dominant_hz(y, t[1] - t[0])) / m.freq_hz
        self.info.setText(f"Mass ratio m* = {m.mass_ratio:.3g}; shedding alone at St U/D = "
                          f"{m.strouhal * m.reduced:.3g} f<sub>n</sub>; lock-in expected near U<sub>r</sub> = "
                          f"{m.lock_in_at:.3g}.")
        self.readout.setText(f"U<sub>r</sub> = {m.reduced:.3g}: settled amplitude {amp:.3g} D at {f:.3g} "
                             f"f<sub>n</sub> (the shedding alone would be at {m.strouhal * m.reduced:.3g} "
                             "f<sub>n</sub>).")

    def _draw_flutter(self) -> None:
        m = self.flutter()
        Uf, Ud = m.flutter_speed(), m.divergence_speed()
        top = 1.3 * (Uf if np.isfinite(Uf) else min(Ud, 100.0))
        U = np.linspace(0.0, top, 160)
        f, z = m.modes(U)
        relabel(self.top_plot, "left", "Frequency", "Hz")
        relabel(self.top_plot, "bottom", "Speed U", "m/s")
        relabel(self.bottom_plot, "left", "Damping ratio ζ", None)
        relabel(self.bottom_plot, "bottom", "Speed U", "m/s")
        for j, name in enumerate(("mode 1 (plunge at rest)", "mode 2 (pitch at rest)")):
            self.top_items[j].setData(U, f[:, j], connect="finite")
            self.bottom_items[j].setData(U, z[:, j], connect="finite")
            self.top_plot.getPlotItem().legend.addItem(self.top_items[j], name)
            self.bottom_plot.getPlotItem().legend.addItem(self.bottom_items[j], name)
        self.top_items[1].setPen(pg.mkPen(colors.mass[1], width=1.5))
        self.bottom_zero.setVisible(True)
        for line in (self.top_line, self.bottom_line):
            line.setVisible(True)
            line.setValue(self.v("f_speed"))
        self.top_plot.setXRange(0.0, top, padding=0)
        self.bottom_plot.setXRange(0.0, top, padding=0)
        self.top_plot.enableAutoRange(axis="y")
        finite = z[np.isfinite(z)]
        self.bottom_plot.setYRange(max(-0.1, float(finite.min()) - 0.02), float(finite.max()) + 0.02, padding=0)
        lam = m.poles(self.v("f_speed"))
        up = lam[lam.imag > 0]
        self.info.setText(f"Flutter at {Uf:.3g} m/s; the pitch spring alone diverges at {Ud:.3g} m/s.")
        self.readout.setText(f"At {self.v('f_speed'):.3g} m/s: " + ", ".join(
            f"{abs(p) / (2 * np.pi):.3g} Hz with ζ = {-p.real / abs(p):.3g}" for p in sorted(up, key=abs)) + ".")

    def _draw_sweep(self) -> None:
        if self.tabs.currentIndex() != TAB_SWEEP or not hasattr(self, "current"):
            return
        for item in (self.sweep_curve, self.sweep_theory, self.sweep_stable, self.sweep_unstable,
                     self.sweep2_curve, self.sweep2_ref):
            item.setData([], [])
        for p in (self.sweep_plot, self.sweep2_plot):
            p.getPlotItem().legend.clear()
        legend = self.sweep_plot.getPlotItem().legend
        self.sweep2_plot.setVisible(self.current in (Kind.VIV, Kind.FLUTTER))
        if self.current is Kind.VDP:
            m = self.vdp()
            eps = np.linspace(-0.5, max(3.0, m.eps * 1.2), 36)
            amp = m.sweep(eps)
            self.sweep_curve.setData(eps, amp * 1e3, symbol="o", symbolSize=5)
            fine = np.linspace(eps[0], eps[-1], 300)
            self.sweep_theory.setData(fine, m.theory_amplitude(fine) * 1e3)
            legend.addItem(self.sweep_curve, "settled (stepped)")
            legend.addItem(self.sweep_theory, "2x<sub>r</sub>√ε (averaging)")
            relabel(self.sweep_plot, "left", "Amplitude (mm)")
            relabel(self.sweep_plot, "bottom", "ε")
            self.sweep_line.setValue(m.eps)
            self.sweep_note.setText("Zero below ε = 0 (the Hopf point), then growing as √ε: supercritical.")
        elif self.current is Kind.GALLOP:
            m = self.galloping()
            U = np.linspace(0.0, max(15.0, 1.3 * m.speed), 241)
            stable_x, stable_y, unstable_x, unstable_y = [], [], [], []
            for b in galloping_branches(m, U):
                ok = np.isfinite(b.amplitude)
                (stable_x if b.stable else unstable_x).extend(b.speed[ok])
                (stable_y if b.stable else unstable_y).extend(b.amplitude[ok] / m.side)
            self.sweep_stable.setData(stable_x, stable_y)
            self.sweep_unstable.setData(unstable_x, unstable_y)
            legend.addItem(self.sweep_stable, "stable")
            legend.addItem(self.sweep_unstable, "unstable")
            relabel(self.sweep_plot, "left", "Amplitude / D")
            relabel(self.sweep_plot, "bottom", "Wind speed U (m/s)")
            self.sweep_line.setValue(m.speed)
            self.sweep_note.setText(f"Averaging (Parkinson's polynomial). Rest is stable below U<sub>g</sub> = "
                                    f"{m.onset:.3g} m/s; above it the amplitude grows with U, on two stable branches "
                                    "where they overlap.")
        elif self.current is Kind.VIV:
            m = self.viv()
            key = (m.diameter, m.mass, m.freq_hz, m.zeta)
            if self._viv_cache is None or self._viv_cache[0] != key:
                self._viv_cache = (key, *m.sweep(VIV_SWEEP))
            _, amp, freq = self._viv_cache
            self.sweep_curve.setData(VIV_SWEEP, amp, symbol="o", symbolSize=5)
            legend.addItem(self.sweep_curve, "settled amplitude")
            self.sweep2_curve.setData(VIV_SWEEP, freq, symbol="o", symbolSize=5)
            self.sweep2_ref.setData(VIV_SWEEP, m.strouhal * VIV_SWEEP)
            l2 = self.sweep2_plot.getPlotItem().legend
            l2.addItem(self.sweep2_curve, "response")
            l2.addItem(self.sweep2_ref, "shedding alone, St U/D")
            relabel(self.sweep_plot, "left", "Amplitude / D")
            relabel(self.sweep_plot, "bottom", "Reduced velocity U/f<sub>n</sub>D")
            relabel(self.sweep2_plot, "left", "Frequency / f<sub>n</sub>")
            relabel(self.sweep2_plot, "bottom", "Reduced velocity U/f<sub>n</sub>D")
            self.sweep2_plot.setYRange(0.0, 3.0, padding=0)
            self.sweep_line.setValue(m.reduced)
            self.sweep2_line.setValue(m.reduced)
            self.sweep_note.setText("Lock-in: where the response frequency stays near f<sub>n</sub> instead of "
                                    "following the Strouhal line, the amplitude is large.")
        else:
            m = self.flutter()
            Uf = m.flutter_speed()
            top = 1.3 * (Uf if np.isfinite(Uf) else 100.0)
            U = np.linspace(0.0, top, 160)
            poles = np.array([m.poles(float(u)) for u in U])
            up = np.where(poles.imag > 0, poles, np.nan)
            self.sweep_stable.setData(up.real.ravel(), (up.imag / (2 * np.pi)).ravel())
            legend.addItem(self.sweep_stable, "poles, 0 to 1.3 × the flutter speed")
            relabel(self.sweep_plot, "left", "Im λ / 2π (Hz)")
            relabel(self.sweep_plot, "bottom", "Re λ (1/s)")
            self.sweep_line.setValue(0.0)
            self.sweep2_plot.setVisible(False)
            self.sweep_note.setText("The root locus: as the speed rises the two modes' poles approach, and one "
                                    "crosses Re λ = 0 (dotted) at the flutter speed.")
        self.sweep_plot.enableAutoRange()
        self.sweep2_plot.enableAutoRange(axis="x")
