"""Damping models page: viscous, hysteretic, Coulomb friction and a viscoelastic (Zener) element on one oscillator.

Left: the oscillator, the damping they're all matched to at f_n, the loop's amplitude and frequency, and
the drive for the response. Centre: the hysteresis loops (force against displacement over a cycle) and the
energy per cycle against frequency. Right: storage stiffness and loss factor against frequency, the
frequency response, the free decay, and the Theory. The page has its own model (core/damping_models.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.damping_models import MODELS, Matched, Model, coulomb_stop
from ..core.sdof import TWO_PI
from . import settle_plots
from .layouts import ColumnScroll, FlowLayout
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

RATIOS = np.geomspace(0.05, 20.0, 400)  # frequency over f_n on the energy and loss plots
TAB_LOSS, TAB_FRF, TAB_DECAY = range(3)
SHORT = {Model.VISCOUS: "Viscous", Model.HYSTERETIC: "Hysteretic", Model.COULOMB: "Coulomb",
         Model.ZENER: "Zener"}


@dataclass(frozen=True)
class DampingTrial:
    name: str
    note: str
    zeta: float = 0.05
    amplitude_mm: float = 10.0
    relax: float = 1.0
    ratio: float = 1.0  # the loop's frequency over f_n
    damping_only: bool = True
    force: float = 1.0
    tab: int = TAB_LOSS


TRIALS = (
    DampingTrial(
        "Four loops, one energy",
        "m = 1 kg, k = 400 N/m (f<sub>n</sub> = 3.18 Hz), each model matched to ζ = 0.05 there. Cycled at "
        "f<sub>n</sub> through ±10 mm, every loop of damping force encloses the same πcω<sub>n</sub>X² = 12.6 mJ: "
        "that is what matched means. Viscous, hysteretic and Zener damping make ellipses, their force a quarter "
        "cycle ahead of the motion; friction makes a rectangle, ±F<sub>f</sub> = ±0.31 N, flipping at each end "
        "of the stroke, which doesn't shrink as the motion does.",
        tab=TAB_LOSS),
    DampingTrial(
        "Twice the frequency",
        "The same cycle at 2f<sub>n</sub>. The viscous loop is twice as fat (25.1 mJ: its force grows with "
        "the velocity), the hysteretic and friction loops are unchanged (12.6 mJ), and the Zener element's, "
        "past its loss peak, is thinner (10.7 mJ).",
        ratio=2.0),
    DampingTrial(
        "With the spring: what a test machine draws",
        "The whole force, spring and damping together, as a test machine measures it on a sample: each loop is "
        "the spring's line kx (dashed) opened out by the damping. The tilt is the stiffness (4 N at 10 mm), "
        "the width the loss (0.4 N): a lightly damped part draws a thin loop.",
        damping_only=False),
    DampingTrial(
        "Friction can't stop a resonance",
        "1 N at every frequency. The viscous, hysteretic and Zener peaks stop at 25 mm, (F/k)/2ζ. Friction, "
        "matched to them at 10 mm, takes at most 4F<sub>f</sub>X a cycle, which grows only as X while the force "
        "puts in πFX: since F = 1 N &gt; 4F<sub>f</sub>/π = 0.4 N, the resonance runs away. Set the drive below "
        "0.4 N and friction holds the mass still (to this approximation).",
        tab=TAB_FRF),
    DampingTrial(
        "A straight-line decay",
        "Released from 10 mm. The viscous swing falls by the same fraction each cycle, e<sup>−δ</sup> = 0.73, "
        "for ever. Friction's falls by the same amount, 4F<sub>f</sub>/k = 3.1 mm, and stops dead after three "
        "cycles, 0.58 mm off centre, where the spring can no longer beat the friction.",
        tab=TAB_DECAY),
    DampingTrial(
        "A viscoelastic loss peak",
        "The Zener element: a spring k<sub>0</sub> beside a spring k<sub>1</sub> and a dashpot in series. Slowly "
        "the dashpot gives and only k<sub>0</sub> = 364 N/m holds; fast it locks and k<sub>0</sub> + "
        "k<sub>1</sub> = 444 N/m does. In between, at its relaxation frequency (here f<sub>n</sub>), the loss "
        "factor peaks at 0.1. Rubber mounts and polymers behave like this.",
        tab=TAB_LOSS),
    DampingTrial(
        "Relaxing well above resonance",
        "The Zener element's peak moved to 10 f<sub>n</sub>. Below it, its loss grows in proportion to "
        "frequency, as a viscous damper's does: through the oscillator's band it is a viscous damper. It "
        "needs a stiffer arm for the same damping at f<sub>n</sub>.",
        relax=10.0, tab=TAB_LOSS),
)

TIPS = {
    "zeta": "<p>The damping ratio every model is matched to at f<sub>n</sub>: the same energy per cycle there as "
            "a viscous damper c = 2ζ√(km), and friction at the amplitude X.</p>",
    "amplitude": "<p>The amplitude X of the cycle the loops trace, where friction is matched, and the release "
                 "of the free decay.</p>",
    "relax": "<p>Where the Zener element's loss factor peaks, as a multiple of f<sub>n</sub>.</p>",
    "ratio": "<p>The frequency of the cycle the loops trace, over f<sub>n</sub>.</p>",
    "force": "<p>The harmonic force on the Frequency response tab.</p>",
    "damping_only": "<p>Draw only the damping part of the force, without the spring's kx.</p>",
}
LOOP_TIP = (
    "<p>The force the spring and the damping carry, against the displacement, over one steady cycle "
    "x = X sin Ωt. The area inside each loop is the energy it takes out per cycle. The spring alone would be the "
    "straight line kx.</p>")
ENERGY_TIP = (
    "<p>The energy each model takes per cycle of amplitude X, against the frequency over f<sub>n</sub>. They "
    "meet at f<sub>n</sub>, where they are matched.</p>")
LOSS_TIP = (
    "<p>Top: the storage stiffness K' (the in-phase part of the force per metre) over k. Bottom: the loss "
    "factor η = K''/K', the out-of-phase part over the in-phase part (2ζ at resonance for a viscous "
    "damper).</p>")
FRF_TIP = (
    "<p>The steady amplitude under the harmonic force, against frequency, for each model. Friction's is the "
    "describing function's, which can't bound a resonance once F &gt; 4F<sub>f</sub>/π.</p>")
DECAY_TIP = (
    "<p>Released from rest at X (held there long, so the Zener element's dashpot is relaxed). Hysteretic "
    "damping has no causal time-domain model: its curve is the viscous one of the same loss at "
    "f<sub>n</sub>.</p>")

NOTES_HTML = """
<!--BEFORE-->
<h3><a name="why"></a>Where the energy goes</h3>
<p>The viscous damper, a force proportional to velocity, is a model chosen because it makes the equations
linear and easy. Real structures lose energy in other ways: internal friction in the material, rubbing in
joints and bolted connections, the shearing of rubber and polymers, the air around them. Each has its own
force law, and they differ in how the loss depends on the <b>amplitude</b> and on the <b>frequency</b>. This
page puts four of them on the same mass and spring, matched so that at f<sub>n</sub> they all take the same
energy per cycle as a viscous damper of ratio ζ.</p>

<h3><a name="hysteresis-loop"></a>The hysteresis loop</h3>
<p>Cycle the oscillator's spring and damper through x = X sin Ωt and plot the force they carry against x.
A spring alone traces the line kx out and back. Anything that loses energy traces a <b>loop</b>, because the
force is larger going one way than coming back, and the area inside it is the energy lost per cycle:</p>
<p align="center">W<sub>d</sub> = ∮ F dx.</p>
<p>A material test machine draws exactly this: the loop's tilt is the stiffness, its width the damping.</p>

<h3><a name="viscous"></a>Viscous damping</h3>
<p>F = cẋ. Over a harmonic cycle the damping force is cΩX cos Ωt, a quarter cycle ahead of x, so the loop is
an ellipse and</p>
<p align="center">W<sub>d</sub> = πcΩX².</p>
<p>The loss grows in proportion to the frequency. Written as a <b>complex stiffness</b>, F = K*X with
K* = k + iΩc: the real part is the <b>storage stiffness</b> K', the imaginary part the <b>loss
stiffness</b> K''.</p>

<h3><a name="hysteretic"></a>Hysteretic (structural) damping</h3>
<p>Tests on most metals, and on built-up structures, find a loss per cycle that hardly depends on frequency.
The model for it is a complex stiffness that doesn't depend on frequency either:</p>
<p align="center">K* = k(1 + iη), &nbsp; W<sub>d</sub> = πηkX².</p>
<p>η is the <b>loss factor</b>. At resonance it gives the same peak as a viscous damper with η = 2ζ, so the two
are often used interchangeably (as the matching here does), but away from resonance they differ: the
hysteretic loss is the same at every frequency. It works for steady harmonic motion (and in frequency-domain
analysis, where it is standard), but it has no honest time-domain form: a constant K'' at every frequency,
negative ones included, would make the mass respond before it is struck. The free decay draws its viscous
equivalent instead.</p>

<h3><a name="coulomb"></a>Coulomb friction</h3>
<p>A dry contact pushes back with a force of fixed size against the sliding, F<sub>f</sub> sign(ẋ). The loop
is a rectangle (with the spring, a parallelogram) and</p>
<p align="center">W<sub>d</sub> = 4F<sub>f</sub>X,</p>
<p>proportional to X, not X². Matching it to a viscous damper gives the <b>equivalent viscous damping</b>
c<sub>eq</sub> = 4F<sub>f</sub>/πΩX: friction is heavy damping for small motions and light for large ones.
So it decays in a straight line, losing 4F<sub>f</sub>/k of amplitude a cycle, and stops dead within
F<sub>f</sub>/k of the centre (the <a href="page:sim_page#friction">Simulation page's friction</a> steps it
exactly). Under a harmonic force F it can't limit a resonance at all once F &gt; 4F<sub>f</sub>/π: the force
then puts in πFX per cycle at resonance, more than the 4F<sub>f</sub>X friction takes out, at every
amplitude.</p>

<h3><a name="viscoelastic"></a>Viscoelastic materials: the Zener element</h3>
<p>Rubber, polymers and damping layers store and lose energy in a way that depends strongly on frequency.
The simplest model that captures it is the <b>standard linear solid</b> (Zener): a spring k<sub>0</sub> in
parallel with a spring k<sub>1</sub> and a dashpot c<sub>1</sub> in series (a Maxwell arm). With the
relaxation time τ = c<sub>1</sub>/k<sub>1</sub>,</p>
<p align="center">K*(Ω) = k<sub>0</sub> + k<sub>1</sub> iΩτ/(1 + iΩτ).</p>
<ul>
<li>Slowly (Ωτ ≪ 1) the dashpot gives way and only k<sub>0</sub> carries load: the <b>relaxed</b> stiffness.</li>
<li>Fast (Ωτ ≫ 1) the dashpot locks: k<sub>0</sub> + k<sub>1</sub>, the <b>unrelaxed</b> stiffness.</li>
<li>In between the stiffness climbs and the loss factor peaks, at Ωτ = 1/√(1 + k<sub>1</sub>/k<sub>0</sub>).</li>
</ul>
<p>Far below its peak the Zener element acts like a viscous damper (η ∝ Ω), far above it like a stiffer
spring with little loss. Real materials have many such arms with a spread of τ, and temperature shifts them
all; designers put a damping material's loss peak where the structure's modes are.</p>

<h3><a name="loss-factor"></a>The loss factor, compared</h3>
<p>For any of them the <b>loss factor</b> η = K''/K' = W<sub>d</sub>/(2πU), U = ½K'X² the peak stored energy,
measures damping without saying where it comes from; at resonance η = 2ζ = 1/Q. Viscous: η = cΩ/k, rising
with frequency. Hysteretic: constant. Friction: 4F<sub>f</sub>/πkX, constant in frequency but falling with
amplitude. Zener: a peak.</p>

<h3><a name="computing"></a>How this page computes it</h3>
<p>The loops, the energies and the loss factors come from each model's complex stiffness (friction's true
force for its loop, its describing function elsewhere). The frequency response is 1/(K*(Ω) − Ω²m), and
friction's √(F² − (4F<sub>f</sub>/π)²)/|k − Ω²m|. The free decays are exact: the viscous one in closed form,
friction's half cycle by half cycle, the Zener element's (three states: the position, the velocity and the
dashpot's end) by the matrix exponential.</p>

<h3><a name="try-it"></a>Things to try</h3>
<p>Each of these sets up the oscillator, the matching and the plots (Things to try, top left).</p>
<!--TRIALS-->
<!--TAKEAWAYS-->
"""
NOTES_HTML = NOTES_HTML.replace("<!--BEFORE-->", before(
    ("sdof_page", "free-vibration", "Free vibration"),
    ("sdof_page", "log-decrement", "Measuring damping from a decay"),
    ("sdof_page", "forced-response", "Forced response"),
)).replace("<!--TRIALS-->", trials_html(TRIALS)).replace("<!--TAKEAWAYS-->", takeaways(
    "The area of the force–displacement loop is the energy lost per cycle; the loss factor η = K''/K' "
    "measures it, and at resonance η = 2ζ.",
    "Viscous damping loses energy in proportion to frequency, hysteretic damping not at all; they agree at "
    "resonance, which is why η = 2ζ swaps between them.",
    "Hysteretic damping is a frequency-domain model: it has no causal free decay.",
    "Friction loses 4F<sub>f</sub>X per cycle: it decays in a straight line, stops off centre, and can't "
    "bound a resonance driven harder than 4F<sub>f</sub>/π.",
    "A viscoelastic material stiffens across its relaxation frequency, where its loss peaks: viscous below, "
    "elastic above.",
))


class DampingModelsPage(QtWidgets.QWidget):
    """Controls on the left, the loops and energies in the centre, loss, response and decay on the right."""

    follows_chain = False  # MainWindow: this page has its own model, not the chain

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.model = Matched()

        self.trials = TrialBox([t.name for t in TRIALS],
                               "<p>Each sets up the oscillator and the plots to show one idea from the Theory "
                               "tab.</p>")
        self.trials.picked.connect(self.load_trial)

        osc = Form("Oscillator and matching")
        self.mass = osc.number("Mass m:", 0.01, 1e4, 1.0, 3, " kg", "<p>The mass m.</p>")
        self.stiffness = osc.number("Stiffness k:", 0.01, 1e8, 400.0, 2, " N/m", "<p>The spring's stiffness k.</p>")
        self.zeta = osc.number("Matched to ζ:", 0.001, 0.5, 0.05, 4, "", TIPS["zeta"])
        self.amplitude = osc.number("Amplitude X:", 0.01, 1000.0, 10.0, 2, " mm", TIPS["amplitude"])
        self.relax = osc.number("Zener peak at:", 0.01, 100.0, 1.0, 3, " × fₙ", TIPS["relax"])
        self.info = note_label()
        osc.row(self.info)
        osc.changed.connect(self._on_model)

        loop = Form("Loops")
        self.ratio = loop.number("Cycle at:", 0.01, 100.0, 1.0, 3, " × fₙ", TIPS["ratio"])
        self.damping_only = QtWidgets.QCheckBox("Damping force only")
        self.damping_only.setChecked(True)
        self.damping_only.setToolTip(TIPS["damping_only"])
        loop.row(self.damping_only)
        self.shown: dict[Model, QtWidgets.QCheckBox] = {}
        row = FlowLayout()
        for m in MODELS:
            box = self.shown[m] = QtWidgets.QCheckBox(SHORT[m])
            box.setChecked(True)
            box.setToolTip(f"<p>Draw the {m.value.lower()} model.</p>")
            box.toggled.connect(self.redraw)
            row.add(box)
        loop.row(row)
        loop.changed.connect(self.redraw)
        self.damping_only.toggled.connect(self.redraw)

        drive = Form("Drive")
        self.force = drive.number("Force F:", 0.0, 1e6, 1.0, 4, " N", TIPS["force"])
        self.drive_note = note_label()
        drive.row(self.drive_note)
        drive.changed.connect(self.redraw)

        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        for w in (self.trials, osc, loop, drive):
            lv.addWidget(w)
        lv.addStretch(1)

        # --- centre
        self.loop_plot = plot("Force", "N", "Displacement x", "m", height=220)
        self.loop_plot.setToolTip(LOOP_TIP)
        self.loop_legend = add_legend(self.loop_plot.getPlotItem(), offset=(10, 10))
        self.spring_item = pg.PlotDataItem()
        self.loop_items = {m: pg.PlotDataItem() for m in MODELS}
        self.loop_plot.addItem(self.spring_item)
        for item in self.loop_items.values():
            self.loop_plot.addItem(item)
        self.loop_note = readout_label()
        self.energy_plot = plot("Energy per cycle", "J", "Frequency / fₙ", None, log_x=True, log_y=True, height=150)
        self.energy_plot.setToolTip(ENERGY_TIP)
        self.energy_plot.setMouseEnabled(x=False, y=False)
        self.energy_items = {m: pg.PlotDataItem() for m in MODELS}
        for item in self.energy_items.values():
            self.energy_plot.addItem(item)
        self.energy_now = pg.InfiniteLine(angle=90)
        self.energy_plot.addItem(self.energy_now)

        center = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(center)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.addWidget(self.loop_plot, 3)
        cv.addWidget(self.loop_note)
        cv.addWidget(self.energy_plot, 2)

        # --- right
        loss = QtWidgets.QWidget()
        lsv = QtWidgets.QVBoxLayout(loss)
        self.storage_plot = plot("K' / k", None, "Frequency / fₙ", None, log_x=True, height=130)
        self.eta_plot = plot("Loss factor η", None, "Frequency / fₙ", None, log_x=True, log_y=True, height=150)
        self.eta_plot.setXLink(self.storage_plot)
        add_legend(self.eta_plot.getPlotItem(), offset=(10, 10))
        for p in (self.storage_plot, self.eta_plot):
            p.setToolTip(LOSS_TIP)
            p.setMouseEnabled(x=False, y=False)
        self.storage_items = {m: pg.PlotDataItem() for m in MODELS}
        self.eta_items = {m: pg.PlotDataItem(name=SHORT[m]) for m in MODELS}
        for m in MODELS:
            self.storage_plot.addItem(self.storage_items[m])
            self.eta_plot.addItem(self.eta_items[m])
        lsv.addWidget(self.storage_plot, 2)
        lsv.addWidget(self.eta_plot, 3)
        self.loss_note = note_label()
        lsv.addWidget(self.loss_note)

        frf = QtWidgets.QWidget()
        frv = QtWidgets.QVBoxLayout(frf)
        self.frf_plot = plot("|X|", "m", "Frequency", "Hz", log_x=True, log_y=True, height=200)
        self.frf_plot.setToolTip(FRF_TIP)
        self.frf_plot.setMouseEnabled(x=False, y=False)
        add_legend(self.frf_plot.getPlotItem(), offset=(-10, 10))
        self.frf_items = {m: pg.PlotDataItem(name=SHORT[m], connect="finite") for m in MODELS}
        for item in self.frf_items.values():
            self.frf_plot.addItem(item)
        frv.addWidget(self.frf_plot, 1)
        self.frf_note = note_label()
        frv.addWidget(self.frf_note)

        decay = QtWidgets.QWidget()
        dv = QtWidgets.QVBoxLayout(decay)
        self.decay_plot = plot("Displacement x", "m", "Time", "s", height=200)
        self.decay_plot.setToolTip(DECAY_TIP)
        add_legend(self.decay_plot.getPlotItem(), offset=(-10, 10))
        self.decay_items = {m: pg.PlotDataItem(name=SHORT[m] + (" (viscous equivalent)" if m is Model.HYSTERETIC else ""))
                            for m in MODELS}
        self.stick_band = pg.LinearRegionItem(orientation="horizontal", movable=False)
        self.decay_plot.addItem(self.stick_band)
        for item in self.decay_items.values():
            self.decay_plot.addItem(item)
        dv.addWidget(self.decay_plot, 1)
        self.decay_note = note_label()
        dv.addWidget(self.decay_note)

        self.notes = QtWidgets.QTextBrowser()
        self.notes.setHtml(NOTES_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(loss, "Stiffness && loss")
        self.tabs.addTab(frf, "Frequency response")
        self.tabs.addTab(decay, "Free decay")
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
        for box, value in ((self.mass, 1.0), (self.stiffness, 400.0), (self.zeta, t.zeta),
                           (self.amplitude, t.amplitude_mm), (self.relax, t.relax), (self.ratio, t.ratio),
                           (self.damping_only, t.damping_only), (self.force, t.force)):
            set_quietly(box, value)
        for box in self.shown.values():
            set_quietly(box, True)
        self.trials.show_note(t.note)
        self.tabs.setCurrentIndex(t.tab)
        self._on_model()

    def _on_model(self) -> None:
        self.model = Matched(self.mass.value(), self.stiffness.value(), self.zeta.value(),
                             1e-3 * self.amplitude.value(), self.relax.value())
        self.redraw()

    # ------------------------------------------------------------- drawing
    def _colors(self) -> dict[Model, str]:
        c = colors
        return {Model.VISCOUS: c.mass[0], Model.HYSTERETIC: c.mode[1], Model.COULOMB: c.force, Model.ZENER: c.mode[3]}

    def apply_theme(self) -> None:
        c = colors
        cols = self._colors()
        for m in MODELS:
            pen = pg.mkPen(cols[m], width=2)
            for items in (self.loop_items, self.energy_items, self.storage_items, self.eta_items, self.frf_items,
                          self.decay_items):
                items[m].setPen(pen)
        for items in (self.energy_items, self.storage_items, self.eta_items):  # on the hysteretic one there
            items[Model.COULOMB].setPen(pg.mkPen(cols[Model.COULOMB], width=2, style=QtCore.Qt.PenStyle.DashLine))
        self.decay_items[Model.HYSTERETIC].setPen(pg.mkPen(cols[Model.HYSTERETIC], width=2,
                                                           style=QtCore.Qt.PenStyle.DashLine))
        self.spring_item.setPen(pg.mkPen(c.grey, width=1, style=QtCore.Qt.PenStyle.DashLine))
        self.energy_now.setPen(pg.mkPen(c.muted, width=1, style=QtCore.Qt.PenStyle.DotLine))
        self.stick_band.setBrush(pg.mkBrush(*pg.mkColor(c.force).getRgb()[:3], 30))
        for line in self.stick_band.lines:
            line.setPen(pg.mkPen(c.force, width=1, style=QtCore.Qt.PenStyle.DotLine))
        self.redraw()

    def _visible(self) -> list[Model]:
        return [m for m in MODELS if self.shown[m].isChecked()]

    def redraw(self) -> None:
        mm = self.model
        k0, k1, tau = mm.zener()
        self.info.setText(f"f<sub>n</sub> = {mm.natural_hz:.4g} Hz. Matched: c = {mm.c:.4g} N·s/m, η = {mm.eta:.4g}, "
                          f"F<sub>f</sub> = {mm.friction:.4g} N, Zener k<sub>0</sub> = {k0:.4g} N/m, "
                          f"k<sub>1</sub> = {k1:.4g} N/m, τ = {tau:.4g} s.")
        self._draw_loops()
        self._draw_energy()
        self._draw_tab()
        settle_plots(self)

    def _draw_loops(self) -> None:
        mm = self.model
        w = self.ratio.value() * mm.omega_n
        only = self.damping_only.isChecked()
        self.loop_legend.clear()
        X = mm.amplitude
        lines = []
        for m in MODELS:
            item = self.loop_items[m]
            if not self.shown[m].isChecked():
                item.setData([], [])
                continue
            x, total, damp = mm.loop(m, w, 600)
            item.setData(x, damp if only else total)
            e = float(mm.energy_per_cycle(m, np.array([w]))[0])
            self.loop_legend.addItem(item, f"{SHORT[m]}: {fmt(e, 'J')}")
            lines.append(f"{SHORT[m]} {fmt(e, 'J')}")
        if only:
            self.spring_item.setData([], [])
        else:
            self.spring_item.setData([-X, X], [-mm.stiffness * X, mm.stiffness * X])
            self.loop_legend.addItem(self.spring_item, "spring kx")
        self.loop_plot.setXRange(-1.1 * X, 1.1 * X, padding=0)
        self.loop_plot.enableAutoRange(axis="y")
        self.loop_note.setText(f"Energy per cycle at {self.ratio.value():.3g} f<sub>n</sub>, X = {fmt(X, 'm')}: "
                               + ", ".join(lines) + ". (The area inside each loop.)")

    def _draw_energy(self) -> None:
        mm = self.model
        w = RATIOS * mm.omega_n
        for m in MODELS:
            item = self.energy_items[m]
            item.setData(RATIOS, mm.energy_per_cycle(m, w)) if self.shown[m].isChecked() else item.setData([], [])
        self.energy_now.setPos(math.log10(self.ratio.value()))
        self.energy_plot.setXRange(math.log10(RATIOS[0]), math.log10(RATIOS[-1]), padding=0)

    def _draw_tab(self) -> None:
        i = self.tabs.currentIndex()
        if i == TAB_LOSS:
            self._draw_loss()
        elif i == TAB_FRF:
            self._draw_frf()
        elif i == TAB_DECAY:
            self._draw_decay()

    def _draw_loss(self) -> None:
        mm = self.model
        w = RATIOS * mm.omega_n
        for m in MODELS:
            K = mm.dynamic_stiffness(m, w)
            if self.shown[m].isChecked():
                self.storage_items[m].setData(RATIOS, K.real / mm.stiffness)
                self.eta_items[m].setData(RATIOS, K.imag / K.real)
            else:
                self.storage_items[m].setData([], [])
                self.eta_items[m].setData([], [])
        self.storage_plot.setXRange(math.log10(RATIOS[0]), math.log10(RATIOS[-1]), padding=0)
        k0, k1, tau = mm.zener()
        peak = 1.0 / (tau * math.sqrt(1.0 + k1 / k0)) / mm.omega_n
        self.loss_note.setText(
            f"Zener: relaxed stiffness k<sub>0</sub> = {k0:.4g} N/m, unrelaxed k<sub>0</sub> + k<sub>1</sub> = "
            f"{k0 + k1:.4g} N/m; loss peak {k1 / (2 * math.sqrt(k0 * (k0 + k1))):.3g} at {peak:.3g} f<sub>n</sub>. "
            f"Friction's η = 4F<sub>f</sub>/πkX = {mm.eta:.3g} holds at X = {fmt(mm.amplitude, 'm')} only.")

    def _draw_frf(self) -> None:
        mm = self.model
        F = self.force.value()
        f = np.geomspace(0.1 * mm.natural_hz, 10.0 * mm.natural_hz, 1500)
        f = f[np.abs(f / mm.natural_hz - 1.0) > 1e-9]
        bottom = math.inf
        for m in MODELS:
            if not self.shown[m].isChecked() or F <= 0:
                self.frf_items[m].setData([], [])
                continue
            X = np.abs(mm.receptance(m, f, F)) * F
            X = np.where(X > 0, X, np.nan)
            self.frf_items[m].setData(f, X)
            if np.isfinite(X).any():
                bottom = min(bottom, float(np.nanmin(X)))
        top = F / mm.stiffness / mm.eta * 4.0
        if math.isfinite(bottom):
            self.frf_plot.setYRange(math.log10(bottom), math.log10(top), padding=0.02)
        self.frf_plot.setXRange(math.log10(f[0]), math.log10(f[-1]), padding=0)
        threshold = 4.0 * mm.friction / math.pi
        if F > threshold:
            text = (f"F = {fmt(F, 'N')} &gt; 4F<sub>f</sub>/π = {fmt(threshold, 'N')}: friction can't stop the "
                    "resonance (its curve runs off the top). ")
        else:
            text = (f"F = {fmt(F, 'N')} ≤ 4F<sub>f</sub>/π = {fmt(threshold, 'N')}: friction holds the mass still "
                    "(the describing function's answer; really it sticks and slips). ")
        text += f"The others peak at (F/k)/η = {fmt(F / mm.stiffness / mm.eta, 'm')}."
        self.frf_note.setText(text)
        self.drive_note.setText(f"Friction's threshold 4F<sub>f</sub>/π = {fmt(threshold, 'N')}.")

    def _draw_decay(self) -> None:
        mm = self.model
        T = TWO_PI / mm.omega_n
        span = min(max(5.0 / (mm.zeta * mm.omega_n), 3 * T), 60 * T)
        t = np.linspace(0.0, span, 3000)
        for m in MODELS:
            if self.shown[m].isChecked():
                self.decay_items[m].setData(t, mm.free_decay(m, mm.amplitude, t))
            else:
                self.decay_items[m].setData([], [])
        d = mm.friction / mm.stiffness
        self.stick_band.setRegion((-d, d))
        self.stick_band.setVisible(self.shown[Model.COULOMB].isChecked())
        self.decay_plot.setXRange(0, span, padding=0)
        halves, stop = coulomb_stop(mm.stiffness, mm.friction, mm.amplitude)
        self.decay_note.setText(
            f"Viscous: each peak e<sup>−δ</sup> = {math.exp(-2 * math.pi * mm.zeta / math.sqrt(1 - mm.zeta ** 2)):.3g}"
            f" of the last. Friction: {fmt(4 * d, 'm')} less each cycle; it stops after {halves / 2:g} cycles, "
            f"{fmt(abs(stop), 'm')} off centre, inside the band ±F<sub>f</sub>/k = ±{fmt(d, 'm')}.")
