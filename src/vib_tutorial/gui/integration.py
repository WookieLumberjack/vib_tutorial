"""Numerical integration page: what a time-stepping scheme does to a vibration, and why the app steps exactly.

Left: the oscillator (optionally with a stiff second mode), the time step, the start and the force, and which
schemes to run. Centre: the first mass's motion by each scheme against the closed form, and each one's error.
Right: each scheme's spectral radius, period error and numerical damping against Δt/T, with the step set marked
for each mode. The page has its own model (core/integration.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.integration import SCHEMES, STABLE_LIMIT, TWO_PI, Load, Model, Scheme, exact, simulate, spectral_curves
from . import settle_plots
from .lesson import (
    Form,
    TrialBox,
    column,
    connect_all,
    fmt,
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
from .style import colors
from .theming import add_legend

TAB_RADIUS, TAB_PERIOD, TAB_DAMPING = range(3)
MOST_STEPS = 20000  # per scheme per run
RATIOS = np.logspace(-2, 1, 241)  # Δt/T for the spectral plots
POINTS = 2000  # of the closed form drawn


@dataclass(frozen=True)
class IntegrationTrial:
    name: str
    note: str
    per_period: float = 10.0  # Δt = T₁ / this
    periods: float = 10.0
    damping: float = 0.0
    stiff_hz: float = 0.0
    x0: float = 0.01
    force: float = 0.0
    freq_hz: float = 2.0
    schemes: tuple[Scheme, ...] = (Scheme.CENTRAL, Scheme.AVERAGE, Scheme.LINEAR, Scheme.RK4)
    alpha: float = -0.1
    tab: int = TAB_PERIOD


TRIALS = (
    IntegrationTrial(
        "Ten steps a period",
        "1 kg on 400 N/m (3.18 Hz), let go from 10 mm, ten steps a period. No scheme blows up, but each gets the "
        "period wrong: central difference 1.7 % short, Newmark's average acceleration 3.2 % long, linear "
        "acceleration 1.6 % long, RK4 0.11 % long. Over ten periods those errors pile up as phase: the "
        "average-acceleration solution is a third of a cycle behind. RK4's amplitude also shrinks, by "
        "0.065 % damping that nobody put in."),
    IntegrationTrial(
        "Central difference's limit: Δt < T/π",
        "Central difference is explicit, and stable only for ωΔt ≤ 2, Δt ≤ T/π. At 3.0 steps a period "
        "(ωΔt = 2.09) its error doubles every few steps and the run stops; at 3.2 (ωΔt = 1.96) it is stable "
        "but its period is 29 % short. Newmark's average acceleration runs at any step, its period 30 % long.",
        per_period=3.0, schemes=(Scheme.CENTRAL, Scheme.AVERAGE), tab=TAB_RADIUS),
    IntegrationTrial(
        "One step a period",
        "Δt = T. Average acceleration is unconditionally stable: the amplitude holds at 10 mm, but the motion it "
        "computes has a period 2.5 times too long, nothing like the oscillator's. Linear acceleration needs "
        "ωΔt ≤ 2√3 (Δt ≤ 0.55 T) and blows up. Stable is not the same as accurate.",
        per_period=1.0, periods=10.0, schemes=(Scheme.AVERAGE, Scheme.LINEAR), tab=TAB_RADIUS),
    IntegrationTrial(
        "RK4's numerical damping",
        "Five steps a period. RK4 is the most accurate here in period (1.1 % long), but it shrinks the "
        "amplitude by 2.2 % a step: numerical damping of ζ = 1.8 %, so after ten periods only 33 % is left. "
        "Average acceleration keeps the amplitude exactly, its period 12 % long.",
        per_period=5.0, schemes=(Scheme.AVERAGE, Scheme.RK4), tab=TAB_DAMPING),
    IntegrationTrial(
        "Explicit Euler grows",
        "Fifty steps a period, which would be plenty for any other scheme. Explicit Euler steps along the "
        "tangent, which always points outwards from the circle the motion should go round: the amplitude grows "
        "by 0.8 % a step, 48 % a period, 50 times over ten periods. Its spectral radius is above 1 at every "
        "Δt for an undamped oscillator.",
        per_period=50.0, schemes=(Scheme.EULER, Scheme.CENTRAL), tab=TAB_RADIUS),
    IntegrationTrial(
        "A stiff mode breaks the explicit schemes",
        "A 0.1 kg mass on a stiff spring hangs from the first: a 1049 Hz mode beside the 3.03 Hz one, as a fine "
        "mesh has. The run takes 50 steps per period of the slow mode, which gives the stiff one ωΔt = 43, a "
        "seventh of a step per period. Central difference and RK4 blow up within 20 ms, although nothing that "
        "matters happens up there. Average acceleration doesn't: what limits an explicit step is the fastest mode "
        "in the model, not the one you want.",
        per_period=50.0, periods=5.0, stiff_hz=1000.0, schemes=(Scheme.CENTRAL, Scheme.RK4, Scheme.AVERAGE),
        tab=TAB_RADIUS),
    IntegrationTrial(
        "HHT-α damps what it can't resolve",
        "The same stiff mode. It is in the closed form too (let go from rest, m1 sets it ringing), but no step "
        "this long can follow it, so average acceleration turns it into jagged noise that never dies (ρ = 1 at "
        "any step). HHT-α with α = −0.3 has ρ = 0.57 a step at ωΔt = 43: the noise is gone within a few steps, "
        "and the 3.03 Hz mode loses only ζ = 0.004 %. That is what finite-element codes use it for.",
        per_period=50.0, periods=5.0, stiff_hz=1000.0, schemes=(Scheme.AVERAGE, Scheme.HHT), alpha=-0.3,
        tab=TAB_DAMPING),
    IntegrationTrial(
        "The exact update and a sine force",
        "A 1 N sine at 2 Hz from rest, ten steps a period. The first-order-hold update every other page uses is "
        "exact for the oscillator at any Δt; its only error comes from treating the force as straight between "
        "samples: 1.3 % of the swing here. RK4's error is 3 %; central difference's and average "
        "acceleration's, 41 and 61 %: the start sets the undamped mass ringing at f<sub>n</sub> as well, and their "
        "period errors put that ringing out of phase.",
        x0=0.0, force=1.0, schemes=(Scheme.FOH, Scheme.CENTRAL, Scheme.AVERAGE, Scheme.RK4), tab=TAB_PERIOD),
)

TIPS = {
    "mass": "<p>The mass m.</p>",
    "stiffness": "<p>The spring k.</p>",
    "damping": "<p>The damper c.</p>",
    "stiff": "<p>A second mass, a tenth of the first, hung from it on a spring stiff enough that alone it would "
             "vibrate at this frequency: a mode far above the first, as a fine mesh has. 0: none.</p>",
    "per_period": "<p>The time step, as steps per period of the first mode: Δt = T₁ / this.</p>",
    "periods": "<p>How long to run, in periods of the first mode.</p>",
    "x0": "<p>The first mass starts here, at rest.</p>",
    "force": "<p>A sine force F sin 2πft on the first mass.</p>",
    "freq": "<p>Its frequency.</p>",
    "alpha": "<p>HHT-α's α, from −1/3 to 0 (0 is the average acceleration scheme): the more negative, the more "
             "it damps motion it can't resolve.</p>",
    "schemes": "<p>Which schemes to run.</p>",
}
TIME_TIP = ("<p>The first mass's displacement by each scheme (dots at the steps), against the closed form "
            "(dashed). A scheme that blows up stops.</p>")
ERROR_TIP = "<p>Each scheme's error against the closed form.</p>"
RADIUS_TIP = ("<p>The spectral radius ρ of each scheme's one-step map of a free, undamped vibration against "
              "Δt/T: above 1 the motion grows every step (unstable); below 1 the scheme damps it. The dotted lines "
              "are the step set, for each mode.</p>")
PERIOD_TIP = "<p>The period of the motion each scheme computes, against the true one, against Δt/T.</p>"
DAMPING_TIP = ("<p>The damping ratio of the motion each scheme computes for an undamped oscillator (numerical "
               "damping), against Δt/T. Negative: it grows.</p>")

NOTES_HTML = """
<!--BEFORE-->
<h3><a name="stepping"></a>Stepping an equation of motion</h3>
<p>Most structures are solved in time by stepping Mẍ + Cẋ + Kx = f(t) with a fixed Δt. A scheme replaces the
derivatives with differences, and each choice gets the motion wrong in its own way. For a linear system the error
is easy to see: let the scheme step a free, undamped oscillator. Its exact motion is a circle in (x, ẋ/ω); one step
of any scheme is a fixed 2 × 2 (or 3 × 3) matrix, the <b>amplification matrix</b>, and its eigenvalues
λ = ρe<sup>±iθ</sup> say everything:</p>
<ul>
<li><b>ρ &gt; 1</b>: the motion grows every step, the scheme is <b>unstable</b> at this Δt.</li>
<li><b>ρ &lt; 1</b>: it shrinks: <b>numerical damping</b>, ζ̄ = −ln ρ / |ln λ|.</li>
<li><b>θ ≠ ωΔt</b>: the period is wrong, <b>period elongation</b> (T̄ − T)/T = ωΔt/θ − 1.</li>
</ul>
<p>All of these depend only on Ω = ωΔt = 2πΔt/T, so each scheme's behaviour is a set of curves against Δt/T
(the tabs on the right).</p>

<h3><a name="central"></a>Central difference</h3>
<p>ẍ<sub>n</sub> ≈ (x<sub>n+1</sub> − 2x<sub>n</sub> + x<sub>n−1</sub>)/Δt²: explicit (no equations to solve when
M is diagonal), second order, no numerical damping, and the period comes out short. It is stable only for
<b>ωΔt ≤ 2</b>, for the <i>highest</i> ω in the model: Δt ≤ T<sub>min</sub>/π. That is the Courant condition of
explicit crash and blast codes: a fine mesh has a very fast highest mode, so the step has to be tiny however slow
the motion of interest.</p>

<h3><a name="newmark"></a>Newmark and HHT-α</h3>
<p>Newmark assumes how the acceleration varies over a step: x<sub>n+1</sub> = x<sub>n</sub> + Δt ẋ<sub>n</sub> +
Δt²[(½ − β)ẍ<sub>n</sub> + βẍ<sub>n+1</sub>], ẋ<sub>n+1</sub> = ẋ<sub>n</sub> + Δt[(1 − γ)ẍ<sub>n</sub> +
γẍ<sub>n+1</sub>], and the equation of motion at n + 1 closes it (an implicit scheme: a linear solve per step).</p>
<ul>
<li><b>Average acceleration</b> (γ = ½, β = ¼, the trapezoidal rule): <b>unconditionally stable</b>, no numerical
damping, second order; the period comes out long, by about (ωΔt)²/12.</li>
<li><b>Linear acceleration</b> (γ = ½, β = ⅙): more accurate, but stable only for ωΔt ≤ 2√3.</li>
<li>γ &gt; ½ damps, but drops to first order. <b>HHT-α</b> (Hilber, Hughes and Taylor) instead evaluates the
stiffness and damping forces at t<sub>n+1+α</sub>, with γ = ½ − α, β = (1 − α)²/4, α ∈ [−⅓, 0]: still second order and
unconditionally stable, with damping that grows with Ω. It removes the unresolved high modes' noise and barely
touches the low ones: the default in many finite-element codes.</li>
</ul>

<h3><a name="rk4"></a>Runge–Kutta</h3>
<p>Classical RK4 on the first-order form ż = Az + Bf is fourth order: very accurate in period at small steps. It is
explicit, so conditionally stable (ωΔt ≤ 2√2 for an undamped oscillator), and it damps: ρ &lt; 1 at any step.
<b>Explicit Euler</b>, z<sub>n+1</sub> = z<sub>n</sub> + Δt ż<sub>n</sub>, is the cautionary tale: for an undamped
oscillator ρ = √(1 + Ω²) &gt; 1 at every Δt, and the energy grows every step.</p>

<h3><a name="exact"></a>The exact update</h3>
<p>For a linear model there is a better way. Over a step the free motion is exactly z<sub>n+1</sub> =
e<sup>AΔt</sup>z<sub>n</sub>; if the force is taken as linear between samples (a first-order hold), its part is
exact too. This <b>FOH update</b> is what the Simulation page and the One mass pages use: stable and exact in
period and amplitude at any Δt, for any stiffness (ρ = e<sup>−ζωΔt</sup>, the oscillator's own). Its only error
is in the force between samples, which is second order in Δt. It needs the matrix exponential, which is cheap for
small models and why big nonlinear codes still use Newmark or central difference.</p>

<h3><a name="try-it"></a>Things to try</h3>
<p>Each of these sets up the oscillator, the step and the schemes (Things to try, top left).</p>
<!--TRIALS-->
<!--TAKEAWAYS-->
"""
NOTES_HTML = NOTES_HTML.replace("<!--BEFORE-->", before(
    ("sdof_page", "equation", "The equation of motion"),
    ("sim_page", "undamped-normal-modes-real", "Undamped normal modes"),
    ("continuum_page", "dispersion", "A mesh's modes against the continuum's"),
)).replace("<!--TRIALS-->", trials_html(TRIALS)).replace("<!--TAKEAWAYS-->", takeaways(
    "One step of a scheme on a free oscillator is a matrix: its eigenvalues give stability (ρ ≤ 1), period error "
    "and numerical damping, all functions of ωΔt.",
    "Explicit schemes are conditionally stable, and the limit is set by the fastest mode in the model: "
    "ωΔt ≤ 2 for central difference.",
    "Newmark's average acceleration is stable at any step but lengthens the period; HHT-α adds damping that "
    "grows with frequency, removing unresolved modes.",
    "Stable is not accurate: resolve the modes you care about with ten or more steps a period.",
    "For a linear model the FOH update is exact at any step, apart from the force between samples.",
))


class IntegrationPage(QtWidgets.QWidget):
    """Controls on the left, the motion and its error in the centre, the schemes' curves on the right."""

    follows_chain = False  # MainWindow: this page has its own model, not the chain

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.model = Model()
        self.load = Load()
        self.T = self.h = 1.0  # set from the inputs below

        self.trials = TrialBox([t.name for t in TRIALS],
                               "<p>Each sets up the oscillator, the step and the schemes to show one idea from "
                               "the Theory tab.</p>")
        self.trials.picked.connect(self.load_trial)

        osc = Form("Oscillator")
        self.mass = osc.number("Mass m:", 0.001, 1e4, 1.0, 3, " kg", TIPS["mass"])
        self.stiffness = osc.number("Stiffness k:", 0.01, 1e8, 400.0, 2, " N/m", TIPS["stiffness"])
        self.damping = osc.number("Damping c:", 0.0, 1e6, 0.0, 3, " N·s/m", TIPS["damping"])
        self.stiff = osc.number("Stiff mode:", 0.0, 1e5, 0.0, 1, " Hz", TIPS["stiff"])
        self.osc_info = note_label()
        osc.row(self.osc_info)

        run = Form("Run")
        self.per_period = run.number("Steps a period:", 0.2, 1000.0, 10.0, 2, "", TIPS["per_period"])
        self.periods = run.number("Periods:", 0.5, 100.0, 10.0, 1, "", TIPS["periods"])
        self.x0 = run.number("Start x₀:", -1.0, 1.0, 0.01, 4, " m", TIPS["x0"])
        self.force = run.number("Force F:", 0.0, 1e4, 0.0, 3, " N", TIPS["force"])
        self.freq = run.number("at f:", 0.01, 1e4, 2.0, 3, " Hz", TIPS["freq"])
        self.step_info = note_label()
        run.row(self.step_info)

        pick = Form("Schemes")
        self.checks: dict[Scheme, QtWidgets.QCheckBox] = {}
        for s in SCHEMES:
            box = QtWidgets.QCheckBox(s.value)
            box.setToolTip(TIPS["schemes"])
            self.checks[s] = box
            pick.row(box)
        self.alpha = pick.number("HHT α:", -1 / 3, 0.0, -0.1, 3, "", TIPS["alpha"])
        self.alpha.setSingleStep(0.05)

        connect_all((self.mass, self.stiffness, self.damping, self.stiff, self.per_period, self.periods, self.x0,
                     self.force, self.freq, self.alpha, *self.checks.values()), self._on_inputs)

        # --- centre
        self.time_plot = plot("Displacement x₁", "m", "Time", "s", height=150)
        self.time_plot.setToolTip(TIME_TIP)
        add_legend(self.time_plot.getPlotItem(), offset=(-10, 10))
        self.exact_item = pg.PlotDataItem(name="closed form")
        self.time_plot.addItem(self.exact_item)
        self.err_plot = plot("Error", "m", "Time", "s", log_y=True, height=120)
        self.err_plot.setToolTip(ERROR_TIP)
        self.err_plot.setXLink(self.time_plot)
        self.run_items: dict[Scheme, pg.PlotDataItem] = {}
        self.err_items: dict[Scheme, pg.PlotDataItem] = {}
        for s in SCHEMES:
            self.run_items[s] = pg.PlotDataItem(name=s.value)
            self.err_items[s] = pg.PlotDataItem()
            self.time_plot.addItem(self.run_items[s])
            self.err_plot.addItem(self.err_items[s])
        self.readout = readout_label()

        # --- right
        def curve_plot(left: str, units: str | None, tip: str) -> tuple[pg.PlotWidget, dict, list]:
            p = plot(left, units, "Δt / T", None, log_x=True, height=200)
            p.setToolTip(tip)
            p.setMouseEnabled(x=False, y=False)
            add_legend(p.getPlotItem(), offset=(10, 10))
            items = {s: pg.PlotDataItem(name=s.value) for s in SCHEMES}
            marks = [pg.InfiniteLine(angle=90), pg.InfiniteLine(angle=90)]
            for item in (*marks, *items.values()):
                p.addItem(item)
            return p, items, marks

        self.radius_plot, self.radius_items, self.radius_marks = curve_plot("Spectral radius ρ", None, RADIUS_TIP)
        self.unit_line = pg.InfiniteLine(angle=0, pos=1.0)
        self.radius_plot.addItem(self.unit_line)
        self.period_plot, self.period_items, self.period_marks = curve_plot("Period error", "%", PERIOD_TIP)
        self.damp_plot, self.damp_items, self.damp_marks = curve_plot("Numerical damping ζ̄", "%", DAMPING_TIP)
        self.curve_note = note_label()

        self.notes = theory(NOTES_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(column(self.radius_plot, stretch=False), "Spectral radius")
        self.tabs.addTab(column(self.period_plot, stretch=False), "Period error")
        self.tabs.addTab(column(self.damp_plot, stretch=False), "Numerical damping")
        self.tabs.addTab(self.notes, "Theory")
        self.tabs.currentChanged.connect(lambda _: self._draw_curves())
        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(self.tabs, 1)
        rv.addWidget(self.curve_note)

        three_panes(self, column(self.trials, osc, run, pick),
                    stack((self.time_plot, 3), (self.err_plot, 2), (self.readout, 0)), right)
        self.load_trial(0, note=False)
        self.apply_theme()

    # ------------------------------------------------------------- inputs
    def load_trial(self, index: int, note: bool = True) -> None:
        t = TRIALS[index]
        for box, value in ((self.mass, 1.0), (self.stiffness, 400.0), (self.damping, t.damping),
                           (self.stiff, t.stiff_hz), (self.per_period, t.per_period), (self.periods, t.periods),
                           (self.x0, t.x0), (self.force, t.force), (self.freq, t.freq_hz), (self.alpha, t.alpha)):
            set_quietly(box, value)
        for s, box in self.checks.items():
            set_quietly(box, s in t.schemes)
        if note:
            self.trials.show_note(t.note)
        self.tabs.setCurrentIndex(t.tab)
        self._on_inputs()

    def _on_inputs(self) -> None:
        self.model = Model(self.mass.value(), self.stiffness.value(), self.damping.value(), self.stiff.value())
        self.load = Load(self.x0.value(), self.force.value(), self.freq.value())
        f = self.model.frequencies_hz()
        self.T = 1.0 / f[0] if f[0] > 0 else 1.0
        self.h = self.T / self.per_period.value()
        self.osc_info.setText(
            "Modes: " + ", ".join(f"{v:.4g} Hz" for v in f) + f"; T₁ = {fmt(self.T, 's')}, ζ = "
            f"{self.model.damping / (2 * math.sqrt(self.model.stiffness * self.model.mass)):.3g}.")
        self.step_info.setText(
            f"Δt = {fmt(self.h, 's')}: ωΔt = " + ", ".join(f"{TWO_PI * v * self.h:.3g}" for v in f)
            + " (mode by mode); Δt/T = " + ", ".join(f"{v * self.h:.3g}" for v in f) + ".")
        self.redraw()

    # ------------------------------------------------------------- drawing
    def apply_theme(self) -> None:
        c = colors
        self.exact_item.setPen(pg.mkPen(c.strong, width=2, style=QtCore.Qt.PenStyle.DashLine))
        for i, s in enumerate(SCHEMES):
            color = c.mass[i % len(c.mass)]
            for item in (self.run_items[s], self.radius_items[s], self.period_items[s], self.damp_items[s]):
                item.setPen(pg.mkPen(color, width=2))
            self.run_items[s].setSymbolBrush(pg.mkBrush(color))
            self.run_items[s].setSymbolPen(None)
            self.err_items[s].setPen(pg.mkPen(color, width=1.5))
        for line in (*self.radius_marks, *self.period_marks, *self.damp_marks):
            line.setPen(pg.mkPen(c.grey, width=1.5, style=QtCore.Qt.PenStyle.DotLine))
        self.unit_line.setPen(pg.mkPen(c.faint, width=1))
        self.redraw()

    def redraw(self) -> None:
        self._draw_runs()
        self._draw_curves()
        settle_plots(self)

    def _chosen(self) -> list[Scheme]:
        return [s for s, box in self.checks.items() if box.isChecked()]

    @staticmethod
    def _legend(p: pg.PlotWidget, entries: list[tuple[pg.PlotDataItem, str]]) -> None:
        legend = p.getPlotItem().legend
        legend.clear()
        for item, name in entries:
            legend.addItem(item, name)

    def _draw_runs(self) -> None:
        h, model, load = self.h, self.model, self.load
        span = self.periods.value() * self.T
        steps = min(MOST_STEPS, max(1, int(round(span / h))))
        span = steps * h
        tf = np.linspace(0.0, span, POINTS + 1)
        self.exact_item.setData(tf, exact(model, load, tf))
        t = np.arange(steps + 1) * h
        reference = exact(model, load, t)
        swing = float(np.max(np.abs(reference))) or 1.0
        lines = []
        dots = steps <= 200
        for s in SCHEMES:
            item, err = self.run_items[s], self.err_items[s]
            if s not in self._chosen():
                item.setData([], [])
                err.setData([], [])
                continue
            x = simulate(s, model, load, h, steps, self.alpha.value())
            ok = np.isfinite(x)
            item.setData(t[ok], x[ok], symbol="o" if dots else None, symbolSize=5)
            e = np.abs(x - reference)
            good = ok & (e > 0)
            err.setData(t[good], e[good])
            if ok.all():
                lines.append(f"{s.value}: largest error {100 * float(np.max(e)) / swing:.3g} % of the swing")
            else:
                gone = t[np.argmin(ok)]
                lines.append(f"{s.value}: <b>blew up</b> by t = {fmt(float(gone), 's')}")
        chosen = self._chosen()
        self._legend(self.time_plot, [(self.exact_item, "closed form")]
                     + [(self.run_items[s], s.value) for s in chosen])
        errors = [item.yData for s, item in self.err_items.items() if s in chosen and item.yData is not None
                  and item.yData.size]
        if errors:
            top = max(float(np.max(e)) for e in errors)
            floor = max(min(float(np.min(e)) for e in errors), 1e-7 * swing)
            self.err_plot.setYRange(math.log10(floor), math.log10(max(top, floor * 10)) + 0.3, padding=0)
        self.time_plot.setXRange(0.0, span, padding=0.01)
        lo, hi = float(np.min(reference)), float(np.max(reference))
        pad = 0.25 * (hi - lo or abs(hi) or 1.0)
        self.time_plot.setYRange(lo - pad, hi + pad, padding=0)
        self.readout.setText(f"{steps} steps of {fmt(h, 's')}. " + "; ".join(lines) + "." if lines
                             else "Pick a scheme on the left.")

    def _draw_curves(self) -> None:
        i = self.tabs.currentIndex()
        if i > TAB_DAMPING:
            return
        zeta = 0.0
        alpha = self.alpha.value()
        O = TWO_PI * RATIOS
        plot_, items, marks = ((self.radius_plot, self.radius_items, self.radius_marks),
                               (self.period_plot, self.period_items, self.period_marks),
                               (self.damp_plot, self.damp_items, self.damp_marks))[i]
        for s in SCHEMES:
            if s not in self._chosen():
                items[s].setData([], [])
                continue
            radius, period, damping = spectral_curves(s, O, zeta, alpha)
            y = (radius, 100 * period, 100 * damping)[i]
            ok = np.isfinite(y)
            items[s].setData(RATIOS[ok], y[ok], connect="finite")
        self._legend(plot_, [(items[s], s.value) for s in self._chosen()])
        f = self.model.frequencies_hz()
        for line, v in zip(marks, (*f, None, None)):
            line.setVisible(v is not None)
            if v is not None:
                line.setValue(math.log10(max(v * self.h, 1e-6)))
        plot_.setXRange(-2, 1, padding=0)
        if i == TAB_RADIUS:
            plot_.setYRange(0.0, 2.0, padding=0)
        elif i == TAB_PERIOD:
            plot_.setYRange(-50.0, 100.0, padding=0)
        else:
            plot_.setYRange(-10.0, 20.0, padding=0)
        limits = [f"{s.value} ωΔt ≤ {STABLE_LIMIT[s]:.3g}" for s in self._chosen()
                  if 0 < STABLE_LIMIT[s] < math.inf]
        self.curve_note.setText(
            "Undamped oscillator; dotted: this Δt for each mode. "
            + ("Stable while " + ", ".join(limits) + "." if limits else ""))
