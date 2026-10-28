"""Energy methods page: Rayleigh's quotient, Dunkerley's sum, Rayleigh–Ritz, Holzer's table and Lagrange's M and K,
on the Simulation page's chain.

Left: the trial shape for Rayleigh's quotient (with some of mode 2 mixed in, and passes of inverse iteration),
and the shapes Rayleigh–Ritz keeps. Centre: the trial shape against mode 1's, and every estimate of f₁ on one
axis. Right: Rayleigh's error against the shape's, Rayleigh–Ritz's estimates as shapes are added, Holzer's
residual, the energies Lagrange's equations start from, and the Theory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.energy_methods import (
    BASES,
    SHAPES,
    TWO_PI,
    Basis,
    Shape,
    contaminated,
    dunkerley,
    dunkerley_terms,
    exact,
    holzer,
    holzer_residual,
    holzer_roots,
    iterate,
    rayleigh,
    rayleigh_ritz,
    ritz_basis,
    ritz_table,
    trial_shape,
)
from ..core.model import ChainSystem
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
    set_quietly,
    stack,
    theory,
    three_panes,
    trials_html,
)
from .links import before, takeaways
from .style import MAX_DOF, colors
from .theming import add_legend, mute

TAB_ERROR, TAB_RITZ, TAB_HOLZER, TAB_LAGRANGE = range(4)
EPS_SPAN = 0.5  # the Shape error tab's ε runs over ±this


def chain(masses=(1.0, 1.0, 1.0, 1.0), stiffness=(400.0, 400.0, 400.0, 400.0)) -> ChainSystem:
    n = len(masses)
    return ChainSystem(np.array(masses, dtype=float), np.array(stiffness, dtype=float), np.full(n, 2.0))


@dataclass(frozen=True)
class EnergyTrial:
    name: str
    note: str
    shape: Shape = Shape.STATIC
    eps: float = 0.0
    iterations: int = 0
    basis: Basis = Basis.KRYLOV
    shapes: int = 2
    tab: int = TAB_ERROR
    masses: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    stiffness: tuple[float, ...] = (400.0, 400.0, 400.0, 400.0)
    holzer_hz: float = 1.0

    def system(self) -> ChainSystem:
        return chain(self.masses, self.stiffness)


TRIALS = (
    EnergyTrial(
        "Rayleigh from the static deflection",
        "The uniform chain (four 1 kg masses, 400 N/m springs) sagging under its own weight, as a guess at mode "
        "1's shape. Its Rayleigh quotient xᵀKx / xᵀMx gives f₁ = 1.112 Hz against the exact 1.105 Hz: 0.55 % "
        "high. The shapes are hard to tell apart; the error in frequency is smaller still, because it is "
        "second order in the shape's."),
    EnergyTrial(
        "A straight line, then one pass of iteration",
        "A straight line, x<sub>i</sub> ∝ i, is a worse guess: 5.1 % high. One pass of inverse iteration "
        "(x ← K⁻¹Mx: the static deflection under the inertia forces of the guessed motion) brings it to 0.063 % "
        "high.",
        shape=Shape.LINEAR, iterations=1),
    EnergyTrial(
        "The error is second order",
        "On a uniform chain the quarter sine is mode 1 exactly. With 10 % of mode 2 mixed in (ε = 0.1) the "
        "estimate is 4.6 % high; with 20 %, 16.6 %, nearly four times as much. The Shape error tab draws the "
        "parabola: it is flat at ε = 0, which is why a rough shape still gives a good frequency, and it never "
        "dips below the exact f₁.",
        shape=Shape.SINE, eps=0.1, tab=TAB_ERROR),
    EnergyTrial(
        "Inverse iteration from a poor guess",
        "Only the last mass moving: 3.18 Hz, 188 % high (it is the last mass on its own spring). One pass of "
        "inverse iteration: 5.1 % high; two: 0.063 %. Each pass shrinks mode r's share by ω₁²/ω<sub>r</sub>², "
        "0.12 for mode 2 here, and the frequency's error by its square.",
        shape=Shape.TIP, iterations=1),
    EnergyTrial(
        "Dunkerley from below",
        "Dunkerley adds up each mass alone on the massless chain: 1/ω₁² ≈ Σ m<sub>i</sub>f<sub>ii</sub>, with "
        "f = K⁻¹. On the uniform chain the four alone are at 3.18, 2.25, 1.84 and 1.59 Hz and the sum gives "
        "1.007 Hz, 8.9 % low. With Rayleigh's static estimate the exact f₁ is bracketed: 1.007 ≤ 1.105 ≤ 1.112 "
        "Hz."),
    EnergyTrial(
        "A heavy tip: Dunkerley does better",
        "The last mass 5 kg. It now dominates Dunkerley's sum (alone it is at 0.71 Hz), and the estimate is "
        "0.624 Hz against the exact 0.653, 4.4 % low. Rayleigh's static shape: 0.25 % high. The more one mass "
        "dominates, the closer Dunkerley comes.",
        masses=(1.0, 1.0, 1.0, 5.0)),
    EnergyTrial(
        "Rayleigh–Ritz from above",
        "Two Ritz shapes (the static deflection and one more pass of K⁻¹M) on the uniform chain: f₁ is exact to "
        "four figures, and f₂ is 3.30 Hz against 3.18, 3.6 % high. With three, f₂ is 3.184 and f₃ 5.02 Hz "
        "against 4.88. Every estimate sits above its mode and comes down as shapes are added; with all N it is "
        "exact.",
        tab=TAB_RITZ, shapes=2),
    EnergyTrial(
        "A poor basis: a stiff first spring",
        "k₁ = 4000 N/m, ten times the others, holds m1 nearly still. Powers of i start with a straight line, "
        "which moves m1 a quarter as far as m4, far more than the stiff spring lets it: with two of them f₁ is 1.63 Hz "
        "against 1.38, 18 % high, and f₂ 6.04 against 3.89. The Krylov shapes come from the chain itself, and "
        "one of them is already within 0.8 %.",
        basis=Basis.POWERS, shapes=2, tab=TAB_RITZ, stiffness=(4000.0, 400.0, 400.0, 400.0)),
    EnergyTrial(
        "Holzer's table",
        "Holzer guesses ω, sets x₁ = 1 and walks along the chain: each mass takes its inertia force m ω² x out "
        "of the spring force reaching it and passes the rest on. At 1.00 Hz, 61 N are left at the free end; the "
        "residual changes sign at 1.105 Hz, then at 3.183, 4.877 and 5.982 Hz: the four natural frequencies, "
        "found without an eigenvalue solver.",
        tab=TAB_HOLZER, holzer_hz=1.0),
)

TIPS = {
    "dof": "<p>The number of masses: the same chain as the Simulation page, so changing it here changes it there "
           "too.</p>",
    "shape": "<p>The shape Rayleigh's quotient is worked out for.</p>",
    "eps": "<p>How much of mode 2's shape (its largest entry 1) is added to the trial shape: a known error in the "
           "shape, to see what it costs in frequency.</p>",
    "iterations": "<p>Passes of inverse iteration, x ← K⁻¹Mx: each is the static deflection under the inertia "
                  "forces of the shape before.</p>",
    "basis": "<p>The shapes Rayleigh–Ritz restricts the chain to.</p>",
    "shapes": "<p>How many of them it keeps (r ≤ N).</p>",
    "holzer": "<p>The trial frequency Holzer's table is worked out at.</p>",
}
SHAPE_TIP = ("<p>Mode 1's exact shape (dashed) and the trial shape Rayleigh's quotient is worked out for, both "
             "scaled to a largest entry of 1; 0 is the ground.</p>")
ESTIMATES_TIP = ("<p>Every estimate of f₁ on one axis, against the exact one (the dotted line): Dunkerley's "
                 "below it, Rayleigh's and Rayleigh–Ritz's above.</p>")
ERROR_TIP = ("<p>Rayleigh's f₁ for the trial shape plus ε of mode 2's, against ε. The exact f₁ is dotted; the "
             "dot is the ε set on the left.</p>")
RITZ_TIP = ("<p>Each mode's Rayleigh–Ritz frequency against the number of shapes kept, r; the exact frequencies "
            "dotted. Mode j appears once r ≥ j.</p>")
HOLZER_TIP = ("<p>The force left at the free end of Holzer's table (for x₁ = 1), against the trial frequency; "
              "it changes sign at each natural frequency (dotted). Drawn as sign · log₁₀(1 + |R|/k₁), since "
              "it grows as ω<sup>2N</sup>.</p>")

NOTES_HTML = """
<!--BEFORE-->
<h3><a name="lagrange"></a>The equations from the energies</h3>
<p>The chain's kinetic and potential energies are</p>
<p align="center">T = ½ Σ m<sub>i</sub>ẋ<sub>i</sub>² = ½ ẋᵀMẋ, &nbsp;
V = ½ Σ k<sub>i</sub>(x<sub>i</sub> − x<sub>i−1</sub>)² = ½ xᵀKx &nbsp; (x₀ = 0, the ground).</p>
<p><b>Lagrange's equations</b>, d/dt(∂T/∂ẋ<sub>i</sub>) − ∂T/∂x<sub>i</sub> + ∂V/∂x<sub>i</sub> = Q<sub>i</sub>,
turn these into Mẍ + Kx = f with no free-body diagrams: M and K are just the energies' second derivatives
(the <i>Lagrange</i> tab works them out for the chain you have). The same recipe works for any coordinates:
angles, a beam's modal amplitudes, the generalized coordinates of the pages that follow.</p>

<h3><a name="rayleigh"></a>Rayleigh's quotient: an upper bound</h3>
<p>In a mode x = X sin ωt, the largest kinetic energy ½ω²XᵀMX equals the largest potential energy ½XᵀKX, so</p>
<p align="center">ω² = XᵀKX / XᵀMX.</p>
<p>Put in any shape X and this <b>Rayleigh quotient</b> R(X) is an estimate of ω₁². Written in the modes,
X = φ₁ + Σ ε<sub>r</sub>φ<sub>r</sub> (mass-normalized),</p>
<p align="center">R(X) = (ω₁² + Σ ε<sub>r</sub>²ω<sub>r</sub>²) / (1 + Σ ε<sub>r</sub>²)
= ω₁² + Σ ε<sub>r</sub>²(ω<sub>r</sub>² − ω₁²) / (1 + Σ ε<sub>r</sub>²).</p>
<p>So R is never below ω₁² (every added term is positive), and its error is <b>second order</b> in the shape's:
a shape 10 % wrong gives a frequency about 1 % wrong, times (ω<sub>r</sub>/ω₁)² − 1. A good shape to try is the
static deflection under the masses' weights: the inertia forces of mode 1 are distributed much like the
weights.</p>

<h3><a name="iteration"></a>Improving the shape</h3>
<p>The static deflection under the inertia forces of a guessed motion, X' = K⁻¹MX, is a better guess: each
mode's share is divided by its ω<sub>r</sub>², so mode r's share against mode 1's falls by ω₁²/ω<sub>r</sub>² per
pass (<b>inverse iteration</b>, or Stodola's method). The static-deflection shape is one pass from a rigid
guess.</p>

<h3><a name="dunkerley"></a>Dunkerley: a lower bound</h3>
<p>With the flexibility matrix F = K⁻¹, the trace of FM is Σ 1/ω<sub>r</sub>², so</p>
<p align="center">1/ω₁² &lt; Σ<sub>r</sub> 1/ω<sub>r</sub>² = Σ<sub>i</sub> m<sub>i</sub>f<sub>ii</sub>.</p>
<p>Each term m<sub>i</sub>f<sub>ii</sub> is 1/ω² of mass i alone on the massless chain, so Dunkerley's estimate
"adds up the masses one at a time". It is below ω₁ by how much the higher modes contribute to the trace: it does
best when one mass dominates. With Rayleigh's it brackets the exact value.</p>

<h3><a name="ritz"></a>Rayleigh–Ritz</h3>
<p>Restrict the motion to a combination of r shapes, x = Qa, and make Rayleigh's quotient stationary in a: that is
the small eigenproblem</p>
<p align="center">(QᵀKQ) a = ω² (QᵀMQ) a.</p>
<p>It gives r frequencies at once, and each is above its mode's exact value (the interlacing theorem), falling as
shapes are added; with all N shapes it is exact. The quality depends on the shapes: shapes the structure makes
itself (static deflections, and K⁻¹M applied to them again: <b>Krylov</b> or load-dependent Ritz vectors) do far
better than generic polynomials that ignore where the structure is stiff. Every reduced model on the later pages
(substructuring, modal coupling, finite elements) is a Rayleigh–Ritz model.</p>

<h3><a name="holzer"></a>Holzer's table</h3>
<p>For a chain, guess ω, set x₁ = 1 and walk along it: the ground spring pulls k₁x₁; mass i keeps
m<sub>i</sub>ω²x<sub>i</sub> of the force reaching it and passes the rest, F<sub>i+1</sub>, to the next spring,
which stretches by F<sub>i+1</sub>/k<sub>i+1</sub>. At the free end nothing should be left. The force that is left,
R(ω), is a polynomial in ω² whose roots are the natural frequencies; it changes sign at each one, so they can be
found by bisection, mode by mode, with a pencil. Holzer's method was the standard for torsional drive trains
(see the Torsional vibration page) before computers.</p>

<h3><a name="try-it"></a>Things to try</h3>
<p>Each of these loads a chain into the Simulation page and sets up the estimates (Things to try, top left).</p>
<!--TRIALS-->
<!--TAKEAWAYS-->
"""
NOTES_HTML = NOTES_HTML.replace("<!--BEFORE-->", before(
    ("sim_page", "undamped-normal-modes-real", "Undamped normal modes"),
    ("sim_page", "rayleigh-quotient", "Rayleigh's quotient"),
    ("sim_page", "energy", "Energy"),
)).replace("<!--TRIALS-->", trials_html(TRIALS)).replace("<!--TAKEAWAYS-->", takeaways(
    "M and K are the second derivatives of the kinetic and potential energies (Lagrange).",
    "Rayleigh's quotient is never below ω₁², and its error is second order in the shape's.",
    "Dunkerley's sum Σ m<sub>i</sub>f<sub>ii</sub> bounds ω₁ from below; with Rayleigh's it brackets it.",
    "Rayleigh–Ritz gives r frequencies, each from above; good shapes come from the structure itself.",
    "Holzer's residual changes sign at each natural frequency of a chain.",
))


def matrix_html(A: np.ndarray, digits: int = 4) -> str:
    rows = "".join("<tr>" + "".join(f"<td align='right'>&nbsp;{v:.{digits}g}&nbsp;</td>" for v in row) + "</tr>"
                   for row in A)
    return f"<table cellspacing='0' cellpadding='1'>{rows}</table>"


class EnergyMethodsPage(QtWidgets.QWidget):
    """Controls on the left, the shapes and estimates in the centre, the methods' own plots on the right."""

    dof_requested = QtCore.Signal(int)  # the user changed N here; the main window owns the system
    system_requested = QtCore.Signal(object)  # a trial's chain, for the Simulation page
    edit_parameters = QtCore.Signal()  # the user wants the Simulation page's parameter panel

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.system = chain()
        self._dirty = True

        self.trials = TrialBox([t.name for t in TRIALS],
                               "<p>Each loads a chain into the Simulation page and sets up the estimates to show "
                               "one idea from the Theory tab.</p>")
        self.trials.picked.connect(self.load_trial)

        ch = Form("Chain")
        self.dof = QtWidgets.QSpinBox()
        self.dof.setRange(1, MAX_DOF)
        self.dof.setToolTip(TIPS["dof"])
        self.dof.valueChanged.connect(self.dof_requested)
        ch.add("Number of masses:", self.dof, TIPS["dof"])
        link = QtWidgets.QLabel("Masses and springs are edited on the <a href='#sim'>Simulation page</a>.")
        link.setWordWrap(True)
        mute(link)
        link.linkActivated.connect(lambda _: self.edit_parameters.emit())
        ch.row(link)
        self.chain_info = note_label()
        ch.row(self.chain_info)

        ray = Form("Rayleigh's quotient")
        self.shape = combo([s.value for s in SHAPES], TIPS["shape"])
        ray.add("Trial shape:", self.shape, TIPS["shape"])
        self.eps = ray.number("Plus ε of mode 2:", -1.0, 1.0, 0.0, 2, "", TIPS["eps"])
        self.eps.setSingleStep(0.05)
        self.iterations = ray.number("Iteration passes:", 0, 6, 0, 0, "", TIPS["iterations"])

        ritz = Form("Rayleigh–Ritz")
        self.basis = combo([b.value for b in BASES], TIPS["basis"])
        ritz.add("Shapes:", self.basis, TIPS["basis"])
        self.r = ritz.number("Keep r:", 1, MAX_DOF, 2, 0, "", TIPS["shapes"])

        connect_all((self.shape, self.eps, self.iterations, self.basis, self.r), self.redraw)

        # --- centre
        self.shape_plot = plot("Displacement (largest 1)", None, "Mass", None, height=150)
        self.shape_plot.setToolTip(SHAPE_TIP)
        self.shape_plot.setMouseEnabled(x=False, y=False)
        add_legend(self.shape_plot.getPlotItem(), offset=(10, 10))
        self.mode_item = pg.PlotDataItem(name="mode 1 (exact)")
        self.trial_item = pg.PlotDataItem(name="trial shape", symbol="o", symbolSize=9)
        self.ritz_item = pg.PlotDataItem(name="Rayleigh–Ritz mode 1", symbol="s", symbolSize=7)
        for item in (self.mode_item, self.ritz_item, self.trial_item):
            self.shape_plot.addItem(item)
        self.est_plot = plot("", None, "Frequency f₁", "Hz", height=120)
        self.est_plot.setToolTip(ESTIMATES_TIP)
        self.est_plot.setMouseEnabled(x=False, y=False)
        self.est_plot.getAxis("left").setTicks([[(0, "Dunkerley"), (1, "Rayleigh"), (2, "Rayleigh–Ritz")]])
        self.est_plot.getAxis("left").setWidth(90)
        self.exact_line = pg.InfiniteLine(angle=90)
        self.est_items = [marker(12, "d"), marker(12, "o"), marker(12, "s")]
        self.est_labels = [pg.TextItem(anchor=(0.5, 1.3)) for _ in range(3)]
        self.est_plot.addItem(self.exact_line)
        for item in (*self.est_items, *self.est_labels):
            self.est_plot.addItem(item)
        self.readout = readout_label()

        # --- right
        self.err_plot = plot("Rayleigh's f₁", "Hz", "ε (mode 2's share in the shape)", None, height=200)
        self.err_plot.setToolTip(ERROR_TIP)
        self.err_plot.setMouseEnabled(x=False, y=False)
        self.err_curve = pg.PlotDataItem()
        self.err_exact = pg.InfiniteLine(angle=0)
        self.err_now = marker(12)
        for item in (self.err_exact, self.err_curve, self.err_now):
            self.err_plot.addItem(item)
        self.err_note = note_label()

        self.ritz_plot = plot("Frequency", "Hz", "Shapes kept r", None, height=200)
        self.ritz_plot.setToolTip(RITZ_TIP)
        self.ritz_plot.setMouseEnabled(x=False, y=False)
        self.ritz_curves: list[pg.PlotDataItem] = []
        self.ritz_exact: list[pg.InfiniteLine] = []
        self.ritz_now = pg.InfiniteLine(angle=90)
        self.ritz_plot.addItem(self.ritz_now)
        self.ritz_note = note_label()

        self.holzer_plot = plot("Residual (scaled)", None, "Trial frequency", "Hz", height=180)
        self.holzer_plot.setToolTip(HOLZER_TIP)
        self.holzer_plot.setMouseEnabled(x=True, y=False)
        self.holzer_curve = pg.PlotDataItem()
        self.holzer_zero = pg.InfiniteLine(angle=0, pos=0)
        self.holzer_trial = pg.InfiniteLine(angle=90, movable=True)
        self.holzer_trial.sigPositionChanged.connect(lambda *_: self._on_holzer_line())
        self.holzer_roots: list[pg.InfiniteLine] = []
        for item in (self.holzer_zero, self.holzer_curve, self.holzer_trial):
            self.holzer_plot.addItem(item)
        hz_row = Form("")
        hz_row.setFlat(True)
        self.holzer_hz = hz_row.number("Trial frequency:", 0.01, 1e4, 1.0, 3, " Hz", TIPS["holzer"])
        self.holzer_hz.valueChanged.connect(lambda *_: self._draw_holzer())
        self.holzer_table = readout_label()
        self.holzer_table.setTextFormat(QtCore.Qt.TextFormat.RichText)

        self.lagrange = theory("")

        self.notes = theory(NOTES_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(column(self.err_plot, self.err_note, stretch=False), "Shape error")
        self.tabs.addTab(column(self.ritz_plot, self.ritz_note, stretch=False), "Rayleigh–Ritz")
        self.tabs.addTab(column(self.holzer_plot, hz_row, self.holzer_table, stretch=False), "Holzer")
        self.tabs.addTab(self.lagrange, "Lagrange")
        self.tabs.addTab(self.notes, "Theory")
        self.tabs.currentChanged.connect(lambda _: self._draw_tab())

        three_panes(self, column(self.trials, ch, ray, ritz),
                    stack((self.shape_plot, 3), (self.est_plot, 2), (self.readout, 0)), self.tabs)
        self.apply_theme()
        self.set_system(self.system)

    # ------------------------------------------------------------- inputs
    def set_system(self, system: ChainSystem, result=None) -> None:
        """The Simulation page's chain. Drawn now if the page shows, else when it does."""
        self.system = system
        set_quietly(self.dof, system.n)
        self.r.setMaximum(system.n)
        self._dirty = True
        if self.isVisible():
            self.redraw()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._dirty:
            self.redraw()

    def load_trial(self, index: int) -> None:
        t = TRIALS[index]
        system = t.system()
        self.system_requested.emit(system)
        if (self.system.n != system.n or not np.allclose(self.system.masses, system.masses)
                or not np.allclose(self.system.stiffness, system.stiffness)):
            self.set_system(system)  # nobody passed it back: show it here alone
        for box, value in ((self.shape, SHAPES.index(t.shape)), (self.eps, t.eps), (self.iterations, t.iterations),
                           (self.basis, BASES.index(t.basis)), (self.r, t.shapes), (self.holzer_hz, t.holzer_hz)):
            set_quietly(box, value)
        self.trials.show_note(t.note)
        self.tabs.setCurrentIndex(t.tab)
        self.redraw()

    def _on_holzer_line(self) -> None:
        f = float(self.holzer_trial.value())
        if abs(f - self.holzer_hz.value()) > 1e-9 and f > 0:
            self.holzer_hz.setValue(f)

    # ------------------------------------------------------------- drawing
    def apply_theme(self) -> None:
        c = colors
        dot = QtCore.Qt.PenStyle.DotLine
        self.mode_item.setPen(pg.mkPen(c.strong, width=2, style=QtCore.Qt.PenStyle.DashLine))
        self.trial_item.setPen(pg.mkPen(c.mass[0], width=2))
        self.trial_item.setSymbolBrush(pg.mkBrush(c.mass[0]))
        self.trial_item.setSymbolPen(None)
        self.ritz_item.setPen(pg.mkPen(c.mass[1], width=1))
        self.ritz_item.setSymbolPen(pg.mkPen(c.mass[1], width=1.5))
        self.ritz_item.setSymbolBrush(None)
        self.exact_line.setPen(pg.mkPen(c.strong, width=1.5, style=dot))
        for item, color in zip(self.est_items, (c.mass[2], c.mass[0], c.mass[1])):
            mark(item, color, filled=True)
        for label in self.est_labels:
            label.setColor(c.strong)
        self.err_curve.setPen(pg.mkPen(c.mass[0], width=2))
        self.err_exact.setPen(pg.mkPen(c.strong, width=1.5, style=dot))
        mark(self.err_now, c.force)
        self.ritz_now.setPen(pg.mkPen(c.grey, width=1, style=dot))
        self.holzer_curve.setPen(pg.mkPen(c.mass[0], width=2))
        self.holzer_zero.setPen(pg.mkPen(c.grey, width=1))
        self.holzer_trial.setPen(pg.mkPen(c.force, width=1.5))
        self._ritz_built = -1  # rebuild the per-mode items in the new colours
        self._dirty = True
        if self.isVisible():
            self.redraw()

    def redraw(self) -> None:
        self._dirty = False
        s = self.system
        self.w, self.phi = exact(s)
        self.f = self.w / TWO_PI
        self.held = bool(np.all(self.w > 1e-9))
        self.chain_info.setText(
            "Exact: " + ", ".join(f"{f:.4g}" for f in self.f) + " Hz"
            + ("" if self.held else " (a rigid-body mode at 0 Hz: the chain isn't held to the ground)"))
        self._draw_main()
        self._draw_tab()
        self._draw_lagrange()
        settle_plots(self)

    def _trial(self) -> np.ndarray:
        s = self.system
        x = trial_shape(s, SHAPES[self.shape.currentIndex()])
        x = contaminated(s, x, self.eps.value())
        x = iterate(s, x, int(self.iterations.value())) if self.held else x
        return x / np.max(np.abs(x))

    def _draw_main(self) -> None:
        s, f = self.system, self.f
        n = s.n
        idx = np.arange(n + 1)
        phi1 = self.phi[:, 0] / self.phi[np.argmax(np.abs(self.phi[:, 0])), 0]
        x = self._trial()
        if x @ phi1 < 0:
            x = -x
        self.mode_item.setData(idx, np.concatenate([[0.0], phi1]))
        self.trial_item.setData(idx, np.concatenate([[0.0], x]))
        r = min(int(self.r.value()), n)
        wr, shapes = rayleigh_ritz(s, ritz_basis(s, BASES[self.basis.currentIndex()], r))
        y = shapes[:, 0] / shapes[np.argmax(np.abs(shapes[:, 0])), 0]
        self.ritz_item.setData(idx, np.concatenate([[0.0], y]))
        self.shape_plot.setXRange(0, n, padding=0.03)
        self.shape_plot.getAxis("bottom").setTicks([[(i, "ground" if i == 0 else f"m{i}") for i in idx]])

        f_ray = math.sqrt(max(rayleigh(s, x), 0.0)) / TWO_PI
        f_ritz = wr[0] / TWO_PI
        f_dunk = math.sqrt(dunkerley(s)) / TWO_PI if self.held else float("nan")
        self.exact_line.setValue(f[0])
        values = (f_dunk, f_ray, f_ritz)
        for row, (item, label, v) in enumerate(zip(self.est_items, self.est_labels, values)):
            if np.isfinite(v):
                item.setData([v], [row])
                label.setText(f"{v:.4g} Hz ({100 * (v / f[0] - 1):+.3g} %)" if f[0] > 0 else f"{v:.4g} Hz")
                label.setPos(v, row)
            else:
                item.setData([], [])
                label.setText("")
        finite = [v for v in (*values, f[0]) if np.isfinite(v)]
        lo, hi = min(finite), max(finite)
        pad = 0.15 * (hi - lo) + 0.02 * hi
        self.est_plot.setXRange(lo - pad, hi + pad, padding=0)
        self.est_plot.setYRange(-0.6, 2.8, padding=0)

        parts = [f"Exact f₁ = {f[0]:.4g} Hz.",
                 f"Rayleigh: {f_ray:.4g} Hz ({self._pct(f_ray)})."]
        if self.held:
            parts.append(f"Dunkerley: {f_dunk:.4g} Hz ({self._pct(f_dunk)}).")
        parts.append(f"Rayleigh–Ritz with {r} shape{'s' if r > 1 else ''}: " + ", ".join(
            f"f{j + 1} = {wr[j] / TWO_PI:.4g} ({self._pct(wr[j] / TWO_PI, j)})" for j in range(r)) + ".")
        self.readout.setText(" ".join(parts))

    def _pct(self, value: float, mode: int = 0) -> str:
        exact_f = self.f[mode]
        if exact_f <= 0:
            return "exact 0"
        pct = 100 * (value / exact_f - 1)
        return "exact" if abs(pct) < 1e-9 else f"{pct:+.3g} %"  # round-off reads as exact

    def _draw_tab(self) -> None:
        i = self.tabs.currentIndex()
        if i == TAB_ERROR:
            self._draw_error()
        elif i == TAB_RITZ:
            self._draw_ritz()
        elif i == TAB_HOLZER:
            self._draw_holzer()

    def _draw_error(self) -> None:
        s = self.system
        if s.n < 2:
            self.err_curve.setData([], [])
            self.err_now.setData([], [])
            self.err_note.setText("One mass: its only shape is exact.")
            return
        base = trial_shape(s, SHAPES[self.shape.currentIndex()])
        passes = int(self.iterations.value()) if self.held else 0
        eps = np.linspace(-EPS_SPAN, EPS_SPAN, 201)
        now = float(self.eps.value())
        eps = np.union1d(eps, [now])

        def estimate(e: float) -> float:
            x = iterate(s, contaminated(s, base, e), passes)
            return math.sqrt(max(rayleigh(s, x), 0.0)) / TWO_PI

        fs = np.array([estimate(e) for e in eps])
        self.err_curve.setData(eps, fs)
        self.err_exact.setValue(self.f[0])
        fn = estimate(now)
        self.err_now.setData([now], [fn])
        self.err_plot.setXRange(min(-EPS_SPAN, now), max(EPS_SPAN, now), padding=0.02)
        self.err_note.setText(
            f"At ε = {now:.3g}: {fn:.4g} Hz, {self._pct(fn)}. The lowest point of the curve is "
            f"{fs.min():.4g} Hz ({self._pct(float(fs.min()))}), never below the exact {self.f[0]:.4g} Hz."
            + (f" After {passes} pass{'es' if passes > 1 else ''} of iteration the curve is flatter: each pass "
               "shrinks mode 2's share." if passes else ""))

    def _draw_ritz(self) -> None:
        s = self.system
        n = s.n
        c = colors
        if self._ritz_built != n:
            for item in (*self.ritz_curves, *self.ritz_exact):
                self.ritz_plot.removeItem(item)
            self.ritz_curves, self.ritz_exact = [], []
            for j in range(n):
                color = c.mode[j % len(c.mode)]
                line = pg.InfiniteLine(angle=0, pen=pg.mkPen(color, width=1, style=QtCore.Qt.PenStyle.DotLine))
                curve = pg.PlotDataItem(pen=pg.mkPen(color, width=2), symbol="o", symbolSize=7,
                                        symbolBrush=pg.mkBrush(color), symbolPen=None)
                self.ritz_plot.addItem(line)
                self.ritz_plot.addItem(curve)
                self.ritz_exact.append(line)
                self.ritz_curves.append(curve)
            self._ritz_built = n
        table = ritz_table(s, BASES[self.basis.currentIndex()])
        rs = np.arange(1, n + 1)
        for j in range(n):
            ok = np.isfinite(table[:, j])
            self.ritz_curves[j].setData(rs[ok], table[ok, j])
            self.ritz_exact[j].setValue(self.f[j])
        r = min(int(self.r.value()), n)
        self.ritz_now.setValue(r)
        top = float(np.nanmax(table))
        self.ritz_plot.setXRange(0.7, n + 0.3, padding=0)
        self.ritz_plot.setYRange(0.0, top * 1.08, padding=0)
        self.ritz_plot.getAxis("bottom").setTicks([[(int(v), str(int(v))) for v in rs]])
        row = table[r - 1, :r]
        self.ritz_note.setText(
            f"With r = {r}: " + ", ".join(f"f{j + 1} = {row[j]:.4g} Hz ({self._pct(row[j], j)})" for j in range(r))
            + ". Each estimate is above its mode and comes down as shapes are added.")

    def _draw_holzer(self) -> None:
        s = self.system
        if self.tabs.currentIndex() != TAB_HOLZER:
            return
        top = 1.2 * float(self.f[-1]) if self.f[-1] > 0 else 1.0
        fs = np.linspace(0.0, top, 1500)
        k1 = s.stiffness[0] or 1.0
        res = holzer_residual(s, TWO_PI * fs) / k1
        self.holzer_curve.setData(fs, np.sign(res) * np.log10(1 + np.abs(res)))
        roots = holzer_roots(s, top)
        for line in self.holzer_roots:
            self.holzer_plot.removeItem(line)
        pen = pg.mkPen(colors.strong, width=1, style=QtCore.Qt.PenStyle.DotLine)
        self.holzer_roots = [pg.InfiniteLine(angle=90, pos=r, pen=pen) for r in roots]
        for line in self.holzer_roots:
            self.holzer_plot.addItem(line)
        fh = float(self.holzer_hz.value())
        self.holzer_trial.blockSignals(True)
        self.holzer_trial.setValue(fh)
        self.holzer_trial.blockSignals(False)
        self.holzer_plot.setXRange(0.0, max(top, fh * 1.05), padding=0)
        rows = holzer(s, TWO_PI * fh)
        body = "".join(
            f"<tr><td>m{i + 1}</td><td align='right'>{r.x:.4g}</td><td align='right'>{r.inertia:.4g}</td>"
            f"<td align='right'>{r.spring:.4g}</td></tr>" for i, r in enumerate(rows))
        self.holzer_table.setText(
            f"<p>Holzer's table at {fh:.4g} Hz (x₁ = 1; the ground spring pulls {s.stiffness[0]:.4g} N):</p>"
            "<table cellspacing='0' cellpadding='3'><tr><th>mass</th><th>x</th><th>m ω² x (N)</th>"
            f"<th>passed on (N)</th></tr>{body}</table>"
            f"<p>Left at the free end: {rows[-1].spring:.4g} N. Its sign changes at "
            + ", ".join(f"{r:.4g}" for r in roots) + " Hz.</p>")

    def _draw_lagrange(self) -> None:
        s = self.system
        terms_T = " + ".join(f"{m:.3g} ẋ{i + 1}²" for i, m in enumerate(s.masses))
        terms_V = " + ".join(
            f"{k:.3g} ({'x1' if i == 0 else f'x{i + 1} − x{i}'})²" for i, k in enumerate(s.stiffness))
        M, _, K = s.matrices()
        self.lagrange.setHtml(
            "<p>For the chain on the Simulation page:</p>"
            f"<p>T = ½ ({terms_T})</p><p>V = ½ ({terms_V})</p>"
            "<p>M<sub>ij</sub> = ∂²T/∂ẋ<sub>i</sub>∂ẋ<sub>j</sub> (kg):</p>" + matrix_html(M)
            + "<p>K<sub>ij</sub> = ∂²V/∂x<sub>i</sub>∂x<sub>j</sub> (N/m):</p>" + matrix_html(K)
            + "<p>Each spring adds k to the two diagonal entries of the masses it joins and −k to the two entries "
              "between them: the assembly the Simulation page's chain uses. Lagrange's equations then give "
              "Mẍ + Kx = f.</p>"
            + ("<p>Dunkerley's terms, each mass alone on the massless chain: "
               + ", ".join(f"m{i + 1} {math.sqrt(v) / TWO_PI:.3g} Hz" for i, v in enumerate(dunkerley_terms(s)))
               + ".</p>" if self.held else ""))
