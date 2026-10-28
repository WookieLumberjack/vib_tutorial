"""Variability & localization page: the chain with scattered properties, and a mistuned ring of blades.

Left: the scatter on the Simulation page's chain (each property's coefficient of variation, how many chains), and
the ring (blades, coupling, mistuning, the engine order, the mode). Centre: the chain's receptance at its last mass,
nominal and the band the scattered chains span, and each mode's spread by Monte Carlo and to first order. Right:
the frequencies' histograms, the ring's modes tuned and mistuned, its forced response and the amplification against
mistuning, and the Theory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.model import ChainSystem
from ..core.random_vibration import linear
from ..core.variability import (
    Ring,
    Scatter,
    amplification,
    amplification_curve,
    forced,
    monte_carlo,
    natural_hz,
    participation,
    ring_modes,
    sensitivity_sigma,
    sweep_grid,
    tuned_band,
    whitehead,
)
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

TAB_HISTOGRAM, TAB_MODES, TAB_FORCED = range(3)
LEVELS = np.array([0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1])
SEEDS = 12  # mistuning patterns per level


@dataclass(frozen=True)
class VariabilityTrial:
    name: str
    note: str
    mass: float = 0.0  # CoV, %
    stiffness: float = 5.0
    damping: float = 0.0
    blades: int = 12
    coupling: float = 5.0  # k_c / k, %
    mistuning: float = 0.0  # %
    engine_order: int = 3
    mode: int = 1
    tab: int = TAB_HISTOGRAM


TRIALS = (
    VariabilityTrial(
        "5 % scatter in the springs",
        "Two hundred copies of the uniform chain, each spring scattered by 5 % (lognormal). Each frequency "
        "scatters by about 1.4 % (mode 1: σ = 0.015 Hz on 1.105 Hz), and the first-order estimate from the "
        "modes' strain energies, σ<sub>f</sub>/f = σ<sub>λ</sub>/2λ, gives 1.44 % for every mode. The receptance's band is narrow "
        "between the peaks and wide at them: a sharp peak moves sideways, so near it the response can be several "
        "times the nominal one."),
    VariabilityTrial(
        "Masses scatter too",
        "5 % on the masses as well as the springs: the frequencies scatter by about 2 % (mode 1: σ = 0.023 Hz). "
        "Independent sources add in quadrature: √2 times the springs' alone.",
        mass=5.0),
    VariabilityTrial(
        "Damping scatter only",
        "30 % scatter on the dampers and none on the rest: the frequencies don't move at all, but the peaks "
        "do, by about the damping's scatter. The band only opens at the resonances.",
        stiffness=0.0, damping=30.0),
    VariabilityTrial(
        "A tuned bladed disc",
        "Twelve identical blades, each on a spring to the disc, joined by springs of 5 % of theirs round the "
        "ring (Modes tab). Its modes are waves round the ring: 0 to 6 nodal diameters, all but the first and "
        "last in pairs, in a band from 15.9 to 17.4 Hz. Every mode moves 6 to 12 blades.",
        tab=TAB_MODES),
    VariabilityTrial(
        "Weak coupling: 1 % mistuning localizes",
        "The coupling cut to 0.5 % and each blade's stiffness scattered by 1 %: the scatter is now twice the "
        "coupling, the waves can't get past the blades that differ, and the lowest mode moves 1.3 blades: "
        "it is one blade's. A mode localized on a few blades puts all the energy there.",
        coupling=0.5, mistuning=1.0, tab=TAB_MODES),
    VariabilityTrial(
        "Strong coupling resists",
        "The same 1 % mistuning with a 20 % coupling: every mode still spreads over 6 to 12 blades. What "
        "matters is the mistuning over the coupling, not the mistuning alone.",
        coupling=20.0, mistuning=1.0, tab=TAB_MODES),
    VariabilityTrial(
        "Forced response: one blade takes more",
        "Engine order 3 on the ring with a 2 % coupling (Forced tab). Tuned, every blade answers alike. With "
        "this 1 % pattern the worst blade peaks only 1.06 times the tuned one, but over 12 patterns the worst "
        "reaches 1.34 at 1 % and 1.42 at 2 %, where the mistuning matches the coupling: still below "
        "Whitehead's bound (1 + √12)/2 = 2.23. More mistuning helps again (1.28 at 10 %): the blades "
        "decouple.",
        coupling=2.0, mistuning=1.0, tab=TAB_FORCED),
)

TIPS = {
    "dof": "<p>The chain's number of masses: the Simulation page's chain is the nominal structure.</p>",
    "cov": "<p>The coefficient of variation (σ over the mean) of each mass, spring or damper, drawn "
           "independently and lognormal.</p>",
    "count": "<p>How many scattered chains to draw.</p>",
    "blades": "<p>The number of blades round the disc.</p>",
    "coupling": "<p>The springs between neighbouring blades, as a share of each blade's own.</p>",
    "mistuning": "<p>The scatter (σ) of the blades' stiffnesses; the pattern is seeded, its mean removed.</p>",
    "eo": "<p>The engine order E: the force on blade j has phase 2πEj/N, a wave E times round the disc.</p>",
    "mode": "<p>Which of the ring's modes to draw, from the lowest.</p>",
}
FRF_TIP = ("<p>The receptance at the last mass from a force there: nominal (bold), the 5–95 % band of the "
           "scattered chains (shaded), and a few of them.</p>")
HIST_TIP = "<p>Each mode's frequency deviation in the scattered chains, with the first-order normal curve.</p>"
MODES_TIP = ("<p>Top: the chosen mode's blade amplitudes, tuned (outline) and mistuned (filled). Bottom: how many "
             "blades each mode moves (participation number).</p>")
FORCED_TIP = ("<p>Top: each blade's amplitude under the engine-order force, mistuned (thin) and tuned (dashed). "
              "Bottom: the worst blade's peak over the tuned one against the mistuning, mean and worst of the "
              "patterns, and Whitehead's bound.</p>")

NOTES_HTML = """
<!--BEFORE-->
<h3><a name="scatter"></a>No two structures are alike</h3>
<p>Real masses, stiffnesses and joints scatter: from part to part, assembly to assembly, day to day. A model
with nominal values predicts the average structure at best. <b>Monte Carlo</b> treats the properties as random:
draw many structures, analyse each, and look at the spread of what matters. The page scatters every mass, spring
and damper independently (lognormal, so they stay positive) and draws each copy's receptance and frequencies.</p>
<p>For small scatter the spread follows from the sensitivities (the Model updating page's): with φ
mass-normalized, a spring's change moves λ = ω² by its strain energy share, (φ<sub>i</sub> − φ<sub>i−1</sub>)² δk,
and a mass's by −λφ<sub>i</sub>² δm, so to first order</p>
<p align="center">σ<sub>λ</sub>² = Σ ((φ<sub>i</sub> − φ<sub>i−1</sub>)² k<sub>i</sub> c<sub>k</sub>)² +
Σ (λ φ<sub>i</sub>² m<sub>i</sub> c<sub>m</sub>)², &nbsp; σ<sub>f</sub>/f = σ<sub>λ</sub>/2λ.</p>
<p>Frequencies scatter by less than the properties (each mode averages over many of them); peaks scatter
sideways, so near a resonance the response's spread is huge. Damping's scatter moves only the peaks' heights.
Design to the band, not the nominal curve.</p>

<h3><a name="localization"></a>Mode localization</h3>
<p>A structure made of identical, weakly coupled parts (a bladed disc, a truss of identical bays, a row of
identical rooms) has modes that are waves through all the parts, in a narrow band of frequencies. Make the parts
slightly different (<b>mistuning</b>) and, when the differences are comparable to the coupling, the waves can no
longer pass: each mode collapses onto one or a few parts (Anderson localization, first met in disordered
crystals). The measure is the mistuning over the coupling, so a structure that is <i>more</i> precisely made but
more weakly coupled can be worse.</p>
<p>The <b>participation number</b> (Σφ²)²/Σφ⁴ counts the parts a mode moves: two-thirds of N for a standing wave,
N for the uniform one, 1 for a mode on one part.</p>

<h3><a name="mistuning"></a>Forced response of a mistuned disc</h3>
<p>A turbine's blades are driven by the flow past the vanes ahead of them: the same force on every blade, its
phase stepping 2πE/N from blade to blade (engine order E). A tuned disc answers with one travelling wave, every
blade alike. Mistuned, the response spreads over several modes and concentrates on some blades: the worst blade's
peak rises above the tuned one, which matters because fatigue is set by the worst blade. The amplification peaks
when the mistuning is comparable to the coupling, and Whitehead's bound (1 + √N)/2 caps it for light damping.
Intentional mistuning (alternating blades) is one of the cures.</p>

<h3><a name="try-it"></a>Things to try</h3>
<p>Each of these sets up the scatter or the ring (Things to try, top left); the first three load the uniform
chain into the Simulation page.</p>
<!--TRIALS-->
<!--TAKEAWAYS-->
"""
NOTES_HTML = NOTES_HTML.replace("<!--BEFORE-->", before(
    ("updating_page", "sensitivities", "Updating by sensitivities"),
    ("test_page", "measured", "A measured FRF"),
)).replace("<!--TRIALS-->", trials_html(TRIALS)).replace("<!--TAKEAWAYS-->", takeaways(
    "Scattered properties scatter the frequencies by less, and the response near a peak by much more.",
    "First-order sensitivities predict the frequencies' spread from the strain and kinetic energy shares.",
    "Weakly coupled, nearly identical parts localize their modes when the mistuning matches the coupling.",
    "A mistuned disc's worst blade can answer up to (1 + √N)/2 times the tuned one.",
))


class VariabilityPage(QtWidgets.QWidget):
    """Controls on the left, the chain's band in the centre, histograms and the ring on the right."""

    dof_requested = QtCore.Signal(int)
    system_requested = QtCore.Signal(object)
    edit_parameters = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.system = ChainSystem.uniform(4)
        self._dirty = True

        self.trials = TrialBox([t.name for t in TRIALS],
                               "<p>Each sets up the scatter or the ring to show one idea from the Theory tab.</p>")
        self.trials.picked.connect(self.load_trial)

        ch = Form("Chain")
        self.dof = QtWidgets.QSpinBox()
        self.dof.setRange(1, MAX_DOF)
        self.dof.valueChanged.connect(self.dof_requested)
        ch.add("Number of masses:", self.dof, TIPS["dof"])
        link = QtWidgets.QLabel("Its nominal masses, springs and dampers are edited on the <a href='#sim'>"
                                "Simulation page</a>.")
        link.setWordWrap(True)
        mute(link)
        link.linkActivated.connect(lambda _: self.edit_parameters.emit())
        ch.row(link)

        sc = Form("Scatter (CoV)")
        self.cov_m = sc.number("Masses:", 0.0, 30.0, 0.0, 1, " %", TIPS["cov"])
        self.cov_k = sc.number("Springs:", 0.0, 30.0, 5.0, 1, " %", TIPS["cov"])
        self.cov_c = sc.number("Dampers:", 0.0, 100.0, 0.0, 1, " %", TIPS["cov"])
        self.count = sc.number("Chains drawn:", 20, 2000, 200, 0, "", TIPS["count"])

        ring = Form("Ring of blades")
        self.blades = ring.number("Blades:", 2, 48, 12, 0, "", TIPS["blades"])
        self.coupling = ring.number("Coupling:", 0.05, 100.0, 5.0, 2, " %", TIPS["coupling"])
        self.mistuning = ring.number("Mistuning:", 0.0, 20.0, 0.0, 2, " %", TIPS["mistuning"])
        self.eo = ring.number("Engine order:", 0, 24, 3, 0, "", TIPS["eo"])
        self.mode = ring.number("Mode:", 1, 48, 1, 0, "", TIPS["mode"])

        connect_all((self.cov_m, self.cov_k, self.cov_c, self.count), self._on_scatter)
        connect_all((self.blades, self.coupling, self.mistuning, self.eo, self.mode), self._on_ring)

        # --- centre
        self.frf_plot = plot("|Receptance|", "m/N", "Frequency", "Hz", log_x=True, log_y=True, height=200)
        self.frf_plot.setToolTip(FRF_TIP)
        add_legend(self.frf_plot.getPlotItem(), offset=(-10, 10))
        self.band_low, self.band_high = pg.PlotDataItem(), pg.PlotDataItem()
        self.band_fill = pg.FillBetweenItem(self.band_low, self.band_high)
        self.sample_items = [pg.PlotDataItem() for _ in range(5)]
        self.nominal_item = pg.PlotDataItem(name="nominal")
        for item in (self.band_fill, self.band_low, self.band_high, *self.sample_items, self.nominal_item):
            self.frf_plot.addItem(item)
        self.readout = readout_label()

        # --- right: histograms
        self.hist_plot = plot("Chains (density)", None, "Frequency deviation", "%", height=200)
        self.hist_plot.setToolTip(HIST_TIP)
        add_legend(self.hist_plot.getPlotItem(), offset=(-10, 10))
        self.hist_items: list[pg.PlotDataItem] = []
        self.hist_note = note_label()

        # --- right: ring modes
        self.shape_plot = plot("Blade amplitude", None, "Blade", None, height=150)
        self.shape_tuned = pg.BarGraphItem(x=[], height=[], width=0.8)
        self.shape_mis = pg.BarGraphItem(x=[], height=[], width=0.5)
        self.shape_plot.addItem(self.shape_tuned)
        self.shape_plot.addItem(self.shape_mis)
        self.pn_plot = plot("Blades moving", None, "Mode frequency", "Hz", height=150)
        add_legend(self.pn_plot.getPlotItem(), offset=(-10, 10))
        self.pn_tuned = marker(9, name="tuned")
        self.pn_mis = marker(9, "s", name="mistuned")
        self.pn_now = marker(15)
        for item in (self.pn_tuned, self.pn_mis, self.pn_now):
            self.pn_plot.addItem(item)
        for p in (self.shape_plot, self.pn_plot):
            p.setToolTip(MODES_TIP)
        self.modes_note = note_label()

        # --- right: forced ring
        self.sweep_plot = plot("Blade amplitude", "m/N", "Frequency", "Hz", height=150)
        self.sweep_items: list[pg.PlotDataItem] = []
        self.sweep_tuned = pg.PlotDataItem()
        self.sweep_plot.addItem(self.sweep_tuned)
        self.amp_plot = plot("Worst blade / tuned", None, "Mistuning", "%", log_x=True, height=150)
        add_legend(self.amp_plot.getPlotItem(), offset=(10, 10))
        self.amp_mean = pg.PlotDataItem(name="mean", symbol="o", symbolSize=5)
        self.amp_top = pg.PlotDataItem(name="worst pattern", symbol="t", symbolSize=6)
        self.amp_bound = pg.InfiniteLine(angle=0)
        self.amp_now = pg.InfiniteLine(angle=90)
        for item in (self.amp_bound, self.amp_now, self.amp_mean, self.amp_top):
            self.amp_plot.addItem(item)
        for p in (self.sweep_plot, self.amp_plot):
            p.setToolTip(FORCED_TIP)
        self.forced_note = note_label()
        self._curve_key = None

        self.notes = theory(NOTES_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(column(self.hist_plot, self.hist_note, stretch=False), "Frequencies")
        self.tabs.addTab(column(self.shape_plot, self.pn_plot, self.modes_note, stretch=False), "Ring modes")
        self.tabs.addTab(column(self.sweep_plot, self.amp_plot, self.forced_note, stretch=False), "Forced ring")
        self.tabs.addTab(self.notes, "Theory")
        self.tabs.currentChanged.connect(lambda _: self._draw_tab())

        three_panes(self, column(self.trials, ch, sc, ring), stack((self.frf_plot, 1), (self.readout, 0)),
                    self.tabs)
        self.apply_theme()
        self.set_system(self.system)

    # ------------------------------------------------------------- inputs
    def set_system(self, system: ChainSystem, result=None) -> None:
        self.system = system
        set_quietly(self.dof, system.n)
        self._dirty = True
        if self.isVisible():
            self.redraw()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._dirty:
            self.redraw()

    def load_trial(self, index: int) -> None:
        t = TRIALS[index]
        if index < 3:
            system = ChainSystem.uniform(4)
            self.system_requested.emit(system)
            s = self.system
            if (s.n != 4 or not np.allclose(s.masses, 1.0) or not np.allclose(s.stiffness, 400.0)
                    or not np.allclose(s.damping, 2.0)):
                self.set_system(system)
        for box, value in ((self.cov_m, t.mass), (self.cov_k, t.stiffness), (self.cov_c, t.damping),
                           (self.count, 200), (self.blades, t.blades), (self.coupling, t.coupling),
                           (self.mistuning, t.mistuning), (self.eo, t.engine_order), (self.mode, t.mode)):
            set_quietly(box, value)
        self.trials.show_note(t.note)
        self.tabs.setCurrentIndex(t.tab)
        self.redraw()

    @property
    def scatter(self) -> Scatter:
        return Scatter(self.cov_m.value() / 100, self.cov_k.value() / 100, self.cov_c.value() / 100)

    @property
    def ring(self) -> Ring:
        return Ring(int(self.blades.value()), coupling=self.coupling.value() / 100,
                    mistuning=self.mistuning.value() / 100)

    def _on_scatter(self) -> None:
        self._draw_chain()
        if self.tabs.currentIndex() == TAB_HISTOGRAM:
            self._draw_histogram()
        settle_plots(self)

    def _on_ring(self) -> None:
        self.mode.setMaximum(int(self.blades.value()))
        self.eo.setMaximum(int(self.blades.value()) // 2)
        self._draw_tab()
        settle_plots(self)

    # ------------------------------------------------------------- drawing
    def apply_theme(self) -> None:
        c = colors
        self.nominal_item.setPen(pg.mkPen(c.strong, width=2.5))
        for item in (self.band_low, self.band_high):
            item.setPen(pg.mkPen(c.mass[0], width=1))
        fill = pg.mkColor(c.mass[0])
        fill.setAlpha(60)
        self.band_fill.setBrush(pg.mkBrush(fill))
        for item in self.sample_items:
            item.setPen(pg.mkPen(c.grey, width=1))
        self.shape_tuned.setOpts(pen=pg.mkPen(c.grey, width=1.5), brush=pg.mkBrush(0, 0, 0, 0))
        self.shape_mis.setOpts(brush=pg.mkBrush(c.mass[0]), pen=None)
        mark(self.pn_tuned, c.grey)
        mark(self.pn_mis, c.mass[0], filled=True)
        mark(self.pn_now, c.force)
        self.sweep_tuned.setPen(pg.mkPen(c.strong, width=2.5, style=QtCore.Qt.PenStyle.DashLine))
        for item, color in ((self.amp_mean, c.mass[0]), (self.amp_top, c.force)):
            item.setPen(pg.mkPen(color, width=2))
            item.setSymbolBrush(pg.mkBrush(color))
            item.setSymbolPen(None)
        self.amp_bound.setPen(pg.mkPen(c.grey, width=1.5, style=QtCore.Qt.PenStyle.DotLine))
        self.amp_now.setPen(pg.mkPen(c.fit, width=1, style=QtCore.Qt.PenStyle.DotLine))
        self._dirty = True
        if self.isVisible():
            self.redraw()

    def redraw(self) -> None:
        self._dirty = False
        self._draw_chain()
        self._draw_tab()
        settle_plots(self)

    def _draw_chain(self) -> None:
        s = self.system
        n = s.n
        f0 = natural_hz(linear(s))[0]
        fpos = f0[f0 > 1e-6]
        lo = 0.3 * float(fpos[0]) if fpos.size else 0.1
        hi = 1.8 * float(f0[-1]) if f0[-1] > 0 else 10.0
        f = np.geomspace(lo, hi, 900)
        self.mc = mc = monte_carlo(s, self.scatter, int(self.count.value()), f, n - 1, n - 1)
        low, high = mc.band()
        self.band_low.setData(f, low)
        self.band_high.setData(f, high)
        for i, item in enumerate(self.sample_items):
            item.setData(f, mc.frf[i] if i < mc.frf.shape[0] else np.full(f.size, np.nan))
        self.nominal_item.setData(f, mc.nominal_frf)
        self.frf_plot.setXRange(np.log10(lo), np.log10(hi), padding=0)
        sigma = sensitivity_sigma(s, self.scatter)
        rows = "".join(
            f"<tr><td>{r + 1}</td><td align='right'>{mc.nominal_freqs[r]:.4g}</td>"
            f"<td align='right'>{mc.freqs[:, r].mean():.4g}</td><td align='right'>{mc.freqs[:, r].std():.3g}</td>"
            f"<td align='right'>{sigma[r]:.3g}</td>"
            f"<td align='right'>{100 * mc.freqs[:, r].std() / mc.nominal_freqs[r]:.3g} %</td></tr>"
            for r in range(n))
        peak = int(np.argmax(mc.nominal_frf))
        self.readout.setText(
            "<table cellspacing='5'><tr><th>mode</th><th>nominal (Hz)</th><th>mean</th><th>σ (Monte Carlo)</th>"
            f"<th>σ (first order)</th><th>σ/f</th></tr>{rows}</table>"
            f"At the nominal peak ({f[peak]:.3g} Hz) the band spans {low[peak] / mc.nominal_frf[peak]:.2g} to "
            f"{high[peak] / mc.nominal_frf[peak]:.2g} times the nominal receptance; {mc.freqs.shape[0]} chains.")

    def _draw_tab(self) -> None:
        i = self.tabs.currentIndex()
        if i == TAB_HISTOGRAM:
            self._draw_histogram()
        elif i == TAB_MODES:
            self._draw_modes()
        elif i == TAB_FORCED:
            self._draw_forced()

    def _draw_histogram(self) -> None:
        mc = self.mc
        for item in self.hist_items:
            self.hist_plot.removeItem(item)
        self.hist_items = []
        legend = self.hist_plot.getPlotItem().legend
        legend.clear()
        dev = 100 * (mc.freqs / mc.nominal_freqs - 1)
        sigma = 100 * sensitivity_sigma(self.system, self.scatter) / mc.nominal_freqs
        span = max(float(np.max(np.abs(dev))), 1e-3) * 1.1
        edges = np.linspace(-span, span, 41)
        x = np.linspace(-span, span, 300)
        for r in range(min(mc.freqs.shape[1], 4)):
            color = colors.mode[r % len(colors.mode)]
            hist, _ = np.histogram(dev[:, r], edges, density=True)
            item = pg.PlotDataItem(edges, hist, stepMode="center", pen=pg.mkPen(color, width=1.5))
            self.hist_plot.addItem(item)
            legend.addItem(item, f"mode {r + 1}")
            self.hist_items.append(item)
            if sigma[r] > 0:
                pdf = np.exp(-0.5 * (x / sigma[r]) ** 2) / (sigma[r] * np.sqrt(2 * np.pi))
                curve = pg.PlotDataItem(x, pdf, pen=pg.mkPen(color, width=1.5, style=QtCore.Qt.PenStyle.DashLine))
                self.hist_plot.addItem(curve)
                self.hist_items.append(curve)
        self.hist_plot.setXRange(-span, span, padding=0)
        self.hist_plot.enableAutoRange(axis="y")
        self.hist_note.setText("Solid: the scattered chains' frequency deviations from nominal; dashed: the "
                               "first-order normal curve. A mode with σ = 0 doesn't move.")

    def _draw_modes(self) -> None:
        ring = self.ring
        n = ring.blades
        mode = min(int(self.mode.value()), n) - 1
        ft, pt = ring_modes(ring, np.zeros(n))
        fm, pm = ring_modes(ring)
        x = np.arange(1, n + 1)
        shape_t = pt[:, mode] / np.max(np.abs(pt[:, mode]))
        shape_m = pm[:, mode] / np.max(np.abs(pm[:, mode]))
        self.shape_tuned.setOpts(x=x, height=shape_t)
        self.shape_mis.setOpts(x=x, height=shape_m)
        self.shape_plot.setXRange(0.4, n + 0.6, padding=0)
        self.shape_plot.setYRange(-1.1, 1.1, padding=0)
        nt, nm = participation(pt), participation(pm)
        self.pn_tuned.setData(ft, nt)
        self.pn_mis.setData(fm, nm)
        self.pn_now.setData([fm[mode]], [nm[mode]])
        self.pn_plot.setYRange(0, n + 0.5, padding=0)
        lo, hi = tuned_band(ring)
        self.modes_note.setText(
            f"Mode {mode + 1}: tuned {ft[mode]:.4g} Hz, moving {nt[mode]:.2g} blades; mistuned {fm[mode]:.4g} Hz, "
            f"moving {nm[mode]:.2g}. The tuned band runs {lo:.4g} to {hi:.4g} Hz; mistuning "
            f"{100 * ring.mistuning:.3g} % against a coupling of {100 * ring.coupling:.3g} % (ratio "
            f"{ring.mistuning / ring.coupling:.2g}).")

    def _draw_forced(self) -> None:
        ring = self.ring
        n = ring.blades
        E = int(self.eo.value())
        f = sweep_grid(ring, 900)
        x = forced(ring, E, f)
        tuned = forced(ring, E, f, np.zeros(n))
        for item in self.sweep_items:
            self.sweep_plot.removeItem(item)
        self.sweep_items = []
        pen = pg.mkPen(colors.mass[0], width=1)
        for j in range(n):
            item = pg.PlotDataItem(f, x[:, j], pen=pen)
            self.sweep_plot.addItem(item)
            self.sweep_items.append(item)
        self.sweep_tuned.setData(f, tuned.max(axis=1))
        self.sweep_plot.setXRange(float(f[0]), float(f[-1]), padding=0)
        key = (n, ring.coupling, E)
        if key != self._curve_key:
            self._curve = amplification_curve(ring, E, LEVELS[1:], SEEDS)
            self._curve_key = key
        mean, top = self._curve
        levels = 100 * LEVELS[1:]
        self.amp_mean.setData(levels, mean)
        self.amp_top.setData(levels, top)
        bound = whitehead(n)
        self.amp_bound.setPos(bound)
        self.amp_now.setPos(np.log10(max(100 * ring.mistuning, levels[0])))
        self.amp_plot.setYRange(0.9, bound + 0.15, padding=0)
        a = amplification(ring, E)
        self.forced_note.setText(
            f"This pattern: the worst blade peaks at {a:.3g} times the tuned ring's. Over {SEEDS} patterns per "
            f"level the worst reaches {float(top.max()):.3g} (at {levels[int(np.argmax(top))]:.3g} %); Whitehead's "
            f"bound for {n} blades is {bound:.3g} (dotted).")
