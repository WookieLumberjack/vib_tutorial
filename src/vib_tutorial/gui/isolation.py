"""Vibration isolation page: a machine with a rotating unbalance on mounts, and the force that reaches the floor.

Left: the machine, the mounts (by their static deflection and damping, of three kinds), an inertia block,
and a design target. Centre: the transmissibility against frequency (with a family of damping ratios
behind it) and the force on the floor against the speed. Right: a run up to speed through the mounts'
resonance, the design chart (static deflection against speed for each isolation), and the Theory. The page
has its own model (core/isolation.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.isolation import G, Isolator, Mount, deflection_for, start_up
from ..core.sdof import TWO_PI
from . import settle_plots
from .layouts import ColumnScroll
from .lesson import (
    Form,
    TrialBox,
    db,
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

MOUNTS = list(Mount)
FAMILY = (0.0, 0.1, 0.3, 0.7)  # the damping ratios drawn behind the isolator's own curve
RATIOS = np.geomspace(0.1, 20.0, 1500)
CHART_SPEEDS = np.geomspace(5.0, 200.0, 200)  # Hz on the design chart
CHART_LEVELS = (0.5, 0.8, 0.9, 0.95, 0.99)
TAB_START, TAB_CHART = range(2)


@dataclass(frozen=True)
class IsolationTrial:
    name: str
    note: str
    deflection_mm: float = 5.0
    zeta: float = 0.05
    mount: Mount = Mount.VISCOUS
    stiffness_ratio: float = 3.0
    stage_mass: float = 0.5
    block: float = 0.0
    rpm: float = 1500.0
    ramp: float = 2.0
    tab: int = TAB_START


TRIALS = (
    IsolationTrial(
        "Soft mounts: isolation",
        "A 100 kg machine with a 0.01 kg·m unbalance at 1500 rpm (25 Hz) on mounts that sag 5 mm: f<sub>n</sub> "
        "= 15.76/√5 = 7.05 Hz, so the speed is 3.5 f<sub>n</sub>. The unbalance pulls with 247 N; only 23 N, 9 %, "
        "reaches the floor (−21 dB): 91 % isolation."),
    IsolationTrial(
        "Stiff mounts: amplification",
        "The same machine on mounts that sag only 0.3 mm: f<sub>n</sub> = 28.8 Hz, just above the running "
        "speed. The mounts now amplify: 3.9 times the unbalance's force reaches the floor. Below √2 f<sub>n</sub> "
        "a mount can only make things worse.",
        deflection_mm=0.3),
    IsolationTrial(
        "Damping costs isolation",
        "Back on the soft mounts, with ζ = 0.3 instead of 0.05. Above √2 f<sub>n</sub> the damper passes force "
        "to the floor in proportion to the speed: 20 % now reaches it instead of 9 %. Damping only helps at "
        "resonance, which the machine passes through at start-up.",
        zeta=0.3),
    IsolationTrial(
        "Running up through resonance",
        "A 2 s run up to 1500 rpm on the soft, lightly damped mounts. As the speed passes f<sub>n</sub> = 7 Hz "
        "(420 rpm) the machine and the floor force swell, then settle to the isolated running state. The "
        "unbalance's force grows as the speed squared, so the resonance is passed at a twelfth of the running "
        "force, which is what keeps it tolerable.",
        tab=TAB_START),
    IsolationTrial(
        "An elastically supported damper",
        "A relaxation mount at 3000 rpm: the damper sits on a spring 3 times as stiff as the mount's, with "
        "ζ = 0.3. At resonance it passes a little more than the plain mount of the same ζ (2.1 times against "
        "2.0), but high above it the spring in series lets the damper float, and the transmissibility falls as "
        "1/r² again, not 1/r: 7.0 % reaches the floor against the plain mount's 8.8 % (at 6000 rpm, 1.9 % "
        "against 4.3 %).",
        zeta=0.3, mount=Mount.RELAXATION, rpm=3000.0),
    IsolationTrial(
        "Two stages",
        "The machine on mounts on a 50 kg intermediate mass, itself on mounts sagging 5 mm under both. Two resonances, but "
        "above them the transmissibility falls as 1/r⁴ (80 dB a decade) instead of 1/r²: 3.8 % reaches the "
        "floor here (one stage: 9 %), and at 3000 rpm the two stages pass 0.2 % against one stage's 2.5 %.",
        mount=Mount.TWO_STAGE),
    IsolationTrial(
        "An inertia block",
        "The machine bolted to a 300 kg block, the mounts stiffened to sag the same 5 mm. The transmissibility "
        "doesn't change (f<sub>n</sub> depends only on δ), nor does the force on the floor, but the machine "
        "moves a quarter as much (±27 µm instead of ±109 µm): a block steadies the machine, it doesn't "
        "isolate better.",
        block=300.0),
)

TIPS = {
    "mass": "<p>The machine's mass M.</p>",
    "rpm": "<p>Its running speed.</p>",
    "unbalance": "<p>Its unbalance m<sub>e</sub>e: an off-centre mass times its radius. Its force is "
                 "m<sub>e</sub>eΩ².</p>",
    "deflection": "<p>How far the mounts sag under the machine's weight. Since k = Mg/δ, the natural frequency "
                  "is √(g/δ)/2π = 15.76/√δ<sub>mm</sub> Hz, whatever the mass: the softer the mount, the lower.</p>",
    "zeta": "<p>The mounts' damping ratio.</p>",
    "mount": "<p><b>Spring + damper:</b> side by side. <b>Relaxation:</b> the damper in series with a spring N "
             "times the mount's, so it can't stiffen the mount at high frequency. <b>Two stages:</b> the machine on "
             "mounts on an intermediate mass, itself on mounts.</p>",
    "ratio": "<p>The relaxation damper's spring, over the mount's (N).</p>",
    "stage": "<p>The intermediate mass, over the machine's.</p>",
    "block": "<p>An inertia block bolted to the machine. The mounts are stiffened to keep their sag.</p>",
    "target": "<p>The isolation wanted at the running speed: the share of the unbalance's force kept off the "
              "floor.</p>",
    "ramp": "<p>How long the run up to speed takes.</p>",
}
TR_TIP = (
    "<p>The force on the floor over the machine's force (equally, the motion that reaches an instrument over the "
    "floor's), against the frequency over f<sub>n</sub>. Thin: plain mounts at ζ = 0, 0.1, 0.3, 0.7; bold: this "
    "isolator. It crosses 1 at √2 f<sub>n</sub>; the circle is the running speed.</p>")
FORCE_TIP = (
    "<p>The force reaching the floor against the machine's speed, against the unbalance's own force (the "
    "dashed line, what a rigid mount would pass). The circle is the running speed.</p>")
START_TIP = (
    "<p>The machine run up from rest at a steady acceleration and held at speed: its speed, its displacement and "
    "the force on the floor, stepped exactly.</p>")
CHART_TIP = (
    "<p>The static deflection undamped mounts need, against the running speed, for each isolation. The circle is "
    "this machine.</p>")

NOTES_HTML = """
<!--BEFORE-->
<h3><a name="problem"></a>The problem</h3>
<p>A machine that runs with a little unbalance shakes with the force m<sub>e</sub>eΩ². Bolted straight to the
floor, all of it goes into the building; set on mounts, some of it is taken up by the machine's own motion. The
same question arises the other way round: an instrument on a floor that shakes. Both are answered by one
number, the <b>transmissibility</b>.</p>

<h3><a name="transmissibility"></a>Transmissibility</h3>
<p>The mount (k and c) pushes on the floor with kx + cẋ. For a machine of mass M on it, under F cos Ωt,</p>
<p align="center">T = |F<sub>floor</sub>|/|F| = |k + iΩc| / |k − Ω²M + iΩc| = √(1 + (2ζr)²) / √((1 − r²)² + (2ζr)²),</p>
<p>with r = Ω/ω<sub>n</sub>. It is exactly the same as the motion that reaches a mass on a moving floor, X/X<sub>g</sub>
(the Simulation page's ground motion), so one curve serves for both. It is 1 at low speed (the mount passes the
force on), peaks at resonance, and crosses 1 again at</p>
<p align="center"><b>r = √2,</b> &nbsp; whatever the damping.</p>
<p>Above √2 f<sub>n</sub> the mount isolates: the machine's inertia takes most of the force. Far above, T ≈ 1/r²
(40 dB a decade). Below √2 f<sub>n</sub> it amplifies. So isolation needs a mount whose f<sub>n</sub> is well
<i>below</i> the running speed: a soft one. The <b>isolation</b> is 1 − T, often quoted in dB, 20 log T.</p>

<h3><a name="static-deflection"></a>Static deflection</h3>
<p>A mount's stiffness under a machine of weight Mg is set by how far it sags, δ = Mg/k, and so</p>
<p align="center">f<sub>n</sub> = (1/2π)√(g/δ) = 15.76/√δ<sub>mm</sub> Hz.</p>
<p>The mass has gone: isolation is a matter of sag. 1 mm gives 15.8 Hz, 5 mm 7 Hz, 25 mm 3.2 Hz. For 90 %
isolation (T = 0.1) undamped mounts need r² = 1 + 1/T = 11, f<sub>n</sub> = f/3.3; at 1500 rpm that is 7.5 Hz,
4.4 mm. The <i>Design chart</i> tab draws this for every speed. A very slow machine needs a very soft mount,
which is why isolating low frequencies is hard.</p>

<h3><a name="damping-trade-off"></a>The damping trade-off</h3>
<p>Damping lowers the resonant peak, T ≈ 1/2ζ there, which matters while the machine runs up and down
through it. But above √2 f<sub>n</sub> the damper passes force in proportion to speed, cΩX, and isolation gets
<i>worse</i>: at high r, T ≈ 2ζ/r instead of 1/r². Mounts are a compromise: light damping, or damping that
only works at resonance.</p>

<h3><a name="relaxation"></a>The relaxation isolator</h3>
<p>Put the damper in series with a spring N k (an <b>elastically supported damper</b>). Near resonance the
spring is stiff compared with the damper and the damper works; at high frequency the damper hardly moves, so
the mount is just k and Nk side by side and T falls as 1/r² again, not 2ζ/r. Near resonance it passes a little
more than a plain mount of the same ζ; it pays off well above it. Its dynamic
stiffness is that of the Damping models page's Zener element.</p>

<h3><a name="two-stage"></a>Two stages</h3>
<p>Mount the machine on an intermediate mass (a concrete block, a sub-frame), and mount that. There are two
resonances, but above both each stage isolates on its own, and the transmissibility falls as 1/r⁴, 80 dB a
decade. Ships' engines and sensitive instruments are mounted this way.</p>

<h3><a name="inertia-block"></a>An inertia block</h3>
<p>Bolting the machine to a heavy block and stiffening the mounts to keep the same sag leaves f<sub>n</sub>,
and so T, unchanged. What it changes is the machine's own motion, m<sub>e</sub>e/M well above resonance: a
heavier M moves less. A block steadies the machine and lowers its centre of gravity; it doesn't isolate
better.</p>

<h3><a name="start-up"></a>Starting up</h3>
<p>Running up to speed, the machine passes through the mounts' resonance. The unbalance's force there is
smaller than at speed, (f<sub>n</sub>/f)² of it, and the faster the run up the less time the resonance has to build
(the Single oscillator page's time constant): so lightly damped mounts are usually tolerable, and a slow
run up or coast down through resonance is the hazard. The <i>Start-up</i> tab steps it exactly, the
unbalance's force along the mount m<sub>e</sub>e(Ω² cos φ + Ω' sin φ) as the speed climbs.</p>

<h3><a name="try-it"></a>Things to try</h3>
<p>Each of these sets up the machine, the mounts and the plots (Things to try, top left).</p>
<!--TRIALS-->
<!--TAKEAWAYS-->
"""
NOTES_HTML = NOTES_HTML.replace("<!--BEFORE-->", before(
    ("sdof_page", "forced-response", "Forced response"),
    ("sdof_page", "unbalance", "A rotating unbalance"),
    ("damping_page", "viscoelastic", "The Zener element"),
)).replace("<!--TRIALS-->", trials_html(TRIALS)).replace("<!--TAKEAWAYS-->", takeaways(
    "Transmissibility is the same for force into the floor and motion from it; it falls below 1 only above "
    "√2 f<sub>n</sub>, whatever the damping.",
    "A mount's f<sub>n</sub> is set by its static deflection alone: 15.76/√δ<sub>mm</sub> Hz. Isolation needs a soft "
    "mount.",
    "Damping lowers the resonant peak but worsens isolation above √2 f<sub>n</sub>; a relaxation mount gets the "
    "first without most of the second.",
    "Two stages fall as 1/r⁴; an inertia block steadies the machine but doesn't change the transmissibility.",
))


class IsolationPage(QtWidgets.QWidget):
    """Controls on the left, transmissibility and floor force in the centre, start-up and design chart on the right."""

    follows_chain = False  # MainWindow: this page has its own model, not the chain

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.iso = Isolator()

        self.trials = TrialBox([t.name for t in TRIALS],
                               "<p>Each sets up the machine and its mounts to show one idea from the Theory "
                               "tab.</p>")
        self.trials.picked.connect(self.load_trial)

        machine = Form("Machine")
        self.mass = machine.number("Mass M:", 0.1, 1e6, 100.0, 1, " kg", TIPS["mass"])
        self.rpm = machine.number("Speed:", 1.0, 1e5, 1500.0, 0, " rpm", TIPS["rpm"])
        self.unbalance = machine.number("Unbalance mₑe:", 0.0, 100.0, 0.01, 5, " kg·m", TIPS["unbalance"])
        self.block = machine.number("Inertia block:", 0.0, 1e6, 0.0, 1, " kg", TIPS["block"])
        machine.changed.connect(self._on_inputs)

        mounts = Form("Mounts")
        self.mount = QtWidgets.QComboBox()
        self.mount.addItems([m.value for m in MOUNTS])
        self.mount.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        mounts.add("Kind:", self.mount, TIPS["mount"])
        self.deflection = mounts.number("Static sag δ:", 0.01, 1000.0, 5.0, 2, " mm", TIPS["deflection"])
        self.zeta = mounts.number("Damping ζ:", 0.0, 2.0, 0.05, 3, "", TIPS["zeta"])
        self.stiffness_ratio = mounts.number("Damper's spring N:", 0.01, 1000.0, 3.0, 2, " × k", TIPS["ratio"])
        self.stage_mass = mounts.number("Intermediate mass:", 0.01, 100.0, 0.5, 2, " × M", TIPS["stage"])
        self.mount_info = note_label()
        mounts.row(self.mount_info)
        mounts.changed.connect(self._on_inputs)
        self.mount.currentIndexChanged.connect(self._on_inputs)

        design = Form("Design")
        self.target = design.number("Isolation wanted:", 1.0, 99.99, 90.0, 2, " %", TIPS["target"])
        self.design_note = note_label()
        design.row(self.design_note)
        self.apply_design = QtWidgets.QPushButton("Use that sag")
        self.apply_design.setToolTip("<p>Set the mounts' static deflection to the one the target needs (for "
                                     "undamped single-stage mounts).</p>")
        self.apply_design.clicked.connect(self._on_apply_design)
        design.row(self.apply_design)
        design.changed.connect(self._show_design)

        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        for w in (self.trials, machine, mounts, design):
            lv.addWidget(w)
        lv.addStretch(1)

        # --- centre
        self.tr_plot = plot("Transmissibility T", None, "Frequency / fₙ", None, log_x=True, log_y=True, height=200)
        self.tr_plot.setToolTip(TR_TIP)
        self.tr_plot.setMouseEnabled(x=False, y=False)
        add_legend(self.tr_plot.getPlotItem(), offset=(-10, 10))
        self.family_items = [pg.PlotDataItem(name=f"ζ = {z:g}") for z in FAMILY]
        self.tr_item = pg.PlotDataItem(name="this isolator")
        self.unit_line = pg.InfiniteLine(angle=0, pos=0.0)
        self.root2_line = pg.InfiniteLine(angle=90, pos=math.log10(math.sqrt(2)))
        self.tr_now = pg.PlotDataItem(pen=None, symbol="o", symbolSize=12)
        for item in (self.unit_line, self.root2_line, *self.family_items, self.tr_item, self.tr_now):
            self.tr_plot.addItem(item)
        self.tr_note = readout_label()
        self.force_plot = plot("Force on the floor", "N", "Speed", "rpm", log_y=True, height=150)
        self.force_plot.setToolTip(FORCE_TIP)
        self.force_plot.setMouseEnabled(x=False, y=False)
        add_legend(self.force_plot.getPlotItem(), offset=(10, 10))
        self.rigid_item = pg.PlotDataItem(name="rigid mount (mₑeΩ²)")
        self.floor_item = pg.PlotDataItem(name="on these mounts")
        self.force_now = pg.PlotDataItem(pen=None, symbol="o", symbolSize=12)
        for item in (self.rigid_item, self.floor_item, self.force_now):
            self.force_plot.addItem(item)

        center = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(center)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.addWidget(self.tr_plot, 3)
        cv.addWidget(self.tr_note)
        cv.addWidget(self.force_plot, 2)

        # --- right
        start = QtWidgets.QWidget()
        sv = QtWidgets.QVBoxLayout(start)
        row = Form("Run up")
        self.ramp = row.number("Run up in:", 0.1, 600.0, 2.0, 1, " s", TIPS["ramp"])
        row.changed.connect(self._draw_tab)
        sv.addWidget(row)
        self.speed_plot = plot("Speed", "rpm", "Time", "s", height=80)
        self.x_plot = plot("Machine x", "m", "Time", "s", height=110)
        self.floor_plot = plot("Floor force", "N", "Time", "s", height=110)
        for p in (self.x_plot, self.floor_plot):
            p.setXLink(self.speed_plot)
        for p in (self.speed_plot, self.x_plot, self.floor_plot):
            p.setToolTip(START_TIP)
            sv.addWidget(p, 1)
        self.speed_item = pg.PlotDataItem()
        self.res_line = pg.InfiniteLine(angle=0)
        self.speed_plot.addItem(self.speed_item)
        self.speed_plot.addItem(self.res_line)
        self.sx_item = pg.PlotDataItem()
        self.x_plot.addItem(self.sx_item)
        self.sf_item = pg.PlotDataItem()
        self.sf_rigid = pg.PlotDataItem()
        self.floor_plot.addItem(self.sf_rigid)
        self.floor_plot.addItem(self.sf_item)
        self.start_note = note_label()
        sv.addWidget(self.start_note)

        chart = QtWidgets.QWidget()
        chv = QtWidgets.QVBoxLayout(chart)
        self.chart_plot = plot("Static deflection", "mm", "Running speed", "rpm", log_x=True, log_y=True, height=200)
        self.chart_plot.setToolTip(CHART_TIP)
        self.chart_plot.setMouseEnabled(x=False, y=False)
        add_legend(self.chart_plot.getPlotItem(), offset=(-10, 10))
        self.chart_items = [pg.PlotDataItem(name=f"{100 * level:g} %") for level in CHART_LEVELS]
        for item in self.chart_items:
            self.chart_plot.addItem(item)
        self.chart_now = pg.PlotDataItem(pen=None, symbol="o", symbolSize=12)
        self.chart_plot.addItem(self.chart_now)
        chv.addWidget(self.chart_plot, 1)
        self.chart_note = note_label()
        chv.addWidget(self.chart_note)

        self.notes = QtWidgets.QTextBrowser()
        self.notes.setHtml(NOTES_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(start, "Start-up")
        self.tabs.addTab(chart, "Design chart")
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
        for box, value in ((self.mass, 100.0), (self.rpm, t.rpm), (self.unbalance, 0.01), (self.block, t.block),
                           (self.mount, MOUNTS.index(t.mount)), (self.deflection, t.deflection_mm),
                           (self.zeta, t.zeta), (self.stiffness_ratio, t.stiffness_ratio),
                           (self.stage_mass, t.stage_mass), (self.ramp, t.ramp)):
            set_quietly(box, value)
        self.trials.show_note(t.note)
        self.tabs.setCurrentIndex(t.tab)
        self._on_inputs()

    def _on_inputs(self) -> None:
        mount = MOUNTS[self.mount.currentIndex()]
        self.iso = Isolator(self.mass.value(), self.rpm.value(), self.unbalance.value(), 1e-3 * self.deflection.value(),
                            self.zeta.value(), mount, self.stiffness_ratio.value(), self.stage_mass.value(),
                            self.block.value())
        for box in (self.stiffness_ratio,):
            box.setEnabled(mount is Mount.RELAXATION)
        self.stage_mass.setEnabled(mount is Mount.TWO_STAGE)
        self.redraw()

    def _on_apply_design(self) -> None:
        self.deflection.setValue(round(1e3 * deflection_for(self.iso.speed_hz, self.target.value() / 100.0), 2))

    # ------------------------------------------------------------- drawing
    def apply_theme(self) -> None:
        c = colors
        greys = (c.faint, c.grey, c.muted, c.structure)
        for item, g in zip(self.family_items, greys):
            item.setPen(pg.mkPen(g, width=1))
        self.tr_item.setPen(pg.mkPen(c.mass[0], width=2.5))
        self.unit_line.setPen(pg.mkPen(c.faint, width=1))
        self.root2_line.setPen(pg.mkPen(c.fit, width=1, style=QtCore.Qt.PenStyle.DotLine))
        for item in (self.tr_now, self.force_now, self.chart_now):
            item.setSymbolPen(pg.mkPen(c.force, width=2))
            item.setSymbolBrush(None)
        self.rigid_item.setPen(pg.mkPen(c.grey, width=1.5, style=QtCore.Qt.PenStyle.DashLine))
        self.floor_item.setPen(pg.mkPen(c.mass[0], width=2))
        self.speed_item.setPen(pg.mkPen(c.strong, width=1.5))
        self.res_line.setPen(pg.mkPen(c.fit, width=1, style=QtCore.Qt.PenStyle.DotLine))
        self.sx_item.setPen(pg.mkPen(c.mass[0], width=1))
        self.sf_item.setPen(pg.mkPen(c.force, width=1))
        self.sf_rigid.setPen(pg.mkPen(c.grey, width=1, style=QtCore.Qt.PenStyle.DashLine))
        for item, color in zip(self.chart_items, (c.mode[0], c.mode[1], c.mode[2], c.mode[3], c.mass[4])):
            item.setPen(pg.mkPen(color, width=2))
        self.redraw()

    def redraw(self) -> None:
        iso = self.iso
        fn = iso.natural_hz
        text = (f"k = {iso.stiffness:.4g} N/m, c = {iso.damping:.4g} N·s/m, f<sub>n</sub> = {fn:.4g} Hz "
                f"({60 * fn:.0f} rpm); running at r = {iso.speed_hz / fn:.3g}.")
        self.mount_info.setText(text)
        self._draw_transmissibility()
        self._show_design()
        self._draw_tab()
        settle_plots(self)

    def _draw_transmissibility(self) -> None:
        iso = self.iso
        fn = iso.natural_hz
        for item, z in zip(self.family_items, FAMILY):
            plain = replace(iso, zeta=z, mount=Mount.VISCOUS, block=0.0)
            T = plain.transmissibility(RATIOS * fn)
            item.setData(RATIOS, np.where(np.isfinite(T), T, np.nan), connect="finite")
        f = RATIOS * fn
        f = f[np.abs(f / fn - 1) > 1e-9] if iso.zeta == 0 else f
        T = iso.transmissibility(f)
        self.tr_item.setData(f / fn, T, connect="finite")
        X, Ff, Tn = iso.at_speed()
        r = iso.speed_hz / fn
        self.tr_now.setData([r], [Tn]) if RATIOS[0] <= r <= RATIOS[-1] else self.tr_now.setData([], [])
        self.tr_plot.setXRange(math.log10(RATIOS[0]), math.log10(RATIOS[-1]), padding=0)
        self.tr_plot.setYRange(-4, math.log10(30.0), padding=0)
        F = float(iso.force_at(iso.speed_hz))
        verdict = (f"isolation {100 * (1 - Tn):.3g} % ({db(Tn):.3g} dB)" if Tn < 1 else
                   f"amplified {Tn:.3g} times ({db(Tn):+.3g} dB)")
        self.tr_note.setText(
            f"At {iso.rpm:.0f} rpm the unbalance pulls with {fmt(F, 'N')}; {fmt(Ff, 'N')} reaches the floor: T = "
            f"{Tn:.3g}, {verdict}. The machine moves ±{fmt(X, 'm')}.")
        rpm = np.geomspace(max(iso.rpm / 50.0, 1.0), iso.rpm * 3.0, 800)
        fr = rpm / 60.0
        if iso.zeta == 0:
            fr = fr[np.abs(fr / fn - 1) > 1e-6]
            rpm = fr * 60.0
        force = iso.force_at(fr)
        self.rigid_item.setData(rpm, force)
        self.floor_item.setData(rpm, iso.transmissibility(fr) * force, connect="finite")
        self.force_now.setData([iso.rpm], [Ff])
        self.force_plot.setXRange(rpm[0], rpm[-1], padding=0)

    def _show_design(self) -> None:
        iso = self.iso
        target = self.target.value() / 100.0
        try:
            d = deflection_for(iso.speed_hz, target)
        except ValueError:
            self.design_note.setText("")
            return
        self.design_note.setText(f"{100 * target:.4g} % at {iso.rpm:.0f} rpm needs f<sub>n</sub> ≤ "
                                 f"{math.sqrt(G / d) / TWO_PI:.3g} Hz: a sag of {1e3 * d:.3g} mm or more "
                                 "(undamped, one stage).")
        if self.tabs.currentIndex() == TAB_CHART:
            self._draw_chart()

    def _draw_tab(self) -> None:
        i = self.tabs.currentIndex()
        if i == TAB_START:
            self._draw_start()
        elif i == TAB_CHART:
            self._draw_chart()

    def _draw_start(self) -> None:
        iso = self.iso
        ramp = self.ramp.value()
        hold = max(0.25 * ramp, 1.0, 6.0 / max(iso.zeta * TWO_PI * iso.natural_hz, 1e-3) / 3)
        hold = min(hold, 4.0 * ramp + 2.0)
        s = start_up(iso, ramp, hold)
        stride = max(1, s.t.size // 6000)
        self.speed_item.setData(s.t[::stride], 60.0 * s.speed_hz[::stride])
        self.res_line.setPos(60.0 * iso.natural_hz)
        self.sx_item.setData(s.t[::stride], s.x[::stride])
        self.sf_item.setData(s.t[::stride], s.floor[::stride])
        self.sf_rigid.setData(s.t[::stride], iso.force_at(s.speed_hz[::stride]))
        self.speed_plot.setXRange(0, s.t[-1], padding=0)
        during = s.t <= ramp
        peak_f = float(np.max(np.abs(s.floor[during]))) if during.any() else 0.0
        peak_x = float(np.max(np.abs(s.x[during]))) if during.any() else 0.0
        X, Ff, _ = iso.at_speed()
        self.start_note.setText(
            f"Through resonance ({60 * iso.natural_hz:.0f} rpm, dotted) the floor force peaks at {fmt(peak_f, 'N')} "
            f"and the machine at ±{fmt(peak_x, 'm')}; running, {fmt(Ff, 'N')} and ±{fmt(X, 'm')}. Dashed: the "
            "unbalance's own force.")

    def _draw_chart(self) -> None:
        iso = self.iso
        rpm = 60.0 * CHART_SPEEDS
        for item, level in zip(self.chart_items, CHART_LEVELS):
            item.setData(rpm, [1e3 * deflection_for(f, level) for f in CHART_SPEEDS])
        self.chart_now.setData([iso.rpm], [1e3 * iso.deflection])
        self.chart_plot.setXRange(math.log10(rpm[0]), math.log10(rpm[-1]), padding=0)
        T = iso.at_speed()[2]
        self.chart_note.setText(f"This machine: {iso.rpm:.0f} rpm on {1e3 * iso.deflection:.3g} mm mounts "
                                f"({100 * (1 - T):.3g} % with its damping and kind). Above a line, better isolated "
                                "than it.")
