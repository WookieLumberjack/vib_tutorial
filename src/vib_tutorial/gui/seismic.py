"""Seismic response spectrum page: the Simulation page's chain as a building, a design spectrum, and the modes'
peaks combined, against the building stepped on records that match the spectrum.

Left: the design spectrum (peak ground acceleration, ground type, damping) and the combination rule. Centre: the
spectrum with each mode on it, and the modes' table (period, participation, effective mass, base shear). Right:
each floor's peak displacement and each storey's shear by every rule and by the records, the modes' correlation,
the ground motions, and the Theory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.model import ChainSystem
from ..core.random_vibration import G
from ..core.seismic import (
    GROUNDS,
    RECORDS,
    RULES,
    Design,
    Rule,
    building,
    combine,
    correlation,
    design_sa,
    modal,
    records,
    suite,
    with_tank,
)
from . import settle_plots
from .lesson import (
    Form,
    TrialBox,
    column,
    combo,
    connect_all,
    fmt,
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

TAB_FLOORS, TAB_CORRELATION, TAB_GROUND = range(3)
PERIODS = np.linspace(0.0, 4.0, 801)[1:]


@dataclass(frozen=True)
class SeismicTrial:
    name: str
    note: str
    storeys: int = 5
    pga: float = 0.3
    ground: int = 2
    zeta: float = 5.0  # %
    tank: float = 0.0  # the rooftop tank's share of mode 1's modal mass there
    rule: Rule = Rule.CQC
    tab: int = TAB_FLOORS


TRIALS = (
    SeismicTrial(
        "A five-storey building",
        "Five 10 t floors on 5 MN/m storeys: T₁ = 0.99 s. Mode 1 carries 88 % of the mass and modes 1 and 2 "
        "together 97 % (codes want 90 %), so a few modes are enough. The spectrum gives mode 1 S<sub>a</sub> = "
        "0.52 g, and the modes together a base shear of 229 kN (CQC or SRSS); stepped on seven records that match "
        "the spectrum, the building's base shear averages 235 kN.",
        rule=Rule.CQC),
    SeismicTrial(
        "ABS is too safe",
        "The same building by ABS: every mode's peak added as if they all arrived together, 277 kN, 18 % over "
        "the records' mean. Modes this far apart (ρ₁₂ = 0.007) peak at different moments, and SRSS (229 kN) is "
        "right.",
        rule=Rule.ABS),
    SeismicTrial(
        "Close modes need CQC",
        "A tank on the roof, 1 % of mode 1's modal mass there, tuned to mode 1, splits it into two modes "
        "0.96 and 1.06 Hz apart, with ρ = 0.50: they move together much of the time. SRSS treats them as "
        "independent: 21 % low on the base shear, and 54 % high on the tank's own swing, where the two modes "
        "nearly cancel. CQC keeps the cross term, with its sign, and is within 11 % of the records' mean on both.",
        tank=0.01, rule=Rule.SRSS),
    SeismicTrial(
        "Soft soil",
        "Ground type D: S = 1.35 and the plateau runs to T<sub>C</sub> = 0.8 s, so mode 1 (0.99 s) sits near it. "
        "Its S<sub>a</sub> rises from 0.52 g to 0.82 g, and the base shear from 229 to 357 kN: a site's soil "
        "matters as much as its seismicity.",
        ground=3),
    SeismicTrial(
        "A taller building",
        "Eight storeys: T₁ = 1.52 s, and mode 1 carries 86 % of the mass, under the 90 % codes want, so mode 2 "
        "is needed too. Mode 1 now sits on the falling branch (0.34 g) and modes 2 to 4 on the plateau (0.86 g): "
        "mode 1 alone gives 96 % of the base shear but only 74 % of the top storey's.",
        storeys=8),
    SeismicTrial(
        "More damping",
        "ζ = 10 % in every mode (dampers added): η = √(10/(5 + 10)) = 0.82 lowers the whole spectrum by 18 %, and "
        "every peak with it.",
        zeta=10.0),
)

TIPS = {
    "dof": "<p>The number of floors: the Simulation page's chain is the building, m1 the first floor.</p>",
    "pga": "<p>The design peak ground acceleration a<sub>g</sub> on rock, in g.</p>",
    "ground": "<p>The ground type: soft soil amplifies the motion (S) and widens the plateau (T<sub>C</sub>).</p>",
    "zeta": "<p>The damping ratio of every mode, for the spectrum (η) and the records' response.</p>",
    "rule": "<p>How the modes' peaks are combined for the readout and the bold profile.</p>",
}
SPECTRUM_TIP = ("<p>The design spectrum: the peak acceleration of a single oscillator of period T and damping ζ. "
                "Dots: the building's modes.</p>")
FLOOR_TIP = ("<p>Each floor's peak displacement relative to the ground (left) and each storey's peak shear "
             "(right): by each combination rule, and the mean over the records (dots).</p>")
GROUND_TIP = ("<p>One of the records (top), and every record's spectrum against the design spectrum (bottom).</p>")


NOTES_HTML = """
<!--BEFORE-->
<h3><a name="design-spectrum"></a>A design spectrum</h3>
<p>The Shock spectrum page's idea, used for earthquakes: a single oscillator's peak response to the ground's
motion, plotted against its period. An earthquake's spectrum is jagged and no two are alike, so a design code
smooths and envelopes many into a <b>design spectrum</b>: a rise from the ground's own acceleration
a<sub>g</sub>S at T = 0, a plateau of 2.5 η a<sub>g</sub>S from T<sub>B</sub> to T<sub>C</sub>, then falls as
1/T (constant velocity) and 1/T² (constant displacement). The page uses the shape of the European code's elastic
spectrum. The ground type sets S and the corners: soft soil amplifies and widens the plateau. Damping scales it
by η = √(10/(5 + ζ%)).</p>

<h3><a name="participation"></a>Participation and effective mass</h3>
<p>A building on moving ground is pushed by −M1 a<sub>g</sub> (each floor's inertia, in coordinates relative to
the ground), so each mode r is a single oscillator driven by Γ<sub>r</sub> a<sub>g</sub>, with the
<b>participation factor</b></p>
<p align="center">Γ<sub>r</sub> = φ<sub>r</sub>ᵀM1 / φ<sub>r</sub>ᵀMφ<sub>r</sub>.</p>
<p>Its peak modal coordinate is Γ<sub>r</sub>S<sub>a</sub>(T<sub>r</sub>)/ω<sub>r</sub>², its floor displacements
φ<sub>r</sub> times that, and its base shear M<sub>eff,r</sub> S<sub>a</sub>(T<sub>r</sub>) with the <b>effective
modal mass</b> M<sub>eff,r</sub> = (φ<sub>r</sub>ᵀM1)²/φ<sub>r</sub>ᵀMφ<sub>r</sub>. The effective masses add up
to the building's mass, so their running sum says how many modes are enough: codes ask for 90 %.</p>

<h3><a name="combination"></a>Combining the modes</h3>
<p>The spectrum gives each mode's peak, not when it happens. Adding them (<b>ABS</b>) assumes they all peak
together: safe, usually far too safe. <b>SRSS</b>, √Σ R<sub>r</sub>², assumes they are independent, which holds when
their frequencies are well apart. <b>NRL</b> (the largest plus the SRSS of the rest) sits between. When two modes
are close they move together much of the time, and SRSS underestimates; the <b>complete quadratic
combination</b></p>
<p align="center">R = √(Σ<sub>i</sub>Σ<sub>j</sub> R<sub>i</sub> ρ<sub>ij</sub> R<sub>j</sub>)</p>
<p>keeps the cross terms, with Der Kiureghian's correlation ρ<sub>ij</sub> (1 on the diagonal, falling off as
the frequencies part, faster the less the damping): ρ = 8ζ²(1 + r)r<sup>3/2</sup> / ((1 − r²)² +
4ζ²r(1 + r)²) for equal damping and r = ω<sub>j</sub>/ω<sub>i</sub>. CQC is SRSS when the modes are far apart, so
codes now use it throughout.</p>

<h3><a name="records"></a>Checking against records</h3>
<p>The page synthesizes seven ground motions whose spectra match the design spectrum (sines at random phases
under a rise, strong-shaking and decay envelope, each scaled until its spectrum meets the target), steps the
building on each mode by mode, exactly, and takes the mean of their peaks, as a code does with a set of
records. The rules are estimates of that mean; each record scatters about it by some 10 %.</p>

<h3><a name="try-it"></a>Things to try</h3>
<p>Each of these loads a building into the Simulation page and sets up the spectrum (Things to try, top
left).</p>
<!--TRIALS-->
<!--TAKEAWAYS-->
"""
NOTES_HTML = NOTES_HTML.replace("<!--BEFORE-->", before(
    ("shock_page", "srs", "The shock response spectrum"),
    ("shock_page", "chain", "The chain's modes on a pulse"),
)).replace("<!--TRIALS-->", trials_html(TRIALS)).replace("<!--TAKEAWAYS-->", takeaways(
    "A design spectrum smooths many earthquakes' spectra into a plateau and 1/T, 1/T² falls; soil and damping "
    "scale it.",
    "Each mode is an oscillator driven by Γ a<sub>g</sub>; its effective mass is its share of the base shear, and "
    "the shares add up to the whole mass.",
    "ABS adds the modes' peaks and overestimates; SRSS is right for well-separated modes.",
    "Close modes are correlated: CQC keeps the cross terms and SRSS underestimates.",
))


def _same(a: ChainSystem, b: ChainSystem) -> bool:
    return (a.n == b.n and np.allclose(a.masses, b.masses) and np.allclose(a.stiffness, b.stiffness)
            and np.allclose(a.damping, b.damping))


class SeismicPage(QtWidgets.QWidget):
    """Controls on the left, the spectrum and the modes in the centre, the floors, correlation and records on the
    right."""

    dof_requested = QtCore.Signal(int)
    system_requested = QtCore.Signal(object)
    edit_parameters = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.system = ChainSystem.uniform(4)
        self._dirty = True

        self.trials = TrialBox([t.name for t in TRIALS],
                               "<p>Each loads a building into the Simulation page and sets up the spectrum to show "
                               "one idea from the Theory tab.</p>")
        self.trials.picked.connect(self.load_trial)

        ch = Form("Building (the chain)")
        self.dof = QtWidgets.QSpinBox()
        self.dof.setRange(1, MAX_DOF)
        self.dof.valueChanged.connect(self.dof_requested)
        ch.add("Floors:", self.dof, TIPS["dof"])
        link = QtWidgets.QLabel("Masses and storey stiffnesses are edited on the <a href='#sim'>Simulation "
                                "page</a>.")
        link.setWordWrap(True)
        mute(link)
        link.linkActivated.connect(lambda _: self.edit_parameters.emit())
        ch.row(link)

        spec = Form("Design spectrum")
        self.pga = spec.number("Peak ground acc.:", 0.01, 2.0, 0.3, 2, " g", TIPS["pga"])
        self.ground = combo(GROUNDS, TIPS["ground"])
        self.ground.setCurrentIndex(2)
        spec.add("Ground type:", self.ground, TIPS["ground"])
        self.zeta = spec.number("Damping ζ:", 0.5, 30.0, 5.0, 1, " %", TIPS["zeta"])
        self.rule = combo([r.value for r in RULES], TIPS["rule"])
        self.rule.setCurrentIndex(RULES.index(Rule.CQC))
        spec.add("Combine by:", self.rule, TIPS["rule"])
        self.spec_info = note_label()
        spec.row(self.spec_info)
        connect_all((self.pga, self.ground, self.zeta, self.rule), self.redraw)

        # --- centre
        self.spectrum_plot = plot("S<sub>a</sub> (g)", None, "Period T", "s", height=150)
        self.spectrum_plot.setToolTip(SPECTRUM_TIP)
        self.spectrum_item = pg.PlotDataItem()
        self.mode_dots = marker(11)
        self.corner_lines = [pg.InfiniteLine(angle=90) for _ in range(3)]
        for item in (*self.corner_lines, self.spectrum_item, self.mode_dots):
            self.spectrum_plot.addItem(item)
        self.mode_labels: list[pg.TextItem] = []
        self.table = readout_label()
        self.table.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)

        # --- right: floors
        floors = QtWidgets.QWidget()
        fl = QtWidgets.QHBoxLayout(floors)
        fl.setContentsMargins(0, 0, 0, 0)
        self.disp_plot = plot("Floor", None, "Peak displacement", "m", height=200)
        self.shear_plot = plot("Storey", None, "Peak shear", "N", height=200)
        self.rule_items: dict[Rule, tuple[pg.PlotDataItem, pg.PlotDataItem]] = {}
        add_legend(self.disp_plot.getPlotItem(), offset=(10, 10))
        for r in RULES:
            self.rule_items[r] = (pg.PlotDataItem(name=r.value), pg.PlotDataItem())
            self.disp_plot.addItem(self.rule_items[r][0])
            self.shear_plot.addItem(self.rule_items[r][1])
        self.th_items = (marker(9, name="records' mean"), marker(9))
        self.disp_plot.addItem(self.th_items[0])
        self.shear_plot.addItem(self.th_items[1])
        for p in (self.disp_plot, self.shear_plot):
            p.setToolTip(FLOOR_TIP)
            p.setMouseEnabled(x=False, y=False)
            p.getAxis("left").setTickSpacing(1, 1)
            fl.addWidget(p)
        self.floor_note = note_label()

        self.rho_label = readout_label()
        self.rho_note = note_label()

        self.record_plot = plot("Ground acc.", "m/s²", "Time", "s", height=120)
        self.record_item = pg.PlotDataItem()
        self.record_plot.addItem(self.record_item)
        self.match_plot = plot("S<sub>a</sub> (g)", None, "Period T", "s", log_x=True, height=150)
        self.match_items: list[pg.PlotDataItem] = []
        self.target_item = pg.PlotDataItem(name="design")
        self.match_mean = pg.PlotDataItem(name="the records' mean")
        add_legend(self.match_plot.getPlotItem(), offset=(-10, 10))
        for item in (self.target_item, self.match_mean):
            self.match_plot.addItem(item)
        for p in (self.record_plot, self.match_plot):
            p.setToolTip(GROUND_TIP)
        self.ground_note = note_label()

        self.notes = theory(NOTES_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(column(floors, self.floor_note, stretch=False), "Floors")
        self.tabs.addTab(column(self.rho_label, self.rho_note), "Correlation")
        self.tabs.addTab(column(self.record_plot, self.match_plot, self.ground_note, stretch=False), "Records")
        self.tabs.addTab(self.notes, "Theory")
        self.tabs.currentChanged.connect(lambda _: self._draw_tab())

        three_panes(self, column(self.trials, ch, spec),
                    stack((self.spectrum_plot, 1), (self.table, 0)), self.tabs)
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
        system = building(t.storeys)
        if t.tank:
            system = with_tank(system, t.tank)
        self.system_requested.emit(system)
        if not _same(self.system, system):
            self.set_system(system)
        for box, value in ((self.pga, t.pga), (self.ground, t.ground), (self.zeta, t.zeta),
                           (self.rule, RULES.index(t.rule))):
            set_quietly(box, value)
        self.trials.show_note(t.note)
        self.tabs.setCurrentIndex(t.tab)
        self.redraw()

    @property
    def design(self) -> Design:
        return Design(self.pga.value(), GROUNDS[self.ground.currentIndex()], self.zeta.value() / 100)

    # ------------------------------------------------------------- drawing
    def apply_theme(self) -> None:
        c = colors
        self.spectrum_item.setPen(pg.mkPen(c.strong, width=2))
        mark(self.mode_dots, c.force, filled=True)
        for line in self.corner_lines:
            line.setPen(pg.mkPen(c.faint, width=1, style=QtCore.Qt.PenStyle.DotLine))
        styles = {Rule.ABS: (c.grey, QtCore.Qt.PenStyle.DashLine), Rule.SRSS: (c.mass[1], QtCore.Qt.PenStyle.SolidLine),
                  Rule.NRL: (c.mode[2], QtCore.Qt.PenStyle.DashDotLine), Rule.CQC: (c.mass[0], QtCore.Qt.PenStyle.SolidLine)}
        for r, items in self.rule_items.items():
            color, style = styles[r]
            for item in items:
                item.setPen(pg.mkPen(color, width=2, style=style))
                item.setSymbol("o")
                item.setSymbolSize(5)
                item.setSymbolBrush(pg.mkBrush(color))
                item.setSymbolPen(None)
        for item in self.th_items:
            mark(item, c.strong)
        self.record_item.setPen(pg.mkPen(c.mass[0], width=1))
        self.target_item.setPen(pg.mkPen(c.strong, width=2.5))
        self.match_mean.setPen(pg.mkPen(c.force, width=2, style=QtCore.Qt.PenStyle.DashLine))
        self._dirty = True
        if self.isVisible():
            self.redraw()

    def redraw(self) -> None:
        self._dirty = False
        d = self.design
        S, TB, TC, TD = d.corners
        self.spec_info.setText(f"S = {S:g}, T<sub>B</sub> = {TB:g} s, T<sub>C</sub> = {TC:g} s, T<sub>D</sub> = "
                               f"{TD:g} s, η = {d.eta:.3g}: a plateau of {2.5 * d.eta * S * d.pga_g:.3g} g.")
        self.modal = modal(self.system, d)
        self.rho = correlation(self.modal.f, d.zeta)
        self._draw_spectrum()
        self._draw_table()
        self._draw_tab()
        settle_plots(self)

    def _draw_spectrum(self) -> None:
        d, m = self.design, self.modal
        self.spectrum_item.setData(PERIODS, design_sa(d, PERIODS) / G)
        self.mode_dots.setData(m.T, m.sa / G)
        for line, T in zip(self.corner_lines, d.corners[1:]):
            line.setPos(T)
        for label in self.mode_labels:
            self.spectrum_plot.removeItem(label)
        self.mode_labels = []
        for r in range(min(m.f.size, 4)):
            label = pg.TextItem(str(r + 1), color=colors.force, anchor=(0.5, 1.3))
            label.setPos(float(m.T[r]), float(m.sa[r] / G))
            self.spectrum_plot.addItem(label)
            self.mode_labels.append(label)
        self.spectrum_plot.setXRange(0.0, max(4.0, 1.05 * float(m.T[0])), padding=0)
        self.spectrum_plot.setYRange(0.0, 1.15 * float(np.max(design_sa(d, PERIODS))) / G, padding=0)

    def _draw_table(self) -> None:
        m = self.modal
        rule = RULES[self.rule.currentIndex()]
        cum = np.cumsum(m.mass_share)
        rows = "".join(
            f"<tr><td>{r + 1}</td><td align='right'>{m.f[r]:.3g}</td><td align='right'>{m.T[r]:.3g}</td>"
            f"<td align='right'>{m.gamma[r] * np.sign(m.phi[-1, r]):.3g}</td>"
            f"<td align='right'>{100 * m.mass_share[r]:.3g} %</td><td align='right'>{100 * cum[r]:.3g} %</td>"
            f"<td align='right'>{m.sa[r] / G:.3g}</td><td align='right'>{fmt(abs(m.base_shear[r]), 'N')}</td></tr>"
            for r in range(m.f.size))
        totals = " · ".join(f"{'<b>' if r is rule else ''}{r.value} {fmt(float(combine(m.shear, r, self.rho)[0]), 'N')}"
                            f"{'</b>' if r is rule else ''}" for r in RULES)
        need = int(np.searchsorted(cum, 0.9 - 1e-9)) + 1
        self.table.setText(
            "<table cellspacing='5'><tr><th>mode</th><th>f (Hz)</th><th>T (s)</th><th>Γ</th><th>M<sub>eff</sub></th>"
            f"<th>Σ M<sub>eff</sub></th><th>S<sub>a</sub> (g)</th><th>V<sub>base</sub></th></tr>{rows}</table>"
            f"Base shear: {totals}. {need} mode{'s' if need > 1 else ''} reach{'' if need > 1 else 'es'} 90 % of "
            f"the mass ({m.total_mass / 1000:.4g} t).")

    def _draw_tab(self) -> None:
        i = self.tabs.currentIndex()
        if i == TAB_FLOORS:
            self._draw_floors()
        elif i == TAB_CORRELATION:
            self._draw_correlation()
        elif i == TAB_GROUND:
            self._draw_ground()

    def _draw_floors(self) -> None:
        m, rho = self.modal, self.rho
        rule = RULES[self.rule.currentIndex()]
        floors = np.arange(1, m.f.size + 1)
        for r, (di, si) in self.rule_items.items():
            width = 3 if r is rule else 1.5
            for item in (di, si):
                pen = item.opts["pen"]
                pen.setWidthF(width)
                item.setPen(pen)
            di.setData(combine(m.displacement, r, rho), floors)
            si.setData(combine(m.shear, r, rho), floors)
        s = suite(self.system, self.design)
        self.th_items[0].setData(s.displacement, floors)
        self.th_items[1].setData(s.shear, floors)
        for p in (self.disp_plot, self.shear_plot):
            p.setYRange(0.5, m.f.size + 0.5, padding=0)
            p.enableAutoRange(axis="x")
        mine = float(combine(m.shear, rule, rho)[0])
        top = float(combine(m.displacement, rule, rho)[-1])
        self.floor_note.setText(
            f"Base shear by {rule.value} {fmt(mine, 'N')}, {100 * (mine / s.base_shear - 1):+.3g} % against the "
            f"mean of {RECORDS} records ({fmt(s.base_shear, 'N')}); the top floor's displacement "
            f"{fmt(top, 'm')}, {100 * (top / s.displacement[-1] - 1):+.3g} %.")

    def _draw_correlation(self) -> None:
        rho = self.rho
        n = rho.shape[0]
        head = "<tr><th></th>" + "".join(f"<th>{j + 1}</th>" for j in range(n)) + "</tr>"
        rows = []
        for i in range(n):
            cells = "".join(
                f"<td align='right'>{'<b>' if i != j and rho[i, j] > 0.1 else ''}{rho[i, j]:.3f}"
                f"{'</b>' if i != j and rho[i, j] > 0.1 else ''}</td>" for j in range(n))
            rows.append(f"<tr><th>{i + 1}</th>{cells}</tr>")
        self.rho_label.setText(f"<p><b>ρ<sub>ij</sub></b> at ζ = {100 * self.design.zeta:.3g} %</p>"
                               f"<table cellspacing='6'>{head}{''.join(rows)}</table>")
        off = rho - np.diag(np.diag(rho))
        i, j = np.unravel_index(int(np.argmax(off)), off.shape)
        f = self.modal.f
        self.rho_note.setText(f"The most correlated pair: modes {i + 1} and {j + 1} ({f[i]:.3g} and {f[j]:.3g} Hz, "
                              f"ρ = {off[i, j]:.3g}). Above about 0.1 SRSS and CQC part; bold.")

    def _draw_ground(self) -> None:
        d = self.design
        recs = records(d)
        g = recs[0]
        self.record_item.setData(g.t, g.a)
        self.record_plot.setXRange(0.0, float(g.t[-1]), padding=0)
        for item in self.match_items:
            self.match_plot.removeItem(item)
        self.match_items = []
        pen = pg.mkPen(colors.grey, width=1)
        for rec in recs:
            item = pg.PlotDataItem(1 / rec.f, rec.spectrum / G, pen=pen)
            self.match_plot.addItem(item)
            self.match_items.append(item)
        T = 1 / recs[0].f
        self.target_item.setData(T, design_sa(d, T) / G)
        mean = np.mean([rec.spectrum for rec in recs], axis=0)
        self.match_mean.setData(T, mean / G)
        err = mean / design_sa(d, T) - 1
        pga = np.mean([np.max(np.abs(rec.a)) for rec in recs]) / G
        self.ground_note.setText(f"{len(recs)} synthesized records (grey); their mean spectrum is within "
                                 f"{100 * float(np.max(np.abs(err))):.2g} % of the design one. Their peak ground "
                                 f"acceleration averages {pga:.2g} g.")
