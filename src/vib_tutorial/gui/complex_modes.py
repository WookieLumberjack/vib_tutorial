"""Complex modes page: when the chain's damping is not proportional, and what that does to its modes.

Left: the chain's damping (its coupling index, a button that makes it proportional), chains to
try, and the mode looked at. Centre: that mode's exact shape ψ in the complex plane with its
best-fit line, its snapshots over half a cycle, and every mode's exact pole and shape against
the proportional approximation's. Right: the FRF, exact and approximated; a free decay from an
undamped mode's shape, and what leaks to the others; each mode's phase scatter; the notes.
Follows the chain from the Simulation page (friction left out: modes are linear).
"""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.complex_modes import (
    ModeComparison,
    compare,
    free_decay,
    frf_grid,
    frf_pair,
    made_proportional,
    strobe,
)
from ..core.modal import ModalResult, modal_analysis
from ..core.model import ChainSystem
from ..core.presets import tuned_mass_damper
from .axes import log_axes
from .layouts import ColumnScroll, group, plot_height
from .style import MAX_DOF, colors
from .theming import add_legend, mute

SNAPSHOTS = 8  # over half a cycle, on the snapshots plot

TIPS = {
    "coupling": "<p>The largest off-diagonal term of Φ<sup>T</sup>CΦ against its diagonal's, "
                "|c<sub>rs</sub>|/√(c<sub>rr</sub>c<sub>ss</sub>): 0 when the damping is proportional, near 1 or "
                "more when it couples the modes strongly.</p>",
    "proportional": "<p>Set every damper in proportion to its spring, c<sub>i</sub> = βk<sub>i</sub>, so "
                    "C = βK, which the undamped modes diagonalize. β keeps mode 1's damping ratio.</p>",
    "mode": "<p>The mode drawn in the complex plane and the snapshots, and released in the free decay.</p>",
    "frf": "<p>The receptance from a force on one mass to another's displacement.</p>",
}
ARGAND_TIP = (
    "<p>The exact mode shape ψ in the complex plane: one arrow per mass. A real mode's arrows lie on one line "
    "(the dashed best-fit line); a complex mode's spread from it, each mass with its own phase.</p>")
STROBE_TIP = (
    "<p>The mode's shape at eight instants over half a cycle, Re(ψe<sup>iθ</sup>): a real mode's are one shape, "
    "scaled, and its nodes stay put (a standing wave); a complex mode's change shape as they go, and the nodes "
    "move along the chain. Grey: ±|ψ|, the envelope.</p>")
TABLE_TIP = (
    "<p>Each mode: the exact pole's frequency |λ|/2π and damping ratio, against the proportional approximation's "
    "(ζ = φ<sup>T</sup>Cφ/2ω, the undamped frequency); the exact shape's MAC with the undamped one, its MPC "
    "(1 real, 0 spread evenly) and its mean phase deviation from the best-fit line.</p>")
FRF_TIP = (
    "<p>The receptance, exact (K − ω²M + iωC)<sup>−1</sup>, and with the proportional approximation's damping "
    "C<sub>p</sub> = MΦ diag(Φ<sup>T</sup>CΦ) Φ<sup>T</sup>M: the off-diagonal terms dropped.</p>")
DECAY_TIP = (
    "<p>The chain released at rest in an undamped mode's shape, its modal coordinates q<sub>s</sub> = "
    "φ<sub>s</sub><sup>T</sup>Mx over time, exact. The approximation keeps the released mode alone, decaying at "
    "its ζ. With proportional damping the others stay at 0; otherwise the damping passes motion to them.</p>")
SCATTER_TIP = (
    "<p>Each mode's entries' angles from its best-fit line (one dot per mass, mass colours), and their mean, "
    "weighted by size (the MPD, a bar). All on 0 for real modes.</p>")


def _chain(damping: list[float]) -> ChainSystem:
    return ChainSystem(np.ones(4), np.full(4, 400.0), np.array(damping, dtype=float))


# Chains to try: (name, chain, note)
TRIALS = (
    ("A damper per spring, in proportion", ChainSystem.uniform(4),
     "Four equal masses, springs and dampers: c<sub>i</sub> = k<sub>i</sub>/200, so C = K/200 and the damping is "
     "proportional. Every mode is real (MPC 1, MPD 0°), the approximation is exact, and a mode released "
     "stays alone."),
    ("One damper, at the base", _chain([20.0, 0, 0, 0]),
     "The same chain with a single 20 N·s/m damper, on spring 1. Mode 3 is the most complex (MPC 0.20, "
     "MPD 29°), and the approximation puts mode 4's ζ at 0.049, twice the exact 0.023. Released in mode 2's "
     "shape, the chain passes 45 % of it to mode 1."),
    ("One damper, at the top", _chain([0, 0, 0, 20.0]),
     "The damper between m3 and m4: mode 4's ζ is 0.017, not the approximation's 0.17, ten times less. "
     "The approximation counts the damper's stretch in the undamped mode 4, but the exact mode 4 shakes m1 "
     "and m2 and leaves m3 and m4 moving nearly together (under a third of the stretch), so the damper barely "
     "works."),
    ("A strong damper locks a joint", _chain([0, 200.0, 0, 0]),
     "200 N·s/m between m1 and m2: so stiff a damper that the two move together, and it hardly works. Mode 1's "
     "ζ is 0.050, not the approximation's 0.58; mode 3's 0.0018, not 2.6; mode 4 has become a pair of real "
     "roots. Mode 2, where m1 and m2 already move together, is untouched."),
    ("Tuned mass damper", tuned_mass_damper(ChainSystem.uniform(4), 0.05),
     "The default chain with a Den Hartog damper on m4 (m5). Modes 1 and 2 are the structure's mode 1 split in "
     "two, close together, and the damper couples them: the most complex (MPD 10° and 17°), while the "
     "modes above, far from them, stay nearly real."),
)

NOTES_HTML = """
<h3>Proportional damping</h3>
<p>The undamped modes φ<sub>r</sub> diagonalize the mass and stiffness matrices: in modal coordinates the
chain is N separate oscillators. Damping keeps them separate only if Φ<sup>T</sup>CΦ is diagonal too:
<b>proportional</b> (or classical) damping. C = αM + βK (Rayleigh damping) is one such C; on this chain, a
damper per spring in proportion to it (c<sub>i</sub> = βk<sub>i</sub>) gives C = βK. Then each mode keeps its
real shape: every mass passes through zero at the same instant, and the nodes stand still. Its damping ratio
is ζ<sub>r</sub> = φ<sub>r</sub><sup>T</sup>Cφ<sub>r</sub>/2ω<sub>r</sub> (φ mass-normalized), its pole
−ζ<sub>r</sub>ω<sub>r</sub> ± iω<sub>r</sub>√(1 − ζ<sub>r</sub>²).</p>

<h3>Complex modes</h3>
<p>A damper on its own (a shock absorber, a joint, a tuned mass damper) is not proportional: Φ<sup>T</sup>CΦ
has off-diagonal terms c<sub>rs</sub> that couple the modes. The exact modes are then the eigenvectors of the
2N × 2N state matrix, and their shapes ψ are complex: each mass has its own phase and peaks at its own
time, the nodes move, and the motion is partly a wave travelling along the chain. To first order in the
damping,</p>
<p align="center">ψ<sub>r</sub> ≈ φ<sub>r</sub> + Σ<sub>s≠r</sub> iω<sub>r</sub>c<sub>sr</sub>
/ (ω<sub>r</sub>² − ω<sub>s</sub>²) φ<sub>s</sub>:</p>
<p>the imaginary part grows with the coupling and as the other modes come close. Close modes, as a tuned mass
damper makes, go complex first; a mode far from the others stays nearly real even under a heavy coupling.</p>

<h3>How complex</h3>
<ul>
<li><b>MPC</b>, the modal phase collinearity: (s<sub>xx</sub> − s<sub>yy</sub>)² + 4s<sub>xy</sub>² over
(s<sub>xx</sub> + s<sub>yy</sub>)², the s the sums of products of ψ's real and imaginary parts. 1 when the
entries lie on a line through the origin, 0 when they spread evenly round it.</li>
<li><b>MPD</b>, the mean phase deviation: the entries' angles from their best-fit line, weighted by their size.
0° for a real mode.</li>
<li><b>MAC</b>(φ, ψ): how much of the exact shape the undamped one explains.</li>
</ul>
<p>Measured modes are always a little complex (noise, a slightly non-linear structure, a sensor out of
phase), so a small MPD says little; a large one in an analysis says the damping couples the modes.</p>

<h3>What ignoring it costs</h3>
<p>The usual approximation keeps Φ<sup>T</sup>CΦ's diagonal and drops the rest. It is exact for proportional
damping and good when the damping is light and the modes are well apart. It fails where a damper is
strong enough to change the shapes: a damper that hardly moves in the exact mode (its two ends move
together) is counted as if it did, and the approximation's ζ can be ten times too high. It cannot see a mode
go overdamped, or a damper lock a joint. The <i>FRF</i> tab compares the receptances; the <i>Free decay</i> tab
releases the chain in an undamped mode's shape: exactly that mode, if the damping is proportional; otherwise
the damping hands some of the motion to the other modes, close ones most.</p>
"""


def _plot(left: str, left_units: str | None, bottom: str, bottom_units: str | None, log: bool = False
          ) -> pg.PlotWidget:
    p = pg.PlotWidget(axisItems=log_axes()) if log else pg.PlotWidget()
    p.setMenuEnabled(False)
    p.hideButtons()
    p.showGrid(x=True, y=True, alpha=0.15)
    p.setLabel("left", left, units=left_units)
    p.setLabel("bottom", bottom, units=bottom_units)
    p.getAxis("left").enableAutoSIPrefix(False)
    return p


def _cell(text: str) -> QtWidgets.QTableWidgetItem:
    item = QtWidgets.QTableWidgetItem(text)
    item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
    return item


def _percent(approx: float, exact: float) -> str:
    return f"{100 * (approx / exact - 1):+.3g} %" if exact and np.isfinite(exact) and np.isfinite(approx) else "–"


class ComplexModesPage(QtWidgets.QWidget):
    """The damping and the mode on the left, the mode's shape and the table in the centre, the rest on the right."""

    dof_requested = QtCore.Signal(int)  # the user changed N here; the main window owns the system
    edit_parameters = QtCore.Signal()  # the user wants the Simulation page's parameter panel
    system_requested = QtCore.Signal(object)  # a ChainSystem for the Simulation page to load

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.system: ChainSystem | None = None
        self.result: ModalResult | None = None
        self.rows: list[ModeComparison] = []
        self._dirty = False

        # --- left: the chain's damping, chains to try, the mode
        box = QtWidgets.QGroupBox("Damping")
        form = QtWidgets.QFormLayout(box)
        self.dof = QtWidgets.QSpinBox()
        self.dof.setRange(1, MAX_DOF)
        self.dof.setToolTip("Same chain as the Simulation page; changing it here changes it there too.")
        self.dof.valueChanged.connect(self.dof_requested)
        form.addRow("Number of masses:", self.dof)
        self.coupling = QtWidgets.QLabel()
        self.coupling.setWordWrap(True)
        self.coupling.setToolTip(TIPS["coupling"])
        form.addRow(self.coupling)
        self.make_proportional = QtWidgets.QPushButton("Make it proportional")
        self.make_proportional.setToolTip(TIPS["proportional"])
        form.addRow(self.make_proportional)
        link = QtWidgets.QLabel("Masses, springs and dampers are edited on the <a href='#sim'>Simulation page</a>.")
        mute(link)
        link.setWordWrap(True)
        link.linkActivated.connect(lambda _: self.edit_parameters.emit())
        form.addRow(link)

        trials = QtWidgets.QGroupBox("Chains to try")
        tv = QtWidgets.QVBoxLayout(trials)
        self.trial = QtWidgets.QComboBox()
        self.trial.addItem("Load a chain…")
        self.trial.addItems([name for name, _, _ in TRIALS])
        self.trial.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.trial.setToolTip("<p>Chains that show something about complex modes. Loading one replaces the "
                              "Simulation page's chain.</p>")
        tv.addWidget(self.trial)
        self.trial_note = QtWidgets.QLabel()
        self.trial_note.setWordWrap(True)
        self.trial_note.setTextFormat(QtCore.Qt.TextFormat.RichText)
        mute(self.trial_note)
        self.trial_note.setVisible(False)
        tv.addWidget(self.trial_note)

        mode_box = QtWidgets.QGroupBox("Mode")
        form = QtWidgets.QFormLayout(mode_box)
        self.mode = QtWidgets.QComboBox()
        self.mode.setToolTip(TIPS["mode"])
        label = QtWidgets.QLabel("Show:")
        label.setToolTip(TIPS["mode"])
        form.addRow(label, self.mode)
        self.mode_info = QtWidgets.QLabel()
        self.mode_info.setWordWrap(True)
        self.mode_info.setTextFormat(QtCore.Qt.TextFormat.RichText)
        form.addRow(self.mode_info)

        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        for b in (box, mode_box, trials):
            lv.addWidget(b)
        lv.addStretch(1)
        left_scroll = ColumnScroll(left)

        # --- centre: the mode in the complex plane, its snapshots, the table
        self.argand = _plot("Im ψ", None, "Re ψ", None)
        self.argand.setAspectLocked(True)
        self.argand.setMouseEnabled(x=False, y=False)
        self.argand.setToolTip(ARGAND_TIP)
        self.strobe = _plot("Re(ψ e<sup>iθ</sup>)", None, "Mass", None)
        self.strobe.setMouseEnabled(x=False, y=False)
        self.strobe.setToolTip(STROBE_TIP)
        for p in (self.argand, self.strobe):
            plot_height(p, 220)
        shapes = QtWidgets.QWidget()
        sh = QtWidgets.QHBoxLayout(shapes)
        sh.setContentsMargins(0, 0, 0, 0)
        sh.addWidget(self.argand, 1)
        sh.addWidget(self.strobe, 1)
        self.table = QtWidgets.QTableWidget()
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.setToolTip(TABLE_TIP)
        self.readout = QtWidgets.QLabel()
        self.readout.setWordWrap(True)
        self.readout.setTextFormat(QtCore.Qt.TextFormat.RichText)
        center = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(center)
        cv.addWidget(shapes, 3)
        cv.addWidget(self.table, 2)
        cv.addWidget(self.readout)

        # --- right: the FRF, the free decay, the phase scatter, the notes
        self.frf_in = QtWidgets.QComboBox()
        self.frf_out = QtWidgets.QComboBox()
        for b in (self.frf_in, self.frf_out):
            b.setToolTip(TIPS["frf"])
        self.frf_mag = _plot("|x/F|", None, "Frequency", "Hz", log=True)
        self.frf_mag.setLogMode(x=True, y=True)
        self.frf_mag.getAxis("left").enableAutoSIPrefix(False)
        self.frf_mag.setToolTip(FRF_TIP)
        self.frf_legend = add_legend(self.frf_mag.getPlotItem(), offset=(-10, 10))
        self.frf_phase = _plot("Phase (°)", None, "Frequency", "Hz", log=True)
        self.frf_phase.setLogMode(x=True, y=False)
        self.frf_phase.getAxis("left").enableAutoSIPrefix(False)
        self.frf_phase.setXLink(self.frf_mag)
        self.frf_phase.setToolTip(FRF_TIP)
        plot_height(self.frf_mag, 180)
        plot_height(self.frf_phase, 110)
        self.frf_note = QtWidgets.QLabel()
        self.frf_note.setWordWrap(True)
        self.frf_note.setTextFormat(QtCore.Qt.TextFormat.RichText)
        frf_tab = QtWidgets.QWidget()
        fv = QtWidgets.QVBoxLayout(frf_tab)
        row = QtWidgets.QFormLayout()
        row.addRow("Force on:", group(self.frf_in, "response of", self.frf_out))
        fv.addLayout(row)
        fv.addWidget(self.frf_mag, 3)
        fv.addWidget(self.frf_phase, 2)
        fv.addWidget(self.frf_note)

        self.decay_plot = _plot("Released mode's q", None, "Time", "s")
        self.decay_plot.setToolTip(DECAY_TIP)
        self.decay_legend = add_legend(self.decay_plot.getPlotItem(), offset=(-10, 10))
        self.leak_plot = _plot("The others' q", None, "Time", "s")
        self.leak_plot.setToolTip(DECAY_TIP)
        self.leak_legend = add_legend(self.leak_plot.getPlotItem(), offset=(-10, 10))
        self.leak_plot.setXLink(self.decay_plot)
        plot_height(self.decay_plot, 150)
        plot_height(self.leak_plot, 150)
        self.decay_note = QtWidgets.QLabel()
        self.decay_note.setWordWrap(True)
        self.decay_note.setTextFormat(QtCore.Qt.TextFormat.RichText)
        decay_tab = QtWidgets.QWidget()
        dv = QtWidgets.QVBoxLayout(decay_tab)
        dv.addWidget(self.decay_plot, 1)
        dv.addWidget(self.leak_plot, 1)
        dv.addWidget(self.decay_note)

        self.scatter_plot = _plot("Angle from the best-fit line (°)", None, "Mode", None)
        self.scatter_plot.setMouseEnabled(x=False, y=False)
        self.scatter_plot.setToolTip(SCATTER_TIP)
        plot_height(self.scatter_plot, 200)
        self.scatter_note = QtWidgets.QLabel(
            "Each mass's phase from its mode's best-fit line; bars: the MPD. Real modes put every dot on 0.")
        self.scatter_note.setWordWrap(True)
        mute(self.scatter_note)
        scatter_tab = QtWidgets.QWidget()
        pv = QtWidgets.QVBoxLayout(scatter_tab)
        pv.addWidget(self.scatter_plot, 1)
        pv.addWidget(self.scatter_note)

        self.notes = QtWidgets.QTextBrowser()
        self.notes.setHtml(NOTES_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.frf_tab, self.decay_tab, self.scatter_tab = frf_tab, decay_tab, scatter_tab
        self.tabs.addTab(frf_tab, "FRF")
        self.tabs.addTab(decay_tab, "Free decay")
        self.tabs.addTab(scatter_tab, "Phase scatter")
        self.tabs.addTab(self.notes, "Notes")

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(left_scroll)
        splitter.addWidget(center)
        splitter.addWidget(self.tabs)
        splitter.setSizes([320, 760, 620])
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        # --- wiring
        self.make_proportional.clicked.connect(self._on_make_proportional)
        self.trial.currentIndexChanged.connect(self._on_trial)
        self.mode.currentIndexChanged.connect(self._on_mode)
        self.table.itemSelectionChanged.connect(self._on_table_selection)
        self.frf_in.currentIndexChanged.connect(lambda _: self._draw_frf())
        self.frf_out.currentIndexChanged.connect(lambda _: self._draw_frf())

    # ------------------------------------------------------------- inputs
    def set_system(self, system: ChainSystem, result: ModalResult | None = None) -> None:
        """New chain parameters: worked out again (now if shown)."""
        n_changed = self.system is None or system.n != self.system.n
        self.system = system
        self.result = result if result is not None else modal_analysis(system)
        self.dof.blockSignals(True)
        self.dof.setValue(system.n)
        self.dof.blockSignals(False)
        if n_changed:
            for box, default in ((self.frf_in, system.n - 1), (self.frf_out, 0)):
                box.blockSignals(True)
                box.clear()
                box.addItems([f"m{i + 1}" for i in range(system.n)])
                box.setCurrentIndex(default)
                box.blockSignals(False)
            self.mode.blockSignals(True)
            current = self.mode.currentIndex()
            self.mode.clear()
            self.mode.addItems([f"Mode {i + 1}" for i in range(system.n)])
            self.mode.setCurrentIndex(current if 0 <= current < system.n else 0)
            self.mode.blockSignals(False)
        if self.isVisible():
            self.recompute()
        else:
            self._dirty = True

    def load_trial(self, index: int) -> None:
        """Ask for chain `index` of TRIALS, and show its note."""
        _, system, note = TRIALS[index]
        self.trial_note.setText(note)
        self.trial_note.setVisible(True)
        self.system_requested.emit(system.copy())

    def _on_trial(self, index: int) -> None:
        if index <= 0:
            return
        self.trial.blockSignals(True)
        self.trial.setCurrentIndex(0)
        self.trial.blockSignals(False)
        self.load_trial(index - 1)

    def _on_make_proportional(self) -> None:
        if self.system is not None:
            self.trial_note.setVisible(False)
            self.system_requested.emit(made_proportional(self.system, self.result))

    def _on_mode(self, index: int) -> None:
        if index < 0 or not self.rows:
            return
        self.table.blockSignals(True)
        self.table.selectRow(index)
        self.table.blockSignals(False)
        self._draw_mode()
        self._draw_decay()

    def _on_table_selection(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if rows and rows[0].row() != self.mode.currentIndex():
            self.mode.setCurrentIndex(rows[0].row())

    def showEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().showEvent(event)
        if self._dirty:
            self.recompute()

    # ------------------------------------------------------------- computing
    def recompute(self) -> None:
        self._dirty = False
        if self.system is None:
            return
        self.rows = compare(self.system, self.result)
        self.redraw()

    def selected(self) -> int:
        return min(max(self.mode.currentIndex(), 0), len(self.rows) - 1)

    # ------------------------------------------------------------- drawing
    def apply_theme(self) -> None:
        if self.rows:
            self.redraw()

    def redraw(self) -> None:
        r = self.result
        if r.is_proportional:
            self.coupling.setText("Coupling index <b>0</b>: the damping is <b>proportional</b>, and every mode real.")
        else:
            self.coupling.setText(f"Coupling index <b>{r.coupling:.2f}</b>: the damping is <b>not proportional</b>, "
                                  "and the modes are complex.")
        self.make_proportional.setEnabled(not r.is_proportional)
        self._fill_table()
        self._draw_mode()
        self._draw_frf()
        self._draw_decay()
        self._draw_scatter()
        worst = max((c for c in self.rows if c.measures is not None), key=lambda c: c.measures.mpd, default=None)
        over = [c.index for c in self.rows if c.exact is None and c.fn_hz > 0]
        text = ""
        if worst is not None and worst.measures.mpd > 0.05:
            text = (f"The most complex: mode {worst.index}, MPC {worst.measures.mpc:.2f}, MPD "
                    f"{worst.measures.mpd:.3g}°.")
        elif worst is not None:
            text = "Every mode is real: the approximation is exact."
        if over:
            text += f" Overdamped (no oscillating pole): mode{'s' if len(over) > 1 else ''} " + \
                    ", ".join(map(str, over)) + "."
        self.readout.setText(text)

    def _fill_table(self) -> None:
        t = self.table
        t.blockSignals(True)
        t.clear()
        heads = ["Mode", "f exact (Hz)", "f approx", "ζ exact", "ζ approx", "ζ error", "MAC", "MPC", "MPD"]
        t.setColumnCount(len(heads))
        t.setHorizontalHeaderLabels(heads)
        t.setRowCount(len(self.rows))
        for i, c in enumerate(self.rows):
            cells = [f"{c.index}"]
            if c.exact is None:
                cells += ["overdamped", f"{c.fn_hz:.4g}", "–", f"{c.zeta_approx:.3g}", "–", "–", "–", "–"]
            else:
                m = c.measures
                cells += [f"{c.fn_exact:.4g}", f"{c.fn_hz:.4g}", f"{c.zeta_exact:.3g}", f"{c.zeta_approx:.3g}",
                          _percent(c.zeta_approx, c.zeta_exact), f"{c.mac:.3f}", f"{m.mpc:.3f}", f"{m.mpd:.2f}°"]
            for j, text in enumerate(cells):
                item = _cell(text)
                if j == 0:
                    item.setForeground(pg.mkColor(colors.mode[i % len(colors.mode)]))
                t.setItem(i, j, item)
        t.resizeColumnsToContents()
        t.selectRow(self.selected())
        t.blockSignals(False)

    def _draw_mode(self) -> None:
        if not self.rows:
            return
        c = self.rows[self.selected()]
        n = self.system.n
        a, s = self.argand, self.strobe
        a.clear()
        s.clear()
        if c.shape is None:
            self.mode_info.setText(f"Mode {c.index} has no oscillating pole: the damping has made it a pair of "
                                   "real roots.")
            return
        psi, m = c.shape, c.measures
        circle = np.exp(1j * np.linspace(0, 2 * np.pi, 97))
        a.plot(circle.real, circle.imag, pen=pg.mkPen(colors.faint, width=1, style=QtCore.Qt.PenStyle.DotLine))
        a.addItem(pg.InfiniteLine(0.0, angle=0, pen=pg.mkPen(colors.faint, width=1)))
        a.addItem(pg.InfiniteLine(0.0, angle=90, pen=pg.mkPen(colors.faint, width=1)))
        a.addItem(pg.InfiniteLine((0.0, 0.0), angle=math.degrees(m.angle),
                                  pen=pg.mkPen(colors.grey, width=1.5, style=QtCore.Qt.PenStyle.DashLine)))
        for i, z in enumerate(psi):
            a.plot([0.0, z.real], [0.0, z.imag], pen=pg.mkPen(colors.mass[i], width=2))
        a.addItem(pg.ScatterPlotItem(psi.real, psi.imag, size=8, pen=None,
                                     brush=[pg.mkBrush(colors.mass[i]) for i in range(n)]))
        a.setRange(xRange=(-1.15, 1.15), yRange=(-1.15, 1.15), padding=0)
        a.setTitle(f"Mode {c.index}: MPC {m.mpc:.3f}, MPD {m.mpd:.2f}°", size="9pt")

        xs = np.arange(n + 1)
        env = np.concatenate([[0.0], np.abs(psi)])
        for sign in (1, -1):
            s.plot(xs, sign * env, pen=pg.mkPen(colors.grey, width=1, style=QtCore.Qt.PenStyle.DashLine))
        colour = colors.mode[(c.index - 1) % len(colors.mode)]
        for k, snap in enumerate(strobe(psi, SNAPSHOTS)):
            pen = pg.mkPen(_alpha(colour, 255 if k == 0 else 110), width=2 if k == 0 else 1)
            s.plot(xs, np.concatenate([[0.0], snap]), pen=pen)
        s.addItem(pg.InfiniteLine(0.0, angle=0, pen=pg.mkPen(colors.faint, width=1)))
        s.setXRange(-0.25, n + 0.25, padding=0)
        s.setYRange(-1.1, 1.1, padding=0)
        s.getAxis("bottom").setTicks([[(0, "ground")] + [(i + 1, f"m{i + 1}") for i in range(n)]])
        s.setTitle("Half a cycle, 8 snapshots", size="9pt")
        lam = c.exact
        self.mode_info.setText(
            f"Exact: {c.fn_exact:.4g} Hz, ζ = {c.zeta_exact:.3g} (λ = {lam.real:.3g} ± {lam.imag:.4g}i). "
            f"Approximation: {c.fn_hz:.4g} Hz, ζ = {c.zeta_approx:.3g}.")

    def _draw_frf(self) -> None:
        if not self.rows:
            return
        inp, out = max(self.frf_in.currentIndex(), 0), max(self.frf_out.currentIndex(), 0)
        f = frf_grid(self.result)
        exact, approx = frf_pair(self.system, self.result, f, inp, out)
        p, q = self.frf_mag, self.frf_phase
        p.clear()
        q.clear()
        self.frf_legend.clear()
        e = p.plot(f, np.maximum(np.abs(exact), 1e-300), pen=pg.mkPen(colors.strong, width=2))
        ap = p.plot(f, np.maximum(np.abs(approx), 1e-300), pen=pg.mkPen(colors.fit, width=2,
                                                                         style=QtCore.Qt.PenStyle.DashLine))
        self.frf_legend.addItem(e, "Exact")
        self.frf_legend.addItem(ap, "Proportional approximation")
        q.plot(f, np.degrees(np.angle(exact)), pen=pg.mkPen(colors.strong, width=2))
        q.plot(f, np.degrees(np.angle(approx)), pen=pg.mkPen(colors.fit, width=2, style=QtCore.Qt.PenStyle.DashLine))
        q.setYRange(-185, 185, padding=0)
        p.setLabel("left", f"|x{out + 1}/F{inp + 1}| (m/N)")
        p.setXRange(math.log10(f[0]), math.log10(f[-1]), padding=0)
        peak = float(np.max(np.abs(exact)))
        p.setYRange(math.log10(peak) - 4, math.log10(peak) + 0.5, padding=0)
        err = np.abs(approx - exact) / np.max(np.abs(exact))
        self.frf_note.setText(
            f"Largest difference: <b>{100 * err.max():.3g} %</b> of the exact FRF's peak, at {f[np.argmax(err)]:.3g} Hz."
            if err.max() > 1e-9 else "The two are the same: the damping is proportional.")

    def _draw_decay(self) -> None:
        if not self.rows:
            return
        k = self.selected()
        p, q = self.decay_plot, self.leak_plot
        p.clear()
        q.clear()
        self.decay_legend.clear()
        self.leak_legend.clear()
        try:
            run = free_decay(self.system, self.result, k)
        except ValueError as e:
            self.decay_note.setText(f"Mode {k + 1}: {e}.")
            return
        colour = colors.mode[k % len(colors.mode)]
        c = p.plot(run.t, run.q[:, k], pen=pg.mkPen(colour, width=2))
        a = p.plot(run.t, run.approx, pen=pg.mkPen(colors.strong, width=1.5, style=QtCore.Qt.PenStyle.DashLine))
        self.decay_legend.addItem(c, f"q{k + 1}, exact")
        self.decay_legend.addItem(a, "Approximation")
        p.setYRange(-1.05, 1.05, padding=0)
        p.setXRange(0, run.t[-1], padding=0)
        top = max(float(run.leak.max()), 1e-3)
        for s in range(self.system.n):
            if s != k:
                curve = q.plot(run.t, run.q[:, s], pen=pg.mkPen(colors.mode[s % len(colors.mode)], width=1.5))
                self.leak_legend.addItem(curve, f"q{s + 1}")
        q.setYRange(-1.1 * top, 1.1 * top, padding=0)
        s = int(np.argmax(run.leak))
        self.decay_note.setText(
            f"Released in mode {k + 1}'s undamped shape: <b>{100 * run.leak[s]:.3g} %</b> of it reaches mode "
            f"{s + 1}." if run.leak[s] > 1e-9 else
            f"Released in mode {k + 1}'s undamped shape, it stays there: no other mode moves.")

    def _draw_scatter(self) -> None:
        p = self.scatter_plot
        p.clear()
        n = self.system.n
        for c in self.rows:
            if c.measures is None:
                continue
            dev = c.measures.deviation
            p.addItem(pg.ScatterPlotItem(np.full(n, float(c.index)), dev, size=8, pen=None,
                                         brush=[pg.mkBrush(colors.mass[i]) for i in range(n)]))
            for sign in (1, -1):
                p.plot([c.index - 0.3, c.index + 0.3], [sign * c.measures.mpd] * 2,
                       pen=pg.mkPen(colors.mode[(c.index - 1) % len(colors.mode)], width=2))
        p.addItem(pg.InfiniteLine(0.0, angle=0, pen=pg.mkPen(colors.faint, width=1)))
        p.setXRange(0.5, n + 0.5, padding=0)
        p.setYRange(-92, 92, padding=0)
        p.getAxis("bottom").setTicks([[(i + 1, str(i + 1)) for i in range(n)]])


def _alpha(color: str, alpha: int):
    c = pg.mkColor(color)
    c.setAlpha(alpha)
    return c
