"""Modal parameter extraction on the Virtual modal test page: controls, stabilization diagram, results.

The page runs the extraction (core.identification) on the measured FRF column
and shows the identified modes against the exact ones: natural frequency,
damping ratio and MAC, a MAC matrix, and for LSCF the stabilization diagram,
where poles can be picked by clicking.
"""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.identification import Method, ModeMatch, Stabilization, Stability
from .panels import spin
from .style import colors, text_on
from .theming import add_legend

METHOD_TIPS = {
    Method.PEAK: "<p>One mode per peak of the summed FRF magnitude. f<sub>n</sub> is the frequency "
    "line at the peak, ζ = (f<sub>2</sub> − f<sub>1</sub>)/(2f<sub>n</sub>) from where the peak falls "
    "to 1/√2 of its height, and the shape is the FRF of every response at the peak.</p>"
    "<p>Quick, and it shows why the line spacing matters: f<sub>n</sub> can only land on a line, and "
    "a peak between lines looks lower and wider (too much damping). Close or heavily damped modes "
    "merge into one peak.</p>",
    Method.CIRCLE: "<p>Near a resonance the mobility iωH traces a circle in the complex plane. The "
    "angle each measured point makes about the centre depends on ω<sub>n</sub> and ζ, so both are "
    "fitted to the angles, between the frequency lines. Each response's modal constant comes from a "
    "fit around the peak, with a constant for the other modes.</p>"
    "<p>More accurate than peak picking, but still one mode at a time: it needs a few lines across "
    "each peak and well-separated modes.</p>",
    Method.LSCF: "<p><b>LSCF</b> (least-squares complex frequency; here the single-reference form, which PolyMAX extends to several references) fits one rational "
    "fraction with a common denominator to every response at once, at every model order up to the "
    "maximum. The denominator's roots are the poles.</p><p>Physical poles come back at the same "
    "frequency and damping as the order rises; poles that only fit noise wander. The "
    "<b>stabilization diagram</b> plots them all: pick the stable columns (automatically, or by "
    "clicking).</p><p><b>LSFD</b> then fits the residues of the chosen poles, plus a constant and a "
    "1/ω² term for the modes outside the band, by linear least squares.</p>",
}


def colored(value: float, good: float, fair: float, text: str) -> tuple[str, str]:
    """(text, color) by |value| against two thresholds."""
    v = abs(value)
    return text, colors.good if v <= good else colors.fair if v <= fair else colors.poor


class ExtractionControls(QtWidgets.QGroupBox):
    """Method, fit band and options; `changed` asks for the extraction to run again."""

    changed = QtCore.Signal()
    auto_requested = QtCore.Signal()
    clear_requested = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("Modal parameters", parent)
        form = QtWidgets.QFormLayout(self)
        self.method = QtWidgets.QComboBox()
        for i, m in enumerate(Method):
            self.method.addItem(m.value, m)
            self.method.setItemData(i, METHOD_TIPS[m], QtCore.Qt.ItemDataRole.ToolTipRole)
        self.method.setCurrentIndex(list(Method).index(Method.LSCF))
        form.addRow("Method:", self.method)
        band_row = QtWidgets.QHBoxLayout()
        self.lo = spin(0.0, 1e4, 0.0, 3, " Hz")
        self.hi = spin(0.0, 1e4, 1.0, 3, " Hz")
        band_row.addWidget(self.lo)
        band_row.addWidget(QtWidgets.QLabel("to"))
        band_row.addWidget(self.hi)
        band_tip = ("Only the FRF lines in this band are fitted. Drag the green band on the FRF "
                    "plot, or type the limits.")
        for w in (self.lo, self.hi):
            w.setToolTip(band_tip)
        form.addRow("Fit band:", band_row)
        self.order = QtWidgets.QSpinBox()
        self.order.setRange(4, 80)
        self.order.setValue(30)
        self.order.setKeyboardTracking(False)
        self.order.setToolTip("Highest LSCF model order. Order p gives up to p poles, so a few times "
                              "the number of modes: the extra poles soak up noise and leakage.")
        self.order_label = QtWidgets.QLabel("Max model order:")
        form.addRow(self.order_label, self.order)
        self.auto = QtWidgets.QPushButton("Auto-select poles")
        self.auto.setToolTip("Pick one pole from each column of the diagram that stays stable over "
                             "at least 5 orders. Click poles in the Stabilization tab to add or remove them.")
        self.auto.clicked.connect(self.auto_requested)
        self.clear = QtWidgets.QPushButton("Clear poles")
        self.clear.setToolTip("Deselect every pole, then click poles in the Stabilization tab to add "
                              "them one at a time and watch each mode join the fitted FRF.")
        self.clear.clicked.connect(self.clear_requested)
        self.pole_row = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(self.pole_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.auto)
        row.addWidget(self.clear)
        form.addRow(self.pole_row)
        self.correct = QtWidgets.QCheckBox("Remove the exponential window's damping")
        self.correct.setChecked(True)
        self.correct.setToolTip("The exponential window moves every pole 1/τ<sub>w</sub> to the left. Moving the "
                                "identified poles back gives the structure's own damping.")
        form.addRow(self.correct)
        self.show_fit = QtWidgets.QCheckBox("Draw the fitted FRF")
        self.show_fit.setChecked(True)
        self.show_fit.setToolTip("The FRF rebuilt from the identified modes (and residuals), drawn "
                                 "over the fit band. Where it misses the measurement, the model does too.")
        form.addRow(self.show_fit)
        self.method.currentIndexChanged.connect(self._on_method)
        for w in (self.lo, self.hi):
            w.valueChanged.connect(self.changed)
        self.order.valueChanged.connect(self.changed)
        self.correct.toggled.connect(self.changed)
        self.show_fit.toggled.connect(self.changed)
        self._on_method(emit=False)

    @property
    def current(self) -> Method:
        return self.method.currentData()

    @property
    def band(self) -> tuple[float, float]:
        return self.lo.value(), self.hi.value()

    def set_band(self, lo: float, hi: float, top: float | None = None) -> None:
        """Set the fit band without asking for a new extraction (`top` limits both spins)."""
        for box, value in ((self.lo, lo), (self.hi, hi)):
            box.blockSignals(True)
            if top is not None:
                box.setMaximum(top)
            box.setValue(value)
            box.blockSignals(False)

    def set_window_correction(self, available: bool) -> None:
        self.correct.setVisible(available)

    def _on_method(self, *_, emit: bool = True) -> None:
        lscf = self.current is Method.LSCF
        for w in (self.order_label, self.order, self.pole_row):
            w.setVisible(lscf)
        if emit:
            self.changed.emit()


class StabilizationPlot(pg.PlotWidget):
    """Poles of every LSCF model order against frequency; click one to select or deselect it."""

    pole_clicked = QtCore.Signal(int, int)  # order, index among that order's poles

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setLabel("bottom", "Frequency", units="Hz")
        self.setLabel("left", "Model order")
        self.showGrid(x=True, y=True, alpha=0.2)
        self.setMouseEnabled(x=False, y=False)
        self.hideButtons()
        self.legend = add_legend(self, offset=(-5, -5))
        self.mif = self.plot()
        self.scatter = {
            Stability.NEW: pg.ScatterPlotItem(symbol="o", size=4, pen=None),
            Stability.FREQUENCY: pg.ScatterPlotItem(symbol="t", size=7, brush=None),
            Stability.STABLE: pg.ScatterPlotItem(symbol="o", size=7, pen=None),
        }
        names = {Stability.NEW: "new", Stability.FREQUENCY: "stable f", Stability.STABLE: "stable f and ζ"}
        for st, item in self.scatter.items():
            self.addItem(item)
            self.legend.addItem(item, names[st])
            item.sigClicked.connect(self._on_click)
        self.selected = pg.ScatterPlotItem(symbol="o", size=15, brush=None)
        # The ring sits over its pole and takes the click, so it deselects the pole.
        self.selected.sigClicked.connect(self._on_click)
        self.addItem(self.selected)
        self.legend.addItem(self.selected, "selected")
        self.lines: list[pg.InfiniteLine] = []
        self.message = pg.TextItem("", anchor=(0.5, 0.5))
        self.addItem(self.message)
        self.apply_theme()

    def apply_theme(self) -> None:
        """Pole markers and the indicator curve; the mode lines are drawn in set_data."""
        self.legend.setBrush(colors.legend_brush())
        self.mif.setPen(pg.mkPen(colors.grey, width=1))
        self.scatter[Stability.NEW].setBrush(pg.mkBrush(colors.stab_new))
        self.scatter[Stability.FREQUENCY].setPen(pg.mkPen(colors.stab_freq))
        self.scatter[Stability.STABLE].setBrush(pg.mkBrush(colors.stab_stable))
        self.selected.setPen(pg.mkPen(colors.force, width=2))
        self.message.setColor(colors.muted)

    def set_data(
        self,
        stab: Stabilization | None,
        selection: list[tuple[int, int]],
        freqs: np.ndarray,
        indicator: np.ndarray,
        exact_hz: list[float],
    ) -> None:
        for line in self.lines:
            self.removeItem(line)
        self.lines = []
        if stab is None or not stab.orders:
            for item in (*self.scatter.values(), self.selected):
                item.setData([], [])
            self.mif.setData([], [])
            self.message.setText("The stabilization diagram belongs to the LSCF method.")
            self.message.setPos(0.5, 0.5)
            self.setRange(xRange=(0, 1), yRange=(0, 1), padding=0)
            return
        self.message.setText("")
        top = max(stab.orders)
        for st, item in self.scatter.items():
            spots = [
                {"pos": (abs(l) / (2 * math.pi), order), "data": (order, i)}
                for order, poles, status in zip(stab.orders, stab.poles, stab.status)
                for i, (l, s) in enumerate(zip(poles, status))
                if s is st
            ]
            item.setData(spots)
        self.selected.setData([{"pos": (abs(stab.pole(o, i)) / (2 * math.pi), o), "data": (o, i)} for o, i in selection])
        lo, hi = stab.band
        if indicator.size:
            y = np.log10(np.maximum(indicator, 1e-300))
            span = y.max() - y.min()
            y = (y - y.min()) / (span if span > 0 else 1.0) * top
            self.mif.setData(freqs, y)
        for r, f in enumerate(exact_hz):
            if lo <= f <= hi:
                line = pg.InfiniteLine(pos=f, angle=90, pen=pg.mkPen(colors.mode[r % len(colors.mode)], width=1,
                                                                     style=QtCore.Qt.PenStyle.DotLine))
                self.addItem(line)
                self.lines.append(line)
        self.setRange(xRange=(lo, hi), yRange=(0, top + 1), padding=0.02)

    def _on_click(self, _item, points, _event=None) -> None:
        if len(points):
            order, i = points[0].data()
            self.pole_clicked.emit(int(order), int(i))


class MacPlot(pg.PlotWidget):
    """MAC of each identified shape (rows) with each exact mode shape (columns)."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMouseEnabled(x=False, y=False)
        self.hideButtons()
        self.setMenuEnabled(False)
        self.getViewBox().invertY(True)
        self.setLabel("bottom", "Exact mode")
        self.setLabel("left", "Identified")
        self.setTitle("MAC: identified vs exact shapes", size="9pt")
        self.image = pg.ImageItem()
        self.addItem(self.image)
        self.labels: list[pg.TextItem] = []
        self.apply_theme()

    def apply_theme(self) -> None:
        """The colour map, from the background at MAC 0 to colors.mac_high at 1; labels follow in set_matrix."""
        self.cmap = pg.ColorMap([0.0, 1.0], [pg.mkColor(colors.background), pg.mkColor(colors.mac_high)])
        self.image.setLookupTable(self.cmap.getLookupTable(nPts=256))

    def set_matrix(self, macs: np.ndarray, rows: list[str], cols: list[str]) -> None:
        for t in self.labels:
            self.removeItem(t)
        self.labels = []
        if macs.size == 0:
            self.image.clear()
            return
        self.image.setImage(macs.T, levels=(0.0, 1.0))
        self.image.setRect(QtCore.QRectF(0, 0, macs.shape[1], macs.shape[0]))
        for i in range(macs.shape[0]):
            for j in range(macs.shape[1]):
                v = macs[i, j]
                t = pg.TextItem(f"{v:.2f}", color=text_on(self.cmap.map(v, mode="qcolor").name()), anchor=(0.5, 0.5))
                t.setPos(j + 0.5, i + 0.5)
                self.addItem(t)
                self.labels.append(t)
        self.getAxis("bottom").setTicks([[(j + 0.5, c) for j, c in enumerate(cols)]])
        self.getAxis("left").setTicks([[(i + 0.5, r) for i, r in enumerate(rows)]])
        self.setRange(xRange=(0, macs.shape[1]), yRange=(0, macs.shape[0]), padding=0)


class ResultsView(QtWidgets.QWidget):
    """Identified vs exact: a table per mode, the MAC matrix and notes."""

    HEADERS = ["Mode", "fₙ exact", "fₙ found", "error", "ζ exact", "ζ found", "error", "MAC"]

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.table = QtWidgets.QTableWidget(0, len(self.HEADERS))
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        tips = [
            "Exact mode number; 'extra' is an identified mode that matches none (a noise or computational pole)",
            "|λ|/2π of the exact damped eigenvalue [Hz]",
            "|λ|/2π of the identified pole [Hz]",
            "Relative frequency error",
            "Exact damping ratio −Re λ/|λ|",
            "Identified damping ratio",
            "Relative damping error. Damping is always the hardest parameter to measure.",
            "Modal assurance criterion between the identified and exact complex shapes: 1 = same shape",
        ]
        for c, tip in enumerate(tips):
            self.table.horizontalHeaderItem(c).setToolTip(tip)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.notes = QtWidgets.QLabel()
        self.notes.setWordWrap(True)
        self.notes.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.mac = MacPlot()
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.table, 3)
        layout.addWidget(self.notes)
        layout.addWidget(self.mac, 2)

    def set_results(self, rows: list[ModeMatch], macs: np.ndarray, notes: str) -> None:
        self.table.setRowCount(len(rows))
        for r, m in enumerate(rows):
            cells: list[tuple[str, str | None]] = []
            color = colors.mode[(m.mode - 1) % len(colors.mode)] if m.mode else colors.muted
            cells.append((str(m.mode) if m.mode else "extra", color))
            if m.exact is not None:
                fe, ze = abs(m.exact) / (2 * math.pi), -m.exact.real / abs(m.exact)
                cells.append((f"{fe:.4g}", None))
            else:
                fe = ze = None
                cells.append(("—", None))
            ident = m.identified
            if ident is None:
                cells += [("missed", colors.poor), ("", None), (f"{ze:.4f}", None), ("", None), ("", None), ("", None)]
            else:
                cells.append((f"{ident.fn_hz:.4g}", None))
                cells.append(colored(100 * (ident.fn_hz / fe - 1), 0.5, 2.0, f"{100 * (ident.fn_hz / fe - 1):+.2f}%")
                             if fe else ("", None))
                cells.append((f"{ze:.4f}" if ze is not None else "—", None))
                cells.append((f"{ident.zeta:.4f}", None))
                cells.append(colored(100 * (ident.zeta / ze - 1), 5.0, 20.0, f"{100 * (ident.zeta / ze - 1):+.1f}%")
                             if ze else ("", None))
                cells.append(colored(1 - m.mac, 0.01, 0.1, f"{m.mac:.3f}") if m.mode else ("", None))
            for c, (text, col) in enumerate(cells):
                item = QtWidgets.QTableWidgetItem(text)
                item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                if col:
                    item.setForeground(pg.mkColor(col))
                self.table.setItem(r, c, item)
        self.notes.setText(notes)


EXTRACTION_THEORY_HTML = """
<h3>Modal parameter extraction</h3>
<p>The measured FRF column is only data. <b>Modal parameter extraction</b> (curve fitting) finds
the modal model behind it: for each mode a pole λ<sub>r</sub> (natural frequency and damping)
and residues R<sub>jr</sub> (the mode shape), with</p>
<p>&nbsp;&nbsp;H<sub>jk</sub>(ω) ≈ Σ<sub>r</sub> [ R<sub>jr</sub>/(iω − λ<sub>r</sub>) +
R<sub>jr</sub>*/(iω − λ<sub>r</sub>*) ] + U<sub>j</sub> − L<sub>j</sub>/ω²</p>
<p>For a force at mass k, R<sub>jr</sub> ∝ ψ<sub>jr</sub>ψ<sub>kr</sub>: one column of H gives
every mode's shape, as long as ψ<sub>kr</sub> ≠ 0 (the force is not at a node of that mode).
U and L, the <b>residuals</b>, stand in for the modes above and below the fitted band.</p>
<ul>
<li><b>Peak picking</b> treats each peak as a single-DOF resonance: f<sub>n</sub> at the peak,
ζ = (f<sub>2</sub> − f<sub>1</sub>)/(2f<sub>n</sub>) from the half-power points, the shape from
the FRF at the peak. It needs well-separated, lightly damped modes and a fine line spacing.</li>
<li><b>Circle fit</b>: for a single mode the mobility iωH (velocity per unit force) is exactly
a circle in the complex plane. How fast the
points move round it peaks at resonance, and the angles give ω<sub>n</sub> and ζ even when
ω<sub>n</sub> falls between two frequency lines. Still one mode at a time.</li>
<li><b>LSCF + LSFD</b> fit every mode and every response at once. The model order (number of
poles) is not known in advance, so the fit is repeated at every order and the poles are
plotted in a <b>stabilization diagram</b>. Physical modes form vertical columns of stable poles;
noise and leakage poles scatter. The residues of the chosen poles then follow from a linear
least-squares fit (LSFD).</li>
</ul>
<p>The <b>Modal parameters</b> tab compares the result with the exact modes: frequency and
damping errors, and the <b>MAC</b> (modal assurance criterion)
|ψ<sub>a</sub><sup>H</sup>ψ<sub>b</sub>|² / (|ψ<sub>a</sub>|²|ψ<sub>b</sub>|²) between shapes,
which is 1 for the same shape and 0 for orthogonal ones.</p>
<h3>Things to try</h3>
<ul>
<li>Peak picking with a short block: the frequency snaps to a line and ζ comes out too
high. Lengthen the block, or switch to the circle fit or LSCF.</li>
<li>Add noise and watch the stabilization diagram: the weak, well-damped high modes lose
their columns first. More averages bring them back.</li>
<li>Use the exponential window and untick <i>Remove the exponential window's damping</i>:
every ζ is too high by 1/(τ<sub>w</sub>ω<sub>n</sub>).</li>
<li>Put the force at m3 of the default chain (a node of mode 2, as in <i>Where to look</i>
above): mode 2 is missing from every FRF, so no method can find it.</li>
</ul>
"""
