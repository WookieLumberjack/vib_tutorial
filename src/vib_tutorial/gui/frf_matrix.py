"""FRF matrix page: every receptance H_jk, and how modes and inputs build it up.

The centre is an N x N grid of small plots, one per term of the receptance
matrix H(w) (row j = response x_j, column k = force at m_k). The controls
choose which modal terms are summed, and which forces act together; the
selected cell is drawn large on the right with its individual terms, their
sum, and the full solution.
"""

from __future__ import annotations

import enum
import math

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ..core import ChainSystem, ModalResult, ModalTerm, frf_matrix, modal_analysis, modal_frf_terms
from .plots import frequency_grid
from .style import MAX_DOF, colors
from .theming import add_legend, mute

# Colours: the full solution in colors.strong; the sum of the modal terms, the
# selected cells and the worst band in colors.force.
FREQ_POINTS = 600  # per curve; the grid draws up to N^2 (2 + modes) curves
WORST_FRACTION = 0.5  # the shaded band is where the difference is at least this share of its largest


class Quantity(enum.Enum):
    MAGNITUDE = "Magnitude |H|"
    PHASE = "Phase ∠H"
    REAL = "Real part Re H"
    IMAG = "Imaginary part Im H"

    @property
    def partner(self) -> Quantity:
        """The quantity drawn with this one in the large plot."""
        return {
            Quantity.MAGNITUDE: Quantity.PHASE,
            Quantity.PHASE: Quantity.MAGNITUDE,
            Quantity.REAL: Quantity.IMAG,
            Quantity.IMAG: Quantity.REAL,
        }[self]

    @property
    def axis_label(self) -> str:
        return {
            Quantity.MAGNITUDE: "|H|  [m/N]",
            Quantity.PHASE: "∠H  [deg]",
            Quantity.REAL: "Re H  [m/N]",
            Quantity.IMAG: "Im H  [m/N]",
        }[self]


class Expansion(enum.Enum):
    EXACT = "Exact (complex modes)"
    CLASSICAL = "Classical (real modes, ζ modal)"


def part(h: np.ndarray, q: Quantity) -> np.ndarray:
    """The plotted real quantity of complex FRF values (along axis 0)."""
    if q is Quantity.MAGNITUDE:
        m = np.abs(h)
        return np.where(m > 0, m, np.nan)  # log axis
    if q is Quantity.PHASE:
        # Unwrapped, so a phase lag that builds up over several resonances stays visible.
        return np.degrees(np.unwrap(np.angle(h), axis=0))
    return h.real.copy() if q is Quantity.REAL else h.imag.copy()


def y_range(q: Quantity, values: list[np.ndarray]) -> tuple[float, float]:
    """Shared y range (log10 units for magnitude) covering `values`."""
    finite = np.concatenate([v[np.isfinite(v)].ravel() for v in values])
    if q is Quantity.PHASE:
        if not finite.size:
            return -195.0, 195.0
        return 90.0 * math.floor(finite.min() / 90.0) - 15.0, 90.0 * math.ceil(finite.max() / 90.0) + 15.0
    if q is Quantity.MAGNITUDE:
        finite = finite[finite > 0]
        if not finite.size:
            return -6.0, 0.0
        return math.log10(finite.min()) - 0.3, math.log10(finite.max()) + 0.3
    m = float(np.abs(finite).max()) if finite.size else 1.0
    return -1.1 * m, 1.1 * m


def worst_band(f: np.ndarray, full: np.ndarray, partial: np.ndarray) -> tuple[float, float, float] | None:
    """(low, high, peak) frequencies of the band around the largest |partial - full|.

    The band extends from the peak for as long as the difference stays above
    WORST_FRACTION of its largest value. None if the two agree to rounding.
    """
    d = np.abs(partial - full)
    i = int(np.argmax(d))
    if d[i] <= 1e-9 * max(np.abs(full).max(), 1e-300):
        return None
    above = d >= WORST_FRACTION * d[i]
    lo = i
    while lo > 0 and above[lo - 1]:
        lo -= 1
    hi = i
    while hi < d.size - 1 and above[hi + 1]:
        hi += 1
    return float(f[max(lo - 1, 0)]), float(f[min(hi + 1, d.size - 1)]), float(f[i])


def set_phase_ticks(axis: pg.AxisItem, phase: bool) -> None:
    """Degrees on multiples of 90 (labels every 180 when the range is wide); otherwise automatic."""
    if phase:
        axis.setTickSpacing(180.0, 90.0)
    else:
        axis.setTickSpacing()


def log_ticks(lo: float, hi: float, mantissas: tuple[int, ...]) -> list[list[tuple[float, str]]]:
    """Labelled ticks at mantissa x 10^e and unlabelled minor ticks, in log10 view units."""
    major, minor = [], []
    for e in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1):
        for m in range(1, 10):
            v = m * 10.0**e
            if lo <= v <= hi:
                (major if m in mantissas else minor).append((math.log10(v), f"{v:g}" if m in mantissas else ""))
    return [major, minor]


def term_color(term: ModalTerm, roots_seen: int) -> str:
    if term.mode is not None:
        return colors.mode[(term.mode - 1) % len(colors.mode)]
    return colors.roots[roots_seen % len(colors.roots)]


def swatch(color: str) -> QtGui.QIcon:
    pix = QtGui.QPixmap(12, 12)
    pix.fill(QtGui.QColor(color))
    return QtGui.QIcon(pix)


class FrfGrid(pg.GraphicsLayoutWidget):
    """N x N small plots of H_jk: row j = response x_j, column k = force at m_k."""

    cell_clicked = QtCore.Signal(int, int, bool)  # output j, input k, add to inputs (Ctrl held)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.n = 0
        self.cells: list[list[pg.PlotItem]] = []
        self.full: list[list[pg.PlotDataItem]] = []
        self.total: list[list[pg.PlotDataItem]] = []
        self.terms: list[list[list[pg.PlotDataItem]]] = []
        self.left_axes: list[pg.AxisItem] = []
        self.bottom_axes: list[pg.AxisItem] = []
        self.corner: pg.LabelItem | None = None
        self.scene().sigMouseClicked.connect(self._on_click)

    def build(self, n: int, n_terms: int, force: bool = False) -> None:
        """Lay out the grid for n masses and n_terms modal terms per cell (again if `force`)."""
        if not force and n == self.n and self.terms and len(self.terms[0][0]) == n_terms:
            return
        self.clear()
        self.n = n
        font = QtGui.QFont()
        font.setPointSize(7)
        self.corner = pg.LabelItem("", size="8pt")
        self.addItem(self.corner, row=0, col=0, colspan=2)
        for k in range(n):
            self.addItem(pg.LabelItem(f"<b>F at m{k + 1}</b>", size="9pt", color=colors.mass[k]), row=0, col=k + 2)
        self.cells, self.full, self.total, self.terms, self.left_axes = [], [], [], [], []
        thin = pg.mkPen(colors.cell_border, width=1)
        for j in range(n):
            self.addItem(pg.LabelItem(f"<b>x{j + 1}</b>", size="9pt", color=colors.mass[j]), row=j + 1, col=0)
            cells, full, total, terms = [], [], [], []
            for k in range(n):
                p = pg.PlotItem()
                p.hideAxis("left")
                p.hideAxis("bottom")
                p.hideButtons()
                p.setMenuEnabled(False)
                p.setMouseEnabled(x=False, y=False)
                p.setLogMode(x=True, y=False)
                vb = p.getViewBox()
                vb.setBorder(thin)
                if j == k:
                    vb.setBackgroundColor(colors.diagonal_tint)
                if cells or self.cells:
                    p.setXLink(self.cells[0][0] if self.cells else cells[0])
                    p.setYLink(self.cells[0][0] if self.cells else cells[0])
                self.addItem(p, row=j + 1, col=k + 2)
                # 1 px pens: up to a few hundred curves are drawn at once.
                terms.append([p.plot(pen=pg.mkPen(colors.grey, width=1)) for _ in range(n_terms)])
                full.append(p.plot(pen=pg.mkPen(colors.strong, width=1)))
                total.append(p.plot(pen=pg.mkPen(colors.force, width=1, style=QtCore.Qt.PenStyle.DashLine)))
                cells.append(p)
            self.cells.append(cells)
            self.full.append(full)
            self.total.append(total)
            self.terms.append(terms)
            axis = pg.AxisItem("left", linkView=cells[0].getViewBox())
            axis.setStyle(tickFont=font, tickTextOffset=2)
            axis.setWidth(44)
            axis.enableAutoSIPrefix(False)
            self.addItem(axis, row=j + 1, col=1)
            self.left_axes.append(axis)
        self.bottom_axes = []
        for k in range(n):
            axis = pg.AxisItem("bottom", linkView=self.cells[0][k].getViewBox())
            axis.setStyle(tickFont=font, tickTextOffset=2)
            axis.setLogMode(x=True, y=False)
            self.addItem(axis, row=n + 1, col=k + 2)
            self.bottom_axes.append(axis)
        self.addItem(pg.LabelItem("Frequency [Hz]", size="8pt"), row=n + 2, col=2, colspan=n)

    def set_curves(
        self,
        f: np.ndarray,
        q: Quantity,
        full: np.ndarray,
        total: np.ndarray,
        terms: list[tuple[str, np.ndarray] | None],
    ) -> None:
        """Real (F, n, n) arrays of the plotted quantity; `terms` entries are (color, values) or None (hidden)."""
        log_y = q is Quantity.MAGNITUDE
        for row in self.cells:
            for p in row:
                p.setLogMode(x=True, y=log_y)
        for axis in self.left_axes:
            axis.setLogMode(x=False, y=log_y)
            set_phase_ticks(axis, q is Quantity.PHASE)
        ticks = log_ticks(f[0], f[-1], (1, 2, 5) if self.n <= 4 else (1,))
        for axis in self.bottom_axes:
            axis.setTicks(ticks)
        for j in range(self.n):
            for k in range(self.n):
                self.full[j][k].setData(f, full[:, j, k], connect="finite")
                self.total[j][k].setData(f, total[:, j, k], connect="finite")
                for curve, term in zip(self.terms[j][k], terms):
                    curve.setVisible(term is not None)
                    if term is not None:
                        color, values = term
                        curve.setPen(pg.mkPen(color, width=1))
                        curve.setData(f, values[:, j, k], connect="finite")
        lo, hi = y_range(q, [full, total])
        self.cells[0][0].setXRange(math.log10(f[0]), math.log10(f[-1]), padding=0)
        self.cells[0][0].setYRange(lo, hi, padding=0)
        self.corner.setText(f"<i>{q.axis_label}</i>", size="8pt")

    def set_selection(self, output: int, inputs: set[int]) -> None:
        thin = pg.mkPen(colors.cell_border, width=1)
        bold = pg.mkPen(colors.force, width=3)
        for j, row in enumerate(self.cells):
            for k, p in enumerate(row):
                p.getViewBox().setBorder(bold if j == output and k in inputs else thin)

    def _on_click(self, event) -> None:
        pos = event.scenePos()
        for j, row in enumerate(self.cells):
            for k, p in enumerate(row):
                if p.sceneBoundingRect().contains(pos):
                    add = bool(event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier)
                    self.cell_clicked.emit(j, k, add)
                    return


class FrfDetail(QtWidgets.QWidget):
    """The selected term large: its parts, their sum, and the full solution."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.header = QtWidgets.QLabel()
        self.header.setWordWrap(True)
        self.header.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.plots = pg.GraphicsLayoutWidget()
        self.top = self.plots.addPlot(row=0, col=0)
        self.bottom = self.plots.addPlot(row=1, col=0)
        self.bottom.setXLink(self.top)
        self.bottom.setLabel("bottom", "Frequency", units="Hz")
        for p in (self.top, self.bottom):
            p.showGrid(x=True, y=True, alpha=0.3)
            p.getAxis("left").enableAutoSIPrefix(False)
        self.legend = add_legend(self.top, offset=(-5, 5))
        self.plots.ci.layout.setRowStretchFactor(0, 3)
        self.plots.ci.layout.setRowStretchFactor(1, 2)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.header)
        layout.addWidget(self.plots, 1)

    def set_data(
        self,
        f: np.ndarray,
        q: Quantity,
        full: np.ndarray,
        total: np.ndarray,
        terms: list[tuple[str, str, np.ndarray]],
        lines: list[tuple[float, str]],
        header: str,
        worst: tuple[float, float, float] | None = None,
    ) -> None:
        """Complex (F,) curves; `terms` are (name, color, values); `lines` are (frequency, color).

        `worst` = (low, high, peak) frequencies of the band to shade, where the sum
        differs most from the full solution.
        """
        self.header.setText(header)
        first = q if q in (Quantity.MAGNITUDE, Quantity.REAL) else q.partner
        self.legend.clear()
        for p, pq in ((self.top, first), (self.bottom, first.partner)):
            p.clear()
            log_y = pq is Quantity.MAGNITUDE
            p.setLogMode(x=True, y=log_y)
            p.setLabel("left", pq.axis_label)
            set_phase_ticks(p.getAxis("left"), pq is Quantity.PHASE)
            for fn, color in lines:
                p.addItem(pg.InfiniteLine(
                    pos=math.log10(fn), angle=90,
                    pen=pg.mkPen(color, width=1, style=QtCore.Qt.PenStyle.DotLine),
                ))
            if worst is not None:
                lo_f, hi_f, peak_f = worst
                brush = pg.mkColor(colors.force)
                brush.setAlpha(40)
                band = pg.LinearRegionItem(
                    (math.log10(lo_f), math.log10(hi_f)), movable=False, brush=brush,
                    pen=pg.mkPen(None),
                )
                band.setZValue(-10)
                p.addItem(band)
                p.addItem(pg.InfiniteLine(
                    pos=math.log10(peak_f), angle=90,
                    pen=pg.mkPen(colors.force, width=1.5, style=QtCore.Qt.PenStyle.DashLine),
                ))
            for name, color, values in terms:
                p.plot(f, part(values, pq), pen=pg.mkPen(color, width=1.5), name=name, connect="finite")
            p.plot(f, part(full, pq), pen=pg.mkPen(colors.strong, width=2.5), name="Full solution", connect="finite")
            p.plot(f, part(total, pq), pen=pg.mkPen(colors.force, width=2, style=QtCore.Qt.PenStyle.DashLine),
                   name="Sum of the terms shown", connect="finite")
            lo, hi = y_range(pq, [part(full, pq), part(total, pq)])
            p.setYRange(lo, hi, padding=0)
        ticks = log_ticks(f[0], f[-1], (1, 2, 5))
        for p in (self.top, self.bottom):
            p.getAxis("bottom").setTicks(ticks)
        self.top.setXRange(math.log10(f[0]), math.log10(f[-1]), padding=0)


class FrfMatrixPage(QtWidgets.QWidget):
    """Controls on the left, the N x N grid in the centre, the selected term and theory on the right."""

    dof_requested = QtCore.Signal(int)  # the user changed N here; the main window owns the system
    edit_parameters = QtCore.Signal()  # the user wants the Simulation page's parameter panel

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.system: ChainSystem | None = None
        self.result: ModalResult | None = None
        self.freqs = np.zeros(0)
        self.H = np.zeros((0, 0, 0), dtype=complex)  # (F, n, n) full solution
        self.terms: list[ModalTerm] = []
        self.term_H = np.zeros((0, 0, 0, 0), dtype=complex)  # (R, F, n, n)
        self.static = None  # K^-1, the static compliance; None if the chain can move as a rigid body
        self.term_static = np.zeros((0, 0, 0))  # (R, n, n) each term at w = 0
        self.term_colors: list[str] = []
        self.expansion_note = ""
        self.output = 0
        self.inputs: set[int] = {0}
        self._dirty = False
        self._unchecked: set[str] = set()  # remembered by label across recomputes
        self._rebuild_grid = False  # the theme changed: build the grid again

        # --- controls
        box = QtWidgets.QGroupBox("FRF matrix H(ω) = (K − ω²M + iωC)⁻¹")
        form = QtWidgets.QFormLayout(box)
        self.dof = QtWidgets.QSpinBox()
        self.dof.setRange(1, MAX_DOF)
        self.dof.setToolTip("Same chain as the Simulation page; changing it here changes it there too.")
        self.dof.valueChanged.connect(self.dof_requested)
        form.addRow("Number of masses:", self.dof)
        self.quantity = QtWidgets.QComboBox()
        for q in Quantity:
            self.quantity.addItem(q.value, q)
        self.quantity.setToolTip(
            "<p>What each small plot shows. The large plot on the right shows the pair: "
            "magnitude with phase, or real with imaginary part.</p>"
            "<p>The modal terms add as complex numbers, so their <b>real and imaginary parts add "
            "directly</b>; their magnitudes and phases do not.</p>"
        )
        self.quantity.currentIndexChanged.connect(self.redraw)
        form.addRow("Show:", self.quantity)
        self.expansion = QtWidgets.QComboBox()
        for e in Expansion:
            self.expansion.addItem(e.value, e)
        self.expansion.setToolTip(
            "<p><b>Exact:</b> each eigenvalue pair of the state-space model contributes "
            "R/(iω − λ) + R*/(iω − λ*). The sum of all of them is the full solution for any "
            "damping.</p>"
            "<p><b>Classical:</b> each undamped mode contributes φφᵀ/(ω<sub>r</sub>² − ω² + "
            "2iζ<sub>r</sub>ω<sub>r</sub>ω). With non-proportional damping even the sum of all modes "
            "misses the full solution, because the coupling terms of ΦᵀCΦ are dropped.</p>"
        )
        self.expansion.currentIndexChanged.connect(self._recompute_terms)
        form.addRow("Modal terms:", self.expansion)
        link = QtWidgets.QLabel(
            "Masses, springs and dampers are edited on the <a href='#sim'>Simulation page</a>."
        )
        mute(link)
        link.setWordWrap(True)
        link.linkActivated.connect(lambda _: self.edit_parameters.emit())
        form.addRow(link)

        self.mode_box = QtWidgets.QGroupBox("Modes included in the sum")
        mv = QtWidgets.QVBoxLayout(self.mode_box)
        self.mode_list = QtWidgets.QVBoxLayout()
        mv.addLayout(self.mode_list)
        buttons = QtWidgets.QHBoxLayout()
        self.all_button = QtWidgets.QPushButton("All")
        self.none_button = QtWidgets.QPushButton("None")
        self.all_button.clicked.connect(lambda: self._check_all(True))
        self.none_button.clicked.connect(lambda: self._check_all(False))
        buttons.addWidget(self.all_button)
        buttons.addWidget(self.none_button)
        mv.addLayout(buttons)
        self.show_terms = QtWidgets.QCheckBox("Draw each mode's term in the grid too")
        self.show_terms.setToolTip("The large plot always draws them. In the grid they can crowd the small plots.")
        self.show_terms.toggled.connect(self.redraw)
        mv.addWidget(self.show_terms)
        self.mode_note = QtWidgets.QLabel()
        self.mode_note.setWordWrap(True)
        mute(self.mode_note)
        mv.addWidget(self.mode_note)
        self.mode_checks: list[QtWidgets.QCheckBox] = []

        io_box = QtWidgets.QGroupBox("Response and forces")
        iv = QtWidgets.QVBoxLayout(io_box)
        out_row = QtWidgets.QHBoxLayout()
        out_row.addWidget(QtWidgets.QLabel("Response of:"))
        self.output_combo = QtWidgets.QComboBox()
        self.output_combo.currentIndexChanged.connect(self._on_output)
        out_row.addWidget(self.output_combo, 1)
        iv.addLayout(out_row)
        iv.addWidget(QtWidgets.QLabel("Unit forces (in phase) at:"))
        self.input_grid = QtWidgets.QGridLayout()
        iv.addLayout(self.input_grid)
        self.input_checks: list[QtWidgets.QCheckBox] = []
        io_note = QtWidgets.QLabel(
            "Click a plot to select it. Ctrl+click more plots in the same row, or tick more "
            "forces, to apply several forces at once: x<sub>j</sub> = Σ<sub>k</sub> H<sub>jk</sub>F<sub>k</sub>, "
            "the sum along the row."
        )
        io_note.setWordWrap(True)
        mute(io_note)
        iv.addWidget(io_note)

        for combo in (self.quantity, self.expansion, self.output_combo):
            combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(10)
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.addWidget(box)
        lv.addWidget(self.mode_box)
        lv.addWidget(io_box)
        lv.addStretch(1)
        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidget(left)
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(340)
        left_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # --- centre: the grid
        self.grid = FrfGrid()
        self.grid.cell_clicked.connect(self._on_cell)
        self.grid_title = QtWidgets.QLabel()
        self.grid_title.setWordWrap(True)
        centre = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(centre)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.addWidget(self.grid_title)
        cv.addWidget(self.grid, 1)

        # --- right: the selected term, and theory
        self.detail = FrfDetail()
        self.theory = QtWidgets.QTextBrowser()
        self.theory.setHtml(THEORY_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.detail, "Selected")
        self.tabs.addTab(self.theory, "Theory")

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(left_scroll)
        splitter.addWidget(centre)
        splitter.addWidget(self.tabs)
        splitter.setSizes([350, 800, 550])
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

    def apply_theme(self) -> None:
        """Swatches now; the grid, its curves and the large plot are redrawn when the page shows."""
        for k, c in enumerate(self.input_checks):
            c.setIcon(swatch(colors.mass[k]))
        self._rebuild_grid = True
        if self.system is not None:
            self._dirty = True
            if self.isVisible():
                self.refresh()

    # ------------------------------------------------------------ inputs
    def set_system(self, system: ChainSystem, result: ModalResult | None = None) -> None:
        """New chain parameters. Recomputed now if visible, else when the page is shown."""
        self.system = system
        self.result = result
        self._dirty = True
        if self.isVisible():
            self.refresh()

    def showEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().showEvent(event)
        if self._dirty:
            self.refresh()

    def _on_cell(self, j: int, k: int, add: bool) -> None:
        if add and j == self.output:
            inputs = self.inputs ^ {k}
            self.inputs = inputs or {k}
        else:
            self.output, self.inputs = j, {k}
        self._sync_io()
        self.redraw()

    def _on_output(self, j: int) -> None:
        if j >= 0 and j != self.output:
            self.output = j
            self.redraw()

    def _on_input_toggled(self) -> None:
        inputs = {k for k, c in enumerate(self.input_checks) if c.isChecked()}
        if not inputs:  # keep at least one force
            self._sync_io()
            return
        self.inputs = inputs
        self.redraw()

    def _on_mode_toggled(self) -> None:
        self._unchecked = {t.label for t, c in zip(self.terms, self.mode_checks) if not c.isChecked()}
        self.redraw()

    def _check_all(self, on: bool) -> None:
        for c in self.mode_checks:
            c.blockSignals(True)
            c.setChecked(on)
            c.blockSignals(False)
        self._on_mode_toggled()

    # ----------------------------------------------------------- compute
    def refresh(self) -> None:
        if self.system is None:
            return
        self._dirty = False
        system = self.system
        n = system.n
        if self.result is None:
            self.result = modal_analysis(system)
        self.dof.blockSignals(True)
        self.dof.setValue(n)
        self.dof.blockSignals(False)
        if len(self.input_checks) != n:
            self._build_io(n)
        self.freqs = frequency_grid(self.result, FREQ_POINTS)
        self.H = frf_matrix(system, self.freqs)
        K = system.matrices()[2]
        self.static = np.linalg.inv(K) if np.linalg.cond(K) < 1e12 else None
        self._recompute_terms()

    def _build_io(self, n: int) -> None:
        self.output = n - 1
        self.inputs = {n - 1}
        self.output_combo.blockSignals(True)
        self.output_combo.clear()
        self.output_combo.addItems([f"m{j + 1}  (x{j + 1})" for j in range(n)])
        self.output_combo.blockSignals(False)
        for c in self.input_checks:
            self.input_grid.removeWidget(c)
            c.setParent(None)
            c.deleteLater()
        self.input_checks = []
        for k in range(n):
            c = QtWidgets.QCheckBox(f"m{k + 1}")
            c.setIcon(swatch(colors.mass[k]))
            c.toggled.connect(self._on_input_toggled)
            self.input_grid.addWidget(c, k // 4, k % 4)
            self.input_checks.append(c)
        self._sync_io()

    def _sync_io(self) -> None:
        self.output_combo.blockSignals(True)
        self.output_combo.setCurrentIndex(self.output)
        self.output_combo.blockSignals(False)
        for k, c in enumerate(self.input_checks):
            c.blockSignals(True)
            c.setChecked(k in self.inputs)
            c.blockSignals(False)

    def _recompute_terms(self) -> None:
        if self.system is None or self.result is None or not self.freqs.size:
            return
        exact = self.expansion.currentData() is Expansion.EXACT
        self.expansion_note = ""
        try:
            terms = modal_frf_terms(self.system, self.result, exact)
        except ValueError:
            terms = modal_frf_terms(self.system, self.result, exact=False)
            self.expansion_note = (
                "<b>The exact expansion is not available:</b> the chain has a repeated eigenvalue "
                "(a free, unsprung chain moves as a rigid body), which has no pole-residue form. "
                "Showing the classical expansion instead."
            )
        self.terms = terms
        self.term_H = np.stack([t.evaluate(self.freqs) for t in terms])
        with np.errstate(divide="ignore", invalid="ignore"):
            # Imaginary parts cancel at w = 0 (a pair's two terms are conjugates).
            self.term_static = np.stack([t.evaluate(np.zeros(1))[0].real for t in terms])
        self.term_colors = []
        roots = 0
        for t in terms:
            self.term_colors.append(term_color(t, roots))
            roots += t.mode is None
        self._build_mode_checks()
        self.grid.build(self.system.n, len(terms), force=self._rebuild_grid)
        self._rebuild_grid = False
        self.redraw()

    def _build_mode_checks(self) -> None:
        for c in self.mode_checks:
            self.mode_list.removeWidget(c)
            c.setParent(None)
            c.deleteLater()
        self.mode_checks = []
        for t, color in zip(self.terms, self.term_colors):
            detail = f"{t.fn_hz:.3g} Hz" if t.fn_hz > 0 else "non-oscillatory"
            if math.isfinite(t.zeta):
                detail += f", ζ {t.zeta:.3g}"
            c = QtWidgets.QCheckBox(f"{t.label}   ({detail})")
            c.setIcon(swatch(color))
            c.setChecked(t.label not in self._unchecked)
            c.toggled.connect(self._on_mode_toggled)
            self.mode_list.addWidget(c)
            self.mode_checks.append(c)

    def selected_terms(self) -> np.ndarray:
        return np.array([c.isChecked() for c in self.mode_checks], dtype=bool)

    def redraw(self) -> None:
        if self.system is None or not self.term_H.size:
            return
        n = self.system.n
        q = self.quantity.currentData()
        sel = self.selected_terms()
        total = self.term_H[sel].sum(axis=0) if sel.any() else np.zeros_like(self.H)
        grid_terms = [
            (color, part(h, q)) if (on and self.show_terms.isChecked()) else None
            for h, color, on in zip(self.term_H, self.term_colors, sel)
        ]
        self.grid.set_curves(self.freqs, q, part(self.H, q), part(total, q), grid_terms)
        self.grid.set_selection(self.output, self.inputs)
        self.grid_title.setText(self._grid_title(n, sel))

        j, inputs = self.output, sorted(self.inputs)
        full = self.H[:, j, inputs].sum(axis=1)
        partial = total[:, j, inputs].sum(axis=1)
        if len(inputs) == 1:
            k = inputs[0]
            parts = [
                (t.label, color, h[:, j, k])
                for t, h, color, on in zip(self.terms, self.term_H, self.term_colors, sel)
                if on
            ]
        else:
            parts = [(f"H{j + 1}{k + 1}: force at m{k + 1}", colors.mass[k], total[:, j, k]) for k in inputs]
        lines = [(t.fn_hz, color) for t, color in zip(self.terms, self.term_colors) if t.fn_hz >= self.freqs[0]]
        worst = worst_band(self.freqs, full, partial) if sel.any() else None
        header = self._detail_header(j, inputs, sel, full, partial, worst)
        self.detail.set_data(self.freqs, q, full, partial, parts, lines, header, worst)

    # -------------------------------------------------------------- text
    def _grid_title(self, n: int, sel: np.ndarray) -> str:
        used = int(sel.sum())
        return (
            f"<b>All {n}×{n} = {n * n} receptances H<sub>jk</sub> = x<sub>j</sub>/F<sub>k</sub></b> "
            f"(row: response, column: force). <span style='color:{colors.strong}'>{self._strong_name()}: "
            f"full solution</span>; <span style='color:{colors.force}'>dashed red: sum of the {used} of {len(sel)} modal "
            "terms ticked</span>. Shaded: driving point (j = k). The matrix is symmetric, "
            "H<sub>jk</sub> = H<sub>kj</sub>."
        )

    @staticmethod
    def _strong_name() -> str:
        return "White" if colors.dark else "Black"

    def _detail_header(
        self,
        j: int,
        inputs: list[int],
        sel: np.ndarray,
        full: np.ndarray,
        partial: np.ndarray,
        worst: tuple[float, float, float] | None,
    ) -> str:
        if len(inputs) == 1:
            k = inputs[0]
            kind = "driving point" if j == k else "transfer"
            title = (f"<b>H<sub>{j + 1}{k + 1}</sub> = x<sub>{j + 1}</sub> / F<sub>{k + 1}</sub></b> "
                     f"({kind}): response of m{j + 1} to a force at m{k + 1}.")
            parts = (f"Coloured: each ticked mode's term. <span style='color:{colors.force}'>Dashed red</span> = "
                     f"Σ over the ticked modes r of mode r's ({j + 1},{k + 1}) entry only; the other cells "
                     "are responses to other forces.")
            zeros = (j if j <= k else k) + (self.system.n - 1 - max(j, k))
            parts += (" Undamped, it has no antiresonances (see Theory)." if zeros == 0 else
                      f" Undamped, it has {zeros} antiresonance{'s' if zeros > 1 else ''}"
                      f"{', counting any that coincide' if zeros > 1 else ''} (see Theory).")
        else:
            names = ", ".join(f"m{k + 1}" for k in inputs)
            terms = " + ".join(f"H<sub>{j + 1}{k + 1}</sub>" for k in inputs)
            title = (f"<b>x<sub>{j + 1}</sub> for unit in-phase forces at {names}</b> = {terms}.")
            cols = ", ".join(f"({j + 1},{k + 1})" for k in inputs)
            parts = ("Coloured: each force's term (from the ticked modes). "
                     f"<span style='color:{colors.force}'>Dashed red</span> = Σ over these forces and the "
                     f"ticked modes r of mode r's entries {cols}: the sum along row {j + 1}.")
        err = float(np.abs(partial - full).max() / max(np.abs(full).max(), 1e-300))
        color = colors.good if err < 1e-6 else colors.fair if err < 0.05 else colors.poor
        amount = "none (equal to rounding error)" if err < 1e-9 else f"{100 * err:.3g}%"
        verdict = (f"Largest difference from the full solution: <b style='color:{color}'>"
                   f"{amount}</b> of the peak")
        if worst is not None:
            verdict += (f", at <b>{worst[2]:.3g} Hz</b> (dashed red line). <span style='color:{colors.force}'>"
                        f"Shaded: {worst[0]:.3g}–{worst[1]:.3g} Hz</span>, where the difference is at "
                        f"least {WORST_FRACTION:.0%} of that.")
        else:
            verdict += "."
        if int(sel.sum()) == 0:
            verdict = "No modes ticked: the sum is zero."
        note = f"<br><span style='color:{colors.muted}'>{self.expansion_note}</span>" if self.expansion_note else ""
        return f"{title} {parts}<br>{verdict}<br>{self._static_text(j, inputs, sel)}{note}"

    def _static_text(self, j: int, inputs: list[int], sel: np.ndarray) -> str:
        """Static compliance (w = 0): the ticked modes' sum vs K^-1, and each mode's share."""
        if self.static is None:
            return ("<b>Static compliance</b> (ω = 0): none. K is singular, so the chain can move "
                    "as a rigid body and a steady force has no static deflection.")
        full = float(self.static[j, inputs].sum())
        each = self.term_static[:, j, inputs].sum(axis=1)
        partial = float(each[sel].sum())
        err = (partial - full) / full if full else 0.0
        color = colors.good if abs(err) < 1e-6 else colors.fair if abs(err) < 0.05 else colors.poor
        amount = "0% (exact)" if abs(err) < 1e-9 else f"{100 * err:+.3g}%"
        shares = []
        for t, c, value, on in zip(self.terms, self.term_colors, each, sel):
            share = f"{100 * value / full:.3g}%" if full and np.isfinite(value) else "—"
            style = "" if on else f" style='color:{colors.grey}'"
            shares.append(f"<span style='color:{c}'>■</span><span{style}>{t.label} {share}</span>")
        return (
            f"<b>Static compliance</b> (ω = 0): K<sup>−1</sup> gives {full:.4g} m/N; the ticked "
            f"modes give {partial:.4g} m/N, error <b style='color:{color}'>{amount}</b>. "
            f"<span style='color:{colors.muted}'>Each mode's share (grey: not ticked):</span> {', '.join(shares)}."
        )


THEORY_HTML = """
<p><i>Before this page: <b>Start with one mass</b>, <b>Undamped normal modes</b> and
<b>Proportional damping</b> in the Simulation page's Background tab.</i></p>
<h3>The FRF (receptance) matrix</h3>
<p>For a harmonic force f(t) = F e<sup>iωt</sup> the steady response is x(t) = X e<sup>iωt</sup> with
<b>(K − ω²M + iωC) X = F</b>, so</p>
<p>&nbsp;&nbsp;<b>X = H(ω) F</b>, &nbsp;&nbsp; H(ω) = (K − ω²M + iωC)<sup>−1</sup></p>
<p>H is an N × N matrix at every frequency. Its term H<sub>jk</sub> is the displacement of mass j
per unit force at mass k. Column k is the response of every mass to a force at m<sub>k</sub> (the
Frequency response tab on the Simulation page shows one column). Row j is how every force moves
mass j.</p>
<ul>
<li><b>Driving point</b> (diagonal, j = k): response where the force acts.</li>
<li><b>Transfer</b> (off-diagonal, j ≠ k): response somewhere else.</li>
<li><b>Reciprocity:</b> M, C and K are symmetric, so H is too: H<sub>jk</sub> = H<sub>kj</sub>.
Pushing at m<sub>k</sub> and measuring at m<sub>j</sub> gives exactly the same FRF as the
other way round (Maxwell–Betti).</li>
</ul>

<h3>Three forms of the same FRF</h3>
<p>H here is the <b>receptance</b> (or compliance, dynamic flexibility): displacement per unit
force, in m/N. The same information is often given per unit force as velocity or acceleration,
which for harmonic motion only multiplies H by iω or −ω²:</p>
<table border="1" cellspacing="0" cellpadding="3">
<tr><th>Form</th><th>Response</th><th>In terms of H</th><th>Units</th><th>Below / above the modes (driving point)</th></tr>
<tr><td>Receptance</td><td>displacement</td><td>H</td><td>m/N</td><td>flat (static compliance) / falls as 1/ω²</td></tr>
<tr><td>Mobility</td><td>velocity</td><td>iωH</td><td>m/(N·s)</td><td>rises as ω / falls as 1/ω</td></tr>
<tr><td>Accelerance (inertance)</td><td>acceleration</td><td>−ω²H</td><td>1/kg</td><td>rises as ω² / flat at 1/m</td></tr>
</table>
<p>The poles, mode shapes and damping are the same in all three; only the weighting of low and
high frequencies changes. Tests usually measure acceleration, so a measured FRF is most often an
accelerance (see the Virtual modal test page). Plotting Im H against Re H (a <i>Nyquist</i> plot)
turns each lightly damped resonance into a near-circle; for mobility and viscous damping it is
exactly a circle, which the circle fit uses. Transfer FRFs (j ≠ k) fall faster above the modes.</p>

<h3>Several forces at once</h3>
<p>Because the system is linear, the response to several forces is the sum of the responses to
each: x<sub>j</sub> = Σ<sub>k</sub> H<sub>jk</sub> F<sub>k</sub>, a sum along row j. Tick
several forces (or Ctrl+click plots in one row) to see the terms and their sum. The terms are
complex, so where they are out of phase they cancel, and the sum can be <i>smaller</i> than
either term alone.</p>

<h3>Every term is a sum of modes</h3>
<p>With mass-normalized undamped mode shapes φ<sub>r</sub> and proportional damping,</p>
<p>&nbsp;&nbsp;<b>H<sub>jk</sub>(ω) = Σ<sub>r</sub> φ<sub>jr</sub> φ<sub>kr</sub> /
(ω<sub>r</sub>² − ω² + 2iζ<sub>r</sub>ω<sub>r</sub>ω)</b></p>
<p>Each mode is a single-DOF resonance, scaled by the <b>modal constant</b>
φ<sub>jr</sub>φ<sub>kr</sub>: how much mode r is excited at k times how much it shows at j.
In matrix form H = Φ diag(1/(ω<sub>r</sub>² − ω² + 2iζ<sub>r</sub>ω<sub>r</sub>ω)) Φ<sup>T</sup>:
every term of the matrix is built from the same N resonances, only weighted differently.</p>
<ul>
<li>Near ω<sub>r</sub>, mode r's term dominates, so every H<sub>jk</sub> peaks there, unless
m<sub>j</sub> or m<sub>k</sub> sits at a node of that mode (φ = 0).</li>
<li>Below its resonance a mode's term acts like a spring (in phase with the force, or exactly
opposite if its modal constant is negative), above it like a mass (180° further behind). Between two resonances the lower mode is already mass-like and the
upper still spring-like.</li>
<li>If both modal constants have the <b>same sign</b>, the two terms are in anti-phase there and
cancel at some frequency: an <b>antiresonance</b> (a sharp dip). With <b>opposite signs</b> they
add, and the curve just passes through a minimum. (This looks at the two neighbouring modes
only; the others shift the dip, and a large contribution from them can remove it.)</li>
<li>At a driving point every modal constant φ<sub>jr</sub>² is positive, so there is an
antiresonance between every pair of resonances: peaks and dips alternate, and (lightly damped)
the phase swings down by 180° at each resonance and back up at each antiresonance.</li>
<li>For this chain, the undamped H<sub>jk</sub> (j ≤ k) has (j − 1) + (N − k)
antiresonances, one for each mass outside the stretch of chain from m<sub>j</sub> to m<sub>k</sub>
(two can fall at the same frequency, e.g. in a uniform chain). They are exactly the natural
frequencies of the two pieces of chain outside that stretch, with m<sub>j</sub> and
m<sub>k</sub> held still: at those frequencies the outer pieces can vibrate on their own and
absorb the force, as a vibration absorber does. In the undamped uniform 4-mass chain,
H<sub>22</sub>'s dips are at 1.97 and 5.15 Hz (m3 and m4 with m2 held) and 4.50 Hz (m1
between the ground and m2). The end-to-end term
H<sub>1N</sub> has none, so its phase keeps falling by 180° at every resonance.</li>
</ul>

<h3>Truncation</h3>
<p>Untick modes to see a <b>truncated modal model</b>, which is what you get from a finite-element
model with only its lowest modes kept, or from a test that measured only some modes. Near the kept
modes' peaks the sum is close. Elsewhere, especially at antiresonances and at low frequency, it
misses. The missing high modes act almost like springs there (their <i>residual flexibility</i>
Σ φ<sub>jr</sub>φ<sub>kr</sub>/ω<sub>r</sub>²), and leaving them out shifts the antiresonances.
Compare the <b>real part</b>: the terms add directly, so you can see the offset left by a
missing mode.</p>
<p>Leaving out a <i>low</i> mode does the opposite. Well above its frequency a mode's term
is −φ<sub>jr</sub>φ<sub>kr</sub>/ω², mass-like, so the missing low modes leave a
<b>residual mass</b> (residual inertia) term that grows towards low frequency as 1/ω². Untick
mode 1 and the sum is wrong by about that amount above it. A model or test band that starts
above the lowest modes needs both corrections, H ≈ Σ<sub>kept</sub> + U − L/ω² (the
<i>residuals</i> fitted on the Virtual modal test page): U for the modes above, L for those
below.</p>
<p>At ω = 0 the sum of every mode is the <b>static compliance</b> K<sup>−1</sup> =
Σ<sub>r</sub> φ<sub>r</sub>φ<sub>r</sub><sup>T</sup>/ω<sub>r</sub>² (damping plays no part
there, so even the classical sum is exact). Each mode's share of it is
φ<sub>jr</sub>φ<sub>kr</sub>/ω<sub>r</sub>² divided by the total, which is why the high modes
matter so little statically: their share falls as 1/ω<sub>r</sub>². The header of the large plot
lists every mode's share and the error of the ticked ones. That error is the residual
flexibility that a truncated model leaves out.</p>

<h3>Non-proportional damping</h3>
<p>The formula above keeps only the diagonal of Φ<sup>T</sup>CΦ. With non-proportional damping
(say, raise only c<sub>1</sub>), the <b>Classical</b> sum misses the full solution even with
every mode ticked. The <b>Exact</b> expansion uses the complex modes of the state-space form
instead: each eigenvalue pair contributes</p>
<p>&nbsp;&nbsp;R<sub>r</sub>/(iω − λ<sub>r</sub>) + R<sub>r</sub>*/(iω − λ<sub>r</sub>*)</p>
<p>with complex residue matrices R<sub>r</sub> (from the right and left eigenvectors), and their
sum is exactly H for any damping. A heavily damped mode can split into two real
(non-oscillatory) roots; each then contributes its own term.</p>
"""
