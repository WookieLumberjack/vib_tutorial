"""Substructuring page: Craig-Bampton component mode synthesis of the chain.

The chain from the Simulation page is cut at one interface mass into
substructures A and B. The boundary (master) DOFs are the interface and the
tip, where the force is applied. The page compares the reduced model's modes
and tip FRF with the full model's and walks through the matrices.
"""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core import (
    ChainSystem,
    ComponentMode,
    CraigBamptonModel,
    ModalResult,
    ModeComparison,
    compare_modes,
    component_modes,
    craig_bampton,
    frf,
    interior_counts,
    modal_analysis,
    reduced_frf,
)
from .animation import MASS_WIDTH, spring_path
from .cms_notes import THEORY_HTML, matrices_html
from .style import (
    FORCE_COLOR,
    MASS_COLORS,
    MAX_DOF,
    MODE_COLORS,
    STRUCTURE_COLOR,
    SUB_COLORS,
)

SUB_NAMES = "AB"
CB_PEN_STYLE = QtCore.Qt.PenStyle.DashLine


class SubstructureSchematic(pg.PlotWidget):
    """Static drawing of the chain with the substructures and master DOFs marked."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMenuEnabled(False)
        self.setMouseEnabled(x=False, y=False)
        self.hideButtons()
        self.hideAxis("left")
        self.hideAxis("bottom")
        self.setAspectLocked(True)
        self.setFixedHeight(190)
        self._items: list = []

    def _add(self, item):
        self.addItem(item)
        self._items.append(item)
        return item

    def set_model(self, n: int, model: CraigBamptonModel | None) -> None:
        for item in self._items:
            self.removeItem(item)
        self._items = []
        self._add(pg.PlotDataItem([0, 0], [-0.35, 0.35], pen=pg.mkPen(STRUCTURE_COLOR, width=4)))
        boundary = set() if model is None else {int(b) for b in model.boundary}
        owner = np.zeros(n, dtype=int)  # substructure owning each element
        if model is not None:
            for s, sub in enumerate(model.substructures):
                owner[sub.elements] = s
                left = 0.0 if s == 0 else sub.boundary[0] + 1.0
                right = sub.boundary[-1] + 1.0
                y0, y1 = (-0.62, 0.45) if s % 2 == 0 else (-0.55, 0.52)  # overlap at the interface
                x0, x1 = left - 0.3, right + 0.3
                band = QtWidgets.QGraphicsRectItem(QtCore.QRectF(x0, y0, x1 - x0, y1 - y0))
                color = pg.mkColor(SUB_COLORS[s])
                band.setPen(pg.mkPen(color, width=1.5, style=QtCore.Qt.PenStyle.DashLine))
                color.setAlpha(28)
                band.setBrush(pg.mkBrush(color))
                band.setZValue(-5)
                self._add(band)
                # A's label above its left edge, B's below its right edge, so a narrow
                # band's label runs across the other band instead of off the view.
                text = f"{sub.name}: {sub.ni} interior, keep {sub.n_kept} mode{'s' if sub.n_kept != 1 else ''}"
                above = s % 2 == 0
                label = pg.TextItem(text, color=SUB_COLORS[s], anchor=(0.0, 1.0) if above else (1.0, 0.0))
                label.setPos(x0 if above else x1, y1 if above else y0)
                self._add(label)

        right_prev = 0.0
        for i in range(n):
            xc = i + 1.0
            pen = pg.mkPen(SUB_COLORS[owner[i]] if model is not None else STRUCTURE_COLOR, width=2)
            self._add(pg.PlotDataItem(*spring_path(right_prev, xc - MASS_WIDTH / 2, 0.0), pen=pen))
            right_prev = xc + MASS_WIDTH / 2
            master = i in boundary
            rect = QtWidgets.QGraphicsRectItem(QtCore.QRectF(xc - MASS_WIDTH / 2, -0.2, MASS_WIDTH, 0.4))
            rect.setBrush(pg.mkBrush(MASS_COLORS[i]))
            rect.setPen(pg.mkPen(STRUCTURE_COLOR, width=4 if master else 1))
            rect.setZValue(5)
            self._add(rect)
            name = pg.TextItem(f"m{i + 1}", color="w", anchor=(0.5, 0.5))
            name.setPos(xc, 0)
            name.setZValue(6)
            self._add(name)
            if model is not None:
                role = pg.TextItem("boundary" if master else "interior", color="#000" if master else "#777",
                                   anchor=(0.5, 0.0))
                role.setPos(xc, -0.2)
                self._add(role)

        # Force at the tip.
        y = 0.28
        self._add(pg.PlotDataItem([n + MASS_WIDTH / 2, n + 0.6], [y, y], pen=pg.mkPen(FORCE_COLOR, width=3)))
        self._add(pg.ArrowItem(pos=(n + 0.6, y), angle=180, headLen=12, tipAngle=40, brush=FORCE_COLOR, pen=None))
        force = pg.TextItem("F", color=FORCE_COLOR, anchor=(0.0, 0.5))
        force.setPos(n + 0.65, y)
        self._add(force)
        self.setRange(QtCore.QRectF(-0.4, -0.95, n + 1.4, 1.75), padding=0)


class ComparisonTable(QtWidgets.QTableWidget):
    HEADERS = ["Mode", "fₙ true [Hz]", "fₙ CB [Hz]", "Error", "ζ true", "ζ CB", "MAC"]
    TIPS = [
        "Mode number, ordered by frequency",
        "Natural frequency of the full N-DOF model (Kφ = ω²Mφ)",
        "Natural frequency of the Craig–Bampton reduced model (K̂η = ω²M̂η). "
        "Blank when the reduced model has fewer DOFs than this mode number.",
        "(f_CB − f_true) / f_true. Never negative: CB is a Rayleigh–Ritz method, so it can only "
        "over-estimate stiffness.",
        "Exact damping ratio of the full model (from its damped eigenvalue λ: −Re λ / |λ|)",
        "Exact damping ratio of the reduced model M̂η̈ + Ĉη̇ + K̂η = 0, with Ĉ = TᵀCT. "
        "Coloured by its error relative to ζ true, which can have either sign. With stiffness-"
        "proportional damping it tracks the frequency error; with non-proportional damping it "
        "can be much worse.",
        "Modal assurance criterion between the true shape φ and the recovered CB shape Tη: "
        "(φᵀx)² / (φᵀφ · xᵀx). 1 = identical shape, 0 = unrelated.",
    ]

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(0, len(self.HEADERS), parent)
        self.setHorizontalHeaderLabels(self.HEADERS)
        for col, tip in enumerate(self.TIPS):
            self.horizontalHeaderItem(col).setToolTip(tip)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)

    def set_rows(self, comparisons) -> None:
        selected = self.currentRow() if self.selectedItems() else 0
        self.setRowCount(len(comparisons))
        for r, c in enumerate(comparisons):
            err = c.error
            cells = [
                str(c.index),
                f"{c.fn_true:.4g}",
                "—" if c.fn_cb is None else f"{c.fn_cb:.4g}",
                "not in model" if c.fn_cb is None else ("—" if err is None else f"{100 * err:+.3g}%"),
                zeta_text(c.zeta_true),
                "—" if c.fn_cb is None else zeta_text(c.zeta_cb),
                "—" if c.mac is None else f"{c.mac:.3f}",
            ]
            for col, text in enumerate(cells):
                item = QtWidgets.QTableWidgetItem(text)
                item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                if col == 0:
                    item.setForeground(pg.mkColor(MODE_COLORS[r % len(MODE_COLORS)]))
                elif col == 3 and err is not None:
                    item.setForeground(pg.mkColor(error_color(err)))
                elif col == 5 and c.zeta_error is not None:
                    item.setForeground(pg.mkColor(error_color(c.zeta_error)))
                    item.setToolTip(f"{100 * c.zeta_error:+.3g}% vs ζ true")
                elif c.fn_cb is None:
                    item.setForeground(pg.mkColor("#999"))
                self.setItem(r, col, item)
        height = self.horizontalHeader().height() + 2 * self.frameWidth()
        height += sum(self.rowHeight(r) for r in range(self.rowCount()))
        self.setFixedHeight(height)
        self.selectRow(min(selected, len(comparisons) - 1))


class ComponentTable(QtWidgets.QTableWidget):
    """Each substructure's own modes (boundary clamped) and the coupled mode each one becomes."""

    HEADERS = ["Mode", "fₙ alone [Hz]", "ζ alone", "Kept", "Becomes", "fₙ [Hz]", "Share"]
    TIPS = [
        "Fixed-interface mode r of substructure A or B",
        "Natural frequency of the substructure on its own, with its boundary masses held fixed: "
        "K_ii φ = ω² M_ii φ",
        "Exact damping ratio of the substructure on its own (boundary held), from the damped "
        "eigenvalues of M_ii, C_ii, K_ii",
        "Whether this mode is one of the fixed-interface modes kept in the reduced model",
        "The mode of the full, coupled chain that this component mode contributes most to",
        "Natural frequency of that coupled mode: compare with fₙ alone to see how coupling "
        "through the interface shifts it",
        "Fraction of the coupled mode's strain energy carried by this component mode. Each coupled "
        "mode is written in Craig–Bampton coordinates (fixed-interface amplitudes q plus boundary "
        "motion); component mode r holds ω_r² q_r² of its energy ω². Well below 100% means the "
        "coupled mode mixes several component modes and boundary motion.",
    ]

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(0, len(self.HEADERS), parent)
        self.setHorizontalHeaderLabels(self.HEADERS)
        for col, tip in enumerate(self.TIPS):
            self.horizontalHeaderItem(col).setToolTip(tip)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.setToolTip("Click a row to draw this component mode over the coupled mode it becomes "
                        "(Modes & FRF tab). Click it again to clear.")
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        # Clicking the selected row again clears the overlay.
        index = self.indexAt(event.position().toPoint())
        if index.isValid() and self.selectionModel().isRowSelected(index.row()):
            self.clearSelection()
            return
        super().mousePressEvent(event)

    def set_rows(self, modes: list[ComponentMode]) -> None:
        self.clearSelection()
        self.setRowCount(len(modes))
        for r, m in enumerate(modes):
            cells = [
                f"{m.substructure}{m.index}",
                f"{m.fn_hz:.4g}",
                zeta_text(m.zeta),
                "yes" if m.kept else "no",
                f"mode {m.closest}",
                f"{m.closest_fn_hz:.4g}",
                f"{100 * m.share:.0f}%",
            ]
            color = SUB_COLORS[SUB_NAMES.index(m.substructure)]
            for col, text in enumerate(cells):
                item = QtWidgets.QTableWidgetItem(text)
                item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                if col == 0:
                    item.setForeground(pg.mkColor(color))
                elif col == 4:
                    item.setForeground(pg.mkColor(MODE_COLORS[(m.closest - 1) % len(MODE_COLORS)]))
                elif col == 3 and not m.kept:
                    item.setForeground(pg.mkColor("#999"))
                self.setItem(r, col, item)
        height = self.horizontalHeader().height() + 2 * self.frameWidth()
        height += sum(self.rowHeight(r) for r in range(self.rowCount()))
        self.setFixedHeight(height)


def zeta_text(zeta: float | None) -> str:
    return "overdamped" if zeta is None else f"{zeta:.4f}"


def error_color(err: float) -> str:
    if abs(err) < 1e-3:
        return "#2a7d2a"
    return "#b07000" if abs(err) < 0.05 else "#c1121f"


class ComparisonPlots(pg.GraphicsLayoutWidget):
    """Selected mode shape (true vs CB) above the tip FRF (true vs CB)."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.shape = self.addPlot(row=0, col=0)
        self.shape.setLabel("bottom", "Position along chain (0 = ground)")
        self.shape.setLabel("left", "Normalized amplitude")
        self.shape.showGrid(x=True, y=True, alpha=0.3)
        self.shape.setMouseEnabled(x=False, y=False)
        # Headroom above +1 for a one-row legend, so it never covers a shape.
        self.shape.setYRange(-1.15, 1.5, padding=0)
        self.shape.getAxis("left").setTicks([[(v, f"{v:g}") for v in (-1, -0.5, 0, 0.5, 1)]])
        self.shape_legend = self.shape.addLegend(offset=(5, 2), colCount=3, brush=pg.mkBrush(255, 255, 255, 230))
        self.frf = self.addPlot(row=1, col=0)
        self.frf.setLogMode(x=True, y=True)
        self.frf.setLabel("left", "|X / F|  [m/N]")
        self.frf.setLabel("bottom", "Frequency", units="Hz")
        self.frf.showGrid(x=True, y=True, alpha=0.3)
        self.frf_legend = self.frf.addLegend(offset=(-5, 5), colCount=2)
        self.ci.layout.setRowStretchFactor(0, 2)
        self.ci.layout.setRowStretchFactor(1, 3)
        self._shape_items: list = []
        self._frf_items: list = []
        self.comparisons = []

    def set_comparisons(self, comparisons, model: CraigBamptonModel | None) -> None:
        self.comparisons = comparisons
        for item in self._shape_items:
            self.shape.removeItem(item)
        self._shape_items = []
        n = len(comparisons[0].shape_true) if comparisons else 0
        if model is not None:
            for b in model.boundary[:-1]:  # interface masses
                line = pg.InfiniteLine(pos=b + 1, angle=90, pen=pg.mkPen("#999", width=1, style=CB_PEN_STYLE),
                                       label="interface", labelOpts={"position": 0.08, "color": "#777"})
                self.shape.addItem(line)
                self._shape_items.append(line)
        self._true_curve = self.shape.plot(symbol="s", symbolSize=9, symbolPen=None)
        self._cb_curve = self.shape.plot(symbol="o", symbolSize=11, symbolBrush=None)
        self._component_curve = self.shape.plot(symbol="t", symbolSize=11)
        self._shape_items += [self._true_curve, self._cb_curve, self._component_curve]
        self.component: ComponentMode | None = None
        self.shape.setXRange(0, n, padding=0.05)
        self.shape.getAxis("bottom").setTicks([[(0, "ground")] + [(i, f"m{i}") for i in range(1, n + 1)]])

    def set_highlight(self, r: int) -> None:
        self.shape_legend.clear()
        if not (0 <= r < len(self.comparisons)):
            return
        c = self.comparisons[r]
        color = MODE_COLORS[r % len(MODE_COLORS)]
        xs = np.arange(c.shape_true.size + 1)
        self._true_curve.setData(xs, np.concatenate([[0.0], c.shape_true]))
        self._true_curve.setPen(pg.mkPen(color, width=3))
        self._true_curve.setSymbolBrush(color)
        self.shape_legend.addItem(self._true_curve, f"Mode {c.index} true: {c.fn_true:.4g} Hz")
        if c.shape_cb is None:
            self._cb_curve.setData([], [])
            self.shape.setTitle(f"Mode {c.index} is not in the reduced model", size="10pt")
        else:
            self._cb_curve.setData(xs, np.concatenate([[0.0], c.shape_cb]))
            self._cb_curve.setPen(pg.mkPen("#000", width=2, style=CB_PEN_STYLE))
            self._cb_curve.setSymbolPen(pg.mkPen("#000", width=2))
            self.shape_legend.addItem(self._cb_curve, f"Mode {c.index} CB: {c.fn_cb:.4g} Hz")
            self.shape.setTitle(f"Mode {c.index}: error {100 * c.error:+.3g}%, MAC {c.mac:.3f}", size="10pt")
        # Keep an overlay only while it belongs to this coupled mode.
        keep = self.component is not None and self.component.closest == c.index
        self.set_component(self.component if keep else None)

    def set_component(self, comp: ComponentMode | None) -> None:
        """Overlay one substructure's own (boundary-held) mode shape, or clear it."""
        self.component = comp
        if comp is None:
            self._component_curve.setData([], [])
            return
        xs, ys = comp.dofs + 1.0, comp.shape
        if comp.substructure == SUB_NAMES[0]:  # A starts at the ground
            xs, ys = np.concatenate([[0.0], xs]), np.concatenate([[0.0], ys])
        color = SUB_COLORS[SUB_NAMES.index(comp.substructure)]
        self._component_curve.setData(xs, ys)
        self._component_curve.setPen(pg.mkPen(color, width=3, style=QtCore.Qt.PenStyle.DotLine))
        self._component_curve.setSymbolBrush(color)
        self._component_curve.setSymbolPen(None)
        self.shape_legend.addItem(
            self._component_curve,
            f"{comp.substructure}{comp.index} alone, boundary held: {comp.fn_hz:.4g} Hz "
            f"({100 * comp.share:.0f}% of mode {comp.closest})",
        )

    def set_frf(self, system: ChainSystem, full: ModalResult, model: CraigBamptonModel | None) -> None:
        for item in self._frf_items:
            self.frf.removeItem(item)
        self._frf_items = []
        self.frf_legend.clear()
        tip = system.n - 1
        fn = np.array([m.fn_hz for m in full.modes if m.fn_hz > 0])
        lo = 0.2 * fn.min() if fn.size else 0.1
        hi = 1.5 * fn.max() if fn.size else 10.0
        f = np.geomspace(lo, hi, 2000)
        peaks = [m.damped.fd_hz for m in full.modes if m.damped is not None]
        if model is not None:
            peaks += list(model.fn_hz[model.fn_hz > 0])
        f = np.unique(np.concatenate([f, fn, [p for p in peaks if lo < p < hi]]))
        H = frf(system, f, tip)
        Hr = reduced_frf(model, f, tip) if model is not None else None
        dofs = [tip] if model is None else [tip] + [int(b) for b in model.boundary[:-1]]
        for d in dofs:
            name = "tip" if d == tip else "interface"
            pen = pg.mkPen(MASS_COLORS[d], width=2)
            self._frf_items.append(self.frf.plot(f, np.abs(H[:, d]), pen=pen, name=f"x{d + 1} ({name}) true"))
            if Hr is not None:
                pen = pg.mkPen("#000" if d == tip else MASS_COLORS[d], width=2, style=CB_PEN_STYLE)
                self._frf_items.append(self.frf.plot(f, np.abs(Hr[:, d]), pen=pen, name=f"x{d + 1} CB"))
        if model is not None:
            for fr in model.fn_hz:
                if lo < fr < hi:
                    line = pg.InfiniteLine(pos=math.log10(fr), angle=90,
                                           pen=pg.mkPen("#aaa", width=1, style=QtCore.Qt.PenStyle.DotLine))
                    self.frf.addItem(line)
                    self._frf_items.append(line)
        self.frf.setTitle(f"Force at m{tip + 1} (tip) · solid: full model · dashed: CB · dotted: CB fₙ",
                          size="10pt")
        self.frf.setXRange(math.log10(lo), math.log10(hi), padding=0)


class BasisPlots(QtWidgets.QWidget):
    """Every column of the global T drawn as a shape along the chain: the reduced model's basis."""

    COLS = 2

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.header = QtWidgets.QLabel()
        self.header.setWordWrap(True)
        self.plots = pg.GraphicsLayoutWidget()
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.header)
        layout.addWidget(self.plots, 1)

    def set_model(self, model: CraigBamptonModel | None) -> None:
        self.plots.clear()
        if model is None:
            self.header.setText("")
            return
        n = model.system.n
        self.header.setText(
            f"<b>The reduced model's basis:</b> x = T [q; x<sub>b</sub>]. Each of the {model.n_red} "
            f"columns of T is one shape over the whole chain, and every motion the reduced model can "
            f"make is a combination of these {model.n_red} shapes (the full model has {n}). "
            "<span style='color:#666'>Modal columns q are the substructures' own clamped modes "
            "(zero outside their substructure and at every boundary mass). Boundary columns "
            "x<sub>b</sub> are constraint modes: 1 at that boundary mass, 0 at the other one, and "
            "the static shape in between. That is why x<sub>b</sub> stays a physical "
            "displacement.</span>"
        )
        xs = np.arange(n + 1)
        ticks = [[(0, "gnd")] + [(i, f"m{i}") for i in range(1, n + 1)]]
        freqs = {f"q_{sub.name}{k + 1}": sub.fixed_omegas[k] / (2 * math.pi)
                 for sub in model.substructures for k in range(sub.n_kept)}
        for c, label in enumerate(model.labels):
            p = self.plots.addPlot(row=c // self.COLS, col=c % self.COLS)
            p.setMouseEnabled(x=False, y=False)
            p.hideButtons()
            p.showGrid(x=True, y=True, alpha=0.25)
            p.setXRange(0, n, padding=0.05)
            p.getAxis("bottom").setTicks(ticks)
            col = model.T[:, c]
            if label.startswith("q_"):
                name = label[2]
                color = SUB_COLORS[SUB_NAMES.index(name)]
                title = (f"q<sub>{label[2:]}</sub>: {name}'s clamped mode {label[3:]} "
                         f"({freqs[label]:.3g} Hz)")
                ys = col / np.abs(col).max()
                p.setYRange(-1.15, 1.15, padding=0)
            else:
                color = "#000"
                title = f"{label}: constraint mode ({label} = 1, other boundary held)"
                ys = col
                p.setYRange(-0.1, 1.15, padding=0)
            p.setTitle(title, size="9pt")
            p.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen("#bbb", width=1)))
            for b in model.boundary:
                p.addItem(pg.InfiniteLine(pos=b + 1, angle=90,
                                          pen=pg.mkPen("#bbb", width=1, style=CB_PEN_STYLE)))
            p.plot(xs, np.concatenate([[0.0], ys]), pen=pg.mkPen(color, width=2),
                   symbol="o", symbolSize=7, symbolBrush=color, symbolPen=None)


class SubstructuringPage(QtWidgets.QWidget):
    """Controls, schematic and comparison on the left; plots, matrices and theory on the right."""

    dof_requested = QtCore.Signal(int)  # the user changed N here; the main window owns the system
    edit_parameters = QtCore.Signal()  # the user wants the Simulation page's parameter panel

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.system: ChainSystem | None = None
        self.full: ModalResult | None = None
        self.model: CraigBamptonModel | None = None
        self.comparisons: list[ModeComparison] = []
        self.components: list[ComponentMode] = []
        self._dirty = False
        self._updating = False  # true while controls are being synced to the model
        self._kept_wanted = [1, 1]  # remembered across changes of N, clipped to the interior size

        # --- controls
        box = QtWidgets.QGroupBox("Craig–Bampton substructuring")
        form = QtWidgets.QFormLayout(box)
        self.dof = QtWidgets.QSpinBox()
        self.dof.setRange(1, MAX_DOF)
        self.dof.setToolTip("Same chain as the Simulation page; changing it here changes it there too.")
        self.dof.valueChanged.connect(self.dof_requested)
        form.addRow("Number of masses:", self.dof)
        self.interface = QtWidgets.QComboBox()
        self.interface.setToolTip(
            "Cut the chain at this mass. It becomes a boundary (master) DOF shared by A (ground side) "
            "and B (tip side). The tip, where the force acts, is always a boundary DOF too."
        )
        self.interface.currentIndexChanged.connect(self._on_interface)
        form.addRow("Interface at:", self.interface)
        self.kept: list[QtWidgets.QSpinBox] = []
        self.kept_info: list[QtWidgets.QLabel] = []
        for s, name in enumerate(SUB_NAMES):
            row = QtWidgets.QHBoxLayout()
            spin = QtWidgets.QSpinBox()
            spin.setToolTip(
                f"Number of fixed-interface normal modes of {name} kept in the reduced model "
                "(lowest first). 0 = Guyan (static) condensation of that substructure."
            )
            spin.valueChanged.connect(lambda v, s=s: self._on_kept(s, v))
            info = QtWidgets.QLabel()
            info.setStyleSheet("color: #666;")
            row.addWidget(spin)
            row.addWidget(info, 1)
            label = QtWidgets.QLabel(f"<b style='color:{SUB_COLORS[s]}'>Modes kept in {name}:</b>")
            form.addRow(label, row)
            self.kept.append(spin)
            self.kept_info.append(info)
        presets = QtWidgets.QHBoxLayout()
        self.guyan_button = guyan = QtWidgets.QPushButton("Guyan (0 modes)")
        guyan.setToolTip("Keep no fixed-interface modes: only the boundary DOFs remain (static condensation).")
        guyan.clicked.connect(lambda: self._set_all_kept(0))
        self.exact_button = exact = QtWidgets.QPushButton("All modes (exact)")
        exact.setToolTip("Keep every fixed-interface mode: no reduction, so the result must be exact.")
        exact.clicked.connect(lambda: self._set_all_kept(MAX_DOF))
        presets.addWidget(guyan)
        presets.addWidget(exact)
        form.addRow(presets)
        link = QtWidgets.QLabel(
            "Masses, springs and dampers are edited on the <a href='#sim'>Simulation page</a>."
        )
        link.setStyleSheet("color: #666;")
        link.linkActivated.connect(lambda _: self.edit_parameters.emit())
        form.addRow(link)

        self.schematic = SubstructureSchematic()
        self.table = ComparisonTable()
        self.component_title = QtWidgets.QLabel(
            "<b>Substructures on their own</b> (boundary masses held fixed), and the coupled mode "
            "each one becomes"
        )
        self.component_title.setWordWrap(True)
        self.component_table = ComponentTable()
        self.component_note = QtWidgets.QLabel()
        self.component_note.setWordWrap(True)
        self.component_note.setStyleSheet("color: #666;")
        self.table.itemSelectionChanged.connect(self._on_row)
        self.component_table.itemSelectionChanged.connect(self._on_component_row)
        self.summary = QtWidgets.QLabel()
        self.summary.setWordWrap(True)
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.addWidget(box)
        lv.addWidget(self.schematic)
        lv.addWidget(self.summary)
        lv.addWidget(self.table)
        lv.addSpacing(8)
        lv.addWidget(self.component_title)
        lv.addWidget(self.component_table)
        lv.addWidget(self.component_note)
        lv.addStretch(1)
        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidget(left)
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(520)

        # --- right: plots, matrices, theory
        self.plots = ComparisonPlots()
        self.matrices = QtWidgets.QTextBrowser()
        self.theory = QtWidgets.QTextBrowser()
        self.theory.setHtml(THEORY_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.basis = BasisPlots()
        self.tabs.addTab(self.plots, "Modes && FRF")
        self.tabs.addTab(self.basis, "Basis (T)")
        self.tabs.addTab(self.matrices, "Matrices (step by step)")
        self.tabs.addTab(self.theory, "Theory")

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(left_scroll)
        splitter.addWidget(self.tabs)
        splitter.setSizes([620, 1080])
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

    # ------------------------------------------------------------ inputs
    def set_system(self, system: ChainSystem, full: ModalResult | None = None) -> None:
        """New chain parameters. Recomputed now if visible, else when the page is shown."""
        self.system = system
        self.full = full
        self._dirty = True
        if self.isVisible():
            self.refresh()

    def showEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().showEvent(event)
        if self._dirty:
            self.refresh()

    def _on_interface(self) -> None:
        if self.system is not None and self.interface.count():
            self.refresh()

    def _on_kept(self, s: int, value: int) -> None:
        if self._updating:
            return
        self._kept_wanted[s] = value
        self.refresh()

    def _set_all_kept(self, value: int) -> None:
        self._kept_wanted = [value] * len(SUB_NAMES)
        self.refresh()

    def _on_component_row(self) -> None:
        rows = self.component_table.selectionModel().selectedRows()
        if not rows or not self.components:
            self.plots.set_component(None)
            self.plots.set_highlight(self.table.currentRow())
            return
        comp = self.components[rows[0].row()]
        self.plots.component = comp
        self.table.selectRow(comp.closest - 1)  # shows the coupled mode, which keeps the overlay
        self.plots.set_highlight(comp.closest - 1)
        self.tabs.setCurrentWidget(self.plots)

    def _on_row(self) -> None:
        if self.plots.component is not None and self.table.currentRow() != self.plots.component.closest - 1:
            self.component_table.clearSelection()  # a different coupled mode: drop the overlay
        self.plots.set_highlight(self.table.currentRow())

    # ----------------------------------------------------------- compute
    def refresh(self) -> None:
        if self.system is None:
            return
        self._dirty = False
        system = self.system
        n = system.n
        self._updating = True
        try:
            self.dof.blockSignals(True)
            self.dof.setValue(n)
            self.dof.blockSignals(False)
            self._sync_controls(n)
        finally:
            self._updating = False
        full = self.full if self.full is not None else modal_analysis(system)
        self.full = full

        model, error = None, None
        if n >= 2:
            j = self.interface.currentIndex()
            try:
                model = craig_bampton(system, [j], [s.value() for s in self.kept])
            except ValueError as exc:
                error = str(exc)
        self.model = model
        self.comparisons = compare_modes(model, full) if model is not None else []

        self.schematic.set_model(n, model)
        for s, info in enumerate(self.kept_info):
            info.setText(self._fixed_text(model.substructures[s]) if model else "")
        if model is None:
            reason = error or "Substructuring needs at least 2 masses (an interface and a tip)."
            self.summary.setText(f"<b style='color:#c1121f'>{reason}</b>")
            self.table.setRowCount(0)
            self.component_table.setRowCount(0)
            self.component_note.setText("")
            self.matrices.setHtml(f"<p>{reason}</p>")
            self.plots.set_comparisons([], None)
            self.plots.set_frf(system, full, None)
            self.basis.set_model(None)
            return

        comparisons = self.comparisons
        self.summary.setText(self._summary(model, comparisons))
        self.plots.set_comparisons(comparisons, model)
        self.table.set_rows(comparisons)
        self.plots.component = None
        self.components = components = component_modes(model, full)
        self.component_table.set_rows(components)
        self.component_note.setText(self._component_note(model, components))
        self._on_row()
        self.plots.set_frf(system, full, model)
        self.basis.set_model(model)
        scroll = self.matrices.verticalScrollBar().value()
        self.matrices.setHtml(matrices_html(model, full))
        self.matrices.verticalScrollBar().setValue(scroll)

    def _sync_controls(self, n: int) -> None:
        """Fit the interface choices and kept-mode ranges to the current N."""
        if self.interface.count() != n - 1:  # N changed: start again from the middle
            self.interface.blockSignals(True)
            self.interface.clear()
            self.interface.addItems([f"m{i + 1}" for i in range(n - 1)])
            self.interface.setCurrentIndex(max(0, n // 2 - 1))
            self.interface.blockSignals(False)
        if n < 2:
            return
        counts = interior_counts(n, [self.interface.currentIndex()])
        for spin, count, wanted in zip(self.kept, counts, self._kept_wanted):
            spin.blockSignals(True)
            spin.setRange(0, count)
            spin.setValue(min(wanted, count))
            spin.setSuffix(f" of {count}")
            spin.setEnabled(count > 0)
            spin.blockSignals(False)

    @staticmethod
    def _fixed_text(sub) -> str:
        if not sub.ni:
            return "no interior DOFs"
        freqs = [f"{f:.3g}" for f in sub.fixed_omegas / (2 * math.pi)]
        kept = ", ".join(freqs[: sub.n_kept]) or "none"
        return f"kept: {kept} Hz" + (f"  (next: {freqs[sub.n_kept]} Hz)" if sub.n_kept < sub.ni else "")

    @staticmethod
    def _component_note(model: CraigBamptonModel, components: list[ComponentMode]) -> str:
        empty = [s.name for s in model.substructures if not s.ni]
        note = ""
        if empty:
            note = f"{' and '.join(empty)} has no interior masses, so no modes of its own. "
        if not components:
            return note
        return note + (
            "Coupling through the interface shifts each substructure's modes and mixes them: a "
            "share well below 100% means the coupled mode is built from several component modes "
            "plus boundary motion. The Craig–Bampton model keeps the <i>kept</i> rows as its "
            "modal coordinates q and lets the boundary DOFs do the coupling."
        )

    @staticmethod
    def _summary(model: CraigBamptonModel, comparisons) -> str:
        n = model.system.n
        coords = ", ".join(
            f"q<sub>{l[2:]}</sub>" if l.startswith("q_") else l for l in model.labels
        )
        kind = (
            "Guyan (static) condensation: no fixed-interface modes kept."
            if model.n_modal == 0
            else "All fixed-interface modes kept: no reduction, so the result is exact."
            if model.n_red == n
            else ""
        )
        worst = max((c.error for c in comparisons if c.error is not None), default=0.0)
        worst_zeta = max((c.zeta_error for c in comparisons if c.zeta_error is not None), key=abs, default=0.0)
        return (
            f"Reduced model: <b>{model.n_red} DOFs</b> instead of {n} "
            f"({model.n_modal} modal + {model.boundary.size} boundary): [{coords}]. {kind} "
            f"Largest frequency error: <b style='color:{error_color(worst)}'>{100 * worst:.3g}%</b>; "
            f"damping ratio: <b style='color:{error_color(worst_zeta)}'>{100 * worst_zeta:+.3g}%</b>."
        )
