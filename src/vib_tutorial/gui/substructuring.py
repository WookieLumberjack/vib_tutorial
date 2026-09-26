"""Substructuring page: component mode synthesis of the chain.

The chain from the Simulation page is cut at one interface mass into
substructures A and B. The boundary (master) DOFs are the interface and the
tip, where the force is applied. The page reduces the chain by Craig-Bampton
(fixed interface), Rubin or MacNeal (free interface), compares the reduced
model's modes and tip FRF with the full model's, compares the three methods,
and walks through the matrices.
"""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core import (
    METHOD_NAMES,
    METHODS,
    ChainSystem,
    CMSModel,
    ComponentMode,
    ModalResult,
    ModeComparison,
    compare_modes,
    component_mode_synthesis,
    component_modes,
    frf,
    kept_ranges,
    modal_analysis,
    reduced_frf,
)
from ..core.substructure import METHOD_SHORT
from .animation import MASS_WIDTH, spring_path
from .cms_notes import THEORY_HTML, matrices_html
from .style import (
    FORCE_COLOR,
    MASS_COLORS,
    MAX_DOF,
    METHOD_COLORS,
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

    def set_model(self, n: int, model: CMSModel | None) -> None:
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
                kind = "free" if sub.free else "clamped"
                text = f"{sub.name}: {sub.ni} interior, keep {sub.n_kept} {kind} mode{'s' if sub.n_kept != 1 else ''}"
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
    HEADERS = ["Mode", "fₙ true\n[Hz]", "fₙ {m}\n[Hz]", "Error", "ζ true", "ζ {m}", "MAC"]
    TIPS = [
        "Mode number, ordered by frequency",
        "Natural frequency of the full N-DOF model (Kφ = ω²Mφ)",
        "Natural frequency of the {name} reduced model (K̂η = ω²M̂η). "
        "Blank when the reduced model has fewer modes than this mode number.",
        "(f_{m} − f_true) / f_true. {bound}",
        "Exact damping ratio of the full model (from its damped eigenvalue λ: −Re λ / |λ|)",
        "Exact damping ratio of the reduced model M̂η̈ + Ĉη̇ + K̂η = 0. "
        "Coloured by its error relative to ζ true, which can have either sign. With stiffness-"
        "proportional damping it tracks the frequency error; with non-proportional damping it "
        "can be much worse.",
        "Modal assurance criterion between the true shape φ and the recovered {m} shape Tη: "
        "(φᵀx)² / (φᵀφ · xᵀx). 1 = identical shape, 0 = unrelated.",
    ]
    BOUNDS = {
        "craig-bampton": "Never negative: Craig–Bampton is a Rayleigh–Ritz method, so it can only "
                         "over-estimate stiffness.",
        "rubin": "Never negative: Rubin's method is a Rayleigh–Ritz method, so it can only "
                 "over-estimate stiffness.",
        "macneal": "Either sign in principle: MacNeal drops the residual mass, so it is not a "
                   "Rayleigh–Ritz method and gives no bound.",
    }

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(0, len(self.HEADERS), parent)
        self.set_method("craig-bampton")
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)

    def set_method(self, method: str) -> None:
        fill = {"m": METHOD_SHORT[method], "name": METHOD_NAMES[method], "bound": self.BOUNDS[method]}
        self.setHorizontalHeaderLabels([h.format(**fill) for h in self.HEADERS])
        for col, tip in enumerate(self.TIPS):
            self.horizontalHeaderItem(col).setToolTip(tip.format(**fill))

    def set_rows(self, comparisons) -> None:
        selected = self.currentRow() if self.selectedItems() else 0
        self.setRowCount(len(comparisons))
        for r, c in enumerate(comparisons):
            err = c.error
            cells = [
                str(c.index),
                f"{c.fn_true:.4g}",
                "—" if c.fn_red is None else f"{c.fn_red:.4g}",
                "not in model" if c.fn_red is None else ("—" if err is None else f"{100 * err:+.3g}%"),
                zeta_text(c.zeta_true),
                "—" if c.fn_red is None else zeta_text(c.zeta_red),
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
                elif c.fn_red is None:
                    item.setForeground(pg.mkColor("#999"))
                self.setItem(r, col, item)
        height = self.horizontalHeader().height() + 2 * self.frameWidth()
        height += sum(self.rowHeight(r) for r in range(self.rowCount()))
        self.setFixedHeight(height)
        self.selectRow(min(selected, len(comparisons) - 1))


class ComponentTable(QtWidgets.QTableWidget):
    """Each substructure's own modes (boundary clamped) and the coupled mode each one becomes."""

    HEADERS = ["Mode", "fₙ alone [Hz]", "ζ alone", "Kept", "Becomes", "fₙ [Hz]", "Share"]
    TIPS = {
        False: [
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
        ],
        True: [
            "Free-interface mode r of substructure A or B",
            "Natural frequency of the substructure on its own, with its boundary masses free: "
            "Kφ = ω²Mφ over all its DOFs (the interface mass split half and half). 0 = rigid body.",
            "Exact damping ratio of the substructure on its own (boundary free), from the damped "
            "eigenvalues of its M, C, K",
            "Whether this mode is one of the free-interface modes kept in the reduced model. "
            "Rigid-body modes are always kept.",
            "The mode of the full, coupled chain that this component mode contributes most to",
            "Natural frequency of that coupled mode: compare with fₙ alone to see how coupling "
            "through the interface shifts it",
            "Fraction of the coupled mode's kinetic energy carried by this component mode. The free "
            "modes of a substructure describe any motion of it, so each coupled mode's motion there "
            "is a sum of free modes with amplitudes q; mode r holds q_r² of its kinetic energy "
            "(rigid-body modes included). Summed over both substructures the shares make 100%.",
        ],
    }

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(0, len(self.HEADERS), parent)
        self.setHorizontalHeaderLabels(self.HEADERS)
        self.set_free(False)
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

    def set_free(self, free: bool) -> None:
        for col, tip in enumerate(self.TIPS[free]):
            self.horizontalHeaderItem(col).setToolTip(tip)

    def set_rows(self, modes: list[ComponentMode]) -> None:
        self.clearSelection()
        self.setRowCount(len(modes))
        for r, m in enumerate(modes):
            rigid = m.fn_hz < 1e-9
            cells = [
                f"{m.substructure}{m.index}",
                "0 (rigid)" if rigid else f"{m.fn_hz:.4g}",
                "—" if rigid else zeta_text(m.zeta),
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
    """Selected mode shape (true vs reduced) above the tip FRF (true vs reduced)."""

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
        self.short = "CB"
        self.free = False

    def set_comparisons(self, comparisons, model: CMSModel | None) -> None:
        self.comparisons = comparisons
        self.short = model.short_name if model is not None else "CB"
        self.free = model is not None and model.free
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
        if c.shape_red is None:
            self._cb_curve.setData([], [])
            self.shape.setTitle(f"Mode {c.index} is not in the reduced model", size="10pt")
        else:
            self._cb_curve.setData(xs, np.concatenate([[0.0], c.shape_red]))
            self._cb_curve.setPen(pg.mkPen("#000", width=2, style=CB_PEN_STYLE))
            self._cb_curve.setSymbolPen(pg.mkPen("#000", width=2))
            self.shape_legend.addItem(self._cb_curve, f"Mode {c.index} {self.short}: {c.fn_red:.4g} Hz")
            # No relative error for a rigid-body mode (f_true = 0 when k1 = 0).
            err = "—" if c.error is None else f"{100 * c.error:+.3g}%"
            self.shape.setTitle(f"Mode {c.index}: error {err}, MAC {c.mac:.3f}", size="10pt")
        # Keep an overlay only while it belongs to this coupled mode.
        keep = self.component is not None and self.component.closest == c.index
        self.set_component(self.component if keep else None)

    def set_component(self, comp: ComponentMode | None) -> None:
        """Overlay one substructure's own mode shape (boundary held or free), or clear it."""
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
            f"{comp.substructure}{comp.index} alone, boundary {'free' if self.free else 'held'}: "
            f"{comp.fn_hz:.4g} Hz "
            f"({100 * comp.share:.0f}% of mode {comp.closest})",
        )

    def set_frf(self, system: ChainSystem, full: ModalResult, model: CMSModel | None) -> None:
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
                self._frf_items.append(self.frf.plot(f, np.abs(Hr[:, d]), pen=pen, name=f"x{d + 1} {model.short_name}"))
        if model is not None:
            for fr in model.fn_hz:
                if lo < fr < hi:
                    line = pg.InfiniteLine(pos=math.log10(fr), angle=90,
                                           pen=pg.mkPen("#aaa", width=1, style=QtCore.Qt.PenStyle.DotLine))
                    self.frf.addItem(line)
                    self._frf_items.append(line)
        short = "reduced" if model is None else model.short_name
        self.frf.setTitle(f"Force at m{tip + 1} (tip) · solid: full model · dashed: {short} · dotted: {short} fₙ",
                          size="10pt")
        self.frf.setXRange(math.log10(lo), math.log10(hi), padding=0)


class BasisPlots(QtWidgets.QWidget):
    """Every column of the global T drawn as a shape along the chain: the reduced model's basis.

    The plots are pooled and reused rather than rebuilt: destroying a PlotItem
    that has been in a GraphicsLayout can leave the scene pointing at freed
    C++ objects, which crashed the app after a few changes of N.
    """

    COLS = 2

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.header = QtWidgets.QLabel()
        self.header.setWordWrap(True)
        self.plots = pg.GraphicsLayoutWidget()
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.header)
        layout.addWidget(self.plots, 1)
        self._slots: list[_BasisSlot] = []  # every plot ever made; the first `shown` are in the layout
        self.shown = 0

    def _slot(self, c: int) -> _BasisSlot:
        while len(self._slots) <= c:
            self._slots.append(_BasisSlot())
        return self._slots[c]

    def set_model(self, model: CMSModel | None) -> None:
        # Take the plots out of the layout, but keep them (and so their C++ objects) alive.
        for slot in self._slots[: self.shown]:
            self.plots.ci.removeItem(slot.plot)
        self.shown = 0
        if model is None:
            self.header.setText("")
            return
        n = model.system.n
        if model.free:
            detail = (
                "Modal columns q start from the substructures' own <i>free</i> modes, with the part "
                "that the boundary columns already describe taken out, so they are zero at every "
                "boundary mass (and outside their substructure). Boundary columns x<sub>b</sub> are "
                "residual attachment modes: the static deflection under a force at that boundary "
                "mass, using only the flexibility of the discarded modes, scaled to 1 there and 0 at "
                "the other boundary mass. That is why x<sub>b</sub> stays a physical displacement. "
                "Compare them with Craig–Bampton's straight-line constraint modes."
            )
        else:
            detail = (
                "Modal columns q are the substructures' own clamped modes (zero outside their "
                "substructure and at every boundary mass). Boundary columns x<sub>b</sub> are "
                "constraint modes: 1 at that boundary mass, 0 at the other one, and the static shape "
                "in between. That is why x<sub>b</sub> stays a physical displacement."
            )
        self.header.setText(
            f"<b>The reduced model's basis ({model.name}):</b> x = T [q; x<sub>b</sub>]. Each of the "
            f"{model.n_red} columns of T is one shape over the whole chain, and every motion the reduced "
            f"model can make is a combination of these {model.n_red} shapes (the full model has {n}). "
            f"<span style='color:#666'>{detail}</span>"
        )
        xs = np.arange(n + 1)
        ticks = [[(0, "gnd")] + [(i, f"m{i}") for i in range(1, n + 1)]]
        freqs = {f"q_{sub.name}{k + 1}": sub.omegas[k] / (2 * math.pi)
                 for sub in model.substructures for k in range(sub.n_kept)}
        for c, label in enumerate(model.labels):
            slot = self._slot(c)
            p = slot.plot
            self.plots.addItem(p, row=c // self.COLS, col=c % self.COLS)
            p.setXRange(0, n, padding=0.05)
            p.getAxis("bottom").setTicks(ticks)
            col = model.T[:, c]
            if label.startswith("q_"):
                name = label[2]
                color = SUB_COLORS[SUB_NAMES.index(name)]
                f = freqs[label]
                if not model.free:
                    what = f"clamped mode {label[3:]} ({f:.3g} Hz)"
                elif f < 1e-9:
                    what = f"free mode {label[3:]}, rigid body, minus its boundary part"
                else:
                    what = f"free mode {label[3:]} ({f:.3g} Hz) minus its boundary part"
                title = f"q<sub>{label[2:]}</sub>: {name}'s {what}"
                ys = col / np.abs(col).max()
                p.setYRange(-1.15, 1.15, padding=0)
            else:
                color = "#000"
                kind = "residual attachment" if model.free else "constraint"
                title = f"{label}: {kind} mode ({label} = 1, other boundary held)"
                ys = col
                p.setYRange(min(-0.1, 1.1 * ys.min()), max(1.15, 1.1 * ys.max()), padding=0)
            p.setTitle(title, size="9pt")
            slot.set_boundaries(model.boundary)
            slot.curve.setData(xs, np.concatenate([[0.0], ys]))
            slot.curve.setPen(pg.mkPen(color, width=2))
            slot.curve.setSymbolBrush(color)
        self.shown = len(model.labels)


class _BasisSlot:
    """One reusable basis plot: its curve and its boundary-mass lines."""

    def __init__(self) -> None:
        self.plot = p = pg.PlotItem()
        p.setMouseEnabled(x=False, y=False)
        p.hideButtons()
        p.showGrid(x=True, y=True, alpha=0.25)
        p.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen("#bbb", width=1)))
        self.lines: list[pg.InfiniteLine] = []
        self.curve = p.plot(symbol="o", symbolSize=7, symbolPen=None)

    def set_boundaries(self, boundary) -> None:
        while len(self.lines) < len(boundary):
            line = pg.InfiniteLine(angle=90, pen=pg.mkPen("#bbb", width=1, style=CB_PEN_STYLE))
            self.plot.addItem(line)
            self.lines.append(line)
        for i, line in enumerate(self.lines):
            line.setVisible(i < len(boundary))
            if i < len(boundary):
                line.setPos(boundary[i] + 1)


ERROR_FLOOR = 1e-9  # relative; smaller errors (exact models) are drawn at this level on the log plot


class MethodComparison(QtWidgets.QWidget):
    """The same cut reduced by all three methods: errors now, and how they converge."""

    HEADERS = ["Mode", "fₙ true [Hz]"] + [f"{METHOD_SHORT[m]} error" for m in METHODS]

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.header = QtWidgets.QLabel()
        self.header.setWordWrap(True)
        self.table = QtWidgets.QTableWidget(0, len(self.HEADERS))
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        for col, m in enumerate(METHODS, start=2):
            self.table.horizontalHeaderItem(col).setToolTip(
                f"(f − f_true) / f_true for the {METHOD_NAMES[m]} model with the same interface and "
                "the same number of modes kept in each substructure as on the left")
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.plot = pg.PlotWidget()
        self.plot.setLogMode(y=True)
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel("bottom", "Reduced-model coordinates (modes kept + boundary DOFs)")
        self.plot.setLabel("left", "|frequency error|  [%]")
        self.plot.setMouseEnabled(x=False, y=False)
        self.legend = self.plot.addLegend(offset=(-5, 5))
        self.note = QtWidgets.QLabel()
        self.note.setWordWrap(True)
        self.note.setStyleSheet("color: #666;")
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.header)
        layout.addWidget(self.table)
        layout.addWidget(self.plot, 1)
        layout.addWidget(self.note)
        self.sweeps: dict[str, list[tuple[int, np.ndarray]]] = {}  # method -> [(n_red, errors per mode)]
        self.current: dict[str, CMSModel | None] = {}
        self.errors: dict[str, str] = {}  # method -> why it has no model
        self.full: ModalResult | None = None

    def set_system(self, system: ChainSystem, full: ModalResult, interface: int, kept: list[int]) -> None:
        """Reduce with every method at these settings, and sweep the number of modes kept."""
        self.full = full
        self.current, self.sweeps, self.errors = {}, {}, {}
        for method in METHODS:
            ranges = kept_ranges(system, [interface], method)
            try:
                self.current[method] = component_mode_synthesis(system, [interface], kept, method)
            except ValueError as exc:
                self.current[method], self.errors[method] = None, str(exc)
            points = []
            configs = self._sweep(ranges)
            for config in configs:
                try:
                    model = component_mode_synthesis(system, [interface], config, method)
                except ValueError:
                    continue
                points.append((model.n_red, self._errors(model)))
            self.sweeps[method] = points
        self._fill_table()

    def clear(self, reason: str) -> None:
        self.full, self.current, self.sweeps, self.errors = None, {}, {}, {}
        self.table.setRowCount(0)
        self.header.setText(reason)
        self.set_mode(0)

    @staticmethod
    def _sweep(ranges: list[tuple[int, int]]) -> list[list[int]]:
        """From the fewest modes to all of them, one more at a time, taking turns between substructures."""
        kept = [lo for lo, _ in ranges]
        configs = [list(kept)]
        while any(k < hi for k, (_, hi) in zip(kept, ranges)):
            for s, (_, hi) in enumerate(ranges):
                if kept[s] < hi:
                    kept[s] += 1
                    configs.append(list(kept))
        return configs

    def _errors(self, model: CMSModel) -> np.ndarray:
        """Relative frequency error of each true mode; NaN where the model has no such mode."""
        return np.array([np.nan if c.error is None else c.error for c in compare_modes(model, self.full)])

    def _fill_table(self) -> None:
        modes = self.full.modes
        self.table.setRowCount(len(modes))
        for r, mode in enumerate(modes):
            cells = [(str(r + 1), MODE_COLORS[r % len(MODE_COLORS)]), (f"{mode.fn_hz:.4g}", None)]
            for method in METHODS:
                model = self.current[method]
                err = self._errors(model)[r] if model is not None else np.nan
                if model is None:
                    cells.append(("—", "#999"))
                elif r >= model.omegas.size:
                    cells.append(("not in model", "#999"))
                elif np.isnan(err):
                    cells.append(("—", None))
                else:
                    cells.append((f"{100 * err:+.3g}%", error_color(err)))
            for col, (text, color) in enumerate(cells):
                item = QtWidgets.QTableWidgetItem(text)
                item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                if color:
                    item.setForeground(pg.mkColor(color))
                self.table.setItem(r, col, item)
        height = self.table.horizontalHeader().height() + 2 * self.table.frameWidth()
        height += sum(self.table.rowHeight(r) for r in range(self.table.rowCount()))
        self.table.setFixedHeight(height)
        sizes = {m: model.n_red for m, model in self.current.items() if model is not None}
        n_red = next(iter(sizes.values()), None)
        macneal = self.current.get("macneal")
        self.header.setText(
            "<b>Three methods, same cut, same modes kept</b> (as set on the left). "
            + ("" if n_red is None else f"Each reduced model has {n_red} coordinates. ")
            + ("" if macneal is None else
               f"MacNeal's boundary coordinates have no mass, so it has only {macneal.omegas.size} modes. ")
            + "".join(f"<br><span style='color:#c1121f'>{METHOD_NAMES[m]}: {e}</span>"
                      for m, e in self.errors.items())
        )

    def set_mode(self, r: int) -> None:
        """Draw how the error of true mode r converges as each method keeps more modes."""
        self.plot.clear()
        self.legend.clear()
        if self.full is None or not (0 <= r < len(self.full.modes)):
            self.plot.setTitle("")
            self.note.setText("")
            return
        for method in METHODS:
            pts = [(n, e[r]) for n, e in self.sweeps.get(method, []) if not np.isnan(e[r])]
            color = METHOD_COLORS[method]
            if pts:
                xs, ys = np.array(pts).T
                ys = 100 * np.maximum(np.abs(ys), ERROR_FLOOR)
                self.plot.plot(xs, ys, pen=pg.mkPen(color, width=2), symbol="o", symbolSize=7,
                               symbolBrush=color, symbolPen=None, name=METHOD_NAMES[method])
            model = self.current.get(method)
            if model is not None and r < model.omegas.size:
                err = self._errors(model)[r]
                if not np.isnan(err):
                    self.plot.plot([model.n_red], [100 * max(abs(err), ERROR_FLOOR)], pen=None, symbol="o",
                                   symbolSize=15, symbolBrush=None, symbolPen=pg.mkPen(color, width=2))
        sizes = [n for pts in self.sweeps.values() for n, _ in pts]
        if sizes:
            self.plot.getAxis("bottom").setTicks([[(n, str(n)) for n in range(min(sizes), max(sizes) + 1)]])
        self.plot.setTitle(f"Mode {r + 1} ({self.full.modes[r].fn_hz:.4g} Hz): error as more modes are kept",
                           size="10pt")
        self.note.setText(
            "Each curve starts from the fewest modes (none for Craig–Bampton; B's rigid-body mode for "
            "the free-interface methods) and adds one mode at a time, taking turns between A and B. "
            "Large open circles: the settings on the left. Errors below 10⁻⁷% (exact models) are "
            "drawn at 10⁻⁷%. Select a mode in the table on the left to plot it."
        )


class SubstructuringPage(QtWidgets.QWidget):
    """Controls, schematic and comparison on the left; plots, matrices and theory on the right."""

    dof_requested = QtCore.Signal(int)  # the user changed N here; the main window owns the system
    edit_parameters = QtCore.Signal()  # the user wants the Simulation page's parameter panel

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.system: ChainSystem | None = None
        self.full: ModalResult | None = None
        self.model: CMSModel | None = None
        self.comparisons: list[ModeComparison] = []
        self.components: list[ComponentMode] = []
        self._dirty = False
        self._updating = False  # true while controls are being synced to the model
        self._kept_wanted = [1, 1]  # remembered across changes of N, clipped to the interior size

        # --- controls
        box = QtWidgets.QGroupBox("Component mode synthesis")
        form = QtWidgets.QFormLayout(box)
        self.dof = QtWidgets.QSpinBox()
        self.dof.setRange(1, MAX_DOF)
        self.dof.setToolTip("Same chain as the Simulation page; changing it here changes it there too.")
        self.dof.valueChanged.connect(self.dof_requested)
        form.addRow("Number of masses:", self.dof)
        self.method = QtWidgets.QComboBox()
        for m, text in zip(METHODS, ("Craig–Bampton (fixed interface)", "Rubin (free interface)",
                                     "MacNeal (free interface, massless residual)")):
            self.method.addItem(text, m)
        self.method.setToolTip(
            "Craig–Bampton: modes of each substructure with its boundary held, plus static constraint "
            "modes.\nRubin: modes with the boundary free, plus the residual flexibility of the "
            "discarded modes (and their residual mass).\nMacNeal: as Rubin, but the residual "
            "flexibility has no mass, so the boundary DOFs carry no inertia.\n"
            "The Compare methods tab shows all three side by side."
        )
        self.method.currentIndexChanged.connect(self._on_method)
        form.addRow("Method:", self.method)
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
        self.guyan_button = guyan = QtWidgets.QPushButton()
        guyan.clicked.connect(lambda: self._set_all_kept(0))
        self.exact_button = exact = QtWidgets.QPushButton()
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
        self.component_title = QtWidgets.QLabel()
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
        self.compare = MethodComparison()
        self.tabs.addTab(self.plots, "Modes && FRF")
        self.tabs.addTab(self.compare, "Compare methods")
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
        self._apply_method()

    @property
    def method_key(self) -> str:
        return self.method.currentData()

    def _apply_method(self) -> None:
        """Labels, tooltips and presets that depend on the chosen method."""
        method = self.method_key
        free = method != "craig-bampton"
        for spin, name in zip(self.kept, SUB_NAMES):
            spin.setToolTip(
                f"Number of free-interface modes of {name} kept in the reduced model (lowest first). "
                "Rigid-body modes are always kept, so a floating substructure keeps at least one."
                if free else
                f"Number of fixed-interface normal modes of {name} kept in the reduced model "
                "(lowest first). 0 = Guyan (static) condensation of that substructure."
            )
        if free:
            self.guyan_button.setText("Fewest modes")
            self.guyan_button.setToolTip("Keep only the rigid-body modes (none in A): the residual "
                                         "flexibility stands in for every other mode.")
        else:
            self.guyan_button.setText("Guyan (0 modes)")
            self.guyan_button.setToolTip("Keep no fixed-interface modes: only the boundary DOFs remain "
                                         "(static condensation).")
        if method == "macneal":
            self.exact_button.setText("All modes")
            self.exact_button.setToolTip("Keep as many free-interface modes as there are interior DOFs. "
                                         "MacNeal is still not exact: the discarded modes' mass is dropped.")
        else:
            self.exact_button.setText("All modes (exact)")
            self.exact_button.setToolTip("Keep every mode there is room for: no reduction, so the result "
                                         "must be exact.")
        self.component_title.setText(
            f"<b>Substructures on their own</b> (boundary masses {'free' if free else 'held fixed'}), "
            "and the coupled mode each one becomes"
        )
        self.table.set_method(method)
        self.component_table.set_free(free)

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

    def _on_method(self) -> None:
        self._apply_method()
        if self.system is not None:
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
        self.compare.set_mode(self.table.currentRow())

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
            kept = [s.value() for s in self.kept]
            try:
                model = component_mode_synthesis(system, [j], kept, self.method_key)
            except ValueError as exc:
                error = str(exc)
            self.compare.set_system(system, full, j, kept)
        else:
            self.compare.clear("Substructuring needs at least 2 masses (an interface and a tip).")
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
            self.compare.set_mode(0)  # the other methods may still work
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
        ranges = kept_ranges(self.system, [self.interface.currentIndex()], self.method_key)
        for spin, (lo, hi), wanted in zip(self.kept, ranges, self._kept_wanted):
            spin.blockSignals(True)
            spin.setRange(lo, hi)
            spin.setValue(int(np.clip(wanted, lo, hi)))
            spin.setSuffix(f" of {hi}")
            spin.setEnabled(hi > lo)
            spin.blockSignals(False)

    @staticmethod
    def _fixed_text(sub) -> str:
        if not sub.ni:
            return "no interior DOFs"
        freqs = [f"{f:.3g}" if f > 1e-9 else "0" for f in sub.omegas / (2 * math.pi)]
        kept = ", ".join(freqs[: sub.n_kept]) or "none"
        rigid = f", {sub.n_rigid} rigid" if sub.n_rigid else ""
        nxt = f"  (next: {freqs[sub.n_kept]} Hz)" if sub.n_kept < sub.omegas.size else ""
        return f"kept: {kept} Hz{rigid}{nxt}"

    @staticmethod
    def _component_note(model: CMSModel, components: list[ComponentMode]) -> str:
        empty = [s.name for s in model.substructures if not s.ni]
        note = ""
        if empty:
            note = (f"{' and '.join(empty)} has no interior masses, so it is kept as its physical "
                    "boundary DOFs, with no modes of its own. ")
        if not components:
            return note
        if model.free:
            return note + (
                "Each substructure's free modes include its interface masses, so the interface moves "
                "in every one of them. A floating substructure (B) has a rigid-body mode at 0 Hz: it "
                "carries B's share of every low coupled mode, and is always kept. The "
                f"{model.name} model keeps the <i>kept</i> rows as its modal coordinates q, and the "
                "residual flexibility of the others ties them to the boundary DOFs."
            )
        return note + (
            "Coupling through the interface shifts each substructure's modes and mixes them: a "
            "share well below 100% means the coupled mode is built from several component modes "
            "plus boundary motion. The Craig–Bampton model keeps the <i>kept</i> rows as its "
            "modal coordinates q and lets the boundary DOFs do the coupling."
        )

    @staticmethod
    def _summary(model: CMSModel, comparisons) -> str:
        n = model.system.n
        coords = ", ".join(
            f"q<sub>{l[2:]}</sub>" if l.startswith("q_") else l for l in model.labels
        )
        fewest = model.n_modal == sum(s.n_rigid for s in model.substructures)
        if not model.free:
            kind = ("Guyan (static) condensation: no fixed-interface modes kept." if fewest
                    else "All fixed-interface modes kept: no reduction, so the result is exact."
                    if model.n_red == n else "")
        elif model.method == "rubin":
            kind = ("Fewest modes: only rigid-body modes kept; the residual flexibility stands in for "
                    "the rest." if fewest
                    else "All the modes there is room for: no reduction, so the result is exact."
                    if model.n_red == n else "")
        else:
            kind = (f"The boundary DOFs have no mass and are condensed out, so the model has "
                    f"<b>{model.omegas.size} modes</b>. ")
            if model.n_red == n:
                kind += "Not exact even now: the mass of the discarded modes is dropped."
        worst = max((c.error for c in comparisons if c.error is not None), key=abs, default=0.0)
        worst_zeta = max((c.zeta_error for c in comparisons if c.zeta_error is not None), key=abs, default=0.0)
        return (
            f"{model.name} reduced model: <b>{model.n_red} DOFs</b> instead of {n} "
            f"({model.n_modal} modal + {model.boundary.size} boundary): [{coords}]. {kind} "
            f"Largest frequency error: <b style='color:{error_color(worst)}'>{100 * worst:+.3g}%</b>; "
            f"damping ratio: <b style='color:{error_color(worst_zeta)}'>{100 * worst_zeta:+.3g}%</b>."
        )
