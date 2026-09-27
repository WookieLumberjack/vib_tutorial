"""Modal coupling page: two subsystems, one mode each, joined as a 2-DOF system.

The chain from the Simulation page is split into A (grounded, masses 1..j) and
B (the rest, its first spring and damper tied to ground). One mode of each
becomes an equivalent oscillator, and the two oscillators are joined B's on
A's. The page compares the 2-DOF model's modes with the chain's, and shows how
the two modes veer apart as B's frequency or mass is swept: the mode splitting
of a vibration absorber, which happens whether or not it was intended.
"""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core import ChainSystem, ModalResult, frf, modal_analysis
from ..core.modal_coupling import (
    A_MASS_NAMES,
    A_MASSES,
    SWEEPS,
    CoupledComparison,
    CoupledModel,
    Subsystem,
    Sweep,
    alone_frf,
    compare_coupled,
    coupled_frf,
    coupled_model,
    subsystems,
    sweep,
)
from .animation import MASS_WIDTH, spring_path
from .coupling_notes import THEORY_HTML
from .style import MAX_DOF, colors, text_on
from .substructuring import error_color, zeta_text
from .theming import mute

DASH = QtCore.Qt.PenStyle.DashLine
DOT = QtCore.Qt.PenStyle.DotLine
SPLIT_TIP = (
    "<p>Split the chain after this mass. <b>A</b> is the masses up to it, on the ground as "
    "before. <b>B</b> is the rest; the spring and damper that joined it to A are tied to the "
    "ground instead.</p>"
    "<p>Each is solved on its own, one mode of each becomes a single-DOF oscillator, and the two "
    "oscillators are joined again, B's on A's.</p>"
)
A_MASS_TIP = (
    "<p>The mass of A's oscillator.</p>"
    "<p><b>Modal mass at the interface</b>, 1/φ<sub>tip</sub>² (φ mass-normalized): the mass of "
    "A's mode as its tip feels it. B is joined to A at A's tip, so this is the right one.</p>"
    "<p><b>Effective mass (base)</b>, Γ² with Γ = φᵀM1: the mass of A's mode as a shaking ground "
    "feels it. Right for B, which is joined at its base, but not for A. Choose it to see the "
    "difference.</p>"
)
RESIDUAL_TIP = (
    "<p>B's mass that is not in its chosen mode, m<sub>B</sub> − m<sub>eff</sub>, belongs to its "
    "other modes. Well below their frequencies they move rigidly with B's base, which is A's "
    "tip, so this mass rides on A's oscillator.</p>"
    "<p>With it (and the modal mass at the interface), the 2-DOF model is a Rayleigh–Ritz model "
    "of the chain, so its frequencies can only be too high.</p>"
)
SWEEP_TIP = (
    "<p><b>B's frequency:</b> B's springs are scaled so that f<sub>B</sub>/f<sub>A</sub> runs "
    "across the plot while the mass ratio μ stays as it is.</p>"
    "<p><b>Mass ratio:</b> all of B (masses and springs) is scaled so that μ runs across the plot "
    "while f<sub>B</sub> stays as it is.</p>"
    "<p>B's dampers are scaled with it, so its damping ratios stay the same.</p>"
)
SWEEP_NAMES = {"frequency": "B's frequency (f_B / f_A)", "mass": "Mass ratio (μ = m_b / m_a)"}


def sub_color(name: str) -> str:
    return colors.sub["AB".index(name)]


def _fit_table(table: QtWidgets.QTableWidget) -> None:
    # The header's size hint, not its height: a hidden table has not laid its header out yet.
    header = table.horizontalHeader()
    height = max(header.height(), header.sizeHint().height()) + 2 * table.frameWidth()
    height += sum(table.rowHeight(r) for r in range(table.rowCount()))
    table.setFixedHeight(height)


def _table(headers: list[str], tips: list[str], selectable: bool = False) -> QtWidgets.QTableWidget:
    table = QtWidgets.QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(headers)
    for col, tip in enumerate(tips):
        table.horizontalHeaderItem(col).setToolTip(tip)
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
    if selectable:
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
    else:
        table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
    table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
    return table


def _set_row(table: QtWidgets.QTableWidget, r: int, cells: list[tuple[str, str | None]]) -> None:
    for col, (text, color) in enumerate(cells):
        item = QtWidgets.QTableWidgetItem(text)
        item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        if color:
            item.setForeground(pg.mkColor(color))
        table.setItem(r, col, item)


def _g(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}g}"


class CouplingSchematic(pg.PlotWidget):
    """A and B on their own (top), and the equivalent 2-DOF system they become (bottom)."""

    GAP = 0.6  # extra room before B, for its own ground

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMenuEnabled(False)
        self.setMouseEnabled(x=False, y=False)
        self.hideButtons()
        self.hideAxis("left")
        self.hideAxis("bottom")
        self.setAspectLocked(True)
        self.setFixedHeight(250)
        self._items: list = []

    def _add(self, item):
        self.addItem(item)
        self._items.append(item)
        return item

    def _wall(self, x: float, y: float) -> None:
        pen = pg.mkPen(colors.structure, width=2)
        self._add(pg.PlotDataItem([x, x], [y - 0.3, y + 0.3], pen=pg.mkPen(colors.structure, width=4)))
        hx, hy = [], []
        for yy in np.linspace(y - 0.25, y + 0.3, 6):
            hx += [x, x - 0.08, np.nan]
            hy += [yy, yy - 0.08, np.nan]
        self._add(pg.PlotDataItem(hx, hy, pen=pen, connect="finite"))

    def _mass(self, xc: float, y: float, fill: str, text: str, outline: str | None = None, w: float = MASS_WIDTH):
        rect = QtWidgets.QGraphicsRectItem(QtCore.QRectF(xc - w / 2, y - 0.2, w, 0.4))
        rect.setBrush(pg.mkBrush(fill))
        rect.setPen(pg.mkPen(outline or colors.structure, width=2 if outline else 1))
        rect.setZValue(5)
        self._add(rect)
        label = pg.TextItem(text, color=text_on(fill), anchor=(0.5, 0.5))
        label.setPos(xc, y)
        label.setZValue(6)
        self._add(label)

    def _spring(self, x0: float, x1: float, y: float, color: str) -> None:
        self._add(pg.PlotDataItem(*spring_path(x0, x1, y), pen=pg.mkPen(color, width=2)))

    def _text(self, text: str, x: float, y: float, color: str, anchor=(0.5, 0.5)) -> None:
        item = pg.TextItem(text, color=color, anchor=anchor)
        item.setPos(x, y)
        self._add(item)

    def set_model(self, n: int, split: int, model: CoupledModel | None) -> None:
        for item in self._items:
            self.removeItem(item)
        self._items = []
        half = MASS_WIDTH / 2
        # --- top: A and B on their own
        y = 0.75
        self._text("on their own", -0.25, y + 0.5, colors.muted, anchor=(0.0, 0.5))
        self._wall(0.0, y)
        right = 0.0
        for i in range(n):
            b = i >= split
            name = "B" if b else "A"
            xc = i + 1.0 + (self.GAP if b else 0.0)
            if i == split:  # B's own ground, where A's tip was
                right = split + 0.35 + self.GAP / 2
                self._wall(right, y)
            self._spring(right, xc - half, y, sub_color(name))
            self._mass(xc, y, colors.mass[i], f"m{i + 1}")
            right = xc + half
        self._text("A", split / 2 + 0.5, y - 0.4, sub_color("A"))
        self._text("B", split + 1 + self.GAP + (n - split - 1) / 2, y - 0.4, sub_color("B"))
        # --- bottom: the equivalent 2-DOF system
        y = -0.85
        self._text("equivalent 2-DOF", -0.25, y + 0.55, colors.muted, anchor=(0.0, 0.5))
        self._wall(0.0, y)
        xa, xb = 1.5, 3.2
        w = 0.6
        if model is None:
            self._text("—", 1.5, y, colors.muted)
        else:
            self._spring(0.0, xa - w / 2, y, sub_color("A"))
            self._spring(xa + w / 2, xb - w / 2, y, sub_color("B"))
            self._mass(xa, y, sub_color("A"), "m_a", w=w)
            self._mass(xb, y, sub_color("B"), "m_b", w=w)
            self._text(f"A{model.mode_a + 1}", xa / 2 - 0.1, y + 0.3, sub_color("A"))
            self._text(f"B{model.mode_b + 1}", (xa + xb) / 2, y + 0.3, sub_color("B"))
            res = f" + {model.m_residual:.3g}" if model.residual and model.m_residual > 1e-9 else ""
            self._text(f"m_a = {model.osc_a.m:.3g}{res} kg", xa, y - 0.25, colors.foreground, anchor=(0.5, 0.0))
            self._text(f"m_b = {model.osc_b.m:.3g} kg", xb + 0.2, y - 0.25, colors.foreground, anchor=(0.0, 0.0))
        self.setRange(QtCore.QRectF(-0.4, -1.45, n + 1.0 + self.GAP + 0.2, 2.55), padding=0)


class SubsystemTable(QtWidgets.QTableWidget):
    """The modes of A and of B on their own, with the masses a joint would feel. A click couples that mode."""

    HEADERS = ["Mode", "fₙ [Hz]", "ζ modal", "Effective\nmass [kg]", "Share of\nits mass", "Modal mass at\ninterface [kg]"]
    TIPS = [
        "Mode r of A (grounded, tip free) or of B (its first spring and damper tied to ground). "
        "Click a row to couple that mode.",
        "Natural frequency of the subsystem on its own: Kφ = ω²Mφ",
        "Modal damping ratio φᵀCφ / (2ω), φ mass-normalized: the damping of the equivalent oscillator",
        "Γ² with Γ = φᵀM1: the mass of this mode that a motion of the subsystem's base (the ground) "
        "feels. They add up to the subsystem's mass. For B, whose base is A's tip, this is the mass "
        "of its oscillator.",
        "Effective mass as a fraction of the subsystem's total mass",
        "1/φ_tip² for A: the mass of this mode that A's tip feels, as for a vibration absorber's "
        "primary. B is joined to A at A's tip, so this is the mass of A's oscillator (by default).",
    ]

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(0, len(self.HEADERS), parent)
        self.setHorizontalHeaderLabels(self.HEADERS)
        for col, tip in enumerate(self.TIPS):
            self.horizontalHeaderItem(col).setToolTip(tip)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.rows: list[tuple[str, int]] = []  # (subsystem, 0-based mode) of each row

    def set_subsystems(self, A: Subsystem, B: Subsystem, mode_a: int, mode_b: int, a_mass: str) -> None:
        self.rows = [("A", r) for r in range(A.omegas.size)] + [("B", r) for r in range(B.omegas.size)]
        self.setRowCount(len(self.rows))
        for row, (name, r) in enumerate(self.rows):
            sub = A if name == "A" else B
            used = r == (mode_a if name == "A" else mode_b)
            meff = sub.effective_masses[r]
            rigid = sub.omegas[r] < 1e-9
            tip = sub.tip_masses()[r] if name == "A" else None
            # The mass the oscillator takes is shown bold.
            used_col = (5 if a_mass == "tip" else 3) if name == "A" else 3
            cells = [
                (f"{name}{r + 1}" + ("  ◀" if used else ""), sub_color(name)),
                ("0 (rigid)" if rigid else _g(sub.fn_hz[r]), None),
                ("—" if rigid else f"{sub.zetas[r]:.4f}", None),
                (_g(meff, 3), None),
                (f"{100 * meff / sub.total_mass:.1f}%", None),
                ("—" if tip is None else ("∞" if not np.isfinite(tip) else _g(tip, 3)), None),
            ]
            _set_row(self, row, cells)
            if used:
                for col in (0, used_col):
                    font = self.item(row, col).font()
                    font.setBold(True)
                    self.item(row, col).setFont(font)
        _fit_table(self)


class VeeringPlot(QtWidgets.QWidget):
    """Natural frequencies of the chain and of the 2-DOF model as B's frequency or mass is swept."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.kind = QtWidgets.QComboBox()
        for k in SWEEPS:
            self.kind.addItem(SWEEP_NAMES[k], k)
        self.kind.setToolTip(SWEEP_TIP)
        top = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel("Sweep:")
        label.setToolTip(SWEEP_TIP)
        top.addWidget(label)
        top.addWidget(self.kind)
        top.addStretch(1)
        self.plot = pg.PlotWidget()
        self.plot.setLogMode(x=True, y=True)
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel("left", "Natural frequency", units="Hz")
        self.plot.setMouseEnabled(x=False, y=False)
        self.legend = self.plot.addLegend(offset=(5, 5))
        self.note = QtWidgets.QLabel()
        self.note.setWordWrap(True)
        mute(self.note)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.plot, 1)
        layout.addWidget(self.note)
        self.data: Sweep | None = None

    def apply_theme(self) -> None:
        self.legend.setBrush(colors.legend_brush())

    def clear(self, reason: str) -> None:
        self.data = None
        self.plot.clear()
        self.legend.clear()
        self.plot.setTitle("")
        self.note.setText(reason)

    def set_model(self, model: CoupledModel, comparisons: list[CoupledComparison]) -> None:
        self.plot.clear()
        self.legend.clear()
        kind = self.kind.currentData()
        try:
            self.data = s = sweep(model, kind)
        except ValueError as exc:
            self.clear(str(exc))
            return
        paired = {c.full_index - 1 for c in comparisons}
        for r in range(s.full.shape[1]):
            color = colors.mode[r % len(colors.mode)]
            width = 3 if r in paired else 1.5
            self.plot.plot(s.x, s.full[:, r], pen=pg.mkPen(color, width=width),
                           name="chain (every mode)" if r == 0 else None)
        for r in range(2):
            self.plot.plot(s.x, s.two_dof[:, r], pen=pg.mkPen(colors.strong, width=2, style=DASH),
                           name="2-DOF model" if r == 0 else None)
        self.plot.plot(s.x, s.f_a, pen=pg.mkPen(sub_color("A"), width=2, style=DOT),
                       name=f"A{model.mode_a + 1} alone")
        self.plot.plot(s.x, s.f_b, pen=pg.mkPen(sub_color("B"), width=2, style=DOT),
                       name=f"B{model.mode_b + 1} alone")
        now = pg.InfiniteLine(pos=math.log10(s.x_now), angle=90, pen=pg.mkPen(colors.grey, width=1),
                              label="now", labelOpts={"position": 0.95, "color": colors.muted})
        self.plot.addItem(now)
        self.plot.plot([s.x_now, s.x_now], list(model.fn_hz), pen=None, symbol="o", symbolSize=10,
                       symbolBrush=None, symbolPen=pg.mkPen(colors.strong, width=2))
        top = 1.4 * max(s.two_dof.max(), 1e-9)
        bottom = max(min(s.two_dof.min(), s.f_a.min()), 1e-9) / 1.4
        self.plot.setYRange(math.log10(bottom), math.log10(top), padding=0)
        self.plot.setXRange(math.log10(s.x[0]), math.log10(s.x[-1]), padding=0)
        if kind == "frequency":
            self.plot.setLabel("bottom", "f_B / f_A (B's springs scaled)")
            self.plot.setTitle(f"μ = {model.mass_ratio:.3g} fixed", size="10pt")
            gap = self._gap_at_tuning(s)
            self.note.setText(
                "Solid: the chain's natural frequencies (thick: the two the 2-DOF modes stand for). "
                "Dashed: the 2-DOF model. Dotted: A's and B's modes on their own. Far from f<sub>B</sub> = "
                "f<sub>A</sub> each coupled mode follows one of them. Near it the two modes <i>veer</i> "
                "apart instead of crossing, and trade shapes: the mode splitting of a vibration absorber. "
                + (f"At f<sub>B</sub> = f<sub>A</sub> the 2-DOF modes are {100 * gap:.3g}% of f<sub>A</sub> "
                   f"apart; two oscillators tuned exactly are √μ = {100 * math.sqrt(model.mass_ratio):.3g}% "
                   "apart, whatever μ. "
                   if gap is not None else "")
                + "Circles: the current parameters."
            )
        else:
            self.plot.setLabel("bottom", "μ = m_b / m_a (all of B scaled)")
            self.plot.setTitle(f"f_B / f_A = {model.freq_ratio:.3g} fixed", size="10pt")
            self.note.setText(
                "Solid: the chain's natural frequencies (thick: the two the 2-DOF modes stand for). "
                "Dashed: the 2-DOF model. Dotted: A's and B's modes on their own, which do not change. "
                "A light B barely moves A's mode; the heavier B, the more the two modes are pushed apart. "
                "The closer f<sub>B</sub>/f<sub>A</sub> is to 1, the smaller the mass ratio it takes. "
                "Circles: the current parameters."
            )

    @staticmethod
    def _gap_at_tuning(s: Sweep) -> float | None:
        """(f2 - f1) / f_A where f_B = f_A, if the sweep reaches it."""
        if not (s.x[0] <= 1.0 <= s.x[-1]):
            return None
        f1 = np.interp(0.0, np.log(s.x), s.two_dof[:, 0])
        f2 = np.interp(0.0, np.log(s.x), s.two_dof[:, 1])
        return float((f2 - f1) / s.f_a[0])


class ShapePlots(pg.GraphicsLayoutWidget):
    """Each 2-DOF mode drawn on the chain, over the chain's own mode it stands for."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.plots = []
        self.legends = []
        for r in range(2):
            p = self.addPlot(row=r, col=0)
            p.showGrid(x=True, y=True, alpha=0.3)
            p.setMouseEnabled(x=False, y=False)
            p.setYRange(-1.15, 1.6, padding=0)
            p.getAxis("left").setTicks([[(v, f"{v:g}") for v in (-1, -0.5, 0, 0.5, 1)]])
            p.setLabel("left", "Normalized amplitude")
            self.plots.append(p)
            self.legends.append(p.addLegend(offset=(5, 2), colCount=2))
        self.plots[1].setLabel("bottom", "Position along chain (0 = ground)")
        self.apply_theme()

    def apply_theme(self) -> None:
        for legend in self.legends:
            legend.setBrush(colors.legend_brush())

    def set_model(self, model: CoupledModel | None, full: ModalResult, comparisons: list[CoupledComparison]) -> None:
        for p, legend in zip(self.plots, self.legends):
            p.clear()
            legend.clear()
            p.setTitle("")
        if model is None:
            return
        n = model.system.n
        xs = np.arange(n + 1)
        for r, (p, c) in enumerate(zip(self.plots, comparisons)):
            line = pg.InfiniteLine(pos=model.split + 0.5, angle=90, pen=pg.mkPen(colors.grey, width=1, style=DASH),
                                   label="split", labelOpts={"position": 0.08, "color": colors.muted})
            p.addItem(line)
            p.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(colors.faint, width=1)))
            true = full.modes[c.full_index - 1].shape
            red = model.shapes[:, r]
            if red @ true < 0:
                red = -red
            color = colors.mode[(c.full_index - 1) % len(colors.mode)]
            p.plot(xs, np.concatenate([[0.0], true]), pen=pg.mkPen(color, width=3), symbol="s", symbolSize=9,
                   symbolPen=None, symbolBrush=color, name=f"chain mode {c.full_index}: {c.fn_full:.4g} Hz")
            p.plot(xs, np.concatenate([[0.0], red]), pen=pg.mkPen(colors.strong, width=2, style=DASH), symbol="o",
                   symbolSize=11, symbolBrush=None, symbolPen=pg.mkPen(colors.strong, width=2),
                   name=f"2-DOF mode {r + 1}: {c.fn:.4g} Hz")
            ua, ub = model.eta[:, r] / np.abs(model.eta[:, r]).max()
            phase = "in phase" if ua * ub > 0 else "in opposite phase"
            err = "—" if c.error is None else f"{100 * c.error:+.3g}%"
            p.setTitle(f"2-DOF mode {r + 1}: m_a and m_b {phase} (u_b/u_a = {ub / ua:.3g}) · "
                       f"error {err}, MAC {c.mac:.3f}" if abs(ua) > 1e-12 else
                       f"2-DOF mode {r + 1}: m_a still · error {err}, MAC {c.mac:.3f}", size="10pt")
            p.setXRange(0, n, padding=0.05)
            p.getAxis("bottom").setTicks([[(0, "ground")] + [(i, f"m{i}") for i in range(1, n + 1)]])


class InterfaceFrf(pg.PlotWidget):
    """Receptance at A's tip for a force there: the chain, the 2-DOF model, and A alone."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setLogMode(x=True, y=True)
        self.showGrid(x=True, y=True, alpha=0.3)
        self.setLabel("left", "|X / F|  [m/N]")
        self.setLabel("bottom", "Frequency", units="Hz")
        self.legend = self.addLegend(offset=(-5, 5))

    def apply_theme(self) -> None:
        self.legend.setBrush(colors.legend_brush())

    def set_model(self, model: CoupledModel | None, system: ChainSystem, full: ModalResult) -> None:
        self.clear()
        self.legend.clear()
        if model is None:
            self.setTitle("")
            return
        tip = model.split - 1
        fn = np.array([m.fn_hz for m in full.modes if m.fn_hz > 0])
        ref = np.concatenate([fn, model.fn_hz[model.fn_hz > 0]])
        lo = 0.2 * ref.min() if ref.size else 0.1
        hi = 1.5 * ref.max() if ref.size else 10.0
        f = np.unique(np.concatenate([np.geomspace(lo, hi, 2000), ref[(ref > lo) & (ref < hi)]]))
        self.plot(f, np.abs(frf(system, f, tip)[:, tip]), pen=pg.mkPen(colors.strong, width=2),
                  name=f"chain: x{tip + 1} / F{tip + 1}")
        self.plot(f, np.abs(coupled_frf(model, f)), pen=pg.mkPen(colors.force, width=2, style=DASH),
                  name="2-DOF: u_a / F")
        self.plot(f, np.abs(alone_frf(model, f)), pen=pg.mkPen(sub_color("A"), width=1.5, style=DOT),
                  name="A alone (B removed)")
        for fr in model.fn_hz:
            if lo < fr < hi:
                self.addItem(pg.InfiniteLine(pos=math.log10(fr), angle=90,
                                             pen=pg.mkPen(colors.grey, width=1, style=DOT)))
        self.setTitle(f"Force at m{tip + 1}, A's tip (the interface) · dotted lines: 2-DOF fₙ", size="10pt")
        self.setXRange(math.log10(lo), math.log10(hi), padding=0)


class ModalCouplingPage(QtWidgets.QWidget):
    """Controls, schematic and tables on the left; veering, shapes, FRF and theory on the right."""

    dof_requested = QtCore.Signal(int)  # the user changed N here; the main window owns the system
    edit_parameters = QtCore.Signal()  # the user wants the Simulation page's parameter panel

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.system: ChainSystem | None = None
        self.full: ModalResult | None = None
        self.model: CoupledModel | None = None
        self.comparisons: list[CoupledComparison] = []
        self._dirty = False
        self._split_wanted: int | None = None  # None: the middle of the chain
        self._modes_wanted = [0, 0]

        box = QtWidgets.QGroupBox("Modal coupling")
        form = QtWidgets.QFormLayout(box)
        self.dof = QtWidgets.QSpinBox()
        self.dof.setRange(1, MAX_DOF)
        self.dof.setToolTip("Same chain as the Simulation page; changing it here changes it there too.")
        self.dof.valueChanged.connect(self.dof_requested)
        form.addRow("Number of masses:", self.dof)
        splits = QtWidgets.QWidget()
        splits.setToolTip(SPLIT_TIP)
        self.split_row = QtWidgets.QHBoxLayout(splits)
        self.split_row.setContentsMargins(0, 0, 0, 0)
        self.split_row.setSpacing(3)
        self.split_group = QtWidgets.QButtonGroup(self)
        self.split_group.setExclusive(True)
        self.split_group.idClicked.connect(self._on_split)
        self.split_buttons: list[QtWidgets.QPushButton] = []
        split_label = QtWidgets.QLabel("Split after:")
        split_label.setToolTip(SPLIT_TIP)
        form.addRow(split_label, splits)
        self.mode_a = QtWidgets.QComboBox()
        self.mode_b = QtWidgets.QComboBox()
        for s, combo in enumerate((self.mode_a, self.mode_b)):
            combo.setToolTip("The mode of this subsystem that becomes its oscillator. Or click its row "
                             "in the table below.")
            combo.activated.connect(lambda i, s=s: self._on_mode(s, i))
        self.mode_labels = [QtWidgets.QLabel(), QtWidgets.QLabel()]
        form.addRow(self.mode_labels[0], self.mode_a)
        form.addRow(self.mode_labels[1], self.mode_b)
        self.a_mass = QtWidgets.QComboBox()
        for key in A_MASSES:
            self.a_mass.addItem(A_MASS_NAMES[key][0].upper() + A_MASS_NAMES[key][1:], key)
        self.a_mass.setToolTip(A_MASS_TIP)
        self.a_mass.currentIndexChanged.connect(self.refresh)
        a_mass_label = QtWidgets.QLabel("Mass of A's oscillator:")
        a_mass_label.setToolTip(A_MASS_TIP)
        form.addRow(a_mass_label, self.a_mass)
        self.residual = QtWidgets.QCheckBox("Add B's residual mass to A")
        self.residual.setChecked(True)
        self.residual.setToolTip(RESIDUAL_TIP)
        self.residual.toggled.connect(self.refresh)
        form.addRow(self.residual)
        link = QtWidgets.QLabel(
            "Masses, springs and dampers are edited on the <a href='#sim'>Simulation page</a>."
        )
        mute(link)
        link.linkActivated.connect(lambda _: self.edit_parameters.emit())
        form.addRow(link)

        self.schematic = CouplingSchematic()
        self.summary = QtWidgets.QLabel()
        self.summary.setWordWrap(True)
        self.sub_title = QtWidgets.QLabel("<b>A and B on their own</b> (click a row to couple that mode)")
        self.sub_table = SubsystemTable()
        self.sub_table.cellClicked.connect(self._on_sub_row)
        self.osc_title = QtWidgets.QLabel("<b>Equivalent oscillators</b>")
        self.osc_table = _table(
            ["Oscillator", "m [kg]", "k [N/m]", "c [N·s/m]", "fₙ [Hz]", "ζ"],
            ["The mode each oscillator stands for",
             "m_a: A's modal mass at the interface (or its effective mass), plus B's residual mass if "
             "added. m_b: B's effective mass.",
             "k = m ω²: the oscillator keeps its mode's natural frequency",
             "c = m φᵀCφ = 2ζωm: the oscillator keeps its mode's modal damping ratio",
             "Natural frequency of the oscillator on its own (without the residual mass)",
             "Damping ratio of the oscillator on its own"],
        )
        self.result_title = QtWidgets.QLabel("<b>Coupled: the 2-DOF model against the whole chain</b>")
        self.result_table = _table(
            ["2-DOF\nmode", "fₙ 2-DOF\n[Hz]", "ζ 2-DOF", "Chain\nmode", "fₙ chain\n[Hz]", "ζ chain", "Error", "MAC"],
            ["Mode of the 2-DOF model (ground – k_a – m_a – k_b – m_b)",
             "Natural frequency of the 2-DOF model",
             "Exact damping ratio of the 2-DOF model (from its damped eigenvalues)",
             "The chain's mode whose shape the 2-DOF mode matches best (each 2-DOF mode a different one)",
             "Natural frequency of that mode of the whole chain",
             "Exact damping ratio of that mode of the whole chain",
             "(f_2DOF − f_chain) / f_chain. With the modal mass at the interface and the residual mass "
             "the 2-DOF model is a Rayleigh–Ritz model, so its mode 1 is never below the chain's mode 1, "
             "nor its mode 2 below the chain's mode 2. Paired by shape with higher chain modes, the "
             "error can be negative.",
             "Modal assurance criterion between the chain's mode and the 2-DOF mode drawn on the chain "
             "(Mode shapes tab). Low: the two modes are too far apart for this pairing to mean much."],
        )
        self.result_note = QtWidgets.QLabel()
        self.result_note.setWordWrap(True)
        mute(self.result_note)
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.addWidget(box)
        lv.addWidget(self.schematic)
        lv.addWidget(self.summary)
        lv.addSpacing(6)
        lv.addWidget(self.result_title)
        lv.addWidget(self.result_table)
        lv.addWidget(self.result_note)
        lv.addSpacing(6)
        lv.addWidget(self.osc_title)
        lv.addWidget(self.osc_table)
        lv.addSpacing(6)
        lv.addWidget(self.sub_title)
        lv.addWidget(self.sub_table)
        lv.addStretch(1)
        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidget(left)
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(520)

        self.veering = VeeringPlot()
        self.veering.kind.currentIndexChanged.connect(self._on_sweep_kind)
        self.shapes = ShapePlots()
        self.frf = InterfaceFrf()
        self.theory = QtWidgets.QTextBrowser()
        self.theory.setHtml(THEORY_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.veering, "Veering && splitting")
        self.tabs.addTab(self.shapes, "Mode shapes")
        self.tabs.addTab(self.frf, "Interface FRF")
        self.tabs.addTab(self.theory, "Theory")

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(left_scroll)
        splitter.addWidget(self.tabs)
        splitter.setSizes([620, 1080])
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)
        self.apply_theme()

    def apply_theme(self) -> None:
        """Recolour the labels now; everything drawn from the model is redrawn when the page shows."""
        for name, label in zip("AB", self.mode_labels):
            label.setText(f"<b style='color:{sub_color(name)}'>Mode of {name}:</b>")
        self.veering.apply_theme()
        self.shapes.apply_theme()
        self.frf.apply_theme()
        scroll = self.theory.verticalScrollBar().value()
        self.theory.setHtml(THEORY_HTML)  # its links, in the new link colour
        self.theory.verticalScrollBar().setValue(scroll)
        if self.system is not None:
            self._dirty = True
            if self.isVisible():
                self.refresh()

    # ------------------------------------------------------------ inputs
    @property
    def split(self) -> int:
        """Number of masses in A (split after mass `split`); 0 when the chain cannot be split."""
        return self.split_group.checkedId() + 1

    def set_split(self, split: int) -> None:
        """Split the chain after mass `split` (1-based: A is masses 1..split)."""
        if self.system is None or not 1 <= split <= self.system.n - 1:
            raise ValueError("the split must leave at least one mass on each side")
        self._on_split(split - 1)

    def set_modes(self, mode_a: int, mode_b: int) -> None:
        """Couple A's mode `mode_a` with B's mode `mode_b` (0-based; clipped to what there is)."""
        self._modes_wanted = [mode_a, mode_b]
        self.refresh()

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

    def _on_split(self, i: int) -> None:
        self._split_wanted = i + 1
        self.refresh()

    def _on_mode(self, s: int, index: int) -> None:
        self._modes_wanted[s] = index
        self.refresh()

    def _on_sub_row(self, row: int, _col: int) -> None:
        name, r = self.sub_table.rows[row]
        self._on_mode("AB".index(name), r)

    def _on_sweep_kind(self) -> None:
        if self.model is not None:
            self.veering.set_model(self.model, self.comparisons)

    def _build_split_buttons(self, n: int) -> None:
        for b in self.split_buttons:
            self.split_group.removeButton(b)
            self.split_row.removeWidget(b)
            b.deleteLater()
        self.split_buttons = []
        for i in range(n - 1):
            b = QtWidgets.QPushButton(f"m{i + 1}")
            b.setCheckable(True)
            b.setMinimumWidth(10)
            b.setToolTip(SPLIT_TIP)
            self.split_group.addButton(b, i)
            self.split_row.addWidget(b, 1)
            self.split_buttons.append(b)

    def _sync_controls(self, n: int) -> None:
        if len(self.split_buttons) != n - 1:
            self._build_split_buttons(n)
        if n < 2:
            return
        split = int(np.clip(self._split_wanted or n // 2, 1, n - 1))
        for i, b in enumerate(self.split_buttons):
            b.blockSignals(True)
            b.setChecked(i == split - 1)
            b.setText(f"m{i + 1} ✂" if i == split - 1 else f"m{i + 1}")
            font = b.font()
            font.setBold(i == split - 1)
            b.setFont(font)
            b.blockSignals(False)

    def _fill_mode_combos(self, A: Subsystem, B: Subsystem) -> list[int]:
        chosen = []
        for s, (combo, sub) in enumerate(zip((self.mode_a, self.mode_b), (A, B))):
            r = int(np.clip(self._modes_wanted[s], 0, sub.omegas.size - 1))
            combo.blockSignals(True)
            combo.clear()
            for k, f in enumerate(sub.fn_hz):
                combo.addItem(f"{sub.name}{k + 1}: {f:.4g} Hz")
            combo.setCurrentIndex(r)
            combo.blockSignals(False)
            chosen.append(r)
        return chosen

    # ----------------------------------------------------------- compute
    def refresh(self) -> None:
        if self.system is None:
            return
        self._dirty = False
        system = self.system
        n = system.n
        self.dof.blockSignals(True)
        self.dof.setValue(n)
        self.dof.blockSignals(False)
        self._sync_controls(n)
        full = self.full if self.full is not None else modal_analysis(system)
        self.full = full
        if n < 2:
            self._clear("Modal coupling needs two subsystems: use at least 2 masses.", n)
            return
        split = self.split
        A, B = subsystems(system, split)
        mode_a, mode_b = self._fill_mode_combos(A, B)
        a_mass = self.a_mass.currentData()
        self.sub_table.set_subsystems(A, B, mode_a, mode_b, a_mass)
        try:
            model = coupled_model(system, split, mode_a, mode_b, a_mass, self.residual.isChecked())
        except ValueError as exc:
            self._clear(str(exc), n, split)
            return
        self.model = model
        self.comparisons = comparisons = compare_coupled(model, full)
        self.schematic.set_model(n, split, model)
        self.summary.setText(self._summary(model))
        self._fill_oscillators(model)
        self.result_table.setRowCount(2)
        for r, c in enumerate(comparisons):
            color = colors.mode[(c.full_index - 1) % len(colors.mode)]
            _set_row(self.result_table, r, [
                (str(c.index), None),
                (_g(c.fn), None),
                (zeta_text(c.zeta), None),
                (f"mode {c.full_index}", color),
                (_g(c.fn_full), None),
                (zeta_text(c.zeta_full), None),
                ("—" if c.error is None else f"{100 * c.error:+.3g}%",
                 None if c.error is None else error_color(c.error)),
                (f"{c.mac:.3f}", colors.poor if c.mac < 0.8 else None),
            ])
        _fit_table(self.result_table)
        self.result_note.setText(self._result_note(model, full, comparisons))
        self.veering.set_model(model, comparisons)
        self.shapes.set_model(model, full, comparisons)
        self.frf.set_model(model, system, full)

    def _clear(self, reason: str, n: int, split: int = 0) -> None:
        self.model, self.comparisons = None, []
        self.schematic.set_model(n, split if split else n, None)
        self.summary.setText(f"<b style='color:{colors.poor}'>{reason}</b>")
        self.osc_table.setRowCount(0)
        self.result_table.setRowCount(0)
        self.result_note.setText("")
        if not split:
            self.sub_table.setRowCount(0)
            for combo in (self.mode_a, self.mode_b):
                combo.clear()
        self.veering.clear(reason)
        self.shapes.set_model(None, self.full, [])
        self.frf.set_model(None, self.system, self.full)

    def _fill_oscillators(self, model: CoupledModel) -> None:
        self.osc_table.setRowCount(2)
        a, b = model.osc_a, model.osc_b
        res = f" + {model.m_residual:.3g}" if model.residual and model.m_residual > 1e-9 else ""
        for r, (name, mode, osc, m_text) in enumerate(
                (("A", model.mode_a, a, _g(a.m, 3) + res), ("B", model.mode_b, b, _g(b.m, 3)))):
            _set_row(self.osc_table, r, [
                (f"{name}{mode + 1}", sub_color(name)),
                (m_text, None),
                (_g(osc.k, 4), None),
                (_g(osc.c, 3), None),
                (_g(osc.fn_hz), None),
                (zeta_text(osc.zeta) if osc.k > 0 else "—", None),
            ])
        _fit_table(self.osc_table)

    @staticmethod
    def _summary(model: CoupledModel) -> str:
        mu, ratio = model.mass_ratio, model.freq_ratio
        f1, f2 = model.fn_hz
        fa, fb = model.osc_a.fn_hz, model.osc_b.fn_hz
        tuned = ""
        if np.isfinite(ratio) and ratio > 0:
            detune = abs(math.log(ratio))
            if detune < math.sqrt(mu):
                tuned = (f"f<sub>B</sub>/f<sub>A</sub> is within about √μ of 1, so the modes are "
                         "<b>strongly coupled</b>: each coupled mode is a mix of A's and B's, and they split "
                         "apart either side.")
            else:
                tuned = ("The two modes are far apart compared with √μ, so they are <b>weakly coupled</b>: "
                         "each coupled mode is mostly A's or B's, shifted a little.")
        return (
            f"A (m1–m{model.split}) mode {model.mode_a + 1} at {fa:.4g} Hz and B "
            f"(m{model.split + 1}–m{model.system.n}) mode {model.mode_b + 1} at {fb:.4g} Hz, "
            f"f<sub>B</sub>/f<sub>A</sub> = <b>{ratio:.3g}</b>, mass ratio "
            f"μ = m<sub>b</sub>/m<sub>a</sub> = <b>{mu:.3g}</b>. Joined, they become <b>{f1:.4g}</b> and "
            f"<b>{f2:.4g} Hz</b>. {tuned}"
        )

    @staticmethod
    def _result_note(model: CoupledModel, full: ModalResult, comparisons: list[CoupledComparison]) -> str:
        if model.rayleigh_ritz:
            note = ("With the modal mass at the interface and B's residual mass, the 2-DOF model is a "
                    "Rayleigh–Ritz model of the chain (A's mode shape, and B's on a fixed base), so its "
                    "frequencies are upper bounds of the chain's lowest two")
            if [c.full_index for c in comparisons] == [1, 2]:
                note += ". "
            else:
                note += (f" ({model.fn_hz[0]:.4g} ≥ {full.modes[0].fn_hz:.4g} Hz and {model.fn_hz[1]:.4g} ≥ "
                         f"{full.modes[1].fn_hz:.4g} Hz), not of the modes they are paired with here. ")
        else:
            note = ("Without " + ("B's residual mass" if model.a_mass == "tip" else "A's modal mass at the "
                                  "interface") + " the 2-DOF model is no longer a Rayleigh–Ritz model, so "
                    "its errors can have either sign. ")
        if min(c.mac for c in comparisons) < 0.8:
            note += ("A MAC well below 1: the other modes of A or B, which the 2-DOF model leaves out, "
                     "take part in that chain mode. ")
        return note + "The chain's other modes come from the modes of A and B that are not coupled here."
