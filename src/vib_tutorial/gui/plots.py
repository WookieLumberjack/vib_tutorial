"""Strip charts, mode-shape plot, modal table, and frequency-response plot."""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core import ChainSystem, ModalResult, frf
from .style import FORCE_COLOR, MASS_COLORS, MODE_COLORS


class TimeHistoryPlot(pg.GraphicsLayoutWidget):
    """Displacement of every mass and the applied force vs. time."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.x_plot = self.addPlot(row=0, col=0)
        self.x_plot.setLabel("left", "Displacement", units="m")
        self.x_plot.addLegend(offset=(5, 5), colCount=4)
        self.f_plot = self.addPlot(row=1, col=0)
        self.f_plot.setLabel("left", "Force", units="N")
        self.f_plot.setLabel("bottom", "Time", units="s")
        self.f_plot.setXLink(self.x_plot)
        self.ci.layout.setRowStretchFactor(0, 3)
        self.ci.layout.setRowStretchFactor(1, 1)
        # Performance: these plots scroll and redraw every frame. No grid (its
        # lines cost more than the data) and 1 px pens (Qt's fast path; wider
        # antialiased lines were ~2x slower, and much worse on Retina displays).
        for p in (self.x_plot, self.f_plot):
            p.setMouseEnabled(x=False, y=True)
            p.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen("#bbb", width=1)))
        self.f_curve = self.f_plot.plot(pen=pg.mkPen(FORCE_COLOR, width=1))
        self.auto_range = True
        self.x_curves: list[pg.PlotDataItem] = []

    def set_dof(self, n: int) -> None:
        for c in self.x_curves:
            self.x_plot.removeItem(c)
        self.x_curves = [
            self.x_plot.plot(pen=pg.mkPen(MASS_COLORS[i], width=1), name=f"x{i + 1}") for i in range(n)
        ]

    def set_auto_range(self, on: bool) -> None:
        """Auto-fit the y axes each frame, or freeze them at their current range."""
        self.auto_range = on
        for p in (self.x_plot, self.f_plot):
            if on:
                p.enableAutoRange(axis="y")
            else:
                p.disableAutoRange(axis="y")

    def update_data(self, t: np.ndarray, x: np.ndarray, f: np.ndarray, window: float) -> None:
        # Hand Qt at most MAX_POINTS per curve: drawing tens of thousands of
        # antialiased segments every frame is what made the app lag.
        idx = decimation_index(np.column_stack([x, f]), MAX_POINTS)
        td = t[idx]
        for i, c in enumerate(self.x_curves):
            c.setData(td, x[idx, i])
        self.f_curve.setData(td, f[idx])
        t_end = t[-1] if t.size else 0.0
        self.x_plot.setXRange(max(0.0, t_end - window), max(window, t_end), padding=0)
        if self.auto_range:
            # Auto-fit, but never zoom in below +/-MIN_Y_SPAN (same reason as
            # MIN_AUTO_PEAK in the animation: decaying motion -> float noise).
            peak = float(np.abs(x).max()) if x.size else 0.0
            if peak < MIN_Y_SPAN:
                self.x_plot.setYRange(-MIN_Y_SPAN, MIN_Y_SPAN, padding=0.05)
            else:
                self.x_plot.enableAutoRange(axis="y")


MAX_POINTS = 3000
MIN_Y_SPAN = 1e-6  # m


def decimation_index(y: np.ndarray, max_points: int) -> np.ndarray:
    """Indices that keep the min and max of y within each bucket.

    Preserves the visual envelope of oscillations (unlike plain striding, which
    can alias). With multiple columns, the extremes of every column are kept.
    """
    k = y.shape[0]
    if k <= max_points:
        return np.arange(k)
    blocks_per_sample = 1 + 2 * (y.shape[1] if y.ndim > 1 else 1)  # start + min + max per column
    bucket = int(math.ceil(blocks_per_sample * k / max_points))
    m = k // bucket
    blocks = y[: m * bucket].reshape(m, bucket, -1)
    base = (np.arange(m) * bucket)[:, None]
    picks = [base, base + blocks.argmin(axis=1), base + blocks.argmax(axis=1)]
    idx = np.concatenate([p.ravel() for p in picks] + [np.arange(m * bucket, k)])
    return np.unique(idx)


class ModalTable(QtWidgets.QTableWidget):
    HEADERS = ["Mode", "fₙ [Hz]", "ζ modal", "ζ exact", "f_d [Hz]"]

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(0, len(self.HEADERS), parent)
        self.setHorizontalHeaderLabels(self.HEADERS)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        tips = [
            "Mode number (ordered by undamped natural frequency)",
            "Undamped natural frequency: Kφ = ω²Mφ",
            "Modal damping ratio φᵀCφ / (2ωₙ) using the undamped (mass-normalized) mode shape. "
            "Exact only for proportional damping.",
            "Exact damping ratio from the complex eigenvalue λ of the state matrix: -Re(λ)/|λ|",
            "Damped natural frequency Im(λ)/2π",
        ]
        for col, tip in enumerate(tips):
            self.horizontalHeaderItem(col).setToolTip(tip)

    def set_result(self, result: ModalResult) -> None:
        selected = self.currentRow()
        self.setRowCount(len(result.modes))
        for r, mode in enumerate(result.modes):
            d = mode.damped
            cells = [
                str(mode.index),
                f"{mode.fn_hz:.4g}",
                "rigid" if math.isnan(mode.zeta_modal) else f"{mode.zeta_modal:.4f}",
                f"{d.zeta:.4f}" if d else "overdamped",
                f"{d.fd_hz:.4g}" if d else "—",
            ]
            for c, text in enumerate(cells):
                item = QtWidgets.QTableWidgetItem(text)
                item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                if c == 0:
                    item.setForeground(pg.mkColor(MODE_COLORS[r % len(MODE_COLORS)]))
                self.setItem(r, c, item)
        if 0 <= selected < len(result.modes):
            self.selectRow(selected)
        # Size to fit the rows so the mode-shape plot gets the spare space.
        height = self.horizontalHeader().height() + 2 * self.frameWidth()
        height += sum(self.rowHeight(r) for r in range(self.rowCount()))
        self.setFixedHeight(height)


class ModeShapePlot(pg.PlotWidget):
    """Undamped mode shapes vs. position along the chain (0 = ground)."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setLabel("bottom", "Position along chain (0 = ground)")
        self.setLabel("left", "Normalized amplitude")
        self.showGrid(x=True, y=True, alpha=0.3)
        self.setYRange(-1.1, 1.1)
        self.setMouseEnabled(x=False, y=False)
        self.legend = self.addLegend(offset=(5, -5), brush=pg.mkBrush(255, 255, 255, 210), colCount=2)
        self.curves: list[pg.PlotDataItem] = []
        self.shapes: list[np.ndarray] = []
        self.highlight: int | None = None

    def set_result(self, result: ModalResult) -> None:
        self.legend.clear()
        for c in self.curves:
            self.removeItem(c)
        n = len(result.modes)
        self.shapes = [np.concatenate([[0.0], m.shape]) for m in result.modes]
        self.curves = []
        for r, mode in enumerate(result.modes):
            color = MODE_COLORS[r % len(MODE_COLORS)]
            curve = self.plot(
                pen=pg.mkPen(color, width=2),
                symbol="s",
                symbolBrush=color,
                symbolPen=None,
                symbolSize=9,
                name=f"Mode {mode.index}: {mode.fn_hz:.3g} Hz",
            )
            self.curves.append(curve)
        self.setXRange(0, n, padding=0.05)
        self.getAxis("bottom").setTicks([[(0, "ground")] + [(i, f"m{i}") for i in range(1, n + 1)]])
        self.set_highlight(self.highlight)
        self.animate(1.0)

    def set_highlight(self, r: int | None) -> None:
        self.highlight = r if r is not None and r < len(self.curves) else None
        for i, c in enumerate(self.curves):
            color = pg.mkColor(MODE_COLORS[i % len(MODE_COLORS)])
            dim = self.highlight is not None and i != self.highlight
            if dim:
                color.setAlpha(50)
            c.setPen(pg.mkPen(color, width=4 if i == self.highlight else 2))
            c.setSymbolBrush(color)

    def animate(self, factor: float) -> None:
        """Scale every shape by factor (e.g. cos(wt)) to show it oscillating."""
        xs = np.arange(len(self.shapes[0])) if self.shapes else []
        for c, s in zip(self.curves, self.shapes):
            c.setData(xs, factor * s)


class FrfPlot(pg.GraphicsLayoutWidget):
    """Receptance |X_i / F| and phase for a force applied at the target mass."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.mag = self.addPlot(row=0, col=0)
        self.mag.setLogMode(x=True, y=True)
        self.mag.setLabel("left", "|X / F|  [m/N]")
        self.mag.showGrid(x=True, y=True, alpha=0.3)
        self.mag.addLegend(offset=(-5, 5), colCount=2)
        self.phase = self.addPlot(row=1, col=0)
        self.phase.setLogMode(x=True, y=False)
        self.phase.setLabel("left", "Phase", units="deg")
        self.phase.setLabel("bottom", "Frequency", units="Hz")
        self.phase.showGrid(x=True, y=True, alpha=0.3)
        self.phase.setXLink(self.mag)
        self.ci.layout.setRowStretchFactor(0, 2)
        self.ci.layout.setRowStretchFactor(1, 1)
        self.mag_curves: list[pg.PlotDataItem] = []
        self.phase_curves: list[pg.PlotDataItem] = []
        self.mode_lines: list[pg.InfiniteLine] = []
        self.drive_lines = [
            pg.InfiniteLine(angle=90, pen=pg.mkPen(FORCE_COLOR, width=2, style=QtCore.Qt.PenStyle.DashLine))
            for _ in range(2)
        ]
        self.mag.addItem(self.drive_lines[0])
        self.phase.addItem(self.drive_lines[1])

    def set_system(self, system: ChainSystem, result: ModalResult, input_dof: int) -> None:
        freqs_n = np.array([m.fn_hz for m in result.modes if m.fn_hz > 0])
        lo = 0.2 * freqs_n.min() if freqs_n.size else 0.1
        hi = 2.5 * freqs_n.max() if freqs_n.size else 10.0
        f = np.geomspace(lo, hi, 1500)
        # Include the damped peaks so lightly damped resonances are not clipped.
        peaks = [m.damped.fd_hz for m in result.modes if m.damped is not None and lo < m.damped.fd_hz < hi]
        f = np.unique(np.concatenate([f, peaks, freqs_n]))
        H = frf(system, f, input_dof)

        for c in self.mag_curves + self.mode_lines:
            self.mag.removeItem(c)
        for c in self.phase_curves:
            self.phase.removeItem(c)
        self.mag.legend.clear()
        self.mag_curves, self.phase_curves = [], []
        for i in range(system.n):
            pen = pg.mkPen(MASS_COLORS[i], width=2)
            self.mag_curves.append(self.mag.plot(f, np.abs(H[:, i]), pen=pen, name=f"x{i + 1}"))
            ph = np.degrees(np.unwrap(np.angle(H[:, i])))
            self.phase_curves.append(self.phase.plot(f, ph, pen=pen))

        self.mode_lines = []
        for r, fn in enumerate(freqs_n):
            line = pg.InfiniteLine(
                pos=math.log10(fn),
                angle=90,
                pen=pg.mkPen(MODE_COLORS[r % len(MODE_COLORS)], width=1, style=QtCore.Qt.PenStyle.DotLine),
            )
            self.mag.addItem(line)
            self.mode_lines.append(line)
        self.mag.setTitle(f"Force at m{input_dof + 1} · dotted: fₙ · dashed: drive", size="10pt")
        self.mag.setXRange(math.log10(lo), math.log10(hi), padding=0)

    def set_drive(self, freq_hz: float | None) -> None:
        for line in self.drive_lines:
            line.setVisible(freq_hz is not None and freq_hz > 0)
            if freq_hz:
                line.setValue(math.log10(freq_hz))
