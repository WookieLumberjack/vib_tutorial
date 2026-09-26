"""Strip charts, mode-shape plot, modal table, and frequency-response plot."""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ..core import ChainSystem, ModalResult, frf
from .modes import Method, ModeEntry, time_constant_text
from .style import FORCE_COLOR, MASS_COLORS, MODE_COLORS


class TimeHistoryPlot(pg.GraphicsLayoutWidget):
    """Displacements (of every mass, or of every mode) and the applied force vs. time."""

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
        """One curve per mass displacement."""
        self.set_curves("Displacement", [(f"x{i + 1}", MASS_COLORS[i]) for i in range(n)])

    def set_curves(self, label: str, curves: list[tuple[str, str]]) -> None:
        """Replace the upper plot's curves with one per (legend name, color)."""
        for c in self.x_curves:
            self.x_plot.removeItem(c)
        self.x_plot.setLabel("left", label, units="m")
        self.x_curves = [self.x_plot.plot(pen=pg.mkPen(color, width=1), name=name) for name, color in curves]

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
    CLASSICAL_HEADERS = ["Mode", "fₙ [Hz]", "ζ modal", "ζ exact", "f_d [Hz]"]
    CLASSICAL_TIPS = [
        "Mode number (ordered by undamped natural frequency)",
        "Undamped natural frequency: Kφ = ω²Mφ",
        "Modal damping ratio φᵀCφ / (2ωₙ) using the undamped (mass-normalized) mode shape. "
        "Exact only for proportional damping.",
        "Exact damping ratio from the complex eigenvalue λ of the state matrix: -Re(λ)/|λ|",
        "Damped natural frequency Im(λ)/2π",
    ]
    STATE_SPACE_HEADERS = ["λ #", "λ = σ ± iω_d [1/s]", "|λ|/2π [Hz]", "ζ", "f_d [Hz]"]
    STATE_SPACE_TIPS = [
        "Eigenvalue number: all 2N eigenvalues of the state matrix A, ordered by |λ|. "
        "λ* marks the complex conjugate of another row.",
        "Eigenvalue of A. Re(λ) = σ is the decay rate, Im(λ) = ±ω_d the oscillation frequency. "
        "Oscillatory eigenvalues come in conjugate pairs; real ones are non-oscillatory.",
        "Natural frequency ωₙ = |λ| (for a real eigenvalue this is only a decay rate)",
        "Damping ratio -Re(λ)/|λ|, exact for any damping",
        "Damped frequency |Im(λ)|/2π. For a real eigenvalue: its decay time constant τ = -1/λ",
    ]
    MAX_VISIBLE_ROWS = 10  # the 2N table scrolls beyond this

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(0, len(self.CLASSICAL_HEADERS), parent)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.method: Method | None = None

    def set_result(self, result: ModalResult, method: Method, entries: list[ModeEntry]) -> None:
        if method is not self.method:
            self.method = method
            self.clearSelection()
            self.setCurrentCell(-1, -1)
            classical = method is Method.CLASSICAL
            headers = self.CLASSICAL_HEADERS if classical else self.STATE_SPACE_HEADERS
            tips = self.CLASSICAL_TIPS if classical else self.STATE_SPACE_TIPS
            self.setHorizontalHeaderLabels(headers)
            for col, tip in enumerate(tips):
                self.horizontalHeaderItem(col).setToolTip(tip)
        rows = self._classical_rows(result) if method is Method.CLASSICAL else self._state_space_rows(result)
        selected = self.currentRow() if self.selectedItems() else -1
        self.setRowCount(len(rows))
        for r, (cells, entry) in enumerate(zip(rows, entries)):
            for c, text in enumerate(cells):
                item = QtWidgets.QTableWidgetItem(text)
                item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                if c == 0:
                    item.setForeground(pg.mkColor(entry.color))
                self.setItem(r, c, item)
        if 0 <= selected < len(rows):
            self.selectRow(selected)
        # Size to fit the rows so the mode-shape plot gets the spare space.
        height = self.horizontalHeader().height() + 2 * self.frameWidth()
        height += sum(self.rowHeight(r) for r in range(min(self.rowCount(), self.MAX_VISIBLE_ROWS)))
        self.setFixedHeight(height)

    @staticmethod
    def _classical_rows(result: ModalResult) -> list[list[str]]:
        rows = []
        for mode in result.modes:
            d = mode.damped
            rows.append([
                str(mode.index),
                f"{mode.fn_hz:.4g}",
                "rigid" if math.isnan(mode.zeta_modal) else f"{mode.zeta_modal:.4f}",
                f"{d.zeta:.4f}" if d else "overdamped",
                f"{d.fd_hz:.4g}" if d else "—",
            ])
        return rows

    @staticmethod
    def _state_space_rows(result: ModalResult) -> list[list[str]]:
        rows = []
        for m in result.complex_modes:
            lam = m.eigenvalue
            if m.is_oscillatory:
                conj = m.conjugate < m.index
                rows.append([
                    f"{m.index} = λ{m.conjugate}*" if conj else str(m.index),
                    f"{lam.real:.4g} {'−' if lam.imag < 0 else '+'} {abs(lam.imag):.4g}i",
                    f"{m.fn_hz:.4g}",
                    f"{m.zeta:.4f}",
                    f"{m.fd_hz:.4g}",
                ])
            else:
                rows.append([str(m.index), f"{lam.real:.4g}", "—", "real", time_constant_text(lam.real)])
        return rows


class ModeShapePlot(pg.PlotWidget):
    """Mode shapes vs. position along the chain (0 = ground).

    Complex shapes are drawn as Re(ψ e^{±iθ}), which is the physical motion
    over one cycle; an envelope ±|ψ| can be shown for the highlighted shape.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setLabel("bottom", "Position along chain (0 = ground)")
        self.setLabel("left", "Normalized amplitude")
        self.showGrid(x=True, y=True, alpha=0.3)
        self.setYRange(-1.1, 1.1)
        self.setMouseEnabled(x=False, y=False)
        self.legend = self.addLegend(offset=(5, -5), brush=pg.mkBrush(255, 255, 255, 210), colCount=2)
        self.curves: list[pg.PlotDataItem] = []
        self.entries: list[ModeEntry] = []
        self.highlight: int | None = None
        self.show_envelope = False
        envelope_pen = pg.mkPen("#888", width=1, style=QtCore.Qt.PenStyle.DashLine)
        self.envelope = [self.plot(pen=envelope_pen) for _ in range(2)]

    def set_entries(self, entries: list[ModeEntry], show_envelope: bool = False) -> None:
        self.legend.clear()
        for c in self.curves:
            self.removeItem(c)
        self.entries = entries
        self.show_envelope = show_envelope
        n = len(entries[0].shape) if entries else 0
        self.curves = []
        for e in entries:
            curve = self.plot(
                pen=pg.mkPen(e.color, width=2),
                symbol="s",
                symbolBrush=e.color,
                symbolPen=None,
                symbolSize=9,
            )
            if e.legend:
                self.legend.addItem(curve, e.legend)
            self.curves.append(curve)
        self.setXRange(0, n, padding=0.05)
        self.getAxis("bottom").setTicks([[(0, "ground")] + [(i, f"m{i}") for i in range(1, n + 1)]])
        self.set_highlight(self.highlight)
        self.animate(0.0)

    def set_highlight(self, r: int | None) -> None:
        self.highlight = r if r is not None and r < len(self.curves) else None
        for i, (c, e) in enumerate(zip(self.curves, self.entries)):
            color = pg.mkColor(e.color)
            dim = self.highlight is not None and i != self.highlight
            if dim:
                color.setAlpha(50)
            c.setPen(pg.mkPen(color, width=4 if i == self.highlight else 2))
            c.setSymbolBrush(color)
        if self.show_envelope and self.highlight is not None:
            mag = np.concatenate([[0.0], np.abs(self.entries[self.highlight].shape)])
            xs = np.arange(mag.size)
            for sign, c in zip((1, -1), self.envelope):
                c.setData(xs, sign * mag)
        else:
            for c in self.envelope:
                c.setData([], [])

    def animate(self, theta: float) -> None:
        """Draw every shape at phase theta of its cycle (theta = 0: the shape itself)."""
        if not self.entries:
            return
        xs = np.arange(len(self.entries[0].shape) + 1)
        for c, e in zip(self.curves, self.entries):
            c.setData(xs, np.concatenate([[0.0], rotate(e, theta).real]))


def rotate(entry: ModeEntry, theta: float) -> np.ndarray:
    """ψ e^{±iθ}: a mode's complex amplitudes at phase theta (static if non-oscillatory)."""
    return entry.shape * complex(math.cos(theta), entry.spin * math.sin(theta))


class PhasorPlot(pg.PlotWidget):
    """Complex-plane view of one mode shape: one arrow ψᵢ per mass.

    Each arrow rotates at the mode's frequency (anticlockwise for Im λ > 0,
    clockwise for its conjugate); its projection on the real axis is that
    mass's displacement. Arrows on one line (0° / 180°) mean a real mode;
    spread-out arrows mean the masses peak at different times.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAspectLocked(True)
        self.setRange(xRange=(-1.15, 1.15), yRange=(-1.15, 1.15), padding=0)
        self.setMouseEnabled(x=False, y=False)
        self.hideButtons()
        self.setLabel("bottom", "Re")
        self.setLabel("left", "Im")
        self.setTitle("Complex plane", size="9pt")
        circle = np.exp(1j * np.linspace(0, 2 * np.pi, 97))
        faint = pg.mkPen("#bbb", width=1)
        self.plot(circle.real, circle.imag, pen=pg.mkPen("#bbb", width=1, style=QtCore.Qt.PenStyle.DotLine))
        self.addItem(pg.InfiniteLine(pos=0, angle=0, pen=faint))
        self.addItem(pg.InfiniteLine(pos=0, angle=90, pen=faint))
        self.arrows: list[pg.PlotDataItem] = []
        self.tips = pg.ScatterPlotItem(size=8, pen=None)
        self.projections = pg.ScatterPlotItem(size=7, symbol="o", brush=None)
        self.addItem(self.tips)
        self.addItem(self.projections)
        self.entries: list[ModeEntry] = []
        self.highlight: int | None = None
        self._brushes: list[QtGui.QBrush] = []
        self._pens: list[QtGui.QPen] = []

    def set_entries(self, entries: list[ModeEntry]) -> None:
        self.entries = entries
        n = len(entries[0].shape) if entries else 0
        while len(self.arrows) < n:
            self.arrows.append(self.plot(pen=pg.mkPen(MASS_COLORS[len(self.arrows)], width=2)))
        for i, a in enumerate(self.arrows):
            a.setVisible(i < n)
        self._brushes = [pg.mkBrush(MASS_COLORS[i]) for i in range(n)]
        self._pens = [pg.mkPen(MASS_COLORS[i], width=2) for i in range(n)]
        self.set_highlight(self.highlight)

    def set_highlight(self, r: int | None) -> None:
        self.highlight = r if r is not None and r < len(self.entries) else None
        entry = self._entry()
        self.setTitle(f"Complex plane: {entry.key}" if entry else "Complex plane", size="9pt")
        self.animate(0.0)

    def _entry(self) -> ModeEntry | None:
        """The highlighted entry, or the first one when nothing is selected."""
        if not self.entries:
            return None
        return self.entries[self.highlight if self.highlight is not None else 0]

    def animate(self, theta: float) -> None:
        entry = self._entry()
        if entry is None:
            return
        z = rotate(entry, theta)
        for a, zi in zip(self.arrows, z):
            a.setData([0.0, zi.real], [0.0, zi.imag])
        self.tips.setData(z.real, z.imag, brush=self._brushes)
        self.projections.setData(z.real, np.zeros(z.size), brush=None, pen=self._pens)


class PhasorPanel(QtWidgets.QWidget):
    """The complex-plane plot (kept square) above a table of |ψᵢ| and phase per mass."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.plot = PhasorPlot()
        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Mass", "|ψ|", "Phase"])
        self.table.horizontalHeaderItem(2).setToolTip(
            "Phase of each mass relative to the largest one. 0° / 180° everywhere means a real "
            "mode; anything else means the masses peak at different times."
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot)
        layout.addWidget(self.table, 1)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        # Keep the aspect-locked plot square so the unit circle fills it.
        self.plot.setFixedHeight(max(120, min(self.width(), self.height() - 80)))
        super().resizeEvent(event)

    def set_entries(self, entries: list[ModeEntry]) -> None:
        self.plot.set_entries(entries)
        self._fill_table()

    def set_highlight(self, r: int | None) -> None:
        self.plot.set_highlight(r)
        self._fill_table()

    def animate(self, theta: float) -> None:
        self.plot.animate(theta)

    def _fill_table(self) -> None:
        entry = self.plot._entry()
        shape = entry.shape if entry else np.zeros(0)
        self.table.setRowCount(shape.size)
        for i, z in enumerate(shape):
            phase = round(math.degrees(math.atan2(z.imag, z.real)), 1) + 0.0  # no "-0.0"
            if phase == -180.0:
                phase = 180.0
            cells = [f"m{i + 1}", f"{abs(z):.3f}", f"{phase:+.1f}°"]
            for c, text in enumerate(cells):
                item = QtWidgets.QTableWidgetItem(text)
                item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                if c == 0:
                    item.setForeground(pg.mkColor(MASS_COLORS[i]))
                self.table.setItem(i, c, item)


def frequency_grid(result: ModalResult, points: int) -> np.ndarray:
    """Log-spaced FRF frequencies [Hz] spanning the modes, plus every peak."""
    freqs_n = np.array([m.fn_hz for m in result.modes])
    # A rigid-body mode's frequency is 0 up to rounding; H is singular there.
    freqs_n = freqs_n[freqs_n > 1e-6 * max(freqs_n.max(), 1e-300)]
    lo = 0.2 * freqs_n.min() if freqs_n.size else 0.1
    hi = 2.5 * freqs_n.max() if freqs_n.size else 10.0
    f = np.geomspace(lo, hi, points)
    # Include the damped peaks so lightly damped resonances are not clipped.
    peaks = [m.damped.fd_hz for m in result.modes if m.damped is not None and lo < m.damped.fd_hz < hi]
    return np.unique(np.concatenate([f, peaks, freqs_n]))


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
        f = frequency_grid(result, 1500)
        lo, hi = f[0], f[-1]
        freqs_n = np.array([m.fn_hz for m in result.modes if lo < m.fn_hz])
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
