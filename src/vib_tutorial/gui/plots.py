"""Strip charts, mode-shape plot, modal table, and frequency-response plot."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ..core import ChainSystem, ModalResult, frf, modal_analysis, transmissibility
from .axes import log_axes
from .modes import Method, ModeEntry, time_constant_text
from .style import colors

MAX_POINTS = 3000
MIN_Y_SPAN = 1e-6  # m


class TimeHistoryPlot(pg.GraphicsLayoutWidget):
    """Displacements (of every mass or mode), or energies, and the applied force vs. time."""

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
        self.zero_lines = []
        for p in (self.x_plot, self.f_plot):
            p.setMouseEnabled(x=False, y=True)
            self.zero_lines.append(pg.InfiniteLine(pos=0, angle=0))
            p.addItem(self.zero_lines[-1])
        self.f_curve = self.f_plot.plot()
        self.auto_range = True
        self.min_span = MIN_Y_SPAN
        self.x_curves: list[pg.PlotDataItem] = []
        self.decimator = Decimator()
        self.apply_theme()

    def apply_theme(self) -> None:
        """Zero lines and the force curve; the upper curves are set with set_curves."""
        for line in self.zero_lines:
            line.setPen(pg.mkPen(colors.faint, width=1))
        self.f_curve.setPen(pg.mkPen(colors.force, width=1))

    def set_dof(self, n: int) -> None:
        """One curve per mass displacement."""
        self.set_curves("Displacement", [(f"x{i + 1}", colors.mass[i]) for i in range(n)])

    def set_curves(
        self, label: str, curves: list[tuple[str, str]], units: str = "m", min_span: float = MIN_Y_SPAN
    ) -> None:
        """Replace the upper plot's curves with one per (legend name, color).

        Auto-ranging never zooms in below +/-min_span (in `units`).
        """
        for c in self.x_curves:
            self.x_plot.removeItem(c)
        self.x_plot.setLabel("left", label, units=units)
        self.min_span = min_span
        self.decimator.reset()  # new curves: the cached picks were for other values
        self.x_curves = [self.x_plot.plot(pen=pg.mkPen(color, width=1), name=name) for name, color in curves]
        # pyqtgraph sizes the legend from its items' current widths, which lag behind
        # new, longer names and squeeze the entries together; use the preferred size.
        legend = self.x_plot.legend
        size = legend.layout.effectiveSizeHint(QtCore.Qt.SizeHint.PreferredSize)
        legend.setGeometry(0, 0, size.width(), size.height())

    def set_input(self, base: bool) -> None:
        """Label the lower plot for a force (N) or a ground displacement (m)."""
        if base:
            self.f_plot.setLabel("left", "Ground x_g", units="m")
        else:
            self.f_plot.setLabel("left", "Force", units="N")

    def set_auto_range(self, on: bool) -> None:
        """Auto-fit the y axes each frame, or freeze them at their current range."""
        self.auto_range = on
        for p in (self.x_plot, self.f_plot):
            if on:
                p.enableAutoRange(axis="y")
            else:
                p.disableAutoRange(axis="y")

    def update_data(
        self, t: np.ndarray, values: Callable[[slice | np.ndarray], np.ndarray], window: float, start: int = 0
    ) -> None:
        """Plot the samples t, numbered from `start` (History.offset numbering).

        values(index) returns the plotted rows at index (a slice or index array into t):
        one column per curve, then the force. Only the decimated rows are derived each
        frame, and the decimation itself only scans samples it hasn't seen before.
        """
        # Hand Qt at most MAX_POINTS per curve: drawing tens of thousands of
        # antialiased segments every frame is what made the app lag.
        idx = self.decimator.index(start, t.size, values, len(self.x_curves) + 1)
        yd = values(idx)
        td = t[idx]
        for i, c in enumerate(self.x_curves):
            c.setData(td, yd[:, i])
        self.f_curve.setData(td, yd[:, -1])
        t_end = t[-1] if t.size else 0.0
        self.x_plot.setXRange(max(0.0, t_end - window), max(window, t_end), padding=0)
        if self.auto_range:
            # Auto-fit, but never zoom in below +/-min_span (same reason as
            # MIN_AUTO_PEAK in the animation: decaying motion -> float noise).
            # The decimation keeps every extreme, so yd has the window's peak.
            peak = float(np.abs(yd[:, :-1]).max()) if yd.size else 0.0
            if peak < self.min_span:
                self.x_plot.setYRange(-self.min_span, self.min_span, padding=0.05)
            else:
                self.x_plot.enableAutoRange(axis="y")


def decimation_index(y: np.ndarray, max_points: int) -> np.ndarray:
    """Indices that keep the min and max of y within each bucket.

    Preserves the visual envelope of oscillations (unlike plain striding, which
    can alias). With multiple columns, the extremes of every column are kept.
    """
    y = y.reshape(len(y), -1)
    return Decimator(max_points).index(0, len(y), y.__getitem__, y.shape[1])


class Decimator:
    """decimation_index for a record that grows at its end and loses samples at its start.

    Samples are grouped in buckets aligned to their absolute numbers, and each whole
    bucket's picks are cached, so a frame only scans the samples that are new since the
    last one plus the two partial buckets at the window's edges. Call reset() when the
    values of samples already seen change (other curves, another modal map).
    """

    def __init__(self, max_points: int = MAX_POINTS) -> None:
        self.max_points = max_points
        self.bucket = 0
        self.reset()

    def reset(self) -> None:
        self.first = 0  # absolute number of the first cached bucket
        # One row per whole bucket: its first sample, then the argmin and argmax of
        # every column, as absolute sample numbers.
        self.picks = np.empty((0, 0), dtype=np.intp)

    def index(
        self, start: int, k: int, values: Callable[[slice | np.ndarray], np.ndarray], columns: int
    ) -> np.ndarray:
        """Sorted indices into a window of k samples numbered from start.

        values(slice(i, j)) returns the (j - i, columns) values of window samples i..j-1.
        """
        if k <= self.max_points:
            return np.arange(k)
        picks_per_bucket = 1 + 2 * columns
        needed = math.ceil(picks_per_bucket * k / self.max_points)
        # Keep the bucket size while the window grows or shrinks a little, so the
        # cache survives; refit with some headroom when it drifts out of range.
        if not needed <= self.bucket <= 1.5 * needed or self.picks.shape[1:] != (picks_per_bucket,):
            self.bucket = math.ceil(1.2 * needed)
            self.reset()
            self.picks = np.empty((0, picks_per_bucket), dtype=np.intp)
        b = self.bucket
        end = start + k
        j0, j1 = -(-start // b), end // b  # whole buckets in the window: j0..j1-1
        if j1 <= j0:
            return np.unique(self._edge(0, k, values))
        cached_end = self.first + len(self.picks)
        if not self.first <= j0 <= cached_end:
            self.first, self.picks = j0, self.picks[:0]  # nothing cached is in the window
        self.picks = self.picks[j0 - self.first : max(0, j1 - self.first)]
        self.first = j0
        lo = self.first + len(self.picks)
        if lo < j1:
            i = lo * b - start
            y = values(slice(i, i + (j1 - lo) * b)).reshape(j1 - lo, b, columns)
            base = (np.arange(lo, j1) * b)[:, None]
            new = np.hstack([base, base + y.argmin(axis=1), base + y.argmax(axis=1)])
            self.picks = np.vstack([self.picks, new])
        head, tail = j0 * b - start, j1 * b - start
        parts = [self._edge(0, head, values), self.picks.ravel() - start, self._edge(tail, k, values), [k - 1]]
        return np.unique(np.concatenate(parts))

    @staticmethod
    def _edge(i: int, j: int, values: Callable[[slice | np.ndarray], np.ndarray]) -> np.ndarray:
        """The first and last sample and the extremes of every column in window samples i..j-1."""
        if j <= i:
            return np.empty(0, dtype=np.intp)
        y = values(slice(i, j))
        return np.concatenate([[i, j - 1], i + y.argmin(axis=0), i + y.argmax(axis=0)])


class ModalTable(QtWidgets.QTableWidget):
    CLASSICAL_HEADERS = ["Mode", "fₙ [Hz]", "ζ modal", "ζ exact", "f_d [Hz]", "ζ friction"]
    CLASSICAL_TIPS = [
        "Mode number (ordered by undamped natural frequency)",
        "Undamped natural frequency: Kφ = ω²Mφ",
        "Modal damping ratio φᵀCφ / (2ωₙ) using the undamped (mass-normalized) mode shape. "
        "Exact only for proportional damping.",
        "Exact damping ratio from the complex eigenvalue λ of the state matrix: -Re(λ)/|λ|",
        "Damped natural frequency Im(λ)/2π",
        "<p>Equivalent viscous damping ratio of the Coulomb friction, for the mode vibrating with "
        "the <i>Initial displacement</i> below at the mass that moves most.</p>"
        "<p>Friction takes a fixed amplitude ΔA per cycle out of the mode (4F<sub>f</sub>/k for a "
        "single mass), whatever its size; a damper with ratio ζ takes about 2πζA. Matching the "
        "two gives ζ = ΔA / 2πA, the energy-equivalent damper c<sub>eq</sub> = 4F<sub>f</sub>/πωX. "
        "It falls as the amplitude grows: friction damps small motions most.</p>"
        "<p>Add it to ζ exact for the total. It assumes the motion stays in the mode shape, which "
        "friction, being nonlinear, only roughly allows.</p>",
    ]
    FRICTION_COLUMN = 5
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

    def set_result(
        self, result: ModalResult, method: Method, entries: list[ModeEntry], friction_zeta: np.ndarray | None = None
    ) -> None:
        """Show `result`; `friction_zeta` (one per mode) adds the classical table's friction column."""
        if method is not self.method:
            self.method = method
            self.clearSelection()
            self.setCurrentCell(-1, -1)
            classical = method is Method.CLASSICAL
            headers = self.CLASSICAL_HEADERS if classical else self.STATE_SPACE_HEADERS
            tips = self.CLASSICAL_TIPS if classical else self.STATE_SPACE_TIPS
            self.setColumnCount(len(headers))
            self.setHorizontalHeaderLabels(headers)
            for col, tip in enumerate(tips):
                self.horizontalHeaderItem(col).setToolTip(tip)
        if method is Method.CLASSICAL:
            rows = self._classical_rows(result, friction_zeta)
            self.setColumnHidden(self.FRICTION_COLUMN, friction_zeta is None)
        else:
            rows = self._state_space_rows(result)
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
    def _classical_rows(result: ModalResult, friction_zeta: np.ndarray | None) -> list[list[str]]:
        rows = []
        for r, mode in enumerate(result.modes):
            d = mode.damped
            rows.append([
                str(mode.index),
                f"{mode.fn_hz:.4g}",
                "rigid" if math.isnan(mode.zeta_modal) else f"{mode.zeta_modal:.4f}",
                f"{d.zeta:.4f}" if d else "overdamped",
                f"{d.fd_hz:.4g}" if d else "—",
                "—" if friction_zeta is None or math.isnan(friction_zeta[r]) else f"{friction_zeta[r]:.4f}",
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
        self.legend = self.addLegend(offset=(5, -5), colCount=2)
        self.curves: list[pg.PlotDataItem] = []
        self.entries: list[ModeEntry] = []
        self.highlight: int | None = None
        self.show_envelope = False
        self.envelope = [self.plot() for _ in range(2)]
        self.apply_theme()

    def apply_theme(self) -> None:
        """The envelope; the shapes take their colours from the entries (set_entries)."""
        self.legend.setBrush(colors.legend_brush())
        for c in self.envelope:
            c.setPen(pg.mkPen(colors.grey, width=1, style=QtCore.Qt.PenStyle.DashLine))

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
        self.circle = self.plot(circle.real, circle.imag)
        self.axes = [pg.InfiniteLine(pos=0, angle=0), pg.InfiniteLine(pos=0, angle=90)]
        for line in self.axes:
            self.addItem(line)
        self.arrows: list[pg.PlotDataItem] = []
        self.tips = pg.ScatterPlotItem(size=8, pen=None)
        self.projections = pg.ScatterPlotItem(size=7, symbol="o", brush=None)
        self.addItem(self.tips)
        self.addItem(self.projections)
        self.entries: list[ModeEntry] = []
        self.highlight: int | None = None
        self._brushes: list[QtGui.QBrush] = []
        self._pens: list[QtGui.QPen] = []
        self.apply_theme()

    def apply_theme(self) -> None:
        """The unit circle and axes; the arrows are coloured in set_entries."""
        self.circle.setPen(pg.mkPen(colors.faint, width=1, style=QtCore.Qt.PenStyle.DotLine))
        for line in self.axes:
            line.setPen(pg.mkPen(colors.faint, width=1))

    def set_entries(self, entries: list[ModeEntry]) -> None:
        self.entries = entries
        n = len(entries[0].shape) if entries else 0
        while len(self.arrows) < n:
            self.arrows.append(self.plot())
        for i, a in enumerate(self.arrows):
            a.setVisible(i < n)
            a.setPen(pg.mkPen(colors.mass[i], width=2))
        self._brushes = [pg.mkBrush(colors.mass[i]) for i in range(n)]
        self._pens = [pg.mkPen(colors.mass[i], width=2) for i in range(n)]
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
                    item.setForeground(pg.mkColor(colors.mass[i]))
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
    """Receptance |X_i / F| and phase for a force at the target mass, or transmissibility |X_i / X_g|."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.mag = self.addPlot(row=0, col=0, axisItems=log_axes())
        self.mag.setLogMode(x=True, y=True)
        self.mag.setLabel("left", "|X / F|  [m/N]")
        self.mag.showGrid(x=True, y=True, alpha=0.3)
        self.mag.addLegend(offset=(-5, 5), colCount=2)
        self.phase = self.addPlot(row=1, col=0, axisItems=log_axes())
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
        self.drive_lines = [pg.InfiniteLine(angle=90) for _ in range(2)]
        self.mag.addItem(self.drive_lines[0])
        self.phase.addItem(self.drive_lines[1])
        self.apply_theme()

    def apply_theme(self) -> None:
        """The drive lines; the curves are drawn by set_system."""
        for line in self.drive_lines:
            line.setPen(pg.mkPen(colors.force, width=2, style=QtCore.Qt.PenStyle.DashLine))

    def set_system(
        self,
        system: ChainSystem,
        result: ModalResult,
        input_dof: int,
        base: bool = False,
        reference: ChainSystem | None = None,
    ) -> None:
        """Receptance for a force at input_dof, or with base=True the transmissibility from the ground.

        A reference system (fewer masses: the chain without its absorber) is drawn thin and dashed,
        unless the force is on a mass it lacks.
        """
        f = frequency_grid(result, 1500)
        lo, hi = f[0], f[-1]
        if reference is not None and not base and input_dof >= reference.n:
            reference = None
        if reference is not None:
            # Its resonant peaks, so a lightly damped one is not clipped (its undamped fₙ is
            # left out: an undamped absorber tuned to it would put an exact zero there).
            peaks = [m.damped.fd_hz for m in modal_analysis(reference).modes if m.damped is not None]
            f = np.union1d(f, [p for p in peaks if lo < p < hi])
        freqs_n = np.array([m.fn_hz for m in result.modes if lo < m.fn_hz])
        H = transmissibility(system, f) if base else frf(system, f, input_dof)
        self.mag.setLabel("left", "|X / X_g|" if base else "|X / F|  [m/N]")

        for c in self.mag_curves + self.mode_lines:
            self.mag.removeItem(c)
        for c in self.phase_curves:
            self.phase.removeItem(c)
        self.mag.legend.clear()
        self.mag_curves, self.phase_curves = [], []
        for i in range(system.n):
            pen = pg.mkPen(colors.mass[i], width=2)
            self.mag_curves.append(self.mag.plot(f, np.abs(H[:, i]), pen=pen, name=f"x{i + 1}"))
            ph = np.degrees(np.unwrap(np.angle(H[:, i])))
            self.phase_curves.append(self.phase.plot(f, ph, pen=pen))
        if reference is not None:
            H0 = transmissibility(reference, f) if base else frf(reference, f, input_dof)
            for i in range(reference.n):
                pen = pg.mkPen(colors.mass[i], width=1.5, style=QtCore.Qt.PenStyle.DashLine)
                self.mag_curves.append(self.mag.plot(f, np.abs(H0[:, i]), pen=pen))

        self.mode_lines = []
        for r, fn in enumerate(freqs_n):
            line = pg.InfiniteLine(
                pos=math.log10(fn),
                angle=90,
                pen=pg.mkPen(colors.mode[r % len(colors.mode)], width=1, style=QtCore.Qt.PenStyle.DotLine),
            )
            self.mag.addItem(line)
            self.mode_lines.append(line)
        source = "Ground motion" if base else f"Force at m{input_dof + 1}"
        without = f" · thin dashed: without m{system.n}" if reference is not None else ""
        self.mag.setTitle(f"{source} · dotted: fₙ · dashed: drive{without}", size="10pt")
        self.mag.setXRange(math.log10(lo), math.log10(hi), padding=0)

    def set_drive(self, freq_hz: float | None) -> None:
        for line in self.drive_lines:
            line.setVisible(freq_hz is not None and freq_hz > 0)
            if freq_hz:
                line.setValue(math.log10(freq_hz))
