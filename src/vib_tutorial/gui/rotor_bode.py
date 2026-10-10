"""Bode and polar plots for the Jeffcott rotor page: the steady-state 1X response and tracked sweeps.

Both plots draw the steady-state 1X response (to the unbalance, skew and bow)
at one station, in one direction, against speed, with a marker at the current
speed, and over it the 1X vectors tracked once per revolution during run-ups
and coast-downs. A sweep pair (its run-up and its coast-down) shares a colour:
the run-up is drawn solid, the coast-down dashed. The page's summary table under the plot is
their key, so the legend names only the steady state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.rotor import RPM
from ..core.rotor_sweep import TrackedSweep, lag_degrees
from .style import colors
from .theming import add_legend

DASH = QtCore.Qt.PenStyle.DashLine
SWEEP_COLORS = (1, 3, 5, 2, 4, 6, 7, 0)  # indices into colors.mode, one per sweep pair in turn

BODE_TIP = (
    "<p>Top: the 1X amplitude at the chosen station and direction against speed; bottom: its phase lag, "
    "the degrees the shaft turns from the keyphasor (the heavy spot passing +x) to the probe's positive "
    "peak. The thick line is the steady state, each speed held until the response settles; the red line "
    "and dot are the speed now.</p><p>The thin lines are run-ups (solid) and coast-downs (dashed), "
    "tracked once per revolution like a tracking filter on a real machine. The dots, if shown, are the "
    "largest displacement over each revolution, free vibration included.</p>"
)
POLAR_TIP = (
    "<p>The same 1X vector drawn as a point: its length is the amplitude, and the angle clockwise from "
    "the right is the phase lag (the direction of rotation is counterclockwise). As the speed passes a "
    "critical, the steady-state vector traces a circle (exactly one, for a single lightly damped mode); "
    "the critical speed is at the bottom of the circle, where the lag is 90°.</p>"
    "<p>The labelled dots on the steady-state curve mark the speed in rpm.</p>"
)


@dataclass
class SweepCurves:
    """What one tracked leg draws: on the amplitude, phase and polar plots."""

    sweep: TrackedSweep
    color: int  # index into SWEEP_COLORS
    amp: pg.PlotDataItem
    peaks: pg.ScatterPlotItem
    lag: pg.PlotDataItem
    polar: pg.PlotDataItem
    drawn: int = 0  # revolutions drawn so far


def _pen(sweep: TrackedSweep, color: int, width: float = 1.5) -> QtCore.QObject:
    c = colors.mode[SWEEP_COLORS[color % len(SWEEP_COLORS)]]
    return pg.mkPen(c, width=width, style=QtCore.Qt.PenStyle.SolidLine if sweep.up else DASH)


class BodePlots(pg.GraphicsLayoutWidget):
    """1X amplitude and phase lag against speed."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setToolTip(BODE_TIP)
        self.amp = self.addPlot(row=0, col=0)
        self.amp.setLabel("left", "1X amplitude", units="m")
        self.lag = self.addPlot(row=1, col=0)
        self.lag.setLabel("left", "Phase lag", units="°")
        self.lag.getAxis("left").enableAutoSIPrefix(False)
        self.lag.setLabel("bottom", "Spin speed", units="rpm")
        self.lag.getAxis("bottom").enableAutoSIPrefix(False)
        self.lag.setXLink(self.amp)
        for p in (self.amp, self.lag):
            p.showGrid(x=True, y=True, alpha=0.15)
            p.setAutoVisible(y=True)
            p.setMenuEnabled(False)
        self.ci.layout.setRowStretchFactor(0, 3)
        self.ci.layout.setRowStretchFactor(1, 2)
        self.legend = add_legend(self.amp, offset=(-5, 5))
        self.steady_amp = self.amp.plot(name="Steady state")
        self.steady_lag = self.lag.plot()
        self.here_lines = [pg.InfiniteLine(angle=90) for _ in range(2)]
        self.here_dots = [pg.ScatterPlotItem(size=9) for _ in range(2)]
        for p, line, dot in zip((self.amp, self.lag), self.here_lines, self.here_dots):
            p.addItem(line)
            p.addItem(dot)
        self.apply_theme()

    def apply_theme(self) -> None:
        self.legend.setBrush(colors.legend_brush())
        for curve in (self.steady_amp, self.steady_lag):
            curve.setPen(pg.mkPen(colors.strong, width=2.5))
        for line, dot in zip(self.here_lines, self.here_dots):
            line.setPen(pg.mkPen(colors.force, width=1))
            dot.setBrush(pg.mkBrush(colors.force))
            dot.setPen(pg.mkPen(colors.background, width=1))
        self.lag.getAxis("left").setTickSpacing(90.0, 45.0)

    def set_steady(self, rpm: np.ndarray, values: np.ndarray, lag: np.ndarray) -> None:
        self.steady_amp.setData(rpm, np.abs(values))
        self.steady_lag.setData(rpm, lag)

    def set_here(self, rpm: float, amp: float, lag: float) -> None:
        for line in self.here_lines:
            line.setValue(rpm)
        self.here_dots[0].setData([rpm], [amp])
        self.here_dots[1].setData([rpm], [lag])


class PolarPlot(pg.PlotWidget):
    """The 1X vector as a point: amplitude out from the centre, phase lag clockwise from +x."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setToolTip(POLAR_TIP)
        p = self.getPlotItem()
        p.setAspectLocked(True)
        p.setMenuEnabled(False)
        p.setLabel("bottom", "In phase with the heavy spot", units="m")
        p.setLabel("left", "Quadrature (down: lagging 90°)", units="m")
        p.showGrid(x=True, y=True, alpha=0.15)
        self.axes = [pg.InfiniteLine(pos=0, angle=a) for a in (0, 90)]
        for line in self.axes:
            p.addItem(line)
        self.steady = p.plot()
        self.marks = pg.ScatterPlotItem(size=5)
        p.addItem(self.marks)
        self.mark_labels: list[pg.TextItem] = []
        self.here = pg.ScatterPlotItem(size=9)
        p.addItem(self.here)
        self.here.setZValue(10)
        self.apply_theme()

    def apply_theme(self) -> None:
        for line in self.axes:
            line.setPen(pg.mkPen(colors.faint, width=1))
        self.steady.setPen(pg.mkPen(colors.strong, width=2.5))
        self.marks.setBrush(pg.mkBrush(colors.strong))
        self.marks.setPen(pg.mkPen(None))
        for label in self.mark_labels:
            label.setColor(colors.muted)
        self.here.setBrush(pg.mkBrush(colors.force))
        self.here.setPen(pg.mkPen(colors.background, width=1))

    @staticmethod
    def xy(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Plot coordinates of 1X vectors V = |V| e^{-i lag}: the lag runs clockwise from +x."""
        return np.real(values), np.imag(values)

    def set_steady(self, rpm: np.ndarray, values: np.ndarray, marks: np.ndarray) -> None:
        """The steady-state curve over the speeds shown, and dots labelled at the speeds `marks` (rpm)."""
        self.steady.setData(*self.xy(values))
        p = self.getPlotItem()
        for label in self.mark_labels:
            p.removeItem(label)
        self.mark_labels = []
        if not rpm.size or not marks.size:
            self.marks.setData([], [])
            return
        at = np.interp(marks, rpm, values.real) + 1j * np.interp(marks, rpm, values.imag)
        self.marks.setData(*self.xy(at))
        for r, v in zip(marks, at):
            label = pg.TextItem(f"{r:,.0f}", anchor=(-0.1, 1.0))
            label.setPos(float(v.real), float(v.imag))
            p.addItem(label)
            self.mark_labels.append(label)
        self.apply_theme()

    def set_here(self, value: complex) -> None:
        self.here.setData([value.real], [value.imag])


class SweepPlots:
    """Draws tracked sweeps on a BodePlots and a PolarPlot, adding new revolutions as they arrive."""

    def __init__(self, bode: BodePlots, polar: PolarPlot) -> None:
        self.bode, self.polar = bode, polar
        self.curves: list[SweepCurves] = []
        self.show_peaks = True

    def add(self, sweep: TrackedSweep, color: int) -> None:
        c = SweepCurves(sweep, color, self.bode.amp.plot(), pg.ScatterPlotItem(size=4),
                        self.bode.lag.plot(), self.polar.plot())
        self.bode.amp.addItem(c.peaks)
        c.peaks.setVisible(self.show_peaks)
        self.curves.append(c)
        self._style(c)

    def clear(self) -> None:
        for c in self.curves:
            for plot, item in ((self.bode.amp, c.amp), (self.bode.amp, c.peaks), (self.bode.lag, c.lag),
                               (self.polar.getPlotItem(), c.polar)):
                plot.removeItem(item)
        self.curves = []

    def _style(self, c: SweepCurves) -> None:
        for item in (c.amp, c.lag, c.polar):
            item.setPen(_pen(c.sweep, c.color))
        c.peaks.setBrush(pg.mkBrush(colors.mode[SWEEP_COLORS[c.color % len(SWEEP_COLORS)]]))
        c.peaks.setPen(pg.mkPen(None))

    def apply_theme(self) -> None:
        for c in self.curves:
            self._style(c)

    def set_show_peaks(self, on: bool) -> None:
        self.show_peaks = on
        for c in self.curves:
            c.peaks.setVisible(on)

    def redraw(self, station: int, direction: str, steady_rpm: np.ndarray, steady_lag: np.ndarray,
               full: bool) -> None:
        """Draw each sweep's revolutions; only the sweeps with new ones unless `full`."""
        for c in self.curves:
            if not full and len(c.sweep) == c.drawn:
                continue
            c.drawn = len(c.sweep)
            w, v, peak = c.sweep.trace(station, direction)
            rpm = w * RPM
            lag = lag_degrees(v, np.interp(rpm, steady_rpm, steady_lag)) if steady_rpm.size else lag_degrees(v)
            c.amp.setData(rpm, np.abs(v))
            c.peaks.setData(rpm, peak)
            c.lag.setData(rpm, lag)
            c.polar.setData(*PolarPlot.xy(v))


def nice_marks(lo: float, hi: float, count: int = 8) -> np.ndarray:
    """Round speeds (rpm) strictly inside (lo, hi), about `count` of them."""
    if hi <= lo:
        return np.empty(0)
    raw = (hi - lo) / count
    step = 10 ** math.floor(math.log10(raw))
    step *= next(m for m in (1, 2, 2.5, 5, 10) if m * step >= raw)
    first = math.floor(lo / step) + 1
    marks = step * np.arange(first, math.ceil(hi / step))
    return marks[(marks > lo) & (marks < hi)]
