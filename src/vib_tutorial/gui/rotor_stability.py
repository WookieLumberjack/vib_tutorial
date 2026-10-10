"""Stability tab of the Jeffcott rotor page: the stability map and the full spectrum of an orbit.

The stability map draws each lightly damped mode's damping ratio (or log
decrement) against speed, coloured by whirl direction as on the Campbell
diagram, with the unstable region below zero shaded and the onset speed marked.
The full spectrum transforms x + iy at one station over the strip chart's
window: forward whirl at positive frequencies, backward at negative, with the
±1X lines and the free whirl frequencies at the current speed.
"""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ..core.rotor import RPM, Campbell, WhirlModes
from ..core.rotor_stability import Onset, log_decrement
from .style import colors

DASH = QtCore.Qt.PenStyle.DashLine
DOT = QtCore.Qt.PenStyle.DotLine
MEASURES = ("Damping ratio ζ", "Log decrement δ")
FLOOR = 1e-4  # the spectrum's log scale starts this far below its highest line

MAP_TIP = (
    "<p>The damping of each lightly damped mode against speed, coloured by whirl direction as on the "
    "Campbell diagram. Below zero (shaded) a mode grows by itself instead of dying away: the rotor is "
    "unstable. The dashed line marks the onset speed, where the first mode crosses zero.</p>"
    "<p>Cross-coupled bearing stiffness and the shaft's internal damping lower the forward modes' "
    "damping and raise the backward ones'.</p>"
)
SPECTRUM_TIP = (
    "<p>The full spectrum of the orbit at the strip chart's station, over the strip chart's window: "
    "the Fourier transform of x + iy. Forward whirl (with the spin) shows at positive frequencies, "
    "backward whirl at negative ones; a straight-line vibration shows at both, half each.</p>"
    "<p>The red lines are ±1X, the speed: the unbalance drives +1X. The thin dashed lines are the free "
    "whirl frequencies at this speed. An unstable rotor whirls at its natural frequency, below +1X: a "
    "subsynchronous line that grows.</p>"
)


def _tinted(color: str, alpha: int) -> QtGui.QColor:
    c = QtGui.QColor(color)
    c.setAlpha(alpha)
    return c


class StabilityMap(pg.GraphicsLayoutWidget):
    """Damping ratio (or log decrement) of each mode against speed, and the onset of instability."""

    def __init__(self, damped_limit: float, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setToolTip(MAP_TIP)
        self.damped_limit = damped_limit  # modes more damped than this are left off
        self.log_dec = False
        self.plot = p = self.addPlot(row=0, col=0)
        p.setLabel("bottom", "Spin speed", units="rpm")
        p.getAxis("bottom").enableAutoSIPrefix(False)
        p.getAxis("left").enableAutoSIPrefix(False)
        p.showGrid(x=True, y=True, alpha=0.15)
        p.setMenuEnabled(False)
        self.legend = pg.LegendItem(colCount=3, brush=colors.legend_brush(), labelTextColor=colors.foreground)
        self.addItem(self.legend, row=1, col=0)
        self.region = pg.LinearRegionItem(orientation="horizontal", movable=False)
        self.region.setZValue(-10)
        p.addItem(self.region)
        self.zero = pg.InfiniteLine(pos=0.0, angle=0)
        p.addItem(self.zero)
        self.points = {key: pg.ScatterPlotItem(size=size, name=name)
                       for key, name, size in (("bw", "backward whirl", 6), ("fw", "forward whirl", 4),
                                               ("planar", "straight-line whirl", 4))}
        for item in self.points.values():
            p.addItem(item)
            self.legend.addItem(item, item.name())
        self.onset_line = pg.InfiniteLine(angle=90)
        self.onset_label = pg.TextItem(anchor=(-0.05, 0.0))
        self.speed_line = pg.InfiniteLine(angle=90)
        for item in (self.onset_line, self.onset_label, self.speed_line):
            p.addItem(item)
        self.result: Campbell | None = None
        self.onset: Onset | None = None
        self.max_rpm = 1.0
        self.apply_theme()

    def apply_theme(self) -> None:
        brushes = {"fw": colors.mode[1], "bw": colors.mode[3], "planar": colors.grey}
        for key, item in self.points.items():
            item.setBrush(pg.mkBrush(brushes[key]))
            item.setPen(pg.mkPen(None))
        self.region.setBrush(pg.mkBrush(_tinted(colors.force, 28)))
        for line in self.region.lines:
            line.setPen(pg.mkPen(None))
        self.zero.setPen(pg.mkPen(colors.structure, width=1))
        self.onset_line.setPen(pg.mkPen(colors.force, width=1.5, style=DASH))
        self.onset_label.setColor(colors.force)
        self.speed_line.setPen(pg.mkPen(colors.force, width=2))

    def set_measure(self, log_dec: bool) -> None:
        self.log_dec = log_dec
        if self.result is not None:
            self.set_result(self.result, self.onset, self.max_rpm)

    def set_result(self, result: Campbell, onset: Onset | None, max_rpm: float) -> None:
        self.result, self.onset, self.max_rpm = result, onset, max_rpm
        p = self.plot
        p.setLabel("left", MEASURES[self.log_dec])
        rpm = np.repeat(result.omega * RPM, result.zeta.shape[1])
        zeta = result.zeta.ravel()
        whirl = result.whirl.ravel()
        ok = np.isfinite(zeta) & (zeta <= self.damped_limit)
        y = log_decrement(zeta) if self.log_dec else zeta
        classes = {"fw": whirl > 0.1, "bw": whirl < -0.1}
        classes["planar"] = ~(classes["fw"] | classes["bw"])
        for key, mask in classes.items():
            self.points[key].setData(rpm[mask & ok], y[mask & ok])
        shown = y[ok]
        top = float(shown.max()) if shown.size else 1.0
        bottom = min(float(shown.min()) if shown.size else 0.0, 0.0)
        span = max(top - bottom, 1e-3)
        lo, hi = bottom - 0.15 * span, top + 0.08 * span
        self.region.setRegion((lo - span, 0.0))
        p.setRange(xRange=(0.0, max_rpm), yRange=(lo, hi), padding=0)
        if onset is None:
            self.onset_line.setVisible(False)
            self.onset_label.setText("")
        else:
            x = onset.omega * RPM
            self.onset_line.setVisible(True)
            self.onset_line.setValue(x)
            self.onset_label.setText(f"onset {x:,.0f} rpm")
            self.onset_label.setPos(x, hi)

    def set_speed(self, rpm: float) -> None:
        self.speed_line.setValue(rpm)


class FullSpectrum(pg.PlotWidget):
    """Amplitude of the orbit's forward (+) and backward (−) whirl components against frequency."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setToolTip(SPECTRUM_TIP)
        p = self.getPlotItem()
        p.setLabel("bottom", "Whirl frequency (− backward, + forward)", units="Hz")
        p.getAxis("bottom").enableAutoSIPrefix(False)
        p.setLabel("left", "Amplitude", units="m")
        p.setLogMode(x=False, y=True)
        p.showGrid(x=True, y=True, alpha=0.15)
        p.setMenuEnabled(False)
        p.setMouseEnabled(x=False, y=False)
        p.hideButtons()
        self.axis = pg.InfiniteLine(pos=0.0, angle=90)
        p.addItem(self.axis)
        self.curve = p.plot()
        self.sync = [pg.InfiniteLine(angle=90, label=text, labelOpts={"position": 0.92})
                     for text in ("+1X", "−1X")]
        for line in self.sync:
            p.addItem(line)
        self.free: list[pg.InfiniteLine] = []
        self.free_forward: list[bool] = []
        self.apply_theme()

    def apply_theme(self) -> None:
        self.axis.setPen(pg.mkPen(colors.faint, width=1))
        self.curve.setPen(pg.mkPen(colors.strong, width=1.5))
        for line, style in zip(self.sync, (QtCore.Qt.PenStyle.SolidLine, DOT)):
            line.setPen(pg.mkPen(colors.force, width=1.5, style=style))
            line.label.setColor(colors.force)
        for line, forward in zip(self.free, self.free_forward):
            line.setPen(pg.mkPen(colors.mode[1] if forward else colors.mode[3], width=1, style=DASH))

    def update_spectrum(self, freqs: np.ndarray, amps: np.ndarray, omega: float, modes: WhirlModes | None,
                        damped_limit: float) -> None:
        """The spectrum, ±1X at speed Ω (rad/s), and the lightly damped modes' frequencies."""
        f1 = omega / (2.0 * math.pi)
        for line, sign in zip(self.sync, (1.0, -1.0)):
            line.setVisible(f1 > 0)
            line.setValue(sign * f1)
        p = self.getPlotItem()
        natural, marks = [], []
        if modes is not None:
            for f, z, w in zip(modes.freq_hz, modes.zeta, modes.whirl):
                if z <= damped_limit:
                    natural.append(f)
                    # Each mode shows on its own whirl side; a straight-line whirl on both.
                    if w >= -0.1:
                        marks.append((f, True))
                    if w <= 0.1:
                        marks.append((-f, False))
        while len(self.free) < len(marks):  # a pool of lines, reused from one update to the next
            self.free.append(pg.InfiniteLine(angle=90))
            self.free_forward.append(True)
            p.addItem(self.free[-1])
        for i, line in enumerate(self.free):
            line.setVisible(i < len(marks))
            if i < len(marks):
                line.setValue(marks[i][0])
                self.free_forward[i] = marks[i][1]
        reach = max([1.6 * f1, 1.3 * min(natural, default=0.0), 20.0])
        p.setXRange(-reach, reach, padding=0)
        if freqs.size:
            inside = np.abs(freqs) <= reach
            top = float(amps[inside].max()) if inside.any() else 0.0
            floor = max(top * FLOOR, 1e-12)
            self.curve.setData(freqs[inside], np.maximum(amps[inside], floor))
            p.setYRange(math.log10(floor), math.log10(max(top, floor) * 2.0), padding=0)
        else:
            self.curve.setData([], [])
        self.apply_theme()
