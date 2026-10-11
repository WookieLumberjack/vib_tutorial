"""Waterfall tab of the Jeffcott rotor page: full spectra stacked against speed or time.

Each line is the full spectrum of one block of the orbit at a station (see
core/rotor_waterfall.py), drawn at a height for its speed (a cascade) or its
time (a waterfall), with its amplitude rising above that baseline. Lines are
filled down to their baseline in the background colour, front to back, so a
tall peak hides the lines behind it as on an analyser's 3D plot. On the base of
each line, the +1X line and the free whirl frequencies at its speed are marked:
where the peaks fall on them, the spectrum is the unbalance or a free mode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.rotor import RPM
from ..core.rotor_waterfall import SpectrumBlock
from .style import colors

DOT = QtCore.Qt.PenStyle.DotLine
MAX_LINES = 120  # at most this many spectra are drawn; more are thinned evenly
DECADES = 3.0  # the log scale's range: the floor is this many decades below the tallest line
AGAINST = ("Speed: latest run-up", "Speed: latest coast-down", "Time")
SCALES = ("Log (60 dB)", "Linear")

WATERFALL_TIP = (
    "<p>Each line is the full spectrum of the orbit over one block of time: forward whirl to the right, "
    "backward to the left, as on the <i>Stability</i> tab. Against speed the lines make a <i>cascade</i>, "
    "against time a <i>waterfall</i>; each rises from its baseline by its amplitude.</p>"
    "<p>The red line on the base is +1X at each line's speed (dotted: −1X), where the unbalance drives. "
    "The dots are the free whirl frequencies at that speed, from the Campbell diagram and coloured as "
    "there, forward and backward. Tap the disc as it runs and peaks show up on them.</p>"
)


@dataclass
class Tallest:
    """The tallest peak drawn: its amplitude (m), frequency (Hz) and line."""

    amp: float
    freq: float
    block: SpectrumBlock


class WaterfallPlot(pg.GraphicsLayoutWidget):
    """Full spectra of a station's orbit stacked against speed or time, with 1X and the free whirl marked."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setToolTip(WATERFALL_TIP)
        self.plot = p = self.addPlot(row=0, col=0)
        p.setLabel("bottom", "Whirl frequency (− backward, + forward)", units="Hz")
        p.getAxis("bottom").enableAutoSIPrefix(False)
        p.getAxis("left").enableAutoSIPrefix(False)
        p.setMenuEnabled(False)
        p.setMouseEnabled(x=False, y=False)
        p.hideButtons()
        self.legend = pg.LegendItem(colCount=3, brush=colors.legend_brush(), labelTextColor=colors.foreground)
        self.addItem(self.legend, row=1, col=0)
        self.axis = pg.InfiniteLine(pos=0.0, angle=90)
        p.addItem(self.axis)
        self.lines: list[pg.PlotCurveItem] = []
        self.sync = pg.PlotCurveItem()
        self.sync_back = pg.PlotCurveItem()
        self.forward = pg.ScatterPlotItem(size=5, name="free whirl, forward")
        self.backward = pg.ScatterPlotItem(size=5, name="free whirl, backward")
        for z, item in enumerate((self.sync, self.sync_back, self.forward, self.backward)):
            item.setZValue(MAX_LINES + 10 + z)
            p.addItem(item)
        self.sync_sample = pg.PlotDataItem()  # the legend's sample of the 1X line
        self.legend.addItem(self.sync_sample, "1X at each line's speed")
        for item in (self.forward, self.backward):
            self.legend.addItem(item, item.name())
        self.apply_theme()

    def apply_theme(self) -> None:
        self.sync_sample.setPen(pg.mkPen(colors.force, width=2))
        self.axis.setPen(pg.mkPen(colors.faint, width=1))
        for line in self.lines:
            line.setPen(pg.mkPen(colors.strong, width=1))
            line.setBrush(pg.mkBrush(colors.background))
        self.sync.setPen(pg.mkPen(colors.force, width=2))
        self.sync_back.setPen(pg.mkPen(colors.force, width=1.5, style=DOT))
        for item, c in ((self.forward, colors.mode[1]), (self.backward, colors.mode[3])):
            item.setBrush(pg.mkBrush(c))
            item.setPen(pg.mkPen(colors.background, width=0.5))

    def _pool(self, n: int) -> None:
        while len(self.lines) < n:
            line = pg.PlotCurveItem()
            line.setPen(pg.mkPen(colors.strong, width=1))
            line.setBrush(pg.mkBrush(colors.background))
            self.plot.addItem(line)
            self.lines.append(line)

    def set_blocks(self, freqs: np.ndarray, blocks: list[SpectrumBlock], ys: np.ndarray, station: int,
                   log: bool, marks: bool, y_label: tuple[str, str]) -> Tallest | None:
        """Draw `blocks` (in time order) at baselines `ys`, the spectra at `station`; return the tallest peak."""
        p = self.plot
        p.setLabel("left", *y_label)
        if not blocks:
            for line in self.lines:
                line.setVisible(False)
            for item in (self.sync, self.sync_back):
                item.setData([], [])
            for item in (self.forward, self.backward):
                item.setData([], [])
            return None
        ys = np.asarray(ys, dtype=float)
        f1 = np.array([b.omega for b in blocks]) / (2.0 * math.pi)
        natural = [np.abs(b.natural) for b in blocks if b.natural.size]
        lowest = min((float(n.min()) for n in natural), default=0.0)
        reach = min(max(1.6 * float(f1.max()), 1.3 * lowest, 20.0), float(np.abs(freqs).max()))
        inside = np.abs(freqs) <= reach
        f = freqs[inside]

        pick = np.arange(len(blocks))
        if pick.size > MAX_LINES:
            pick = np.unique(np.round(np.linspace(0, len(blocks) - 1, MAX_LINES)).astype(int))
        amps = np.array([blocks[i].amps[station][inside] for i in pick], dtype=float)
        top = float(amps.max())
        k, j = np.unravel_index(int(np.argmax(amps)), amps.shape)
        tallest = Tallest(top, float(f[j]), blocks[pick[k]])
        if top <= 0.0:
            heights = np.zeros_like(amps)
        elif log:
            floor = top * 10.0**-DECADES
            heights = np.log10(np.maximum(amps, floor) / floor) / DECADES
        else:
            heights = amps / top

        base = ys[pick]
        span = float(np.ptp(base))
        spacing = span / (base.size - 1) if base.size > 1 else 0.0
        scale = max(0.2 * span, 2.5 * spacing) or max(0.05 * abs(float(base[0])), 1.0)
        self._pool(pick.size)
        rank = np.empty(pick.size, dtype=int)
        rank[np.argsort(base, kind="stable")] = np.arange(pick.size)
        for i, line in enumerate(self.lines):
            line.setVisible(i < pick.size)
            if i < pick.size:
                # The lowest line is in front: it hides the ones behind (above) it.
                line.setData(f, base[i] + scale * heights[i], fillLevel=float(base[i]))
                line.setZValue(pick.size - rank[i])

        for item, sign in ((self.sync, 1.0), (self.sync_back, -1.0)):
            item.setData(sign * f1, ys) if marks else item.setData([], [])
        for item, forward in ((self.forward, True), (self.backward, False)):
            pts = [(n, base[i]) for i, b in enumerate(blocks[j] for j in pick)
                   for n in b.natural if (n > 0) == forward and abs(n) <= reach]
            if marks and pts:
                x, y = np.array(pts).T
                item.setData(x, y)
            else:
                item.setData([], [])
        lo, hi = float(base.min()), float(base.max()) + scale
        pad = 0.03 * (hi - lo)
        p.setRange(xRange=(-reach, reach), yRange=(lo - pad, hi + pad), padding=0)
        return tallest


def against_speed(mode: int) -> bool:
    return mode < 2


def baselines(blocks: list[SpectrumBlock], mode: int) -> tuple[np.ndarray, tuple[str, str]]:
    """The lines' baselines for an AGAINST mode, and the vertical axis's label and units."""
    if against_speed(mode):
        return np.array([b.omega * RPM for b in blocks]), ("Spin speed", "rpm")
    return np.array([b.t for b in blocks]), ("Time", "s")
