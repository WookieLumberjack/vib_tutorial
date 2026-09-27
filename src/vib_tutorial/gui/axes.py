"""A log axis whose labels do not run into each other.

pyqtgraph labels every minor tick (2, 3, ... 9 x 10^e) on a log axis spanning
less than about three decades, so on a narrow plot they overlap ("0.60.70.8").
LogAxis draws every tick but labels all of them only when they fit, else 1,
2 and 5 of each decade, else only the decades, chosen from the pixels per
decade at each redraw (so it stays readable while zooming). It also turns off pyqtgraph's automatic
SI prefix in log mode, which scales the labels and appends a confusing
"(x0.001)" to the axis title.
"""

from __future__ import annotations

import math

import pyqtgraph as pg
from PySide6 import QtGui

# Labelled mantissas, the first set that fits: every tick, then 1, 2 and 5, then only the decades.
TIERS = (tuple(range(1, 10)), (1, 2, 5), (1,))
SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def log_label(value: float) -> str:
    """0.0001 ... 9999 as plain numbers; smaller or larger as m·10ⁿ."""
    e = math.floor(math.log10(value) + 1e-9)
    if -4 <= e <= 3:
        return f"{value:.4g}"
    m = round(value / 10.0**e)
    if m == 10:
        m, e = 1, e + 1
    power = f"10{str(e).translate(SUPERSCRIPT)}"
    return power if m == 1 else f"{m}·{power}"


class LogAxis(pg.AxisItem):
    """An AxisItem that behaves as usual in linear mode and labels sparsely in log mode."""

    def __init__(self, orientation: str, **kwargs) -> None:
        super().__init__(orientation, **kwargs)
        self._px_per_decade = math.inf

    def setLogMode(self, *args, **kwargs) -> None:  # noqa: N802 (Qt-style override)
        super().setLogMode(*args, **kwargs)
        self.enableAutoSIPrefix(not self.logMode)

    def logTickValues(self, minVal, maxVal, size, stdTicks):  # noqa: N802, N803 (pyqtgraph API)
        span = maxVal - minVal
        self._px_per_decade = size / span if span > 0 else math.inf
        levels = super().logTickValues(minVal, maxVal, size, stdTicks)
        # The minor level repeats the decades of the major levels: drawn twice, their labels look bold.
        major = {round(v, 9) for spacing, values in levels if spacing is not None for v in values}
        return [(spacing, values if spacing is not None else [v for v in values if round(v, 9) not in major])
                for spacing, values in levels]

    def logTickStrings(self, values, scale, spacing):  # noqa: N802 (pyqtgraph API)
        labelled = next((t for t in TIERS if self._room_for(t)), (1,))
        out = []
        for x in values:
            v = 10.0**x * scale
            e = math.floor(x + 1e-9)
            m = round(10.0 ** (x - e))
            out.append(log_label(v) if m in labelled or m == 10 else "")
        return out

    def _room_for(self, mantissas: tuple[int, ...]) -> bool:
        """Whether labels at these mantissas fit: the closest pair is log10(2) (or 10/5) apart."""
        font = self.style.get("tickFont") or QtGui.QFont()
        metrics = QtGui.QFontMetrics(font)
        if self.orientation in ("left", "right"):
            need = 1.6 * metrics.height()
        else:
            need = metrics.horizontalAdvance("0.005") + 10
        gap = min(math.log10(b / a) for a, b in zip(mantissas, (*mantissas[1:], 10)))
        return gap * self._px_per_decade >= need


def log_axes() -> dict[str, LogAxis]:
    """Left and bottom axes for a plot with a log axis: pass as axisItems to addPlot or PlotWidget."""
    return {"left": LogAxis("left"), "bottom": LogAxis("bottom")}
