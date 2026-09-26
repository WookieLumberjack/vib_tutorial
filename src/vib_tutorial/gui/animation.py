"""Schematic animation of the chain: wall, springs, dampers, masses, force arrow."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from .style import FORCE_COLOR, MASS_COLORS, STRUCTURE_COLOR

SPACING = 1.0  # display units between mass centers at rest
MASS_WIDTH = 0.4
BASE_HEIGHT = 0.5
SPRING_Y = 0.13
DAMPER_Y = -0.13
MAX_SWING = 0.3 * SPACING  # auto-scale target for the largest displacement
# Never magnify motion smaller than this: as vibration decays toward zero the
# gain would otherwise grow without bound (magnifying float noise and driving
# Qt's drawing transforms to absurd values).
MIN_AUTO_PEAK = 1e-6  # m


def spring_path(x0: float, x1: float, y: float, coils: int = 6, amp: float = 0.05):
    """Zig-zag spring from x0 to x1 with straight leads at both ends."""
    lead = 0.12 * (x1 - x0)
    xs = [x0, x0 + lead]
    ys = [y, y]
    zig = np.linspace(x0 + lead, x1 - lead, 2 * coils + 1)
    for j, xz in enumerate(zig[1:-1]):
        xs.append(xz)
        ys.append(y + (amp if j % 2 == 0 else -amp))
    xs += [x1 - lead, x1]
    ys += [y, y]
    return np.array(xs), np.array(ys)


def damper_path(x0: float, x1: float, y: float, rest_gap: float, h: float = 0.045):
    """Dashpot: open cylinder fixed to x0, piston rod fixed to x1.

    The piston tracks the relative motion but is clamped inside the cylinder,
    and the rod always runs from the piston to x1, so the two halves stay
    connected however far the ends move (the rod just gets longer/shorter).

    Returned as one polyline with NaN breaks (drawn with connect='finite').
    """
    cyl0 = x0 + 0.15 * rest_gap
    cyl1 = x0 + 0.7 * rest_gap
    piston = x1 - 0.5 * rest_gap  # sits mid-cylinder at rest
    piston = min(max(piston, cyl0 + 0.05 * rest_gap), cyl1 - 0.02 * rest_gap)
    if x1 < piston:  # ends have crossed (extreme compression): collapse the rod
        piston = x1
    nan = np.nan
    xs = [x0, cyl0, nan,
          cyl1, cyl0, cyl0, cyl1, nan,  # cylinder: top, back wall, bottom
          piston, piston, nan,  # piston plate
          piston, x1]  # rod
    ys = [y, y, nan,
          y + h, y + h, y - h, y - h, nan,
          y + 0.75 * h, y - 0.75 * h, nan,
          y, y]
    return np.array(xs), np.array(ys)


class ChainView(pg.PlotWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMenuEnabled(False)
        self.setMouseEnabled(x=False, y=False)
        self.hideButtons()
        self.hideAxis("left")
        self.hideAxis("bottom")
        self.setAspectLocked(True)
        self.gain = 10.0  # display units per meter
        self.auto_scale = True
        self._n = 0
        self._masses: np.ndarray = np.ones(1)

        pen = pg.mkPen(STRUCTURE_COLOR, width=2)
        # Ground wall with hatching.
        self.addItem(pg.PlotDataItem([0, 0], [-0.45, 0.45], pen=pg.mkPen(STRUCTURE_COLOR, width=4)))
        hx, hy = [], []
        for yy in np.linspace(-0.4, 0.45, 9):
            hx += [0, -0.1, np.nan]
            hy += [yy, yy - 0.1, np.nan]
        self.addItem(pg.PlotDataItem(hx, hy, pen=pen, connect="finite"))

        self._pen = pen
        self._springs: list[pg.PlotDataItem] = []
        self._dampers: list[pg.PlotDataItem] = []
        self._rects: list[QtWidgets.QGraphicsRectItem] = []
        self._labels: list[pg.TextItem] = []
        self._rest_marks: list[pg.PlotDataItem] = []

        self._force_shaft = pg.PlotDataItem(pen=pg.mkPen(FORCE_COLOR, width=4))
        self._force_head = pg.ArrowItem(angle=180, headLen=18, tipAngle=40, brush=FORCE_COLOR, pen=None)
        self._force_text = pg.TextItem(color=FORCE_COLOR, anchor=(0.5, 1.0))
        for item in (self._force_shaft, self._force_head, self._force_text):
            self.addItem(item)

        self._scale_bar = pg.PlotDataItem(pen=pg.mkPen("#666", width=2))
        self._scale_text = pg.TextItem(color="#444", anchor=(0, 0.5))
        self.addItem(self._scale_bar)
        self.addItem(self._scale_text)

    # ------------------------------------------------------------- structure
    def set_masses(self, masses: np.ndarray) -> None:
        """(Re)build the per-mass graphics; mass sizes hint at relative mass."""
        n = masses.size
        self._masses = masses.copy()
        if n != self._n:
            for item in self._springs + self._dampers + self._rects + self._labels + self._rest_marks:
                self.removeItem(item)
            self._springs = [self._add(pg.PlotDataItem(pen=self._pen)) for _ in range(n)]
            self._dampers = [self._add(pg.PlotDataItem(pen=self._pen, connect="finite")) for _ in range(n)]
            self._rest_marks = []
            self._rects = []
            self._labels = []
            for i in range(n):
                xr = (i + 1) * SPACING
                self._rest_marks.append(
                    self._add(pg.PlotDataItem([xr, xr], [-0.5, -0.42], pen=pg.mkPen("#999", width=1)))
                )
                rect = QtWidgets.QGraphicsRectItem()
                rect.setBrush(pg.mkBrush(MASS_COLORS[i]))
                rect.setPen(pg.mkPen(STRUCTURE_COLOR, width=1.5))
                rect.setZValue(5)
                self._rects.append(self._add(rect))
                label = pg.TextItem(f"m{i + 1}", color="w", anchor=(0.5, 0.5))
                label.setZValue(6)
                self._labels.append(self._add(label))
            self._n = n
            self.setXRange(-0.25, n * SPACING + 0.6, padding=0)
            self.setYRange(-0.6, 0.75, padding=0)
            self._scale_bar.setData([0.05, 0.05 + MAX_SWING], [-0.55, -0.55])
            self._scale_text.setPos(0.1 + MAX_SWING, -0.55)

    def _add(self, item):
        self.addItem(item)
        return item

    def _height(self, i: int) -> float:
        rel = self._masses[i] / self._masses.mean()
        return BASE_HEIGHT * float(np.clip(rel ** (1 / 3), 0.6, 1.5))

    # ---------------------------------------------------------------- update
    def update_state(self, x: np.ndarray, peak: float, force: float, force_target: int, force_scale: float) -> None:
        """Redraw for displacements x (m).

        peak is the largest recent |x| used for auto-scaling; force_scale is
        the force magnitude that maps to a full-length arrow.
        """
        if self.auto_scale:
            target = MAX_SWING / max(peak, MIN_AUTO_PEAK)
            # Shrink immediately, grow slowly: keeps the view calm as motion decays.
            self.gain = target if target < self.gain else self.gain + 0.03 * (target - self.gain)
        self._scale_text.setText(f"= {_fmt_len(MAX_SWING / self.gain)}")

        rest_gap = SPACING - MASS_WIDTH
        right_prev = 0.0
        for i in range(self._n):
            xc = (i + 1) * SPACING + self.gain * x[i]
            left = xc - MASS_WIDTH / 2
            hgt = self._height(i)
            self._rects[i].setRect(QtCore.QRectF(left, -hgt / 2, MASS_WIDTH, hgt))
            self._labels[i].setPos(xc, 0)
            gap_rest = rest_gap if i > 0 else SPACING - MASS_WIDTH / 2
            self._springs[i].setData(*spring_path(right_prev, left, SPRING_Y))
            self._dampers[i].setData(*damper_path(right_prev, left, DAMPER_Y, gap_rest))
            right_prev = left + MASS_WIDTH

        if abs(force) > 1e-12 and force_scale > 0 and 0 <= force_target < self._n:
            xc = (force_target + 1) * SPACING + self.gain * x[force_target]
            y = self._height(force_target) / 2 + 0.12
            length = 0.45 * SPACING * min(1.0, abs(force) / force_scale)
            tip = xc + np.sign(force) * length
            self._force_shaft.setData([xc, tip], [y, y])
            self._force_head.setPos(tip, y)
            self._force_head.setStyle(angle=180 if force > 0 else 0)
            self._force_text.setText(f"F = {force:+.3g} N")
            self._force_text.setPos(xc, y + 0.14)
            for item in (self._force_shaft, self._force_head, self._force_text):
                item.setVisible(True)
        else:
            for item in (self._force_shaft, self._force_head, self._force_text):
                item.setVisible(False)


def _fmt_len(meters: float) -> str:
    if meters >= 1:
        return f"{meters:.3g} m"
    if meters >= 1e-3:
        return f"{meters * 1e3:.3g} mm"
    return f"{meters * 1e6:.3g} µm"
