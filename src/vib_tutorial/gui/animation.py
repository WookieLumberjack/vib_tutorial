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
# Farthest a mass can be dragged from rest (display units): the auto-scale
# target, so letting go does not rescale the view.
MAX_DRAG = MAX_SWING
# Magnification when a mass is grabbed at rest: a full drag is then 30 mm,
# rather than the micrometres the auto-scale grows to while nothing moves.
DRAG_GAIN = 10.0  # display units per meter
DRAG_TIP = (
    "<p><b>Pluck:</b> drag any mass sideways with the mouse and let go.</p>"
    "<p>While you hold it the force is switched off and time stands still; the other masses "
    "take the static shape a slow pull gives (every spring balanced), and the red arrow is "
    "the force your hand needs. Let go and the chain vibrates freely from that shape, "
    "which excites every mode, most of all the low ones.</p>"
)


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
    mass_grabbed = QtCore.Signal(int)  # mass index
    mass_dragged = QtCore.Signal(int, float)  # mass index, displacement (m)
    mass_released = QtCore.Signal(int)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setToolTip(DRAG_TIP)
        self.setMouseTracking(True)
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
        self._centers = np.zeros(0)  # drawn mass centers (display units)
        self.held: int | None = None  # mass being dragged

        pen = pg.mkPen(STRUCTURE_COLOR, width=2)
        # Ground wall with hatching; it moves with base excitation.
        self._wall = [pg.PlotDataItem([0, 0], [-0.45, 0.45], pen=pg.mkPen(STRUCTURE_COLOR, width=4))]
        hx, hy = [], []
        for yy in np.linspace(-0.4, 0.45, 9):
            hx += [0, -0.1, np.nan]
            hy += [yy, yy - 0.1, np.nan]
        self._wall.append(pg.PlotDataItem(hx, hy, pen=pen, connect="finite"))
        for item in self._wall:
            self.addItem(item)

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
            # Set both axes in one call: with the aspect ratio locked, separate
            # setXRange/setYRange calls let the second one shrink the first,
            # cropping the chain once it's wider than the widget's shape.
            self.setRange(
                QtCore.QRectF(-0.25, -0.6, n * SPACING + 0.85, 1.35), padding=0
            )
            self._scale_bar.setData([0.05, 0.05 + MAX_SWING], [-0.55, -0.55])
            self._scale_text.setPos(0.1 + MAX_SWING, -0.55)

    def _add(self, item):
        self.addItem(item)
        return item

    def _height(self, i: int) -> float:
        rel = self._masses[i] / self._masses.mean()
        return BASE_HEIGHT * float(np.clip(rel ** (1 / 3), 0.6, 1.5))

    # ---------------------------------------------------------------- update
    def update_state(
        self, x: np.ndarray, peak: float, force: float, force_target: int, force_scale: float, ground: float = 0.0
    ) -> None:
        """Redraw for displacements x (m) and the ground at `ground` (m).

        peak is the largest recent |x| used for auto-scaling; force_scale is
        the force magnitude that maps to a full-length arrow.
        """
        if self.auto_scale and self.held is None:  # hold the scale still under the mouse
            target = MAX_SWING / max(peak, MIN_AUTO_PEAK)
            # Shrink immediately, grow slowly: keeps the view calm as motion decays.
            self.gain = target if target < self.gain else self.gain + 0.03 * (target - self.gain)
        self._scale_text.setText(f"= {_fmt_len(MAX_SWING / self.gain)}")

        rest_gap = SPACING - MASS_WIDTH
        right_prev = self.gain * ground
        for item in self._wall:
            item.setPos(right_prev, 0)
        self._centers = (np.arange(self._n) + 1) * SPACING + self.gain * np.asarray(x[: self._n])
        for i in range(self._n):
            xc = self._centers[i]
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

    # ------------------------------------------------------------------ drag
    def _view_pos(self, event) -> QtCore.QPointF:
        return self.plotItem.vb.mapSceneToView(self.mapToScene(event.position().toPoint()))

    def mass_at(self, pos: QtCore.QPointF) -> int | None:
        """Index of the mass drawn under a point in view coordinates, if any."""
        for i, xc in enumerate(self._centers):
            if abs(pos.x() - xc) <= MASS_WIDTH / 2 and abs(pos.y()) <= self._height(i) / 2:
                return i
        return None

    def _drag_to(self, pos: QtCore.QPointF) -> None:
        i = self.held
        offset = float(np.clip(pos.x() - (i + 1) * SPACING, -MAX_DRAG, MAX_DRAG))
        self.mass_dragged.emit(i, offset / self.gain)

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        i = self.mass_at(self._view_pos(event)) if event.button() == QtCore.Qt.MouseButton.LeftButton else None
        if i is None:
            super().mousePressEvent(event)
            return
        if self.auto_scale:
            self.gain = min(self.gain, DRAG_GAIN)
        self.held = i
        self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
        self.mass_grabbed.emit(i)
        self._drag_to(self._view_pos(event))
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 (Qt override)
        pos = self._view_pos(event)
        if self.held is not None:
            self._drag_to(pos)
            event.accept()
            return
        if self.mass_at(pos) is None:
            self.unsetCursor()
        else:
            self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if self.held is None:
            super().mouseReleaseEvent(event)
            return
        i, self.held = self.held, None
        self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
        self.mass_released.emit(i)
        event.accept()


def _fmt_len(meters: float) -> str:
    if meters >= 1:
        return f"{meters:.3g} m"
    if meters >= 1e-3:
        return f"{meters * 1e3:.3g} mm"
    return f"{meters * 1e6:.3g} µm"
