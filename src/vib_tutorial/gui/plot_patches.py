"""Two changes to how every pyqtgraph plot draws, installed once by make_app (`install`).

Grids. pyqtgraph draws a plot's grid as its axes' ticks stretched across the plot, in the
foreground colour at a low alpha, and above the data (the axes sit at z 0.5, over the ViewBox).
Blending those long translucent lines is what a plot that rescales every frame spent its paint
on: on the Duffing page's phase plane and strip chart, ~35 ms a frame, against ~5 ms with the
same lines drawn opaque. So a grid axis draws them in the colour they would have come out over
the background, and goes behind the ViewBox, so that the lines pass under the curves, not over.

Titles. A LabelItem's minimum width is its text's, so a long plot title (the Virtual modal
test's FRF says what each line style means) made its plot wider than a 1280 px window's pane
leaves it, and the layout overflowed its view: the plot's right side and legend were cut off.
A level label (not a legend's) now wraps to the width it's given, centred, and only asks for TITLE_MIN_WIDTH.
"""

from __future__ import annotations

import pyqtgraph as pg
from PySide6 import QtCore, QtGui

from . import style

GRID_Z = -200.0  # below the ViewBox (z -100): the grid goes under the data
AXIS_Z = 0.5  # pyqtgraph's own
TITLE_MIN_WIDTH = 80.0  # px a title asks for at least; narrower than its text, it wraps


def _opaque_grid_specs(self, p):
    specs = _generate_draw_specs(self, p)
    if specs is None or self.grid is False:
        return specs
    axis_spec, tick_specs, text_specs = specs
    bg = QtGui.QColor(style.colors.background)
    blended: dict[tuple[int, int], QtGui.QPen] = {}  # (pen id, alpha) -> opaque pen; the ticks share a few
    out = []
    for pen, p1, p2 in tick_specs:
        c = pen.color()
        a = c.alphaF()
        if a < 1.0:
            key = (id(pen), c.alpha())
            if key not in blended:
                opaque = QtGui.QPen(pen)
                opaque.setColor(QtGui.QColor.fromRgbF(c.redF() * a + bg.redF() * (1 - a),
                                                      c.greenF() * a + bg.greenF() * (1 - a),
                                                      c.blueF() * a + bg.blueF() * (1 - a)))
                blended[key] = opaque
            pen = blended[key]
        out.append((pen, p1, p2))
    return axis_spec, out, text_specs


def _set_grid(self, grid) -> None:
    _set_grid_original(self, grid)
    self.setZValue(AXIS_Z if self.grid is False else GRID_Z)


def _wraps(label) -> bool:
    """A level label that isn't a legend's entry (a legend sizes itself to its entries)."""
    return getattr(label, "angle", 0) == 0 and not isinstance(label.parentItem(), pg.LegendItem)


def _wrap_to_width(self, ev) -> None:
    if _wraps(self):
        item = self.item
        item.setTextWidth(-1)  # its natural width
        width = self.rect().width()
        if width > 0 and item.boundingRect().width() > width + 0.5:
            item.setTextWidth(max(width, TITLE_MIN_WIDTH))
            option = item.document().defaultTextOption()
            option.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
            item.document().setDefaultTextOption(option)
    _resize_original(self, ev)


def _update_min(self) -> None:
    _update_min_original(self)
    if _wraps(self) and self.minimumWidth() > TITLE_MIN_WIDTH:
        self.setMinimumWidth(TITLE_MIN_WIDTH)
        self._sizeHint[QtCore.Qt.SizeHint.MinimumSize] = (TITLE_MIN_WIDTH, self.minimumHeight())


_generate_draw_specs = pg.AxisItem.generateDrawSpecs
_set_grid_original = pg.AxisItem.setGrid
_resize_original = pg.LabelItem.resizeEvent
_update_min_original = pg.LabelItem.updateMin


def install() -> None:
    """Grids opaque and under the data, titles that wrap; see the module's docstring."""
    pg.AxisItem.generateDrawSpecs = _opaque_grid_specs
    pg.AxisItem.setGrid = _set_grid
    pg.LabelItem.resizeEvent = _wrap_to_width
    pg.LabelItem.updateMin = _update_min
