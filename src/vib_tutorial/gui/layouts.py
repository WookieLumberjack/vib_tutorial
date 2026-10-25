"""Layouts that keep the panes, and so the window, narrow.

The window can't be narrower than its widest page, and a page is as wide as its
splitter's panes' minimums together. A QHBoxLayout's minimum width is the sum of its
controls', so one long row of them sets the width of its whole pane. A FlowLayout lays
its items out on one line when there's room and wraps between them when there isn't:
its minimum width is its widest item's. Keep a label with its control by putting them
in one item (`group`). A ColumnScroll holds a column of controls (a page's parameters,
a tall panel): it scrolls up and down only, and is as narrow as what it holds allows.
It makes what it holds at least as tall as that asks for, so give its plots a height
to ask for (`plot_height`), or it scrolls even where the plots could have shrunk.
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets


class FlowLayout(QtWidgets.QLayout):
    """Items left to right, wrapping onto a new line when the next doesn't fit."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._items: list[QtWidgets.QLayoutItem] = []
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item: QtWidgets.QLayoutItem) -> None:  # noqa: N802 (Qt override)
        self._items.append(item)

    def add(self, *items: QtWidgets.QWidget | QtWidgets.QLayout) -> None:
        """Add each widget or layout as one item (a layout is wrapped in a widget, as by `group`)."""
        for item in items:
            self.addWidget(item if isinstance(item, QtWidgets.QWidget) else _holder(item))

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> QtWidgets.QLayoutItem | None:  # noqa: N802 (Qt override)
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int) -> QtWidgets.QLayoutItem | None:  # noqa: N802 (Qt override)
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self) -> QtCore.Qt.Orientation:  # noqa: N802 (Qt override)
        return QtCore.Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802 (Qt override)
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802 (Qt override)
        return self._place(QtCore.QRect(0, 0, width, 0), move=False)

    def setGeometry(self, rect: QtCore.QRect) -> None:  # noqa: N802 (Qt override)
        super().setGeometry(rect)
        self._place(rect, move=True)

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802 (Qt override)
        """Everything on one line."""
        shown = self._shown()
        m = self.contentsMargins()
        width = sum(i.sizeHint().width() for i in shown) + self._gap() * max(len(shown) - 1, 0)
        height = max((i.sizeHint().height() for i in shown), default=0)
        return QtCore.QSize(width + m.left() + m.right(), height + m.top() + m.bottom())

    def minimumSize(self) -> QtCore.QSize:  # noqa: N802 (Qt override)
        """The widest item: anything wider wraps."""
        size = QtCore.QSize()
        for item in self._shown():
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        return size + QtCore.QSize(m.left() + m.right(), m.top() + m.bottom())

    def _shown(self) -> list[QtWidgets.QLayoutItem]:
        # A hidden widget's item is empty; a group whose widgets are all hidden has no size.
        return [i for i in self._items if not i.isEmpty() and i.sizeHint().width() > 0]

    def _gap(self, vertical: bool = False) -> int:
        gap = self.spacing()
        if gap >= 0:
            return gap
        style = self.parentWidget().style() if self.parentWidget() else QtWidgets.QApplication.style()
        metric = (QtWidgets.QStyle.PixelMetric.PM_LayoutVerticalSpacing if vertical
                  else QtWidgets.QStyle.PixelMetric.PM_LayoutHorizontalSpacing)
        return max(style.pixelMetric(metric), 6)

    def _place(self, rect: QtCore.QRect, move: bool) -> int:
        """Lay the items out in `rect` (if `move`); the height they take.

        A line takes items while their least widths fit, as a QHBoxLayout would squeeze them,
        then shares what's left over between them, up to the widths they ask for.
        """
        m = self.contentsMargins()
        area = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        gap, line_gap = self._gap(), self._gap(vertical=True)
        lines: list[list[tuple[QtWidgets.QLayoutItem, int, QtCore.QSize]]] = [[]]
        x = 0
        for item in self._shown():
            least = item.minimumSize().width()
            if lines[-1] and x + gap + least > area.width():
                lines.append([])
                x = 0
            x += (gap if lines[-1] else 0) + least
            lines[-1].append((item, least, item.sizeHint().expandedTo(item.minimumSize())))
        y = area.y()
        for line in lines:
            height = max((size.height() for _, _, size in line), default=0)
            spare = area.width() - sum(least for _, least, _ in line) - gap * (len(line) - 1)
            wanted = sum(size.width() - least for _, least, size in line)
            share = min(1.0, spare / wanted) if wanted > 0 else 0.0
            x = area.x()
            for item, least, size in line:
                width = min(least + int(share * (size.width() - least)), max(area.width(), least))
                if move:  # centred on the line, so a checkbox lines up with a taller combo box
                    item.setGeometry(QtCore.QRect(x, y + (height - size.height()) // 2, width, size.height()))
                x += width + gap
            y += height + line_gap
        return y - line_gap - area.y() + m.top() + m.bottom() if self._shown() else 0


def group(*items: QtWidgets.QWidget | str) -> QtWidgets.QWidget:
    """Widgets (or label texts) kept together on one line of a FlowLayout."""
    row = QtWidgets.QHBoxLayout()
    for item in items:
        row.addWidget(QtWidgets.QLabel(item) if isinstance(item, str) else item)
    return _holder(row)


def _holder(layout: QtWidgets.QLayout) -> QtWidgets.QWidget:
    widget = QtWidgets.QWidget()
    layout.setContentsMargins(0, 0, 0, 0)
    widget.setLayout(layout)
    return widget


class ColumnScroll(QtWidgets.QScrollArea):
    """A column of controls: scrolls up and down, never sideways, and is as narrow as they allow."""

    def __init__(self, widget: QtWidgets.QWidget) -> None:
        super().__init__()
        self.setWidget(widget)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def minimumSizeHint(self) -> QtCore.QSize:  # noqa: N802 (Qt override)
        width = (self.widget().minimumSizeHint().width() + self.verticalScrollBar().sizeHint().width()
                 + 2 * self.frameWidth())
        return QtCore.QSize(width, super().minimumSizeHint().height())


def plot_height(plot: QtWidgets.QWidget, height: int) -> None:
    """Let a plot be as short as `height` px, and ask for no more (pyqtgraph's views ask for 480)."""
    plot.setMinimumHeight(height)
    plot.sizeHint = lambda: QtCore.QSize(height, height)  # read by the layouts through PySide
