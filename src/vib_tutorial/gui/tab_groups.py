"""Tabs in groups: a row of group tabs, each with its own row of tabs below.

One row of ten tabs needs ~970 px, far more than a side pane has, so the rotor page
groups its views. Reach a page through `show_page` and `current_page`, which work
whichever group it's in, and see through a scroll area put round a page; `changed`
fires whenever the page showing changes.
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets


class TabGroups(QtWidgets.QTabWidget):
    changed = QtCore.Signal()  # the page showing changed (either row)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.currentChanged.connect(lambda _: self.changed.emit())

    def add_group(self, title: str, *pages: tuple[QtWidgets.QWidget, str]) -> None:
        """A group tab: one page goes in as it is, several get their own row of tabs."""
        if len(pages) == 1:
            self.addTab(pages[0][0], title)
            return
        inner = QtWidgets.QTabWidget()
        inner.setDocumentMode(True)  # no second frame inside the first
        for page, name in pages:
            inner.addTab(page, name)
        inner.currentChanged.connect(lambda _: self.changed.emit())
        self.addTab(inner, title)

    def show_page(self, page: QtWidgets.QWidget) -> None:
        """Bring `page` to the front, in its group."""
        while page is not self:
            tabs = page.parentWidget().parentWidget()  # a tab's page sits in its tab widget's stack
            if isinstance(tabs, QtWidgets.QTabWidget) and tabs.indexOf(page) >= 0:
                tabs.setCurrentWidget(page)
            page = page.parentWidget()

    def current_page(self) -> QtWidgets.QWidget:
        page = self.currentWidget()
        if self._is_group(page):
            page = page.currentWidget()
        return page.widget() if isinstance(page, QtWidgets.QScrollArea) else page

    def _is_group(self, widget: QtWidgets.QWidget) -> bool:
        return isinstance(widget, QtWidgets.QTabWidget) and widget.parentWidget().parentWidget() is self


def tucked_away(widget: QtWidgets.QWidget) -> bool:
    """True if `widget` is hidden, or a tab (at any level) hides it, whether or not its window is shown yet."""
    while widget is not None:
        if widget.isHidden():
            return True
        widget = widget.parentWidget()
        if widget is not None and widget.isWindow():
            return False  # a window never shown is hidden, but its tabs say what would show
    return False


class PageGroups(QtWidgets.QWidget):
    """The window's pages, in a row of group tabs, each group with its own row of pages.

    Fourteen pages in one row need ~1,800 px, more than most screens, so they are grouped
    (Foundations, Responses, …). The pages are also numbered in one flat order, the order
    they are meant to be read in, and the methods below mirror QTabWidget's over that
    order, so a caller can count, find and show a page without knowing its group.
    """

    currentChanged = QtCore.Signal(int)  # the flat index of the page now showing

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.groups = QtWidgets.QTabWidget()
        self.groups.setDocumentMode(True)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.groups)
        self._rows: list[QtWidgets.QTabWidget] = []
        self._current = -1
        self.groups.currentChanged.connect(self._changed)

    def add_group(self, title: str, *pages: tuple[QtWidgets.QWidget, str]) -> None:
        row = QtWidgets.QTabWidget()
        row.setDocumentMode(True)
        for page, name in pages:
            row.addTab(page, name)
        row.currentChanged.connect(self._changed)
        self._rows.append(row)
        self.groups.addTab(row, title)

    def setCornerWidget(self, widget: QtWidgets.QWidget, corner=QtCore.Qt.Corner.TopRightCorner) -> None:  # noqa: N802
        self.groups.setCornerWidget(widget, corner)

    # --- QTabWidget's API over the flat order
    def count(self) -> int:
        return sum(row.count() for row in self._rows)

    def _locate(self, index: int) -> tuple[QtWidgets.QTabWidget, int]:
        for row in self._rows:
            if index < row.count():
                return row, index
            index -= row.count()
        raise IndexError(index)

    def widget(self, index: int) -> QtWidgets.QWidget | None:
        try:
            row, i = self._locate(index)
        except IndexError:
            return None
        return row.widget(i)

    def tabText(self, index: int) -> str:  # noqa: N802
        row, i = self._locate(index)
        return row.tabText(i)

    def group_title(self, index: int) -> str:
        """The title of the group the page at `index` is in."""
        row, _ = self._locate(index)
        return self.groups.tabText(self._rows.index(row))

    def indexOf(self, widget: QtWidgets.QWidget) -> int:  # noqa: N802
        start = 0
        for row in self._rows:
            i = row.indexOf(widget)
            if i >= 0:
                return start + i
            start += row.count()
        return -1

    def currentIndex(self) -> int:  # noqa: N802
        row = self.groups.currentWidget()
        if row not in self._rows:
            return -1
        return sum(r.count() for r in self._rows[: self._rows.index(row)]) + row.currentIndex()

    def currentWidget(self) -> QtWidgets.QWidget | None:  # noqa: N802
        row = self.groups.currentWidget()
        return row.currentWidget() if row is not None else None

    def setCurrentIndex(self, index: int) -> None:  # noqa: N802
        row, i = self._locate(index)
        row.setCurrentIndex(i)
        self.groups.setCurrentWidget(row)

    def setCurrentWidget(self, widget: QtWidgets.QWidget) -> None:  # noqa: N802
        index = self.indexOf(widget)
        if index >= 0:
            self.setCurrentIndex(index)

    def replace(self, old: QtWidgets.QWidget, new: QtWidgets.QWidget) -> None:
        """Put `new` in `old`'s tab, with its title, showing if `old` was; `old` is taken out (not deleted)."""
        row, i = self._locate(self.indexOf(old))
        showing = row.currentIndex() == i
        row.blockSignals(True)  # the page showing doesn't change, only what's in its tab
        row.insertTab(i, new, row.tabText(i))
        if showing:
            row.setCurrentIndex(i)
        row.removeTab(i + 1)
        row.blockSignals(False)

    def _changed(self, _: int) -> None:
        index = self.currentIndex()
        if index != self._current:
            self._current = index
            self.currentChanged.emit(index)
