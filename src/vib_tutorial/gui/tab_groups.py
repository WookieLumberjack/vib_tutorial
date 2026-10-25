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
