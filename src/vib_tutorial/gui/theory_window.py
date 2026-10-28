"""The Theory window: every page's Theory notes in a window of their own, beside the main one.

A contents tree (groups, pages, each page's sections) and the notes, with Back and Forward. Links to
another page's notes stay in the window; *Open this page* shows the page in the main window. With
*Follow the page* on, it shows the notes of each page the main window turns to. Each page keeps the
place it was read to, so going back to one carries on where it left off.

The notes are the pages' own module-level HTML, so a page needn't be built to read its notes.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Callable

from PySide6 import QtCore, QtGui, QtWidgets

from . import links
from .rotor_figures import FigureBrowser
from .theming import SETTINGS

HEADING = re.compile(r"<h3>(.*?)</h3>", re.S)
ANCHOR = re.compile(r'<a name="([^"]+)"></a>')
PAGE_ROLE = QtCore.Qt.ItemDataRole.UserRole  # (page attribute, anchor) of a tree item


def page_html(name: str) -> str:
    """Page `name`'s Theory notes (an attribute from links.COURSE), as its Theory tab shows them."""
    if name == "sim_page":
        from .background import BACKGROUND_HTML as html
    else:
        from .main_window import _page_class

        module = sys.modules[_page_class(name).__module__]
        html = getattr(module, "NOTES_HTML", None) or module.THEORY_HTML
    return anchored(html)


def anchored(html: str) -> str:
    """`html` with an anchor on every <h3> that lacks one (section-1, section-2, …), so the contents reach it."""
    count = 0

    def add(match: re.Match) -> str:
        nonlocal count
        count += 1
        inner = match.group(1)
        return match.group(0) if ANCHOR.search(inner) else f'<h3><a name="section-{count}"></a>{inner}</h3>'

    return HEADING.sub(add, html)


def sections(html: str) -> list[tuple[str, str]]:
    """(anchor, plain title) of each <h3> in `html`, in order."""
    found = []
    for match in HEADING.finditer(html):
        anchor = ANCHOR.search(match.group(1))
        if anchor:
            title = QtGui.QTextDocumentFragment.fromHtml(match.group(1)).toPlainText().strip()
            found.append((anchor.group(1), title))
    return found


class TheoryWindow(QtWidgets.QWidget):
    """The notes of any page, in their own window. `open_page(name)` shows a page in the main window."""

    def __init__(self, open_page: Callable[[str], None], parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent, QtCore.Qt.WindowType.Window)
        self.setWindowTitle("Theory")
        self._open_page = open_page
        self.page: str | None = None
        self._places: dict[str, int] = {}  # page → where it was read to (scroll bar value)
        self._back: list[tuple[str, int]] = []
        self._forward: list[tuple[str, int]] = []

        self.back_button = QtWidgets.QToolButton()
        self.back_button.setText("◀ Back")
        self.back_button.clicked.connect(self.back)
        self.forward_button = QtWidgets.QToolButton()
        self.forward_button.setText("Forward ▶")
        self.forward_button.clicked.connect(self.forward)
        self.follow = QtWidgets.QCheckBox("Follow the page")
        self.follow.setToolTip("<p>Show the notes of each page the main window turns to. Off, the notes stay "
                               "where they are as you move between pages.</p>")
        self.open_button = QtWidgets.QToolButton()
        self.open_button.setText("Open this page")
        self.open_button.setToolTip("Show the page these notes belong to in the main window.")
        self.open_button.clicked.connect(lambda: self.page and self._open_page(self.page))
        bar = QtWidgets.QHBoxLayout()
        for w in (self.back_button, self.forward_button, self.follow):
            bar.addWidget(w)
        bar.addStretch(1)
        bar.addWidget(self.open_button)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.itemClicked.connect(self._on_item)
        self.tree.itemActivated.connect(self._on_item)
        self.tree.itemExpanded.connect(self._fill)
        self._items: dict[str, QtWidgets.QTreeWidgetItem] = {}
        number = 1
        for group, pages in links.COURSE:
            top = QtWidgets.QTreeWidgetItem([group])
            font = top.font(0)
            font.setBold(True)
            top.setFont(0, font)
            top.setFlags(top.flags() & ~QtCore.Qt.ItemFlag.ItemIsSelectable)
            self.tree.addTopLevelItem(top)
            for name, title, about in pages:
                item = QtWidgets.QTreeWidgetItem([f"{number}. {title.replace('&&', '&')}"])
                item.setData(0, PAGE_ROLE, (name, ""))
                item.setToolTip(0, about)
                item.setChildIndicatorPolicy(QtWidgets.QTreeWidgetItem.ChildIndicatorPolicy.ShowIndicator)
                top.addChild(item)
                self._items[name] = item
                number += 1
            top.setExpanded(True)

        self.browser = FigureBrowser()
        links.wire(self.browser, self.show_page)  # another page's notes stay in this window

        split = QtWidgets.QSplitter()
        split.addWidget(self.tree)
        split.addWidget(self.browser)
        split.setStretchFactor(1, 1)
        split.setSizes([260, 640])
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(bar)
        layout.addWidget(split)
        self.resize(950, 800)
        geometry = QtCore.QSettings(*SETTINGS).value("theory_window/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        self.follow.setChecked(QtCore.QSettings(*SETTINGS).value("theory_window/follow", "false") == "true")
        self.follow.toggled.connect(lambda on: QtCore.QSettings(*SETTINGS).setValue(
            "theory_window/follow", "true" if on else "false"))
        QtGui.QShortcut(QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Back), self, self.back)
        QtGui.QShortcut(QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Forward), self, self.forward)
        self._update_buttons()

    # --------------------------------------------------------------- navigation
    def show_page(self, name: str, anchor: str = "", remember: bool = True) -> None:
        """Show page `name`'s notes: at `anchor` if given, else where it was last read to."""
        if name != self.page:
            if remember and self.page is not None:
                self._back.append((self.page, self._scroll()))
                self._forward.clear()
            self._load(name)
        if anchor:
            QtCore.QTimer.singleShot(0, lambda: self.browser.scrollToAnchor(anchor))  # once it's laid out
        self._update_buttons()

    def follow_page(self, name: str) -> None:
        """The main window turned to page `name`: show its notes if following."""
        if self.follow.isChecked():
            self.show_page(name)

    def back(self) -> None:
        if self._back:
            self._forward.append((self.page, self._scroll()))
            self._step_to(*self._back.pop())

    def forward(self) -> None:
        if self._forward:
            self._back.append((self.page, self._scroll()))
            self._step_to(*self._forward.pop())

    def _step_to(self, name: str, place: int) -> None:
        self._load(name)
        self._places[name] = place
        self._restore(place)
        self._update_buttons()

    def _load(self, name: str) -> None:
        if self.page is not None:
            self._places[self.page] = self._scroll()
            self._items[self.page].setExpanded(False)
        self.page = name
        self.browser.setHtml(page_html(name))
        self._restore(self._places.get(name, 0))
        item = self._items[name]
        item.setExpanded(True)  # its sections
        self.tree.setCurrentItem(item)
        self.tree.scrollToItem(item)
        self.setWindowTitle(f"Theory: {links.PAGE_TITLES[name].removesuffix(' page')}")

    def _restore(self, place: int) -> None:
        bar = self.browser.verticalScrollBar()
        bar.setValue(place)
        QtCore.QTimer.singleShot(0, lambda: bar.setValue(place))  # again once it's laid out

    def _scroll(self) -> int:
        return self.browser.verticalScrollBar().value()

    def _fill(self, item: QtWidgets.QTreeWidgetItem) -> None:
        """Put a page's sections under it, the first time it's expanded."""
        target = item.data(0, PAGE_ROLE)
        if target and not target[1] and item.childCount() == 0:
            for anchor, title in sections(page_html(target[0])):
                child = QtWidgets.QTreeWidgetItem([title])
                child.setData(0, PAGE_ROLE, (target[0], anchor))
                child.setToolTip(0, title)  # cut short when the tree is narrow
                item.addChild(child)

    def _on_item(self, item: QtWidgets.QTreeWidgetItem) -> None:
        target = item.data(0, PAGE_ROLE)
        if target:
            self.show_page(*target)

    def _update_buttons(self) -> None:
        self.back_button.setEnabled(bool(self._back))
        self.forward_button.setEnabled(bool(self._forward))
        self.open_button.setEnabled(self.page is not None)

    # --------------------------------------------------------------- window
    def apply_theme(self) -> None:
        """Set the notes again: their links' colour and the figures are fixed when the HTML is set."""
        if self.page is not None:
            place = self._scroll()
            self.browser.setHtml(page_html(self.page))
            self._restore(place)

    def save_geometry(self) -> None:
        QtCore.QSettings(*SETTINGS).setValue("theory_window/geometry", self.saveGeometry())

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802 (Qt's name)
        self.save_geometry()
        super().closeEvent(event)
