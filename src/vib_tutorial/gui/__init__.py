"""Qt (PySide6 + pyqtgraph) user interface."""

from __future__ import annotations

import sys


def make_app(theme: str | None = None):
    """Create (or reuse) the QApplication in a colour theme (default: the one last picked).

    `theme` is a name from theming.names(): "System" follows the desktop's light
    or dark scheme.
    """
    import pyqtgraph as pg
    from PySide6 import QtWidgets

    from . import theming

    pg.setConfigOptions(antialias=True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Vibration Tutorial")
    theming.apply(theme or theming.saved_name())
    return app


def run() -> int:
    from .main_window import MainWindow

    app = make_app()
    window = MainWindow()
    window.show()
    return app.exec()
