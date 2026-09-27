"""Qt (PySide6 + pyqtgraph) user interface."""

from __future__ import annotations

import sys


def make_app():
    """Create (or reuse) the QApplication, set up for the app's light colours."""
    import pyqtgraph as pg
    from PySide6 import QtCore, QtWidgets

    pg.setConfigOptions(antialias=True, background="w", foreground="k")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Vibration Tutorial")
    # The plots, animation and energy bars are drawn in fixed light colours, so
    # keep the widgets light on a dark desktop too (dark theme: see the README's
    # ideas for extension).
    app.styleHints().setColorScheme(QtCore.Qt.ColorScheme.Light)
    return app


def run() -> int:
    from .main_window import MainWindow

    app = make_app()
    window = MainWindow()
    window.show()
    return app.exec()
