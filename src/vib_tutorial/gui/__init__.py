"""Qt (PySide6 + pyqtgraph) user interface."""

from __future__ import annotations

import sys


def run() -> int:
    import pyqtgraph as pg
    from PySide6 import QtWidgets

    from .main_window import MainWindow

    pg.setConfigOptions(antialias=True, background="w", foreground="k")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Vibration Tutorial")
    window = MainWindow()
    window.show()
    return app.exec()
