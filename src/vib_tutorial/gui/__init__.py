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


def close_graphics(widget) -> None:
    """Tear down every pyqtgraph view inside `widget`, as pyqtgraph expects before exit.

    GraphicsView.close() clears its scene; closing the window does not call it. Items
    left in a scene until PySide's own teardown at interpreter exit can segfault there.

    Stops the widget's timers first, so no frame is drawn into a cleared scene.

    PlotWidget.close() also does setParent(None), which makes Qt delete the view's layout
    item. PySide wraps those items when a filled layout is added to another (addLayout,
    addRow), and isn't told when Qt deletes one, so its wrapper stays registered at the
    freed address. A QWidgetAction later allocated there (pyqtgraph builds some for every
    plot's menu) resolves to that wrapper in QMenu.addAction and segfaults. Taking the item
    out with takeAt() first hands it to Python, which deletes it and unregisters it.
    """
    import pyqtgraph as pg
    from PySide6 import QtCore, QtWidgets

    for timer in widget.findChildren(QtCore.QTimer):
        timer.stop()
    for view in widget.findChildren(pg.GraphicsView):
        if view.closed:
            continue
        parent = view.parentWidget()
        if parent is not None:
            for layout in parent.findChildren(QtWidgets.QLayout):
                index = layout.indexOf(view)
                if index >= 0:
                    layout.takeAt(index)  # the returned item is Python's and freed right away
                    break
        view.close()


def run() -> int:
    from PySide6 import QtCore

    from .main_window import MainWindow

    app = make_app()
    window = MainWindow()
    window.show()
    if "--smoke-test" in sys.argv:  # release builds check the packaged app starts, then quit
        QtCore.QTimer.singleShot(2000, app.quit)
    code = app.exec()
    close_graphics(window)
    return code
