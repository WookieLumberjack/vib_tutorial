"""Qt (PySide6 + pyqtgraph) user interface."""

from __future__ import annotations

import os
import sys
import threading
import time


def make_app(theme: str | None = None):
    """Create (or reuse) the QApplication in a colour theme (default: the one last picked).

    `theme` is a name from theming.names(): "System" follows the desktop's light
    or dark scheme.
    """
    import pyqtgraph as pg
    from PySide6 import QtWidgets

    from . import axes, theming

    pg.setConfigOptions(antialias=True)
    axes.install_opaque_grid()
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


def settle_plots(widget) -> None:
    """Apply the new view ranges of the plots showing in `widget` now, at the end of a frame.

    pyqtgraph defers a range change to the start of the next paint, where moving the items
    to it marks them dirty again, so every plot whose range a frame changes (the strip
    charts scroll, the orbits rescale) was painted twice. Call it after a frame's drawing.
    """
    import pyqtgraph as pg

    for view in widget.findChildren(pg.GraphicsView):
        if view.isVisible():
            view.scene().prepareForPaint()


def _preload_scipy() -> None:
    """Import the SciPy modules that only the Virtual modal test uses.

    core imports them on first use so they stay out of startup (~0.9 s); loading
    them in the background once the window is up keeps that page's first visit
    from stalling.
    """
    import scipy.optimize
    import scipy.signal  # noqa: F401


def _analysis_worker_runs(timeout: float = 60.0) -> bool:
    """True if the rotor page's analysis process starts and answers (in a packaged app it's the app itself)."""
    from ..core.rotor import RotorSystem
    from ..core.rotor_analysis import AnalysisWorker

    worker = AnalysisWorker()
    try:
        ticket = worker.submit(RotorSystem(), 6000.0)
        end = time.perf_counter() + timeout
        while time.perf_counter() < end:
            if worker.result(ticket) is not None:
                return worker.running  # not worked out here, because the process didn't start
            time.sleep(0.05)
        return False
    finally:
        worker.close()


def record_startup(window, path: str, launched_at: float | None = None) -> None:
    """Write how long startup took to `path` (JSON) once `window` first paints.

    Seconds since the interpreter reached the package (`python`), and, when `launched_at`
    (a time.time() taken just before launching) is given, since then (`launch`), which also
    counts the frozen app's bootloader. `window` is the time MainWindow() took to build.
    """
    import json

    from PySide6 import QtCore, QtWidgets

    from .. import STARTED

    built = time.perf_counter() - STARTED

    class FirstPaint(QtCore.QObject):
        def eventFilter(self, obj, event):
            if event.type() == QtCore.QEvent.Type.Paint and obj.isWidgetType() and obj.window() is window:
                QtWidgets.QApplication.instance().removeEventFilter(self)
                timings = {"window": round(built, 3), "python": round(time.perf_counter() - STARTED, 3)}
                if launched_at is not None:
                    timings["launch"] = round(time.time() - launched_at, 3)
                with open(path, "w") as f:
                    json.dump(timings, f)
            return False

    window._first_paint = FirstPaint(window)
    QtWidgets.QApplication.instance().installEventFilter(window._first_paint)


def run() -> int:
    from PySide6 import QtCore

    from . import frame_stats
    from .main_window import MainWindow

    app = make_app()
    window = MainWindow()
    if path := os.environ.get("VIB_TUTORIAL_STARTUP_FILE"):  # set by the release builds' smoke test
        launched = os.environ.get("VIB_TUTORIAL_LAUNCHED_AT")
        record_startup(window, path, float(launched) if launched else None)
    window.show()
    preload = threading.Thread(target=_preload_scipy, name="preload-scipy", daemon=True)
    QtCore.QTimer.singleShot(0, preload.start)  # after the first paint
    if "--smoke-test" in sys.argv:  # release builds check the packaged app starts, then quit
        from .rotor_figures import FIGURES, render  # the rotor page is built lazily: check QtSvg is packaged

        if any(render(build()).isNull() for build in FIGURES.values()):
            return 1
        if not _analysis_worker_runs():
            return 1
        from ..core import native

        if native.module is None:  # the build must ship the Rust kernels, not fall back to the Python
            return 1
        QtCore.QTimer.singleShot(2000, app.quit)
    code = app.exec()
    close_graphics(window)
    frame_stats.uninstall()
    if preload.is_alive():  # don't tear the interpreter down mid-import
        preload.join()
    return code
