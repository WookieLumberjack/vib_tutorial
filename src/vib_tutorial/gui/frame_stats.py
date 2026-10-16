"""Frame timing overlay, for measuring the app's frame rate on a real display.

Off unless the environment variable VIB_FRAME_STATS is set when the window is
built: ``VIB_FRAME_STATS=1`` shows the overlay, and any other value is also a
CSV file to log every frame to (``VIB_FRAME_STATS=frames.csv``).

Each page's frame loop reports the computing it does (`frame`): simulating,
updating the plots' data. The window reports how long each repaint takes
(`paint`): the widgets drawing themselves and the result handed to the
display. A frame's paint is everything painted between its start and the next
frame's. The overlay shows the page whose frames ran last, over the last
`WINDOW` frames.
"""

from __future__ import annotations

import os
import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

ENV = "VIB_FRAME_STATS"
WINDOW = 120  # frames the overlay summarizes (~2 s at 60 fps)
GAP = 0.5  # s between frames that starts a new run (a page shown again, a pause)
LATE = 1.5  # a frame is late when it starts this many target intervals after the last
REFRESH_S = 0.25  # s between overlay updates


@dataclass
class Frame:
    start: float  # s, perf_counter
    interval: float  # s since the page's previous frame started (nan for a run's first)
    work: float  # s computing in the page's frame loop
    paint: float  # s repainting until the next frame started
    sim_t: float | None  # s, the page's simulated time after the frame


class FrameStats:
    """Collects frame timings per page; `summary` gives the numbers the overlay shows."""

    def __init__(self, log_path: str | None = None) -> None:
        self.frames: dict[str, deque[Frame]] = {}
        self.targets: dict[str, float] = {}
        self.page: str | None = None
        self._last: dict[str, Frame] = {}
        self._log = open(log_path, "w", buffering=1) if log_path else None
        if self._log:
            self._log.write("wall_s,page,interval_ms,work_ms,paint_ms,sim_t\n")
        self._t0 = time.perf_counter()

    def close(self) -> None:
        if self._log:
            self._flush_log(self._last.get(self.page)) if self.page else None
            self._log.close()
            self._log = None

    def record(self, page: str, start: float, work: float, sim_t: float | None, target: float) -> None:
        """A frame of `page` started at `start` (perf_counter) and computed for `work` s."""
        previous = self._last.get(page)
        interval = start - previous.start if previous else float("nan")
        if interval > GAP:
            interval = float("nan")
        if previous is not None:
            self._flush_log(previous)
        frame = Frame(start, interval, work, 0.0, sim_t)
        self.frames.setdefault(page, deque(maxlen=WINDOW)).append(frame)
        self._last[page] = frame
        self.targets[page] = target
        self.page = page

    def paint(self, seconds: float) -> None:
        """A repaint took `seconds`: charged to the frame that ran last."""
        if self.page is not None:
            self._last[self.page].paint += seconds

    def _flush_log(self, f: Frame | None) -> None:
        if self._log is None or f is None:
            return
        sim = "" if f.sim_t is None else f"{f.sim_t:.6f}"
        self._log.write(f"{f.start - self._t0:.4f},{self.page},{f.interval * 1e3:.3f},"
                        f"{f.work * 1e3:.3f},{f.paint * 1e3:.3f},{sim}\n")

    def summary(self, page: str | None = None) -> dict[str, float] | None:
        """fps, interval/work/paint medians and 95th percentiles (ms), late share, sim speed."""
        page = page or self.page
        frames = list(self.frames.get(page, ()))[:-1]  # the last frame's paint is still to come
        if len(frames) < 2:
            return None
        interval = np.array([f.interval for f in frames])
        interval = interval[np.isfinite(interval)]
        work = np.array([f.work for f in frames])
        paint = np.array([f.paint for f in frames])
        busy = work + paint
        out = {
            "frames": len(frames),
            "fps": 1.0 / interval.mean() if interval.size else float("nan"),
            "interval_p95": np.percentile(interval, 95) * 1e3 if interval.size else float("nan"),
            "work_med": np.median(work) * 1e3,
            "work_p95": np.percentile(work, 95) * 1e3,
            "paint_med": np.median(paint) * 1e3,
            "paint_p95": np.percentile(paint, 95) * 1e3,
            "busy_p95": np.percentile(busy, 95) * 1e3,
            "late": float(np.mean(interval > LATE * self.targets[page])) if interval.size else 0.0,
            "sim_speed": float("nan"),
        }
        sims = [(f.start, f.sim_t) for f in frames if f.sim_t is not None]
        if len(sims) >= 2 and sims[-1][0] > sims[0][0]:
            out["sim_speed"] = (sims[-1][1] - sims[0][1]) / (sims[-1][0] - sims[0][0])
        return out


STATS: FrameStats | None = None


@contextmanager
def frame(page: str, sim_t: Callable[[], float | None] | None = None, target_ms: float = 16.0) -> Iterator[None]:
    """Time one frame of `page`'s loop; `sim_t` reads its simulated time after it. Free when off."""
    if STATS is None:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        work = time.perf_counter() - start
        STATS.record(page, start, work, sim_t() if sim_t else None, target_ms * 1e-3)


class Overlay(QtWidgets.QLabel):
    """The numbers, in the window's bottom-right corner, never in the way of the mouse."""

    def __init__(self, window: QtWidgets.QWidget, stats: FrameStats) -> None:
        super().__init__(window)
        self.stats = stats
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAutoFillBackground(True)  # the theme's window colour, so the text stays legible
        self.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        self.setMargin(6)
        self.setText("frame stats: waiting for frames")
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(int(REFRESH_S * 1000))
        window.installEventFilter(self)
        self._place()

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:  # noqa: N802 (Qt override)
        if event.type() == QtCore.QEvent.Type.Resize:
            self._place()
        return False

    def _place(self) -> None:
        self.adjustSize()
        parent = self.parentWidget()
        self.move(parent.width() - self.width() - 12, parent.height() - self.height() - 30)
        self.raise_()

    def refresh(self) -> None:
        s = self.stats.summary()
        if s is None:
            return
        screen = self.screen()
        hz = screen.refreshRate() if screen else 60.0
        budget = 1000.0 / hz
        lines = [
            f"{self.stats.page}  ·  display {hz:.0f} Hz, {budget:.1f} ms per refresh",
            f"fps        {s['fps']:6.1f}   interval p95 {s['interval_p95']:6.1f} ms",
            f"work  med {s['work_med']:6.1f}   p95 {s['work_p95']:6.1f} ms",
            f"paint med {s['paint_med']:6.1f}   p95 {s['paint_p95']:6.1f} ms",
            f"busy p95  {s['busy_p95']:6.1f} ms ({s['busy_p95'] / budget:.0%} of a refresh)",
            f"late      {s['late']:6.0%} of frames",
        ]
        if np.isfinite(s["sim_speed"]):
            lines.append(f"sim speed {s['sim_speed']:6.2f}x real time")
        self.setText("\n".join(lines))
        self._place()


def install(window: QtWidgets.QWidget) -> Overlay | None:
    """Turn the stats on for `window` if VIB_FRAME_STATS is set; the overlay, or None."""
    global STATS
    value = os.environ.get(ENV)
    if not value:
        return None
    STATS = FrameStats(None if value == "1" else value)
    return Overlay(window, STATS)


def uninstall() -> None:
    global STATS
    if STATS is not None:
        STATS.close()
    STATS = None
