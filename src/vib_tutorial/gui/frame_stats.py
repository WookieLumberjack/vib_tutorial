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

The CSV log has a row per frame, and between them rows saying what changed, so
the frames can be put down to what was on screen (`scripts/frame_report.py`
summarizes a log that way):

- ``view``: what shows (the page, its sub-tabs as far down as they go, a
  page's own `frame_context()`, such as the rotor's preset), the window's size,
  its device pixel ratio and the display's refresh rate. Checked every
  `REFRESH_S`, logged when it changes.
- ``input``: what the user did: a button pressed or toggled, a choice in a
  list, a value typed or stepped, a tab picked, a click or drag on a plot, each
  with what it's labelled and its new value.
"""

from __future__ import annotations

import csv
import os
import re
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
REFRESH_S = 0.25  # s between overlay updates (and checks of what shows)
COLUMNS = ("wall_s", "kind", "page", "interval_ms", "work_ms", "paint_ms", "sim_t", "detail")
DRAG_PX = 5  # a press and release this far apart is a drag, not a click


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
        self._log = open(log_path, "w", buffering=1, newline="", encoding="utf-8") if log_path else None
        self._csv = csv.writer(self._log) if self._log else None
        if self._csv:
            self._csv.writerow(COLUMNS)
        self._t0 = time.perf_counter()

    def close(self) -> None:
        if self._log:
            self._flush_log(self._last.get(self.page)) if self.page else None
            self._log.close()
            self._log = self._csv = None

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
        if self._csv is None or f is None:
            return
        sim = "" if f.sim_t is None else f"{f.sim_t:.6f}"
        self._csv.writerow((f"{f.start - self._t0:.4f}", "frame", self.page, f"{f.interval * 1e3:.3f}",
                            f"{f.work * 1e3:.3f}", f"{f.paint * 1e3:.3f}", sim, ""))

    def note(self, kind: str, detail: str, page: str | None = None) -> None:
        """Log a ``view`` or ``input`` row: what changed, now (between the frames around it)."""
        if self._csv is not None:
            self._csv.writerow((f"{time.perf_counter() - self._t0:.4f}", kind, page or self.page or "", "", "", "",
                                "", detail))

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
        self._view = ""
        self.inputs = InputLog(window, stats)
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
        self.check_view()
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


    def check_view(self) -> None:
        """Log a ``view`` row if what shows changed since the last check."""
        window = self.parentWidget()
        screen = window.screen()
        view = (f"{describe_view(window)} | {window.width()}x{window.height()} @{window.devicePixelRatioF():g}x"
                f" {screen.refreshRate() if screen else 0:.0f} Hz")
        if view != self._view:
            self._view = view
            self.stats.note("view", view, page=_page_title(window))


def _plain(text: str) -> str:
    """`text` without HTML, accelerator ampersands or runs of whitespace."""
    text = re.sub(r"<[^>]+>", " ", text).replace("&&", "\0").replace("&", "").replace("\0", "&").replace("■", "")
    return " ".join(text.split())


def _page_title(window: QtWidgets.QWidget) -> str:
    tabs = getattr(window, "pages", None)
    return _plain(tabs.tabText(tabs.currentIndex())) if isinstance(tabs, QtWidgets.QTabWidget) else ""


def describe_view(window: QtWidgets.QWidget) -> str:
    """What shows: the tabs picked, outermost first ("Jeffcott rotor > Campbell"), and the page's own context."""
    parts: list[str] = []
    widget = window
    while True:
        tabs = next((t for t in widget.findChildren(QtWidgets.QTabWidget) if t.isVisible()
                     and _tab_parent(t, widget)), None)
        if tabs is None or tabs.currentWidget() is None:
            break
        parts.append(_plain(tabs.tabText(tabs.currentIndex())))
        widget = tabs.currentWidget()
        context = getattr(widget, "frame_context", None)
        if callable(context) and (text := context()):
            parts.append(f"[{text}]")
    return " > ".join(parts)


def _tab_parent(tabs: QtWidgets.QTabWidget, widget: QtWidgets.QWidget) -> bool:
    """True if no other tab widget stands between `tabs` and `widget` (it's the next level down)."""
    parent = tabs.parentWidget()
    while parent is not None and parent is not widget:
        if isinstance(parent, QtWidgets.QTabWidget):
            return False
        parent = parent.parentWidget()
    return parent is widget


class InputLog(QtCore.QObject):
    """Logs what the user does in `window` as ``input`` rows: an application-wide event filter."""

    KINDS = (QtWidgets.QAbstractButton, QtWidgets.QComboBox, QtWidgets.QAbstractSpinBox, QtWidgets.QAbstractSlider,
             QtWidgets.QTabBar, QtWidgets.QGraphicsView, QtWidgets.QLineEdit)

    def __init__(self, window: QtWidgets.QWidget, stats: FrameStats) -> None:
        super().__init__(window)
        self.window, self.stats = window, stats
        self._values: dict[int, str] = {}  # what each widget was last logged as, by id
        self._last: tuple[int, str, float] = (0, "", 0.0)  # the last row's widget, value and time
        self._press: QtCore.QPointF | None = None
        QtWidgets.QApplication.instance().installEventFilter(self)

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:  # noqa: N802 (Qt override)
        if STATS is not self.stats:  # turned off (uninstall)
            return False
        t = event.type()
        if t == QtCore.QEvent.Type.MouseButtonPress and isinstance(obj, QtWidgets.QWidget):
            self._press = event.globalPosition()
        elif t in (QtCore.QEvent.Type.MouseButtonRelease, QtCore.QEvent.Type.KeyRelease, QtCore.QEvent.Type.Wheel):
            widget = self._control(obj)
            if widget is not None:
                how = "key" if t == QtCore.QEvent.Type.KeyRelease else "wheel" if t == QtCore.QEvent.Type.Wheel else (
                    "drag" if self._press is not None and (event.globalPosition() - self._press).manhattanLength()
                    > DRAG_PX else "click")
                # After the widget has handled it: its value is the new one then.
                QtCore.QTimer.singleShot(0, lambda w=widget, h=how: self._log(w, h))
        return False

    def _control(self, obj: QtCore.QObject) -> QtWidgets.QWidget | None:
        """The control `obj` belongs to (a spin box's line edit, a plot's viewport), if it's in the window."""
        widget = obj if isinstance(obj, QtWidgets.QWidget) else None
        while widget is not None:
            if isinstance(widget, self.KINDS):
                if isinstance(widget, QtWidgets.QLineEdit) and isinstance(widget.parentWidget(),
                                                                          (QtWidgets.QAbstractSpinBox,
                                                                           QtWidgets.QComboBox)):
                    widget = widget.parentWidget()
                    continue
                return widget if widget.window() is self.window or _is_popup_of(widget, self.window) else None
            widget = widget.parentWidget()
        return None

    def _log(self, widget: QtWidgets.QWidget, how: str) -> None:
        try:
            label, value = describe_input(widget, how)
        except RuntimeError:  # deleted meanwhile
            return
        key = id(widget)
        valued = isinstance(widget, (QtWidgets.QComboBox, QtWidgets.QAbstractSpinBox, QtWidgets.QLineEdit,
                                     QtWidgets.QAbstractSlider))
        if (how in ("key", "wheel") or valued) and self._values.get(key) == value:
            return  # nothing changed (a key that moved the cursor, a click into the box)
        now = time.perf_counter()
        if self._last[:2] == (key, value) and now - self._last[2] < 0.05:
            return  # the same click, passed on to a parent
        self._values[key], self._last = value, (key, value, now)
        self.stats.note("input", f"{label}: {value}" if label else value, page=_page_title(self.window))


def _is_popup_of(widget: QtWidgets.QWidget, window: QtWidgets.QWidget) -> bool:
    """True for a combo box's or menu's popup opened from `window`."""
    parent = widget.window().parentWidget()
    return parent is not None and parent.window() is window


def describe_input(widget: QtWidgets.QWidget, how: str) -> tuple[str, str]:
    """(what the control is called, its value now) for an ``input`` row."""
    if isinstance(widget, QtWidgets.QTabBar):
        return "tab", _plain(widget.tabText(widget.currentIndex()))
    if isinstance(widget, QtWidgets.QAbstractButton):
        text = _plain(widget.text()) or _plain(widget.toolTip())[:40]
        if widget.isCheckable() and not isinstance(widget, QtWidgets.QRadioButton):
            return _label(widget) or text, ("on" if widget.isChecked() else "off") if _label(widget) else (
                f"{text} {'on' if widget.isChecked() else 'off'}")
        return _label(widget), text
    if isinstance(widget, QtWidgets.QComboBox):
        return _label(widget), widget.currentText()
    if isinstance(widget, (QtWidgets.QAbstractSpinBox, QtWidgets.QLineEdit)):
        return _label(widget), widget.text()
    if isinstance(widget, QtWidgets.QAbstractSlider):
        return _label(widget) or "slider", str(widget.value())
    if isinstance(widget, QtWidgets.QGraphicsView):
        return _plot_name(widget), how
    return "", how


def _label(widget: QtWidgets.QWidget) -> str:
    """What `widget` is labelled in the window: its form row's label, the label before it in a row of
    controls, or its grid column's heading (and row's label); else its tooltip's start."""
    w = widget
    for _ in range(3):  # the control itself, or the box of controls it is in
        parent = w.parentWidget()
        if parent is None:
            break
        layout = parent.layout()
        found = _holding(layout, w) if layout is not None else None
        if isinstance(layout, QtWidgets.QFormLayout) and (label := _form_label(layout, w)):
            return label
        if isinstance(found, QtWidgets.QGridLayout):
            row, col, _, _ = found.getItemPosition(found.indexOf(w))
            heading = _text_at(found, 0, col) if row > 0 else ""
            side = next((t for c in range(col - 1, -1, -1) if (t := _text_at(found, row, c))), "")
            if heading or side:
                return " ".join(t for t in (side, heading) if t)
        if isinstance(found, QtWidgets.QBoxLayout):
            before = [found.itemAt(i).widget() for i in range(found.indexOf(w))]
            label = next((x for x in reversed(before) if isinstance(x, QtWidgets.QLabel) and x.text()), None)
            if label is not None:
                return _plain(label.text()).rstrip(":")
        w = parent
    tip = _plain(widget.toolTip())
    return re.split(r"(?<=[a-z])[.:;(]", tip, maxsplit=1)[0][:40] if tip else widget.objectName()


def _holding(layout: QtWidgets.QLayout, widget: QtWidgets.QWidget) -> QtWidgets.QLayout | None:
    """The layout (`layout` or one nested in it) with `widget` among its own items."""
    if layout.indexOf(widget) >= 0:
        return layout
    for i in range(layout.count()):
        inner = layout.itemAt(i).layout()
        if inner is not None and (found := _holding(inner, widget)) is not None:
            return found
    return None


def _text_at(grid: QtWidgets.QGridLayout, row: int, col: int) -> str:
    item = grid.itemAtPosition(row, col)
    label = item.widget() if item is not None else None
    return _plain(label.text()).rstrip(":") if isinstance(label, QtWidgets.QLabel) else ""


def _form_label(form: QtWidgets.QFormLayout, widget: QtWidgets.QWidget) -> str:
    """The label of the form's row holding `widget`, itself or in a box or layout of controls."""
    for row in range(form.rowCount()):
        field = form.itemAt(row, QtWidgets.QFormLayout.ItemRole.FieldRole)
        label = form.itemAt(row, QtWidgets.QFormLayout.ItemRole.LabelRole)
        if field is None or label is None or not isinstance(label.widget(), QtWidgets.QLabel):
            continue
        held = field.widget()
        if (held is not None and (held is widget or held.isAncestorOf(widget))) or (
                field.layout() is not None and _in_layout(field.layout(), widget)):
            return _plain(label.widget().text()).rstrip(":")
    return ""


def _in_layout(layout: QtWidgets.QLayout, widget: QtWidgets.QWidget) -> bool:
    for i in range(layout.count()):
        item = layout.itemAt(i)
        if item.widget() is not None and (item.widget() is widget or item.widget().isAncestorOf(widget)):
            return True
        if item.layout() is not None and _in_layout(item.layout(), widget):
            return True
    return False


def _plot_name(view: QtWidgets.QGraphicsView) -> str:
    """A plot's title (or its axes' labels), for a click or drag on it."""
    scene = view.scene()
    names = []
    for item in scene.items() if scene is not None else ():
        title = getattr(item, "titleLabel", None)
        if title is not None and getattr(title, "text", ""):
            names.append(_plain(title.text))
    if names:
        return "plot " + " / ".join(sorted(set(names)))[:60]
    return "plot " + (view.objectName() or type(view).__name__)


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
