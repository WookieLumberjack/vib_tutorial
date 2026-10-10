"""Time response tab of the Substructuring page: the reduced model simulated beside the full one.

Both models feel the same force at the tip. The two chains are animated one above the
other at the same scale, and the tip and interface displacements are plotted with their
differences, so the reduction error shows up as motion rather than as a number in a table.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core import CMSModel, ForceController, ForceKind, ForceSettings, ModalResult
from ..core.cms_response import CMSResponse
from .animation import ChainView
from .panels import spin
from .style import colors
from .theming import add_legend, mute

FRAME_MS = 16
KINDS = (ForceKind.STEP, ForceKind.HARMONIC, ForceKind.PULSE)
SPEEDS = (0.1, 0.25, 0.5, 1.0, 2.0)
MIN_SPAN = 1e-9  # m; smaller motions are float noise
DASH = QtCore.Qt.PenStyle.DashLine
TIP = (
    "<p>The full chain and the reduced model, stepped side by side with the same exact "
    "(first-order-hold) update and the same force F(t) at the tip. The reduced model's "
    "motion is recovered in physical coordinates, x = T r, so every difference between the two "
    "is the reduction error.</p>"
    "<p>A step shows the static error first (none for Craig–Bampton: its constraint modes are "
    "the exact static shapes) and then the ringing of the modes it gets wrong. Drive at a true "
    "natural frequency that the reduced model has shifted and the two drift out of phase; keep "
    "more modes and they lock together.</p>"
    "<p>Changing the model (method, interface, modes kept, or the chain) restarts both from rest.</p>"
)


class CMSRunView(QtWidgets.QWidget):
    """The full chain and a reduced model run side by side: force controls, frame timer, model changes.

    Subclasses build the display (`_build`), make the response (`_make_sim`), set
    up for a new model (`_on_model`), store what each advance returns (`_advance`)
    and draw (`_draw`).
    """

    tip = TIP

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.force = ForceController(ForceSettings(kind=ForceKind.STEP, amplitude=10.0))
        self.sim: CMSResponse | None = None
        self.model: CMSModel | None = None
        self._key: tuple | None = None

        self.header = QtWidgets.QLabel()
        self.header.setWordWrap(True)
        self.header.setToolTip(self.tip)

        self.controls = row = QtWidgets.QHBoxLayout()
        self.kind = QtWidgets.QComboBox()
        for k in KINDS:
            self.kind.addItem(k.value.split("  ")[0], k)
        self.amplitude = spin(-1e5, 1e5, 10.0, 3, " N")
        self.freq = spin(0.001, 1000.0, 1.0, 3, " Hz")
        self.tune = QtWidgets.QComboBox()
        self.tune.setToolTip("Drive at a natural frequency of the full model or of the reduced one")
        self.duration = spin(1e-4, 10.0, 0.05, 4, " s")
        self.button = QtWidgets.QPushButton()
        self.button.setMinimumWidth(130)
        self.reset_button = QtWidgets.QPushButton("Reset")
        self.reset_button.setToolTip("Both models back to rest, force off")
        self.speed = QtWidgets.QComboBox()
        for s in SPEEDS:
            self.speed.addItem(f"{s:g}×", s)
        self.speed.setCurrentIndex(SPEEDS.index(1.0))
        self.speed.setToolTip("Simulation speed (× real time)")
        self.window = spin(0.1, 60.0, 5.0, 1, " s")
        self.window.setToolTip("Plot window")
        self.freq_label, self.duration_label = QtWidgets.QLabel("at"), QtWidgets.QLabel("for")
        for w in (QtWidgets.QLabel("Force at tip:"), self.kind, self.amplitude, self.freq_label, self.freq,
                  self.tune, self.duration_label, self.duration, self.button, self.reset_button):
            row.addWidget(w)
        row.addStretch(1)
        for w in (QtWidgets.QLabel("Speed:"), self.speed, QtWidgets.QLabel("Window:"), self.window):
            row.addWidget(w)

        self.kind.currentIndexChanged.connect(self._apply)
        for box in (self.amplitude, self.freq, self.duration):
            box.valueChanged.connect(self._apply)
        self.tune.activated.connect(self._on_tune)
        self.button.clicked.connect(self._on_button)
        self.reset_button.clicked.connect(self.reset)

        self._clock = QtCore.QElapsedTimer()
        self._last = 0.0
        self._target = 0.0  # simulated time to catch up to
        # Single-shot timer re-armed after each frame: if a frame runs long, the
        # event loop still gets to handle input before the next one starts.
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(FRAME_MS)
        self._timer.timeout.connect(self._frame)
        self._build()
        self._apply()
        self.apply_theme()

    # ------------------------------------------------------------- display (subclasses)
    def _build(self) -> None:
        """Lay out the tab: the header and the controls row, then the display."""
        raise NotImplementedError

    def apply_theme(self) -> None:
        raise NotImplementedError

    def _make_sim(self, model: CMSModel) -> CMSResponse:
        return CMSResponse(model, self.force)

    def _on_model(self, model: CMSModel) -> None:
        """Set up the display for a new model (the simulation is already at rest)."""

    def _advance(self, duration: float) -> None:
        """Advance the simulation by about `duration` s and keep what is plotted."""
        raise NotImplementedError

    def _clear_display(self) -> None:
        """Drop the plotted history."""

    def _draw(self) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------- model
    def set_model(self, model: CMSModel | None, full: ModalResult | None, reason: str = "") -> None:
        """A new reduced model; restarts from rest unless it is the same model as before."""
        key = None if model is None else _model_key(model)
        if key is not None and key == self._key:
            self.model = model  # e.g. a theme change: keep the motion going
            return
        self._key, self.model = key, model
        if model is None:
            self.sim = None
            self.header.setText(f"<b style='color:{colors.poor}'>{reason}</b>")
            self._timer.stop()
            self._clear_history()
            self._draw()
            return
        on = self.force.on
        self.sim = self._make_sim(model)
        self.sim.reset()
        if on:
            self.force.switch_on()
        self._clear_history()
        self._on_model(model)
        self.tune.clear()
        self.tune.addItem("Tune to…")
        for r, m in enumerate(full.modes if full is not None else []):
            if m.fn_hz > 0:
                self.tune.addItem(f"Mode {r + 1} true ({m.fn_hz:.4g} Hz)", m.fn_hz)
        for r, f in enumerate(model.fn_hz):
            if f > 0:
                self.tune.addItem(f"Mode {r + 1} {model.short_name} ({f:.4g} Hz)", float(f))
        self._draw()
        self._start_timer()

    def reset(self) -> None:
        if self.sim is not None:
            self.sim.reset()
        self._clear_history()
        self._refresh_button()
        self._draw()

    def _clear_history(self) -> None:
        self._clear_display()
        self._target = self.sim.t if self.sim is not None else 0.0

    # ------------------------------------------------------------- force
    def _apply(self) -> None:
        s = self.force.settings
        kind = self.kind.currentData()
        if kind is not s.kind:
            self.force.switch_off()
        s.kind = kind
        s.amplitude = self.amplitude.value()
        s.freq_hz = self.freq.value()
        s.pulse_duration = self.duration.value()
        harmonic, pulse = kind is ForceKind.HARMONIC, kind is ForceKind.PULSE
        for w in (self.freq_label, self.freq, self.tune):
            w.setVisible(harmonic)
        self.duration_label.setVisible(pulse)
        self.duration.setVisible(pulse)
        self._refresh_button()

    def _on_tune(self, index: int) -> None:
        f = self.tune.itemData(index)
        if f:
            self.freq.setValue(f)
        self.tune.setCurrentIndex(0)

    def _on_button(self) -> None:
        if self.force.settings.kind is ForceKind.PULSE:
            self.force.fire_pulse()
        elif self.force.on:
            self.force.switch_off()
        else:
            self.force.switch_on()
        self._refresh_button()

    def _refresh_button(self) -> None:
        if self.force.settings.kind is ForceKind.PULSE:
            self.button.setText("Fire pulse")
        else:
            self.button.setText("Stop force" if self.force.on else "Apply force")

    # ------------------------------------------------------------- loop
    def showEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().showEvent(event)
        self._start_timer()

    def hideEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().hideEvent(event)
        self._timer.stop()

    def _start_timer(self) -> None:
        if self.sim is not None and self.isVisible() and not self._timer.isActive():
            self._clock.restart()
            self._last = 0.0
            self._target = self.sim.t
            self._timer.start()

    def _frame(self) -> None:
        now = self._clock.elapsed() / 1000.0
        dt = min(now - self._last, 0.1)
        self._last = now
        self.step(dt * float(self.speed.currentData()))
        if self.sim is not None and self.isVisible():
            self._timer.start()

    def step(self, duration: float) -> None:
        """Advance both models by `duration` s of simulated time and redraw."""
        if self.sim is None:
            return
        self._target = min(self._target + duration, self.sim.t + 0.25 + duration)
        self._advance(self._target - self.sim.t)
        if self.force.settings.kind is ForceKind.PULSE or not self.force.on:
            self._refresh_button()
        self._draw()


class CMSTimeView(CMSRunView):
    """Controls, two animated chains (full above, reduced below), and time histories of both."""

    def _build(self) -> None:
        self._dofs = [0]  # plotted masses: the tip, then each interface
        self.history = _Buffer(2)  # the plotted masses: full model, then reduced
        self.full_view = ChainView(draggable=False)
        self.red_view = ChainView(draggable=False)
        for view in (self.full_view, self.red_view):
            view.setMinimumHeight(110)
            view.setToolTip(TIP)

        self.plots = pg.GraphicsLayoutWidget()
        self.x_plot = self.plots.addPlot(row=0, col=0)
        self.x_plot.setLabel("left", "Displacement", units="m")
        add_legend(self.x_plot, offset=(-5, 5), colCount=2)
        self.e_plot = self.plots.addPlot(row=1, col=0)
        self.e_plot.setLabel("left", "Reduced − full", units="m")
        self.e_plot.setLabel("bottom", "Time", units="s")
        self.e_plot.setXLink(self.x_plot)
        self.plots.ci.layout.setRowStretchFactor(0, 3)
        self.plots.ci.layout.setRowStretchFactor(1, 2)
        # Performance: these plots scroll and redraw every frame. No grid and 1 px pens
        # (Qt's fast path): wide pens made each frame ~90 ms once the window filled,
        # several times that on a Retina display. See TimeHistoryPlot.
        for p in (self.x_plot, self.e_plot):
            p.setClipToView(True)
            p.setDownsampling(auto=True, mode="peak")
            p.setMouseEnabled(x=False, y=False)
            p.hideButtons()
        self.x_curves: list[pg.PlotDataItem] = []  # per plotted mass: full, reduced
        self.e_curves: list[pg.PlotDataItem] = []  # per plotted mass: reduced - full
        self.error_note = QtWidgets.QLabel()
        mute(self.error_note)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.header)
        layout.addLayout(self.controls)
        layout.addWidget(self.full_view, 2)
        layout.addWidget(self.red_view, 2)
        layout.addWidget(self.plots, 5)
        layout.addWidget(self.error_note)

    # ------------------------------------------------------------- setup
    def apply_theme(self) -> None:
        """Chains and curve pens in the current theme."""
        for view in (self.full_view, self.red_view):
            view.apply_theme()
        self.x_plot.legend.setBrush(colors.legend_brush())  # opaque, over the curves
        self._set_pens()

    def _make_curves(self) -> None:
        """Two displacement curves and one difference curve per plotted mass."""
        for c in self.x_curves:
            self.x_plot.removeItem(c)
        for c in self.e_curves:
            self.e_plot.removeItem(c)
        self.x_curves = [self.x_plot.plot() for _ in range(2 * len(self._dofs))]
        self.e_curves = [self.e_plot.plot() for _ in self._dofs]

    def _set_pens(self) -> None:
        short = self.model.short_name if self.model is not None else "reduced"
        self.x_plot.legend.clear()
        for k, d in enumerate(self._dofs):
            where = "tip" if k == 0 else "interface"
            styles = [
                (colors.mass[d], None, f"x{d + 1} ({where}) full"),
                (colors.strong if k == 0 else colors.mass[d], DASH, f"x{d + 1} {short}"),
            ]
            for curve, (color, style, name) in zip(self.x_curves[2 * k : 2 * k + 2], styles):
                # Full model solid, reduced dashed: the full one shows through the gaps.
                curve.setPen(pg.mkPen(color, width=1, style=style or QtCore.Qt.PenStyle.SolidLine))
                self.x_plot.legend.addItem(curve, name)
        for curve, d in zip(self.e_curves, self._dofs):
            curve.setPen(pg.mkPen(colors.mass[d], width=1))

    def _on_model(self, model: CMSModel) -> None:
        n = model.system.n
        self._dofs = [n - 1] + [int(b) for b in model.boundary[:-1]]
        self.history = _Buffer(2 * len(self._dofs))
        self._make_curves()
        for view in (self.full_view, self.red_view):
            view.set_masses(model.system.masses)
            view.gain = 10.0
        self.full_view.setTitle(f"Full model: {n} DOFs", size="10pt")
        self.red_view.setTitle(f"{model.name} reduced model: {model.n_red} DOF{'s' if model.n_red != 1 else ''} "
                               f"({model.n_modal} modal + {model.boundary.size} boundary)", size="10pt")
        self.header.setText(
            f"<b>The {model.name} model against the full chain, in time.</b> The same force acts at the "
            f"tip (m{n}) of both; each is stepped exactly, so any difference is the reduction error. "
            "Change the modes kept on the left and watch the two lock together."
        )
        self._set_pens()

    def _clear_display(self) -> None:
        self.history.clear()

    def _advance(self, duration: float) -> None:
        ts, xf, xr, _ = self.sim.advance(duration)
        cols = self._dofs
        self.history.extend(ts, np.hstack([xf[:, cols], xr[:, cols]]))

    def _draw(self) -> None:
        sim = self.sim
        if sim is None:
            for c in self.x_curves + self.e_curves:
                c.setData([], [])
            self.error_note.setText("")
            return
        t, y = self.history.window(self.window.value())
        k = len(self._dofs)
        xf, xr = y[:, :k], y[:, k:]
        x_now, r_now = sim.x_full, sim.x_reduced
        peak = max(float(np.abs(x_now).max()), float(np.abs(r_now).max()),
                   float(np.abs(xf).max()) if xf.size else 0.0, float(np.abs(xr).max()) if xr.size else 0.0)
        f = self.force.value()
        tip = sim.model.system.n - 1
        amp = abs(self.force.settings.amplitude)
        self.full_view.update_state(x_now, peak, f, tip, amp)
        self.red_view.update_state(r_now, peak, f, tip, amp)
        err = xr - xf
        for i, curve in enumerate(self.x_curves):
            curve.setData(t, (xf, xr)[i % 2][:, i // 2])
        for i, curve in enumerate(self.e_curves):
            curve.setData(t, err[:, i])
        if t.size:
            w = self.window.value()
            x0 = max(t[-1] - w, 0.0) if t[-1] > w else 0.0
            self.x_plot.setXRange(x0, x0 + w, padding=0)
            span = max(float(np.abs(np.concatenate([xf.ravel(), xr.ravel()])).max()), MIN_SPAN)
            self.x_plot.setYRange(-span, span, padding=0.05)
            espan = max(float(np.abs(err).max()), MIN_SPAN * 1e-3)
            self.e_plot.setYRange(-espan, espan, padding=0.05)
            rms = float(np.sqrt(np.mean(xf[:, 0] ** 2)))
            rel = float(np.sqrt(np.mean(err[:, 0] ** 2))) / rms if rms > MIN_SPAN else 0.0
            self.error_note.setText(
                f"Over the last {min(w, t[-1] - t[0]):.3g} s: tip error {100 * rel:.3g}% RMS of the full "
                f"model's tip motion; largest |error| {_fmt(float(np.abs(err[:, 0]).max()))} at the tip"
                + (f", {_fmt(float(np.abs(err[:, 1:]).max()))} at the interface{'s' if k > 2 else ''}" if k > 1 else "")
                + f".   t = {sim.t:.2f} s"
            )


def _model_key(model: CMSModel) -> tuple:
    s = model.system
    return (tuple(s.masses), tuple(s.stiffness), tuple(s.damping), model.method,
            tuple(int(b) for b in model.boundary), tuple(sub.n_kept for sub in model.substructures))


def _fmt(meters: float) -> str:
    if meters >= 1e-3:
        return f"{meters * 1e3:.3g} mm"
    if meters >= 1e-6:
        return f"{meters * 1e6:.3g} µm"
    return f"{meters * 1e9:.3g} nm"


class _Buffer:
    """Recent samples (t, columns), dropping the oldest half when full: amortized O(1) per sample."""

    def __init__(self, columns: int, capacity: int = 200_000) -> None:
        self.t = np.empty(capacity)
        self.y = np.empty((capacity, columns))
        self.size = 0

    def clear(self) -> None:
        self.size = 0

    def extend(self, ts: np.ndarray, ys: np.ndarray) -> None:
        cap, k = self.t.size, min(ts.size, self.t.size)
        if self.size + k > cap:
            keep = min(self.size, cap // 2, cap - k)
            self.t[:keep] = self.t[self.size - keep : self.size]
            self.y[:keep] = self.y[self.size - keep : self.size]
            self.size = keep
        self.t[self.size : self.size + k] = ts[-k:] if k else ts[:0]
        self.y[self.size : self.size + k] = ys[-k:] if k else ys[:0]
        self.size += k

    def window(self, seconds: float) -> tuple[np.ndarray, np.ndarray]:
        t = self.t[: self.size]
        i0 = int(np.searchsorted(t, t[-1] - seconds)) if self.size else 0
        return t[i0:], self.y[i0 : self.size]
