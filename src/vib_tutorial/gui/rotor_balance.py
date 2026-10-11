"""Single-plane balancing exercise for the Jeffcott rotor page.

The steps are those of a field balance: read the slow roll, read the 1X vector
as found at the balancing speed, add a trial weight and read again, then put
the correction weight on (the trial taken off) and check. Each reading runs the
simulation: the page ramps to the speed, waits for the transient to die away
and averages the 1X vector over some revolutions. The vectors are kept at
every station, so the probe can be changed afterwards and the arithmetic redone.
"""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.rotor import RPM
from ..core.rotor_balance import PROBES, correction, influence, weight, weight_parts
from ..core.rotor_sweep import STATIONS, lag_degrees, probe, steady_vectors
from .panels import spin
from .style import colors
from .theming import mute

DASH = QtCore.Qt.PenStyle.DashLine
CURVE_POINTS = 600
BALANCE_RPM = 1200.0
SLOW_RPM = 100.0
MYSTERY_RANGE = (10e-6, 60e-6)  # kg·m, the hidden unbalance's size
# Each reading: its symbol and the colour it is drawn in (an attribute of colors, or an index into colors.mode).
READINGS = (("slow", "S", "grey"), ("found", "V₀", 1), ("trial", "V₁", 3), ("verify", "V₂", "strong"))

STEPS_TIP = (
    "<p>1. Turn slowly and read the runout (slow roll): a bowed shaft reads its bow here, and no unbalance "
    "shows yet.<br>2. Run at the balancing speed and read the 1X vector as found.<br>3. Add a known trial "
    "weight and read again: the change, per unit of weight, is the influence coefficient.<br>4. Take the "
    "trial off, put on the weight that cancels the reading (less its slow roll, if ticked), and read again "
    "to check.</p><p>Each reading ramps to the speed, waits for the transient to die away, and averages "
    "ten revolutions, at no less than 2× real time.</p>"
)
MYSTERY_TIP = (
    "<p>Give the disc an unbalance of unknown size at an unknown angle, and hide the heavy spot: find it by "
    "balancing. <i>Reveal</i> shows it in the panel again.</p>"
)


def _color(key: str | int) -> str:
    return colors.mode[key] if isinstance(key, int) else getattr(colors, key)


def _vec(v: complex) -> str:
    """A 1X reading as amplitude ∠ phase lag."""
    return f"{abs(v) * 1e6:.3g} µm ∠ {float(lag_degrees(np.array(v))) % 360.0:.0f}°"


class BalancingPanel(QtWidgets.QWidget):
    """The balancing tab: settings, the four steps, the vectors, and the response before and after."""

    def __init__(self, page, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.page = page
        self.vectors: dict[str, np.ndarray | None] = {key: None for key, _, _ in READINGS}
        self.found_weights: tuple[complex, ...] = ()
        self.found_system = None  # the rotor as found, for the curve
        self.trial: complex | None = None  # the trial weight used (kg·m)
        self.correction: complex | None = None  # the correction weight worked out (kg·m)
        self._rng = np.random.default_rng()

        form = QtWidgets.QFormLayout()
        self.station = QtWidgets.QComboBox()
        self.station.addItems(STATIONS)
        self.direction = QtWidgets.QComboBox()
        self.direction.addItems([f"{d} probe" for d in PROBES])
        for c in (self.station, self.direction):
            c.currentIndexChanged.connect(self.refresh)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.station, 1)
        row.addWidget(self.direction, 1)
        form.addRow("Probe:", row)
        self.speed = spin(60.0, 10_000.0, BALANCE_RPM, 0, " rpm")
        self.speed.setToolTip("The speed the rotor is balanced at")
        self.slow_speed = spin(10.0, 10_000.0, SLOW_RPM, 0, " rpm")
        self.slow_speed.setToolTip(
            "<p>Slow roll: slow enough that the unbalance makes no force to speak of. Faster, the reading "
            "holds some unbalance response too, and subtracting it biases the correction.</p>"
        )
        self.subtract = QtWidgets.QCheckBox("Subtract slow roll")
        self.subtract.setChecked(True)
        self.subtract.setToolTip(
            "<p>Work out the correction from the readings less the slow-roll runout, which no weight "
            "removes. Untick to balance the raw reading to zero at the balancing speed instead.</p>"
        )
        self.subtract.toggled.connect(self.refresh)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.speed, 1)
        row.addWidget(QtWidgets.QLabel("slow roll"))
        row.addWidget(self.slow_speed, 1)
        row.addWidget(self.subtract)
        form.addRow("Balancing at:", row)
        self.trial_mass = spin(0.01, 100.0, 0.5, 2, " g")
        self.trial_radius = spin(1.0, 500.0, 20.0, 0, " mm")
        self.trial_angle = spin(0.0, 359.0, 0.0, 0, "°")
        self.trial_angle.setWrapping(True)
        self.trial_angle.setStepType(QtWidgets.QAbstractSpinBox.StepType.DefaultStepType)
        self.trial_angle.setSingleStep(15.0)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.trial_mass, 1)
        row.addWidget(QtWidgets.QLabel("at radius"))
        row.addWidget(self.trial_radius, 1)
        row.addWidget(QtWidgets.QLabel("angle"))
        row.addWidget(self.trial_angle, 1)
        form.addRow("Trial weight:", row)

        steps = QtWidgets.QGridLayout()
        self.step_buttons = [QtWidgets.QPushButton(t) for t in
                             ("1. Slow roll", "2. As found", "3. Trial run", "4. Correct and check")]
        for i, (b, action) in enumerate(zip(self.step_buttons, (self.measure_slow, self.measure_found,
                                                                  self.measure_trial, self.measure_verify))):
            b.setToolTip(STEPS_TIP)
            b.clicked.connect(action)
            steps.addWidget(b, 0, i)
        self.mystery = QtWidgets.QPushButton("Mystery unbalance")
        self.mystery.setToolTip(MYSTERY_TIP)
        self.mystery.clicked.connect(self.new_mystery)
        self.reveal = QtWidgets.QPushButton("Reveal")
        self.reveal.setToolTip("Show the hidden unbalance in the panel, and the heavy spot")
        self.reveal.clicked.connect(page.reveal_unbalance)
        self.remove = QtWidgets.QPushButton("Remove weights")
        self.remove.setToolTip("Take every balance weight off the disc and start again")
        self.remove.clicked.connect(self.remove_weights)
        self.show_heavy = QtWidgets.QCheckBox("Show heavy spot")
        self.show_heavy.setChecked(True)
        self.show_heavy.setToolTip("Draw the heavy spot on the disc and its line on the disc's orbit")
        self.show_heavy.toggled.connect(page.set_heavy_shown)
        for i, w in enumerate((self.mystery, self.reveal, self.remove, self.show_heavy)):
            steps.addWidget(w, 1, i)
        self.status = QtWidgets.QLabel()
        mute(self.status)

        self.vector_plot = pg.PlotWidget()
        vp = self.vector_plot.getPlotItem()
        vp.setAspectLocked(True)
        vp.setMenuEnabled(False)
        vp.setLabel("bottom", "In phase with the keyphasor", units="m")
        vp.setLabel("left", "Quadrature (down: lagging 90°)", units="m")
        vp.showGrid(x=True, y=True, alpha=0.15)
        self.vector_plot.setToolTip(
            "<p>The readings as vectors, as on the polar plot: length the amplitude, angle clockwise from the "
            "right the phase lag. The dashed line from V₀ to V₁ is the trial weight's effect.</p>"
        )
        self.axes = [pg.InfiniteLine(pos=0, angle=a) for a in (0, 90)]
        for line in self.axes:
            vp.addItem(line)
        self.arrows = {key: vp.plot() for key, _, _ in READINGS}
        self.tips = {key: pg.ScatterPlotItem(size=8) for key, _, _ in READINGS}
        # The slow roll's label goes under its tip: a bowed rotor balanced reads it again as V₂.
        self.labels = {key: pg.TextItem(sym, anchor=(0.5, -0.1) if key == "slow" else (0.5, 1.1))
                       for key, sym, _ in READINGS}
        for key, _, _ in READINGS:
            vp.addItem(self.tips[key])
            vp.addItem(self.labels[key])
        self.effect = vp.plot()

        self.curve_plot = pg.PlotWidget()
        cp = self.curve_plot.getPlotItem()
        cp.setMenuEnabled(False)
        cp.setLabel("left", "1X amplitude", units="m")
        cp.setLabel("bottom", "Speed (rpm)")
        cp.getAxis("bottom").enableAutoSIPrefix(False)
        cp.showGrid(x=True, y=True, alpha=0.15)
        cp.addLegend(offset=(-5, 5))
        self.curve_found = cp.plot(name="As found")
        self.curve_now = cp.plot(name="Now")
        self.speed_lines = [pg.InfiniteLine(angle=90) for _ in range(2)]
        for line in self.speed_lines:
            cp.addItem(line)
        self.curve_plot.setToolTip(
            "<p>The steady-state 1X amplitude at the probe against speed: as found (at step 2), and with the "
            "weights on now. The lines mark the balancing and slow-roll speeds.</p>"
        )
        self.results = QtWidgets.QLabel()
        self.results.setWordWrap(True)
        self.results.setTextFormat(QtCore.Qt.TextFormat.RichText)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(steps)
        layout.addWidget(self.status)
        layout.addWidget(self.vector_plot, 3)
        layout.addWidget(self.results)
        layout.addWidget(self.curve_plot, 2)
        for w in (self.speed, self.slow_speed):
            w.valueChanged.connect(self._place_speed_lines)
        self.apply_theme()
        self._place_speed_lines()

    # ------------------------------------------------------------- theme
    def apply_theme(self) -> None:
        for line in self.axes:
            line.setPen(pg.mkPen(colors.faint, width=1))
        for key, _, c in READINGS:
            color = _color(c)
            self.arrows[key].setPen(pg.mkPen(color, width=2.5))
            self.tips[key].setBrush(pg.mkBrush(color))
            self.tips[key].setPen(pg.mkPen(colors.background, width=1))
            self.labels[key].setColor(color)
        self.effect.setPen(pg.mkPen(_color(3), width=1.5, style=DASH))
        self.curve_found.setPen(pg.mkPen(colors.grey, width=2))
        self.curve_now.setPen(pg.mkPen(colors.strong, width=2))
        self.curve_plot.getPlotItem().legend.setBrush(colors.legend_brush())
        self.speed_lines[0].setPen(pg.mkPen(colors.force, width=1.5))
        self.speed_lines[1].setPen(pg.mkPen(colors.grey, width=1, style=DASH))
        self.refresh()

    # ------------------------------------------------------------- readings
    def reading(self, key: str) -> complex | None:
        """A stored reading at the chosen probe, or None if not taken."""
        v = self.vectors[key]
        if v is None:
            return None
        return complex(probe(v[self.station.currentIndex()], PROBES[self.direction.currentIndex()]))

    def _runout(self) -> complex:
        slow = self.reading("slow")
        return slow if self.subtract.isChecked() and slow is not None else 0j

    def influence(self) -> complex | None:
        v0, v1 = self.reading("found"), self.reading("trial")
        if v0 is None or v1 is None or self.trial is None or v1 == v0:
            return None
        return influence(v0, v1, self.trial)

    def correction_weight(self) -> complex | None:
        alpha, v0 = self.influence(), self.reading("found")
        return None if alpha is None else correction(v0, alpha, self._runout())

    # ------------------------------------------------------------- steps
    def _measure(self, rpm: float, key: str) -> None:
        def measured(vectors: np.ndarray) -> None:
            self.vectors[key] = vectors
            self.refresh()

        self.page.start_measurement(rpm / RPM, measured)
        self._enable(False)

    def measure_slow(self) -> None:
        self._measure(self.slow_speed.value(), "slow")

    def measure_found(self) -> None:
        """Read as found, with whatever weights are on now; earlier trial and check readings go."""
        self.found_weights = self.page.sim.system.weights
        self.found_system = self.page.sim.system
        self.trial = self.correction = None
        self.vectors["trial"] = self.vectors["verify"] = None
        self._measure(self.speed.value(), "found")

    def measure_trial(self) -> None:
        r = self.trial_radius.value() / 1e3
        self.trial = weight(self.trial_mass.value() / 1e3, r, math.radians(self.trial_angle.value()))
        self.vectors["verify"] = None
        self.page.set_weights(self.found_weights + (self.trial,))
        self._measure(self.speed.value(), "trial")

    def measure_verify(self) -> None:
        w = self.correction_weight()
        if w is None:
            return
        self.correction = w
        self.page.set_weights(self.found_weights + (w,))
        self._measure(self.speed.value(), "verify")

    def remove_weights(self) -> None:
        self.page.set_weights(())
        self.clear()

    def new_mystery(self) -> None:
        u = float(self._rng.uniform(*MYSTERY_RANGE))
        angle = float(self._rng.uniform(0.0, 2.0 * math.pi))
        self.page.set_weights(())
        self.page.hide_unbalance(u, angle)
        self.clear()

    def clear(self) -> None:
        """Forget the readings (the slow roll too) and the weights worked out."""
        self.vectors = {key: None for key, _, _ in READINGS}
        self.found_weights, self.found_system = (), None
        self.trial = self.correction = None
        self.refresh()

    def measurement_ended(self) -> None:
        self._enable(True)
        self.refresh()

    def _enable(self, on: bool) -> None:
        for b in (*self.step_buttons, self.mystery, self.remove):
            b.setEnabled(on)
        if on:
            self.step_buttons[2].setEnabled(self.vectors["found"] is not None)
            self.step_buttons[3].setEnabled(self.correction_weight() is not None)

    def update_status(self) -> None:
        text = self.page.measurement_status()
        if text != self.status.text():
            self.status.setText(text)

    # ------------------------------------------------------------- drawing
    def _place_speed_lines(self) -> None:
        self.speed_lines[0].setValue(self.speed.value())
        self.speed_lines[1].setValue(self.slow_speed.value())
        self.refresh()

    def refresh(self) -> None:
        """Redraw the vectors, the results and the curves from the readings kept."""
        if self.page.measurement is None:
            self._enable(True)
        values = {key: self.reading(key) for key, _, _ in READINGS}
        for key, _, _ in READINGS:
            v = values[key]
            if v is None:
                self.arrows[key].setData([], [])
                self.tips[key].setData([], [])
                self.labels[key].setVisible(False)
            else:
                self.arrows[key].setData([0.0, v.real], [0.0, v.imag])
                self.tips[key].setData([v.real], [v.imag])
                self.labels[key].setPos(v.real, v.imag)
                self.labels[key].setVisible(True)
        v0, v1 = values["found"], values["trial"]
        if v0 is not None and v1 is not None:
            self.effect.setData([v0.real, v1.real], [v0.imag, v1.imag])
        else:
            self.effect.setData([], [])
        self.results.setText(self._results_html(values))
        self._draw_curves()

    def _results_html(self, values: dict[str, complex | None]) -> str:
        rows = []
        runout = self._runout()
        names = {"slow": f"Slow roll ({self.slow_speed.value():,.0f} rpm)", "found": "As found",
                 "trial": "Trial run", "verify": "Check"}
        for key, sym, c in READINGS:
            v = values[key]
            if v is None:
                continue
            comp = "" if key == "slow" or not runout else _vec(v - runout)
            rows.append(f"<tr><td><span style='color:{_color(c)}'><b>{sym}</b></span> {names[key]}</td>"
                        f"<td>{_vec(v)}</td><td>{comp}</td></tr>")
        if not rows:
            return ("<small>Take the readings in order: 1 to 4. The influence coefficient and the correction "
                    "weight appear here. Try <i>Mystery unbalance</i> to balance a rotor whose heavy spot "
                    "you cannot see.</small>")
        head = ("<table cellspacing='0' cellpadding='2'><tr><th align='left'>Reading</th><th>1X</th>"
                + ("<th>less slow roll</th>" if runout else "<th></th>") + "</tr>")
        out = head + "".join(rows) + "</table>"
        alpha = self.influence()
        r = self.trial_radius.value() / 1e3
        if alpha is not None:
            mass, angle = weight_parts(self.trial, r)
            out += (f"Trial {mass * 1e3:.3g} g at {math.degrees(angle):.0f}°: influence coefficient "
                    f"α = (V₁ − V₀)/T = {abs(alpha):.3g} µm per g·mm ∠ "
                    f"{float(lag_degrees(np.array(alpha))) % 360.0:.0f}°<br>")
            w = self.correction_weight()
            mass, angle = weight_parts(w, r)
            less = "(V₀ − S)" if runout else "V₀"
            out += (f"<b>Correction</b> W = −{less}/α: <b>{mass * 1e3:.3g} g at {r * 1e3:.0f} mm, "
                    f"{math.degrees(angle):.0f}°</b> from the mark (trial off)")
        v0, v2 = values["found"], values["verify"]
        if v2 is not None and v0 is not None and abs(v0 - runout) > 0:
            out += (f"<br>Check: {_vec(v2)}, {100 * abs(v2 - runout) / abs(v0 - runout):.1f}% of the "
                    f"{'compensated ' if runout else ''}reading as found.")
        return out

    def _draw_curves(self) -> None:
        top = max(4000.0, 2.0 * self.speed.value())
        rpm = np.linspace(top / CURVE_POINTS, top, CURVE_POINTS)
        station, direction = self.station.currentIndex(), PROBES[self.direction.currentIndex()]

        def amplitude(system) -> np.ndarray:
            return np.abs(probe(steady_vectors(system, rpm / RPM)[:, station], direction))

        self.curve_now.setData(rpm, amplitude(self.page.sim.system))
        if self.found_system is not None:
            self.curve_found.setData(rpm, amplitude(self.found_system))
        else:
            self.curve_found.setData([], [])
