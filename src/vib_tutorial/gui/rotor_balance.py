"""Single- and two-plane balancing exercise for the Jeffcott rotor page.

The steps are those of a field balance: read the slow roll, read the 1X vector
as found at the balancing speed, add a trial weight and read again (in two
planes, a trial in each plane in turn), then put the correction weights on (the
trials taken off) and check. Each reading runs the simulation: the page ramps
to the speed, waits for the transient to die away and averages the 1X vector
over some revolutions. The vectors are kept at every station, so the probes can
be changed afterwards and the arithmetic redone.
"""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.rotor import RPM, PlaneWeight
from ..core.rotor_balance import PROBES, corrections, influence_matrix, weight, weight_parts
from ..core.rotor_sweep import STATIONS, lag_degrees, probe, steady_vectors
from .panels import spin
from .style import colors
from .theming import mute

DASH = QtCore.Qt.PenStyle.DashLine
CURVE_POINTS = 600
BALANCE_RPM = 1200.0
SLOW_RPM = 100.0
MYSTERY_RANGE = (10e-6, 60e-6)  # kg·m, the hidden unbalance's size
MYSTERY_SKEW = (0.05, 0.3)  # degrees, the hidden skew's size (two planes)
PLANE_SPACING = 40.0  # mm between the planes on the disc
SHAFT_PLANES = (200.0, 300.0)  # mm from bearing A, the planes on the shaft
ILL_CONDITIONED = 20.0  # cond(α) above which the planes are hard to tell apart
# The ways to balance: one weight on the disc, two in planes on the disc's faces, or two on the shaft.
MODES = ("One plane: the disc", "Two planes: the disc's faces", "Two planes on the shaft")
# Each reading: its symbol and the colour it is drawn in (an attribute of colors, or an index into colors.mode).
READINGS = (("slow", "S", "grey"), ("found", "V₀", 1), ("trial", "V₁", 3), ("trial2", "V₂", 4),
            ("verify", "V₂", "strong"))
TRIALS = ("trial", "trial2")  # the readings with a trial weight in plane 1, plane 2
SECOND = "′"  # marks the second probe's readings

STEPS_TIP = (
    "<p>1. Turn slowly and read the runout (slow roll): a bowed shaft reads its bow here, and no unbalance "
    "shows yet.<br>2. Run at the balancing speed and read the 1X vector as found.<br>3. Add a known trial "
    "weight and read again: the change, per unit of weight, is the influence coefficient. With two planes, "
    "do this for each plane in turn (the first trial taken off): each gives a column of the influence "
    "matrix.<br>Last, take the trials off, put on the weights that cancel the readings (less their slow "
    "roll, if ticked), and read again to check.</p><p>Each reading ramps to the speed, waits for the "
    "transient to die away, and averages ten revolutions, at no less than 2× real time.</p>"
)
MYSTERY_TIP = (
    "<p>Give the disc an unbalance of unknown size at an unknown angle, and hide the heavy spot: find it by "
    "balancing. With two planes the disc is skewed too, by a hidden angle: a couple unbalance. "
    "<i>Reveal</i> shows them in the panel again.</p>"
)
MODE_TIP = (
    "<p><b>One plane</b>: one weight on the disc, read at one probe.<br><b>Two planes on the disc</b>: "
    "weights on its two faces, a set distance apart, read at two probes. Together they make a force and "
    "a couple, so they can cancel the disc's unbalance and its skew at once, at every speed.<br><b>Two "
    "planes on the shaft</b>: weights clamped to the shaft wherever you choose. They cancel both readings "
    "at the balancing speed, but on a flexible shaft only there, unless the planes are close to the "
    "unbalance.</p>"
)


def _color(key: str | int) -> str:
    return colors.mode[key] if isinstance(key, int) else getattr(colors, key)


def _vec(v: complex) -> str:
    """A 1X reading as amplitude ∠ phase lag."""
    return f"{abs(v) * 1e6:.3g} µm ∠ {float(lag_degrees(np.array(v))) % 360.0:.0f}°"


class BalancingPanel(QtWidgets.QWidget):
    """The balancing tab: settings, the steps, the vectors, and the response before and after."""

    def __init__(self, page, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.page = page
        self.vectors: dict[str, np.ndarray | None] = {key: None for key, _, _ in READINGS}
        self.found_weights: tuple[complex, ...] = ()
        self.found_planes: tuple[PlaneWeight, ...] = ()  # plane weights on as found
        self.found_system = None  # the rotor as found, for the curve
        self.trial: complex | None = None  # the trial weight used (kg·m)
        self.planes_used: tuple[PlaneWeight, ...] = ()  # the planes the trials went in (two planes)
        self.correction: np.ndarray | None = None  # the correction weights put on (kg·m), one per plane
        self._rng = np.random.default_rng()

        form = QtWidgets.QFormLayout()
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(MODES)
        self.mode.setToolTip(MODE_TIP)
        self.spacing = spin(1.0, 1000.0, PLANE_SPACING, 0, " mm apart")
        self.spacing.setToolTip("<p>The distance between the disc's two balancing planes, its faces: "
                                "the further apart, the smaller the weights a couple needs.</p>")
        self.plane_z = [spin(0.0, 1e4, z, 0, " mm") for z in SHAFT_PLANES]
        for box in self.plane_z:
            box.setToolTip("<p>Where the plane is along the shaft, from bearing A.</p>")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.mode, 2)
        row.addWidget(self.spacing, 1)
        self.plane_labels = [QtWidgets.QLabel(t) for t in ("at", "and")]
        for label, box in zip(self.plane_labels, self.plane_z):
            row.addWidget(label)
            row.addWidget(box, 1)
        form.addRow("Planes:", row)
        self.mode.currentIndexChanged.connect(self._on_mode)
        for box in (self.spacing, *self.plane_z):
            box.valueChanged.connect(self._on_planes)

        self.station = QtWidgets.QComboBox()
        self.station.addItems(STATIONS)
        self.direction = QtWidgets.QComboBox()
        self.direction.addItems([f"{d} probe" for d in PROBES])
        self.station2 = QtWidgets.QComboBox()
        self.station2.addItems(STATIONS)
        self.station2.setCurrentIndex(len(STATIONS) - 1)  # bearing B
        self.direction2 = QtWidgets.QComboBox()
        self.direction2.addItems([f"{d} probe" for d in PROBES])
        for c in (self.station, self.direction, self.station2, self.direction2):
            c.currentIndexChanged.connect(self.refresh)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.station, 1)
        row.addWidget(self.direction, 1)
        self.second_label = QtWidgets.QLabel("and")
        self.second_label.setToolTip("The second probe's readings are drawn dashed, their names primed")
        row.addWidget(self.second_label)
        row.addWidget(self.station2, 1)
        row.addWidget(self.direction2, 1)
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
        self._buttons = {key: QtWidgets.QPushButton() for key in ("slow", "found", "trial", "trial2", "verify")}
        actions = {"slow": self.measure_slow, "found": self.measure_found, "trial": lambda: self.measure_trial(0),
                   "trial2": lambda: self.measure_trial(1), "verify": self.measure_verify}
        for i, (key, b) in enumerate(self._buttons.items()):
            b.setToolTip(STEPS_TIP)
            b.clicked.connect(actions[key])
            steps.addWidget(b, 0, i)
        self.mystery = QtWidgets.QPushButton("Mystery unbalance")
        self.mystery.setToolTip(MYSTERY_TIP)
        self.mystery.clicked.connect(self.new_mystery)
        self.reveal = QtWidgets.QPushButton("Reveal")
        self.reveal.setToolTip("Show the hidden unbalance in the panel again, and the heavy spot")
        self.reveal.clicked.connect(page.reveal_unbalance)
        self.remove = QtWidgets.QPushButton("Remove weights")
        self.remove.setToolTip("Take every balance weight off the rotor and start again")
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
            "right the phase lag. The dashed lines from V₀ are the trial weights' effects. With two probes "
            "the second's readings are dashed, their names primed.</p>"
        )
        self.axes = [pg.InfiniteLine(pos=0, angle=a) for a in (0, 90)]
        for line in self.axes:
            vp.addItem(line)
        self.arrows, self.tips, self.labels = {}, {}, {}
        for key, sym, _ in READINGS:
            for k in range(2):
                self.arrows[key, k] = vp.plot()
                self.tips[key, k] = pg.ScatterPlotItem(size=8, symbol="o" if k == 0 else "d")
                # The slow roll's label goes under its tip: a bowed rotor balanced reads it again at the check.
                self.labels[key, k] = pg.TextItem(sym + SECOND * k,
                                                  anchor=(0.5, -0.1) if key == "slow" else (0.5, 1.1))
                vp.addItem(self.tips[key, k])
                vp.addItem(self.labels[key, k])
        self.effects = {(key, k): vp.plot() for key in TRIALS for k in range(2)}

        self.curve_plot = pg.PlotWidget()
        cp = self.curve_plot.getPlotItem()
        cp.setMenuEnabled(False)
        cp.setLabel("left", "1X amplitude", units="m")
        cp.setLabel("bottom", "Speed (rpm)")
        cp.getAxis("bottom").enableAutoSIPrefix(False)
        cp.showGrid(x=True, y=True, alpha=0.15)
        cp.addLegend(offset=(-5, 5))
        self.curve_found = [cp.plot(name="As found"), cp.plot()]
        self.curve_now = [cp.plot(name="Now"), cp.plot()]
        self.speed_lines = [pg.InfiniteLine(angle=90) for _ in range(2)]
        for line in self.speed_lines:
            cp.addItem(line)
        self.curve_plot.setToolTip(
            "<p>The steady-state 1X amplitude at the probe against speed: as found (at step 2), and with the "
            "weights on now; the second probe's dashed. The lines mark the balancing and slow-roll speeds.</p>"
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
        self._show_mode()
        self.apply_theme()
        self._place_speed_lines()

    # ------------------------------------------------------------- theme
    def apply_theme(self) -> None:
        for line in self.axes:
            line.setPen(pg.mkPen(colors.faint, width=1))
        for key, _, c in READINGS:
            color = _color(c)
            for k in range(2):
                self.arrows[key, k].setPen(pg.mkPen(color, width=2.5 if k == 0 else 2, style=DASH if k else None))
                self.tips[key, k].setBrush(pg.mkBrush(color))
                self.tips[key, k].setPen(pg.mkPen(colors.background, width=1))
                self.labels[key, k].setColor(color)
        for (key, _), line in self.effects.items():
            line.setPen(pg.mkPen(_color(dict((r[0], r[2]) for r in READINGS)[key]), width=1.5, style=DASH))
        for k in range(2):
            style = DASH if k else QtCore.Qt.PenStyle.SolidLine
            self.curve_found[k].setPen(pg.mkPen(colors.grey, width=2, style=style))
            self.curve_now[k].setPen(pg.mkPen(colors.strong, width=2, style=style))
        self.curve_plot.getPlotItem().legend.setBrush(colors.legend_brush())
        self.speed_lines[0].setPen(pg.mkPen(colors.force, width=1.5))
        self.speed_lines[1].setPen(pg.mkPen(colors.grey, width=1, style=DASH))
        self.refresh()

    # ------------------------------------------------------------- planes and probes
    @property
    def two_planes(self) -> bool:
        return self.mode.currentIndex() > 0

    @property
    def step_buttons(self) -> list[QtWidgets.QPushButton]:
        """The steps' buttons, in order, for the planes chosen."""
        return [b for key, b in self._buttons.items() if key != "trial2" or self.two_planes]

    def planes(self) -> tuple[PlaneWeight, ...]:
        """The two balancing planes chosen (with no weight yet), for the rotor on the page."""
        s = self.page.sim.system
        if self.mode.currentIndex() == 1:
            h = self.spacing.value() / 2e3
            return PlaneWeight(s.a - h, 0j, True), PlaneWeight(s.a + h, 0j, True)
        return tuple(PlaneWeight(box.value() / 1e3, 0j) for box in self.plane_z)

    def probes(self) -> list[tuple[int, str]]:
        """(station, direction) of each probe in use: one for one plane, two for two."""
        pairs = [(self.station, self.direction), (self.station2, self.direction2)][: 2 if self.two_planes else 1]
        return [(s.currentIndex(), PROBES[d.currentIndex()]) for s, d in pairs]

    def _probe_name(self, k: int) -> str:
        station, direction = self.probes()[k]
        return f"{STATIONS[station]}, {direction}"

    def _show_mode(self) -> None:
        mode = self.mode.currentIndex()
        two = mode > 0
        self.spacing.setVisible(mode == 1)
        for w in (*self.plane_labels, *self.plane_z):
            w.setVisible(mode == 2)
        for w in (self.second_label, self.station2, self.direction2):
            w.setVisible(two)
        self._buttons["trial2"].setVisible(two)
        names = (["1. Slow roll", "2. As found", "3. Trial, plane 1", "4. Trial, plane 2", "5. Correct and check"]
                 if two else ["1. Slow roll", "2. As found", "3. Trial run", "", "4. Correct and check"])
        for b, name in zip(self._buttons.values(), names):
            b.setText(name)
        for k in range(2):
            self.labels["verify", k].setText(("V₃" if two else "V₂") + SECOND * k)
        lo, hi = self.page.sim.system.extent
        for box in self.plane_z:
            box.blockSignals(True)
            box.setRange(lo * 1e3, hi * 1e3)
            box.blockSignals(False)

    def _forget_trials(self) -> None:
        """The trial runs belong to the planes they were made in: forget them when the planes change."""
        self.vectors.update(trial=None, trial2=None, verify=None)
        self.trial, self.planes_used, self.correction = None, (), None

    def _on_mode(self) -> None:
        self._forget_trials()
        self._show_mode()
        self.refresh()

    def _on_planes(self) -> None:
        if self.two_planes:
            self._forget_trials()
        self.refresh()

    def rotor_changed(self) -> None:
        """The rotor's length or disc moved: keep the shaft planes on the shaft."""
        self._show_mode()

    # ------------------------------------------------------------- readings
    def readings(self, key: str) -> np.ndarray | None:
        """A stored reading at each probe in use (complex), or None if not taken."""
        v = self.vectors[key]
        if v is None:
            return None
        return np.array([complex(probe(v[station], direction)) for station, direction in self.probes()])

    def reading(self, key: str, k: int = 0) -> complex | None:
        """A stored reading at probe k, or None if not taken."""
        v = self.readings(key)
        return None if v is None else complex(v[k])

    def _runout(self) -> np.ndarray | complex:
        slow = self.readings("slow")
        return slow if self.subtract.isChecked() and slow is not None else 0j

    def influence(self) -> np.ndarray | None:
        """α (probes x planes) from the trial runs, or None until they are all in."""
        keys = TRIALS if self.two_planes else TRIALS[:1]
        v0, runs = self.readings("found"), [self.readings(k) for k in keys]
        if v0 is None or any(r is None for r in runs) or self.trial is None:
            return None
        if any(np.array_equal(r, v0) for r in runs):
            return None
        return influence_matrix(v0, np.array(runs), np.full(len(keys), self.trial))

    def correction_weights(self) -> np.ndarray | None:
        """W = -α⁻¹ (V₀ - S), one weight (kg·m) per plane, or None until the trial runs are in."""
        alpha = self.influence()
        if alpha is None:
            return None
        try:
            return corrections(self.readings("found"), alpha, self._runout())
        except ValueError:
            return None

    # ------------------------------------------------------------- steps
    def _measure(self, rpm: float, key: str) -> None:
        def measured(vectors: np.ndarray) -> None:
            self.vectors[key] = vectors
            self.refresh()

        self.page.start_measurement(rpm / RPM, measured)
        self._enable(False)

    def _put_on(self, weights: np.ndarray) -> None:
        """Put these weights on, one per plane, beside the weights on as found."""
        if self.two_planes:
            planes = tuple(p._replace(weight=complex(w)) for p, w in zip(self.planes_used, weights) if w)
            self.page.set_weights(self.found_weights, self.found_planes + planes)
        else:
            self.page.set_weights(self.found_weights + (complex(weights[0]),), self.found_planes)

    def measure_slow(self) -> None:
        self._measure(self.slow_speed.value(), "slow")

    def measure_found(self) -> None:
        """Read as found, with whatever weights are on now; earlier trial and check readings go."""
        system = self.page.sim.system
        self.found_weights, self.found_planes, self.found_system = system.weights, system.plane_weights, system
        self._forget_trials()
        self._measure(self.speed.value(), "found")

    def measure_trial(self, plane: int = 0) -> None:
        """Run with the trial weight in this plane alone (0 or 1)."""
        r = self.trial_radius.value() / 1e3
        if plane == 0:
            self.trial = weight(self.trial_mass.value() / 1e3, r, math.radians(self.trial_angle.value()))
            self.planes_used = self.planes() if self.two_planes else ()
            self.vectors.update(trial2=None, verify=None)
        self.vectors["verify"], self.correction = None, None
        self._put_on(np.eye(2)[plane] * self.trial)
        self._measure(self.speed.value(), TRIALS[plane])

    def measure_verify(self) -> None:
        w = self.correction_weights()
        if w is None:
            return
        self.correction = w
        self._put_on(w)
        self._measure(self.speed.value(), "verify")

    def remove_weights(self) -> None:
        self.page.set_weights((), ())
        self.clear()

    def new_mystery(self) -> None:
        u = float(self._rng.uniform(*MYSTERY_RANGE))
        angle = float(self._rng.uniform(0.0, 2.0 * math.pi))
        skew = None
        if self.two_planes:
            skew = (math.radians(float(self._rng.uniform(*MYSTERY_SKEW))), float(self._rng.uniform(0.0, 2 * math.pi)))
        self.page.set_weights((), ())
        self.page.hide_unbalance(u, angle, skew)
        self.clear()

    def clear(self) -> None:
        """Forget the readings (the slow roll too) and the weights worked out."""
        self.vectors = {key: None for key, _, _ in READINGS}
        self.found_weights, self.found_planes, self.found_system = (), (), None
        self._forget_trials()
        self.refresh()

    def measurement_ended(self) -> None:
        self._enable(True)
        self.refresh()

    def _enable(self, on: bool) -> None:
        for b in (*self._buttons.values(), self.mystery, self.remove, self.mode):
            b.setEnabled(on)
        if on:
            self._buttons["trial"].setEnabled(self.vectors["found"] is not None)
            self._buttons["trial2"].setEnabled(self.vectors["trial"] is not None)
            self._buttons["verify"].setEnabled(self.correction_weights() is not None)

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
        values = {key: self.readings(key) for key, _, _ in READINGS}
        shown = {key for key, _, _ in READINGS if key != "trial2" or self.two_planes}
        n = len(self.probes())
        for key, _, _ in READINGS:
            for k in range(2):
                v = values[key]
                if v is None or key not in shown or k >= n:
                    self.arrows[key, k].setData([], [])
                    self.tips[key, k].setData([], [])
                    self.labels[key, k].setVisible(False)
                else:
                    self.arrows[key, k].setData([0.0, v[k].real], [0.0, v[k].imag])
                    self.tips[key, k].setData([v[k].real], [v[k].imag])
                    self.labels[key, k].setPos(v[k].real, v[k].imag)
                    self.labels[key, k].setVisible(True)
        v0 = values["found"]
        for (key, k), line in self.effects.items():
            v1 = values[key]
            if v0 is not None and v1 is not None and key in shown and k < n:
                line.setData([v0[k].real, v1[k].real], [v0[k].imag, v1[k].imag])
            else:
                line.setData([], [])
        self.results.setText(self._results_html(values))
        self._draw_curves()

    def _results_html(self, values: dict[str, np.ndarray | None]) -> str:
        two = self.two_planes
        runout = self._runout()
        subtract = np.any(runout != 0)
        names = {"slow": f"Slow roll ({self.slow_speed.value():,.0f} rpm)", "found": "As found",
                 "trial": "Trial, plane 1" if two else "Trial run", "trial2": "Trial, plane 2", "verify": "Check"}
        rows = []
        for key, sym, c in READINGS:
            v = values[key]
            if v is None or (key == "trial2" and not two):
                continue
            if key == "verify" and two:
                sym = "V₃"
            cells = [_vec(x) for x in v]
            if subtract:
                cells += ["" if key == "slow" else _vec(x) for x in v - runout]
            rows.append(f"<tr><td><span style='color:{_color(c)}'><b>{sym}</b></span> {names[key]}</td>"
                        + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
        if not rows:
            return ("<small>Take the readings in order. The influence coefficients and the correction "
                    "weights appear here. Try <i>Mystery unbalance</i> to balance a rotor whose heavy spot "
                    "you cannot see.</small>")
        if two:
            heads = [f"{self._probe_name(k)}{SECOND * k}" for k in range(2)]
            heads += [f"less slow roll{SECOND * k}" for k in range(2)] if subtract else []
        else:
            heads = ["1X", "less slow roll" if subtract else ""]
        out = ("<table cellspacing='0' cellpadding='2'><tr><th align='left'>Reading</th>"
               + "".join(f"<th>{h}</th>" for h in heads) + "</tr>" + "".join(rows) + "</table>")
        alpha = self.influence()
        r = self.trial_radius.value() / 1e3
        if alpha is not None:
            mass, angle = weight_parts(self.trial, r)
            if two:
                out += self._matrix_html(alpha, mass, angle)
            else:
                a = complex(alpha[0, 0])
                out += (f"Trial {mass * 1e3:.3g} g at {math.degrees(angle):.0f}°: influence coefficient "
                        f"α = (V₁ − V₀)/T = {_per_weight(a)}<br>")
            w = self.correction_weights()
            less = "(V₀ − S)" if subtract else "V₀"
            if w is None:
                out += "The trial runs cannot tell the planes apart: try planes further apart."
            elif two:
                parts = [f"W<sub>{j + 1}</sub> = {_weight(x, r)}" for j, x in enumerate(w)]
                out += f"<b>Correction</b> W = −α<sup>−1</sup>{less}: <b>{'; '.join(parts)}</b> (trials off)"
            else:
                out += f"<b>Correction</b> W = −{less}/α: <b>{_weight(complex(w[0]), r)}</b> from the mark (trial off)"
        v2 = values["verify"]
        if v2 is not None and v0_ok(values["found"], runout):
            v0 = values["found"]
            share = [100 * abs(a - b) / abs(c - b) for a, b, c in zip(v2, np.broadcast_to(runout, v2.shape), v0)]
            kind = "compensated " if subtract else ""
            if two:
                where = "; ".join(f"{p:.1f}% at {self._probe_name(k)}" for k, p in enumerate(share))
                out += f"<br>Check: {where}, of the {kind}readings as found."
            else:
                out += f"<br>Check: {_vec(v2[0])}, {share[0]:.1f}% of the {kind}reading as found."
        return out

    def _matrix_html(self, alpha: np.ndarray, mass: float, angle: float) -> str:
        """The influence matrix, a column per plane, and how well it tells the planes apart."""
        cells = [[f"{abs(a):.3g} ∠ {float(lag_degrees(np.array(a))) % 360.0:.0f}°" for a in row] for row in alpha]
        cond = np.linalg.cond(alpha)
        note = (": the planes are hard to tell apart, so the weights come out large and small errors in the "
                "readings swing them" if cond > ILL_CONDITIONED else "")
        return (f"Trial {mass * 1e3:.3g} g at {math.degrees(angle):.0f}° in each plane in turn. Influence matrix "
                f"α (µm per g·mm ∠ lag), column j = (V<sub>j</sub> − V₀)/T: "
                f"[{cells[0][0]}, {cells[0][1]}; {cells[1][0]}, {cells[1][1]}], condition number {cond:.3g}{note}.<br>")

    def _draw_curves(self) -> None:
        top = max(4000.0, 2.0 * self.speed.value())
        rpm = np.linspace(top / CURVE_POINTS, top, CURVE_POINTS)
        probes = self.probes()

        def amplitude(system) -> np.ndarray:
            v = steady_vectors(system, rpm / RPM)
            return [np.abs(probe(v[:, station], direction)) for station, direction in probes]

        now = amplitude(self.page.sim.system)
        found = amplitude(self.found_system) if self.found_system is not None else []
        for k in range(2):
            self.curve_now[k].setData(rpm, now[k]) if k < len(now) else self.curve_now[k].setData([], [])
            self.curve_found[k].setData(rpm, found[k]) if k < len(found) else self.curve_found[k].setData([], [])


def v0_ok(v0: np.ndarray | None, runout) -> bool:
    """True if there is an as-found reading, compensated, to compare the check with."""
    return v0 is not None and bool(np.all(np.abs(v0 - runout) > 0))


def _per_weight(a: complex) -> str:
    """An influence coefficient, in µm per g·mm with its lag."""
    return f"{abs(a):.3g} µm per g·mm ∠ {float(lag_degrees(np.array(a))) % 360.0:.0f}°"


def _weight(w: complex, radius: float) -> str:
    """A weight as grams at the radius and an angle from the mark."""
    mass, angle = weight_parts(w, radius)
    return f"{mass * 1e3:.3g} g at {radius * 1e3:.0f} mm, {round(math.degrees(angle)) % 360}°"
