"""Fluid-film bearing tab of the Jeffcott rotor page: a plain journal bearing by short-bearing theory.

From the bearing's dimensions, its oil and its load (core/fluid_bearing.py): the
journal's locus in its clearance circle as the speed rises, the eight
linearized stiffness and damping coefficients against the speed or the
modified Sommerfeld number, and the readings at the rotor's speed now. With *Oil-film
journal bearings* ticked under the rotor's bearing supports, the rotor runs in
this bearing at both ends; each then carries its share of the rotor's weight
in place of the load set here, and the tab shows either.
"""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.fluid_bearing import GRAVITY, JournalBearing, attitude, rigid_threshold
from ..core.rotor import RPM
from ..core.rotor_stability import Onset
from .panels import spin
from .style import colors
from .theming import mute

DASH = QtCore.Qt.PenStyle.DashLine
SPEEDS = 300  # points on the coefficient curves
LOCUS_POINTS = 200
LOCUS_SIZE = 230  # px
AXIS_WIDTH = 56  # px, the coefficient plots' left axes
LABEL_GAP = 0.14  # of the clearance: speed labels on the locus are at least this far apart
SHORT_RANGE = (0.001, 10.0)  # the dimensionless chart's span of S(L_b/D_b)²
LOW_SPEED = 0.02  # of the top speed: the speed plot's range is scaled to the curves above this
SHORT_LIMIT = 0.5  # L_b/D_b past which short-bearing theory is stretched
THRESHOLD_TOP = 60_000.0  # rpm: the rigid rotor's threshold is looked for up to this (the page's widest range)
AGAINST = ("Speed (N/m, N·s/m)", "Modified Sommerfeld number (dimensionless)")
# The coefficients drawn: (row, column) in K or C, label, colour index into colors.mode.
STIFFNESS = (((0, 0), "k<sub>xx</sub>", 0), ((0, 1), "k<sub>xy</sub>", 1), ((1, 0), "k<sub>yx</sub>", 3),
             ((1, 1), "k<sub>yy</sub>", 5))
DAMPING = (((0, 0), "c<sub>xx</sub>", 0), ((0, 1), "c<sub>xy</sub> = c<sub>yx</sub>", 1),
           ((1, 1), "c<sub>yy</sub>", 5))

LOCUS_TIP = (
    "<p>Where the journal's centre sits in its clearance circle (radius c<sub>r</sub>), seen from the B end: x to "
    "the right, y up, spinning counterclockwise, the load straight down. At rest it lies on the bottom; "
    "as the speed rises the oil film lifts it and pushes it sideways, in the direction of spin, along "
    "this path toward the centre. The solid part is the page's speed range; the dot is the speed now.</p>"
    "<p>The angle from the load line to the journal is the <i>attitude angle</i>: nearly 90° when "
    "lightly loaded. The film holds the load up by pushing the journal sideways, and that sideways push "
    "is the cross-coupled stiffness.</p><p>Short-bearing theory gives the same path for every bearing: "
    "only how fast the journal travels along it depends on the dimensions, oil and load.</p>"
)
COEFF_TIP = (
    "<p>The linearized coefficients for small motions about the journal's equilibrium: the film pushes "
    "back with f = −K Δq − C Δq′. k<sub>xy</sub> is the force in x from a displacement in y.</p>"
    "<p>K is not symmetric: k<sub>xy</sub> and k<sub>yx</sub> have opposite signs, and that skew part "
    "pushes a forward whirl along its path, as the rotor's k<sub>xy</sub> does. It grows with the speed "
    "while the direct damping levels off, so past some speed the film feeds a whirl faster than it damps "
    "it: oil whirl.</p><p>Made dimensionless, K c<sub>r</sub> / W and C c<sub>r</sub> Ω / W, the "
    "coefficients of a short bearing depend only on S(L<sub>b</sub>/D<sub>b</sub>)², the Sommerfeld number S times the square "
    "of the length-to-diameter ratio (the modified Sommerfeld number), so against it the curves are the "
    "same for every short bearing. Against S alone they would move with L<sub>b</sub>/D<sub>b</sub>. The scales are log, and dashed curves are negative.</p>"
)
INPUT_TIPS = {
    "diameter": "Journal diameter D<sub>b</sub>",
    "length": "<p>Bearing length L<sub>b</sub>. Short-bearing theory drops the pressure flow round the "
              "bearing and holds for L<sub>b</sub>/D<sub>b</sub> up to about 0.5; longer bearings carry more than it says.</p>",
    "clearance": "<p>Radial clearance c<sub>r</sub>, the gap with the journal centred: typically 1/1000 to 2/1000 of "
                 "the radius.</p>",
    "viscosity": "<p>The oil's dynamic viscosity μ. ISO VG 32 turbine oil is about 30 mPa·s at 40 °C, "
                 "and about 10 mPa·s at 70 °C.</p>",
    "load": "<p>The steady load W on the bearing, straight down: on a horizontal rotor, the weight it "
            "carries.</p>",
}


class BearingPanel(QtWidgets.QWidget):
    """Inputs, the journal locus, the coefficient plots and the readings at the rotor's speed."""

    changed = QtCore.Signal()  # the user changed the bearing

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.top_rpm = 10_000.0
        self.loads: tuple[float, float] | None = None  # the rotor's bearings' loads (N), when it runs on this one
        self.onset: Onset | None = None  # where the rotor running on it goes unstable
        self._own_load = 0.0  # N: the load set here, kept while the box shows the rotor's
        self.rpm = 0.0  # the rotor's speed now
        self._shown_rpm = math.nan  # the speed the readings were written for
        self._threshold: float | None = None  # rad/s, where a rigid rotor on it whirls, if it does
        self._stale = False  # changed while hidden: drawn when the tab shows
        b = JournalBearing()
        self.diameter = spin(1.0, 1000.0, b.diameter * 1e3, 1, " mm")
        self.length = spin(0.5, 1000.0, b.length * 1e3, 1, " mm")
        self.clearance = spin(1.0, 2000.0, b.clearance * 1e6, 1, " µm")
        self.viscosity = spin(0.5, 2000.0, b.viscosity * 1e3, 1, " mPa·s")
        self.load = spin(0.0, 1e6, b.load, 2, " N")
        self.which = QtWidgets.QComboBox()
        self.which.addItems(["Bearing A", "Bearing B"])
        self.which.setToolTip("<p>The rotor's bearing to show: each carries its own share of the weight.</p>")
        self.which.currentIndexChanged.connect(lambda _: self._redraw())
        self.rotor_note = QtWidgets.QLabel()
        mute(self.rotor_note)
        self.against = QtWidgets.QComboBox()
        self.against.addItems(AGAINST)
        self.against.setToolTip(COEFF_TIP)
        form = QtWidgets.QGridLayout()
        boxes = (("Diameter D<sub>b</sub>:", self.diameter, "diameter"), ("Length L<sub>b</sub>:", self.length, "length"),
                 ("Clearance c<sub>r</sub>:", self.clearance, "clearance"), ("Viscosity μ:", self.viscosity, "viscosity"),
                 ("Load W:", self.load, "load"))
        for i, (text, box, key) in enumerate(boxes):
            label = QtWidgets.QLabel(text)
            for w in (label, box):
                w.setToolTip(INPUT_TIPS[key])
            form.addWidget(label, i // 2, 2 * (i % 2))
            form.addWidget(box, i // 2, 2 * (i % 2) + 1)
            box.valueChanged.connect(self._on_params)
        form.addWidget(QtWidgets.QLabel("Against:"), 2, 2)
        form.addWidget(self.against, 2, 3)
        form.addWidget(self.which, 3, 1)
        form.addWidget(self.rotor_note, 3, 2, 1, 2)
        self.which.setVisible(False)
        self.against.currentIndexChanged.connect(lambda _: self._draw_coefficients())
        form.setColumnStretch(1, 1)
        form.setColumnStretch(3, 1)

        self.locus = pg.PlotWidget()
        self.locus.setToolTip(LOCUS_TIP)
        lp = self.locus.getPlotItem()
        lp.setAspectLocked(True)
        lp.setMenuEnabled(False)
        lp.setMouseEnabled(x=False, y=False)
        lp.hideButtons()
        lp.hideAxis("left")
        lp.hideAxis("bottom")
        lp.setRange(xRange=(-1.15, 1.15), yRange=(-1.15, 1.15), padding=0)
        self.circle = pg.PlotCurveItem()
        self.axes = [pg.InfiniteLine(pos=0.0, angle=a) for a in (0, 90)]
        self.full_path = pg.PlotCurveItem()  # the whole of the universal locus
        self.path = pg.PlotCurveItem()  # the page's speed range on it
        self.unstable_path = pg.PlotCurveItem()  # where a rigid rotor on it would whirl
        self.marks = pg.ScatterPlotItem(size=5)
        self.dot = pg.ScatterPlotItem(size=9)
        self.load_arrow = pg.ArrowItem(angle=-90, headLen=12, tailLen=26, tailWidth=2, pos=(0.0, -0.25))
        self.mark_labels: list[pg.TextItem] = []
        for item in (self.circle, *self.axes, self.full_path, self.path, self.unstable_path, self.marks,
                     self.dot, self.load_arrow):
            lp.addItem(item)
        self.locus.setFixedSize(LOCUS_SIZE, LOCUS_SIZE)
        self.readout = QtWidgets.QLabel()
        self.readout.setWordWrap(True)
        self.readout.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.readout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.locus, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        top.addWidget(self.readout, 1)

        self.coeffs = pg.GraphicsLayoutWidget()
        self.coeffs.setToolTip(COEFF_TIP)
        self.k_plot = self.coeffs.addPlot(row=1, col=0)
        self.c_plot = self.coeffs.addPlot(row=3, col=0)
        self.c_plot.setXLink(self.k_plot)
        # Each coefficient: its positive part, and its negative part (dashed, on the log scale).
        self.k_curves: list[tuple[pg.PlotDataItem, pg.PlotDataItem]] = []
        self.c_curves: list[tuple[pg.PlotDataItem, pg.PlotDataItem]] = []
        self.speed_lines: list[pg.InfiniteLine] = []
        self.legends: list[pg.LegendItem] = []
        for row, p, spec, curves in ((0, self.k_plot, STIFFNESS, self.k_curves),
                                     (2, self.c_plot, DAMPING, self.c_curves)):
            p.setMenuEnabled(False)
            p.setMouseEnabled(x=False, y=False)
            p.hideButtons()
            p.showGrid(x=True, y=True, alpha=0.15)
            p.getAxis("left").setWidth(AXIS_WIDTH)  # the same in both, so that their speeds line up
            legend = pg.LegendItem(colCount=len(spec), brush=colors.legend_brush(),
                                   labelTextColor=colors.foreground)
            self.coeffs.addItem(legend, row=row, col=0)
            self.legends.append(legend)
            for _, name, _ in spec:
                pair = (pg.PlotDataItem(connect="finite"), pg.PlotDataItem(connect="finite"))
                for curve in pair:
                    p.addItem(curve)
                legend.addItem(pair[0], name)
                curves.append(pair)
            line = pg.InfiniteLine(angle=90)
            p.addItem(line)
            self.speed_lines.append(line)
        self.range_band = pg.LinearRegionItem(movable=False)  # the page's speeds, on the dimensionless chart
        self.range_band.setZValue(-10)
        self.k_plot.addItem(self.range_band)
        self.note = QtWidgets.QLabel()
        self.note.setWordWrap(True)
        mute(self.note)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(top, 2)
        layout.addWidget(self.coeffs, 3)
        layout.addWidget(self.note)
        self.apply_theme()
        self._redraw()

    # ------------------------------------------------------------- inputs
    def bearing(self, rotor: bool = True) -> JournalBearing:
        """The bearing described by the inputs (SI): under the shown rotor bearing's load, if `rotor`."""
        load = self.load.value() if self.loads is None else self._own_load
        if rotor and self.loads is not None:
            load = abs(self.loads[self.which.currentIndex()])
        return JournalBearing(length=self.length.value() / 1e3, diameter=self.diameter.value() / 1e3,
                              clearance=self.clearance.value() / 1e6, viscosity=self.viscosity.value() / 1e3,
                              load=load)

    def set_bearing(self, b: JournalBearing, notify: bool = True) -> None:
        """Show and use `b` (for scripts and tests); `notify` tells the rotor, if it runs on it."""
        for box, value in ((self.length, b.length * 1e3), (self.diameter, b.diameter * 1e3),
                           (self.clearance, b.clearance * 1e6), (self.viscosity, b.viscosity * 1e3),
                           (self.load, b.load if self.loads is None else self.load.value())):
            box.blockSignals(True)
            box.setValue(value)
            box.blockSignals(False)
        if self.loads is not None:
            self._own_load = b.load
        self._redraw()
        if notify:
            self.changed.emit()

    def set_rotor(self, loads: tuple[float, float] | None) -> None:
        """The rotor runs on this bearing, its two carrying `loads` (N); or None: it does not."""
        if loads == self.loads:
            return
        if self.loads is None:
            self._own_load = self.load.value()  # the box shows the rotor's loads meanwhile
        elif loads is None:
            self.load.blockSignals(True)
            self.load.setValue(self._own_load)
            self.load.blockSignals(False)
        self.loads = loads
        on = loads is not None
        self.which.setVisible(on)
        self.load.setEnabled(not on)
        self.rotor_note.setText("the rotor's bearings, each under its share of the weight" if on else "")
        self._redraw()

    def set_onset(self, onset: Onset | None) -> None:
        """Where the rotor, running on this bearing, goes unstable (None: it does not, or it does not run on it)."""
        self.onset = onset
        self._show_speed()

    def set_top_rpm(self, rpm: float) -> None:
        self.top_rpm = rpm
        self._redraw()

    def set_speed(self, rpm: float) -> None:
        """The rotor's speed now: marked on the plots, and the readings are at it (call while the tab shows)."""
        self.rpm = rpm
        if abs(rpm - self._shown_rpm) >= 0.5:
            self._show_speed()

    def _on_params(self) -> None:
        self._redraw()
        self.changed.emit()

    def _redraw(self) -> None:
        b = self.bearing()
        if self.loads is not None:
            self.load.blockSignals(True)
            self.load.setValue(b.load)
            self.load.blockSignals(False)
        self._stale = self.isHidden()  # a tab not showing: draw when it does
        if self._stale:
            return
        threshold = b.threshold(max(self.top_rpm, THRESHOLD_TOP) / RPM)
        self._threshold = threshold
        self._draw_locus(b)
        self._draw_coefficients()
        ratio = b.length / b.diameter
        self.note.setText(
            f"L<sub>b</sub>/D<sub>b</sub> = {ratio:.2g}: past {SHORT_LIMIT:g} short-bearing theory underrates the film's load capacity, "
            "and a finite-length solution is needed." if ratio > SHORT_LIMIT else
            "Short-bearing (Ocvirk) theory, the film cavitating where its pressure would go below zero "
            "(the π-film). Choose <i>Oil-film journal bearings</i> under the rotor's bearing supports to run "
            "the rotor in it." if self.loads is None else
            "Short-bearing (Ocvirk) theory, the film cavitating where its pressure would go below zero "
            "(the π-film). The rotor runs in it: below the speed where ε = 0.9 it takes the film as it is "
            "there, the journal nearly on its bottom. With <i>Nonlinear film</i> ticked the time response "
            "takes the film's whole force; the coefficients here are its linearization.")
        self._show_speed()

    # ------------------------------------------------------------- drawing
    def showEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().showEvent(event)
        self.catch_up()

    def catch_up(self) -> None:
        """Draw what changed while the tab was hidden (its drawing waits till it shows)."""
        if self._stale:
            self._redraw()

    def apply_theme(self) -> None:
        self.circle.setPen(pg.mkPen(colors.structure, width=1.5))
        for line in self.axes:
            line.setPen(pg.mkPen(colors.faint, width=1))
        self.full_path.setPen(pg.mkPen(colors.grey, width=1, style=DASH))
        self.path.setPen(pg.mkPen(colors.strong, width=2))
        self.unstable_path.setPen(pg.mkPen(colors.poor, width=2.5))
        self.marks.setBrush(pg.mkBrush(colors.strong))
        self.marks.setPen(pg.mkPen(None))
        self.dot.setBrush(pg.mkBrush(colors.force))
        self.dot.setPen(pg.mkPen(colors.background, width=1))
        self.load_arrow.setStyle(brush=pg.mkBrush(colors.muted), pen=pg.mkPen(colors.muted))
        for label in self.mark_labels:
            label.setColor(colors.foreground)
        for legend in self.legends:
            legend.setBrush(colors.legend_brush())
            legend.setLabelTextColor(colors.foreground)
        for line in self.speed_lines:
            line.setPen(pg.mkPen(colors.force, width=1.5))
        band = pg.mkColor(colors.grey)
        band.setAlpha(40)
        self.range_band.setBrush(pg.mkBrush(band))
        for line in self.range_band.lines:
            line.setPen(pg.mkPen(None))
        self._draw_coefficients()

    def _draw_locus(self, b: JournalBearing) -> None:
        a = np.linspace(0.0, 2.0 * math.pi, 181)
        self.circle.setData(np.cos(a), np.sin(a))
        eps = np.linspace(1.0, 0.0, LOCUS_POINTS + 1)[:-1]
        x, y = _locus(eps)
        self.full_path.setData(x, y)
        top = self.top_rpm / RPM
        eps_top = b.state(top).eccentricity
        inside = eps >= eps_top
        self.path.setData(*_locus(np.append(eps[inside], eps_top)))
        if self._threshold is not None and self._threshold < top:
            e0 = b.state(self._threshold).eccentricity
            run = np.concatenate(([e0], eps[(eps < e0) & (eps > eps_top)], [eps_top]))
            self.unstable_path.setData(*_locus(run))
        else:
            self.unstable_path.setData([], [])
        # Round speeds along the range, labelled where they are not crowded (near the centre they bunch up).
        marks, pts = [], []
        for rpm in _round_speeds(self.top_rpm):
            pt = np.array(b.state(rpm / RPM).position())
            if not pts or np.hypot(*(pt - pts[-1])) >= LABEL_GAP:
                marks.append(rpm)
                pts.append(pt)
        pts = np.array(pts).reshape(-1, 2)
        self.marks.setData(pts[:, 0], pts[:, 1])
        plot = self.locus.getPlotItem()
        for label in self.mark_labels:
            plot.removeItem(label)
        self.mark_labels = []
        for rpm, (px, py) in zip(marks, pts):
            label = pg.TextItem(f"{rpm:,.0f}", color=colors.foreground, anchor=(0.0, 0.5))
            label.setPos(px + 0.05, py)
            plot.addItem(label)
            self.mark_labels.append(label)

    def _draw_coefficients(self) -> None:
        b = self.bearing()
        dimensionless = self.against.currentIndex() == 1
        loaded = b.load > 0.0
        if dimensionless and not loaded:
            b = b.with_load(1.0)  # the dimensionless curves are every bearing's: draw them at any load
        if dimensionless:
            x = np.geomspace(*SHORT_RANGE, SPEEDS)
            omega = np.asarray(b.omega_at(x / _aspect2(b)))
        else:
            top = self.top_rpm / RPM
            omega = np.linspace(top / SPEEDS, top, SPEEDS)
            x = omega * RPM
        states = [b.state(float(w)) for w in omega]
        k = np.array([st.k for st in states])
        c = np.array([st.c for st in states])
        if dimensionless:
            k = k * b.clearance / b.load
            c = c * (b.clearance * omega / b.load)[:, None, None]
        for p in (self.k_plot, self.c_plot):
            if p.ctrl.logXCheck.isChecked() != dimensionless:
                p.setRange(xRange=(0.0, 1.0), yRange=(0.0, 1.0), padding=0)  # no 10^(4e6) on the way
                p.setLogMode(x=dimensionless, y=dimensionless)
        units = (("K c<sub>r</sub> / W", ""), ("C c<sub>r</sub> Ω / W", "")) if dimensionless else (("Stiffness", "N/m"),
                                                                             ("Damping", "N·s/m"))
        for p, (label, unit) in zip((self.k_plot, self.c_plot), units):
            p.setLabel("left", label, units=unit)
        self.c_plot.setLabel("bottom", "S(L<sub>b</sub>/D<sub>b</sub>)²" if dimensionless else "Spin speed (rpm)")
        for curves, spec, values in ((self.k_curves, STIFFNESS, k), (self.c_curves, DAMPING, c)):
            for (pos, neg), ((i, j), _, color) in zip(curves, spec):
                v = values[:, i, j]
                pos.setPen(pg.mkPen(colors.mode[color], width=2))
                neg.setPen(pg.mkPen(colors.mode[color], width=2, style=DASH))
                if dimensionless:
                    # Log scale: the size, dashed where the coefficient is negative.
                    pos.setData(x, np.where(v > 0.0, v, np.nan))
                    neg.setData(x, np.where(v < 0.0, -v, np.nan))
                else:
                    pos.setData(x, v)
                    neg.setData([], [])
        for p, values in ((self.k_plot, k), (self.c_plot, c)):
            if dimensionless:
                v = np.abs(values).reshape(-1)
                v = np.log10(v[v > 0.0])
                p.setRange(xRange=np.log10(SHORT_RANGE), yRange=(max(float(v.min()), -3.0), float(v.max())),
                           padding=0.04)
            else:
                # Scale to the curves above the lowest speeds, where the film's stiffness climbs without bound.
                keep = x >= LOW_SPEED * self.top_rpm
                v = values[keep].reshape(-1)
                lo, hi = min(float(v.min()), 0.0), max(float(v.max()), 0.0)
                pad = 0.06 * (hi - lo)
                p.setRange(xRange=(0.0, self.top_rpm), yRange=(lo - pad, hi + pad), padding=0)
        if dimensionless:
            lo, hi = (_short_sommerfeld(b, LOW_SPEED * self.top_rpm / RPM),
                      _short_sommerfeld(b, self.top_rpm / RPM))
            self.range_band.setRegion((math.log10(lo), math.log10(hi)))
        self.range_band.setVisible(dimensionless and loaded)
        self._place_speed_lines()

    def _place_speed_lines(self) -> None:
        dimensionless = self.against.currentIndex() == 1
        b = self.bearing()
        if dimensionless and b.load > 0.0:
            value = math.log10(max(_short_sommerfeld(b, self.rpm / RPM), 1e-12))
        else:
            value = self.rpm
        for line in self.speed_lines:
            line.setValue(value)
            line.setVisible(self.rpm > 0.0 and (b.load > 0.0 or not dimensionless))

    def _show_speed(self) -> None:
        self._shown_rpm = self.rpm
        self._place_speed_lines()
        b = self.bearing()
        omega = self.rpm / RPM
        if omega <= 0.0:
            self.dot.setData([0.0], [-1.0])
            self.readout.setText(
                "<p>At rest the journal lies on the bottom of its clearance: there is no film to lift it, "
                "and no coefficients to speak of. Run the rotor up to see them.</p>" + self._threshold_html(b))
            return
        s = b.state(omega)
        px, py = s.position()
        self.dot.setData([px], [py])
        k, c = s.k, s.c
        swirl = (k[0, 1] - k[1, 0]) / (omega * (c[0, 0] + c[1, 1]))
        rows = "".join(
            f"<tr><td>{name}</td><td align='right'>{_si(k[i, j], 'N/m')}</td><td>&nbsp;&nbsp;c{name[1:]}</td>"
            f"<td align='right'>{_si(c[i, j], 'N·s/m')}</td></tr>"
            for (i, j), name in (((0, 0), "k<sub>xx</sub>"), ((0, 1), "k<sub>xy</sub>"),
                                 ((1, 0), "k<sub>yx</sub>"), ((1, 1), "k<sub>yy</sub>")))
        self.readout.setText(
            f"<p><b>At {self.rpm:,.0f} rpm</b>: Sommerfeld number "
            + (f"S = {b.sommerfeld(omega):.3g} (S(L<sub>b</sub>/D<sub>b</sub>)² = {_short_sommerfeld(b, omega):.3g}), " if b.load > 0.0
               else "S = ∞ (no load: a vertical rotor), ")
            + f"eccentricity ε = {s.eccentricity:.3f}, attitude angle χ = {math.degrees(s.attitude):.1f}°, "
            f"thinnest film {b.clearance * (1.0 - s.eccentricity) * 1e6:.1f} µm.</p>"
            f"<table cellspacing='0' cellpadding='1'>{rows}</table>"
            f"<p>(k<sub>xy</sub> − k<sub>yx</sub>) / Ω(c<sub>xx</sub> + c<sub>yy</sub>) = {swirl:.2f}: the "
            "swirl ratio λ of the panel's bearings, ½ at any load in short-bearing theory.</p>"
            + self._threshold_html(b))

    def _threshold_html(self, b: JournalBearing) -> str:
        return self._rigid_html(b) + self._rotor_html()

    def _rotor_html(self) -> str:
        if self.loads is None:
            return ""
        if self.onset is None:
            return f"<p>The rotor running on it stays stable up to {self.top_rpm:,.0f} rpm.</p>"
        if not any(self.loads):
            return ("<p>Unloaded (a vertical rotor), the film has no direct stiffness, only the swirl's "
                    "cross-coupling: the rotor running on it whirls at any speed.</p>")
        return (f"<p>The rotor running on it, its shaft flexible, goes unstable at <b>{self.onset.omega * RPM:,.0f} "
                f"rpm</b>, whirling at {self.onset.freq_hz:.3g} Hz ({self.onset.order:.2f}×): oil whip, near "
                "twice its first critical speed (the <i>Stability</i> tab).</p>")

    def _rigid_html(self, b: JournalBearing) -> str:
        mass = b.load / GRAVITY
        if self._threshold is None:
            return (f"<p>A rigid rotor whose weight is the load ({mass:.3g} kg per bearing) stays stable "
                    f"up to {max(self.top_rpm, THRESHOLD_TOP):,.0f} rpm.</p>")
        s = b.state(self._threshold)
        _, nu2 = rigid_threshold(s.k, s.c)
        ratio = math.sqrt(max(nu2, 0.0)) / self._threshold
        where = " (red on the locus)" if self._threshold * RPM < self.top_rpm else ", past the page's speed range"
        return (f"<p>A rigid rotor whose weight is the load ({mass:.3g} kg per bearing) whirls unstably "
                f"above <b>{self._threshold * RPM:,.0f} rpm</b>{where}, at {ratio:.2f}× the "
                f"speed: oil whirl. Past ε ≈ 0.76 it is stable at any speed.</p>")


def _round_speeds(top: float) -> list[float]:
    """1, 2, 5 times powers of ten (rpm) up to `top`, from 100 rpm."""
    return [m * 10.0**d for d in range(2, 6) for m in (1, 2, 5) if m * 10.0**d <= top]


def _aspect2(b: JournalBearing) -> float:
    """(L_b/D_b)²: S times it is what a short bearing's state depends on."""
    return (b.length / b.diameter) ** 2


def _short_sommerfeld(b: JournalBearing, omega: float) -> float:
    """S(L_b/D_b)² at the speed `omega` (rad/s), the dimensionless chart's x."""
    return float(b.sommerfeld(omega)) * _aspect2(b)


def _locus(eps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The journal's position (fractions of c) at eccentricities `eps`."""
    psi = np.array([attitude(float(e)) for e in eps])
    return eps * np.sin(psi), -eps * np.cos(psi)


def _si(value: float, unit: str) -> str:
    """`value` with an SI prefix: 1.23 MN/m."""
    if value == 0.0:
        return f"0 {unit}"
    exp = min(max(math.floor(math.log10(abs(value)) / 3.0), -2), 3)
    prefix = {-2: "µ", -1: "m", 0: "", 1: "k", 2: "M", 3: "G"}[exp]
    return f"{value / 10.0 ** (3 * exp):.3g} {prefix}{unit}"
