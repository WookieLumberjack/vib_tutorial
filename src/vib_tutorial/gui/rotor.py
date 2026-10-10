"""Jeffcott rotor page: a disc on a flexible shaft in flexible bearings, spun up through its criticals.

Left: the speed controls, the run-up and coast-down sweep, and the rotor's
parameters. Centre: a 3D view of the whirling shaft (projected onto a 2D
pyqtgraph scene, no OpenGL), the orbits at the bearings, the disc and midspan,
and a strip chart. Right: the Campbell diagram, the Bode and polar plots of the
1X response (steady state and tracked sweeps), the stability map and full
spectrum, the matrices of the equations of motion at the current speed, and the
theory notes.
"""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ..core.rotor import (
    RPM,
    XA,
    XB,
    XD,
    YA,
    YB,
    YD,
    Bearing,
    Campbell,
    RotorSimulator,
    RotorSystem,
    campbell,
    whirl_modes,
)
from ..core.rotor_sweep import (
    DIRECTIONS,
    STATIONS,
    RunUpCoastDown,
    TrackedSweep,
    lag_degrees,
    probe,
    steady_vectors,
)
from ..core.rotor_stability import Onset, full_spectrum, least_damped, stability_onset
from .cms_time import SampleBuffer
from .panels import spin
from .rotor_bode import SWEEP_COLORS, BodePlots, PolarPlot, SweepPlots, nice_marks
from .rotor_notes import THEORY_HTML, matrices_html, whirl_name
from .rotor_stability import MEASURES, FullSpectrum, StabilityMap
from .style import colors
from .theming import add_legend, mute

FRAME_MS = 16
SPEEDS = (0.1, 0.25, 0.5, 1.0, 2.0)  # simulation speed, × real time
MAX_RPM = 10_000.0
DIRECTION_NAMES = ("x (horizontal probe)", "y (vertical probe)", "Orbit (major axis)")
# Columns of the history: (x, y) at each station, then the unbalance's angle φ and the speed Ω.
PHASE, OMEGA = 8, 9
STORE_DT = 2.5e-4  # s; samples kept for the plots are at least this far apart
MIN_PEAK = 1e-9  # m; never magnify motion smaller than this
TAP_IMPULSE = 0.005  # N·s, downward on the disc
ORBIT_REVS = 3  # revolutions in each orbit trail
ORBIT_MAX_TIME = 1.0  # s, the longest orbit trail (at low speed)
AXIAL = 4.0  # display length of the shaft in the 3D view
SWING = 0.5  # display units: the largest whirl is magnified to this
GROUND = 1.1  # display units from the shaft's axis to the supports
DAMPED = 0.3  # modes with more damping than this are drawn hollow and are not called critical
DASH = QtCore.Qt.PenStyle.DashLine
DOT = QtCore.Qt.PenStyle.DotLine
STEADY_POINTS = 2000  # speeds on the steady-state curve, plus as many again around the criticals
RUB = 0.02  # of the span: a whirl this large pauses the simulation (a real rotor would rub)
SPECTRUM_FRAMES = 6  # the full spectrum is recomputed every this many frames

SPEED_TIP = (
    "<p>The speed the rotor ramps toward, at the ramp rate. Drag the red line on the Campbell diagram "
    "to set it too.</p><p>Ramp slowly through a critical speed and the whirl builds up to its "
    "steady-state peak; ramp quickly and it passes before the whirl can grow.</p>"
)
SWEEP_TIP = (
    "<p>Ramp to the first speed and let the start-up transient die away, then run up to the second "
    "speed and coast back down, at the ramp rate above. Once per revolution the 1X vector is measured "
    "at the chosen station, and drawn over the steady-state curve on the Bode and polar plots.</p>"
    "<p>Run it again at another ramp rate to compare: the faster the ramp, the lower the peak, and the "
    "further past the critical speed it comes.</p>"
)
HOLD_SWEEPS_TIP = (
    "<p>Keep the earlier sweeps on the plots when a new one starts, to compare ramp rates. Unticked, a "
    "new sweep replaces them. Changing the rotor clears them, since the steady-state curve changes.</p>"
)
TAP_TIP = (
    "<p>Hit the disc downward (an impulse of 5 mN·s). The tap sets every mode ringing at its own "
    "natural frequency: both the forward and the backward whirl of each mode. With the disc off "
    "centre, at speed, the gyroscopic effect pulls their frequencies apart and the orbit becomes a "
    "rosette (watch the Campbell diagram at the current speed).</p>"
)
VIEW_TIP = (
    "<p>The shaft's centre line, the disc and the bearings' springs and dampers, with the whirl "
    "magnified (the scale is in the top corner). Drag to turn the view; double-click to reset it.</p>"
    "<p>The red dot on the disc is the heavy spot, where its unbalance is: it turns with the shaft. "
    "The thin lines are the orbits of the four stations (bearings, disc, midspan).</p>"
)
INTERNAL_TIP = (
    "<p>Damping inside the rotating shaft (material hysteresis, slip in shrink fits and couplings), as "
    "the damping c<sub>i</sub> of a damper across the shaft at the disc. It resists the bending rate "
    "the spinning shaft sees, so below the critical speed it damps the forward whirl, and above it "
    "(the shaft then turns faster than it whirls) it drives it.</p>"
    "<p>Try 20 N·s/m: the rotor becomes unstable at 3,910 rpm and, above that, whirls at its first "
    "natural frequency, about 27 Hz, whatever the speed.</p>"
)
KXY_TIP = (
    "<p>Cross-coupled stiffness: a displacement in x pushes the journal in y, and one in y pushes it "
    "back in −x (k<sub>yx</sub> = −k<sub>xy</sub>). Fluid-film bearings, seals and impeller "
    "clearances do this, the fluid being dragged round by the shaft. Positive k<sub>xy</sub> pushes a "
    "forward whirl along its path: it lowers the forward modes' damping, and past the bearings' own "
    "damping, the rotor is unstable.</p><p>Here k<sub>xy</sub> is constant; in a real bearing it "
    "grows with speed, so read it as the value at the speed you look at. Try 15,000 N/m on both.</p>"
)
BOW_TIP = (
    "<p>A shaft bent for good (by a thermal bow, a sag left after standing, or a bent repair), by δ at "
    "the disc, toward an angle measured from the heavy spot in the direction of spin. It turns with the "
    "shaft, so it drives the rotor once per revolution like an unbalance, but with a force k δ that does "
    "not grow with speed.</p><p>Turning slowly, the probes read the bow itself (the <i>slow-roll "
    "runout</i>), and the polar plot starts from it rather than from the origin. Well above the critical "
    "speed the disc stays on the bearings' axis and the shaft bends round it. Put the bow at 180° to the "
    "heavy spot and the two cancel at one speed.</p>"
)
SKEW_TIP = (
    "<p>The disc mounted out of square: its own axis tilted by τ from the shaft's, toward an angle "
    "measured from the heavy spot in the direction of spin. Spinning, the tilted disc's inertia makes a "
    "moment (I<sub>d</sub> − I<sub>p</sub>) τ Ω² that turns with the shaft: a <i>couple unbalance</i>. "
    "It drives the disc's tilt directly, and pushes a thin disc (I<sub>p</sub> &gt; I<sub>d</sub>) toward "
    "square to the spin axis, bending the shaft at the disc by −τ at high speed.</p><p>At midspan on "
    "identical bearings it only tilts the disc; move the disc off centre and it moves it too.</p>"
)
ORBIT_TIP = (
    "<p>The path of the shaft's centre at each station over the last few revolutions, looking from "
    "the B end toward A: x to the right, y up, and the shaft spinning counterclockwise (x toward y). "
    "Each plot has its own scale unless <i>Same scale</i> is ticked.</p>"
    "<p>The small hollow circles are once-per-revolution marks (a keyphasor): where the centre was "
    "each time the heavy spot passed +x. On the disc's orbit the red line points at the heavy spot "
    "now; the angle from the high spot (where the disc is) to the heavy spot is the phase lag, "
    "0° well below the critical speed, 90° at it and 180° well above.</p>"
)


class RotorView(pg.PlotWidget):
    """The rotor in 3D, orthographically projected; drag to turn it."""

    YAW, PITCH = math.radians(62.0), math.radians(18.0)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setToolTip(VIEW_TIP)
        self.setAspectLocked(True)
        for side in ("left", "bottom"):
            self.hideAxis(side)
        self.setMouseEnabled(x=False, y=False)
        self.setMenuEnabled(False)
        self.hideButtons()
        self.yaw, self.pitch = self.YAW, self.PITCH
        self._drag: QtCore.QPointF | None = None
        self.system = RotorSystem()
        self.gain = 1.0  # display units per metre of lateral motion
        self.disc_radius = 0.4  # display units

        plot = self.getPlotItem()
        self.frame = pg.PlotCurveItem(connect="finite")  # base plate, side walls, axis
        self.axis = pg.PlotCurveItem()
        self.supports = pg.PlotCurveItem(connect="finite")  # springs and dampers
        self.trails = [pg.PlotCurveItem() for _ in STATIONS]
        self.shaft = pg.PlotCurveItem()
        self.disc = QtWidgets.QGraphicsPathItem()
        self.spoke = pg.PlotCurveItem()
        self.heavy = pg.ScatterPlotItem(size=9)
        self.stations = pg.ScatterPlotItem(size=8)
        self.triad = pg.PlotCurveItem(connect="pairs")
        self.triad_labels = [pg.TextItem(t, anchor=(0.5, 0.5)) for t in ("x", "y", "z")]
        self.end_labels = [pg.TextItem(t, anchor=(0.5, 0.0)) for t in ("A", "B")]
        for item in (self.frame, self.axis, self.supports, *self.trails, self.shaft, self.disc,
                     self.spoke, self.heavy, self.stations, self.triad, *self.triad_labels, *self.end_labels):
            plot.addItem(item)
        self.scale_label = pg.TextItem(anchor=(1.0, 0.0))
        plot.addItem(self.scale_label)
        self.apply_theme()
        self._place_static()

    # ------------------------------------------------------------- geometry
    def project(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Screen (x, y) of display points p (..., 3) = (lateral x, vertical y, axial z from midspan)."""
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        xr = x * cy + z * sy
        zr = -x * sy + z * cy  # toward the viewer
        return xr, y * cp - zr * sp

    def _z(self, z_m: float | np.ndarray) -> np.ndarray:
        """Display axial coordinate (from midspan) of an axial position in metres."""
        return (np.asarray(z_m) / self.system.length - 0.5) * AXIAL

    def set_system(self, system: RotorSystem) -> None:
        self.system = system
        r = math.sqrt(2.0 * system.ip / system.disc_mass) if system.ip > 0 else 0.05
        self.disc_radius = float(np.clip(r / system.length * AXIAL, 0.2, 0.9))
        self._place_static()

    def _segments(self, parts: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Screen coordinates of polylines joined with NaN breaks (for connect='finite')."""
        gap = np.full((1, 3), np.nan)
        pts = np.vstack([np.vstack([p, gap]) for p in parts]) if parts else np.empty((0, 3))
        return self.project(pts)

    def _place_static(self) -> None:
        """The base plate, the side walls, the undeflected axis, the triad and the view's range."""
        h, half = GROUND, AXIAL / 2 + 0.5
        w = 0.8
        plate = np.array([[-w, -h, -half], [w, -h, -half], [w, -h, half], [-w, -h, half], [-w, -h, -half]])
        walls = []
        for zb in (-AXIAL / 2, AXIAL / 2):
            walls.append(np.array([[-h, -0.35, zb - 0.3], [-h, 0.35, zb - 0.3], [-h, 0.35, zb + 0.3],
                                   [-h, -0.35, zb + 0.3], [-h, -0.35, zb - 0.3]]))
        self.frame.setData(*self._segments([plate, *walls]))
        self.axis.setData(*self.project(np.array([[0.0, 0.0, -half], [0.0, 0.0, half]])))
        o = np.array([w + 0.3, -h, half - 0.2])
        ends = [o + d for d in (np.array([0.45, 0, 0]), np.array([0, 0.45, 0]), np.array([0, 0, -0.45]))]
        tx, ty = self.project(np.array([v for e in ends for v in (o, e)]))
        self.triad.setData(tx, ty)
        for label, e in zip(self.triad_labels, ends):
            label.setPos(*[float(c) for c in self.project(o + 1.3 * (e - o))])
        for label, zb in zip(self.end_labels, (-AXIAL / 2, AXIAL / 2)):
            label.setPos(*[float(c) for c in self.project(np.array([0.0, -h - 0.05, zb]))])
        # Range: everything the rotor can reach.
        top = SWING + self.disc_radius + 0.1
        corners = np.array([[sx * (h + 0.2), sy, sz * half] for sx in (-1, 1) for sy in (-h - 0.15, top)
                            for sz in (-1, 1)])
        px, py = self.project(corners)
        self.setRange(xRange=(px.min(), px.max()), yRange=(py.min(), py.max()), padding=0.02)
        vr = self.getViewBox().viewRange()
        self.scale_label.setPos(vr[0][1], vr[1][1])

    def apply_theme(self) -> None:
        self.frame.setPen(pg.mkPen(colors.faint, width=1))
        self.axis.setPen(pg.mkPen(colors.faint, width=1, style=DASH))
        self.supports.setPen(pg.mkPen(colors.structure, width=1))
        for curve, i in zip(self.trails, range(len(STATIONS))):
            curve.setPen(pg.mkPen(colors.mass[i], width=1))
        self.shaft.setPen(pg.mkPen(colors.structure, width=3))
        fill = QtGui.QColor(colors.mass[1])
        fill.setAlpha(150)
        self.disc.setBrush(fill)
        pen = QtGui.QPen(QtGui.QColor(colors.mass[1]))
        pen.setCosmetic(True)
        pen.setWidth(2)
        self.disc.setPen(pen)
        self.spoke.setPen(pg.mkPen(colors.force, width=2))
        self.heavy.setBrush(pg.mkBrush(colors.force))
        self.heavy.setPen(pg.mkPen(colors.background, width=1))
        self.stations.setPen(pg.mkPen(colors.background, width=1))
        self.triad.setPen(pg.mkPen(colors.muted, width=1))
        for label in (*self.triad_labels, *self.end_labels, self.scale_label):
            label.setColor(colors.muted)

    # ------------------------------------------------------------- drawing
    def update_state(self, q: np.ndarray, phase: float, trails: list[np.ndarray], peak: float) -> None:
        """Draw displacements q (8,), the heavy spot at angle `phase`, and each station's recent (x, y)."""
        s = self.system
        # The disc's own axis: the shaft's slope there plus the skew. Its rim's wobble counts as motion.
        axis = np.array([q[2], q[6]]) + s.skew * np.array([math.cos(phase + s.skew_angle),
                                                           math.sin(phase + s.skew_angle)])
        peak = max(peak, float(np.hypot(*axis)) * self.disc_radius * s.length / AXIAL)
        target = SWING / max(peak, MIN_PEAK)
        # Shrink at once (keep the motion on screen), grow slowly (no flicker as it settles).
        self.gain = target if target < self.gain else self.gain + 0.08 * (target - self.gain)
        g = self.gain
        self.scale_label.setText(f"motion ×{_round_gain(g * s.length / AXIAL)}")

        # Shaft centre line.
        z = np.linspace(0.0, s.length, 41)
        defl = s.deflection(q, z) * g
        line = np.column_stack([defl, self._z(z)])
        self.shaft.setData(*self.project(line))

        # Stations and their orbits.
        zs = np.array([0.0, s.a, 0.5 * s.length, s.length])
        here = np.column_stack([s.deflection(q, zs) * g, self._z(zs)])
        # The midspan is drawn only when the disc is not there.
        shown = [i for i in range(len(STATIONS)) if i != 2 or abs(s.position - 0.5) > 1e-9]
        sx, sy = self.project(here[shown])
        self.stations.setData(sx, sy, brush=[pg.mkBrush(colors.mass[i]) for i in shown])
        for i, (curve, xy, zc) in enumerate(zip(self.trails, trails, self._z(zs))):
            if xy.size and i in shown:
                pts = np.column_stack([xy * g, np.full(len(xy), zc)])
                curve.setData(*self.project(pts))
            else:
                curve.setData([], [])

        # Disc: a circle square to its own (magnified) axis.
        scale = g * s.length / AXIAL  # display slope per radian of real slope
        n = np.array([axis[0] * scale, axis[1] * scale, 1.0])
        n /= np.linalg.norm(n)
        u = np.cross([0.0, 1.0, 0.0], n)
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        c = here[1]
        t = np.linspace(0.0, 2 * math.pi, 49)
        rim = c + self.disc_radius * (np.cos(t)[:, None] * u + np.sin(t)[:, None] * v)
        rx, ry = self.project(rim)
        path = QtGui.QPainterPath()
        path.addPolygon(QtGui.QPolygonF([QtCore.QPointF(float(a), float(b)) for a, b in zip(rx, ry)]))
        self.disc.setPath(path)
        spot = c + 0.8 * self.disc_radius * (math.cos(phase) * u + math.sin(phase) * v)
        self.spoke.setData(*self.project(np.vstack([c, spot])))
        hx, hy = self.project(spot[None, :])
        self.heavy.setData(hx, hy)

        # Supports: a spring and a damper to ground below (y) and to the side wall (x) at each bearing.
        parts = []
        for station in (0, 3):
            j = here[station]
            ground_y = np.array([0.0, -GROUND, j[2]])
            ground_x = np.array([-GROUND, 0.0, j[2]])
            for end in (ground_y, ground_x):
                parts += _spring_and_damper(j, end)
        self.supports.setData(*self._segments(parts))

    # ------------------------------------------------------------- mouse
    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._drag = event.position()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if self._drag is None:
            super().mouseMoveEvent(event)
            return
        d = event.position() - self._drag
        self._drag = event.position()
        self.yaw = (self.yaw - 0.01 * d.x()) % (2 * math.pi)
        self.pitch = float(np.clip(self.pitch + 0.01 * d.y(), -math.pi / 2, math.pi / 2))
        self._place_static()
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self._drag = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self.yaw, self.pitch = self.YAW, self.PITCH
        self._place_static()
        event.accept()


def _spring_and_damper(p0: np.ndarray, p1: np.ndarray) -> list[np.ndarray]:
    """Polylines for a spring and a damper side by side from p0 (the journal) to p1 (ground)."""
    d = p1 - p0
    side = np.array([0.0, 0.0, 0.12])  # spring and damper sit either side along the shaft
    s = np.linspace(0.0, 1.0, 15)
    zig = np.zeros_like(s)
    zig[2:-2:2], zig[3:-2:2] = 1.0, -1.0
    off = 0.06 * np.cross(d / np.linalg.norm(d), [0.0, 0.0, 1.0]) if abs(d[2]) < 1e-9 else np.zeros(3)
    spring = p0 - side + s[:, None] * d + zig[:, None] * off
    rod = np.vstack([p0 + side, p0 + side + 0.55 * d])
    w = off * 1.6
    cup = np.vstack([p0 + side + 0.45 * d + w, p0 + side + 0.7 * d + w, p0 + side + 0.7 * d - w,
                     p0 + side + 0.45 * d - w])
    piston = np.vstack([p0 + side + 0.55 * d + 0.8 * w, p0 + side + 0.55 * d - 0.8 * w])
    tail = np.vstack([p0 + side + 0.7 * d, p1 + side])
    bar = np.vstack([p0 - side * 1.5, p0 + side * 1.5])
    return [spring, rod, cup, piston, tail, bar]


def _round_gain(g: float) -> str:
    """A magnification for the label, to two significant figures: 0.52, 9.7, 970, 12,000."""
    g = float(f"{g:.2g}")
    return f"{g:,.0f}" if g >= 10 else f"{g:.2g}"


class OrbitPlots(pg.GraphicsLayoutWidget):
    """The orbit at each station, seen from the B end; the disc's with its heavy spot."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setToolTip(ORBIT_TIP)
        self.plots: list[pg.PlotItem] = []
        self.trails: list[pg.PlotDataItem] = []
        self.dots: list[pg.ScatterPlotItem] = []
        self.keys: list[pg.ScatterPlotItem] = []
        self.spans = [MIN_PEAK] * len(STATIONS)
        self.same_scale = False
        for i, name in enumerate(STATIONS):
            p = self.addPlot(row=0, col=i)
            p.setAspectLocked(True)
            p.setMouseEnabled(x=False, y=False)
            p.hideButtons()
            p.setMenuEnabled(False)
            p.setLabel("bottom", "x", units="m")
            p.setLabel("left", "y", units="m")
            p.getAxis("bottom").setStyle(maxTickLevel=0)
            p.getAxis("left").setStyle(maxTickLevel=0)
            for angle in (0, 90):
                line = pg.InfiniteLine(pos=0, angle=angle)
                p.addItem(line)
            self.trails.append(p.plot())
            self.keys.append(pg.ScatterPlotItem(size=6, symbol="o"))
            self.dots.append(pg.ScatterPlotItem(size=8))
            p.addItem(self.keys[-1])
            p.addItem(self.dots[-1])
            self.plots.append(p)
        self.heavy = self.plots[1].plot()
        self.apply_theme()

    def apply_theme(self) -> None:
        for i, p in enumerate(self.plots):
            p.setTitle(f"<span style='color:{colors.mass[i]}'><b>{self._title(i)}</b></span>", size="9pt")
            for item in p.items:
                if isinstance(item, pg.InfiniteLine):
                    item.setPen(pg.mkPen(colors.faint, width=1))
            self.trails[i].setPen(pg.mkPen(colors.mass[i], width=1))
            self.dots[i].setBrush(pg.mkBrush(colors.mass[i]))
            self.dots[i].setPen(pg.mkPen(colors.foreground, width=1))
            self.keys[i].setBrush(pg.mkBrush(None))
            self.keys[i].setPen(pg.mkPen(colors.foreground, width=1))
        self.heavy.setPen(pg.mkPen(colors.force, width=2))

    def _title(self, i: int) -> str:
        if i == 1 and not self.plots[2].isVisible():
            return "Disc (midspan)"
        return STATIONS[i]

    def set_midspan_shown(self, shown: bool) -> None:
        """Hide the midspan orbit when the disc is there."""
        if shown == self.plots[2].isVisible():
            return
        self.plots[2].setVisible(shown)
        layout = self.ci.layout
        layout.setColumnStretchFactor(2, 1 if shown else 0)
        layout.setColumnMaximumWidth(2, 1e6 if shown else 0)
        self.apply_theme()

    def update_orbits(self, trails: list[np.ndarray], keys: list[np.ndarray], heavy_angle: float | None) -> None:
        """Each station's trail (k, 2), its keyphasor points, and the heavy spot's angle now."""
        peaks = [float(np.abs(xy).max()) if xy.size else 0.0 for xy in trails]
        if self.same_scale:
            shown = [p for i, p in enumerate(peaks) if self.plots[i].isVisible()]
            peaks = [max(shown)] * len(peaks)
        for i, p in enumerate(self.plots):
            xy = trails[i]
            target = 1.15 * max(peaks[i], MIN_PEAK)
            span = self.spans[i]
            self.spans[i] = span = target if target > span else span + 0.1 * (target - span)
            if xy.size:
                self.trails[i].setData(xy[:, 0], xy[:, 1])
                self.dots[i].setData(xy[-1:, 0], xy[-1:, 1])
            else:
                self.trails[i].setData([], [])
                self.dots[i].setData([], [])
            k = keys[i]
            self.keys[i].setData(k[:, 0], k[:, 1]) if k.size else self.keys[i].setData([], [])
            p.setRange(xRange=(-span, span), yRange=(-span, span), padding=0)
        if heavy_angle is None:
            self.heavy.setData([], [])
        else:
            r = 0.9 * self.spans[1]
            self.heavy.setData([0.0, r * math.cos(heavy_angle)], [0.0, r * math.sin(heavy_angle)])


class TimePlots(pg.GraphicsLayoutWidget):
    """x and y at one station against time, and the speed."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.x_plot = self.addPlot(row=0, col=0)
        self.x_plot.setLabel("left", "Displacement", units="m")
        add_legend(self.x_plot, offset=(-5, 5), colCount=2)
        self.w_plot = self.addPlot(row=1, col=0)
        self.w_plot.setLabel("left", "Speed", units="rpm")
        self.w_plot.getAxis("left").enableAutoSIPrefix(False)
        self.w_plot.setLabel("bottom", "Time", units="s")
        self.w_plot.setXLink(self.x_plot)
        self.ci.layout.setRowStretchFactor(0, 2)
        self.ci.layout.setRowStretchFactor(1, 1)
        # As the other strip charts: no grid, 1 px pens, clipped and downsampled.
        for p in (self.x_plot, self.w_plot):
            p.setClipToView(True)
            p.setDownsampling(auto=True, mode="peak")
            p.setMouseEnabled(x=False, y=False)
            p.hideButtons()
        self.x_curve = self.x_plot.plot(name="x (horizontal)")
        self.y_curve = self.x_plot.plot(name="y (vertical)")
        self.w_curve = self.w_plot.plot()
        self.crit_lines: list[pg.InfiniteLine] = []
        self.apply_theme()

    def apply_theme(self) -> None:
        self.x_plot.legend.setBrush(colors.legend_brush())
        self.x_curve.setPen(pg.mkPen(colors.mode[1], width=1))
        self.y_curve.setPen(pg.mkPen(colors.mode[3], width=1))
        self.w_curve.setPen(pg.mkPen(colors.strong, width=1))
        for line in self.crit_lines:
            line.setPen(pg.mkPen(colors.force, width=1, style=DOT))

    def set_criticals(self, rpms: list[float]) -> None:
        for line in self.crit_lines:
            self.w_plot.removeItem(line)
        self.crit_lines = [pg.InfiniteLine(pos=r, angle=0) for r in rpms]
        for line in self.crit_lines:
            self.w_plot.addItem(line)
        self.apply_theme()

    def update_curves(self, t: np.ndarray, xy: np.ndarray, rpm: np.ndarray, window: float) -> None:
        self.x_curve.setData(t, xy[:, 0])
        self.y_curve.setData(t, xy[:, 1])
        self.w_curve.setData(t, rpm)
        if t.size:
            x0 = max(t[-1] - window, 0.0) if t[-1] > window else 0.0
            self.x_plot.setXRange(x0, x0 + window, padding=0)
            span = max(float(np.abs(xy).max()), MIN_PEAK)
            self.x_plot.setYRange(-span, span, padding=0.05)
            top = max(float(rpm.max()), *(line.value() for line in self.crit_lines[:1]), 100.0)
            self.w_plot.setYRange(0.0, 1.1 * top, padding=0)


class CampbellPlot(pg.GraphicsLayoutWidget):
    """Whirl frequencies against speed: forward and backward branches, the 1X line, the criticals."""

    speed_dragged = QtCore.Signal(float)  # rpm

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.plot = p = self.addPlot(row=0, col=0)
        p.setLabel("bottom", "Spin speed", units="rpm")
        p.setLabel("left", "Whirl frequency", units="Hz")
        p.getAxis("bottom").enableAutoSIPrefix(False)
        p.showGrid(x=True, y=True, alpha=0.15)
        p.setMenuEnabled(False)
        # The legend sits below the plot: inside, it would hide a branch wherever it went.
        self.legend = pg.LegendItem(colCount=2, brush=colors.legend_brush(), labelTextColor=colors.foreground)
        self.addItem(self.legend, row=1, col=0)
        self.points: dict[str, pg.ScatterPlotItem] = {}
        # Backward drawn first and larger: where the two branches coincide (an axisymmetric
        # rotor not spinning, or a mode the spin cannot split) both still show.
        for key, name, size in (("bw", "backward whirl", 7), ("fw", "forward whirl", 3),
                                ("planar", "straight-line whirl", 4)):
            self.points[key] = pg.ScatterPlotItem(size=size, name=name)
            p.addItem(self.points[key])
        self.damped = pg.ScatterPlotItem(size=3, name=f"heavily damped (ζ > {DAMPED:g})")
        p.addItem(self.damped)
        self.sync = p.plot(name="1X (synchronous)")
        self.uncrit = pg.ScatterPlotItem(size=11, symbol="o", name="backward crossing, not driven")
        p.addItem(self.uncrit)
        self.crit = pg.ScatterPlotItem(size=11, symbol="o", name="critical speed")
        p.addItem(self.crit)
        self.crit_labels: list[pg.TextItem] = []
        self.unstable = pg.ScatterPlotItem(size=7, symbol="x", name="growing (ζ &lt; 0)")
        p.addItem(self.unstable)
        self.onset_line = pg.InfiniteLine(angle=90)
        self.onset_label = pg.TextItem(anchor=(1.05, 0.0))
        p.addItem(self.onset_line)
        p.addItem(self.onset_label)
        self.speed_line = pg.InfiniteLine(angle=90, movable=True)
        self.speed_line.setToolTip("The speed now. Drag it to set the target speed.")
        self.speed_line.sigPositionChangeFinished.connect(lambda line: self.speed_dragged.emit(max(0.0, line.value())))
        p.addItem(self.speed_line)
        for item in (*self.points.values(), self.damped, self.sync, self.crit, self.uncrit, self.unstable):
            self.legend.addItem(item, item.name())
        self.result: Campbell | None = None
        self.apply_theme()

    def apply_theme(self) -> None:
        brushes = {"fw": colors.mode[1], "bw": colors.mode[3], "planar": colors.grey}
        for key, item in self.points.items():
            item.setBrush(pg.mkBrush(brushes[key]))
            item.setPen(pg.mkPen(None))
        self.damped.setBrush(pg.mkBrush(None))
        self.damped.setPen(pg.mkPen(colors.grey, width=1))
        self.sync.setPen(pg.mkPen(colors.strong, width=1, style=DASH))
        self.crit.setBrush(pg.mkBrush(None))
        self.crit.setPen(pg.mkPen(colors.force, width=2))
        self.uncrit.setBrush(pg.mkBrush(None))
        self.uncrit.setPen(pg.mkPen(colors.grey, width=2))
        self.speed_line.setPen(pg.mkPen(colors.force, width=2))
        self.speed_line.setHoverPen(pg.mkPen(colors.force, width=4))
        self.unstable.setBrush(pg.mkBrush(colors.force))
        self.unstable.setPen(pg.mkPen(None))
        self.onset_line.setPen(pg.mkPen(colors.force, width=1.5, style=DASH))
        self.onset_label.setColor(colors.force)
        for label in self.crit_labels:
            label.setColor(colors.force)

    def set_result(self, result: Campbell, max_rpm: float, isotropic: bool, onset: Onset | None = None) -> None:
        self.result = result
        rpm = np.repeat(result.omega * RPM, result.freq_hz.shape[1])
        f = result.freq_hz.ravel()
        whirl = result.whirl.ravel()
        damped = result.zeta.ravel() > DAMPED
        ok = np.isfinite(f)
        classes = {"fw": whirl > 0.1, "bw": whirl < -0.1}
        classes["planar"] = ~(classes["fw"] | classes["bw"])
        for key, mask in classes.items():
            m = mask & ok & ~damped
            self.points[key].setData(rpm[m], f[m])
        self.damped.setData(rpm[ok & damped], f[ok & damped])
        growing = ok & (result.zeta.ravel() < 0.0)
        self.unstable.setData(rpm[growing], f[growing])
        self.onset_line.setVisible(onset is not None)
        if onset is not None:
            self.onset_line.setValue(onset.omega * RPM)
            self.onset_label.setText(f"instability onset\n{onset.omega * RPM:,.0f} rpm")
        else:
            self.onset_label.setText("")
        self.sync.setData([0.0, max_rpm], [0.0, max_rpm / 60.0])
        p = self.plot
        for label in self.crit_labels:
            p.removeItem(label)
        driven = critical_speeds(result, isotropic)
        crits = [w * RPM for w, _ in driven]
        self.crit.setData(crits, [c / 60.0 for c in crits])
        others = [w * RPM for w, r in lightly_damped(result) if (w, r) not in driven]
        self.uncrit.setData(others, [c / 60.0 for c in others])
        self.crit_labels = []
        for rpm, names in _grouped(critical_speeds(result, isotropic)):
            label = pg.TextItem(f"{rpm:,.0f} rpm ({' + '.join(names)})", anchor=(-0.05, 1.1))
            label.setPos(rpm, rpm / 60.0)
            p.addItem(label)
            self.crit_labels.append(label)
        top = np.nanmax(f[ok & (f < 2.5 * max_rpm / 60.0)]) if np.any(ok & (f < 2.5 * max_rpm / 60.0)) else 1.0
        y_top = 1.1 * max(top, max_rpm / 60.0)
        p.setRange(xRange=(0.0, max_rpm), yRange=(0.0, y_top), padding=0)
        if onset is not None:
            self.onset_label.setAnchor((1.05, 0.0) if onset.omega * RPM > 0.5 * max_rpm else (-0.05, 0.0))
            self.onset_label.setPos(onset.omega * RPM, 0.97 * y_top)
        self.apply_theme()

    def set_speed(self, rpm: float) -> None:
        if not self.speed_line.moving:
            self.speed_line.setValue(rpm)


class RotorPage(QtWidgets.QWidget):
    """Parameters and speed on the left; 3D view, orbits and time on the centre; Campbell, matrices, theory."""

    follows_chain = False  # MainWindow: this page has its own model, not the chain

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.sim = RotorSimulator()
        self.sim.target = 1000.0 / RPM
        self.history = SampleBuffer(10)
        self.campbell: Campbell | None = None
        self.running = True
        self._stride = 1
        self._carry = 0  # steps since the last stored sample
        self._matrices_speed = -1.0  # rpm the matrices were drawn at
        self.run_up: RunUpCoastDown | None = None  # the sweep running, if any
        self.sweeps: list[TrackedSweep] = []  # every leg on the plots, the running one's too
        self._sweep_pairs = 0  # sweeps started since the plots were cleared (each pair's colour)
        self._steady_rpm = np.empty(0)  # the steady-state curve: speeds (rpm)
        self._steady_vectors = np.empty((0, len(STATIONS), 2), dtype=complex)  # and 1X vectors there
        self.onset: Onset | None = None  # where the rotor goes unstable, if it does up to MAX_RPM
        self.rubbed = False  # the whirl grew past RUB of the span and the simulation paused
        self._frames = 0

        # --- parameters
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.addWidget(self._speed_box())
        self.readout = QtWidgets.QLabel()
        self.readout.setWordWrap(True)
        self.readout.setTextFormat(QtCore.Qt.TextFormat.RichText)
        lv.addWidget(self.readout)
        lv.addWidget(self._sweep_box())
        lv.addWidget(self._rotor_box())
        lv.addWidget(self._bearing_box())
        self.crit_label = QtWidgets.QLabel()
        self.crit_label.setWordWrap(True)
        lv.addWidget(self.crit_label)
        lv.addStretch(1)
        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidget(left)
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(470)

        # --- centre
        self.view = RotorView()
        self.orbits = OrbitPlots()
        self.time = TimePlots()
        self.station = QtWidgets.QComboBox()
        self.station.addItems(STATIONS)
        self.station.setCurrentIndex(1)
        self.window = spin(0.2, 20.0, 3.0, 1, " s")
        self.window.setToolTip("Strip chart window")
        self.same_scale = QtWidgets.QCheckBox("Same scale")
        self.same_scale.setToolTip("Draw the orbits at one scale, to compare their sizes")
        self.same_scale.toggled.connect(self._on_same_scale)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Orbits, seen from the B end:"))
        row.addWidget(self.same_scale)
        row.addStretch(1)
        row.addWidget(QtWidgets.QLabel("Strip chart:"))
        row.addWidget(self.station)
        row.addWidget(QtWidgets.QLabel("Window:"))
        row.addWidget(self.window)
        center = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(center)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.addWidget(self.view, 5)
        cv.addLayout(row)
        cv.addWidget(self.orbits, 3)
        cv.addWidget(self.time, 3)

        # --- right
        self.campbell_plot = CampbellPlot()
        self.campbell_plot.speed_dragged.connect(self.set_target_rpm)
        self.campbell_note = QtWidgets.QLabel()
        self.campbell_note.setWordWrap(True)
        mute(self.campbell_note)
        campbell_tab = QtWidgets.QWidget()
        ct = QtWidgets.QVBoxLayout(campbell_tab)
        ct.addWidget(self.campbell_plot, 1)
        ct.addWidget(self.campbell_note)
        self.bode = BodePlots()
        self.polar = PolarPlot()
        self.sweep_plots = SweepPlots(self.bode, self.polar)
        self.sweep_summary = QtWidgets.QLabel()
        self.sweep_summary.setWordWrap(True)
        self.sweep_summary.setTextFormat(QtCore.Qt.TextFormat.RichText)
        bode_tab = QtWidgets.QWidget()
        bt = QtWidgets.QVBoxLayout(bode_tab)
        bt.addWidget(self.bode, 1)
        bt.addWidget(self.sweep_summary)
        self.polar_note = QtWidgets.QLabel(
            "The 1X vector at the chosen station: its length is the amplitude, its angle clockwise from "
            "the right the phase lag. Through a lightly damped critical speed the steady-state vector "
            "traces a circle, lagging 90° at its lowest point. With a bowed shaft the curve starts from the "
            "bow (the slow-roll runout), not from the origin. The labels give the speed in rpm."
        )
        self.polar_note.setWordWrap(True)
        mute(self.polar_note)
        polar_tab = QtWidgets.QWidget()
        pt = QtWidgets.QVBoxLayout(polar_tab)
        pt.addWidget(self.polar, 1)
        pt.addWidget(self.polar_note)
        self.stability_map = StabilityMap(DAMPED)
        self.measure = QtWidgets.QComboBox()
        self.measure.addItems(MEASURES)
        self.measure.setToolTip(
            "<p>The log decrement δ = 2πζ/√(1 − ζ²) is the natural log of the ratio of successive peaks of "
            "a decaying vibration; API 684 asks for δ ≥ 0.1 at the running speed.</p>"
        )
        self.measure.currentIndexChanged.connect(lambda i: self.stability_map.set_measure(i == 1))
        self.stability_note = QtWidgets.QLabel()
        self.stability_note.setWordWrap(True)
        self.stability_note.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.spectrum = FullSpectrum()
        self.spectrum_note = QtWidgets.QLabel()
        self.spectrum_note.setWordWrap(True)
        mute(self.spectrum_note)
        stability_tab = QtWidgets.QWidget()
        st = QtWidgets.QVBoxLayout(stability_tab)
        measure_row = QtWidgets.QHBoxLayout()
        measure_row.addWidget(QtWidgets.QLabel("Stability map:"))
        measure_row.addWidget(self.measure)
        measure_row.addStretch(1)
        st.addLayout(measure_row)
        st.addWidget(self.stability_map, 3)
        st.addWidget(self.stability_note)
        st.addWidget(self.spectrum, 2)
        st.addWidget(self.spectrum_note)
        self.matrices = QtWidgets.QTextBrowser()
        self.theory = QtWidgets.QTextBrowser()
        self.theory.setHtml(THEORY_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(campbell_tab, "Campbell diagram")
        self.tabs.addTab(bode_tab, "Bode plot")
        self.tabs.addTab(polar_tab, "Polar plot")
        self.tabs.addTab(stability_tab, "Stability")
        self.tabs.addTab(self.matrices, "Equations && matrices")
        self.tabs.addTab(self.theory, "Theory")
        self.tabs.currentChanged.connect(lambda _: self._refresh_matrices())

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(left_scroll)
        splitter.addWidget(center)
        splitter.addWidget(self.tabs)
        splitter.setSizes([470, 720, 510])
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        # --- loop
        self._clock = QtCore.QElapsedTimer()
        self._last = 0.0
        self._target_t = 0.0
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(FRAME_MS)
        self._timer.timeout.connect(self._frame)
        # Parameter edits are applied at once; the Campbell diagram is recomputed once they settle.
        self._analysis_timer = QtCore.QTimer(self)
        self._analysis_timer.setSingleShot(True)
        self._analysis_timer.setInterval(150)
        self._analysis_timer.timeout.connect(self._refresh_analysis)

        self._on_params()
        self._refresh_analysis()
        self._draw()

    # ------------------------------------------------------------- panels
    def _speed_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Speed")
        form = QtWidgets.QFormLayout(box)
        self.speed_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.speed_slider.setRange(0, int(MAX_RPM))
        self.speed_slider.setSingleStep(10)
        self.speed_slider.setPageStep(250)
        self.speed_slider.setToolTip(SPEED_TIP)
        self.target_rpm = spin(0.0, MAX_RPM, 1000.0, 0, " rpm")
        self.target_rpm.setToolTip(SPEED_TIP)
        self.speed_slider.setValue(1000)
        self.speed_slider.valueChanged.connect(lambda v: self.target_rpm.setValue(float(v)))
        self.target_rpm.valueChanged.connect(self._on_target)
        form.addRow("Target speed:", self.target_rpm)
        form.addRow(self.speed_slider)
        self.ramp = spin(10.0, 20_000.0, 500.0, 0, " rpm/s")
        self.ramp.setToolTip("How fast the speed changes toward the target (run-up or coast-down rate)")
        self.ramp.valueChanged.connect(self._on_ramp)
        form.addRow("Ramp rate:", self.ramp)
        self.sim_speed = QtWidgets.QComboBox()
        for s in SPEEDS:
            self.sim_speed.addItem(f"{s:g}× real time", s)
        self.sim_speed.setCurrentIndex(SPEEDS.index(0.25))
        self.sim_speed.setToolTip("Slow motion, so the whirl stays visible at speed")
        form.addRow("Simulation:", self.sim_speed)
        buttons = QtWidgets.QHBoxLayout()
        self.run_button = QtWidgets.QPushButton("Pause")
        self.run_button.clicked.connect(self._on_run)
        self.reset_button = QtWidgets.QPushButton("Reset")
        self.reset_button.setToolTip("Back to rest, not spinning; it then ramps up to the target speed")
        self.reset_button.clicked.connect(self.reset)
        self.tap_button = QtWidgets.QPushButton("Tap disc")
        self.tap_button.setToolTip(TAP_TIP)
        self.tap_button.clicked.connect(self.tap)
        for b in (self.run_button, self.reset_button, self.tap_button):
            buttons.addWidget(b)
        form.addRow(buttons)
        return box

    def _sweep_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Run-up and coast-down")
        form = QtWidgets.QFormLayout(box)
        self.sweep_from = spin(60.0, MAX_RPM, 800.0, 0, " rpm")
        self.sweep_to = spin(60.0, MAX_RPM, 2400.0, 0, " rpm")
        for w in (self.sweep_from, self.sweep_to):
            w.setToolTip(SWEEP_TIP)
            w.valueChanged.connect(self._fit_tracking_range)
        speeds = QtWidgets.QHBoxLayout()
        speeds.addWidget(self.sweep_from, 1)
        speeds.addWidget(QtWidgets.QLabel("to"))
        speeds.addWidget(self.sweep_to, 1)
        form.addRow("From:", speeds)
        buttons = QtWidgets.QHBoxLayout()
        self.sweep_button = QtWidgets.QPushButton("Run up and coast down")
        self.sweep_button.setToolTip(SWEEP_TIP)
        self.sweep_button.clicked.connect(self._on_sweep_button)
        self.clear_sweeps_button = QtWidgets.QPushButton("Clear")
        self.clear_sweeps_button.setToolTip("Take every sweep off the Bode and polar plots")
        self.clear_sweeps_button.clicked.connect(self.clear_sweeps)
        buttons.addWidget(self.sweep_button, 2)
        buttons.addWidget(self.clear_sweeps_button, 1)
        form.addRow(buttons)
        self.hold_sweeps = QtWidgets.QCheckBox("Hold earlier sweeps for comparison")
        self.hold_sweeps.setChecked(True)
        self.hold_sweeps.setToolTip(HOLD_SWEEPS_TIP)
        form.addRow(self.hold_sweeps)
        self.probe_station = QtWidgets.QComboBox()
        self.probe_station.addItems(STATIONS)
        self.probe_station.setCurrentIndex(1)
        self.probe_direction = QtWidgets.QComboBox()
        self.probe_direction.addItems(DIRECTION_NAMES)
        self.probe_direction.setToolTip(
            "<p>What the Bode and polar plots show: a probe reading x or y, or the orbit's major axis. "
            "The phase lag is the angle the shaft turns from the keyphasor (the heavy spot passing +x) "
            "to the probe's positive peak, so on a circular orbit the y probe reads 90° more than the x "
            "probe. For the orbit, it is the angle from the heavy spot back to the high spot.</p>"
        )
        for c in (self.probe_station, self.probe_direction):
            c.currentIndexChanged.connect(self._on_probe)
        probe_row = QtWidgets.QHBoxLayout()
        probe_row.addWidget(self.probe_station, 1)
        probe_row.addWidget(self.probe_direction, 2)
        form.addRow("Plot:", probe_row)
        self.show_peaks = QtWidgets.QCheckBox("Per-revolution peaks (dots)")
        self.show_peaks.setChecked(True)
        self.show_peaks.setToolTip(
            "<p>Also draw the largest displacement over each revolution (the largest orbit radius, for the "
            "orbit). Unlike the 1X vector it includes the free vibration at the natural frequency, so "
            "just past the critical speed it shows the beating.</p>"
        )
        self.show_peaks.toggled.connect(self._on_show_peaks)
        form.addRow(self.show_peaks)
        self.sweep_status = QtWidgets.QLabel()
        self.sweep_status.setWordWrap(True)
        mute(self.sweep_status)
        form.addRow(self.sweep_status)
        return box

    def _rotor_box(self) -> QtWidgets.QGroupBox:
        s = RotorSystem()
        box = QtWidgets.QGroupBox("Shaft and disc")
        form = QtWidgets.QFormLayout(box)
        self.length = spin(0.05, 5.0, s.length, 3, " m")
        self.length.setToolTip("Bearing span L")
        self.diameter = spin(1.0, 500.0, s.diameter * 1e3, 1, " mm")
        self.diameter.setToolTip("Shaft diameter d; EI = E π d⁴/64")
        self.youngs = spin(1.0, 1000.0, s.youngs / 1e9, 1, " GPa")
        self.youngs.setToolTip("Young's modulus E of the shaft (steel ≈ 200 GPa)")
        self.position = spin(0.05, 0.95, s.position, 3)
        self.position.setSingleStep(0.05)
        self.position.setToolTip(
            "<p>Where the disc sits, a/L from bearing A. At 0.5 (midspan) with identical bearings the "
            "disc's translation and tilt are uncoupled, so the spin's gyroscopic moment never shows in "
            "the translational whirl: the classic Jeffcott rotor. Move it off centre and the disc tilts "
            "as it whirls.</p>"
        )
        self.disc_mass = spin(0.01, 1000.0, s.disc_mass, 3, " kg")
        self.ip = spin(1e-6, 100.0, s.ip, 6, " kg·m²")
        self.ip.setToolTip("Polar moment of inertia, about the spin axis (a thin disc: m R²/2)")
        self.id = spin(1e-6, 100.0, s.id, 6, " kg·m²")
        self.id.setToolTip("Diametral moment of inertia, about a diameter (a thin disc: m R²/4 = Ip/2)")
        self.unbalance = spin(0.0, 1e6, s.unbalance * 1e6, 1, " g·mm")
        self.unbalance.setToolTip(
            "<p>Mass unbalance U = m e: the disc's mass times the distance e of its centre of mass from "
            "the shaft's axis. 20 g·mm on a 1 kg disc is e = 20 µm, about balance grade G6.3 at "
            "3000 rpm (e Ω = 6.3 mm/s).</p>"
        )
        form.addRow("Span L:", self.length)
        form.addRow("Shaft diameter:", self.diameter)
        form.addRow("Young's modulus:", self.youngs)
        form.addRow("Disc position a/L:", self.position)
        form.addRow("Disc mass m:", self.disc_mass)
        form.addRow("Polar inertia Ip:", self.ip)
        form.addRow("Diametral inertia Id:", self.id)
        form.addRow("Unbalance m·e:", self.unbalance)
        self.bow = spin(0.0, 5000.0, s.bow * 1e6, 1, " µm")
        self.bow_angle = spin(-180.0, 180.0, math.degrees(s.bow_angle), 0, "°")
        self.skew = spin(0.0, 10.0, math.degrees(s.skew), 3, "°")
        self.skew_angle = spin(-180.0, 180.0, math.degrees(s.skew_angle), 0, "°")
        for size, angle, tip, label in ((self.bow, self.bow_angle, BOW_TIP, "Shaft bow δ<sub>b</sub>:"),
                                        (self.skew, self.skew_angle, SKEW_TIP, "Disc skew τ:")):
            angle.setWrapping(True)
            angle.setStepType(QtWidgets.QAbstractSpinBox.StepType.DefaultStepType)
            angle.setSingleStep(15.0)
            row = QtWidgets.QHBoxLayout()
            row.addWidget(size, 3)
            row.addWidget(QtWidgets.QLabel("at"))
            row.addWidget(angle, 2)
            for w in (size, angle):
                w.setToolTip(tip)
            form.addRow(label, row)
        self.internal_damping = spin(0.0, 1e5, s.internal_damping, 1, " N·s/m")
        self.internal_damping.setToolTip(INTERNAL_TIP)
        form.addRow("Internal damping c<sub>i</sub>:", self.internal_damping)
        self.shaft_note = QtWidgets.QLabel()
        self.shaft_note.setWordWrap(True)
        mute(self.shaft_note)
        form.addRow(self.shaft_note)
        for box_ in (self.length, self.diameter, self.youngs, self.position, self.disc_mass, self.ip,
                     self.id, self.unbalance, self.bow, self.bow_angle, self.skew, self.skew_angle,
                     self.internal_damping):
            box_.valueChanged.connect(self._on_params)
        return box

    def _bearing_box(self) -> QtWidgets.QGroupBox:
        b = Bearing()
        box = QtWidgets.QGroupBox("Bearing supports")
        grid = QtWidgets.QGridLayout(box)
        heads = (("k<sub>x</sub> [N/m]", "Horizontal support stiffness"),
                 ("k<sub>y</sub> [N/m]", "Vertical support stiffness"),
                 ("c<sub>x</sub> [N·s/m]", "Horizontal support damping"),
                 ("c<sub>y</sub> [N·s/m]", "Vertical support damping"),
                 ("m [kg]", "Mass of the journal and whatever moves with it on the support"),
                 ("k<sub>xy</sub> [N/m]", KXY_TIP))
        for c, (text, tip) in enumerate(heads):
            label = QtWidgets.QLabel(text)
            label.setToolTip(tip)
            grid.addWidget(label, 0, c + 1)
        self.bearing_rows: list[list[QtWidgets.QDoubleSpinBox]] = []
        for r, name in enumerate("AB"):
            grid.addWidget(QtWidgets.QLabel(f"<b>{name}</b>"), r + 1, 0)
            row = [spin(1e2, 1e10, b.kx, 0), spin(1e2, 1e10, b.ky, 0), spin(0.0, 1e6, b.cx, 1),
                   spin(0.0, 1e6, b.cy, 1), spin(0.001, 100.0, b.mass, 3), spin(-1e9, 1e9, b.kxy, 0)]
            row[5].setToolTip(KXY_TIP)
            for c, w in enumerate(row):
                w.setMinimumWidth(52)
                w.valueChanged.connect(self._on_params)
                grid.addWidget(w, r + 1, c + 1)
            self.bearing_rows.append(row)
        self.isotropic = QtWidgets.QCheckBox("Isotropic (ky = kx, cy = cx)")
        self.isotropic.setChecked(True)
        self.isotropic.setToolTip(
            "<p>Untick to give the supports a different stiffness or damping vertically. The single "
            "critical speed then splits in two, the orbits become ellipses, and between the two "
            "criticals the shaft whirls backward, against its spin.</p>"
        )
        self.same_bearings = QtWidgets.QCheckBox("B same as A")
        self.same_bearings.setChecked(True)
        self.same_bearings.setToolTip(
            "<p>Untick to give the two bearings different supports. Like an off-centre disc, unequal "
            "supports couple the disc's translation and tilt, so the gyroscopic moment shows.</p>"
        )
        for c in (self.isotropic, self.same_bearings):
            c.toggled.connect(self._on_params)
        grid.addWidget(self.isotropic, 3, 0, 1, 7)
        grid.addWidget(self.same_bearings, 4, 0, 1, 7)
        return box

    # ------------------------------------------------------------- parameters
    def system(self) -> RotorSystem:
        """The rotor described by the panels, after applying the isotropic and same-as-A ties."""
        iso, same = self.isotropic.isChecked(), self.same_bearings.isChecked()
        for r, row in enumerate(self.bearing_rows):
            src = self.bearing_rows[0] if same and r == 1 else row
            for c, w in enumerate(row):
                value = src[c].value()
                if iso and c in (1, 3):
                    value = src[c - 1].value()
                if w.value() != value:
                    w.blockSignals(True)
                    w.setValue(value)
                    w.blockSignals(False)
                w.setEnabled(not (same and r == 1) and not (iso and c in (1, 3)))
        bearings = [Bearing(*(w.value() for w in row)) for row in self.bearing_rows]
        return RotorSystem(
            length=self.length.value(),
            diameter=self.diameter.value() / 1e3,
            youngs=self.youngs.value() * 1e9,
            position=self.position.value(),
            disc_mass=self.disc_mass.value(),
            ip=self.ip.value(),
            id=self.id.value(),
            bearing_a=bearings[0],
            bearing_b=bearings[1],
            unbalance=self.unbalance.value() / 1e6,
            bow=self.bow.value() / 1e6,
            bow_angle=math.radians(self.bow_angle.value()),
            skew=math.radians(self.skew.value()),
            skew_angle=math.radians(self.skew_angle.value()),
            internal_damping=self.internal_damping.value(),
        )

    def set_system(self, system: RotorSystem) -> None:
        """Show and use `system` (for scripts and tests)."""
        boxes = (self.length, self.diameter, self.youngs, self.position, self.disc_mass, self.ip, self.id,
                 self.unbalance, self.bow, self.bow_angle, self.skew, self.skew_angle, self.internal_damping,
                 self.isotropic, self.same_bearings, *(w for row in self.bearing_rows for w in row))
        for w in boxes:
            w.blockSignals(True)
        self.length.setValue(system.length)
        self.diameter.setValue(system.diameter * 1e3)
        self.youngs.setValue(system.youngs / 1e9)
        self.position.setValue(system.position)
        self.disc_mass.setValue(system.disc_mass)
        self.ip.setValue(system.ip)
        self.id.setValue(system.id)
        self.unbalance.setValue(system.unbalance * 1e6)
        self.bow.setValue(system.bow * 1e6)
        self.bow_angle.setValue(math.degrees(system.bow_angle))
        self.skew.setValue(math.degrees(system.skew))
        self.skew_angle.setValue(math.degrees(system.skew_angle))
        self.internal_damping.setValue(system.internal_damping)
        A, B = system.bearing_a, system.bearing_b
        self.isotropic.setChecked(all(b.kx == b.ky and b.cx == b.cy for b in (A, B)))
        self.same_bearings.setChecked(A == B)
        for row, b in zip(self.bearing_rows, (A, B)):
            for w, v in zip(row, (b.kx, b.ky, b.cx, b.cy, b.mass, b.kxy)):
                w.setValue(v)
        for w in boxes:
            w.blockSignals(False)
        self._on_params()

    def _on_params(self) -> None:
        system = self.system()
        if self.sweeps or self.run_up is not None:
            self.clear_sweeps()  # their steady-state curve is about to change
        self.sim.set_system(system)
        self.view.set_system(system)
        self.orbits.set_midspan_shown(abs(system.position - 0.5) > 1e-9)
        self._stride = max(1, round(STORE_DT / self.sim.step_size()))
        ks = 48.0 * system.ei / system.length**3
        self.shaft_note.setText(
            f"EI = {system.ei:.4g} N·m²; shaft stiffness at the disc (bearings rigid) "
            f"{3 * system.ei * system.length / (system.a * system.b) ** 2:,.0f} N/m"
            + (" = 48EI/L³" if abs(system.position - 0.5) < 1e-9 else f" (48EI/L³ = {ks:,.0f} N/m at midspan)")
        )
        self._analysis_timer.start()

    def _refresh_analysis(self) -> None:
        """The Campbell diagram and critical speeds for the current parameters."""
        system = self.sim.system
        self.campbell = campbell(system, MAX_RPM / RPM, n_modes=8)
        self.onset = stability_onset(system, MAX_RPM / RPM)
        iso = _isotropic(system)
        self.campbell_plot.set_result(self.campbell, MAX_RPM, iso, self.onset)
        self.stability_map.set_result(self.campbell, self.onset, MAX_RPM)
        crits = critical_speeds(self.campbell, iso)
        self.time.set_criticals([w * RPM for w, _ in crits])
        lines = [f"{w * RPM:,.0f} rpm ({w / (2 * math.pi):.4g} Hz), {whirl_name(r)} whirl" for w, r in crits]
        jeff = system.jeffcott_speed() * RPM
        self.crit_label.setText(
            "<b>Critical speeds</b> (where a whirl frequency equals the spin speed, up to "
            f"{MAX_RPM:,.0f} rpm):<br>" + ("<br>".join(lines) if lines else "none")
            + ("<br><small>The backward whirl crossings are not critical: an isotropic rotor's unbalance "
               "drives only forward whirl.</small>" if iso else "")
            + f"<br><small>Textbook Jeffcott estimate (massless journals, no tilt): {jeff:,.0f} rpm</small>"
            + f"<br><b>Stability:</b> {self._onset_text()}"
        )
        self.stability_note.setText(self._stability_note())
        self.campbell_note.setText(
            "Each dot is a damped natural frequency at that speed, coloured by its whirl direction "
            f"(modes damped more than ζ = {DAMPED:g}, here the journals bouncing on their supports, are "
            "hollow). A critical speed (circled) is where a lightly damped whirl frequency meets the 1X "
            "line: the unbalance, turning once per revolution, drives that mode at its natural "
            "frequency. An isotropic rotor's unbalance drives only its forward whirl."
            + (" Crosses mark growing modes (ζ < 0), above the onset of instability (dashed)." if self.onset else "")
        )
        self._matrices_speed = -1.0
        self._refresh_matrices()
        self._refresh_steady()

    def _onset_text(self) -> str:
        o = self.onset
        if o is None:
            return f"stable up to {MAX_RPM:,.0f} rpm."
        whirl = whirl_name(o.whirl)
        if o.omega == 0.0:
            return f"<b>unstable even at rest</b>: the {whirl} whirl at {o.freq_hz:.3g} Hz grows."
        return (f"<b>unstable above {o.omega * RPM:,.0f} rpm</b>, where the {whirl} whirl at "
                f"{o.freq_hz:.3g} Hz ({o.order:.2f}×) starts to grow.")

    def _stability_note(self) -> str:
        s = self.sim.system
        coupled = [f"k<sub>xy</sub> = {b.kxy:,.0f} N/m at {n}" for n, b in (("A", s.bearing_a), ("B", s.bearing_b))
                   if b.kxy]
        if s.internal_damping:
            coupled.append(f"internal damping c<sub>i</sub> = {s.internal_damping:g} N·s/m")
        cause = ("With " + " and ".join(coupled) + ": " if coupled else
                 "No cross-coupling and no internal damping: every mode stays damped, however fast it spins. "
                 "Set c<sub>i</sub> or k<sub>xy</sub> to see one go unstable. ")
        return (f"<small>{cause}{self._onset_text()} Modes damped more than ζ = {DAMPED:g} are left off. "
                "Below: the full spectrum of the strip chart's station.</small>")

    def _refresh_steady(self) -> None:
        """The steady-state 1X vectors at every station, finely around each critical speed."""
        w = [np.linspace(MAX_RPM / STEADY_POINTS, MAX_RPM, STEADY_POINTS) / RPM]
        crits = [c for c, _, _ in self.campbell.criticals] if self.campbell else []
        w += [np.linspace(0.8 * c, 1.2 * c, STEADY_POINTS // max(len(crits), 1)) for c in crits]
        w = np.unique(np.concatenate(w))
        self._steady_rpm = w * RPM
        self._steady_vectors = steady_vectors(self.sim.system, w)
        self._draw_tracking(full=True)

    def _refresh_matrices(self) -> None:
        rpm = self.sim.omega * RPM
        if self.tabs.currentWidget() is not self.matrices or abs(rpm - self._matrices_speed) < 25.0:
            return
        self._matrices_speed = rpm
        scroll = self.matrices.verticalScrollBar().value()
        self.matrices.setHtml(matrices_html(self.sim.system, self.sim.omega))
        self.matrices.verticalScrollBar().setValue(scroll)

    def apply_theme(self) -> None:
        self.view.apply_theme()
        self.orbits.apply_theme()
        self.time.apply_theme()
        self.campbell_plot.apply_theme()
        self.stability_map.apply_theme()
        self.spectrum.apply_theme()
        self.bode.apply_theme()
        self.polar.apply_theme()
        self.sweep_plots.apply_theme()
        self._draw_tracking(full=True)
        scroll = self.theory.verticalScrollBar().value()
        self.theory.setHtml(THEORY_HTML)
        self.theory.verticalScrollBar().setValue(scroll)
        self._matrices_speed = -1.0
        self._refresh_matrices()
        self._draw()

    # ------------------------------------------------------------- speed and actions
    def set_target_rpm(self, rpm: float) -> None:
        self.target_rpm.setValue(min(max(rpm, 0.0), MAX_RPM))

    def _on_target(self, rpm: float) -> None:
        self.stop_sweep(hold_speed=False)  # the user took over the speed
        self.sim.target = rpm / RPM
        self.speed_slider.blockSignals(True)
        self.speed_slider.setValue(round(rpm))
        self.speed_slider.blockSignals(False)
        self._stride = max(1, round(STORE_DT / self.sim.step_size()))

    def _on_ramp(self, rate: float) -> None:
        self.sim.accel = rate / RPM

    def _on_run(self) -> None:
        self.running = not self.running
        if self.running:
            self.rubbed = False
        self.run_button.setText("Pause" if self.running else "Run")
        if self.running:
            self._start_timer()

    def _on_same_scale(self, on: bool) -> None:
        self.orbits.same_scale = on
        self._draw()

    def reset(self) -> None:
        self.stop_sweep()
        self.sim.reset()
        self.rubbed = False
        self.history.clear()
        self._target_t = 0.0
        self._carry = 0
        self._draw()

    def tap(self) -> None:
        self.sim.tap(0.0, -TAP_IMPULSE)

    # ------------------------------------------------------------- run-up and coast-down
    def _on_sweep_button(self) -> None:
        if self.run_up is None:
            self.start_sweep()
        else:
            self.stop_sweep()

    def start_sweep(self) -> None:
        """Ramp to From, settle, run up to To and coast back down, tracking the 1X vector."""
        start, turn = self.sweep_from.value() / RPM, self.sweep_to.value() / RPM
        if start == turn:
            self.sweep_status.setText("Pick two different speeds.")
            return
        self.stop_sweep()
        if not self.hold_sweeps.isChecked():
            self.clear_sweeps()
        self.run_up = RunUpCoastDown(self.sim, start, turn)
        self._sync_target()
        self._sweep_pairs += 1
        self.sweep_button.setText("Stop sweep")
        self.ramp.setEnabled(False)  # each sweep is labelled with its rate
        if self.tabs.currentWidget() is not self.polar.parentWidget():
            self.tabs.setCurrentWidget(self.bode.parentWidget())
        self._fit_tracking_range()
        self._update_sweep_status()
        self._start_timer()

    def stop_sweep(self, hold_speed: bool = True) -> None:
        """Stop the sweep running (its legs so far stay on the plots), holding the speed it got to."""
        if self.run_up is None:
            return
        self.run_up = None
        self.sweep_button.setText("Run up and coast down")
        self.ramp.setEnabled(True)
        if hold_speed:
            self.sim.target = self.sim.omega
            self._sync_target()
        self._update_sweep_status()

    def clear_sweeps(self) -> None:
        self.stop_sweep()
        self.sweeps = []
        self._sweep_pairs = 0
        self.sweep_plots.clear()
        self._update_sweep_status()
        self._draw_tracking(full=True)

    def _sync_target(self) -> None:
        """Show the simulator's target speed (set by the sweep) in the speed box, without acting on it."""
        rpm = self.sim.target * RPM
        for w in (self.target_rpm, self.speed_slider):
            w.blockSignals(True)
        self.target_rpm.setValue(rpm)
        self.speed_slider.setValue(round(rpm))
        for w in (self.target_rpm, self.speed_slider):
            w.blockSignals(False)
        self._stride = max(1, round(STORE_DT / self.sim.step_size()))

    def _advance_sweep(self, duration: float):
        run_up = self.run_up
        before = len(run_up.sweeps)
        r = run_up.advance(duration)
        for sweep in run_up.sweeps[before:]:
            self.sweeps.append(sweep)
            self.sweep_plots.add(sweep, self._sweep_pairs - 1)
        if run_up.sim.target * RPM != self.target_rpm.value():
            self._sync_target()
        if run_up.done:
            self.run_up = None
            self.sweep_button.setText("Run up and coast down")
            self.ramp.setEnabled(True)
        self._update_sweep_status()
        return r

    def _update_sweep_status(self) -> None:
        r = self.run_up
        if r is None:
            n = len(self.sweeps)
            text = f"{n} sweep{'s' if n != 1 else ''} on the Bode and polar plots." if n else ""
        elif r.stage == "approach":
            text = f"Going to {r.start * RPM:,.0f} rpm…"
        elif r.stage == "settle":
            text = f"Settling at {r.start * RPM:,.0f} rpm…"
        else:
            leg = r.sweeps[-1]
            text = f"{leg.label}: {self.sim.omega * RPM:,.0f} rpm, {len(leg)} revolutions tracked"
        self.sweep_status.setText(text)

    def _on_probe(self) -> None:
        self._fit_tracking_range()
        self._draw_tracking(full=True)

    def _on_show_peaks(self, on: bool) -> None:
        self.sweep_plots.set_show_peaks(on)

    def _probe(self) -> tuple[int, str]:
        return self.probe_station.currentIndex(), DIRECTIONS[self.probe_direction.currentIndex()]

    def _tracking_range(self) -> tuple[float, float]:
        """Speeds (rpm) the Bode and polar plots show: the sweep's, and those of the sweeps on them."""
        speeds = [self.sweep_from.value(), self.sweep_to.value()]
        for sweep in self.sweeps:
            if len(sweep):
                speeds += [min(sweep.speed) * RPM, max(sweep.speed) * RPM]
        return min(speeds), max(speeds)

    def _fit_tracking_range(self) -> None:
        lo, hi = self._tracking_range()
        pad = 0.04 * max(hi - lo, 1.0)
        self.bode.amp.setXRange(max(lo - pad, 0.0), hi + pad, padding=0)
        self._draw_tracking(full=True)
        self.polar.getPlotItem().enableAutoRange()

    def _steady_probe(self) -> tuple[np.ndarray, np.ndarray]:
        """The steady-state 1X values at the chosen station and direction, and their unwrapped lag."""
        station, direction = self._probe()
        values = probe(self._steady_vectors[:, station], direction)
        return values, lag_degrees(values)

    def _draw_tracking(self, full: bool = False) -> None:
        """The Bode and polar plots: the steady-state curve and summary (if `full`), the sweeps, the marker."""
        if not self._steady_rpm.size:
            return
        rpm = self._steady_rpm
        values, lag = self._steady_probe()
        if full:
            self.bode.set_steady(rpm, values, lag)
            lo, hi = self._tracking_range()
            shown = (rpm >= lo) & (rpm <= hi)
            self.polar.set_steady(rpm[shown], values[shown], nice_marks(lo, hi))
        self.sweep_plots.redraw(*self._probe(), rpm, lag, full)
        here = self.sim.omega * RPM
        v = complex(np.interp(here, rpm, values.real), np.interp(here, rpm, values.imag))
        self.bode.set_here(here, abs(v), float(np.interp(here, rpm, lag)))
        self.polar.set_here(v)
        if full or (self.run_up is not None and self.run_up.stage in ("out", "back")):
            text = self._summary_html(rpm, values)
            if text != self.sweep_summary.text():
                self.sweep_summary.setText(text)

    def _summary_html(self, rpm: np.ndarray, values: np.ndarray) -> str:
        """The steady-state peak over the sweeps' speeds, and each sweep's peak against it."""
        station, direction = self._probe()
        lo, hi = self._tracking_range()
        inside = np.flatnonzero((rpm >= lo) & (rpm <= hi))
        if not inside.size:
            return ""
        i = inside[np.argmax(np.abs(values[inside]))]
        ss_amp, ss_rpm = float(abs(values[i])), float(rpm[i])
        wc = ss_rpm / RPM
        where = f"{STATIONS[station].lower()}, {DIRECTION_NAMES[self.probe_direction.currentIndex()].lower()}"
        head = f"<b>Steady state</b> ({where}): peak {_fmt_len(ss_amp)} at {ss_rpm:,.0f} rpm"
        zetas = [(abs(w - wc), z) for w, _, z in (self.campbell.criticals if self.campbell else [])]
        if zetas and min(zetas)[0] < 0.05 * wc:
            z = min(zetas)[1]
            head += f"; its mode has ζ = {z:.3g}, ζ² = {z * z:.2g}"
        if not self.sweeps:
            return head + ("<br><small>Run up and coast down to track the 1X response over this curve.</small>")
        rows = []
        for c in self.sweep_plots.curves:
            sweep = c.sweep
            amp, w = sweep.peak(station, direction)
            if not len(sweep):
                continue
            color = colors.mode[SWEEP_COLORS[c.color % len(SWEEP_COLORS)]]
            line = "━━" if sweep.up else "╍╍"
            alpha = sweep.rate / RPM
            rows.append(
                f"<tr><td><span style='color:{color}'><b>{line}</b></span> {sweep.label}</td>"
                f"<td align='right'>{alpha / wc**2:.2g}</td><td align='right'>{_fmt_len(amp)}</td>"
                f"<td align='right'>{100 * amp / ss_amp:.0f}%</td><td align='right'>{w * RPM:,.0f}</td>"
                f"<td align='right'>{(w - wc) * RPM:+,.0f}</td></tr>"
            )
        return (
            head + "<table cellspacing='0' cellpadding='2'><tr><th align='left'>Sweep</th>"
            "<th>α/ω<sub>c</sub>²</th><th>1X peak</th><th>of steady</th><th>at rpm</th><th>vs steady</th></tr>"
            + "".join(rows) + "</table>"
        )

    # ------------------------------------------------------------- loop
    def showEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().showEvent(event)
        self._start_timer()

    def hideEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().hideEvent(event)
        self._timer.stop()

    def _start_timer(self) -> None:
        if self.isVisible() and self.running and not self._timer.isActive():
            self._clock.restart()
            self._last = 0.0
            self._target_t = self.sim.t
            self._timer.start()

    def _frame(self) -> None:
        now = self._clock.elapsed() / 1000.0
        dt = min(now - self._last, 0.1)
        self._last = now
        self.step(dt * float(self.sim_speed.currentData()))
        if self.isVisible() and self.running:
            self._timer.start()

    def step(self, duration: float) -> None:
        """Advance by `duration` s of simulated time and redraw."""
        self._target_t = min(self._target_t + duration, self.sim.t + 0.1 + duration)
        if self.run_up is not None:
            r = self._advance_sweep(self._target_t - self.sim.t)
        else:
            r = self.sim.advance(self._target_t - self.sim.t)
        if r.t.size:
            # Keep every stride-th sample, continuing the stride across advances.
            first = (self._stride - self._carry - 1) % self._stride
            pick = np.arange(first, r.t.size, self._stride)
            self._carry = (r.t.size - 1 - pick[-1]) if pick.size else self._carry + r.t.size
            if pick.size:
                self.history.extend(r.t[pick], self._columns(r.q[pick], r.phase[pick], r.omega[pick]))
        if self.running and np.abs(self.sim.q[[XA, XD, XB, YA, YD, YB]]).max() > RUB * self.sim.system.length:
            self.stop_sweep()
            self._on_run()  # pause
            self.rubbed = True
        self._draw()
        self._refresh_matrices()

    def _columns(self, q: np.ndarray, phase: np.ndarray, omega: np.ndarray) -> np.ndarray:
        s = self.sim.system
        mid = s.deflection(q, [0.5 * s.length])[:, 0, :]
        return np.column_stack([q[:, XA], q[:, YA], q[:, XD], q[:, YD], mid, q[:, XB], q[:, YB], phase, omega])

    def _draw(self) -> None:
        sim = self.sim
        w = abs(sim.omega)
        trail_time = min(ORBIT_MAX_TIME, ORBIT_REVS * 2 * math.pi / w) if w > 0 else ORBIT_MAX_TIME
        _, y = self.history.window(trail_time)
        trails = [y[:, 2 * i : 2 * i + 2] for i in range(len(STATIONS))]
        keys = [np.empty((0, 2))] * len(STATIONS)
        if y.shape[0] > 1:
            turns = np.floor(y[:, PHASE] / (2 * math.pi))
            idx = np.flatnonzero(np.diff(turns) != 0) + 1
            keys = [t[idx] for t in trails]
        here = self._columns(sim.q[None, :], np.array([sim.phase]), np.array([sim.omega]))[0]
        peak = max([float(np.abs(t).max()) for t in trails if t.size] + [float(np.abs(here[:8]).max())])
        self.view.update_state(sim.q, sim.phase, trails, peak)
        steady = sim.omega == sim.target and w > 0
        r_disc = math.hypot(here[2], here[3])
        self.orbits.update_orbits(trails, keys, sim.phase if w > 0 else None)

        t, hist = self.history.window(self.window.value())
        st = self.station.currentIndex()
        self.time.update_curves(t, hist[:, 2 * st : 2 * st + 2], hist[:, OMEGA] * RPM, self.window.value())
        self.campbell_plot.set_speed(sim.omega * RPM)
        self.stability_map.set_speed(sim.omega * RPM)
        self._draw_tracking()
        self._frames += 1
        if self.tabs.currentWidget() is self.spectrum.parentWidget() and (
                self._frames % SPECTRUM_FRAMES == 0 or not self.running):
            self._draw_spectrum()

        # Readout.
        rpm = sim.omega * RPM
        parts = [f"<b>Speed {rpm:,.0f} rpm</b> ({sim.omega / (2 * math.pi):.3g} Hz)"]
        if not steady and sim.omega != sim.target:
            parts[0] += f" → {sim.target * RPM:,.0f} rpm"
        crits = [c for c, _ in critical_speeds(self.campbell, _isotropic(sim.system))] if self.campbell else []
        if crits and w > 0:
            parts.append(f"Ω / first critical = {sim.omega / crits[0]:.2f}")
        disc = trails[1]
        if disc.size:
            text = f"disc orbit up to {_fmt_len(float(np.hypot(disc[:, 0], disc[:, 1]).max()))} from centre"
            if w > 0 and disc.shape[0] > 2:
                # Signed area swept (x dy - y dx): positive turns x toward y, the way the shaft spins.
                swept = float(np.sum(disc[:-1, 0] * np.diff(disc[:, 1]) - disc[:-1, 1] * np.diff(disc[:, 0])))
                text += ", whirling " + ("forward" if swept > 0 else "backward")
            parts.append(text)
        growing = self.onset is not None and sim.omega >= self.onset.omega and least_damped(sim.system, w)[0] < 0
        if steady and not growing and r_disc > 1e-3 * max(sim.system.unbalance / sim.system.disc_mass, MIN_PEAK):
            lag = math.degrees(sim.phase - math.atan2(here[3], here[2])) % 360.0
            parts.append(f"heavy spot leads the high spot by {lag:.0f}°")
        if growing:
            zeta, f, _ = least_damped(sim.system, w)
            if w > 0:
                grow = -zeta * 2 * math.pi * f / math.sqrt(max(1 - zeta * zeta, 1e-12))
                parts.append(f"<span style='color:{colors.force}'>unstable: a whirl at {f:.3g} Hz "
                             f"({2 * math.pi * f / w:.2f}×) grows ×e every {1 / grow:.2g} s</span>")
        if self.rubbed:
            parts.append(
                f"<span style='color:{colors.force}'><b>Paused:</b> the whirl passed "
                f"{_fmt_len(RUB * sim.system.length)} ({RUB:.0%} of the span), where a real rotor would rub its "
                "seals or wreck its bearings. The model is linear, so it would grow for ever. Lower the target "
                "below the onset and Reset.</span>")
        self.readout.setText(" · ".join(parts))

    def _draw_spectrum(self) -> None:
        """The full spectrum at the strip chart's station over the strip chart's window."""
        t, hist = self.history.window(self.window.value())
        st = self.station.currentIndex()
        freqs, amps = full_spectrum(t, hist[:, 2 * st : 2 * st + 2])
        omega = self.sim.omega
        self.spectrum.update_spectrum(freqs, amps, omega, whirl_modes(self.sim.system, omega), DAMPED)
        self.spectrum_note.setText(
            f"Full spectrum of the {STATIONS[st].lower()} orbit over the last {self.window.value():g} s "
            "(the strip chart's station and window): forward whirl on the right, backward on the left.")


def lightly_damped(result: Campbell) -> list[tuple[float, float]]:
    """(speed in rad/s, whirl ratio) of the 1X crossings whose mode is damped less than DAMPED."""
    return [(w, r) for w, r, z in result.criticals if z <= DAMPED]


def critical_speeds(result: Campbell, isotropic: bool) -> list[tuple[float, float]]:
    """The 1X crossings the unbalance drives: lightly damped, and forward if the rotor is isotropic."""
    return [(w, r) for w, r in lightly_damped(result) if not isotropic or r > 0.1]


def _isotropic(system: RotorSystem) -> bool:
    return all(b.kx == b.ky and b.cx == b.cy for b in (system.bearing_a, system.bearing_b))


def _grouped(crits: list[tuple[float, float]]) -> list[tuple[float, list[str]]]:
    """Critical speeds (rpm) with the whirl names of those within 1% of each other merged."""
    out: list[tuple[float, list[str]]] = []
    for w, r in crits:
        rpm = w * RPM
        if out and rpm < 1.01 * out[-1][0]:
            out[-1][1].append(whirl_name(r))
        else:
            out.append((rpm, [whirl_name(r)]))
    return out


def _fmt_len(meters: float) -> str:
    if meters >= 1e-3:
        return f"{meters * 1e3:.3g} mm"
    if meters >= 1e-6:
        return f"{meters * 1e6:.3g} µm"
    return f"{meters * 1e9:.3g} nm"
