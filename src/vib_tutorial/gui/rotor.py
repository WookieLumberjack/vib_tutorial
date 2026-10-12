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

import cmath
import math
import time
from dataclasses import fields

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
    CRACK_DEPTH_MAX,
    MODE_LIMIT,
    STEEL,
    Bearing,
    Crack,
    PlaneWeight,
    Campbell,
    RotorSimulator,
    RotorSystem,
    campbell,
    steady_harmonics,
    whirl_modes,
)
from ..core.rotor_sweep import (
    DIRECTIONS,
    STATIONS,
    RunUpCoastDown,
    TrackedSweep,
    lag_degrees,
    probe,
    station_motion,
)
from ..core.fluid_bearing import JournalBearing
from ..core.magnetic_bearing import MagneticBearing
from ..core.rotor_balance import SteadyMeasurement
from ..core.rotor_presets import rotor_presets
from ..core.rotor_stability import AsymmetryChart, Onset, full_spectrum, least_damped, stability_onset
from ..core.rotor_waterfall import Waterfall
from .cms_time import SampleBuffer
from .panels import spin
from .rotor_balance import BalancingPanel
from .rotor_bearing import BearingPanel
from .rotor_magnet import MagnetPanel
from .rotor_bode import SWEEP_COLORS, BodePlots, PolarPlot, SweepPlots, nice_marks
from .rotor_figures import FigureBrowser
from .rotor_notes import THEORY_HTML, matrices_html, whirl_name
from .rotor_stability import MEASURES, AsymmetryPlot, FullSpectrum, StabilityMap
from .rotor_waterfall import AGAINST, SCALES, WaterfallPlot, against_speed, baselines
from .style import colors
from .theming import add_legend, mute

FRAME_MS = 16
SPEEDS = (0.1, 0.25, 0.5, 1.0, 2.0)  # simulation speed, × real time
TOP_RPM = 10_000.0  # the default speed range
MAX_TOP_RPM = 60_000.0  # the widest it can be set to
ORDERS = {2.0: "2X", 3.0: "3X", 0.5: "½X"}
CUSTOM = "Custom"  # the preset box's entry for a rotor that is none of the presets  # other orders of the speed the Campbell diagram can draw
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
SHAFT_DOT = 0.1  # display units: how far off the shaft's centre line a weight on it is drawn
CRACK_TICK = 0.22  # display units: the longest tick a crack is drawn as, from the shaft's centre line
GROUND = 1.1  # display units from the shaft's axis to the supports
HOUSING = 0.16  # display units: an oil-film bearing's housing, drawn round the bearing's axis
STATOR = 0.22  # display units: a magnetic bearing's stator ring, round the bearing's axis
SUPPORTS = ("Springs and dampers", "Oil-film journal bearings", "Magnetic bearings")
SPRINGS, FILMS, MAGNETS = range(len(SUPPORTS))
DAMPED = 0.3  # modes with more damping than this are drawn hollow and are not called critical
DASH = QtCore.Qt.PenStyle.DashLine
DOT = QtCore.Qt.PenStyle.DotLine
STEADY_POINTS = 2000  # speeds on the steady-state curve, plus as many again around the criticals
BEARING_WIDTHS = (56, 56, 50, 50, 46, 56, 40)  # px, the bearing table's columns: kx, ky, cx, cy, m, kxy, λ
BEARING_GAP = 0.02  # of the span: the closest the disc may be put to a bearing
RUB = 0.02  # of the span: a whirl this large pauses the simulation (a real rotor would rub)
GRAVITY = 9.81  # m/s²
SAG_SPEEDS = 201  # speeds the static sag is worked out at, with internal damping, and interpolated
SPECTRUM_FRAMES = 6  # the full spectrum is recomputed every this many frames
MEASURE_SPEED = 2.0  # × real time, at least, while a balancing measurement runs
FRAME_BUDGET = 0.012  # s of computing per frame: past it (the films' whole force is slow) the page runs slower
WATERFALL_FRAMES = 15  # new spectra are drawn at most every this many frames (each repaint fills every line)
WATERFALL_RANGE = 2.0  # the waterfall keeps frequencies up to this many times the top speed
CHART_BUDGET = 0.012  # s per frame spent filling in the asymmetry chart while it shows
CHART_DELTA = 0.5  # the asymmetry chart's top Δ (taller only for a rotor past it)
ORDER_NAMES = ("1X", "2X", "3X")  # what the Bode and polar plots can track

ORDER_TIPS = {
    2.0: "<p>Twice the speed. A shaft stiffer one way than the other (a two-pole rotor's slots, a "
         "keyway, a crack) bends twice per turn under its weight, and misalignment pulls twice per turn "
         "too. Where a forward branch meets this line, at about half its critical speed, a horizontal "
         "rotor with such a shaft whirls at 2X: a <i>secondary critical</i> (set an asymmetry Δ or a "
         "crack, and tick <i>Horizontal rotor</i>).</p>",
    3.0: "<p>Three times the speed. A breathing crack, opened and closed by the shaft's weight once per "
         "turn, changes the stiffness with 3X in it too, so the weight drives a 3X whirl where a forward "
         "branch meets this line, at about a third of the critical speed. An asymmetric shaft's stiffness "
         "changes only at 2X: no 3X.</p>",
    0.5: "<p>Half the speed. Oil in a plain journal bearing swirls at just under half the shaft's speed, "
         "so its whirl starts near ½X (<i>oil whirl</i>) and, once ½X has passed the first natural "
         "frequency, locks onto it (<i>oil whip</i>): set the bearings' swirl ratio λ to 0.5. A rub "
         "or loose parts can also respond at ½X.</p>",
}
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
REPEAT_TAP_TIP = (
    "<p>Tap the disc again and again as it runs, as in an impact test at speed. On the <i>Waterfall</i> "
    "tab each tap rings the free whirl frequencies at the speed it came at, so over a run-up the peaks "
    "trace the Campbell diagram's branches.</p>"
)
BLOCK_TIP = (
    "<p>How long each spectrum's block of signal is. The frequency resolution is 1 / block: 0.5 s "
    "resolves 2 Hz. But while the speed ramps, it changes over a block, smearing the 1X peak: at "
    "500 rpm/s, by 250 rpm (4 Hz) over 0.5 s. Each block overlaps the last by half.</p>"
)
VIEW_TIP = (
    "<p>The shaft's centre line, the disc and the bearings' springs and dampers, with the whirl "
    "magnified (the scale is in the top corner). Drag to turn the view; double-click to reset it.</p>"
    "<p>The red dot on the disc is the heavy spot, where its unbalance is: it turns with the shaft. The "
    "tick across the rim is the keyphasor mark that angles are measured from, and squares are balance "
    "weights. An asymmetric shaft's stiff axis is the dashed diameter across the disc; a crack is the "
    "dark tick on the shaft, pointing at its mouth and thick while it is open. "
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
SHAFT_MASS_TIP = (
    "<p>The Jeffcott model's shaft has no mass: it is a spring holding the disc. Tick this to give it "
    "its mass (and its sections' rotary inertia and gyroscopic moment), spread along it: it then "
    "becomes a beam finite element model, and has bending modes of its own.</p>"
    "<p>A shaft light beside its disc changes little: the first critical speed drops a little, as if "
    "part of the shaft's mass sat on the disc. Make the shaft long and the disc light and the "
    "massless model fails: the shaft's own bending modes come down into the speed range.</p>"
)
ELEMENTS_TIP = (
    "<p>How many beam elements model the shaft, shared between its stretches by length. Each has a "
    "displacement and a slope at both ends, bending as a cubic in between. More elements, closer to "
    "the true beam's modes.</p>"
    "<p><i>Consistent</i> mass spreads each element's mass with the same cubic shapes as its "
    "stiffness. It is a Rayleigh–Ritz model, so at rest and undamped its frequencies are never too "
    "low, and come down toward the true ones as elements are added. <i>Lumped</i> puts half of each "
    "element's mass at each end. It is simpler, but it has no such bound: with few elements much of the "
    "mass sits on the bearings, where it hardly moves, and the frequencies come out high.</p>"
    "<p>The stations are kept as they are; the rest of the shaft follows them, plus its own modes "
    "with the stations held, up to " + f"{MODE_LIMIT:,.0f}" + " Hz (a Craig–Bampton reduction). The "
    "modes far above those would only slow the simulation down.</p>"
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
    "<p>A shaft bent for good (by a thermal bow, a sag left after standing, or a bent repair), by δ<sub>b</sub> at "
    "the disc, toward an angle measured from the keyphasor mark in the direction of spin. It turns with the "
    "shaft, so it drives the rotor once per revolution like an unbalance, but with a force kδ<sub>b</sub> that does "
    "not grow with speed.</p><p>Turning slowly, the probes read the bow itself (the <i>slow-roll "
    "runout</i>), and the polar plot starts from it rather than from the origin. Well above the critical "
    "speed the disc stays on the bearings' axis and the shaft bends round it. Put the bow at 180° to the "
    "heavy spot and the two cancel at one speed.</p>"
)
SKEW_TIP = (
    "<p>The disc mounted out of square: its own axis tilted by τ from the shaft's, toward an angle "
    "measured from the keyphasor mark in the direction of spin. Spinning, the tilted disc's inertia makes a "
    "moment (I<sub>d</sub> − I<sub>p</sub>) τ Ω² that turns with the shaft: a <i>couple unbalance</i>. "
    "It drives the disc's tilt directly, and pushes a thin disc (I<sub>p</sub> &gt; I<sub>d</sub>) toward "
    "square to the spin axis, bending the shaft at the disc by −τ at high speed.</p><p>At midspan on "
    "identical bearings it only tilts the disc; move the disc off centre and it moves it too.</p>"
)
ASYMMETRY_TIP = (
    "<p>A shaft stiffer about one axis than the other, as the slots of a two-pole generator's rotor make "
    "it (or a keyway, or a crack): bending toward its stiff axis it has EI(1 + Δ), across it EI(1 − Δ). "
    "The angle is the stiff axis's from the keyphasor mark, in the direction of spin.</p>"
    "<p>Its stiffness turns with it, so it has two critical speeds, one per axis, and between them it is "
    "unstable unless the damping holds it: the whirl grows locked to 1X. Laid horizontal, its weight acts "
    "on a stiffness that changes twice per turn and drives a 2X whirl, largest near half the critical "
    "speed (the <i>gravity critical</i>; plot 2X on the Bode plot). The 1X response depends on where the "
    "heavy spot sits from the stiff axis.</p>"
)
CRACK_TIP = (
    "<p>A transverse crack in the shaft, cut to a depth a<sub>c</sub> from the surface (a<sub>c</sub>/d of the "
    "shaft's diameter), its mouth at the angle "
    "given from the keyphasor mark. Open, it is a hinge: the shaft's slope jumps across it, much more "
    "under bending that pulls its mouth open than across (its compliance comes from fracture "
    "mechanics, for depths up to half the shaft).</p>"
    "<p><i>Breathing</i>, it is opened and closed by the shaft's weight as it turns (tick "
    "<i>Horizontal rotor</i>): fully open with its mouth on the side the sag stretches, shut half a turn "
    "later. That stiffness changes with 1X, 2X and 3X in it, so the weight drives 2X and 3X whirls, "
    "largest near a half and a third of the critical speed, and changes the 1X. Unticked, it is held "
    "open (a notch): then it is an asymmetric shaft, softened.</p>"
    "<p>The position is z/L from bearing A, as the disc's.</p>"
)
GRAVITY_TIP = (
    "<p>A horizontal rotor: gravity pulls the disc and the journals down. The shaft sags and the "
    "supports give, so the rotor whirls about a point below the bearings' centre. The model is linear, "
    "so the vibration about that point is just as without gravity: on these supports gravity alone "
    "makes no whirl. With cross-coupled bearings or internal damping the sag also goes sideways.</p>"
    "<p>An asymmetric or cracked shaft is the exception: its stiffness turns under its weight, so "
    "gravity drives it at 2X (and a breathing crack at 1X and 3X too). Without gravity nothing opens a "
    "breathing crack.</p>"
)
SAG_TIP = (
    "<p>Draw the motion with the static sag, as a DC-coupled probe reads the shaft's position in its "
    "bearing: the orbits sit below the centre. Untick to draw only the vibration about where gravity "
    "holds the rotor, as an AC-coupled probe does. The tracking, the spectrum and the phase are always "
    "of the vibration.</p>"
)
FILM_TIP = (
    "<p>Run both journals in plain journal bearings, oil films on rigid housings, in place of the springs "
    "and dampers: the bearing set on the <i>Fluid-film bearing</i> tab, each carrying its share of the "
    "rotor's weight (none on a vertical rotor). Their stiffness and damping change with the speed as the "
    "film lifts the journal, and their cross-coupling, the oil swirling at half the speed, makes oil whirl "
    "and whip from the bearing itself rather than from λ.</p><p>Laid horizontal, the journals sit off "
    "centre where the film holds them, and rise along the bearing's locus as the speed rises: tick "
    "<i>Static sag</i> to see them move. Only the journal's mass is left to set.</p><p>Upright, unloaded, "
    "a plain bearing has no direct stiffness, only the swirl's cross-coupling: a vertical rotor in such "
    "bearings whirls at any speed.</p>"
)
MAGNET_TIP = (
    "<p>Float both journals in active magnetic bearings, set on the <i>Magnetic bearing</i> tab: on each "
    "axis a pair of electromagnets, whose pull grows as the journal comes nearer (a negative stiffness), "
    "and a PID controller that sets their current from the journal's position. The bearing's stiffness "
    "and damping come from the gains, and change with the frequency: the controller's lag makes the "
    "damping negative at high frequencies, and a mode that gets there grows.</p><p>With the integral, "
    "gravity cannot move the journals: they stay centred, and the shaft alone sags. Only the journal's "
    "mass is left to set.</p>"
)
NONLINEAR_FILM_TIP = (
    "<p>Step the rotor with the films' whole force, not their linearization about where the journals rest. "
    "Off centre the film stiffens and damps ever harder toward the wall, so the journals stay inside their "
    "clearance: through a critical speed, and in oil whirl, which settles to a limit cycle at half the "
    "speed. In oil whip the shaft's whirl still grows, the films holding the journals but not the shaft's "
    "bending, until it would rub.</p><p>Only the time response: the Campbell diagram, stability, Bode "
    "and polar curves are always of the linearized films. Untick to step the linear films, as they "
    "predict. A shaft that turns its stiffness (asymmetric or cracked) keeps the films linear. The whole "
    "force is slow to step: past what a frame can compute, the page runs slower than the speed set.</p>"
)
SWIRL_TIP = (
    "<p>Fluid swirl ratio λ of a fluid-film bearing: the oil in the clearance is dragged round at λ times the "
    "shaft's speed: ½ in short-bearing theory, just under it (0.42 to 0.48) in real plain bearings. It makes a cross-coupled stiffness that grows "
    "with speed, k<sub>xy</sub> = λcΩ on top of the constant one (c the bearing's damping). Once the swirl "
    "λΩ outruns a forward whirl's frequency, the oil feeds it faster than c damps it: with λ = 0.5 the "
    "rotor goes unstable at twice its first natural frequency (<i>oil whip</i>).</p>"
)
ORBIT_TIP = (
    "<p>The path of the shaft's centre at each station over the last few revolutions, looking from "
    "the B end toward A: x to the right, y up, and the shaft spinning counterclockwise (x toward y). "
    "Each plot has its own scale unless <i>Same scale</i> is ticked.</p>"
    "<p>The small hollow circles are once-per-revolution marks (a keyphasor): where the centre was "
    "each time the mark on the shaft passed +x (the heavy spot is at the mark unless you move it). On "
    "the disc's orbit the red line points at the heavy spot now; the angle from the high spot (where the disc is) to the heavy spot is the phase lag, "
    "0° well below the critical speed, about 90° near it and 180° well above.</p>"
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
        self.weight_dots = pg.ScatterPlotItem(size=7, symbol="s")  # balance weights, on the disc or the shaft
        self.mark = pg.PlotCurveItem()  # the keyphasor mark: a tick across the disc's rim
        self.stiff_axis = pg.PlotCurveItem()  # an asymmetric shaft's stiff axis: a diameter across the disc
        self.crack_mark = pg.PlotCurveItem()  # a crack: a tick on the shaft toward its mouth
        self.show_heavy = True
        self.stations = pg.ScatterPlotItem(size=8)
        self.triad = pg.PlotCurveItem(connect="pairs")
        self.triad_labels = [pg.TextItem(t, anchor=(0.5, 0.5)) for t in ("x", "y", "z")]
        self.end_labels = [pg.TextItem(t, anchor=(0.5, 0.0)) for t in ("A", "B")]
        for item in (self.frame, self.axis, self.supports, *self.trails, self.shaft, self.disc,
                     self.spoke, self.stiff_axis, self.crack_mark, self.heavy, self.weight_dots, self.mark, self.stations, self.triad, *self.triad_labels, *self.end_labels):
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

    @property
    def _metres(self) -> float:
        """Metres of shaft per display unit along it: the whole shaft, overhang too, spans AXIAL."""
        z0, z1 = self.system.extent
        return (z1 - z0) / AXIAL

    def _z(self, z_m: float | np.ndarray) -> np.ndarray:
        """Display axial coordinate (from the shaft's middle) of an axial position in metres."""
        z0, z1 = self.system.extent
        return (np.asarray(z_m) - 0.5 * (z0 + z1)) / self._metres

    def set_system(self, system: RotorSystem) -> None:
        self.system = system
        r = math.sqrt(2.0 * system.ip / system.disc_mass) if system.ip > 0 else 0.05
        self.disc_radius = float(np.clip(r / self._metres, 0.2, 0.9))
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
        bearings = self._z(np.array([0.0, self.system.length]))
        for zb in bearings:
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
        for label, zb in zip(self.end_labels, bearings):
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
        self.weight_dots.setBrush(pg.mkBrush(colors.strong))
        self.weight_dots.setPen(pg.mkPen(colors.background, width=1))
        self.mark.setPen(pg.mkPen(colors.strong, width=2))
        self.stiff_axis.setPen(pg.mkPen(colors.strong, width=1.5, style=DASH))
        self.stations.setPen(pg.mkPen(colors.background, width=1))
        self.triad.setPen(pg.mkPen(colors.muted, width=1))
        for label in (*self.triad_labels, *self.end_labels, self.scale_label):
            label.setColor(colors.muted)

    # ------------------------------------------------------------- drawing
    def update_state(self, q: np.ndarray, phase: float, trails: list[np.ndarray], peak: float) -> None:
        """Draw displacements q (n_dof,) with the keyphasor mark at angle `phase`, and each station's recent (x, y)."""
        s = self.system
        # The disc's own axis: the shaft's slope there plus the skew. Its rim's wobble counts as motion.
        axis = np.array([q[2], q[6]]) + s.skew * np.array([math.cos(phase + s.skew_angle),
                                                           math.sin(phase + s.skew_angle)])
        peak = max(peak, float(np.hypot(*axis)) * self.disc_radius * self._metres)
        target = SWING / max(peak, MIN_PEAK)
        # Shrink at once (keep the motion on screen), grow slowly (no flicker as it settles).
        self.gain = target if target < self.gain else self.gain + 0.08 * (target - self.gain)
        g = self.gain
        self.scale_label.setText(f"motion ×{_round_gain(g * self._metres)}")

        # Shaft centre line.
        z = np.union1d(np.linspace(*s.extent, 81), s.nodes)
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
        scale = g * self._metres  # display slope per radian of real slope
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

        def on_disc(angle: float, r: float) -> np.ndarray:
            return c + r * self.disc_radius * (math.cos(angle) * u + math.sin(angle) * v)

        net = s.mass_unbalance
        if self.show_heavy and net:
            spot = on_disc(phase + math.atan2(net.imag, net.real), 0.8)
            self.spoke.setData(*self.project(np.vstack([c, spot])))
            self.heavy.setData(*self.project(spot[None, :]))
        else:
            self.spoke.setData([], [])
            self.heavy.setData([], [])
        dots = [on_disc(phase + cmath.phase(w), 0.92) for w in s.weights]
        unit = n / np.linalg.norm(n)
        for p in s.plane_weights:
            if s.on_disc(p):  # on the disc, at its axial offset
                dots.append(on_disc(phase + cmath.phase(p.weight), 0.92) + unit * (p.z - s.a) / self._metres)
            else:  # on the shaft, just off its centre line
                angle = phase + cmath.phase(p.weight)
                centre = s.deflection(q, [p.z])[0] * g
                dots.append([centre[0] + SHAFT_DOT * math.cos(angle), centre[1] + SHAFT_DOT * math.sin(angle),
                             float(self._z(p.z))])
        if dots:
            self.weight_dots.setData(*self.project(np.array(dots)))
        else:
            self.weight_dots.setData([], [])
        self.mark.setData(*self.project(np.vstack([on_disc(phase, 0.88), on_disc(phase, 1.12)])))
        if s.asymmetry:
            angle = phase + s.asymmetry_angle
            self.stiff_axis.setData(*self.project(np.vstack([on_disc(angle, 0.75), on_disc(angle + math.pi, 0.75)])))
        else:
            self.stiff_axis.setData([], [])
        if s.crack is not None and s.crack.depth:
            angle = phase + s.crack.angle
            centre = s.deflection(q, [s.crack_z])[0] * g
            reach = CRACK_TICK * (0.4 + 0.6 * s.crack.depth / CRACK_DEPTH_MAX)
            tick = np.array([[centre[0], centre[1]], [centre[0] + reach * math.cos(angle),
                                                      centre[1] + reach * math.sin(angle)]])
            self.crack_mark.setData(*self.project(np.column_stack([tick, np.full(2, self._z(s.crack_z))])))
            self.crack_mark.setPen(pg.mkPen(colors.strong, width=1.5 + 3.0 * float(s.crack_open(phase))))
        else:
            self.crack_mark.setData([], [])

        # Supports: a spring and a damper to ground below (y) and to the side wall (x) at each bearing;
        # or an oil film's housing round the bearing's axis, on a pedestal; or a magnetic bearing's
        # stator ring with its four poles (a magnet pair on each axis), on a pedestal.
        parts = []
        ring = np.linspace(0.0, 2.0 * math.pi, 49)
        for station, bearing in ((0, s.bearing_a), (3, s.bearing_b)):
            j = here[station]
            if bearing.magnet is not None:
                zb = np.full_like(ring, j[2])
                parts.append(np.column_stack([STATOR * np.cos(ring), STATOR * np.sin(ring), zb]))
                for a in np.arange(4) * 0.5 * math.pi:
                    for side in (-0.25, 0.25):  # each pole a pair of lines toward the journal
                        u = np.array([math.cos(a + side * 0.6), math.sin(a + side * 0.6)])
                        parts.append(np.array([[*(STATOR * u), j[2]], [*(0.55 * STATOR * u), j[2]]]))
                parts.append(np.array([[0.0, -STATOR, j[2]], [0.0, -GROUND, j[2]]]))
                continue
            if bearing.film is not None:
                zb = np.full_like(ring, j[2])
                parts.append(np.column_stack([HOUSING * np.cos(ring), HOUSING * np.sin(ring), zb]))
                parts.append(np.array([[0.0, -HOUSING, j[2]], [0.0, -GROUND, j[2]]]))
                continue
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

    def update_orbits(self, trails: list[np.ndarray], keys: list[np.ndarray], heavy_angle: float | None,
                      centre: tuple[float, float] = (0.0, 0.0)) -> None:
        """Each station's trail (k, 2), its keyphasor points, and the heavy spot's angle now, drawn from
        `centre` on the disc's orbit (where it whirls about: its sag, if drawn)."""
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
            x0, y0 = centre
            self.heavy.setData([x0, x0 + r * math.cos(heavy_angle)], [y0, y0 + r * math.sin(heavy_angle)])


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
        # The same rotor with a massless shaft, when its shaft has mass: one curve per branch.
        self.reference: list[pg.PlotDataItem] = []
        self.reference_sample = pg.PlotDataItem(name="massless shaft (Jeffcott)")
        # Other orders of the speed, drawn on request: lines but not in the legend, labelled at their ends.
        self.orders = {order: p.plot() for order in ORDERS}
        self.order_labels = {order: pg.TextItem(f"{name}", anchor=(1.1, 1.0)) for order, name in ORDERS.items()}
        for order in ORDERS:
            self.orders[order].setVisible(False)
            self.order_labels[order].setVisible(False)
            p.addItem(self.order_labels[order])
        self.uncrit = pg.ScatterPlotItem(size=11, symbol="o", name="backward crossing, not driven")
        p.addItem(self.uncrit)
        self.crit = pg.ScatterPlotItem(size=11, symbol="o", name="critical speed")
        p.addItem(self.crit)
        self.crit_labels: list[pg.TextItem] = []
        self.unstable = pg.ScatterPlotItem(size=7, symbol="x", name="growing (ζ &lt; 0)")
        p.addItem(self.unstable)
        self.onset_line = pg.InfiniteLine(angle=90)
        self.end_line = pg.InfiniteLine(angle=90)  # where an unstable band ends (an asymmetric shaft's)
        self.onset_label = pg.TextItem(anchor=(1.05, 0.0))
        p.addItem(self.onset_line)
        p.addItem(self.end_line)
        p.addItem(self.onset_label)
        self.speed_line = pg.InfiniteLine(angle=90, movable=True)
        self.speed_line.setToolTip("The speed now. Drag it to set the target speed.")
        self.speed_line.sigPositionChangeFinished.connect(lambda line: self.speed_dragged.emit(max(0.0, line.value())))
        p.addItem(self.speed_line)
        for item in (*self.points.values(), self.damped, self.sync, self.crit, self.uncrit, self.unstable):
            self.legend.addItem(item, item.name())
        self.result: Campbell | None = None
        self._reference_shown = False
        self.apply_theme()

    def apply_theme(self) -> None:
        brushes = {"fw": colors.mode[1], "bw": colors.mode[3], "planar": colors.grey}
        for key, item in self.points.items():
            item.setBrush(pg.mkBrush(brushes[key]))
            item.setPen(pg.mkPen(None))
        self.damped.setBrush(pg.mkBrush(None))
        self.damped.setPen(pg.mkPen(colors.grey, width=1))
        self.sync.setPen(pg.mkPen(colors.strong, width=1, style=DASH))
        for line in (*self.reference, self.reference_sample):
            line.setPen(pg.mkPen(colors.muted, width=1.5, style=DASH))
        for order, line in self.orders.items():
            line.setPen(pg.mkPen(colors.muted, width=1, style=DOT))
            self.order_labels[order].setColor(colors.muted)
        self.crit.setBrush(pg.mkBrush(None))
        self.crit.setPen(pg.mkPen(colors.force, width=2))
        self.uncrit.setBrush(pg.mkBrush(None))
        self.uncrit.setPen(pg.mkPen(colors.grey, width=2))
        self.speed_line.setPen(pg.mkPen(colors.force, width=2))
        self.speed_line.setHoverPen(pg.mkPen(colors.force, width=4))
        self.unstable.setBrush(pg.mkBrush(colors.force))
        self.unstable.setPen(pg.mkPen(None))
        self.onset_line.setPen(pg.mkPen(colors.force, width=1.5, style=DASH))
        self.end_line.setPen(pg.mkPen(colors.force, width=1.5, style=DASH))
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
        self.end_line.setVisible(onset is not None and math.isfinite(onset.end))
        if onset is not None and math.isfinite(onset.end):
            self.onset_line.setValue(onset.omega * RPM)
            self.end_line.setValue(onset.end * RPM)
            self.onset_label.setText(f"unstable\n{onset.omega * RPM:,.0f}–{onset.end * RPM:,.0f} rpm")
        elif onset is not None:
            self.onset_line.setValue(onset.omega * RPM)
            self.onset_label.setText(f"instability onset\n{onset.omega * RPM:,.0f} rpm")
        else:
            self.onset_label.setText("")
        self.sync.setData([0.0, max_rpm], [0.0, max_rpm / 60.0])
        for order, line in self.orders.items():
            line.setData([0.0, max_rpm], [0.0, order * max_rpm / 60.0])
        p = self.plot
        for label in self.crit_labels:
            p.removeItem(label)
        driven = critical_speeds(result, isotropic)
        crits = [w * RPM for w, _ in driven]
        self.crit.setData(crits, [c / 60.0 for c in crits])
        others = [w * RPM for w, r in lightly_damped(result) if (w, r) not in driven]
        self.uncrit.setData(others, [c / 60.0 for c in others])
        self.crit_labels = []
        groups = _grouped(critical_speeds(result, isotropic))
        for i, (rpm, names) in enumerate(groups):
            # Right of the circle and above it, unless the next one is close: then below it.
            crowded = i + 1 < len(groups) and groups[i + 1][0] - rpm < 0.3 * max_rpm
            label = pg.TextItem(f"{rpm:,.0f} rpm ({' + '.join(names)})",
                                anchor=(-0.05, -0.1) if crowded else (-0.05, 1.1))
            label.setPos(rpm, rpm / 60.0)
            p.addItem(label)
            self.crit_labels.append(label)
        top = np.nanmax(f[ok & (f < 2.5 * max_rpm / 60.0)]) if np.any(ok & (f < 2.5 * max_rpm / 60.0)) else 1.0
        y_top = 1.1 * max(top, max_rpm / 60.0)
        p.setRange(xRange=(0.0, max_rpm), yRange=(0.0, y_top), padding=0)
        for order, label in self.order_labels.items():
            end = min(max_rpm, 60.0 * y_top / order)  # where the line leaves the plot
            label.setPos(end, order * end / 60.0)
            label.setAnchor((1.1, 0.0) if end < max_rpm else (1.1, 1.0))  # out the top: hang below
        if onset is not None:
            self.onset_label.setAnchor((1.05, 0.0) if onset.omega * RPM > 0.5 * max_rpm else (-0.05, 0.0))
            self.onset_label.setPos(onset.omega * RPM, 0.97 * y_top)
        self.apply_theme()

    def set_reference(self, result: Campbell | None) -> None:
        """Draw another Campbell diagram's lightly damped branches as dashed curves (None: none)."""
        p = self.plot
        branches = [] if result is None else [j for j in range(result.freq_hz.shape[1])
                                              if np.any(result.zeta[:, j] <= DAMPED)]
        while len(self.reference) < len(branches):
            line = p.plot()
            line.setZValue(-1)
            self.reference.append(line)
        for i, line in enumerate(self.reference):
            if i < len(branches):
                j = branches[i]
                f = np.where(result.zeta[:, j] <= DAMPED, result.freq_hz[:, j], np.nan)
                line.setData(result.omega * RPM, f, connect="finite")
            line.setVisible(i < len(branches))
        if (result is not None) != self._reference_shown:
            if result is not None:
                self.legend.addItem(self.reference_sample, self.reference_sample.name())
            else:
                self.legend.removeItem(self.reference_sample)
            self._reference_shown = result is not None
        self.apply_theme()

    def set_orders(self, shown: set[float]) -> None:
        """Show the lines of these orders of the speed (keys of ORDERS)."""
        for order in ORDERS:
            self.orders[order].setVisible(order in shown)
            self.order_labels[order].setVisible(order in shown)

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
        self.top_rpm = TOP_RPM  # the speed range: Campbell diagram, stability map, Bode plot, speed controls
        self.running = True
        self._stride = 1
        self._carry = 0  # steps since the last stored sample
        self._matrices_speed = -math.inf  # rpm the matrices were drawn at
        self.run_up: RunUpCoastDown | None = None  # the sweep running, if any
        self.sweeps: list[TrackedSweep] = []  # every leg on the plots, the running one's too
        self._sweep_pairs = 0  # sweeps started since the plots were cleared (each pair's colour)
        self._steady_rpm = np.empty(0)  # the steady-state curve: speeds (rpm)
        self._steady_vectors = np.empty((0, len(STATIONS), 2), dtype=complex)  # and 1X vectors there
        self._steady_orders = [self._steady_vectors] * len(ORDER_NAMES)  # and the 1X, 2X and 3X vectors
        self.asym_chart: AsymmetryChart | None = None  # the stability tab's asymmetry chart, as far as it got
        self.onset: Onset | None = None  # where the rotor goes unstable, if it does up to top_rpm
        self.rubbed = False  # the whirl grew past RUB of the span and the simulation paused
        self.touched: tuple[str, float] | None = None  # or a journal reached its film's clearance: which, how far
        self._sag: tuple[np.ndarray, np.ndarray] | None = None  # see _set_sag
        self._weights: tuple[complex, ...] = ()  # balance weights on the disc (kg·m)
        self._plane_weights: tuple[PlaneWeight, ...] = ()  # weights in balancing planes
        self._hidden: dict[str, float] | None = None  # RotorSystem fields kept out of the panels (SI)
        self.measurement: SteadyMeasurement | None = None  # a balancing measurement running, if any
        self._measured = None  # its callback, given the mean 1X vectors (4, 2)
        self._frames = 0
        self._next_tap = 0.0  # simulated time of the next repeated tap
        self.waterfall: Waterfall | None = None  # spectra block by block, made by _new_waterfall
        self._waterfall_drawn = False  # the waterfall tab shows the latest blocks

        # --- parameters
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.addWidget(self._preset_box())
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
        self.show_sag = QtWidgets.QCheckBox("Static sag")
        self.show_sag.setChecked(True)
        self.show_sag.setEnabled(False)
        self.show_sag.setToolTip(SAG_TIP)
        self.show_sag.toggled.connect(lambda _: self._draw())
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Orbits, seen from the B end:"))
        row.addWidget(self.same_scale)
        row.addWidget(self.show_sag)
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
        self.top_speed = spin(1000.0, MAX_TOP_RPM, TOP_RPM, 0, " rpm")
        self.top_speed.setSingleStep(1000.0)
        self.top_speed.setToolTip(
            "<p>The highest speed on the Campbell diagram, the stability map and the Bode plot, and the top "
            "of the speed controls.</p>")
        self.top_speed.valueChanged.connect(self.set_top_rpm)
        self.order_boxes = {}
        order_row = QtWidgets.QHBoxLayout()
        order_row.addWidget(QtWidgets.QLabel("Speeds up to:"))
        order_row.addWidget(self.top_speed)
        order_row.addSpacing(12)
        order_row.addWidget(QtWidgets.QLabel("Also draw:"))
        for order, name in ORDERS.items():
            box_ = QtWidgets.QCheckBox(name)
            box_.setToolTip(ORDER_TIPS[order])
            box_.toggled.connect(self._on_orders)
            order_row.addWidget(box_)
            self.order_boxes[order] = box_
        order_row.addStretch(1)
        campbell_tab = QtWidgets.QWidget()
        ct = QtWidgets.QVBoxLayout(campbell_tab)
        ct.addLayout(order_row)
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
        self.asymmetry_plot = AsymmetryPlot()
        self.measure.currentIndexChanged.connect(self._on_measure)
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
        self.stability_stack = QtWidgets.QStackedWidget()
        self.stability_stack.addWidget(self.stability_map)
        self.stability_stack.addWidget(self.asymmetry_plot)
        st.addWidget(self.stability_stack, 3)
        st.addWidget(self.stability_note)
        st.addWidget(self.spectrum, 2)
        st.addWidget(self.spectrum_note)
        self.waterfall_plot = WaterfallPlot()
        self.wf_station = QtWidgets.QComboBox()
        self.wf_station.addItems(STATIONS)
        self.wf_station.setCurrentIndex(1)
        self.wf_against = QtWidgets.QComboBox()
        self.wf_against.addItems(AGAINST)
        self.wf_against.setToolTip(
            "<p>Against speed (a <i>cascade</i>): the latest run-up's or coast-down's spectra, each at its "
            "speed. Against time (a <i>waterfall</i>): every spectrum kept, steady running and taps "
            "included.</p>")
        self.wf_scale = QtWidgets.QComboBox()
        self.wf_scale.addItems(SCALES)
        self.wf_scale.setToolTip(
            "<p>Log: each line's height is its amplitude in decades above a floor 1000 times (60 dB) "
            "below the tallest peak, so small peaks show beside the 1X one. Linear: heights in proportion "
            "to the amplitude, as most cascades are drawn.</p>")
        self.wf_block = spin(0.1, 2.0, 0.5, 2, " s")
        self.wf_block.setSingleStep(0.1)
        self.wf_block.setToolTip(BLOCK_TIP)
        self.wf_block.valueChanged.connect(lambda _: self._new_waterfall())
        self.wf_marks = QtWidgets.QCheckBox("1X and free whirl")
        self.wf_marks.setChecked(True)
        self.wf_marks.setToolTip("Mark 1X and the free whirl frequencies (from the Campbell diagram) under the lines")
        for c in (self.wf_station, self.wf_against, self.wf_scale):
            c.currentIndexChanged.connect(lambda _: self._draw_waterfall())
        self.wf_marks.toggled.connect(lambda _: self._draw_waterfall())
        self.wf_clear = QtWidgets.QPushButton("Clear")
        self.wf_clear.setToolTip("Drop every spectrum taken so far")
        self.wf_clear.clicked.connect(self._new_waterfall)
        self.waterfall_note = QtWidgets.QLabel()
        self.waterfall_note.setWordWrap(True)
        mute(self.waterfall_note)
        waterfall_tab = QtWidgets.QWidget()
        wt = QtWidgets.QVBoxLayout(waterfall_tab)
        for widgets in ((QtWidgets.QLabel("Orbit at:"), self.wf_station, QtWidgets.QLabel("Against:"),
                         self.wf_against),
                        (QtWidgets.QLabel("Amplitude:"), self.wf_scale, QtWidgets.QLabel("Block:"), self.wf_block,
                         self.wf_marks, self.wf_clear)):
            row = QtWidgets.QHBoxLayout()
            for w_ in widgets:
                row.addWidget(w_)
            row.addStretch(1)
            wt.addLayout(row)
        wt.addWidget(self.waterfall_plot, 1)
        wt.addWidget(self.waterfall_note)
        self.balancing = BalancingPanel(self)
        self.bearing_panel = BearingPanel()
        self.bearing_panel.set_top_rpm(self.top_rpm)
        self.bearing_panel.changed.connect(lambda: self._on_bearing_tab(FILMS))
        self.magnet_panel = MagnetPanel()
        self.magnet_panel.set_top_rpm(self.top_rpm)
        self.magnet_panel.changed.connect(lambda: self._on_bearing_tab(MAGNETS))
        self.matrices = QtWidgets.QTextBrowser()
        self.theory = FigureBrowser()
        self.theory.setHtml(THEORY_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(campbell_tab, "Campbell diagram")
        self.tabs.addTab(bode_tab, "Bode plot")
        self.tabs.addTab(polar_tab, "Polar plot")
        self.tabs.addTab(stability_tab, "Stability")
        self.tabs.addTab(waterfall_tab, "Waterfall")
        self.tabs.addTab(self.balancing, "Balancing")
        self.tabs.addTab(self.bearing_panel, "Fluid-film bearing")
        self.tabs.addTab(self.magnet_panel, "Magnetic bearing")
        self.tabs.addTab(self.matrices, "Equations && matrices")
        self.tabs.addTab(self.theory, "Theory")
        self.tabs.currentChanged.connect(lambda _: self._refresh_matrices())
        self.tabs.currentChanged.connect(lambda _: self._draw_waterfall(only_new=True))
        self.tabs.currentChanged.connect(lambda _: self._show_bearing_speed())

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
        self._cost = 1e-9  # s of computing per simulated s (see step)
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
    def _preset_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Preset")
        v = QtWidgets.QVBoxLayout(box)
        self.presets = rotor_presets()
        self.preset = QtWidgets.QComboBox()
        self.preset.addItems([p.name for p in self.presets] + [CUSTOM])
        self.preset.setToolTip(
            "<p>Load a rotor that shows one effect clearly, with a speed range to suit it. It starts from "
            "rest and runs up to its target speed. Change any parameter and it becomes your own rotor.</p>")
        self.preset.activated.connect(self._on_preset)
        self.preset_note = QtWidgets.QLabel()
        self.preset_note.setWordWrap(True)
        mute(self.preset_note)
        v.addWidget(self.preset)
        v.addWidget(self.preset_note)
        return box

    def _speed_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Speed")
        form = QtWidgets.QFormLayout(box)
        self.speed_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.speed_slider.setRange(0, int(TOP_RPM))
        self.speed_slider.setSingleStep(10)
        self.speed_slider.setPageStep(250)
        self.speed_slider.setToolTip(SPEED_TIP)
        self.target_rpm = spin(0.0, TOP_RPM, 1000.0, 0, " rpm")
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
        self.repeat_taps = QtWidgets.QCheckBox("Tap the disc every")
        self.repeat_taps.setToolTip(REPEAT_TAP_TIP)
        self.repeat_taps.toggled.connect(self._on_repeat_taps)
        self.tap_every = spin(0.1, 10.0, 0.5, 1, " s")
        self.tap_every.setSingleStep(0.1)
        self.tap_every.setToolTip(REPEAT_TAP_TIP)
        taps = QtWidgets.QHBoxLayout()
        taps.addWidget(self.repeat_taps)
        taps.addWidget(self.tap_every)
        taps.addWidget(QtWidgets.QLabel("of simulated time"))
        taps.addStretch(1)
        form.addRow(taps)
        return box

    def _sweep_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Run-up and coast-down")
        form = QtWidgets.QFormLayout(box)
        self.sweep_from = spin(60.0, TOP_RPM, 800.0, 0, " rpm")
        self.sweep_to = spin(60.0, TOP_RPM, 2400.0, 0, " rpm")
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
            "The phase lag is the angle the shaft turns from the keyphasor (the mark on the shaft passing +x) "
            "to the probe's positive peak, so on a circular orbit the y probe reads 90° more than the x "
            "probe. For the orbit, it is the angle from the mark back to the high spot.</p>"
        )
        self.probe_order = QtWidgets.QComboBox()
        self.probe_order.addItems(ORDER_NAMES)
        self.probe_order.setToolTip(
            "<p>Track the 1X vector (once per revolution: the unbalance, the bow, the skew), the 2X vector "
            "(twice per revolution) or the 3X. On a round shaft nothing drives 2X or 3X; an asymmetric "
            "shaft's weight drives 2X, most near half its critical speeds, and a breathing crack's drives "
            "2X and 3X, most near a half and a third of them.</p>")
        for c in (self.probe_station, self.probe_direction, self.probe_order):
            c.currentIndexChanged.connect(self._on_probe)
        probe_row = QtWidgets.QHBoxLayout()
        probe_row.addWidget(self.probe_station, 2)
        probe_row.addWidget(self.probe_direction, 3)
        probe_row.addWidget(self.probe_order, 1)
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
        self.position = PositionBox()
        self.position.setValue(s.position)
        self.position.setToolTip(
            "<p>Where the disc sits, a/L from bearing A. At 0.5 (midspan) with identical bearings the "
            "disc's translation and tilt are uncoupled, so the spin's gyroscopic moment never shows in "
            "the translational whirl: the classic Jeffcott rotor. Move it off centre and the disc tilts "
            "as it whirls.</p><p>Above 1 the disc overhangs bearing B (below 0, bearing A): an "
            "<i>overhung</i> rotor, like a fan or a pump impeller. The disc then tilts as much as it "
            "moves, and the gyroscopic moment moves even the first critical speed.</p>"
        )
        self.disc_mass = spin(0.01, 1000.0, s.disc_mass, 3, " kg")
        self.ip = spin(1e-6, 100.0, s.ip, 6, " kg·m²")
        self.ip.setToolTip("Polar moment of inertia I<sub>p</sub>, about the spin axis (a thin disc of radius r<sub>d</sub>: m r<sub>d</sub>²/2)")
        self.id = spin(1e-6, 100.0, s.id, 6, " kg·m²")
        self.id.setToolTip("Diametral moment of inertia I<sub>d</sub>, about a diameter (a thin disc: m r<sub>d</sub>²/4 = I<sub>p</sub>/2)")
        self.unbalance = spin(0.0, 1e6, s.unbalance * 1e6, 1, " g·mm")
        self.unbalance.setToolTip(
            "<p>Mass unbalance U = m e: the disc's mass times the distance e of its centre of mass from "
            "the shaft's axis. 20 g·mm on a 1 kg disc is e = 20 µm, about balance grade G6.3 at "
            "3000 rpm (e Ω = 6.3 mm/s).</p>"
        )
        form.addRow("Span L:", self.length)
        form.addRow("Shaft diameter d:", self.diameter)
        form.addRow("Young's modulus:", self.youngs)
        self.shaft_mass = QtWidgets.QCheckBox("Shaft mass, ρ")
        self.shaft_mass.setToolTip(SHAFT_MASS_TIP)
        self.density = spin(100.0, 30000.0, STEEL, 0, " kg/m³")
        self.density.setToolTip("The shaft's density (steel ≈ 7850 kg/m³, aluminium ≈ 2700)")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.shaft_mass)
        row.addWidget(self.density, 1)
        form.addRow("Shaft:", row)
        self.elements = QtWidgets.QSpinBox()
        self.elements.setRange(2, 60)
        self.elements.setValue(s.elements)
        self.elements.setKeyboardTracking(False)
        self.mass_matrix = QtWidgets.QComboBox()
        self.mass_matrix.addItems(["Consistent", "Lumped"])
        for w in (self.elements, self.mass_matrix):
            w.setToolTip(ELEMENTS_TIP)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.elements, 1)
        row.addWidget(self.mass_matrix, 1)
        form.addRow("Beam elements:", row)
        self.shaft_mass.toggled.connect(self._on_params)
        self.mass_matrix.currentIndexChanged.connect(self._on_params)
        form.addRow("Disc position a/L:", self.position)
        form.addRow("Disc mass m:", self.disc_mass)
        form.addRow("Polar inertia I<sub>p</sub>:", self.ip)
        form.addRow("Diametral inertia I<sub>d</sub>:", self.id)
        self.unbalance_angle = spin(-180.0, 180.0, math.degrees(s.unbalance_angle), 0, "°")
        self.unbalance_angle.setToolTip(
            "<p>Where the heavy spot is: its angle from the keyphasor mark on the shaft, in the direction "
            "of spin. The probes' phase is measured from the mark, as on a real machine.</p>"
        )
        self.bow = spin(0.0, 5000.0, s.bow * 1e6, 1, " µm")
        self.bow_angle = spin(-180.0, 180.0, math.degrees(s.bow_angle), 0, "°")
        self.skew = spin(0.0, 10.0, math.degrees(s.skew), 3, "°")
        self.skew_angle = spin(-180.0, 180.0, math.degrees(s.skew_angle), 0, "°")
        self.asymmetry = spin(0.0, 90.0, s.asymmetry * 100, 1, " %")
        self.asymmetry_angle = spin(-180.0, 180.0, math.degrees(s.asymmetry_angle), 0, "°")
        crack = Crack()
        self.crack_depth = spin(0.0, CRACK_DEPTH_MAX * 100, 0.0, 1, " %")
        self.crack_depth.setSingleStep(5.0)
        self.crack_angle = spin(-180.0, 180.0, math.degrees(crack.angle), 0, "°")
        for size, angle, tip, label in ((self.unbalance, self.unbalance_angle, None, "Unbalance m·e:"),
                                        (self.bow, self.bow_angle, BOW_TIP, "Shaft bow δ<sub>b</sub>:"),
                                        (self.skew, self.skew_angle, SKEW_TIP, "Disc skew τ:"),
                                        (self.asymmetry, self.asymmetry_angle, ASYMMETRY_TIP,
                                         "Asymmetry ±Δ:"),
                                        (self.crack_depth, self.crack_angle, CRACK_TIP, "Crack depth a<sub>c</sub>/d:")):
            angle.setWrapping(True)
            angle.setStepType(QtWidgets.QAbstractSpinBox.StepType.DefaultStepType)
            angle.setSingleStep(15.0)
            row = QtWidgets.QHBoxLayout()
            row.addWidget(size, 3)
            row.addWidget(QtWidgets.QLabel("at"))
            row.addWidget(angle, 2)
            for w in (size, angle) if tip else ():
                w.setToolTip(tip)
            form.addRow(label, row)
        self.crack_position = spin(-1.0, 2.0, crack.position, 3, "")
        self.crack_position.setSingleStep(0.05)
        self.crack_breathing = QtWidgets.QCheckBox("Breathing")
        self.crack_breathing.setChecked(crack.breathing)
        self.crack_breathing.toggled.connect(self._on_params)
        for w in (self.crack_position, self.crack_breathing):
            w.setToolTip(CRACK_TIP)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.crack_position, 3)
        row.addWidget(self.crack_breathing, 2)
        form.addRow("Crack at z/L:", row)
        self.internal_damping = spin(0.0, 1e5, s.internal_damping, 1, " N·s/m")
        self.internal_damping.setToolTip(INTERNAL_TIP)
        form.addRow("Internal damping c<sub>i</sub>:", self.internal_damping)
        self.gravity = QtWidgets.QCheckBox(f"Horizontal rotor (g = {GRAVITY} m/s²)")
        self.gravity.setToolTip(GRAVITY_TIP)
        self.gravity.toggled.connect(self._on_params)
        form.addRow("Gravity:", self.gravity)
        self.shaft_note = QtWidgets.QLabel()
        self.shaft_note.setWordWrap(True)
        mute(self.shaft_note)
        form.addRow(self.shaft_note)
        for box_ in (self.length, self.diameter, self.youngs, self.position, self.disc_mass, self.ip,
                     self.id, self.unbalance, self.unbalance_angle, self.bow, self.bow_angle, self.skew, self.skew_angle,
                     self.asymmetry, self.asymmetry_angle, self.crack_depth, self.crack_angle, self.crack_position,
                     self.internal_damping, self.density, self.elements):
            box_.valueChanged.connect(self._on_params)
        return box

    def _bearing_box(self) -> QtWidgets.QGroupBox:
        b = Bearing()
        box = QtWidgets.QGroupBox("Bearing supports")
        grid = QtWidgets.QGridLayout(box)
        grid.setHorizontalSpacing(4)
        # Units on a second line, so that the headers are no wider than the boxes below them.
        heads = (("k<sub>x</sub><br>[N/m]", "Horizontal support stiffness"),
                 ("k<sub>y</sub><br>[N/m]", "Vertical support stiffness"),
                 ("c<sub>x</sub><br>[N·s/m]", "Horizontal support damping"),
                 ("c<sub>y</sub><br>[N·s/m]", "Vertical support damping"),
                 ("m<br>[kg]", "Mass of the journal and whatever moves with it on the support"),
                 ("k<sub>xy</sub><br>[N/m]", KXY_TIP),
                 ("λ<br>&nbsp;", SWIRL_TIP))
        for c, width in enumerate(BEARING_WIDTHS):
            grid.setColumnStretch(c + 1, width)
        for c, (text, tip) in enumerate(heads):
            label = QtWidgets.QLabel(text)
            label.setToolTip(tip)
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(label, 0, c + 1)
        self.bearing_rows: list[list[QtWidgets.QDoubleSpinBox]] = []
        for r, name in enumerate("AB"):
            grid.addWidget(QtWidgets.QLabel(f"<b>{name}</b>"), r + 1, 0)
            row = [spin(1e2, 1e10, b.kx, 0), spin(1e2, 1e10, b.ky, 0), spin(0.0, 1e6, b.cx, 1),
                   spin(0.0, 1e6, b.cy, 1), spin(0.001, 100.0, b.mass, 3), spin(-1e9, 1e9, b.kxy, 0),
                   spin(0.0, 1.0, b.swirl, 2)]
            row[5].setToolTip(KXY_TIP)
            row[6].setToolTip(SWIRL_TIP)
            row[6].setSingleStep(0.05)
            for c, w in enumerate(row):
                # No arrows (the wheel and the arrow keys still step): seven columns leave no room for them.
                w.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
                w.setMinimumWidth(BEARING_WIDTHS[c])
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
        self.supports = QtWidgets.QComboBox()
        self.supports.addItems(SUPPORTS)
        self.supports.setToolTip("<p>What holds the journals: the springs and dampers in the table, or "
                                 "these bearings at both ends.</p>")
        self.supports.setItemData(FILMS, FILM_TIP, QtCore.Qt.ItemDataRole.ToolTipRole)
        self.supports.setItemData(MAGNETS, MAGNET_TIP, QtCore.Qt.ItemDataRole.ToolTipRole)
        self.supports.currentIndexChanged.connect(lambda _: self._on_params())
        self.nonlinear_film = QtWidgets.QCheckBox("Nonlinear film")
        self.nonlinear_film.setChecked(True)
        self.nonlinear_film.setToolTip(NONLINEAR_FILM_TIP)
        for c in (self.isotropic, self.same_bearings, self.nonlinear_film):
            c.toggled.connect(self._on_params)
        grid.addWidget(self.isotropic, 3, 0, 1, 8)
        grid.addWidget(self.same_bearings, 4, 0, 1, 8)
        grid.addWidget(self.supports, 5, 0, 1, 5)
        grid.addWidget(self.nonlinear_film, 5, 5, 1, 3)
        return box

    # ------------------------------------------------------------- parameters
    def system(self) -> RotorSystem:
        """The rotor described by the panels, after applying the isotropic and same-as-A ties."""
        iso, same = self.isotropic.isChecked(), self.same_bearings.isChecked()
        kind = self.supports.currentIndex()
        film = kind != SPRINGS  # a film or a magnet holds the journal
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
                # A film or a magnet holds the journal in place of the springs: only its mass is left to set.
                w.setEnabled(not (same and r == 1) and not (iso and c in (1, 3)) and not (film and c != 4))
        for w in (self.isotropic, self.same_bearings):
            w.setEnabled(not film)
        # A turning shaft's split time step keeps the films linearized.
        self.nonlinear_film.setEnabled(kind == FILMS and not self.asymmetry.value() and not self.crack_depth.value())
        self.nonlinear_film.setVisible(kind == FILMS)
        oil = self.bearing_panel.bearing(rotor=False) if kind == FILMS else None
        magnet = self.magnet_panel.bearing() if kind == MAGNETS else None
        bearings = [Bearing(*(w.value() for w in row), film=oil, magnet=magnet) for row in self.bearing_rows]
        for w in (self.density, self.elements, self.mass_matrix):
            w.setEnabled(self.shaft_mass.isChecked())
        cracked = self.crack_depth.value() > 0.0
        for w in (self.crack_angle, self.crack_position, self.crack_breathing):
            w.setEnabled(cracked)
        return RotorSystem(
            length=self.length.value(),
            diameter=self.diameter.value() / 1e3,
            youngs=self.youngs.value() * 1e9,
            position=_off_bearing(self.position.value()),
            disc_mass=self.disc_mass.value(),
            ip=self.ip.value(),
            id=self.id.value(),
            bearing_a=bearings[0],
            bearing_b=bearings[1],
            unbalance=self.unbalance.value() / 1e6,
            unbalance_angle=math.radians(self.unbalance_angle.value()),
            weights=self._weights,
            plane_weights=self._plane_weights,
            gravity=GRAVITY if self.gravity.isChecked() else 0.0,
            bow=self.bow.value() / 1e6,
            bow_angle=math.radians(self.bow_angle.value()),
            skew=math.radians(self.skew.value()),
            skew_angle=math.radians(self.skew_angle.value()),
            asymmetry=self.asymmetry.value() / 100,
            asymmetry_angle=math.radians(self.asymmetry_angle.value()),
            crack=self._crack() if cracked else None,
            internal_damping=self.internal_damping.value(),
            density=self.density.value() if self.shaft_mass.isChecked() else 0.0,
            elements=self.elements.value(),
            lumped=self.mass_matrix.currentIndex() == 1,
            nonlinear_films=kind == FILMS and self.nonlinear_film.isChecked(),
        ).with_(**(self._hidden or {}))

    def _crack(self) -> Crack:
        """The crack the panel describes, kept inside the shaft (and off its ends)."""
        lo, hi = (x / self.length.value() for x in (min(0.0, self.position.value() * self.length.value()),
                                                    max(self.length.value(), self.position.value() * self.length.value())))
        margin = 1e-3 * (hi - lo)
        return Crack(position=min(max(self.crack_position.value(), lo + margin), hi - margin),
                     depth=self.crack_depth.value() / 100, angle=math.radians(self.crack_angle.value()),
                     breathing=self.crack_breathing.isChecked())

    def set_system(self, system: RotorSystem) -> None:
        """Show and use `system` (for scripts and tests)."""
        boxes = (self.length, self.diameter, self.youngs, self.position, self.disc_mass, self.ip, self.id,
                 self.unbalance, self.unbalance_angle, self.bow, self.bow_angle, self.skew, self.skew_angle,
                 self.asymmetry, self.asymmetry_angle, self.crack_depth, self.crack_angle, self.crack_position,
                 self.crack_breathing, self.internal_damping,
                 self.gravity, self.isotropic, self.same_bearings, self.supports, self.nonlinear_film,
                 self.shaft_mass, self.density,
                 self.elements, self.mass_matrix, *(w for row in self.bearing_rows for w in row))
        for w in boxes:
            w.blockSignals(True)
        self.length.setValue(system.length)
        self.diameter.setValue(system.diameter * 1e3)
        self.youngs.setValue(system.youngs / 1e9)
        self.position.setValue(system.position)
        self.disc_mass.setValue(system.disc_mass)
        self.ip.setValue(system.ip)
        self.id.setValue(system.id)
        self._clear_hidden()
        self.unbalance.setValue(system.unbalance * 1e6)
        self.unbalance_angle.setValue(math.degrees(system.unbalance_angle))
        self._weights = system.weights
        self._plane_weights = system.plane_weights
        self.gravity.setChecked(system.gravity > 0.0)
        self.bow.setValue(system.bow * 1e6)
        self.bow_angle.setValue(math.degrees(system.bow_angle))
        self.skew.setValue(math.degrees(system.skew))
        self.skew_angle.setValue(math.degrees(system.skew_angle))
        self.asymmetry.setValue(system.asymmetry * 100)
        self.asymmetry_angle.setValue(math.degrees(system.asymmetry_angle))
        crack = system.crack
        self.crack_depth.setValue(0.0 if crack is None else crack.depth * 100)
        if crack is not None:
            self.crack_angle.setValue(math.degrees(crack.angle))
            self.crack_position.setValue(crack.position)
            self.crack_breathing.setChecked(crack.breathing)
        self.internal_damping.setValue(system.internal_damping)
        self.shaft_mass.setChecked(system.density > 0.0)
        if system.density > 0.0:
            self.density.setValue(system.density)
        self.elements.setValue(system.elements)
        self.mass_matrix.setCurrentIndex(1 if system.lumped else 0)
        A, B = system.bearing_a, system.bearing_b
        self.isotropic.setChecked(all(b.kx == b.ky and b.cx == b.cy for b in (A, B)))
        self.same_bearings.setChecked(A == B)
        for row, b in zip(self.bearing_rows, (A, B)):
            for w, v in zip(row, (b.kx, b.ky, b.cx, b.cy, b.mass, b.kxy, b.swirl)):
                w.setValue(v)
        self.supports.setCurrentIndex(FILMS if A.film is not None else MAGNETS if A.magnet is not None else SPRINGS)
        self.nonlinear_film.setChecked(system.nonlinear_films or A.film is None)
        if A.film is not None:
            self.bearing_panel.set_bearing(A.film, notify=False)
        if A.magnet is not None:
            self.magnet_panel.set_bearing(A.magnet, notify=False)
        for w in boxes:
            w.blockSignals(False)
        self._on_params()

    def load_preset(self, name: str) -> None:
        """Load the preset called `name`: its rotor, speed range and target speed, starting from rest."""
        preset = next(p for p in self.presets if p.name == name)
        self.set_top_rpm(preset.top_rpm)
        self.set_system(preset.system)
        self.set_target_rpm(preset.target_rpm)
        self.reset()

    def _on_preset(self, index: int) -> None:
        if index < len(self.presets):
            self.load_preset(self.presets[index].name)
        else:
            self._show_preset(self.sim.system)  # Custom is not a rotor: show what is loaded

    def _show_preset(self, system: RotorSystem) -> None:
        """Select the preset this rotor is, or Custom, and show its note."""
        index = next((i for i, p in enumerate(self.presets) if _same_rotor(p.system, system)), len(self.presets))
        self.preset.blockSignals(True)
        self.preset.setCurrentIndex(index)
        self.preset.blockSignals(False)
        self.preset_note.setText(self.presets[index].note if index < len(self.presets)
                                 else "Your own rotor: pick a preset to start again from one.")

    def set_weights(self, weights: tuple[complex, ...], plane_weights: tuple[PlaneWeight, ...] = ()) -> None:
        """Put these balance weights on the disc (kg·m each) and in these planes, in place of any there."""
        self._weights = tuple(weights)
        self._plane_weights = tuple(plane_weights)
        self._on_params()

    def _hideable(self) -> dict[str, tuple[QtWidgets.QDoubleSpinBox, float]]:
        """The boxes a balancing exercise can hide, by RotorSystem field, with the factor from SI to the box."""
        deg = 180.0 / math.pi
        return {"unbalance": (self.unbalance, 1e6), "unbalance_angle": (self.unbalance_angle, deg),
                "skew": (self.skew, deg), "skew_angle": (self.skew_angle, deg)}

    def hide_unbalance(self, unbalance: float, angle: float, skew: tuple[float, float] | None = None) -> None:
        """Give the disc this unbalance (kg·m at `angle` rad) without showing it: a balancing exercise.

        With `skew` (τ, its angle in rad), hide a couple unbalance too; otherwise the skew stays as set.
        """
        self._clear_hidden()
        self._hidden = {"unbalance": unbalance, "unbalance_angle": angle}
        if skew is not None:
            self._hidden.update(skew=skew[0], skew_angle=skew[1])
        boxes = self._hideable()
        for name in self._hidden:
            box = boxes[name][0]
            box.blockSignals(True)
            box.setSpecialValueText("hidden")
            box.setValue(box.minimum())
            box.setEnabled(False)
            box.blockSignals(False)
        self.set_heavy_shown(False)
        self._on_params()

    def reveal_unbalance(self) -> None:
        """Show the hidden unbalance (and couple) in the panel again, and the heavy spot."""
        if self._hidden is None:
            return
        hidden, boxes = self._hidden, self._hideable()
        self._clear_hidden()
        for name, value in hidden.items():
            box, scale = boxes[name]
            box.blockSignals(True)
            box.setValue(value * scale)
            box.blockSignals(False)
        self.set_heavy_shown(True)
        self._on_params()

    def _clear_hidden(self) -> None:
        if self._hidden is not None:
            for name in self._hidden:
                box = self._hideable()[name][0]
                box.setSpecialValueText("")
                box.setEnabled(True)
        self._hidden = None

    def set_heavy_shown(self, shown: bool) -> None:
        self.view.show_heavy = shown
        self.balancing.show_heavy.blockSignals(True)
        self.balancing.show_heavy.setChecked(shown)
        self.balancing.show_heavy.blockSignals(False)
        self._draw()

    def _on_params(self) -> None:
        self._matrices_speed = -math.inf  # redraw the matrices: the numbers changed
        system = self.system()
        if self.sweeps or self.run_up is not None:
            self.clear_sweeps()  # their steady-state curve is about to change
        self.cancel_measurement()
        self.sim.set_system(system)
        self.bearing_panel.set_rotor(system.bearing_loads if system.films else None)
        self.magnet_panel.set_rotor(system if system.magnets else None)
        self._show_preset(system)
        self.view.set_system(system)
        self.orbits.set_midspan_shown(abs(system.position - 0.5) > 1e-9)
        self._stride = max(1, round(STORE_DT / self.sim.step_size()))
        self._set_sag(system)
        self._new_waterfall()  # its spectra and free whirl frequencies were of the old rotor
        ks = 48.0 * system.ei / system.length**3
        if system.overhung:
            formula = f" = 3EI/(c²(L + c)), overhung by c = {system.overhang * 1e3:.4g} mm"
        elif abs(system.position - 0.5) < 1e-9:
            formula = " = 48EI/L³"
        else:
            formula = f" (48EI/L³ = {ks:,.0f} N/m at midspan)"
        self.shaft_note.setText(
            f"EI = {system.ei:.4g} N·m²; shaft stiffness at the disc (bearings rigid) "
            f"{system.disc_stiffness:,.0f} N/m" + formula + self._asymmetry_text(system) + self._crack_text(system)
            + self._shaft_mass_text(system) + self._sag_text(system)
        )
        self._analysis_timer.start()

    @staticmethod
    def _asymmetry_text(system: RotorSystem) -> str:
        if not system.asymmetry:
            return ""
        d = system.asymmetry
        return (f"<br>Asymmetric: EI = {system.ei * (1 + d):.4g} N·m² bending toward the stiff axis "
                f"({math.degrees(system.asymmetry_angle):.0f}° from the mark), {system.ei * (1 - d):.4g} across it")

    @staticmethod
    def _crack_text(system: RotorSystem) -> str:
        crack = system.crack
        if crack is None:
            return ""
        c_open, c_across = system.crack_compliances
        soft, stiff = (system.axis_system(k).stiffness(0.0) for k in (False, True))
        loss = [1.0 - k[XD, XD] / system.with_(crack=None).stiffness(0.0)[XD, XD] for k in (soft, stiff)]
        state = ("held open" if not crack.breathing else
                 "breathing, opened by the weight" if system.crack_opening else
                 "breathing, but shut: nothing opens it without gravity")
        return (f"<br>Crack {crack.depth:.0%} deep at z = {system.crack_z * 1e3:.4g} mm ({state}): compliance "
                f"{c_open:.3g} rad/(N·m) opening it, {c_across:.3g} across; on average over a turn it takes "
                f"{loss[0]:.2%} and {loss[1]:.2%} off the disc's stiffness along and across its mouth")

    @staticmethod
    def _shaft_mass_text(system: RotorSystem) -> str:
        if not system.density:
            return ""
        f = system.fixed_interface_freqs
        held = (f"its lowest {f.size} mode{'s' if f.size != 1 else ''} with the stations held "
                f"({', '.join(f'{x:,.0f}' for x in f)} Hz)" if f.size else
                f"no modes with the stations held below {MODE_LIMIT:,.0f} Hz")
        return (f"<br>Shaft {system.shaft_mass:.3g} kg ({system.shaft_mass / system.disc_mass:.0%} of the disc), "
                f"{system.nodes.size - 1} elements, reduced to the stations and {held}")

    def _refresh_analysis(self) -> None:
        """The Campbell diagram and critical speeds for the current parameters."""
        system = self.sim.system
        top = self.top_rpm
        self.campbell = campbell(system, top / RPM, n_modes=8)
        # With the shaft's mass: the same rotor's Jeffcott model, for comparison.
        massless = campbell(system.with_(density=0.0), top / RPM, n_modes=8) if system.density else None
        self.onset = stability_onset(system, top / RPM)
        self.bearing_panel.set_onset(self.onset if system.films else None)
        self.magnet_panel.set_onset(self.onset if system.magnets else None)
        iso = _isotropic(system)
        self.campbell_plot.set_result(self.campbell, top, iso, self.onset)
        self.campbell_plot.set_reference(massless)
        self.stability_map.set_result(self.campbell, self.onset, top)
        self._new_asymmetry_chart()
        crits = critical_speeds(self.campbell, iso)
        self.time.set_criticals([w * RPM for w, _ in crits])
        lines = [f"{w * RPM:,.0f} rpm ({w / (2 * math.pi):.4g} Hz), {whirl_name(r)} whirl" for w, r in crits]
        # On oil films, far stiffer than most shafts, the textbook estimate takes the bearings as rigid.
        jeff = (math.sqrt(system.disc_stiffness / system.disc_mass) if system.films else system.jeffcott_speed()) * RPM
        self.crit_label.setText(
            "<b>Critical speeds</b> (where a whirl frequency equals the spin speed, up to "
            f"{top:,.0f} rpm):<br>" + ("<br>".join(lines) if lines else "none")
            + ("<br><small>The backward whirl crossings are not critical: an isotropic rotor's unbalance "
               "drives only forward whirl.</small>" if iso else "")
            + f"<br><small>Textbook Jeffcott estimate (massless journals, no tilt"
            + (", rigid bearings" if system.films else "") + f"): {jeff:,.0f} rpm</small>"
            + ("<br><small>An asymmetric or cracked shaft has two critical speeds for each mode: those of a "
               "round shaft as stiff as each of its axes (on average over a turn).</small>" if system.periodic else "")
            + (f"<br><small>With a massless shaft (the Jeffcott model, dashed): "
               + (", ".join(f"{w * RPM:,.0f}" for w, _ in critical_speeds(massless, iso)) or "none") + " rpm</small>"
               if massless is not None else "")
            + f"<br><b>Stability:</b> {self._onset_text()}"
        )
        self.stability_note.setText(self._stability_note())
        self.campbell_note.setText(
            "Each dot is a damped natural frequency at that speed, coloured by its whirl direction "
            f"(modes damped more than ζ = {DAMPED:g}, here the journals bouncing on their supports, are "
            "hollow). A critical speed (circled) is where a lightly damped whirl frequency meets the 1X "
            "line: the unbalance, turning once per revolution, drives that mode at its natural "
            "frequency. An isotropic rotor's unbalance drives only its forward whirl."
            + (" Crosses mark growing modes (ζ < 0), " + ("between the dashed lines." if math.isfinite(self.onset.end)
                                                        else "above the onset of instability (dashed).")
               if self.onset else "")
        )
        self._matrices_speed = -math.inf
        self._refresh_matrices()
        self._refresh_steady()

    def _onset_text(self) -> str:
        o = self.onset
        if o is None:
            return f"stable up to {self.top_rpm:,.0f} rpm."
        whirl = whirl_name(o.whirl)
        if o.omega == 0.0:
            return f"<b>unstable even at rest</b>: the {whirl} whirl at {o.freq_hz:.3g} Hz grows."
        if math.isfinite(o.end):
            return (f"<b>unstable from {o.omega * RPM:,.0f} to {o.end * RPM:,.0f} rpm</b>, where the {whirl} whirl "
                    f"at {o.freq_hz:.3g} Hz ({o.order:.2f}×) grows; stable again above.")
        return (f"<b>unstable above {o.omega * RPM:,.0f} rpm</b>, where the {whirl} whirl at "
                f"{o.freq_hz:.3g} Hz ({o.order:.2f}×) starts to grow.")

    def _stability_note(self) -> str:
        s = self.sim.system
        bearings = (("A", s.bearing_a), ("B", s.bearing_b))
        coupled = [f"k<sub>xy</sub> = {b.kxy:,.0f} N/m at {n}" for n, b in bearings if b.kxy and b.film is None]
        coupled += [f"swirl λ = {b.swirl:g} at {n}" for n, b in bearings if b.swirling]
        if s.films:
            coupled.append("oil-film bearings (their oil swirling at half the speed)")
        if s.magnets:
            coupled.append("magnetic bearings (their controllers' lag)")
        if s.internal_damping:
            coupled.append(f"internal damping c<sub>i</sub> = {s.internal_damping:g} N·s/m")
        if s.asymmetry:
            coupled.append(f"the shaft's asymmetry Δ = {s.asymmetry:.3g}")
        if s.crack_opening:
            coupled.append(f"a {'breathing' if s.crack.breathing else 'open'} crack {s.crack.depth:.0%} deep")
        cause = ("With " + " and ".join(coupled) + ": " if coupled else
                 "No cross-coupling, internal damping, asymmetry or crack: every mode stays damped, however fast it "
                 "spins. Set c<sub>i</sub>, k<sub>xy</sub>, λ or Δ to see one go unstable. ")
        if self.measure.currentIndex() == 2:
            chart = (" The chart shades where the rotor is unstable against the shaft's asymmetry Δ and the speed "
                     "(the rest of the rotor as it is), between the two axes' critical speeds (dashed).")
            if self.asym_chart is not None and not self.asym_chart.done:
                chart += f" Working it out: {self.asym_chart._next / self.asym_chart.growth.size:.0%}."
        else:
            chart = f" Modes damped more than ζ = {DAMPED:g} are left off."
        return f"<small>{cause}{self._onset_text()}{chart} Below: the full spectrum of the strip chart's station.</small>"

    def _on_measure(self, index: int) -> None:
        if index < 2:
            self.stability_map.set_measure(index == 1)
        self.stability_stack.setCurrentIndex(1 if index == 2 else 0)
        self._new_asymmetry_chart()
        self.stability_note.setText(self._stability_note())

    def _new_asymmetry_chart(self) -> None:
        """Start the asymmetry chart afresh if the rotor (other than its asymmetry) or the speed range changed."""
        system = self.sim.system
        top = max(CHART_DELTA, math.ceil(system.asymmetry * 12.0) / 10.0)
        chart = self.asym_chart
        if (chart is not None and chart.system.with_(asymmetry=0.0) == system.with_(asymmetry=0.0)
                and chart.omegas[-1] == self.top_rpm / RPM and chart.asymmetries[-1] == top):
            return
        self.asym_chart = AsymmetryChart(system, self.top_rpm / RPM, top) if self.measure.currentIndex() == 2 else None
        self.asymmetry_plot.set_chart(self.asym_chart)

    def _work_asymmetry_chart(self) -> None:
        """Fill in some more of the asymmetry chart, if it shows, and draw what is new."""
        chart = self.asym_chart
        if (chart is None or self.measure.currentIndex() != 2
                or self.tabs.currentWidget() is not self.spectrum.parentWidget()):
            return
        if not chart.done:
            chart.work(CHART_BUDGET)
        elif self.asymmetry_plot.drawn == chart._next:
            return
        self.asymmetry_plot.set_chart(chart)
        if chart.done or self._frames % SPECTRUM_FRAMES == 0:
            self.stability_note.setText(self._stability_note())

    def _refresh_steady(self) -> None:
        """The steady-state 1X vectors at every station, finely around each critical speed."""
        w = [np.linspace(self.top_rpm / STEADY_POINTS, self.top_rpm, STEADY_POINTS) / RPM]
        crits = [c for c, _, _ in self.campbell.criticals] if self.campbell else []
        system = self.sim.system
        if system.periodic and system.gravity:
            # Gravity's 2X peaks near half of each (and a breathing crack's 3X near a third).
            crits += [c / k for c in crits for k in ((2, 3) if system.spacing == 1 else (2,))]
        w += [np.linspace(0.8 * c, 1.2 * c, STEADY_POINTS // max(len(crits), 1)) for c in crits]
        w = np.unique(np.concatenate(w))
        self._steady_rpm = w * RPM
        V = steady_harmonics(system, w, len(ORDER_NAMES))[1]
        self._steady_orders = [station_motion(system, V[:, k]) for k in range(len(ORDER_NAMES))]
        self._steady_vectors = self._steady_orders[0]
        self._draw_tracking(full=True)
        self.balancing.rotor_changed()
        self.balancing.refresh()

    def _on_bearing_tab(self, kind: int) -> None:
        if self.supports.currentIndex() == kind:
            self._on_params()  # the rotor's bearings are the tab's

    def _show_bearing_speed(self) -> None:
        if self.tabs.currentWidget() is self.bearing_panel:
            self.bearing_panel.set_speed(self.sim.omega * RPM)
        elif self.tabs.currentWidget() is self.magnet_panel:
            self.magnet_panel.set_speed(self.sim.omega * RPM)

    def _refresh_matrices(self) -> None:
        rpm = self.sim.omega * RPM
        if self.tabs.currentWidget() is not self.matrices or abs(rpm - self._matrices_speed) < 25.0:
            return
        self._matrices_speed = rpm
        scroll = self.matrices.verticalScrollBar().value()
        self.matrices.setHtml(matrices_html(self.sim.system, self.sim.omega, hidden=tuple(self._hidden or ())))
        self.matrices.verticalScrollBar().setValue(scroll)

    def apply_theme(self) -> None:
        self.view.apply_theme()
        self.orbits.apply_theme()
        self.time.apply_theme()
        self.campbell_plot.apply_theme()
        self.stability_map.apply_theme()
        self.spectrum.apply_theme()
        self.waterfall_plot.apply_theme()
        self.bode.apply_theme()
        self.polar.apply_theme()
        self.sweep_plots.apply_theme()
        self.balancing.apply_theme()
        self.bearing_panel.apply_theme()
        self.magnet_panel.apply_theme()
        self._draw_tracking(full=True)
        scroll = self.theory.verticalScrollBar().value()
        self.theory.setHtml(THEORY_HTML)
        self.theory.verticalScrollBar().setValue(scroll)
        self._matrices_speed = -math.inf
        self._refresh_matrices()
        self._draw()

    # ------------------------------------------------------------- speed and actions
    def set_top_rpm(self, rpm: float) -> None:
        """Show speeds up to `rpm`: on the plots, and as the top of the speed controls."""
        if self.top_speed.value() != rpm:
            self.top_speed.setValue(rpm)  # comes back here
            return
        self.top_rpm = rpm
        self.speed_slider.setRange(0, int(rpm))
        for w in (self.target_rpm, self.sweep_from, self.sweep_to):
            w.setMaximum(rpm)  # a target above it comes down (and the speed with it)
        self._set_sag(self.sim.system)
        self._new_waterfall()
        self.bearing_panel.set_top_rpm(rpm)
        self.magnet_panel.set_top_rpm(rpm)
        self._analysis_timer.start()

    def _on_orders(self) -> None:
        self.campbell_plot.set_orders({order for order, box_ in self.order_boxes.items() if box_.isChecked()})

    def set_target_rpm(self, rpm: float) -> None:
        self.target_rpm.setValue(min(max(rpm, 0.0), self.top_rpm))

    def _on_target(self, rpm: float) -> None:
        self.stop_sweep(hold_speed=False)  # the user took over the speed
        self.cancel_measurement()
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
        self.cancel_measurement()
        self.sim.reset()
        self.rubbed = False
        self.history.clear()
        self._new_waterfall()
        self._target_t = 0.0
        self._carry = 0
        self._next_tap = 0.0
        self._draw()

    def tap(self) -> None:
        self.sim.tap(0.0, -TAP_IMPULSE)

    def _on_repeat_taps(self, on: bool) -> None:
        if on:
            self._next_tap = self.sim.t  # the first one now

    # ------------------------------------------------------------- balancing measurements
    def start_measurement(self, omega: float, measured) -> None:
        """Run at `omega` rad/s, settle and average the 1X vectors; then call measured(vectors (4, 2))."""
        self.stop_sweep(hold_speed=False)
        self.cancel_measurement()
        if not self.running:
            self._on_run()
        self.measurement = SteadyMeasurement(self.sim, omega)
        self._measured = measured
        self._sync_target()
        self._start_timer()

    def cancel_measurement(self) -> None:
        if self.measurement is None:
            return
        self.measurement = self._measured = None
        self.balancing.measurement_ended()

    def _advance_measurement(self, duration: float):
        m = self.measurement
        r = m.advance(duration)
        if m.sim.target * RPM != self.target_rpm.value():
            self._sync_target()
        if m.done:
            measured, self.measurement, self._measured = self._measured, None, None
            measured(m.result)
            self.balancing.measurement_ended()
        return r

    def measurement_status(self) -> str:
        m = self.measurement
        if m is None:
            return ""
        rpm = m.omega * RPM
        if m.stage == "approach":
            return f"Going to {rpm:,.0f} rpm…"
        if m.stage == "settle":
            return f"Settling at {rpm:,.0f} rpm ({max(m._settle_end - self.sim.t, 0.0):.1f} s)…"
        return f"Measuring at {rpm:,.0f} rpm: {min(m.tracked, m.revs)} of {m.revs} revolutions"

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
        self.cancel_measurement()
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
        order = self._order()
        self.sweep_plots.order = order
        self.bode.set_order(order)
        self._fit_tracking_range()
        self._draw_tracking(full=True)

    def _on_show_peaks(self, on: bool) -> None:
        self.sweep_plots.set_show_peaks(on)

    def _probe(self) -> tuple[int, str]:
        return self.probe_station.currentIndex(), DIRECTIONS[self.probe_direction.currentIndex()]

    def _order(self) -> int:
        """The harmonic the Bode and polar plots track: 1 (1X), 2 (2X) or 3 (3X)."""
        return self.probe_order.currentIndex() + 1

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
        """The steady-state 1X (or 2X, 3X) values at the chosen station and direction, and their unwrapped lag."""
        station, direction = self._probe()
        vectors = self._steady_orders[self._order() - 1]
        values = probe(vectors[:, station], direction)
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
        order = self._order()
        system = self.sim.system
        if order > 1:
            where += f", {order}X"
            if not ss_amp:
                return (f"<b>Steady state</b> ({where}): nothing drives {order}X"
                        + (" on a round shaft." if not system.periodic else
                           " on an asymmetric shaft (only a breathing crack's stiffness changes at 3X)."
                           if order == 3 and system.spacing == 2 else
                           " without gravity: tick <i>Horizontal rotor</i>."))
        head = f"<b>Steady state</b> ({where}): peak {_fmt_len(ss_amp)} at {ss_rpm:,.0f} rpm"
        crits = self.campbell.criticals if self.campbell else []
        zetas = [(abs(w - wc), z) for w, _, z in crits]
        near = min(crits, key=lambda c: abs(c[0] - order * wc), default=None)
        if order > 1:
            if near is not None and abs(near[0] - order * wc) < 0.15 * order * wc:
                head += (f", about {'half' if order == 2 else 'a third of'} the critical speed "
                         f"{near[0] * RPM:,.0f} rpm" + (": the <i>gravity critical</i>" if order == 2 else ""))
        elif zetas and min(zetas)[0] < 0.05 * wc:
            z = min(zetas)[1]
            head += f"; its mode has ζ = {z:.3g}, ζ² = {z * z:.2g}"
        if not self.sweeps:
            return head + (f"<br><small>Run up and coast down to track the {order}X response over this curve.</small>")
        rows = []
        for c in self.sweep_plots.curves:
            sweep = c.sweep
            amp, w = sweep.peak(station, direction, order)
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
            f"<th>Ω'/ω<sub>c</sub>²</th><th>{order}X peak</th><th>of steady</th><th>at rpm</th><th>vs steady</th></tr>"
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
        speed = float(self.sim_speed.currentData())
        self.step(dt * (max(speed, MEASURE_SPEED) if self.measurement is not None else speed), FRAME_BUDGET)
        if self.isVisible() and self.running:
            self._timer.start()

    def step(self, duration: float, budget: float | None = None) -> None:
        """Advance by `duration` s of simulated time and redraw.

        With a `budget` (s of computing), advance no further than it allows at the
        cost seen so far: where the simulation is slow (the films' whole force),
        the page falls behind real time rather than stalling.
        """
        self._target_t = min(self._target_t + duration, self.sim.t + 0.1 + duration)
        if budget is not None and self._target_t - self.sim.t > budget / self._cost:
            self._target_t = self.sim.t + budget / self._cost
        if self.repeat_taps.isChecked() and self.measurement is None and self.sim.t >= self._next_tap:
            self.tap()
            self._next_tap = self.sim.t + self.tap_every.value()
        start, began = time.perf_counter(), self.sim.t
        if self.run_up is not None:
            r = self._advance_sweep(self._target_t - self.sim.t)
        elif self.measurement is not None:
            r = self._advance_measurement(self._target_t - self.sim.t)
        else:
            r = self.sim.advance(self._target_t - self.sim.t)
        if self.sim.t > began:  # s of computing per simulated s, smoothed
            cost = (time.perf_counter() - start) / (self.sim.t - began)
            self._cost = 0.8 * self._cost + 0.2 * cost if self._cost > 1e-9 else cost
        if r.t.size:
            # Keep every stride-th sample, continuing the stride across advances.
            first = (self._stride - self._carry - 1) % self._stride
            pick = np.arange(first, r.t.size, self._stride)
            self._carry = (r.t.size - 1 - pick[-1]) if pick.size else self._carry + r.t.size
            if pick.size:
                cols = self._columns(r.q[pick], r.phase[pick], r.omega[pick])
                self.history.extend(r.t[pick], cols)
                vib = self._vibration(cols)
                if self.waterfall.feed(r.t[pick], vib[:, 0::2] + 1j * vib[:, 1::2], cols[:, OMEGA]):
                    self._waterfall_drawn = False
        touched = self._journal_touched()
        if self.running and (touched or np.abs(self.sim.q[[XA, XD, XB, YA, YD, YB]]).max()
                             > RUB * self.sim.system.length):
            self.stop_sweep()
            self.cancel_measurement()
            self._on_run()  # pause
            self.rubbed = True
            self.touched = touched
        self._draw()
        self._refresh_matrices()

    def _journal_touched(self) -> tuple[str, float] | None:
        """The journal ("A" or "B") that has moved its oil film's whole clearance off centre, or onto its
        magnetic bearing's backup bearing, and that clearance."""
        q, s = self.sim.q, self.sim.system
        for name, (jx, jy), b in (("A", (XA, YA), s.bearing_a), ("B", (XB, YB), s.bearing_b)):
            clearance = (b.film.clearance if b.film is not None else
                         b.magnet.touchdown if b.magnet is not None else math.inf)
            if math.hypot(q[jx], q[jy]) >= clearance:
                return name, clearance
        return None

    def _columns(self, q: np.ndarray, phase: np.ndarray, omega: np.ndarray) -> np.ndarray:
        s = self.sim.system
        mid = s.deflection(q, [0.5 * s.length])[:, 0, :]
        return np.column_stack([q[:, XA], q[:, YA], q[:, XD], q[:, YD], mid, q[:, XB], q[:, YB], phase, omega])

    def _draw(self) -> None:
        sim = self.sim
        w = abs(sim.omega)
        trail_time = min(ORBIT_MAX_TIME, ORBIT_REVS * 2 * math.pi / w) if w > 0 else ORBIT_MAX_TIME
        _, y = self.history.window(trail_time)
        shown = self._shown(y)
        trails = [shown[:, 2 * i : 2 * i + 2] for i in range(len(STATIONS))]
        keys = [np.empty((0, 2))] * len(STATIONS)
        if y.shape[0] > 1:
            turns = np.floor(y[:, PHASE] / (2 * math.pi))
            idx = np.flatnonzero(np.diff(turns) != 0) + 1
            keys = [t[idx] for t in trails]
        q_vib = sim.q - sim.system.static_deflection(sim.omega) if sim.system.gravity else sim.q
        q_shown = sim.q if self.show_sag.isChecked() else q_vib
        here = self._columns(q_shown[None, :], np.array([sim.phase]), np.array([sim.omega]))[0]
        peak = max([float(np.abs(t).max()) for t in trails if t.size] + [float(np.abs(here[:8]).max())])
        self.view.update_state(q_shown, sim.phase, trails, peak)
        steady = sim.omega == sim.target and w > 0
        here = self._columns(q_vib[None, :], np.array([sim.phase]), np.array([sim.omega]))[0]
        r_disc = math.hypot(here[2], here[3])
        heavy = self._heavy_angle()
        centre = (0.0, 0.0)
        if self._sag is not None and self.show_sag.isChecked():
            sag = self._sag_at(np.array([sim.omega]))[0]
            centre = (float(sag[2]), float(sag[3]))
        self.orbits.update_orbits(trails, keys, heavy if w > 0 else None, centre)

        t, hist = self.history.window(self.window.value())
        st = self.station.currentIndex()
        self.time.update_curves(t, self._shown(hist)[:, 2 * st : 2 * st + 2], hist[:, OMEGA] * RPM,
                                self.window.value())
        self.campbell_plot.set_speed(sim.omega * RPM)
        self.stability_map.set_speed(sim.omega * RPM)
        self._show_bearing_speed()
        self.asymmetry_plot.set_here(sim.omega * RPM, sim.system.asymmetry)
        self._work_asymmetry_chart()
        self._draw_tracking()
        self.balancing.update_status()
        self._frames += 1
        if self.tabs.currentWidget() is self.spectrum.parentWidget() and (
                self._frames % SPECTRUM_FRAMES == 0 or not self.running):
            self._draw_spectrum()
        if self._frames % WATERFALL_FRAMES == 0 or not self.running:
            self._draw_waterfall(only_new=True)

        # Readout.
        rpm = sim.omega * RPM
        parts = [f"<b>Speed {rpm:,.0f} rpm</b> ({sim.omega / (2 * math.pi):.3g} Hz)"]
        if not steady and sim.omega != sim.target:
            parts[0] += f" → {sim.target * RPM:,.0f} rpm"
        crits = [c for c, _ in critical_speeds(self.campbell, _isotropic(sim.system))] if self.campbell else []
        if crits and w > 0:
            parts.append(f"Ω / first critical = {sim.omega / crits[0]:.2f}")
        disc = self._vibration(y)[:, 2:4]
        if disc.size:
            text = f"disc whirling up to {_fmt_len(float(np.hypot(disc[:, 0], disc[:, 1]).max()))}"
            if sim.system.gravity:
                text += " about its sag"
            if w > 0 and disc.shape[0] > 2:
                # Signed area swept (x dy - y dx): positive turns x toward y, the way the shaft spins.
                swept = float(np.sum(disc[:-1, 0] * np.diff(disc[:, 1]) - disc[:-1, 1] * np.diff(disc[:, 0])))
                text += ", whirling " + ("forward" if swept > 0 else "backward")
            parts.append(text)
        growing = self.onset is not None and sim.omega >= self.onset.omega and least_damped(sim.system, w)[0] < 0
        # (Not where a turning shaft's weight drives 2X: the disc's position then says little of the 1X.)
        if (heavy is not None and steady and not growing and not (sim.system.periodic and sim.system.gravity)
                and r_disc > 1e-3 * max(abs(sim.system.mass_unbalance) / sim.system.disc_mass, MIN_PEAK)):
            lag = math.degrees(heavy - math.atan2(here[3], here[2])) % 360.0
            parts.append(f"heavy spot leads the high spot by {lag:.0f}°")
        if growing:
            zeta, f, _ = least_damped(sim.system, w)
            if f == 0.0:  # a magnetic bearing too weak for its magnets
                parts.append(f"<span style='color:{colors.force}'>unstable: the journals fall toward the "
                             "magnets</span>")
            elif w > 0:
                grow = -zeta * 2 * math.pi * f / math.sqrt(max(1 - zeta * zeta, 1e-12))
                # With the films' whole force that is only how it starts: oil whirl settles to a limit cycle.
                what = "linearized, the equilibrium is unstable" if sim.system.whole_films else "unstable"
                parts.append(f"<span style='color:{colors.force}'>{what}: a whirl at {f:.3g} Hz "
                             f"({2 * math.pi * f / w:.2f}×) grows ×e every {1 / grow:.2g} s</span>")
        if self.rubbed:
            where = (f"journal {self.touched[0]} landed on its backup bearing, {_fmt_len(self.touched[1])} off "
                     "centre (half the magnets' air gap). A real machine drops onto these when its magnetic "
                     "bearings lose control; the model here is linear, and has no contact. "
                     if self.touched and sim.system.magnets else
                     f"journal {self.touched[0]} reached the wall of its bearing, a clearance of "
                     f"{_fmt_len(self.touched[1])} off centre. "
                     "A real oil film stiffens sharply there and keeps the journal off the wall; the film here is "
                     "linearized (about where the journal rests), so it would let it through. Tick <i>Nonlinear "
                     "film</i> to step the film's whole force. "
                     if self.touched else
                     f"the whirl passed {_fmt_len(RUB * sim.system.length)} ({RUB:.0%} of the span), where a real "
                     "rotor would rub its seals or wreck its bearings. "
                     + ("The oil films hold the journals inside their clearance, but not the shaft's bending, so "
                        "oil whip grows on. " if sim.system.whole_films else
                        "The model is linear, so it would grow for ever. "))
            parts.append(
                f"<span style='color:{colors.force}'><b>Paused:</b> {where}"
                + ("Set the target outside the unstable range" if self.onset is not None and math.isfinite(self.onset.end)
                   else "Lower the target below the onset") + " and Reset.</span>")
        self.readout.setText(" · ".join(parts))

    def _heavy_angle(self) -> float | None:
        """The heavy spot's angle now (rad), or None if it is hidden or there is no unbalance."""
        net = self.sim.system.mass_unbalance
        if not self.view.show_heavy or not net:
            return None
        return self.sim.phase + math.atan2(net.imag, net.real)

    # ------------------------------------------------------------- static sag
    def _set_sag(self, system: RotorSystem) -> None:
        """The static sag at the history's columns, at a grid of speeds (one row if it does not vary)."""
        self.show_sag.setEnabled(system.gravity > 0.0)
        if not system.gravity:
            self._sag = None
            return
        speeds = np.linspace(0.0, self.top_rpm / RPM, SAG_SPEEDS if system.sag_varies else 1)
        qs = system.static_deflection(speeds)
        self._sag = (speeds, self._columns(qs, speeds, speeds)[:, :8])

    def _sag_at(self, omega: np.ndarray) -> np.ndarray:
        """The static sag (len(omega), 8) in the history's station columns."""
        speeds, cols = self._sag
        if speeds.size == 1:
            return np.broadcast_to(cols[0], (np.size(omega), 8))
        return np.column_stack([np.interp(np.abs(omega), speeds, c) for c in cols.T])

    def _vibration(self, y: np.ndarray) -> np.ndarray:
        """History rows' station columns less the static sag: the vibration."""
        return y[:, :8] if self._sag is None else y[:, :8] - self._sag_at(y[:, OMEGA])

    def _shown(self, y: np.ndarray) -> np.ndarray:
        """History rows' station columns as drawn: with the sag or without, as ticked."""
        return y[:, :8] if self.show_sag.isChecked() else self._vibration(y)

    def _sag_text(self, system: RotorSystem) -> str:
        if not system.gravity:
            return ""
        q = system.static_deflection(0.0)
        return (f"<br>Static sag at rest: disc {_fmt_len(-q[YD])}, journals {_fmt_len(-q[YA])} (A) and "
                f"{_fmt_len(-q[YB])} (B)")

    def _draw_spectrum(self) -> None:
        """The full spectrum at the strip chart's station over the strip chart's window."""
        t, hist = self.history.window(self.window.value())
        st = self.station.currentIndex()
        freqs, amps = full_spectrum(t, self._vibration(hist)[:, 2 * st : 2 * st + 2])
        omega = self.sim.omega
        self.spectrum.update_spectrum(freqs, amps, omega, whirl_modes(self.sim.system, omega), DAMPED,
                                      top=(3 if self.sim.system.spacing == 1 else 2) if self.sim.system.periodic else 1)
        self.spectrum_note.setText(
            f"Full spectrum of the {STATIONS[st].lower()} orbit over the last {self.window.value():g} s "
            "(the strip chart's station and window): forward whirl on the right, backward on the left.")

    # ------------------------------------------------------------- waterfall
    def _new_waterfall(self) -> None:
        """Start the spectra afresh: for a new rotor, speed range or block length, or on Clear."""
        f_max = max(WATERFALL_RANGE * self.top_rpm / 60.0, 50.0)
        self.waterfall = Waterfall(self.wf_block.value(), f_max, self.sim.system)
        self._waterfall_drawn = False
        self._draw_waterfall()

    def _draw_waterfall(self, only_new: bool = False) -> None:
        """Draw the waterfall if its tab is showing (with `only_new`, only if blocks came in since)."""
        if self.tabs.currentWidget() is not self.waterfall_plot.parentWidget():
            return
        if only_new and self._waterfall_drawn:
            return
        self._waterfall_drawn = True
        wf = self.waterfall
        mode = self.wf_against.currentIndex()
        blocks = wf.latest_ramp(up=mode == 0) if against_speed(mode) else wf.blocks
        ys, label = baselines(blocks, mode)
        st = self.wf_station.currentIndex()
        log = self.wf_scale.currentIndex() == 0
        tallest = self.waterfall_plot.set_blocks(wf.freqs, blocks, ys, st, log, self.wf_marks.isChecked(), label)
        where = f"the {STATIONS[st].lower()} orbit"
        if tallest is None:
            if against_speed(mode):
                text = (f"No {'run-up' if mode == 0 else 'coast-down'} since the spectra were cleared: change "
                        "the target speed, or run up and coast down, and the spectra are drawn at each speed. "
                        "Spectra taken at a steady speed are shown against time.")
            else:
                text = f"The first {wf.block:g} s block is being taken."
            self.waterfall_note.setText(text)
            return
        b = tallest.block
        at = f"{b.omega * RPM:,.0f} rpm" if against_speed(mode) else f"{b.t:.1f} s ({b.omega * RPM:,.0f} rpm)"
        order = f" ({tallest.freq / (b.omega / (2 * math.pi)):+.2f}×)" if b.omega > 0 else ""
        self.waterfall_note.setText(
            f"{len(blocks)} full spectra of {where}, each of a {wf.block:g} s block ({1 / wf.block:.2g} Hz "
            f"resolution), overlapping by half. Tallest peak {_fmt_len(tallest.amp)} at "
            f"{tallest.freq:+.1f} Hz{order}, at {at}"
            + (": heights are in decades above 1/1000 of it (60 dB)." if log else ".")
            + ("" if self.repeat_taps.isChecked() else
               " Tap the disc, or tick <i>Tap the disc every</i>, to ring the free whirl."))


def lightly_damped(result: Campbell) -> list[tuple[float, float]]:
    """(speed in rad/s, whirl ratio) of the 1X crossings whose mode is damped less than DAMPED."""
    return [(w, r) for w, r, z in result.criticals if z <= DAMPED]


def critical_speeds(result: Campbell, isotropic: bool) -> list[tuple[float, float]]:
    """The 1X crossings the unbalance drives: lightly damped, and forward if the rotor is isotropic."""
    return [(w, r) for w, r in lightly_damped(result) if not isotropic or r > 0.1]


def _isotropic(system: RotorSystem) -> bool:
    """True if the supports act alike in x and y, so that unbalance drives only forward whirl.

    A magnetic bearing does (the same controller on both axes), though its controller's
    states keep the rotor from being time-invariant in the rotating frame (isotropic_supports).
    """
    if not system.magnets:
        return system.isotropic_supports
    plain = [Bearing(mass=b.mass) if b.magnet is not None else b for b in (system.bearing_a, system.bearing_b)]
    return system.with_(bearing_a=plain[0], bearing_b=plain[1]).isotropic_supports


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


class PositionBox(QtWidgets.QDoubleSpinBox):
    """The disc's position a/L: between the bearings or overhung past either, stepping over the bearings."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setRange(-1.0, 2.0)
        self.setDecimals(3)
        self.setSingleStep(0.05)
        self.setKeyboardTracking(False)

    def stepBy(self, steps: int) -> None:  # noqa: N802 (Qt override)
        super().stepBy(steps)
        if _off_bearing(self.value()) != self.value():
            super().stepBy(1 if steps > 0 else -1)


def _same_rotor(a: RotorSystem, b: RotorSystem) -> bool:
    """True if the two rotors are the same up to round-off (the panels' values pass through decimals)."""
    def close(u, v) -> bool:
        if u is None or v is None:
            return u is v
        if isinstance(u, (Bearing, JournalBearing, MagneticBearing, Crack)):
            # A film's own load is not used on the rotor: each bearing carries its share of the weight.
            return type(u) is type(v) and all(close(getattr(u, f.name), getattr(v, f.name)) for f in fields(u)
                                              if not (isinstance(u, JournalBearing) and f.name == "load"))
        if isinstance(u, tuple):
            return len(u) == len(v) and all(close(x, y) for x, y in zip(u, v))
        return cmath.isclose(u, v, rel_tol=1e-9, abs_tol=1e-15)

    # A massless shaft is the same however it is meshed.
    skip = {"elements", "lumped"} if a.density == b.density == 0.0 else set()
    return all(close(getattr(a, f.name), getattr(b, f.name)) for f in fields(RotorSystem) if f.name not in skip)


def _off_bearing(position: float) -> float:
    """a/L moved off a bearing: the disc cannot sit on one (it would have no shaft on one side)."""
    for bearing in (0.0, 1.0):
        if abs(position - bearing) < BEARING_GAP:
            inward = 0.5 - bearing  # toward midspan, if it is right on the bearing
            return bearing + math.copysign(BEARING_GAP, position - bearing if position != bearing else inward)
    return position
