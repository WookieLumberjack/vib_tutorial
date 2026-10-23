"""Jeffcott rotor on flexible bearings: model, whirl analysis and exact time stepping.

A massless, uniform Euler–Bernoulli shaft runs through two bearings, A at z = 0
and B at z = L, and carries a rigid disc at z = a: between the bearings
(0 < a < L), or overhung past one of them (a > L past B, a < 0 past A), the
shaft then running on from the bearing to the disc. The shaft bends in
two planes, x (horizontal) and y (vertical). Each plane has four degrees of
freedom:

    the bearing journal A's displacement, the disc's displacement, the disc's
    slope (dx/dz or dy/dz), and the bearing journal B's displacement,

so q = [x_A, x_D, θx, x_B, y_A, y_D, θy, y_B] (m, m, rad, m, ...). θx = dx/dz
and θy = dy/dz are the slopes of the shaft at the disc; with these slopes the
two planes have the same matrices. The bearings are pinned (they take no
moment), so the slopes at A and B are condensed out of the shaft's two beam
elements (A to the disc and the disc to B, or for an overhung disc the span and
the overhang).

Each journal has a mass and is held to ground by a spring and a damper in x and
in y, which may differ (anisotropic supports) and may differ between A and B,
with cross terms (k_xy, k_yx, c_xy, c_yx) too. Or it runs in a plain journal
bearing, an oil film on a rigid housing (core/fluid_bearing.py): its stiffness
and damping then come from the film, under the bearing's share of the weight,
and change with the speed. The film is linearized about where its journal
settles, so with gravity it also adds a constant load that holds the journal
there (`_film_terms`): the rotor rests with its journals on the film's locus.
So in general the stiffness and the damping are K(Ω) and C(Ω), not only K + ΩH.
Or it floats in an active magnetic bearing (core/magnetic_bearing.py): the
magnets' negative stiffness -k_m is in K, and a PID controller behind an
amplifier sets the current that holds the journal. The controllers have states
of their own, x_c, after q and q' in the state (n_states long); the steady
states take the bearings' dynamic stiffness k_i G(s) at each harmonic instead
(support_stiffness), and with integral action a steady load cannot move the
journals at all.
The disc has mass m, polar moment of inertia Ip (about the spin axis) and
diametral moment Id (about a diameter).

The shaft may also have mass (density ρ > 0), which the Jeffcott model leaves
out. It is then a beam finite element model: `elements` Euler–Bernoulli
elements along it (Rayleigh beams: the sections' rotary inertia and their own
gyroscopic moment included), with a consistent or a lumped mass matrix. Every
node has a displacement and a slope in each plane, so the model has many more
DOFs, and its highest modes are far above anything the page shows (tens of
kHz, which the time stepping would have to resolve). So it is reduced, Craig–
Bampton style: the four stations' DOFs are kept as they are, and the rest of
the shaft moves as their static shape (exactly what the massless shaft does)
plus its lowest few modes with the stations held (the fixed-interface modes,
up to MODE_LIMIT). q is then

    [x_A, x_D, θx, x_B, y_A, y_D, θy, y_B, ξ_x1, ..., ξ_xr, ξ_y1, ..., ξ_yr],

the stations first, as before, then the r modal coordinates of each plane. With
a massless shaft r = 0 and the reduction is exactly the condensation above.

The rotor spins at Ω (rad/s) about +z, turning x toward y. A tilted, spinning
disc has angular momentum Ip Ω along its tilted axis; turning that axis takes a
moment, the gyroscopic moment. It enters the equations of motion through the
velocities:

    M q'' + (C + Ω G) q' + (K + Ω H) q = f(t)

G is skew-symmetric (G^T = -G), so q'^T G q' = 0: the gyroscopic forces do no
work and dissipate nothing, though they sit next to the damping matrix. They
couple the two planes' slopes, and make the natural frequencies depend on Ω:
each whirl mode splits into a forward (with the spin) and a backward (against
it) branch, the Campbell diagram.

Two terms can make the rotor unstable. A bearing (or seal) may have
cross-coupled stiffness: a displacement in x pushes the journal in y and vice
versa, k_xy = -k_yx, so K is not symmetric. In a fluid-film bearing that
coupling comes from the oil being dragged round at a fraction λ of the shaft's
speed, so it grows with the speed, as λ c Ω on top of any constant k_xy (c the
bearing's damping). With λ = 1/2, the forward whirl is fed faster than the
bearing damps it once Ω passes twice its frequency: oil whirl and whip. And the
shaft may have internal
(rotating) damping, which resists the bending rate seen by the spinning shaft,
q' - Ω J q in fixed coordinates (J turns x toward y). Part of it is ordinary
damping in C; the rest is Ω H, a skew stiffness growing with the speed, which
also holds the bearings' λ c. Both
skew stiffnesses push a forward whirl along its path, feeding it energy; above
the speed where that outweighs the damping, the whirl grows by itself.

Angles on the rotor are measured from a keyphasor mark on the shaft, at angle
φ in fixed coordinates (φ' = Ω), in the direction of spin. Three sources turn
with the shaft and drive it once per revolution (1X):

- the disc's mass unbalance U = m e (kg·m): its centre of mass is a distance e
  from the shaft's centre at angle φ + ψ (ψ = 0 by default: the heavy spot is at
  the mark). Keeping the centre of mass on its path takes

      f_x = U (Ω² cos φ + Ω' sin φ),   f_y = U (Ω² sin φ - Ω' cos φ)

  on the disc for ψ = 0, or in general f_x + i f_y = U e^{iψ} (Ω² - iΩ') e^{iφ}.
  Balance weights m_k r_k at angles θ_k add to U e^{iψ} the same way. A weight
  at an axial offset h from the disc's centre plane adds a moment w h too; one
  on the shaft at z is a point force there (two-plane balancing).
- the disc's skew τ (rad): its principal axis is tilted from the shaft's by τ,
  toward angle φ + γ. The disc's inertia acts on its own axis, the shaft's slope
  plus the skew, so the skew moves to the right side as a moment on the slopes,
  M_x + i M_y = (I_d - I_p) τ (Ω² - iΩ') e^{i(φ+γ)}: a couple unbalance. A thin
  disc (I_p > I_d) is pushed flat, toward square to the spin axis.
- the shaft's bow: a residual bend, δ at the disc toward angle φ + β, in the
  shape a load at the disc gives with the bearings held (q_bow). The shaft's
  elastic force is K_shaft (q - q_bow(φ)), so the bow is a force K_shaft q_bow
  turning with the shaft, constant in size. It loads only the shaft: its
  forces on the disc and the journals balance, so at rest the shaft takes its
  bent shape without loading the bearings.

The internal damping acts on q - q_bow too, but the bow does not move in the
shaft, so it adds nothing there: C_i q_bow' + Ω H q_bow = 0.

On a horizontal rotor gravity pulls the disc and the journals down (-y), a
constant force f_g. It moves the equilibrium to the static deflection
q_s = (K + Ω H)^{-1} f_g, about which the rotor vibrates exactly as without it
(the model is linear): the static sag, sideways too where K + ΩH is not
symmetric. The tracking and the spectra work on q - q_s, the vibration.

The shaft may be asymmetric (a two-pole generator's rotor, slotted along two
sides): EI(1 + Δ) bending toward its stiff axis, at angle φ + α, and EI(1 - Δ)
across it. Its stiffness then turns with it, K + ΔK(φ) with ΔK changing at 2X,
and the equations have a periodic coefficient. Turning with the shaft on
isotropic supports they are time-invariant, so its modes are exact there
(_rotating_modes); otherwise they are found by Hill's method (_hill_modes).
Either way they are Floquet modes. Its steady states come by harmonic balance
(harmonic_response): a 1X load stays 1X on isotropic supports, and gravity gives
a mean sag and a 2X whirl (gravity_response), largest near half a critical
speed. Between the critical speeds of its two axes it may be unstable.

The shaft may be cracked (`crack`): a straight-fronted transverse crack, its
mouth toward angle φ + β_c, cut to a depth a from the surface. Where it is open it
is a hinge: the shaft's slope jumps across it by c M (M the bending moment
there), c its compliance from fracture mechanics (crack_compliance), large
bending that opens it and small across. The crack is condensed out exactly, a
rank-one loss of stiffness on the stations (_crack_deficits). A breathing crack
is opened and closed by the shaft's weight alone (weight dominance, as in Mayes
and Davies' model): fully open where the weight's bending puts its mouth in
tension, shut half a turn later, open by (1 + cos)/2 between. So its stiffness
is periodic in φ with 1X, 2X and 3X in it, not only the asymmetric shaft's 2X,
and even on isotropic supports it changes in axes turning with the shaft (the
weight does not turn), so its modes come by Hill's method and its steady states
by harmonic balance over every harmonic. Its weight then drives 1X, 2X and 3X,
the last two most near a half and a third of a critical speed. An open crack
(not breathing, a notch) turns with the shaft as the asymmetric shaft does.

Time stepping uses the state z = [q, q'], z' = A(Ω) z + B u, with the same exact
first-order-hold matrix-exponential update as the chain's Simulator. A depends
on Ω, so while the speed ramps A is held at the speed at the middle of each short
chunk of steps (CHUNK_TIME), and the unbalance force follows the exact phase.
An asymmetric or cracked shaft's turning stiffness is split off each step as a
kick to the velocities at mid-step, between two exact half steps of the rest.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import cached_property, lru_cache
from typing import NamedTuple

import numpy as np
import scipy.linalg

from . import native
from .fluid_bearing import BearingState, JournalBearing, full_film_damping, pi_film
from .magnetic_bearing import MagneticBearing
from .simulator import MAX_STEP, MAX_STEPS_PER_ADVANCE, STEPS_PER_PERIOD, foh_discretize

N_DOF = 8  # the stations' DOFs, the first of q (a shaft with mass adds its modal coordinates after them)
# Index of each station DOF in q.
XA, XD, TX, XB, YA, YD, TY, YB = range(N_DOF)
DOF_NAMES = ("x_A", "x_D", "θx", "x_B", "y_A", "y_D", "θy", "y_B")
STEEL = 7850.0  # kg/m³
MODE_LIMIT = 1000.0  # Hz: the shaft's fixed-interface modes are kept up to this frequency (the top speed: 60,000 rpm)
MAX_MODES = 12  # and at most this many in each plane
CHUNK_TIME = 2e-3  # s; while the speed ramps, A(Ω) is held over chunks this long
HARMONICS = 3  # an asymmetric shaft on anisotropic supports: harmonics kept each side of the 1X (or 0X) one
HILL_ORDER = 3  # and harmonics of 2Ω (Ω with a breathing crack) kept each side in Hill's method for its modes
RPM = 60.0 / (2.0 * math.pi)  # rpm per rad/s
MIN_GAP = 1e-3  # of the span: the closest the disc may come to a bearing
POISSON = 0.3  # the shaft's Poisson's ratio, for a crack's compliance
CRACK_DEPTH_MAX = 0.5  # a / D: the strip model's crack compliance holds to half the shaft's diameter
# Nonlinear oil films (RotorSystem.nonlinear_films) in the time step, see RotorSimulator._film_steps:
OVERDAMPED = 0.05  # a root with |Im λ| below this times |Re λ| (ζ > 0.9987) is taken as not oscillating
FILM_MODE_STEPS = 10  # steps per period of the fastest mode (the spin keeps STEPS_PER_PERIOD)
FILM_DRIFT = 0.3  # the films' damping bound is taken afresh once it has moved this much (of its largest term)
FILM_TOL = 1e-7  # Newton stops when the remainder's residual is this small, over (1 N + the largest load)
FILM_ITERATIONS = 12  # and after this many iterations at most
FILM_CORRECTION = 1e-4  # a Newton correction this small (over the same) is taken without checking: it converges


def _edge_crack_factor(s: np.ndarray, bending: bool) -> np.ndarray:
    """F(a/h): K = σ √(πa) F for an edge crack of depth a in a strip of height h (Tada's fits, 0.5%)."""
    x = 0.5 * math.pi * s
    root = np.sqrt(np.tan(x) / x) / np.cos(x)
    if bending:
        return root * (0.923 + 0.199 * (1.0 - np.sin(x)) ** 4)
    return root * (0.752 + 2.02 * s + 0.37 * (1.0 - np.sin(x)) ** 3)


# Speeds a system keeps its films' states for (an analysis refresh asks for ~5,000), and their
# terms (whole matrices; a sweep asks for each a few times in a row).
CACHE_STATES = 8192
CACHE_TERMS = 64


def _cached[T](system: RotorSystem, name: str, key: float, make: Callable[[], T], size: int) -> T:
    """`make()` for speed `key`, kept in the system's cache `name`, the least recently used dropped first."""
    cache = system.__dict__.setdefault(name, OrderedDict())
    try:
        cache.move_to_end(key)
        return cache[key]
    except KeyError:
        value = cache[key] = make()
        if len(cache) > size:
            cache.popitem(last=False)
        return value


@lru_cache(maxsize=64)
def crack_compliance(depth: float) -> tuple[float, float]:
    """A straight-fronted crack's bending compliance in a round shaft, per (1 - ν²)/(E R³): (opening, across).

    Its depth is a/D, from the surface (Dimarogonas and Papadopoulos). The crack is
    cut into strips across its front, each an edge crack in a strip as tall as the
    shaft there; the energy released as the crack grows, ∫ (1 - ν²) K²/E dA, gives
    the extra rotation c M across the crack under a bending moment M. Bending that
    pulls the crack's mouth open (the moment along its front) loads every strip in
    bending; across, each strip is in tension or compression by its distance off
    the middle. So the crack gives much more to the first, until it is deep.
    """
    if depth <= 0.0:
        return 0.0, 0.0
    radius, a = 1.0, 2.0 * depth
    half = math.sqrt(a * (2.0 * radius - a))  # half the crack front's length
    inertia = math.pi * radius**4 / 4.0
    x, w = np.polynomial.legendre.leggauss(96)
    xi, wx = half * x, half * w
    chord = np.sqrt(radius**2 - xi**2)
    height = 2.0 * chord  # each strip's height, and its crack's depth
    deep = chord - (radius - a)
    alpha = 0.5 * deep[:, None] * (x[None, :] + 1.0)
    wa = 0.5 * deep[:, None] * w[None, :]
    s = alpha / height[:, None]
    opening = (0.5 * height[:, None] / inertia) ** 2 * math.pi * alpha * _edge_crack_factor(s, True) ** 2
    across = (xi[:, None] / inertia) ** 2 * math.pi * alpha * _edge_crack_factor(s, False) ** 2
    return float(2.0 * wx @ np.sum(wa * opening, axis=1)), float(2.0 * wx @ np.sum(wa * across, axis=1))


@dataclass(frozen=True)
class Crack:
    """A transverse crack in the shaft, at z = position L, its mouth at `angle` from the keyphasor mark.

    Breathing, it is opened and closed by the shaft's weight as it turns; not
    breathing, it is held open (as a sawn notch). `compliance` (rad/(N·m), opening
    and across) overrides the one its depth gives.
    """

    position: float = 0.5  # z / L, as the disc's position (past 0 or 1 on an overhang)
    depth: float = 0.3  # a / D, from the shaft's surface
    angle: float = 0.0  # rad, of the crack's mouth from the keyphasor mark, in the direction of spin
    breathing: bool = True
    compliance: tuple[float, float] | None = None


@dataclass(frozen=True)
class Bearing:
    """A bearing journal on its support: springs and dampers to ground in x and y, an oil film or a magnet.

    The support pushes back on the journal with f = -K [x, y] - C [x', y'], K =
    [[kx, k_xy], [k_yx, ky]] and C = [[cx, c_xy], [c_yx, cy]]. With a `film` the
    journal runs in a plain journal bearing on a rigid housing instead: K and C
    are then the film's, at the speed and under the bearing's share of the
    rotor's weight (the film's own `load` is not used), and the springs, dampers
    and swirl are left out. With a `magnet` it floats in an active magnetic
    bearing (core/magnetic_bearing.py): K is then the magnets' negative stiffness
    -k_m, and the rest of the push comes from the controller's states.
    """

    kx: float = 5e4  # N/m
    ky: float = 5e4  # N/m
    cx: float = 100.0  # N·s/m
    cy: float = 100.0  # N·s/m
    mass: float = 0.1  # kg, the journal and whatever moves with it
    kxy: float = 0.0  # N/m, cross-coupled stiffness: f_x = -k_xy y
    swirl: float = 0.0  # fluid swirl ratio λ: the cross-coupling grows with speed by λ c Ω (see damping)
    kyx: float | None = None  # N/m: f_y = -k_yx x; None for -k_xy (a pure skew coupling, as a seal's)
    cxy: float = 0.0  # N·s/m: f_x = -c_xy y'
    cyx: float = 0.0  # N·s/m: f_y = -c_yx x'
    film: JournalBearing | None = None  # an oil film in place of the springs and dampers
    magnet: MagneticBearing | None = None  # or an active magnetic bearing

    def __post_init__(self) -> None:
        if self.film is not None and self.magnet is not None:
            raise ValueError("a bearing is an oil film or a magnet, not both")

    @property
    def damping(self) -> float:
        """c (N·s/m), the mean of c_x and c_y: the fluid's drag-round takes k_xy up by λ c Ω."""
        return 0.5 * (self.cx + self.cy)

    @property
    def k_yx(self) -> float:
        return -self.kxy if self.kyx is None else self.kyx

    @property
    def swirling(self) -> bool:
        """True if the springs' cross-coupling grows with the speed (λ c ≠ 0, and no film or magnet)."""
        return self.film is None and self.magnet is None and bool(self.swirl * self.damping)

    def springs(self) -> tuple[np.ndarray, np.ndarray]:
        """(K, C), each 2 x 2: the springs and dampers at rest (none with a film; a magnet's -k_m)."""
        if self.film is not None:
            return np.zeros((2, 2)), np.zeros((2, 2))
        if self.magnet is not None:
            return -self.magnet.negative_stiffness * np.eye(2), np.zeros((2, 2))
        return (np.array([[self.kx, self.kxy], [self.k_yx, self.ky]]),
                np.array([[self.cx, self.cxy], [self.cyx, self.cy]]))


class PlaneWeight(NamedTuple):
    """A balance weight in a plane at axial position z (m from bearing A): m r e^{iθ} (kg·m) from the mark.

    On the disc (`disc`), it sits on the rigid disc at an axial offset z - a from its
    centre plane, so it adds w to the disc's unbalance and w (z - a) to its couple.
    On the shaft, it is a point force on the shaft at z.
    """

    z: float
    weight: complex
    disc: bool = False


@dataclass(frozen=True)
class RotorSystem:
    """A Jeffcott rotor: shaft, disc, two bearings and the disc's unbalance. SI units."""

    length: float = 0.5  # m, between the bearings
    diameter: float = 0.01  # m, of the shaft
    youngs: float = 200e9  # Pa
    position: float = 0.5  # disc position a / L (0.5: midspan; > 1 overhung past B, < 0 past A)
    disc_mass: float = 1.0  # kg
    ip: float = 1.25e-3  # kg·m², polar (about the spin axis)
    id: float = 6.25e-4  # kg·m², diametral (about a diameter)
    bearing_a: Bearing = field(default_factory=Bearing)
    bearing_b: Bearing = field(default_factory=Bearing)
    unbalance: float = 20e-6  # kg·m (20 g·mm)
    unbalance_angle: float = 0.0  # rad, ψ: the heavy spot's angle from the keyphasor mark
    internal_damping: float = 0.0  # N·s/m, the shaft's rotating damping, as a damper at the disc
    bow: float = 0.0  # m, the shaft's residual bow at the disc
    bow_angle: float = 0.0  # rad, of the bow from the keyphasor mark (in the direction of spin)
    skew: float = 0.0  # rad, the disc's principal axis's tilt from the shaft's
    skew_angle: float = 0.0  # rad, of the skew's direction from the keyphasor mark
    weights: tuple[complex, ...] = ()  # balance weights on the disc, each m r e^{iθ} (kg·m) from the mark
    plane_weights: tuple[PlaneWeight, ...] = ()  # weights in planes along the rotor (two-plane balancing)
    gravity: float = 0.0  # m/s², pulling in -y (9.81 for a horizontal rotor; 0 for a vertical one)
    density: float = 0.0  # kg/m³ of the shaft: 0 for the Jeffcott model's massless shaft (steel: STEEL)
    elements: int = 8  # beam elements along the shaft, shared between its stretches by length
    lumped: bool = False  # the shaft's mass lumped at the nodes, rather than the consistent mass matrix
    asymmetry: float = 0.0  # Δ: the shaft's EI is EI(1 + Δ) bending toward its stiff axis, EI(1 - Δ) across it
    asymmetry_angle: float = 0.0  # rad, α: the stiff axis's angle from the keyphasor mark
    crack: Crack | None = None  # a transverse crack in the shaft
    nonlinear_films: bool = False  # the time response takes the oil films' whole force, not their linearization

    def __post_init__(self) -> None:
        if min(abs(self.position), abs(self.position - 1.0)) < MIN_GAP:
            raise ValueError("the disc cannot sit on a bearing (a/L = 0 or 1)")
        positive = (self.length, self.diameter, self.youngs, self.disc_mass, self.id,
                    self.bearing_a.mass, self.bearing_b.mass)
        if min(positive) <= 0.0:
            raise ValueError("lengths, Young's modulus, masses and Id must be positive")
        if self.internal_damping < 0.0:
            raise ValueError("the internal damping cannot be negative")
        if min(self.bearing_a.swirl, self.bearing_b.swirl) < 0.0:
            raise ValueError("the fluid swirls with the shaft: the swirl ratio cannot be negative")
        if self.gravity < 0.0:
            raise ValueError("gravity pulls down (-y): give its size")
        if self.bow < 0.0 or self.skew < 0.0:
            raise ValueError("the bow and the skew are sizes; give their direction as an angle")
        if self.density < 0.0:
            raise ValueError("the shaft's density cannot be negative")
        if not 0.0 <= self.asymmetry < 1.0:
            raise ValueError("the asymmetry Δ is in [0, 1): EI(1 - Δ) about the soft axis must stay positive")
        if self.elements < 2:
            raise ValueError("the shaft needs at least two elements, one each side of the disc")
        lo, hi = self.extent
        if any(not p.disc and not lo <= p.z <= hi for p in self.plane_weights):
            raise ValueError("balance weights go on the shaft, between its ends")
        if self.crack is not None:
            if not 0.0 <= self.crack.depth <= CRACK_DEPTH_MAX:
                raise ValueError(f"the crack's depth a/D is in [0, {CRACK_DEPTH_MAX}]")
            if not lo < self.crack.position * self.length < hi:
                raise ValueError("the crack is in the shaft, between its ends")

    def with_(self, **changes) -> RotorSystem:
        return replace(self, **changes)

    def on_disc(self, p: PlaneWeight) -> bool:
        """True if the plane weight p is on the disc: put there, or on the shaft at the disc's centre plane."""
        return p.disc or abs(p.z - self.a) <= 1e-9 * self.length

    @property
    def mass_unbalance(self) -> complex:
        """The disc's net mass unbalance (kg·m), the unbalance and the weights: |·| m e, its angle the heavy spot's.

        Plane weights on the disc count, whatever their axial offset.
        """
        return self.unbalance * complex(math.cos(self.unbalance_angle), math.sin(self.unbalance_angle)) + sum(
            self.weights, 0j) + sum((p.weight for p in self.plane_weights if self.on_disc(p)), 0j)

    @property
    def couple_unbalance(self) -> complex:
        """The disc's net couple unbalance (kg·m²), turning with the shaft: the skew's and the plane weights'.

        Its moment on the disc's slopes is this times (Ω² - iΩ') e^{iφ}.
        """
        return (self.id - self.ip) * self.skew * np.exp(1j * self.skew_angle) + sum(
            (p.weight * (p.z - self.a) for p in self.plane_weights if self.on_disc(p)), 0j)

    # ------------------------------------------------------------- the shaft
    @property
    def ei(self) -> float:
        """Bending stiffness EI (N·m²) of the round shaft."""
        return self.youngs * math.pi * self.diameter**4 / 64.0

    @property
    def a(self) -> float:
        """Disc position from bearing A (m)."""
        return self.position * self.length

    @property
    def b(self) -> float:
        """Disc position from bearing B (m), L - a: negative when the disc overhangs B."""
        return self.length - self.a

    @property
    def overhung(self) -> bool:
        """True if the disc sits outside the bearings rather than between them."""
        return not 0.0 < self.position < 1.0

    @property
    def overhang(self) -> float:
        """The disc's distance past the nearer bearing (m), 0 between the bearings."""
        return max(-self.a, self.a - self.length, 0.0)

    @property
    def area(self) -> float:
        """The shaft's cross-section (m²)."""
        return math.pi * self.diameter**2 / 4.0

    @property
    def shaft_mass(self) -> float:
        """The whole shaft's mass (kg), 0 for the Jeffcott model's massless shaft."""
        lo, hi = self.extent
        return self.density * self.area * (hi - lo)

    @cached_property
    def _mesh(self) -> tuple[np.ndarray, tuple[int, int, int]]:
        """The shaft's nodes' z (m) in axial order, and the nodes at bearing A, the disc and bearing B.

        Each stretch between the stations gets elements in proportion to its length, at
        least one. A crack has a node of its own, unless it is next to a station.
        """
        stations = [(0.0, 0), (self.a, 1), (self.length, 2)]
        if self.crack is not None and min(abs(self.crack_z - z) for z, _ in stations) > MIN_GAP * self.length:
            stations.append((self.crack_z, 3))
        stations.sort()
        span = stations[-1][0] - stations[0][0]
        z, node = [stations[0][0]], {stations[0][1]: 0}
        for (z0, _), (z1, n1) in zip(stations, stations[1:]):
            k = max(1, round(self.elements * (z1 - z0) / span))
            z.extend(np.linspace(z0, z1, k + 1)[1:])
            node[n1] = len(z) - 1
        return np.array(z), (node[0], node[1], node[2])

    @property
    def crack_z(self) -> float:
        """The crack's axial position (m from bearing A); NaN without one."""
        return math.nan if self.crack is None else self.crack.position * self.length

    @cached_property
    def _assembled(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(K, M, R): one plane's whole shaft, on every node's [w, θ] (see _reduced)."""
        z = self._mesh[0]
        n = 2 * z.size
        K, M, R = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
        rho_a, rho_i = self.density * self.area, self.density * math.pi * self.diameter**4 / 64.0
        for e, length in enumerate(np.diff(z)):
            d = slice(2 * e, 2 * e + 4)
            translation, rotation = beam_mass(rho_a, rho_i, length, self.lumped)
            K[d, d] += beam_stiffness(self.ei, length)
            M[d, d] += translation + rotation
            R[d, d] += rotation
        return K, M, R

    @property
    def _station_dofs(self) -> list[int]:
        """The stations' [w_A, w_D, θ_D, w_B] in the whole shaft's DOFs."""
        na, nd, nb = self._mesh[1]
        return [2 * na, 2 * nd, 2 * nd + 1, 2 * nb]

    @property
    def crack_compliances(self) -> tuple[float, float]:
        """The crack's compliance (rad/(N·m)): to bending that opens it, and across; zeros without one."""
        if self.crack is None:
            return 0.0, 0.0
        if self.crack.compliance is not None:
            return self.crack.compliance
        scale = (1.0 - POISSON**2) / (self.youngs * (0.5 * self.diameter) ** 3)
        c_open, c_across = crack_compliance(self.crack.depth)
        return c_open * scale, c_across * scale

    @cached_property
    def _crack_hinge(self) -> tuple[np.ndarray, float]:
        """(b (4,), s): the crack as a hinge, a jump Δθ in the shaft's slope there, condensed onto the stations.

        The shaft's energy on [stations, Δθ], everything else free (statically), is
        ½ [q; Δθ]^T [[K_ss, b], [b^T, s]] [q; Δθ]: b is the bending moment at the
        crack the stations' displacements make (per unit of each), s the shaft's
        stiffness against the jump with the stations held.
        """
        z, _ = self._mesh
        K = self._assembled[0]
        n = K.shape[0]
        node = int(np.argmin(np.abs(z - self.crack_z)))
        # The jump turns the element past the crack at its near end (or, at the shaft's end, the one before it).
        e, j = (node, 1) if node < z.size - 1 else (node - 1, 3)
        ke = beam_stiffness(self.ei, z[e + 1] - z[e])
        g = np.arange(2 * e, 2 * e + 4)
        Kx = np.zeros((n + 1, n + 1))
        Kx[:n, :n] = K
        Kx[n, g] += ke[j]
        Kx[g, n] += ke[:, j]
        Kx[n, n] += ke[j, j]
        kept = self._station_dofs + [n]
        inner = [i for i in range(n + 1) if i not in kept]
        Kc = Kx[np.ix_(kept, kept)] - Kx[np.ix_(kept, inner)] @ np.linalg.solve(
            Kx[np.ix_(inner, inner)], Kx[np.ix_(inner, kept)])
        return Kc[:4, 4].copy(), float(Kc[4, 4])

    def _hinge_loss(self, compliance: float) -> np.ndarray:
        """One plane's stiffness lost (4 + r square) to a hinge of this compliance at the crack.

        Condensing the jump out, the stations' stiffness is K_ss - b b^T c/(1 + c s):
        exact for the massless shaft. With the shaft's mass its fixed-interface modes
        are those of the uncracked shaft (the crack's own inertia is left out).
        """
        loss = np.zeros((4 + self.modes_kept,) * 2)
        if compliance > 0.0:
            b, s = self._crack_hinge
            loss[:4, :4] = np.outer(b, b) * compliance / (1.0 + compliance * s)
        return loss

    @cached_property
    def crack_sign(self) -> int:
        """Where the weight opens a breathing crack: +1 its mouth down (the shaft sags there), -1 up, 0 never.

        The sign of the shaft's curvature at the crack under its weight, the journals
        held (the bending moment of a shaft on two bearings does not depend on them).
        """
        if self.crack is None or not self.gravity:
            return 0
        z, (na, nd, nb) = self._mesh
        K, M, _ = self._assembled
        n = K.shape[0]
        f = -M @ np.tile([1.0, 0.0], z.size)
        f[2 * nd] -= self.disc_mass
        free = [i for i in range(n) if i not in (2 * na, 2 * nb)]
        u = np.zeros(n)
        u[free] = np.linalg.solve(K[np.ix_(free, free)], f[free])
        node = int(np.argmin(np.abs(z - self.crack_z)))
        e, end = (node, 0.0) if node < z.size - 1 else (node - 1, 1.0)
        length = z[e + 1] - z[e]
        # w'' of the element's cubic at its end, from its [w1, θ1, w2, θ2].
        curve = np.array([12 * end - 6, length * (6 * end - 4), 6 - 12 * end, length * (6 * end - 2)]) / length**2
        w2 = float(curve @ u[2 * e:2 * e + 4])
        scale = float(np.abs(curve) @ np.abs(u[2 * e:2 * e + 4])) + 1e-300
        return 0 if abs(w2) <= 1e-9 * scale else (1 if w2 > 0 else -1)

    @property
    def crack_opening(self) -> float:
        """How open the crack is on average over a turn: 1 held open, ½ breathing, 0 shut (or none)."""
        if self.crack is None or not self.crack.depth and self.crack.compliance is None:
            return 0.0
        if not self.crack.breathing:
            return 1.0
        return 0.5 if self.crack_sign else 0.0

    def crack_open(self, phase: float | np.ndarray) -> np.ndarray:
        """How open the crack is (0 to 1) with the keyphasor mark at angle φ: (1 - s sin θ)/2, θ its mouth's angle."""
        if not self.crack_opening:
            return np.zeros(np.shape(phase))
        if not self.crack.breathing:
            return np.ones(np.shape(phase))
        return 0.5 * (1.0 - self.crack_sign * np.sin(np.asarray(phase, dtype=float) + self.crack.angle))

    @property
    def nodes(self) -> np.ndarray:
        """z (m) of the shaft's finite element nodes, in axial order."""
        return self._mesh[0].copy()

    @cached_property
    def _reduced(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(T, K, M, R, freqs): one plane's shaft, reduced to the stations and its fixed-interface modes.

        The beam elements are assembled on every node's [w, θ]. The stations' DOFs
        [w_A, w_D, θ_D, w_B] are kept; the others (the slopes at the bearings, and
        every node in between) are the static shape the stations give them,
        -K_ii^{-1} K_ib, plus the modes of the shaft with the stations held, mass
        normalized, up to MODE_LIMIT: full = T @ [stations, ξ]. K, M and R are
        T^T (·) T for the shaft's stiffness, mass and rotary inertia (its
        gyroscopic matrix is 2R: a round section's polar inertia is twice its
        diametral). freqs (Hz) are the modes kept.
        """
        K, M, R = self._assembled
        n = K.shape[0]
        kept = self._station_dofs
        inner = [j for j in range(n) if j not in kept]
        static = -np.linalg.solve(K[np.ix_(inner, inner)], K[np.ix_(inner, kept)])
        if self.density > 0.0:
            w2, shapes = scipy.linalg.eigh(K[np.ix_(inner, inner)], M[np.ix_(inner, inner)])
            r = min(MAX_MODES, int(np.sum(w2 <= (2.0 * math.pi * MODE_LIMIT) ** 2)))
            w2, shapes = w2[:r], shapes[:, :r]
            shapes *= np.sign(shapes[np.argmax(np.abs(shapes), axis=0), np.arange(r)])  # largest entry positive
        else:
            w2, shapes = np.empty(0), np.empty((len(inner), 0))
        T = np.zeros((n, 4 + w2.size))
        T[kept, :4] = np.eye(4)
        T[inner, :4] = static
        T[inner, 4:] = shapes
        return T, T.T @ K @ T, T.T @ M @ T, T.T @ R @ T, np.sqrt(w2) / (2.0 * math.pi)

    @property
    def modes_kept(self) -> int:
        """r, the shaft's fixed-interface modes kept in each plane (0 for a massless shaft)."""
        return self._reduced[0].shape[1] - 4

    @property
    def fixed_interface_freqs(self) -> np.ndarray:
        """The frequencies (Hz) of the shaft's modes kept, with the stations held."""
        return self._reduced[4].copy()

    @property
    def n_dof(self) -> int:
        """The length of q: the 8 stations' DOFs and the shaft's modal coordinates in each plane."""
        return N_DOF + 2 * self.modes_kept

    @cached_property
    def _planes(self) -> tuple[np.ndarray, np.ndarray]:
        r = self.modes_kept
        x = np.r_[0:4, N_DOF:N_DOF + r]
        return x, np.r_[4:8, N_DOF + r:N_DOF + 2 * r]

    @property
    def plane_dofs(self) -> tuple[np.ndarray, np.ndarray]:
        """The indices in q of the x plane's DOFs and the y plane's, each [stations, modal coordinates]."""
        x, y = self._planes
        return x.copy(), y.copy()

    def merge(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """q (..., n_dof) from the two planes' parts (..., 4 + r): the inverse of q[..., plane_dofs]."""
        ix, iy = self._planes
        x, y = np.asarray(x), np.asarray(y)
        out = np.empty(np.broadcast_shapes(x.shape, y.shape)[:-1] + (self.n_dof,), dtype=np.result_type(x, y))
        out[..., ix], out[..., iy] = x, y
        return out

    def _blocks(self, xx: np.ndarray, yy: np.ndarray, xy: np.ndarray | None = None,
                yx: np.ndarray | None = None) -> np.ndarray:
        """An n_dof square matrix from its plane blocks: x on x, y on y, x's row on y's column, and back."""
        ix, iy = self._planes
        A = np.zeros((self.n_dof, self.n_dof), dtype=np.result_type(xx, yy))
        A[np.ix_(ix, ix)], A[np.ix_(iy, iy)] = xx, yy
        if xy is not None:
            A[np.ix_(ix, iy)] = xy
        if yx is not None:
            A[np.ix_(iy, ix)] = yx
        return A

    def shaft_stiffness(self) -> np.ndarray:
        """One plane's shaft stiffness on [w_A, w_D, θ_D, w_B, ξ_1, ..., ξ_r]: 4x4 for a massless shaft.

        The stations' block is the shaft's exact static stiffness (it does not depend
        on the elements); each fixed-interface mode adds its ω² alone on the diagonal.
        """
        return self._reduced[1].copy()

    def plane_mass(self) -> np.ndarray:
        """One plane's mass matrix on [stations, ξ]: the shaft's, the journals', the disc's m and I_d."""
        M = self._reduced[2].copy()
        M[[0, 1, 2, 3], [0, 1, 2, 3]] += [self.bearing_a.mass, self.disc_mass, self.id, self.bearing_b.mass]
        return M

    @property
    def disc_stiffness(self) -> float:
        """The shaft's stiffness at the disc with the bearings held (N/m).

        3EIL/(a²b²) between the bearings; 3EI/(c²(L + c)) for a disc overhung by c.
        """
        k = self.shaft_stiffness()[1:3, 1:3]
        return 1.0 / np.linalg.inv(k)[0, 0]

    @property
    def loss_time(self) -> float:
        """η (s): the internal damping is η K_shaft, so that it is c_i on the disc's own stiffness."""
        return self.internal_damping / self.disc_stiffness

    def bow_shape(self) -> np.ndarray:
        """One plane's bow (4 + r,) on [w_A, w_D, θ_D, w_B, ξ], per metre of bow at the disc.

        The shape a load at the disc gives with the bearings held: no moment at the
        disc, so K_shaft times it is a force at the disc and the bearings' reactions.
        It is a static shape, so the modal coordinates are zero.
        """
        k = self.shaft_stiffness()[1:3, 1:3]
        w, theta = np.linalg.solve(k, [1.0, 0.0])
        shape = np.zeros(4 + self.modes_kept)
        shape[1:3] = 1.0, theta / w
        return shape

    def rotating_load(self, omega: float | np.ndarray, alpha: float | np.ndarray = 0.0) -> np.ndarray:
        """The 1X load on one plane's DOFs as f_x + i f_y = P e^{iφ}: P (..., 4 + r) complex.

        The unbalance and the balance weights on the disc, its couple (the skew's and
        the weights' moment on its slopes), the weights on the shaft (each a point
        force, shared over the DOFs as S(z)^T) and the bow's force K_shaft q_bow, at
        speed Ω and acceleration Ω' = alpha.
        """
        drive = np.asarray(omega, dtype=float) ** 2 - 1j * np.asarray(alpha, dtype=float)
        per_drive, bow = self._load_parts
        return drive[..., None] * per_drive + bow

    @cached_property
    def _load_parts(self) -> tuple[np.ndarray, np.ndarray]:
        """(the load per unit Ω² - iΩ', the bow's constant load), each (4 + r,) complex."""
        per_drive = np.zeros(4 + self.modes_kept, dtype=complex)
        per_drive[1:3] = self.mass_unbalance, self.couple_unbalance
        shaft = [p for p in self.plane_weights if not self.on_disc(p)]
        if shaft:
            per_drive += np.array([p.weight for p in shaft]) @ self.shape_matrix(np.array([p.z for p in shaft]))
        # K_shaft(φ) q_bow(φ): the deviatoric part Δ k e^{2i(φ+α)} conj(·) of the bow turns with it too.
        bow = self.shaft_stiffness() @ self.bow_shape() * self.bow * (
            np.exp(1j * self.bow_angle) + self.asymmetry * np.exp(1j * (2.0 * self.asymmetry_angle - self.bow_angle)))
        return per_drive, bow

    def plane_load(self, P: np.ndarray) -> np.ndarray:
        """The real load f (..., n_dof) of a turning one f_x + i f_y = P (..., 4 + r): f_x = Re P, f_y = Im P."""
        P = np.asarray(P)
        return self.merge(P.real, P.imag)

    def gravity_load(self, omega: float = 0.0) -> np.ndarray:
        """f_g (n_dof,): the weights of the journals, the disc and the shaft, in -y (N).

        With oil films, also the constant load p(Ω) that holds each journal where its
        film settles it (see _film_terms), so it depends on the speed.
        """
        # The stations moving together, the slope at the disc held, is a rigid shift (modal coordinates 0).
        shift = np.zeros(4 + self.modes_kept)
        shift[[0, 1, 3]] = 1.0
        f = self.merge(np.zeros_like(shift), -self.gravity * self.plane_mass() @ shift)
        return f + self._film_terms(omega)[2] if self.films and self.gravity else f

    def static_deflection(self, omega: float | np.ndarray = 0.0) -> np.ndarray:
        """q_s = (K + ΩH)^{-1} f_g: where gravity holds the rotor, (n_dof,) or (len(omega), n_dof).

        On magnets, from the whole state (static_state): with integral action they hold
        their journals at the centre.

        Unless the stiffness depends on the speed (speed_dependent), neither does this.
        With an asymmetric or cracked shaft the rotor never holds still under
        gravity: q_s is then the mean of its periodic response (gravity_response),
        the sag about which the gravity vibration swings, and it depends on the speed.
        """
        omega = np.asarray(omega, dtype=float)
        n = self.n_dof
        if not self.gravity:
            return np.zeros(omega.shape + (n,))
        if self.periodic:
            return gravity_response(self, omega.ravel())[0].real.reshape(omega.shape + (n,))
        if self.magnets:  # the magnets' push has states of their own
            if not self.speed_dependent or omega.ndim == 0:
                q = self.static_state(float(omega) if omega.ndim == 0 else 0.0)[:n]
                return q if omega.ndim == 0 else np.broadcast_to(q, omega.shape + (n,)).copy()
            out = np.array([self.static_state(w)[:n] for w in omega.ravel()])
            return out.reshape(omega.shape + (n,))
        if not self.speed_dependent or omega.ndim == 0:
            w = float(omega) if omega.ndim == 0 else 0.0
            q = np.linalg.solve(self.stiffness(w), self.gravity_load(w))
            return q if omega.ndim == 0 else np.broadcast_to(q, omega.shape + (n,)).copy()
        if self.films:
            flat = omega.ravel()
            out = np.array([np.linalg.solve(self.stiffness(w), self.gravity_load(w)) for w in flat])
            return out.reshape(omega.shape + (n,))
        K, H, f = self.matrices()[3], self._circulatory, self.gravity_load()
        return np.linalg.solve(K + omega[..., None, None] * H, np.broadcast_to(f, omega.shape + (n,))[..., None])[..., 0]

    @property
    def sag_varies(self) -> bool:
        """True if the static sag depends on the speed: a speed-dependent stiffness, or a turning one."""
        return bool(self.gravity and (self.speed_dependent or self.periodic))

    def rest_deflection(self, phase: float = 0.0) -> np.ndarray:
        """Where gravity holds the rotor at rest, its keyphasor mark at angle φ (n_dof,).

        As static_deflection(0), except that an asymmetric or cracked shaft at rest
        sags on the stiffness it has at that angle, not on its mean.
        """
        if not self.gravity or not self.periodic:
            return self.static_deflection(0.0)
        return self.static_state(0.0, phase)[:self.n_dof]

    def bow_displacement(self, phase: float) -> np.ndarray:
        """q_bow (n_dof,): the bow at shaft angle φ, as displacements in fixed coordinates."""
        b = self.bow_shape() * self.bow * np.exp(1j * (phase + self.bow_angle))
        return self.merge(b.real, b.imag)

    def shape_matrix(self, z: np.ndarray) -> np.ndarray:
        """S (len(z) x (4 + r)): the shaft's deflection at axial positions z (m) is S @ one plane's DOFs.

        Within each element the shaft bends as a cubic through its nodes' displacements and
        slopes (exact for the massless shaft); beyond the shaft's ends the end element carries on.
        """
        z = np.atleast_1d(np.asarray(z, dtype=float))
        nodes = self._mesh[0]
        T = self._reduced[0]
        e = np.clip(np.searchsorted(nodes, z, side="right") - 1, 0, nodes.size - 2)
        length = nodes[e + 1] - nodes[e]
        s = (z - nodes[e]) / length
        h = np.column_stack([1 - 3 * s**2 + 2 * s**3, length * (s - 2 * s**2 + s**3),
                             3 * s**2 - 2 * s**3, length * (s**3 - s**2)])
        rows = 2 * e[:, None] + np.arange(4)
        return np.einsum("zk,zkj->zj", h, T[rows])

    @property
    def extent(self) -> tuple[float, float]:
        """The shaft's ends (m): the bearings, or the disc where it overhangs one."""
        return min(0.0, self.a), max(self.length, self.a)

    def deflection(self, q: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Shaft centre (x, y) at axial positions z for displacements q (..., n_dof): shape (..., len(z), 2)."""
        S = self.shape_matrix(z)
        q = np.asarray(q)
        ix, iy = self._planes
        return np.stack([q[..., ix] @ S.T, q[..., iy] @ S.T], axis=-1)

    # ------------------------------------------------------------- matrices
    def matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(M, C, G, K), each n_dof square on q = [x_A, x_D, θx, x_B, y_A, y_D, θy, y_B, ξ_x, ξ_y].

        The equations of motion are M q'' + (C + Ω G) q' + (K + Ω H) q = f, H from
        circulatory(). K holds the bearings' cross-coupled stiffness (so it is not
        symmetric unless k_xy = 0) and C the shaft's internal damping. An oil film's
        stiffness and damping are not in them: they change with the speed, so use
        stiffness(Ω) and damping(Ω) wherever the speed is known.
        """
        return tuple(m.copy() for m in self._matrices)

    @cached_property
    def _matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        A, B = self.bearing_a, self.bearing_b
        ks = self.shaft_stiffness()
        mp = self.plane_mass()
        M = self._blocks(mp, mp)
        K = self._blocks(ks, ks)
        C = self._blocks(self.loss_time * ks, self.loss_time * ks)
        for j, bearing in ((XA, A), (XB, B)):
            k, c = bearing.springs()
            K[np.ix_([j, j + 4], [j, j + 4])] += k
            C[np.ix_([j, j + 4], [j, j + 4])] += c
        # Id θx'' + Ω Ip θy' = moment in x-z;  Id θy'' - Ω Ip θx' = moment in y-z. The shaft's own
        # sections do the same, with twice their diametral rotary inertia.
        gp = 2.0 * self._reduced[3]
        gp[2, 2] += self.ip
        G = self._blocks(np.zeros_like(gp), np.zeros_like(gp), gp, -gp)
        return M, C, G, K + self._periodic_terms[0]

    def circulatory(self) -> np.ndarray:
        """H (8x8): the skew stiffness per unit speed, Ω H in the equations.

        The shaft's internal damping η K_shaft acts on the bending rate seen in the
        rotating shaft, q' - Ω J q, J turning x toward y. Its -Ω J part moves to the
        left side as Ω H, with H = η [0 K_shaft; -K_shaft 0]. Each fluid-film
        bearing's cross-coupling λ c Ω adds λ c where K has k_xy.
        """
        return self._circulatory.copy()

    @cached_property
    def _circulatory(self) -> np.ndarray:
        eta_ks = self.loss_time * self.shaft_stiffness()
        zero = np.zeros_like(eta_ks)
        H = self._blocks(zero, zero, eta_ks, -eta_ks)
        for j, bearing in ((XA, self.bearing_a), (XB, self.bearing_b)):
            if bearing.swirling:
                H[j, j + 4] += bearing.swirl * bearing.damping
                H[j + 4, j] -= bearing.swirl * bearing.damping
        return H

    @property
    def films(self) -> bool:
        """True if either bearing is an oil film."""
        return self.bearing_a.film is not None or self.bearing_b.film is not None

    @property
    def whole_films(self) -> bool:
        """True if the time response takes the oil films' whole force (`nonlinear_films`, on a round shaft).

        A turning shaft's split time step keeps them linearized.
        """
        return self.nonlinear_films and self.films and not self.periodic

    @property
    def speed_dependent(self) -> bool:
        """True if the stiffness changes with the speed: internal damping, a swirling bearing or an oil film."""
        return bool(self.internal_damping or self.bearing_a.swirling or self.bearing_b.swirling or self.films)

    def cross_coupling(self, omega: float) -> tuple[float, float]:
        """Each bearing's cross-coupled stiffness k_xy at speed Ω (N/m), A then B: k_xy + λ c Ω, or the film's."""
        states = self.bearing_states(omega)
        return tuple(float(st.k[0, 1]) if st is not None else b.kxy + b.swirling * b.swirl * b.damping * omega
                     for b, st in zip((self.bearing_a, self.bearing_b), states))

    @cached_property
    def bearing_loads(self) -> tuple[float, float]:
        """Each bearing's share of the rotor's weight (N), A then B: the support's upward push on its journal.

        With the journals held, the shaft carries the disc's and its own weight to
        them, and each journal's weight is its own. Negative where the support has
        to pull the journal down (the far bearing of an overhung rotor).
        """
        if not self.gravity:
            return 0.0, 0.0
        ks = self.shaft_stiffness()
        f = -self.gravity * self.plane_mass() @ np.r_[1.0, 1.0, 0.0, 1.0, np.zeros(self.modes_kept)]
        j = [0, 3]
        r = [i for i in range(ks.shape[0]) if i not in j]
        held = np.linalg.solve(ks[np.ix_(r, r)], f[r])
        push = ks[np.ix_(j, r)] @ held - f[j]
        return float(push[0]), float(push[1])

    def bearing_states(self, omega: float) -> tuple[BearingState | None, BearingState | None]:
        """Each oil film's state at speed Ω under its share of the weight, A then B (None: springs)."""
        key = abs(float(omega))
        return _cached(self, "_film_states", key, lambda: tuple(
            None if b.film is None else b.film.with_load(abs(w)).on_rotor(key)
            for b, w in zip((self.bearing_a, self.bearing_b), self.bearing_loads)), CACHE_STATES)

    def _film_terms(self, omega: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(K_f, C_f, p): the films' stiffness and damping at speed Ω, and the load that holds them there.

        A film is linearized about where its journal settles, q_J at its eccentricity
        and attitude, not about the bearing's centre: f = W - K_f (q - q_J) - C_f q',
        W its share of the weight held up. So it adds K_f to the stiffness and the
        constant p = W + K_f q_J to the load (with gravity, beside f_g): the rotor
        then rests with its journals on the film's locus.

        Cached per speed, read-only: a sweep asks for each speed several times.
        """
        return _cached(self, "_film_terms_at", abs(float(omega)), lambda: self._new_film_terms(omega), CACHE_TERMS)

    def _new_film_terms(self, omega: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = self.n_dof
        Kf, Cf, p = np.zeros((n, n)), np.zeros((n, n)), np.zeros(n)
        for j, b, w, st in zip((XA, XB), (self.bearing_a, self.bearing_b), self.bearing_loads,
                               self.bearing_states(omega)):
            if st is None:
                continue
            d = slice(j, j + 5, 4)  # [x, y] of the journal
            Kf[d, d] = st.k
            Cf[d, d] = st.c
            # An upward pull mirrors the journal through the centre; the coefficients stay the same.
            q_j = math.copysign(b.film.clearance, w) * np.array(st.position())
            p[d] = np.array([0.0, w]) + st.k @ q_j
        for a in (Kf, Cf, p):
            a.flags.writeable = False
        return Kf, Cf, p

    def stiffness(self, omega: float) -> np.ndarray:
        """K + Ω H + the films': the whole stiffness at speed Ω (with an asymmetric shaft, its mean)."""
        K = self._matrices[3]
        if not self.speed_dependent:
            return K.copy()
        K = K + omega * self._circulatory
        return K + self._film_terms(omega)[0] if self.films else K

    def damping(self, omega: float) -> np.ndarray:
        """C + the films': the damping at speed Ω (without the gyroscopic Ω G)."""
        C = self._matrices[1]
        return C + self._film_terms(omega)[1] if self.films else C.copy()

    @property
    def deviatoric(self) -> tuple[np.ndarray, np.ndarray]:
        """(K_c, K_s): an asymmetric shaft's stiffness less its mean is cos 2θ K_c + sin 2θ K_s, θ = φ + α.

        Bending toward the stiff axis (at θ) meets Δ k more than the mean stiffness k
        of the shaft, across it Δ k less: in each plane pair that is Δ k [cos 2θ,
        sin 2θ; sin 2θ, -cos 2θ]. As f_x + i f_y it is Δ k e^{2iθ} (x - iy). The
        shaft's reduced stiffness all scales with EI, so the modal coordinates
        share the same Δ.
        """
        Kc, Ks = self._deviatoric
        return Kc.copy(), Ks.copy()

    @cached_property
    def _deviatoric(self) -> tuple[np.ndarray, np.ndarray]:
        dk = self.asymmetry * self.shaft_stiffness()
        zero = np.zeros_like(dk)
        return self._blocks(dk, -dk), self._blocks(zero, zero, dk, dk)

    @cached_property
    def _periodic_terms(self) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        """(the crack's mean loss of stiffness (n_dof square, in K), {k: E_k}): the turning part's harmonics.

        The stiffness less its mean is Σ_k E_k e^{ikφ} + conj, k ≥ 1. The asymmetric
        shaft's deviatoric part is E_2 alone. An open crack takes D_o off the shaft's
        stiffness bending along its mouth's direction θ = φ + β_c, D_a across: in fixed
        axes P + Re(W' e^{2iθ}), P = (D_o + D_a)/2 on both planes and W' the
        deviatoric part of (D_o - D_a)/2, as for the asymmetric shaft. Breathing, it
        is open by g = (1 - s sin θ)/2, so g (P + W e^{2iθ} + W̄ e^{-2iθ}) takes
        harmonics 0 to 3 of θ.
        """
        n = self.n_dof
        mean, E = np.zeros((n, n)), {}
        if self.asymmetry:
            Kc, Ks = self._deviatoric
            E[2] = 0.5 * (Kc - 1j * Ks) * np.exp(2j * self.asymmetry_angle)
        opening = self.crack_opening
        if opening:
            c_open, c_across = self.crack_compliances
            d_open, d_across = self._hinge_loss(c_open), self._hinge_loss(c_across)
            dm, dd = 0.5 * (d_open + d_across), 0.5 * (d_open - d_across)
            zero = np.zeros_like(dm)
            P = self._blocks(dm, dm)
            W = 0.5 * (self._blocks(dd, -dd) - 1j * self._blocks(zero, zero, dd, dd))
            turn = np.exp(1j * self.crack.angle)
            mean -= opening * P
            E[2] = E.get(2, 0.0) - opening * W * turn**2
            if self.crack.breathing:  # g = ½ + (is/4) e^{iθ} - (is/4) e^{-iθ}
                quarter = 0.25j * self.crack_sign
                E[1] = -quarter * (P - W) * turn
                E[3] = -quarter * W * turn**3
        return mean, {k: E[k] for k in sorted(E) if np.any(E[k])}

    @property
    def periodic(self) -> bool:
        """True if the stiffness changes as the shaft turns: an asymmetric shaft, or an open or breathing crack."""
        return bool(self._periodic_terms[1])

    @property
    def spacing(self) -> int:
        """The harmonics of φ the stiffness changes by are multiples of this: 2 (turning with the shaft) or 1."""
        return math.gcd(*self._periodic_terms[1]) if self.periodic else 2

    @property
    def rotating_frame(self) -> bool:
        """True if the rotor is time-invariant in axes turning with the shaft (see rotating_state_matrix).

        Its stiffness has to turn rigidly with the shaft (only 2X: not a breathing
        crack, opened by the weight that stays put), on isotropic supports.
        """
        return self.spacing == 2 and self.isotropic_supports

    @cached_property
    def _kick(self) -> tuple[np.ndarray, np.ndarray]:
        """(k, S): the turning stiffness's pull on the accelerations is M^{-1} ΔK(φ) q = (cos kφ, sin kφ) S q.

        k (m,) the harmonics and S (2m x n x n) stacked [2 M^{-1} Re E_k, ..., -2 M^{-1} Im E_k, ...].
        """
        E = self._periodic_terms[1]
        Minv = self.mass_inverse
        ks = np.array(list(E), dtype=float)
        S = np.array([2.0 * Minv @ e.real for e in E.values()] + [-2.0 * Minv @ e.imag for e in E.values()])
        return ks, S

    def stiffness_at(self, omega: float, phase: float | np.ndarray) -> np.ndarray:
        """K + Ω H + the deviatoric part at shaft angle φ: the whole stiffness at that instant, (..., n, n)."""
        K = self.stiffness(omega)
        phase = np.asarray(phase, dtype=float)
        K = np.broadcast_to(K, phase.shape + K.shape).copy()
        for k, e in self._periodic_terms[1].items():
            K += 2.0 * (np.exp(1j * k * phase)[..., None, None] * e).real
        return K

    @property
    def isotropic_supports(self) -> bool:
        """True if both bearings have the same stiffness and damping in x as in y, and turn alike.

        Their cross terms then have to be a pure turn (k_yx = -k_xy, no c_xy), and
        no film (whose stiffness depends on the load's direction) nor magnet (whose
        controller's states the rotating frame does not turn).

        Then an asymmetric shaft is time-invariant in coordinates turning with it,
        so a few harmonics give its response and modes exactly (rotating_frame).
        """
        return all(b.film is None and b.magnet is None and b.kx == b.ky and b.cx == b.cy and b.cxy == b.cyx == 0.0
                   and b.k_yx == -b.kxy for b in (self.bearing_a, self.bearing_b))

    def axis_system(self, stiff: bool) -> RotorSystem:
        """The same rotor with a round shaft as stiff as this one's stiff (or soft) axis: EI(1 ± Δ).

        Its critical speed is that axis's: an asymmetric shaft has two. A crack's
        axes are along its mouth (soft) and across: in axes turning with the shaft
        it takes ḡ D_o and ḡ D_a off them on average (ḡ = crack_opening), so each
        axis's round shaft has a crack that takes that off both planes. With both an
        asymmetry and a crack, whose axes need not line up, the crack's mean loss
        goes on both axes: approximate.
        """
        f = 1.0 + (self.asymmetry if stiff else -self.asymmetry)
        out = self.with_(youngs=self.youngs * f, asymmetry=0.0)
        opening = self.crack_opening
        if not opening:
            return out.with_(crack=None) if self.crack is not None else out
        _, s = self._crack_hinge
        taus = [c / (1.0 + c * s) for c in self.crack_compliances]  # each loss is b b^T τ
        tau = opening * (0.5 * sum(taus) if self.asymmetry else taus[1] if stiff else taus[0])
        # Its hinge's b and s scale with EI, by f: f² τ(c') = τ.
        c = tau / (f * (f - tau * s)) if f > tau * s else 1e12
        return out.with_(crack=replace(self.crack, breathing=False, compliance=(c, c)))

    @cached_property
    def mass_inverse(self) -> np.ndarray:
        """M^{-1}: diagonal for the massless shaft, full once the shaft's mass couples the DOFs."""
        M = self._matrices[0]
        if np.count_nonzero(M - np.diag(np.diag(M))) == 0:
            return np.diag(1.0 / np.diag(M))
        return np.linalg.inv(M)

    # ------------------------------------------------------------- magnetic bearings
    @property
    def magnets(self) -> bool:
        """True if either bearing is an active magnetic bearing."""
        return self.bearing_a.magnet is not None or self.bearing_b.magnet is not None

    @cached_property
    def _axes(self) -> list[tuple[int, MagneticBearing]]:
        """Each magnet's axes: (the journal's DOF in q, its bearing), x then y, A then B."""
        return [(d, b.magnet) for j, b in ((XA, self.bearing_a), (XB, self.bearing_b)) if b.magnet is not None
                for d in (j, j + 4)]

    @property
    def n_control(self) -> int:
        """The controllers' states, after q and q' in the state: 0 without magnets."""
        return sum(m.n_states for _, m in self._axes)

    @property
    def n_states(self) -> int:
        """The length of the state z = [q, q', the controllers' states]."""
        return 2 * self.n_dof + self.n_control

    @cached_property
    def _control(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(A_c, B_c, F_c): the controllers' x_c' = A_c x_c + B_c q, and their push on q, F_c x_c.

        Each axis's journal displacement drives its controller (a, b, c); its current
        pushes the journal back by -k_i c x_c (the magnets' -k_m x is in K).
        """
        n, nc = self.n_dof, self.n_control
        Ac, Bc, Fc = np.zeros((nc, nc)), np.zeros((nc, n)), np.zeros((n, nc))
        k = 0
        for d, m in self._axes:
            a, b, c = m.state_space()
            sl = slice(k, k + b.size)
            Ac[sl, sl], Bc[sl, d], Fc[d, sl] = a, b, -m.current_gain * c
            k += b.size
        return Ac, Bc, Fc

    def support_stiffness(self, s: complex) -> tuple[np.ndarray, list[int]]:
        """(D, pinned): the magnets' controlled push at s (on top of K's -k_m), -D(s) q, n_dof square.

        D(s) = k_i G(s) on each axis. With integral action it is infinite at s = 0: a
        steady load cannot move the journal there, so those DOFs are `pinned`, held
        at 0 (their equation's unknown is then the force the magnet takes).
        """
        n = self.n_dof
        D, pinned = np.zeros((n, n), dtype=complex), []
        for d, m in self._axes:
            if s == 0 and m.integral:
                pinned.append(d)
            else:
                D[d, d] = m.current_gain * m.gain(s)
        return D, pinned

    def dynamic_solve(self, Z: np.ndarray, f: np.ndarray, s: complex) -> np.ndarray:
        """Q from (Z + D(s)) Q = f: Z the rest of the dynamic stiffness at s, D the magnets' (support_stiffness)."""
        if not self.magnets:
            return np.linalg.solve(Z, f)
        D, pinned = self.support_stiffness(s)
        Z = Z + D
        f = np.array(f, dtype=Z.dtype)
        Z[pinned] = 0.0
        Z[pinned, pinned] = 1.0
        f[pinned] = 0.0
        return np.linalg.solve(Z, f)

    def state_matrix(self, omega: float) -> np.ndarray:
        """A(Ω) (n_states square) for z = [q, q', x_c]: z' = A z + B u."""
        G = self._matrices[2]
        Minv = self.mass_inverse
        n = self.n_dof
        A = np.zeros((self.n_states, self.n_states))
        A[:n, n:2 * n] = np.eye(n)
        A[n:2 * n, :n] = -Minv @ self.stiffness(omega)
        A[n:2 * n, n:2 * n] = -Minv @ (self.damping(omega) + omega * G)
        if self.magnets:
            Ac, Bc, Fc = self._control
            A[n:2 * n, 2 * n:] = Minv @ Fc
            A[2 * n:, :n], A[2 * n:, 2 * n:] = Bc, Ac
        return A

    def input_matrix(self) -> np.ndarray:
        """B (n_states x n): the inputs u = f (n,), a force or moment on each DOF."""
        n = self.n_dof
        B = np.zeros((self.n_states, n))
        B[n:2 * n] = self.mass_inverse
        return B

    def static_state(self, omega: float = 0.0, phase: float | None = None) -> np.ndarray:
        """The state z (n_states,) where gravity alone holds the rotor still at speed Ω: A z + B f_g = 0.

        Its q is static_deflection's (rest_deflection's, at shaft angle `phase`); its
        controllers' states hold the weight on the magnets. With an asymmetric or
        cracked shaft turning, q is not still: use static_deflection for its mean.
        """
        if not self.gravity:
            return np.zeros(self.n_states)
        A = self.state_matrix(omega)
        if phase is not None:
            n = self.n_dof
            A[n:2 * n, :n] = -self.mass_inverse @ self.stiffness_at(omega, phase)
        return np.linalg.solve(A, -self.input_matrix() @ self.gravity_load(omega))

    # ------------------------------------------------------------- checks
    def jeffcott_speed(self) -> float:
        """Critical speed (rad/s) of the textbook Jeffcott rotor with these parts.

        The disc's mass on the shaft's midspan-equivalent stiffness in series with
        the two bearings in parallel, all massless and isotropic (x-direction
        stiffnesses), the disc's tilt ignored: ω² = k_eq / m. Exact for a disc at
        midspan on identical bearings as the journal masses go to zero; poor for an
        overhung disc, whose tilt (and so the gyroscopic moment) matters most.
        """
        a, b, L = self.a, self.b, self.length
        k_shaft = self.disc_stiffness
        # The bearings carry b/L and a/L of the load (one of them negative for an
        # overhung disc: the far bearing is pulled up); the disc moves by the weighted sum.
        flex_bearings = (b / L) ** 2 / self.bearing_a.kx + (a / L) ** 2 / self.bearing_b.kx
        return math.sqrt(1.0 / (1.0 / k_shaft + flex_bearings) / self.disc_mass)


def beam_stiffness(ei: float, length: float) -> np.ndarray:
    """Euler–Bernoulli beam element stiffness on [w1, θ1, w2, θ2], θ = dw/dz."""
    L = length
    return ei / L**3 * np.array([
        [12.0, 6 * L, -12.0, 6 * L],
        [6 * L, 4 * L * L, -6 * L, 2 * L * L],
        [-12.0, -6 * L, 12.0, -6 * L],
        [6 * L, 2 * L * L, -6 * L, 4 * L * L],
    ])


def beam_mass(rho_a: float, rho_i: float, length: float, lumped: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """A beam element's mass on [w1, θ1, w2, θ2]: (translation, rotary inertia), ρA and ρI per length.

    Consistent: the cubic shape functions' ∫ρA N^T N and ∫ρI N'^T N' (a Rayleigh beam).
    Lumped: half of each to each node, on its displacement and its slope.
    """
    L = length
    if lumped:
        half_a, half_i = 0.5 * rho_a * L, 0.5 * rho_i * L
        return np.diag([half_a, 0.0, half_a, 0.0]), np.diag([0.0, half_i, 0.0, half_i])
    translation = rho_a * L / 420.0 * np.array([
        [156.0, 22 * L, 54.0, -13 * L],
        [22 * L, 4 * L * L, 13 * L, -3 * L * L],
        [54.0, 13 * L, 156.0, -22 * L],
        [-13 * L, -3 * L * L, -22 * L, 4 * L * L],
    ])
    rotation = rho_i / (30.0 * L) * np.array([
        [36.0, 3 * L, -36.0, 3 * L],
        [3 * L, 4 * L * L, -3 * L, -L * L],
        [-36.0, -3 * L, 36.0, -3 * L],
        [3 * L, -L * L, -3 * L, 4 * L * L],
    ])
    return translation, rotation


# ------------------------------------------------------------------ whirl analysis


def whirl_ratio(system: RotorSystem, Q: np.ndarray) -> np.ndarray:
    """Whirl direction of complex amplitudes Q (..., n_dof), x = Re(X e^{iωt}) for ω > 0.

    +1 is a circular forward whirl (with the spin, x toward y), -1 a circular
    backward whirl, 0 a straight line. The motion is weighted by one plane's mass
    matrix, i.e. by kinetic energy: the stations (journals, disc, disc tilt) and,
    with a shaft of some mass, the shaft between them.
    """
    mp = system.plane_mass()
    ix, iy = system.plane_dofs
    X, Y = Q[..., ix], Q[..., iy]
    num = 2.0 * np.imag(np.einsum("...i,ij,...j->...", np.conj(Y), mp, X))
    den = np.real(np.einsum("...i,ij,...j->...", np.conj(X), mp, X) + np.einsum("...i,ij,...j->...", np.conj(Y), mp, Y))
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def _whirl_forms(system: RotorSystem) -> tuple[np.ndarray, np.ndarray]:
    """Hermitian (H, D) with v^H H v and v^H D v the numerator and denominator of whirl_ratio."""
    mp = system.plane_mass().astype(complex)
    return system._blocks(0 * mp, 0 * mp, 1j * mp, -1j * mp), system._blocks(mp, mp)


def _circularize(system: RotorSystem, lam: np.ndarray, vec: np.ndarray, tol: float = 1e-7) -> None:
    """Within each group of repeated eigenvalues, pick the purely forward and backward shapes.

    An axisymmetric rotor not spinning (or with no tilt coupling) has each whirl
    frequency twice, and the solver returns any mix of the pair. The combinations
    that extremize the whirl ratio are its circular forward and backward whirls.
    """
    H, D = None, None
    i = 0
    while i < lam.size:
        j = i + 1
        while j < lam.size and abs(lam[j] - lam[i]) <= tol * abs(lam[i]):
            j += 1
        if j - i > 1:
            if H is None:
                H, D = _whirl_forms(system)
            V = vec[:, i:j]
            _, c = scipy.linalg.eigh(V.conj().T @ H @ V, V.conj().T @ D @ V)
            vec[:, i:j] = V @ c
        i = j


@dataclass
class WhirlModes:
    """The rotor's damped modes at one speed, in order of frequency (one per conjugate pair)."""

    omega: float  # spin speed (rad/s)
    freq_hz: np.ndarray  # damped natural frequencies
    zeta: np.ndarray  # damping ratios, -Re λ / |λ|
    whirl: np.ndarray  # whirl ratios (see whirl_ratio): > 0 forward, < 0 backward
    shapes: np.ndarray  # (n, n_dof) complex displacement shapes, largest station entry 1
    drift: float = -math.inf  # 1/s, the fastest growth of a root that does not oscillate (> 0: it diverges)

    @property
    def forward(self) -> np.ndarray:
        return self.whirl >= 0.0


def whirl_modes(system: RotorSystem, omega: float) -> WhirlModes:
    """Eigen-solution of A(Ω): the oscillatory modes (Im λ > 0), lowest first.

    A root that does not oscillate is left out, but the fastest growth among them
    is kept (`drift`): a magnetic bearing too weak for its magnets lets its journal
    fall away without oscillating.

    An asymmetric or cracked shaft's stiffness turns with it, so A depends on
    time and its modes are Floquet's: e^{λt} times a motion periodic in φ. Each is
    listed by the harmonic of it that carries the most kinetic energy (see
    _rotating_modes and _hill_modes): that is its frequency here, and its shape
    and whirl are that harmonic's.
    """
    drift = -math.inf
    if system.periodic:
        if system.rotating_frame:
            lam, vec = _rotating_modes(system, omega)
        else:
            lam, vec, drift = _hill_modes(system, omega)
        order = np.argsort(lam.imag)
        lam, vec = lam[order], vec[:, order]
    else:
        lam, vec = np.linalg.eig(system.state_matrix(omega))
        keep, drift = _oscillation(lam)
        lam, vec = lam[keep], vec[:system.n_dof, keep]
        order = np.argsort(lam.imag)
        lam, vec = lam[order], vec[:, order]
        _circularize(system, lam, vec)
    shapes = vec.T.copy()
    for s in shapes:
        # By a station's largest entry (a shaft mode may barely move them: then by its largest).
        big = s[:N_DOF] if np.abs(s[:N_DOF]).max() > 1e-6 * np.abs(s).max() else s
        s /= big[np.argmax(np.abs(big))]
    return WhirlModes(
        omega=omega,
        freq_hz=lam.imag / (2.0 * math.pi),
        zeta=-lam.real / np.abs(lam),
        whirl=whirl_ratio(system, shapes),
        shapes=shapes,
        drift=drift,
    )


def _oscillation(lam: np.ndarray) -> tuple[np.ndarray, float]:
    """(Im λ > 0 and oscillating, the fastest growth of the roots that do not oscillate (-inf if none)).

    A magnet's controller has repeated slow real roots (its integrals), which the
    solver and the gyroscopic coupling split by a hair: a root this overdamped
    (|Im λ| < OVERDAMPED |Re λ|, ζ > 0.9987) does not oscillate.
    """
    limit = np.maximum(_oscillating(lam), OVERDAMPED * np.abs(lam.real))
    real = np.abs(lam.imag) <= limit
    return lam.imag > limit, float(lam.real[real].max()) if real.any() else -math.inf


def _oscillating(lam: np.ndarray) -> float:
    """The least Im λ of a mode that oscillates (smaller is round-off on a real eigenvalue)."""
    return 1e-9 * max(1.0, float(np.abs(lam).max())) if lam.size else 0.0


def rotating_state_matrix(system: RotorSystem, omega: float) -> np.ndarray:
    """A_r(Ω) (2n x 2n) in coordinates turning with the shaft, q = R(φ) r: constant on isotropic supports.

    R(φ) turns each plane pair by φ, R' = Ω R J. Every matrix but the asymmetric
    shaft's commutes with R when the supports are isotropic, so R^T(...)R leaves
    them, the shaft's deviatoric part is held at φ = 0 (so is an open crack's), and
        M r'' + (C + ΩG + 2ΩMJ) r' + (K + ΩH + ΔK(0) - Ω²M + ΩCJ + Ω²GJ) r = R^T f.
    """
    M, _, G, _ = system._matrices
    C = system.damping(omega)
    n = system.n_dof
    J = _turn(system)
    Kr = system.stiffness_at(omega, 0.0) - omega**2 * M + omega * C @ J + omega**2 * G @ J
    Cr = C + omega * G + 2.0 * omega * M @ J
    Minv = system.mass_inverse
    A = np.zeros((2 * n, 2 * n))
    A[:n, n:] = np.eye(n)
    A[n:, :n] = -Minv @ Kr
    A[n:, n:] = -Minv @ Cr
    return A


def _turn(system: RotorSystem) -> np.ndarray:
    """J (n x n): turns every plane pair x toward y, J [x; y] = [-y; x]."""
    eye = np.eye(system.n_dof // 2)
    return system._blocks(0 * eye, 0 * eye, -eye, eye)


def _rotating_modes(system: RotorSystem, omega: float) -> tuple[np.ndarray, np.ndarray]:
    """An asymmetric shaft's modes on isotropic supports, exactly: (λ (k,), shapes (n_dof, k)), Im λ > 0.

    Turning with the shaft the rotor is time-invariant (rotating_state_matrix).
    A mode r = V e^{μt} there, V = [X; Y] by plane, is seen from the fixed frame
    as a forward circle (X + iY) e^{(μ + iΩ)t} plus a backward one (X - iY)
    e^{(μ - iΩ)t}. The one carrying more kinetic energy names the mode: λ = μ ± iΩ,
    its shape that circle (conjugated if need be, so that Im λ > 0). Its growth
    rate Re λ = Re μ is exact. A mode that stands still turning with the shaft
    (μ real, as between the two critical speeds) is a forward whirl locked to 1X.
    """
    mu, vec = np.linalg.eig(rotating_state_matrix(system, omega))
    # A conjugate pair there is one real motion, and maps to one λ: keep its Im μ ≥ 0 member.
    half = mu.imag >= -_oscillating(mu)
    mu, vec = mu[half], vec[:, half]
    ix, iy = system.plane_dofs
    V = vec[:system.n_dof]
    a, b = V[ix] + 1j * V[iy], V[ix] - 1j * V[iy]
    mp = system.plane_mass()
    fwd = np.real(np.einsum("ik,ij,jk->k", a.conj(), mp, a))
    bwd = np.real(np.einsum("ik,ij,jk->k", b.conj(), mp, b))
    forward = fwd >= bwd
    lam = mu + np.where(forward, 1j, -1j) * omega
    shapes = np.where(forward, system.merge(a.T, -1j * a.T).T, system.merge(b.T, 1j * b.T).T)
    flip = lam.imag < 0
    lam[flip], shapes[:, flip] = lam[flip].conj(), shapes[:, flip].conj()
    keep = lam.imag > _oscillating(lam)
    return lam[keep], shapes[:, keep]


def _hill_modes(system: RotorSystem, omega: float) -> tuple[np.ndarray, np.ndarray, float]:
    """A turning shaft's modes, by Hill's method: (λ, shapes (n_dof, k), drift), Im λ > 0.

    A(t) = A_0 + Σ_k A_k e^{ikφ}, k = ±p, ±2p, ... (p = spacing: 2 for an
    asymmetric shaft, 1 for a breathing crack); a Floquet mode z = e^{λt} Σ c_j
    e^{ijpΩt} has λ c_j = (A_0 - ijpΩ) c_j + Σ_k A_k c_{j-k/p}, an eigenproblem in
    the c_j, |j| ≤ HILL_ORDER. Each
    mode turns up once for every j, shifted by ipΩ (and so does its conjugate);
    the copies kept are those whose kinetic energy is centred on c_0, within half
    a harmonic (the copies near the truncation are the inaccurate ones, and their
    energy is off at the edges). A whirl locked to pX/2 has its energy split
    evenly between two harmonics, so two of its copies sit on that edge; turned
    to Im λ > 0 they are one λ, kept once. Its shape is c_0. drift is as
    whirl_modes's, from the roots that do not oscillate.
    """
    n, p, H = system.n_dof, system.spacing, HILL_ORDER
    A0 = system.state_matrix(omega).astype(complex)
    size, blocks = A0.shape[0], 2 * H + 1
    big = np.zeros((blocks * size, blocks * size), dtype=complex)
    for i in range(blocks):
        d = slice(i * size, (i + 1) * size)
        big[d, d] = A0 - 1j * (i - H) * p * omega * np.eye(size)
    for k, e in system._periodic_terms[1].items():
        Ak = -system.mass_inverse @ e
        for i in range(blocks - k // p):  # A_k takes c_j to c_{j + k/p}, its conjugate back
            lo, hi = slice(i * size, (i + 1) * size), slice((i + k // p) * size, (i + k // p + 1) * size)
            big[hi, lo][n:2 * n, :n] = Ak
            big[lo, hi][n:2 * n, :n] = Ak.conj()
    lam, vec = np.linalg.eig(big)
    disp = vec.reshape(blocks, size, -1)[:, :n]  # (harmonic, DOF, mode)
    M = system._matrices[0]
    energy = np.real(np.einsum("hik,ij,hjk->hk", disp.conj(), M, disp))
    centre = np.arange(-H, H + 1) @ energy / energy.sum(axis=0)
    keep = np.abs(centre) <= 0.5 + 1e-6
    lam, shapes = lam[keep], disp[H][:, keep]
    flip = lam.imag < 0
    lam[flip], shapes[:, flip] = lam[flip].conj(), shapes[:, flip].conj()
    keep, drift = _oscillation(lam)
    lam, shapes, flip = lam[keep], shapes[:, keep], flip[keep]
    # A mode and its conjugate (turned) are one: pair each turned copy with an unturned one of the same λ. A
    # double root (a round rotor's two whirls) has two of each, and stays double.
    tol = 1e-7 * max(1.0, float(np.abs(lam).max())) if lam.size else 0.0
    single, paired = [], set()
    for i in range(lam.size):
        mate = next((j for j in single if j not in paired and flip[j] != flip[i]
                     and abs(lam[j] - lam[i]) <= tol), None)
        if mate is None:
            single.append(i)
        else:
            paired.add(mate)
    return lam[single], shapes[:, single], drift


@dataclass
class Campbell:
    """Damped natural frequencies against spin speed, and the critical speeds."""

    omega: np.ndarray  # (n_speeds,) rad/s
    freq_hz: np.ndarray  # (n_speeds, n_modes), each column one frequency-ordered branch
    zeta: np.ndarray  # (n_speeds, n_modes)
    whirl: np.ndarray  # (n_speeds, n_modes), > 0 forward
    criticals: list[tuple[float, float, float]]  # (speed in rad/s, whirl ratio, ζ) where a branch meets 1X


def campbell(system: RotorSystem, omega_max: float, n_speeds: int = 241, n_modes: int = 6) -> Campbell:
    """Sweep the speed 0..omega_max and find where each branch crosses the synchronous (1X) line.

    An asymmetric shaft's modes lock onto 1X between its two critical speeds, so
    its branches do not cross the line cleanly. Its critical speeds (and a cracked
    shaft's) are taken as those of a round shaft as stiff as each axis
    (RotorSystem.axis_system): where,
    turning with the shaft, the rotor has no stiffness left along that axis
    against the centrifugal force. With no damping and no gyroscopic coupling
    they are exactly the ends of the band where it is unstable.
    """
    omegas = np.linspace(0.0, omega_max, n_speeds)
    freq = np.full((n_speeds, n_modes), np.nan)
    zeta = np.full((n_speeds, n_modes), np.nan)
    whirl = np.zeros((n_speeds, n_modes))
    for i, w in enumerate(omegas):
        m = whirl_modes(system, w)
        k = min(n_modes, m.freq_hz.size)
        freq[i, :k], zeta[i, :k], whirl[i, :k] = m.freq_hz[:k], m.zeta[:k], m.whirl[:k]
    criticals = []
    if system.periodic:
        for stiff in (False, True):
            criticals += campbell(system.axis_system(stiff), omega_max, n_speeds, n_modes).criticals
        criticals.sort()
        return Campbell(omegas, freq, zeta, whirl, criticals)
    for j in range(n_modes):
        gap = 2.0 * math.pi * freq[:, j] - omegas
        for i in range(n_speeds - 1):
            g0, g1 = gap[i], gap[i + 1]
            if np.isfinite(g0) and np.isfinite(g1) and g0 > 0.0 >= g1:
                w = omegas[i] + (omegas[i + 1] - omegas[i]) * g0 / (g0 - g1)
                w = _refine_critical(system, j, w, omegas[i], omegas[i + 1])
                m = whirl_modes(system, w)
                if j < m.whirl.size:
                    criticals.append((w, float(m.whirl[j]), float(m.zeta[j])))
    criticals.sort()
    return Campbell(omegas, freq, zeta, whirl, criticals)


def _refine_critical(system: RotorSystem, branch: int, guess: float, lo: float, hi: float) -> float:
    """A few secant steps on 2π f_branch(Ω) = Ω, kept within [lo, hi]."""
    def gap(w: float) -> float:
        f = whirl_modes(system, w).freq_hz
        return 2.0 * math.pi * f[branch] - w if branch < f.size else -w

    x0, x1 = lo, hi
    g0, g1 = gap(x0), gap(x1)
    for _ in range(30):
        if g1 == g0:
            break
        x2 = min(max(x1 - g1 * (x1 - x0) / (g1 - g0), lo), hi)
        x0, g0, x1, g1 = x1, g1, x2, gap(x2)
        if abs(x1 - x0) <= 1e-10 * max(1.0, abs(x1)):
            break
    return x1 if lo <= x1 <= hi else guess


def synchronous_response(system: RotorSystem, omegas: np.ndarray) -> np.ndarray:
    """Steady-state 1X response at constant speeds: Q (len(omegas), n_dof), complex.

    The unbalance, the disc's skew and the shaft's bow together. q(t) = Re(Q e^{iΩt}),
    with the keyphasor mark at angle φ = Ωt (along +x at t = 0). Above
    the stability onset this particular solution still exists, but the free whirl
    grows over it, so the rotor never settles to it.

    With an asymmetric or cracked shaft the response to a 1X load has other
    harmonics in it too (harmonic_response; on isotropic supports none, for the
    asymmetric shaft); this is its 1X part. A breathing crack's weight drives 1X
    as well: that is gravity_response's, and steady_harmonics adds the two.
    """
    M, _, G, _ = system.matrices()
    omegas = np.atleast_1d(np.asarray(omegas, dtype=float))
    out = np.empty((omegas.size, system.n_dof), dtype=complex)
    f = _rotating_force(system, omegas)
    if system.periodic:
        orders = harmonic_orders(system, odd=True)
        F = np.zeros((omegas.size, orders.size, system.n_dof), dtype=complex)
        F[:, _index(orders, 1)], F[:, _index(orders, -1)] = 0.5 * f, 0.5 * f.conj()
        return 2.0 * harmonic_response(system, omegas, orders, F)[:, _index(orders, 1)]
    for i, w in enumerate(omegas):
        Z = system.stiffness(w) - w * w * M + 1j * w * (system.damping(w) + w * G)
        out[i] = system.dynamic_solve(Z, f[i], 1j * w)
    return out


def _rotating_force(system: RotorSystem, omegas: np.ndarray) -> np.ndarray:
    """F (len(omegas), n_dof) complex: the 1X load at each speed is Re(F e^{iφ})."""
    load = system.rotating_load(omegas)
    # f_x + i f_y = P e^{iφ}: f_x = Re(P e^{iφ}), f_y = Re(-iP e^{iφ}).
    return system.merge(load, -1j * load)


def _index(orders: np.ndarray, k: int) -> int:
    return int(np.searchsorted(orders, k))


def harmonic_orders(system: RotorSystem, odd: bool | None) -> np.ndarray:
    """The harmonics of φ a turning shaft's steady state needs: odd (to a 1X load), even (to gravity) or all.

    An asymmetric shaft's deviatoric stiffness shifts a harmonic k to k ± 2, so a
    load at 1X reaches the odd ones and a constant load the even ones. On
    isotropic supports the chain closes at once: a forward whirl at 1X stays 1X
    (in x and y, ±1), and the 0X sag gives only a forward 2X (0, ±2). Otherwise
    HARMONICS each side. A breathing crack's shifts them by 1 too, so every load
    reaches every harmonic: 2 HARMONICS each side, whichever is asked.
    """
    if system.spacing == 1:
        return np.arange(-2 * HARMONICS, 2 * HARMONICS + 1)
    top = 1 if system.rotating_frame else 2 * HARMONICS - 1  # the top odd harmonic, the even one past it
    if odd is None:
        return np.arange(-top - 1, top + 2)
    return np.arange(-top, top + 1, 2) if odd else np.arange(-top - 1, top + 2, 2)


def harmonic_response(system: RotorSystem, omega: float | np.ndarray, orders: np.ndarray,
                      loads: np.ndarray) -> np.ndarray:
    """Harmonic balance at constant speed Ω: the periodic q = Σ Q_k e^{ikφ} under f = Σ F_k e^{ikφ}.

    orders (m,) are the harmonics k, in order; loads (..., m, n_dof) the F_k (F_{-k}
    = conj(F_k) for a real load). Returns the Q_k, (m, n_dof) for one speed or
    (len(omega), m, n_dof) for several (loads then one set for all, or one per
    speed). Row k reads (K(Ω) - k²Ω²M + ikΩ(C(Ω) + ΩG)) Q_k + Σ_j E_j Q_{k-j} =
    F_k, the turning stiffness being Σ_j E_j e^{ijφ} (E_{-j} = conj(E_j));
    harmonics past the ends are taken as zero. Magnetic bearings add their push
    D(ikΩ) to row k (support_stiffness); at k = 0 with integral action their
    journals are held at the centre.
    """
    M, C, G, K = system._matrices
    H = system._circulatory if system.speed_dependent else np.zeros_like(K)
    E = system._periodic_terms[1]
    n, m = system.n_dof, len(orders)
    w = np.atleast_1d(np.asarray(omega, dtype=float))
    F = np.broadcast_to(np.asarray(loads, dtype=complex), (w.size, m, n)).reshape(w.size, m * n)
    out = np.empty((w.size, m * n), dtype=complex)
    chunk = max(1, 2_000_000 // (m * n) ** 2)  # speeds solved at once, to bound the memory
    for c in range(0, w.size, chunk):
        wc = w[c:c + chunk, None, None]
        if system.films:  # the films' stiffness and damping are not linear in Ω
            Kw = np.array([system.stiffness(x) for x in w[c:c + chunk]])
            Cw = np.array([system.damping(x) for x in w[c:c + chunk]])
        else:
            Kw, Cw = K + wc * H, C
        A = np.zeros((wc.shape[0], m * n, m * n), dtype=complex)
        for i, k in enumerate(orders):
            s = k * wc
            d = slice(i * n, (i + 1) * n)
            A[:, d, d] = Kw - s * s * M + 1j * s * (Cw + wc * G)
            for j, kj in enumerate(orders):
                gap = int(k - kj)
                if gap and abs(gap) in E:
                    A[:, d, j * n:(j + 1) * n] = E[gap] if gap > 0 else E[-gap].conj()
        Fc = F[c:c + chunk].copy()
        if system.magnets:
            for i, k in enumerate(orders):
                for j, x in enumerate(wc[:, 0, 0]):
                    D, pinned = system.support_stiffness(1j * k * x)
                    A[j, i * n:(i + 1) * n, i * n:(i + 1) * n] += D
                    rows = [i * n + p for p in pinned]
                    A[j, rows] = 0.0
                    A[j, rows, rows] = 1.0
                    Fc[j, rows] = 0.0
        out[c:c + chunk] = np.linalg.solve(A, Fc[..., None])[..., 0]
    out = out.reshape(w.size, m, n)
    return out[0] if np.ndim(omega) == 0 else out


def gravity_response(system: RotorSystem, omegas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gravity's steady state at constant speeds: (mean sag Q_0, 2X vector Q_2), each (len(omegas), n_dof).

    q = Q_0 + Re(Q_2 e^{2iφ}) + (4X, ...: only on anisotropic supports; and 1X, 3X,
    ... with a breathing crack: see steady_harmonics). A round shaft just sags: Q_0
    = (K + ΩH)^{-1} f_g and Q_2 = 0. An asymmetric shaft's weight acts on a
    stiffness that changes twice per revolution, so it vibrates at 2X, most where
    2Ω meets a natural frequency: the gravity critical, near half the critical speed.
    """
    Q0, V = periodic_steady(system, omegas, gravity=True, rotating=False, top=2)
    return Q0, V[:, 1]


def steady_harmonics(system: RotorSystem, omegas: np.ndarray, top: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """The whole steady state, gravity and the 1X loads together: (Q_0 (len(omegas), n_dof), V (len, top, n_dof)).

    q = Q_0 + Re Σ_k V_k e^{ikφ}, k = 1 ... top (V[:, k - 1]). A round shaft has
    only the sag and the 1X (synchronous_response); a turning one has the others
    from harmonic balance. A breathing crack mixes the two: its weight drives 1X,
    2X and 3X, and the unbalance 2X and the mean too.
    """
    return periodic_steady(system, omegas, gravity=True, rotating=True, top=top)


def periodic_steady(system: RotorSystem, omegas: np.ndarray, gravity: bool, rotating: bool,
                    top: int) -> tuple[np.ndarray, np.ndarray]:
    """As steady_harmonics, from gravity's load, the 1X loads, or both: (Q_0, V_k for k = 1 ... top)."""
    omegas = np.atleast_1d(np.asarray(omegas, dtype=float))
    n = system.n_dof
    Q0 = np.zeros((omegas.size, n), dtype=complex)
    V = np.zeros((omegas.size, top, n), dtype=complex)
    gravity = gravity and bool(system.gravity)
    if not system.periodic:
        if gravity:
            Q0[:] = system.static_deflection(omegas)
        if rotating and top:
            V[:, 0] = synchronous_response(system, omegas)
        return Q0, V
    if not gravity and not rotating:
        return Q0, V
    orders = harmonic_orders(system, None if gravity and rotating else not gravity)
    # Each speed once: a steady run asks for the same one again and again.
    speeds, back = np.unique(omegas, return_inverse=True)
    F = np.zeros((speeds.size, orders.size, n), dtype=complex)
    if gravity:
        F[:, _index(orders, 0)] = [system.gravity_load(w) for w in speeds] if system.films else system.gravity_load()
    if rotating:
        f = _rotating_force(system, speeds)
        F[:, _index(orders, 1)], F[:, _index(orders, -1)] = 0.5 * f, 0.5 * f.conj()
    Q = harmonic_response(system, speeds, orders, F)[back.ravel()]
    if 0 in orders:
        Q0 = Q[:, _index(orders, 0)]
    for k in range(1, top + 1):
        if k in orders:
            V[:, k - 1] = 2.0 * Q[:, _index(orders, k)]
    return Q0, V


# ------------------------------------------------------------------ time stepping


@dataclass
class RotorSamples:
    """What one advance returns, one row per step."""

    t: np.ndarray  # (k,) s
    q: np.ndarray  # (k, n_dof) displacements
    omega: np.ndarray  # (k,) spin speed (rad/s)
    phase: np.ndarray  # (k,) angle φ of the keyphasor mark (rad, unwrapped)


class _Films:
    """The oil films' whole force on the journals, for RotorSimulator._film_steps.

    The exact update carries each film linearized: its stiffness K_f and load p
    at the held speed (as the linear films have), and in place of its damping
    C_f the bound C_b, the whole uncavitated film's damping where the journal
    was when it was last taken. The rest of the film's force, the remainder r_f
    = f - (p - K_f q - C_b q'), goes in as a force on the journals.
    """

    def __init__(self, system: RotorSystem) -> None:
        self.system = system
        n = system.n_dof
        films = [(j, b) for j, b in ((XA, system.bearing_a), (XB, system.bearing_b)) if b.film is not None]
        loads = dict(zip((XA, XB), system.bearing_loads))
        self.dofs = [d for j, _ in films for d in (j, j + 4)]  # x and y of each film's journal
        self.rates = [d + n for d in self.dofs]
        # (k, c, the speed below which the film is held): per film.
        self.films = [(b.film.film_scale, b.film.clearance, b.film.with_load(abs(loads[j])).slowest())
                      for j, b in films]
        self.scale = 1.0 + max(abs(w) for w in system.bearing_loads)
        self.held = math.nan
        self.key: tuple | None = None
        self.kernel = None  # the Rust steps on the update, built when first stepped (`stepper`)

    def __getstate__(self) -> dict:
        # The Rust kernel doesn't copy or pickle: a copy builds its own from the same matrices.
        return {**self.__dict__, "kernel": None}

    def hold(self, omega: float) -> None:
        """Take the films' K_f and p at the speed Ω the exact update holds."""
        if omega == self.held:
            return
        Kf, Cf, p = self.system._film_terms(omega)
        d = self.dofs
        self.held = omega
        self.Kf, self.Cf = Kf[np.ix_(d, d)], Cf[np.ix_(d, d)]
        self.p = p[d] if self.system.gravity else np.zeros(len(d))
        self.A = self.system.state_matrix(omega)  # each take changes only the films' damping in it

    def bound(self, z: np.ndarray) -> list[tuple[float, float, float]]:
        """Each film's whole damping (c_xx, c_xy, c_yy) where its journal is in the state z."""
        x = z[self.dofs].tolist()
        return [full_film_damping(k, c, x[2 * i], x[2 * i + 1]) for i, (k, c, _) in enumerate(self.films)]

    def take(self, z: np.ndarray, h: float, omega: float) -> None:
        """Bound the damping where the journals are in z, and discretize over h at the held speed Ω."""
        system, n, d = self.system, self.system.n_dof, self.dofs
        self.hold(omega)
        self.taken = self.bound(z)
        self.size = max(abs(c) for b in self.taken for c in b)
        self.Cb = np.zeros((len(d), len(d)))
        for i, (cxx, cxy, cyy) in enumerate(self.taken):
            self.Cb[2 * i:2 * i + 2, 2 * i:2 * i + 2] = ((cxx, cxy), (cxy, cyy))
        A = self.A.copy()
        A[n:2 * n, self.rates] -= system.mass_inverse[:, d] @ (self.Cb - self.Cf)
        self.Phi, self.G0, self.G1 = foh_discretize(A, system.input_matrix(), h)
        self.R0, self.R1 = self.G0[:, d], self.G1[:, d]
        self.Ex, self.Ev = self.R1[d], self.R1[self.rates]  # the journals' x and x' at the step's end, per force
        self.E = np.vstack([self.Ex, self.Ev])
        self.Es = 0.5 * (self.Ev + self.Ev.T)  # ½ Δvᵀ sym(E_v⁻¹) Δv = ½ rᵀ sym(E_v) r
        self.lin = np.hstack([self.Kf, self.Cb])  # the update's films: p - K_f x - C_b x'
        self.eye = np.eye(len(d))
        self.key = (h, omega)
        self.kernel = None

    def stepper(self):
        """The Rust kernel's steps on this bound's update (`native.module.FilmSteps`)."""
        if self.kernel is None:
            self.kernel = native.module.FilmSteps(
                self.Phi, self.G0, self.G1, self.R0, self.R1, self.E, self.Es, self.Ev, self.lin, self.Kf,
                self.Cb, self.p, self.dofs + self.rates, self.films, self.taken, FILM_DRIFT * self.size,
                FILM_TOL * self.scale, FILM_CORRECTION * self.scale, FILM_ITERATIONS)
        return self.kernel

    def drifted(self, z: np.ndarray) -> bool:
        """True once a journal has moved far enough that the damping bound should be taken afresh."""
        limit = FILM_DRIFT * self.size
        return any(abs(a - b) > limit for new, old in zip(self.bound(z), self.taken) for a, b in zip(new, old))

    def forces(self, y: np.ndarray, omega: float) -> tuple[np.ndarray, np.ndarray, float]:
        """(f, D, Ψ) with the journals at y = [x, x']: the films' force, their stiffness and damping
        [K C] (each film's rows), and their dissipation."""
        m = len(self.dofs)
        f, D = np.empty(m), np.zeros((m, 2 * m))
        psi = 0.0
        for i, (k, c, slowest) in enumerate(self.films):
            j = 2 * i
            o = pi_film(k, c, y[j], y[j + 1], y[m + j], y[m + j + 1], max(abs(omega), slowest))
            f[j:j + 2] = o[:2]
            D[j:j + 2, j:j + 2] = (o[2:4], o[4:6])
            D[j:j + 2, m + j:m + j + 2] = ((o[6], o[7]), (o[7], o[8]))
            psi += o[9]
        return f, D, psi

    def remainder(self, z: np.ndarray, omega: float) -> np.ndarray:
        """r at the state z: the films' force less its part in the exact update."""
        y = z[self.dofs + self.rates]
        return self.forces(y.tolist(), omega)[0] - self.p + self.lin @ y

    def solve(self, zb: np.ndarray, r: np.ndarray, omega: float) -> np.ndarray:
        """r at the step's end, where z = zb + R1 r: Newton from the guess r.

        The journals' velocities settle where a convex function is least, Q =
        their inertia over the step, ½ rᵀ E_v r, plus the films' dissipation Ψ, less
        what the update carries: a Newton step that does not lower it, nor halve
        the residual, is cut back.
        """
        m = len(self.dofs)
        yb = zb[self.dofs + self.rates]
        tol = FILM_TOL * self.scale

        def at(r: np.ndarray) -> tuple:
            y = yb + self.E @ r
            f, D, psi = self.forces(y.tolist(), omega)
            x, v = y[:m], y[m:]
            q = 0.5 * r @ self.Es @ r + psi - 0.5 * v @ self.Cb @ v + (self.p - self.Kf @ x) @ v
            return r - f + self.p - self.lin @ y, D, q

        R, D, q = at(r)
        for _ in range(FILM_ITERATIONS):
            size = np.abs(R).max()
            if size <= tol:
                break
            dr = -np.linalg.solve(self.eye + (D - self.lin) @ self.E, R)
            if np.abs(dr).max() <= FILM_CORRECTION * self.scale:  # what is left is far smaller still
                return r + dr
            slope = R @ (self.Ev @ dr)
            t = 1.0
            while True:
                Rt, Dt, qt = at(r + t * dr)
                if qt <= q + 1e-4 * t * slope or np.abs(Rt).max() <= 0.5 * size or t < 1e-3:
                    break
                t *= 0.5
            r, R, D, q = r + t * dr, Rt, Dt, qt
        return r


class RotorSimulator:
    """Steps the rotor exactly (FOH matrix exponential) while its speed follows a target."""

    def __init__(self, system: RotorSystem | None = None) -> None:
        self.system = system or RotorSystem()
        self.t = 0.0
        self.phase = 0.0  # φ: angle of the keyphasor mark (rad, unwrapped)
        self.state = self._rest()
        self.omega = 0.0  # spin speed now (rad/s)
        self.target = 0.0  # the speed it ramps toward (rad/s)
        self.accel = 500.0 / RPM  # ramp rate (rad/s²)
        self._alpha = 0.0  # Ω' over the last step
        self._cache: dict[tuple[float, float], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._gravity_at: tuple[float, np.ndarray] = (math.nan, np.empty(0))  # gravity's load, and its speed
        self._f_at: dict[float, float] = {}  # with oil films: the highest frequency at each top speed asked
        self._f_still = self._highest_freq(0.0)
        self._films = _Films(self.system) if self.system.whole_films else None

    # ------------------------------------------------------------- setup
    def set_system(self, system: RotorSystem) -> None:
        """New parameters, keeping the motion (live editing).

        If the shaft's modal coordinates change in number, the stations keep their
        motion and the shaft starts again from its static shape between them. If
        the magnets' controllers change their states, those start where they hold
        the rotor's weight.
        """
        old = self.system
        if system.n_dof != old.n_dof or system.n_control != old.n_control:
            n0, n1 = old.n_dof, system.n_dof
            state = np.zeros(system.n_states)
            if n1 == n0:
                state[:2 * n1] = self.state[:2 * n0]
            else:
                state[:N_DOF], state[n1:n1 + N_DOF] = self.state[:N_DOF], self.state[n0:n0 + N_DOF]
            if system.n_control == old.n_control:
                state[2 * n1:] = self.state[2 * n0:]
            else:
                state[2 * n1:] = system.static_state(self.omega)[2 * n1:]
            self.state = state
        self.system = system
        self._cache.clear()
        self._gravity_at = (math.nan, np.empty(0))
        self._f_at.clear()
        self._f_still = self._highest_freq(0.0)
        self._films = _Films(system) if system.whole_films else None

    def _rest(self) -> np.ndarray:
        """The state at rest: the shaft in its bowed shape (which loads nothing), sagging under gravity."""
        s = self.system
        q = s.bow_displacement(self.phase) + s.rest_deflection(self.phase)
        z = np.concatenate([q, np.zeros_like(q), np.zeros(s.n_control)])
        if s.magnets:  # their controllers already holding the weight
            z[2 * s.n_dof:] = s.static_state(0.0, self.phase if s.periodic else None)[2 * s.n_dof:]
        return z

    def reset(self) -> None:
        """Back to rest, not spinning (the target speed is kept)."""
        self.t = self.phase = self.omega = self._alpha = 0.0
        self.state = self._rest()

    def tap(self, impulse_x: float = 0.0, impulse_y: float = 0.0) -> None:
        """Hit the disc: an impulse (N·s) in x and y changes the velocities at once, by M^{-1} times it."""
        Minv = self.system.mass_inverse
        n = self.system.n_dof
        self.state[n:2 * n] += Minv[:, XD] * impulse_x + Minv[:, YD] * impulse_y

    @property
    def q(self) -> np.ndarray:
        return self.state[:self.system.n_dof]

    @property
    def qdot(self) -> np.ndarray:
        return self.state[self.system.n_dof:2 * self.system.n_dof]

    @property
    def control(self) -> np.ndarray:
        """The magnets' controllers' states (see RotorSystem._control); empty without magnets."""
        return self.state[2 * self.system.n_dof:]

    def energy(self) -> tuple[float, float]:
        """(kinetic, potential) energy of the vibration (J); the spin's own energy left out."""
        M = self.system.matrices()[0]
        # Measured from the equilibrium: the shaft's strain from its bow (zero at the journals, where
        # the supports act), the springs' from where gravity holds them (gravity's own potential energy
        # cancels the rest). Only K's symmetric part stores energy: the cross-coupling does work instead.
        # An asymmetric shaft's strain energy is on its stiffness at this angle, about its mean sag. An oil
        # film's is on its stiffness at this speed (its skew part drops out of q K q).
        K = self.system.stiffness_at(self.omega, self.phase)
        q = self.q - self.system.bow_displacement(self.phase) - self.system.static_deflection(self.omega)
        return 0.5 * float(self.qdot @ M @ self.qdot), 0.5 * float(q @ K @ q)

    # ------------------------------------------------------------- stepping
    def _highest_freq(self, omega: float) -> float:
        """The fastest root at speed Ω (Hz), |λ|/2π; with oil films, the fastest oscillation, |Im λ|/2π.

        A film damps its journal so hard that its roots decay at once without
        oscillating, far faster than anything else: the exact update needs no
        short step for them, only for what oscillates. So with a magnet's
        controller, whose filter and amplifier are fast real roots.
        """
        lam = np.linalg.eigvals(self.system.state_matrix(omega))
        fast = np.abs(lam.imag) if self.system.films or self.system.magnets else np.abs(lam)
        return float(fast.max()) / (2.0 * math.pi)

    def step_size(self) -> float:
        """h resolving the fastest mode at any speed up to the target, and the spin.

        A forward mode's frequency rises with speed by at most (Ip/Id) Ω (the shaft's
        own sections: 2Ω). An oil film's stiffness changes with the speed, so with
        films the modes at the top speed are looked at too. With the films' whole
        force the exact update still carries every mode, and only the force on the
        journals is stepped: the step resolves the spin, and the modes only to
        FILM_MODE_STEPS a period. Rounded down to a power of two so the
        discretization is reused.
        """
        top = max(abs(self.omega), abs(self.target))
        spin = top / (2.0 * math.pi)
        gyro = max(self.system.ip / self.system.id, 2.0 if self.system.density else 0.0)
        fmax = max(self._f_still + spin * gyro, spin)
        if self.system.films:
            if top not in self._f_at:
                if len(self._f_at) > 64:
                    self._f_at.clear()
                self._f_at[top] = self._highest_freq(top)
            fmax = max(fmax, self._f_at[top])
        if self._films is not None:
            fmax = max(spin, fmax * FILM_MODE_STEPS / STEPS_PER_PERIOD)
        h = MAX_STEP if fmax <= 0 else min(MAX_STEP, 1.0 / (STEPS_PER_PERIOD * fmax))
        return 2.0 ** math.floor(math.log2(h))

    def _discrete(self, h: float, omega: float, cache: bool,
                  phase: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The FOH update over h at speed Ω; with `phase`, an asymmetric shaft standing still at that angle."""
        key = (h, omega, phase)
        if key in self._cache:
            return self._cache[key]
        A = self.system.state_matrix(omega)
        if phase is not None:
            n = self.system.n_dof
            A[n:2 * n, :n] = -self.system.mass_inverse @ self.system.stiffness_at(omega, phase)
        out = foh_discretize(A, self.system.input_matrix(), h)
        if cache:
            if len(self._cache) > 8:
                self._cache.clear()
            self._cache[key] = out
        return out

    def force(self, phase: np.ndarray, omega: np.ndarray, alpha: np.ndarray, held: float = 0.0) -> np.ndarray:
        """The load f (..., n_dof) on q: the 1X unbalance, skew moment and bow force, and gravity.

        Gravity's load with oil films is taken at the speed `held`, as A(Ω) is.
        """
        p = self.system.rotating_load(omega, alpha) * np.exp(1j * np.asarray(phase))[..., None]
        return self.system.plane_load(p) + self._gravity(held)

    def _gravity(self, omega: float) -> np.ndarray:
        if not self.system.films:
            omega = 0.0
        if self._gravity_at[0] != omega:
            self._gravity_at = (omega, self.system.gravity_load(omega))
        return self._gravity_at[1]

    def _speeds(self, h: float, n: int) -> np.ndarray:
        """Ω at the end of each of the next n steps, ramping toward the target at self.accel."""
        gap = self.target - self.omega
        if gap == 0.0:
            return np.full(n, self.target)
        ramp =self.omega + math.copysign(self.accel * h, gap) * np.arange(1, n + 1)
        return np.minimum(ramp, self.target) if gap > 0 else np.maximum(ramp, self.target)

    def advance(self, duration: float) -> RotorSamples:
        """Advance by about `duration` s of simulated time (at most MAX_STEPS_PER_ADVANCE steps)."""
        h = self.step_size()
        steps = min(MAX_STEPS_PER_ADVANCE, max(0, round(duration / h)))
        n_dof = self.system.n_dof
        ts, qs = np.empty(steps), np.empty((steps, n_dof))
        ws, phis = np.empty(steps), np.empty(steps)
        chunk = max(1, int(CHUNK_TIME / h))
        k = 0
        while k < steps:
            steady = self.omega == self.target
            n = steps - k if steady else min(steps - k, chunk)
            w1 = self._speeds(h, n)
            w0 = np.concatenate([[self.omega], w1[:-1]])
            alpha = (w1 - w0) / h
            phi1 = self.phase + np.cumsum(0.5 * h * (w0 + w1))  # exact for Ω linear over each step
            # A is held at the chunk's mean speed (exactly the speed when it is steady).
            held = float(np.mean(w1)) if not steady else self.omega
            u0 = self.force(self.phase, self.omega, self._alpha, held)
            u = self.force(phi1, w1, alpha, held)
            # A turning shaft standing still has a fixed stiffness: stepped exactly, like a round one.
            still = self.system.periodic and steady and self.omega == 0.0
            if self._films is not None:
                z = self._film_steps(h, held, w1, u0, u, qs[k:k + n])
            elif self.system.periodic and not still:
                z = self._split_steps(h, held, steady, w0, w1, alpha, u0, u, qs[k:k + n])
            else:
                Phi, G0, G1 = self._discrete(h, held, steady, self.phase if still else None)
                drive = G0 @ np.vstack([u0, u[:-1]]).T + G1 @ u.T
                z = self.state
                for j in range(n):
                    z = Phi @ z + drive[:, j]
                    qs[k + j] = z[:n_dof]
            self.state = z
            ts[k : k + n] = self.t + h * np.arange(1, n + 1)
            ws[k : k + n], phis[k : k + n] = w1, phi1
            self.t = float(ts[k + n - 1])
            self.omega, self.phase, self._alpha = float(w1[-1]), float(phi1[-1]), float(alpha[-1])
            k += n
        return RotorSamples(ts, qs, ws, phis)

    def _film_steps(self, h: float, held: float, w1: np.ndarray, u0: np.ndarray, u: np.ndarray,
                    out: np.ndarray) -> np.ndarray:
        """`_film_steps_reference`'s steps, by the Rust kernel (`_Films.stepper`) where it's built.

        It steps until a journal drifts; the bound is taken afresh here (a matrix
        exponential), and it carries on from there.
        """
        if native.module is None:
            return self._film_steps_reference(h, held, w1, u0, u, out)
        films = self._films
        films.hold(held)
        z = self.state
        if films.key != (h, held) or films.drifted(z):
            films.take(z, h, held)
        r = films.remainder(z, self.omega)
        last = older = r
        u0, u = np.ascontiguousarray(u0, dtype=float), np.ascontiguousarray(u, dtype=float)
        j = 0
        while True:
            j, z, r, last, older = films.stepper().steps(z, r, last, older, u0, u, w1, j, out)
            z = np.asarray(z)
            if j == w1.size:
                return z
            films.take(z, h, held)
            r = last = older = films.remainder(z, w1[j - 1])

    def _film_steps_reference(self, h: float, held: float, w1: np.ndarray, u0: np.ndarray, u: np.ndarray,
                              out: np.ndarray) -> np.ndarray:
        """Steps with the oil films' whole force (`whole_films`); fills out (n, n_dof), returns the last state.

        The film's force depends on where the journal is and how it moves, far from
        linearly once it moves a fair part of its clearance. Each step is the exact
        update of the films linearized (see _Films), plus their remainder r_f as a
        force on the journals, linear over the step (as the FOH takes the loads):

            z1 = Φ z0 + Γ0 (u0 + r_f0) + Γ1 (u1 + r_f1),

        r_f1 found by Newton, since it depends on z1. The films damp their journals
        far faster than the step, so r1 must be implicit; and their damping, the
        stiff part of the remainder, is no more than the bound C_b in the update
        carries, so the remainder is never stiffer than the exact part (the
        trapezoidal rule would ring if it were). C_b is taken afresh, and the
        update with it, as the journals move round; second order in h.
        """
        films, n_dof = self._films, self.system.n_dof
        films.hold(held)
        z = self.state
        if films.key != (h, held):
            films.take(z, h, held)
        r = films.remainder(z, self.omega)
        last = older = r
        for j in range(w1.size):
            if films.drifted(z):
                films.take(z, h, held)
                r = last = older = films.remainder(z, w1[j - 1] if j else self.omega)
            zb = films.Phi @ z + films.G0 @ (u[j - 1] if j else u0) + films.G1 @ u[j] + films.R0 @ r
            guess = 3.0 * (r - last) + older  # r on the parabola through the last three steps
            r, last, older = films.solve(zb, guess, w1[j]), r, last
            z = zb + films.R1 @ r
            out[j] = z[:n_dof]
        return z

    def _split_steps(self, h: float, held: float, steady: bool, w0: np.ndarray, w1: np.ndarray,
                     alpha: np.ndarray, u0: np.ndarray, u: np.ndarray, out: np.ndarray) -> np.ndarray:
        """Steps with an asymmetric or cracked shaft (Strang splitting); fills out (n, n_dof), returns the last state.

        Its stiffness turns with it, so no one matrix exponential steps it. Each step
        is half a step of everything else (the exact FOH update, as for a round
        shaft, on the mean stiffness), then the turning part at mid-step as a kick
        to the velocities (exact for it alone: v -= h M^{-1} ΔK(φ) q), then the
        other half step. Second order in h, and stable: ΔK is smaller than the
        shaft's stiffness, whose fastest mode the step already resolves.
        """
        system, n_dof = self.system, self.system.n_dof
        n = w1.size
        Phi, G0, G1 = self._discrete(0.5 * h, held, steady)
        wm = 0.5 * (w0 + w1)
        phi0 = np.concatenate([[self.phase], self.phase + np.cumsum(0.5 * h * (w0 + w1))[:-1]])
        phim = phi0 + 0.25 * h * (w0 + wm)  # exact for Ω linear over the step
        um = self.force(phim, wm, alpha, held)
        before = np.vstack([u0, u[:-1]])
        first = G0 @ before.T + G1 @ um.T
        second = G0 @ um.T + G1 @ u.T
        ks, S = system._kick
        angle = np.outer(phim, ks)
        weights = h * np.hstack([np.cos(angle), np.sin(angle)])  # (n, 2m)
        z = self.state
        for j in range(n):
            z = Phi @ z + first[:, j]
            z[n_dof:2 * n_dof] -= weights[j] @ (S @ z[:n_dof])
            z = Phi @ z + second[:, j]
            out[j] = z[:n_dof]
        return z
