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
in y, which may differ (anisotropic supports) and may differ between A and B.
The disc has mass m, polar moment of inertia Ip (about the spin axis) and
diametral moment Id (about a diameter).

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
  Balance weights m_k r_k at angles θ_k add to U e^{iψ} the same way.
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

Time stepping uses the state z = [q, q'], z' = A(Ω) z + B u, with the same exact
first-order-hold matrix-exponential update as the chain's Simulator. A depends
on Ω, so while the speed ramps A is held at the speed at the middle of each short
chunk of steps (CHUNK_TIME), and the unbalance force follows the exact phase.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from functools import cached_property

import numpy as np
import scipy.linalg

from .simulator import MAX_STEP, MAX_STEPS_PER_ADVANCE, STEPS_PER_PERIOD, foh_discretize

N_DOF = 8
# Index of each DOF in q.
XA, XD, TX, XB, YA, YD, TY, YB = range(N_DOF)
DOF_NAMES = ("x_A", "x_D", "θx", "x_B", "y_A", "y_D", "θy", "y_B")
CHUNK_TIME = 2e-3  # s; while the speed ramps, A(Ω) is held over chunks this long
RPM = 60.0 / (2.0 * math.pi)  # rpm per rad/s
MIN_GAP = 1e-3  # of the span: the closest the disc may come to a bearing


@dataclass(frozen=True)
class Bearing:
    """A bearing journal on its support: springs and dampers to ground in x and y."""

    kx: float = 5e4  # N/m
    ky: float = 5e4  # N/m
    cx: float = 100.0  # N·s/m
    cy: float = 100.0  # N·s/m
    mass: float = 0.1  # kg, the journal and whatever moves with it
    kxy: float = 0.0  # N/m, cross-coupled stiffness: f_x = -k_xy y, f_y = +k_xy x (k_yx = -k_xy)
    swirl: float = 0.0  # fluid swirl ratio λ: the cross-coupling grows with speed by λ c Ω (see damping)

    @property
    def damping(self) -> float:
        """c (N·s/m), the mean of c_x and c_y: the fluid's drag-round takes k_xy up by λ c Ω."""
        return 0.5 * (self.cx + self.cy)


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
    gravity: float = 0.0  # m/s², pulling in -y (9.81 for a horizontal rotor; 0 for a vertical one)

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

    def with_(self, **changes) -> RotorSystem:
        return replace(self, **changes)

    @property
    def mass_unbalance(self) -> complex:
        """The disc's net mass unbalance (kg·m), the unbalance and the weights: |·| m e, its angle the heavy spot's."""
        return self.unbalance * complex(math.cos(self.unbalance_angle), math.sin(self.unbalance_angle)) + sum(
            self.weights, 0j)

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

    def _elements(self) -> list[tuple[int, int, float, float]]:
        """The shaft's two beam elements in axial order: (first node, second node, z of the first, length).

        Nodes are 0 (bearing A), 1 (the disc) and 2 (bearing B).
        """
        nodes = sorted(((0.0, 0), (self.a, 1), (self.length, 2)))
        return [(n0, n1, z0, z1 - z0) for (z0, n0), (z1, n1) in zip(nodes, nodes[1:])]

    def _shaft_condensation(self) -> tuple[np.ndarray, np.ndarray]:
        """(K_shaft, T): one plane's 4x4 shaft stiffness on [w_A, w_D, θ_D, w_B], and T (6x4)."""
        K, T = self._condensed
        return K.copy(), T.copy()

    @cached_property
    def _condensed(self) -> tuple[np.ndarray, np.ndarray]:
        """_shaft_condensation's result, worked out once (the system is frozen).

        The shaft is two beam elements (see _elements) on [w_A, θ_A, w_D, θ_D, w_B, θ_B].
        The bearings take no moment, so θ_A and θ_B are condensed out: full = T @ retained.
        """
        k6 = np.zeros((6, 6))
        for n0, n1, _, length in self._elements():
            dofs = [2 * n0, 2 * n0 + 1, 2 * n1, 2 * n1 + 1]
            k6[np.ix_(dofs, dofs)] += beam_stiffness(self.ei, length)
        keep, drop = [0, 2, 3, 4], [1, 5]
        krr = k6[np.ix_(keep, keep)]
        krc = k6[np.ix_(keep, drop)]
        kcc = k6[np.ix_(drop, drop)]
        recover = -np.linalg.solve(kcc, krc.T)  # θ_A, θ_B from the retained DOFs
        T = np.zeros((6, 4))
        T[keep] = np.eye(4)
        T[drop] = recover
        return krr + krc @ recover, T

    def shaft_stiffness(self) -> np.ndarray:
        """One plane's shaft stiffness (4x4) on [w_A, w_D, θ_D, w_B]."""
        return self._shaft_condensation()[0]

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
        """One plane's bow (4,) on [w_A, w_D, θ_D, w_B], per metre of bow at the disc.

        The shape a load at the disc gives with the bearings held: no moment at the
        disc, so K_shaft times it is a force at the disc and the bearings' reactions.
        """
        k = self.shaft_stiffness()[1:3, 1:3]
        w, theta = np.linalg.solve(k, [1.0, 0.0])
        return np.array([0.0, 1.0, theta / w, 0.0])

    def rotating_load(self, omega: float | np.ndarray, alpha: float | np.ndarray = 0.0) -> np.ndarray:
        """The 1X load on [A, D, θ, B] as f_x + i f_y = P e^{iφ}: P (..., 4) complex.

        The unbalance and the balance weights on the disc, the skew's moment on its
        slopes and the bow's force K_shaft q_bow, at speed Ω and acceleration Ω' = alpha.
        """
        drive = np.asarray(omega, dtype=float) ** 2 - 1j * np.asarray(alpha, dtype=float)
        per_drive, bow = self._load_parts
        return drive[..., None] * per_drive + bow

    @cached_property
    def _load_parts(self) -> tuple[np.ndarray, np.ndarray]:
        """(the load per unit Ω² - iΩ', the bow's constant load), each (4,) complex."""
        per_drive = np.array([0.0, self.mass_unbalance,
                              (self.id - self.ip) * self.skew * np.exp(1j * self.skew_angle), 0.0])
        bow = self.shaft_stiffness() @ self.bow_shape() * self.bow * np.exp(1j * self.bow_angle)
        return per_drive, bow

    def gravity_load(self) -> np.ndarray:
        """f_g (8,): the weights of the journals and the disc, in -y (N)."""
        f = np.zeros(N_DOF)
        f[[YA, YD, YB]] = -self.gravity * np.array([self.bearing_a.mass, self.disc_mass, self.bearing_b.mass])
        return f

    def static_deflection(self, omega: float | np.ndarray = 0.0) -> np.ndarray:
        """q_s = (K + ΩH)^{-1} f_g: where gravity holds the rotor, (8,) or (len(omega), 8).

        Unless the stiffness depends on the speed (speed_dependent), neither does this.
        """
        omega = np.asarray(omega, dtype=float)
        if not self.gravity:
            return np.zeros(omega.shape + (N_DOF,))
        f = self.gravity_load()
        if not self.speed_dependent or omega.ndim == 0:
            q = np.linalg.solve(self.stiffness(float(omega) if omega.ndim == 0 else 0.0), f)
            return q if omega.ndim == 0 else np.broadcast_to(q, omega.shape + (N_DOF,)).copy()
        K, H = self.matrices()[3], self._circulatory
        return np.linalg.solve(K + omega[..., None, None] * H, np.broadcast_to(f, omega.shape + (N_DOF,))[..., None])[..., 0]

    def bow_displacement(self, phase: float) -> np.ndarray:
        """q_bow (8,): the bow at shaft angle φ, as displacements in fixed coordinates."""
        b = self.bow_shape() * self.bow * np.exp(1j * (phase + self.bow_angle))
        return np.concatenate([b.real, b.imag])

    def shape_matrix(self, z: np.ndarray) -> np.ndarray:
        """S (len(z) x 4): the shaft's deflection at axial positions z (m) is S @ [w_A, w_D, θ_D, w_B].

        Exact for the massless shaft: between the stations it bends as a cubic.
        """
        z = np.atleast_1d(np.asarray(z, dtype=float))
        T = self._shaft_condensation()[1]
        S = np.zeros((z.size, 4))
        (n0, n1, z0, l0), (m0, m1, z1, l1) = self._elements()
        # Each element takes its own stretch; beyond the shaft's ends the nearer one carries on.
        for (a, b, start, length), inside in (((n0, n1, z0, l0), z < z1), ((m0, m1, z1, l1), z >= z1)):
            s = (z[inside] - start) / length
            h = np.column_stack([1 - 3 * s**2 + 2 * s**3, length * (s - 2 * s**2 + s**3),
                                 3 * s**2 - 2 * s**3, length * (s**3 - s**2)])
            S[inside] = h @ T[[2 * a, 2 * a + 1, 2 * b, 2 * b + 1]]
        return S

    @property
    def extent(self) -> tuple[float, float]:
        """The shaft's ends (m): the bearings, or the disc where it overhangs one."""
        return min(0.0, self.a), max(self.length, self.a)

    def deflection(self, q: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Shaft centre (x, y) at axial positions z for displacements q: shape (..., len(z), 2)."""
        S = self.shape_matrix(z)
        q = np.asarray(q)
        return np.stack([q[..., :4] @ S.T, q[..., 4:] @ S.T], axis=-1)

    # ------------------------------------------------------------- matrices
    def matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(M, C, G, K), each 8x8 on q = [x_A, x_D, θx, x_B, y_A, y_D, θy, y_B].

        The equations of motion are M q'' + (C + Ω G) q' + (K + Ω H) q = f, H from
        circulatory(). K holds the bearings' cross-coupled stiffness (so it is not
        symmetric unless k_xy = 0) and C the shaft's internal damping.
        """
        A, B = self.bearing_a, self.bearing_b
        ks = self.shaft_stiffness()
        M = np.diag([A.mass, self.disc_mass, self.id, B.mass] * 2)
        K = np.zeros((N_DOF, N_DOF))
        K[:4, :4] = ks + np.diag([A.kx, 0.0, 0.0, B.kx])
        K[4:, 4:] = ks + np.diag([A.ky, 0.0, 0.0, B.ky])
        for j, bearing in ((XA, A), (XB, B)):
            K[j, j + 4], K[j + 4, j] = bearing.kxy, -bearing.kxy
        C = np.diag([A.cx, 0.0, 0.0, B.cx, A.cy, 0.0, 0.0, B.cy])
        C[:4, :4] += self.loss_time * ks
        C[4:, 4:] += self.loss_time * ks
        G = np.zeros((N_DOF, N_DOF))
        # Id θx'' + Ω Ip θy' = moment in x-z;  Id θy'' - Ω Ip θx' = moment in y-z.
        G[TX, TY] = self.ip
        G[TY, TX] = -self.ip
        return M, C, G, K

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
        H = np.zeros((N_DOF, N_DOF))
        eta_ks = self.loss_time * self.shaft_stiffness()
        H[:4, 4:], H[4:, :4] = eta_ks, -eta_ks
        for j, bearing in ((XA, self.bearing_a), (XB, self.bearing_b)):
            H[j, j + 4] += bearing.swirl * bearing.damping
            H[j + 4, j] -= bearing.swirl * bearing.damping
        return H

    @property
    def speed_dependent(self) -> bool:
        """True if the stiffness changes with the speed: internal damping or a swirling bearing."""
        return bool(self.internal_damping or self.bearing_a.swirl * self.bearing_a.damping
                    or self.bearing_b.swirl * self.bearing_b.damping)

    def cross_coupling(self, omega: float) -> tuple[float, float]:
        """Each bearing's cross-coupled stiffness k_xy at speed Ω (N/m), A then B: k_xy + λ c Ω."""
        return tuple(b.kxy + b.swirl * b.damping * omega for b in (self.bearing_a, self.bearing_b))

    def stiffness(self, omega: float) -> np.ndarray:
        """K + Ω H: the whole stiffness at speed Ω."""
        K = self.matrices()[3]
        return K + omega * self._circulatory if self.speed_dependent else K

    def state_matrix(self, omega: float) -> np.ndarray:
        """A(Ω) (16x16) for z = [q, q']: z' = A z + B u."""
        M, C, G, _ = self.matrices()
        Minv = np.diag(1.0 / np.diag(M))
        A = np.zeros((2 * N_DOF, 2 * N_DOF))
        A[:N_DOF, N_DOF:] = np.eye(N_DOF)
        A[N_DOF:, :N_DOF] = -Minv @ self.stiffness(omega)
        A[N_DOF:, N_DOF:] = -Minv @ (C + omega * G)
        return A

    def input_matrix(self) -> np.ndarray:
        """B (16x8): the inputs u = f (8,), a force or moment on each DOF."""
        B = np.zeros((2 * N_DOF, N_DOF))
        A, Bb = self.bearing_a, self.bearing_b
        B[N_DOF:] = np.diag(1.0 / np.array([A.mass, self.disc_mass, self.id, Bb.mass] * 2))
        return B

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


# ------------------------------------------------------------------ whirl analysis


def whirl_ratio(system: RotorSystem, Q: np.ndarray) -> np.ndarray:
    """Whirl direction of complex amplitudes Q (..., 8), x = Re(X e^{iωt}) for ω > 0.

    +1 is a circular forward whirl (with the spin, x toward y), -1 a circular
    backward whirl, 0 a straight line. The four stations (journals, disc, disc
    tilt) are weighted by their masses and Id, i.e. by kinetic energy.
    """
    M = np.diag(system.matrices()[0])[:4]
    X, Y = Q[..., :4], Q[..., 4:]
    num = 2.0 * np.sum(M * np.imag(X * np.conj(Y)), axis=-1)
    den = np.sum(M * (np.abs(X) ** 2 + np.abs(Y) ** 2), axis=-1)
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def _whirl_forms(system: RotorSystem) -> tuple[np.ndarray, np.ndarray]:
    """Hermitian (H, D) with v^H H v and v^H D v the numerator and denominator of whirl_ratio."""
    m = np.diag(system.matrices()[0])[:4]
    H = np.zeros((N_DOF, N_DOF), dtype=complex)
    H[:4, 4:] = 1j * np.diag(m)
    H[4:, :4] = -1j * np.diag(m)
    return H, np.diag(np.concatenate([m, m])).astype(complex)


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
    shapes: np.ndarray  # (n, 8) complex displacement shapes, largest entry 1

    @property
    def forward(self) -> np.ndarray:
        return self.whirl >= 0.0


def whirl_modes(system: RotorSystem, omega: float) -> WhirlModes:
    """Eigen-solution of A(Ω): the oscillatory modes (Im λ > 0), lowest first."""
    lam, vec = np.linalg.eig(system.state_matrix(omega))
    keep = lam.imag > 1e-9 * max(1.0, float(np.abs(lam).max()))
    lam, vec = lam[keep], vec[:N_DOF, keep]
    order = np.argsort(lam.imag)
    lam, vec = lam[order], vec[:, order]
    _circularize(system, lam, vec)
    shapes = vec.T.copy()
    for s in shapes:
        s /= s[np.argmax(np.abs(s))]
    return WhirlModes(
        omega=omega,
        freq_hz=lam.imag / (2.0 * math.pi),
        zeta=-lam.real / np.abs(lam),
        whirl=whirl_ratio(system, shapes),
        shapes=shapes,
    )


@dataclass
class Campbell:
    """Damped natural frequencies against spin speed, and the critical speeds."""

    omega: np.ndarray  # (n_speeds,) rad/s
    freq_hz: np.ndarray  # (n_speeds, n_modes), each column one frequency-ordered branch
    zeta: np.ndarray  # (n_speeds, n_modes)
    whirl: np.ndarray  # (n_speeds, n_modes), > 0 forward
    criticals: list[tuple[float, float, float]]  # (speed in rad/s, whirl ratio, ζ) where a branch meets 1X


def campbell(system: RotorSystem, omega_max: float, n_speeds: int = 241, n_modes: int = 6) -> Campbell:
    """Sweep the speed 0..omega_max and find where each branch crosses the synchronous (1X) line."""
    omegas = np.linspace(0.0, omega_max, n_speeds)
    freq = np.full((n_speeds, n_modes), np.nan)
    zeta = np.full((n_speeds, n_modes), np.nan)
    whirl = np.zeros((n_speeds, n_modes))
    for i, w in enumerate(omegas):
        m = whirl_modes(system, w)
        k = min(n_modes, m.freq_hz.size)
        freq[i, :k], zeta[i, :k], whirl[i, :k] = m.freq_hz[:k], m.zeta[:k], m.whirl[:k]
    criticals = []
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
    """Steady-state 1X response at constant speeds: Q (len(omegas), 8), complex.

    The unbalance, the disc's skew and the shaft's bow together. q(t) = Re(Q e^{iΩt}),
    with the keyphasor mark at angle φ = Ωt (along +x at t = 0). Above
    the stability onset this particular solution still exists, but the free whirl
    grows over it, so the rotor never settles to it.
    """
    M, C, G, _ = system.matrices()
    omegas = np.atleast_1d(np.asarray(omegas, dtype=float))
    out = np.empty((omegas.size, N_DOF), dtype=complex)
    load = system.rotating_load(omegas)
    for i, w in enumerate(omegas):
        # f_x + i f_y = P e^{iφ}: f_x = Re(P e^{iφ}), f_y = Re(-iP e^{iφ}).
        f = np.concatenate([load[i], -1j * load[i]])
        out[i] = np.linalg.solve(system.stiffness(w) - w * w * M + 1j * w * (C + w * G), f)
    return out


# ------------------------------------------------------------------ time stepping


@dataclass
class RotorSamples:
    """What one advance returns, one row per step."""

    t: np.ndarray  # (k,) s
    q: np.ndarray  # (k, 8) displacements
    omega: np.ndarray  # (k,) spin speed (rad/s)
    phase: np.ndarray  # (k,) angle φ of the keyphasor mark (rad, unwrapped)


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
        self._f_still = self._highest_freq_still()

    # ------------------------------------------------------------- setup
    def set_system(self, system: RotorSystem) -> None:
        """New parameters, keeping the motion (live editing)."""
        self.system = system
        self._cache.clear()
        self._f_still = self._highest_freq_still()

    def _rest(self) -> np.ndarray:
        """The state at rest: the shaft in its bowed shape (which loads nothing), sagging under gravity."""
        q = self.system.bow_displacement(self.phase) + self.system.static_deflection(0.0)
        return np.concatenate([q, np.zeros(N_DOF)])

    def reset(self) -> None:
        """Back to rest, not spinning (the target speed is kept)."""
        self.t = self.phase = self.omega = self._alpha = 0.0
        self.state = self._rest()

    def tap(self, impulse_x: float = 0.0, impulse_y: float = 0.0) -> None:
        """Hit the disc: an impulse (N·s) in x and y changes its velocity at once."""
        self.state[N_DOF + XD] += impulse_x / self.system.disc_mass
        self.state[N_DOF + YD] += impulse_y / self.system.disc_mass

    @property
    def q(self) -> np.ndarray:
        return self.state[:N_DOF]

    @property
    def qdot(self) -> np.ndarray:
        return self.state[N_DOF:]

    def energy(self) -> tuple[float, float]:
        """(kinetic, potential) energy of the vibration (J); the spin's own energy left out."""
        M, _, _, K = self.system.matrices()
        # Measured from the equilibrium: the shaft's strain from its bow (zero at the journals, where
        # the supports act), the springs' from where gravity holds them (gravity's own potential energy
        # cancels the rest). Only K's symmetric part stores energy: the cross-coupling does work instead.
        q = self.q - self.system.bow_displacement(self.phase) - self.system.static_deflection(self.omega)
        return 0.5 * float(self.qdot @ M @ self.qdot), 0.5 * float(q @ K @ q)

    # ------------------------------------------------------------- stepping
    def _highest_freq_still(self) -> float:
        lam = np.linalg.eigvals(self.system.state_matrix(0.0))
        return float(np.abs(lam).max()) / (2.0 * math.pi)

    def step_size(self) -> float:
        """h resolving the fastest mode at any speed up to the target, and the spin.

        A forward mode's frequency rises with speed by at most (Ip/Id) Ω. Rounded
        down to a power of two so the discretization is reused.
        """
        spin = max(abs(self.omega), abs(self.target)) / (2.0 * math.pi)
        fmax = max(self._f_still + spin * self.system.ip / self.system.id, spin)
        h = MAX_STEP if fmax <= 0 else min(MAX_STEP, 1.0 / (STEPS_PER_PERIOD * fmax))
        return 2.0 ** math.floor(math.log2(h))

    def _discrete(self, h: float, omega: float, cache: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = (h, omega)
        if key in self._cache:
            return self._cache[key]
        out = foh_discretize(self.system.state_matrix(omega), self.system.input_matrix(), h)
        if cache:
            if len(self._cache) > 8:
                self._cache.clear()
            self._cache[key] = out
        return out

    def force(self, phase: np.ndarray, omega: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """The load f (..., 8) on q: the 1X unbalance, skew moment and bow force, and gravity."""
        p = self.system.rotating_load(omega, alpha) * np.exp(1j * np.asarray(phase))[..., None]
        return np.concatenate([p.real, p.imag], axis=-1) + self.system.gravity_load()

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
        ts, qs = np.empty(steps), np.empty((steps, N_DOF))
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
            u0 = self.force(self.phase, self.omega, self._alpha)
            u = self.force(phi1, w1, alpha)
            # A is held at the chunk's mean speed (exactly the speed when it is steady).
            Phi, G0, G1 = self._discrete(h, float(np.mean(w1)) if not steady else self.omega, steady)
            drive = G0 @ np.vstack([u0, u[:-1]]).T + G1 @ u.T
            z = self.state
            for j in range(n):
                z = Phi @ z + drive[:, j]
                qs[k + j] = z[:N_DOF]
            self.state = z
            ts[k : k + n] = self.t + h * np.arange(1, n + 1)
            ws[k : k + n], phis[k : k + n] = w1, phi1
            self.t = float(ts[k + n - 1])
            self.omega, self.phase, self._alpha = float(w1[-1]), float(phi1[-1]), float(alpha[-1])
            k += n
        return RotorSamples(ts, qs, ws, phis)
