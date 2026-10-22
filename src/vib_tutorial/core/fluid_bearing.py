"""Plain journal bearing by short-bearing (Ocvirk) theory: its equilibrium and linearized coefficients.

A journal of radius R = D/2 turns at Ω in a bearing of length L with radial
clearance c, the gap full of oil of viscosity μ, and carries a steady load W
straight down (-y). x is horizontal, y up, and the journal spins from x toward
y, as on the rotor page. The film is h = c - x cos θ - y sin θ round the bearing
(θ from +x, in the direction of spin), for the journal's centre at (x, y).

Reynolds' equation for the film's pressure, with the bearing short beside its
diameter (L/D up to about 1/2), keeps only the axial pressure flow:

    ∂/∂z (h³ ∂p/∂z) = 6μ (Ω ∂h/∂θ + 2 ∂h/∂t),

so p = 3μ (z² - L²/4)(Ω ∂h/∂θ + 2 ∂h/∂t) / h³, a parabola along the bearing,
zero at its ends. Where that comes out negative the oil cavitates instead, and
the pressure is taken as zero there (the Gümbel, or π-film, condition: at rest
the film carries load over the half where it narrows). The pressure, summed
over the journal's surface, is the film's force on it (`film_force`).

At rest the journal settles where that force holds the load. Its eccentricity
ε = e/c follows from the load alone through Ocvirk's (modified Sommerfeld)
number S_s = D Ω μ L³ / (8 W c²),

    S_s ε √(π² (1 - ε²) + 16 ε²) = (1 - ε²)²,

and it sits off the load line by the attitude angle χ, turned in the direction
of spin: tan χ = π √(1 - ε²) / (4 ε). Slow or heavily loaded, ε → 1 and χ → 0,
the journal sinking to the bottom of its clearance; fast or lightly loaded,
ε → 0 and χ → 90°, the journal going sideways to the centre. The film lifts the
load by pushing sideways: that sideways push is where the cross-coupled
stiffness comes from.

Small motions about there are resisted by the eight linearized coefficients,
f = -K Δq - C Δq' on the journal (Δq = [Δx, Δy]), in closed form for the short
bearing (Friswell, Penny, Garvey & Lees, Dynamics of Rotating Machines, 2010,
§5.5):

    K = (W / c) a(ε),    C = (W / (c Ω)) b(ε).

K is not symmetric, and its skew part feeds a forward whirl (oil whirl); C is
symmetric. At every load k_xy - k_yx = (Ω/2)(c_xx + c_yy): the skew stiffness
is what a film swirling at half the shaft's speed gives, the rotor page's
λ = 1/2.

The classical Sommerfeld number S = (μ N L D / W)(R / c)² (N in rev/s) is the
one bearing charts use; S_s = π (L/D)² S.

Away from equilibrium the film's whole force has a closed form too (`pi_film`).
Turned to the line of centres (the journal at eccentricity ε, angle ψ), Ω ∂h/∂θ
+ 2 ∂h/∂t = c (a sin φ + b cos φ) for φ from the line of centres, with a =
ε(Ω - 2ψ') and b = -2ε'. The film carries pressure over the half where that is
negative, φ from π - δ to 2π - δ (δ the angle of (a, b)): the cavitation
boundary moves with the journal's velocity. The force is then k (a, b) times
integrals of sin²φ, sin φ cos φ and cos²φ over (1 - ε cos φ)³ on that half,
k = R μ L³ / (2 c³) (`film_scale`). Written in the angle u with tan(u/2) =
√((1 + ε)/(1 - ε)) tan(φ/2) (as the eccentric anomaly is in an orbit), each is
a short trigonometric polynomial in u, so they and their derivatives come in
closed form, whatever ε < 1.

The force is minus the gradient, in the journal's velocity, of the film's
dissipation Ψ = (k/4)(a² I_ss + 2ab I_sc + b² I_cc): its damping, the Hessian,
is symmetric and positive wherever the journal is, and Ψ is convex. Its damping
is never more than the whole (uncavitated) film's at that position
(`full_film_damping`): that integrates the same positive terms round the whole
bearing, not half of it. The rotor's nonlinear time step leans on both.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import scipy.optimize

from . import native

GRAVITY = 9.81  # m/s²
PI2 = math.pi**2
LIGHTEST = 1e-6  # the least S_s⁻¹ taken: an unloaded bearing is worked out as one this lightly loaded
ECCENTRICITY_LIMIT = 0.9  # on the rotor, slower than this ε the film is held as it is here (see `on_rotor`)
SLOWEST = 1.0  # rad/s: and never slower than this
ECCENTRICITY_MAX = 1.0 - 1e-6  # the whole film's force is worked out no nearer the wall than this


@dataclass(frozen=True)
class JournalBearing:
    """A plain journal bearing, short-bearing theory: dimensions, oil and load. SI units."""

    length: float = 5e-3  # m, L
    diameter: float = 10e-3  # m, D
    clearance: float = 15e-6  # m, radial clearance c
    viscosity: float = 0.03  # Pa·s, μ (ISO VG 32 at about 40 °C)
    load: float = 5.0  # N, W, straight down

    @property
    def radius(self) -> float:
        return 0.5 * self.diameter

    def modified_sommerfeld(self, omega: float | np.ndarray) -> float | np.ndarray:
        """Ocvirk's number S_s = D Ω μ L³ / (8 W c²): the short bearing's ε depends on it alone."""
        return self.diameter * omega * self.viscosity * self.length**3 / (8.0 * self.load * self.clearance**2)

    def sommerfeld(self, omega: float | np.ndarray) -> float | np.ndarray:
        """The classical Sommerfeld number S = (μ N L D / W)(R / c)², N = Ω / 2π in rev/s."""
        return self.modified_sommerfeld(omega) / (math.pi * (self.length / self.diameter) ** 2)

    def omega_at(self, sommerfeld: float | np.ndarray) -> float | np.ndarray:
        """The speed Ω (rad/s) at which the Sommerfeld number S is `sommerfeld`."""
        return sommerfeld / self.sommerfeld(1.0)

    def state(self, omega: float) -> BearingState:
        """The bearing's equilibrium and coefficients at speed Ω (rad/s, > 0).

        An unloaded (or barely loaded) bearing is worked out as one with S_s =
        1/LIGHTEST: its journal at the centre, its coefficients their limits as
        the load goes to zero.
        """
        s = float(self.modified_sommerfeld(omega)) if self.load > 0.0 else math.inf
        load = self.load
        if not s < 1.0 / LIGHTEST:
            s = 1.0 / LIGHTEST
            load = float(self.with_load(1.0).modified_sommerfeld(omega)) * LIGHTEST
        eps = eccentricity(s)
        a, b = coefficients(eps)
        w_c = load / self.clearance
        return BearingState(omega, eps, attitude(eps), w_c * a, w_c / omega * b)

    def with_load(self, load: float) -> JournalBearing:
        return replace(self, load=load)

    def slowest(self) -> float:
        """The speed (rad/s) below which `on_rotor` holds the film: where ε = ECCENTRICITY_LIMIT."""
        e = ECCENTRICITY_LIMIT
        s = (1.0 - e * e) ** 2 / (e * math.sqrt(PI2 * (1.0 - e * e) + 16.0 * e * e))
        return max(s / float(self.modified_sommerfeld(1.0)), SLOWEST) if self.load > 0.0 else SLOWEST

    def on_rotor(self, omega: float) -> BearingState:
        """The state at speed Ω as the rotor uses it: held at `slowest()` below it.

        Slower, the journal is nearly on its bottom (mixed lubrication), where the
        film's stiffness and damping grow without bound and the short-bearing
        model is not to be trusted; the rotor takes it as it is at ε = 0.9.
        """
        return self.state(max(abs(omega), self.slowest()))

    def film_force(self, x: float, y: float, vx: float, vy: float, omega: float,
                   points: int = 4000) -> np.ndarray:
        """The film's force [f_x, f_y] (N) on the journal at (x, y) moving at (vx, vy), by quadrature.

        The short-bearing pressure, cavitated where it would be negative, summed
        round the bearing at `points` angles. Only for checks: K and C, and
        the force itself (`force`), come in closed form.
        """
        theta = np.arange(points) * (2.0 * math.pi / points)
        cos, sin = np.cos(theta), np.sin(theta)
        h = self.clearance - x * cos - y * sin
        if np.any(h <= 0.0):
            raise ValueError("the journal touches the bearing")
        g = omega * (x * sin - y * cos) - 2.0 * (vx * cos + vy * sin)  # Ω ∂h/∂θ + 2 ∂h/∂t
        p = np.maximum(-self.viscosity * self.length**3 / (2.0 * h**3) * g, 0.0)  # summed along z
        d = self.radius * 2.0 * math.pi / points
        return -d * np.array([p @ cos, p @ sin])

    @property
    def film_scale(self) -> float:
        """k = R μ L³ / (2 c³) (N·s/m): the whole film's force is k times c (a, b) times the arc's integrals."""
        return self.radius * self.viscosity * self.length**3 / (2.0 * self.clearance**3)

    def force(self, x: float, y: float, vx: float, vy: float, omega: float) -> np.ndarray:
        """The film's force [f_x, f_y] (N) on the journal at (x, y) moving at (vx, vy), in closed form.

        The same force as `film_force`, from `pi_film`: any position inside the
        clearance, any velocity.
        """
        return np.array(pi_film(self.film_scale, self.clearance, x, y, vx, vy, omega)[:2])

    def threshold(self, top: float, mass: float | None = None) -> float | None:
        """The speed (rad/s) up to `top` where a rigid rotor on this bearing whirls unstably, or None.

        The rotor's mass per bearing is `mass` (kg), by default its load's
        weight W / g. See `rigid_threshold`.
        """
        mass = self.load / GRAVITY if mass is None else mass
        speeds = np.geomspace(top * 1e-3, top, 400)

        def margin(omega: float) -> float:
            s = self.state(omega)
            k_eq, nu2 = rigid_threshold(s.k, s.c)
            return mass * nu2 / k_eq - 1.0  # > 0: unstable

        values = [margin(w) for w in speeds]
        for i, v in enumerate(values):
            if v > 0.0:
                if i == 0:
                    return float(speeds[0])
                return float(scipy.optimize.brentq(margin, speeds[i - 1], speeds[i], xtol=1e-9 * top))
        return None


@dataclass(frozen=True)
class BearingState:
    """The bearing at one speed: its journal's equilibrium and its linearized coefficients."""

    omega: float  # rad/s
    eccentricity: float  # ε = e / c
    attitude: float  # rad, χ: from the load line (-y), turned with the spin
    k: np.ndarray  # (2, 2) N/m, [[k_xx, k_xy], [k_yx, k_yy]]
    c: np.ndarray  # (2, 2) N·s/m

    def position(self) -> tuple[float, float]:
        """The journal centre's (x, y) as fractions of the clearance."""
        return self.eccentricity * math.sin(self.attitude), -self.eccentricity * math.cos(self.attitude)


def eccentricity(modified_sommerfeld: float) -> float:
    """The short bearing's eccentricity ε for Ocvirk's number S_s (0 for an unloaded one, → 1 at rest)."""
    s = modified_sommerfeld
    if s <= 0.0:
        return 1.0
    if not math.isfinite(s):
        return 0.0

    def balance(e: float) -> float:  # rises from -1 at ε = 0 to > 0 at ε = 1
        return s * e * math.sqrt(PI2 * (1.0 - e * e) + 16.0 * e * e) - (1.0 - e * e) ** 2

    return float(scipy.optimize.brentq(balance, 0.0, 1.0, xtol=1e-14, rtol=1e-14))


def attitude(eps: float) -> float:
    """The attitude angle χ (rad) of the short bearing's journal at eccentricity ε."""
    return math.atan2(math.pi * math.sqrt(max(1.0 - eps * eps, 0.0)), 4.0 * eps)


def coefficients(eps: float) -> tuple[np.ndarray, np.ndarray]:
    """The short bearing's dimensionless coefficients a = K c / W and b = C c Ω / W at eccentricity ε.

    In x (horizontal) and y (up), the load down -y. ε must be strictly between 0 and 1.
    """
    e2 = eps * eps
    s = math.sqrt(1.0 - e2)
    h0 = 1.0 / (PI2 * (1.0 - e2) + 16.0 * e2) ** 1.5
    a = h0 * np.array([
        [4.0 * (PI2 * (2.0 - e2) + 16.0 * e2),
         math.pi * (PI2 * (1.0 - e2) ** 2 - 16.0 * e2 * e2) / (eps * s)],
        [-math.pi * (PI2 * (1.0 - e2) * (1.0 + 2.0 * e2) + 32.0 * e2 * (1.0 + e2)) / (eps * s),
         4.0 * (PI2 * (1.0 + 2.0 * e2) + 32.0 * e2 * (1.0 + e2) / (1.0 - e2))],
    ])
    off = -8.0 * (PI2 * (1.0 + 2.0 * e2) - 16.0 * e2)
    b = h0 * np.array([
        [2.0 * math.pi * s * (PI2 * (1.0 + 2.0 * e2) - 16.0 * e2) / eps, off],
        [off, 2.0 * math.pi * (PI2 * (1.0 - e2) ** 2 + 48.0 * e2) / (eps * s)],
    ])
    return a, b


def pi_film_reference(k: float, c: float, x: float, y: float, vx: float, vy: float, omega: float) -> tuple[float, ...]:
    """The π-film's whole force on the journal, its derivatives and its dissipation, in closed form.

    k is `film_scale`, c the clearance, the journal at (x, y) moving at (vx, vy),
    turning at Ω. Returns (f_x, f_y, k_xx, k_xy, k_yx, k_yy, c_xx, c_xy, c_yy, Ψ):
    the force (N), its stiffness and damping there (minus its derivatives in the
    position and the velocity) and Ψ (W), whose gradient in the velocity is
    -f. Plain floats, for the rotor's time step. Nearer the wall than
    ECCENTRICITY_MAX the film is worked out as it is there.
    """
    e = math.hypot(x, y)
    if e < 1e-9 * c:  # centred: the line of centres is any direction
        cp, sp, e = 1.0, 0.0, 1e-9 * c
    else:
        cp, sp = x / e, y / e
    eps = e / c
    if eps > ECCENTRICITY_MAX:
        eps = ECCENTRICITY_MAX
        e = eps * c
    vr = vx * cp + vy * sp  # e'
    vt = vy * cp - vx * sp  # e ψ'
    a = omega * e - 2.0 * vt
    b = -2.0 * vr
    # The film carries pressure for φ from p1 to p1 + π; u is φ's eccentric anomaly.
    p1 = math.pi - math.atan2(b, a)
    s2 = 1.0 - eps * eps
    s = math.sqrt(s2)
    beta = eps / (1.0 + s)
    sn, cs = math.sin(p1), math.cos(p1)
    u1 = p1 + 2.0 * math.atan(beta * sn / (1.0 - beta * cs))
    u2 = p1 + math.pi - 2.0 * math.atan(beta * sn / (1.0 + beta * cs))
    s1, c1, s2u, c2u = math.sin(u1), math.cos(u1), math.sin(u2), math.cos(u2)
    du, ds, dc = u2 - u1, s2u - s1, c2u - c1
    dsc = s2u * c2u - s1 * c1
    ds3 = s2u**3 - s1**3
    s4 = s2 * s2
    # ∫ (sin φ cos φ, cos²φ, sin²φ) / (1 - ε cos φ)³ dφ over the arc, and their ε-derivatives.
    isc = (0.5 * (s2u * s2u - s1 * s1) - eps * dc) / s4
    icc = ((0.5 + eps * eps) * du + 0.5 * dsc + 2.0 * eps * ds) / (s4 * s)
    iss = 0.5 * (du - dsc) / (s2 * s)
    disc = -((c2u + eps) ** 3 - (c1 + eps) ** 3) / (s4 * s2)
    dicc = 3.0 * (ds - ds3 / 3.0 + 1.5 * eps * (du + dsc) + 3.0 * eps * eps * ds + eps**3 * du) / (s4 * s2 * s)
    diss = 3.0 * (ds3 / 3.0 + 0.5 * eps * (du - dsc)) / (s4 * s)
    fr = k * (a * isc + b * icc)  # along the line of centres
    ft = k * (a * iss + b * isc)  # across it, the way ψ turns
    psi = 0.25 * k * (a * a * iss + 2.0 * a * b * isc + b * b * icc)
    # Derivatives in e and in ψ (the latter turning the frame too); the arc's own motion adds
    # nothing, the pressure being zero at both its ends.
    fr_e = k * (omega * isc + (a * disc + b * dicc) / c)
    ft_e = k * (omega * iss + (a * diss + b * disc) / c)
    am = a - omega * e
    fr_p = k * (am * icc - b * isc) - ft
    ft_p = k * (am * isc - b * iss) + fr
    fx, fy = fr * cp - ft * sp, fr * sp + ft * cp
    gx_e, gy_e = fr_e * cp - ft_e * sp, fr_e * sp + ft_e * cp
    gx_p, gy_p = (fr_p * cp - ft_p * sp) / e, (fr_p * sp + ft_p * cp) / e
    crr, crt, ctt = 2.0 * k * icc, 2.0 * k * isc, 2.0 * k * iss
    return (fx, fy,
            gx_p * sp - gx_e * cp, -gx_e * sp - gx_p * cp,
            gy_p * sp - gy_e * cp, -gy_e * sp - gy_p * cp,
            crr * cp * cp - 2.0 * crt * cp * sp + ctt * sp * sp,
            (crr - ctt) * cp * sp + crt * (cp * cp - sp * sp),
            crr * sp * sp + 2.0 * crt * cp * sp + ctt * cp * cp,
            psi)


pi_film = native.kernel("pi_film", pi_film_reference)


def full_film_damping(k: float, c: float, x: float, y: float) -> tuple[float, float, float]:
    """(c_xx, c_xy, c_yy): the whole uncavitated film's damping at (x, y), more than the π-film's at any velocity.

    2k ∫ n nᵀ / (1 - ε cos φ)³ round the whole bearing: π(1 + 2ε²)/(1 - ε²)^{5/2}
    along the line of centres, π/(1 - ε²)^{3/2} across it.
    """
    e = math.hypot(x, y)
    cp, sp = (x / e, y / e) if e > 0.0 else (1.0, 0.0)
    eps = min(e / c, ECCENTRICITY_MAX)
    s2 = 1.0 - eps * eps
    crr = 2.0 * math.pi * k * (1.0 + 2.0 * eps * eps) / s2**2.5
    ctt = 2.0 * math.pi * k / s2**1.5
    return crr * cp * cp + ctt * sp * sp, (crr - ctt) * cp * sp, crr * sp * sp + ctt * cp * cp


def rigid_threshold(k: np.ndarray, c: np.ndarray) -> tuple[float, float]:
    """A rigid rotor on bearings with coefficients K, C: its threshold stiffness K_eq and whirl ν² (Lund).

    At the onset of instability the rotor whirls at ν, and m ν² = K_eq, where

        K_eq = (k_xx c_yy + k_yy c_xx - k_xy c_yx - k_yx c_xy) / (c_xx + c_yy),
        ν² = ((K_eq - k_xx)(K_eq - k_yy) - k_xy k_yx) / (c_xx c_yy - c_xy c_yx).

    K, C and the mass m are per bearing. A rotor heavier than K_eq / ν² whirls
    unstably; ν² ≤ 0 means none does, at any mass.
    """
    k_eq = (k[0, 0] * c[1, 1] + k[1, 1] * c[0, 0] - k[0, 1] * c[1, 0] - k[1, 0] * c[0, 1]) / (c[0, 0] + c[1, 1])
    nu2 = ((k_eq - k[0, 0]) * (k_eq - k[1, 1]) - k[0, 1] * k[1, 0]) / (c[0, 0] * c[1, 1] - c[0, 1] * c[1, 0])
    return float(k_eq), float(nu2)
