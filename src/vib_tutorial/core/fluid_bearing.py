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
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.optimize

GRAVITY = 9.81  # m/s²
PI2 = math.pi**2


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
        """The bearing's equilibrium and coefficients at speed Ω (rad/s, > 0)."""
        eps = eccentricity(float(self.modified_sommerfeld(omega)))
        a, b = coefficients(eps)
        w_c = self.load / self.clearance
        return BearingState(omega, eps, attitude(eps), w_c * a, w_c / omega * b)

    def film_force(self, x: float, y: float, vx: float, vy: float, omega: float,
                   points: int = 4000) -> np.ndarray:
        """The film's force [f_x, f_y] (N) on the journal at (x, y) moving at (vx, vy), by quadrature.

        The short-bearing pressure, cavitated where it would be negative, summed
        round the bearing at `points` angles. Only for checks (and for a
        nonlinear bearing): K and C come in closed form.
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
