"""Pendulums and other rotational oscillators: gravity as the spring, and what large swings do to it.

The *Pendulums* page. A `Pendulum` turns about a pivot with moment of inertia J, a torsion spring k_t and a
weight m g acting at an arm d from the pivot, below it (hanging) or above it (inverted):

    J θ'' + c θ' + k_t θ ± m g d sin θ = 0.

Linearized (sin θ ≈ θ) it is the single oscillator with J for m and k_t ± m g d for k. Kept whole, the hanging
pendulum's restoring moment grows more slowly than θ, so it softens (its period lengthens with the swing, as a
Duffing oscillator's with a negative cubic, γ = ∓ m g d / 6), and the inverted one's hardens. `period` is the exact
undamped period by quadrature of the energy, `duffing_period` the cubic's harmonic balance, and `swing` the free
motion from rest at θ₀, the whole equation stepped by `solve_ivp` beside the linear one.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass

import numpy as np
import scipy.integrate
import scipy.optimize
import scipy.special

TWO_PI = 2.0 * math.pi
G = 9.81


class Kind(enum.Enum):
    SIMPLE = "Simple pendulum"
    ROD = "Rod on a pivot"
    TORSION = "Disc on a wire"
    INVERTED = "Inverted, on a spring"


KINDS = tuple(Kind)


@dataclass(frozen=True)
class Pendulum:
    """`length` is the string (simple), the rod (rod), the disc's radius (torsion) or the arm to the mass
    (inverted); `pivot` is the rod's pivot's distance from its centre; `spring` k_t (N·m/rad) acts on the
    disc and the inverted pendulum; `zeta` is the linearized oscillator's damping ratio."""

    kind: Kind = Kind.SIMPLE
    mass: float = 1.0
    length: float = 1.0
    pivot: float = 0.5
    spring: float = 15.0
    zeta: float = 0.0
    gravity: float = G

    @property
    def inertia(self) -> float:
        """J about the pivot."""
        m, L = self.mass, self.length
        if self.kind is Kind.ROD:
            return m * L * L / 12 + m * self.pivot ** 2
        if self.kind is Kind.TORSION:
            return m * L * L / 2
        return m * L * L

    @property
    def arm(self) -> float:
        """d: from the pivot to the centre of mass."""
        if self.kind is Kind.ROD:
            return self.pivot
        if self.kind is Kind.TORSION:
            return 0.0
        return self.length

    @property
    def weight_moment(self) -> float:
        """± m g d: the gravity term's coefficient (sin θ), negative for the inverted pendulum."""
        w = self.mass * self.gravity * self.arm
        return -w if self.kind is Kind.INVERTED else w

    @property
    def torsion(self) -> float:
        return self.spring if self.kind in (Kind.TORSION, Kind.INVERTED) else 0.0

    @property
    def stiffness(self) -> float:
        """The linearized k_t ± m g d."""
        return self.torsion + self.weight_moment

    @property
    def cubic(self) -> float:
        """γ in k θ + γ θ³: sin θ ≈ θ − θ³/6."""
        return -self.weight_moment / 6

    @property
    def stable(self) -> bool:
        return self.stiffness > 0

    @property
    def omega0(self) -> float:
        """The linearized natural frequency (rad/s); NaN when it isn't stable."""
        return math.sqrt(self.stiffness / self.inertia) if self.stable else math.nan

    @property
    def natural_hz(self) -> float:
        return self.omega0 / TWO_PI

    @property
    def period0(self) -> float:
        return TWO_PI / self.omega0

    @property
    def damping(self) -> float:
        """c = 2ζ√(|k| J)."""
        return 2 * self.zeta * math.sqrt(abs(self.stiffness) * self.inertia)

    @property
    def equivalent_length(self) -> float:
        """The simple pendulum with the same linear period: J / m d."""
        return self.inertia / (self.mass * self.arm) if self.arm > 0 else math.inf

    def moment(self, theta: np.ndarray) -> np.ndarray:
        """The restoring moment k_t θ ± m g d sin θ."""
        return self.torsion * theta + self.weight_moment * np.sin(theta)

    def potential(self, theta: np.ndarray) -> np.ndarray:
        """V(θ) = k_t θ²/2 ± m g d (1 − cos θ)."""
        theta = np.asarray(theta, dtype=float)
        return 0.5 * self.torsion * theta ** 2 + self.weight_moment * (1 - np.cos(theta))

    def leaning(self) -> float:
        """The inverted pendulum's tilted rest angle when the spring is too weak (k_t θ = m g d sin θ); 0
        otherwise."""
        if self.stable or self.kind is not Kind.INVERTED or self.torsion <= 0:
            return 0.0
        return float(scipy.optimize.brentq(lambda t: self.moment(t), 1e-9, math.pi))


def period(p: Pendulum, amplitude: float) -> float:
    """The exact undamped period of a swing to ±`amplitude` (rad), by quadrature of the energy:
    T = 4 ∫₀^A dθ / √(2(V(A) − V(θ))/J). With θ = A sin φ and the drop in energy factored,
    V(A) − V(θ) = (A − θ) Q, Q = k_t (A + θ)/2 ± m g d sin((A + θ)/2) sinc((A − θ)/2), the integrand is
    √(A J (1 + sin φ) / 2Q): smooth, with no cancellation near the turning point. NaN if the motion doesn't
    turn at A."""
    A = abs(amplitude)
    if A == 0:
        return p.period0
    if not p.moment(A) > 0:
        return math.nan
    J, kt, w = p.inertia, p.torsion, p.weight_moment

    def integrand(phi: float) -> float:
        theta = A * math.sin(phi)
        u = 0.5 * (A - theta)
        Q = 0.5 * kt * (A + theta) + w * math.sin(0.5 * (A + theta)) * (math.sin(u) / u if u > 1e-12 else 1.0)
        return math.sqrt(A * J * (1 + math.sin(phi)) / (2 * Q))

    value, _ = scipy.integrate.quad(integrand, 0.0, math.pi / 2, limit=200, epsrel=1e-11)
    return 4 * value


def elliptic_period(p: Pendulum, amplitude: float) -> float:
    """The simple (spring-free, hanging) pendulum's period in closed form: T₀ · (2/π) K(sin²(A/2))."""
    return p.period0 * 2 / math.pi * float(scipy.special.ellipk(math.sin(amplitude / 2) ** 2))


def duffing_period(p: Pendulum, amplitude: float) -> float:
    """The cubic's first harmonic balance: ω² = ω₀² (1 + 3γA²/4k), so the swing softens if γ < 0."""
    r = 1 + 0.75 * p.cubic * amplitude ** 2 / p.stiffness
    return p.period0 / math.sqrt(r) if r > 0 else math.nan


def periods(p: Pendulum, amplitudes: np.ndarray) -> np.ndarray:
    return np.array([period(p, float(a)) for a in amplitudes])


def swing(p: Pendulum, theta0: float, duration: float, points: int = 2000,
          omega0: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(t, θ, θ_linear) from θ₀ (rad) and θ'₀ = `omega0`: the whole equation and the linearized one."""
    J, c = p.inertia, p.damping

    def whole(_t, y):
        return [y[1], -(c * y[1] + float(p.moment(y[0]))) / J]

    def linear(_t, y):
        return [y[1], -(c * y[1] + p.stiffness * y[0]) / J]

    t = np.linspace(0.0, duration, points)
    out = []
    for f in (whole, linear):
        sol = scipy.integrate.solve_ivp(f, (0.0, duration), [theta0, omega0], method="DOP853", t_eval=t,
                                        rtol=1e-10, atol=1e-12)
        y = np.full(t.size, np.nan)
        y[:sol.y.shape[1]] = sol.y[0]
        out.append(y)
    return t, out[0], out[1]


def measured_period(t: np.ndarray, theta: np.ndarray) -> float:
    """The mean spacing of θ's downward zero crossings (interpolated); NaN with fewer than two."""
    s = np.sign(theta - float(np.mean(theta)))
    i = np.nonzero((s[:-1] > 0) & (s[1:] <= 0))[0]
    if i.size < 2:
        return math.nan
    th = theta - float(np.mean(theta))
    tc = t[i] + (t[i + 1] - t[i]) * th[i] / (th[i] - th[i + 1])
    return float(np.mean(np.diff(tc)))


def phase_curve(p: Pendulum, energy: float, theta: np.ndarray) -> np.ndarray:
    """θ' ≥ 0 on the undamped orbit of the given energy (NaN where V(θ) > E)."""
    drop = energy - p.potential(theta)
    with np.errstate(invalid="ignore"):
        return np.where(drop >= 0, np.sqrt(2 * np.maximum(drop, 0) / p.inertia), np.nan)


def orbit(p: Pendulum, energy: float, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The closed curves of constant energy over the grid `theta`, as (θ, θ') with NaN between pieces: the
    upper half, then the lower half back, with the turning points (V = E) found exactly so each piece closes."""
    theta = np.asarray(theta, dtype=float)
    drop = energy - p.potential(theta)
    cross = np.nonzero(np.sign(drop[:-1]) * np.sign(drop[1:]) < 0)[0]
    turns = [scipy.optimize.brentq(lambda t: energy - float(p.potential(t)), theta[i], theta[i + 1])
             for i in cross]
    x = np.sort(np.concatenate([theta, turns]))
    y = phase_curve(p, energy, x)
    y[np.isin(x, turns)] = 0.0
    # pieces where the orbit exists, each drawn up and back down
    ok = np.isfinite(y)
    xs, ys = [], []
    edges = np.flatnonzero(np.diff(np.concatenate([[0], ok.astype(int), [0]])))
    for start, stop in zip(edges[::2], edges[1::2]):
        px, py = x[start:stop], y[start:stop]
        xs += [px, px[::-1], [np.nan]]
        ys += [py, -py[::-1], [np.nan]]
    if not xs:
        return np.array([]), np.array([])
    return np.concatenate(xs), np.concatenate(ys)


def rod_period(length: float, pivot: np.ndarray, gravity: float = G) -> np.ndarray:
    """A uniform rod's linear period against the pivot's distance d from its centre: 2π √((L²/12 + d²)/(g d))."""
    d = np.asarray(pivot, dtype=float)
    with np.errstate(divide="ignore"):
        return TWO_PI * np.sqrt((length ** 2 / 12 + d * d) / (gravity * d))


def conjugate_pivot(length: float, pivot: float) -> float:
    """The other distance on the same side with the same period (Kater): k_G² / d."""
    return length ** 2 / 12 / pivot
