"""Numerical integration of Mẍ + Cẋ + Kx = f: what the usual time-stepping schemes do to a vibration.

The *Numerical integration* page. A `Model` is one mass on a spring and a damper, optionally with a second,
light mass on a stiff spring (a mode far above the first, as a fine mesh has). Each `Scheme` steps it with a
fixed Δt: central difference, Newmark's average and linear acceleration, HHT-α, classical Runge–Kutta, explicit
Euler, and the exact first-order-hold update the rest of the app uses. `exact` is the closed form (free
vibration plus the harmonic force's steady state). For one undamped or damped oscillator, `amplification` is
each scheme's one-step map of the free motion, and `spectral` reads from it the spectral radius (stable when
≤ 1), the period's error and the damping the scheme adds, against Ω = ωΔt.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np
import scipy.linalg

from .simulator import foh_discretize

TWO_PI = 2.0 * math.pi


class Scheme(Enum):
    FOH = "Exact update (FOH)"
    CENTRAL = "Central difference"
    AVERAGE = "Newmark, average acceleration"
    LINEAR = "Newmark, linear acceleration"
    HHT = "HHT-α"
    RK4 = "Runge–Kutta 4"
    EULER = "Explicit Euler"


SCHEMES = tuple(Scheme)

# The largest stable Ω = ωΔt of an undamped oscillator (inf: unconditionally stable; 0: never).
STABLE_LIMIT = {
    Scheme.FOH: math.inf,
    Scheme.CENTRAL: 2.0,
    Scheme.AVERAGE: math.inf,
    Scheme.LINEAR: 2.0 * math.sqrt(3.0),
    Scheme.HHT: math.inf,
    Scheme.RK4: 2.0 * math.sqrt(2.0),
    Scheme.EULER: 0.0,
}


def newmark_parameters(scheme: Scheme, alpha: float = 0.0) -> tuple[float, float, float]:
    """(α, γ, β): HHT-α's (α ≤ 0, γ = ½ − α, β = (1 − α)²/4), or plain Newmark's (α = 0)."""
    if scheme is Scheme.HHT:
        return alpha, 0.5 - alpha, 0.25 * (1.0 - alpha) ** 2
    if scheme is Scheme.LINEAR:
        return 0.0, 0.5, 1.0 / 6.0
    return 0.0, 0.5, 0.25


@dataclass(frozen=True)
class Model:
    """A mass on a spring and a damper; with `stiff_hz` > 0 a second mass, a tenth of the first, hangs from it on a
    spring stiff enough that alone it would vibrate at `stiff_hz`."""

    mass: float = 1.0
    stiffness: float = 400.0
    damping: float = 0.0
    stiff_hz: float = 0.0

    @property
    def n(self) -> int:
        return 2 if self.stiff_hz > 0 else 1

    def matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        m, k, c = self.mass, self.stiffness, self.damping
        if self.n == 1:
            return np.array([[m]]), np.array([[c]]), np.array([[k]])
        m2 = 0.1 * m
        k2 = m2 * (TWO_PI * self.stiff_hz) ** 2
        return (np.diag([m, m2]), np.array([[c, 0.0], [0.0, 0.0]]),
                np.array([[k + k2, -k2], [-k2, k2]]))

    def state_space(self) -> tuple[np.ndarray, np.ndarray]:
        M, C, K = self.matrices()
        n = self.n
        Mi = np.linalg.inv(M)
        A = np.zeros((2 * n, 2 * n))
        A[:n, n:] = np.eye(n)
        A[n:, :n] = -Mi @ K
        A[n:, n:] = -Mi @ C
        B = np.zeros((2 * n, n))
        B[n:] = Mi
        return A, B

    def frequencies_hz(self) -> np.ndarray:
        M, _, K = self.matrices()
        return np.sqrt(np.clip(scipy.linalg.eigh(K, M, eigvals_only=True), 0.0, None)) / TWO_PI


@dataclass(frozen=True)
class Load:
    """The first mass let go from `x0` at rest, and a force F sin 2πf t on it."""

    x0: float = 0.01
    force: float = 0.0
    freq_hz: float = 1.0

    def at(self, model: Model) -> Callable[[float], np.ndarray]:
        n = model.n
        e = np.zeros(n)
        e[0] = 1.0
        w = TWO_PI * self.freq_hz
        F = self.force
        return lambda t: e * (F * math.sin(w * t))

    def start(self, model: Model) -> np.ndarray:
        x = np.zeros(model.n)
        x[0] = self.x0
        return x


def exact(model: Model, load: Load, t: np.ndarray) -> np.ndarray:
    """The first mass's displacement in closed form: the steady state under the sine plus the free motion that
    makes up the start, z = e^{At}(z₀ − z_p(0)) + z_p(t)."""
    A, B = model.state_space()
    n = model.n
    w = TWO_PI * load.freq_hz
    b = np.zeros(n)
    b[0] = load.force
    # F sin wt = Im(F e^{iwt}): z_p = Im(Z e^{iwt}), (iw − A) Z = B F
    Z = np.linalg.solve(1j * w * np.eye(2 * n) - A, B @ b) if load.force else np.zeros(2 * n, dtype=complex)
    z0 = np.concatenate([load.start(model), np.zeros(n)]) - Z.imag
    t = np.asarray(t, dtype=float)
    # e^{At} by the eigen-decomposition (A is small); fall back to expm each step if it's defective
    lam, V = np.linalg.eig(A)
    if np.linalg.cond(V) < 1e10:
        c = np.linalg.solve(V, z0.astype(complex))
        free = ((V[0] * c)[None, :] * np.exp(np.outer(t, lam))).sum(axis=1).real
    else:
        free = np.array([(scipy.linalg.expm(A * s) @ z0)[0] for s in t])
    return free + (Z[0] * np.exp(1j * w * t)).imag


def simulate(scheme: Scheme, model: Model, load: Load, h: float, steps: int, alpha: float = -0.1,
             limit: float = 1e6) -> np.ndarray:
    """The first mass's displacement at t = 0, h, …, steps·h. Once it passes `limit` times the start (the scheme
    has blown up) the rest is NaN."""
    M, C, K = model.matrices()
    n = model.n
    f = load.at(model)
    x = load.start(model)
    v = np.zeros(n)
    out = np.full(steps + 1, np.nan)
    out[0] = x[0]
    bound = limit * (abs(load.x0) + abs(load.force) / model.stiffness + 1e-12)
    Mi = np.linalg.inv(M)
    a = Mi @ (f(0.0) - C @ v - K @ x)

    if scheme is Scheme.CENTRAL:
        left = M / h ** 2 + C / (2 * h)
        right_prev = M / h ** 2 - C / (2 * h)
        mid = K - 2 * M / h ** 2
        solve = scipy.linalg.lu_factor(left)
        x_prev = x - h * v + 0.5 * h * h * a
        for i in range(1, steps + 1):
            x_next = scipy.linalg.lu_solve(solve, f((i - 1) * h) - mid @ x - right_prev @ x_prev)
            x_prev, x = x, x_next
            out[i] = x[0]
            if not abs(x[0]) < bound:
                out[i] = np.nan
                break
        return out

    if scheme in (Scheme.AVERAGE, Scheme.LINEAR, Scheme.HHT):
        al, g, b = newmark_parameters(scheme, alpha)
        solve = scipy.linalg.lu_factor(M + (1 + al) * (g * h * C + b * h * h * K))
        for i in range(1, steps + 1):
            t0, t1 = (i - 1) * h, i * h
            xt = x + h * v + h * h * (0.5 - b) * a
            vt = v + h * (1 - g) * a
            rhs = f((1 + al) * t1 - al * t0) - (1 + al) * (C @ vt + K @ xt) + al * (C @ v + K @ x)
            a = scipy.linalg.lu_solve(solve, rhs)
            x, v = xt + b * h * h * a, vt + g * h * a
            out[i] = x[0]
            if not abs(x[0]) < bound:
                out[i] = np.nan
                break
        return out

    A, B = model.state_space()
    z = np.concatenate([x, v])
    if scheme is Scheme.FOH:
        Phi, G0, G1 = foh_discretize(A, B, h)
        for i in range(1, steps + 1):
            z = Phi @ z + G0 @ f((i - 1) * h) + G1 @ f(i * h)
            out[i] = z[0]
        return out

    def rate(t: float, z: np.ndarray) -> np.ndarray:
        return A @ z + B @ f(t)

    for i in range(1, steps + 1):
        t = (i - 1) * h
        if scheme is Scheme.EULER:
            z = z + h * rate(t, z)
        else:
            k1 = rate(t, z)
            k2 = rate(t + h / 2, z + h / 2 * k1)
            k3 = rate(t + h / 2, z + h / 2 * k2)
            k4 = rate(t + h, z + h * k3)
            z = z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out[i] = z[0]
        if not abs(z[0]) < bound:
            out[i] = np.nan
            break
    return out


def amplification(scheme: Scheme, Omega: float, zeta: float = 0.0, alpha: float = -0.1) -> np.ndarray:
    """The free motion's one-step map for an oscillator with ω = 1 and Δt = Ω (so Ω = ωΔt)."""
    h, w = Omega, 1.0
    A = np.array([[0.0, 1.0], [-w * w, -2 * zeta * w]])
    if scheme is Scheme.FOH:
        return scipy.linalg.expm(A * h)
    if scheme is Scheme.EULER:
        return np.eye(2) + h * A
    if scheme is Scheme.RK4:
        hA = h * A
        return np.eye(2) + hA + hA @ hA / 2 + hA @ hA @ hA / 6 + hA @ hA @ hA @ hA / 24
    if scheme is Scheme.CENTRAL:  # on [x_n, x_{n−1}]
        d = 1 + zeta * h
        return np.array([[(2 - h * h) / d, -(1 - zeta * h) / d], [1.0, 0.0]])
    al, g, b = newmark_parameters(scheme, alpha)  # on [x, v, a]
    c, k = 2 * zeta, 1.0
    lhs = 1 + (1 + al) * (g * h * c + b * h * h * k)
    cols = []
    for x, v, a in np.eye(3):
        xt = x + h * v + h * h * (0.5 - b) * a
        vt = v + h * (1 - g) * a
        a1 = (-(1 + al) * (c * vt + k * xt) + al * (c * v + k * x)) / lhs
        cols.append([xt + b * h * h * a1, vt + g * h * a1, a1])
    return np.array(cols).T


@dataclass(frozen=True)
class Spectral:
    """What a scheme does to a free vibration per step: the spectral radius ρ (stable if ≤ 1), the period's
    relative error (T̄ − T)/T, and the damping ratio of the motion it computes (NaN once it no longer
    oscillates)."""

    radius: float
    period_error: float
    damping: float


def spectral(scheme: Scheme, Omega: float, zeta: float = 0.0, alpha: float = -0.1) -> Spectral:
    lam = np.linalg.eigvals(amplification(scheme, Omega, zeta, alpha))
    radius = float(np.max(np.abs(lam)))
    oscillating = lam[lam.imag > 1e-12]
    if oscillating.size == 0:
        return Spectral(radius, math.nan, math.nan)
    principal = oscillating[np.argmax(np.abs(oscillating))]
    log = np.log(principal)  # σ + iθ per step
    theta = float(log.imag)
    exact_period = TWO_PI / math.sqrt(max(1 - zeta * zeta, 1e-300))  # damped period for ω = 1
    period = TWO_PI * Omega / theta
    return Spectral(radius, period / exact_period - 1, float(-log.real / abs(log)))


def spectral_curves(scheme: Scheme, Omegas: np.ndarray, zeta: float = 0.0,
                    alpha: float = -0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(ρ, period error, damping) at each Ω."""
    rows = [spectral(scheme, float(o), zeta, alpha) for o in Omegas]
    return (np.array([r.radius for r in rows]), np.array([r.period_error for r in rows]),
            np.array([r.damping for r in rows]))


def stable_limit(scheme: Scheme, zeta: float = 0.0, alpha: float = -0.1) -> float:
    """The largest stable Ω (inf if none is found up to 100), by bisection on ρ ≤ 1."""
    if scheme is Scheme.EULER and zeta == 0:
        return 0.0

    def stable(o: float) -> bool:
        return spectral(scheme, o, zeta, alpha).radius <= 1 + 1e-12

    if stable(100.0) and scheme is not Scheme.EULER:
        return math.inf
    lo, hi = 1e-6, 100.0
    if not stable(lo):
        return 0.0
    # walk up to the first unstable Ω, then bisect
    o = lo
    while o < hi and stable(o):
        o *= 1.05
    lo, hi = o / 1.05, o
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if stable(mid):
            lo = mid
        else:
            hi = mid
    return lo
