"""One mass on a spring and a damper, in closed form: the model the One mass pages are built on.

m x'' + c x' + k x = f(t). The free response, the response to a harmonic force from rest, the
receptance and its half-power points, the impulse and step responses, and the exact (first-order
hold) response to any sampled force, which is what convolution with the impulse response gives
when the force is linear between samples. `isolation`, `damping_models` and `transient` build on
these.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .simulator import foh_discretize

TWO_PI = 2.0 * math.pi
CRITICAL_BAND = 1e-6  # |zeta - 1| below this is critical damping (the two roots' formula loses precision)


@dataclass(frozen=True)
class Oscillator:
    """m x'' + c x' + k x = f: SI units (kg, N/m, N·s/m)."""

    mass: float = 1.0
    stiffness: float = 400.0
    damping: float = 2.0

    def __post_init__(self) -> None:
        if not (self.mass > 0 and self.stiffness > 0 and self.damping >= 0):
            raise ValueError("the mass and the stiffness must be positive and the damping not negative")

    @classmethod
    def from_modal(cls, mass: float, natural_hz: float, zeta: float) -> Oscillator:
        wn = TWO_PI * natural_hz
        return cls(mass, mass * wn * wn, 2.0 * zeta * mass * wn)

    @property
    def omega_n(self) -> float:
        return math.sqrt(self.stiffness / self.mass)

    @property
    def natural_hz(self) -> float:
        return self.omega_n / TWO_PI

    @property
    def critical(self) -> float:
        """The critical damping 2√(km): the least that stops it oscillating."""
        return 2.0 * math.sqrt(self.stiffness * self.mass)

    @property
    def zeta(self) -> float:
        return self.damping / self.critical

    @property
    def omega_d(self) -> float:
        """The damped natural frequency ω_n √(1 − ζ²); 0 at and above critical damping."""
        z = self.zeta
        return self.omega_n * math.sqrt(1.0 - z * z) if z < 1.0 else 0.0

    @property
    def damped_hz(self) -> float:
        return self.omega_d / TWO_PI

    @property
    def period(self) -> float:
        """The undamped natural period 2π/ω_n."""
        return TWO_PI / self.omega_n

    @property
    def time_constant(self) -> float:
        """How long the slowest part of a free decay takes to fall by e: 1/ζω_n below critical damping."""
        if self.damping == 0:
            return math.inf
        return -1.0 / float(np.max(self.poles().real))

    @property
    def log_decrement(self) -> float:
        """ln(x_k / x_{k+1}) between successive peaks: 2πζ/√(1 − ζ²); inf at and above critical damping."""
        z = self.zeta
        return TWO_PI * z / math.sqrt(1.0 - z * z) if z < 1.0 else math.inf

    @property
    def static(self) -> float:
        """The static compliance 1/k (m/N)."""
        return 1.0 / self.stiffness

    def poles(self) -> np.ndarray:
        """The roots of m s² + c s + k: a conjugate pair below critical damping, two real roots above."""
        wn, z = self.omega_n, self.zeta
        if z < 1.0:
            wd = wn * math.sqrt(1.0 - z * z)
            return np.array([complex(-z * wn, wd), complex(-z * wn, -wd)])
        root = wn * math.sqrt(z * z - 1.0)
        return np.array([complex(-z * wn + root, 0.0), complex(-z * wn - root, 0.0)])

    def peak_hz(self) -> float | None:
        """Where |H| peaks, f_n √(1 − 2ζ²); None for ζ ≥ 1/√2 (it falls from zero frequency)."""
        z = self.zeta
        return self.natural_hz * math.sqrt(1.0 - 2.0 * z * z) if 2.0 * z * z < 1.0 else None

    def half_power_hz(self) -> tuple[float, float] | None:
        """The two frequencies where |H| is its peak over √2 (exact); None without a peak or with no damping.

        |H|⁻² ∝ (1 − r²)² + (2ζr)² = 2 × 4ζ²(1 − ζ²) gives r² = 1 − 2ζ² ± 2ζ√(1 − ζ²).
        """
        z = self.zeta
        if z <= 0 or 2.0 * z * z >= 1.0:
            return None
        lo2 = 1.0 - 2.0 * z * z - 2.0 * z * math.sqrt(1.0 - z * z)
        hi2 = 1.0 - 2.0 * z * z + 2.0 * z * math.sqrt(1.0 - z * z)
        if lo2 <= 0:
            return None
        return self.natural_hz * math.sqrt(lo2), self.natural_hz * math.sqrt(hi2)

    def matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """(A, B) of z' = A z + B f with z = [x, x']."""
        m, c, k = self.mass, self.damping, self.stiffness
        return np.array([[0.0, 1.0], [-k / m, -c / m]]), np.array([[0.0], [1.0 / m]])


def receptance(osc: Oscillator, freqs_hz: np.ndarray) -> np.ndarray:
    """H = X/F = 1/(k − ω²m + iωc). Infinite (complex inf) on an undamped oscillator's natural frequency."""
    w = TWO_PI * np.asarray(freqs_hz, dtype=float)
    z = osc.stiffness - w * w * osc.mass + 1j * w * osc.damping
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(z == 0, complex(np.inf, 0.0), 1.0 / np.where(z == 0, 1.0, z))


def frequency_grid(osc: Oscillator, lo_ratio: float = 0.05, hi_ratio: float = 5.0, points: int = 1200) -> np.ndarray:
    """Log-spaced frequencies (Hz) round f_n, with points packed across the peak; never exactly on an undamped f_n."""
    fn = osc.natural_hz
    f = np.geomspace(lo_ratio * fn, hi_ratio * fn, points)
    z = max(osc.zeta, 1e-4)
    near = fn * (1.0 + z * np.linspace(-6.0, 6.0, 241))
    f = np.unique(np.concatenate([f, near[(near > f[0]) & (near < f[-1])]]))
    if osc.damping == 0:
        f = f[np.abs(f / fn - 1.0) > 1e-9]
    return f


def free_response(osc: Oscillator, x0: float, v0: float, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(x, v) of the free motion from x(0) = x0, x'(0) = v0, in closed form for every damping."""
    t = np.asarray(t, dtype=float)
    wn, z = osc.omega_n, osc.zeta
    if abs(z - 1.0) < CRITICAL_BAND:  # x = (x0 + (v0 + wn x0) t) e^{-wn t}
        e = np.exp(-wn * t)
        b = v0 + wn * x0
        x = (x0 + b * t) * e
        v = (b - wn * (x0 + b * t)) * e
        return x, v
    if z < 1.0:
        wd = wn * math.sqrt(1.0 - z * z)
        s = z * wn
        e = np.exp(-s * t)
        b = (v0 + s * x0) / wd
        cos, sin = np.cos(wd * t), np.sin(wd * t)
        x = e * (x0 * cos + b * sin)
        v = e * ((wd * b - s * x0) * cos - (wd * x0 + s * b) * sin)
        return x, v
    s1, s2 = osc.poles().real  # s1 the slower (nearer zero)
    a1 = (v0 - s2 * x0) / (s1 - s2)
    a2 = (s1 * x0 - v0) / (s1 - s2)
    e1, e2 = np.exp(s1 * t), np.exp(s2 * t)
    return a1 * e1 + a2 * e2, a1 * s1 * e1 + a2 * s2 * e2


def impulse_response(osc: Oscillator, t: np.ndarray) -> np.ndarray:
    """h(t): the displacement after a unit impulse (1 N·s) at t = 0, the free motion from x' = 1/m."""
    t = np.asarray(t, dtype=float)
    x, _ = free_response(osc, 0.0, 1.0 / osc.mass, np.maximum(t, 0.0))
    return np.where(t >= 0, x, 0.0)


def step_response(osc: Oscillator, t: np.ndarray) -> np.ndarray:
    """The displacement under a unit force switched on at t = 0: (1/k)(1 − the free decay from x = 1)."""
    t = np.asarray(t, dtype=float)
    x, _ = free_response(osc, 1.0, 0.0, np.maximum(t, 0.0))
    return np.where(t >= 0, (1.0 - x) / osc.stiffness, 0.0)


def step_overshoot(osc: Oscillator) -> float:
    """The step response's first peak over its final value, minus 1: e^{−ζπ/√(1 − ζ²)}; 0 at and above critical."""
    z = osc.zeta
    return math.exp(-z * math.pi / math.sqrt(1.0 - z * z)) if z < 1.0 else 0.0


def harmonic_from_rest(
    osc: Oscillator, force: float, freq_hz: float, t: np.ndarray, x0: float = 0.0, v0: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """(total, steady) displacement under F cos ωt switched on at t = 0 from (x0, v0).

    The total is the steady response plus the free motion that makes up the start; at resonance with no
    damping there is no steady state, and the swing grows as (F/2mω_n) t sin ω_n t.
    """
    t = np.asarray(t, dtype=float)
    w = TWO_PI * freq_hz
    h = receptance(osc, np.array([freq_hz]))[0]
    if not np.isfinite(h):
        grow = force / (2.0 * osc.mass * w) * t * np.sin(w * t)
        x, _ = free_response(osc, x0, v0, t)
        return grow + x, np.full_like(t, np.nan)
    X = force * h  # x_ss = Re(X e^{iωt})
    steady = (X * np.exp(1j * w * t)).real
    xs0, vs0 = X.real, (1j * w * X).real
    x, _ = free_response(osc, x0 - xs0, v0 - vs0, t)
    return steady + x, steady


def peaks(t: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The positive local maxima of a sampled decay, each refined by a parabola: (times, heights)."""
    x = np.asarray(x)
    i = np.nonzero((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]) & (x[1:-1] > 0))[0] + 1
    if not i.size:
        return np.empty(0), np.empty(0)
    y0, y1, y2 = x[i - 1], x[i], x[i + 1]
    den = y0 - 2 * y1 + y2
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(den != 0, 0.5 * (y0 - y2) / den, 0.0)
    h = t[1] - t[0]
    return t[i] + d * h, y1 - 0.25 * (y0 - y2) * d


def measured_decrement(t: np.ndarray, x: np.ndarray) -> float | None:
    """The mean log decrement between successive positive peaks of a free decay, or None with fewer than two."""
    _, p = peaks(t, x)
    p = p[p > 1e-12 * (abs(p[0]) if p.size else 1.0)]
    if p.size < 2:
        return None
    return float(math.log(p[0] / p[-1]) / (p.size - 1))


def respond(A: np.ndarray, B: np.ndarray, h: float, u: np.ndarray, z0: np.ndarray | None = None) -> np.ndarray:
    """The states (len(u), n) of z' = A z + B u under u sampled every h and linear between samples (FOH).

    Exact for such an input: the same update the chain's simulator uses. z[0] is z0 (rest by default).
    """
    Phi, G0, G1 = foh_discretize(A, B, h)
    u = np.asarray(u, dtype=float)
    n = A.shape[0]
    z = np.empty((u.size, n))
    z[0] = 0.0 if z0 is None else z0
    g0, g1 = G0[:, 0], G1[:, 0]
    for k in range(1, u.size):
        z[k] = Phi @ z[k - 1] + g0 * u[k - 1] + g1 * u[k]
    return z


def response(osc: Oscillator, h: float, u: np.ndarray, x0: float = 0.0, v0: float = 0.0) -> np.ndarray:
    """The displacement under the force samples u (h apart, linear between them), exactly."""
    A, B = osc.matrices()
    return respond(A, B, h, u, np.array([x0, v0]))[:, 0]
