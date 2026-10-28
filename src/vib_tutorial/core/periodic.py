"""Periodic forcing: a force that repeats every T = 1/f₀ as a Fourier series, each harmonic through H(kω₀).

The *Periodic forcing* page. A `Wave` (square, sawtooth, triangle, a rectified sine, a train of rectangular
pulses or of impulses) is written exactly as f(t) = Re Σ_k C_k e^{ikω₀t}, its coefficients in closed form, so
the series can be cut at any number of harmonics. The oscillator's steady response is then the same sum with
each term multiplied by H(kω₀). `periodic_steady` gives the steady state exactly, without the series: one
period stepped by the first-order-hold update from the state that repeats (z₀ = (I − Φ_T)⁻¹ z_T), so the
series' convergence can be measured against it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .sdof import TWO_PI, Oscillator, free_response, receptance, respond

STEPS = 4000  # first-order-hold steps per period for the exact steady state


class Wave(Enum):
    SQUARE = "Square wave"
    SAWTOOTH = "Sawtooth"
    TRIANGLE = "Triangle"
    RECTIFIED = "Rectified sine"
    PULSES = "Rectangular pulses"
    IMPULSES = "Impulse train (hammer blows)"


WAVES = tuple(Wave)


@dataclass(frozen=True)
class Forcing:
    """A periodic force of `amplitude` F (N; for impulses, each blow's impulse in N·s) at f₀ = `freq_hz`.

    `duty` is the rectangular pulses' fraction of the period that they're on.
    """

    wave: Wave = Wave.SQUARE
    amplitude: float = 1.0
    freq_hz: float = 1.0
    duty: float = 0.25

    @property
    def period(self) -> float:
        return 1.0 / self.freq_hz

    def coefficients(self, harmonics: int) -> np.ndarray:
        """C_0 … C_N, with f(t) = Re Σ C_k e^{ikω₀t} (C_0 the mean)."""
        F = self.amplitude
        k = np.arange(harmonics + 1, dtype=float)
        C = np.zeros(harmonics + 1, dtype=complex)
        n = k[1:]
        odd = (n % 2) == 1
        if self.wave is Wave.SQUARE:  # ±F, rising through 0 at t = 0: Σ (4F/πk) sin kω₀t over odd k
            C[1:] = np.where(odd, -1j * 4 * F / (math.pi * n), 0.0)
        elif self.wave is Wave.SAWTOOTH:  # rising from −F to F over the period, centred on t = 0
            C[1:] = -1j * 2 * F * (-1.0) ** (n + 1) / (math.pi * n)
        elif self.wave is Wave.TRIANGLE:  # ±F, rising through 0 at t = 0
            sign = np.where((np.floor((n - 1) / 2) % 2) == 0, 1.0, -1.0)
            C[1:] = np.where(odd, -1j * 8 * F * sign / (math.pi * n) ** 2, 0.0)
        elif self.wave is Wave.RECTIFIED:  # F |sin(π t/T)|: mean 2F/π, then −4F/π(4k² − 1) cos kω₀t
            C[0] = 2 * F / math.pi
            C[1:] = -4 * F / (math.pi * (4 * n * n - 1))
        elif self.wave is Wave.PULSES:  # F for the first duty × T of each period, 0 after
            d = self.duty
            C[0] = F * d
            C[1:] = F * (1 - np.exp(-2j * math.pi * n * d)) / (1j * math.pi * n)
        else:  # an impulse F every period: F/T, then 2F/T cos kω₀t
            C[0] = F / self.period
            C[1:] = 2 * F / self.period
        return C

    def force(self, t: np.ndarray) -> np.ndarray:
        """The waveform itself (impulses drawn as nothing between the blows)."""
        F = self.amplitude
        u = np.mod(np.asarray(t, dtype=float) / self.period, 1.0)
        if self.wave is Wave.SQUARE:
            return np.where(u < 0.5, F, -F)
        if self.wave is Wave.SAWTOOTH:
            return F * 2.0 * np.where(u < 0.5, u, u - 1.0)
        if self.wave is Wave.TRIANGLE:
            return F * np.where(u < 0.25, 4 * u, np.where(u < 0.75, 2 - 4 * u, 4 * u - 4))
        if self.wave is Wave.RECTIFIED:
            return F * np.abs(np.sin(math.pi * u))
        if self.wave is Wave.PULSES:
            return np.where(u < self.duty, F, 0.0)
        return np.zeros_like(u)


def series(C: np.ndarray, freq_hz: float, t: np.ndarray) -> np.ndarray:
    """Re Σ C_k e^{ikω₀t}."""
    k = np.arange(C.size)
    return (C[None, :] * np.exp(1j * TWO_PI * freq_hz * np.outer(np.asarray(t, dtype=float), k))).real.sum(axis=1)


def response_coefficients(osc: Oscillator, forcing: Forcing, harmonics: int) -> np.ndarray:
    """X_k = H(kω₀) C_k: the steady response's own series (H(0) = 1/k)."""
    C = forcing.coefficients(harmonics)
    H = receptance(osc, forcing.freq_hz * np.arange(harmonics + 1))
    with np.errstate(invalid="ignore"):
        return np.where(C == 0, 0.0, H * C)


def periodic_steady(osc: Oscillator, forcing: Forcing, t: np.ndarray) -> np.ndarray:
    """The steady periodic displacement at the instants t, exactly (no series).

    One period from rest gives z_T = Φ_T z₀ + z_p; the state that repeats is z₀ = (I − Φ_T)⁻¹ z_p. The force is
    stepped by the FOH update at `STEPS` a period (a sample on a jump takes its mean), impulses as
    jumps in velocity. NaN if the oscillator has no damping and a harmonic sits on f_n (no steady state).
    """
    T = forcing.period
    A, B = osc.matrices()
    if forcing.wave is Wave.IMPULSES:
        # just after each blow: z = Φ_T z + [0, I/m]
        Phi = _transition(osc, T)
        b = np.array([0.0, forcing.amplitude / osc.mass])
        z0 = _repeat(Phi, b)
        if z0 is None:
            return np.full(np.shape(t), np.nan)
        u = np.mod(np.asarray(t, dtype=float), T)
        x, _ = free_response(osc, z0[0], z0[1], u)
        return x
    h = T / STEPS
    grid = np.arange(STEPS + 1) * h
    f = _sampled(forcing, grid)
    zp = respond(A, B, h, f)[-1]
    Phi = _transition(osc, T)
    z0 = _repeat(Phi, zp)
    if z0 is None:
        return np.full(np.shape(t), np.nan)
    x = respond(A, B, h, f, z0)[:, 0]
    return np.interp(np.mod(np.asarray(t, dtype=float), T), grid, x)


def _sampled(forcing: Forcing, t: np.ndarray) -> np.ndarray:
    """The force at the samples t for the FOH update: a sample on a jump takes the mean of the two sides, so the
    force linear between samples has the true one's impulse."""
    eps = 1e-9 * forcing.period
    return 0.5 * (forcing.force(t - eps) + forcing.force(t + eps))


def _transition(osc: Oscillator, T: float) -> np.ndarray:
    from scipy.linalg import expm

    A, _ = osc.matrices()
    return expm(A * T)


def _repeat(Phi: np.ndarray, zp: np.ndarray) -> np.ndarray | None:
    M = np.eye(2) - Phi
    if abs(np.linalg.det(M)) < 1e-12:
        return None
    return np.linalg.solve(M, zp)


def from_rest(osc: Oscillator, forcing: Forcing, periods: float, per_period: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """(t, x) from rest over `periods` periods, the force stepped exactly (impulses as velocity jumps)."""
    T = forcing.period
    h = T / per_period
    n = round(periods * per_period)
    t = np.arange(n + 1) * h
    A, B = osc.matrices()
    if forcing.wave is Wave.IMPULSES:
        Phi, _, _ = _step(osc, h)
        z = np.zeros((n + 1, 2))
        state = np.array([0.0, forcing.amplitude / osc.mass])
        z[0] = state
        for i in range(1, n + 1):
            state = Phi @ state
            if i % per_period == 0:
                state = state + np.array([0.0, forcing.amplitude / osc.mass])
            z[i] = state
        return t, z[:, 0]
    f = _sampled(forcing, t)
    f[0] = forcing.force(t[:1])[0]  # switched on at t = 0: nothing before
    return t, respond(A, B, h, f)[:, 0]


def _step(osc: Oscillator, h: float):
    from .simulator import foh_discretize

    A, B = osc.matrices()
    return foh_discretize(A, B, h)


def series_error(osc: Oscillator, forcing: Forcing, harmonics: int, points: int = 2000) -> float:
    """The largest difference over a period between the series cut at `harmonics` and the exact steady state,
    over the steady state's largest swing."""
    t = np.linspace(0.0, forcing.period, points, endpoint=False)
    exact = periodic_steady(osc, forcing, t)
    approx = series(response_coefficients(osc, forcing, harmonics), forcing.freq_hz, t)
    span = float(np.max(np.abs(exact - np.mean(exact)))) or 1.0
    return float(np.max(np.abs(approx - exact))) / span
