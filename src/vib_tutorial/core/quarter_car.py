"""A quarter-car on a rough road: ride comfort, road holding and suspension travel from the road's PSD.

The *Quarter-car* page. The body (sprung mass m_s) rides on the suspension (k_s, c_s) over the wheel (unsprung
m_u), which rides on the tyre (k_t) over the road's height z_r(t). The road is an ISO 8608 class: a displacement
PSD in space G_d(n) = G_d(n₀)(n/n₀)^−2 (n in cycles/m, n₀ = 0.1), so at speed v its height's PSD in time is
G_d(f/v)/v and its vertical velocity is white. Each output's PSD is |H(f)|² G_z(f) (`response`) and its RMS the
square root of the area, worked out on a grid packed round the two modes (body bounce and wheel hop). `tradeoff`
sweeps the damper (and the spring) for the classic conflict between comfort and grip. `drive` synthesizes a road
with that PSD and steps the car over it by the exact FOH update, as a check.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, replace

import numpy as np
import scipy.special

from .simulator import foh_discretize

TWO_PI = 2.0 * math.pi
G = 9.80665
N0 = 0.1  # cycles/m, ISO 8608's reference spatial frequency
# ISO 8608 classes: G_d(n₀), the geometric mean of each class's range (m³)
CLASSES = {c: 16e-6 * 4.0 ** i for i, c in enumerate("ABCDEFGH")}
CLASS_NAMES = tuple(CLASSES)
BAND = (0.05, 60.0)  # Hz, the band the RMS values are integrated over
GRID = 3000


class Output(enum.Enum):
    BODY = "Body acceleration"
    TRAVEL = "Suspension travel"
    TYRE = "Dynamic tyre load"


OUTPUTS = tuple(Output)


@dataclass(frozen=True)
class Car:
    sprung: float = 300.0  # kg, a quarter of the body
    unsprung: float = 40.0  # kg, wheel, brake, half the axle
    spring: float = 20e3  # N/m
    damper: float = 1.5e3  # N·s/m
    tyre: float = 200e3  # N/m

    @property
    def weight(self) -> float:
        """The tyre's static load (N)."""
        return (self.sprung + self.unsprung) * G

    def matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """M, C, K for q = [x_body, x_wheel] (up, from rest)."""
        M = np.diag([self.sprung, self.unsprung])
        C = self.damper * np.array([[1.0, -1.0], [-1.0, 1.0]])
        K = np.array([[self.spring, -self.spring], [-self.spring, self.spring + self.tyre]])
        return M, C, K

    def modes(self) -> tuple[float, float]:
        """The undamped body-bounce and wheel-hop frequencies (Hz)."""
        M, _, K = self.matrices()
        w2 = np.sort(np.linalg.eigvals(np.linalg.solve(M, K)).real)
        return float(np.sqrt(w2[0]) / TWO_PI), float(np.sqrt(w2[1]) / TWO_PI)

    def body_zeta(self) -> float:
        """The body's damping ratio on the suspension alone (wheel held): c / 2√(k_s m_s)."""
        return self.damper / (2 * math.sqrt(self.spring * self.sprung))


@dataclass(frozen=True)
class Road:
    grade: str = "C"
    speed: float = 20.0  # m/s

    @property
    def gd0(self) -> float:
        return CLASSES[self.grade]

    def psd(self, f: np.ndarray) -> np.ndarray:
        """The road height's one-sided PSD in time (m²/Hz) at f: G_d(f/v)/v = G_d(n₀) n₀² v / f²."""
        f = np.asarray(f, dtype=float)
        return self.gd0 * N0 * N0 * self.speed / (f * f)


def transfer(car: Car, f: np.ndarray) -> dict[Output, np.ndarray]:
    """Each output per unit road height, at f (Hz): body acceleration (m/s² per m), travel x_b − x_w (m/m), and
    the tyre's dynamic load k_t (z_r − x_w) (N/m, positive pressing harder). The 2 × 2 solve in closed form."""
    w = TWO_PI * np.asarray(f, dtype=float)
    a11 = car.spring - w * w * car.sprung + 1j * w * car.damper
    a12 = -car.spring - 1j * w * car.damper
    a22 = car.spring + car.tyre - w * w * car.unsprung + 1j * w * car.damper
    det = a11 * a22 - a12 * a12
    xb, xw = -a12 * car.tyre / det, a11 * car.tyre / det
    return {Output.BODY: -w * w * xb, Output.TRAVEL: xb - xw, Output.TYRE: car.tyre * (1 - xw)}


def grid(car: Car) -> np.ndarray:
    """Frequencies over BAND, packed round the two modes."""
    f = [np.geomspace(*BAND, GRID)]
    for fn in car.modes():
        f.append(fn * np.geomspace(0.5, 2.0, 400))
    g = np.unique(np.concatenate(f))
    return g[(g >= BAND[0]) & (g <= BAND[1])]


@dataclass(frozen=True)
class Response:
    f: np.ndarray
    psd: dict[Output, np.ndarray]
    rms: dict[Output, float]
    weight: float

    @property
    def load_ratio(self) -> float:
        """σ of the dynamic tyre load over the static load."""
        return self.rms[Output.TYRE] / self.weight

    @property
    def lift_probability(self) -> float:
        """P(dynamic load < −static load): the tyre leaving the road, Gaussian."""
        return 0.5 * float(scipy.special.erfc(1 / (math.sqrt(2) * self.load_ratio)))


def response(car: Car, road: Road) -> Response:
    f = grid(car)
    Gz = road.psd(f)
    H = transfer(car, f)
    psd = {o: np.abs(h) ** 2 * Gz for o, h in H.items()}
    rms = {o: float(math.sqrt(np.trapezoid(p, f))) for o, p in psd.items()}
    return Response(f, psd, rms, car.weight)


@dataclass(frozen=True)
class Tradeoff:
    """For each spring (rows) and damper (columns): the RMS body acceleration, tyre load ratio and travel."""

    springs: np.ndarray
    dampers: np.ndarray
    body: np.ndarray
    tyre: np.ndarray
    travel: np.ndarray


def tradeoff(car: Car, road: Road, springs: np.ndarray, dampers: np.ndarray) -> Tradeoff:
    shape = (len(springs), len(dampers))
    body, tyre, travel = np.empty(shape), np.empty(shape), np.empty(shape)
    f = np.geomspace(*BAND, 1200)
    for i, k in enumerate(springs):
        for j, c in enumerate(dampers):
            variant = replace(car, spring=float(k), damper=float(c))
            fg = np.unique(np.concatenate([f, *(fn * np.geomspace(0.6, 1.6, 120) for fn in variant.modes())]))
            H = transfer(variant, fg)
            G_ = road.psd(fg)
            body[i, j] = math.sqrt(np.trapezoid(np.abs(H[Output.BODY]) ** 2 * G_, fg))
            tyre[i, j] = math.sqrt(np.trapezoid(np.abs(H[Output.TYRE]) ** 2 * G_, fg)) / variant.weight
            travel[i, j] = math.sqrt(np.trapezoid(np.abs(H[Output.TRAVEL]) ** 2 * G_, fg))
    return Tradeoff(np.asarray(springs), np.asarray(dampers), body, tyre, travel)


def best_damper(car: Car, road: Road, output: Output, dampers: np.ndarray) -> float:
    """The damper (on the grid) that minimizes the output's RMS."""
    t = tradeoff(car, road, np.array([car.spring]), dampers)
    values = {Output.BODY: t.body, Output.TYRE: t.tyre, Output.TRAVEL: t.travel}[output][0]
    return float(dampers[int(np.argmin(values))])


@dataclass(frozen=True)
class Drive:
    t: np.ndarray
    road: np.ndarray  # m
    body: np.ndarray  # m/s²
    travel: np.ndarray  # m
    tyre: np.ndarray  # N


def road_profile(road: Road, duration: float, fs: float, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """(t, z_r): a road height with the PSD over BAND, random phases on lines 1/duration apart."""
    n = round(duration * fs)
    n += n % 2
    f = np.arange(n // 2 + 1) * fs / n
    psd = np.zeros_like(f)
    inside = (f >= BAND[0]) & (f <= BAND[1])
    psd[inside] = road.psd(f[inside])
    amp = np.sqrt(2 * psd * fs / n)
    rng = np.random.default_rng(seed)
    z = np.fft.irfft(amp * np.exp(1j * rng.uniform(0, TWO_PI, f.size)) * n / 2, n)
    return np.arange(n) / fs, z


def drive(car: Car, road: Road, duration: float = 60.0, fs: float = 400.0, seed: int = 1) -> Drive:
    """The car over a synthesized road, by the exact FOH update (the road linear between samples)."""
    t, z = road_profile(road, duration, fs, seed)
    M, C, K = car.matrices()
    Mi = np.linalg.inv(M)
    A = np.block([[np.zeros((2, 2)), np.eye(2)], [-Mi @ K, -Mi @ C]])
    B = np.concatenate([np.zeros(2), Mi @ np.array([0.0, car.tyre])])[:, None]
    Phi, G0, G1 = foh_discretize(A, B, 1.0 / fs)
    # start on the road's steady state at its first height (both masses lifted by z[0])
    s = np.array([z[0], z[0], 0.0, 0.0])
    states = np.empty((z.size, 4))
    states[0] = s
    for k in range(1, z.size):
        s = Phi @ s + G0[:, 0] * z[k - 1] + G1[:, 0] * z[k]
        states[k] = s
    acc = states @ A[2].T + B[2, 0] * z
    return Drive(t, z, acc, states[:, 0] - states[:, 1], car.tyre * (z - states[:, 1]))
