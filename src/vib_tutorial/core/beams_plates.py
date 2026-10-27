"""Thick beams and plates in closed form: Timoshenko's corrections, and two-dimensional modes.

The *Beams & plates* page. A simply supported beam of rectangular section has the modes sin(kx), k = nπ/L, in
every theory, so each theory's frequency is a root of a small determinant: Euler–Bernoulli (bending only),
Rayleigh (with the sections' rotary inertia), shear only (with the shear deflection), and Timoshenko (both, whose
2 × 2 determinant has a second root, the thickness-shear branch). A simply supported rectangular plate's modes are
sin(mπx/a) sin(nπy/b), a clamped circular plate's Bessel functions J_n and I_n with cos nθ. `counting` gives the
number of modes below f, against the asymptotes: ∝ √f for a beam, ∝ f for a plate.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass

import numpy as np
import scipy.optimize
import scipy.special

TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class Material:
    E: float = 210e9  # Pa
    nu: float = 0.3
    rho: float = 7850.0  # kg/m³

    @property
    def G(self) -> float:
        return self.E / (2 * (1 + self.nu))


class Theory(enum.Enum):
    EULER = "Euler–Bernoulli"
    RAYLEIGH = "Rayleigh (rotary inertia)"
    SHEAR = "Shear only"
    TIMOSHENKO = "Timoshenko (both)"


THEORIES = tuple(Theory)


@dataclass(frozen=True)
class Beam:
    """A simply supported beam, length L, rectangular section depth h × width b."""

    length: float = 1.0
    depth: float = 0.1
    width: float = 0.05
    material: Material = Material()
    kappa: float = 5.0 / 6.0  # shear coefficient of a rectangle

    @property
    def area(self) -> float:
        return self.depth * self.width

    @property
    def inertia(self) -> float:
        return self.width * self.depth ** 3 / 12

    @property
    def gyration(self) -> float:
        """r = √(I/A) = h/√12."""
        return math.sqrt(self.inertia / self.area)

    @property
    def slenderness(self) -> float:
        return self.length / self.gyration


def beam_omega(beam: Beam, k: np.ndarray, theory: Theory = Theory.TIMOSHENKO, branch: int = 0) -> np.ndarray:
    """ω for the wave sin(kx) (k = nπ/L on the simply supported beam): the lower root (bending), or with
    `branch` 1 Timoshenko's upper one (thickness shear).

    Timoshenko's det [[κGAk² − ρAω², −κGAk], [−κGAk, EIk² + κGA − ρIω²]] = 0, a quadratic in ω²; Rayleigh is its
    κG → ∞ limit, shear-only its ρI → 0 limit."""
    k = np.asarray(k, dtype=float)
    m = beam.material
    A, I, E, rho = beam.area, beam.inertia, m.E, m.rho
    S = beam.kappa * m.G * A
    EI4 = E * I * k ** 4
    if theory is Theory.EULER:
        return np.sqrt(EI4 / (rho * A))
    if theory is Theory.RAYLEIGH:
        return np.sqrt(EI4 / (rho * A + rho * I * k * k))
    if theory is Theory.SHEAR:  # ρAω² = S k² EIk² / (EIk² + S)
        return np.sqrt(S * k * k * E * I * k * k / (E * I * k * k + S) / (rho * A))
    # ρA ρI ω⁴ − [ρA(EIk² + S) + ρI S k²] ω² + S k² EI k² = 0
    a = rho * A * rho * I
    b = rho * A * (E * I * k * k + S) + rho * I * S * k * k
    c = S * k * k * E * I * k * k
    disc = np.sqrt(b * b - 4 * a * c)
    lower = 2 * c / (b + disc)  # stable form of (b − disc)/2a
    upper = (b + disc) / (2 * a)
    return np.sqrt(upper if branch else lower)


def beam_frequencies(beam: Beam, count: int, theory: Theory = Theory.TIMOSHENKO) -> np.ndarray:
    """The first `count` bending frequencies (Hz) of the simply supported beam."""
    k = np.arange(1, count + 1) * math.pi / beam.length
    return beam_omega(beam, k, theory) / TWO_PI


def timoshenko_determinant(beam: Beam, k: float, omega: float) -> float:
    """The 2 × 2 determinant, scaled by its first diagonal term squared (zero at each Timoshenko root)."""
    m = beam.material
    A, I = beam.area, beam.inertia
    S = beam.kappa * m.G * A
    a11 = S * k * k - m.rho * A * omega ** 2
    a22 = m.E * I * k * k + S - m.rho * I * omega ** 2
    return (a11 * a22 - (S * k) ** 2) / (S * k * k) ** 2


def wave_speeds(beam: Beam) -> tuple[float, float]:
    """(√(E/ρ), √(κG/ρ)): the speeds Timoshenko's bending wave and its shear branch tend to at short wavelength."""
    m = beam.material
    return math.sqrt(m.E / m.rho), math.sqrt(beam.kappa * m.G / m.rho)


# --------------------------------------------------------------------------------------------- plates
@dataclass(frozen=True)
class Plate:
    """A plate a × b (or radius a when circular), thickness h."""

    a: float = 0.6
    b: float = 0.4
    thickness: float = 0.003
    material: Material = Material()
    circular: bool = False

    @property
    def rigidity(self) -> float:
        """D = E h³ / 12(1 − ν²)."""
        m = self.material
        return m.E * self.thickness ** 3 / (12 * (1 - m.nu ** 2))

    @property
    def surface_mass(self) -> float:
        return self.material.rho * self.thickness

    @property
    def area(self) -> float:
        return math.pi * self.a ** 2 if self.circular else self.a * self.b

    @property
    def scale(self) -> float:
        """√(D/ρh)."""
        return math.sqrt(self.rigidity / self.surface_mass)


@dataclass(frozen=True)
class PlateMode:
    m: int  # rectangular: half waves along a; circular: nodal diameters n
    n: int  # rectangular: half waves along b; circular: the root's index s (s − 1 nodal circles)
    freq_hz: float
    root: float = 0.0  # circular: λ


def rect_modes(plate: Plate, count: int) -> list[PlateMode]:
    """The simply supported rectangle's lowest `count` modes: ω = π² (m²/a² + n²/b²) √(D/ρh)."""
    top = math.ceil(2 * math.sqrt(count)) + 4
    modes = [PlateMode(m, n, math.pi ** 2 * (m * m / plate.a ** 2 + n * n / plate.b ** 2) * plate.scale / TWO_PI)
             for m in range(1, top + 1) for n in range(1, top + 1)]
    modes.sort(key=lambda p: (p.freq_hz, p.m, p.n))
    return modes[:count]


def rect_shape(mode: PlateMode, plate: Plate, x: np.ndarray, y: np.ndarray, mix: int = 0) -> np.ndarray:
    """sin(mπx/a) sin(nπy/b) on the grid (x along a, rows along y); `mix` ±1 adds ± its twin (n, m), which has
    the same frequency on a square plate."""
    X, Y = np.meshgrid(x, y)
    w = np.sin(mode.m * math.pi * X / plate.a) * np.sin(mode.n * math.pi * Y / plate.b)
    if mix and mode.m != mode.n:
        w = w + mix * np.sin(mode.n * math.pi * X / plate.a) * np.sin(mode.m * math.pi * Y / plate.b)
    return w


def _clamped(n: int, lam: float) -> float:
    """J_n(λ) I_{n+1}(λ) + I_n(λ) J_{n+1}(λ): zero at the clamped circular plate's roots (scaled by e^−λ)."""
    jv, ive = scipy.special.jv, scipy.special.ive
    return jv(n, lam) * ive(n + 1, lam) + ive(n, lam) * jv(n + 1, lam)


def circular_roots(n: int, count: int) -> list[float]:
    """The first `count` λ (ω = λ² √(D/ρh) / a²) with n nodal diameters."""
    grid = np.linspace(0.5, 4.0 * (count + n + 2), 4000)
    vals = np.array([_clamped(n, x) for x in grid])
    roots = []
    for i in np.nonzero(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)[0]:
        roots.append(float(scipy.optimize.brentq(lambda t: _clamped(n, t), grid[i], grid[i + 1], xtol=1e-14)))
        if len(roots) == count:
            break
    return roots


def circular_modes(plate: Plate, count: int) -> list[PlateMode]:
    """The clamped circular plate's lowest `count` modes (each n ≥ 1 is a pair, cos nθ and sin nθ, listed once)."""
    modes = []
    for n in range(count + 1):
        for s, lam in enumerate(circular_roots(n, count), 1):
            modes.append(PlateMode(n, s, lam * lam * plate.scale / (plate.a ** 2 * TWO_PI), lam))
    modes.sort(key=lambda p: (p.freq_hz, p.m))
    return modes[:count]


def circular_shape(mode: PlateMode, x: np.ndarray, y: np.ndarray, edge: float = 1.0) -> np.ndarray:
    """(J_n(λr) − J_n(λ)/I_n(λ) I_n(λr)) cos nθ on the grid (in radii); NaN outside the plate. With `edge` < 1,
    the points beyond r = edge take the value at r = edge instead (the clamped edge touches zero without
    crossing it, so a nodal-line finder needs it kept off)."""
    X, Y = np.meshgrid(x, y)
    r, th = np.hypot(X, Y), np.arctan2(Y, X)
    n, lam = mode.m, mode.root
    ratio = scipy.special.jv(n, lam) / scipy.special.iv(n, lam)
    rr = np.minimum(r, edge)
    w = (scipy.special.jv(n, lam * rr) - ratio * scipy.special.iv(n, lam * rr)) * np.cos(n * th)
    return np.where(r <= 1.0, w, np.nan) if edge >= 1.0 else w


# --------------------------------------------------------------------------------------------- counting
def counting(freqs: np.ndarray, f: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """N(f): how many modes lie at or below each f (a mode's weight is 2 for a circular pair)."""
    freqs = np.asarray(freqs, dtype=float)
    w = np.ones_like(freqs) if weights is None else np.asarray(weights, dtype=float)
    order = np.argsort(freqs)
    cum = np.cumsum(w[order])
    idx = np.searchsorted(freqs[order], f, side="right")
    return np.where(idx > 0, cum[np.maximum(idx - 1, 0)], 0.0)


def plate_asymptote(plate: Plate, f: np.ndarray) -> np.ndarray:
    """N ≈ S ω / 4π √(ρh/D): a plate's modal density, S/2 √(ρh/D) modes per Hz, doesn't depend on f."""
    return plate.area * TWO_PI * np.asarray(f) / (4 * math.pi * plate.scale)


def beam_asymptote(beam: Beam, f: np.ndarray) -> np.ndarray:
    """Euler–Bernoulli: k = (ρAω²/EI)^¼ and N ≈ kL/π, so ∝ √f."""
    m = beam.material
    k = (m.rho * beam.area * (TWO_PI * np.asarray(f)) ** 2 / (m.E * beam.inertia)) ** 0.25
    return k * beam.length / math.pi
