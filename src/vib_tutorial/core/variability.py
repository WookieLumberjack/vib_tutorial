"""Scatter in a structure's properties: Monte Carlo on the chain, and mode localization in a mistuned ring.

The *Variability & localization* page. Two models:

- The Simulation page's chain with every mass, spring and damper scattered (lognormal, so it stays positive, with
  the given coefficient of variation): `monte_carlo` draws `count` chains and gives each one's natural frequencies
  and receptance, for bands on the FRF. `sensitivity_sigma` is the first-order estimate of each frequency's spread:
  with φ mass-normalized, λ = ω² moves by (φ_i − φ_{i−1})² δk_i and −λ φ_i² δm_i, so
  σ_λ² = Σ ((φ_i − φ_{i−1})² k_i c_k)² + Σ (λ φ_i² m_i c_m)² and σ_f/f = σ_λ/2λ.
- A `Ring` of identical blades (a bladed disc): each blade a mass on its own spring k to the disc (held still),
  neighbours joined by weak springs k_c round the ring. Tuned, its modes are travelling waves with n nodal diameters
  in a narrow band; mistuned (each blade's k scattered by a CoV), a weak coupling lets them localize on a few blades
  (`participation`), and an engine-order force (the same on every blade, its phase stepping 2πE/N) can drive one
  blade far harder than the tuned ring would (`amplification`; Whitehead's bound (1 + √N)/2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.linalg

from .model import ChainSystem
from .random_vibration import linear

TWO_PI = 2.0 * math.pi


# ------------------------------------------------------------- scatter on the chain
@dataclass(frozen=True)
class Scatter:
    """Coefficients of variation of the masses, springs and dampers."""

    mass: float = 0.0
    stiffness: float = 0.05
    damping: float = 0.0


def lognormal(rng: np.random.Generator, cov: float, size: int) -> np.ndarray:
    """Factors with mean 1 and the given coefficient of variation (lognormal)."""
    if cov <= 0:
        return np.ones(size)
    s2 = math.log(1 + cov * cov)
    return np.exp(rng.normal(-s2 / 2, math.sqrt(s2), size))


def samples(system: ChainSystem, scatter: Scatter, count: int, seed: int = 1) -> list[ChainSystem]:
    rng = np.random.default_rng(seed)
    lin = linear(system)
    n = lin.n
    return [ChainSystem(lin.masses * lognormal(rng, scatter.mass, n),
                        lin.stiffness * lognormal(rng, scatter.stiffness, n),
                        lin.damping * lognormal(rng, scatter.damping, n)) for _ in range(count)]


def natural_hz(system: ChainSystem) -> tuple[np.ndarray, np.ndarray]:
    """(f, Φ): undamped, Φ mass-normalized."""
    M, _, K = system.matrices()
    w2, phi = scipy.linalg.eigh(K, M)
    return np.sqrt(np.clip(w2, 0, None)) / TWO_PI, phi


def receptance(system: ChainSystem, f: np.ndarray, out: int, inp: int) -> np.ndarray:
    M, C, K = system.matrices()
    w = TWO_PI * np.asarray(f, dtype=float)
    Z = K[None] - (w * w)[:, None, None] * M[None] + 1j * w[:, None, None] * C[None]
    rhs = np.zeros((w.size, system.n, 1), dtype=complex)
    rhs[:, inp, 0] = 1.0
    return np.linalg.solve(Z, rhs)[:, out, 0]


@dataclass(frozen=True)
class MonteCarlo:
    freqs: np.ndarray  # (count, modes) Hz
    f: np.ndarray  # the FRF's grid
    frf: np.ndarray  # (count, grid) |receptance|
    nominal_freqs: np.ndarray
    nominal_frf: np.ndarray

    def band(self, low: float = 5.0, high: float = 95.0) -> tuple[np.ndarray, np.ndarray]:
        return np.percentile(self.frf, low, axis=0), np.percentile(self.frf, high, axis=0)


def monte_carlo(system: ChainSystem, scatter: Scatter, count: int, f: np.ndarray, out: int, inp: int,
                seed: int = 1) -> MonteCarlo:
    lin = linear(system)
    draws = samples(lin, scatter, count, seed)
    freqs = np.array([natural_hz(s)[0] for s in draws])
    frf = np.array([np.abs(receptance(s, f, out, inp)) for s in draws])
    return MonteCarlo(freqs, np.asarray(f), frf, natural_hz(lin)[0], np.abs(receptance(lin, f, out, inp)))


def sensitivity_sigma(system: ChainSystem, scatter: Scatter) -> np.ndarray:
    """First-order σ_f (Hz) of each mode from the springs' and masses' scatter."""
    lin = linear(system)
    f, phi = natural_hz(lin)
    lam = (TWO_PI * f) ** 2
    dphi = np.diff(phi, prepend=0.0, axis=0)  # φ_i − φ_{i−1}, the ground before m1
    var = ((dphi ** 2 * lin.stiffness[:, None] * scatter.stiffness) ** 2).sum(axis=0)
    var += ((lam * phi ** 2 * lin.masses[:, None] * scatter.mass) ** 2).sum(axis=0)
    return f * np.sqrt(var) / (2 * lam)


# ------------------------------------------------------------- a mistuned ring
@dataclass(frozen=True)
class Ring:
    blades: int = 12
    mass: float = 1.0  # kg
    stiffness: float = 1e4  # N/m, each blade to the disc
    coupling: float = 0.05  # k_c / k
    zeta: float = 0.003  # every mode's (a mass-proportional damper)
    mistuning: float = 0.0  # CoV of the blades' k
    seed: int = 1

    @property
    def blade_hz(self) -> float:
        return math.sqrt(self.stiffness / self.mass) / TWO_PI

    def deltas(self) -> np.ndarray:
        """Each blade's stiffness factor − 1: normal with σ = `mistuning`, mean removed (so the mean blade is
        the tuned one)."""
        if self.mistuning <= 0:
            return np.zeros(self.blades)
        d = np.random.default_rng(self.seed).normal(0.0, self.mistuning, self.blades)
        return d - d.mean()

    def matrices(self, deltas: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n, k = self.blades, self.stiffness
        d = self.deltas() if deltas is None else deltas
        kc = self.coupling * k
        K = np.diag(k * (1 + d) + 2 * kc)
        for j in range(n):
            K[j, (j + 1) % n] -= kc
            K[(j + 1) % n, j] -= kc
        if n == 2:  # the two neighbours are the same blade: one spring, not two
            K = np.diag(k * (1 + d) + kc) - kc * (np.ones((2, 2)) - np.eye(2))
        M = self.mass * np.eye(n)
        C = 2 * self.zeta * self.mass * TWO_PI * self.blade_hz * np.eye(n)
        return M, C, K


def ring_modes(ring: Ring, deltas: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    M, _, K = ring.matrices(deltas)
    w2, phi = scipy.linalg.eigh(K, M)
    return np.sqrt(w2) / TWO_PI, phi


def participation(phi: np.ndarray) -> np.ndarray:
    """Each mode's participation number (Σφ²)²/Σφ⁴: how many blades move; N/… for a spread wave, 1 for a mode
    on one blade."""
    p2 = (phi ** 2).sum(axis=0)
    return p2 * p2 / (phi ** 4).sum(axis=0)


def tuned_band(ring: Ring) -> tuple[float, float]:
    """The tuned ring's frequencies run from the blade's (0 nodal diameters) to √(1 + 4k_c/k) times it."""
    return ring.blade_hz, ring.blade_hz * math.sqrt(1 + 4 * ring.coupling)


def forced(ring: Ring, engine_order: int, f: np.ndarray, deltas: np.ndarray | None = None) -> np.ndarray:
    """Each blade's |x| (m) per newton of an engine-order force, F_j = e^{i 2π E j / N}: (frequencies, blades)."""
    M, C, K = ring.matrices(deltas)
    n = ring.blades
    w = TWO_PI * np.asarray(f, dtype=float)
    Z = K[None] - (w * w)[:, None, None] * M[None] + 1j * w[:, None, None] * C[None]
    F = np.exp(2j * math.pi * engine_order * np.arange(n) / n)
    return np.abs(np.linalg.solve(Z, np.broadcast_to(F[:, None], (w.size, n, 1)))[..., 0])


def sweep_grid(ring: Ring, points: int = 1500) -> np.ndarray:
    lo, hi = tuned_band(ring)
    spread = 4 * ring.mistuning + 0.02
    return np.linspace(lo * (1 - spread), hi * (1 + spread), points)


def amplification(ring: Ring, engine_order: int, deltas: np.ndarray | None = None) -> float:
    """The largest blade's peak over the sweep, over the tuned ring's peak (every blade alike when tuned)."""
    f = sweep_grid(ring)
    tuned = float(forced(ring, engine_order, f, np.zeros(ring.blades)).max())
    return float(forced(ring, engine_order, f, deltas).max()) / tuned


def whitehead(blades: int) -> float:
    """Whitehead's upper bound on the amplification (light damping, one engine order): (1 + √N)/2."""
    return (1 + math.sqrt(blades)) / 2


def amplification_curve(ring: Ring, engine_order: int, levels: np.ndarray, seeds: int = 20
                        ) -> tuple[np.ndarray, np.ndarray]:
    """(mean, largest) amplification over `seeds` mistuning patterns at each CoV."""
    mean, top = np.empty(levels.size), np.empty(levels.size)
    f = sweep_grid(Ring(ring.blades, ring.mass, ring.stiffness, ring.coupling, ring.zeta, float(levels.max())))
    tuned = float(forced(ring, engine_order, f, np.zeros(ring.blades)).max())
    for i, cov in enumerate(levels):
        values = []
        for s in range(1, seeds + 1):
            r = Ring(ring.blades, ring.mass, ring.stiffness, ring.coupling, ring.zeta, float(cov), s)
            values.append(float(forced(r, engine_order, f).max()) / tuned)
        mean[i], top[i] = np.mean(values), np.max(values)
    return mean, top
