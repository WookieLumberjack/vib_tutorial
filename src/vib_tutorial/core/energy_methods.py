"""Energy methods on the chain: Rayleigh's quotient, Dunkerley's sum, Rayleigh–Ritz and Holzer's table.

The *Energy methods* page. Every estimate here works from the chain's two energies, T = ½ẋᵀMẋ and
V = ½xᵀKx (Lagrange's equations give M and K from them). Rayleigh's quotient xᵀKx / xᵀMx is ω₁² for the
first mode's shape and above it for any other shape, by a margin second order in the shape's error.
Dunkerley's sum Σ mᵢ fᵢᵢ (f = K⁻¹) is above 1/ω₁², so it bounds ω₁ from below. Rayleigh–Ritz restricts the
chain to r shapes and solves that small eigenproblem: each estimate is above its mode's exact value and falls
as r grows. Holzer's table steps along the chain at a trial ω, from the ground to the free end; the force left
over at the free end is zero only at a natural frequency.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import scipy.linalg

from .model import ChainSystem

TWO_PI = 2.0 * np.pi


class Shape(Enum):
    STATIC = "Static deflection (its weight)"
    LINEAR = "A straight line"
    UNIFORM = "All masses equal (rigid)"
    SINE = "A quarter sine"
    TIP = "Only the last mass"


SHAPES = tuple(Shape)


class Basis(Enum):
    KRYLOV = "Static deflections, repeated (Krylov)"
    POWERS = "Powers: i, i², i³, …"
    SINES = "Quarter sines: 1, 3, 5, … quarters"


BASES = tuple(Basis)


def exact(system: ChainSystem) -> tuple[np.ndarray, np.ndarray]:
    """(ω, Φ): the undamped natural frequencies (rad/s), ascending, and mass-normalized shapes."""
    M, _, K = system.matrices()
    w2, phi = scipy.linalg.eigh(K, M)
    return np.sqrt(np.clip(w2, 0.0, None)), phi


def trial_shape(system: ChainSystem, shape: Shape) -> np.ndarray:
    """A first guess at mode 1's shape, its largest entry 1."""
    n = system.n
    i = np.arange(1, n + 1, dtype=float)
    if shape is Shape.STATIC:
        x = static_deflection(system, system.masses)
    elif shape is Shape.LINEAR:
        x = i
    elif shape is Shape.UNIFORM:
        x = np.ones(n)
    elif shape is Shape.SINE:
        x = np.sin(np.pi * i / (2 * n + 1))
    else:
        x = np.zeros(n)
        x[-1] = 1.0
    return x / np.max(np.abs(x))


def static_deflection(system: ChainSystem, load: np.ndarray) -> np.ndarray:
    """K⁻¹ load (least squares if the chain isn't held to the ground)."""
    _, _, K = system.matrices()
    return np.linalg.lstsq(K, np.asarray(load, dtype=float), rcond=None)[0]


def rayleigh(system: ChainSystem, x: np.ndarray) -> float:
    """Rayleigh's quotient xᵀKx / xᵀMx (rad²/s²)."""
    M, _, K = system.matrices()
    return float(x @ K @ x) / float(x @ M @ x)


def iterate(system: ChainSystem, x: np.ndarray, times: int) -> np.ndarray:
    """x ← K⁻¹M x, `times` times (inverse iteration), its largest entry 1: each pass cuts the higher modes'
    share by ω₁²/ωᵣ² against mode 1's."""
    M, _, _ = system.matrices()
    for _ in range(times):
        x = static_deflection(system, M @ x)
        x = x / np.max(np.abs(x))
    return x


def contaminated(system: ChainSystem, x: np.ndarray, eps: float) -> np.ndarray:
    """x plus eps of mode 2's shape (both scaled to a largest entry of 1)."""
    if system.n < 2:
        return x
    _, phi = exact(system)
    second = phi[:, 1] / np.max(np.abs(phi[:, 1]))
    return x + eps * second


def dunkerley(system: ChainSystem) -> float:
    """Dunkerley's estimate of ω₁² (rad²/s²): 1 / Σ mᵢ fᵢᵢ, below the exact one."""
    M, _, K = system.matrices()
    f = np.linalg.inv(K)
    return 1.0 / float(np.sum(np.diag(M) * np.diag(f)))


def dunkerley_terms(system: ChainSystem) -> np.ndarray:
    """Each mass alone on the massless chain: its own ω² = 1 / (mᵢ fᵢᵢ)."""
    M, _, K = system.matrices()
    return 1.0 / (np.diag(M) * np.diag(np.linalg.inv(K)))


def ritz_basis(system: ChainSystem, basis: Basis, r: int) -> np.ndarray:
    """r shapes as columns (r ≤ N), orthonormalized."""
    n = system.n
    r = max(1, min(r, n))
    i = np.arange(1, n + 1, dtype=float)
    if basis is Basis.KRYLOV:
        M, _, _ = system.matrices()
        cols = [static_deflection(system, system.masses)]
        for _ in range(r - 1):
            cols.append(static_deflection(system, M @ cols[-1]))
        Q = np.column_stack(cols)
    elif basis is Basis.POWERS:
        Q = np.column_stack([(i / n) ** p for p in range(1, r + 1)])
    else:
        Q = np.column_stack([np.sin((2 * p - 1) * np.pi * i / (2 * n + 1)) for p in range(1, r + 1)])
    return np.linalg.qr(Q)[0]  # the same span, orthonormal: powers of i are nearly parallel


def rayleigh_ritz(system: ChainSystem, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(ω, shapes): the chain restricted to the columns of Q, (QᵀKQ) a = ω² (QᵀMQ) a; shapes = Q a."""
    M, _, K = system.matrices()
    w2, a = scipy.linalg.eigh(Q.T @ K @ Q, Q.T @ M @ Q)
    return np.sqrt(np.clip(w2, 0.0, None)), Q @ a


def ritz_table(system: ChainSystem, basis: Basis) -> np.ndarray:
    """table[r - 1, j]: mode j's Rayleigh–Ritz frequency (Hz) with r shapes (NaN for j ≥ r)."""
    n = system.n
    table = np.full((n, n), np.nan)
    for r in range(1, n + 1):
        w, _ = rayleigh_ritz(system, ritz_basis(system, basis, r))
        table[r - 1, :r] = w / TWO_PI
    return table


@dataclass(frozen=True)
class HolzerRow:
    """One mass's line in Holzer's table at a trial ω: its displacement, its inertia force mω²x, and the
    force in the spring after it (the sum of the inertia forces up to it, less the ground spring's pull)."""

    x: float
    inertia: float
    spring: float


def holzer(system: ChainSystem, omega: float) -> list[HolzerRow]:
    """Holzer's table at ω, from the ground: x₁ = 1 (the ground spring then pulls k₁), and each mass passes on
    the force it doesn't use, Fᵢ₊₁ = Fᵢ − mᵢω²xᵢ, which stretches the next spring by Fᵢ₊₁/kᵢ₊₁."""
    m, k = system.masses, system.stiffness
    rows = []
    x = 1.0
    force = k[0] * x  # the ground spring's tension
    for i in range(system.n):
        inertia = m[i] * omega ** 2 * x
        after = force - inertia  # left for the spring beyond mass i
        rows.append(HolzerRow(x, inertia, after))
        if i + 1 < system.n:
            x = x + after / k[i + 1] if k[i + 1] > 0 else np.nan
            force = after
    return rows


def holzer_residual(system: ChainSystem, omega: np.ndarray) -> np.ndarray:
    """The force left at the free end (N, for x₁ = 1) at each ω: zero only at a natural frequency.

    It is −det(K − ω²M) / (k₂ k₃ … k_N), so its sign changes at each mode (a test checks).
    """
    m, k = system.masses, system.stiffness
    w2 = np.asarray(omega, dtype=float) ** 2
    x = np.ones_like(w2)
    force = np.full_like(w2, k[0])
    for i in range(system.n):
        after = force - m[i] * w2 * x
        if i + 1 < system.n:
            x = x + after / k[i + 1]
        force = after
    return force


def holzer_roots(system: ChainSystem, f_max: float, points: int = 4000) -> np.ndarray:
    """The natural frequencies (Hz) found from Holzer's residual: its sign changes, refined by bisection."""
    f = np.linspace(1e-6, f_max, points)
    res = holzer_residual(system, TWO_PI * f)
    roots = []
    for j in np.nonzero(np.sign(res[:-1]) * np.sign(res[1:]) < 0)[0]:
        lo, hi = f[j], f[j + 1]
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if np.sign(holzer_residual(system, np.array([TWO_PI * mid]))[0]) == np.sign(res[j]):
                lo = mid
            else:
                hi = mid
        roots.append(0.5 * (lo + hi))
    return np.array(roots)


def lagrange_matrices(system: ChainSystem) -> tuple[np.ndarray, np.ndarray]:
    """M and K as the Hessians of T = ½Σ mᵢẋᵢ² and V = ½Σ kᵢ(xᵢ − xᵢ₋₁)², by differencing the energies (exact,
    since they are quadratic): what Lagrange's equations put in Mẍ + Kx = 0."""
    n = system.n
    m, k = system.masses, system.stiffness

    def T(v):
        return 0.5 * float(np.sum(m * v ** 2))

    def V(x):
        return 0.5 * float(np.sum(k * np.diff(x, prepend=0.0) ** 2))

    def hessian(energy):
        H = np.zeros((n, n))
        e = np.eye(n)
        for a in range(n):
            for b in range(n):
                H[a, b] = energy(e[a] + e[b]) - energy(e[a]) - energy(e[b])
        return H

    return hessian(T), hessian(V)
