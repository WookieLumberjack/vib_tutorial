"""Lumped-parameter model of a spring-mass-damper chain.

Topology (N masses, N spring/damper elements)::

    ground --[k1,c1]-- m1 --[k2,c2]-- m2 --[k3,c3]-- m3 ... --[kN,cN]-- mN

Element 1 ties mass 1 to ground; element i (i >= 2) ties mass i-1 to mass i.
The equations of motion are

    M x'' + C x' + K x = f(t) + f_friction

with diagonal M and tridiagonal C, K assembled element by element. Each
mass may also slide with Coulomb friction against the fixed floor: a force of
size friction_i (N, mu N) opposing its velocity, or anything up to that size
while it sticks. Friction makes the chain nonlinear; everything built on M, C
and K (modes, FRFs, substructuring) ignores it, and only the simulator applies it.
All quantities are SI: kg, N/m, N*s/m, m, N.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_MASS = 1.0  # kg
DEFAULT_STIFFNESS = 400.0  # N/m
DEFAULT_DAMPING = 2.0  # N*s/m


@dataclass
class ChainSystem:
    masses: np.ndarray
    stiffness: np.ndarray
    damping: np.ndarray
    friction: np.ndarray | None = None  # N, Coulomb friction on each mass; zeros if None
    _matrices: tuple[np.ndarray, np.ndarray, np.ndarray] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        self.masses = np.asarray(self.masses, dtype=float).copy()
        self.stiffness = np.asarray(self.stiffness, dtype=float).copy()
        self.damping = np.asarray(self.damping, dtype=float).copy()
        n = self.masses.size
        self.friction = np.zeros(n) if self.friction is None else np.asarray(self.friction, dtype=float).copy()
        if not (self.stiffness.size == self.damping.size == self.friction.size == n) or n == 0:
            raise ValueError("masses, stiffness, damping and friction must have the same nonzero length")
        if np.any(self.masses <= 0):
            raise ValueError("masses must be positive")
        if np.any(self.stiffness < 0) or np.any(self.damping < 0) or np.any(self.friction < 0):
            raise ValueError("stiffness, damping and friction must be non-negative")

    @classmethod
    def uniform(
        cls,
        n: int = 4,
        mass: float = DEFAULT_MASS,
        stiffness: float = DEFAULT_STIFFNESS,
        damping: float = DEFAULT_DAMPING,
        friction: float = 0.0,
    ) -> ChainSystem:
        return cls(np.full(n, mass), np.full(n, stiffness), np.full(n, damping), np.full(n, friction))

    @property
    def n(self) -> int:
        return self.masses.size

    def copy(self) -> ChainSystem:
        return ChainSystem(self.masses, self.stiffness, self.damping, self.friction)

    def resized(self, n: int) -> ChainSystem:
        """Return a copy with n masses, padding with the last element's values."""

        def fit(a: np.ndarray) -> np.ndarray:
            return a[:n] if n <= a.size else np.concatenate([a, np.full(n - a.size, a[-1])])

        return ChainSystem(fit(self.masses), fit(self.stiffness), fit(self.damping), fit(self.friction))

    def matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (M, C, K). Cached; treat the arrays as read-only."""
        if self._matrices is None:
            self._matrices = (
                np.diag(self.masses),
                assemble_chain(self.damping),
                assemble_chain(self.stiffness),
            )
        return self._matrices


def assemble_chain(values: np.ndarray) -> np.ndarray:
    """Assemble the tridiagonal matrix for a grounded chain of two-node elements.

    Element i connects node i-1 to node i; element 0 connects ground to node 0.
    Each element contributes [[v, -v], [-v, v]] to its two nodes (the ground
    node's row/column is dropped).
    """
    n = len(values)
    mat = np.zeros((n, n))
    for i, v in enumerate(values):
        mat[i, i] += v
        if i > 0:
            mat[i - 1, i - 1] += v
            mat[i - 1, i] -= v
            mat[i, i - 1] -= v
    return mat


def element_forces(system: ChainSystem, x: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Tension in each spring k_i (x_i - x_{i-1}) and damper c_i (v_i - v_{i-1}) (N); x_0 = ground.

    x and v have the masses along the last axis; so do the results. Positive
    means the element is stretched (or stretching) and pulls its two masses together.
    """
    spring = system.stiffness * np.diff(x, prepend=0.0, axis=-1)
    damper = system.damping * np.diff(v, prepend=0.0, axis=-1)
    return spring, damper


def pluck_shape(system: ChainSystem, dof: int, x_dof: float) -> np.ndarray:
    """Static displacements (m) with mass `dof` held at `x_dof` and no other load.

    This is the shape a chain takes when one mass is pulled aside slowly by
    hand: the other masses sit where their springs balance, K_ff x_f = -K_fd x_d.
    A part of the chain joined to neither the ground nor the held mass by a
    spring has no unique rest position; it stays at zero (least squares).
    """
    K = system.matrices()[2]
    free = np.arange(system.n) != dof
    x = np.zeros(system.n)
    x[dof] = x_dof
    if free.any():
        rhs = -K[np.ix_(free, [dof])][:, 0] * x_dof
        x[free] = np.linalg.lstsq(K[np.ix_(free, free)], rhs, rcond=None)[0]
    return x


def state_space(system: ChainSystem) -> tuple[np.ndarray, np.ndarray]:
    """First-order form z' = A z + B f with z = [x, v].

    B maps a force vector (one entry per mass) into the state derivative.
    """
    M, C, K = system.matrices()
    n = system.n
    Minv = np.diag(1.0 / system.masses)
    A = np.zeros((2 * n, 2 * n))
    A[:n, n:] = np.eye(n)
    A[n:, :n] = -Minv @ K
    A[n:, n:] = -Minv @ C
    B = np.zeros((2 * n, n))
    B[n:, :] = Minv
    return A, B
