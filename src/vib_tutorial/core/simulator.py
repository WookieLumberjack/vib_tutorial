"""Time integration of the chain using an exact discrete-time state transition.

For z' = A z + B u with u varying linearly across each step of length h
(first-order hold), the exact update is

    z[k+1] = Phi z[k] + (G1 - G2) u[k] + G2 u[k+1]

where Phi = e^{A h} and G1, G2 come from one matrix exponential of an
augmented matrix. Because the homogeneous part is exact, the scheme is
unconditionally stable and has no numerical damping regardless of how stiff
the user makes the springs; h only needs to be small enough to resolve the
forcing and to give smooth plots.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.linalg

from .forcing import ForceController, ForceKind
from .model import ChainSystem, state_space

MAX_STEP = 1e-3  # s
STEPS_PER_PERIOD = 40
MAX_STEPS_PER_ADVANCE = 20_000


def foh_discretize(A: np.ndarray, B: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (Phi, Gamma0, Gamma1): z1 = Phi z0 + Gamma0 u0 + Gamma1 u1."""
    ns, nu = B.shape
    aug = np.zeros((ns + 2 * nu, ns + 2 * nu))
    aug[:ns, :ns] = A * h
    aug[:ns, ns : ns + nu] = B * h
    aug[ns : ns + nu, ns + nu :] = np.eye(nu)
    E = scipy.linalg.expm(aug)
    Phi = E[:ns, :ns]
    G1 = E[:ns, ns : ns + nu]
    G2 = E[:ns, ns + nu :]
    return Phi, G1 - G2, G2


class Simulator:
    def __init__(self, system: ChainSystem, force: ForceController | None = None) -> None:
        self.force = force or ForceController()
        self.system = system
        self.t = 0.0
        self.state = np.zeros(2 * system.n)
        self._A, self._B = state_space(system)
        self._fmax_natural = self._highest_natural_freq()
        self._cache: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    # ----------------------------------------------------------------- setup
    def set_system(self, system: ChainSystem) -> None:
        """Swap in new parameters, keeping the current state (live editing)."""
        if system.n != self.system.n:
            old = self.state
            n_old, n = self.system.n, system.n
            k = min(n_old, n)
            self.state = np.zeros(2 * n)
            self.state[:k] = old[:k]
            self.state[n : n + k] = old[n_old : n_old + k]
            self.force.settings.target = min(self.force.settings.target, n - 1)
        self.system = system
        self._A, self._B = state_space(system)
        self._fmax_natural = self._highest_natural_freq()
        self._cache.clear()

    def reset(self) -> None:
        self.t = 0.0
        self.state = np.zeros(2 * self.system.n)
        self.force.reset()

    def set_displacement(self, x: np.ndarray) -> None:
        """Set the displacements (m) and zero the velocities, e.g. to release a mode shape."""
        n = self.system.n
        self.state[:n] = x
        self.state[n:] = 0.0

    def set_state(self, x: np.ndarray, v: np.ndarray) -> None:
        """Set displacements (m) and velocities (m/s), e.g. to release a complex mode."""
        n = self.system.n
        self.state[:n] = x
        self.state[n:] = v

    @property
    def displacement(self) -> np.ndarray:
        return self.state[: self.system.n]

    @property
    def velocity(self) -> np.ndarray:
        return self.state[self.system.n :]

    # ------------------------------------------------------------ stepping
    def _highest_natural_freq(self) -> float:
        # Largest |eigenvalue| of the state matrix bounds the fastest dynamics.
        lam = np.linalg.eigvals(self._A)
        return float(np.abs(lam).max()) / (2.0 * math.pi) if lam.size else 0.0

    def step_size(self) -> float:
        """Choose h to resolve the fastest mode, the forcing, and any pulse.

        Rounded down to a power of two so the discretization cache gets reused
        while the user drags the frequency around.
        """
        s = self.force.settings
        fmax = self._fmax_natural
        if s.kind is ForceKind.HARMONIC:
            fmax = max(fmax, s.freq_hz)
        h = MAX_STEP if fmax <= 0 else min(MAX_STEP, 1.0 / (STEPS_PER_PERIOD * fmax))
        if s.kind is ForceKind.PULSE:
            h = min(h, s.pulse_duration / 10.0)
        return 2.0 ** math.floor(math.log2(h))

    def _discrete(self, h: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if h not in self._cache:
            self._cache[h] = foh_discretize(self._A, self._B, h)
        return self._cache[h]

    def advance(self, duration: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Advance by approximately `duration` seconds of simulated time.

        Returns per-step samples (t, x, v, force) with shapes (k,), (k, n),
        (k, n), (k,), which the GUI appends to its history buffers.
        """
        h = self.step_size()
        steps = min(MAX_STEPS_PER_ADVANCE, max(0, int(round(duration / h))))
        n = self.system.n
        ts = np.empty(steps)
        zs = np.empty((steps, 2 * n))
        fs = np.empty(steps)
        if steps == 0:
            return ts, zs[:, :n], zs[:, n:], fs

        Phi, G0, G1 = self._discrete(h)
        j = self.force.settings.target
        g0 = G0[:, j].copy()
        g1 = G1[:, j].copy()
        z = self.state
        f0 = self.force.value()
        for k in range(steps):
            self.force.advance(h)
            f1 = self.force.value()
            z = Phi @ z + g0 * f0 + g1 * f1
            self.t += h
            ts[k] = self.t
            zs[k] = z
            fs[k] = f1
            f0 = f1
        self.state = z
        return ts, zs[:, :n], zs[:, n:], fs
