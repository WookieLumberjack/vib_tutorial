"""Time integration of the chain using an exact discrete-time state transition.

For z' = A z + B u with u varying linearly across each step of length h
(first-order hold), the exact update is

    z[k+1] = Phi z[k] + (G1 - G2) u[k] + G2 u[k+1]

where Phi = e^{A h} and G1, G2 come from one matrix exponential of an
augmented matrix. Because the homogeneous part is exact, the scheme is
unconditionally stable and has no numerical damping regardless of how stiff
the user makes the springs; h only needs to be small enough to resolve the
forcing and to give smooth plots.

The simulator also keeps an exact energy ledger. Over each step the work done
by the force (integral of f v) and the energy taken out by the dampers
(integral of v^T C v) are quadratic forms in the step's starting state and
force, found with Van Loan's matrix exponential. So, to rounding error,

    energy_added + work - dissipated = T + V

where energy_added counts the jumps in stored energy when the state is set
(a mode release) or the parameters are edited.

With base excitation the input is the ground displacement u = x_g instead of
a force. It reaches mass 1 through spring 1 and damper 1, as
k1 x_g + c1 x_g', so the step also depends on the input's slope du:

    w' = Aw w,  w = [z, u, du],  z[k+1] = Phi z[k] + g_u u[k] + g_du du

(for a force g_du only carries the FOH interpolation). The ground is then
piecewise linear in time and simulated exactly. x stays the absolute
displacement; spring 1 stretches by x1 - x_g, and "work" is the work the
moving ground does on the chain, -integral of (tension in element 1) x_g' dt.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.linalg

from .energy import stored_energy
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


def input_system(A: np.ndarray, b: np.ndarray, b_du: np.ndarray | None = None) -> np.ndarray:
    """Aw for w = [z, u, du] over a step with u(s) = u0 + du s: z' = A z + b u + b_du du."""
    ns = A.shape[0]
    Aw = np.zeros((ns + 2, ns + 2))
    Aw[:ns, :ns] = A
    Aw[:ns, ns] = b
    if b_du is not None:
        Aw[:ns, ns + 1] = b_du
    Aw[ns, ns + 1] = 1.0
    return Aw


def foh_input_step(
    A: np.ndarray, b: np.ndarray, b_du: np.ndarray | None, h: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (Phi, g0, g1): z1 = Phi z0 + g0 u0 + g1 u1 for a scalar input linear over the step."""
    ns = A.shape[0]
    E = scipy.linalg.expm(input_system(A, b, b_du) * h)
    g_du = E[:ns, ns + 1] / h
    return E[:ns, :ns], E[:ns, ns] - g_du, g_du


def foh_quadratic_integrals(
    A: np.ndarray, b: np.ndarray, h: float, Qs: list[np.ndarray], b_du: np.ndarray | None = None
) -> list[np.ndarray]:
    """Exact integrals of quadratic forms over one first-order-hold step.

    Over the step, w(s) = [z(s), u(s), du] with u(s) = u0 + du s and
    du = (u1 - u0) / h obeys the linear system w' = Aw w (see input_system;
    b_du feeds the input's slope, as a damper to a moving ground does). For each symmetric
    Q (size len(z) + 2) this returns W with

        integral_0^h w(s)^T Q w(s) ds = w0^T W w0,   w0 = [z0, u0, du],

    using Van Loan's result: expm([[-Aw^T, Q], [0, Aw]] h) = [[., F12], [0, F22]]
    and W = F22^T F12.
    """
    Aw = input_system(A, b, b_du)
    m = Aw.shape[0]
    out = []
    for Q in Qs:
        big = np.zeros((2 * m, 2 * m))
        big[:m, :m] = -Aw.T
        big[:m, m:] = Q
        big[m:, m:] = Aw
        E = scipy.linalg.expm(big * h)
        W = E[m:, m:].T @ E[:m, m:]
        out.append(0.5 * (W + W.T))
    return out


class Simulator:
    def __init__(self, system: ChainSystem, force: ForceController | None = None) -> None:
        self.force = force or ForceController()
        self.system = system
        self.t = 0.0
        self.state = np.zeros(2 * system.n)
        self._A, self._B = state_space(system)
        self._fmax_natural = self._highest_natural_freq()
        self._cache: dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._energy_cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        # The input where the last step ended, and whether it was a ground motion.
        # The next advance starts from it, so a switch or a jump in the input
        # is a ramp over one step whose work is counted, not a jump in stored energy.
        self._u = 0.0
        self._u_base = self.force.settings.base
        self.ground_velocity = np.empty(0)  # x_g' (m/s) over each step of the last advance
        self.work = 0.0  # J, done by the applied force since reset
        self.dissipated = 0.0  # J, taken out by the dampers since reset
        self.energy_added = 0.0  # J, jumps in stored energy from set_state and parameter edits
        self.ledger = np.empty((0, 3))  # (added, work, dissipated) at each sample of the last advance

    @property
    def ground(self) -> float:
        """Ground displacement x_g (m) now; 0 unless the input is a ground motion."""
        return self._u if self._u_base else 0.0

    @property
    def stored_energy(self) -> float:
        """Kinetic plus potential energy now (J); spring 1 stretches by x1 - x_g."""
        return stored_energy(self.system, self.displacement - self.ground, self.velocity)

    # ----------------------------------------------------------------- setup
    def set_system(self, system: ChainSystem) -> None:
        """Swap in new parameters, keeping the current state (live editing).

        The stored energy jumps (a stiffer spring holds more energy at the same
        stretch); the jump is counted in ``energy_added``.
        """
        before = self.stored_energy
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
        self._energy_cache.clear()
        self.energy_added += self.stored_energy - before

    def reset(self) -> None:
        self.t = 0.0
        self.state = np.zeros(2 * self.system.n)
        self.force.reset()
        self._u = 0.0
        self.work = self.dissipated = self.energy_added = 0.0

    def set_displacement(self, x: np.ndarray) -> None:
        """Set the displacements (m) and zero the velocities, e.g. to release a mode shape."""
        self.set_state(x, np.zeros(self.system.n))

    def set_state(self, x: np.ndarray, v: np.ndarray) -> None:
        """Set displacements (m) and velocities (m/s), e.g. to release a complex mode."""
        n = self.system.n
        before = self.stored_energy
        self.state[:n] = x
        self.state[n:] = v
        self.energy_added += self.stored_energy - before

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
        fmax = max(self._fmax_natural, s.max_freq_hz)
        h = MAX_STEP if fmax <= 0 else min(MAX_STEP, 1.0 / (STEPS_PER_PERIOD * fmax))
        if s.kind is ForceKind.PULSE:
            h = min(h, s.pulse_duration / 10.0)
        return 2.0 ** math.floor(math.log2(h))

    def _input(self) -> tuple[tuple, np.ndarray, np.ndarray | None]:
        """(key, b, b_du): how the scalar input u and its slope enter z' = A z + b u + b_du u'."""
        if self.force.settings.base:
            # The ground pulls mass 1 through k1 and c1: A[n:, :n] @ 1 = -M^-1 K 1 = -k1/m1 e1.
            n = self.system.n
            ones = np.ones(n)
            b = np.zeros(2 * n)
            b_du = np.zeros(2 * n)
            b[n:] = -self._A[n:, :n] @ ones
            b_du[n:] = -self._A[n:, n:] @ ones
            return ("base",), b, b_du
        j = self.force.settings.target
        return ("force", j), self._B[:, j], None

    def _discrete(self, h: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        key, b, b_du = self._input()
        if (h, key) not in self._cache:
            self._cache[h, key] = foh_input_step(self._A, b, b_du, h)
        return self._cache[h, key]

    def _energy_forms(self, h: float) -> tuple[np.ndarray, np.ndarray]:
        """(W_dissipated, W_work) for steps of length h with the current input."""
        key, b, b_du = self._input()
        if (h, key) not in self._energy_cache:
            n = self.system.n
            m = 2 * n + 2
            u, du = 2 * n, 2 * n + 1  # indices of the input and its slope in w = [x, v, u, du]
            C = self.system.matrices()[1]
            Qw = np.zeros((m, m))
            if key[0] == "base":
                # Dampers see v - x_g' 1 (C 1 = c1 e1): (v - du 1)^T C (v - du 1).
                S = np.zeros((n, m))
                S[:, n : 2 * n] = np.eye(n)
                S[:, du] = -1.0
                Qd = S.T @ C @ S
                # Work by the ground: -(k1 (x1 - u) + c1 (v1 - du)) du.
                k1, c1 = self.system.stiffness[0], self.system.damping[0]
                for i, coef in ((0, -k1), (u, k1), (n, -c1)):
                    Qw[i, du] += 0.5 * coef
                    Qw[du, i] += 0.5 * coef
                Qw[du, du] += c1
            else:
                Qd = np.zeros((m, m))
                Qd[n : 2 * n, n : 2 * n] = C  # v^T C v
                Qw[u, n + key[1]] = Qw[n + key[1], u] = 0.5  # u v_j
            Wd, Ww = foh_quadratic_integrals(self._A, b, h, [Qd, Qw], b_du)
            self._energy_cache[h, key] = (Wd, Ww)
        return self._energy_cache[h, key]

    def advance(self, duration: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Advance by approximately `duration` seconds of simulated time.

        Returns per-step samples (t, x, v, force) with shapes (k,), (k, n),
        (k, n), (k,), which the GUI appends to its history buffers; force is
        the ground displacement (m) with base excitation. The energy ledger at
        each of those samples is left in ``self.ledger``, shape (k, 3):
        columns energy_added, work, dissipated, and the ground's velocity over
        each step in ``self.ground_velocity``.
        """
        h = self.step_size()
        steps = min(MAX_STEPS_PER_ADVANCE, max(0, int(round(duration / h))))
        n = self.system.n
        ts = np.empty(steps)
        zs = np.empty((steps, 2 * n))
        fs = np.empty(steps)
        self.ledger = np.empty((steps, 3))
        self.ground_velocity = np.zeros(steps)
        base = self.force.settings.base
        if base != self._u_base:
            # The new kind of input starts from 0. A displaced ground snaps back,
            # which changes spring 1's energy at once: book it like a set_state.
            before = self.stored_energy
            self._u, self._u_base = 0.0, base
            self.energy_added += self.stored_energy - before
        if steps == 0:
            return ts, zs[:, :n], zs[:, n:], fs

        Phi, g0, g1 = self._discrete(h)
        z = self.state
        z_start, f_start = z.copy(), self._u
        f0 = f_start
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
        self._u = float(fs[-1])

        # Energy ledger: each step's integrals are quadratic in w0 = [z0, f0, (f1 - f0) / h].
        z0s = np.vstack([z_start, zs[:-1]])
        f0s = np.concatenate([[f_start], fs[:-1]])
        w = np.column_stack([z0s, f0s, (fs - f0s) / h])
        Wd, Ww = self._energy_forms(h)
        dissipated = self.dissipated + np.cumsum(np.einsum("ki,ij,kj->k", w, Wd, w))
        work = self.work + np.cumsum(np.einsum("ki,ij,kj->k", w, Ww, w))
        self.dissipated, self.work = float(dissipated[-1]), float(work[-1])
        self.ledger = np.column_stack([np.full(steps, self.energy_added), work, dissipated])
        if base:
            self.ground_velocity = w[:, -1]
        return ts, zs[:, :n], zs[:, n:], fs
