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

Coulomb friction on the masses (``ChainSystem.friction``) makes the chain
nonlinear, but only piecewise: while every mass keeps sliding the same way or
stays stuck, friction is a constant force on each sliding mass and a stuck
mass has no acceleration (its row of A is zeroed). That is again a linear
system, w' = Aw w with w = [z, u, du, e] and e the friction forces, stepped
exactly with one matrix exponential. Within a step the simulator looks for
the first moment a sliding mass stops (its velocity changes sign) or a stuck
one breaks free (the pull of its springs, dampers and the force exceeds the
friction), finds it on a cubic through the step's ends, steps exactly to it,
updates which masses stick and carries on. At a stop the mass sticks if the
pull on it is no more than its friction, and slides back otherwise (the
static and sliding friction are equal). The heat friction makes over a
sub-step is exactly -e . (change in x), so the energy ledger stays exact.

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
EVENTS_PER_MASS = 4  # stick-slip events handled within one step, per mass (then the rest is stepped as is)


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
    return quadratic_integrals(input_system(A, b, b_du), h, Qs)


def quadratic_integrals(Aw: np.ndarray, h: float, Qs: list[np.ndarray]) -> list[np.ndarray]:
    """W with integral_0^h w^T Q w ds = w0^T W w0 for each Q, along w' = Aw w (see foh_quadratic_integrals)."""
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


def first_root(g0: float, d0: float, g1: float, d1: float, h: float) -> float:
    """First time in (0, h] where the cubic with g(0) = g0, g'(0) = d0, g(h) = g1, g'(h) = d1 is zero.

    g0 >= 0 > g1. It locates a stick-slip event within a step; the error is O(h^4).
    """
    a, b = h * d0, h * d1
    coeffs = [2 * g0 + a - 2 * g1 + b, -3 * g0 - 2 * a + 3 * g1 - b, a, g0]
    ts = [r.real for r in np.roots(coeffs) if abs(r.imag) <= 1e-9 and 1e-12 < r.real <= 1 + 1e-9]
    if ts:
        return h * min(min(ts), 1.0)
    return h * (g0 / (g0 - g1) if g0 > 0 else 1.0)


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
        self._friction_cache: dict[tuple, np.ndarray | tuple[np.ndarray, np.ndarray]] = {}
        # The input where the last step ended, and whether it was a ground motion.
        # The next advance starts from it, so a switch or a jump in the input
        # is a ramp over one step whose work is counted, not a jump in stored energy.
        self._u = 0.0
        self._u_base = self.force.settings.base
        self.ground_velocity = np.empty(0)  # x_g' (m/s) over each step of the last advance
        self.work = 0.0  # J, done by the applied force since reset
        self.dissipated = 0.0  # J, taken out by the dampers since reset
        self.friction_loss = 0.0  # J, taken out by friction since reset
        self.energy_added = 0.0  # J, jumps in stored energy from set_state and parameter edits
        # (added, work, dissipated, friction_loss) at each sample of the last advance
        self.ledger = np.empty((0, 4))

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
        self._friction_cache.clear()
        self.energy_added += self.stored_energy - before

    def reset(self) -> None:
        self.t = 0.0
        self.state = np.zeros(2 * self.system.n)
        self.force.reset()
        self._u = 0.0
        self.work = self.dissipated = self.friction_loss = self.energy_added = 0.0

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
            Qd, Qw = self._energy_quadratics(key)
            self._energy_cache[h, key] = tuple(foh_quadratic_integrals(self._A, b, h, [Qd, Qw], b_du))
        return self._energy_cache[h, key]

    def _energy_quadratics(self, key: tuple) -> tuple[np.ndarray, np.ndarray]:
        """(Q_dissipated, Q_work): the dampers' power and the input's, as quadratic forms in w = [x, v, u, du]."""
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
        return Qd, Qw

    # ------------------------------------------------------------ friction
    @property
    def has_friction(self) -> bool:
        return bool(np.any(self.system.friction > 0))

    def _pull_rows(self, _key: tuple, b: np.ndarray, b_du: np.ndarray | None) -> np.ndarray:
        """P: P @ [x, v, u, du] is the force of the springs, dampers and input on each mass (friction aside)."""
        n = self.system.n
        Aw = input_system(self._A, b, b_du)
        return self.system.masses[:, None] * Aw[n : 2 * n]

    def _friction_system(self, key: tuple, b: np.ndarray, b_du: np.ndarray | None, stuck: np.ndarray) -> np.ndarray:
        """Aw for w = [x, v, u, du, e], e the friction force on each mass, with the stuck masses held still."""
        ck = ("Aw", key, stuck.tobytes())
        if ck not in self._friction_cache:
            n = self.system.n
            Aw = np.zeros((3 * n + 2, 3 * n + 2))
            Aw[: 2 * n + 2, : 2 * n + 2] = input_system(self._A, b, b_du)
            Aw[n : 2 * n, 2 * n + 2 :] = np.diag(1.0 / self.system.masses)
            Aw[n + np.flatnonzero(stuck)] = 0.0
            self._friction_cache[ck] = Aw
        return self._friction_cache[ck]

    def _friction_step_maps(self, key: tuple, Aw: np.ndarray, stuck: np.ndarray, s: float, h: float, ledger: bool):
        """(Phi, W_dissipated, W_work) over s (the W are None without the ledger); cached for whole steps."""
        ck = ("step", key, stuck.tobytes(), ledger, h)
        if s == h and ck in self._friction_cache:
            return self._friction_cache[ck]
        out: tuple = (scipy.linalg.expm(Aw * s), None, None)
        if ledger:
            n = self.system.n
            Qs = []
            for Q in self._energy_quadratics(key):
                padded = np.zeros((3 * n + 2, 3 * n + 2))
                padded[: 2 * n + 2, : 2 * n + 2] = Q
                Qs.append(padded)
            out = (out[0], *quadratic_integrals(Aw, s, Qs))
        if s == h:
            self._friction_cache[ck] = out
        return out

    def _slip(self, w: np.ndarray, P: np.ndarray, forced: dict[int, float]) -> tuple[np.ndarray, np.ndarray]:
        """(stuck, e): which masses stick, and the friction force on each sliding one.

        A moving mass slides the way it moves. One at rest sticks while the pull
        on it is no more than its friction, and otherwise starts to slide the way
        it is pulled. ``forced`` overrides this for a mass that just broke free.
        """
        n = self.system.n
        F = self.system.friction
        v = w[n : 2 * n]
        way = np.sign(v)
        rest = (v == 0.0) & (F > 0)
        stuck = np.zeros(n, dtype=bool)
        if forced or rest.any():
            pull = P @ w[: 2 * n + 2]
            way = np.where(rest, np.sign(pull), way)
            stuck = rest & (np.abs(pull) <= F)
        for i, d in forced.items():
            stuck[i], way[i] = False, d
        return stuck, np.where(stuck, 0.0, -F * way)

    def _first_event(self, w0, w1, Aw, P, stuck, e, s) -> tuple[float, int, float] | None:
        """(time, mass, way) of the first stop (way 0) or break-free (way ±1) within the step w0 -> w1."""
        n = self.system.n
        m = 2 * n + 2
        F = self.system.friction
        # A sliding mass whose velocity changed sign stopped on the way: its velocity
        # now has the sign of the friction force (which opposed the sliding).
        reversed_ = e * w1[n : 2 * n] > 0.0
        # A stuck mass whose pull grew beyond its friction broke free on the way.
        frees, p1 = np.empty(0, dtype=int), np.empty(0)
        if stuck.any():
            frees = np.flatnonzero(stuck)
            p1 = P[frees] @ w1[:m]
            keep = np.abs(p1) > F[frees]
            frees, p1 = frees[keep], p1[keep]
        elif not reversed_.any():
            return None
        stops = np.flatnonzero(reversed_)
        if not stops.size and not frees.size:
            return None
        way = -np.sign(e)
        g1 = way * w1[n : 2 * n]
        d0, d1 = Aw @ w0, Aw @ w1
        best: tuple[float, int, float] | None = None
        for i in stops:
            t = first_root(way[i] * w0[n + i], way[i] * d0[n + i], g1[i], way[i] * d1[n + i], s)
            if best is None or t < best[0]:
                best = (t, int(i), 0.0)
        for i, p in zip(frees, p1):
            sg = float(np.sign(p))
            p0 = P[i] @ w0[:m]
            t = first_root(F[i] - sg * p0, -sg * (P[i] @ d0[:m]), F[i] - sg * p, -sg * (P[i] @ d1[:m]), s)
            if best is None or t < best[0]:
                best = (t, int(i), sg)
        return best

    def _stored_w(self, w: np.ndarray) -> float:
        n = self.system.n
        ground = w[2 * n] if self.force.settings.base else 0.0
        return stored_energy(self.system, w[:n] - ground, w[n : 2 * n])

    def _friction_step(self, w: np.ndarray, h: float, inp: tuple, P: np.ndarray, ledger: bool = True) -> np.ndarray:
        """Step w = [x, v, u, du, e] by h in place; return the step's (dissipated, work, friction loss).

        Without the ledger the energies are left at 0, which saves the quadratic
        integrals over each part of a step cut by an event.
        """
        n = self.system.n
        key, b, b_du = inp
        totals = np.zeros(3)
        left, forced = h, {}  # time left in the step, and a mass that just broke free
        tries = EVENTS_PER_MASS * n + 1
        for attempt in range(tries):
            stuck, e = self._slip(w, P, forced)
            forced = {}
            w[2 * n + 2 :] = e
            Aw = self._friction_system(key, b, b_du, stuck)
            Phi, Wd, Ww = self._friction_step_maps(key, Aw, stuck, left, h, ledger)
            w1 = Phi @ w
            event = self._first_event(w, w1, Aw, P, stuck, e, left) if attempt < tries - 1 else None
            s = left
            if event is not None and event[0] < left:
                s = event[0]
                Phi, Wd, Ww = self._friction_step_maps(key, Aw, stuck, s, h, ledger)
                w1 = Phi @ w
            if ledger:
                totals += (w @ Wd @ w, w @ Ww @ w, -e @ (w1[:n] - w[:n]))
            # Hold the stuck masses exactly, and stop the one that came to rest; the
            # (tiny) energy this takes out is friction's.
            if event is not None or stuck.any():
                st = np.flatnonzero(stuck)
                before = self._stored_w(w1) if ledger else 0.0
                w1[st], w1[n + st] = w[st], 0.0
                if event is not None:
                    if event[2] == 0.0:
                        w1[n + event[1]] = 0.0
                    else:
                        forced[event[1]] = event[2]
                if ledger:
                    totals[2] += before - self._stored_w(w1)
            w[:] = w1
            left -= s
            if event is None or left <= 0.0:
                break
        return totals

    def _advance_friction(self, h: float, steps: int) -> tuple[np.ndarray, ...]:
        """(t, z, input, per-step (dissipated, work, friction loss)) for `steps` steps with friction."""
        n = self.system.n
        inp = self._input()
        P = self._pull_rows(*inp)
        ts = np.empty(steps)
        zs = np.empty((steps, 2 * n))
        fs = np.empty(steps)
        losses = np.empty((steps, 3))
        w = np.zeros(3 * n + 2)
        w[: 2 * n] = self.state
        f0 = self._u
        for k in range(steps):
            self.force.advance(h)
            f1 = self.force.value()
            w[2 * n], w[2 * n + 1] = f0, (f1 - f0) / h
            losses[k] = self._friction_step(w, h, inp, P)
            self.t += h
            ts[k] = self.t
            zs[k] = w[: 2 * n]
            fs[k] = f1
            f0 = f1
        return ts, zs, fs, losses

    def drive(self, f: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
        """Step h at a time through given force samples, with friction; for the virtual modal test.

        f[k] (N, on the force's target mass) is the force at the end of step k;
        the first step starts from the force where the last call ended. The
        force controller, the clock and the energy ledger are left alone.
        Returns the state z = [x, v] and the accelerations after each step,
        shapes (len(f), 2n) and (len(f), n).
        """
        n = self.system.n
        f = np.asarray(f, dtype=float)
        inp = self._input()
        P = self._pull_rows(*inp)
        zs = np.empty((f.size, 2 * n))
        w = np.zeros(3 * n + 2)
        w[: 2 * n] = self.state
        F = self.system.friction
        f0 = self._u
        k = 0
        while k < f.size:
            f1 = f[k]
            if f1 == f0 and not w[n : 2 * n].any():
                w[2 * n], w[2 * n + 1] = f0, 0.0
                if np.all(np.abs(P @ w[: 2 * n + 2]) <= F):
                    # At rest, every mass held by its friction (or with no pull on it),
                    # and the force not changing: nothing moves until it does.
                    changes = np.flatnonzero(f[k:] != f0)
                    stop = f.size if not changes.size else k + int(changes[0])
                    zs[k:stop] = w[: 2 * n]
                    k = stop
                    continue
            w[2 * n], w[2 * n + 1] = f0, (f1 - f0) / h
            self._friction_step(w, h, inp, P, ledger=False)
            zs[k] = w[: 2 * n]
            f0 = f1
            k += 1
        if f.size:
            self.state = zs[-1].copy()
            self._u = float(f[-1])
        # The acceleration at each sample: none while stuck, else (pull - friction) / m.
        v = zs[:, n:]
        pull = np.column_stack([zs, f, np.zeros(f.size)]) @ P.T  # a force has no slope term (b_du = 0)
        way = np.where(v != 0.0, np.sign(v), np.sign(pull))
        stuck = (v == 0.0) & (F > 0) & (np.abs(pull) <= F)
        return zs, np.where(stuck, 0.0, (pull - F * way) / self.system.masses)

    def advance(self, duration: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Advance by approximately `duration` seconds of simulated time.

        Returns per-step samples (t, x, v, force) with shapes (k,), (k, n),
        (k, n), (k,), which the GUI appends to its history buffers; force is
        the ground displacement (m) with base excitation. The energy ledger at
        each of those samples is left in ``self.ledger``, shape (k, 4):
        columns energy_added, work, dissipated, friction_loss, and the ground's
        velocity over each step in ``self.ground_velocity``.
        """
        h = self.step_size()
        steps = min(MAX_STEPS_PER_ADVANCE, max(0, int(round(duration / h))))
        n = self.system.n
        ts = np.empty(steps)
        zs = np.empty((steps, 2 * n))
        fs = np.empty(steps)
        self.ledger = np.empty((steps, 4))
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

        z_start, f_start = self.state.copy(), self._u
        if self.has_friction:
            ts, zs, fs, losses = self._advance_friction(h, steps)
        else:
            Phi, g0, g1 = self._discrete(h)
            z = self.state
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
        self.state = zs[-1].copy()
        self._u = float(fs[-1])

        # Energy ledger: each step's integrals are quadratic in w0 = [z0, f0, (f1 - f0) / h].
        z0s = np.vstack([z_start, zs[:-1]])
        f0s = np.concatenate([[f_start], fs[:-1]])
        w = np.column_stack([z0s, f0s, (fs - f0s) / h])
        if not self.has_friction:
            Wd, Ww = self._energy_forms(h)
            losses = np.column_stack(
                [np.einsum("ki,ij,kj->k", w, Wd, w), np.einsum("ki,ij,kj->k", w, Ww, w), np.zeros(steps)]
            )
        dissipated = self.dissipated + np.cumsum(losses[:, 0])
        work = self.work + np.cumsum(losses[:, 1])
        friction = self.friction_loss + np.cumsum(losses[:, 2])
        self.dissipated, self.work, self.friction_loss = float(dissipated[-1]), float(work[-1]), float(friction[-1])
        self.ledger = np.column_stack([np.full(steps, self.energy_added), work, dissipated, friction])
        if base:
            self.ground_velocity = w[:, -1]
        return ts, zs[:, :n], zs[:, n:], fs
