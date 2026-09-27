"""Time response of a reduced (CMS) model beside the full chain, under the same force.

Both models are stepped with the exact first-order-hold update the simulator
uses, sample for sample, with one force F(t) on one mass (the tip by default).
The reduced model's physical displacements are recovered as x = T r, so any
difference between the two is the reduction error, not a numerical one.

The reduced model, with coordinates r and force vector g = T^T e_dof, is

    M^ r'' + C^ r' + K^ r = g F

MacNeal's boundary coordinates have no mass (M^ has zero rows). They are
condensed statically, as for its modes: with the massed coordinates m and the
massless ones z,

    r = S m + d F,   S = [I; -K_zz^-1 K_zm],   d = [0; K_zz^-1 g_z]

and projecting on S (S^T K d = 0) leaves

    S^T M^ S m'' + S^T C^ S m' + S^T K^ S m = S^T g F - S^T C^ d F'

This drops the damping in the massless rows' own balance (it treats
C_zz z' as small next to K_zz z), which is exact without damping.
"""

from __future__ import annotations

import math

import numpy as np

from .forcing import ForceController, ForceKind
from .model import state_space
from .simulator import MAX_STEP, MAX_STEPS_PER_ADVANCE, STEPS_PER_PERIOD, foh_input_step
from .substructure import CMSModel


def second_order_state_space(M: np.ndarray, C: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(A, Minv): z' = A z + [0; Minv] u for z = [x, v], with a full (non-diagonal) M."""
    n = M.shape[0]
    Minv = np.linalg.inv(M)
    A = np.zeros((2 * n, 2 * n))
    A[:n, n:] = np.eye(n)
    A[n:, :n] = -Minv @ K
    A[n:, n:] = -Minv @ C
    return A, Minv


def condensation(model: CMSModel, dof: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(S, d, g): the reduced coordinates are r = S m + d F, with g = T^T e_dof the force vector."""
    e = np.zeros(model.system.n)
    e[dof] = 1.0
    g = model.T.T @ e
    M, K = model.M, model.K
    n_red = M.shape[0]
    z = np.all(M == 0.0, axis=1)
    m = ~z
    S = np.zeros((n_red, int(m.sum())))
    S[m] = np.eye(int(m.sum()))
    d = np.zeros(n_red)
    if z.any():
        S[z] = -np.linalg.solve(K[np.ix_(z, z)], K[np.ix_(z, m)])
        d[z] = np.linalg.solve(K[np.ix_(z, z)], g[z])
    return S, d, g


def condensed(model: CMSModel, dof: int) -> tuple[np.ndarray, ...]:
    """(M, C, K, g, g_du, X, x_f) of the reduced model with massless coordinates condensed out.

    It obeys M m'' + C m' + K m = g F + g_du F', and the physical displacements
    are x = X m + x_f F.
    """
    S, d, g = condensation(model, dof)
    M, C, K = model.M, model.C, model.K
    return (S.T @ M @ S, S.T @ C @ S, S.T @ K @ S, S.T @ g, -S.T @ C @ d, model.T @ S, model.T @ d)


class _Stepper:
    """One linear model z' = A z + b F + b_du F' stepped exactly for F linear across each step."""

    def __init__(self, A: np.ndarray, b: np.ndarray, b_du: np.ndarray | None) -> None:
        self.A, self.b, self.b_du = A, b, b_du
        self.z = np.zeros(A.shape[0])
        self._cache: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        lam = np.linalg.eigvals(A)
        self.fmax = float(np.abs(lam).max()) / (2 * math.pi) if lam.size else 0.0

    def discrete(self, h: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if h not in self._cache:
            self._cache[h] = foh_input_step(self.A, self.b, self.b_du, h)
        return self._cache[h]


class CMSResponse:
    """The full chain and a reduced model, stepped side by side under one force at mass `dof`."""

    def __init__(self, model: CMSModel, force: ForceController | None = None, dof: int | None = None) -> None:
        self.model = model
        self.force = force or ForceController()
        n = model.system.n
        self.dof = n - 1 if dof is None else dof
        A, B = state_space(model.system)
        self._full = _Stepper(A, B[:, self.dof], None)
        M, C, K, g, g_du, self._X, self._xf = condensed(model, self.dof)
        Ar, Minv = second_order_state_space(M, C, K)
        nm = M.shape[0]
        b = np.concatenate([np.zeros(nm), Minv @ g])
        b_du = np.concatenate([np.zeros(nm), Minv @ g_du]) if np.any(g_du) else None
        self._red = _Stepper(Ar, b, b_du)
        self.t = 0.0
        self._u = 0.0  # the force where the last step ended

    def reset(self) -> None:
        self.t = 0.0
        self._u = 0.0
        self.force.reset()
        self._full.z[:] = 0.0
        self._red.z[:] = 0.0

    @property
    def x_full(self) -> np.ndarray:
        return self._full.z[: self.model.system.n]

    @property
    def x_reduced(self) -> np.ndarray:
        """The reduced model's motion in physical coordinates, x = T r."""
        return self._X @ self._red.z[: self._X.shape[1]] + self._xf * self._u

    def step_size(self) -> float:
        """As Simulator.step_size: resolve the fastest mode of either model, the forcing and a pulse."""
        s = self.force.settings
        fmax = max(self._full.fmax, self._red.fmax, s.max_freq_hz)
        h = MAX_STEP if fmax <= 0 else min(MAX_STEP, 1.0 / (STEPS_PER_PERIOD * fmax))
        if s.kind is ForceKind.PULSE:
            h = min(h, s.pulse_duration / 10.0)
        return 2.0 ** math.floor(math.log2(h))

    def advance(self, duration: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Advance by about `duration` s; per-step samples (t, x_full, x_reduced, F), shapes (k,), (k, n), (k, n), (k,)."""
        ts, fs, zf, zr = self._run(duration)
        n = self.model.system.n
        nm = self._X.shape[1]
        return ts, zf[:, :n], zr[:, :nm] @ self._X.T + np.outer(fs, self._xf), fs

    def _run(self, duration: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Step both models; per-step (t, F, full state, reduced state)."""
        h = self.step_size()
        steps = min(MAX_STEPS_PER_ADVANCE, max(0, int(round(duration / h))))
        ts, fs = np.empty(steps), np.empty(steps)
        zf = np.empty((steps, self._full.z.size))
        zr = np.empty((steps, self._red.z.size))
        if steps == 0:
            return ts, fs, zf, zr
        Pf, f0f, f1f = self._full.discrete(h)
        Pr, f0r, f1r = self._red.discrete(h)
        a, b = self._full.z, self._red.z
        u0 = self._u
        for k in range(steps):
            self.force.advance(h)
            u1 = self.force.value()
            a = Pf @ a + f0f * u0 + f1f * u1
            b = Pr @ b + f0r * u0 + f1r * u1
            self.t += h
            ts[k], fs[k], zf[k], zr[k] = self.t, u1, a, b
            u0 = u1
        self._full.z, self._red.z, self._u = a, b, u0
        return ts, fs, zf, zr
