"""Back expansion (data recovery): the interior motion of each substructure after the coupled solve.

The coupled (system-level) analysis only solves for the reduced coordinates
r = [q_A, q_B, ..., x_b]: the kept component modes and the boundary DOFs. The
interior DOFs are not in it. They are recovered afterwards, substructure by
substructure, from that substructure's own rows of T:

    x_i = T_ib x_b + T_iq q

For Craig-Bampton T_ib = Psi (the constraint modes: the static shape the
boundary drags the interior into) and T_iq = Phi_k (the kept fixed-interface
modes: the interior's own vibration relative to that). For the free-interface
methods T_ib = R and T_iq = Phi_ik - R Phi_bk. This is the recovery the
reduced model implies, and for Craig-Bampton and Rubin with every mode kept
it is exact.

Two other recoveries use less, or more, than the coupled solve provides:

* Boundary only (static, Guyan): x_i = -K_ii^-1 K_ib x_b, from the boundary
  motion alone, as when the modal coordinates were not kept from the system
  run (or the substructure was Guyan-reduced). For Craig-Bampton it is the
  Psi x_b part of the recovery above; what it misses is the q part.
* Enhanced: re-solve the substructure's interior on its own with the boundary
  motion x_b(t) of the coupled solve imposed, keeping *every* interior DOF
  (all its fixed-interface modes, not just the kept ones):

      M_ii x_i'' + C_ii x_i' + K_ii x_i = -K_ib x_b - C_ib x_b'

  (M_ib = 0: the chain's masses are lumped). Its only error is then the error
  in x_b, so it recovers the dynamics of the modes the reduction truncated.
  These interior states are stepped together with the reduced model, in one
  exact first-order-hold update: x_b is a linear function of the reduced state
  (and, for MacNeal, of F), so the whole thing is one linear system.

Recovered loads are the spring forces k_e (x_e - x_{e-1}), from each recovered
displacement (a displacement-based loads transformation matrix).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cms_response import CMSResponse, _Stepper, condensation
from .forcing import ForceController
from .substructure import CMSModel

RECOVERIES = ("coupled", "boundary", "enhanced")
RECOVERY_NAMES = {
    "coupled": "Coupled solution (T r)",
    "boundary": "Boundary only (static)",
    "enhanced": "Enhanced (re-solved)",
}


class Recovery:
    """The recovery matrices of a reduced model, all in global (chain) DOF order.

    Each method takes the reduced coordinates r (shape (k, n_red), rows are time
    samples) and returns physical displacements (k, n). Boundary rows are x_b
    itself in every recovery; they differ only in the interior.
    """

    def __init__(self, model: CMSModel) -> None:
        self.model = model
        n = model.system.n
        self.nq = model.n_modal
        self.boundary = model.boundary
        self.interior = np.setdiff1d(np.arange(n), model.boundary)
        self.T_q = model.T[:, : self.nq]  # interior columns: Phi_k (CB) or Phi_ik - R Phi_bk
        self.T_b = model.T[:, self.nq :]  # Psi (CB) or R, and the identity on the boundary rows
        K = model.system.matrices()[2]
        i, b = self.interior, self.boundary
        self.guyan: np.ndarray | None = self.T_b.copy()
        if i.size:
            Kii = K[np.ix_(i, i)]
            if np.linalg.cond(Kii) > 1e12:
                self.guyan = None  # an interior mass joined to no boundary by a spring: no static shape
            else:
                self.guyan[i] = -np.linalg.solve(Kii, K[np.ix_(i, b)])

    def owner(self, dof: int) -> int | None:
        """Index of the substructure whose interior this DOF is, None for a boundary DOF."""
        for s, sub in enumerate(self.model.substructures):
            if dof in sub.interior:
                return s
        return None

    def coupled(self, r: np.ndarray) -> np.ndarray:
        return r @ self.model.T.T

    def from_boundary(self, r: np.ndarray) -> np.ndarray:
        """T_b x_b: the part of the coupled recovery that the boundary motion drags along."""
        return r[:, self.nq :] @ self.T_b.T

    def from_modes(self, r: np.ndarray) -> np.ndarray:
        """T_q q: the part that the kept component modes add (zero on the boundary)."""
        return r[:, : self.nq] @ self.T_q.T

    def boundary_only(self, r: np.ndarray) -> np.ndarray | None:
        """-K_ii^-1 K_ib x_b in the interior; None when K_ii is singular."""
        return None if self.guyan is None else r[:, self.nq :] @ self.guyan.T

    def spring_forces(self, x: np.ndarray) -> np.ndarray:
        """k_e (x_e - x_{e-1}) for every spring, from displacements (k, n)."""
        return self.model.system.stiffness * np.diff(x, prepend=0.0, axis=-1)


@dataclass(frozen=True)
class RecoverySamples:
    t: np.ndarray  # (k,)
    force: np.ndarray  # (k,)
    x_full: np.ndarray  # (k, n): the full chain, the truth
    r: np.ndarray  # (k, n_red): the coupled solve's reduced coordinates [q, x_b]
    x_enhanced: np.ndarray  # (k, n): boundary from r, interior re-solved with it imposed


class RecoveryResponse(CMSResponse):
    """CMSResponse that also returns the reduced coordinates and the enhanced interior recovery."""

    def __init__(self, model: CMSModel, force: ForceController | None = None, dof: int | None = None) -> None:
        super().__init__(model, force, dof)
        self.recovery = Recovery(model)
        system = model.system
        S, d, _ = condensation(model, self.dof)
        self._S, self._d = S, d
        X, xf = self._X, self._xf
        i, b = self.recovery.interior, self.recovery.boundary
        ni, nm = i.size, X.shape[1]
        self._ni = ni
        red = self._red
        ny = red.A.shape[0]  # [m, m']
        _, C, K = system.matrices()
        minv = 1.0 / system.masses[i]
        A = np.zeros((ny + 2 * ni, ny + 2 * ni))
        A[:ny, :ny] = red.A
        w, v = slice(ny, ny + ni), slice(ny + ni, ny + 2 * ni)
        A[w, v] = np.eye(ni)
        A[v, w] = -minv[:, None] * K[np.ix_(i, i)]
        A[v, v] = -minv[:, None] * C[np.ix_(i, i)]
        # The imposed boundary motion: x_b = X_b m + xf_b F and x_b' = X_b m' + xf_b F'.
        Kib, Cib = minv[:, None] * K[np.ix_(i, b)], minv[:, None] * C[np.ix_(i, b)]
        A[v, :nm] = -Kib @ X[b]
        A[v, nm:ny] = -Cib @ X[b]
        bvec = np.concatenate([red.b, np.zeros(ni), -Kib @ xf[b]])
        if self.dof in i:
            bvec[ny + ni + int(np.searchsorted(i, self.dof))] += minv[np.searchsorted(i, self.dof)]
        b_du = np.concatenate([red.b_du if red.b_du is not None else np.zeros(ny), np.zeros(ni), -Cib @ xf[b]])
        self._red = _Stepper(A, bvec, b_du if np.any(b_du) else None)

    @property
    def x_enhanced(self) -> np.ndarray:
        x = self.x_reduced.copy()
        ny = self._red.A.shape[0] - 2 * self._ni
        x[self.recovery.interior] = self._red.z[ny : ny + self._ni]
        return x

    @property
    def r(self) -> np.ndarray:
        """The reduced coordinates now, r = S m + d F."""
        return self._S @ self._red.z[: self._S.shape[1]] + self._d * self._u

    def record(self, duration: float) -> RecoverySamples:
        """Advance by about `duration` s, returning every step's samples."""
        ts, fs, zf, zr = self._run(duration)
        n = self.model.system.n
        nm = self._S.shape[1]
        r = zr[:, :nm] @ self._S.T + np.outer(fs, self._d)
        x_enh = r @ self.model.T.T
        ny = 2 * nm
        x_enh[:, self.recovery.interior] = zr[:, ny : ny + self._ni]
        return RecoverySamples(ts, fs, zf[:, :n], r, x_enh)
