"""Modal analysis of the chain: undamped normal modes and exact damped eigenvalues.

Two complementary views are computed:

1. Classical (undamped) modal analysis -- solve K phi = w^2 M phi. The mass-
   normalized mode shapes diagonalize M and K. Projecting C onto them gives
   the *modal damping ratio* zeta_r = phi_r^T C phi_r / (2 w_r). This is exact
   only when damping is proportional (Phi^T C Phi is diagonal); otherwise it
   is the usual engineering approximation that ignores modal coupling.

2. Exact damped (state-space) eigenvalues -- the eigenvalues of the 2N x 2N
   state matrix come in pairs lambda = -zeta w_n +/- i w_n sqrt(1 - zeta^2).
   These are exact for any damping, and the corresponding mode shapes are in
   general complex (masses do not all pass through zero at the same instant).
   All 2N eigenpairs, conjugates and real (overdamped) roots included, are
   kept in ``ModalResult.complex_modes``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg

from .model import ChainSystem, state_space

TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class ComplexMode:
    """One eigenpair (lambda, psi) of the 2N x 2N state matrix.

    Oscillatory eigenvalues come in conjugate pairs; both members are kept so
    the full set of 2N solutions can be shown. A real eigenvalue is a
    non-oscillatory (overdamped, or rigid-body when lambda = 0) solution.
    """

    index: int  # 1-based position among all 2N eigenvalues
    eigenvalue: complex
    shape: np.ndarray  # complex displacement part psi, largest entry normalized to 1+0j
    state_vector: np.ndarray  # full eigenvector [psi, lambda psi], same normalization
    conjugate: int | None  # index of the conjugate partner; None for a real eigenvalue

    @property
    def is_oscillatory(self) -> bool:
        return self.conjugate is not None

    @property
    def omega_n(self) -> float:
        """rad/s, |lambda|."""
        return abs(self.eigenvalue)

    @property
    def zeta(self) -> float:
        """-Re(lambda) / |lambda|."""
        wn = self.omega_n
        return -self.eigenvalue.real / wn if wn > 0 else float("nan")

    @property
    def omega_d(self) -> float:
        """rad/s, |Im(lambda)| (the same for both members of a pair)."""
        return abs(self.eigenvalue.imag)

    @property
    def fn_hz(self) -> float:
        return self.omega_n / TWO_PI

    @property
    def fd_hz(self) -> float:
        return self.omega_d / TWO_PI


@dataclass(frozen=True)
class Mode:
    """Undamped normal mode r, with its matching damped pole if one exists."""

    index: int  # 1-based mode number
    omega_n: float  # rad/s, undamped natural frequency
    shape: np.ndarray  # real shape, largest |entry| normalized to +1
    shape_mass_normalized: np.ndarray  # phi with phi^T M phi = 1
    zeta_modal: float  # phi^T C phi / (2 w); NaN for a rigid-body mode
    damped: ComplexMode | None  # matching eigenvalue with Im > 0; None if overdamped

    @property
    def fn_hz(self) -> float:
        return self.omega_n / TWO_PI


@dataclass(frozen=True)
class ModalResult:
    modes: list[Mode]  # N undamped normal modes
    complex_modes: list[ComplexMode]  # all 2N state-space eigenpairs, ordered by |lambda|
    coupling: float  # max normalized off-diagonal of Phi^T C Phi; 0 => proportional

    @property
    def overdamped_roots(self) -> list[float]:
        """Real eigenvalues (1/s) of non-oscillatory motion, slowest first."""
        return sorted((m.eigenvalue.real for m in self.complex_modes if not m.is_oscillatory), reverse=True)

    @property
    def is_proportional(self) -> bool:
        return self.coupling < 1e-6


def _normalize_real(v: np.ndarray) -> np.ndarray:
    return v / v[np.argmax(np.abs(v))]


def modal_analysis(system: ChainSystem) -> ModalResult:
    M, C, K = system.matrices()

    # --- Undamped normal modes. eigh(K, M) returns M-orthonormal eigenvectors.
    w2, Phi = scipy.linalg.eigh(K, M)
    omegas = np.sqrt(np.clip(w2, 0.0, None))
    Cm = Phi.T @ C @ Phi
    d = np.sqrt(np.abs(np.outer(np.diag(Cm), np.diag(Cm))))
    off = np.abs(Cm - np.diag(np.diag(Cm)))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(d > 0, off / d, 0.0)
    coupling = float(ratios.max()) if ratios.size else 0.0

    # --- Exact damped eigenpairs of the state matrix.
    complex_modes = _state_space_modes(system)
    poles = [m for m in complex_modes if m.is_oscillatory and m.eigenvalue.imag > 0]
    n = system.n

    # Pair each damped pole with the undamped mode whose shape it most resembles
    # (modal assurance criterion), so a heavily damped mode that drops out does
    # not shift the pairing of the others.
    matched: dict[int, ComplexMode] = {}
    unmatched = set(range(n))
    for pole in poles:
        if not unmatched:
            break
        best = max(unmatched, key=lambda r: _mac(Phi[:, r], pole.shape) - 1e-3 * r)
        matched[best] = pole
        unmatched.discard(best)

    modes = []
    for r in range(n):
        phi = Phi[:, r]
        wn = float(omegas[r])
        zeta = float(Cm[r, r] / (2.0 * wn)) if wn > 1e-9 else float("nan")
        modes.append(
            Mode(
                index=r + 1,
                omega_n=wn,
                shape=_normalize_real(phi),
                shape_mass_normalized=phi.copy(),
                zeta_modal=zeta,
                damped=matched.get(r),
            )
        )
    return ModalResult(modes=modes, complex_modes=complex_modes, coupling=coupling)


def _state_space_modes(system: ChainSystem) -> list[ComplexMode]:
    """All 2N eigenpairs of A, each conjugate pair adjacent (Im > 0 member first)."""
    A, _ = state_space(system)
    lam, vecs = scipy.linalg.eig(A)
    n = system.n
    # A repeated (defective) eigenvalue such as the lambda = 0 of a free chain
    # splits into +/- ~sqrt(eps) i numerically, so "real" needs a loose tolerance.
    tol = 1e-6 * max(1.0, float(np.abs(lam).max()))
    upper = [i for i in range(lam.size) if lam[i].imag > tol]
    lower = {i for i in range(lam.size) if lam[i].imag < -tol}
    real = [i for i in range(lam.size) if abs(lam[i].imag) <= tol]

    # Each group is one oscillatory pair or one real root, ordered by |lambda|.
    groups: list[tuple[complex, np.ndarray, bool]] = []
    for i in upper:
        partner = min(lower, key=lambda j: abs(lam[j] - np.conj(lam[i])))
        lower.discard(partner)
        groups.append((complex(lam[i]), vecs[:, i], True))
    for i in real:
        groups.append((complex(lam[i].real), vecs[:, i].real.astype(complex), False))
    groups.sort(key=lambda g: abs(g[0]))

    out: list[ComplexMode] = []
    for li, v, oscillatory in groups:
        v = v / v[np.argmax(np.abs(v[:n]))]  # largest displacement entry -> 1+0j
        k = len(out) + 1
        if oscillatory:
            # Store the partner as the exact conjugate so the pair is consistent.
            out.append(ComplexMode(k, li, v[:n].copy(), v, conjugate=k + 1))
            vc = v.conj()
            out.append(ComplexMode(k + 1, li.conjugate(), vc[:n].copy(), vc, conjugate=k))
        else:
            out.append(ComplexMode(k, li, v[:n].copy(), v, conjugate=None))
    return out


def modal_coordinate_map(system: ChainSystem, result: ModalResult, complex_modes: bool = False) -> np.ndarray:
    """Real matrix P, shape (2N, m), mapping states to modal coordinates: y = z @ P.

    z holds states [x, v] as rows. Each coordinate is scaled to metres: it is
    the mode's contribution to the displacement of the mass the mode moves
    most (the mass where its normalized shape is 1).

    Classical (N columns, in mode order): with mass-normalized phi_r,
    q_r = phi_r^T M x and x = sum_r phi_r q_r = sum_r shape_r y_r, so
    y_r = phi_r[max] q_r. Velocities do not enter.

    Complex (one column per conjugate pair or real root, in the order of
    ``result.complex_modes`` without the second member of each pair): with the
    state eigenvectors V as columns, z = V eta, so eta = V^-1 z. A pair adds
    psi eta + conj(psi eta) = 2 Re(psi eta) to x, which is 2 Re(eta) at the
    mass where psi = 1; a real root adds eta there. Unlike the classical q,
    these coordinates decouple for any damping.
    """
    n = system.n
    if not complex_modes:
        M = system.matrices()[0]
        Phi = np.column_stack([m.shape_mass_normalized for m in result.modes])
        peak = Phi[np.argmax(np.abs(Phi), axis=0), np.arange(n)]
        P = np.zeros((2 * n, n))
        P[:n] = M @ Phi * peak
        return P

    V = np.column_stack([m.state_vector for m in result.complex_modes])
    # pinv, not inv: a defective eigenvalue (the lambda = 0 of a free chain)
    # makes V singular, and pinv still gives finite coordinates there.
    Vinv = np.linalg.pinv(V)
    cols = []
    for m in result.complex_modes:
        row = Vinv[m.index - 1]
        if m.conjugate is None:
            cols.append(row.real)
        elif m.conjugate > m.index:
            cols.append(2.0 * row.real)
    return np.column_stack(cols)


def _mac(real_shape: np.ndarray, complex_shape: np.ndarray) -> float:
    num = abs(np.vdot(real_shape, complex_shape)) ** 2
    den = np.vdot(real_shape, real_shape).real * np.vdot(complex_shape, complex_shape).real
    return float(num / den) if den > 0 else 0.0


def frf(system: ChainSystem, freqs_hz: np.ndarray, input_dof: int) -> np.ndarray:
    """Receptance H(w) = X/F for a force at `input_dof`, shape (len(freqs), n).

    H(w) = (K - w^2 M + i w C)^-1 e_input
    """
    e = np.zeros(system.n)
    e[input_dof] = 1.0
    return receptance(*system.matrices(), freqs_hz, e)


def transmissibility(system: ChainSystem, freqs_hz: np.ndarray) -> np.ndarray:
    """X_i / X_g for a harmonic ground motion x_g: shape (len(freqs), n), complex.

    The ground acts on mass 1 through spring and damper 1 only, as a force
    (k1 + i w c1) X_g, so this is receptance column 1 times that stiffness.
    """
    w = TWO_PI * np.asarray(freqs_hz, dtype=float)
    k1, c1 = system.stiffness[0], system.damping[0]
    return frf(system, freqs_hz, 0) * (k1 + 1j * w * c1)[:, None]


def receptance(M: np.ndarray, C: np.ndarray, K: np.ndarray, freqs_hz: np.ndarray, f: np.ndarray) -> np.ndarray:
    """(K - w^2 M + i w C)^-1 f at each frequency, shape (len(freqs), len(f))."""
    w = TWO_PI * np.asarray(freqs_hz, dtype=float)
    Z = K[None, :, :] - (w**2)[:, None, None] * M[None] + 1j * w[:, None, None] * C[None]
    rhs = np.broadcast_to(np.asarray(f, dtype=complex)[None, :, None], (w.size, f.size, 1))
    return np.linalg.solve(Z, rhs)[:, :, 0]
