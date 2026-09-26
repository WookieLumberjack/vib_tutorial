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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg

from .model import ChainSystem, state_space

TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class DampedPole:
    """One oscillatory eigenvalue pair of the damped system."""

    eigenvalue: complex  # the member of the pair with positive imaginary part
    omega_n: float  # rad/s, |lambda|
    zeta: float  # -Re(lambda) / |lambda|
    omega_d: float  # rad/s, Im(lambda)
    shape: np.ndarray  # complex displacement shape, largest entry normalized to 1+0j

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
    damped: DampedPole | None  # None if this mode is overdamped

    @property
    def fn_hz(self) -> float:
        return self.omega_n / TWO_PI


@dataclass(frozen=True)
class ModalResult:
    modes: list[Mode]
    overdamped_roots: list[float]  # real eigenvalues (1/s) of non-oscillatory motion
    coupling: float  # max normalized off-diagonal of Phi^T C Phi; 0 => proportional

    @property
    def is_proportional(self) -> bool:
        return self.coupling < 1e-6


def _normalize_real(v: np.ndarray) -> np.ndarray:
    return v / v[np.argmax(np.abs(v))]


def _normalize_complex(v: np.ndarray) -> np.ndarray:
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

    # --- Exact damped eigenvalues from the state matrix.
    A, _ = state_space(system)
    lam, vecs = scipy.linalg.eig(A)
    n = system.n
    scale = max(1.0, float(np.abs(lam).max()))
    poles: list[DampedPole] = []
    overdamped: list[float] = []
    for i, li in enumerate(lam):
        if li.imag > 1e-9 * scale:
            wn = abs(li)
            poles.append(
                DampedPole(
                    eigenvalue=complex(li),
                    omega_n=wn,
                    zeta=-li.real / wn,
                    omega_d=li.imag,
                    shape=_normalize_complex(vecs[:n, i]),
                )
            )
        elif abs(li.imag) <= 1e-9 * scale:
            overdamped.append(float(li.real))
    poles.sort(key=lambda p: p.omega_n)
    overdamped.sort(reverse=True)

    # Pair each damped pole with the undamped mode whose shape it most resembles
    # (modal assurance criterion), so a heavily damped mode that drops out does
    # not shift the pairing of the others.
    matched: dict[int, DampedPole] = {}
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
    return ModalResult(modes=modes, overdamped_roots=overdamped, coupling=coupling)


def _mac(real_shape: np.ndarray, complex_shape: np.ndarray) -> float:
    num = abs(np.vdot(real_shape, complex_shape)) ** 2
    den = np.vdot(real_shape, real_shape).real * np.vdot(complex_shape, complex_shape).real
    return float(num / den) if den > 0 else 0.0


def frf(system: ChainSystem, freqs_hz: np.ndarray, input_dof: int) -> np.ndarray:
    """Receptance H(w) = X/F for a force at `input_dof`, shape (len(freqs), n).

    H(w) = (K - w^2 M + i w C)^-1 e_input
    """
    M, C, K = system.matrices()
    w = TWO_PI * np.asarray(freqs_hz, dtype=float)
    Z = K[None, :, :] - (w**2)[:, None, None] * M[None] + 1j * w[:, None, None] * C[None]
    rhs = np.zeros((w.size, system.n, 1), dtype=complex)
    rhs[:, input_dof, 0] = 1.0
    return np.linalg.solve(Z, rhs)[:, :, 0]
