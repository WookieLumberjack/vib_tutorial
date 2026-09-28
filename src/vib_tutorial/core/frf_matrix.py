"""The full receptance (compliance) matrix H(w) and its modal decomposition.

H(w) = (K - w^2 M + i w C)^-1 is an N x N matrix at every frequency: H_jk is
the displacement of mass j per unit force at mass k. It is symmetric
(H_jk = H_kj, Maxwell-Betti reciprocity) because M, C and K are.

H is also a sum of one term per mode. Two expansions are offered:

* Exact (state-space). With A = V diag(lambda) V^-1 and the output/input
  maps of the first-order form, (iw - A)^-1 = sum_r v_r w_r^T / (iw - lambda_r),
  so each eigenvalue contributes a residue matrix R_r / (iw - lambda_r). An
  oscillatory pair (lambda, lambda*) is kept together as one term; each real
  (overdamped) root is its own term. Summing every term gives H exactly, for
  any damping.

* Classical (real modes). With mass-normalized undamped shapes phi_r,
  H ~ sum_r phi_r phi_r^T / (w_r^2 - w^2 + i w c_r), c_r = phi_r^T C phi_r.
  This drops the off-diagonal terms of Phi^T C Phi, so it is exact only for
  proportional damping.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .modal import TWO_PI, ModalResult
from .model import ChainSystem

# V is effectively singular when an eigenvalue is defective (a free chain's
# repeated lambda = 0, or two coincident overdamped roots). Rounding splits such
# a pair by ~sqrt(eps), leaving cond(V) as low as ~3e7 on some CPUs (seen on a
# Windows CI runner); real chains stay below ~1e5, even within 1e-8 of critical
# damping. A chain past the limit falls back safely (classical terms, direct stepping).
MAX_EIGVEC_CONDITION = 1e6


@dataclass(frozen=True)
class ModalTerm:
    """One mode's contribution to H(w): an N x N matrix at every frequency."""

    label: str  # e.g. "Mode 2" or "Real root λ = -3.1"
    mode: int | None  # 1-based undamped mode number it belongs to, if any
    fn_hz: float  # undamped natural frequency; 0 for a non-oscillatory root
    zeta: float  # damping ratio (exact or modal, matching the expansion); NaN if none
    residues: np.ndarray  # (P, n, n): one residue per pole, or phi phi^T for a real mode
    poles: np.ndarray  # (P,) state-space eigenvalues; empty for a classical real mode
    stiffness: float = 0.0  # classical only: w_r^2
    damping: float = 0.0  # classical only: c_r = phi_r^T C phi_r

    def evaluate(self, freqs_hz: np.ndarray) -> np.ndarray:
        """This term's H(w), shape (len(freqs), n, n)."""
        w = TWO_PI * np.asarray(freqs_hz, dtype=float)
        if self.poles.size:
            den = 1j * w[:, None] - self.poles[None, :]  # (F, P)
            return np.einsum("fp,pjk->fjk", 1.0 / den, self.residues)
        den = self.stiffness - w**2 + 1j * w * self.damping
        return self.residues[0][None, :, :] / den[:, None, None]


def frf_matrix(system: ChainSystem, freqs_hz: np.ndarray) -> np.ndarray:
    """Receptance matrix H(w) = (K - w^2 M + i w C)^-1, shape (len(freqs), n, n)."""
    M, C, K = system.matrices()
    w = TWO_PI * np.asarray(freqs_hz, dtype=float)
    Z = K[None, :, :] - (w**2)[:, None, None] * M[None] + 1j * w[:, None, None] * C[None]
    return np.linalg.inv(Z)


def modal_frf_terms(system: ChainSystem, result: ModalResult, exact: bool = True) -> list[ModalTerm]:
    """H(w) split into one term per mode, lowest first.

    `exact` selects the state-space expansion (raises ValueError if an
    eigenvalue is defective, so no modal expansion exists); otherwise the
    classical real-mode one.
    """
    return _exact_terms(system, result) if exact else _classical_terms(system, result)


def _classical_terms(system: ChainSystem, result: ModalResult) -> list[ModalTerm]:
    _, C, _ = system.matrices()
    terms = []
    for m in result.modes:
        phi = m.shape_mass_normalized
        c = float(phi @ C @ phi)
        terms.append(
            ModalTerm(
                label=f"Mode {m.index}",
                mode=m.index,
                fn_hz=m.fn_hz,
                zeta=m.zeta_modal,
                residues=np.outer(phi, phi)[None].astype(complex),
                poles=np.zeros(0, dtype=complex),
                stiffness=m.omega_n**2,
                damping=c,
            )
        )
    return terms


def _exact_terms(system: ChainSystem, result: ModalResult) -> list[ModalTerm]:
    n = system.n
    cms = result.complex_modes
    V = np.column_stack([m.state_vector for m in cms])
    if np.linalg.cond(V) > MAX_EIGVEC_CONDITION:
        raise ValueError("a repeated (defective) eigenvalue has no modal expansion")
    # Residue of pole r: (displacement part of v_r) x (row r of V^-1 B), B = [0; M^-1].
    W = np.linalg.inv(V)[:, n:] / system.masses[None, :]
    residues = np.einsum("jr,rk->rjk", V[:n], W)
    mode_of = {m.damped.index: m for m in result.modes if m.damped is not None}

    terms = []
    for r, cm in enumerate(cms):
        if cm.is_oscillatory and cm.eigenvalue.imag < 0:
            continue  # added together with its partner
        idx = [r] if not cm.is_oscillatory else [r, cm.conjugate - 1]
        mode = mode_of.get(cm.index)
        if mode is not None:
            label, number, fn = f"Mode {mode.index}", mode.index, cm.fn_hz
        elif cm.is_oscillatory:
            label, number, fn = f"λ{cm.index}, λ{cm.conjugate}", None, cm.fn_hz
        else:
            label, number, fn = f"Real root λ = {cm.eigenvalue.real:.3g}", None, 0.0
        terms.append(
            ModalTerm(
                label=label,
                mode=number,
                fn_hz=fn,
                zeta=cm.zeta if cm.is_oscillatory else float("nan"),
                residues=residues[idx],
                poles=np.array([cms[i].eigenvalue for i in idx]),
            )
        )
    # Modes in order, then the non-oscillatory roots, slowest first.
    terms.sort(key=lambda t: (t.mode is None, t.mode or 0, -max(t.poles.real)))
    return terms
