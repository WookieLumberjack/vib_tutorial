"""Modal coupling of two subsystems, each reduced to one equivalent oscillator.

The chain is split after mass j into two subsystems, each analysed on its own::

    full:   ground --k1-- m1 --k2-- m2 --k3-- m3 --k4-- m4
    A:      ground --k1-- m1 --k2-- m2                         (tip m2 free)
    B:                           ground --k3-- m3 --k4-- m4    (k3, c3 to ground)

A keeps its own ground; B's first spring and damper (element j+1), which join
it to A, are tied to ground instead. Each has its own modes. One mode of each
is replaced by a single-DOF oscillator, and the two oscillators are joined the
way A and B are, B's on A's::

    ground --[k_a, c_a]-- m_a --[k_b, c_b]-- m_b

B is joined to A at its base, so the mass of B's mode that the base feels is
its *effective mass* along x (the participation of a base motion), with
mass-normalized phi and the influence vector 1 (every mass moves with the base):

    Gamma = phi^T M 1,   m_eff = Gamma^2,   sum over all modes of m_eff = total mass.

The rest of B's mass (its other modes, m_B - m_eff) moves rigidly with the base
below their frequencies: the *residual mass*, added to m_a. B's oscillator plus
its residual mass is Craig-Bampton for B with one fixed-interface mode kept.

A is joined to B at its free tip, so the mass of A's mode that the tip feels is
its *modal mass at the tip* 1/phi_tip^2 (the same as an absorber's primary). A's
effective mass along x, which is what a shaking ground would feel, can be chosen
instead to see the difference.

Each oscillator keeps its mode's frequency and modal damping ratio:
k = m omega^2 and c = m phi^T C phi (= 2 zeta omega m). With the tip modal mass
and the residual mass, the 2-DOF model is a Rayleigh-Ritz model of the chain
(A's mode shape, carried rigidly into B, and B's mode shape on a fixed base), so
its natural frequencies are upper bounds of the chain's.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg

from .modal import TWO_PI, ModalResult, receptance
from .model import ChainSystem, assemble_chain
from .substructure import damped_poles

A_MASSES = ("tip", "effective")  # how A's oscillator mass is chosen
A_MASS_NAMES = {"tip": "modal mass at the interface", "effective": "effective mass (base)"}
SWEEPS = ("frequency", "mass")


@dataclass(frozen=True)
class Subsystem:
    """One side of the split, grounded, with its own modes (mass-normalized)."""

    name: str  # "A" or "B"
    dofs: np.ndarray  # global 0-based DOFs, ascending
    M: np.ndarray
    C: np.ndarray
    K: np.ndarray
    omegas: np.ndarray  # rad/s, undamped
    Phi: np.ndarray  # columns: mass-normalized modes, signed so that Gamma >= 0

    @property
    def total_mass(self) -> float:
        return float(np.trace(self.M))

    @property
    def fn_hz(self) -> np.ndarray:
        return self.omegas / TWO_PI

    @property
    def participation(self) -> np.ndarray:
        """Gamma_r = phi_r^T M 1: how much a rigid motion of the base drives mode r."""
        return self.Phi.T @ self.M @ np.ones(self.dofs.size)

    @property
    def effective_masses(self) -> np.ndarray:
        """Gamma_r^2 (kg); they sum to the total mass."""
        return self.participation**2

    @property
    def modal_damping(self) -> np.ndarray:
        """phi_r^T C phi_r (1/s): 2 zeta_r omega_r for a mass-normalized mode."""
        return np.einsum("ir,ij,jr->r", self.Phi, self.C, self.Phi)

    @property
    def zetas(self) -> np.ndarray:
        """Modal damping ratio phi^T C phi / (2 omega); NaN for a rigid-body mode."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.omegas > 1e-9, self.modal_damping / (2.0 * self.omegas), np.nan)

    def tip_masses(self) -> np.ndarray:
        """1 / phi_tip^2 (kg): each mode's modal mass at the last DOF; inf where the tip does not move."""
        tip = self.Phi[-1]
        with np.errstate(divide="ignore"):
            return np.where(np.abs(tip) > 1e-12, 1.0 / tip**2, np.inf)


def subsystems(system: ChainSystem, split: int) -> tuple[Subsystem, Subsystem]:
    """A = masses 1..split on the ground; B = the rest, with its first element tied to ground."""
    n = system.n
    if not 1 <= split <= n - 1:
        raise ValueError("the split must leave at least one mass on each side")
    out = []
    for name, sl in (("A", slice(0, split)), ("B", slice(split, n))):
        m, c, k = system.masses[sl], system.damping[sl], system.stiffness[sl]
        M, C, K = np.diag(m), assemble_chain(c), assemble_chain(k)
        w2, Phi = scipy.linalg.eigh(K, M)
        gamma = Phi.T @ M @ np.ones(m.size)
        # Sign: Gamma positive; if it is 0, the largest entry positive.
        ref = np.where(np.abs(gamma) > 1e-12, gamma, Phi[np.argmax(np.abs(Phi), axis=0), np.arange(m.size)])
        Phi = Phi * np.sign(ref)
        out.append(Subsystem(name, np.arange(n)[sl], M, C, K, np.sqrt(np.clip(w2, 0.0, None)), Phi))
    return out[0], out[1]


@dataclass(frozen=True)
class Oscillator:
    """An equivalent single-DOF oscillator: mass, stiffness and damping."""

    m: float  # kg
    k: float  # N/m
    c: float  # N s/m

    @property
    def omega(self) -> float:
        return float(np.sqrt(self.k / self.m))

    @property
    def fn_hz(self) -> float:
        return self.omega / TWO_PI

    @property
    def zeta(self) -> float | None:
        return self.c / (2.0 * np.sqrt(self.k * self.m)) if self.k > 0 else None


@dataclass(frozen=True)
class CoupledModel:
    """The two equivalent oscillators joined, B's on A's, and the 2-DOF model's modes."""

    system: ChainSystem
    split: int  # number of masses in A
    mode_a: int  # 0-based mode of A
    mode_b: int  # 0-based mode of B
    a_mass: str  # one of A_MASSES
    residual: bool  # B's residual mass added to A's oscillator
    A: Subsystem
    B: Subsystem
    osc_a: Oscillator  # without the residual mass
    osc_b: Oscillator
    m_residual: float  # kg, B's mass not in its chosen mode (added to m_a if residual)
    M: np.ndarray  # 2x2, coordinates [u_a, u_b]
    C: np.ndarray
    K: np.ndarray
    omegas: np.ndarray  # rad/s, the two undamped natural frequencies
    zetas: list[float | None]  # exact damping ratio of each (None: overdamped)
    eta: np.ndarray  # 2x2, undamped mode shapes in [u_a, u_b], columns
    shapes: np.ndarray  # N x 2, the same modes on the chain (largest |entry| = +1)

    @property
    def m_a(self) -> float:
        """Mass of A's oscillator in the 2-DOF model, with the residual mass if added."""
        return float(self.M[0, 0])

    @property
    def fn_hz(self) -> np.ndarray:
        return self.omegas / TWO_PI

    @property
    def mass_ratio(self) -> float:
        """mu = m_b / m_a (with the residual mass, if added)."""
        return self.osc_b.m / self.m_a

    @property
    def freq_ratio(self) -> float:
        """f_B / f_A of the two uncoupled modes; inf if A's is 0."""
        return self.osc_b.omega / self.osc_a.omega if self.osc_a.omega > 0 else np.inf

    @property
    def rayleigh_ritz(self) -> bool:
        """With the tip modal mass and the residual mass the model is Rayleigh-Ritz: an upper bound."""
        return self.a_mass == "tip" and self.residual


def coupled_model(system: ChainSystem, split: int, mode_a: int = 0, mode_b: int = 0,
                  a_mass: str = "tip", residual: bool = True) -> CoupledModel:
    """Reduce A to its mode `mode_a` and B to its mode `mode_b` (0-based), and join them."""
    if a_mass not in A_MASSES:
        raise ValueError(f"a_mass must be one of {A_MASSES}")
    A, B = subsystems(system, split)
    if not (0 <= mode_a < A.omegas.size and 0 <= mode_b < B.omegas.size):
        raise ValueError("no such mode")
    if a_mass == "tip":
        m_a = float(A.tip_masses()[mode_a])
        if not np.isfinite(m_a):
            raise ValueError(f"A's mode {mode_a + 1} does not move m{split} (its tip), so it cannot "
                             "couple to B there")
    else:
        m_a = float(A.effective_masses[mode_a])
    m_b = float(B.effective_masses[mode_b])
    small = 1e-9 * system.masses.sum()
    if m_a < small:
        raise ValueError(f"A's mode {mode_a + 1} has no effective mass along x")
    if m_b < small:
        raise ValueError(f"B's mode {mode_b + 1} has (almost) no effective mass along x: a motion of its "
                         "base does not excite it, so it cannot couple to A")
    osc_a = Oscillator(m_a, m_a * A.omegas[mode_a] ** 2, m_a * A.modal_damping[mode_a])
    osc_b = Oscillator(m_b, m_b * B.omegas[mode_b] ** 2, m_b * B.modal_damping[mode_b])
    m_res = B.total_mass - m_b
    M = np.diag([m_a + (m_res if residual else 0.0), m_b])
    K = two_dof(osc_a.k, osc_b.k)
    C = two_dof(osc_a.c, osc_b.c)
    w2, eta = scipy.linalg.eigh(K, M)
    shapes = chain_shapes(A, B, mode_a, mode_b, eta)
    return CoupledModel(system, split, mode_a, mode_b, a_mass, residual, A, B, osc_a, osc_b, m_res,
                        M, C, K, np.sqrt(np.clip(w2, 0.0, None)), _zetas(M, C, K, w2), eta, shapes)


def two_dof(a: float, b: float) -> np.ndarray:
    """Stiffness (or damping) matrix of ground --a-- 1 --b-- 2."""
    return np.array([[a + b, -b], [-b, b]])


def chain_shapes(A: Subsystem, B: Subsystem, mode_a: int, mode_b: int, eta: np.ndarray) -> np.ndarray:
    """The 2-DOF mode shapes [u_a, u_b] drawn on the chain.

    A moves in its mode, scaled to u_a at its tip. B's base is A's tip, so B
    moves rigidly with it plus its own mode relative to it: an oscillator of
    effective mass Gamma^2 and relative displacement y = u_b - u_a stands for
    the relative motion Gamma phi y (its base force is then the same).
    """
    phi_a = A.Phi[:, mode_a]
    tip = phi_a[-1] if abs(phi_a[-1]) > 1e-12 else 1.0
    phi_b = B.Phi[:, mode_b] * B.participation[mode_b]
    u_a, u_b = eta
    x = np.vstack([np.outer(phi_a / tip, u_a), np.outer(np.ones(B.dofs.size), u_a) + np.outer(phi_b, u_b - u_a)])
    return x / x[np.argmax(np.abs(x), axis=0), np.arange(x.shape[1])]


def _zetas(M: np.ndarray, C: np.ndarray, K: np.ndarray, w2: np.ndarray) -> list[float | None]:
    """Exact damping ratio of the damped pole nearest each undamped frequency."""
    poles = list(damped_poles(M, C, K))
    out: list[float | None] = []
    for w in np.sqrt(np.clip(w2, 0.0, None)):
        if not poles or w <= 1e-9:
            out.append(None)
            continue
        p = min(poles, key=lambda lam: abs(abs(lam) - w))
        poles.remove(p)
        out.append(float(-p.real / abs(p)))
    return out


@dataclass(frozen=True)
class CoupledComparison:
    """One 2-DOF mode against the chain's mode it stands for."""

    index: int  # 1-based 2-DOF mode
    fn: float  # Hz, 2-DOF
    zeta: float | None
    full_index: int  # 1-based chain mode with the largest MAC
    fn_full: float  # Hz
    zeta_full: float | None  # exact
    mac: float

    @property
    def error(self) -> float | None:
        return (self.fn - self.fn_full) / self.fn_full if self.fn_full > 0 else None

    @property
    def zeta_error(self) -> float | None:
        if self.zeta is None or not self.zeta_full:
            return None
        return (self.zeta - self.zeta_full) / self.zeta_full


def compare_coupled(model: CoupledModel, full: ModalResult) -> list[CoupledComparison]:
    """Pair each 2-DOF mode with a different chain mode, the pair with the larger total MAC."""
    Phi = np.array([m.shape for m in full.modes]).T  # (n, n_modes)
    mac = np.array([[_mac(model.shapes[:, r], Phi[:, s]) for s in range(Phi.shape[1])] for r in range(2)])
    if Phi.shape[1] > 1:
        pairs = [(s, t) for s in range(Phi.shape[1]) for t in range(Phi.shape[1]) if s != t]
        best = max(pairs, key=lambda p: mac[0, p[0]] + mac[1, p[1]] - 1e-9 * (p[0] + p[1]))
    else:
        best = (0, 0)
    out = []
    for r, s in enumerate(best):
        mode = full.modes[s]
        out.append(CoupledComparison(r + 1, float(model.fn_hz[r]), model.zetas[r], s + 1, mode.fn_hz,
                                     mode.damped.zeta if mode.damped is not None else None, float(mac[r, s])))
    return out


def _mac(a: np.ndarray, b: np.ndarray) -> float:
    den = (a @ a) * (b @ b)
    return float((a @ b) ** 2 / den) if den > 0 else 0.0


def coupled_frf(model: CoupledModel, freqs_hz: np.ndarray) -> np.ndarray:
    """Receptance u_a / F of the 2-DOF model for a force on m_a (the interface), complex."""
    return receptance(model.M, model.C, model.K, freqs_hz, np.array([1.0, 0.0]))[:, 0]


def alone_frf(model: CoupledModel, freqs_hz: np.ndarray) -> np.ndarray:
    """Receptance of A on its own at its tip (the interface), with every mode of A, complex."""
    tip = np.zeros(model.split)
    tip[-1] = 1.0
    return receptance(model.A.M, model.A.C, model.A.K, freqs_hz, tip)[:, -1]


@dataclass(frozen=True)
class Sweep:
    """Natural frequencies as B is made stiffer (frequency) or heavier (mass), everything else fixed."""

    kind: str  # one of SWEEPS
    x: np.ndarray  # f_B / f_A, or mu = m_b / m_a
    f_a: np.ndarray  # Hz, A's mode alone, at each x
    f_b: np.ndarray  # Hz, B's mode alone
    two_dof: np.ndarray  # (len(x), 2) Hz
    full: np.ndarray  # (len(x), N) Hz, the whole chain
    x_now: float


def sweep(model: CoupledModel, kind: str, points: int = 241) -> Sweep:
    """Scale B's springs (frequency: f_B changes, mu does not) or all of B (mass: mu changes, f_B does not).

    B's dampers are scaled to keep its damping ratios. The 2-DOF model and the
    whole chain are solved at each point (undamped).
    """
    if kind not in SWEEPS:
        raise ValueError(f"kind must be one of {SWEEPS}")
    if kind == "frequency":
        x_now = model.freq_ratio
        if not np.isfinite(x_now) or x_now <= 0:
            raise ValueError("a mode at 0 Hz has no frequency ratio to sweep")
        lo, hi = min(0.25, x_now / 1.5), max(4.0, 1.5 * x_now)
    else:
        x_now = model.mass_ratio
        lo, hi = min(0.005, x_now / 2), max(2.0, 2 * x_now)
    x = np.unique(np.concatenate([np.geomspace(lo, hi, points), [x_now]]))
    scale = (x / x_now) ** 2 if kind == "frequency" else x / x_now
    B = slice(model.split, model.system.n)
    fa, fb, two, full = [], [], [], []
    for s in scale:
        k_scale, m_scale = (s, 1.0) if kind == "frequency" else (s, s)
        sys_ = model.system.copy()
        sys_.masses[B] *= m_scale
        sys_.stiffness[B] *= k_scale
        sys_.damping[B] *= np.sqrt(k_scale * m_scale)
        cm = coupled_model(sys_, model.split, model.mode_a, model.mode_b, model.a_mass, model.residual)
        fa.append(cm.osc_a.fn_hz)
        fb.append(cm.osc_b.fn_hz)
        two.append(cm.fn_hz)
        M, _, K = sys_.matrices()
        full.append(np.sqrt(np.clip(scipy.linalg.eigh(K, M, eigvals_only=True), 0.0, None)) / TWO_PI)
    return Sweep(kind, x, np.array(fa), np.array(fb), np.array(two), np.array(full), x_now)
