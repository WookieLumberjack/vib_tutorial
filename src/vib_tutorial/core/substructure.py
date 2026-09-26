"""Component mode synthesis of the chain: the Craig-Bampton (Hurty) method.

The chain is cut at one or more *interface* masses into substructures::

    ground --k1-- m1 --k2-- m2 --k3-- m3 --k4-- m4 --k5-- m5
    |--------- A ---------|
                          |------------- B -------------|
                        (interface)                  (tip)

Each substructure owns the springs/dampers between its nodes and the masses
of its nodes, except the interface mass on its left, which belongs to the
previous substructure (the split is arbitrary: the assembly just sums them).
The global matrices are recovered exactly by summing the substructures,
K = sum_s L_s^T K_s L_s, with Boolean localization matrices L_s.

The *boundary* (master) DOFs are the interface masses plus the last mass of
the chain, where the force is applied, so the loaded DOF stays physical.
Every other DOF is *interior* to exactly one substructure. For each
substructure, ordered [interior i | boundary b]:

* fixed-interface modes:  K_ii phi = w^2 M_ii phi  (boundary held fixed);
  keep the lowest n_kept, mass-normalized, as Phi_k.
* constraint modes:       Psi = -K_ii^-1 K_ib  (static shape of the interior
  when one boundary DOF moves by 1 and the others are held).
* transformation:         [x_i; x_b] = T [q; x_b],  T = [[Phi_k, Psi], [0, I]]
* reduced matrices:       M^ = T^T M T,  K^ = T^T K T,  C^ = T^T C T.
  K^ is block diagonal: diag(w_k^2) for q and K_bb + K_bi Psi for x_b.

The reduced substructures are assembled on their shared boundary DOFs into a
model with coordinates [q_A, q_B, ..., x_b]. Keeping no fixed-interface
modes is Guyan (static) condensation; keeping all of them is exact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg

from .modal import TWO_PI, ModalResult, receptance
from .model import ChainSystem, assemble_chain

SUBSTRUCTURE_NAMES = "ABCDEFGH"


@dataclass(frozen=True)
class Substructure:
    """One component, with its matrices in local [interior | boundary] order."""

    name: str
    elements: np.ndarray  # 0-based element indices (element e joins node e-1 to e)
    interior: np.ndarray  # global 0-based DOFs
    boundary: np.ndarray  # global 0-based DOFs
    M: np.ndarray
    C: np.ndarray
    K: np.ndarray
    fixed_omegas: np.ndarray  # rad/s, all fixed-interface natural frequencies
    Phi: np.ndarray  # all fixed-interface modes (n_i x n_i), Phi^T M_ii Phi = I
    n_kept: int
    Psi: np.ndarray  # constraint modes (n_i x n_b)
    T: np.ndarray  # (n_i + n_b) x (n_kept + n_b)
    M_red: np.ndarray
    C_red: np.ndarray
    K_red: np.ndarray

    @property
    def dofs(self) -> np.ndarray:
        """Global DOFs in local order: interior then boundary."""
        return np.concatenate([self.interior, self.boundary])

    @property
    def ni(self) -> int:
        return self.interior.size

    @property
    def nb(self) -> int:
        return self.boundary.size

    def block(self, mat: np.ndarray, rows: str, cols: str) -> np.ndarray:
        """Partition of a local matrix, e.g. block(K, "i", "b") = K_ib."""
        sl = {"i": slice(0, self.ni), "b": slice(self.ni, self.ni + self.nb)}
        return mat[sl[rows], sl[cols]]


@dataclass(frozen=True)
class CraigBamptonModel:
    system: ChainSystem
    substructures: list[Substructure]
    boundary: np.ndarray  # global boundary DOFs, sorted
    labels: list[str]  # one per reduced coordinate: "q_A1", ..., "x3"
    T: np.ndarray  # N x n_red, maps reduced coordinates to physical x
    M: np.ndarray  # assembled reduced matrices
    C: np.ndarray
    K: np.ndarray
    omegas: np.ndarray  # rad/s, reduced-model natural frequencies
    shapes: np.ndarray  # N x n_red physical mode shapes T eta, largest |entry| = +1

    @property
    def n_red(self) -> int:
        return len(self.labels)

    @property
    def fn_hz(self) -> np.ndarray:
        return self.omegas / TWO_PI

    @property
    def n_modal(self) -> int:
        return sum(s.n_kept for s in self.substructures)


def interior_counts(n: int, interfaces: list[int]) -> list[int]:
    """Number of interior DOFs in each substructure for these interface masses."""
    return [sub.size for sub in _partition(n, interfaces)[1]]


def _partition(n: int, interfaces: list[int]):
    """Per substructure: element indices, interior DOFs, boundary DOFs."""
    cuts = sorted({int(j) for j in interfaces})
    if not cuts or cuts[0] < 0 or cuts[-1] >= n - 1:
        raise ValueError("need at least one interface, each between the first and the last mass")
    lefts = [None] + cuts  # interface node on the left of each substructure (None = ground)
    rights = cuts + [n - 1]
    elements, interiors, boundaries = [], [], []
    for left, right in zip(lefts, rights):
        first = 0 if left is None else left + 1
        elements.append(np.arange(first, right + 1))
        interiors.append(np.arange(first, right))  # right-hand node is always a boundary DOF
        boundaries.append(np.array(([] if left is None else [left]) + [right], dtype=int))
    return elements, interiors, boundaries


def craig_bampton(system: ChainSystem, interfaces: list[int], n_kept: list[int]) -> CraigBamptonModel:
    """Reduce the chain with Craig-Bampton substructures cut at `interfaces`.

    interfaces: 0-based interface masses (each in 0 .. n-2); n_kept: the number
    of fixed-interface modes kept in each substructure (clipped to its interior).
    """
    n = system.n
    elements, interiors, boundaries = _partition(n, interfaces)
    if len(n_kept) != len(elements):
        raise ValueError(f"n_kept needs one entry per substructure ({len(elements)})")

    subs = []
    for s, (el, inner, bnd) in enumerate(zip(elements, interiors, boundaries)):
        # Element e ends at node e, so the same mask also picks the masses this
        # substructure owns (all its nodes except the interface on its left).
        mask = np.zeros(n)
        mask[el] = 1.0
        dofs = np.concatenate([inner, bnd])
        ix = np.ix_(dofs, dofs)
        M = np.diag(system.masses * mask)[ix]
        C = assemble_chain(system.damping * mask)[ix]
        K = assemble_chain(system.stiffness * mask)[ix]
        subs.append(_reduce(SUBSTRUCTURE_NAMES[s], el, inner, bnd, M, C, K, n_kept[s]))

    # Reduced coordinates: every substructure's kept modal q, then the boundary DOFs.
    boundary = np.unique(np.concatenate(boundaries))
    labels = [f"q_{sub.name}{k + 1}" for sub in subs for k in range(sub.n_kept)]
    labels += [f"x{b + 1}" for b in boundary]
    n_red = len(labels)
    T = np.zeros((n, n_red))
    Mr, Cr, Kr = (np.zeros((n_red, n_red)) for _ in range(3))
    q0 = 0
    nq = sum(sub.n_kept for sub in subs)
    for sub in subs:
        # Localization: where this substructure's reduced coordinates sit globally.
        cols = np.concatenate([
            np.arange(q0, q0 + sub.n_kept),
            nq + np.searchsorted(boundary, sub.boundary),
        ])
        q0 += sub.n_kept
        ix = np.ix_(cols, cols)
        Mr[ix] += sub.M_red
        Cr[ix] += sub.C_red
        Kr[ix] += sub.K_red
        T[np.ix_(sub.interior, cols)] = sub.T[: sub.ni]
    T[boundary, nq + np.arange(boundary.size)] = 1.0

    w2, eta = scipy.linalg.eigh(Kr, Mr)
    shapes = T @ eta
    shapes = shapes / shapes[np.argmax(np.abs(shapes), axis=0), np.arange(n_red)]
    return CraigBamptonModel(
        system=system,
        substructures=subs,
        boundary=boundary,
        labels=labels,
        T=T,
        M=Mr,
        C=Cr,
        K=Kr,
        omegas=np.sqrt(np.clip(w2, 0.0, None)),
        shapes=shapes,
    )


def _reduce(name, elements, interior, boundary, M, C, K, n_kept) -> Substructure:
    ni, nb = interior.size, boundary.size
    i, b = slice(0, ni), slice(ni, ni + nb)
    if ni:
        w2, Phi = scipy.linalg.eigh(K[i, i], M[i, i])
        if w2[0] <= 1e-9 * max(1.0, abs(w2[-1])):
            raise ValueError(
                f"substructure {name} has an interior mass that is free to move with its "
                "boundary fixed (a zero spring); Craig-Bampton needs K_ii to be invertible"
            )
        Psi = -np.linalg.solve(K[i, i], K[i, b])
    else:
        w2, Phi, Psi = np.zeros(0), np.zeros((0, 0)), np.zeros((0, nb))
    k = int(np.clip(n_kept, 0, ni))
    T = np.zeros((ni + nb, k + nb))
    T[i, :k] = Phi[:, :k]
    T[i, k:] = Psi
    T[b, k:] = np.eye(nb)
    return Substructure(
        name=name,
        elements=elements,
        interior=interior,
        boundary=boundary,
        M=M,
        C=C,
        K=K,
        fixed_omegas=np.sqrt(np.clip(w2, 0.0, None)),
        Phi=Phi,
        n_kept=k,
        Psi=Psi,
        T=T,
        M_red=T.T @ M @ T,
        C_red=T.T @ C @ T,
        K_red=T.T @ K @ T,
    )


@dataclass(frozen=True)
class ComponentMode:
    """A fixed-interface mode of one substructure, and the coupled mode it contributes most to."""

    substructure: str
    index: int  # 1-based within the substructure
    fn_hz: float  # boundary clamped, undamped
    zeta: float | None  # exact, boundary clamped; None if overdamped
    kept: bool
    closest: int  # 1-based full-system mode that this component mode carries the most energy of
    closest_fn_hz: float
    share: float  # that fraction of the coupled mode's strain energy, 0..1


def component_modes(model: CraigBamptonModel, full: ModalResult) -> list[ComponentMode]:
    """Every substructure's fixed-interface modes, each paired with a full-system mode.

    Each coupled mode phi (mass-normalized) is written in the substructure's
    Craig-Bampton coordinates: inside the substructure its elastic motion is
    d = x_i - Psi x_b (the interior motion minus what the boundary drags along
    statically), with fixed-interface amplitudes q = Phi^T M_ii d. Component
    mode r then holds w_r^2 q_r^2 of the coupled mode's strain energy
    phi^T K phi = w^2. The pairing picks the coupled mode where that share is
    largest: "which mode of the assembled chain does this component mode become?"
    """
    Phi = np.array([m.shape_mass_normalized for m in full.modes]).T  # (n, n_modes)
    energy = np.array([max(m.omega_n, 1e-12) ** 2 for m in full.modes])
    out = []
    for sub in model.substructures:
        if not sub.ni:
            continue
        Mii, Cii, Kii = (sub.block(mat, "i", "i") for mat in (sub.M, sub.C, sub.K))
        zetas = _match_zetas(damped_poles(Mii, Cii, Kii), sub.fixed_omegas)
        d = Phi[sub.interior] - sub.Psi @ Phi[sub.boundary]
        q = sub.Phi.T @ Mii @ d  # (ni, n_modes)
        shares = sub.fixed_omegas[:, None] ** 2 * q**2 / energy[None, :]
        for r in range(sub.ni):
            best = int(np.argmax(shares[r]))
            out.append(ComponentMode(sub.name, r + 1, float(sub.fixed_omegas[r] / TWO_PI), zetas[r],
                                     r < sub.n_kept, best + 1, full.modes[best].fn_hz, float(shares[r, best])))
    return out


@dataclass(frozen=True)
class ModeComparison:
    index: int  # 1-based mode number
    fn_true: float  # Hz
    fn_cb: float | None  # Hz; None when the reduced model has fewer modes
    mac: float | None
    shape_true: np.ndarray  # largest |entry| = +1
    shape_cb: np.ndarray | None  # sign aligned with shape_true
    zeta_true: float | None = None  # exact damping ratio; None if overdamped or rigid
    zeta_cb: float | None = None  # exact damping ratio of the reduced model's matching pole

    @property
    def error(self) -> float | None:
        """Relative frequency error (f_CB - f_true) / f_true; >= 0 (Rayleigh-Ritz)."""
        if self.fn_cb is None or self.fn_true <= 0:
            return None
        return (self.fn_cb - self.fn_true) / self.fn_true

    @property
    def zeta_error(self) -> float | None:
        """Relative damping-ratio error (zeta_CB - zeta_true) / zeta_true; either sign."""
        if self.zeta_cb is None or not self.zeta_true:
            return None
        return (self.zeta_cb - self.zeta_true) / self.zeta_true


def compare_modes(model: CraigBamptonModel, full: ModalResult) -> list[ModeComparison]:
    """Pair reduced mode r with true mode r (both ordered by frequency).

    Damping ratios are exact on both sides: the true one from the full model's
    matching damped pole, the CB one from the damped pole of M^, C^, K^ nearest
    in |lambda| to reduced mode r (so an overdamped mode does not shift the rest).
    """
    zeta_cb = _reduced_zetas(model)
    out = []
    for r, mode in enumerate(full.modes):
        true = mode.shape
        zeta_true = mode.damped.zeta if mode.damped is not None else None
        if r < model.n_red:
            cb = model.shapes[:, r].copy()
            if cb @ true < 0:
                cb = -cb
            mac = float((cb @ true) ** 2 / ((cb @ cb) * (true @ true)))
            out.append(ModeComparison(r + 1, mode.fn_hz, float(model.fn_hz[r]), mac, true, cb,
                                      zeta_true, zeta_cb[r]))
        else:
            out.append(ModeComparison(r + 1, mode.fn_hz, None, None, true, None, zeta_true))
    return out


def _reduced_zetas(model: CraigBamptonModel) -> list[float | None]:
    """Exact damping ratio of each reduced mode, None where it has no oscillatory pole."""
    return _match_zetas(damped_poles(model.M, model.C, model.K), model.omegas)


def _match_zetas(damped: np.ndarray, omegas: np.ndarray) -> list[float | None]:
    """Damping ratio of the damped pole nearest in |lambda| to each undamped omega."""
    poles = list(damped)
    out: list[float | None] = []
    for w in omegas:
        if not poles or w <= 1e-9:
            out.append(None)
            continue
        p = min(poles, key=lambda lam: abs(abs(lam) - w))
        poles.remove(p)
        out.append(float(-p.real / abs(p)))
    return out


def reduced_frf(model: CraigBamptonModel, freqs_hz: np.ndarray, input_dof: int) -> np.ndarray:
    """Receptance of the reduced model in physical DOFs, shape (len(freqs), n).

    H(w) = T (K^ - w^2 M^ + i w C^)^-1 T^T e_input
    """
    e = np.zeros(model.system.n)
    e[input_dof] = 1.0
    return receptance(model.M, model.C, model.K, freqs_hz, model.T.T @ e) @ model.T.T


def damped_poles(M: np.ndarray, C: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Oscillatory eigenvalues (Im > 0) of M x'' + C x' + K x = 0, ordered by |lambda|.

    Works for a full (non-diagonal) M such as the reduced M^.
    """
    n = M.shape[0]
    A = np.zeros((2 * n, 2 * n))
    A[:n, n:] = np.eye(n)
    A[n:, :n] = -np.linalg.solve(M, K)
    A[n:, n:] = -np.linalg.solve(M, C)
    lam = scipy.linalg.eigvals(A)
    tol = 1e-6 * max(1.0, float(np.abs(lam).max()))
    upper = lam[lam.imag > tol]
    return upper[np.argsort(np.abs(upper))]


def substructure_damping(sub: Substructure) -> tuple[float, float]:
    """How far C^ is from the block-diagonal form of K^.

    Returns (modal coupling, boundary coupling): the largest off-diagonal term
    of C^_qq and the largest term of C^_qb, each relative to the diagonal
    sqrt(C^_rr C^_ss). Both are 0 when the substructure's damping is
    proportional to its stiffness (C = beta K gives C^ = beta K^).
    """
    Cr = sub.C_red
    d = np.sqrt(np.abs(np.outer(np.diag(Cr), np.diag(Cr))))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(d > 0, np.abs(Cr) / d, 0.0)
    k = sub.n_kept
    qq = rel[:k, :k] - np.diag(np.diag(rel[:k, :k]))
    qb = rel[:k, k:]
    return (float(qq.max()) if qq.size else 0.0, float(qb.max()) if qb.size else 0.0)
