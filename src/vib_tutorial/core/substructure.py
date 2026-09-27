"""Component mode synthesis of the chain: Craig-Bampton, Rubin and MacNeal.

The chain is cut at *interface* masses into substructures (no cut at all makes
the whole chain one substructure, whose only boundary DOF is the tip)::

    ground --k1-- m1 --k2-- m2 --k3-- m3 --k4-- m4 --k5-- m5
    |--------- A ---------|
                          |------------- B -------------|
                        (interface)                  (tip)

Each substructure owns the springs/dampers between its nodes and the masses
of its nodes. For Craig-Bampton the interface mass belongs to the substructure
on its left; the free-interface methods split it half and half, so that every
substructure has mass at every DOF (the split is arbitrary: assembly just sums
them). The global matrices are recovered exactly by summing the substructures,
K = sum_s L_s^T K_s L_s, with Boolean localization matrices L_s.

The *boundary* (master) DOFs are the interface masses plus the last mass of
the chain, where the force is applied, so the loaded DOF stays physical.
Every other DOF is *interior* to exactly one substructure. For each
substructure, ordered [interior i | boundary b]:

Craig-Bampton (fixed interface):

* fixed-interface modes:  K_ii phi = w^2 M_ii phi  (boundary held fixed);
  keep the lowest n_kept, mass-normalized, as Phi_k.
* constraint modes:       Psi = -K_ii^-1 K_ib  (static shape of the interior
  when one boundary DOF moves by 1 and the others are held).
* transformation:         [x_i; x_b] = T [q; x_b],  T = [[Phi_k, Psi], [0, I]]
* reduced matrices:       M^ = T^T M T,  K^ = T^T K T,  C^ = T^T C T.
  K^ is block diagonal: diag(w_k^2) for q and K_bb + K_bi Psi for x_b.

Free interface with residual flexibility (Rubin; MacNeal):

* free-interface modes:   K phi = w^2 M phi  over the whole substructure,
  boundary free; keep the lowest n_kept (every rigid-body mode among them).
* residual flexibility:   G_d = Phi_d W_d^-2 Phi_d^T, the static flexibility
  of the discarded modes (= K^-1 - Phi_k W_k^-2 Phi_k^T without rigid modes).
* x = Phi_k q + G_d[:, b] f_b; the boundary rows give x_b = Phi_bk q + G_bb f_b,
  so f_b = G_bb^-1 (x_b - Phi_bk q) and again [x_i; x_b] = T [q; x_b] with
  T = [[Phi_ik - R Phi_bk, R], [0, I]],  R = G_ib G_bb^-1.
* K^ = T^T K T = [[W_k^2 + Phi_bk^T G_bb^-1 Phi_bk, -Phi_bk^T G_bb^-1],
  [-G_bb^-1 Phi_bk, G_bb^-1]]: the modes couple to the boundary by stiffness.
* Rubin: M^ = T^T M T (Rayleigh-Ritz, keeps the residual mass).
  MacNeal: the residual flexibility is massless, so M^ = diag(I, 0); the
  massless x_b are condensed statically when solving for the modes.
  Both project the damping like the stiffness, C^ = T^T C T.

The reduced substructures are assembled on their shared boundary DOFs into a
model with coordinates [q_A, q_B, ..., x_b]. Craig-Bampton with no modes kept
is Guyan (static) condensation; Craig-Bampton and Rubin with all modes kept
are exact. MacNeal never is: it drops the inertia of the discarded modes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg

from .modal import TWO_PI, ModalResult, receptance
from .model import ChainSystem, assemble_chain

SUBSTRUCTURE_NAMES = "ABCDEFGH"
METHODS = ("craig-bampton", "rubin", "macneal")
METHOD_NAMES = {"craig-bampton": "Craig–Bampton", "rubin": "Rubin", "macneal": "MacNeal"}
METHOD_SHORT = {"craig-bampton": "CB", "rubin": "Rubin", "macneal": "MacNeal"}


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
    free: bool  # component modes with the boundary free (Rubin, MacNeal) or held (Craig-Bampton)
    omegas: np.ndarray  # rad/s, all component-mode natural frequencies
    Phi: np.ndarray  # all component modes, mass-normalized: fixed (n_i x n_i) or free (local, n x n)
    n_kept: int
    Psi: np.ndarray  # (n_i x n_b) boundary columns of T: constraint or residual attachment modes
    T: np.ndarray  # (n_i + n_b) x (n_kept + n_b)
    M_red: np.ndarray
    C_red: np.ndarray
    K_red: np.ndarray
    n_rigid: int = 0  # free-interface rigid-body modes (always kept)
    G: np.ndarray | None = None  # free interface: residual flexibility of the discarded modes, local order

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
class CMSModel:
    system: ChainSystem
    method: str  # one of METHODS
    substructures: list[Substructure]
    boundary: np.ndarray  # global boundary DOFs, sorted
    labels: list[str]  # one per reduced coordinate: "q_A1", ..., "x3"
    T: np.ndarray  # N x n_red, maps reduced coordinates to physical x
    M: np.ndarray  # assembled reduced matrices
    C: np.ndarray
    K: np.ndarray
    omegas: np.ndarray  # rad/s, reduced-model natural frequencies (fewer than n_red for MacNeal)
    shapes: np.ndarray  # N x len(omegas) physical mode shapes T eta, largest |entry| = +1

    @property
    def n_red(self) -> int:
        return len(self.labels)

    @property
    def fn_hz(self) -> np.ndarray:
        return self.omegas / TWO_PI

    @property
    def n_modal(self) -> int:
        return sum(s.n_kept for s in self.substructures)

    @property
    def name(self) -> str:
        return METHOD_NAMES[self.method]

    @property
    def short_name(self) -> str:
        return METHOD_SHORT[self.method]

    @property
    def free(self) -> bool:
        return self.method != "craig-bampton"


CraigBamptonModel = CMSModel  # the name before the free-interface methods were added


def interior_counts(n: int, interfaces: list[int]) -> list[int]:
    """Number of interior DOFs in each substructure for these interface masses."""
    return [sub.size for sub in _partition(n, interfaces)[1]]


def kept_ranges(system: ChainSystem, interfaces: list[int], method: str) -> list[tuple[int, int]]:
    """(fewest, most) component modes each substructure can keep with this method.

    The most is the number of interior DOFs for every method (then Craig-Bampton
    and Rubin are exact). The fewest is 0 for Craig-Bampton, and the number of
    rigid-body modes for the free-interface methods, which must always be kept
    (and at least one mode in all for MacNeal, whose boundary DOFs have no mass).
    """
    elements, interiors, boundaries = _partition(system.n, interfaces)
    out = []
    for el, inner, bnd in zip(elements, interiors, boundaries):
        lo = 0
        if method != "craig-bampton" and inner.size:
            M, _, K = _substructure_matrices(system, el, inner, bnd, interfaces, split=True)
            lo = min(_rigid_count(scipy.linalg.eigh(K, M, eigvals_only=True)), inner.size)
        out.append((lo, inner.size))
    if method == "macneal" and not any(lo for lo, _ in out):
        # MacNeal's boundary DOFs are massless, so with no mode kept the model would have no mass at
        # all (e.g. one grounded substructure): the first substructure with an interior keeps one.
        s = next((s for s, (_, hi) in enumerate(out) if hi), None)
        if s is not None:
            out[s] = (1, out[s][1])
    return out


def _partition(n: int, interfaces: list[int]):
    """Per substructure: element indices, interior DOFs, boundary DOFs."""
    cuts = sorted({int(j) for j in interfaces})
    if cuts and (cuts[0] < 0 or cuts[-1] >= n - 1):
        raise ValueError("each interface must be between the first and the last mass")
    lefts = [None] + cuts  # interface node on the left of each substructure (None = ground)
    rights = cuts + [n - 1]
    elements, interiors, boundaries = [], [], []
    for left, right in zip(lefts, rights):
        first = 0 if left is None else left + 1
        elements.append(np.arange(first, right + 1))
        interiors.append(np.arange(first, right))  # right-hand node is always a boundary DOF
        boundaries.append(np.array(([] if left is None else [left]) + [right], dtype=int))
    return elements, interiors, boundaries


def _substructure_matrices(system, elements, interior, boundary, interfaces, split: bool):
    """M, C, K of one substructure in local [interior | boundary] order.

    Element e ends at node e, so the element mask also picks the masses this
    substructure owns: all its nodes except the interface on its left. With
    split=True each interface mass is shared half and half instead.
    """
    n = system.n
    mask = np.zeros(n)
    mask[elements] = 1.0
    share = mask.copy()
    if split:
        for j in interfaces:
            if j in boundary:
                share[j] = 0.5
    dofs = np.concatenate([interior, boundary])
    ix = np.ix_(dofs, dofs)
    return (
        np.diag(system.masses * share)[ix],
        assemble_chain(system.damping * mask)[ix],
        assemble_chain(system.stiffness * mask)[ix],
    )


def _rigid_count(w2: np.ndarray) -> int:
    return int(np.sum(w2 <= 1e-9 * max(1.0, abs(float(w2[-1])))))


def component_mode_synthesis(system: ChainSystem, interfaces: list[int], n_kept: list[int],
                             method: str = "craig-bampton") -> CMSModel:
    """Reduce the chain with substructures cut at `interfaces`, by any of METHODS.

    interfaces: 0-based interface masses (each in 0 .. n-2; none = one substructure); n_kept: the number
    of component modes kept in each substructure (clipped to kept_ranges).
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    elements, interiors, boundaries = _partition(system.n, interfaces)
    if len(n_kept) != len(elements):
        raise ValueError(f"n_kept needs one entry per substructure ({len(elements)})")
    free = method != "craig-bampton"
    subs = []
    for s, (el, inner, bnd) in enumerate(zip(elements, interiors, boundaries)):
        M, C, K = _substructure_matrices(system, el, inner, bnd, interfaces, split=free)
        name = SUBSTRUCTURE_NAMES[s]
        if free:
            subs.append(_reduce_free(name, el, inner, bnd, M, C, K, n_kept[s], method == "rubin"))
        else:
            subs.append(_reduce_fixed(name, el, inner, bnd, M, C, K, n_kept[s]))
    return _assemble(system, method, subs, np.unique(np.concatenate(boundaries)))


def craig_bampton(system: ChainSystem, interfaces: list[int], n_kept: list[int]) -> CMSModel:
    """Reduce the chain with Craig-Bampton (fixed-interface) substructures."""
    return component_mode_synthesis(system, interfaces, n_kept, "craig-bampton")


def free_interface(system: ChainSystem, interfaces: list[int], n_kept: list[int],
                   residual_mass: bool = True) -> CMSModel:
    """Reduce the chain with free-interface modes plus residual flexibility.

    residual_mass=True is Rubin's method (a Rayleigh-Ritz projection);
    False is MacNeal's, which treats the residual flexibility as massless.
    """
    return component_mode_synthesis(system, interfaces, n_kept, "rubin" if residual_mass else "macneal")


def _assemble(system: ChainSystem, method: str, subs: list[Substructure], boundary: np.ndarray) -> CMSModel:
    """Join the reduced substructures on their shared boundary DOFs and solve for the modes."""
    n = system.n
    # Reduced coordinates: every substructure's kept modal q, then the boundary DOFs.
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

    w2, eta = undamped_modes(Kr, Mr)
    shapes = T @ eta
    shapes = shapes / shapes[np.argmax(np.abs(shapes), axis=0), np.arange(shapes.shape[1])]
    return CMSModel(
        system=system,
        method=method,
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


def undamped_modes(K: np.ndarray, M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """K eta = w^2 M eta, allowing coordinates without mass (rows of M that are all 0).

    Massless coordinates (MacNeal's boundary DOFs) carry no inertia, so they
    follow the others statically: they are condensed out, and their part of
    each eigenvector is recovered as -K_zz^-1 K_zm eta_m.
    """
    massless = np.all(M == 0.0, axis=1)
    if not massless.any():
        return scipy.linalg.eigh(K, M)
    m, z = ~massless, massless
    if not m.any():
        raise ValueError("the reduced model has no mass")
    back = -np.linalg.solve(K[np.ix_(z, z)], K[np.ix_(z, m)])
    Kc = K[np.ix_(m, m)] + K[np.ix_(m, z)] @ back
    w2, eta_m = scipy.linalg.eigh(Kc, M[np.ix_(m, m)])
    eta = np.zeros((K.shape[0], w2.size))
    eta[m] = eta_m
    eta[z] = back @ eta_m
    return w2, eta


def _reduce_fixed(name, elements, interior, boundary, M, C, K, n_kept) -> Substructure:
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
        free=False,
        omegas=np.sqrt(np.clip(w2, 0.0, None)),
        Phi=Phi,
        n_kept=k,
        Psi=Psi,
        T=T,
        M_red=T.T @ M @ T,
        C_red=T.T @ C @ T,
        K_red=T.T @ K @ T,
    )


def _reduce_free(name, elements, interior, boundary, M, C, K, n_kept, residual_mass: bool) -> Substructure:
    ni, nb = interior.size, boundary.size
    i, b = slice(0, ni), slice(ni, ni + nb)
    if not ni:
        # Nothing to reduce: the substructure is its boundary DOFs, kept physical.
        T = np.eye(nb)
        return Substructure(name, elements, interior, boundary, M, C, K, True, np.zeros(0), np.zeros((nb, 0)),
                            0, np.zeros((0, nb)), T, M.copy(), C.copy(), K.copy())
    w2, Phi = scipy.linalg.eigh(K, M)
    # Sign: the largest entry positive (the first of them, if tied), e.g. a rigid-body mode all +.
    peak = np.argmax(np.abs(Phi) >= np.abs(Phi).max(axis=0) - 1e-9, axis=0)
    Phi = Phi * np.sign(Phi[peak, np.arange(Phi.shape[1])])
    n_rigid = _rigid_count(w2)
    w2[:n_rigid] = 0.0
    if n_rigid > ni:
        raise ValueError(
            f"substructure {name} has {n_rigid} rigid-body modes but only {ni} interior DOFs "
            "(zero springs); the free-interface methods must keep every rigid-body mode"
        )
    k = int(np.clip(n_kept, n_rigid, ni))
    # Residual flexibility: the static flexibility of the discarded modes. (Without
    # rigid-body modes this is K^-1 - Phi_k W_k^-2 Phi_k^T, which needs no discarded modes.)
    Phi_d = Phi[:, k:]
    G = Phi_d @ np.diag(1.0 / w2[k:]) @ Phi_d.T
    Gbb = G[b, b]
    if np.linalg.cond(Gbb) > 1e10:
        raise ValueError(
            f"substructure {name}: the discarded modes give no flexibility at a boundary DOF "
            "(G_bb is singular), so its displacement cannot be a coordinate; keep fewer modes"
        )
    R = np.linalg.solve(Gbb, G[b, i]).T  # G_ib G_bb^-1 (G is symmetric)
    T = np.zeros((ni + nb, k + nb))
    T[i, :k] = Phi[i, :k] - R @ Phi[b, :k]
    T[i, k:] = R
    T[b, k:] = np.eye(nb)
    if residual_mass:
        M_red = T.T @ M @ T
    else:
        # MacNeal: only the kept modes carry inertia; the residual flexibility is a massless spring.
        M_red = np.zeros((k + nb, k + nb))
        M_red[:k, :k] = np.eye(k)
    return Substructure(
        name=name,
        elements=elements,
        interior=interior,
        boundary=boundary,
        M=M,
        C=C,
        K=K,
        free=True,
        omegas=np.sqrt(w2),
        Phi=Phi,
        n_kept=k,
        Psi=R,
        T=T,
        M_red=M_red,
        C_red=T.T @ C @ T,
        K_red=T.T @ K @ T,
        n_rigid=n_rigid,
        G=G,
    )


@dataclass(frozen=True)
class ComponentMode:
    """A mode of one substructure on its own, and the coupled mode it contributes most to."""

    substructure: str
    index: int  # 1-based within the substructure
    fn_hz: float  # undamped; boundary clamped (fixed interface) or free (free interface)
    zeta: float | None  # exact, same boundary condition; None if overdamped or rigid
    kept: bool
    closest: int  # 1-based full-system mode that this component mode carries the most energy of
    closest_fn_hz: float
    share: float  # that fraction of the coupled mode's energy, 0..1
    dofs: np.ndarray  # global DOFs of the substructure, ascending along the chain
    shape: np.ndarray  # mode shape on those DOFs (0 at a held boundary), peak |1|, signed to
    # move the same way as the coupled mode


def component_modes(model: CMSModel, full: ModalResult) -> list[ComponentMode]:
    """Every substructure's own modes, each paired with a full-system mode.

    Fixed interface: each coupled mode phi (mass-normalized) is written in the
    substructure's Craig-Bampton coordinates: inside the substructure its
    elastic motion is d = x_i - Psi x_b (the interior motion minus what the
    boundary drags along statically), with fixed-interface amplitudes
    q = Phi^T M_ii d. Component mode r then holds w_r^2 q_r^2 of the coupled
    mode's strain energy phi^T K phi = w^2.

    Free interface: the free modes of a substructure are a complete basis for
    its motion, so q = Phi^T M u with u = phi on the substructure's DOFs, and
    component mode r holds q_r^2 of the coupled mode's kinetic energy
    phi^T M phi = 1 (this also covers rigid-body modes, which have no strain).

    The pairing picks the coupled mode where that share is largest: "which
    mode of the assembled chain does this component mode become?"
    """
    Phi = np.array([m.shape_mass_normalized for m in full.modes]).T  # (n, n_modes)
    energy = np.array([max(m.omega_n, 1e-12) ** 2 for m in full.modes])
    out = []
    for sub in model.substructures:
        if not sub.ni:
            continue
        if sub.free:
            zetas = _match_zetas(damped_poles(sub.M, sub.C, sub.K), sub.omegas)
            q = sub.Phi.T @ sub.M @ Phi[sub.dofs]  # (n_s, n_modes)
            shares = q**2
            rows = sub.dofs
        else:
            Mii, Cii, Kii = (sub.block(mat, "i", "i") for mat in (sub.M, sub.C, sub.K))
            zetas = _match_zetas(damped_poles(Mii, Cii, Kii), sub.omegas)
            d = Phi[sub.interior] - sub.Psi @ Phi[sub.boundary]
            q = sub.Phi.T @ Mii @ d  # (ni, n_modes)
            shares = sub.omegas[:, None] ** 2 * q**2 / energy[None, :]
            rows = sub.interior
        dofs = np.sort(sub.dofs)
        for r in range(sub.omegas.size):
            best = int(np.argmax(shares[r]))
            shape = np.zeros(dofs.size)
            shape[np.searchsorted(dofs, rows)] = sub.Phi[:, r]
            shape *= np.sign(q[r, best]) or 1.0
            shape /= np.abs(shape).max()
            out.append(ComponentMode(sub.name, r + 1, float(sub.omegas[r] / TWO_PI), zetas[r],
                                     r < sub.n_kept, best + 1, full.modes[best].fn_hz, float(shares[r, best]),
                                     dofs, shape))
    return out


@dataclass(frozen=True)
class ModeComparison:
    index: int  # 1-based mode number
    fn_true: float  # Hz
    fn_red: float | None  # Hz, reduced model; None when it has fewer modes
    mac: float | None
    shape_true: np.ndarray  # largest |entry| = +1
    shape_red: np.ndarray | None  # sign aligned with shape_true
    zeta_true: float | None = None  # exact damping ratio; None if overdamped or rigid
    zeta_red: float | None = None  # exact damping ratio of the reduced model's matching pole

    @property
    def error(self) -> float | None:
        """Relative frequency error (f_red - f_true) / f_true; >= 0 for Rayleigh-Ritz methods."""
        if self.fn_red is None or self.fn_true <= 0:
            return None
        return (self.fn_red - self.fn_true) / self.fn_true

    @property
    def zeta_error(self) -> float | None:
        """Relative damping-ratio error (zeta_red - zeta_true) / zeta_true; either sign."""
        if self.zeta_red is None or not self.zeta_true:
            return None
        return (self.zeta_red - self.zeta_true) / self.zeta_true


def compare_modes(model: CMSModel, full: ModalResult) -> list[ModeComparison]:
    """Pair reduced mode r with true mode r (both ordered by frequency).

    Damping ratios are exact on both sides: the true one from the full model's
    matching damped pole, the reduced one from the damped pole of M^, C^, K^
    nearest in |lambda| to reduced mode r (so an overdamped mode does not
    shift the rest).
    """
    zeta_red = _reduced_zetas(model)
    out = []
    for r, mode in enumerate(full.modes):
        true = mode.shape
        zeta_true = mode.damped.zeta if mode.damped is not None else None
        if r < model.omegas.size:
            red = model.shapes[:, r].copy()
            if red @ true < 0:
                red = -red
            mac = float((red @ true) ** 2 / ((red @ red) * (true @ true)))
            out.append(ModeComparison(r + 1, mode.fn_hz, float(model.fn_hz[r]), mac, true, red,
                                      zeta_true, zeta_red[r]))
        else:
            out.append(ModeComparison(r + 1, mode.fn_hz, None, None, true, None, zeta_true))
    return out


def _reduced_zetas(model: CMSModel) -> list[float | None]:
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


def reduced_frf(model: CMSModel, freqs_hz: np.ndarray, input_dof: int) -> np.ndarray:
    """Receptance of the reduced model in physical DOFs, shape (len(freqs), n).

    H(w) = T (K^ - w^2 M^ + i w C^)^-1 T^T e_input
    """
    e = np.zeros(model.system.n)
    e[input_dof] = 1.0
    return receptance(model.M, model.C, model.K, freqs_hz, model.T.T @ e) @ model.T.T


def damped_poles(M: np.ndarray, C: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Oscillatory eigenvalues (Im > 0) of M x'' + C x' + K x = 0, ordered by |lambda|.

    Works for a full (non-diagonal) M such as the reduced M^, and for a
    singular one (MacNeal's massless boundary DOFs): the pencil
    [[0, I], [-K, -C]] - lambda [[I, 0], [0, M]] then has infinite
    eigenvalues, which are dropped.
    """
    n = M.shape[0]
    A = np.zeros((2 * n, 2 * n))
    A[:n, n:] = np.eye(n)
    A[n:, :n] = -K
    A[n:, n:] = -C
    B = np.eye(2 * n)
    B[n:, n:] = M
    lam = scipy.linalg.eigvals(A, B)
    lam = lam[np.isfinite(lam)]
    tol = 1e-6 * max(1.0, float(np.abs(lam).max())) if lam.size else 0.0
    upper = lam[lam.imag > tol]
    return upper[np.argsort(np.abs(upper))]


def substructure_damping(sub: Substructure) -> tuple[float, float]:
    """How far C^ is from diagonal in the kept modes, and how much it couples them to the boundary.

    Returns (modal coupling, boundary coupling): the largest off-diagonal term
    of C^_qq and the largest term of C^_qb, each relative to the diagonal
    sqrt(C^_rr C^_ss). For Craig-Bampton both are 0 when the substructure's
    damping is proportional to its stiffness (C = beta K gives C^ = beta K^).
    """
    Cr = sub.C_red
    d = np.sqrt(np.abs(np.outer(np.diag(Cr), np.diag(Cr))))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(d > 0, np.abs(Cr) / d, 0.0)
    k = sub.n_kept
    qq = rel[:k, :k] - np.diag(np.diag(rel[:k, :k]))
    qb = rel[:k, k:]
    return (float(qq.max()) if qq.size else 0.0, float(qb.max()) if qb.size else 0.0)
