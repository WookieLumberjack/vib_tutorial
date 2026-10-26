"""A string, a bar or a beam, and the chains of masses and springs that approximate it.

The continua (SI throughout):

* a **string** under tension T with mass μ per length: transverse waves, μ w_tt = T w_xx;
* a **bar** of modulus E, density ρ and section A: axial waves, ρA u_tt = EA u_xx;
* a **beam** (Euler–Bernoulli) of bending stiffness EI: ρA w_tt = −EI w_xxxx.

The string and the bar are one equation, with the wave speed c = √(T/μ) or √(E/ρ): every
wavenumber κ travels at c, ω = cκ (non-dispersive). The beam's waves go as ω = aκ², with
a = √(EI/ρA): short waves outrun long ones (dispersive). The bar and the beam here have a
rectangular section, width × depth; the string's `tension` and `line_density` stand in for
the bar's EA and ρA.

Each is cut into N equal elements of length h = L/N. The string and the bar get linear
elements (a spring T/h or EA/h between neighbouring nodes), the beam Hermite cubics (a
deflection and a slope at each node). Each element's mass is shared out one of three ways
(`Scheme`):

* **lumped**: half to each end node (the beam's slopes get none, so they are condensed
  out exactly). Lumped, the string is beads on a massless string and the bar is the
  chain of the first page, k = EA/h, m = ρAh;
* **consistent**: from the same shape functions as the stiffness (Rayleigh–Ritz), so the
  frequencies are upper bounds;
* **mixed**: the average of the two. For the bar and string lumped is low by (κh)²/24
  and consistent high by as much, so their average's error is O((κh)⁴).

On an endless uniform mesh a wave e^{i(κx − ωt)} sampled at the nodes is exact in shape,
and its frequency comes from one node's equation (`dispersion`): for the string and bar
with the mass stencil (β, 1 − 2β, β) (β = 0 lumped, 1/6 consistent, 1/12 mixed),

    ω² = (c/h)² 2(1 − cos κh) / (1 − 2β(1 − cos κh)),

which for κh ≪ 1 is c²κ² but stops at κh = π: two elements per wavelength, the shortest
wave a mesh can carry (its cut-off, where the group velocity dω/dκ falls to zero). The
modes of a string or bar with either end fixed or free are such waves, κ_n from the ends,
so the mesh's frequencies are exactly the dispersion curve's at κ_n; likewise a pinned–pinned
beam's. Low modes span many elements and come out nearly right; the high ones crowd towards
the cut-off and are wrong by tens of per cent.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

import numpy as np
import scipy.linalg
import scipy.optimize

TWO_PI = 2.0 * math.pi
MAX_ELEMENTS = 200
CONVERGENCE_COUNTS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192)
NOISE = 1e-14  # the eigensolver's relative round-off in the largest eigenvalue


class Kind(Enum):
    STRING = "String"
    BAR = "Bar"
    BEAM = "Beam"


class End(Enum):
    FIXED = "Fixed"  # string and bar: no displacement
    FREE = "Free"  # every kind: no force (and, a beam's, no moment)
    PINNED = "Pinned"  # beam: no deflection, free to turn
    CLAMPED = "Clamped"  # beam: no deflection, no slope


ENDS = {Kind.STRING: (End.FIXED, End.FREE), Kind.BAR: (End.FIXED, End.FREE),
        Kind.BEAM: (End.CLAMPED, End.PINNED, End.FREE)}


class Scheme(Enum):
    LUMPED = "Lumped"
    CONSISTENT = "Consistent"
    MIXED = "Mixed"


SCHEMES = tuple(Scheme)
STENCIL = {Scheme.LUMPED: 0.0, Scheme.CONSISTENT: 1.0 / 6.0, Scheme.MIXED: 1.0 / 12.0}  # β, string and bar


@dataclass(frozen=True)
class Continuum:
    """The medium, its length (m) and its ends. A string uses `tension` (N) and `line_density` (kg/m); a bar or beam
    `modulus` (Pa), `density` (kg/m³) and its section, `width` × `depth` (m; a beam bends across its depth)."""

    kind: Kind
    length: float
    left: End
    right: End
    tension: float = 100.0
    line_density: float = 0.01
    modulus: float = 210e9
    density: float = 7850.0
    width: float = 0.02
    depth: float = 0.01

    def __post_init__(self) -> None:
        if not self.length > 0:
            raise ValueError("length must be positive")
        for end in (self.left, self.right):
            if end not in ENDS[self.kind]:
                raise ValueError(f"a {self.kind.value.lower()} can't have a {end.value.lower()} end")
        for name in ("tension", "line_density", "modulus", "density", "width", "depth"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive")

    def with_(self, **changes) -> Continuum:
        return dataclasses.replace(self, **changes)

    @property
    def area(self) -> float:
        return self.width * self.depth

    @property
    def rigidity(self) -> float:
        """T (N), EA (N) or EI (N·m²): what resists the deformation."""
        if self.kind is Kind.STRING:
            return self.tension
        if self.kind is Kind.BAR:
            return self.modulus * self.area
        return self.modulus * self.width * self.depth**3 / 12.0

    @property
    def mass_per_length(self) -> float:
        return self.line_density if self.kind is Kind.STRING else self.density * self.area

    @property
    def speed(self) -> float:
        """The string's or bar's wave speed c (m/s), or the beam's a = √(EI/ρA) (m²/s), with ω = aκ²."""
        return math.sqrt(self.rigidity / self.mass_per_length)

    @property
    def dofs_per_node(self) -> int:
        return 2 if self.kind is Kind.BEAM else 1

    def omega(self, kappa):
        """The continuum's dispersion: ω (rad/s) at wavenumber κ (rad/m)."""
        kappa = np.asarray(kappa, dtype=float)
        return self.speed * (kappa**2 if self.kind is Kind.BEAM else kappa)

    @property
    def rigid_modes(self) -> int:
        """How many rigid-body modes the ends allow: a free–free bar slides, a free–free beam slides and tips, a
        pinned–free one swings."""
        ends = {self.left, self.right}
        if self.kind is not Kind.BEAM:
            return int(ends == {End.FREE})
        return {frozenset({End.FREE}): 2, frozenset({End.FREE, End.PINNED}): 1}.get(frozenset(ends), 0)

    # ------------------------------------------------------------- the exact modes
    def exact_wavenumbers(self, count: int) -> np.ndarray:
        """The first `count` flexible modes' wavenumbers κ_n (rad/m), ascending."""
        n = np.arange(1, count + 1, dtype=float)
        if self.kind is not Kind.BEAM:
            same = (self.left is End.FIXED) == (self.right is End.FIXED)
            return (n if same else n - 0.5) * math.pi / self.length
        return _beam_roots(self.left, self.right, count) / self.length

    def exact_hz(self, count: int) -> np.ndarray:
        return self.omega(self.exact_wavenumbers(count)) / TWO_PI

    def exact_shape(self, kappa: float, x: np.ndarray) -> np.ndarray:
        """A flexible mode's shape at x, the mode with wavenumber κ (from `exact_wavenumbers`), largest 1."""
        x = np.asarray(x, dtype=float)
        if self.kind is not Kind.BEAM:
            w = np.sin(kappa * x) if self.left is End.FIXED else np.cos(kappa * x)
        else:
            beta_l = kappa * self.length
            a = scipy.linalg.null_space(_beam_matrix(self.left, self.right, np.array([beta_l]))[0], rcond=1e-8)
            if a.shape[1] == 0:  # not quite on the root: the smallest singular vector
                a = np.linalg.svd(_beam_matrix(self.left, self.right, np.array([beta_l]))[0])[2][-1:].T
            w = _beam_basis(kappa * x, beta_l, 0) @ a[:, 0]
        return w / np.max(np.abs(w))


def _beam_basis(bx, beta_l, order: int) -> np.ndarray:
    """The beam's free solutions cos βx, sin βx, e^{−βx}, e^{−β(L−x)} (scaled so none overflows), their `order`th
    derivatives over β^order, a row per βx (βL broadcast against it)."""
    bx = np.atleast_1d(np.asarray(bx, dtype=float))
    c, s, d, g = np.cos(bx), np.sin(bx), np.exp(-bx), np.exp(bx - beta_l)
    rows = {0: (c, s, d, g), 1: (-s, c, -d, g), 2: (-c, -s, d, g), 3: (s, -c, -d, g)}[order]
    return np.stack(rows, axis=-1)


_CONDITIONS = {End.CLAMPED: (0, 1), End.PINNED: (0, 2), End.FREE: (2, 3)}  # the derivatives that vanish


def _beam_matrix(left: End, right: End, beta_l: np.ndarray) -> np.ndarray:
    """The end conditions on the four coefficients, a 4 × 4 matrix per βL."""
    rows = []
    for end, at in ((left, np.zeros_like(beta_l)), (right, beta_l)):
        for order in _CONDITIONS[end]:
            rows.append(_beam_basis(at, beta_l, order))
    return np.stack(rows, axis=1)


@lru_cache(maxsize=32)
def _beam_roots(left: End, right: End, count: int) -> np.ndarray:
    """The first `count` roots βL > 0 of the beam's frequency equation (its end conditions' determinant)."""
    step = 0.05
    top = (count + 2) * math.pi + 1.0
    grid = np.arange(step, top, step)
    det = np.linalg.det(_beam_matrix(left, right, grid))

    def f(b: float) -> float:
        return float(np.linalg.det(_beam_matrix(left, right, np.array([b]))[0]))

    roots = [scipy.optimize.brentq(f, grid[i], grid[i + 1], xtol=1e-14)
             for i in np.flatnonzero(np.sign(det[:-1]) * np.sign(det[1:]) < 0)]
    roots = np.array(roots[:count])
    if roots.size < count:
        raise RuntimeError("missed a root of the beam's frequency equation")
    return roots


# ------------------------------------------------------------- the elements
def element_matrices(c: Continuum, h: float, scheme: Scheme) -> tuple[np.ndarray, np.ndarray]:
    """One element's stiffness and mass on its end nodes' DOFs (the beam's: w₁, θ₁, w₂, θ₂)."""
    r, mu = c.rigidity, c.mass_per_length
    if c.kind is not Kind.BEAM:
        k = r / h * np.array([[1.0, -1.0], [-1.0, 1.0]])
        beta = STENCIL[scheme]
        m = mu * h * np.array([[0.5 - beta, beta], [beta, 0.5 - beta]])
        return k, m
    k = r / h**3 * np.array([[12, 6 * h, -12, 6 * h],
                             [6 * h, 4 * h * h, -6 * h, 2 * h * h],
                             [-12, -6 * h, 12, -6 * h],
                             [6 * h, 2 * h * h, -6 * h, 4 * h * h]])
    consistent = mu * h / 420 * np.array([[156, 22 * h, 54, -13 * h],
                                          [22 * h, 4 * h * h, 13 * h, -3 * h * h],
                                          [54, 13 * h, 156, -22 * h],
                                          [-13 * h, -3 * h * h, -22 * h, 4 * h * h]])
    lumped = mu * h / 2 * np.diag([1.0, 0.0, 1.0, 0.0])
    m = {Scheme.LUMPED: lumped, Scheme.CONSISTENT: consistent, Scheme.MIXED: 0.5 * (lumped + consistent)}[scheme]
    return k, m


def _condense(k: np.ndarray, m: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Condense out the DOFs without mass (the lumped beam's slopes) exactly: K on the rest, and the map from the rest
    to every DOF (None when nothing was massless)."""
    massless = np.all(np.abs(m) == 0.0, axis=0)
    if not massless.any():
        return k, None
    keep = ~massless
    t = np.zeros((k.shape[0], int(keep.sum())), dtype=k.dtype)
    t[keep] = np.eye(t.shape[1])
    t[massless] = -np.linalg.solve(k[np.ix_(massless, massless)], k[np.ix_(massless, keep)])
    return t.conj().T @ k @ t, t


def dispersion(c: Continuum, h: float, scheme: Scheme, kappa: np.ndarray) -> np.ndarray:
    """The endless mesh's frequencies (rad/s) at each wavenumber κ (rad/m), a column per branch, lowest first (the
    beam's consistent and mixed masses give a second, the slopes', branch). From one node's equation with its
    neighbours a phase e^{±iκh} away (Bloch)."""
    ke, me = element_matrices(c, h, scheme)
    d = c.dofs_per_node
    a, b = slice(0, d), slice(d, 2 * d)
    out = []
    for q in np.atleast_1d(np.asarray(kappa, dtype=float)):
        z = np.exp(1j * q * h)
        k = ke[a, a] + ke[b, b] + ke[a, b] * z + ke[b, a] / z
        m = me[a, a] + me[b, b] + me[a, b] * z + me[b, a] / z
        k, t = _condense(k, m)
        if t is not None:
            m = t.conj().T @ m @ t
        lam = scipy.linalg.eigvalsh(k, m) if k.shape[0] > 1 else np.real(k / m).ravel()
        out.append(np.sqrt(np.clip(np.real(lam), 0.0, None)))
    return np.array(out)


def string_dispersion(c: Continuum, h: float, scheme: Scheme, kappa) -> np.ndarray:
    """The string's or bar's mesh dispersion in closed form, ω (rad/s): see the module's notes."""
    s = 1.0 - np.cos(np.asarray(kappa, dtype=float) * h)
    return c.speed / h * np.sqrt(2.0 * s / (1.0 - 2.0 * STENCIL[scheme] * s))


# ------------------------------------------------------------- the mesh
@dataclass(frozen=True)
class FEModel:
    """N equal elements: the constrained DOFs dropped, the massless ones condensed out."""

    continuum: Continuum
    elements: int
    scheme: Scheme

    def __post_init__(self) -> None:
        if not 1 <= self.elements <= MAX_ELEMENTS:
            raise ValueError(f"between 1 and {MAX_ELEMENTS} elements")

    @property
    def h(self) -> float:
        return self.continuum.length / self.elements

    @property
    def nodes(self) -> np.ndarray:
        return np.linspace(0.0, self.continuum.length, self.elements + 1)

    def free_dofs(self) -> np.ndarray:
        """The DOFs (node × dofs_per_node + which) the ends leave free."""
        c, d = self.continuum, self.continuum.dofs_per_node
        n = (self.elements + 1) * d
        held = set()
        for node, end in ((0, c.left), (self.elements, c.right)):
            if end in (End.FIXED, End.PINNED, End.CLAMPED):
                held.add(node * d)
            if end is End.CLAMPED:
                held.add(node * d + 1)
        return np.array([i for i in range(n) if i not in held], dtype=int)

    def matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """K and M on every DOF, ends included."""
        c, d = self.continuum, self.continuum.dofs_per_node
        n = (self.elements + 1) * d
        ke, me = element_matrices(c, self.h, self.scheme)
        k = np.zeros((n, n))
        m = np.zeros((n, n))
        for e in range(self.elements):
            s = slice(e * d, (e + 2) * d)
            k[s, s] += ke
            m[s, s] += me
        return k, m

    def modes(self) -> tuple[np.ndarray, np.ndarray]:
        """Every mode's frequency (Hz, ascending, the rigid ones 0) and shape on every DOF (a column each, ends
        included, largest deflection +1)."""
        return _modes(self)

    def field(self, shape: np.ndarray, x: np.ndarray) -> np.ndarray:
        """A shape (on every DOF) between the nodes, as the elements' own shape functions interpolate it."""
        x = np.clip(np.asarray(x, dtype=float), 0.0, self.continuum.length)
        h = self.h
        e = np.minimum((x / h).astype(int), self.elements - 1)
        s = x / h - e
        if self.continuum.dofs_per_node == 1:
            return shape[e] * (1 - s) + shape[e + 1] * s
        w1, t1, w2, t2 = shape[2 * e], shape[2 * e + 1], shape[2 * e + 2], shape[2 * e + 3]
        return (w1 * (1 - 3 * s**2 + 2 * s**3) + t1 * h * (s - 2 * s**2 + s**3)
                + w2 * (3 * s**2 - 2 * s**3) + t2 * h * (-s**2 + s**3))

    def deflections(self, shape: np.ndarray) -> np.ndarray:
        """A shape's deflection at the nodes (the beam's slopes left out)."""
        return shape[::self.continuum.dofs_per_node]


@lru_cache(maxsize=256)  # the convergence plot's meshes as well as the page's
def _modes(model: FEModel) -> tuple[np.ndarray, np.ndarray]:
    k, m = model.matrices()
    free = model.free_dofs()
    if free.size == 0:  # one element between two held ends: nothing moves
        return np.zeros(0), np.zeros((k.shape[0], 0))
    kf, mf = k[np.ix_(free, free)], m[np.ix_(free, free)]
    kc, t = _condense(kf, mf)
    mc = mf if t is None else t.T @ mf @ t
    lam, v = scipy.linalg.eigh(kc, mc)
    if t is not None:
        v = t @ v
    shapes = np.zeros((k.shape[0], v.shape[1]))
    shapes[free] = v
    # largest deflection +1; a mode of the slopes alone (no deflection at any node) by its largest slope
    d = model.continuum.dofs_per_node
    for j in range(shapes.shape[1]):
        w = shapes[::d, j]
        big = w if np.max(np.abs(w)) > 1e-9 * np.max(np.abs(shapes[:, j])) else shapes[:, j]
        shapes[:, j] /= big[np.argmax(np.abs(big))]
    hz = np.sqrt(np.clip(lam, 0.0, None)) / TWO_PI
    hz[:model.continuum.rigid_modes] = 0.0
    hz.setflags(write=False)
    shapes.setflags(write=False)
    return hz, shapes


def flexible_hz(model: FEModel) -> np.ndarray:
    """The mesh's flexible modes' frequencies (Hz), the rigid ones left out."""
    return model.modes()[0][model.continuum.rigid_modes:]


def errors(model: FEModel) -> tuple[np.ndarray, np.ndarray]:
    """Each flexible mode's exact frequency (Hz) and the mesh's relative error, (f_mesh − f) / f."""
    f = flexible_hz(model)
    exact = model.continuum.exact_hz(f.size)
    return exact, f / exact - 1.0


def convergence(c: Continuum, scheme: Scheme, mode: int,
                counts: tuple[int, ...] = CONVERGENCE_COUNTS) -> tuple[np.ndarray, np.ndarray]:
    """Flexible mode `mode`'s (1 the first) relative error against the number of elements: the counts that have the
    mode, and their errors. It depends only on the kind, the ends and the scheme: the rest scales out."""
    unit = Continuum(c.kind, 1.0, c.left, c.right)
    return _convergence(unit, scheme, mode, counts)


@lru_cache(maxsize=64)
def _convergence(c: Continuum, scheme: Scheme, mode: int, counts: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    exact = c.exact_hz(mode)[-1]
    ns, errs = [], []
    for n in counts:
        f = flexible_hz(FEModel(c, n, scheme))
        if f.size < mode:
            continue
        err = f[mode - 1] / exact - 1.0
        # The eigensolver gets λ to round-off of the largest one: below that the error isn't the mesh's.
        if abs(err) < NOISE * (f[-1] / f[mode - 1])**2:
            break
        ns.append(n)
        errs.append(err)
    return np.array(ns, dtype=int), np.array(errs)


def order(counts: np.ndarray, errs: np.ndarray) -> float | None:
    """The error's rate of convergence p (error ∝ N^−p), from the three finest meshes."""
    if counts.size < 3:
        return None
    return float(-np.polyfit(np.log(counts[-3:]), np.log(np.abs(errs[-3:])), 1)[0])


# ------------------------------------------------------------- presets
@dataclass(frozen=True)
class Preset:
    name: str
    continuum: Continuum
    elements: int
    note: str


PRESETS = (
    Preset("Guitar string (low E)",
           Continuum(Kind.STRING, 0.648, End.FIXED, End.FIXED, tension=73.0, line_density=6.40e-3),
           12,
           "A 648 mm scale at 73 N: 82.4 Hz, every harmonic a whole multiple. As 12 beads (lumped) the "
           "harmonics fall flat, more the higher they go: the 6th, at two elements a half wavelength, "
           "is 10.0 % low and the 11th 31 %."),
    Preset("Steel bar, fixed–free",
           Continuum(Kind.BAR, 1.0, End.FIXED, End.FREE, modulus=210e9, density=7850.0, width=0.02, depth=0.02),
           10,
           "A 1 m steel bar held at one end, rung lengthwise: c = 5172 m/s, so a quarter wavelength fits at "
           "1293 Hz and the modes go 1 : 3 : 5. Ten elements: lumped is low and consistent high by the same, "
           "and their average (mixed) far closer than either."),
    Preset("Steel ruler, clamped–free",
           Continuum(Kind.BEAM, 0.3, End.CLAMPED, End.FREE, modulus=210e9, density=7850.0, width=0.025,
                     depth=0.001),
           8,
           "300 mm of a 1 mm steel ruler held over a desk's edge: 9.28 Hz, then 58.2 and 163 Hz, not "
           "whole multiples (a beam's waves are dispersive, ω ∝ κ²). Consistent Hermite beams converge as "
           "N⁻⁴, lumped masses only as N⁻²."),
    Preset("Vibraphone bar, free–free",
           Continuum(Kind.BEAM, 0.3, End.FREE, End.FREE, modulus=69e9, density=2700.0, width=0.04, depth=0.012),
           12,
           "A plain 300 × 12 mm aluminium bar on its nodes: 693 Hz, overtones 2.76 and 5.40 times that "
           "(a real vibraphone's bars are undercut to bring them to 4 and 10). Two rigid modes: it slides "
           "and tips."),
    Preset("Pinned–pinned beam",
           Continuum(Kind.BEAM, 1.0, End.PINNED, End.PINNED, modulus=210e9, density=7850.0, width=0.05,
                     depth=0.01),
           6,
           "Pinned at both ends a beam's modes are sines, the endless mesh's waves exactly, so every mode "
           "lands on the mesh's dispersion curve. Consistent masses give a second, slopes', branch: six "
           "elements have 12 modes, and modes 6 to 12 are on it, 11 % high at the cut-off and 27 % at the top."),
)
