"""A pulse on a string, a bar or a beam, on the mesh and on the continuum, and its reflections.

A smooth pulse (a raised cosine, its full width a fraction of the length) starts on the
medium, either at rest (it splits into two halves, each going its own way) or, on a string
or bar, travelling right (its velocity −c u₀′: d'Alembert's right-going wave alone).

On the **continuum** a string's or bar's waves keep their shape: u = r(x − ct) + l(x + ct).
The reference carries the two exactly, as a digital waveguide: each is sampled at cells
one time step τ of travel long and moved on a cell a step, which is exact (to the sampling
of the pulse) at any τ. Where they meet something they scatter in closed form, with Z the
impedance, √(Tμ) or ρcA:

* an end, reflected = ρ × incident: fixed −1 (the pulse comes back upside down), free +1,
  a dashpot c_d (1 − r)/(1 + r) with r = c_d/Z: matched (r = 1) it takes the whole wave;
* a step from Z₁ to Z₂: reflected (Z₁ − Z₂)/(Z₁ + Z₂), transmitted 2Z₁/(Z₁ + Z₂). Into a
  heavier string (or a thicker bar) the reflection comes back inverted.

A beam's bending waves are dispersive, so the continuum's pulse spreads too; its reference
is the sum of its exact modes, each turning at its own frequency (ends clamped, pinned or
free: no dashpot or step on a beam here).

The **mesh** is the page's: N elements and one way of sharing out the mass, its end a
dashpot or its section stepped as the continuum's (the step at the nearest node). It is
stepped exactly, z ← e^{AΔ} z, a frame's Δ at a time, from the pulse sampled at its nodes
(and, a beam's, its slopes). Its short waves travel slower than they should (lumped) or
faster (consistent), and none shorter than two elements travels at all, so the pulse
trails (or leads) ripples the continuum doesn't have, and a matched dashpot, matched to
the continuum's long waves, reflects the mesh's short ones.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np
import scipy.linalg

from .continuum import (
    Continuum,
    End,
    FEModel,
    Kind,
    Scheme,
    _condense,
    element_matrices,
)

FRAMES_PER_TRANSIT = 240  # frames for the pulse to cross the length, at 1× playback
CELLS_PER_FRAME = 8  # the waveguide's steps a frame at 1×: ~2000 cells along the length
PLAYBACK = (0.25, 0.5, 1.0, 2.0)  # × the base rate; each a whole number of waveguide steps
REFERENCE_POINTS = 1201  # where the beam's exact solution is drawn
MAX_BEAM_MODES = 800
BEAM_REACH = 40.0  # the beam's modes are summed up to κ = BEAM_REACH / width: the pulse to ~0.2 %


@dataclass(frozen=True)
class Waves:
    """The pulse (full width and centre as fractions of the length; travelling right or released at rest), the mesh's
    mass, and what the wave meets: a dashpot on the right end (c_d / Z, None: the medium's own end), a section
    step (where, as a fraction of the length, None: none, and Z₂/Z₁)."""

    width: float = 0.125
    at: float = 0.25
    moving: bool = True
    scheme: Scheme = Scheme.LUMPED
    dashpot: float | None = None
    step_at: float | None = None
    step_ratio: float = 2.0

    def __post_init__(self) -> None:
        if not 0 < self.width <= 1:
            raise ValueError("the pulse's width is a fraction of the length")
        if not 0 <= self.at <= 1:
            raise ValueError("the pulse's centre is on the length")
        if self.dashpot is not None and self.dashpot < 0:
            raise ValueError("a dashpot can't be negative")
        if self.step_at is not None and not 0 < self.step_at < 1:
            raise ValueError("the step is inside the length")
        if not self.step_ratio > 0:
            raise ValueError("the impedance ratio must be positive")

    def with_(self, **changes) -> Waves:
        return dataclasses.replace(self, **changes)


def pulse(x: np.ndarray, centre: float, width: float) -> np.ndarray:
    """A raised cosine, 1 at the centre, 0 beyond half its width either side."""
    s = (np.asarray(x, dtype=float) - centre) / (0.5 * width)
    return np.where(np.abs(s) < 1.0, 0.5 * (1.0 + np.cos(math.pi * s)), 0.0)


def pulse_slope(x: np.ndarray, centre: float, width: float) -> np.ndarray:
    s = (np.asarray(x, dtype=float) - centre) / (0.5 * width)
    return np.where(np.abs(s) < 1.0, -0.5 * math.pi / (0.5 * width) * np.sin(math.pi * s), 0.0)


def end_reflection(end: End, dashpot: float | None = None) -> float:
    """What an end returns of a displacement wave: fixed −1, free +1, a dashpot (1 − r)/(1 + r)."""
    if dashpot is not None:
        return (1.0 - dashpot) / (1.0 + dashpot)
    return 1.0 if end is End.FREE else -1.0


def step_coefficients(ratio: float) -> tuple[float, float]:
    """A step from Z₁ to Z₂ = ratio Z₁: the reflected and transmitted parts of a wave arriving from the first side."""
    return (1.0 - ratio) / (1.0 + ratio), 2.0 / (1.0 + ratio)


class WaveRun:
    """The pulse on the continuum and on the mesh, stepped a frame at a time."""

    def __init__(self, continuum: Continuum, elements: int, waves: Waves, playback: float = 1.0) -> None:
        c = continuum
        if c.kind is Kind.BEAM and (waves.dashpot is not None or waves.step_at is not None):
            raise ValueError("a beam's waves meet only its ends here")
        self.continuum = c
        self.elements = elements
        self.waves = waves
        self.moving = waves.moving and c.kind is not Kind.BEAM
        self.length = c.length
        # the second side of a step: a string's mass per length × q² (Z × q, c ÷ q), a bar's section × q
        q = waves.step_ratio
        if waves.step_at is None:
            self.far = c
        elif c.kind is Kind.STRING:
            self.far = c.with_(line_density=c.line_density * q * q)
        else:
            self.far = c.with_(width=c.width * q)
        h = c.length / elements
        self.step_node = None if waves.step_at is None else int(np.clip(round(waves.step_at * elements), 1,
                                                                          elements - 1))
        self.step_x = None if self.step_node is None else self.step_node * h
        self.transit = self._transit()
        self.tau = self.transit / (FRAMES_PER_TRANSIT * CELLS_PER_FRAME)  # the waveguide's step
        self.t = 0.0
        self.frames = 0
        self._build_mesh()
        if c.kind is Kind.BEAM:
            self._build_modes()
        else:
            self._build_waveguide()
        self.set_playback(playback)

    # ------------------------------------------------------------- set-up
    def _transit(self) -> float:
        """How long the pulse takes to cross the length: c's for a string or bar (each side's at its own speed), a
        beam's at the group speed 2aκ of the pulse's main wavenumber, κ = 2π / width."""
        c = self.continuum
        if c.kind is Kind.BEAM:
            kappa = 2.0 * math.pi / (self.waves.width * c.length)
            return c.length / (2.0 * c.speed * kappa)
        if self.step_x is None:
            return c.length / c.speed
        return self.step_x / c.speed + (c.length - self.step_x) / self.far.speed

    def set_playback(self, playback: float) -> None:
        """A frame's simulated time, Δ = playback × the base: the mesh's exponential afresh, the waveguide's steps."""
        if playback not in PLAYBACK:
            raise ValueError(f"playback is one of {PLAYBACK}")
        self.playback = playback
        self.cells_per_frame = round(CELLS_PER_FRAME * playback)
        self.dt = self.cells_per_frame * self.tau
        n = self._m.shape[0]
        a = np.zeros((2 * n, 2 * n))
        a[:n, n:] = np.eye(n)
        a[n:, :n] = -np.linalg.solve(self._m, self._k)
        a[n:, n:] = -np.linalg.solve(self._m, self._c)
        self._phi = scipy.linalg.expm(a * self.dt)

    def _build_mesh(self) -> None:
        c, waves = self.continuum, self.waves
        n_el = self.elements
        h = c.length / n_el
        right = End.FREE if waves.dashpot is not None else c.right
        base = FEModel(c.with_(right=right), n_el, waves.scheme)
        self.model = base
        d = c.dofs_per_node
        size = (n_el + 1) * d
        k = np.zeros((size, size))
        m = np.zeros((size, size))
        for e in range(n_el):
            side = self.far if self.step_node is not None and e >= self.step_node else c
            ke, me = element_matrices(side, h, waves.scheme)
            s = slice(e * d, (e + 2) * d)
            k[s, s] += ke
            m[s, s] += me
        cmat = np.zeros((size, size))
        if waves.dashpot is not None:
            last = self.far
            cmat[-d, -d] = waves.dashpot * math.sqrt(last.rigidity * last.mass_per_length)  # r × Z
        free = base.free_dofs()
        self._free = free
        kf, mf, cf = (x[np.ix_(free, free)] for x in (k, m, cmat))
        kc, t = _condense(kf, mf)
        self._t = np.eye(free.size) if t is None else t  # reduced → free DOFs
        self._k = kc
        self._m = self._t.T @ mf @ self._t
        self._c = self._t.T @ cf @ self._t
        self._size = size
        # the pulse at the nodes (a beam's slopes too), its velocity for a right-going wave
        x = base.nodes
        full = np.zeros(size)
        full[::d] = pulse(x, waves.at * c.length, waves.width * c.length)
        vel = np.zeros(size)
        if d == 2:
            full[1::d] = pulse_slope(x, waves.at * c.length, waves.width * c.length)
        if self.moving:
            far = np.zeros(x.size, dtype=bool) if self.step_x is None else x >= self.step_x
            speed = np.where(far, self.far.speed, c.speed)
            vel[::d] = -speed * pulse_slope(x, waves.at * c.length, waves.width * c.length)
        rows = free if t is None else free[np.any(np.abs(mf) > 0, axis=0)]  # the lumped beam's: deflections only
        self._z = np.concatenate([full[rows], vel[rows]])
        self.energy0 = self.energy()

    def _build_waveguide(self) -> None:
        c, waves = self.continuum, self.waves
        sides = [(0.0, self.step_x if self.step_x is not None else c.length, c.speed)]
        if self.step_x is not None:
            sides.append((self.step_x, c.length, self.far.speed))
        xs = []
        for i, (a, b, speed) in enumerate(sides):
            cells = max(1, round((b - a) / (speed * self.tau)))
            xs.append(np.linspace(a, b, cells + 1)[(1 if i else 0):])
        self.guide_x = np.concatenate(xs)
        self._junction = None if self.step_x is None else len(xs[0]) - 1
        u0 = pulse(self.guide_x, waves.at * c.length, waves.width * c.length)
        self._r = u0.copy() if self.moving else 0.5 * u0
        self._l = np.zeros_like(u0) if self.moving else 0.5 * u0
        self._rho_left = end_reflection(c.left)
        self._rho_right = end_reflection(c.right, waves.dashpot)
        self._scatter = step_coefficients(waves.step_ratio)
        self._r_in = 0.0

    def _build_modes(self) -> None:
        c, waves = self.continuum, self.waves
        reach = BEAM_REACH / (waves.width * c.length)
        count = 1
        while count < MAX_BEAM_MODES and c.exact_wavenumbers(count)[-1] < reach:
            count = min(2 * count, MAX_BEAM_MODES)
        kappa = c.exact_wavenumbers(count)
        kappa = kappa[kappa < reach] if np.any(kappa < reach) else kappa[:1]
        x = np.linspace(0.0, c.length, REFERENCE_POINTS)
        shapes = [c.exact_shape(k, x) for k in kappa]
        omega = list(c.omega(kappa))
        rigid = {0: [], 1: [x if c.left is End.PINNED else c.length - x],
                 2: [np.ones_like(x), x - 0.5 * c.length]}[c.rigid_modes]
        shapes = rigid + shapes
        omega = [0.0] * len(rigid) + omega
        phi = np.array(shapes)
        u0 = pulse(x, waves.at * c.length, waves.width * c.length)
        weights = np.full(x.size, x[1] - x[0])
        weights[[0, -1]] *= 0.5
        gram = (phi * weights) @ phi.T  # the rigid shapes aren't orthogonal to each other; the rest are
        self._coef = np.linalg.solve(gram, (phi * weights) @ u0)
        self._phi_modes = phi
        self._omega = np.array(omega)
        self.guide_x = x

    # ------------------------------------------------------------- stepping
    def advance(self, frames: int = 1) -> None:
        for _ in range(frames):
            self._z = self._phi @ self._z
            if self.continuum.kind is not Kind.BEAM:
                for _ in range(self.cells_per_frame):
                    self._guide_step()
            self.t += self.dt
            self.frames += 1

    def _guide_step(self) -> None:
        r, l = self._r, self._l
        j = self._junction
        if j is not None:
            r_in, l_in = r[j - 1], l[j + 1]
        r[1:] = r[:-1].copy()
        l[:-1] = l[1:].copy()
        r[0] = self._rho_left * l[0]
        l[-1] = self._rho_right * r[-1]
        if j is not None:
            refl, trans = self._scatter
            l[j] = refl * r_in + (2.0 - trans) * l_in  # back into the first side, and through from the second
            r[j] = trans * r_in - refl * l_in
            self._r_in = r_in

    # ------------------------------------------------------------- reading
    def exact(self) -> tuple[np.ndarray, np.ndarray]:
        """The continuum's displacement now: positions and values."""
        if self.continuum.kind is Kind.BEAM:
            return self.guide_x, (self._coef * np.cos(self._omega * self.t)) @ self._phi_modes
        u = self._r + self._l
        j = self._junction
        if j is not None and self.frames:
            u = u.copy()
            u[j] = self._r_in + self._l[j]
        return self.guide_x, u

    def mesh_shape(self) -> np.ndarray:
        """The mesh's displacement on every DOF (a beam's slopes too, ends included)."""
        n = self._k.shape[0]
        full = np.zeros(self._size)
        full[self._free] = self._t @ self._z[:n]
        return full

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """The mesh's nodes and their displacement now."""
        return self.model.nodes, self.model.deflections(self.mesh_shape())

    def energy(self) -> float:
        n = self._k.shape[0]
        q, v = self._z[:n], self._z[n:]
        return 0.5 * float(v @ self._m @ v + q @ self._k @ q)

    @property
    def energy_left(self) -> float:
        """The mesh's energy now over its energy at the start."""
        return self.energy() / self.energy0 if self.energy0 > 0 else 1.0


# ------------------------------------------------------------- presets
@dataclass(frozen=True)
class WavePreset:
    name: str
    continuum: Continuum
    elements: int
    waves: Waves
    note: str


_STRING = Continuum(Kind.STRING, 1.0, End.FIXED, End.FIXED, tension=100.0, line_density=0.01)

WAVE_PRESETS = (
    WavePreset("Waves: a pulse on 40 beads", _STRING, 40, Waves(width=0.1, at=0.25),
               "A pulse four beads wide runs along a string at c = 100 m/s. On the string it comes back from "
               "the fixed end upside down and whole; on the beads, after one crossing, it is down to 66 % of "
               "its height and 5 cm behind, the ripples it sheds trailing it: the short waves it is made of go "
               "slower on the beads. Consistent masses make them faster (53 %, ahead), mixed keep 81 %."),
    WavePreset("Waves: a matched end",
               Continuum(Kind.BAR, 1.0, End.FIXED, End.FIXED, modulus=210e9, density=7850.0, width=0.02,
                         depth=0.02),
               20, Waves(width=0.15, at=0.3, dashpot=1.0),
               "A dashpot c = ρcA on the bar's far end is its impedance: the continuum's pulse runs into it and "
               "is gone, as if the bar went on for ever. Matched to the long waves, it isn't to the 20 "
               "elements' short ones, which slow down: 17 % of the mesh's energy comes back (5 % consistent)."),
    WavePreset("Waves: a light string tied to a heavy one", _STRING, 100,
               Waves(width=0.1, at=0.2, step_at=0.5, step_ratio=2.0),
               "Halfway along, the string is four times heavier: twice the impedance, half the speed. The pulse "
               "splits: a third of it comes back upside down (R = −1/3, 1/9 of the energy) and two thirds "
               "go on (T = 2/3), at half the speed and so half as long. There it spans five elements, and the "
               "mesh carries 0.49 of it, not 0.67."),
    WavePreset("Waves: a bending pulse",
               Continuum(Kind.BEAM, 1.0, End.PINNED, End.PINNED, modulus=210e9, density=7850.0, width=0.05,
                         depth=0.01),
               20, Waves(width=0.1, at=0.5, moving=False),
               "A bump released on a beam splits and spreads: bending waves are dispersive (ω = aκ²), the "
               "short ones twice as fast as waves twice as long, so ripples run ahead, on the beam itself. "
               "Half a crossing later the beam's peak is 0.22; 20 lumped elements, slowing the short waves, "
               "still hold 0.26 there (consistent 0.23)."),
)
