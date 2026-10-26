"""Torsional vibration of a drive train: discs on shafts, and gear meshes between them.

The train is a line of discs (inertia J, and an optional damper to ground: a fan's or
pump's load slope, windage), each joined to the next by a shaft (stiffness k, damping c)
or by a gear mesh. Each disc's angle θ is measured in its own shaft's direction of
turning, so a mesh with z_in teeth on the motor's side and z_out on the far side turns
the far shaft at s = z_in / z_out times the near one's speed, whichever way it turns.

A mesh keeps both gears' inertias and joins them by a spring along the line of action:
its stretch is δ = r_in θ_in − r_out θ_out − e, with r the base radii (r_out = r_in z_out
/ z_in) and e the transmission error, the teeth's departure from perfect involutes, here
one harmonic at the tooth-pass frequency (z_in times the near shaft's speed). The force
k_m δ + c_m δ' acts on the near gear as −r_in F and on the far one as +r_out F, so the
mesh stiffens the pair by k_m [r_in², −r_in r_out; −r_in r_out, r_out²].

Referred to the motor's shaft (θ_i = s_i ψ_i, s_i the product of the ratios on the way),
the train is a plain chain: inertias J s², shafts k s², and each mesh a shaft of stiffness
k_m (r_in s_in)², since r_out s_out = r_in s_in. That is only a change of coordinates, so
its frequencies are the geared train's exactly (`referred`; a test checks). Making the
mesh rigid as well merges the two gears into one inertia, J_in s_in² + J_out s_out²: the
textbook n² reflection, which loses the mesh's own mode (`rigid_gears`).

The modes come from K φ = ω² M φ, the first a rigid-body one at 0 Hz (the whole train
turning), and each mode's damping ratio from φᵀCφ / (2ω φᵀMφ). Nothing here depends on
the speed (no gyroscopic terms), so on the interference (Campbell) diagram the natural
frequencies are flat lines, and a speed-proportional excitation (an engine's firing
orders, a reciprocating compressor's, a mesh's tooth-pass) a line through the origin: a
torsional critical speed is where they cross (`criticals`). The steady response to each
order at each speed is one complex solve, (K − ω²M + iωC) Θ = T (`forced_response`).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np
import scipy.linalg

TWO_PI = 2.0 * math.pi
RIGID_HZ = 1e-6  # a mode below this (relative to the highest) is the rigid-body one


@dataclass(frozen=True)
class Disc:
    """An inertia on the train (kg·m²), with an optional damper to ground (N·m·s/rad)."""

    name: str
    inertia: float
    damping: float = 0.0


@dataclass(frozen=True)
class Shaft:
    """A shaft or coupling between two discs: torsional stiffness (N·m/rad) and damping (N·m·s/rad)."""

    stiffness: float
    damping: float = 0.0
    name: str = "Shaft"


@dataclass(frozen=True)
class Mesh:
    """A gear mesh: teeth on the motor's side and the far side, the near gear's base radius (m), the
    mesh's stiffness (N/m) and damping (N·s/m) along the line of action, and the transmission error's
    amplitude (m) at the tooth-pass frequency."""

    teeth_in: int
    teeth_out: int
    base_radius: float
    stiffness: float
    damping: float = 0.0
    error: float = 0.0
    name: str = "Gear mesh"

    @property
    def ratio(self) -> float:
        """The far shaft's speed over the near one's."""
        return self.teeth_in / self.teeth_out

    @property
    def radius_out(self) -> float:
        return self.base_radius * self.teeth_out / self.teeth_in


Link = Shaft | Mesh


@dataclass(frozen=True)
class Excitation:
    """A pulsating torque on one disc: (order, amplitude in N·m) pairs, each order of that disc's own speed."""

    disc: int
    harmonics: tuple[tuple[float, float], ...]
    name: str = "Torque"


@dataclass(frozen=True)
class DriveTrain:
    """Discs in order from the motor, a link between each pair, and the torque that drives it."""

    discs: tuple[Disc, ...]
    links: tuple[Link, ...]
    excitation: Excitation | None = None

    def __post_init__(self) -> None:
        if len(self.discs) < 2:
            raise ValueError("a drive train needs two discs")
        if len(self.links) != len(self.discs) - 1:
            raise ValueError("one link between each pair of discs")
        for d in self.discs:
            if not d.inertia > 0:
                raise ValueError(f"{d.name}: inertia must be positive")
            if d.damping < 0:
                raise ValueError(f"{d.name}: damping can't be negative")
        for link in self.links:
            if not link.stiffness > 0:
                raise ValueError(f"{link.name}: stiffness must be positive")
            if link.damping < 0:
                raise ValueError(f"{link.name}: damping can't be negative")
            if isinstance(link, Mesh):
                if link.teeth_in < 1 or link.teeth_out < 1:
                    raise ValueError(f"{link.name}: a gear needs teeth")
                if not link.base_radius > 0:
                    raise ValueError(f"{link.name}: base radius must be positive")
        if self.excitation is not None and not 0 <= self.excitation.disc < len(self.discs):
            raise ValueError("the excitation's disc isn't on the train")

    def with_(self, **changes) -> DriveTrain:
        return dataclasses.replace(self, **changes)

    # ------------------------------------------------------------- editing
    def with_disc(self, i: int, disc: Disc) -> DriveTrain:
        return self.with_(discs=self.discs[:i] + (disc,) + self.discs[i + 1:])

    def with_link(self, i: int, link: Link) -> DriveTrain:
        return self.with_(links=self.links[:i] + (link,) + self.links[i + 1:])

    def inserted(self, i: int) -> DriveTrain:
        """A copy of disc i added after it, joined by a copy of the shaft on its far side (or near side)."""
        shafts = [link for link in self.links if isinstance(link, Shaft)]
        near = self.links[i] if i < len(self.links) else self.links[i - 1]
        shaft = near if isinstance(near, Shaft) else (shafts[0] if shafts else Shaft(1e5))
        d = self.discs[i]
        exc = self.excitation
        if exc is not None and exc.disc > i:
            exc = dataclasses.replace(exc, disc=exc.disc + 1)
        return DriveTrain(self.discs[:i + 1] + (dataclasses.replace(d, name=f"{d.name} (2)"),) + self.discs[i + 1:],
                          self.links[:i] + (shaft,) + self.links[i:], exc)

    def removed(self, i: int) -> DriveTrain:
        """Without disc i and the link on its far side (the last disc: the link on its near side)."""
        j = min(i, len(self.links) - 1)
        exc = self.excitation
        if exc is not None:
            exc = None if exc.disc == i else dataclasses.replace(exc, disc=exc.disc - (exc.disc > i))
        return DriveTrain(self.discs[:i] + self.discs[i + 1:], self.links[:j] + self.links[j + 1:], exc)

    # ------------------------------------------------------------- model
    @property
    def n(self) -> int:
        return len(self.discs)

    @property
    def speeds(self) -> np.ndarray:
        """Each disc's speed over the motor's (the first disc's)."""
        s = np.ones(self.n)
        for i, link in enumerate(self.links):
            s[i + 1] = s[i] * (link.ratio if isinstance(link, Mesh) else 1.0)
        return s

    def link_terms(self, i: int) -> tuple[np.ndarray, float, float]:
        """Link i's twist as w · (θ_i, θ_i+1) (a mesh's stretch, in m), its stiffness and damping."""
        link = self.links[i]
        if isinstance(link, Mesh):
            return np.array([link.base_radius, -link.radius_out]), link.stiffness, link.damping
        return np.array([1.0, -1.0]), link.stiffness, link.damping

    def matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """M, C, K on the discs' angles."""
        n = self.n
        m = np.diag([d.inertia for d in self.discs])
        c = np.diag([d.damping for d in self.discs])
        k = np.zeros((n, n))
        for i in range(n - 1):
            w, ki, ci = self.link_terms(i)
            ww = np.outer(w, w)
            k[i:i + 2, i:i + 2] += ki * ww
            c[i:i + 2, i:i + 2] += ci * ww
        return m, c, k

    def referred(self) -> DriveTrain:
        """The same train referred to the motor's shaft: a plain chain with the same frequencies."""
        s = self.speeds
        discs = tuple(dataclasses.replace(d, inertia=d.inertia * s[i]**2, damping=d.damping * s[i]**2)
                      for i, d in enumerate(self.discs))
        links = []
        for i, link in enumerate(self.links):
            if isinstance(link, Mesh):
                r = link.base_radius * s[i]
                links.append(Shaft(link.stiffness * r * r, link.damping * r * r, link.name))
            else:
                links.append(Shaft(link.stiffness * s[i]**2, link.damping * s[i]**2, link.name))
        return DriveTrain(discs, tuple(links))

    def rigid_gears(self) -> DriveTrain:
        """Referred to the motor's shaft with every mesh rigid: each pair of gears one inertia (the n²
        reflection). Without a mesh, the referred train itself."""
        ref = self.referred()
        discs = [ref.discs[0]]
        links = []
        for i, link in enumerate(self.links):
            d = ref.discs[i + 1]
            if isinstance(link, Mesh):
                last = discs[-1]
                discs[-1] = Disc(f"{last.name} + {d.name}", last.inertia + d.inertia, last.damping + d.damping)
            else:
                links.append(ref.links[i])
                discs.append(d)
        if len(discs) < 2:
            raise ValueError("with the gears rigid nothing is left to twist")
        return DriveTrain(tuple(discs), tuple(links))

    @property
    def meshes(self) -> list[int]:
        return [i for i, link in enumerate(self.links) if isinstance(link, Mesh)]


# ------------------------------------------------------------------ modes
@dataclass(frozen=True)
class TorsionalMode:
    freq_hz: float
    zeta: float
    shape: np.ndarray  # referred to the motor's shaft (θ_i / s_i), largest swing +1
    twist: np.ndarray  # each link's share of the mode's strain energy

    @property
    def link(self) -> int:
        """The link that twists most."""
        return int(np.argmax(self.twist))


def modes(train: DriveTrain) -> list[TorsionalMode]:
    """The flexible modes, slowest first (the rigid-body mode left out)."""
    m, c, k = train.matrices()
    w2, phi = scipy.linalg.eigh(k, m)
    w2 = np.clip(w2, 0.0, None)
    top = math.sqrt(w2[-1])
    s = train.speeds
    out = []
    for j in range(train.n):
        w = math.sqrt(w2[j])
        if w <= RIGID_HZ * top:
            continue
        v = phi[:, j]
        zeta = float(v @ c @ v) / (2.0 * w * float(v @ m @ v))
        shape = v / s
        shape = shape / shape[np.argmax(np.abs(shape))]
        energy = np.empty(train.n - 1)
        for i in range(train.n - 1):
            wi, ki, _ = train.link_terms(i)
            energy[i] = ki * float(wi @ v[i:i + 2]) ** 2
        out.append(TorsionalMode(w / TWO_PI, zeta, shape, energy / energy.sum()))
    return out


def natural_hz(train: DriveTrain) -> np.ndarray:
    return np.array([md.freq_hz for md in modes(train)])


# ------------------------------------------------------------------ excitation
@dataclass(frozen=True)
class Source:
    """One speed-proportional excitation: its order of the motor's speed and what it does."""

    label: str
    order: float  # of the motor's speed
    disc: int | None = None  # a torque (N·m) on this disc...
    torque: float = 0.0
    mesh: int | None = None  # ... or this mesh's transmission error (m)
    error: float = 0.0


def sources(train: DriveTrain) -> list[Source]:
    """The excitation's orders and each mesh's tooth-pass, each as an order of the motor's speed."""
    s = train.speeds
    out = []
    exc = train.excitation
    if exc is not None:
        for order, amplitude in exc.harmonics:
            if order > 0 and amplitude > 0:
                out.append(Source(f"{exc.name} {order:g}×", order * s[exc.disc], disc=exc.disc, torque=amplitude))
    for i in train.meshes:
        mesh = train.links[i]
        if mesh.error > 0:
            out.append(Source(f"Tooth pass ({mesh.name})", mesh.teeth_in * s[i], mesh=i, error=mesh.error))
    return out


def order_lines(train: DriveTrain) -> list[tuple[str, float]]:
    """Every order the interference diagram draws: the sources', then 1X and 2X of each shaft's speed
    (each shaft named after the last disc on it, or with one shaft, the running speed's)."""
    out = [(src.label, src.order) for src in sources(train)]
    s = train.speeds
    shafts: dict[float, str] = {}
    for i, d in enumerate(train.discs):
        shafts[float(s[i])] = d.name
    if len(shafts) == 1:
        shafts = {1.0: "running speed"}
    for speed, name in shafts.items():
        for k in (1, 2):
            order = k * speed
            if not any(math.isclose(order, o, rel_tol=1e-9) for _, o in out):
                out.append((f"{k}× {name}", order))
    return out


@dataclass(frozen=True)
class Critical:
    mode: int  # index into modes()
    label: str
    order: float
    rpm: float  # the motor's speed
    freq_hz: float


def criticals(train: DriveTrain, top_rpm: float, lines: list[tuple[str, float]] | None = None) -> list[Critical]:
    """Where each order line crosses each natural frequency, up to top_rpm of the motor, slowest first."""
    out = []
    for j, md in enumerate(modes(train)):
        for label, order in (lines if lines is not None else order_lines(train)):
            rpm = 60.0 * md.freq_hz / order
            if rpm <= top_rpm:
                out.append(Critical(j, label, order, rpm, md.freq_hz))
    return sorted(out, key=lambda c: c.rpm)


def separation(critical_rpm: float, operating_rpm: float) -> float:
    """How far a critical is from the operating speed, as a fraction of it."""
    return abs(critical_rpm - operating_rpm) / operating_rpm


# ------------------------------------------------------------------ steady response
@dataclass(frozen=True)
class ForcedResponse:
    rpm: np.ndarray
    sources: list[Source]
    torque: np.ndarray  # (source, link, speed): each link's dynamic torque amplitude (N·m, on its near disc)
    angle: np.ndarray  # (source, disc, speed): each disc's angle amplitude (rad)

    def total(self) -> np.ndarray:
        """(link, speed): the amplitudes summed, the peak when every order's swing lines up."""
        return self.torque.sum(axis=0) if self.torque.size else np.zeros(self.torque.shape[1:])


def forced_response(train: DriveTrain, rpm: np.ndarray) -> ForcedResponse:
    """Each source's steady response at each motor speed: (K − ω²M + iωC) Θ = T at ω = order × Ω."""
    rpm = np.asarray(rpm, dtype=float)
    m, c, k = train.matrices()
    srcs = sources(train)
    n, nl = train.n, train.n - 1
    torque = np.zeros((len(srcs), nl, rpm.size))
    angle = np.zeros((len(srcs), n, rpm.size))
    terms = [train.link_terms(i) for i in range(nl)]
    for a, src in enumerate(srcs):
        for b, speed in enumerate(rpm):
            w = src.order * speed * TWO_PI / 60.0
            if w <= 0:
                continue
            z = k - w * w * m + 1j * w * c
            f = np.zeros(n, dtype=complex)
            if src.disc is not None:
                f[src.disc] = src.torque
            stretch = 0.0
            if src.mesh is not None:
                wi, ki, ci = terms[src.mesh]
                f[src.mesh:src.mesh + 2] += (ki + 1j * w * ci) * src.error * wi
                stretch = src.error
            theta = np.linalg.solve(z, f)
            angle[a, :, b] = np.abs(theta)
            for i, (wi, ki, ci) in enumerate(terms):
                twist = wi @ theta[i:i + 2] - (stretch if i == src.mesh else 0.0)
                torque[a, i, b] = abs((ki + 1j * w * ci) * twist * wi[0])  # a mesh's force times r_in
    return ForcedResponse(rpm, srcs, torque, angle)


def two_inertia_hz(j1: float, j2: float, k: float) -> float:
    """Two discs on one shaft: the textbook f = √(k (J1 + J2) / (J1 J2)) / 2π."""
    return math.sqrt(k * (j1 + j2) / (j1 * j2)) / TWO_PI


def response_speeds(train: DriveTrain, top_rpm: float, points: int = 800) -> np.ndarray:
    """Motor speeds to plot the forced response at: an even spread, packed round each critical (over a
    few half-power widths), so a lightly damped peak isn't missed."""
    rpm = [np.linspace(top_rpm / points, top_rpm, points)]
    ms = modes(train)
    for c in criticals(train, top_rpm, [(s.label, s.order) for s in sources(train)]):
        width = max(ms[c.mode].zeta, 1e-4)
        rpm.append(c.rpm * (1.0 + width * np.linspace(-6.0, 6.0, 61)))
    rpm = np.unique(np.concatenate(rpm))
    return rpm[(rpm > 0) & (rpm <= top_rpm)]


# ------------------------------------------------------------------ presets
@dataclass(frozen=True)
class TorsionalPreset:
    name: str
    train: DriveTrain
    operating_rpm: float  # the motor's
    top_rpm: float  # the speed range shown
    note: str


FAN = DriveTrain((Disc("Motor", 2.0), Disc("Fan", 10.0, 2.0)), (Shaft(1.8e5, 20.0, "Coupling"),),
                 Excitation(1, ((2.0, 50.0),), "Misalignment"))
COMPRESSOR = DriveTrain(
    (Disc("Motor", 10.0), Disc("Bull gear", 3.0), Disc("Pinion", 0.05), Disc("Compressor", 0.3, 0.05)),
    (Shaft(5e5, 20.0, "Low-speed coupling"), Mesh(120, 30, 0.3, 2e9, 2e4, 2e-6),
     Shaft(2e4, 2.0, "High-speed coupling")),
    Excitation(3, ((1.0, 20.0),), "Compressor"))
GENSET = DriveTrain(
    (Disc("Pulley", 0.3),) + tuple(Disc(f"Crank {i + 1}", 0.5) for i in range(6))
    + (Disc("Flywheel", 10.0), Disc("Generator", 15.0, 5.0)),
    (Shaft(3e6, 200.0, "Nose"),) + tuple(Shaft(3e6, 200.0, f"Crank {i + 1}–{i + 2}") for i in range(5))
    + (Shaft(3e6, 200.0, "Crank–flywheel"), Shaft(1e5, 100.0, "Flexible coupling")),
    Excitation(3, ((3.0, 3000.0), (6.0, 300.0)), "Firing"))

PRESETS = (
    TorsionalPreset(
        "Motor and fan: two inertias", FAN, 1480.0, 3600.0,
        "One mode, the motor and the fan twisting the coupling against each other at "
        "√(k (J₁ + J₂) / J₁J₂) / 2π = 52.3 Hz. A misaligned coupling's 2× torque meets it at 1569 rpm, "
        "6 % above the running speed: inside a 10 % margin. A softer coupling moves it below."),
    TorsionalPreset(
        "Geared compressor", COMPRESSOR, 1490.0, 1800.0,
        "A 4:1 speed-up gearbox. The two lower modes twist the couplings; the mesh's own, 2688 Hz, "
        "twists the teeth. The gears made rigid (the n² reflection) keep the lower two to within 0.1 % "
        "but lose that one, which the tooth-pass meets at 1344 rpm, 10 % below the running speed."),
    TorsionalPreset(
        "Diesel generator set", GENSET, 1500.0, 1800.0,
        "A six-cylinder four-stroke engine's firing orders, 3× and 6× (lumped on crank 3), against the "
        "crankshaft's modes and the flexible coupling's (18.9 Hz). The coupling's mode is far below the "
        "running speed, passed on the way up at 378 rpm (3×); the crankshaft's first, 97.3 Hz, at 973 rpm (6×)."),
)
