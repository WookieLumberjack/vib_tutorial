"""Seismic response-spectrum analysis of the chain as a shear building: a design spectrum, each mode's share of
the mass, and the modal combination rules.

The *Seismic response spectrum* page. The chain is a building: m1 the first floor on the ground through k1, each
spring a storey's shear stiffness. Every mode is given the damping ratio ζ the spectrum is drawn for (a code's
practice; the chain's own dampers are set aside). `design_sa` is an elastic design spectrum in the shape the
European code uses (a rise to a plateau of 2.5 η a_g S, then 1/T and 1/T² falls, its corners set by the ground
type). `modal` gives each mode's period, participation Γ = φᵀM1/φᵀMφ, effective mass Γ²φᵀMφ (they add up to the
total mass) and its peak floor displacements, storey drifts and shears from the spectrum. `combine` adds the
modes' peaks by ABS, SRSS, NRL or CQC (Der Kiureghian's correlation, which matters when modes are close).
`accelerogram` synthesizes a ground motion whose 5 % spectrum matches the design one, and `time_history` steps the
building on it, mode by mode, exactly, as the check.
"""

from __future__ import annotations

import enum
import functools
import math
from dataclasses import dataclass

import numpy as np
import scipy.linalg

from .model import ChainSystem
from .random_vibration import G, linear
from .shock import Pulse, Record, oscillator, srs

TWO_PI = 2.0 * math.pi

# Ground types (Type 1 spectrum): S, T_B, T_C, T_D (s)
GROUND = {
    "A (rock)": (1.0, 0.15, 0.4, 2.0),
    "B (dense sand, stiff clay)": (1.2, 0.15, 0.5, 2.0),
    "C (medium-dense soil)": (1.15, 0.20, 0.6, 2.0),
    "D (loose soil)": (1.35, 0.20, 0.8, 2.0),
    "E (soft layer on rock)": (1.4, 0.15, 0.5, 2.0),
}
GROUNDS = tuple(GROUND)

SYNTH_FREQS = 72  # components of the synthesized ground motion, log-spaced
SYNTH_BAND = (0.2, 25.0)  # Hz
SYNTH_ROUNDS = 15
RECORD_FS = 100.0  # Hz
RISE, STRONG, DECAY = 2.0, 10.0, 0.4  # envelope: t² up to RISE, flat to STRONG, then e^{−DECAY (t − STRONG)}
RECORD_LENGTH = 22.0  # s
RECORDS = 7  # averaged, as the codes ask for at least seven


def building(storeys: int = 5, mass: float = 1e4, stiffness: float = 5e6, zeta1: float = 0.05) -> ChainSystem:
    """A uniform shear building: `storeys` floors of `mass` kg on storeys of `stiffness` N/m, its dampers in
    proportion to the springs so mode 1 has ζ₁ (on the Simulation page; the spectrum uses its own ζ)."""
    M, _, K = ChainSystem.uniform(storeys, mass, stiffness, 0.0).matrices()
    w = math.sqrt(scipy.linalg.eigh(K, M, eigvals_only=True)[0])
    return ChainSystem.uniform(storeys, mass, stiffness, 2 * zeta1 * stiffness / w)


def with_tank(system: ChainSystem, ratio: float) -> ChainSystem:
    """A light appendage (a rooftop tank) on the top floor, `ratio` of mode 1's modal mass there, tuned to mode 1
    and undamped: it splits mode 1 into two close modes."""
    from .presets import modal_mass, with_absorber

    lin = linear(system)
    M, _, K = lin.matrices()
    f1 = math.sqrt(scipy.linalg.eigh(K, M, eigvals_only=True)[0]) / TWO_PI
    return with_absorber(system, ratio * modal_mass(lin, 0, lin.n - 1), f1)


@dataclass(frozen=True)
class Design:
    """The design spectrum: peak ground acceleration a_g (in g), ground type, damping ratio ζ."""

    pga_g: float = 0.3
    ground: str = GROUNDS[2]
    zeta: float = 0.05

    @property
    def eta(self) -> float:
        """The damping correction √(10/(5 + ζ%)), at least 0.55."""
        return max(0.55, math.sqrt(10.0 / (5.0 + 100.0 * self.zeta)))

    @property
    def corners(self) -> tuple[float, float, float, float]:
        return GROUND[self.ground]


def design_sa(design: Design, T: np.ndarray) -> np.ndarray:
    """S_a(T) in m/s²: a_g S [1 + T/T_B (2.5η − 1)] up to T_B, a_g S 2.5η to T_C, × T_C/T to T_D, × T_C T_D/T²."""
    S, TB, TC, TD = design.corners
    T = np.asarray(T, dtype=float)
    ag = design.pga_g * G * S
    plateau = 2.5 * design.eta
    with np.errstate(divide="ignore", invalid="ignore"):
        sa = np.where(T < TB, 1 + T / TB * (plateau - 1),
                      np.where(T <= TC, plateau,
                               np.where(T <= TD, plateau * TC / T, plateau * TC * TD / (T * T))))
    return ag * sa


@dataclass(frozen=True)
class Modal:
    """Mode by mode (columns): f (Hz), T (s), φ (mass-normalized), Γ, effective mass, S_a, and the peaks: floor
    displacements u (floors × modes), storey drifts, storey shears, base shear."""

    f: np.ndarray
    T: np.ndarray
    phi: np.ndarray
    gamma: np.ndarray
    effective_mass: np.ndarray
    sa: np.ndarray
    displacement: np.ndarray
    drift: np.ndarray
    shear: np.ndarray
    base_shear: np.ndarray
    total_mass: float

    @property
    def mass_share(self) -> np.ndarray:
        return self.effective_mass / self.total_mass


def modal(system: ChainSystem, design: Design) -> Modal:
    lin = linear(system)
    M, _, K = lin.matrices()
    w2, phi = scipy.linalg.eigh(K, M)
    w = np.sqrt(np.clip(w2, 0.0, None))
    gamma = phi.T @ lin.masses  # φ mass-normalized: Γ = φᵀM1
    meff = gamma ** 2
    T = TWO_PI / w
    sa = design_sa(design, T)
    D = gamma * sa / w2  # each mode's peak modal coordinate
    u = phi * D
    drift = np.diff(u, prepend=0.0, axis=0)
    forces = lin.masses[:, None] * phi * gamma * sa
    shear = np.cumsum(forces[::-1], axis=0)[::-1]  # storey j carries the floors at and above it
    return Modal(w / TWO_PI, T, phi, gamma, meff, sa, u, drift, shear, shear[0], float(np.sum(lin.masses)))


class Rule(enum.Enum):
    ABS = "ABS"
    SRSS = "SRSS"
    NRL = "NRL"
    CQC = "CQC"


RULES = tuple(Rule)


def correlation(f: np.ndarray, zeta: float | np.ndarray) -> np.ndarray:
    """Der Kiureghian's ρ_ij for white noise: 8√(ζ_iζ_j)(ζ_i + rζ_j) r^{3/2} / ((1 − r²)² + 4ζ_iζ_j r(1 + r²)
    + 4(ζ_i² + ζ_j²) r²), r = ω_j/ω_i."""
    f = np.asarray(f, dtype=float)
    z = np.broadcast_to(np.asarray(zeta, dtype=float), f.shape)
    zi, zj = z[:, None], z[None, :]
    r = f[None, :] / f[:, None]
    num = 8 * np.sqrt(zi * zj) * (zi + r * zj) * r ** 1.5
    den = (1 - r * r) ** 2 + 4 * zi * zj * r * (1 + r * r) + 4 * (zi * zi + zj * zj) * r * r
    return num / den


def combine(peaks: np.ndarray, rule: Rule, rho: np.ndarray | None = None) -> np.ndarray:
    """Each output's (rows) combined peak from the modes' (columns)."""
    peaks = np.atleast_2d(peaks)
    a = np.abs(peaks)
    if rule is Rule.ABS:
        return a.sum(axis=1)
    if rule is Rule.SRSS:
        return np.sqrt((a * a).sum(axis=1))
    if rule is Rule.NRL:
        top = a.max(axis=1)
        return top + np.sqrt(np.maximum((a * a).sum(axis=1) - top * top, 0.0))
    if rho is None:
        raise ValueError("CQC needs the modes' correlation")
    return np.sqrt(np.maximum(np.einsum("oi,ij,oj->o", peaks, rho, peaks), 0.0))


# ------------------------------------------------------------- a ground motion to the spectrum
@dataclass(frozen=True)
class Accelerogram:
    t: np.ndarray
    a: np.ndarray  # m/s²
    fs: float
    f: np.ndarray  # the components' frequencies, where the match is checked
    spectrum: np.ndarray  # its S_a at f (ζ of the design)

    @property
    def record(self) -> Record:
        return Record(Pulse(), self.fs, self.a, None)


def envelope(t: np.ndarray) -> np.ndarray:
    return np.where(t < RISE, (t / RISE) ** 2, np.where(t <= STRONG, 1.0, np.exp(-DECAY * (t - STRONG))))


def accelerogram(design: Design, seed: int = 1) -> Accelerogram:
    """A ground motion whose ζ-damped spectrum matches the design spectrum: SYNTH_FREQS sines at random phases
    under a rise–strong–decay envelope, each sine's amplitude scaled round after round by the target over the
    spectrum at its frequency."""
    rng = np.random.default_rng(seed)
    fs = RECORD_FS
    t = np.arange(int(RECORD_LENGTH * fs)) / fs
    f = np.geomspace(*SYNTH_BAND, SYNTH_FREQS)
    phase = rng.uniform(0, TWO_PI, f.size)
    target = design_sa(design, 1 / f)
    amp = 0.2 * target
    env = envelope(t)
    waves = np.sin(TWO_PI * f[:, None] * t[None, :] + phase[:, None])
    best = None
    for _ in range(SYNTH_ROUNDS):
        a = env * (amp @ waves)
        sa = srs(Record(Pulse(), fs, a, None), f, design.zeta).maximax
        worst = float(np.max(np.abs(np.log(sa / target))))
        if best is None or worst < best[0]:
            best = (worst, a, sa)
        amp *= (target / sa) ** 0.8  # a little short of the full step: neighbouring sines interfere
    _, a, sa = best
    return Accelerogram(t, a, fs, f, sa)


@dataclass(frozen=True)
class History:
    """The building on the ground motion: floor displacements relative to the ground (time × floors), and the peaks
    of each floor's displacement, each storey's drift and shear, and the base shear."""

    t: np.ndarray
    displacement: np.ndarray
    peak_displacement: np.ndarray
    peak_drift: np.ndarray
    peak_shear: np.ndarray
    base_shear: np.ndarray  # history

    @property
    def peak_base_shear(self) -> float:
        return float(np.max(np.abs(self.base_shear)))


def time_history(system: ChainSystem, design: Design, ground: Accelerogram, tail: float = 4.0) -> History:
    """Modal superposition with every mode at the design's ζ, each mode's coordinate Γ_r z_r(t) from its own
    oscillator under −a_g (exact for the record's linear segments)."""
    lin = linear(system)
    m = modal(lin, design)
    rec = ground.record
    u = None
    for r in range(m.f.size):
        osc = oscillator(rec, float(m.f[r]), design.zeta, tail)
        q = m.gamma[r] * osc.displacement  # z'' + 2ζωz' + ω²z = −a_g
        part = np.outer(q, m.phi[:, r])
        u = part if u is None else u + part
        t = osc.t
    drift = np.diff(u, prepend=0.0, axis=1)
    shear = drift * lin.stiffness
    return History(t, u, np.max(np.abs(u), axis=0), np.max(np.abs(drift), axis=0),
                   np.max(np.abs(shear), axis=0), shear[:, 0])


@functools.lru_cache(maxsize=8)
def records(design: Design, count: int = RECORDS) -> tuple[Accelerogram, ...]:
    """`count` ground motions to the design spectrum (seeds 1 … count); they depend on the spectrum alone, so
    they're kept while only the building changes."""
    return tuple(accelerogram(design, seed) for seed in range(1, count + 1))


@dataclass(frozen=True)
class Suite:
    """The building on each record, and the mean of their peaks (a code's design value from a set of records)."""

    histories: tuple[History, ...]

    def _mean(self, name: str) -> np.ndarray:
        return np.mean([getattr(h, name) for h in self.histories], axis=0)

    @property
    def displacement(self) -> np.ndarray:
        return self._mean("peak_displacement")

    @property
    def drift(self) -> np.ndarray:
        return self._mean("peak_drift")

    @property
    def shear(self) -> np.ndarray:
        return self._mean("peak_shear")

    @property
    def base_shear(self) -> float:
        return float(np.mean([h.peak_base_shear for h in self.histories]))


def suite(system: ChainSystem, design: Design, count: int = RECORDS) -> Suite:
    return Suite(tuple(time_history(system, design, g) for g in records(design, count)))
