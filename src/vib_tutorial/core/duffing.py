"""Duffing oscillator: a mass on a spring that stiffens (or softens) as it stretches.

    m x'' + c x' + k x + k3 x^3 = F cos(phi),   phi' = Omega(t)

k3 > 0 hardens the spring, k3 < 0 softens it. A softening spring's potential
has a hilltop at the escape amplitude sqrt(k / -k3): past it the spring pushes
outward, and a mass carried over it runs away.

Harmonic balance keeps the first harmonic, x = a cos(phi - theta). The cubic's
first harmonic is (3/4) k3 a^3 cos(phi - theta), so the spring acts as one of
stiffness kappa(a) = k + (3/4) k3 a^2, and

    (kappa(a) - m Omega^2) a = F cos(theta),   c Omega a = F sin(theta).

Squared and added, G(s) = s [(kappa - m Omega^2)^2 + c^2 Omega^2] = F^2 with
s = a^2: a cubic in s, so at one frequency there can be one steady amplitude
or three. With three, the middle one is unstable (a saddle of the averaged
equations: their trace is -c/m < 0 and their determinant has the sign of
dG/ds), and the outer two are both stable, so the amplitude a sweep finds
depends on where it came from. A branch ends where dG/ds = 0 (a fold), and the
response jumps from there to the other stable branch. The backbone,
m Omega^2 = kappa(a), is the free oscillation's frequency at amplitude a,
which the curve leans along.

Parametrized by the amplitude, the response is two branches of
Omega^2 at each a (a quadratic), which meet at the curve's tip: that traces
the whole curve, unstable middle included, in one pass (`response_curve`).

The simulator steps the linear part exactly (first-order hold, as the chain's
simulator does) and treats the cubic's force as an input, linear over the step,
whose end value depends on where the step ends: one scalar Newton solve per
step. The step resolves the drive and the spring's stiffness at the largest
amplitude in reach.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import scipy.linalg
import scipy.optimize

TWO_PI = 2.0 * math.pi
MAX_STEP = 0.01  # s
STEPS_PER_PERIOD = 64  # of the drive, or of the stiffest the spring gets
MAX_STEPS_PER_ADVANCE = 100_000
REACH = 1.5  # the step resolves the stiffness at this many times the largest amplitude in sight
NEWTON_ITERATIONS = 12
ESCAPED = 2.0  # a softening spring's mass this many times the escape amplitude out has run away


@dataclass(frozen=True)
class Duffing:
    """The oscillator and the drive's amplitude: SI throughout (kg, N/m, N·s/m, N/m³, N)."""

    mass: float = 1.0
    stiffness: float = 400.0
    damping: float = 1.0
    cubic: float = 1e5  # k3: > 0 hardening, < 0 softening
    force: float = 2.0  # the drive's amplitude F

    def __post_init__(self) -> None:
        if self.mass <= 0:
            raise ValueError("the mass must be positive")
        if self.stiffness < 0 or self.damping < 0 or self.force < 0:
            raise ValueError("stiffness, damping and force must be non-negative")
        if self.cubic < 0 and self.stiffness <= 0:
            raise ValueError("a softening spring needs a linear stiffness to soften")

    def with_(self, **changes) -> Duffing:
        return dataclasses.replace(self, **changes)

    @property
    def natural_hz(self) -> float:
        """The small-amplitude natural frequency (Hz), sqrt(k/m) / 2 pi."""
        return math.sqrt(self.stiffness / self.mass) / TWO_PI

    @property
    def escape(self) -> float:
        """A softening spring's escape amplitude (m), where its force peaks; inf for a hardening one."""
        return math.sqrt(self.stiffness / -self.cubic) if self.cubic < 0 else math.inf

    def restoring(self, x):
        """The spring's force k x + k3 x^3 (N)."""
        return self.stiffness * x + self.cubic * x**3

    def potential(self, x):
        """The spring's energy k x^2 / 2 + k3 x^4 / 4 (J)."""
        return 0.5 * self.stiffness * x**2 + 0.25 * self.cubic * x**4

    def kappa(self, a):
        """The spring's stiffness to the first harmonic at amplitude a: k + (3/4) k3 a^2 (N/m)."""
        return self.stiffness + 0.75 * self.cubic * np.square(a)

    def backbone_hz(self, a):
        """The free oscillation's frequency at amplitude a (Hz), by harmonic balance; NaN past kappa = 0."""
        kap = np.asarray(self.kappa(a), dtype=float)
        with np.errstate(invalid="ignore"):
            return np.where(kap >= 0, np.sqrt(np.maximum(kap, 0.0) / self.mass), np.nan) / TWO_PI

    def amplitude_of(self, energy: float) -> float:
        """The amplitude (m) a free oscillation with this much energy (J) swings to (inf past a hilltop)."""
        k, k3 = self.stiffness, self.cubic
        if energy <= 0:
            return 0.0
        if k3 == 0:
            return math.sqrt(2.0 * energy / k) if k > 0 else math.inf
        disc = 0.25 * k * k + k3 * energy
        if disc < 0:
            return math.inf  # over the softening spring's hilltop
        s = (-0.5 * k + math.sqrt(disc)) / (0.5 * k3)
        return math.sqrt(s) if s >= 0 else math.inf


# ------------------------------------------------------------------ harmonic balance
def _g_terms(osc: Duffing, omega, s):
    """G(s) - F^2 and dG/ds at frequency omega (rad/s), s = a^2."""
    alpha = osc.kappa(np.sqrt(s)) - osc.mass * omega**2
    damp = (osc.damping * omega) ** 2
    return s * (alpha**2 + damp) - osc.force**2, alpha**2 + damp + 1.5 * osc.cubic * s * alpha


def amplitudes(osc: Duffing, freq_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """The steady amplitudes (m) at one frequency, smallest first, and whether each is stable.

    The real positive roots of G(a^2) = F^2, below a softening spring's escape amplitude.
    """
    w = TWO_PI * freq_hz
    alpha = osc.stiffness - osc.mass * w * w
    beta = 0.75 * osc.cubic
    lin = alpha * alpha + (osc.damping * w) ** 2
    if osc.force == 0:
        return np.zeros(1), np.ones(1, dtype=bool)
    if beta == 0:
        s = np.array([osc.force**2 / lin]) if lin > 0 else np.empty(0)
    else:
        roots = np.roots([beta * beta, 2 * alpha * beta, lin, -osc.force**2])
        s = np.sort(roots[(np.abs(roots.imag) <= 1e-9 * np.abs(roots).max()) & (roots.real > 0)].real)
    a = np.sqrt(s)
    a = a[a < osc.escape]
    return a, _g_terms(osc, w, a * a)[1] > 0


def steady_phase(osc: Duffing, freq_hz: float, a: float) -> float:
    """The lag theta (rad) of the steady response x = a cos(phi - theta) of amplitude a at freq_hz."""
    w = TWO_PI * freq_hz
    return math.atan2(osc.damping * w * a, (float(osc.kappa(a)) - osc.mass * w * w) * a)


@dataclass
class Fold:
    """Where a stable branch ends: the response jumps from `amplitude` to `lands_at` (m) at freq_hz."""

    freq_hz: float
    amplitude: float
    lands_at: float

    @property
    def down(self) -> bool:
        return self.lands_at < self.amplitude


@dataclass
class ResponseCurve:
    """The harmonic-balance response: pieces of the curve, each all stable or all unstable, and its folds."""

    pieces: list[tuple[np.ndarray, np.ndarray, bool]]  # (Hz, m, stable); neighbours share an end point
    folds: list[Fold]
    tip: float  # the largest amplitude on the curve (m); inf if it runs off (no damping)

    @property
    def peak(self) -> float:
        """The largest amplitude drawn (m)."""
        return max((float(np.nanmax(a)) for _, a, _ in self.pieces if a.size), default=0.0)

    def band(self) -> tuple[float, float] | None:
        """The frequencies (Hz) between which two stable amplitudes coexist, if any."""
        if len(self.folds) < 2:
            return None
        f = sorted(fold.freq_hz for fold in self.folds)
        return f[0], f[-1]


def _first_root(fun, xs: np.ndarray) -> float | None:
    """The first sign change of fun along the grid xs, refined; None if it has none."""
    values = fun(xs)
    change = np.flatnonzero(np.sign(values[:-1]) * np.sign(values[1:]) <= 0)
    if not change.size:
        return None
    i = change[0]
    if values[i] == 0:
        return float(xs[i])
    return scipy.optimize.brentq(fun, xs[i], xs[i + 1], xtol=1e-14 * xs[i + 1], rtol=1e-12)


def response_curve(osc: Duffing, freq_max_hz: float, points: int = 1200) -> ResponseCurve:
    """The steady response from 0 to about freq_max_hz, by harmonic balance (see the module's notes).

    For each amplitude a, m^2 W^2 + (c^2 - 2 m kappa) W + kappa^2 - F^2 / a^2 = 0 in W = Omega^2.
    Its two roots meet at the tip (the discriminant's first zero), where the curve turns back.
    """
    m, c, F = osc.mass, osc.damping, osc.force
    if F <= 0:
        return ResponseCurve([], [], 0.0)
    w_max = TWO_PI * max(freq_max_hz, osc.natural_hz, 1e-6)

    def disc(a):
        return c**4 - 4 * m * osc.kappa(a) * c**2 + 4 * m * m * F * F / np.square(a)

    a_min = F / (m * (1.5 * w_max) ** 2 + osc.stiffness + abs(osc.cubic) * 1e-30) * 0.5
    if osc.cubic < 0:
        a_cap = 0.999 * osc.escape
    elif osc.cubic > 0:  # where the backbone passes 3 times the range
        a_cap = math.sqrt(max(m * (3 * w_max) ** 2 - osc.stiffness, 0.0) / (0.75 * osc.cubic))
    else:
        a_cap = 50.0 * F / osc.stiffness if osc.stiffness > 0 else 50.0 * F / (m * w_max**2)
    a_cap = max(a_cap, 4 * a_min)
    grid = np.geomspace(a_min, a_cap, 4 * points)
    tip = _first_root(disc, grid)
    a_end = tip if tip is not None else a_cap
    a = np.geomspace(a_min, a_end, points)
    if tip is not None:  # crowd the points toward the tip, where the branches turn
        a = np.unique(np.concatenate([a, a_end * (1 - np.geomspace(1e-7, 1 - a_min / a_end, points // 2))]))
        a[-1] = a_end
    kap = osc.kappa(a)
    b = c * c - 2 * m * kap
    d = np.maximum(disc(a), 0.0)  # b^2 - 4 m^2 (kappa^2 - F^2 / a^2)
    with np.errstate(invalid="ignore"):
        w2_plus = (-b + np.sqrt(d)) / (2 * m * m)
        w2_minus = (-b - np.sqrt(d)) / (2 * m * m)
    left = w2_minus > 0
    branches = []
    if left.any():  # Omega rising from 0, where kappa(a) a = F, to the tip
        w_left, a_left = np.sqrt(w2_minus[left]), a[left]
        a_static = _first_root(lambda y: osc.kappa(y) * y - F, a)
        if a_static is not None and a_static < a_left[0]:
            w_left, a_left = np.concatenate([[0.0], w_left]), np.concatenate([[a_static], a_left])
        branches.append((w_left, a_left))
    right = w2_plus > 0
    if tip is not None and branches:  # one curve: up the left branch, over the tip, down the right
        w = np.concatenate([branches[0][0], np.sqrt(w2_plus[right])[::-1]])
        aa = np.concatenate([branches[0][1], a[right][::-1]])
        branches = [(w, aa)]
    else:
        branches.append((np.sqrt(w2_plus[right])[::-1], a[right][::-1]))

    pieces: list[tuple[np.ndarray, np.ndarray, bool]] = []
    folds: list[Fold] = []
    for w, aa in branches:
        if w.size < 2:
            continue
        stable = _g_terms(osc, w, aa * aa)[1] > 0
        cuts = np.flatnonzero(stable[1:] != stable[:-1]) + 1
        starts = np.concatenate([[0], cuts])
        ends = np.concatenate([cuts, [w.size]])
        for s0, s1 in zip(starts, ends):
            lo = max(s0 - 1, 0)  # share the end point with the piece before, so the line is unbroken
            pieces.append((w[lo:s1] / TWO_PI, aa[lo:s1], bool(stable[s0])))
        for i in cuts:
            f_fold, a_fold = _fold_point(osc, w[i - 1], aa[i - 1], w[i], aa[i])
            others, _ = amplitudes(osc, f_fold)  # the fold's double root, and the branch it jumps to
            if others.size:
                folds.append(Fold(f_fold, a_fold, float(others[np.argmax(np.abs(others - a_fold))])))
    return ResponseCurve(pieces, folds, tip if tip is not None else math.inf)


def _fold_point(osc: Duffing, w0: float, a0: float, w1: float, a1: float) -> tuple[float, float]:
    """The fold between two neighbouring points of the curve, where dG/ds = 0: (Hz, m).

    Found along the amplitude between them, each amplitude's frequency the root of the
    quadratic in Omega^2 nearer the line between the two points.
    """
    m, c, F = osc.mass, osc.damping, osc.force

    def point(a: float) -> float:
        kap = float(osc.kappa(a))
        b = c * c - 2 * m * kap
        d = math.sqrt(max(b * b - 4 * m * m * (kap * kap - F * F / (a * a)), 0.0))
        roots = [math.sqrt(max(r, 0.0)) for r in ((-b + d) / (2 * m * m), (-b - d) / (2 * m * m))]
        guess = w0 + (a - a0) / (a1 - a0) * (w1 - w0) if a1 != a0 else w0
        return min(roots, key=lambda w: abs(w - guess))

    def slope(a: float) -> float:
        return float(_g_terms(osc, point(a), a * a)[1])

    if a0 == a1 or slope(a0) * slope(a1) > 0:
        return 0.5 * (w0 + w1) / TWO_PI, 0.5 * (a0 + a1)
    a = scipy.optimize.brentq(slope, a0, a1, xtol=1e-15, rtol=1e-13)
    return point(a) / TWO_PI, a


def linear_response(osc: Duffing, freq_hz: np.ndarray) -> np.ndarray:
    """The amplitude (m) the spring would give without its cubic term: F / |k - m Omega^2 + i c Omega|."""
    w = TWO_PI * np.asarray(freq_hz, dtype=float)
    with np.errstate(divide="ignore"):
        return osc.force / np.abs(osc.stiffness - osc.mass * w * w + 1j * osc.damping * w)


# ------------------------------------------------------------------ drive
class Sweep(Enum):
    HOLD = "Hold the frequency"
    UP = "Sweep up"
    DOWN = "Sweep down"
    UP_DOWN = "Sweep up, then down"


@dataclass
class Drive:
    """The drive's frequency, held or swept linearly between two frequencies, and its phase phi.

    The phase is integrated (phi += Omega dt, at the middle of each step's frequency),
    so the frequency can change without a jump in the force. A sweep's legs each take
    `sweep_time`; when the last ends the drive holds the frequency it ended at.
    """

    freq_hz: float = 3.0  # held, or where the sweep is now
    start_hz: float = 2.0
    end_hz: float = 6.0
    sweep_time: float = 100.0  # s per leg
    mode: Sweep = Sweep.HOLD
    phase: float = 0.0
    elapsed: float = 0.0  # s into the current leg
    legs: list[str] = field(default_factory=list)  # "up"/"down" still to run, the current first

    def start(self, mode: Sweep) -> None:
        """Start a sweep (from its first leg's start) or hold the frequency where it is."""
        self.mode = mode
        self.elapsed = 0.0
        self.legs = {Sweep.HOLD: [], Sweep.UP: ["up"], Sweep.DOWN: ["down"], Sweep.UP_DOWN: ["up", "down"]}[mode]
        if self.legs:
            self.freq_hz = self.start_hz if self.legs[0] == "up" else self.end_hz

    @property
    def leg(self) -> str:
        """"up", "down", or "hold"."""
        return self.legs[0] if self.legs else "hold"

    @property
    def max_hz(self) -> float:
        """The highest frequency the drive reaches from here."""
        return max(self.freq_hz, self.start_hz, self.end_hz) if self.legs else self.freq_hz

    def frequency_at(self, elapsed: float) -> float:
        """The frequency (Hz) `elapsed` s into the current leg (the held one if none)."""
        if not self.legs:
            return self.freq_hz
        frac = min(max(elapsed / self.sweep_time, 0.0), 1.0)
        if self.legs[0] == "down":
            frac = 1.0 - frac
        return self.start_hz + (self.end_hz - self.start_hz) * frac

    def advance(self, dt: float) -> float:
        """Move on by dt; return the phase's advance over it (rad)."""
        if not self.legs:
            dphi = TWO_PI * self.freq_hz * dt
        else:
            dphi = TWO_PI * self.frequency_at(self.elapsed + 0.5 * dt) * dt
            self.elapsed += dt
            self.freq_hz = self.frequency_at(self.elapsed)
            if self.elapsed >= self.sweep_time - 1e-12:
                self.legs.pop(0)
                self.elapsed = 0.0
                if not self.legs:
                    self.mode = Sweep.HOLD
        self.phase += dphi
        return dphi


# ------------------------------------------------------------------ simulation
@dataclass
class Cycle:
    """One cycle of the drive: the response's first harmonic over it, and the state where it began."""

    t: float  # s, the cycle's end
    freq_hz: float  # 1 / its length
    amplitude: float  # m, of x's first harmonic over the cycle
    lag: float  # rad, behind the force
    leg: str  # the drive's "up", "down" or "hold" over it
    x: float  # m and m/s at the cycle's end (phi = 0 mod 2 pi): the Poincaré section
    v: float


@dataclass
class Samples:
    """Every step of an advance: time, displacement, velocity, force, the drive's frequency."""

    t: np.ndarray
    x: np.ndarray
    v: np.ndarray
    force: np.ndarray
    freq_hz: np.ndarray
    cycles: list[Cycle]


class DuffingSimulator:
    """Steps the oscillator under its drive (see the module's notes)."""

    def __init__(self, osc: Duffing | None = None, drive: Drive | None = None) -> None:
        self.osc = osc or Duffing()
        self.drive = drive or Drive()
        self.t = 0.0
        self.x = 0.0
        self.v = 0.0
        self.escaped = False
        self._cache: dict[tuple, tuple] = {}
        self._tip = 0.0
        self._sums = [0.0, 0.0, 0.0]  # this cycle's integrals of x cos phi, x sin phi, and its length
        self._set_tip()

    def set_system(self, osc: Duffing) -> None:
        """New parameters; the state carries on."""
        self.osc = osc
        self._cache.clear()
        self._set_tip()

    def _set_tip(self) -> None:
        tip = response_curve(self.osc, max(self.drive.max_hz, self.osc.natural_hz, 0.1), points=200).tip
        self._tip = tip if math.isfinite(tip) else 0.0

    def reset(self) -> None:
        self.t = 0.0
        self.x = self.v = 0.0
        self.escaped = False
        self.drive.phase = 0.0
        self._sums = [0.0, 0.0, 0.0]

    def set_state(self, x: float, v: float) -> None:
        self.x, self.v = float(x), float(v)
        self.escaped = False
        self._sums = [0.0, 0.0, 0.0]

    def start_on(self, a: float) -> None:
        """Put the mass on the steady response of amplitude a at the drive's frequency and phase."""
        theta = steady_phase(self.osc, self.drive.freq_hz, a)
        w = TWO_PI * self.drive.freq_hz
        self.set_state(a * math.cos(self.drive.phase - theta), -a * w * math.sin(self.drive.phase - theta))

    @property
    def energy(self) -> float:
        """Kinetic and spring energy (J)."""
        return 0.5 * self.osc.mass * self.v**2 + float(self.osc.potential(self.x))

    def step_size(self) -> float:
        """The step (s): STEPS_PER_PERIOD to the drive's period and to the stiffest the spring gets in reach.

        Rounded down to a power of two, so the discretizations are reused.
        """
        osc = self.osc
        reach = REACH * max(osc.amplitude_of(self.energy) if osc.cubic > 0 else 0.0, self._tip, abs(self.x))
        stiff = osc.stiffness + 3.0 * max(osc.cubic, 0.0) * reach * reach
        f = max(self.drive.max_hz, math.sqrt(stiff / osc.mass) / TWO_PI)
        h = MAX_STEP if f <= 0 else min(MAX_STEP, 1.0 / (STEPS_PER_PERIOD * f))
        return 2.0 ** math.floor(math.log2(h))

    def _discrete(self, h: float) -> tuple[float, ...]:
        """(Phi, g0, g1) as floats: z1 = Phi z0 + g0 u0 + g1 u1 for the force u, linear over the step."""
        key = (h, self.osc.mass, self.osc.stiffness, self.osc.damping)
        if key not in self._cache:
            m = self.osc.mass
            aug = np.zeros((4, 4))
            aug[0, 1] = 1.0
            aug[1, 0] = -self.osc.stiffness / m
            aug[1, 1] = -self.osc.damping / m
            aug[1, 2] = 1.0 / m
            aug[2, 3] = 1.0 / h
            E = scipy.linalg.expm(aug * h)
            g1 = E[:2, 3]
            g0 = E[:2, 2] - g1
            self._cache[key] = (E[0, 0], E[0, 1], E[1, 0], E[1, 1], g0[0], g0[1], g1[0], g1[1])
        return self._cache[key]

    def advance(self, duration: float) -> Samples:
        """Step on by about `duration` s (whole steps, at most MAX_STEPS_PER_ADVANCE); every step's samples."""
        h = self.step_size()
        steps = min(round(duration / h), MAX_STEPS_PER_ADVANCE)
        p00, p01, p10, p11, g0x, g0v, g1x, g1v = self._discrete(h)
        osc, drive = self.osc, self.drive
        F, k3 = osc.force, osc.cubic
        limit = ESCAPED * osc.escape
        x, v, t = self.x, self.v, self.t
        sc, ss, span = self._sums
        ts, xs, vs, fs, freqs = [], [], [], [], []
        cycles: list[Cycle] = []
        phi = drive.phase % TWO_PI
        f0 = F * math.cos(phi)
        for _ in range(steps):
            if self.escaped:
                break
            leg = drive.leg
            dphi = drive.advance(h)
            phi1 = phi + dphi
            f1 = F * math.cos(phi1)
            u0 = f0 - k3 * x * x * x
            px = p00 * x + p01 * v + g0x * u0
            pv = p10 * x + p11 * v + g0v * u0
            x1 = px + g1x * (f1 - k3 * x * x * x)
            for _ in range(NEWTON_ITERATIONS):
                r = x1 - px - g1x * (f1 - k3 * x1 * x1 * x1)
                x1 -= r / (1.0 + 3.0 * g1x * k3 * x1 * x1)
                if abs(r) <= 1e-13 * (abs(x1) + 1e-12):
                    break
            v1 = pv + g1v * (f1 - k3 * x1 * x1 * x1)
            # the first harmonic over the cycle, by the trapezoidal rule
            sc += 0.5 * h * (x * math.cos(phi) + x1 * math.cos(phi1))
            ss += 0.5 * h * (x * math.sin(phi) + x1 * math.sin(phi1))
            span += h
            if phi1 >= TWO_PI:  # a cycle ends within the step: close it at phi = 2 pi
                frac = (TWO_PI - phi) / dphi
                cycles.append(Cycle(t + frac * h, 1.0 / span, 2.0 * math.hypot(sc, ss) / span,
                                    math.atan2(ss, sc), leg, x + frac * (x1 - x), v + frac * (v1 - v)))
                sc = ss = span = 0.0
                phi1 -= TWO_PI
            x, v, phi, f0 = x1, v1, phi1, f1
            t += h
            ts.append(t)
            xs.append(x)
            vs.append(v)
            fs.append(f1)
            freqs.append(drive.freq_hz)
            if abs(x) > limit:
                self.escaped = True
        self.x, self.v, self.t = x, v, t
        drive.phase = phi
        self._sums = [sc, ss, span]
        return Samples(np.array(ts), np.array(xs), np.array(vs), np.array(fs), np.array(freqs), cycles)


# ------------------------------------------------------------------ measured jumps
def measured_jump(cycles: list[Cycle], leg: str, span: float = 3.0, ratio: float = 1.5) -> float | None:
    """Where one leg of a sweep jumped (Hz), if its amplitude changed by more than `ratio` within `span` s
    (a jump up takes a few of the damping's time constants to grow); None if it changed by less.

    The frequency is where the biggest such change is halfway through (in log amplitude). The leg's first
    2 `span` s are left out: from rest the amplitude grows as it settles, which is no jump.
    A resonance as sharp as the sweep is slow can pass the test too: compare with the folds.
    """
    run = [c for c in cycles if c.leg == leg]
    run = [c for c in run if c.t >= run[0].t + 2 * span] if run else run
    if len(run) < 4:
        return None
    t = np.array([c.t for c in run])
    a = np.log(np.maximum([c.amplitude for c in run], 1e-300))
    later = np.minimum(np.searchsorted(t, t + span), t.size - 1)
    change = a[later] - a
    i = int(np.argmax(np.abs(change)))
    if abs(change[i]) < math.log(ratio):
        return None
    half = 0.5 * (a[i] + a[later[i]])
    past = np.flatnonzero((a[i:later[i] + 1] - half) * np.sign(change[i]) >= 0)
    return run[i + int(past[0])].freq_hz


# ------------------------------------------------------------------ presets
@dataclass(frozen=True)
class DuffingPreset:
    name: str
    osc: Duffing
    start_hz: float
    end_hz: float
    sweep_time: float  # s per leg
    mode: Sweep
    freq_hz: float  # held (HOLD), or where a sweep starts from otherwise
    note: str


PRESETS = (
    DuffingPreset(
        "Hardening spring: the jump", Duffing(), 2.0, 6.0, 120.0, Sweep.UP_DOWN, 2.0,
        "Up the upper branch past the linear resonance (3.18 Hz) and off its end at the fold (4.45 Hz), "
        "a little late as the sweep overruns it; back down along the lower branch to its fold (3.64 Hz) "
        "and up. Between the folds both branches are stable."),
    DuffingPreset(
        "Softening spring: the jump", Duffing(damping=0.8, cubic=-2e4, force=1.25), 1.5, 4.5, 120.0,
        Sweep.UP_DOWN, 1.5,
        "The curve leans the other way: up the lower branch to its fold (2.97 Hz) and a jump up, then down "
        "along the upper branch to its fold (2.55 Hz) and a jump down."),
    DuffingPreset(
        "Linear spring: no jump", Duffing(cubic=0.0), 2.0, 6.0, 120.0, Sweep.UP_DOWN, 2.0,
        "k3 = 0: one amplitude at each frequency, so the two sweeps agree, apart from the ringing a sweep "
        "leaves behind just past the resonance."),
    DuffingPreset(
        "Two stable states", Duffing(), 2.0, 6.0, 120.0, Sweep.HOLD, 4.0,
        "Held at 4 Hz, inside the band: the mass settles on the lower branch from rest. Start it on the "
        "upper branch and it stays; start it on the middle one and it falls off to one or the other."),
    DuffingPreset(
        "Escape from the well", Duffing(cubic=-4e4, force=1.25), 1.0, 4.5, 120.0, Sweep.DOWN, 4.5,
        "A softening spring driven harder: below the resonance the upper branch leans into the escape "
        "amplitude (0.1 m), and the mass is thrown over the potential's hilltop."),
    DuffingPreset(
        "Chaos (Ueda)", Duffing(stiffness=0.0, damping=0.05, cubic=1.0, force=7.5), 0.05, 0.5, 120.0,
        Sweep.HOLD, 1.0 / TWO_PI,
        "x'' + 0.05 x' + x^3 = 7.5 cos t: no linear spring, light damping, a big drive. The response never "
        "repeats: once per drive cycle the Poincaré section's dot lands somewhere new on a strange attractor."),
)
