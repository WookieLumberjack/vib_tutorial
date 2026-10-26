"""A motor starting a drive train from rest: the shafts' torques on the way up.

The train (core/torsional.py) starts at rest with a motor on its first disc and a load on
its last. The load is a fan's or a compressor's, rising as the square of its speed to
`StartUp.load` at the motor's synchronous speed (referred through the gears). The motor is
one of two kinds:

- **Induction**, started direct on line. Its torque follows the steady speed–torque curve,
  here Kloss's, T = 2 T_b s s_b / (s² + s_b²), from the slip s = 1 − Ω/Ω_s (T_b the
  breakdown torque at the slip s_b). Switching on adds a transient: the stator's flux
  starts with a DC part that the rotor's currents pull against, a torque at the line
  frequency that dies away in a few cycles, T_t e^{−t/τ} sin(ω_e t). A mode near the line
  frequency is struck by it at once.
- **Synchronous**, started asynchronously on its damper (amortisseur) cage. The cage gives
  Kloss's average torque, and the salient poles a pulsating one, T_p sin 2δ, with
  δ = ω_e t − p θ the rotor's electrical lag behind the stator's field (p the pole pairs).
  At a slip s it pulses at 2 s f_e: from twice the line frequency at rest to nothing at
  synchronism, sweeping down through every mode below 2 f_e on the way. Below `field_slip`
  the field is switched on, adding T_f sin δ, which pulls the rotor into step.

The discs' angles are the whole turn, not just the twist, so θ grows as the train runs up;
the rigid-body mode (K's null space) carries it. The linear part steps by the exact FOH
update, as everywhere in the app, and the motor's and the load's torques are its input,
linear over each step: their values at the step's end depend on the end's state, found by
Newton on the two torques (`StartUpSimulator.advance`), as `DuffingSimulator` does the
cubic's force. The step resolves twice the line frequency and the modes below `MODE_CAP`;
faster ones (a mesh's) are stepped exactly but sampled coarsely, and nothing here excites
them much: the torques are smooth.

A link's torque is k (w · θ) + c (w · θ′) (a mesh's: the force times the near gear's base
radius, as in the steady response), every step; each link's peak is the largest over the
run. `running_torques` gives what each link carries once the train runs steadily, the
yardstick the peaks are measured against.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
import scipy.optimize

from .simulator import foh_discretize
from .torsional import TWO_PI, DriveTrain, modes

STEPS_PER_PERIOD = 40
MAX_STEP = 1e-3  # s
MODE_CAP = 500.0  # Hz: modes up to this are resolved by the step
MAX_STEPS_PER_ADVANCE = 20_000
NEWTON_ITERATIONS = 8


class MotorKind(Enum):
    INDUCTION = "Induction"
    SYNCHRONOUS = "Synchronous"


@dataclass(frozen=True)
class Motor:
    """The motor on the train's first disc (see the module's notes). Torques in N·m."""

    kind: MotorKind = MotorKind.INDUCTION
    line_hz: float = 50.0
    pole_pairs: int = 2
    breakdown: float = 1000.0  # the cage's peak torque T_b (Kloss)
    breakdown_slip: float = 0.2  # where it peaks, s_b
    transient: float = 0.0  # induction: the switch-on torque's amplitude T_t
    time_constant: float = 0.05  # s, its decay
    pulsating: float = 0.0  # synchronous: the salient poles' T_p, at 2× slip
    field: float = 0.0  # synchronous: the field's pull-out torque T_f
    field_slip: float = 0.05  # the slip below which the field is on

    def __post_init__(self) -> None:
        if not self.line_hz > 0 or self.pole_pairs < 1:
            raise ValueError("the motor needs a line frequency and poles")
        if not self.breakdown > 0 or not 0 < self.breakdown_slip <= 1:
            raise ValueError("the motor's breakdown torque must be positive, at a slip between 0 and 1")
        if min(self.transient, self.pulsating, self.field) < 0 or not self.time_constant > 0:
            raise ValueError("the motor's torques can't be negative")

    @property
    def sync_speed(self) -> float:
        """rad/s of the shaft."""
        return TWO_PI * self.line_hz / self.pole_pairs

    @property
    def sync_rpm(self) -> float:
        return 60.0 * self.line_hz / self.pole_pairs

    def cage(self, slip: float) -> tuple[float, float]:
        """Kloss's torque at a slip, and its slope dT/ds."""
        sb = self.breakdown_slip
        d = slip * slip + sb * sb
        return 2.0 * self.breakdown * slip * sb / d, 2.0 * self.breakdown * sb * (sb * sb - slip * slip) / (d * d)

    def torque(self, t: float, angle: float, speed: float, field_on: bool) -> tuple[float, float, float]:
        """The air-gap torque at time t, shaft angle and speed: (T, dT/dθ, dT/dΩ)."""
        ws = self.sync_speed
        tc, slope = self.cage(1.0 - speed / ws)
        d_speed = -slope / ws
        if self.kind is MotorKind.INDUCTION:
            we = TWO_PI * self.line_hz
            return tc + self.transient * math.exp(-t / self.time_constant) * math.sin(we * t), 0.0, d_speed
        delta = TWO_PI * self.line_hz * t - self.pole_pairs * angle
        tf = self.field if field_on else 0.0
        s2, c2 = math.sin(2.0 * delta), math.cos(2.0 * delta)
        t_poles = self.pulsating * s2 + tf * math.sin(delta)
        d_angle = -self.pole_pairs * (2.0 * self.pulsating * c2 + tf * math.cos(delta))
        return tc + t_poles, d_angle, d_speed

    def pulsation_hz(self, slip: float) -> float:
        """The frequency the motor's torque pulses at, at a slip: the synchronous 2 s f_e; the induction
        motor's switch-on transient at the line frequency."""
        return 2.0 * slip * self.line_hz if self.kind is MotorKind.SYNCHRONOUS else self.line_hz


@dataclass(frozen=True)
class StartUp:
    """A motor on the first disc, a load on the last (N·m on its own shaft with the motor at its synchronous
    speed, ∝ speed²), and how long to run (s)."""

    motor: Motor
    load: float
    duration: float = 5.0

    def __post_init__(self) -> None:
        if self.load < 0 or not self.duration > 0:
            raise ValueError("the load can't be negative, and the run needs a duration")

    def with_(self, **changes) -> StartUp:
        return dataclasses.replace(self, **changes)


def drag(train: DriveTrain, startup: StartUp, speed: float) -> float:
    """The torque the train takes at the motor's shaft running steadily at `speed` (rad/s): the load and
    the discs' dampers to ground, each referred by its speed ratio (power over speed)."""
    s = train.speeds
    ws = startup.motor.sync_speed
    total = startup.load * (speed / ws) ** 2 * s[-1]
    for i, d in enumerate(train.discs):
        total += d.damping * speed * s[i] ** 2
    return total


def steady_speed(train: DriveTrain, startup: StartUp) -> float:
    """The motor's speed (rad/s) once the train runs steadily: synchronous, or where the induction motor's
    curve meets the drag (on the curve's stable side, below the breakdown slip); nan if it can't carry it."""
    m = startup.motor
    ws = m.sync_speed
    if m.kind is MotorKind.SYNCHRONOUS:
        return ws if m.field > 0 or m.pulsating > 0 else math.nan

    def gap(slip: float) -> float:
        return m.cage(slip)[0] - drag(train, startup, ws * (1.0 - slip))

    if gap(m.breakdown_slip) < 0:
        return math.nan
    return ws * (1.0 - scipy.optimize.brentq(gap, 0.0, m.breakdown_slip, xtol=1e-14))


def running_torques(train: DriveTrain, startup: StartUp, speed: float | None = None) -> np.ndarray:
    """Each link's steady torque (N·m, at its near disc) running at the motor speed `speed` (rad/s; the
    steady speed by default): what the discs beyond it take, by power over its speed."""
    if speed is None:
        speed = steady_speed(train, startup)
    s = train.speeds
    ws = startup.motor.sync_speed
    out = np.zeros(train.n - 1)
    for i in range(train.n - 1):
        power = sum(d.damping * (speed * s[j]) ** 2 for j, d in enumerate(train.discs) if j > i)
        power += startup.load * (speed / ws) ** 2 * speed * s[-1]
        out[i] = power / (speed * s[i])
    return out


@dataclass(frozen=True)
class Crossing:
    """Where the motor's pulsation meets a mode on the way up: at this motor speed (rpm)."""

    mode: int
    freq_hz: float
    rpm: float


def crossings(train: DriveTrain, motor: Motor) -> list[Crossing]:
    """A synchronous motor's 2× slip pulsation against the modes below 2 f_e, slowest speed first (the
    fastest mode first); none for an induction motor, whose transient is at the line frequency from rest."""
    if motor.kind is not MotorKind.SYNCHRONOUS:
        return []
    out = []
    for j, md in enumerate(modes(train)):
        slip = md.freq_hz / (2.0 * motor.line_hz)
        if slip < 1.0:
            out.append(Crossing(j, md.freq_hz, motor.sync_rpm * (1.0 - slip)))
    return sorted(out, key=lambda c: c.rpm)


@dataclass
class StartUpSamples:
    """Every step of an advance: time, the motor's speed (rpm) and torque, the load's, each link's torque."""

    t: np.ndarray
    rpm: np.ndarray
    motor: np.ndarray
    load: np.ndarray
    links: np.ndarray  # (link, step), N·m


class StartUpSimulator:
    """Steps a start-up from rest (see the module's notes)."""

    def __init__(self, train: DriveTrain, startup: StartUp, h: float | None = None) -> None:
        self.train = train
        self.startup = startup
        n = train.n
        m, c, k = train.matrices()
        minv = np.diag(1.0 / np.diag(m))
        a = np.zeros((2 * n, 2 * n))
        a[:n, n:] = np.eye(n)
        a[n:, :n] = -minv @ k
        a[n:, n:] = -minv @ c
        b = np.zeros((2 * n, 2))
        b[n, 0] = 1.0 / m[0, 0]  # the motor's torque on the first disc
        b[2 * n - 1, 1] = -1.0 / m[-1, -1]  # the load's against the last
        self.h = self.step_size() if h is None else h
        self._phi, self._g0, self._g1 = foh_discretize(a, b, self.h)
        # each link's torque from the state
        self._links = np.zeros((n - 1, 2 * n))
        for i in range(n - 1):
            w, ki, ci = train.link_terms(i)
            self._links[i, i:i + 2] = ki * w * w[0]
            self._links[i, n + i:n + i + 2] = ci * w * w[0]
        self._load_speed = startup.motor.sync_speed * train.speeds[-1]
        self.peaks = np.zeros(n - 1)  # each link's largest |torque| so far, N·m
        self.peak_t = np.zeros(n - 1)  # and when
        self.reset()

    def reset(self) -> None:
        self.t = 0.0
        self.z = np.zeros(2 * self.train.n)
        self.field_on = False
        self.peaks[:] = 0.0
        self.peak_t[:] = 0.0
        self._u = np.zeros(2)  # the torques at the current step's start

    @property
    def done(self) -> bool:
        return self.t >= self.startup.duration - 0.5 * self.h

    def step_size(self) -> float:
        """STEPS_PER_PERIOD to twice the line frequency and to the modes up to MODE_CAP; a power of two."""
        f = [2.0 * self.startup.motor.line_hz] + [md.freq_hz for md in modes(self.train) if md.freq_hz <= MODE_CAP]
        h = min(MAX_STEP, 1.0 / (STEPS_PER_PERIOD * max(f)))
        return 2.0 ** math.floor(math.log2(h))

    def _load(self, speed: float) -> tuple[float, float]:
        r = speed / self._load_speed
        return self.startup.load * r * abs(r), 2.0 * self.startup.load * abs(r) / self._load_speed

    @property
    def slip(self) -> float:
        return 1.0 - self.z[self.train.n] / self.startup.motor.sync_speed

    def advance(self, duration: float) -> StartUpSamples:
        """Step on by about `duration` s (whole steps, at most MAX_STEPS_PER_ADVANCE, never past the run's
        end); every step's samples."""
        n = self.train.n
        h = self.h
        steps = min(round(duration / h), MAX_STEPS_PER_ADVANCE,
                    max(0, round((self.startup.duration - self.t) / h)))
        motor = self.startup.motor
        phi, g0, g1 = self._phi, self._g0, self._g1
        rows = (0, n, 2 * n - 1)  # the motor's angle and speed, the load's speed
        (a0, a1), (b0, b1), (c0, c1) = (tuple(g1[r]) for r in rows)
        field_slip = motor.field_slip if motor.kind is MotorKind.SYNCHRONOUS and motor.field > 0 else -math.inf
        z, u, t, field_on = self.z, self._u, self.t, self.field_on
        zs = np.empty((steps, 2 * n))
        ts = np.empty(steps)
        us = np.empty((steps, 2))
        if t == 0.0:
            u = np.array([motor.torque(0.0, z[0], z[n], field_on)[0], self._load(z[-1])[0]])
        for i in range(steps):
            if not field_on and 1.0 - z[n] / motor.sync_speed < field_slip:
                field_on = True
            p = phi @ z + g0 @ u
            t1 = t + h
            tm, tl = u  # the start's torques as the first guess
            p0, p1, p2 = p[0], p[n], p[-1]
            for _ in range(NEWTON_ITERATIONS):
                angle = p0 + a0 * tm + a1 * tl
                speed = p1 + b0 * tm + b1 * tl
                fm, dm_a, dm_s = motor.torque(t1, angle, speed, field_on)
                fl, dl = self._load(p2 + c0 * tm + c1 * tl)
                r0, r1 = tm - fm, tl - fl
                # the Jacobian of (r0, r1) in (tm, tl)
                j00 = 1.0 - dm_a * a0 - dm_s * b0
                j01 = -dm_a * a1 - dm_s * b1
                j10 = -dl * c0
                j11 = 1.0 - dl * c1
                det = j00 * j11 - j01 * j10
                dtm = (j11 * r0 - j01 * r1) / det
                dtl = (j00 * r1 - j10 * r0) / det
                tm -= dtm
                tl -= dtl
                if abs(dtm) + abs(dtl) <= 1e-12 * (abs(tm) + abs(tl) + 1.0):
                    break
            u = np.array([tm, tl])
            z = p + g1 @ u
            t = t1
            zs[i] = z
            ts[i] = t
            us[i] = u
        self.z, self._u, self.t, self.field_on = z, u, t, field_on
        links = self._links @ zs.T if steps else np.zeros((n - 1, 0))
        if steps:
            big = np.abs(links)
            at = np.argmax(big, axis=1)
            top = big[np.arange(n - 1), at]
            more = top > self.peaks
            self.peaks[more] = top[more]
            self.peak_t[more] = ts[at[more]]
        return StartUpSamples(ts, zs[:, n] * 60.0 / TWO_PI, us[:, 0], us[:, 1], links)


# ------------------------------------------------------------------ presets
@dataclass(frozen=True)
class StartUpPreset:
    startup: StartUp
    note: str


FAN_START = StartUp(Motor(MotorKind.INDUCTION, 50.0, 2, 1600.0, 0.2, transient=1500.0, time_constant=0.04),
                    300.0, 4.0)
COMPRESSOR_START = StartUp(Motor(MotorKind.SYNCHRONOUS, 50.0, 2, 6000.0, 0.2, pulsating=2000.0, field=10000.0,
                                 field_slip=0.05), 600.0, 3.0)

# The start-up each torsional preset (by name) comes with; a train without one (the diesel set's
# engine drives it) keeps the motor the page has.
STARTUPS = {
    "Motor and fan: two inertias": StartUpPreset(
        FAN_START,
        "Direct on line: the switch-on torque rings at the line's 50 Hz, just below the coupling's 52.3 Hz "
        "mode, and the coupling's torque peaks at 6.0 kN·m in the first 0.07 s, 10× what it carries "
        "running. A softer coupling (120 kN·m/rad: 42.7 Hz, further from 50) takes 7.0×."),
    "Geared compressor": StartUpPreset(
        COMPRESSOR_START,
        "A synchronous motor's start: its poles' torque pulses at twice the slip frequency, 100 Hz at rest "
        "down to nothing at 1500 rpm, passing mode 2 (83.1 Hz) at 254 rpm and mode 1 (38.9 Hz) at 916 rpm. "
        "The low-speed coupling's torque peaks at 11.3 kN·m just after the second, 4.5× its running torque."),
}
