"""Run-up and coast-down of the Jeffcott rotor: steady-state Bode data and once-per-revolution tracking.

A rotor accelerated through a critical speed has no time to build up to the
steady-state peak. To show it, the response is measured as on a real machine:
a keyphasor marks each revolution (here, the heavy spot passing +x, φ = 2πk),
and over each revolution a one-period DFT gives the 1X vector, as a tracking
filter does. The largest displacement over the revolution is kept too.

Both are taken at four stations (the two bearings, the disc and midspan) and
read in one of three directions: a probe in x, a probe in y, or the orbit's
major axis. A 1X vector V is complex: the probe reads Re(V e^{iφ}). |V| is
the amplitude and -arg V the phase lag, the degrees of rotation from the
keyphasor to the probe's positive peak. For the orbit, (x + iy) = F e^{iφ} +
B e^{-iφ}: F is the forward whirl, B the backward, |F| + |B| the major
semi-axis and -arg F the angle from the heavy spot back to the high spot.

RunUpCoastDown drives a RotorSimulator from one speed to another and back at
its ramp rate, recording each leg as its own TrackedSweep.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .rotor import RPM, RotorSamples, RotorSimulator, RotorSystem, unbalance_response, whirl_modes

STATIONS = ("Bearing A", "Disc", "Midspan", "Bearing B")
DIRECTIONS = ("x", "y", "orbit")  # probe directions; "orbit" is the major axis
SETTLE_MAX = 2.0  # s, the longest dwell at the starting speed before a sweep is recorded
TWO_PI = 2.0 * math.pi


def station_positions(system: RotorSystem) -> np.ndarray:
    """Axial positions (m) of the stations: bearing A, the disc, midspan, bearing B."""
    return np.array([0.0, system.a, 0.5 * system.length, system.length])


def station_motion(system: RotorSystem, q: np.ndarray) -> np.ndarray:
    """(x, y) of the shaft's centre at each station for displacements q (..., 8): shape (..., 4, 2)."""
    return system.deflection(q, station_positions(system))


def steady_vectors(system: RotorSystem, omegas: np.ndarray) -> np.ndarray:
    """Steady-state 1X vectors at each station, (len(omegas), 4, 2) complex: (x, y) = Re(V e^{iφ})."""
    return station_motion(system, unbalance_response(system, omegas))


def probe(vectors: np.ndarray, direction: str) -> np.ndarray:
    """One complex number per 1X vector (..., 2): its amplitude |·| and phase lag -arg(·) in `direction`."""
    X, Y = vectors[..., 0], vectors[..., 1]
    if direction == "x":
        return X
    if direction == "y":
        return Y
    if direction != "orbit":
        raise ValueError(f"direction must be one of {DIRECTIONS}")
    F = 0.5 * (X + 1j * Y)  # forward whirl, turning with φ
    B = 0.5 * (np.conj(X) + 1j * np.conj(Y))  # backward whirl
    return (np.abs(F) + np.abs(B)) * np.exp(1j * np.angle(F))


def peak_column(direction: str) -> int:
    """Column of TrackedSweep.peak for a direction: max |x|, max |y| or the largest orbit radius."""
    return DIRECTIONS.index(direction)


def lag_degrees(values: np.ndarray, near: np.ndarray | None = None) -> np.ndarray:
    """Phase lag (degrees) of complex values; unwrapped along the array, or the branch nearest `near`.

    Unwrapped, the first value lies in [-90, 270), so a lag that grows through 180° (and
    beyond, at a second critical) keeps growing on the plot.
    """
    lag = -np.degrees(np.angle(values))
    if near is not None:
        return lag + 360.0 * np.round((near - lag) / 360.0)
    if np.ndim(lag) == 0:
        return lag + 360.0 * math.ceil((-90.0 - lag) / 360.0)
    lag = np.degrees(np.unwrap(np.radians(lag)))
    if lag.size:
        lag += 360.0 * math.ceil((-90.0 - lag[0]) / 360.0)
    return lag


@dataclass
class TrackedSweep:
    """One leg of a run-up or coast-down, one row per revolution."""

    rate: float  # ramp rate (rpm/s)
    up: bool  # True for a run-up
    speed: list[float] = field(default_factory=list)  # rad/s, at the middle of each revolution
    vectors: list[np.ndarray] = field(default_factory=list)  # each (4, 2) complex: the 1X vectors
    peaks: list[np.ndarray] = field(default_factory=list)  # each (4, 3): max |x|, max |y|, max radius

    @property
    def label(self) -> str:
        return f"{'Run-up' if self.up else 'Coast-down'} {self.rate:,.0f} rpm/s"

    def __len__(self) -> int:
        return len(self.speed)

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(speed (n,) rad/s, vectors (n, 4, 2), peaks (n, 4, 3))."""
        n = len(self.speed)
        return (np.array(self.speed), np.array(self.vectors).reshape(n, 4, 2),
                np.array(self.peaks).reshape(n, 4, 3))

    def trace(self, station: int, direction: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(speed rad/s, 1X complex, per-revolution peak) at one station, in one direction."""
        w, v, p = self.arrays()
        return w, probe(v[:, station], direction), p[:, station, peak_column(direction)]

    def peak(self, station: int, direction: str) -> tuple[float, float]:
        """(largest 1X amplitude, the speed it occurred at in rad/s); (0, nan) if empty."""
        w, v, _ = self.trace(station, direction)
        if not w.size:
            return 0.0, math.nan
        i = int(np.argmax(np.abs(v)))
        return float(np.abs(v[i])), float(w[i])


class RevolutionTracker:
    """The 1X vector and peak displacement over each full revolution of samples fed in order.

    A revolution runs from φ = 2πk to 2π(k+1). The partial revolution before the
    first boundary is skipped. The DFT integrates over φ by the trapezoidal rule,
    from the last sample before the revolution to the first one after it.
    """

    def __init__(self, sweep: TrackedSweep) -> None:
        self.sweep = sweep
        self._phase = np.empty(0)
        self._omega = np.empty(0)
        self._xy = np.empty((0, 4, 2))
        self._first: int | None = None  # the first full revolution's number

    def feed(self, phase: np.ndarray, omega: np.ndarray, xy: np.ndarray) -> int:
        """Add samples (phase (k,), Ω (k,), xy (k, 4, 2)); return how many revolutions completed."""
        if not phase.size:
            return 0
        if self._first is None:
            self._first = math.floor(phase[0] / TWO_PI) + 1
            if phase[0] == TWO_PI * (self._first - 1):  # starting exactly on a mark
                self._first -= 1
        ph = np.concatenate([self._phase, phase])
        om = np.concatenate([self._omega, omega])
        xy = np.concatenate([self._xy, xy])
        turn = np.floor(ph / TWO_PI).astype(int)
        done = 0
        k = self._first
        while True:
            inside = np.flatnonzero(turn == k)
            after = np.flatnonzero(turn > k)
            if not inside.size or not after.size:
                break
            lo, hi = max(inside[0] - 1, 0), after[0]
            self._record(ph[lo : hi + 1], om[lo : hi + 1], xy[lo : hi + 1], xy[inside], TWO_PI * (k + 0.5))
            done += 1
            k += 1
        self._first = k
        keep = max(int(np.searchsorted(turn, k)) - 1, 0)  # the last sample before revolution k, and on
        self._phase, self._omega, self._xy = ph[keep:], om[keep:], xy[keep:]
        return done

    def _record(self, ph: np.ndarray, om: np.ndarray, xy: np.ndarray, inside: np.ndarray, middle: float) -> None:
        f = xy * np.exp(-1j * ph)[:, None, None]
        span = ph[-1] - ph[0]
        vec = 2.0 / span * np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(ph)[:, None, None], axis=0)
        peak = np.concatenate([np.abs(inside).max(axis=0), np.hypot(inside[..., 0], inside[..., 1]).max(axis=0)[:, None]],
                              axis=1)
        self.sweep.speed.append(float(np.interp(middle, ph, om)))
        self.sweep.vectors.append(vec)
        self.sweep.peaks.append(peak)


def settle_time(system: RotorSystem, omega: float) -> float:
    """How long to dwell at speed Ω for the start-up transient to die away: four time constants
    of the least damped mode (ζ ≤ 0.3), at most SETTLE_MAX."""
    m = whirl_modes(system, omega)
    wn = TWO_PI * m.freq_hz / np.sqrt(np.maximum(1.0 - m.zeta**2, 1e-12))
    rate = (m.zeta * wn)[(m.zeta <= 0.3) & (m.zeta > 0)]
    return float(min(SETTLE_MAX, 4.0 / rate.min())) if rate.size else 0.0


def concat_samples(parts: list[RotorSamples]) -> RotorSamples:
    if len(parts) == 1:
        return parts[0]
    return RotorSamples(*(np.concatenate([getattr(p, f) for p in parts]) for f in ("t", "q", "omega", "phase")))


class RunUpCoastDown:
    """Drive a RotorSimulator from `start` to `turn` and back (rad/s) at its ramp rate, tracking each leg.

    Stages: "approach" (to the start speed, not recorded), "settle" (a dwell there
    for the transient to die away), "out" (start to turn), "back" (turn to start),
    then "done". The two legs are `sweeps`; each is fed as it runs, so a page can
    draw them live.
    """

    def __init__(self, sim: RotorSimulator, start: float, turn: float, settle: float | None = None) -> None:
        if start == turn or min(start, turn) <= 0.0:
            raise ValueError("the sweep needs two different, positive speeds")
        self.sim = sim
        self.start, self.turn = start, turn
        self.rate = sim.accel * RPM  # rpm/s
        self.settle = settle_time(sim.system, start) if settle is None else settle
        self.stage = "approach"
        self.sweeps: list[TrackedSweep] = []
        self._tracker: RevolutionTracker | None = None
        self._settle_end = 0.0
        sim.target = start

    @property
    def done(self) -> bool:
        return self.stage == "done"

    def _until_next_stage(self) -> float:
        """Simulated time to the next stage change (inf when done)."""
        sim = self.sim
        if self.stage == "settle":
            return self._settle_end - sim.t
        if self.stage == "done":
            return math.inf
        return abs(sim.target - sim.omega) / sim.accel

    def _next_stage(self) -> None:
        sim = self.sim
        if self.stage == "approach":
            self.stage, self._settle_end = "settle", sim.t + self.settle
        elif self.stage == "settle":
            self.stage, sim.target = "out", self.turn
            self._begin_leg(self.turn > self.start)
        elif self.stage == "out":
            self.stage, sim.target = "back", self.start
            self._begin_leg(self.start > self.turn)
        else:
            self.stage, self._tracker = "done", None

    def _begin_leg(self, up: bool) -> None:
        self.sweeps.append(TrackedSweep(rate=self.rate, up=up))
        self._tracker = RevolutionTracker(self.sweeps[-1])

    def advance(self, duration: float) -> RotorSamples:
        """Advance the simulator by about `duration` s, changing stage on the way; return its samples."""
        sim = self.sim
        end = sim.t + duration
        parts = []
        while True:
            h = sim.step_size()
            left = end - sim.t
            if left < 0.5 * h:
                break
            remaining = self._until_next_stage()
            r = sim.advance(min(left, max(remaining, h)))
            if not r.t.size:
                break
            parts.append(r)
            if self._tracker is not None:
                self._tracker.feed(r.phase, r.omega, station_motion(sim.system, r.q))
            if self.stage != "done" and self._until_next_stage() <= 0.5 * h:
                self._next_stage()
        if not parts:
            return sim.advance(0.0)
        return concat_samples(parts)

    def run(self, chunk: float = 0.25) -> list[TrackedSweep]:
        """Run to the end (for scripts and tests); return the two legs."""
        while not self.done:
            t0 = self.sim.t
            self.advance(chunk)
            if self.sim.t == t0:
                break
        return self.sweeps

