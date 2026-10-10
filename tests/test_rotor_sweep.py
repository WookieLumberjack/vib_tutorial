"""Run-up and coast-down of the Jeffcott rotor: 1X tracking against the steady-state response."""

import math

import numpy as np
import pytest

from vib_tutorial.core.rotor import RPM, Bearing, RotorSimulator, RotorSystem
from vib_tutorial.core.rotor_sweep import (
    DIRECTIONS,
    RevolutionTracker,
    RunUpCoastDown,
    TrackedSweep,
    lag_degrees,
    probe,
    station_motion,
    steady_vectors,
)

DAMPED = Bearing(cx=400.0, cy=400.0)  # ζ ≈ 0.09 for the first mode: a slow ramp is quick to simulate


def sweep(system: RotorSystem, rate: float, lo: float = 1000.0, hi: float = 2400.0, back: bool = True):
    """Run up from lo to hi rpm at `rate` rpm/s (and back down); return the tracked legs."""
    sim = RotorSimulator(system)
    sim.accel = rate / RPM
    run = RunUpCoastDown(sim, lo / RPM, hi / RPM)
    if back:
        return run.run()
    while run.stage != "back":
        run.advance(0.25)
    return run.sweeps[:1]


def steady_peak(system: RotorSystem, station: int = 1, direction: str = "x") -> tuple[float, float]:
    """(amplitude, speed in rad/s) of the steady-state 1X peak near the first critical."""
    w = np.linspace(0.6, 1.4, 4001) * system.jeffcott_speed()
    amp = np.abs(probe(steady_vectors(system, w)[:, station], direction))
    i = int(np.argmax(amp))
    return float(amp[i]), float(w[i])


def test_probe_directions_on_an_elliptical_orbit():
    # x = a cos(φ - δ), y = b sin(φ - δ): a forward ellipse with its major axis along x, lagging δ.
    a, b, delta = 3.0, 1.0, 0.7
    v = np.array([a * np.exp(-1j * delta), -1j * b * np.exp(-1j * delta)])
    assert abs(probe(v, "x")) == pytest.approx(a) and abs(probe(v, "y")) == pytest.approx(b)
    assert abs(probe(v, "orbit")) == pytest.approx(a)  # the major semi-axis
    assert lag_degrees(probe(v, "x")) == pytest.approx(math.degrees(delta))
    assert lag_degrees(probe(v, "y")) == pytest.approx(math.degrees(delta) + 90.0)
    # The high spot is on the major axis, at the x probe's peak: the orbit's lag is the x lag.
    assert lag_degrees(probe(v, "orbit")) == pytest.approx(math.degrees(delta))
    # Unwrapped lags keep growing through 180°; a branch can be picked near a reference.
    lag = lag_degrees(np.exp(-1j * np.radians([10.0, 170.0, 190.0, 350.0, 370.0])))
    np.testing.assert_allclose(lag, [10.0, 170.0, 190.0, 350.0, 370.0])
    assert lag_degrees(np.exp(-1j * np.radians(10.0)), near=np.array(355.0)) == pytest.approx(370.0)


def test_tracker_recovers_the_steady_state_vectors():
    s = RotorSystem(position=0.35)
    sim = RotorSimulator(s)
    sim.omega = sim.target = 160.0
    while sim.t < 3.0:  # settle
        sim.advance(0.25)
    leg = TrackedSweep(rate=0.0, up=True)
    tracker = RevolutionTracker(leg)
    for _ in range(4):  # in pieces, so revolutions straddle the advances
        r = sim.advance(0.0137)
        tracker.feed(r.phase, r.omega, station_motion(s, r.q))
    revs = 4 * 0.0137 * 160.0 / (2 * math.pi)  # 1.4 revolutions per piece
    assert int(revs) - 1 <= len(leg) <= int(revs)  # whole revolutions only
    expected = steady_vectors(s, [160.0])[0]
    w, vectors, peaks = leg.arrays()
    np.testing.assert_allclose(w, 160.0)
    for v, p in zip(vectors, peaks):
        np.testing.assert_allclose(v, expected, atol=1e-4 * np.abs(expected).max())
        np.testing.assert_allclose(p[:, :2], np.abs(expected), rtol=1e-3)


def test_sweep_stages_and_legs():
    sim = RotorSimulator()
    sim.accel = 4000.0 / RPM
    run = RunUpCoastDown(sim, 1200.0 / RPM, 2000.0 / RPM, settle=0.1)
    stages = []
    while not run.done:
        run.advance(0.05)
        stages.append(run.stage)
    assert [s for i, s in enumerate(stages) if i == 0 or s != stages[i - 1]] == ["approach", "settle", "out", "back", "done"]
    up, down = run.sweeps
    assert up.up and not down.up and up.label == "Run-up 4,000 rpm/s" and down.label == "Coast-down 4,000 rpm/s"
    assert np.all(np.diff(up.speed) > 0) and np.all(np.diff(down.speed) < 0)
    assert 1200 / RPM < min(up.speed) and max(up.speed) < 2000 / RPM
    assert sim.omega == sim.target == pytest.approx(1200.0 / RPM)
    # Both ways round: a coast-down first.
    run = RunUpCoastDown(sim, 2000.0 / RPM, 1200.0 / RPM, settle=0.0)
    first, second = run.run()
    assert not first.up and second.up


def test_slow_ramp_approaches_the_steady_state():
    s = RotorSystem(bearing_a=DAMPED, bearing_b=DAMPED)
    errors = []
    for rate in (100.0, 300.0):
        (leg,) = sweep(s, rate, 1000.0, 2200.0, back=False)
        w, v, peak = leg.trace(1, "x")
        ss = probe(steady_vectors(s, w)[:, 1], "x")
        scale = np.abs(ss).max()
        errors.append((np.abs(v - ss).max() / scale, np.abs(peak - np.abs(ss)).max() / scale,
                       np.abs(lag_degrees(v, lag_degrees(ss)) - lag_degrees(ss)).max()))
    slow, faster = errors
    assert slow[0] < 0.05 and slow[1] < 0.04 and slow[2] < 3.0  # within 5% and 3° throughout
    assert all(a < 0.5 * b for a, b in zip(slow, faster))  # and closer the slower the ramp


def test_faster_ramps_peak_lower_and_later():
    s = RotorSystem()
    ss_amp, wc = steady_peak(s)
    ups, downs = [], []
    for rate in (250.0, 1000.0, 4000.0):
        up, down = sweep(s, rate)
        ups.append(up.peak(1, "x"))
        downs.append(down.peak(1, "x"))
    for peaks in (ups, downs):
        amps = [a for a, _ in peaks]
        assert amps[0] < 1.02 * ss_amp
        assert amps[0] > amps[1] > amps[2]  # lower, the faster the ramp
    # The response lags the sweep: the peak comes after the critical speed going up, before it
    # coming down, and further from it the faster the ramp.
    assert wc < ups[0][1] < ups[1][1] < ups[2][1]
    assert wc > downs[0][1] > downs[1][1] > downs[2][1]


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_steady_state_lag_passes_90_degrees_at_the_critical(direction):
    s = RotorSystem()
    _, wc = steady_peak(s, direction=direction)
    w = np.array([0.3, 1.0, 3.0]) * wc
    lag = lag_degrees(probe(steady_vectors(s, w)[:, 1], direction))
    offset = 90.0 if direction == "y" else 0.0
    assert np.all(np.abs(lag - offset - [0.0, 90.0, 180.0]) < [5.0, 3.0, 10.0])
