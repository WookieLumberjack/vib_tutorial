"""Single- and two-plane balancing of the Jeffcott rotor: measurement, influence coefficients, corrections."""

import math

import numpy as np
import pytest

from vib_tutorial.core.rotor import (
    RPM,
    STEEL,
    XD,
    YD,
    Bearing,
    PlaneWeight,
    RotorSimulator,
    RotorSystem,
    synchronous_response,
)
from vib_tutorial.core.rotor_balance import (
    SteadyMeasurement,
    correction,
    corrections,
    influence,
    influence_matrix,
    weight,
    weight_parts,
)
from vib_tutorial.core.rotor_presets import rotor_presets
from vib_tutorial.core.rotor_sweep import probe, steady_vectors

BEARING_A, DISC, BEARING_B = 0, 1, 3
PROBES = ((BEARING_A, "x"), (BEARING_B, "x"))
TRIAL = weight(0.5e-3, 0.02, 0.0)
SKEWED = RotorSystem(position=0.3, skew=math.radians(0.5))  # the theory notes' example


def reading(system: RotorSystem, rpm: float, station: int = BEARING_A, direction: str = "x") -> complex:
    """The steady-state 1X reading of a probe."""
    return complex(probe(steady_vectors(system, np.array([rpm / RPM]))[0, station], direction))


def balance(system: RotorSystem, rpm: float, trial: complex, slow_rpm: float | None = None) -> RotorSystem:
    """The influence-coefficient procedure on steady readings; return the rotor with its correction on."""
    slow = reading(system, slow_rpm) if slow_rpm else 0j
    v0 = reading(system, rpm)
    v1 = reading(system.with_(weights=system.weights + (trial,)), rpm)
    w = correction(v0, influence(v0, v1, trial), slow)
    return system.with_(weights=system.weights + (w,))


def test_weights_round_trip_and_add_to_the_unbalance():
    w = weight(0.5e-3, 0.02, math.radians(250.0))
    mass, angle = weight_parts(w, 0.02)
    assert mass == pytest.approx(0.5e-3) and math.degrees(angle) == pytest.approx(250.0)
    s = RotorSystem(unbalance=20e-6, unbalance_angle=math.radians(40.0))
    assert s.mass_unbalance == pytest.approx(20e-6 * np.exp(1j * math.radians(40.0)))
    # A weight equal and opposite takes the unbalance away: nothing moves at any speed.
    still = s.with_(weights=(-s.mass_unbalance,))
    assert np.abs(synchronous_response(still, [100.0, 166.0, 400.0])).max() < 1e-18
    # Moving the heavy spot by ψ turns the whole response by ψ.
    Q0 = synchronous_response(RotorSystem(), [150.0])[0, XD]
    assert synchronous_response(s, [150.0])[0, XD] == pytest.approx(Q0 * np.exp(1j * math.radians(40.0)))


def test_one_plane_balances_a_disc_unbalance_at_every_speed():
    s = RotorSystem(position=0.35, unbalance=35e-6, unbalance_angle=math.radians(200.0))
    fixed = balance(s, 1200.0, weight(0.5e-3, 0.02, math.radians(90.0)))
    assert abs(fixed.mass_unbalance) < 1e-9 * abs(s.mass_unbalance)
    for rpm in (600.0, 1700.0, 4000.0):
        assert abs(reading(fixed, rpm)) < 1e-9 * abs(reading(s, rpm))


def test_a_bowed_rotor_balances_at_every_speed_only_with_the_slow_roll_taken_out():
    # Relative to its bent shape, a bowed shaft sees the bow's inertia: an unbalance m δ_b at the
    # disc. So one weight takes out every force, and the disc's probe is left reading the runout.
    s = RotorSystem(bow=10e-6, bow_angle=math.radians(120.0))
    trial = weight(0.5e-3, 0.02, 0.0)
    slow = reading(s, 100.0, DISC)
    assert abs(slow) == pytest.approx(10e-6, rel=0.01)

    def balanced(compensate: bool) -> RotorSystem:
        v0, v1 = reading(s, 1200.0, DISC), reading(s.with_(weights=(trial,)), 1200.0, DISC)
        return s.with_(weights=(correction(v0, influence(v0, v1, trial), slow if compensate else 0j),))

    comp, raw = balanced(True), balanced(False)
    bow_as_unbalance = s.disc_mass * s.bow * np.exp(1j * s.bow_angle)
    assert comp.mass_unbalance == pytest.approx(-bow_as_unbalance, abs=0.01 * abs(bow_as_unbalance))
    for rpm in (800.0, 1590.0, 4000.0):
        assert abs(reading(comp, rpm, DISC) - slow) < 0.01 * abs(reading(s, rpm, DISC))
    # Balanced on the raw reading, it reads zero at the balancing speed and is out of balance elsewhere.
    assert abs(reading(raw, 1200.0, DISC)) < 1e-12
    assert abs(reading(raw, 1590.0, DISC)) > 0.3 * abs(reading(s, 1590.0, DISC))


def test_one_plane_cannot_cancel_a_couple_unbalance_at_every_speed():
    s = RotorSystem(position=0.3, unbalance=0.0, skew=math.radians(0.2))
    fixed = balance(s, 1200.0, weight(0.5e-3, 0.02, 0.0))
    assert abs(reading(fixed, 1200.0)) < 1e-12
    assert abs(reading(fixed, 3000.0)) > 0.1 * abs(reading(s, 3000.0))


def test_the_measurement_settles_to_the_steady_state():
    s = RotorSystem(position=0.35, bow=5e-6, unbalance_angle=1.0)
    sim = RotorSimulator(s)
    sim.accel = 2000.0 / RPM
    m = SteadyMeasurement(sim, 1200.0 / RPM, revs=8)
    assert m.stage == "approach" and sim.target == 1200.0 / RPM
    result = m.run()
    assert m.done and m.tracked >= 8 and result.shape == (4, 2)
    expected = steady_vectors(s, np.array([1200.0 / RPM]))[0]
    np.testing.assert_allclose(result, expected, atol=5e-3 * np.abs(expected).max())


def test_balancing_by_measurement_takes_out_most_of_the_reading():
    s = RotorSystem(unbalance=30e-6, unbalance_angle=math.radians(300.0))
    sim = RotorSimulator(s)
    sim.accel = 3000.0 / RPM
    w = 1200.0 / RPM

    def measure() -> complex:
        return complex(probe(SteadyMeasurement(sim, w).run()[BEARING_A], "x"))

    v0 = measure()
    trial = weight(0.5e-3, 0.02, 0.0)
    sim.set_system(s.with_(weights=(trial,)))
    v1 = measure()
    sim.set_system(s.with_(weights=(correction(v0, influence(v0, v1, trial)),)))
    v2 = measure()
    assert abs(v2) < 0.01 * abs(v0)


def readings(system: RotorSystem, rpm, probes=PROBES) -> np.ndarray:
    """Steady-state 1X readings (len(rpm), len(probes)) of the probes."""
    v = steady_vectors(system, np.atleast_1d(np.asarray(rpm, dtype=float)) / RPM)
    return np.array([[probe(v[i, station], d) for station, d in probes] for i in range(v.shape[0])])


def balance_two(system: RotorSystem, rpm: float, planes, noise=None) -> tuple[RotorSystem, np.ndarray, np.ndarray]:
    """Two-plane balancing on steady readings: (the rotor with its corrections on, the weights, α)."""
    read = (lambda v: v) if noise is None else noise
    v0 = read(readings(system, rpm)[0])
    runs = read(np.array([readings(system.with_(plane_weights=(p._replace(weight=TRIAL),)), rpm)[0] for p in planes]))
    alpha = influence_matrix(v0, runs, [TRIAL, TRIAL])
    w = corrections(v0, alpha)
    return system.with_(plane_weights=tuple(p._replace(weight=x) for p, x in zip(planes, w))), w, alpha


def faces(system: RotorSystem, apart: float) -> tuple[PlaneWeight, PlaneWeight]:
    return PlaneWeight(system.a - apart / 2, 0j, True), PlaneWeight(system.a + apart / 2, 0j, True)


def grams(w: complex) -> tuple[float, int]:
    mass, angle = weight_parts(w, 0.02)
    return round(mass * 1e3, 2), round(math.degrees(angle)) % 360


def test_plane_weights_on_the_disc_make_a_force_and_a_couple():
    s = RotorSystem(position=0.3, unbalance=0.0)
    w = weight(1e-3, 0.02, 1.0)
    on = s.with_(plane_weights=(PlaneWeight(s.a + 0.03, w, True),))
    assert on.mass_unbalance == pytest.approx(w) and on.couple_unbalance == pytest.approx(0.03 * w)
    # A pair equal and opposite, h apart, is a pure couple: it cancels a skew exactly, at every speed.
    skewed = s.with_(skew=math.radians(0.3), skew_angle=2.0)
    m = skewed.couple_unbalance / 0.04
    fixed = skewed.with_(plane_weights=(PlaneWeight(s.a - 0.02, m, True), PlaneWeight(s.a + 0.02, -m, True)))
    assert abs(fixed.couple_unbalance) < 1e-15 and abs(fixed.mass_unbalance) < 1e-15
    assert np.abs(synchronous_response(fixed, [100.0, 183.0, 600.0])).max() < 1e-18
    # A weight on the shaft at the disc's plane is a weight on the disc.
    at_disc = s.with_(plane_weights=(PlaneWeight(s.a, w),))
    assert at_disc.mass_unbalance == pytest.approx(w)
    np.testing.assert_allclose(synchronous_response(at_disc, [150.0]), synchronous_response(s.with_(weights=(w,)), [150.0]))
    with pytest.raises(ValueError):
        RotorSystem(plane_weights=(PlaneWeight(0.6, w),))  # past the shaft's end


def test_a_weight_on_the_shaft_is_a_point_force_there():
    # Rigid bearings and a massless shaft: the disc moves as a simply supported beam's formula says.
    rigid = Bearing(kx=1e12, ky=1e12)
    s = RotorSystem(bearing_a=rigid, bearing_b=rigid, unbalance=0.0, plane_weights=(PlaneWeight(0.15, 1.0 + 0j),))
    q = np.linalg.solve(s.stiffness(0.0), s.plane_load(s.rotating_load(1.0)))
    L, x, z = s.length, s.a, 0.15
    assert q[XD] == pytest.approx(z * (L - x) * (2 * L * x - x**2 - z**2) / (6 * s.ei * L), rel=1e-6)
    assert abs(q[YD]) < 1e-20


def test_the_influence_matrix_reduces_to_the_coefficient():
    v0, v1, t = 3e-6 + 1e-6j, 5e-6 - 2e-6j, 1e-5j
    alpha = influence_matrix([v0], [[v1]], [t])
    assert alpha.shape == (1, 1) and alpha[0, 0] == pytest.approx(influence(v0, v1, t))
    assert corrections([v0], alpha, 1e-7)[0] == pytest.approx(correction(v0, influence(v0, v1, t), 1e-7))
    with pytest.raises(ValueError):
        corrections([v0, v0], np.ones((2, 2)))  # both planes act alike: no solution
    with pytest.raises(ValueError):
        influence_matrix([v0, v0], [[v1, v1], [v1, v1]], [t, 0j])


def test_two_planes_on_the_disc_balance_it_at_every_speed():
    s = SKEWED
    grid = np.linspace(100.0, 10_000.0, 3961)
    found = np.abs(readings(s, grid, [(DISC, "x")]))[:, 0]
    assert round(found.max() * 1e6) == 72 and grid[found.argmax()] == pytest.approx(1752.5)
    fixed, w, alpha = balance_two(s, 1200.0, faces(s, 0.04))
    assert [grams(x) for x in w] == [(7.32, 180), (6.32, 0)]
    assert round(np.linalg.cond(alpha), 1) == 27.6
    assert abs(fixed.mass_unbalance) < 1e-9 * abs(s.mass_unbalance)
    assert abs(fixed.couple_unbalance) < 1e-9 * abs(s.couple_unbalance)
    assert np.abs(readings(fixed, grid, [(DISC, "x"), *PROBES])).max() < 1e-9 * found.max()
    assert round(np.linalg.cond(balance_two(s, 1200.0, faces(s, 0.1))[2])) == 11


def test_two_planes_on_the_shaft_balance_only_at_the_balancing_speed():
    s = SKEWED
    crit = 1752.5

    def disc(system: RotorSystem, rpm: float) -> float:
        return abs(readings(system, rpm, [(DISC, "x")])[0, 0]) * 1e6

    near, w, _ = balance_two(s, 1200.0, (PlaneWeight(0.1, 0j), PlaneWeight(0.2, 0j)))
    assert [grams(x) for x in w] == [(3.23, 180), (2.22, 0)]
    assert [grams(x) for x in balance_two(s, 1200.0, faces(s, 0.1))[1]] == [(3.23, 180), (2.23, 0)]
    assert np.abs(readings(near, 1200.0)).max() < 1e-12
    assert round(disc(near, crit), 1) == 2.7 and round(disc(near, 6000.0), 1) == 1.2
    far, w, alpha = balance_two(s, 1200.0, (PlaneWeight(0.02, 0j), PlaneWeight(0.48, 0j)))
    assert [grams(x) for x in w] == [(1.25, 180), (0.33, 0)] and round(np.linalg.cond(alpha), 1) == 1.5
    assert np.abs(readings(far, 1200.0)).max() < 1e-12
    assert round(disc(far, crit), 1) == 12.8
    assert round(disc(s, 6000.0), 1) == 13.6


def test_planes_close_together_amplify_errors_in_the_readings():
    s = SKEWED
    rng = np.random.default_rng(0)

    def noisy(v: np.ndarray) -> np.ndarray:
        return v * (1 + 0.01 * (rng.standard_normal(v.shape) + 1j * rng.standard_normal(v.shape)) / math.sqrt(2))

    def left(apart: float) -> float:
        out = [abs(readings(balance_two(s, 1200.0, faces(s, apart), noisy)[0], 1752.5, [(DISC, "x")])[0, 0])
               for _ in range(1000)]
        return float(np.median(out)) * 1e6

    # A sampled median: the notes say about 18 µm and about 8 µm.
    assert 17.0 < left(0.04) < 20.0 and 7.5 < left(0.1) < 8.8


def test_two_planes_on_the_disc_balance_the_long_shaft_and_the_overhung_rotor():
    presets = {p.name: p.system for p in rotor_presets()}
    for name, rpm, top in (("Long shaft, with its mass", 1200.0, 10_000.0), ("Overhung rotor", 1000.0, 3000.0)):
        s = presets[name].with_(skew=math.radians(0.05), skew_angle=1.0)
        fixed, _, _ = balance_two(s, rpm, faces(s, 0.05))
        grid = np.linspace(100.0, top, 400)
        before = np.abs(readings(s, grid, [(DISC, "x"), *PROBES])).max()
        assert np.abs(readings(fixed, grid, [(DISC, "x"), *PROBES])).max() < 1e-8 * before
    assert presets["Long shaft, with its mass"].density == STEEL
