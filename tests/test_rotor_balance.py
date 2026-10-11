"""Single-plane balancing of the Jeffcott rotor: measurement, influence coefficient, correction."""

import math

import numpy as np
import pytest

from vib_tutorial.core.rotor import RPM, XD, RotorSimulator, RotorSystem, synchronous_response
from vib_tutorial.core.rotor_balance import (
    SteadyMeasurement,
    correction,
    influence,
    weight,
    weight_parts,
)
from vib_tutorial.core.rotor_sweep import probe, steady_vectors

BEARING_A, DISC = 0, 1


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
