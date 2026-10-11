"""Gravity on a horizontal Jeffcott rotor: static sag, and vibration about it unchanged."""

import math

import numpy as np
import pytest

from vib_tutorial.core.rotor import RPM, XB, XD, YA, YB, YD, Bearing, RotorSimulator, RotorSystem
from vib_tutorial.core.rotor_balance import SteadyMeasurement
from vib_tutorial.core.rotor_sweep import RunUpCoastDown, vibration

G = 9.81
UNDAMPED = Bearing(cx=0.0, cy=0.0)


def run(sim: RotorSimulator, seconds: float) -> list:
    out = []
    end = sim.t + seconds
    while sim.t < end - 1e-12:
        out.append(sim.advance(min(0.25, end - sim.t)))
    return out


def test_the_static_sag_matches_the_beam_and_springs():
    s = RotorSystem(gravity=G)
    q = s.static_deflection()
    # Each bearing carries half the weight on its spring; the shaft bends under the disc's weight.
    w_journal = (s.disc_mass / 2 + s.bearing_a.mass) * G / s.bearing_a.ky
    assert q[YA] == pytest.approx(-w_journal) and q[YB] == pytest.approx(-w_journal)
    assert q[YD] == pytest.approx(-w_journal - s.disc_mass * G / s.disc_stiffness)
    assert np.abs(q[:4]).max() == 0.0  # nothing sideways
    assert np.abs(RotorSystem().static_deflection(np.array([0.0, 100.0]))).max() == 0.0  # vertical rotor


def test_the_rotor_rests_in_equilibrium_and_vibrates_about_it_as_without_gravity():
    s = RotorSystem(position=0.35, bearing_a=Bearing(kx=2e4, ky=8e4))
    with_g, without = RotorSimulator(s.with_(gravity=G)), RotorSimulator(s)
    np.testing.assert_allclose(with_g.q, s.with_(gravity=G).static_deflection())
    for sim in (with_g, without):
        sim.target = 1500.0 / RPM
        sim.accel = 2000.0 / RPM
    a, b = run(with_g, 1.5), run(without, 1.5)
    # Linear: the response is the same, shifted by the sag.
    for pa, pb in zip(a, b):
        np.testing.assert_allclose(vibration(with_g.system, pa.q, pa.omega), pb.q, atol=1e-12)


def test_gravity_alone_makes_no_whirl_even_on_anisotropic_supports():
    s = RotorSystem(unbalance=0.0, gravity=G, bearing_a=Bearing(kx=2e4, ky=8e4), bearing_b=Bearing(kx=3e4))
    sim = RotorSimulator(s)
    sim.target = 3000.0 / RPM
    sim.accel = 5000.0 / RPM
    pieces = run(sim, 1.0)
    q = np.concatenate([p.q for p in pieces])
    np.testing.assert_allclose(q, np.broadcast_to(s.static_deflection(), q.shape), atol=1e-15)


def test_cross_coupling_and_internal_damping_push_the_sag_sideways():
    # A displacement down pushes the journal sideways: the sag has an attitude angle.
    s = RotorSystem(gravity=G, bearing_a=Bearing(kxy=2e4), bearing_b=Bearing(kxy=2e4))
    q = s.static_deflection()
    assert q[XB] != 0.0 and math.degrees(math.atan2(-q[XD], -q[YD])) != 0.0
    # Internal damping: the shaft turns through its own sag, and the more so the faster.
    s = RotorSystem(gravity=G, internal_damping=20.0)
    q = s.static_deflection(np.array([0.0, 1000.0 / RPM, 3000.0 / RPM]))
    assert q[0, XD] == 0.0 and abs(q[2, XD]) > abs(q[1, XD]) > 0.0
    np.testing.assert_allclose(q[1], s.static_deflection(1000.0 / RPM))


def test_energy_from_equilibrium_is_kept_with_gravity():
    s = RotorSystem(position=0.3, bearing_a=UNDAMPED, bearing_b=UNDAMPED, unbalance=0.0, gravity=G)
    sim = RotorSimulator(s)
    assert sum(sim.energy()) == pytest.approx(0.0, abs=1e-18)
    sim.omega = sim.target = 1200.0
    sim.tap(0.01, -0.004)
    e0 = sum(sim.energy())
    run(sim, 1.0)
    assert sum(sim.energy()) == pytest.approx(e0, rel=1e-9)


def test_tracking_and_measurement_ignore_the_sag():
    s = RotorSystem(bearing_a=Bearing(cx=400.0, cy=400.0), bearing_b=Bearing(cx=400.0, cy=400.0))
    legs = {}
    for g in (0.0, G):
        sim = RotorSimulator(s.with_(gravity=g))
        sim.accel = 3000.0 / RPM
        legs[g] = RunUpCoastDown(sim, 1000.0 / RPM, 2000.0 / RPM).run()
        legs[g, "m"] = SteadyMeasurement(sim, 1200.0 / RPM).run()
    for a, b in zip(legs[0.0], legs[G]):
        np.testing.assert_allclose(np.array(a.vectors), np.array(b.vectors), atol=1e-12)
        np.testing.assert_allclose(np.array(a.peaks), np.array(b.peaks), atol=1e-12)
    np.testing.assert_allclose(legs[0.0, "m"], legs[G, "m"], atol=1e-12)
