"""The Rust kernels (vib_tutorial._native) against the Python they replace, which stays the reference."""

import copy
import math
import os

import numpy as np
import pytest

from vib_tutorial.core import fluid_bearing, native, rotor
from vib_tutorial.core.rotor import RPM, RotorSimulator, RotorSystem
from vib_tutorial.core.rotor_presets import JOURNAL_BEARING

needs_native = pytest.mark.skipif(native.module is None, reason="Rust kernels not loaded")


def test_kernels_are_built():
    # A source install builds them, so a missing module is a broken build, unless asked for the Python.
    if os.environ.get("VIB_NATIVE", "1") == "0":
        assert native.module is None
        assert fluid_bearing.pi_film is fluid_bearing.pi_film_reference
        assert fluid_bearing.full_film_damping is fluid_bearing.full_film_damping_reference
    else:
        assert native.module is not None
        assert fluid_bearing.pi_film is native.module.pi_film
        assert fluid_bearing.full_film_damping is native.module.full_film_damping


@needs_native
def test_constants_agree():
    assert native.module.ECCENTRICITY_MAX == fluid_bearing.ECCENTRICITY_MAX


@needs_native
def test_pi_film_matches_the_reference():
    # Anywhere in the clearance and past the wall, centred, at rest and spinning either way.
    rng = np.random.default_rng(7)
    k, c = 2.4e3, 50e-6
    cases = [(0.0, 0.0, 0.0, 0.0, 300.0), (0.0, 0.0, 1e-3, -2e-3, 300.0), (0.3 * c, 0.0, 0.0, 0.0, 0.0),
             (0.0, -0.99 * c, 0.0, 0.0, 300.0), (2 * c, -c, 0.0, 0.0, 50.0)]
    for _ in range(500):
        eps, angle = rng.choice([rng.uniform(0.0, 0.99), rng.uniform(1.0, 1.5)]), rng.uniform(0.0, 2 * math.pi)
        omega = rng.choice([1.0, -1.0]) * rng.uniform(1.0, 3000.0)
        vx, vy = rng.normal(scale=c * abs(omega), size=2)
        cases.append((eps * c * math.cos(angle), eps * c * math.sin(angle), vx, vy, omega))
    for case in cases:
        want = np.array(fluid_bearing.pi_film_reference(k, c, *case))
        got = native.module.pi_film(k, c, *case)
        assert isinstance(got, tuple) and len(got) == 10
        # Each group (force, stiffness, damping, dissipation) on its own scale. The integrals divide by
        # powers of 1 - ε², up to the stiffness's 3.5, so the libraries' rounding (Rust's sin_cos and
        # powi against Python's math, which differ on macOS) grows near the wall: at ε = 0.99 the
        # stiffness rounds apart by ~1e-9. At the wall (where past it the film is worked out) even the
        # force is good to only ~1e-3 in either, against a 50-digit evaluation, and the stiffness is noise.
        wall = math.hypot(case[0], case[1]) > 0.99 * c
        near = 1e-2 if wall else 1e-10
        for part, tol in ((slice(0, 2), near), (slice(2, 6), None if wall else 1e-8),
                          (slice(6, 9), near), (slice(9, 10), near)):
            if tol is not None:
                scale = np.abs(want[part]).max()
                np.testing.assert_allclose(got[part], want[part], rtol=0, atol=tol * scale, err_msg=str(case))


@needs_native
def test_full_film_damping_matches_the_reference():
    rng = np.random.default_rng(3)
    k, c = 2.4e3, 50e-6
    for eps in [0.0, *rng.uniform(0.0, 1.5, 300)]:
        angle = rng.uniform(0.0, 2 * math.pi)
        x, y = eps * c * math.cos(angle), eps * c * math.sin(angle)
        want = fluid_bearing.full_film_damping_reference(k, c, x, y)
        np.testing.assert_allclose(native.module.full_film_damping(k, c, x, y), want, rtol=0,
                                   atol=1e-13 * max(abs(v) for v in want))


@needs_native
@pytest.mark.parametrize("gravity, density, rpm, drifts", [
    (9.81, 0.0, 9000.0, False),  # Journal bearings (oil whip), past the whip's onset
    (0.0, 0.0, 3500.0, True),  # Oil whirl (vertical rotor): the journals circle past ε = 0.9, the bound retaken often
    (9.81, 7800.0, 6000.0, False),  # a shaft with mass: more states than the stations'
])
def test_film_steps_match_the_reference(gravity, density, rpm, drifts):
    # The kernel steps as _film_steps_reference does, handing back to take the bound afresh as the
    # journals drift: the same motion to rounding error, while ramping and at a steady speed.
    system = RotorSystem(bearing_a=JOURNAL_BEARING, bearing_b=JOURNAL_BEARING, gravity=gravity,
                         density=density, nonlinear_films=True)
    sim = RotorSimulator(system)
    sim.omega, sim.target = 0.9 * rpm / RPM, rpm / RPM
    sim.accel = 0.1 * rpm / RPM / 0.25  # at speed 0.25 s into the first stretch
    sim.advance(0.05)
    take = rotor._Films.take

    def counting(self, *args):
        nonlocal takes
        takes += 1
        return take(self, *args)

    for steady in (False, True):  # ramping, then at speed
        assert (sim.omega == sim.target) == steady
        takes = 0
        ref = copy.deepcopy(sim)
        ref._film_steps = ref._film_steps_reference
        rotor._Films.take = counting
        try:
            got = sim.advance(0.3)
        finally:
            rotor._Films.take = take
        want = ref.advance(0.3)
        scale = np.abs(want.q).max()
        np.testing.assert_allclose(got.q, want.q, rtol=0, atol=1e-11 * scale)
        np.testing.assert_allclose(sim.state, ref.state, rtol=0, atol=1e-11 * np.abs(ref.state).max())
    # At speed the bound is retaken only as the journals drift: the kernel hands back to take it afresh.
    assert (takes > 1) == drifts
