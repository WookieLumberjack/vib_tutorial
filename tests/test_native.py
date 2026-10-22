"""The Rust kernels (vib_tutorial._native) against the Python they replace, which stays the reference."""

import math
import os

import numpy as np
import pytest

from vib_tutorial.core import fluid_bearing, native

needs_native = pytest.mark.skipif(native.module is None, reason="Rust kernels not loaded")


def test_kernels_are_built():
    # A source install builds them, so a missing module is a broken build, unless asked for the Python.
    if os.environ.get("VIB_NATIVE", "1") == "0":
        assert native.module is None
        assert fluid_bearing.pi_film is fluid_bearing.pi_film_reference
    else:
        assert native.module is not None
        assert fluid_bearing.pi_film is native.module.pi_film


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
        # Each group (force, stiffness, damping, dissipation) on its own scale. The stiffness divides
        # by (1 - ε²)^3.5: at ε = 0.99 the two round apart by ~1e-9, and at the wall (where past it
        # the film is worked out) both are rounding noise, so it's left out there.
        wall = math.hypot(case[0], case[1]) > 0.99 * c
        for part, tol in ((slice(0, 2), 1e-10), (slice(2, 6), None if wall else 1e-8),
                          (slice(6, 9), 1e-10), (slice(9, 10), 1e-10)):
            if tol is not None:
                scale = np.abs(want[part]).max()
                np.testing.assert_allclose(got[part], want[part], rtol=0, atol=tol * scale, err_msg=str(case))
