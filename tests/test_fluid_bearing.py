"""Short-bearing (Ocvirk) journal bearing: equilibrium, coefficients and the rigid rotor's threshold."""

import math

import numpy as np
import pytest

from vib_tutorial.core.fluid_bearing import (
    GRAVITY,
    JournalBearing,
    attitude,
    coefficients,
    eccentricity,
    rigid_threshold,
)

RPM = 60.0 / (2.0 * math.pi)
POINTS = 200_000  # the cavitation boundary is a kink: the quadrature converges as 1/points²
BEARING = JournalBearing(length=0.02, diameter=0.05, clearance=50e-6, viscosity=0.03, load=500.0)


@pytest.mark.parametrize("rpm", [200.0, 3000.0, 20000.0])
def test_film_holds_the_load_at_the_equilibrium(rpm):
    b = BEARING
    s = b.state(rpm / RPM)
    x, y = s.position()
    f = b.film_force(x * b.clearance, y * b.clearance, 0.0, 0.0, s.omega, points=POINTS)
    # Straight up, as big as the load: the journal sits off the load line, turned with the spin (x > 0).
    assert f[1] == pytest.approx(b.load, rel=1e-6)
    assert abs(f[0]) < 1e-6 * b.load
    assert x > 0.0 and y < 0.0


@pytest.mark.parametrize("rpm", [200.0, 3000.0, 20000.0])
def test_closed_form_coefficients_match_the_film_force(rpm):
    # Central differences of the quadrature force about the equilibrium: f = -K Δq - C Δq'.
    b = BEARING
    s = b.state(rpm / RPM)
    x0, y0 = (v * b.clearance for v in s.position())
    k, c = np.zeros((2, 2)), np.zeros((2, 2))
    d, v = 1e-5 * b.clearance, 1e-5 * b.clearance * s.omega
    for j in range(2):
        e = np.eye(2)[j]
        k[:, j] = -(b.film_force(x0 + d * e[0], y0 + d * e[1], 0, 0, s.omega, POINTS)
                    - b.film_force(x0 - d * e[0], y0 - d * e[1], 0, 0, s.omega, POINTS)) / (2 * d)
        c[:, j] = -(b.film_force(x0, y0, v * e[0], v * e[1], s.omega, POINTS)
                    - b.film_force(x0, y0, -v * e[0], -v * e[1], s.omega, POINTS)) / (2 * v)
    scale_k, scale_c = np.abs(s.k).max(), np.abs(s.c).max()
    np.testing.assert_allclose(k, s.k, atol=1e-5 * scale_k)
    np.testing.assert_allclose(c, s.c, atol=1e-5 * scale_c)
    np.testing.assert_allclose(s.c, s.c.T)  # the damping is symmetric, the stiffness is not
    assert abs(s.k[0, 1] - s.k[1, 0]) > 0.2 * scale_k


def test_eccentricity_and_attitude_limits():
    # Fast or light: the journal goes to the centre, 90° from the load line. Slow or heavy: to the bottom.
    assert eccentricity(1e4) < 1e-3 and attitude(1e-4) == pytest.approx(math.pi / 2, abs=1e-3)
    assert eccentricity(1e-4) > 0.97 and attitude(0.999) < 0.05
    assert eccentricity(0.0) == 1.0
    # ε falls as the speed rises; the classical and Ocvirk numbers are π (L/D)² apart.
    eps = [BEARING.state(rpm / RPM).eccentricity for rpm in (100, 1000, 10000)]
    assert eps[0] > eps[1] > eps[2]
    assert BEARING.modified_sommerfeld(100.0) == pytest.approx(math.pi * 0.4**2 * BEARING.sommerfeld(100.0))
    assert BEARING.sommerfeld(BEARING.omega_at(0.3)) == pytest.approx(0.3)


@pytest.mark.parametrize("eps", [1e-4, 0.3, 0.6, 0.9, 0.99])
def test_short_bearing_swirls_at_half_speed(eps):
    # At any load, k_xy - k_yx = Ω/2 (c_xx + c_yy): the page's swirl ratio λ = 1/2.
    a, b = coefficients(eps)
    assert (a[0, 1] - a[1, 0]) / (b[0, 0] + b[1, 1]) == pytest.approx(0.5, rel=1e-12)


def test_lightly_loaded_rigid_rotor_whirls_at_half_speed():
    _, nu2 = rigid_threshold(*coefficients(1e-4))
    assert math.sqrt(nu2) == pytest.approx(0.5, rel=1e-3)


def test_rigid_rotor_is_stable_at_any_speed_past_the_eccentricity_limit():
    # The short bearing's known result: past ε ≈ 0.756 no whirl can be fed (ν² < 0).
    nu2 = lambda e: rigid_threshold(*coefficients(e))[1]  # noqa: E731
    assert nu2(0.74) > 0.0 > nu2(0.77)


def test_threshold_matches_the_rigid_rotors_eigenvalues():
    b = JournalBearing()  # the tab's: a 0.51 kg rotor per bearing goes unstable at about 21,000 rpm
    omega = b.threshold(60_000 / RPM)
    assert omega is not None and 15_000 < omega * RPM < 30_000
    m = b.load / GRAVITY

    def growth(w):
        s = b.state(w)
        a = np.block([[np.zeros((2, 2)), np.eye(2)], [-s.k / m, -s.c / m]])
        lam = np.linalg.eigvals(a)
        return lam[np.argmax(lam.real)]

    below, at, above = growth(0.98 * omega), growth(omega), growth(1.02 * omega)
    assert below.real < 0.0 < above.real
    assert abs(at.real) < 1e-6 * abs(at.imag)
    s = b.state(omega)
    assert abs(at.imag) == pytest.approx(math.sqrt(rigid_threshold(s.k, s.c)[1]), rel=1e-6)
    assert abs(at.imag) / omega == pytest.approx(0.5, abs=0.03)  # oil whirl, at about half the speed
    assert b.threshold(10_000 / RPM) is None  # not within a narrower range


def test_theory_numbers():
    # The numbers quoted in the rotor page's Theory, for the tab's default bearing.
    b = JournalBearing()
    eps = {rpm: b.state(rpm / RPM).eccentricity for rpm in (100, 1000, 3000)}
    assert [round(eps[r], 2) for r in (100, 1000, 3000)] == [0.80, 0.44, 0.22]
    s = b.state(3000 / RPM)
    assert b.clearance * (1 - s.eccentricity) * 1e6 == pytest.approx(11.7, abs=0.05)
    assert s.k[0, 1] / 1e6 == pytest.approx(1.36, abs=0.005) and s.k[1, 0] / 1e6 == pytest.approx(-1.81, abs=0.005)
    assert b.load / GRAVITY == pytest.approx(0.51, abs=0.005)
    omega = b.threshold(60_000 / RPM)
    assert round(omega * RPM) == 21_324
    t = b.state(omega)
    assert math.sqrt(rigid_threshold(t.k, t.c)[1]) / omega == pytest.approx(0.50, abs=0.005)
    nu2 = lambda e: rigid_threshold(*coefficients(e))[1]  # noqa: E731
    assert nu2(0.7555) > 0.0 > nu2(0.7565)
