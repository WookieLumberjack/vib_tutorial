"""Short-bearing (Ocvirk) journal bearing: equilibrium, coefficients, the whole film's force and the rigid rotor's threshold."""

import math

import numpy as np
import pytest

from vib_tutorial.core.fluid_bearing import (
    GRAVITY,
    JournalBearing,
    attitude,
    coefficients,
    eccentricity,
    full_film_damping,
    pi_film,
    rigid_threshold,
)

RPM = 60.0 / (2.0 * math.pi)
POINTS = 200_000  # the cavitation boundary is a kink: the quadrature converges as 1/points²
BEARING = JournalBearing(length=0.02, diameter=0.05, clearance=50e-6, viscosity=0.03, load=500.0)


def nu2(eps):
    """The rigid rotor's threshold whirl ν² on a short bearing at ε, in units of Ω²."""
    return rigid_threshold(*coefficients(eps))[1]


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
    # The search: the heaviest rotor the film holds, K_eq/ω_w², falls fast as the speed rises.
    for rpm, eps, ratio, mass in ((1000, 0.44, 0.52, 195.0), (5000, 0.14, 0.51, 9.0)):
        s = b.state(rpm / RPM)
        k_eq, w2 = rigid_threshold(s.k, s.c)
        assert (round(s.eccentricity, 2), round(math.sqrt(w2) / s.omega, 2)) == (eps, ratio)
        assert round(k_eq / w2, 1 if mass < 100 else 0) == mass
    assert nu2(0.7555) > 0.0 > nu2(0.7565)


def test_short_bearing_depends_on_s_times_l_over_d_squared():
    # Made dimensionless, a short bearing's state is the same at the same S(L/D)², not at the same S.
    wide, narrow = JournalBearing(), JournalBearing(length=2.5e-3)
    a, b = wide.state(wide.omega_at(1.0)), narrow.state(narrow.omega_at(4.0))
    assert a.eccentricity == pytest.approx(b.eccentricity, rel=1e-9)
    def dimensionless(bearing, s):
        return s.k * bearing.clearance / bearing.load, s.c * bearing.clearance * s.omega / bearing.load

    for x, y in zip(dimensionless(wide, a), dimensionless(narrow, b)):
        np.testing.assert_allclose(x, y, rtol=1e-9)
    # At the same S = 1, halving L moves ε from 0.32 to 0.60 (the Theory).
    assert round(wide.state(wide.omega_at(1.0)).eccentricity, 2) == 0.32
    assert round(narrow.state(narrow.omega_at(1.0)).eccentricity, 2) == 0.60


def test_lunds_whirl_ratio_depends_on_the_eccentricity_alone():
    # K scales as W/c and C as W/(cΩ), so ω_w/Ω is the same wherever ε is, whatever the load or speed.
    light, heavy = JournalBearing(), JournalBearing(load=20.0, clearance=30e-6)
    a, b = light.state(light.omega_at(0.3)), heavy.state(heavy.omega_at(0.3))
    assert a.eccentricity == pytest.approx(b.eccentricity, rel=1e-9) and a.omega != pytest.approx(b.omega)
    ratio = [math.sqrt(rigid_threshold(s.k, s.c)[1]) / s.omega for s in (a, b)]
    assert ratio[0] == pytest.approx(ratio[1], rel=1e-9)


def _states(count, seed, top=0.98):
    """Random journal positions (ε up to `top`), velocities and speeds for BEARING."""
    rng = np.random.default_rng(seed)
    c = BEARING.clearance
    for _ in range(count):
        e, a = rng.uniform(0.0, top) * c, rng.uniform(0.0, 2 * math.pi)
        yield e * math.cos(a), e * math.sin(a), *rng.normal(0.0, 0.02, 2), rng.uniform(10.0, 2000.0)


def test_whole_force_matches_the_quadrature():
    # The closed form against the film's pressure summed round the bearing, cavitation and all, at any
    # position and velocity: the arc carrying pressure turns with the journal's velocity.
    b = BEARING
    for x, y, vx, vy, w in _states(40, 1, top=0.9):
        f = b.film_force(x, y, vx, vy, w, points=POINTS)
        np.testing.assert_allclose(b.force(x, y, vx, vy, w), f, atol=1e-7 * np.abs(f).max())


def test_whole_force_linearizes_to_the_coefficients():
    # At the equilibrium the whole force holds the load, and its derivatives are the closed-form K and C.
    b = BEARING
    for rpm in (200.0, 3000.0, 20000.0):
        s = b.state(rpm / RPM)
        x, y = (v * b.clearance for v in s.position())
        o = pi_film(b.film_scale, b.clearance, x, y, 0.0, 0.0, s.omega)
        np.testing.assert_allclose(o[:2], [0.0, b.load], atol=1e-9 * b.load)
        np.testing.assert_allclose(np.reshape(o[2:6], (2, 2)), s.k, rtol=1e-9)
        np.testing.assert_allclose([[o[6], o[7]], [o[7], o[8]]], s.c, rtol=1e-9)


def test_whole_force_derivatives_and_dissipation():
    # K and C are minus the force's derivatives; the force is minus the gradient of Ψ in the velocity, so
    # C is symmetric and positive, and never more than the whole uncavitated film's damping.
    b = BEARING
    k, c = b.film_scale, b.clearance
    for x, y, vx, vy, w in _states(60, 2):
        o = pi_film(k, c, x, y, vx, vy, w)
        K, C = np.reshape(o[2:6], (2, 2)), np.array([[o[6], o[7]], [o[7], o[8]]])
        d, dv = 1e-7 * c, 1e-7 * max(abs(vx), abs(vy), c * w)
        Kn, Cn, grad = np.zeros((2, 2)), np.zeros((2, 2)), np.zeros(2)
        for j in range(2):
            e = np.eye(2)[j]
            Kn[:, j] = -(b.force(x + d * e[0], y + d * e[1], vx, vy, w) - b.force(x - d * e[0], y - d * e[1], vx, vy, w)) / (2 * d)
            Cn[:, j] = -(b.force(x, y, vx + dv * e[0], vy + dv * e[1], w) - b.force(x, y, vx - dv * e[0], vy - dv * e[1], w)) / (2 * dv)
            grad[j] = (pi_film(k, c, x, y, vx + dv * e[0], vy + dv * e[1], w)[9]
                       - pi_film(k, c, x, y, vx - dv * e[0], vy - dv * e[1], w)[9]) / (2 * dv)
        np.testing.assert_allclose(K, Kn, atol=1e-5 * np.abs(K).max())
        np.testing.assert_allclose(C, Cn, atol=1e-5 * np.abs(C).max())
        np.testing.assert_allclose(grad, -np.array(o[:2]), atol=1e-5 * np.abs(o[:2]).max())
        cxx, cxy, cyy = full_film_damping(k, c, x, y)
        assert np.linalg.eigvalsh(C).min() >= -1e-9 * np.abs(C).max()
        assert np.linalg.eigvalsh(np.array([[cxx, cxy], [cxy, cyy]]) - C).min() >= -1e-9 * cxx


def test_whole_force_vanishes_at_half_speed_whirl():
    # A centred-orbit whirl at exactly half the speed leaves the film nothing to squeeze or drag: no force.
    b = BEARING
    w, e = 300.0, 0.6 * b.clearance
    assert np.abs(b.force(e, 0.0, 0.0, 0.5 * w * e, w)).max() < 1e-9 * b.load
    assert b.force(e, 0.0, 0.0, 0.0, w)[1] > 0.0  # still, it pushes the journal ahead of it
