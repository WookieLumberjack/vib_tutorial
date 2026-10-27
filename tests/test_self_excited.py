"""The Self-excited vibration page's models (core/self_excited.py)."""

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from vib_tutorial.core.self_excited import (
    Flutter,
    Galloping,
    VanDerPol,
    Viv,
    galloping_branches,
    rk4,
    settled_amplitude,
)


def test_rk4_matches_solve_ivp_on_van_der_pol():
    m = VanDerPol(eps=1.0)
    t, x, _ = m.simulate(5, 400)
    f = m.rhs(m.eps)
    ref = solve_ivp(lambda _, z: f(z), (0, t[-1]), [m.start, 0.0], t_eval=t, rtol=1e-11, atol=1e-14).y[0]
    assert np.max(np.abs(x - ref)) < 1e-7


def test_rk4_steps_many_systems_at_once():
    m = VanDerPol()
    eps = np.array([0.1, 0.5])
    z = rk4(m.rhs(eps), np.array([[1e-3, 1e-3], [0.0, 0.0]]), 0.01, 200)
    for j, e in enumerate(eps):
        alone = rk4(m.rhs(e), np.array([1e-3, 0.0]), 0.01, 200)
        assert z[:, :, j] == pytest.approx(alone)


def test_van_der_pol_limit_cycle_is_2_xr_root_eps():
    for eps in (0.05, 0.2):
        m = VanDerPol(eps=eps)
        assert m.sweep(np.array([eps]), periods=200)[0] == pytest.approx(float(m.theory_amplitude(eps)), rel=0.01)
    assert VanDerPol().sweep(np.array([-0.2]))[0] < 1e-6


def test_galloping_onset_is_where_the_net_damping_changes_sign():
    g = Galloping()
    small = 1e-4 * g.side
    for U, sign in ((0.99 * g.onset, -1), (1.01 * g.onset, 1)):
        assert np.sign(g.net_power(np.array([small]), U)[0]) == sign


def test_galloping_branches_are_where_the_motion_settles():
    g = Galloping()
    for U, start in ((6.0, 0.001), (6.0, 0.12), (12.0, 0.001)):
        _t, y, _ = Galloping(speed=U, start=start).simulate(600)
        amp = settled_amplitude(y)
        stable = [b.amplitude[0] for b in galloping_branches(g, np.array([U])) if b.stable]
        assert min(abs(a - amp) for a in stable) < 0.01 * amp


def test_net_power_averages_the_force():
    g = Galloping()
    a, U = 0.05, 6.0
    theta = np.linspace(0, 2 * np.pi, 20001)[:-1]
    v = -a * g.omega * np.sin(theta)
    from vib_tutorial.core.self_excited import force_coefficient

    F = 0.5 * g.rho * U * U * g.side * force_coefficient(v / U)
    wind = np.mean(F * v) * 2 * np.pi / g.omega
    damper = np.pi * 2 * g.mass * g.zeta * g.omega ** 2 * a * a
    assert g.net_power(np.array([a]), U)[0] == pytest.approx(wind - damper, rel=1e-9)


def test_viv_locks_in_near_the_strouhal_speed():
    v = Viv()
    amp, freq = v.sweep(np.array([3.0, 5.5, 9.0]))
    assert amp[1] > 5 * max(amp[0], amp[2])
    assert freq[1] == pytest.approx(1.0, abs=0.03) and freq[2] > 1.5


def test_flutter_poles_without_wind_are_the_structures():
    f = Flutter(x_cg=0.0, zeta=0.0)
    lam = f.poles(0.0)
    assert sorted(abs(lam.imag[lam.imag > 0]) / (2 * math.pi)) == pytest.approx([3.0, 6.0])
    assert f.divergence_speed() == pytest.approx(Flutter().divergence_speed())
    Uf = Flutter().flutter_speed()
    assert np.max(Flutter().poles(0.999 * Uf).real) < 0 < np.max(Flutter().poles(1.001 * Uf).real)
