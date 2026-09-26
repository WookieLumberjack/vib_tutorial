import math

import numpy as np
import pytest

from vib_tutorial.core import (
    ChainSystem,
    ForceController,
    ForceKind,
    ForceSettings,
    Simulator,
    frf,
    modal_analysis,
)


def test_assembly_two_dof():
    s = ChainSystem([1.0, 2.0], [10.0, 20.0], [0.1, 0.2])
    M, C, K = s.matrices()
    np.testing.assert_allclose(M, np.diag([1.0, 2.0]))
    np.testing.assert_allclose(K, [[30.0, -20.0], [-20.0, 20.0]])
    np.testing.assert_allclose(C, [[0.3, -0.2], [-0.2, 0.2]])


def test_single_dof_matches_textbook():
    m, k, c = 2.0, 800.0, 4.0
    res = modal_analysis(ChainSystem([m], [k], [c]))
    wn = math.sqrt(k / m)
    zeta = c / (2 * math.sqrt(k * m))
    mode = res.modes[0]
    assert mode.omega_n == pytest.approx(wn)
    assert mode.zeta_modal == pytest.approx(zeta)
    assert mode.damped.zeta == pytest.approx(zeta)
    assert mode.damped.omega_d == pytest.approx(wn * math.sqrt(1 - zeta**2))


def test_two_dof_uniform_chain_frequencies():
    # Grounded chain m-k-m-k: w^2 = (k/m) (3 -/+ sqrt5)/2
    m, k = 1.0, 100.0
    res = modal_analysis(ChainSystem.uniform(2, m, k, 0.0))
    expected = np.sqrt(k / m * np.array([(3 - math.sqrt(5)) / 2, (3 + math.sqrt(5)) / 2]))
    np.testing.assert_allclose([md.omega_n for md in res.modes], expected)
    # First mode: both masses in phase; second: out of phase.
    assert np.all(res.modes[0].shape > 0)
    assert res.modes[1].shape[0] * res.modes[1].shape[1] < 0


def test_stiffness_proportional_damping_is_exact():
    # C = beta K  =>  zeta_r = beta w_r / 2, and the approximation is exact.
    beta = 0.002
    k = np.array([400.0, 300.0, 500.0, 200.0])
    s = ChainSystem([1.0, 1.5, 0.8, 1.2], k, beta * k)
    res = modal_analysis(s)
    assert res.is_proportional
    for md in res.modes:
        assert md.zeta_modal == pytest.approx(beta * md.omega_n / 2)
        assert md.damped.zeta == pytest.approx(md.zeta_modal, rel=1e-8)
        assert md.damped.omega_n == pytest.approx(md.omega_n, rel=1e-8)


def test_nonproportional_damping_flagged():
    s = ChainSystem([1.0, 1.0, 1.0, 1.0], [400.0] * 4, [20.0, 0.0, 0.0, 0.0])
    res = modal_analysis(s)
    assert not res.is_proportional
    assert all(md.damped is not None for md in res.modes)


def test_overdamped_single_dof():
    res = modal_analysis(ChainSystem([1.0], [100.0], [50.0]))  # zeta = 2.5
    assert res.modes[0].damped is None
    assert len(res.overdamped_roots) == 2


def test_free_vibration_matches_analytic():
    m, k, c = 1.0, 400.0, 2.0
    sim = Simulator(ChainSystem([m], [k], [c]))
    x0 = 0.1
    sim.set_displacement(np.array([x0]))
    ts, xs, _ = sim.advance(1.0)
    wn = math.sqrt(k / m)
    z = c / (2 * math.sqrt(k * m))
    wd = wn * math.sqrt(1 - z**2)
    exact = x0 * np.exp(-z * wn * ts) * (np.cos(wd * ts) + z * wn / wd * np.sin(wd * ts))
    np.testing.assert_allclose(xs[:, 0], exact, atol=1e-9)


def test_static_deflection_under_step():
    k = np.array([400.0, 200.0, 100.0, 50.0])
    s = ChainSystem([1.0] * 4, k, [20.0] * 4)
    force = ForceController(ForceSettings(target=3, kind=ForceKind.STEP, amplitude=10.0))
    sim = Simulator(s, force)
    force.switch_on()
    while sim.t < 60.0:
        sim.advance(5.0)
    # Springs in series all carry the full load.
    np.testing.assert_allclose(sim.displacement, np.cumsum(10.0 / k), rtol=1e-6)


def test_harmonic_steady_state_matches_frf():
    s = ChainSystem.uniform(4, 1.0, 400.0, 3.0)
    f_hz = 2.3
    force = ForceController(ForceSettings(target=1, kind=ForceKind.HARMONIC, amplitude=5.0, freq_hz=f_hz))
    sim = Simulator(s, force)
    force.switch_on()
    while sim.t < 60.0:  # let transients decay (advance() caps steps per call)
        sim.advance(5.0)
    _, xs, _ = sim.advance(5.0)
    measured = (xs.max(axis=0) - xs.min(axis=0)) / 2
    expected = 5.0 * np.abs(frf(s, np.array([f_hz]), 1)[0])
    np.testing.assert_allclose(measured, expected, rtol=2e-3)


def test_live_parameter_change_keeps_state():
    sim = Simulator(ChainSystem.uniform(4))
    sim.set_displacement(np.array([0.01, 0.02, 0.03, 0.04]))
    sim.advance(0.1)
    before = sim.state.copy()
    sim.set_system(ChainSystem.uniform(4, stiffness=800.0))
    np.testing.assert_array_equal(sim.state, before)
    sim.set_system(sim.system.resized(3))
    assert sim.state.size == 6
