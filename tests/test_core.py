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
    state_space,
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


def test_state_space_modes_are_2n_eigenpairs():
    s = ChainSystem([1.0, 2.0, 0.5, 1.0], [400.0, 300.0, 500.0, 200.0], [20.0, 0.0, 1.0, 0.0])
    res = modal_analysis(s)
    cm = res.complex_modes
    assert len(cm) == 2 * s.n
    assert [m.index for m in cm] == list(range(1, 2 * s.n + 1))
    A, _ = state_space(s)
    for m in cm:
        # Each is an eigenpair of A, with the state vector = [psi, lambda psi].
        np.testing.assert_allclose(A @ m.state_vector, m.eigenvalue * m.state_vector, atol=1e-9)
        np.testing.assert_allclose(m.state_vector[s.n :], m.eigenvalue * m.shape, atol=1e-9)
        assert np.abs(m.shape).max() == pytest.approx(1.0)
        # Conjugate partners are adjacent and exact conjugates.
        partner = cm[m.conjugate - 1]
        assert abs(partner.index - m.index) == 1 and partner.conjugate == m.index
        assert partner.eigenvalue == m.eigenvalue.conjugate()
        np.testing.assert_array_equal(partner.shape, m.shape.conj())
    # The classical table's damped poles are the Im > 0 members.
    assert {id(md.damped) for md in res.modes} == {id(m) for m in cm if m.eigenvalue.imag > 0}


def test_state_space_modes_include_real_roots():
    res = modal_analysis(ChainSystem([1.0, 1.0], [100.0, 100.0], [80.0, 0.0]))
    real = [m for m in res.complex_modes if not m.is_oscillatory]
    assert len(res.complex_modes) == 4 and len(real) == 2
    assert all(m.eigenvalue.imag == 0 and m.eigenvalue.real < 0 for m in real)
    assert res.overdamped_roots == sorted((m.eigenvalue.real for m in real), reverse=True)


def test_releasing_complex_mode_excites_only_that_mode():
    # Non-proportional damping: the real undamped shape is not a mode, the complex one is.
    s = ChainSystem([1.0] * 4, [400.0] * 4, [15.0, 2.0, 2.0, 2.0])
    res = modal_analysis(s)
    assert not res.is_proportional
    m = res.complex_modes[4]
    sim = Simulator(s)
    z0 = m.state_vector.real
    sim.set_state(z0[: s.n], z0[s.n :])
    ts, xs, _ = sim.advance(1.0)
    exact = (m.shape[None, :] * np.exp(m.eigenvalue * ts)[:, None]).real
    np.testing.assert_allclose(xs, exact, atol=1e-9)


def _cb_system():
    return ChainSystem([1.0, 2.0, 0.5, 1.5, 1.0, 0.8, 1.2], [400.0, 300.0, 500.0, 200.0, 350.0, 450.0, 250.0], [2.0] * 7)


def test_substructures_assemble_to_full_matrices():
    from vib_tutorial.core import craig_bampton

    s = _cb_system()
    model = craig_bampton(s, [2, 4], [1, 1, 1])
    for full, attr in zip(s.matrices(), "MCK"):
        total = np.zeros((s.n, s.n))
        for sub in model.substructures:
            ix = np.ix_(sub.dofs, sub.dofs)
            total[ix] += getattr(sub, attr)
        np.testing.assert_allclose(total, full)
    # Every DOF is interior to one substructure or on the boundary; the tip is a boundary DOF.
    interior = np.concatenate([sub.interior for sub in model.substructures])
    assert sorted(np.concatenate([interior, model.boundary])) == list(range(s.n))
    assert list(model.boundary) == [2, 4, 6]


def test_craig_bampton_structure_and_assembly():
    from vib_tutorial.core import craig_bampton

    s = _cb_system()
    model = craig_bampton(s, [3], [2, 1])
    for sub in model.substructures:
        # Constraint modes: interior at static equilibrium, K_ii Psi + K_ib = 0.
        np.testing.assert_allclose(sub.block(sub.K, "i", "i") @ sub.Psi + sub.block(sub.K, "i", "b"), 0, atol=1e-9)
        k = sub.n_kept
        np.testing.assert_allclose(sub.K_red[:k, :k], np.diag(sub.fixed_omegas[:k] ** 2), atol=1e-8)
        np.testing.assert_allclose(sub.K_red[:k, k:], 0, atol=1e-8)  # K^ is block diagonal
        np.testing.assert_allclose(sub.M_red[:k, :k], np.eye(k), atol=1e-12)
    # Assembling the reduced substructures = reducing the assembled model with the global T.
    M, C, K = s.matrices()
    for red, full in ((model.M, M), (model.C, C), (model.K, K)):
        np.testing.assert_allclose(red, model.T.T @ full @ model.T, atol=1e-9)
    assert model.labels == ["q_A1", "q_A2", "q_B1", "x4", "x7"]


def test_craig_bampton_is_exact_with_all_modes_and_an_upper_bound_otherwise():
    from vib_tutorial.core import compare_modes, craig_bampton, interior_counts

    s = _cb_system()
    full = modal_analysis(s)
    exact = craig_bampton(s, [3], interior_counts(s.n, [3]))
    assert exact.n_red == s.n
    np.testing.assert_allclose(exact.fn_hz, [m.fn_hz for m in full.modes], rtol=1e-10)
    assert all(c.mac == pytest.approx(1.0) for c in compare_modes(exact, full))

    for kept in ([0, 0], [1, 0], [1, 1], [2, 2]):  # [0, 0] is Guyan condensation
        cmp = compare_modes(craig_bampton(s, [3], kept), full)
        errors = [c.error for c in cmp if c.error is not None]
        assert len(errors) == sum(kept) + 2
        assert all(e >= -1e-12 for e in errors)  # Rayleigh-Ritz: never below the true value
    # More modes kept never makes the first mode worse.
    e = [compare_modes(craig_bampton(s, [3], [k, k]), full)[0].error for k in range(3)]
    assert e[0] >= e[1] >= e[2]


def test_reduced_frf_matches_full_when_exact_and_statically():
    from vib_tutorial.core import craig_bampton, interior_counts, reduced_frf

    s = _cb_system()
    tip = s.n - 1
    exact = craig_bampton(s, [2], interior_counts(s.n, [2]))
    f = np.array([0.5, 1.7, 3.3])
    np.testing.assert_allclose(reduced_frf(exact, f, tip), frf(s, f, tip), rtol=1e-8)
    # Constraint modes are static solutions, so even Guyan is exact at 0 Hz.
    guyan = craig_bampton(s, [2], [0, 0])
    np.testing.assert_allclose(reduced_frf(guyan, np.array([0.0]), tip), frf(s, np.array([0.0]), tip), rtol=1e-10)


def test_craig_bampton_rejects_floating_interior():
    from vib_tutorial.core import craig_bampton

    # A free chain (k1 = 0) is fine: fixing the interface still holds m1 via k2.
    craig_bampton(ChainSystem([1.0] * 5, [0.0, 400.0, 400.0, 400.0, 400.0], [2.0] * 5), [2], [1, 1])
    # With k1 = k2 = 0, m1 floats even when the boundary is held.
    with pytest.raises(ValueError, match="K_ii"):
        craig_bampton(ChainSystem([1.0] * 5, [0.0, 0.0, 400.0, 400.0, 400.0], [2.0] * 5), [2], [1, 1])


def test_craig_bampton_damping():
    from vib_tutorial.core import (
        craig_bampton,
        damped_poles,
        interior_counts,
        substructure_damping,
    )

    # Every c/k equal (C = beta K): C^ = beta K^, block diagonal, no coupling.
    s = ChainSystem.uniform(8)
    model = craig_bampton(s, [3], [2, 2])
    for sub in model.substructures:
        np.testing.assert_allclose(sub.C_red, (2.0 / 400.0) * sub.K_red, atol=1e-10)
        assert substructure_damping(sub) == pytest.approx((0.0, 0.0), abs=1e-9)
    # Non-proportional: C^ couples the modes to the boundary in A (which holds c1).
    s = ChainSystem([1.0] * 8, [400.0] * 8, [15.0] + [2.0] * 7)
    model = craig_bampton(s, [3], [2, 2])
    assert substructure_damping(model.substructures[0])[1] > 0.1
    assert substructure_damping(model.substructures[1]) == pytest.approx((0.0, 0.0), abs=1e-9)
    # All modes kept: the reduced damped poles are the full model's.
    exact = craig_bampton(s, [3], interior_counts(s.n, [3]))
    true = [m.damped.eigenvalue for m in modal_analysis(s).modes]
    np.testing.assert_allclose(damped_poles(exact.M, exact.C, exact.K), true, rtol=1e-8)


def test_compare_modes_damping_ratios():
    from vib_tutorial.core import compare_modes, craig_bampton, interior_counts

    s = ChainSystem([1.0] * 8, [400.0] * 8, [15.0] + [2.0] * 7)
    full = modal_analysis(s)
    exact = compare_modes(craig_bampton(s, [3], interior_counts(s.n, [3])), full)
    for c in exact:
        assert c.zeta_cb == pytest.approx(c.zeta_true, rel=1e-8)
    reduced = compare_modes(craig_bampton(s, [3], [1, 1]), full)
    assert reduced[0].zeta_error == pytest.approx(0.0, abs=0.05)
    assert abs(reduced[3].zeta_error) > 0.2  # truncation loses damping faster than frequency accuracy
    assert reduced[4].zeta_cb is None and reduced[4].zeta_true is not None


def test_component_modes_alone_and_coupled():
    from vib_tutorial.core import component_modes, craig_bampton

    s = ChainSystem.uniform(8)
    model = craig_bampton(s, [3], [1, 1])
    comps = component_modes(model, modal_analysis(s))
    assert [(c.substructure, c.index) for c in comps] == [("A", i) for i in (1, 2, 3)] + [("B", i) for i in (1, 2, 3)]
    assert [c.kept for c in comps] == [True, False, False, True, False, False]
    # Identical halves: the same clamped mode, which splits into two coupled modes when joined.
    a1, b1 = comps[0], comps[3]
    assert a1.fn_hz == pytest.approx(b1.fn_hz)
    assert {a1.closest, b1.closest} == {3, 4}
    assert all(0 < c.share <= 1 for c in comps)
    # Clamped damping ratio: exact for C = beta K, zeta = beta w / 2.
    assert a1.zeta == pytest.approx(0.005 * 2 * math.pi * a1.fn_hz / 2)
