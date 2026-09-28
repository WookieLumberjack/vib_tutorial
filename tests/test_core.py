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
    frf_matrix,
    modal_analysis,
    modal_frf_terms,
    pluck_shape,
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
    ts, xs, _, _ = sim.advance(1.0)
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
    _, xs, _, _ = sim.advance(5.0)
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
    ts, xs, _, _ = sim.advance(1.0)
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
        np.testing.assert_allclose(sub.K_red[:k, :k], np.diag(sub.omegas[:k] ** 2), atol=1e-8)
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
        assert c.zeta_red == pytest.approx(c.zeta_true, rel=1e-8)
    reduced = compare_modes(craig_bampton(s, [3], [1, 1]), full)
    assert reduced[0].zeta_error == pytest.approx(0.0, abs=0.05)
    assert abs(reduced[3].zeta_error) > 0.2  # truncation loses damping faster than frequency accuracy
    assert reduced[4].zeta_red is None and reduced[4].zeta_true is not None


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


def test_free_interface_structure_and_assembly():
    from vib_tutorial.core import free_interface

    s = _cb_system()
    for residual_mass in (True, False):
        model = free_interface(s, [2, 4], [1, 1, 1], residual_mass)
        # The interface masses are split half and half; the pieces still sum to the full model.
        for full, attr in zip(s.matrices(), "MCK"):
            total = np.zeros((s.n, s.n))
            for sub in model.substructures:
                total[np.ix_(sub.dofs, sub.dofs)] += getattr(sub, attr)
            np.testing.assert_allclose(total, full)
        assert [sub.n_rigid for sub in model.substructures] == [0, 1, 1]  # B and C float
        for sub in model.substructures:
            k, ni = sub.n_kept, sub.ni
            np.testing.assert_allclose(sub.Phi.T @ sub.M @ sub.Phi, np.eye(ni + sub.nb), atol=1e-10)
            # x_b stays physical: the boundary rows of T are [0, I].
            np.testing.assert_allclose(sub.T[ni:], np.hstack([np.zeros((sub.nb, k)), np.eye(sub.nb)]))
            # Residual flexibility is orthogonal to the kept modes: K^_qq = W^2 + Phi_bk^T G_bb^-1 Phi_bk.
            Gbb_inv = np.linalg.inv(sub.block(sub.G, "b", "b"))
            Pbk = sub.Phi[ni:, :k]
            np.testing.assert_allclose(sub.K_red[:k, :k], np.diag(sub.omegas[:k] ** 2) + Pbk.T @ Gbb_inv @ Pbk,
                                       atol=1e-8)
            np.testing.assert_allclose(sub.K_red[k:, k:], Gbb_inv, rtol=1e-9)
        M, C, K = s.matrices()
        np.testing.assert_allclose(model.K, model.T.T @ K @ model.T, atol=1e-8)
        if residual_mass:  # Rubin is a Rayleigh-Ritz projection
            np.testing.assert_allclose(model.M, model.T.T @ M @ model.T, atol=1e-10)
        else:  # MacNeal: only the kept modes have mass
            np.testing.assert_allclose(model.M, np.diag([1.0] * 3 + [0.0] * 3))
            assert model.omegas.size == 3


def test_rubin_is_exact_with_all_modes_and_an_upper_bound_otherwise():
    from vib_tutorial.core import compare_modes, free_interface, interior_counts, kept_ranges

    s = _cb_system()
    full = modal_analysis(s)
    assert kept_ranges(s, [3], "rubin") == [(0, 3), (1, 2)]  # B's rigid-body mode is always kept
    assert kept_ranges(s, [3], "craig-bampton") == [(0, 3), (0, 2)]
    exact = free_interface(s, [3], interior_counts(s.n, [3]))
    np.testing.assert_allclose(exact.fn_hz, [m.fn_hz for m in full.modes], rtol=1e-9)
    for kept in ([0, 1], [1, 1], [2, 2]):
        errors = [c.error for c in compare_modes(free_interface(s, [3], kept), full) if c.error is not None]
        assert len(errors) == sum(kept) + 2 and all(e >= -1e-12 for e in errors)
    # Asking for fewer than the rigid-body modes keeps them anyway.
    assert free_interface(s, [3], [0, 0]).substructures[1].n_kept == 1

    # MacNeal drops the residual mass: fewer modes, and not exact even with every mode kept.
    macneal = compare_modes(free_interface(s, [3], interior_counts(s.n, [3]), residual_mass=False), full)
    assert sum(c.fn_red is not None for c in macneal) == 5
    assert max(abs(c.error) for c in macneal if c.error is not None) > 1e-3


def test_free_interface_beats_craig_bampton_on_the_lowest_mode():
    from vib_tutorial.core import compare_modes, craig_bampton, free_interface

    s = ChainSystem.uniform(8)
    full = modal_analysis(s)
    cb = compare_modes(craig_bampton(s, [3], [1, 1]), full)
    rubin = compare_modes(free_interface(s, [3], [1, 1]), full)
    assert 0 < rubin[0].error < cb[0].error / 5


def test_free_interface_frf_is_exact_statically_and_with_all_modes():
    from vib_tutorial.core import free_interface, interior_counts, reduced_frf

    s = _cb_system()
    tip = s.n - 1
    f = np.array([0.5, 1.7, 3.3])
    exact = free_interface(s, [2], interior_counts(s.n, [2]))
    np.testing.assert_allclose(reduced_frf(exact, f, tip), frf(s, f, tip), rtol=1e-8)
    # The residual flexibility restores the static flexibility of the discarded modes.
    for residual_mass in (True, False):
        fewest = free_interface(s, [2], [0, 0], residual_mass)
        np.testing.assert_allclose(reduced_frf(fewest, np.array([0.0]), tip), frf(s, np.array([0.0]), tip),
                                   rtol=1e-9)


def test_free_interface_handles_a_floating_interior():
    from vib_tutorial.core import compare_modes, free_interface

    # k1 = k2 = 0: m1 floats even with the interface held, which Craig-Bampton cannot reduce.
    # Free-interface modes just treat it as a second rigid-body mode of A.
    s = ChainSystem([1.0] * 5, [0.0, 0.0, 400.0, 400.0, 400.0], [2.0] * 5)
    model = free_interface(s, [2], [2, 1])
    assert model.substructures[0].n_rigid == 2
    cmp = compare_modes(model, modal_analysis(s))
    np.testing.assert_allclose([c.fn_red for c in cmp], [c.fn_true for c in cmp], atol=1e-6)


def test_mass_normalized_shapes_have_a_positive_peak():
    # eigh's signs depend on the LAPACK build; fixing them keeps results the same on every OS.
    for s in (ChainSystem.uniform(5), ChainSystem([1.0, 2.0, 1.0], [0.0, 300.0, 500.0], [0.0, 2.0, 1.0])):
        for m in modal_analysis(s).modes:
            phi = m.shape_mass_normalized
            assert phi[np.argmax(np.abs(phi))] > 0
            np.testing.assert_allclose(phi / np.abs(phi).max(), m.shape, atol=1e-12)


def test_free_component_modes_share_the_kinetic_energy():
    from vib_tutorial.core import component_modes, free_interface

    s = ChainSystem.uniform(8)
    model = free_interface(s, [3], [1, 1])
    comps = component_modes(model, modal_analysis(s))
    # Free modes are complete over the whole substructure: 4 in A, 5 in B (with its interface).
    assert [(c.substructure, c.index) for c in comps] == [("A", i) for i in range(1, 5)] + [("B", i) for i in range(1, 6)]
    b1 = comps[4]
    assert b1.fn_hz == pytest.approx(0.0, abs=1e-6) and b1.zeta is None and b1.kept  # rigid body
    assert b1.shape == pytest.approx(np.ones(5))
    assert all(0 < c.share <= 1 for c in comps)


def test_frf_matrix_is_symmetric_and_matches_columns():
    s = ChainSystem([1.0, 2.0, 1.0, 1.5], [400.0, 300.0, 500.0, 200.0], [15.0, 2.0, 0.0, 1.0])
    f = np.geomspace(0.1, 20.0, 200)
    H = frf_matrix(s, f)
    np.testing.assert_allclose(H, H.transpose(0, 2, 1), atol=1e-15)  # reciprocity
    for k in range(s.n):
        np.testing.assert_allclose(H[:, :, k], frf(s, f, k))


@pytest.mark.parametrize(
    "system",
    [
        ChainSystem.uniform(4),  # proportional
        ChainSystem([1.0, 2.0, 1.0, 1.5], [400.0, 300.0, 500.0, 200.0], [15.0, 2.0, 0.0, 1.0]),
        ChainSystem([1.0] * 3, [400.0] * 3, [200.0, 2.0, 2.0]),  # mode 2 overdamped: real roots
    ],
)
def test_exact_modal_terms_sum_to_the_frf_matrix(system):
    f = np.geomspace(0.1, 20.0, 200)
    H = frf_matrix(system, f)
    terms = modal_frf_terms(system, modal_analysis(system), exact=True)
    total = sum(t.evaluate(f) for t in terms)
    np.testing.assert_allclose(total, H, atol=1e-12 * np.abs(H).max())


def test_classical_modal_terms_exact_only_for_proportional_damping():
    f = np.geomspace(0.1, 20.0, 200)
    prop = ChainSystem.uniform(4)
    terms = modal_frf_terms(prop, modal_analysis(prop), exact=False)
    np.testing.assert_allclose(sum(t.evaluate(f) for t in terms), frf_matrix(prop, f), atol=1e-14)
    # Each term is phi_r phi_r^T / (w_r^2 - w^2 + i w c_r); the lowest dominates at its resonance.
    assert [t.mode for t in terms] == [1, 2, 3, 4]

    nonprop = ChainSystem([1.0] * 4, [400.0] * 4, [15.0, 2.0, 2.0, 2.0])
    H = frf_matrix(nonprop, f)
    total = sum(t.evaluate(f) for t in modal_frf_terms(nonprop, modal_analysis(nonprop), exact=False))
    assert np.abs(total - H).max() > 1e-3 * np.abs(H).max()


def test_exact_modal_terms_refuse_a_defective_eigenvalue():
    free = ChainSystem([1.0] * 3, [0.0, 400.0, 400.0], [0.0, 2.0, 2.0])  # rigid-body lambda = 0 (double)
    with pytest.raises(ValueError):
        modal_frf_terms(free, modal_analysis(free), exact=True)
    # Rounding splits the double root differently on each CPU: really or imaginarily, by
    # ~sqrt(eps). Mimic that with eps-sized noise on A; every split must still be refused.
    from vib_tutorial.core import modal as modal_module

    state_space = modal_module.state_space
    rng = np.random.default_rng(0)
    for _ in range(100):
        def noisy(system):
            A, B = state_space(system)
            return A + np.finfo(float).eps * np.abs(A).max() * rng.standard_normal(A.shape), B

        modal_module.state_space = noisy
        try:
            result = modal_analysis(free)
        finally:
            modal_module.state_space = state_space
        with pytest.raises(ValueError):
            modal_frf_terms(free, result, exact=True)
    f = np.geomspace(0.1, 20.0, 100)
    terms = modal_frf_terms(free, modal_analysis(free), exact=False)  # proportional, so still exact
    np.testing.assert_allclose(sum(t.evaluate(f) for t in terms), frf_matrix(free, f), rtol=1e-9)


def _coordinates(s, res, sim, complex_modes):
    """Run 1 s and return (t, x, y) with y the modal coordinates, including t = 0."""
    from vib_tutorial.core import modal_coordinate_map

    z0 = sim.state.copy()
    ts, xs, vs, _ = sim.advance(1.0)
    zs = np.vstack([z0, np.hstack([xs, vs])])
    return np.r_[0.0, ts], zs[:, : s.n], zs @ modal_coordinate_map(s, res, complex_modes)


def test_classical_modal_coordinates_rebuild_the_motion():
    s = ChainSystem([1.0, 2.0, 0.5, 1.0], [400.0, 300.0, 500.0, 200.0], [15.0, 2.0, 2.0, 2.0])
    res = modal_analysis(s)
    sim = Simulator(s)
    sim.set_state(np.array([0.01, -0.02, 0.03, 0.005]), np.array([0.1, 0.0, -0.2, 0.3]))
    _, xs, ys = _coordinates(s, res, sim, complex_modes=False)
    shapes = np.column_stack([m.shape for m in res.modes])
    np.testing.assert_allclose(ys @ shapes.T, xs, atol=1e-12)


def test_classical_release_stays_in_its_coordinate_only_if_proportional():
    for c1, leaks in ((2.0, False), (15.0, True)):
        s = ChainSystem([1.0] * 4, [400.0] * 4, [c1, 2.0, 2.0, 2.0])
        res = modal_analysis(s)
        sim = Simulator(s)
        sim.set_displacement(0.01 * res.modes[2].shape)
        _, _, ys = _coordinates(s, res, sim, complex_modes=False)
        others = np.abs(np.delete(ys, 2, axis=1)).max()
        assert np.abs(ys[0, 2]) == pytest.approx(0.01)
        assert (others > 1e-4) if leaks else (others < 1e-12)


def test_complex_modal_coordinates_decouple_any_damping():
    s = ChainSystem([1.0] * 4, [400.0] * 4, [15.0, 2.0, 2.0, 2.0])
    res = modal_analysis(s)
    m = res.complex_modes[4]
    sim = Simulator(s)
    z0 = m.state_vector.real
    sim.set_state(z0[: s.n], z0[s.n :])
    ts, xs, ys = _coordinates(s, res, sim, complex_modes=True)
    assert ys.shape[1] == s.n  # one coordinate per conjugate pair
    j = m.index // 2  # column of lambda5's pair
    # Released with x0 = Re(psi): the coordinate is Re(e^{lambda t}), since psi = 1 at its peak mass.
    np.testing.assert_allclose(ys[:, j], np.exp(m.eigenvalue * ts).real, atol=1e-9)
    assert np.abs(np.delete(ys, j, axis=1)).max() < 1e-9
    peak = np.argmax(np.abs(m.shape))
    np.testing.assert_allclose(ys[:, j], xs[:, peak], atol=1e-9)


def test_complex_modal_coordinates_with_real_roots():
    from vib_tutorial.core import modal_coordinate_map

    s = ChainSystem([1.0, 1.0], [100.0, 100.0], [80.0, 0.0])  # one pair, two real roots
    res = modal_analysis(s)
    P = modal_coordinate_map(s, res, complex_modes=True)
    assert P.shape == (4, 3)


def _run(sim, seconds):
    end = sim.t + seconds
    while sim.t < end - 1e-9:
        sim.advance(end - sim.t)


def _balance(sim):
    return sim.energy_added + sim.work - sim.dissipated - sim.stored_energy


def test_energy_balance_is_exact():
    s = ChainSystem([1.0, 2.0, 0.5, 1.0], [400.0, 300.0, 500.0, 200.0], [15.0, 2.0, 2.0, 2.0])
    force = ForceController(ForceSettings(target=3, kind=ForceKind.HARMONIC, amplitude=5.0, freq_hz=2.3))
    sim = Simulator(s, force)
    sim.set_displacement(np.array([0.01, -0.02, 0.0, 0.03]))
    e0 = sim.energy_added
    assert e0 == pytest.approx(sim.stored_energy) and e0 > 0
    force.switch_on()
    _run(sim, 3.0)
    assert sim.work > 0 and sim.dissipated > 0
    scale = sim.energy_added + sim.work
    assert abs(_balance(sim)) < 1e-10 * scale

    # A live edit changes the stored energy; the jump is booked as energy added.
    sim.set_system(ChainSystem.uniform(4, stiffness=800.0, damping=3.0))
    assert abs(_balance(sim)) < 1e-10 * scale
    force.settings.target = 1
    force.settings.freq_hz = 7.0
    _run(sim, 2.0)
    sim.set_system(sim.system.resized(3))
    force.switch_off()
    force.settings.kind = ForceKind.PULSE
    force.switch_on()
    _run(sim, 2.0)
    assert abs(_balance(sim)) < 1e-10 * (sim.energy_added + sim.work)

    sim.reset()
    assert sim.work == sim.dissipated == sim.energy_added == 0.0


def test_free_decay_dissipates_what_was_stored():
    s = ChainSystem([1.0], [400.0], [2.0])
    sim = Simulator(s)
    sim.set_displacement(np.array([0.1]))
    _run(sim, 30.0)  # zeta = 0.05: decays to ~1e-26 of the start
    assert sim.dissipated == pytest.approx(0.5 * 400.0 * 0.1**2, rel=1e-12)
    assert sim.work == 0.0


def test_dissipation_matches_quadrature():
    s = ChainSystem.uniform(3, damping=5.0)
    force = ForceController(ForceSettings(target=2, kind=ForceKind.HARMONIC, amplitude=5.0, freq_hz=1.5))
    sim = Simulator(s, force)
    force.switch_on()
    ts, _, vs, fs = sim.advance(1.0)
    C = s.matrices()[1]
    dt = ts[1] - ts[0]
    v = np.vstack([np.zeros(3), vs])
    f = np.concatenate([[0.0], fs])
    # Trapezoid on the step samples is second order: agree to ~ (h * f)^2.
    assert sim.dissipated == pytest.approx(np.trapezoid(np.einsum("ki,ij,kj->k", v, C, v), dx=dt), rel=1e-4)
    assert sim.work == pytest.approx(np.trapezoid(f * v[:, 2], dx=dt), rel=1e-4)


def test_ledger_balances_at_every_sample():
    from vib_tutorial.core import kinetic_energy, potential_energy

    s = ChainSystem.uniform(3, damping=5.0)
    force = ForceController(ForceSettings(target=2, kind=ForceKind.HARMONIC, amplitude=5.0, freq_hz=1.5))
    sim = Simulator(s, force)
    sim.set_displacement(np.array([0.01, 0.0, -0.01]))
    force.switch_on()
    for _ in range(2):  # the second call continues the running totals
        _, xs, vs, _ = sim.advance(1.0)
        added, work, dissipated = sim.ledger.T
        stored = kinetic_energy(s, vs).sum(axis=1) + potential_energy(s, xs).sum(axis=1)
        np.testing.assert_allclose(added + work - dissipated, stored, rtol=0, atol=1e-12)
    assert (work[-1], dissipated[-1]) == (sim.work, sim.dissipated)
    assert sim.advance(0.0)[0].size == 0 and sim.ledger.shape == (0, 3)


def test_element_forces_carry_a_static_load_and_balance_each_mass():
    from vib_tutorial.core import element_forces

    s = ChainSystem([1.0, 2.0, 0.5], [400.0, 300.0, 500.0], [15.0, 2.0, 2.0])
    M, C, K = s.matrices()
    # Static equilibrium under a tip load F: every spring carries F.
    F = 3.0
    x = np.linalg.solve(K, np.array([0.0, 0.0, F]))
    spring, damper = element_forces(s, x, np.zeros(3))
    np.testing.assert_allclose(spring, F)
    assert not damper.any()
    # In motion: each mass feels element i pulling back and element i+1 pulling on.
    xs, vs = np.array([[0.01, -0.02, 0.03], [0.0, 0.01, 0.0]]), np.array([[0.1, 0.0, -0.2], [0.3, 0.0, 0.1]])
    spring, damper = element_forces(s, xs, vs)
    total = spring + damper
    np.testing.assert_allclose(total - np.column_stack([total[:, 1:], np.zeros(2)]), xs @ K + vs @ C)


def test_energy_splits_by_element_and_by_mode():
    from vib_tutorial.core import kinetic_energy, modal_energies, potential_energy, stored_energy

    s = ChainSystem([1.0, 2.0, 0.5], [400.0, 300.0, 500.0], [15.0, 2.0, 2.0])
    x, v = np.array([0.01, -0.02, 0.03]), np.array([0.1, 0.0, -0.2])
    M, _, K = s.matrices()
    np.testing.assert_allclose(kinetic_energy(s, v).sum(), 0.5 * v @ M @ v)
    np.testing.assert_allclose(potential_energy(s, x).sum(), 0.5 * x @ K @ x)
    res = modal_analysis(s)
    E = modal_energies(s, res, x, v)
    assert E.shape == (3,) and np.all(E >= 0)
    assert E.sum() == pytest.approx(stored_energy(s, x, v), rel=1e-12)
    # A mode on its own holds all the energy in its own term.
    E3 = modal_energies(s, res, 0.01 * res.modes[2].shape, np.zeros(3))
    assert E3[2] == pytest.approx(E3.sum()) and E3[:2].max() < 1e-12 * E3.sum()


def test_non_proportional_damping_moves_energy_between_modes():
    from vib_tutorial.core import modal_energies

    for c1, moves in ((2.0, False), (15.0, True)):
        s = ChainSystem([1.0] * 4, [400.0] * 4, [c1, 2.0, 2.0, 2.0])
        res = modal_analysis(s)
        sim = Simulator(s)
        sim.set_displacement(0.01 * res.modes[2].shape)
        _, xs, vs, _ = sim.advance(1.0)
        E = modal_energies(s, res, xs, vs)
        share = np.delete(E, 2, axis=1).max() / sim.energy_added
        assert (share > 1e-3) if moves else (share < 1e-20)


def test_pluck_shape_is_the_static_shape_under_a_point_load():
    s = ChainSystem([1.0, 2.0, 1.0, 3.0], [400.0, 100.0, 250.0, 50.0], [2.0] * 4)
    x = pluck_shape(s, 1, 0.01)
    assert x[1] == 0.01
    load = s.matrices()[2] @ x
    np.testing.assert_allclose(np.delete(load, 1), 0.0, atol=1e-12)  # only the held mass is pushed
    assert x[0] == pytest.approx(0.01 * 100.0 / 500.0)  # springs 1 and 2 share the stretch
    np.testing.assert_allclose(x[2:], 0.01)  # nothing loads the masses beyond: they follow

    # Without k1 the chain floats on the held mass; a part joined to nothing stays put.
    np.testing.assert_allclose(pluck_shape(ChainSystem([1.0] * 3, [0.0, 400.0, 400.0], [0.0] * 3), 2, 0.02), 0.02)
    np.testing.assert_allclose(pluck_shape(ChainSystem([1.0] * 3, [400.0, 0.0, 0.0], [0.0] * 3), 0, 0.02), [0.02, 0, 0])


def test_chirp_sweeps_its_frequency_and_stops():
    for log, f_mid in ((False, 5.0), (True, 3.0)):
        force = ForceController(
            ForceSettings(kind=ForceKind.CHIRP, sweep_start_hz=1.0, sweep_end_hz=9.0, sweep_time=4.0, sweep_log=log)
        )
        force.switch_on()
        assert force.frequency() == 1.0
        for _ in range(2000):
            force.advance(1e-3)
        assert force.frequency() == pytest.approx(f_mid)  # halfway: arithmetic or geometric mean
        # The phase is the integral of the frequency.
        cycles = 2.0 * (1.0 + 5.0) / 2 if not log else 8.0 / math.log(9.0)
        assert abs(np.exp(1j * force.phase) - np.exp(2j * math.pi * cycles)) < 1e-6
        for _ in range(2001):
            force.advance(1e-3)
        assert not force.active and force.value() == 0.0


def test_chirp_steps_resolve_the_highest_frequency():
    force = ForceController(ForceSettings(kind=ForceKind.CHIRP, sweep_start_hz=40.0, sweep_end_hz=2.0))
    sim = Simulator(ChainSystem.uniform(2), force)
    assert sim.step_size() <= 1 / (40 * 40.0)


def test_base_excitation_matches_transmissibility():
    from vib_tutorial.core import transmissibility

    s = ChainSystem([1.0, 2.0, 1.0], [400.0, 300.0, 200.0], [3.0, 1.0, 2.0])
    f_hz, xg = 2.9, 0.004
    force = ForceController(ForceSettings(kind=ForceKind.HARMONIC, freq_hz=f_hz, base=True, base_amplitude=xg))
    sim = Simulator(s, force)
    force.switch_on()
    _run(sim, 60.0)
    _, xs, _, fs = sim.advance(5.0)
    assert (fs.max() - fs.min()) / 2 == pytest.approx(xg, rel=1e-4)  # fs is the ground motion
    measured = (xs.max(axis=0) - xs.min(axis=0)) / 2
    np.testing.assert_allclose(measured, xg * np.abs(transmissibility(s, np.array([f_hz]))[0]), rtol=2e-3)
    # Statically the chain just moves with the ground.
    np.testing.assert_allclose(transmissibility(s, np.array([1e-6]))[0], 1.0, atol=1e-6)


def test_base_step_moves_the_chain_with_the_ground():
    s = ChainSystem.uniform(3, damping=10.0)
    force = ForceController(ForceSettings(kind=ForceKind.STEP, base=True, base_amplitude=0.02))
    sim = Simulator(s, force)
    force.switch_on()
    _run(sim, 40.0)
    np.testing.assert_allclose(sim.displacement, 0.02, atol=1e-9)
    assert sim.ground == 0.02 and sim.stored_energy < 1e-12


def test_energy_balance_with_base_excitation():
    from vib_tutorial.core import element_forces, kinetic_energy, potential_energy

    s = ChainSystem([1.0, 2.0, 0.5], [400.0, 300.0, 500.0], [15.0, 2.0, 2.0])
    force = ForceController(
        ForceSettings(kind=ForceKind.CHIRP, sweep_start_hz=0.5, sweep_end_hz=6.0, sweep_time=3.0, base=True)
    )
    sim = Simulator(s, force)
    sim.set_displacement(np.array([0.01, -0.02, 0.03]))
    force.switch_on()
    _, xs, vs, fs = sim.advance(1.0)
    added, work, dissipated = sim.ledger.T
    xr, vr = xs - fs[:, None], vs - sim.ground_velocity[:, None]
    stored = kinetic_energy(s, vs).sum(axis=1) + potential_energy(s, xr).sum(axis=1)
    np.testing.assert_allclose(added + work - dissipated, stored, rtol=0, atol=1e-12)
    # Work by the ground = -integral of (tension in element 1) x_g'; check by quadrature.
    spring, damper = element_forces(s, xr, vr)
    dt = sim.step_size()
    assert sim.work == pytest.approx(-np.sum((spring[:, 0] + damper[:, 0]) * sim.ground_velocity) * dt, rel=1e-2)

    # Switching the kind of input (or stepping the ground) mid-run keeps the balance.
    for kind, base in ((ForceKind.STEP, True), (ForceKind.HARMONIC, False), (ForceKind.PULSE, True)):
        force.switch_off()
        force.settings.kind, force.settings.base = kind, base
        force.switch_on()
        _run(sim, 1.5)
        assert abs(_balance(sim)) < 1e-10 * (sim.energy_added + abs(sim.work))
    sim.reset()
    assert sim.ground == 0.0


def test_cms_response_matches_the_full_model_when_nothing_is_reduced():
    from vib_tutorial.core import component_mode_synthesis
    from vib_tutorial.core.cms_response import CMSResponse

    s = ChainSystem([1.0, 2.0, 1.5, 1.0, 0.5], [400.0, 300.0, 500.0, 200.0, 350.0], [2.0, 8.0, 1.0, 0.5, 3.0])
    for method in ("craig-bampton", "rubin"):
        model = component_mode_synthesis(s, [1], [1, 2], method)
        force = ForceController(ForceSettings(kind=ForceKind.HARMONIC, freq_hz=2.0))
        resp = CMSResponse(model, force)
        force.switch_on()
        t, x_full, x_red, f = resp.advance(2.0)
        # The full half is the ordinary simulator, sample for sample.
        ref_force = ForceController(ForceSettings(kind=ForceKind.HARMONIC, freq_hz=2.0, target=4))
        sim = Simulator(s, ref_force)
        ref_force.switch_on()
        _, x_sim, _, _ = sim.advance(2.0)
        np.testing.assert_allclose(x_full, x_sim, atol=1e-12)
        np.testing.assert_allclose(x_red, x_full, atol=1e-9 * np.abs(x_full).max())
        np.testing.assert_allclose(resp.x_reduced, x_red[-1])


def test_cms_response_of_a_reduced_model():
    from vib_tutorial.core import component_mode_synthesis
    from vib_tutorial.core.cms_response import CMSResponse

    s = ChainSystem.uniform(6, damping=20.0)  # heavily damped: settles within the run
    for method in ("craig-bampton", "rubin", "macneal"):
        model = component_mode_synthesis(s, [2], [1, 1], method)
        force = ForceController(ForceSettings(kind=ForceKind.STEP, amplitude=10.0))
        resp = CMSResponse(model, force)
        force.switch_on()
        _, x_full, x_red, _ = resp.advance(2.0)
        err = np.abs(x_red - x_full).max() / np.abs(x_full).max()
        assert 1e-4 < err < 0.2  # a reduction error while it rings, but a small one
        for _ in range(3):
            _, x_full, x_red, _ = resp.advance(10.0)
        static = np.linalg.solve(s.matrices()[2], 10.0 * np.eye(6)[5])
        np.testing.assert_allclose(x_full[-1], static, rtol=1e-6)
        if method == "craig-bampton":
            # The constraint modes are the exact static shapes: no static error.
            np.testing.assert_allclose(x_red[-1], static, rtol=1e-6)
    # MacNeal's massless boundary follows the tip force at once.
    model = component_mode_synthesis(s, [2], [1, 1], "macneal")
    force = ForceController(ForceSettings(kind=ForceKind.STEP, amplitude=10.0))
    resp = CMSResponse(model, force)
    force.switch_on()
    _, _, x_red, _ = resp.advance(1e-3)
    assert x_red[0, 5] > 0.0


def test_one_substructure_without_a_cut():
    from vib_tutorial.core import compare_modes, component_mode_synthesis, kept_ranges

    s = ChainSystem.uniform(6)
    full = modal_analysis(s)
    guyan = component_mode_synthesis(s, [], [0])
    assert guyan.labels == ["x6"] and guyan.n_red == 1
    # Guyan onto the tip: omega^2 = the tip's static stiffness / the mass it drags along.
    assert compare_modes(guyan, full)[0].error > 0.01
    assert all(abs(c.error) < 1e-9 for c in compare_modes(component_mode_synthesis(s, [], [5]), full))
    # Free interface: the one substructure's free modes are the chain's modes, so kept ones are exact.
    rubin = component_mode_synthesis(s, [], [2], "rubin")
    errors = [c.error for c in compare_modes(rubin, full)]
    assert abs(errors[0]) < 1e-9 and abs(errors[1]) < 1e-9 and errors[2] > 0.01
    assert kept_ranges(s, [], "macneal") == [(1, 5)]  # else no mass at all
    assert kept_ranges(s, [], "craig-bampton") == [(0, 5)]


def test_back_expansion_is_exact_when_nothing_is_truncated():
    from vib_tutorial.core import component_mode_synthesis
    from vib_tutorial.core.back_expansion import RecoveryResponse

    s = ChainSystem([1.0, 2.0, 1.5, 1.0, 0.5, 1.2], [400.0, 300.0, 500.0, 200.0, 350.0, 250.0],
                    [2.0, 8.0, 1.0, 0.5, 3.0, 1.0])
    for method in ("craig-bampton", "rubin"):
        for dof in (None, 1):  # the tip, and an interior mass
            model = component_mode_synthesis(s, [2], [2, 2], method)
            force = ForceController(ForceSettings(kind=ForceKind.HARMONIC, freq_hz=2.0))
            resp = RecoveryResponse(model, force, dof)
            force.switch_on()
            smp = resp.record(2.0)
            rec = resp.recovery
            tol = 1e-9 * np.abs(smp.x_full).max()
            np.testing.assert_allclose(rec.coupled(smp.r), smp.x_full, atol=tol)
            np.testing.assert_allclose(smp.x_enhanced, smp.x_full, atol=tol)
            np.testing.assert_allclose(rec.from_boundary(smp.r) + rec.from_modes(smp.r), rec.coupled(smp.r), atol=tol)
            assert np.abs(rec.from_modes(smp.r)[:, rec.boundary]).max() == 0.0
            np.testing.assert_allclose(resp.r, smp.r[-1])
            np.testing.assert_allclose(resp.x_enhanced, smp.x_enhanced[-1])
            # The boundary-only (static) recovery misses the interior's own vibration.
            assert np.abs(rec.boundary_only(smp.r) - smp.x_full).max() > 0.01 * np.abs(smp.x_full).max()


def test_back_expansion_of_a_reduced_model():
    from vib_tutorial.core import component_mode_synthesis
    from vib_tutorial.core.back_expansion import Recovery, RecoveryResponse

    s = ChainSystem.uniform(8)
    # Guyan: no q, so the coupled recovery is the boundary-only one.
    guyan = component_mode_synthesis(s, [3], [0, 0])
    rec = Recovery(guyan)
    r = np.random.default_rng(0).normal(size=(5, guyan.n_red))
    np.testing.assert_allclose(rec.coupled(r), rec.boundary_only(r), atol=1e-12)
    np.testing.assert_array_equal(rec.interior, [0, 1, 2, 4, 5, 6])
    assert [rec.owner(d) for d in (0, 3, 5, 7)] == [0, None, 1, None]

    # Driven at mode 4 with 2 modes kept: the coupled recovery of a spring force is poor, and
    # re-solving the interior with the coupled boundary motion recovers most of it.
    model = component_mode_synthesis(s, [3], [2, 2])
    fn = modal_analysis(s).modes[3].fn_hz
    force = ForceController(ForceSettings(kind=ForceKind.HARMONIC, freq_hz=fn, amplitude=10.0))
    resp = RecoveryResponse(model, force)
    force.switch_on()
    for _ in range(3):
        smp = resp.record(4.0)
    rec = resp.recovery
    true_f = rec.spring_forces(smp.x_full)[:, 1]

    def err(x):
        return np.sqrt(np.mean((rec.spring_forces(x)[:, 1] - true_f) ** 2) / np.mean(true_f**2))

    assert err(smp.x_enhanced) < 0.5 * err(rec.coupled(smp.r))

    # A step, heavily damped: every recovery ends at the static solution (for a tip load the
    # boundary-only shape is exact), for every method, including MacNeal's massless boundary.
    s = ChainSystem.uniform(6, damping=20.0)
    static = np.linalg.solve(s.matrices()[2], 10.0 * np.eye(6)[5])
    for method in ("craig-bampton", "rubin", "macneal"):
        model = component_mode_synthesis(s, [2], [1, 1], method)
        force = ForceController(ForceSettings(kind=ForceKind.STEP, amplitude=10.0))
        resp = RecoveryResponse(model, force)
        force.switch_on()
        for _ in range(4):
            smp = resp.record(10.0)
        rec = resp.recovery
        np.testing.assert_allclose(smp.x_full[-1], static, rtol=1e-6)
        np.testing.assert_allclose(rec.boundary_only(smp.r)[-1], static, rtol=1e-6)
        np.testing.assert_allclose(smp.x_enhanced[-1], rec.boundary_only(smp.r)[-1], rtol=1e-6)
