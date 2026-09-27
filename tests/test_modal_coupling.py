import math

import numpy as np
import pytest

from vib_tutorial.core import ChainSystem, compare_coupled, coupled_model, modal_analysis, subsystems
from vib_tutorial.core.modal_coupling import coupled_frf, sweep


def random_chain(rng, n):
    return ChainSystem(rng.uniform(0.2, 3.0, n), rng.uniform(50, 800, n), rng.uniform(0.1, 4.0, n))


def test_effective_masses_add_up_to_the_total():
    s = random_chain(np.random.default_rng(1), 6)
    A, B = subsystems(s, 4)
    assert A.effective_masses.sum() == pytest.approx(s.masses[:4].sum())
    assert B.effective_masses.sum() == pytest.approx(s.masses[4:].sum())
    assert np.all(A.participation >= 0) and np.all(B.participation >= 0)
    # B's first spring is tied to ground: its stiffness matrix starts with k5 + k6.
    assert B.K[0, 0] == pytest.approx(s.stiffness[4] + s.stiffness[5])


def test_two_masses_are_exact():
    """One mass each side: the oscillators are the masses themselves, so the 2-DOF model is the chain."""
    s = ChainSystem([1.0, 0.1], [400.0, 40.0], [1.0, 0.3])
    model = coupled_model(s, 1)
    full = modal_analysis(s)
    np.testing.assert_allclose(model.M, s.matrices()[0])
    np.testing.assert_allclose(model.K, s.matrices()[2])
    np.testing.assert_allclose(model.C, s.matrices()[1])
    for c in compare_coupled(model, full):
        assert c.full_index == c.index and c.error == pytest.approx(0.0, abs=1e-12)
        assert c.zeta == pytest.approx(c.zeta_full) and c.mac == pytest.approx(1.0)
    assert model.m_residual == pytest.approx(0.0, abs=1e-12)


def test_tuned_oscillators_split_by_sqrt_mu():
    for mu in (0.01, 0.1, 1.0, 3.0):
        s = ChainSystem([1.0, mu], [400.0, 400.0 * mu], [0.0, 0.0])
        model = coupled_model(s, 1)
        assert model.freq_ratio == pytest.approx(1.0) and model.mass_ratio == pytest.approx(mu)
        f1, f2 = model.fn_hz
        assert f2 - f1 == pytest.approx(math.sqrt(mu) * model.osc_a.fn_hz)


@pytest.mark.parametrize("seed", range(5))
def test_tip_mass_and_residual_are_rayleigh_ritz(seed):
    """The 2-DOF matrices are the chain's projected on [A's mode, B rigid] and [B's mode on its base]."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 9))
    s = random_chain(rng, n)
    M, C, K = s.matrices()
    full = [m.fn_hz for m in modal_analysis(s).modes]
    split = int(rng.integers(1, n))
    A, B = subsystems(s, split)
    for ra in range(A.omegas.size):
        for rb in range(B.omegas.size):
            if B.effective_masses[rb] < 1e-9 * s.masses.sum():
                with pytest.raises(ValueError, match="no effective mass"):
                    coupled_model(s, split, ra, rb)
                continue
            model = coupled_model(s, split, ra, rb)
            phi_a = A.Phi[:, ra] / A.Phi[-1, ra]
            rel = B.participation[rb] * B.Phi[:, rb]
            T = np.zeros((n, 2))
            T[:split, 0] = phi_a
            T[split:, 0] = 1.0 - rel
            T[split:, 1] = rel
            np.testing.assert_allclose(T.T @ M @ T, model.M, atol=1e-9)
            np.testing.assert_allclose(T.T @ K @ T, model.K, rtol=1e-9, atol=1e-9 * K.max())
            np.testing.assert_allclose(T.T @ C @ T, model.C, rtol=1e-9, atol=1e-9 * C.max())
            # Upper bounds, by mode number.
            assert model.fn_hz[0] >= full[0] * (1 - 1e-9) and model.fn_hz[1] >= full[1] * (1 - 1e-9)
            assert model.rayleigh_ritz
            # The shapes on the chain are T times the 2-DOF eigenvectors.
            x = T @ model.eta
            x /= x[np.argmax(np.abs(x), axis=0), [0, 1]]
            np.testing.assert_allclose(model.shapes, x, atol=1e-9)


def test_choices_that_are_not_rayleigh_ritz():
    s = ChainSystem.uniform(8)
    base = coupled_model(s, 4)
    no_res = coupled_model(s, 4, residual=False)
    eff = coupled_model(s, 4, a_mass="effective")
    assert not no_res.rayleigh_ritz and not eff.rayleigh_ritz
    assert no_res.m_a == pytest.approx(base.m_a - base.m_residual)
    assert eff.osc_a.m == pytest.approx(subsystems(s, 4)[0].effective_masses[0])
    # Each oscillator keeps its mode's frequency and modal damping ratio whatever its mass.
    for m in (base, no_res, eff):
        assert m.osc_a.fn_hz == pytest.approx(m.A.fn_hz[0])
        assert m.osc_a.zeta == pytest.approx(m.A.zetas[0])
        assert m.osc_b.zeta == pytest.approx(m.B.zetas[0])
    # The right masses do better on the fundamental.
    f1 = modal_analysis(s).modes[0].fn_hz
    assert abs(base.fn_hz[0] - f1) < abs(eff.fn_hz[0] - f1)


def test_pairing_and_frf():
    s = ChainSystem([1.0, 1.0, 0.1, 0.1], [400.0, 400.0, 40.0, 40.0], [2.0, 2.0, 0.2, 0.2])
    model = coupled_model(s, 2)
    cmp = compare_coupled(model, modal_analysis(s))
    assert [c.full_index for c in cmp] == [1, 2]
    assert all(abs(c.error) < 0.005 and c.mac > 0.99 for c in cmp)
    # Higher modes pair by shape with higher chain modes, each with a different one.
    s8 = ChainSystem.uniform(8)
    cmp = compare_coupled(coupled_model(s8, 6, 1, 0), modal_analysis(s8))
    assert len({c.full_index for c in cmp}) == 2 and cmp[0].full_index > 1
    # Static receptance at the interface: 1 / k_a (B's spring carries no static load).
    H0 = coupled_frf(model, np.array([1e-6]))[0]
    assert H0.real == pytest.approx(1.0 / model.osc_a.k, rel=1e-6)


def test_sweeps():
    s = ChainSystem([1.0, 1.0, 0.1, 0.1], [400.0, 400.0, 40.0, 40.0], [2.0, 2.0, 0.2, 0.2])
    model = coupled_model(s, 2)
    full = np.array([m.fn_hz for m in modal_analysis(s).modes])
    for kind in ("frequency", "mass"):
        sw = sweep(model, kind, points=41)
        i = int(np.argmin(np.abs(sw.x - sw.x_now)))
        assert sw.x[i] == pytest.approx(sw.x_now)
        np.testing.assert_allclose(sw.two_dof[i], model.fn_hz)
        np.testing.assert_allclose(sw.full[i], full)
        assert np.all(np.diff(sw.two_dof, axis=1) > 0)  # veering: the two never cross
    sw = sweep(model, "frequency", points=41)
    np.testing.assert_allclose(sw.f_b / sw.f_a, sw.x)
    sw = sweep(model, "mass", points=41)
    np.testing.assert_allclose(sw.f_b, model.osc_b.fn_hz)


def test_bad_inputs():
    s = ChainSystem.uniform(4)
    with pytest.raises(ValueError):
        subsystems(s, 0)
    with pytest.raises(ValueError):
        coupled_model(s, 2, mode_a=2)
    with pytest.raises(ValueError):
        coupled_model(s, 2, a_mass="nope")
    # B's first spring removed: its rigid-body mode is at 0 Hz, which has no frequency ratio.
    free = ChainSystem([1.0, 1.0], [400.0, 0.0], [1.0, 0.0])
    model = coupled_model(free, 1)
    assert model.osc_b.fn_hz == 0.0
    with pytest.raises(ValueError):
        sweep(model, "frequency")
