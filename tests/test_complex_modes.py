"""Complex modes: the complexity measures, the proportional approximation, the free decay, and the trials' notes."""

import numpy as np
import pytest
import scipy.linalg

from vib_tutorial.core.complex_modes import (
    compare,
    complexity,
    free_decay,
    frf_grid,
    frf_pair,
    made_proportional,
    proportional_damping,
    strobe,
)
from vib_tutorial.core.modal import modal_analysis
from vib_tutorial.core.model import ChainSystem, state_space

UNIFORM = ChainSystem.uniform(4)
BASE = ChainSystem(np.ones(4), np.full(4, 400.0), np.array([20.0, 0, 0, 0]))


def test_complexity_measures():
    real = np.array([1.0, -0.4, 0.2, 0.9])
    for psi in (real, real * np.exp(0.7j), -2j * real):
        m = complexity(psi)
        assert m.mpc == pytest.approx(1.0) and m.mpd == pytest.approx(0.0, abs=1e-12)
        assert m.deviation == pytest.approx(np.zeros(4), abs=1e-12)
    # two entries θ apart: MPC = cos²θ, each half of θ off the line between them
    for theta in (0.2, 0.6, 1.2):
        psi = np.array([1.0, np.exp(1j * theta)]) * np.exp(0.3j)
        m = complexity(psi)
        assert m.mpc == pytest.approx(np.cos(theta) ** 2)
        assert m.mpd == pytest.approx(np.degrees(theta) / 2)
    # spread evenly round the origin: MPC 0
    assert complexity(np.exp(2j * np.pi * np.arange(3) / 3)).mpc == pytest.approx(0.0, abs=1e-12)
    snaps = strobe(real * np.exp(0.4j), 8)
    assert np.linalg.matrix_rank(snaps, tol=1e-9) == 1  # a real mode: one shape, scaled
    assert np.linalg.matrix_rank(strobe(np.array([1.0, 1j, 0.5]), 8), tol=1e-9) == 2


def test_proportional_damping_is_exact():
    """A damper per spring in proportion: real modes, the approximation's poles and FRF the exact ones, no leak."""
    r = modal_analysis(UNIFORM)
    assert r.is_proportional
    for c in compare(UNIFORM, r):
        assert c.measures.mpc == pytest.approx(1.0) and c.mac == pytest.approx(1.0)
        assert c.exact == pytest.approx(c.approx, rel=1e-10)
    f = frf_grid(r)
    exact, approx = frf_pair(UNIFORM, r, f, 3, 0)
    assert approx == pytest.approx(exact, rel=1e-9)
    assert proportional_damping(UNIFORM, r) == pytest.approx(UNIFORM.matrices()[1], abs=1e-12)
    for k in range(4):
        run = free_decay(UNIFORM, r, k)
        assert run.leak.max() < 1e-10
        assert run.q[:, k] == pytest.approx(run.approx, abs=1e-10)


def test_made_proportional():
    r = modal_analysis(BASE)
    p = made_proportional(BASE, r)
    rp = modal_analysis(p)
    assert rp.is_proportional and rp.modes[0].zeta_modal == pytest.approx(r.modes[0].zeta_modal)
    assert p.damping == pytest.approx(p.damping[0] / p.stiffness[0] * p.stiffness)
    free = ChainSystem(np.ones(2), np.array([0.0, 100.0]), np.array([1.0, 0.0]))  # a rigid-body mode first
    assert modal_analysis(made_proportional(free)).is_proportional


def test_non_proportional():
    """One damper at the base: complex modes, an approximation that misses, and motion handed between modes."""
    r = modal_analysis(BASE)
    rows = compare(BASE, r)
    assert not r.is_proportional
    assert min(c.measures.mpc for c in rows) < 0.5
    # the approximation's C_p is proportional and has the same diagonal ΦᵀCΦ
    phi = np.column_stack([m.shape_mass_normalized for m in r.modes])
    cp = proportional_damping(BASE, r)
    assert np.diag(phi.T @ cp @ phi) == pytest.approx(np.diag(phi.T @ BASE.matrices()[1] @ phi))
    assert modal_analysis(ChainSystem(BASE.masses, BASE.stiffness, np.zeros(4))).modes[0].omega_n == pytest.approx(
        r.modes[0].omega_n)
    f = frf_grid(r)
    exact, approx = frf_pair(BASE, r, f, 3, 3)
    assert np.max(np.abs(approx - exact)) > 0.1 * np.max(np.abs(exact))
    # the exact free decay against the state matrix's eigen-decomposition
    run = free_decay(BASE, r, 1)
    lam, V = scipy.linalg.eig(state_space(BASE)[0])
    z0 = np.concatenate([r.modes[1].shape_mass_normalized, np.zeros(4)])
    eta = np.linalg.solve(V, z0)
    x = (V[:4] @ (eta[:, None] * np.exp(lam[:, None] * run.t[None, :]))).real.T
    q = x @ BASE.matrices()[0] @ phi
    assert run.q == pytest.approx(q, abs=1e-9)
    assert run.leak.max() > 0.3
    with pytest.raises(ValueError):
        free_decay(ChainSystem(np.ones(2), np.array([0.0, 100.0]), np.array([1.0, 0.0])),
                   modal_analysis(ChainSystem(np.ones(2), np.array([0.0, 100.0]), np.array([1.0, 0.0]))), 0)


def test_first_order_shapes():
    """Light damping: ψ ≈ φ_r + Σ iω_r c_sr / (ω_r² − ω_s²) φ_s, the imaginary part the coupling over the spacing."""
    light = ChainSystem(BASE.masses, BASE.stiffness, BASE.damping * 0.01)
    r = modal_analysis(light)
    phi = np.column_stack([m.shape_mass_normalized for m in r.modes])
    w = np.array([m.omega_n for m in r.modes])
    cm = phi.T @ light.matrices()[1] @ phi
    for k, m in enumerate(r.modes):
        guess = phi[:, k] + sum(1j * w[k] * cm[s, k] / (w[k] ** 2 - w[s] ** 2) * phi[:, s]
                                for s in range(4) if s != k)
        psi = m.damped.shape
        scale = np.vdot(guess, psi) / np.vdot(guess, guess)
        assert np.max(np.abs(psi / scale - guess)) < 0.03 * np.max(np.abs(guess.imag))


def test_the_trials_notes():
    from vib_tutorial.gui.complex_modes import TRIALS

    trial = {name: system for name, system, _ in TRIALS}
    note = {name: text for name, _, text in TRIALS}

    def rows(name):
        return compare(trial[name], modal_analysis(trial[name]))

    name = "A damper per spring, in proportion"
    assert modal_analysis(trial[name]).is_proportional

    name = "One damper, at the base"
    r = rows(name)
    assert min(r, key=lambda c: c.measures.mpc).index == 3
    assert (r[2].measures.mpc, r[2].measures.mpd) == pytest.approx((0.20, 29), abs=0.5e-2, rel=0.02)
    assert (r[3].zeta_approx, r[3].zeta_exact) == pytest.approx((0.049, 0.023), abs=0.5e-3)
    assert free_decay(trial[name], modal_analysis(trial[name]), 1).leak[0] == pytest.approx(0.45, abs=0.005)
    assert "MPC 0.20" in note[name] and "MPD 29°" in note[name] and "45 %" in note[name]

    name = "One damper, at the top"
    r = rows(name)
    assert (r[3].zeta_exact, r[3].zeta_approx) == pytest.approx((0.017, 0.17), abs=0.5e-3, rel=0.03)
    phi = modal_analysis(trial[name]).modes[3].shape
    assert 0.25 < abs(r[3].shape[3] - r[3].shape[2]) / abs(phi[3] - phi[2]) < 1 / 3  # both 1 at m2

    name = "A strong damper locks a joint"
    r = rows(name)
    assert (r[0].zeta_exact, r[0].zeta_approx) == pytest.approx((0.050, 0.58), abs=0.5e-3, rel=0.01)
    assert (r[2].zeta_exact, r[2].zeta_approx) == pytest.approx((0.0018, 2.6), abs=0.5e-4, rel=0.02)
    assert r[3].exact is None and r[1].measures.mpd == pytest.approx(0.0, abs=1e-9)

    name = "Tuned mass damper"
    r = rows(name)
    assert [round(c.measures.mpd) for c in r[:2]] == [10, 17]
    assert max(c.measures.mpd for c in r[2:]) < 2.5


def test_a_dampers_root_locus():
    """Sweeping the base damper: from the undamped modes to those of the chain with m1 held (the damper locked),
    one pair going overdamped on the way; each mode's damping peaks between, while the approximation's grows as c."""
    from vib_tutorial.core.complex_modes import Parameter, sweep, sweep_values

    values = sweep_values(BASE, Parameter.DAMPER, 0)
    assert 20.0 in values and values[0] == pytest.approx(0.2) and values[-1] == pytest.approx(2000.0)
    sw = sweep(BASE, Parameter.DAMPER, 0)
    undamped = [m.fn_hz for m in modal_analysis(ChainSystem(BASE.masses, BASE.stiffness, np.zeros(4))).modes]
    assert np.abs(sw.poles[0]) / (2 * np.pi) == pytest.approx(undamped, rel=1e-3)
    held = ChainSystem(np.ones(3), np.full(3, 400.0), np.zeros(3))  # m1 fixed: m2–m4 on k2
    locked = sorted(m.fn_hz for m in modal_analysis(held).modes)
    end = np.abs(sw.poles[-1]) / (2 * np.pi)
    assert np.isnan(end[1]) and sorted(end[[0, 2, 3]]) == pytest.approx(locked, rel=1e-3)
    assert sw.roots[-1].size == 2  # mode 2's pair, overdamped
    # followed by continuity: no jumps between neighbouring values (but near critical damping, where a pair
    # meets on the real axis, its poles move as the square root of the distance)
    step = np.abs(np.diff(sw.poles, axis=0)) / np.abs(sw.poles[1:])
    assert np.nanmax(np.where(sw.zeta_exact[1:] < 0.7, step, np.nan)) < 0.1
    for k in range(4):
        value, zeta = sw.peak(k)
        assert sw.values[0] < value < sw.values[-1]
        assert k == 1 or zeta > 2 * sw.zeta_exact[-1, k]
    assert sw.peak(0) == pytest.approx((59.7, 0.147), abs=0.05)  # the notes' and the README's
    assert sw.zeta_approx[:, 0] / sw.values == pytest.approx(np.full(values.size, sw.zeta_approx[0, 0] / values[0]))
    # nearly real at both ends: barely damped, then a chain with no damping left in it
    assert np.nanmax(sw.mpd[0]) < 1.0 and np.nanmax(sw.mpd[-1]) < 1.0 and np.nanmax(sw.mpd) > 30
    assert sweep(BASE, Parameter.SPRING, 2).values[0] == pytest.approx(20.0)


def test_identification_real_and_complex_modes():
    """LSFD on the exact poles: complex modes fit any damping; real modes only proportional damping."""
    from vib_tutorial.core.complex_modes import identify, real_mode_lsfd

    res = identify(UNIFORM, modal_analysis(UNIFORM))
    assert res.complex_fit.error < 1e-8 and res.real_fit.error < 1e-8
    assert res.real_fit.mac == pytest.approx(np.ones(4)) and res.real_fit.mac_undamped == pytest.approx(np.ones(4))
    r = modal_analysis(BASE)
    res = identify(BASE, r)
    assert res.complex_fit.error < 1e-8 and res.complex_fit.mac == pytest.approx(np.ones(4))
    assert res.real_fit.error == pytest.approx(0.29, abs=0.005)
    assert res.real_fit.mac == pytest.approx([0.997, 0.773, 0.506, 0.886], abs=0.001)
    modes, _ = real_mode_lsfd(res.freqs, res.H, [m.damped.eigenvalue for m in r.modes])
    assert all(np.abs(m.plus.real).max() == 0.0 for m in modes)  # A/(λ − λ*): purely imaginary
    # noise: the complex fit's shapes come out a little complex even where the modes are real
    noisy = identify(UNIFORM, modal_analysis(UNIFORM), noise=0.01, seed=0)
    mpc = [complexity(m.shape).mpc for m in noisy.complex_fit.modes]
    assert min(mpc) < 0.995 and noisy.real_fit.error < 0.01
