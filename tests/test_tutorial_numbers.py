"""The numbers quoted in the tutorial text (Background, Theory tabs, Try it lists, README).

Each test recomputes what a page tells the student they will see, so a change to a
default, a preset or a solver cannot silently make the text wrong. If one fails, fix
the text (or the change), and quote the new value.
"""

import numpy as np
import pytest
from scipy.signal import find_peaks

from vib_tutorial.core import (
    ChainSystem,
    Simulator,
    compare_coupled,
    compare_modes,
    component_mode_synthesis,
    coupled_model,
    frf,
    modal_analysis,
    modal_energies,
    modal_frf_terms,
    transmissibility,
)
from vib_tutorial.core.presets import presets


def peaks(system: ChainSystem, dof: int, scale: float, lo: float = 0.5, hi: float = 8.0) -> list[float]:
    f = np.linspace(lo, hi, 100001)
    h = np.abs(frf(system, f, dof)[:, dof]) * scale
    return [float(h[i]) for i in find_peaks(h)[0]]


def preset(name: str):
    return next(p for p in presets() if name in p.name)


def with_c1(c1: float) -> ChainSystem:
    s = ChainSystem.uniform(4)
    s.damping[0] = c1
    return s


@pytest.mark.parametrize(("c1", "coupling", "modal", "exact"), [
    (5, 0.36, 0.0977, 0.0979),
    (15, 0.71, 0.168, 0.180),
    (50, 0.90, 0.45, 0.98),
])
def test_background_coupling_table(c1, coupling, modal, exact):
    r = modal_analysis(with_c1(c1))
    assert r.coupling == pytest.approx(coupling, abs=0.005)
    pairs = [(m.zeta_modal, m.damped.zeta) for m in r.modes if m.damped]
    assert any(z_m == pytest.approx(modal, rel=0.01) and z_e == pytest.approx(exact, rel=0.01) for z_m, z_e in pairs)


def test_background_try_it_overdamped_at_c1_200():
    roots = modal_analysis(with_c1(200)).overdamped_roots
    assert len(roots) == 2 and roots[0] > -3 and roots[1] < -100  # one slow, one very fast


def test_background_try_it_energy_moves_to_mode_1():
    for c1, mode1_share in ((15, 0.85), (2, None)):
        s = with_c1(c1)
        r = modal_analysis(s)
        sim = Simulator(s)
        sim.set_state(0.02 * r.modes[2].shape, np.zeros(4))
        for _ in range(20):
            sim.advance(0.05)  # 1 s
        e = modal_energies(s, r, sim.displacement, sim.velocity)
        if mode1_share is None:
            assert e[2] / e.sum() == pytest.approx(1.0, abs=1e-6)  # mode 3 keeps it all
        else:
            assert e[0] / e.sum() > mode1_share  # "within a second most of what is left is in mode 1"


def test_presets_absorber_and_tuned_mass_damper():
    absorber = preset("absorber (undamped)").system
    f = modal_analysis(absorber).modes
    assert (f[1].fn_hz - f[0].fn_hz) / 3.183 == pytest.approx(0.1**0.5, abs=0.01)  # "√0.1 ≈ 32% apart"
    tmd = preset("Den Hartog").system
    assert max(peaks(tmd.resized(1), 0, 400.0)) == pytest.approx(40.0, rel=0.01)  # "40 times"
    assert max(peaks(tmd, 0, 400.0)) < 6.0  # "under 6"
    undamped = ChainSystem(tmd.masses, tmd.stiffness, [0.0, tmd.damping[1]])
    assert peaks(undamped, 0, 400.0) == pytest.approx([6.4, 6.4], rel=0.005)  # "about 6.4", equal
    for c2, count in ((0.0, 2), (5.0, 1)):  # "Set c2 to 0: two sharp peaks. Set it to 5: one"
        assert len(peaks(ChainSystem(tmd.masses, tmd.stiffness, [tmd.damping[0], c2]), 0, 400.0)) == count


def coupling_errors(system, split, mode_a=0, mode_b=0, a_mass="tip"):
    m = coupled_model(system, split, mode_a, mode_b, a_mass)
    return m, [(c.error, c.mac) for c in compare_coupled(m, modal_analysis(system))]


def test_modal_coupling_try_it():
    _, errs = coupling_errors(preset("absorber (undamped)").system, 1)
    assert all(abs(e) < 1e-9 for e, _ in errs)
    tuned = ChainSystem([1, 1, 0.1, 0.1], [400, 400, 40, 40], [2, 2, 0.2, 0.2])
    m, errs = coupling_errors(tuned, 2)
    assert m.mass_ratio == pytest.approx(0.14, abs=0.005)
    assert [e for e, _ in errs] == pytest.approx([0.002, 0.0033], abs=0.0005)  # "about 0.2%"
    _, errs = coupling_errors(tuned, 2, a_mass="effective")
    assert max(abs(e) for e, _ in errs) == pytest.approx(0.03, abs=0.002)  # "about 3%", either sign
    _, errs = coupling_errors(ChainSystem.uniform(8), 6, mode_a=1)
    assert errs[0][0] == pytest.approx(-0.14, abs=0.005)  # "14% too low"


def cms(interfaces, kept, method="craig-bampton"):
    s = ChainSystem.uniform(8)
    return [(c.error, c.mac) for c in compare_modes(component_mode_synthesis(s, interfaces, kept, method),
                                                   modal_analysis(s)) if c.error is not None]


@pytest.mark.parametrize(("interfaces", "kept", "method", "mode", "error", "tol"), [
    ([], [0], "craig-bampton", 1, 0.073, 0.001),  # Guyan, +7.3%
    ([], [1], "craig-bampton", 1, 0.0033, 0.0001),
    ([], [1], "craig-bampton", 2, 0.074, 0.001),
    ([], [2], "craig-bampton", 3, 0.044, 0.001),
    ([], [1], "rubin", 2, 0.16, 0.005),
    ([3], [1, 1], "craig-bampton", 4, 0.07, 0.002),
    ([3], [0, 0], "craig-bampton", 2, 0.11, 0.005),
    ([3], [2, 2], "craig-bampton", 6, 0.027, 0.001),
    ([6], [1, 0], "craig-bampton", 3, 0.50, 0.01),
    ([2, 5], [1, 1, 1], "craig-bampton", 4, 0.023, 0.001),
    ([2, 5], [0, 0, 0], "craig-bampton", 3, 0.16, 0.005),
    ([3], [1, 1], "rubin", 4, 0.10, 0.005),
    ([3], [1, 1], "macneal", 2, 0.22, 0.005),
    ([3], [9, 9], "macneal", 6, 0.66, 0.005),
])
def test_substructuring_try_it(interfaces, kept, method, mode, error, tol):
    assert cms(interfaces, kept, method)[mode - 1][0] == pytest.approx(error, abs=tol)


def test_background_single_mass_numbers():
    s = ChainSystem.uniform(1)
    m, k, c = s.masses[0], s.stiffness[0], s.damping[0]
    wn = (k / m) ** 0.5
    zeta = c / (2 * (k * m) ** 0.5)
    assert (wn, wn / (2 * np.pi), zeta) == pytest.approx((20.0, 3.18, 0.05), rel=0.002)
    assert 1 / (zeta * wn) == pytest.approx(1.0)  # time constant
    assert 2 * np.pi * zeta / (1 - zeta**2) ** 0.5 == pytest.approx(0.31, abs=0.005)  # log decrement
    assert max(peaks(s, 0, k, 1.0, 6.0)) == pytest.approx(1 / (2 * zeta), rel=0.002)  # Q = 10
    assert 2 * zeta * wn / (2 * np.pi) == pytest.approx(0.32, abs=0.005)  # half-power bandwidth, Hz


def test_background_rayleigh_damping_of_the_default_chain():
    zetas = [m.zeta_modal for m in modal_analysis(ChainSystem.uniform(4)).modes]
    assert zetas[0] == pytest.approx(0.017, abs=0.0005) and zetas[3] == pytest.approx(0.094, abs=0.0005)
    for m in modal_analysis(ChainSystem.uniform(4)).modes:
        assert m.zeta_modal == pytest.approx(0.005 * m.omega_n / 2)  # zeta = beta omega / 2


def test_background_isolation_above_root_2_fn():
    for c in (0.5, 2.0, 20.0):
        s = ChainSystem([1.0], [400.0], [c])
        f = np.array([0.99, 1.01]) * 2**0.5 * 20 / (2 * np.pi)
        t = np.abs(transmissibility(s, f))[:, 0]
        assert t[0] > 1 > t[1]
    f = np.linspace(0.01, 20, 20001)
    t3 = np.abs(transmissibility(ChainSystem.uniform(4), f))[:, 2]
    assert f[np.nonzero(t3 > 1)[0][-1]] == pytest.approx(1.65, abs=0.01)  # m3's crossover


def test_frf_matrix_three_forms_and_residual_mass():
    s = ChainSystem.uniform(4)
    r = modal_analysis(s)
    hi = np.array([60.0, 120.0])  # far above the modes: driving-point accelerance tends to 1/m
    w = 2 * np.pi * hi
    acc = np.abs(-(w**2) * frf(s, hi, 3)[:, 3])
    assert acc[1] == pytest.approx(1 / s.masses[3], rel=0.01)
    # Dropping mode 1 leaves a mass-like error -phi phi / w^2 above it.
    f = np.array([4.0, 5.5])
    w = 2 * np.pi * f
    terms = modal_frf_terms(s, r, exact=False)
    without_1 = sum(t.evaluate(f) for t in terms[1:])[:, 3, 3]
    error = frf(s, f, 3)[:, 3] - without_1
    phi = r.modes[0].shape_mass_normalized[3]
    assert error.real == pytest.approx(-(phi**2) / w**2, rel=0.1)


def test_background_rigid_body_root_with_k1_zero():
    s = ChainSystem.uniform(4)
    s.stiffness[0] = 0.0
    roots = modal_analysis(s).overdamped_roots
    assert min(roots, key=abs) == pytest.approx(0.0, abs=1e-6)
    second = sorted(roots, key=abs)[1]
    assert second == pytest.approx(-0.50, abs=0.005)  # close to −c1/(m1 + … + mN)


def test_frf_matrix_antiresonances_of_h22():
    s = ChainSystem.uniform(4)
    s.damping[:] = 0.0
    f = np.linspace(0.5, 8.0, 200001)
    h = np.abs(frf(s, f, 1)[:, 1])
    dips = [float(f[i]) for i in find_peaks(-np.log(h))[0]]
    assert dips == pytest.approx([1.97, 4.50, 5.15], abs=0.01)


def test_modal_coupling_absorber_damping_is_shared():
    r = modal_analysis(preset("absorber").system)
    assert [m.zeta_modal for m in r.modes] == pytest.approx([0.012, 0.012], abs=0.0015)


def test_background_friction_trial_loses_5_mm_a_cycle_and_sticks_after_four():
    from vib_tutorial.gui.sim_trials import TRIALS

    note = next(t.note for t in TRIALS if t.name.startswith("Friction"))
    sim = Simulator(ChainSystem([1.0], [400.0], [0.01], [0.5]))
    sim.set_displacement(np.array([0.02]))
    out = []
    while sim.t < 2.0 - 1e-12:
        out.append(sim.advance(2.0 - sim.t)[:3])
    t, x, v = (np.concatenate(a)[:, 0] if a[0].ndim > 1 else np.concatenate(a) for a in zip(*out))
    turns = np.flatnonzero(v[:-1] * v[1:] < 0) + 1
    lost = 0.02 - np.abs(x[turns[1::2]])  # after each whole cycle
    np.testing.assert_allclose(np.diff(np.concatenate([[0.0], lost]))[:3], 0.005, atol=2e-4)
    still = t[np.flatnonzero(v != 0)[-1]]
    assert still == pytest.approx(1.3, abs=0.05) and abs(x[-1]) < 1.25e-3
    assert "= 5 mm" in note and "four cycles (1.3 s)" in note


def test_background_rayleigh_quotient_and_effective_masses():
    s = ChainSystem.uniform(4)
    M, _, K = s.matrices()
    v = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.sqrt(v @ K @ v / (v @ M @ v)) / (2 * np.pi) == pytest.approx(1.162, abs=5e-4)
    assert modal_analysis(s).modes[0].omega_n / (2 * np.pi) == pytest.approx(1.105, abs=5e-4)
    gamma2 = [(m.shape_mass_normalized @ M @ np.ones(4)) ** 2 for m in modal_analysis(s).modes]
    np.testing.assert_allclose(gamma2, [3.57, 0.33, 0.08, 0.01], atol=0.006)


def test_frf_and_coupling_trials():
    from vib_tutorial.core.modal_coupling import coupled_model as model
    from vib_tutorial.gui.coupling_notes import TRIALS as COUPLING

    s = ChainSystem.uniform(4)
    _, _, K = s.matrices()
    phi = np.array([m.shape_mass_normalized for m in modal_analysis(s).modes]).T
    w2 = np.array([m.omega_n**2 for m in modal_analysis(s).modes])
    share = phi[0] ** 2 / w2 / np.linalg.inv(K)[0, 0]  # m1's static compliance, mode by mode
    assert share[2:].sum() == pytest.approx(0.24, abs=0.005)
    assert (phi[3, 0] ** 2 / w2[0]) / np.linalg.inv(K)[3, 3] == pytest.approx(0.89, abs=0.005)
    assert abs(phi[2, 1]) < 1e-12  # m3 is a node of mode 2
    errors = {}
    for t in COUPLING:
        system = t.system()
        comparisons = compare_coupled(model(system, t.split, t.mode_a, t.mode_b, t.a_mass, True), modal_analysis(system))
        errors[t.name] = [100 * c.error for c in comparisons]
    np.testing.assert_allclose(errors["Two equal halves: pushed far apart"][1], 4.9, atol=0.05)
    np.testing.assert_allclose(errors["A light half, tuned: splitting"], [0.2, 0.3], atol=0.05)
    np.testing.assert_allclose(errors["The wrong mass for A"], [2.9, -2.2], atol=0.06)
    assert errors["A mode left out (8 masses)"][0] == pytest.approx(-14, abs=0.5)


# --------------------------------------------------------------------- the One mass and Signals pages' trials
def test_single_oscillator_trials():
    import math

    from vib_tutorial.core.sdof import Oscillator, receptance

    osc = Oscillator()
    assert osc.natural_hz == pytest.approx(3.18, abs=0.005) and osc.zeta == pytest.approx(0.05)
    assert math.exp(osc.log_decrement) == pytest.approx(1.37, abs=0.005)
    assert osc.log_decrement == pytest.approx(0.315, abs=0.0005) and osc.time_constant == pytest.approx(1.0)
    slow = Oscillator(1, 400, 80).poles()
    assert sorted(slow.real) == pytest.approx([-74.6, -5.4], abs=0.05)
    two = Oscillator(1, 400, 80).poles().real.max()  # ζ = 2
    assert -two / osc.omega_n == pytest.approx(0.27, abs=0.005) and -1 / two == pytest.approx(0.19, abs=0.005)
    assert 1 / 20 == pytest.approx(0.05)  # critical: a double root at −ω_n
    assert (1 / 400) / (2 * 0.05) == pytest.approx(0.025)  # 25 mm at resonance, Q = 10
    lo, hi = osc.half_power_hz()
    assert (lo, hi) == pytest.approx((3.01, 3.33), abs=0.005)
    beat = osc.natural_hz - 3.0
    assert beat == pytest.approx(0.18, abs=0.005) and 1 / beat == pytest.approx(5.5, abs=0.05)
    undamped = Oscillator(1, 400, 0)
    assert abs(receptance(undamped, np.array([3.0]))[0]) == pytest.approx(0.022, abs=0.0005)
    assert 1 / (2 * undamped.omega_n) == pytest.approx(0.025)  # F/2mω_n, m/s
    assert (1e-4 / 1) / (2 * 0.05) == pytest.approx(1e-3)
    assert 1 / (2 * osc.omega_n) == pytest.approx(0.025)  # the Nyquist circle's diameter 1/cω_n, m/N


def test_transient_trials():
    from vib_tutorial.core.sdof import Oscillator
    from vib_tutorial.core.transient import PulseKind, peak_ratio, step_metrics

    osc = Oscillator()
    m = step_metrics(osc)
    assert m.overshoot == pytest.approx(0.85, abs=0.005) and 2.5 * (1 + m.overshoot) == pytest.approx(4.64, abs=0.005)
    assert m.peak_time == pytest.approx(0.157, abs=0.0005) and m.settling == pytest.approx(3.8, abs=0.05)
    assert step_metrics(Oscillator(1, 400, 40)).settling == pytest.approx(0.29, abs=0.005)
    assert 2 * np.sin(np.pi / 20) == pytest.approx(0.31, abs=0.005)
    r = 0.26 / osc.period
    assert peak_ratio(0.0, PulseKind.HALF_SINE, np.array([r]))[0] == pytest.approx(1.77, abs=0.005)
    assert peak_ratio(0.05, PulseKind.HALF_SINE, np.array([r]))[0] == pytest.approx(1.65, abs=0.005)
    r = 0.864 / osc.period
    assert r == pytest.approx(2.75, abs=0.005)
    assert peak_ratio(0.0, PulseKind.RAMP, np.array([r]))[0] - 1 == pytest.approx(0.08, abs=0.005)
    assert peak_ratio(0.05, PulseKind.RAMP, np.array([r]))[0] - 1 == pytest.approx(0.06, abs=0.005)


def test_damping_models_trials():
    from vib_tutorial.core.damping_models import Matched, Model, coulomb_stop

    m = Matched()
    wn = m.omega_n
    assert m.energy_per_cycle(Model.VISCOUS, np.array([wn]))[0] == pytest.approx(0.0126, abs=5e-5)
    assert m.friction == pytest.approx(0.31, abs=0.005)
    assert m.energy_per_cycle(Model.VISCOUS, np.array([2 * wn]))[0] == pytest.approx(0.0251, abs=5e-5)
    assert m.energy_per_cycle(Model.ZENER, np.array([2 * wn]))[0] == pytest.approx(0.0107, abs=5e-5)
    assert 4 * m.friction / np.pi == pytest.approx(0.4, abs=0.005)
    assert 4 * m.friction / m.stiffness == pytest.approx(0.0031, abs=5e-5)
    half_cycles, where = coulomb_stop(m.stiffness, m.friction, 0.01)
    assert half_cycles == 6 and abs(where) == pytest.approx(0.00058, abs=5e-6)
    k0, k1, _ = m.zener()
    assert (k0, k0 + k1) == pytest.approx((364, 444), abs=0.5)


def test_isolation_trials():
    from vib_tutorial.core.isolation import Isolator, Mount, deflection_for

    soft = Isolator()
    X, floor, T = soft.at_speed()
    assert soft.natural_hz == pytest.approx(7.05, abs=0.005)
    assert float(soft.force_at(25.0)) == pytest.approx(247, abs=0.5) and floor == pytest.approx(23, abs=0.5)
    assert T == pytest.approx(0.09, abs=0.005) and 20 * np.log10(T) == pytest.approx(-21, abs=0.5)
    assert X == pytest.approx(109e-6, abs=0.5e-6)
    assert deflection_for(25.0, 0.9) == pytest.approx(4.37e-3, abs=5e-6)
    stiff = Isolator(deflection=3e-4)
    assert stiff.natural_hz == pytest.approx(28.8, abs=0.05) and stiff.at_speed()[2] == pytest.approx(3.9, abs=0.05)
    assert Isolator(zeta=0.3).at_speed()[2] == pytest.approx(0.20, abs=0.005)
    assert (soft.natural_hz / soft.speed_hz) ** 2 == pytest.approx(1 / 12, abs=0.005)
    f = np.linspace(1, 30, 20001)
    relax = Isolator(zeta=0.3, mount=Mount.RELAXATION, rpm=3000)
    plain = Isolator(zeta=0.3, rpm=3000)
    assert relax.transmissibility(f).max() == pytest.approx(2.1, abs=0.05)
    assert plain.transmissibility(f).max() == pytest.approx(2.0, abs=0.05)
    assert (relax.at_speed()[2], plain.at_speed()[2]) == pytest.approx((0.070, 0.088), abs=0.0005)
    fast = (Isolator(zeta=0.3, mount=Mount.RELAXATION, rpm=6000), Isolator(zeta=0.3, rpm=6000))
    assert [i.at_speed()[2] for i in fast] == pytest.approx([0.019, 0.043], abs=0.0005)
    assert Isolator(mount=Mount.TWO_STAGE).at_speed()[2] == pytest.approx(0.038, abs=0.0005)
    assert Isolator(mount=Mount.TWO_STAGE, rpm=3000).at_speed()[2] == pytest.approx(0.002, abs=0.0005)
    assert Isolator(rpm=3000).at_speed()[2] == pytest.approx(0.025, abs=0.0005)
    assert Isolator(block=300).at_speed()[0] == pytest.approx(27e-6, abs=0.5e-6)


def test_signals_trials():
    from vib_tutorial.core.signals import Kind, Source, Window, mass_loaded, nearest_line_reading, sample
    from vib_tutorial.core.signals import spectrum as averaged
    from vib_tutorial.core.signals import usable_band, window_props

    def reading(source, f, window, block=1024, averages=1, filtered=True):
        n = block + (averages - 1) * block // 2
        _, x = sample(source, 1024.0, n, filtered)
        return nearest_line_reading(averaged(x, 1024.0, block, window, 0.5), f)[1]

    assert reading(Source(f1=50.0, a2=0.0), 50.0, Window.RECTANGULAR) == pytest.approx(1.0, abs=5e-4)
    pair = Source(f1=50.5, f2=70.0, a2=0.001)
    assert reading(pair, 50.5, Window.RECTANGULAR) == pytest.approx(0.64, abs=0.005)
    assert reading(pair, 70.0, Window.RECTANGULAR) == pytest.approx(0.015, abs=0.0005)
    assert reading(pair, 50.5, Window.HANN) == pytest.approx(0.85, abs=0.005)
    assert reading(pair, 70.0, Window.HANN) == pytest.approx(0.0010, abs=0.00005)
    assert reading(pair, 50.5, Window.FLAT_TOP) == pytest.approx(0.999, abs=0.0005)
    flat = window_props(Window.FLAT_TOP)
    assert flat.main_lobe == 5 and flat.enbw == pytest.approx(3.8, abs=0.05)
    alias = Source(f1=900.0, a2=0.0)
    assert reading(alias, 124.0, Window.HANN, filtered=False) == pytest.approx(1.0, abs=5e-4)
    assert reading(alias, 124.0, Window.HANN) == pytest.approx(0.0015, abs=0.00005)
    assert 1024 / 2.56 == 400
    assert 0.5 ** 2 / 512 == pytest.approx(4.9e-4, abs=5e-6)
    hann = window_props(Window.HANN)
    for block, level in ((4096, 0.019), (256, 0.077)):
        assert 2 * 0.5 * np.sqrt(hann.enbw / block) == pytest.approx(level, abs=0.0005)
    assert 0.05 / 0.019 == pytest.approx(2.6, abs=0.05)
    ringing = Source(Kind.RINGING, f1=50.0, zeta=0.01)
    assert 2 * ringing.zeta * ringing.f1 == pytest.approx(1.0) and 1024 / 4096 == 0.25
    assert usable_band(0.01) == pytest.approx(0.22, abs=0.005) and 0.218 * 30000 == pytest.approx(6.5e3, abs=60)
    assert usable_band(0.7, accelerometer=False) == pytest.approx(1.7, abs=0.05)
    assert usable_band(0.7, accelerometer=False) * 4.5 == pytest.approx(7.6, abs=0.05)
    assert mass_loaded(200.0, 0.5, 0.03) / 200 - 1 == pytest.approx(-0.029, abs=0.0005)


def test_periodic_trials():
    import math

    from vib_tutorial.core.periodic import Forcing, Wave, periodic_steady, response_coefficients, series, series_error
    from vib_tutorial.core.sdof import Oscillator

    osc = Oscillator()
    t = np.linspace(0.0, 1.0, 20001)
    sq = Forcing(Wave.SQUARE)
    assert np.max(series(sq.coefficients(5), 1.0, t)) - 1 == pytest.approx(0.19, abs=0.005)
    fine = np.linspace(0.0, 0.01, 20001)
    assert np.max(series(sq.coefficients(401), 1.0, fine)) - 1 == pytest.approx(0.18, abs=0.005)
    third = Forcing(Wave.SQUARE, freq_hz=osc.natural_hz / 3)
    assert 4 / (3 * math.pi) == pytest.approx(0.42, abs=0.005) and 1 / (2 * osc.omega_n) == pytest.approx(0.025)
    X = np.abs(response_coefficients(osc, third, 3))
    assert (X[3], X[1]) == pytest.approx((0.0106, 0.0036), abs=5e-5)
    swing = lambda fo: float(np.max(periodic_steady(osc, fo, t * fo.period)))  # noqa: E731
    assert swing(third) == pytest.approx(0.0139, abs=5e-5)
    assert swing(Forcing(Wave.SQUARE, freq_hz=osc.natural_hz / 2)) == pytest.approx(0.0050, abs=5e-5)
    blows = periodic_steady(osc, Forcing(Wave.IMPULSES, 0.1), t)
    assert (blows.min(), blows.max()) == pytest.approx((-0.0045, 0.0052), abs=5e-5)
    assert 0.1 / 400 == pytest.approx(0.00025) and osc.time_constant == pytest.approx(1.0)
    assert 2 / math.pi == pytest.approx(0.64, abs=0.005) and 2 / math.pi / 400 == pytest.approx(0.00159, abs=5e-6)
    tri, sq1 = Forcing(Wave.TRIANGLE), Forcing(Wave.SQUARE)
    assert series_error(osc, tri, 3) == pytest.approx(0.019, abs=0.0005)
    assert series_error(osc, sq1, 3) == pytest.approx(0.057, abs=0.0005)
    assert series_error(osc, tri, 9) == pytest.approx(0.0010, abs=0.00005)
    assert series_error(osc, sq1, 9) == pytest.approx(0.0065, abs=0.00005)


def test_energy_methods_trials():
    from vib_tutorial.core.energy_methods import (
        Basis,
        Shape,
        contaminated,
        dunkerley,
        dunkerley_terms,
        exact,
        holzer,
        holzer_roots,
        iterate,
        rayleigh,
        ritz_table,
        trial_shape,
    )
    from vib_tutorial.gui.energy_methods import chain

    def hz(w2):
        return np.sqrt(w2) / (2 * np.pi)

    s = chain()
    f = exact(s)[0] / (2 * np.pi)
    assert f == pytest.approx([1.105, 3.183, 4.877, 5.982], abs=0.0005)

    def ray(shape, eps=0.0, passes=0, system=s):
        return hz(rayleigh(system, iterate(system, contaminated(system, trial_shape(system, shape), eps), passes)))

    assert ray(Shape.STATIC) == pytest.approx(1.112, abs=0.0005)
    assert ray(Shape.STATIC) / f[0] - 1 == pytest.approx(0.0055, abs=0.00005)
    assert ray(Shape.LINEAR) / f[0] - 1 == pytest.approx(0.051, abs=0.0005)
    assert ray(Shape.LINEAR, passes=1) / f[0] - 1 == pytest.approx(0.00063, abs=0.000005)
    assert ray(Shape.SINE) == pytest.approx(f[0], rel=1e-12)
    assert ray(Shape.SINE, 0.1) / f[0] - 1 == pytest.approx(0.046, abs=0.0005)
    assert ray(Shape.SINE, 0.2) / f[0] - 1 == pytest.approx(0.166, abs=0.0005)
    assert ray(Shape.TIP) == pytest.approx(3.18, abs=0.005) and ray(Shape.TIP) / f[0] - 1 == pytest.approx(1.88, abs=0.005)
    assert ray(Shape.TIP, passes=1) / f[0] - 1 == pytest.approx(0.051, abs=0.0005)
    assert ray(Shape.TIP, passes=2) / f[0] - 1 == pytest.approx(0.00063, abs=0.000005)
    assert (f[0] / f[1]) ** 2 == pytest.approx(0.12, abs=0.005)
    assert hz(dunkerley_terms(s)) == pytest.approx([3.18, 2.25, 1.84, 1.59], abs=0.005)
    assert hz(dunkerley(s)) == pytest.approx(1.007, abs=0.0005) and hz(dunkerley(s)) / f[0] - 1 == pytest.approx(
        -0.089, abs=0.0005)
    tip = chain(masses=(1.0, 1.0, 1.0, 5.0))
    ft = exact(tip)[0] / (2 * np.pi)
    assert hz(dunkerley_terms(tip))[-1] == pytest.approx(0.71, abs=0.005)
    assert (hz(dunkerley(tip)), ft[0]) == pytest.approx((0.624, 0.653), abs=0.0005)
    assert hz(dunkerley(tip)) / ft[0] - 1 == pytest.approx(-0.044, abs=0.0005)
    assert ray(Shape.STATIC, system=tip) / ft[0] - 1 == pytest.approx(0.0025, abs=0.00005)
    table = ritz_table(s, Basis.KRYLOV)
    assert table[1, 0] == pytest.approx(f[0], abs=0.00005)
    assert table[1, 1] == pytest.approx(3.30, abs=0.005) and table[1, 1] / f[1] - 1 == pytest.approx(0.036, abs=0.0005)
    assert (table[2, 1], table[2, 2]) == pytest.approx((3.184, 5.02), abs=0.005)
    stiff = chain(stiffness=(4000.0, 400.0, 400.0, 400.0))
    fs = exact(stiff)[0] / (2 * np.pi)
    powers = ritz_table(stiff, Basis.POWERS)
    assert (powers[1, 0], fs[0]) == pytest.approx((1.63, 1.38), abs=0.005)
    assert powers[1, 0] / fs[0] - 1 == pytest.approx(0.18, abs=0.005)
    assert (powers[1, 1], fs[1]) == pytest.approx((6.04, 3.89), abs=0.005)
    assert ritz_table(stiff, Basis.KRYLOV)[0, 0] / fs[0] - 1 < 0.008
    assert holzer(s, 2 * np.pi * 1.0)[-1].spring == pytest.approx(61, abs=0.5)
    assert holzer_roots(s, 7.0) == pytest.approx(f, abs=0.0005)


def test_integration_trials():
    import math

    from vib_tutorial.core.integration import Load, Model, Scheme, exact, simulate, spectral, stable_limit

    def at(per_period, scheme, alpha=-0.1):
        return spectral(scheme, 2 * math.pi / per_period, 0.0, alpha)

    assert Model().frequencies_hz()[0] == pytest.approx(3.18, abs=0.005)
    assert at(10, Scheme.CENTRAL).period_error == pytest.approx(-0.017, abs=0.0005)
    assert at(10, Scheme.AVERAGE).period_error == pytest.approx(0.032, abs=0.0005)
    assert at(10, Scheme.LINEAR).period_error == pytest.approx(0.016, abs=0.0005)
    assert at(10, Scheme.RK4).period_error == pytest.approx(0.0011, abs=0.00005)
    assert at(10, Scheme.RK4).damping == pytest.approx(0.00065, abs=0.000005)
    assert 10 * at(10, Scheme.AVERAGE).period_error == pytest.approx(1 / 3, abs=0.02)
    assert 2 * math.pi / 3.0 == pytest.approx(2.09, abs=0.005) and at(3.0, Scheme.CENTRAL).radius > 1
    assert 2 * math.pi / 3.2 == pytest.approx(1.96, abs=0.005)
    assert at(3.2, Scheme.CENTRAL).radius == pytest.approx(1.0)
    assert at(3.2, Scheme.CENTRAL).period_error == pytest.approx(-0.29, abs=0.005)
    assert at(3.0, Scheme.AVERAGE).period_error == pytest.approx(0.30, abs=0.005)
    assert at(1.0, Scheme.AVERAGE).period_error + 1 == pytest.approx(2.5, abs=0.05)
    assert at(1.0, Scheme.AVERAGE).radius == pytest.approx(1.0)
    assert stable_limit(Scheme.LINEAR) == pytest.approx(2 * math.sqrt(3)) and 2 * math.sqrt(3) / (2 * math.pi) == (
        pytest.approx(0.55, abs=0.005))
    assert at(1.0, Scheme.LINEAR).radius > 1
    rk = at(5, Scheme.RK4)
    assert rk.period_error == pytest.approx(0.011, abs=0.0005) and 1 - rk.radius == pytest.approx(0.022, abs=0.0005)
    assert rk.damping == pytest.approx(0.018, abs=0.0005) and rk.radius ** 50 == pytest.approx(0.33, abs=0.005)
    assert at(5, Scheme.AVERAGE).period_error == pytest.approx(0.12, abs=0.005)
    eu = at(50, Scheme.EULER).radius
    assert eu - 1 == pytest.approx(0.008, abs=0.0005) and eu ** 50 - 1 == pytest.approx(0.48, abs=0.005)
    assert eu ** 500 == pytest.approx(50, abs=1)
    stiff = Model(stiff_hz=1000.0)
    f = stiff.frequencies_hz()
    assert f == pytest.approx([3.03, 1049], abs=0.5) and f[0] == pytest.approx(3.03, abs=0.005)
    h = 1 / f[0] / 50
    assert 2 * math.pi * f[1] * h == pytest.approx(43, abs=0.5) and 1 / (f[1] * h) == pytest.approx(1 / 7, abs=0.005)
    for scheme in (Scheme.CENTRAL, Scheme.RK4):
        x = simulate(scheme, stiff, Load(), h, 250)
        assert np.isnan(x[: int(0.020 / h) + 1]).any()
    assert np.all(np.isfinite(simulate(Scheme.AVERAGE, stiff, Load(), h, 250)))
    assert spectral(Scheme.HHT, 2 * math.pi * f[1] * h, 0.0, -0.3).radius == pytest.approx(0.57, abs=0.005)
    assert spectral(Scheme.HHT, 2 * math.pi * f[0] * h, 0.0, -0.3).damping == pytest.approx(0.00004, abs=0.000005)
    m, load = Model(), Load(0.0, 1.0, 2.0)
    h = 1 / m.frequencies_hz()[0] / 10
    t = np.arange(101) * h
    ref = exact(m, load, t)
    errs = {s: np.max(np.abs(simulate(s, m, load, h, 100) - ref)) / np.max(np.abs(ref))
            for s in (Scheme.FOH, Scheme.CENTRAL, Scheme.AVERAGE, Scheme.RK4)}
    assert errs[Scheme.FOH] == pytest.approx(0.013, abs=0.0005)
    assert [errs[s] for s in (Scheme.RK4, Scheme.CENTRAL, Scheme.AVERAGE)] == pytest.approx([0.03, 0.41, 0.61],
                                                                                           abs=0.005)


def test_fatigue_trials():
    from vib_tutorial.core.fatigue import (
        Curve,
        Stress,
        dirlik_damage,
        life_text,
        narrow_band_damage,
        rates,
        record,
    )

    c, c5 = Curve(), Curve(b=5.0)
    one = Stress()
    r = record(one, 600.0)
    assert rates(one).alpha2 == pytest.approx(0.95, abs=0.005)
    for est in (narrow_band_damage(one, c), dirlik_damage(one, c)):
        assert abs(est / r.damage_rate(c) - 1) < 0.015
    assert life_text(r.damage_rate(c)) == "25.1 days"
    two = Stress(sigma2=10.0)
    r2 = record(two, 600.0)
    assert rates(two).alpha2 == pytest.approx(0.46, abs=0.005) and rates(two).nu0 == pytest.approx(6.9, abs=0.05)
    assert narrow_band_damage(two, c) / r2.damage_rate(c) == pytest.approx(1.6, abs=0.05)
    assert dirlik_damage(two, c) / r2.damage_rate(c) == pytest.approx(0.92, abs=0.005)
    assert narrow_band_damage(two, c5) / r2.damage_rate(c5) == pytest.approx(1.6, abs=0.05)
    assert dirlik_damage(two, c5) / r2.damage_rate(c5) == pytest.approx(0.82, abs=0.005)
    average = np.mean([record(two, 600.0, k).damage_rate(c5) for k in range(1, 21)])
    assert dirlik_damage(two, c5) / average == pytest.approx(0.85, abs=0.005)
    assert life_text(record(Stress(sigma1=40.0), 600.0).damage_rate(c)) == "3.13 days"
    assert 1.1 ** 3 == pytest.approx(1.33, abs=0.005)
    wide = Stress(zeta1=0.3)
    rw = record(wide, 600.0)
    assert rates(wide).alpha2 == pytest.approx(0.64, abs=0.005)
    assert narrow_band_damage(wide, c) / rw.damage_rate(c) == pytest.approx(1.15, abs=0.005)
    assert abs(dirlik_damage(wide, c) / rw.damage_rate(c) - 1) < 0.02
    short = [record(two, 20.0, k).damage_rate(c5) / r2.damage_rate(c5) for k in range(1, 11)]
    assert (min(short), max(short)) == pytest.approx((0.67, 1.53), abs=0.005)


def test_self_excited_trials():
    from vib_tutorial.core.self_excited import (
        Flutter,
        Galloping,
        VanDerPol,
        Viv,
        dominant_hz,
        galloping_branches,
        settled_amplitude,
    )

    v = VanDerPol()
    t, x, _ = v.simulate(60)
    assert settled_amplitude(x) == pytest.approx(0.0089, abs=0.00005) and -v.eps / 2 == pytest.approx(-0.1)
    t, x, _ = VanDerPol(eps=5.0).simulate(20)
    assert 1 / dominant_hz(x, t[1] - t[0]) == pytest.approx(1.84, abs=0.03)
    g = Galloping()
    assert g.onset == pytest.approx(3.81, abs=0.005)
    U = np.linspace(4.0, 7.5, 351)
    pairs = [b for b in galloping_branches(g, U) if not b.stable]
    band = U[np.isfinite(pairs[0].amplitude)]
    assert (band.min(), band.max()) == pytest.approx((4.7, 7.0), abs=0.05)
    at6 = sorted(b.amplitude[0] / g.side for b in galloping_branches(g, np.array([6.0])) if b.stable)
    assert at6 == pytest.approx([0.51, 1.28], abs=0.005)
    for start, want in ((0.001, 0.51), (0.12, 1.28)):
        _, y, _ = Galloping(speed=6.0, start=start).simulate(600)
        assert settled_amplitude(y) / g.side == pytest.approx(want, abs=0.01)
    _, y, _ = Galloping(speed=12.0).simulate(200)
    assert settled_amplitude(y) / g.side == pytest.approx(2.7, abs=0.05) and 12 / g.onset == pytest.approx(3.1, abs=0.1)
    viv = Viv()
    assert viv.mass_ratio == pytest.approx(10.2, abs=0.05) and viv.strouhal * 5.5 == pytest.approx(1.1)
    amp, freq = viv.sweep(np.array([4.0, 4.5, 5.5, 6.25, 6.5, 9.0]))
    assert amp[2] == pytest.approx(0.21, abs=0.005) and freq[2] == pytest.approx(1.0, abs=0.01)
    assert freq[1:4] == pytest.approx([1.0, 1.0, 1.0], abs=0.04) and abs(freq[4] - 1) > 0.04
    assert abs(freq[0] - 1) > 0.04
    assert amp[5] == pytest.approx(0.019, abs=0.0005) and freq[5] == pytest.approx(1.7, abs=0.02)
    f = Flutter()
    assert f.flutter_speed() == pytest.approx(19.1, abs=0.05) and f.divergence_speed() == pytest.approx(39, abs=0.5)
    fr, _ = f.modes(np.array([0.0, 19.0]))
    assert fr[0] == pytest.approx([2.93, 6.71], abs=0.1) and fr[1] == pytest.approx([3.4, 5.0], abs=0.05)
    assert Flutter(x_cg=-0.2).flutter_speed() == pytest.approx(f.divergence_speed(), abs=0.05)
    assert Flutter(x_cg=0.0).flutter_speed() == pytest.approx(17.7, abs=0.05)
