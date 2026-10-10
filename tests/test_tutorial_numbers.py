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
