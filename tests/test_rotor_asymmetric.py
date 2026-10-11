"""The asymmetric shaft: its turning stiffness, Floquet modes, harmonic balance and split time step."""

import math

import numpy as np
import pytest

import vib_tutorial.core.rotor as rotor
from vib_tutorial.core.rotor import (
    RPM,
    XD,
    YD,
    Bearing,
    RotorSimulator,
    RotorSamples,
    RotorSystem,
    campbell,
    gravity_response,
    synchronous_response,
    whirl_modes,
)
from vib_tutorial.core.rotor_presets import rotor_presets
from vib_tutorial.core.rotor_stability import AsymmetryChart, growth_rate, least_damped, stability_onset
from vib_tutorial.core.rotor_sweep import RevolutionTracker, TrackedSweep, probe, station_motion, steady_vectors, vibration

ANISO = Bearing(ky=8e4)
TWO_POLE = next(p for p in rotor_presets() if p.name.startswith("Two-pole")).system


def run(sim: RotorSimulator, duration: float) -> RotorSamples:
    """Advance by `duration` s, looping over the simulator's cap on steps per call."""
    parts, end = [], sim.t + duration
    while sim.t < end - 1e-12:
        r = sim.advance(end - sim.t)
        if not r.t.size:
            break
        parts.append(r)
    return RotorSamples(*(np.concatenate([getattr(p, f) for p in parts]) for f in ("t", "q", "omega", "phase")))


def harmonic(r: RotorSamples, order: int, dof: int) -> tuple[float, complex]:
    """(mean, V) of q[dof] ≈ mean + Re(V e^{i order φ}), least squares over the samples."""
    ph = r.phase
    A = np.column_stack([np.ones_like(ph), np.cos(order * ph), -np.sin(order * ph)])
    c, *_ = np.linalg.lstsq(A, r.q[:, dof], rcond=None)
    return c[0], complex(c[1], c[2])


def steady(system: RotorSystem, omega: float, settle: float = 3.0) -> RotorSamples:
    sim = RotorSimulator(system)
    sim.omega = sim.target = omega
    sim.state[:] = 0.0
    sim.state[:system.n_dof] = system.static_deflection(omega)
    run(sim, settle)
    return run(sim, 0.5)


def test_validation_and_stiffness():
    with pytest.raises(ValueError):
        RotorSystem(asymmetry=1.0)
    with pytest.raises(ValueError):
        RotorSystem(asymmetry=-0.1)
    s = RotorSystem(asymmetry=0.2, asymmetry_angle=0.3)
    ks = s.shaft_stiffness()
    # Bending toward the stiff axis meets EI(1 + Δ); across it EI(1 - Δ).
    for phase in (0.0, 1.1):
        K = s.stiffness_at(0.0, phase) - s.stiffness(0.0)
        theta = phase + s.asymmetry_angle
        e = np.zeros(s.n_dof)
        e[[XD, YD]] = math.cos(theta), math.sin(theta)
        n = np.zeros(s.n_dof)
        n[[XD, YD]] = -math.sin(theta), math.cos(theta)
        assert e @ K @ e == pytest.approx(0.2 * ks[1, 1])
        assert n @ K @ n == pytest.approx(-0.2 * ks[1, 1])
        assert e @ K @ n == pytest.approx(0.0, abs=1e-6 * ks[1, 1])
    # The stiffer axis system's EI.
    assert s.axis_system(True).ei == pytest.approx(1.2 * s.ei) and s.axis_system(False).ei == pytest.approx(0.8 * s.ei)
    assert not s.axis_system(True).asymmetry


def test_a_vanishing_asymmetry_is_the_round_shaft():
    s0 = RotorSystem(position=0.3, bow=10e-6, gravity=9.81)
    s = s0.with_(asymmetry=1e-10)
    for w in (0.0, 80.0, 400.0):
        a, b = whirl_modes(s0, w), whirl_modes(s, w)
        np.testing.assert_allclose(b.freq_hz, a.freq_hz, rtol=1e-6)
        np.testing.assert_allclose(b.zeta, a.zeta, rtol=1e-5)
    w = np.array([50.0, 160.0, 300.0])
    np.testing.assert_allclose(synchronous_response(s, w), synchronous_response(s0, w), atol=1e-12)
    q0, q2 = gravity_response(s, w)
    np.testing.assert_allclose(q0.real, s0.static_deflection(w), atol=1e-12)
    assert np.abs(q2).max() < 1e-12


def test_rotating_frame_and_hill_agree_on_isotropic_supports():
    s = RotorSystem(asymmetry=0.2, asymmetry_angle=0.3, position=0.4)
    for w in (30.0, 150.0, 170.0, 300.0):
        exact, _ = rotor._rotating_modes(s, w)
        hill, _ = rotor._hill_modes(s, w)
        # Hill's copies of a mode locked to 1X are evenly split; it may keep one fewer of those.
        for lam in hill:
            assert np.abs(exact - lam).min() < 1e-6 * abs(lam)
        assert exact.real.max() == pytest.approx(hill.real.max(), abs=1e-6)


def test_hill_is_converged_on_anisotropic_supports(monkeypatch):
    s = RotorSystem(asymmetry=0.2, bearing_a=ANISO, bearing_b=ANISO)
    for w in (80.0, 160.0, 170.0):
        lam3, _ = rotor._hill_modes(s, w)
        monkeypatch.setattr(rotor, "HILL_ORDER", 6)
        lam6, _ = rotor._hill_modes(s, w)
        monkeypatch.setattr(rotor, "HILL_ORDER", 3)
        assert lam3.size == lam6.size
        np.testing.assert_allclose(np.sort(lam3.imag), np.sort(lam6.imag), rtol=1e-6)
        assert lam3.real.max() == pytest.approx(lam6.real.max(), abs=1e-6)


def test_harmonic_balance_closes_on_isotropic_supports():
    # A hair of anisotropy brings in all the harmonics; they stay empty, so the isotropic set was exact.
    s = RotorSystem(asymmetry=0.15, asymmetry_angle=0.4, gravity=9.81, unbalance_angle=1.0)
    hair = s.with_(bearing_a=Bearing(ky=5e4 * (1 + 1e-12)))
    assert s.isotropic_supports and not hair.isotropic_supports
    w = np.array([70.0, 150.0, 250.0])
    np.testing.assert_allclose(synchronous_response(hair, w), synchronous_response(s, w), atol=1e-12)
    for a, b in zip(gravity_response(hair, w), gravity_response(s, w)):
        np.testing.assert_allclose(a, b, atol=1e-12)
    # Gravity's 2X on isotropic supports is a circular forward whirl.
    q2 = gravity_response(s, [82.0])[1][0]
    assert q2[YD] == pytest.approx(-1j * q2[XD])


@pytest.mark.parametrize("supports", [Bearing(), ANISO])
def test_the_split_time_step_matches_the_harmonic_balance(supports):
    s = RotorSystem(asymmetry=0.15, asymmetry_angle=0.4, gravity=9.81, unbalance=0.0,
                    bearing_a=supports, bearing_b=supports)
    w = 75.0  # near the gravity critical
    r = steady(s, w)
    q0, q2 = gravity_response(s, [w])
    for dof in (XD, YD):
        mean, v = harmonic(r, 2, dof)
        assert mean == pytest.approx(q0[0, dof].real, abs=2e-3 * abs(q2[0, dof]))
        assert abs(v - q2[0, dof]) < 1e-3 * abs(q2[0, dof])
    s1 = s.with_(gravity=0.0, unbalance=20e-6, unbalance_angle=1.0)
    r = steady(s1, 150.0)
    Q = synchronous_response(s1, [150.0])[0]
    for dof in (XD, YD):
        _, v = harmonic(r, 1, dof)
        assert abs(v - Q[dof]) < 1e-3 * abs(Q[dof])


def test_rest_and_energy():
    s = RotorSystem(asymmetry=0.3, asymmetry_angle=0.5, gravity=9.81, unbalance=0.0)
    sim = RotorSimulator(s)
    # At rest it sags on the stiffness it has at that angle, and stays there.
    q = sim.q.copy()
    np.testing.assert_allclose(s.stiffness_at(0.0, 0.0) @ q, s.gravity_load(), atol=1e-9)
    sim.target = 0.0
    run(sim, 0.2)
    np.testing.assert_allclose(sim.q, q, atol=1e-12)
    assert sim.energy()[1] == pytest.approx(0.5 * (q - s.static_deflection(0.0)) @ s.stiffness_at(0.0, 0.0)
                                            @ (q - s.static_deflection(0.0)))


def test_the_bow_reads_as_runout_at_slow_roll():
    # Relative to its bent shape the asymmetric shaft is still unloaded, so turning slowly it reads the bow.
    s = RotorSystem(asymmetry=0.3, asymmetry_angle=0.7, unbalance=0.0, bow=20e-6, bow_angle=1.2)
    v = synchronous_response(s, [1e-3])[0]
    assert v[XD] == pytest.approx(20e-6 * np.exp(1.2j), rel=1e-6)


def test_the_undamped_band_lies_exactly_between_the_axes_critical_speeds():
    free = Bearing(cx=0.0, cy=0.0)
    s = RotorSystem(asymmetry=0.15, bearing_a=free, bearing_b=free)
    band = stability_onset(s, 3000.0 / RPM)
    soft, stiff = (min(w for w, r, _ in campbell(s.axis_system(k), 3000.0 / RPM).criticals if r > 0.1)
                   for k in (False, True))
    assert band.omega == pytest.approx(soft, abs=2e-3) and band.end == pytest.approx(stiff, abs=2e-3)
    assert round(band.omega * RPM) == 1479 and round(band.end * RPM) == 1645
    # The criticals Campbell lists are the axes' too.
    crits = [w for w, r, _ in campbell(s, 3000.0 / RPM).criticals if r > 0.1]
    assert crits[:2] == pytest.approx([soft, stiff])


def test_damping_sets_a_threshold_asymmetry():
    def unstable(delta: float, c: float) -> bool:
        b = Bearing(cx=c, cy=c)
        return stability_onset(RotorSystem(asymmetry=delta, bearing_a=b, bearing_b=b), 3000.0 / RPM) is not None

    assert not unstable(0.122, 100.0) and unstable(0.124, 100.0)
    assert not unstable(0.177, 400.0) and unstable(0.179, 400.0)


def test_inside_the_band_a_whirl_grows_locked_to_1x():
    s = RotorSystem(asymmetry=0.2)
    w = 165.0
    zeta, f, whirl = least_damped(s, w)
    assert zeta < 0 and f == pytest.approx(w / (2 * math.pi)) and whirl > 0.9
    sim = RotorSimulator(s)
    sim.omega = sim.target = w
    sim.state[:] = 0.0
    sim.tap(0.0, -0.005)
    run(sim, 1.0)
    r = run(sim, 1.0)
    amp = np.hypot(r.q[:, XD], r.q[:, YD])
    assert np.polyfit(r.t, np.log(amp), 1)[0] == pytest.approx(growth_rate(s, w), rel=0.01)


def test_two_pole_preset_numbers():
    s = TWO_POLE
    assert [round(w * RPM) for w, r, z in campbell(s, 3000.0 / RPM).criticals if r > 0.1] == [1491, 1665]
    band = stability_onset(s, 3000.0 / RPM)
    assert (round(band.omega * RPM), round(band.end * RPM)) == (1527, 1631)
    assert 1.0 / growth_rate(s, 1580.0 / RPM) == pytest.approx(0.56, abs=0.005)
    assert least_damped(s, 1580.0 / RPM)[1] == pytest.approx(26.3, abs=0.05)
    w = np.linspace(500.0, 1100.0, 6001) / RPM
    v2 = np.abs(probe(steady_vectors(s, w, order=2)[:, 1], "orbit"))
    i = int(np.argmax(v2))
    assert round(v2[i] * 1e6) == 447 and round(w[i] * RPM) == 785
    assert round(-s.static_deflection(0.0)[YD] * 1e6) == 384


def test_unbalance_angle_from_the_stiff_axis():
    from scipy.signal import find_peaks

    w = np.linspace(1300.0, 1900.0, 6001) / RPM

    def peaks(angle: float, delta: float = 0.1) -> list[tuple[int, int]]:
        s = RotorSystem(asymmetry=delta, unbalance_angle=math.radians(angle))
        v = np.abs(probe(steady_vectors(s, w)[:, 1], "orbit"))
        return [(round(w[i] * RPM), round(v[i] * 1e6)) for i in find_peaks(v)[0]]

    assert peaks(0) == [(1589, 827)]
    assert peaks(90) == [(1576, 848)]
    assert peaks(45) == [(1547, 201), (1624, 195)]
    assert peaks(0, 0.0) == [(1589, 217)]
    # Not a complex number times the unbalance: a quarter turn of the heavy spot doesn't turn the response.
    a, b = (synchronous_response(RotorSystem(asymmetry=0.1, unbalance_angle=t), [1400 / RPM])[0, XD]
            for t in (0.0, math.pi / 2))
    assert abs(b - 1j * a) > 0.5 * abs(a)


def test_tracking_takes_the_2x_vector():
    s = TWO_POLE.with_(unbalance=0.0)
    w = 785.0 / RPM
    sim = RotorSimulator(s)
    sim.omega = sim.target = w
    sim.state[:] = 0.0
    sim.state[:s.n_dof] = s.static_deflection(w)
    run(sim, 3.0)
    r = run(sim, 0.6)
    sweep = TrackedSweep(rate=0.0, up=True)
    RevolutionTracker(sweep).feed(r.phase, r.omega, station_motion(s, vibration(s, r.q, r.omega)))
    expected = steady_vectors(s, [w], order=2)[0]
    assert len(sweep) >= 3
    for v in sweep.vectors2:
        np.testing.assert_allclose(v, expected, atol=2e-3 * np.abs(expected).max())
    assert np.abs(sweep.vectors).max() < 1e-2 * np.abs(expected).max()  # no unbalance: no 1X
    _, v2, _ = sweep.trace(1, "orbit", order=2)
    assert np.abs(v2) == pytest.approx(447e-6, rel=0.01)


def test_asymmetry_chart():
    chart = AsymmetryChart(RotorSystem(), 3000.0 / RPM, delta_max=0.4, n_deltas=5, n_speeds=31)
    assert not chart.done and chart.work(0.0)  # at least one cell
    chart.run()
    assert chart.done and not np.isnan(chart.growth).any()
    assert np.all(chart.growth[0] < 0)  # a round shaft is stable
    # The tongue: between the axes' critical speeds, once the asymmetry passes the threshold.
    rpm = chart.omegas * RPM
    for i, d in enumerate(chart.asymmetries):
        unstable = rpm[chart.growth[i] > 0]
        if d < 0.12:
            assert not unstable.size
        else:
            assert unstable.size and chart.soft[i] * RPM < unstable.min() and unstable.max() < chart.stiff[i] * RPM
    assert np.all(np.diff(chart.soft) < 0) and np.all(np.diff(chart.stiff) > 0)
