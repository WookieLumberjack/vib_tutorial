"""The rotor on active magnetic bearings: the actuator, the controller's states and the analyses that use them."""

import math

import numpy as np
import pytest

from vib_tutorial.core.magnetic_bearing import MagneticBearing
from vib_tutorial.core.rotor import (
    RPM,
    XA,
    XB,
    XD,
    YA,
    YB,
    YD,
    Bearing,
    RotorSimulator,
    RotorSystem,
    _rotating_force,
    gravity_response,
    synchronous_response,
    whirl_modes,
)
from vib_tutorial.core.rotor_stability import least_damped, stability_onset

MAGNET = MagneticBearing()
ON_MAGNETS = RotorSystem(bearing_a=Bearing(magnet=MAGNET), bearing_b=Bearing(magnet=MAGNET), gravity=9.81)


def on(magnet: MagneticBearing, **changes) -> RotorSystem:
    return ON_MAGNETS.with_(bearing_a=Bearing(magnet=magnet), bearing_b=Bearing(magnet=magnet), **changes)


def run_for(sim, duration):
    out = []
    end = sim.t + duration
    while sim.t < end - 1e-12:
        r = sim.advance(end - sim.t)
        if not r.t.size:
            break
        out.append(r)
    return out


def test_actuator_gains_linearize_the_magnet_pair():
    # f = k [(i0 + i)²/(s0 - x)² - (i0 - i)²/(s0 + x)²]: its slopes at the centre are k_m and k_i.
    m = MAGNET
    k, s0, i0 = m.force_constant, m.gap, m.bias

    def force(x, i):
        return k * ((i0 + i) ** 2 / (s0 - x) ** 2 - (i0 - i) ** 2 / (s0 + x) ** 2)

    dx, di = 1e-9, 1e-6
    assert (force(dx, 0) - force(-dx, 0)) / (2 * dx) == pytest.approx(m.negative_stiffness, rel=1e-6)
    assert (force(0, di) - force(0, -di)) / (2 * di) == pytest.approx(m.current_gain, rel=1e-6)
    assert force(0, i0) == pytest.approx(m.capacity, rel=1e-12)  # one magnet's current run out


@pytest.mark.parametrize("magnet", [MAGNET, MagneticBearing(ki=0.0), MagneticBearing(kd=0.0),
                                    MagneticBearing(ki=0.0, kd=0.0)])
def test_state_space_is_the_transfer_function(magnet):
    a, b, c = magnet.state_space()
    assert b.size == magnet.n_states
    for s in (10j, 3000j, 40.0 + 700j):
        assert c @ np.linalg.solve(s * np.eye(b.size) - a, b) == pytest.approx(magnet.gain(s), rel=1e-10)


def test_damped_band_edges_are_where_the_equivalent_damping_changes_sign():
    lo, hi = MAGNET.damped_band()
    assert 0.0 < lo < hi < math.inf
    c_lo, c_hi = MAGNET.equivalent(np.array([0.99, 1.01]) * lo)[1], MAGNET.equivalent(np.array([0.99, 1.01]) * hi)[1]
    assert c_lo[0] < 0.0 < c_lo[1] and c_hi[0] > 0.0 > c_hi[1]
    # Without the integral nothing lags at low frequency.
    assert MagneticBearing(ki=0.0).damped_band()[0] == 0.0


def test_integral_action_centres_the_journals_under_gravity():
    s = ON_MAGNETS
    q = s.static_deflection(0.0)
    assert abs(q[YA]) < 1e-15 and abs(q[YB]) < 1e-15
    assert q[YD] < -1e-5  # the shaft still bends under the disc
    np.testing.assert_allclose(s.static_state(0.0)[:s.n_dof], q, atol=1e-18)
    # The integrators carry the bearings' shares of the weight, as current: k_i i = W.
    sim = RotorSimulator(s)
    run_for(sim, 0.3)
    np.testing.assert_allclose(sim.q, q, atol=1e-12)


def test_proportional_magnet_sags_like_its_spring():
    # Without the integral, statically the bearing is a spring of k_i k_p - k_m.
    m = MagneticBearing(ki=0.0)
    k = m.current_gain * m.kp - m.negative_stiffness
    springs = ON_MAGNETS.with_(bearing_a=Bearing(kx=k, ky=k), bearing_b=Bearing(kx=k, ky=k))
    np.testing.assert_allclose(on(m).static_deflection(0.0), springs.static_deflection(0.0), rtol=1e-9, atol=1e-18)


def test_fast_controller_acts_as_its_spring_and_damper():
    # With the filter and the amplifier far faster than the rotor, a PD bearing is k_i k_p - k_m and k_i k_d.
    m = MagneticBearing(ki=0.0, filter_hz=1e6, amplifier_hz=1e6)
    k, c = m.current_gain * m.kp - m.negative_stiffness, m.current_gain * m.kd
    springs = ON_MAGNETS.with_(bearing_a=Bearing(kx=k, ky=k, cx=c, cy=c), bearing_b=Bearing(kx=k, ky=k, cx=c, cy=c))
    w = 3000.0 / RPM
    a, b = whirl_modes(on(m), w), whirl_modes(springs, w)
    np.testing.assert_allclose(a.freq_hz[:6], b.freq_hz[:6], rtol=2e-3)
    np.testing.assert_allclose(a.zeta[:6], b.zeta[:6], rtol=1e-2, atol=1e-4)


@pytest.mark.parametrize("periodic", [False, True])
def test_synchronous_response_is_the_state_spaces(periodic):
    s = ON_MAGNETS.with_(position=0.35, unbalance=50e-6, asymmetry=1e-9 if periodic else 0.0)
    w = 2500.0 / RPM
    f = _rotating_force(s, np.array([w]))[0]
    z = np.linalg.solve(1j * w * np.eye(s.n_states) - s.state_matrix(w), s.input_matrix() @ f)
    np.testing.assert_allclose(synchronous_response(s, [w])[0], z[:s.n_dof], rtol=1e-6, atol=1e-12)


def test_asymmetric_shaft_on_magnets_tends_to_the_round_one():
    s = ON_MAGNETS.with_(position=0.35)
    w = 4000.0 / RPM
    sag, two = gravity_response(s.with_(asymmetry=1e-9), np.array([w]))
    np.testing.assert_allclose(sag[0].real, s.static_deflection(w), rtol=1e-6, atol=1e-15)
    assert np.abs(two).max() < 1e-12
    a, b = whirl_modes(s.with_(asymmetry=1e-9), w), whirl_modes(s, w)
    np.testing.assert_allclose(a.freq_hz[:6], b.freq_hz[:6], rtol=1e-6)


def test_simulation_holds_the_steady_state():
    # Started on the whole 1X steady state (the controllers' states too), at constant speed, it stays there.
    s = ON_MAGNETS.with_(position=0.35, unbalance=50e-6)
    w = 2500.0 / RPM
    f = _rotating_force(s, np.array([w]))[0]
    Z = np.linalg.solve(1j * w * np.eye(s.n_states) - s.state_matrix(w), s.input_matrix() @ f)
    zs = s.static_state(w)
    sim = RotorSimulator(s)
    sim.omega = sim.target = w
    sim.state = zs + Z.real
    res = run_for(sim, 0.2)
    t = np.concatenate([r.t for r in res])
    q = np.concatenate([r.q for r in res])
    expect = zs[:s.n_dof] + (Z[None, :s.n_dof] * np.exp(1j * w * t)[:, None]).real
    assert np.abs(q - expect).max() < 1e-4 * np.abs(Z[:s.n_dof]).max()


def test_amplifier_lag_drives_the_forward_mode_where_the_damping_turns_negative():
    # The disc's forward tilting mode rises with the speed; it goes unstable as it passes the band's
    # top edge, where the filter's and the amplifier's lag make the bearings' damping negative.
    m = MagneticBearing(amplifier_hz=300.0)
    onset = stability_onset(on(m), 30_000.0 / RPM)
    assert onset is not None and onset.whirl > 0.9
    assert onset.freq_hz == pytest.approx(m.damped_band()[1], rel=1e-3)
    assert stability_onset(on(MagneticBearing(amplifier_hz=2000.0)), 30_000.0 / RPM) is None


@pytest.mark.parametrize("ki", [0.0, MAGNET.ki])
def test_too_weak_a_magnet_drops_its_journal(ki):
    # k_i k_p < k_m: the magnets outpull the controller. Without the integral the journal falls away
    # without oscillating; with it, the integral pulls it back too late, and it swings ever wider.
    m = MagneticBearing(kp=0.9 * MAGNET.negative_stiffness / MAGNET.current_gain, ki=ki)
    s = on(m)
    zeta, freq, _ = least_damped(s, 0.0)
    if ki:
        assert zeta < 0.0 < freq
    else:
        assert whirl_modes(s, 0.0).drift > 0.0 and zeta == -1.0
    assert stability_onset(s, 5000.0 / RPM).omega == 0.0
    assert whirl_modes(ON_MAGNETS, 0.0).drift < 0.0


def test_changing_the_controller_keeps_the_motion_and_holds_the_weight():
    sim = RotorSimulator(ON_MAGNETS.with_(bearing_a=Bearing(), bearing_b=Bearing()))
    run_for(sim, 0.05)
    q = sim.q.copy()
    sim.set_system(ON_MAGNETS)
    assert sim.state.size == ON_MAGNETS.n_states
    np.testing.assert_array_equal(sim.q, q)
    np.testing.assert_allclose(sim.control, ON_MAGNETS.static_state(0.0)[2 * ON_MAGNETS.n_dof:])
    sim.set_system(on(MagneticBearing(ki=0.0)))  # one state fewer on each axis
    assert sim.state.size == on(MagneticBearing(ki=0.0)).n_states
    np.testing.assert_array_equal(sim.q, q)


def test_magnet_and_film_cannot_share_a_bearing():
    from vib_tutorial.core.fluid_bearing import JournalBearing
    with pytest.raises(ValueError):
        Bearing(film=JournalBearing(), magnet=MAGNET)
    assert not ON_MAGNETS.isotropic_supports
    assert ON_MAGNETS.n_states == 2 * ON_MAGNETS.n_dof + 4 * MAGNET.n_states
    assert XA in (d for d, _ in ON_MAGNETS._axes) and XB in (d for d, _ in ON_MAGNETS._axes)
    assert XD not in (d for d, _ in ON_MAGNETS._axes)
