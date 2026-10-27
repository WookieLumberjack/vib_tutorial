"""The One mass pages' models (core/sdof.py, transient.py, damping_models.py, isolation.py) against
solve_ivp and closed forms."""

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from vib_tutorial.core.damping_models import (
    MODELS,
    Matched,
    Model,
    coulomb_decay,
    coulomb_stop,
)
from vib_tutorial.core.isolation import Isolator, Mount, deflection_for, start_up
from vib_tutorial.core.sdof import (
    Oscillator,
    free_response,
    frequency_grid,
    harmonic_from_rest,
    measured_decrement,
    receptance,
    response,
)
from vib_tutorial.core.transient import (
    Pulse,
    PulseKind,
    convolution,
    impulse_spectrum,
    peak_ratio,
    peak_ratio_undamped,
    pulse_response,
    step_metrics,
)


def integrate(m, c, k, force, x0, v0, t):
    sol = solve_ivp(lambda tt, z: [z[1], (force(tt) - c * z[1] - k * z[0]) / m], (t[0], t[-1]), [x0, v0],
                    t_eval=t, rtol=1e-11, atol=1e-14, max_step=1e-3)
    return sol.y[0]


@pytest.mark.parametrize("c", [0.0, 2.0, 40.0, 80.0])  # undamped, light, critical, overdamped
def test_free_response_matches_the_ode(c):
    osc = Oscillator(1.0, 400.0, c)
    t = np.linspace(0.0, 2.0, 2001)
    x, _ = free_response(osc, 0.01, 0.05, t)
    ref = integrate(1.0, c, 400.0, lambda tt: 0.0, 0.01, 0.05, t)
    assert np.max(np.abs(x - ref)) < 1e-8


def test_oscillator_numbers():
    osc = Oscillator()  # 1 kg, 400 N/m, 2 N·s/m
    assert osc.natural_hz == pytest.approx(20 / (2 * math.pi))
    assert osc.zeta == pytest.approx(0.05)
    lo, hi = osc.half_power_hz()
    peak = np.abs(receptance(osc, np.array([osc.peak_hz()])))[0]
    assert np.abs(receptance(osc, np.array([lo, hi]))) == pytest.approx(peak / math.sqrt(2), rel=1e-9)
    t = np.linspace(0, 5, 50001)
    x, _ = free_response(osc, 0.01, 0.0, t)
    assert measured_decrement(t, x) == pytest.approx(osc.log_decrement, rel=1e-4)
    assert sorted(osc.poles(), key=lambda p: p.imag) == pytest.approx(
        sorted(np.roots([1.0, 2.0, 400.0]), key=lambda p: p.imag))


def test_harmonic_from_rest_matches_the_ode_also_at_undamped_resonance():
    for c, f in ((2.0, 2.7), (0.0, 20 / (2 * math.pi))):
        osc = Oscillator(1.0, 400.0, c)
        t = np.linspace(0.0, 3.0, 3001)
        x, _ = harmonic_from_rest(osc, 1.5, f, t)
        ref = integrate(1.0, c, 400.0, lambda tt, f=f: 1.5 * math.cos(2 * math.pi * f * tt), 0.0, 0.0, t)
        assert np.max(np.abs(x - ref)) < 1e-8 * max(1.0, np.max(np.abs(ref)) * 1e3)


def test_undamped_frequency_grid_skips_the_infinite_peak():
    osc = Oscillator(1.0, 400.0, 0.0)
    f = frequency_grid(osc)
    assert np.all(np.isfinite(receptance(osc, f)))


def test_foh_response_is_exact_for_a_ramped_force():
    osc = Oscillator()
    h = 0.01
    t = np.arange(301) * h
    u = 3.0 * t  # linear: the FOH holds it exactly
    x = response(osc, h, u)
    ref = integrate(1.0, 2.0, 400.0, lambda tt: 3.0 * tt, 0.0, 0.0, t)
    assert np.max(np.abs(x - ref)) < 1e-9


@pytest.mark.parametrize("kind", list(PulseKind))
def test_pulse_responses_and_convolution(kind):
    osc = Oscillator()
    pulse = Pulse(kind, 2.0, 0.13)
    t = np.linspace(0.0, 3.0, 3001)
    x = pulse_response(osc, pulse, t)
    if kind is PulseKind.IMPULSE:
        ref = integrate(1.0, 2.0, 400.0, lambda tt: 0.0, 0.0, 2.0, t)
    else:
        ref = integrate(1.0, 2.0, 400.0, lambda tt: pulse.force(np.array([tt]))[0], 0.0, 0.0, t)
    assert np.max(np.abs(x - ref)) < 1e-4 * np.max(np.abs(ref))
    if kind is not PulseKind.IMPULSE:
        *_, at = convolution(osc, pulse, 1.0)
        assert at == pytest.approx(float(np.interp(1.0, t, x)), rel=1e-3, abs=1e-6 * np.max(np.abs(ref)))


def test_impulse_response_and_receptance_are_a_fourier_pair():
    osc = Oscillator()
    f, H = impulse_spectrum(osc, 12.0, 2 ** 14)
    band = (f > 1) & (f < 6)
    exact = receptance(osc, f[band])
    assert np.max(np.abs(H[band] - exact)) < 1e-4 * np.max(np.abs(exact))


def test_pulse_shock_spectrum_against_closed_forms():
    for r in (0.05, 0.25, 0.7):
        assert peak_ratio(0.0, PulseKind.RECTANGLE, np.array([r]))[0] == pytest.approx(
            peak_ratio_undamped(PulseKind.RECTANGLE, r), rel=2e-3)
    for r in (0.3, 1.5, 5.5):
        assert peak_ratio(0.0, PulseKind.RAMP, np.array([r]))[0] == pytest.approx(
            peak_ratio_undamped(PulseKind.RAMP, r), rel=2e-3)
    r = np.linspace(0.6, 1.0, 81)
    assert np.max(peak_ratio(0.0, PulseKind.HALF_SINE, r)) == pytest.approx(1.768, abs=0.003)


def test_step_metrics():
    m = step_metrics(Oscillator())
    assert m.overshoot == pytest.approx(math.exp(-0.05 * math.pi / math.sqrt(1 - 0.05 ** 2)))
    assert m.settling_estimate == pytest.approx(4.0)
    assert 3.0 < m.settling < 4.2


def test_the_damping_models_lose_the_same_energy_at_fn():
    m = Matched()
    wn = m.omega_n
    expect = math.pi * m.c * wn * m.amplitude ** 2
    for model in MODELS:
        x, force, _ = m.loop(model, wn, 20001)
        assert abs(np.trapezoid(force, x)) == pytest.approx(expect, rel=1e-3), model
        assert m.energy_per_cycle(model, np.array([wn]))[0] == pytest.approx(expect, rel=1e-9), model
    assert m.dynamic_stiffness(Model.ZENER, np.array([wn]))[0].real == pytest.approx(m.stiffness)
    assert m.loss_factor(Model.ZENER, np.array([wn]))[0] == pytest.approx(m.eta)


def test_coulomb_decay_against_the_ode():
    m = Matched()
    t = np.linspace(0.0, 1.2, 12001)
    x = coulomb_decay(m.mass, m.stiffness, m.friction, 0.01, t)

    def rhs(tt, z):  # sliding only: the closed form says where it sticks
        return [z[1], (-m.stiffness * z[0] - m.friction * np.sign(z[1])) / m.mass]

    half = math.pi / m.omega_n
    ref = solve_ivp(rhs, (0, half * 0.999), [0.01, -1e-12], t_eval=t[t < half * 0.999], rtol=1e-11, atol=1e-14)
    assert np.max(np.abs(x[: ref.y.shape[1]] - ref.y[0])) < 1e-7
    _, pos = coulomb_stop(m.stiffness, m.friction, 0.01)
    assert abs(pos) <= m.friction / m.stiffness
    assert x[-1] == pytest.approx(pos)


def test_zener_free_decay_matches_the_ode():
    m = Matched()
    k0, k1, tau = m.zener()
    c1 = k1 * tau
    t = np.linspace(0.0, 2.0, 4001)
    x = m.free_decay(Model.ZENER, 0.01, t)

    def rhs(tt, z):
        x, v, y = z
        return [v, (-k0 * x - k1 * (x - y)) / m.mass, k1 * (x - y) / c1]

    ref = solve_ivp(rhs, (0, 2), [0.01, 0.0, 0.01], t_eval=t, rtol=1e-11, atol=1e-14).y[0]
    assert np.max(np.abs(x - ref)) < 1e-8


@pytest.mark.parametrize("mount", list(Mount))
def test_start_up_settles_to_the_transmissibility(mount):
    iso = Isolator(mount=mount, zeta=0.2)
    _, floor, _ = iso.at_speed()
    run = start_up(iso, 2.0, 6.0)
    tail = run.t > 6.0
    assert np.max(np.abs(run.floor[tail])) == pytest.approx(floor, rel=0.01)


def test_isolation_numbers():
    iso = Isolator()
    assert iso.natural_hz == pytest.approx(15.76 / math.sqrt(5), rel=1e-3)
    assert Isolator(zeta=0.0).transmissibility(np.array([math.sqrt(2) * iso.natural_hz]))[0] == pytest.approx(1.0)
    T = 1 - 0.9
    d = deflection_for(25.0, 0.9)
    assert Isolator(deflection=d, zeta=0.0).at_speed()[2] == pytest.approx(T)
    viscous = Isolator(mount=Mount.RELAXATION, zeta=0.0)
    assert viscous.at_speed()[2] == pytest.approx(Isolator(zeta=0.0).at_speed()[2])
