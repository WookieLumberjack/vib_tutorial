"""Motor start-ups of a torsional train: the stepping against an ODE solver, the referred train, the steady
running state, the synchronous motor's crossings, and the presets' numbers."""

import dataclasses
import math

import numpy as np
import pytest
import scipy.integrate

from vib_tutorial.core.torsional import COMPRESSOR, FAN, TWO_PI, Shaft, modes
from vib_tutorial.core.torsional_startup import (
    COMPRESSOR_START,
    FAN_START,
    STARTUPS,
    Motor,
    StartUp,
    StartUpSimulator,
    crossings,
    running_torques,
    steady_speed,
)


def run(train, startup, h=None, until=None):
    """The whole run's samples, concatenated, and the simulator."""
    sim = StartUpSimulator(train, startup, h)
    parts = []
    end = startup.duration if until is None else until
    while sim.t < end - 0.5 * sim.h:
        parts.append(sim.advance(end - sim.t))
    cat = {f: np.concatenate([getattr(p, f) for p in parts], axis=-1) for f in ("t", "rpm", "motor", "load", "links")}
    return sim, cat


def reference(train, startup, t_end):
    """The same start-up by an implicit ODE solver at tight tolerances: M θ'' + C θ' + K θ = T_m e_0 − T_L e_L."""
    m, c, k = train.matrices()
    n = train.n
    minv = 1.0 / np.diag(m)
    motor = startup.motor
    w_load = motor.sync_speed * train.speeds[-1]

    def f(t, z):
        th, w = z[:n], z[n:]
        torque = -(c @ w + k @ th)
        torque[0] += motor.torque(t, th[0], w[0], False)[0]
        r = w[-1] / w_load
        torque[-1] -= startup.load * r * abs(r)
        return np.concatenate([w, minv * torque])

    sol = scipy.integrate.solve_ivp(f, (0.0, t_end), np.zeros(2 * n), method="Radau", rtol=1e-11, atol=1e-12)
    return sol.y[:, -1]


@pytest.mark.parametrize("train, startup", [
    (FAN, FAN_START),
    (COMPRESSOR, dataclasses.replace(COMPRESSOR_START, motor=dataclasses.replace(COMPRESSOR_START.motor, field=0.0))),
])
def test_the_steps_match_an_ode_solver_and_converge_at_second_order(train, startup):
    t_end = 0.25
    ref = reference(train, startup, t_end)
    errors = []
    for h in (2.0**-12, 2.0**-13):
        sim, _ = run(train, startup, h, t_end)
        assert sim.t == pytest.approx(t_end)
        errors.append(np.abs(sim.z - ref).max() / np.abs(ref).max())
    assert errors[0] < 1e-4
    assert errors[0] / errors[1] == pytest.approx(4.0, rel=0.3)


def test_the_referred_train_starts_up_the_same():
    """Referred to the motor's shaft (the load times its speed ratio), each link's torque is s times the geared one's."""
    s = COMPRESSOR.speeds
    geared, a = run(COMPRESSOR, COMPRESSOR_START, until=1.0)
    _, b = run(COMPRESSOR.referred(), COMPRESSOR_START.with_(load=COMPRESSOR_START.load * s[-1]), until=1.0)
    assert b["rpm"] == pytest.approx(a["rpm"], rel=1e-7)  # only rounding apart
    scale = np.abs(a["links"]).max()
    assert np.abs(b["links"] - s[:-1, None] * a["links"]).max() < 1e-7 * scale * s.max()
    assert geared.field_on


def test_it_settles_on_the_steady_running_state():
    sim, out = run(FAN, FAN_START)
    w = steady_speed(FAN, FAN_START)
    assert out["rpm"][-1] == pytest.approx(60.0 * w / TWO_PI, rel=1e-6)
    m = FAN_START.motor
    # the motor's curve meets the drag: the load and the fan's damper, there
    slip = 1.0 - w / m.sync_speed
    assert m.cage(slip)[0] == pytest.approx(FAN_START.load * (w / m.sync_speed) ** 2 + 2.0 * w, rel=1e-9)
    assert out["links"][:, -1] == pytest.approx(running_torques(FAN, FAN_START), rel=1e-5)
    # through a gearbox: the far side's torque is the near side's over the ratio
    sim, out = run(COMPRESSOR, COMPRESSOR_START)
    assert sim.field_on and out["rpm"][-1] == pytest.approx(1500.0, rel=1e-6)
    run_t = running_torques(COMPRESSOR, COMPRESSOR_START)
    assert out["links"][:, -1] == pytest.approx(run_t, rel=1e-3)
    assert run_t[0] == pytest.approx(4.0 * run_t[2], rel=1e-12)  # the same power at a quarter of the speed
    # a motor too weak for the load can't run
    weak = FAN_START.with_(motor=dataclasses.replace(FAN_START.motor, breakdown=200.0))
    assert math.isnan(steady_speed(FAN, weak))


def test_the_synchronous_pulsation_crosses_the_modes_at_twice_the_slip_frequency():
    m = COMPRESSOR_START.motor
    f = [md.freq_hz for md in modes(COMPRESSOR)]
    cs = crossings(COMPRESSOR, m)
    assert [c.mode for c in cs] == [1, 0]  # the faster mode first, the 2688 Hz mesh mode out of reach
    for c in cs:
        assert m.pulsation_hz(1.0 - c.rpm / m.sync_rpm) == pytest.approx(f[c.mode])
    assert crossings(FAN, FAN_START.motor) == []
    # the motor's torque pulses at 2 s f: over a stretch at near-constant slip, count its cycles
    speed = 0.5 * m.sync_speed  # held at half speed: 50 Hz
    ts = np.linspace(0.0, 0.2, 4001)
    torque = [m.torque(x, speed * x, speed, False)[0] for x in ts]
    spectrum = np.abs(np.fft.rfft(torque - np.mean(torque)))
    assert np.fft.rfftfreq(ts.size, ts[1] - ts[0])[np.argmax(spectrum)] == pytest.approx(50.0, abs=5.0)


def test_the_motor_torque_derivatives():
    for motor in (FAN_START.motor, COMPRESSOR_START.motor):
        for field in (False, True):
            t, a, w = 0.123, 4.5, 60.0
            _, da, dw = motor.torque(t, a, w, field)
            e = 1e-6
            assert da == pytest.approx((motor.torque(t, a + e, w, field)[0] - motor.torque(t, a - e, w, field)[0])
                                       / (2 * e), rel=1e-5, abs=1e-6)
            assert dw == pytest.approx((motor.torque(t, a, w + e, field)[0] - motor.torque(t, a, w - e, field)[0])
                                       / (2 * e), rel=1e-5, abs=1e-6)
    with pytest.raises(ValueError):
        Motor(breakdown=0.0)
    with pytest.raises(ValueError):
        StartUp(FAN_START.motor, -1.0)


def test_preset_notes():
    note = STARTUPS["Motor and fan: two inertias"].note
    sim, out = run(FAN, FAN_START)
    run_t = running_torques(FAN, FAN_START)
    assert round(sim.peaks[0] / 1e3, 1) == 6.0 and sim.peak_t[0] < 0.07 and round(sim.peaks[0] / run_t[0]) == 10
    assert "6.0 kN·m" in note and "10×" in note and "52.3 Hz" in note
    soft = FAN.with_link(0, Shaft(1.2e5, 20.0, "Coupling"))
    sim_soft, _ = run(soft, FAN_START)
    assert round(modes(soft)[0].freq_hz, 1) == 42.7 and round(sim_soft.peaks[0] / run_t[0], 1) == 7.0

    note = STARTUPS["Geared compressor"].note
    cs = crossings(COMPRESSOR, COMPRESSOR_START.motor)
    assert [round(c.rpm) for c in cs] == [254, 916] and [round(c.freq_hz, 1) for c in cs] == [83.1, 38.9]
    sim, out = run(COMPRESSOR, COMPRESSOR_START)
    run_t = running_torques(COMPRESSOR, COMPRESSOR_START)
    i = int(np.argmax(sim.peaks))
    assert COMPRESSOR.links[i].name == "Low-speed coupling"
    assert round(sim.peaks[i] / 1e3, 1) == 11.3 and round(sim.peaks[i] / run_t[i], 1) == 4.5
    passed = out["t"][np.argmax(out["rpm"] >= cs[1].rpm)]
    assert passed < sim.peak_t[i] < passed + 0.1  # just after the second crossing
    assert "11.3 kN·m" in note and "4.5×" in note and "254 rpm" in note and "916 rpm" in note
