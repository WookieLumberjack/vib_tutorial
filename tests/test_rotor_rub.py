"""Rub on a stator ring round the disc: the contact force, the time step, and the presets' notes."""

import math
from dataclasses import replace

import numpy as np
import pytest
import scipy.optimize

from vib_tutorial.core.fluid_bearing import JournalBearing
from vib_tutorial.core.rotor import (RPM, XD, YD, Bearing, RotorSimulator, RotorSystem, Stator, _rotating_force,
                                     stator_force)
from vib_tutorial.core.rotor_presets import rotor_presets

PRESETS = {p.name: p for p in rotor_presets()}


def run(system: RotorSystem, rpm: float, duration: float, accel: float | None = None, steady: bool = False):
    """(t, x_D, y_D, Ω, rub) from rest (or at speed, `steady`) over `duration` s."""
    sim = RotorSimulator(system)
    sim.target = rpm / RPM
    if accel is not None:
        sim.accel = accel / RPM
    if steady:
        sim.omega = sim.target
    out = []
    while True:
        r = sim.advance(min(0.25, duration - sim.t))
        if not r.t.size:  # less than half a step left
            break
        out.append((r.t, r.q[:, XD], r.q[:, YD], r.omega, r.rub))
    return tuple(np.concatenate(a) for a in zip(*out))


def full_spectrum(t: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = x + 1j * y
    z = (z - z.mean()) * np.hanning(z.size)
    return np.fft.fftfreq(z.size, t[1] - t[0]), np.abs(np.fft.fft(z)) * 4 / z.size


def test_contact_force_derivatives_match_finite_differences():
    stator = Stator(clearance=2e-4, stiffness=2e6, damping=100.0, friction=0.3, offset=5e-5, offset_angle=1.0)
    rng = np.random.default_rng(1)
    checked = 0
    for _ in range(100):
        angle, rho = rng.uniform(0, 2 * math.pi), stator.clearance + rng.uniform(1e-6, 3e-5)
        y = np.array([*(np.array(stator.centre) + rho * np.array([math.cos(angle), math.sin(angle)])),
                      *rng.normal(0.0, 0.05, 2)])
        omega = rng.uniform(-50.0, 50.0)
        fx, fy, normal, J = stator_force(stator, 0.05, *y, omega)
        if normal == 0.0:
            continue
        assert math.hypot(fx, fy) >= normal  # the push and the friction
        numeric = np.empty((2, 4))
        for i in range(4):
            h = np.zeros(4)
            h[i] = 1e-9 if i < 2 else 1e-7
            numeric[:, i] = (np.array(stator_force(stator, 0.05, *(y + h), omega)[:2])
                             - np.array(stator_force(stator, 0.05, *(y - h), omega)[:2])) / (2 * h[i])
        assert np.reshape(J, (2, 4)) == pytest.approx(numeric, rel=1e-6, abs=1e-6 * np.abs(numeric).max())
        checked += 1
    assert checked > 50


def test_no_force_inside_the_clearance_or_pulling_away():
    stator = Stator(clearance=1e-4, offset=2e-5, offset_angle=0.0)
    assert stator_force(stator, 0.05, 1.1e-4, 0.0, 0.0, 0.0, 100.0)[2] == 0.0  # 90 µm from the centre
    assert stator_force(stator, 0.05, 1.3e-4, 0.0, 0.0, 0.0, 100.0)[2] == pytest.approx(stator.stiffness * 1e-5)
    # Leaving faster than the ring springs back (c δ' < -k δ): the ring cannot pull.
    assert stator_force(stator, 0.05, 1.21e-4, 0.0, -10.0, 0.0, 100.0)[2] == 0.0


def test_a_ring_never_reached_changes_nothing():
    plain = RotorSimulator(RotorSystem())
    ringed = RotorSimulator(RotorSystem(stator=Stator(clearance=1e-3)))
    for sim in (plain, ringed):
        sim.target = 2000.0 / RPM
        for _ in range(2):
            r = sim.advance(0.5)
    assert ringed.step_size() <= plain.step_size()
    assert np.all(r.rub[:, 2] == 0.0)
    # The same motion, on a step that also resolves the disc bouncing on the ring.
    assert ringed.q[[XD, YD]] == pytest.approx(plain.q[[XD, YD]], rel=1e-3, abs=1e-9)


@pytest.mark.parametrize(("friction", "rpm"), [(0.0, 1500.0), (0.0, 2500.0), (0.1, 2000.0)])
def test_full_rub_settles_on_the_contact_steady_state(friction, rpm):
    """A forward synchronous full rub is a circle round the ring: the ring is then a spring
    κ = k (1 - δ/ρ) at the disc, its friction cross-coupled, μ κ, so the orbit's radius ρ solves
    a linear steady state at that κ."""
    system = RotorSystem(stator=Stator(clearance=30e-6, friction=friction))
    w = rpm / RPM
    M, _, G, _ = system.matrices()
    f = _rotating_force(system, np.array([w]))[0]

    def radius(kappa: float) -> float:
        Z = system.stiffness(w) - w * w * M + 1j * w * (system.damping(w) + w * G)
        Z[np.ix_([XD, YD], [XD, YD])] += kappa * np.array([[1.0, -friction], [friction, 1.0]])
        return abs(system.dynamic_solve(Z, f, 1j * w)[XD])

    st = system.stator
    rho = scipy.optimize.brentq(lambda r: radius(st.stiffness * (1 - st.clearance / r)) - r,
                                st.clearance * (1 + 1e-9), radius(0.0))
    t, x, y, _, rub = run(system, rpm, 3.0, steady=True)
    last = t > 2.5
    assert np.hypot(x[last], y[last]) == pytest.approx(rho, rel=1e-6)
    assert np.all(rub[last, 2] == pytest.approx(st.stiffness * (rho - st.clearance), rel=1e-4))


def test_finer_steps_agree():
    """The rub's first bounces (partial rub, under gravity) on the step chosen and on one four times finer."""
    system = PRESETS["Partial rub (½X)"].system
    ends = []
    for fine in (False, True):
        sim = RotorSimulator(system)
        sim.omega = sim.target = 4400.0 / RPM
        if fine:
            h = sim.step_size()
            sim.step_size = lambda h=h: h / 4
        r = sim.advance(0.1)
        assert r.rub[:, 2].max() > 0.0
        ends.append(sim.q[[XD, YD]].copy())
    assert ends[0] == pytest.approx(ends[1], abs=1e-3 * np.abs(ends[1]).max())


def test_a_stator_needs_a_round_shaft_and_sound_numbers():
    with pytest.raises(ValueError):
        RotorSystem(stator=Stator(clearance=0.0))
    with pytest.raises(ValueError):
        RotorSystem(stator=Stator(friction=-0.1))
    with pytest.raises(ValueError):
        RotorSystem(asymmetry=0.1, stator=Stator())
    film = Bearing(film=JournalBearing())
    with pytest.raises(ValueError):
        RotorSystem(bearing_a=film, bearing_b=film, nonlinear_films=True, stator=Stator())
    RotorSystem(bearing_a=film, bearing_b=film, stator=Stator())  # linearized films are fine
    assert RotorSystem().disc_radius == pytest.approx(0.05)


def test_partial_rub_preset_whirls_at_half_the_speed():
    p = PRESETS["Partial rub (½X)"]
    s = p.system
    assert -s.static_deflection(0.0)[YD] == pytest.approx(378e-6, abs=0.5e-6)
    t, x, y, w, rub = run(s, p.target_rpm, 13.0)
    assert w[-1] * RPM == pytest.approx(4400.0)
    last = t > 10.0
    assert 0.05 < np.mean(rub[last, 2] > 0) < 0.5  # a partial rub: in contact only now and then
    freqs, amps = full_spectrum(t[last], x[last], y[last])
    top = abs(freqs[np.argmax(amps)])
    assert top == pytest.approx(0.5 * 4400.0 / 60.0, abs=0.4)  # 36.7 Hz
    half = amps[np.abs(np.abs(freqs) - top) < 0.5].max()
    once = amps[np.abs(freqs - 4400.0 / 60.0) < 0.5].max()
    assert half > once


def test_full_rub_preset_flips_to_backward_whirl():
    p = PRESETS["Full rub (backward whirl)"]
    s = p.system
    t, x, y, w, rub = run(s, p.target_rpm, 6.0)
    r = np.hypot(x, y)
    swept = x[:-1] * np.diff(y) - y[:-1] * np.diff(x)
    backward = np.flatnonzero((swept < 0) & (r[:-1] > 0.1e-3))
    assert round(w[backward[0]] * RPM, -1) == 1550  # 1545 rpm
    last = t > 4.5
    freqs, amps = full_spectrum(t[last], x[last], y[last])
    assert round(freqs[np.argmax(amps)]) == -227
    # Rolling: the rim's slip R Ω + ω ρ is nil, so ρ = R Ω / |ω|.
    omega = 2000.0 / RPM
    assert r[-1] == pytest.approx(s.disc_radius * omega / (227 * 2 * math.pi), rel=0.01)
    assert round(r[-1] * 1e3, 1) == 7.3


def test_things_to_try():
    """The theory notes' Things to try for rub."""
    s = PRESETS["Full rub (backward whirl)"].system
    t, x, y, _, _ = run(s.with_(stator=replace(s.stator, friction=0.1)), 2000.0, 6.0)
    last = t > 5.0
    swept = x[:-1] * np.diff(y) - y[:-1] * np.diff(x)
    assert np.all(swept[last[:-1]] > 0)  # forward, round the ring
    assert round(np.hypot(x[-1], y[-1]) * 1e6) == 31
    # The partial rub's ½X goes below about 4000 rpm.
    sim = RotorSimulator(PRESETS["Partial rub (½X)"].system)
    sim.target = 4400.0 / RPM
    for target, until in ((4400.0, 12.0), (3000.0, 20.0)):
        sim.target = target / RPM
        out = []
        while sim.t < until - 0.5 * sim.step_size():
            r = sim.advance(min(0.25, until - sim.t))
            out.append((r.t, r.q[:, XD], r.q[:, YD]))
    t, x, y = (np.concatenate(a) for a in zip(*out))
    freqs, amps = full_spectrum(t[t > 17.0], x[t > 17.0], y[t > 17.0])
    assert freqs[np.argmax(amps)] == pytest.approx(50.0, abs=0.4)  # 1X
    assert amps[np.abs(np.abs(freqs) - 25.0) < 0.5].max() < 0.02 * amps.max()
