"""The rotor's time response with the oil films' whole (nonlinear) force: RotorSystem.nonlinear_films."""

import numpy as np
import pytest

from vib_tutorial.core import rotor
from vib_tutorial.core.fluid_bearing import JournalBearing
from vib_tutorial.core.rotor import (
    RPM,
    XA,
    XB,
    XD,
    YA,
    YB,
    YD,
    Bearing,
    Crack,
    RotorSimulator,
    RotorSystem,
    synchronous_response,
)
from vib_tutorial.core.rotor_presets import rotor_presets

FILM = Bearing(film=JournalBearing())
WHOLE = RotorSystem(bearing_a=FILM, bearing_b=FILM, gravity=9.81, nonlinear_films=True)
PRESETS = {p.name: p for p in rotor_presets()}
WHIP = PRESETS["Journal bearings (oil whip)"]
WHIRL = PRESETS["Oil whirl (vertical rotor)"]
CLEARANCE = FILM.film.clearance


def run(system, rpm, duration, kick=0.0):
    """Start at speed on the static deflection, the disc kicked at `kick` m/s in x; (t, q) after `duration`."""
    sim = RotorSimulator(system)
    sim.omega = sim.target = rpm / RPM
    n = system.n_dof
    sim.state = np.concatenate([system.static_deflection(sim.omega), np.zeros(n)])
    sim.state[n + XD] += kick
    out = []
    while sim.t < duration - 1e-12:
        r = sim.advance(duration - sim.t)
        if not r.t.size:
            break
        out.append(r)
    return np.concatenate([r.t for r in out]), np.concatenate([r.q for r in out])


def eccentricity(q, x=XA, y=YA):
    return np.hypot(q[:, x], q[:, y]) / CLEARANCE


def whirl_freq(t, q):
    """The disc's strongest frequency (Hz), from a zero-padded spectrum."""
    x = q[:, XD] - q[:, XD].mean()
    n = 16 * x.size
    f = np.fft.rfftfreq(n, t[1] - t[0])
    return f[np.abs(np.fft.rfft(x * np.hanning(x.size), n)).argmax()]


def test_whole_films_only_on_a_round_shaft():
    assert WHOLE.whole_films and not WHOLE.with_(nonlinear_films=False).whole_films
    assert not RotorSystem(nonlinear_films=True).whole_films  # no films
    # A turning shaft's split step keeps the films linearized.
    assert not WHOLE.with_(asymmetry=0.1).whole_films
    assert not WHOLE.with_(crack=Crack(depth=0.3)).whole_films


def test_at_rest_the_journals_stay_where_the_films_hold_them():
    # Below the slowest speed the film is held as it is at ε = 0.9, the whole force as the linear one;
    # faster, the whole force holds the journals on the locus too.
    _, q = run(WHOLE, 0.0, 0.1)
    np.testing.assert_allclose(q[-1], WHOLE.static_deflection(0.0), atol=1e-12)
    still = WHOLE.with_(unbalance=0.0)
    _, q = run(still, 3000.0, 0.1)
    np.testing.assert_allclose(q[-1], still.static_deflection(3000.0 / RPM), atol=1e-12)


def test_small_motions_are_the_linear_films():
    # Started on the linear films' 1X steady state for a small unbalance, the whole films hold it: they
    # differ from their linearization only by the motion's size squared.
    s = WHOLE.with_(unbalance=2e-6)
    w = 2500.0 / RPM
    Q = synchronous_response(s, [w])[0]
    qs = s.static_deflection(w)
    sim = RotorSimulator(s)
    sim.omega = sim.target = w
    sim.state = np.concatenate([qs + Q.real, (1j * w * Q).real])
    r = sim.advance(0.1)
    expect = qs + (Q[None, :] * np.exp(1j * w * r.t)[:, None]).real
    assert np.abs(r.q - expect).max() < 3e-3 * np.abs(Q).max()


def test_whole_film_step_converges(monkeypatch):
    # Halving the step changes the whirl's growth toward the wall by under 1% of the clearance.
    s = WHOLE.with_(gravity=0.0)
    t1, q1 = run(s, 3500.0, 1.0, kick=0.002)
    monkeypatch.setattr(rotor, "STEPS_PER_PERIOD", 2 * rotor.STEPS_PER_PERIOD)
    monkeypatch.setattr(rotor, "FILM_MODE_STEPS", 2 * rotor.FILM_MODE_STEPS)
    t2, q2 = run(s, 3500.0, 1.0, kick=0.002)
    assert t2[1] - t2[0] == pytest.approx(0.5 * (t1[1] - t1[0]))
    np.testing.assert_allclose(q2[1::2, [XA, YA, XB, YB]], q1[:, [XA, YA, XB, YB]], atol=1e-2 * CLEARANCE)
    assert eccentricity(q1).max() > 0.8


def test_oil_whirl_settles_to_a_limit_cycle():
    # The preset's note. Upright, the unloaded films whirl at any speed. Below twice the first natural
    # frequency it is oil whirl: at half the speed, it settles with the journals circling near the wall.
    s, rpm = WHIRL.system, WHIRL.target_rpm
    assert s == WHOLE.with_(gravity=0.0) and rpm == 3500.0
    t, q = run(s, rpm, 3.0, kick=0.002)
    late, later = (t > 2.0) & (t < 2.5), t > 2.5
    assert whirl_freq(t[t > 1.5], q[t > 1.5]) == pytest.approx(29.2, abs=0.05)
    assert round(0.5 * rpm / 60.0, 1) == 29.2
    e1, e2 = eccentricity(q[late]).max(), eccentricity(q[later]).max()
    assert 0.9 < e1 < e2 < 0.97
    disc = [np.hypot(q[m, XD], q[m, YD]).max() for m in (late, later)]
    assert disc[1] == pytest.approx(disc[0], rel=0.03) and disc[1] == pytest.approx(0.16e-3, abs=0.005e-3)


def test_oil_whip_grows_with_the_journals_inside_their_clearance():
    # The presets' notes. Past twice the first natural frequency the whirl locks at it, oil whip. The
    # films keep the journals in their clearance, but the shaft bends ever more: the linear model's
    # exponential growth slows, but does not stop.
    s = WHIP.system
    assert s == WHOLE
    t, q = run(s, 5000.0, 1.5, kick=0.02)
    qs = s.static_deflection(5000.0 / RPM)
    assert whirl_freq(t[t > 0.5], q[t > 0.5]) == pytest.approx(30.7, abs=0.1)  # as the linear films
    assert eccentricity(q).max() < 0.8
    disc = [np.hypot(q[m, XD] - qs[XD], q[m, YD] - qs[YD]).max() for m in (t < 0.5, t > 1.0)]
    assert disc[1] > 1.5 * disc[0]


@pytest.mark.parametrize("change", [{"position": 1.2}, {"density": 7850.0}, {"bearing_b": Bearing()}],
                         ids=["overhung", "shaft mass", "one film"])
def test_whole_films_on_other_rotors(change):
    # An overhung rotor's far film is pulled up, a shaft with mass couples the journals through M, and one
    # film may run beside springs. After a small tap each moves as on its linear films.
    s = WHOLE.with_(**change)
    t, q = run(s, 1500.0, 0.2, kick=1e-3)
    t0, q0 = run(s.with_(nonlinear_films=False), 1500.0, 0.2, kick=1e-3)
    step = round((t[1] - t[0]) / (t0[1] - t0[0]))
    qs = s.static_deflection(1500.0 / RPM)
    moved = np.abs(q0[:, :8] - qs[:8]).max()
    np.testing.assert_allclose(q[:, :8], q0[step - 1::step, :8], atol=0.02 * moved)


def test_vertical_rotor_whips_past_twice_its_natural_frequency():
    # The Oil whirl preset's note: at 5000 rpm the whirl is locked at 30.8 Hz, and twice that is 3700 rpm.
    t, q = run(WHIRL.system, 5000.0, 1.5, kick=0.002)
    f = whirl_freq(t[t > 0.5], q[t > 0.5])
    assert f == pytest.approx(30.8, abs=0.1) and round(2 * f * 60, -2) == 3700


def test_whole_films_hold_the_journals_through_the_critical():
    # The Theory: run up through the critical at 500 rpm/s from the 1X steady state at 1200 rpm, the
    # linearized films carry the journals within 1% of the wall, the whole ones to ε = 0.63. The disc's
    # peak is much the same.
    for whole, eps in ((False, 0.99), (True, 0.63)):
        s = WHIP.system.with_(nonlinear_films=whole)
        w0 = 1200.0 / RPM
        Q = synchronous_response(s, [w0])[0]
        sim = RotorSimulator(s)
        sim.omega, sim.target, sim.accel = w0, 2500.0 / RPM, 500.0 / RPM
        sim.state = np.concatenate([s.static_deflection(w0) + Q.real, (1j * w0 * Q).real])
        out = []
        while sim.t < 3.0 - 1e-12:
            r = sim.advance(3.0 - sim.t)
            if not r.t.size:
                break
            out.append(r)
        q, w = np.concatenate([r.q for r in out]), np.concatenate([r.omega for r in out])
        vib = q - s.static_deflection(w)
        assert round(eccentricity(q).max(), 2) == eps
        assert np.hypot(vib[:, XD], vib[:, YD]).max() == pytest.approx(0.66e-3, abs=0.01e-3)
