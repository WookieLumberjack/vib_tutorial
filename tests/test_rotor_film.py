"""The rotor on oil-film journal bearings, and the generalized Bearing (full 2 x 2 stiffness and damping)."""

import math

import numpy as np
import pytest

from vib_tutorial.core import rotor
from vib_tutorial.core.fluid_bearing import JournalBearing
from vib_tutorial.core.rotor import (
    RPM,
    XA,
    XB,
    YA,
    YB,
    YD,
    Bearing,
    RotorSimulator,
    RotorSystem,
    campbell,
    synchronous_response,
)
from vib_tutorial.core.rotor_presets import rotor_presets
from vib_tutorial.core.rotor_stability import least_damped, stability_onset

FILM = Bearing(film=JournalBearing())
ON_FILMS = RotorSystem(bearing_a=FILM, bearing_b=FILM, gravity=9.81)


def run_for(sim, duration):
    out = []
    end = sim.t + duration
    while sim.t < end - 1e-12:
        r = sim.advance(end - sim.t)
        if not r.t.size:
            break
        out.append(r)
    return out


def test_bearing_cross_terms():
    # k_yx defaults to -k_xy (the old skew coupling); given, it and c_xy, c_yx go where they belong.
    old = RotorSystem(bearing_a=Bearing(kxy=3e3), bearing_b=Bearing(kxy=3e3))
    same = RotorSystem(bearing_a=Bearing(kxy=3e3, kyx=-3e3), bearing_b=Bearing(kxy=3e3, kyx=-3e3))
    np.testing.assert_array_equal(old.matrices()[3], same.matrices()[3])
    assert old.isotropic_supports and same.isotropic_supports
    b = Bearing(kxy=1e3, kyx=-4e3, cxy=7.0, cyx=-2.0)
    s = RotorSystem(bearing_a=b, bearing_b=Bearing())
    _, C, _, K = s.matrices()
    assert (K[XA, YA], K[YA, XA], C[XA, YA], C[YA, XA]) == (1e3, -4e3, 7.0, -2.0)
    assert K[XB, YB] == K[YB, XB] == C[XB, YB] == 0.0
    assert not s.isotropic_supports
    # A film ignores the springs: only the journal's mass is left of them.
    f = RotorSystem(bearing_a=Bearing(kx=1e9, cx=1e9, kxy=1e9, swirl=0.5, film=JournalBearing()))
    _, C, _, K = f.matrices()
    assert K[XA, XA] == f.shaft_stiffness()[0, 0] and C[XA, XA] == 0.0 and not f.circulatory()[XA, YA]
    assert f.speed_dependent and not f.isotropic_supports


def test_bearing_loads_are_the_weights_shares():
    s = ON_FILMS
    m = s.disc_mass + s.bearing_a.mass + s.bearing_b.mass
    assert sum(s.bearing_loads) == pytest.approx(m * 9.81)
    assert s.bearing_loads[0] == pytest.approx(s.bearing_loads[1])
    # Overhung past B: A is pulled down (negative), B carries more than the whole disc.
    o = s.with_(position=1.2)
    a, b = o.bearing_loads
    L, z = o.length, 1.2 * o.length
    assert b == pytest.approx(9.81 * (o.disc_mass * z / L + o.bearing_b.mass))
    assert a == pytest.approx(9.81 * (o.disc_mass * (1 - z / L) + o.bearing_a.mass)) and a < 0.0
    # With the shaft's own mass too (consistent loads carry it exactly).
    d = s.with_(density=7850.0)
    assert sum(d.bearing_loads) == pytest.approx((m + d.shaft_mass) * 9.81)
    assert RotorSystem(bearing_a=FILM, bearing_b=FILM).bearing_loads == (0.0, 0.0)


@pytest.mark.parametrize("position", [0.5, 0.3, 1.2])
@pytest.mark.parametrize("rpm", [0.0, 1000.0, 4000.0])
def test_journals_rest_where_their_films_hold_them(position, rpm):
    s = ON_FILMS.with_(position=position)
    w = rpm / RPM
    q = s.static_deflection(w)
    for (jx, jy), load, st in zip(((XA, YA), (XB, YB)), s.bearing_loads, s.bearing_states(w)):
        expected = math.copysign(FILM.film.clearance, load) * np.array(st.position())
        np.testing.assert_allclose(q[[jx, jy]], expected, rtol=1e-9, atol=1e-15)
        assert st.eccentricity <= 0.9 + 1e-12  # held at ε = 0.9 when slower
    # The films' coefficients are in the stiffness and damping at that speed.
    K, C = s.stiffness(w), s.damping(w)
    st = s.bearing_states(w)[0]
    K0, C0 = s.matrices()[3], s.matrices()[1]
    np.testing.assert_allclose(K[np.ix_([XA, YA], [XA, YA])] - K0[np.ix_([XA, YA], [XA, YA])], st.k)
    np.testing.assert_allclose(C[np.ix_([XA, YA], [XA, YA])] - C0[np.ix_([XA, YA], [XA, YA])], st.c)


def test_simulation_holds_the_steady_state():
    # Started on the 1X steady state about the sag, at constant speed, it stays there: the films'
    # preload, stiffness and damping are the same in the time stepping as in the harmonic solution.
    s = ON_FILMS.with_(unbalance=50e-6)
    w = 1500.0 / RPM
    Q = synchronous_response(s, [w])[0]
    qs = s.static_deflection(w)
    sim = RotorSimulator(s)
    sim.omega = sim.target = w
    sim.state = np.concatenate([qs + Q.real, (1j * w * Q).real])
    res = run_for(sim, 0.2)
    t = np.concatenate([r.t for r in res])
    q = np.concatenate([r.q for r in res])
    expect = qs + (Q[None, :] * np.exp(1j * w * t)[:, None]).real
    assert np.abs(q - expect).max() < 1e-4 * np.abs(Q).max()


def test_asymmetric_shaft_on_films_tends_to_the_round_one():
    s = ON_FILMS.with_(unbalance=50e-6)
    w = np.array([1000.0, 2500.0]) / RPM
    np.testing.assert_allclose(synchronous_response(s.with_(asymmetry=1e-9), w), synchronous_response(s, w),
                               rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(s.with_(asymmetry=1e-9).static_deflection(2500.0 / RPM),
                               s.static_deflection(2500.0 / RPM), rtol=1e-6, atol=1e-12)


def test_stiff_shaft_whirls_at_lunds_threshold():
    # A shaft far stiffer than the films is a rigid rotor: its onset is the bearing's threshold for the
    # mass each bearing carries, its whirl about half the speed.
    s = ON_FILMS.with_(diameter=0.1)
    film = FILM.film.with_load(s.bearing_loads[0])
    expected = film.threshold(60_000.0 / RPM, mass=s.bearing_loads[0] / 9.81)
    onset = stability_onset(s, 60_000.0 / RPM)
    assert onset.omega == pytest.approx(expected, rel=2e-3)
    assert onset.order == pytest.approx(0.5, abs=0.02)


def test_vertical_rotor_on_unloaded_films_is_never_stable():
    # An unloaded plain bearing has no direct stiffness, only the swirl's skew coupling: it whirls at once.
    onset = stability_onset(ON_FILMS.with_(gravity=0.0), 10_000.0 / RPM)
    assert onset is not None and onset.omega * RPM < 100.0


def test_journal_bearing_preset():
    p = next(p for p in rotor_presets() if p.name == "Journal bearings (oil whip)")
    c = campbell(p.system, p.top_rpm / RPM)
    forward = [round(w * RPM) for w, r, z in c.criticals if z < 0.3 and r > 0]
    assert forward == [1845]
    onset = stability_onset(p.system, p.top_rpm / RPM)
    assert round(onset.omega * RPM) == 3576 and round(onset.freq_hz, 1) == 30.5
    assert least_damped(p.system, p.target_rpm / RPM)[0] > 0  # it starts out stable
    zeta, f, _ = least_damped(p.system, 5000.0 / RPM)
    assert f == pytest.approx(30.7, abs=0.05) and 1 / (-zeta * 2 * math.pi * f) == pytest.approx(0.85, abs=0.01)
    q = p.system.static_deflection(2500.0 / RPM)
    assert q[XA] * 1e6 == pytest.approx(3.98, abs=0.005) and q[YA] * 1e6 == pytest.approx(-1.50, abs=0.005)
    assert q[YD] < q[YA]  # the disc sags below the journals


def test_journal_bearing_preset_splits_its_critical():
    # The Theory: the anisotropic film adds a backward crossing, more lightly damped than the forward one.
    p = next(p for p in rotor_presets() if p.name == "Journal bearings (oil whip)")
    c = campbell(p.system, p.top_rpm / RPM)
    lightly = sorted((round(w * RPM), r > 0, round(z, 3)) for w, r, z in c.criticals if z < 0.3)
    assert lightly == [(1845, True, 0.007), (1853, False, 0.002)]
    # The linear response ignores the clearance: the journals' orbits peak at about 28 µm, beyond 15 µm.
    rpm = np.arange(1800.0, 1900.0, 0.25)
    q = synchronous_response(p.system, rpm / RPM)
    half_length = (np.abs(q[:, XA] + 1j * q[:, YA]) + np.abs(q[:, XA] - 1j * q[:, YA])) / 2
    assert rpm[half_length.argmax()] == pytest.approx(1845.0, abs=0.5)
    assert half_length.max() * 1e6 == pytest.approx(28, abs=0.5) and half_length.max() > p.system.bearing_a.film.clearance
    rigid = Bearing(kx=1e12, ky=1e12, cx=0.0, cy=0.0, mass=1e-3)
    on_rigid = campbell(p.system.with_(bearing_a=rigid, bearing_b=rigid), p.top_rpm / RPM)
    assert min(round(w * RPM) for w, r, _ in on_rigid.criticals if r > 0) == 1854


def test_heavier_loaded_films_whirl_later():
    # The Theory's Things to try: a looser bearing with thinner oil sinks lower and holds off oil whip.
    loose = Bearing(film=JournalBearing(clearance=30e-6, viscosity=0.01))
    s = ON_FILMS.with_(bearing_a=loose, bearing_b=loose)
    assert round(ON_FILMS.bearing_states(3000.0 / RPM)[0].eccentricity, 2) == 0.25
    assert round(s.bearing_states(3000.0 / RPM)[0].eccentricity, 2) == 0.71
    assert round(stability_onset(s, 10_000.0 / RPM).omega * RPM) == 4358
    q = ON_FILMS.static_deflection(0.0)
    assert (round(q[XA] * 1e6, 1), round(q[YA] * 1e6, 1)) == (4.8, -12.6)


def test_film_terms_are_cached_per_speed():
    """A sweep asks for each speed's film terms several times: they're kept, read-only, and match a fresh copy."""
    s = ON_FILMS.with_()
    w = 3000.0 / RPM
    Kf, Cf, p = s._film_terms(w)
    assert s._film_terms(-w)[0] is Kf  # the films don't care which way it spins
    assert not Kf.flags.writeable and not p.flags.writeable
    fresh = s.with_()._new_film_terms(w)
    for a, b in zip((Kf, Cf, p), fresh, strict=True):
        assert np.array_equal(a, b)
    for x in np.linspace(100.0, 9000.0, 300) / RPM:
        s.stiffness(x)
    assert len(s.__dict__["_film_terms_at"]) <= rotor.CACHE_TERMS
    assert len(s.__dict__["_film_states"]) == 301
