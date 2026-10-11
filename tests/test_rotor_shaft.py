"""The shaft's own mass: beam finite elements, reduced to the stations and the shaft's fixed-interface modes."""

import math

import numpy as np
import pytest

import vib_tutorial.core.rotor as rotor
from vib_tutorial.core.rotor import (
    MODE_LIMIT,
    RPM,
    STEEL,
    XD,
    YD,
    Bearing,
    RotorSimulator,
    RotorSystem,
    campbell,
    synchronous_response,
    whirl_modes,
)
from vib_tutorial.core.rotor_presets import rotor_presets
from vib_tutorial.core.rotor_sweep import vibration

UNDAMPED = Bearing(cx=0.0, cy=0.0)
LONG = next(p for p in rotor_presets() if p.name == "Long shaft, with its mass").system


def forward_criticals(s: RotorSystem, top_rpm: float = 10_000.0) -> list[int]:
    crits = campbell(s, top_rpm / RPM, n_modes=8).criticals
    return [round(w * RPM) for w, r, z in crits if r > 0.1 and z < 0.3]


def unreduced(s: RotorSystem, monkeypatch) -> RotorSystem:
    """The same rotor keeping every fixed-interface mode: the full element model."""
    monkeypatch.setattr(rotor, "MODE_LIMIT", math.inf)
    monkeypatch.setattr(rotor, "MAX_MODES", 10**6)
    return s.with_()  # a new instance: nothing cached from the reduced one


def test_a_massless_shaft_is_the_jeffcott_model_however_it_is_meshed():
    base = RotorSystem(position=0.3, bearing_a=Bearing(kxy=2e3), internal_damping=5.0, gravity=9.81, bow=1e-5)
    assert base.n_dof == 8 and base.modes_kept == 0 and base.shaft_mass == 0.0
    for n in (2, 5, 20):
        s = base.with_(elements=n)
        for a, b in zip(s.matrices(), base.matrices()):
            np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9 * np.abs(b).max())
        z = np.linspace(-0.1, 0.6, 15)
        np.testing.assert_allclose(s.shape_matrix(z), base.shape_matrix(z), atol=1e-12)
    M = base.matrices()[0]
    assert np.count_nonzero(M - np.diag(np.diag(M))) == 0  # diagonal, as before
    with pytest.raises(ValueError):
        RotorSystem(elements=1)
    with pytest.raises(ValueError):
        RotorSystem(density=-1.0)


@pytest.mark.parametrize("position", [0.5, 0.3, 1.2])
def test_the_reduction_keeps_the_full_element_models_modes(position, monkeypatch):
    s = RotorSystem(position=position, density=STEEL, elements=16, bearing_a=UNDAMPED, bearing_b=UNDAMPED)
    assert 0 < s.modes_kept and s.fixed_interface_freqs.max() <= MODE_LIMIT
    assert s.n_dof == 8 + 2 * s.modes_kept
    reduced = whirl_modes(s, 3000.0 / RPM).freq_hz
    full = whirl_modes(unreduced(s, monkeypatch), 3000.0 / RPM).freq_hz
    k = int(np.sum(full < MODE_LIMIT))
    error = np.abs(reduced[:k] / full[:k] - 1.0)
    assert error[full[:k] < 500.0].max() < 2e-3 and error.max() < 1e-2  # as the theory notes say


def test_a_light_shaft_adds_about_17_35_of_its_mass_to_the_disc():
    s = RotorSystem(density=STEEL)
    assert s.shaft_mass == pytest.approx(0.308, abs=5e-4)
    assert forward_criticals(s) == [1458]
    assert forward_criticals(RotorSystem()) == [1584]
    rule = 1584 / math.sqrt(1 + 17 / 35 * s.shaft_mass / s.disc_mass)
    assert round(rule) == 1477


def test_the_long_shaft_converges_from_above_with_consistent_mass():
    second = {lumped: [forward_criticals(LONG.with_(elements=n, lumped=lumped))[1] for n in (2, 4, 8)]
              for lumped in (False, True)}
    assert second[False] == [7393, 6591, 6582]
    assert second[True] == [8187, 6733, 6620]
    # At rest and undamped, consistent mass is Rayleigh–Ritz: every frequency falls as elements are added.
    still = LONG.with_(bearing_a=UNDAMPED, bearing_b=UNDAMPED)
    f = [whirl_modes(still.with_(elements=n), 0.0).freq_hz[:6] for n in (2, 4, 8, 16)]
    assert np.all(np.diff(f, axis=0) <= 1e-9)


def test_the_shafts_weight_and_the_sag():
    s = LONG.with_(gravity=9.81)
    _, iy = s.plane_dofs
    shift = np.zeros(4 + s.modes_kept)
    shift[[0, 1, 3]] = 1.0  # all of it moving down together
    weight = -shift @ s.gravity_load()[iy]
    assert weight == pytest.approx(9.81 * (s.disc_mass + 2 * LONG.bearing_a.mass + s.shaft_mass), rel=1e-12)
    assert round(-s.static_deflection()[YD] * 1e6) == 375
    assert round(-s.with_(density=0.0).static_deflection()[YD] * 1e6) == 200


def test_the_shafts_gyroscopic_moment_splits_its_own_modes():
    # A shaft alone (a tiny disc with no polar inertia): its own sections' spin still splits the whirl.
    s = RotorSystem(density=STEEL, disc_mass=1e-3, ip=1e-9, id=1e-9, elements=12, diameter=0.04,
                    bearing_a=UNDAMPED, bearing_b=UNDAMPED)
    at_rest = whirl_modes(s, 0.0)
    spinning = whirl_modes(s, 20_000.0 / RPM)
    fw = spinning.freq_hz[spinning.whirl > 0.5]
    bw = spinning.freq_hz[spinning.whirl < -0.5]
    assert fw[0] > at_rest.freq_hz[0] > bw[0]
    G = s.matrices()[2]
    np.testing.assert_allclose(G, -G.T, atol=1e-15)


def test_simulation_matches_the_steady_state_and_conserves_energy():
    s = LONG.with_(internal_damping=10.0, bearing_a=Bearing(kxy=3e3, cx=200.0, cy=200.0, mass=0.5),
                   gravity=9.81, bow=20e-6, skew=1e-3)
    w = 4000.0 / RPM
    sim = RotorSimulator(s)
    sim.omega = sim.target = w
    ts, qs = [], []
    while sim.t < 3.0:
        r = sim.advance(0.25)
        ts.append(r.t)
        qs.append(r.q)
    t, q = np.concatenate(ts), vibration(s, np.concatenate(qs), w)
    assert q.shape[1] == s.n_dof
    Q = synchronous_response(s, [w])[0]
    expect = np.real(Q[None, :] * np.exp(1j * w * t[-300:, None]))
    np.testing.assert_allclose(q[-300:], expect, atol=1e-5 * np.abs(Q).max())

    free = LONG.with_(bearing_a=UNDAMPED, bearing_b=UNDAMPED, unbalance=0.0)
    sim = RotorSimulator(free)
    sim.omega = sim.target = 3000.0 / RPM
    sim.tap(0.01, -0.005)
    e0 = sum(sim.energy())
    while sim.t < 0.5:
        sim.advance(0.1)
    assert sum(sim.energy()) == pytest.approx(e0, rel=1e-9)


def test_a_tap_moves_the_disc_and_the_shaft_with_it():
    sim = RotorSimulator(LONG)
    sim.tap(0.01, 0.0)
    M = LONG.matrices()[0]
    impulse = M @ sim.qdot  # the momentum given, DOF by DOF
    expect = np.zeros(LONG.n_dof)
    expect[XD] = 0.01
    np.testing.assert_allclose(impulse, expect, atol=1e-15)


def test_switching_the_shaft_mass_on_keeps_the_stations_motion():
    sim = RotorSimulator(RotorSystem())
    sim.omega = sim.target = 1000.0 / RPM
    sim.advance(0.1)
    q, qdot = sim.q.copy(), sim.qdot.copy()
    sim.set_system(LONG)
    assert sim.q.size == LONG.n_dof
    np.testing.assert_array_equal(sim.q[:8], q)
    np.testing.assert_array_equal(sim.qdot[:8], qdot)
    assert not sim.q[8:].any()
    sim.advance(0.05)
    sim.set_system(RotorSystem())
    assert sim.q.size == 8
