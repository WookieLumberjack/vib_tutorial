import math

import numpy as np
import pytest

from vib_tutorial.core import (
    ChainSystem,
    ForceController,
    ForceKind,
    ForceSettings,
    Simulator,
    frf,
    friction_frf,
    friction_transmissibility,
    modal_analysis,
    transmissibility,
)

CHAIN = ChainSystem([1.0] * 3, [400.0] * 3, [2.0] * 3, [0.5, 0.3, 0.8])


def steady_amplitudes(system, target, amplitude, freq_hz, settle=15.0, cycles=4.0):
    """First-harmonic displacement amplitude of each mass, simulated, after `settle` seconds."""
    force = ForceController(ForceSettings(target=target, kind=ForceKind.HARMONIC, amplitude=amplitude, freq_hz=freq_hz))
    sim = Simulator(system, force)
    force.switch_on()
    out = []
    end = settle + round(cycles * freq_hz) / freq_hz
    while sim.t < end - 1e-12:
        t, x = sim.advance(end - sim.t)[:2]
        if not t.size:
            break
        if sim.t > settle:
            out.append((t, x))
    t, x = (np.concatenate(a) for a in zip(*out))
    keep = t > settle
    t, x = t[keep], x[keep]
    return np.abs(2 * np.mean(x * np.exp(-2j * np.pi * freq_hz * t)[:, None], axis=0)), np.ptp(x, axis=0)


def test_single_mass_matches_the_closed_form():
    # k X (1 - r^2) and the friction harmonic 4F_f/pi are in quadrature: |X| = sqrt(F^2 - (4F_f/pi)^2) / |k - w^2 m|.
    m, k, Ff, F = 1.0, 400.0, 1.0, 10.0
    s = ChainSystem([m], [k], [0.0], [Ff])
    f = np.linspace(0.5, 6.0, 40)
    r = friction_frf(s, f, 0, F)
    w = 2 * np.pi * f
    exact = math.sqrt(F**2 - (4 * Ff / math.pi) ** 2) / np.abs(k - w**2 * m)
    np.testing.assert_allclose(np.abs(r.X[:, 0]), exact, rtol=1e-8)
    assert r.converged.all() and not r.stuck.any()


def test_single_mass_sticks_when_friction_beats_the_drive():
    # No sliding solution once 4F_f/pi > F: the describing function holds the mass still.
    s = ChainSystem([1.0], [400.0], [0.0], [8.0])
    r = friction_frf(s, np.linspace(0.5, 6.0, 40), 0, 10.0)
    assert r.stuck.all() and np.all(r.X == 0)


def test_without_friction_it_is_the_linear_frf():
    s = ChainSystem.uniform(4)
    f = np.linspace(0.2, 8.0, 200)
    np.testing.assert_allclose(friction_frf(s, f, 2, 7.0).X, 7.0 * frf(s, f, 2))
    np.testing.assert_allclose(friction_transmissibility(s, f, 0.01).X, 0.01 * transmissibility(s, f))


def test_ground_motion_on_a_single_mass():
    # The ground pulls through the spring: drive amplitude k X_g.
    m, k, Ff, Xg = 1.0, 400.0, 0.5, 0.02
    s = ChainSystem([m], [k], [0.0], [Ff])
    f = np.linspace(0.5, 6.0, 40)
    r = friction_transmissibility(s, f, Xg)
    exact = math.sqrt((k * Xg) ** 2 - (4 * Ff / math.pi) ** 2) / np.abs(k - (2 * np.pi * f) ** 2 * m)
    np.testing.assert_allclose(np.abs(r.X[:, 0]), exact, rtol=1e-8)


def test_strong_drive_approaches_the_linear_response():
    f = np.linspace(0.3, 8.0, 300)
    lin = np.abs(frf(CHAIN, f, 2))
    strong = np.abs(friction_frf(CHAIN, f, 2, 1000.0).X) / 1000.0
    assert np.max(np.abs(strong / lin - 1)) < 0.03


@pytest.mark.parametrize("freq_hz", [1.6, 4.0, 5.5])
def test_matches_a_simulated_steady_state(freq_hz):
    # Every mass slides through the cycle here, where the describing function is at its best;
    # friction takes 5-20% off the linear response, well above the tolerance.
    sim, _ = steady_amplitudes(CHAIN, 2, 10.0, freq_hz)
    df = np.abs(friction_frf(CHAIN, np.array([freq_hz]), 2, 10.0).X[0])
    lin = 10.0 * np.abs(frf(CHAIN, np.array([freq_hz]), 2)[0])
    np.testing.assert_allclose(df, sim, rtol=0.01)
    assert np.max(lin / sim) > 1.04


def test_a_stuck_mass_stays_put_in_the_simulation_too():
    s = ChainSystem([1.0] * 3, [400.0] * 3, [2.0] * 3, [6.0, 0.0, 0.3])
    r = friction_frf(s, np.array([6.0]), 2, 10.0)
    assert list(r.stuck[0]) == [True, False, False]
    sim, swing = steady_amplitudes(s, 2, 10.0, 6.0)
    assert swing[0] == 0.0
    np.testing.assert_allclose(np.abs(r.X[0, 1:]), sim[1:], rtol=0.01)


def test_converges_on_random_chains():
    rng = np.random.default_rng(3)
    unconverged = total = 0
    for _ in range(25):
        n = int(rng.integers(1, 9))
        s = ChainSystem(
            rng.uniform(0.2, 5, n),
            rng.uniform(50, 2000, n),
            rng.choice([0.0, 0.5, 3.0], n) * rng.uniform(0, 1, n),
            rng.uniform(0, 3, n) * (rng.uniform(0, 1, n) > 0.3),
        )
        fn = [mode.fn_hz for mode in modal_analysis(s).modes]
        f = np.geomspace(0.2 * min(fn), 2.5 * max(fn), 300)
        r = friction_frf(s, f, int(rng.integers(n)), float(rng.choice([1.0, 10.0, 100.0])))
        assert np.isnan(r.X[~r.converged]).all() and np.isfinite(r.X[r.converged]).all()
        unconverged += (~r.converged).sum()
        total += f.size
    assert unconverged / total < 0.002
