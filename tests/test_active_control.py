"""Active control: the closed loop's poles, the light-gain estimate, spillover, lag, the TMD and skyhook."""

import numpy as np
import pytest

from vib_tutorial.core.active_control import (
    Controller,
    best_gain,
    closed_loop,
    frf,
    max_real,
    modal_damping_estimate,
    oscillating_poles,
    skyhook,
    tmd_for,
    tracked_damping,
)
from vib_tutorial.core.model import ChainSystem


def chain():
    return ChainSystem.uniform(4)


def onset(system, ctrl, low=0.0, high=1e4):
    """The smallest gain with a pole in the right half plane, by bisection (it must be unstable at `high`)."""
    def unstable(g):
        return max_real(closed_loop(system, Controller(ctrl.sensor, ctrl.actuator, g, ctrl.lag))[0]) > 0
    assert unstable(high)
    for _ in range(60):
        mid = 0.5 * (low + high)
        low, high = (low, mid) if unstable(mid) else (mid, high)
    return high


def test_open_loop_matches_the_chain():
    s = chain()
    A, _, _ = closed_loop(s, None)
    M, C, K = s.matrices()
    lam = np.linalg.eigvals(np.block([[np.zeros((4, 4)), np.eye(4)],
                                      [-np.linalg.solve(M, K), -np.linalg.solve(M, C)]]))
    assert np.allclose(np.sort_complex(np.linalg.eigvals(A)), np.sort_complex(lam))


def test_light_gain_matches_the_estimate():
    s = chain()
    ctrl = Controller(3, 3, 0.05)
    open_z = np.array([p.zeta for p in oscillating_poles(closed_loop(s, None)[0])])
    closed_z = np.array([p.zeta for p in oscillating_poles(closed_loop(s, ctrl)[0])])
    assert np.allclose(closed_z - open_z, modal_damping_estimate(s, ctrl), rtol=0.02)


def test_feedback_is_a_damper_to_ground():
    """Ideal collocated velocity feedback is exactly a dashpot from the mass to the ground."""
    s = chain()
    g = 20.0
    A, B, Cx = closed_loop(s, Controller(3, 3, g))
    grounded = s.copy()
    M, C, K = grounded.matrices()
    C = C.copy()
    C[3, 3] += g
    Mi = np.linalg.inv(M)
    A2 = np.block([[np.zeros((4, 4)), np.eye(4)], [-Mi @ K, -Mi @ C]])
    assert np.allclose(A, A2)
    f = np.array([0.5, 1.2, 3.0])
    h = frf(A, B, Cx, 3, 3, f)
    direct = [np.linalg.solve(K - (2 * np.pi * fk) ** 2 * M + 2j * np.pi * fk * C, np.eye(4)[3])[3] for fk in f]
    assert np.allclose(h, direct)


def test_collocated_is_stable_at_any_gain():
    s = chain()
    for g in np.geomspace(0.1, 1e4, 30):
        assert max_real(closed_loop(s, Controller(3, 3, g))[0]) < 0


def test_tracking_keeps_each_mode_its_own():
    """Mode 1 turns overdamped near its optimum; the others must not take its place (the curves stay smooth)."""
    s = chain()
    g = np.geomspace(0.05, 2000.0, 160)
    z = tracked_damping(s, Controller(3, 3, 0.0), g)
    for j in range(1, 4):
        d = np.abs(np.diff(z[:, j]))
        assert np.nanmax(d) < 0.05


def test_lag_destabilizes():
    s = chain()
    assert onset(s, Controller(3, 3, 0.0, 0.020)) == pytest.approx(96, abs=1)
    assert onset(s, Controller(3, 3, 0.0, 0.010)) == pytest.approx(201, abs=2)


def test_active_control_trials():
    s = chain()
    # Velocity feedback at the tip
    open_z = [p.zeta for p in oscillating_poles(closed_loop(s, None)[0])]
    assert open_z[0] == pytest.approx(0.017, abs=0.001)
    z = [p.zeta for p in oscillating_poles(closed_loop(s, Controller(3, 3, 20.0))[0])]
    assert z[0] == pytest.approx(0.72, abs=0.005)
    # Too much gain pins the mass
    g, zbest = best_gain(s, Controller(3, 3, 0.0))
    assert g == pytest.approx(25, rel=0.1)
    g = np.geomspace(1.0, 1000.0, 600)
    z = tracked_damping(s, Controller(3, 3, 0.0), g)
    assert g[np.isnan(z[:, 0])][0] == pytest.approx(26, abs=0.5)
    f = [p.freq_hz for p in oscillating_poles(closed_loop(s, Controller(3, 3, 200.0))[0])]
    M, _, K = s.matrices()
    fixed = np.sqrt(np.linalg.eigvals(np.linalg.solve(M[:3, :3], K[:3, :3])).real) / (2 * np.pi)
    assert np.sort(fixed) == pytest.approx(f, abs=0.05)
    assert f == pytest.approx([2.4, 4.5, 5.9], abs=0.05)
    # Non-collocated: spillover
    est = modal_damping_estimate(s, Controller(0, 3, 20.0))
    assert est[1] == pytest.approx(-0.17, abs=0.005)
    z = [p.zeta for p in oscillating_poles(closed_loop(s, Controller(0, 3, 20.0))[0])]
    assert z[1] == pytest.approx(-0.10, abs=0.005)
    assert onset(s, Controller(0, 3, 0.0)) == pytest.approx(6.1, abs=0.05)
    # Active against a tuned mass damper
    tmd = tmd_for(s, 3, 0.05)
    assert tmd[1] == pytest.approx(0.116, abs=0.001)
    fgrid = np.geomspace(0.3, 12.0, 4000)
    peak = lambda A, B, Cx: np.max(np.abs(frf(A, B, Cx, 3, 3, fgrid)))  # noqa: E731
    assert peak(*closed_loop(s, None)) == pytest.approx(0.26, abs=0.005)
    assert peak(*closed_loop(s, None, tmd)) == pytest.approx(0.051, abs=0.001)
    assert peak(*closed_loop(s, Controller(3, 3, 2.3))) == pytest.approx(0.051, abs=0.002)
    # Skyhook isolation
    passive, sky, _ = skyhook(np.array([1.0, 5.0]), 0.1, 0.5)
    r = np.geomspace(0.1, 10, 20001)
    assert np.max(skyhook(r, 0.1, 0.5)[0]) == pytest.approx(5.1, abs=0.05)
    assert passive[1] == pytest.approx(0.059, abs=0.001)
    assert np.max(skyhook(r, 0.1, 0.5)[1]) == pytest.approx(1.15, abs=0.005)
    assert sky[1] == pytest.approx(0.041, abs=0.001)
