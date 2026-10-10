"""Jeffcott rotor: matrices, whirl analysis and the exact time stepping."""

import math

import numpy as np
import pytest

from vib_tutorial.core.rotor import (
    RPM,
    TX,
    TY,
    XD,
    YD,
    Bearing,
    RotorSimulator,
    RotorSystem,
    campbell,
    unbalance_response,
    whirl_modes,
)

RIGID = Bearing(kx=1e12, ky=1e12, cx=0.0, cy=0.0, mass=1e-3)
UNDAMPED = Bearing(cx=0.0, cy=0.0)


def run(sim: RotorSimulator, seconds: float) -> list:
    """Advance in pieces (advance caps its steps); return each piece's samples."""
    out = []
    end = sim.t + seconds
    while sim.t < end - 1e-12:
        out.append(sim.advance(min(0.25, end - sim.t)))
    return out


def test_matrices_are_symmetric_and_gyroscopic_matrix_skew():
    M, C, G, K = RotorSystem(position=0.3).matrices()
    for A in (M, C, K):
        np.testing.assert_allclose(A, A.T, atol=1e-9 * np.abs(A).max())
    np.testing.assert_array_equal(G, -G.T)
    assert G[TX, TY] > 0 and np.count_nonzero(G) == 2  # it couples only the disc's two slopes
    assert np.linalg.eigvalsh(K).min() > 0  # held by the bearings
    # The shaft alone can move rigidly: two zero eigenvalues per plane.
    assert np.sum(np.abs(np.linalg.eigvalsh(RotorSystem().shaft_stiffness())) < 1e-6) == 2


@pytest.mark.parametrize("position", [0.5, 0.3, 0.8])
def test_shaft_flexibility_matches_beam_formulas(position):
    s = RotorSystem(position=position)
    a, b, L, EI = s.a, s.b, s.length, s.ei
    # Bearings held still: invert the disc's 2x2 stiffness (deflection, slope).
    flex = np.linalg.inv(s.shaft_stiffness()[1:3, 1:3])
    assert flex[0, 0] == pytest.approx(a * a * b * b / (3 * EI * L))
    assert flex[1, 1] == pytest.approx((a**3 + b**3) / (3 * EI * L * L))
    assert abs(flex[0, 1]) == pytest.approx(a * b * abs(b - a) / (3 * EI * L), abs=1e-15)


def test_shape_matrix_reproduces_stations_and_a_rigid_tilt():
    s = RotorSystem(position=0.35)
    S = s.shape_matrix([0.0, s.a, s.length])
    np.testing.assert_allclose(S, [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], atol=1e-12)
    # A rigid tilt w = z: bearings at 0 and L, the disc at a with slope 1.
    z = np.linspace(0.0, s.length, 11)
    np.testing.assert_allclose(s.shape_matrix(z) @ [0.0, s.a, 1.0, s.length], z, atol=1e-12)


def test_classic_jeffcott_critical_speed():
    # Disc at midspan, identical isotropic bearings, light journals.
    light = Bearing(mass=1e-4, cx=0.0, cy=0.0)  # a damper on a light journal would stiffen it
    s = RotorSystem(bearing_a=light, bearing_b=light)
    assert s.jeffcott_speed() == pytest.approx(math.sqrt((1 / (1 / (48 * s.ei / s.length**3) + 1 / 1e5)) / 1.0))
    c = campbell(s, 2.0 * s.jeffcott_speed())
    forward = [w for w, whirl, _ in c.criticals if whirl > 0.5]
    assert forward[0] == pytest.approx(s.jeffcott_speed(), rel=1e-3)
    # On rigid bearings it is the shaft alone: 48 EI / L^3.
    rigid = RotorSystem(bearing_a=RIGID, bearing_b=RIGID)
    f = whirl_modes(rigid, 0.0).freq_hz[0]
    assert 2 * math.pi * f == pytest.approx(math.sqrt(48 * rigid.ei / rigid.length**3), rel=1e-4)


def test_midspan_disc_has_no_gyroscopic_effect_on_translation():
    s = RotorSystem()
    for w in (0.0, 500.0, 1500.0):
        m = whirl_modes(s, w)
        assert m.freq_hz[0] == pytest.approx(whirl_modes(s, 0.0).freq_hz[0], rel=1e-9)
        assert m.freq_hz[1] == pytest.approx(m.freq_hz[0], rel=1e-9)
        assert sorted(np.sign(m.whirl[:2])) == [-1, 1]  # one forward, one backward, circular
        np.testing.assert_allclose(np.abs(m.whirl[:2]), 1.0, atol=1e-9)


def test_gyroscopic_effect_splits_forward_and_backward_whirl():
    s = RotorSystem(position=0.3)
    still, fast = whirl_modes(s, 0.0), whirl_modes(s, 1000.0)
    assert still.freq_hz[1] == pytest.approx(still.freq_hz[0], rel=1e-9)
    # At speed, the forward whirl stiffens and the backward whirl softens.
    fw = fast.freq_hz[fast.whirl > 0.5]
    bw = fast.freq_hz[fast.whirl < -0.5]
    assert fw[0] > still.freq_hz[0] > bw[0]
    # The undamped conical mode: Id ω² ∓ Ip Ω ω - k_θ = 0 for a disc on rigid bearings at midspan.
    rigid = RotorSystem(bearing_a=RIGID, bearing_b=RIGID)
    k_theta = 12 * rigid.ei / rigid.length
    W = 2000.0
    m = whirl_modes(rigid, W)
    for sign in (1, -1):
        w = (sign * rigid.ip * W + math.sqrt((rigid.ip * W) ** 2 + 4 * rigid.id * k_theta)) / (2 * rigid.id)
        assert np.min(np.abs(2 * math.pi * m.freq_hz - w)) < 1e-4 * w
        i = int(np.argmin(np.abs(2 * math.pi * m.freq_hz - w)))
        assert np.sign(m.whirl[i]) == sign


def test_anisotropic_supports_give_backward_whirl_between_the_split_criticals():
    soft_x = Bearing(kx=2e4, ky=8e4, cx=20.0, cy=20.0)
    s = RotorSystem(bearing_a=soft_x, bearing_b=soft_x)
    c = campbell(s, 400.0)
    # The two lowest modes are elliptical: one mostly in x (lower), one in y.
    m = whirl_modes(s, 0.0)
    assert m.freq_hz[1] > 1.1 * m.freq_hz[0]
    w1, w2 = 2 * math.pi * m.freq_hz[0], 2 * math.pi * m.freq_hz[1]
    between = 0.5 * (w1 + w2)
    Q = unbalance_response(s, [0.5 * w1, between, 2 * w2])
    X, Y = Q[:, XD], Q[:, YD]
    direction = np.imag(X * np.conj(Y))  # > 0: forward (x toward y)
    assert direction[0] > 0 and direction[1] < 0 and direction[2] > 0
    # Both are critical speeds, and their modes whirl in straight lines (x, then y).
    speeds = [w for w, _, _ in c.criticals]
    for w in (w1, w2):
        assert min(abs(np.array(speeds) - w)) < 0.02 * w
    assert all(abs(r) < 0.1 for w, r, _ in c.criticals if w < 1.2 * w2)


def test_simulation_matches_the_steady_unbalance_response():
    s = RotorSystem(position=0.35)
    sim = RotorSimulator(s)
    sim.omega = sim.target = 150.0
    pieces = run(sim, 4.0)
    Q = unbalance_response(s, [sim.omega])[0]
    last = pieces[-1]
    expected = np.real(Q[None, :] * np.exp(1j * last.phase[:, None]))
    np.testing.assert_allclose(last.q, expected, atol=1e-5 * np.abs(Q).max())


def test_gyroscopic_forces_do_no_work():
    s = RotorSystem(position=0.3, bearing_a=UNDAMPED, bearing_b=UNDAMPED, unbalance=0.0)
    sim = RotorSimulator(s)
    sim.omega = sim.target = 1200.0
    sim.tap(0.01, -0.004)
    e0 = sum(sim.energy())
    run(sim, 1.0)
    assert sum(sim.energy()) == pytest.approx(e0, rel=1e-9)
    # The energy moved between the planes and into the tilt: it is not a single mode.
    assert abs(sim.q[TY]) + abs(sim.q[TX]) > 0


def test_speed_ramps_to_the_target_and_the_phase_follows():
    sim = RotorSimulator()
    sim.target = 3000.0 / RPM
    sim.accel = 1000.0 / RPM
    pieces = run(sim, 4.0)
    w = np.concatenate([p.omega for p in pieces])
    t = np.concatenate([p.t for p in pieces])
    phi = np.concatenate([p.phase for p in pieces])
    assert sim.omega == sim.target and w.max() == sim.target
    assert np.all(np.diff(w) >= 0)
    i = np.searchsorted(t, 1.5)
    assert w[i] == pytest.approx(sim.accel * t[i], rel=1e-3)
    # φ = ½ α t² while ramping (3 s), then Ω t.
    ramp = sim.target / sim.accel
    expected = 0.5 * sim.accel * ramp**2 + sim.target * (t[-1] - ramp)
    assert phi[-1] == pytest.approx(expected, rel=1e-6)
    # Back down to rest, and resting stays still.
    sim.target = 0.0
    run(sim, 4.0)
    assert sim.omega == 0.0


def test_passing_the_critical_peaks_and_the_phase_lags_by_180():
    s = RotorSystem()
    wc = s.jeffcott_speed()
    Q = unbalance_response(s, [0.3 * wc, wc, 3.0 * wc])
    amp = np.abs(Q[:, XD])
    assert amp[1] > 5 * amp[0] and amp[1] > 5 * amp[2]
    # Far above the critical the disc spins about its centre of mass: |x| -> e.
    e = s.unbalance / s.disc_mass
    assert amp[2] == pytest.approx(e, rel=0.2)
    lag = -np.degrees(np.angle(Q[:, XD]))  # the heavy spot leads the high spot by this
    assert lag[0] < 20 and 60 < lag[1] < 120 and lag[2] > 150
