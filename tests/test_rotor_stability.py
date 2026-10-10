"""Rotor stability: cross-coupled bearings, internal damping, the onset speed and the full spectrum."""

import math

import numpy as np
import pytest

from vib_tutorial.core.rotor import (
    RPM,
    XD,
    YD,
    Bearing,
    RotorSimulator,
    RotorSystem,
    campbell,
    unbalance_response,
    whirl_modes,
)
from vib_tutorial.core.rotor_stability import full_spectrum, least_damped, log_decrement, stability_onset

UNDAMPED = Bearing(cx=0.0, cy=0.0)
MAX = 10_000.0 / RPM


def run(sim: RotorSimulator, seconds: float):
    """Advance in pieces (advance caps its steps); return (t, disc xy) over the run."""
    ts, xys = [], []
    end = sim.t + seconds
    while sim.t < end - 1e-12:
        r = sim.advance(min(0.25, end - sim.t))
        ts.append(r.t)
        xys.append(r.q[:, [XD, YD]])
    return np.concatenate(ts), np.concatenate(xys)


def cross_coupled(kxy: float, **kw) -> RotorSystem:
    return RotorSystem(bearing_a=Bearing(kxy=kxy), bearing_b=Bearing(kxy=kxy), **kw)


def test_cross_coupling_and_internal_damping_enter_as_skew_stiffness():
    s = RotorSystem(bearing_a=Bearing(kxy=3e3), internal_damping=20.0, position=0.3)
    M, C, G, K = s.matrices()
    skew = 0.5 * (K - K.T)
    assert np.count_nonzero(np.abs(skew) > 1e-9) == 2 and skew[0, 4] == 3e3  # A only, x row y column
    np.testing.assert_allclose(C, C.T, atol=1e-12)
    H = s.circulatory()
    np.testing.assert_allclose(H, -H.T, atol=1e-12)
    np.testing.assert_allclose(C[:4, :4] - np.diag(np.diag(RotorSystem().matrices()[1])[:4]),
                               s.loss_time * s.shaft_stiffness(), atol=1e-12)
    np.testing.assert_allclose(s.stiffness(100.0), K + 100.0 * H)
    # None of it: K symmetric, H zero, stable at every speed.
    assert not RotorSystem().circulatory().any()
    assert stability_onset(RotorSystem(), MAX) is None
    with pytest.raises(ValueError):
        RotorSystem(internal_damping=-1.0)


@pytest.mark.parametrize("position", [0.5, 0.3])
def test_internal_damping_alone_destabilizes_exactly_above_the_forward_critical(position):
    # Isotropic, no other damping: from the shaft a forward whirl at ω moves at ω − Ω, so the
    # internal damping takes energy out below the critical (ω > Ω) and puts it in above.
    s = RotorSystem(position=position, bearing_a=UNDAMPED, bearing_b=UNDAMPED, internal_damping=20.0)
    first = min(w for w, r, _ in campbell(s, MAX).criticals if r > 0)
    onset = stability_onset(s, MAX)
    assert onset.omega == pytest.approx(first, rel=1e-4)
    assert onset.whirl > 0.9 and 2 * math.pi * onset.freq_hz == pytest.approx(first, rel=1e-3)


def test_external_damping_raises_the_onset_and_the_whirl_is_subsynchronous():
    onsets = [stability_onset(RotorSystem(internal_damping=c), MAX) for c in (40.0, 20.0, 10.0)]
    speeds = [o.omega for o in onsets]
    assert speeds == sorted(speeds)  # less internal damping, against the same supports: later
    first = min(w for w, r, _ in campbell(RotorSystem(), MAX).criticals if r > 0)
    assert all(o.omega > first and o.order < 1 and o.whirl > 0.9 for o in onsets)
    assert onsets[1].omega * RPM == pytest.approx(3910, abs=5)


def test_cross_coupling_moves_damping_from_forward_to_backward_and_has_a_threshold():
    def first_pair(kxy):
        m = whirl_modes(cross_coupled(kxy), 1000.0 / RPM)
        fw = m.zeta[(m.whirl > 0.5) & (m.zeta < 0.3)][0]
        bw = m.zeta[(m.whirl < -0.5) & (m.zeta < 0.3)][0]
        return fw, bw

    pairs = [first_pair(k) for k in (0.0, 5e3, 1e4, 1.5e4)]
    fw, bw = np.array(pairs).T
    assert np.all(np.diff(fw) < 0) and np.all(np.diff(bw) > 0)
    # Threshold near c ω: the bearings' damping times the whirl frequency.
    c, w = Bearing().cx, 2 * math.pi * whirl_modes(RotorSystem(), 0.0).freq_hz[0]
    assert least_damped(cross_coupled(0.95 * c * w), 0.0)[0] > 0
    onset = stability_onset(cross_coupled(1.05 * c * w), MAX)
    assert onset is not None and onset.omega == 0.0 and onset.whirl > 0.9 and math.isinf(onset.order)


def test_steady_unbalance_response_matches_the_simulation_with_both_terms():
    s = RotorSystem(position=0.4, bearing_a=Bearing(kxy=4e3), internal_damping=10.0)
    w = 1200.0 / RPM
    sim = RotorSimulator(s)
    sim.omega = sim.target = w
    t, xy = run(sim, 3.0)
    Q = unbalance_response(s, np.array([w]))[0]
    expect = np.real(Q[[XD, YD]][None, :] * np.exp(1j * w * t[-200:, None]))
    np.testing.assert_allclose(xy[-200:], expect, atol=2e-3 * np.abs(Q).max())


def test_beyond_the_onset_a_tap_grows_at_the_natural_frequency():
    s = RotorSystem(internal_damping=20.0, unbalance=0.0)
    w = 5000.0 / RPM
    zeta, f, _ = least_damped(s, w)
    assert zeta < 0
    sim = RotorSimulator(s)
    sim.omega = sim.target = w
    sim.tap(0.0, -0.005)
    t, xy = run(sim, 2.0)
    r = np.hypot(xy[:, 0], xy[:, 1])
    early, late = r[(t > 0.4) & (t < 0.6)].max(), r[(t > 1.4) & (t < 1.6)].max()
    sigma = -zeta * 2 * math.pi * f / math.sqrt(1 - zeta**2)
    assert late / early == pytest.approx(math.exp(sigma * 1.0), rel=0.05)
    freqs, amps = full_spectrum(t[t > 1.0], xy[t > 1.0])
    peak = freqs[np.argmax(amps)]
    assert peak == pytest.approx(f, abs=1.5)  # forward (positive) and below 1X
    assert peak < w / (2 * math.pi)
    # Below the onset the same tap dies away.
    sim = RotorSimulator(s)
    sim.omega = sim.target = 3000.0 / RPM
    sim.tap(0.0, -0.005)
    t, xy = run(sim, 2.0)
    r = np.hypot(xy[:, 0], xy[:, 1])
    assert r[t > 1.8].max() < 0.1 * r[t < 0.2].max()


def test_full_spectrum_separates_forward_and_backward_whirl():
    t = np.linspace(0.0, 2.0, 8000, endpoint=False)
    fwd = 3e-4 * np.column_stack([np.cos(2 * np.pi * 20 * t), np.sin(2 * np.pi * 20 * t)])
    bwd = 1e-4 * np.column_stack([np.cos(2 * np.pi * 50 * t), -np.sin(2 * np.pi * 50 * t)])
    line = np.column_stack([2e-4 * np.cos(2 * np.pi * 80 * t), np.zeros_like(t)])
    freqs, amps = full_spectrum(t, fwd + bwd + line, points=8000)

    def at(f):
        return amps[np.argmin(np.abs(freqs - f))]

    assert at(20) == pytest.approx(3e-4, rel=0.02) and at(-20) < 1e-6
    assert at(-50) == pytest.approx(1e-4, rel=0.02) and at(50) < 1e-6
    assert at(80) == pytest.approx(1e-4, rel=0.02) and at(-80) == pytest.approx(1e-4, rel=0.02)
    assert full_spectrum(t[:3], fwd[:3])[0].size == 0


def test_log_decrement():
    assert log_decrement(0.0) == 0.0
    assert log_decrement(0.046) == pytest.approx(0.2893, abs=1e-4)
    assert log_decrement(np.array([-0.01]))[0] < 0
