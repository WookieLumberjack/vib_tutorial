"""Waterfall and cascade spectra of the Jeffcott rotor: block by block full spectra."""

import math

import numpy as np
import pytest

from vib_tutorial.core.rotor import RPM, RotorSimulator, whirl_modes
from vib_tutorial.core.rotor_presets import rotor_presets
from vib_tutorial.core.rotor_stability import whirl_lines
from vib_tutorial.core.rotor_sweep import station_motion
from vib_tutorial.core.rotor_waterfall import DAMPED, Waterfall


def whirls(t: np.ndarray) -> np.ndarray:
    """Two stations: one whirling forward (0.3 mm at 20 Hz), one backward (0.1 mm at 50 Hz)."""
    return np.column_stack([3e-4 * np.exp(2j * np.pi * 20 * t), 1e-4 * np.exp(-2j * np.pi * 50 * t)])


def at(wf: Waterfall, amps: np.ndarray, f: float) -> float:
    return float(amps[np.argmin(np.abs(wf.freqs - f))])


def test_blocks_resolve_forward_and_backward_whirl_whatever_the_feeding():
    t = np.linspace(0.0, 3.0, 12_001)
    omega = np.linspace(10.0, 40.0, t.size)
    whole = Waterfall(0.5, 100.0)
    assert whole.feed(t, whirls(t), omega) == 11  # (3 − 0.5) / 0.25 + 1
    assert whole.hop == 0.25 and whole.freqs.max() <= 100.0
    pieces = Waterfall(0.5, 100.0)
    rng = np.random.default_rng(1)
    cuts = np.sort(rng.choice(t.size, 40, replace=False))
    for part in np.split(np.arange(t.size), cuts):
        pieces.feed(t[part], whirls(t[part]), omega[part])
    assert len(pieces.blocks) == len(whole.blocks)
    for a, b in zip(whole.blocks, pieces.blocks):
        np.testing.assert_allclose(a.amps, b.amps, atol=1e-9)
        assert a.t == pytest.approx(b.t) and a.omega == pytest.approx(b.omega)
    b = whole.blocks[3]
    assert b.t == pytest.approx(1.0) and b.omega == pytest.approx(np.interp(1.0, t, omega), rel=1e-3)
    assert b.rise == pytest.approx(30.0 * 0.5 / 3.0, rel=1e-2)
    assert at(whole, b.amps[0], 20) == pytest.approx(3e-4, rel=0.01) and at(whole, b.amps[0], -20) < 1e-8
    assert at(whole, b.amps[1], -50) == pytest.approx(1e-4, rel=0.01) and at(whole, b.amps[1], 50) < 1e-8
    assert b.natural.size == 0  # no rotor given


def test_oldest_blocks_are_dropped_and_clear_starts_again():
    t = np.linspace(0.0, 5.0, 5001)
    wf = Waterfall(0.2, 50.0, max_blocks=5)
    wf.feed(t, whirls(t), np.zeros_like(t))
    assert len(wf.blocks) == 5 and wf.blocks[-1].t == pytest.approx(4.9)
    wf.clear()
    assert not wf.blocks
    wf.feed(t[:400] + 10.0, whirls(t[:400]), np.zeros(400))
    assert wf.blocks[0].t == pytest.approx(10.1)


def test_latest_ramp_is_the_last_unbroken_run_up_or_coast_down():
    # Up from 0 to 100 rad/s, steady, down to 50, steady, up to 80, steady.
    knots_t = [0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    knots_w = [0.0, 100.0, 100.0, 50.0, 50.0, 80.0, 80.0]
    t = np.arange(0.0, 7.0, 1e-3)
    omega = np.interp(t, knots_t, knots_w)
    wf = Waterfall(0.2, 50.0)
    wf.feed(t, whirls(t), omega)
    up = wf.latest_ramp(up=True)
    assert up and all(b.rise > 0 for b in up)
    assert up[0].t == pytest.approx(5.0, abs=0.15) and up[-1].t == pytest.approx(6.0, abs=0.15)
    down = wf.latest_ramp(up=False)
    assert down[0].t == pytest.approx(3.0, abs=0.15) and down[-1].t == pytest.approx(4.0, abs=0.15)
    assert [b.omega for b in down] == sorted((b.omega for b in down), reverse=True)
    assert Waterfall(0.2, 50.0).latest_ramp(up=True) == []


def test_a_tap_at_speed_rings_the_free_whirl_frequencies():
    """Overhung rotor at 2000 rpm, tapped: peaks at its forward and backward first-mode whirl and +1X."""
    s = next(p for p in rotor_presets() if p.name == "Overhung rotor").system
    w = 2000.0 / RPM
    sim = RotorSimulator(s)
    sim.omega = sim.target = w
    wf = Waterfall(0.5, 120.0, s)
    while sim.t < 3.0:  # past the start-up transient
        sim.advance(0.1)
    sim.tap(0.0, -0.005)
    while not wf.blocks:
        r = sim.advance(0.05)
        xy = station_motion(s, r.q)
        wf.feed(r.t, xy[..., 0] + 1j * xy[..., 1], r.omega)
    b = wf.blocks[0]
    lines = whirl_lines(whirl_modes(s, w), DAMPED)
    np.testing.assert_allclose(b.natural, lines)
    backward, forward = lines[0], lines[1]  # the first mode's, at about −6.6 and +24 Hz
    assert backward < 0 < forward < w / (2 * math.pi)
    disc = b.amps[1]
    peaks = [i for i in range(1, disc.size - 1) if disc[i] > disc[i - 1] and disc[i] >= disc[i + 1]]
    top = sorted(peaks, key=lambda i: -disc[i])[:3]
    found = sorted(float(wf.freqs[i]) for i in top)
    resolution = 1.0 / wf.block
    assert found[0] == pytest.approx(backward, abs=resolution)
    assert found[1] == pytest.approx(forward, abs=resolution)
    assert found[2] == pytest.approx(w / (2 * math.pi), abs=resolution)  # +1X, the unbalance
