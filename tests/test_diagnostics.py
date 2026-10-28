import math

import numpy as np
import pytest

from vib_tutorial.core.diagnostics import (
    Bearing,
    BearingFault,
    Machine,
    RunRecord,
    RunUp,
    amplitude_spectrum,
    cepstrum,
    envelope_spectrum,
    impacts,
    line_amplitude,
    mode_filter,
    order_map,
    order_track,
    ring_filter,
    run_up,
)


def test_6205_defect_frequencies():
    f = Bearing().frequencies(1.0)  # per unit shaft speed
    assert (f["FTF"], f["BPFO"], f["BPFI"], f["BSF"]) == pytest.approx((0.3983, 3.5848, 5.4152, 2.3568), abs=1e-4)
    assert f["BPFO"] + f["BPFI"] == pytest.approx(Bearing().balls)


def test_ring_filter_is_the_sampled_impulse_response():
    fs, fn, zeta = 20480.0, 3000.0, 0.05
    b, a = ring_filter(fn, zeta, fs)
    x = np.zeros(400)
    x[0] = 1.0
    import scipy.signal

    y = scipy.signal.lfilter(b, a, x)
    t = np.arange(400) / fs
    wn = 2 * math.pi * fn
    wd = wn * math.sqrt(1 - zeta ** 2)
    exact = np.exp(-zeta * wn * t) * np.sin(wd * t)
    scale = y[1] / exact[1]
    assert y[1:] == pytest.approx(scale * exact[1:], abs=1e-12)
    fine = np.linspace(0, 1e-3, 20001)
    assert scale * np.max(np.exp(-zeta * wn * fine) * np.sin(wd * fine)) == pytest.approx(1.0, rel=1e-6)


def test_mode_filter_reaches_the_steady_state():
    import scipy.signal

    run = RunUp(mode=40.0, zeta=0.03, misalignment=1.0)
    fs = 2048.0
    b, a = mode_filter(run.mode, run.zeta, fs)
    for f in (15.0, 39.0, 90.0):
        t = np.arange(int(20 * fs)) / fs
        y = scipy.signal.lfilter(b, a, np.cos(2 * math.pi * f * t))
        tail = y[-int(2 * fs):]
        assert np.max(np.abs(tail)) == pytest.approx(float(run.steady(2, f / 2)), rel=2e-3)


def test_an_order_stays_on_its_line_through_a_run_up():
    run = RunUp(start=5.0, end=60.0, duration=10.0)
    fs = 2048.0
    t = np.arange(int(run.duration * fs) + 1) / fs
    theta = run.angle(t)
    rec = RunRecord(t, 0.7 * np.cos(3 * theta + 0.4), theta, run.speed(t), fs)
    om = order_map(rec)
    assert order_track(om, 3) == pytest.approx(0.7, rel=1e-3)
    assert np.max(order_track(om, 2)) < 1e-3


def test_slow_run_up_tracks_the_steady_state():
    run = RunUp(duration=60.0, noise=0.0)
    om = order_map(run_up(run))
    away = np.abs(om.speed - run.mode) > 8
    ratio = order_track(om, 1)[away] / run.steady(1, om.speed[away])
    assert ratio == pytest.approx(1.0, abs=0.01)


def test_flat_top_reads_a_tone_between_lines():
    fs = 20480.0
    t = np.arange(81920) / fs
    s = amplitude_spectrum(0.8 * np.sin(2 * math.pi * 1001.3 * t), fs)
    assert line_amplitude(s.freqs, s.amplitude, 1001.3) == pytest.approx(0.8, rel=0.01)


def test_envelope_of_a_modulated_carrier():
    fs = 20480.0
    t = np.arange(81920) / fs
    x = (1 + 0.3 * np.cos(2 * math.pi * 37.0 * t)) * np.cos(2 * math.pi * 3000.0 * t)
    s = envelope_spectrum(x, fs, (2500.0, 3500.0))
    assert line_amplitude(s.freqs, s.amplitude, 37.0) == pytest.approx(0.3, rel=0.02)


def test_cepstrum_finds_a_comb_spacing():
    fs = 20480.0
    t = np.arange(81920) / fs
    x = sum(np.cos(2 * math.pi * (500 + 25 * k) * t) for k in range(-8, 9))
    x = x + 0.01 * np.random.default_rng(1).standard_normal(t.size)
    q, c = cepstrum(x, fs)
    keep = q > 0.002
    assert q[keep][np.argmax(c[keep])] == pytest.approx(1 / 25, abs=2 / fs)


def test_impacts_come_at_the_defect_rate():
    m = Machine(bearing_fault=BearingFault.OUTER, slip=0.0)
    fs, n = 20480.0, 81920
    hits = np.flatnonzero(impacts(m, n, fs, np.random.default_rng(1)))
    assert np.mean(np.diff(hits)) / fs == pytest.approx(1 / m.defect_hz(), rel=1e-3)
