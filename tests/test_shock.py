"""Shock response spectra: the oscillators against closed forms, the chain on a shaken base, and the trials' notes."""

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from vib_tutorial.core.modal import modal_analysis
from vib_tutorial.core.model import ChainSystem
from vib_tutorial.core.random_vibration import G
from vib_tutorial.core.shock import (
    CLASSICAL,
    Pulse,
    Response,
    Rule,
    Shape,
    bank,
    chain_peaks,
    chain_response,
    classical,
    oscillator,
    record,
    sample_rate,
    srs,
    target,
)

UNIFORM = ChainSystem.uniform(4)


def _spectrum(pulse, zeta, result=None, system=None):
    f = bank(pulse, result)
    rec = record(pulse, sample_rate(pulse, f[-1]), system, result)
    return rec, srs(rec, f, zeta)


def test_undamped_half_sine_and_square():
    """Against the undamped oscillator's response in closed form, during the pulse and after it."""
    T, A = 0.1, G
    rec, sp = _spectrum(Pulse(Shape.HALF_SINE, 1.0, T), 0.0)
    assert rec.duration == pytest.approx(T) and rec.velocity_change == pytest.approx(2 * A * T / math.pi, rel=1e-4)
    big = 0.0
    for i in range(0, sp.f.size, 10):
        w, Om = 2 * math.pi * sp.f[i], math.pi / T
        t = np.linspace(0.0, T, 100_001)
        z = -A / (w * w - Om * Om) * (np.sin(Om * t) - Om / w * np.sin(w * t))
        zd = -A / (w * w - Om * Om) * (Om * np.cos(Om * T) - Om * np.cos(w * T))
        y = -w * w * z
        assert sp.primary[0, i] == pytest.approx(y.max(), rel=3e-3, abs=3e-3 * A)
        assert sp.primary[1, i] == pytest.approx(max(-y.min(), 0.0), abs=3e-3 * A)
        assert sp.residual[0, i] == pytest.approx(w * w * math.hypot(z[-1], zd / w), rel=3e-3, abs=3e-3 * A)
        big = max(big, sp.maximax[i])
    assert sp.maximax.max() / A == pytest.approx(1.768, abs=1e-3)  # the half-sine's undamped peak
    k = int(np.argmax(sp.maximax))
    assert sp.f[k] * T == pytest.approx(0.81, abs=0.02)
    assert sp.maximax[0] == pytest.approx(2 * math.pi * sp.f[0] * rec.velocity_change, rel=2e-3)  # an impulse
    assert sp.maximax[-1] / A == pytest.approx(1.0, abs=0.03)  # stiff: follows the pulse (undamped, ringing a little)
    assert sp.pseudo_velocity == pytest.approx(2 * math.pi * sp.f * sp.displacement)

    _, sq = _spectrum(Pulse(Shape.SQUARE, 1.0, T), 0.0)
    f = sq.f
    stiff = f * T > 1.0
    assert sq.primary[0, stiff] / A == pytest.approx(np.full(stiff.sum(), 2.0), abs=0.01)  # a step's overshoot
    soft = f * T < 1
    assert sq.residual[0, soft] == pytest.approx(2 * A * np.abs(np.sin(math.pi * f * T))[soft],
                                                 abs=0.02 * A)  # undamped: 2A |sin(ωT/2)|
    # in general, undamped: ω |∫a e^{−iωt} dt|, the record's input linear between samples (its jump up a ramp)
    rec = record(Pulse(Shape.SQUARE, 1.0, T), sample_rate(Pulse(Shape.SQUARE, 1.0, T), f[-1]))
    t = np.linspace(-rec.h, rec.duration, 50 * rec.a.size)
    a = np.interp(t, np.concatenate([[-rec.h], rec.t]), np.concatenate([[0.0], rec.a]))
    w = 2 * math.pi * f
    res = w * np.abs(np.trapezoid(a[None, :] * np.exp(-1j * w[:, None] * t[None, :]), t, axis=1))
    assert sq.residual.max(axis=0) == pytest.approx(res, rel=1e-3, abs=1e-3 * A)


def test_oscillator_and_damping():
    """One oscillator's response peaks at its spectrum's value; damping eases the residual's tail."""
    pulse = Pulse(Shape.HALF_SINE, 10.0, 0.1)
    rec, sp = _spectrum(pulse, 0.05)
    for i in (0, 60, 120, 200):
        osc = oscillator(rec, sp.f[i], 0.05, 2.0 / sp.f[i])
        assert np.max(np.abs(osc.acceleration)) == pytest.approx(sp.maximax[i], rel=2e-3)
        assert np.max(np.abs(osc.displacement)) == pytest.approx(sp.displacement[i], rel=2e-3)
    # the shapes, each peak 1
    s = np.linspace(0.0, 1.0, 1001)
    for shape in CLASSICAL:
        assert np.max(classical(shape, s)) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        srs(rec, sp.f, 1.0)


def test_chain_on_a_shaken_base():
    """chain_response against solve_ivp: relative displacements and absolute accelerations under a base pulse."""
    system = ChainSystem(np.array([1.0, 2.0, 0.5]), np.array([400.0, 300.0, 200.0]), np.array([2.0, 0.0, 5.0]))
    M, C, K = system.matrices()
    T, fs = 0.2, 2000.0
    t = np.arange(int(1.0 * fs) + 1) / fs
    a = np.where(t <= T, 9.0 * np.sin(np.pi * np.minimum(t, T) / T), 0.0)
    resp = chain_response(system, a, fs)

    def rhs(tt, s):
        ag = 9.0 * math.sin(math.pi * tt / T) if tt <= T else 0.0
        x, v = s[:3], s[3:]
        return np.concatenate([v, np.linalg.solve(M, -C @ v - K @ x) - ag])

    sol = solve_ivp(rhs, (0, t[-1]), np.zeros(6), t_eval=t, rtol=1e-10, atol=1e-12, max_step=1e-3)
    x = sol.y[:3].T
    assert resp.displacement == pytest.approx(x, abs=2e-5 * np.abs(x).max())
    acc = -np.linalg.solve(M, C @ sol.y[3:] + K @ sol.y[:3]).T  # absolute: the springs' and dampers' pull
    assert resp.acceleration == pytest.approx(acc, abs=2e-4 * np.abs(acc).max())


def test_measured_records():
    """The transient starts and ends at rest; an offset is a velocity change; noise follows the seed."""
    r = modal_analysis(UNIFORM)
    clean = record(Pulse(Shape.TRANSIENT, 10.0, 0.1), 80_000.0)
    assert np.max(np.abs(clean.a)) == pytest.approx(10.0 * G)
    assert abs(clean.velocity_change) < 5e-4 * np.max(np.abs(clean.velocity()))
    shifted = record(Pulse(Shape.TRANSIENT, 10.0, 0.1, offset=0.01), 80_000.0)
    assert shifted.velocity_change - clean.velocity_change == pytest.approx(0.01 * 10.0 * G * 0.1, rel=1e-3)
    one, two = (record(Pulse(Shape.TRANSIENT, 10.0, 0.1, noise=0.02, seed=s), 80_000.0) for s in (1, 2))
    assert np.std(one.a - clean.a) == pytest.approx(0.02 * 10.0 * G, rel=0.05)
    assert not np.allclose(one.a, two.a)
    at = record(Pulse(Shape.AT_A_MASS, 10.0, 0.1, mass=-1), 4000.0, UNIFORM, r)
    assert at.end is None and at.duration == pytest.approx(0.1 + 4.0 / r.modes[0].fn_hz, abs=1e-3)
    with pytest.raises(ValueError):
        record(Pulse(Shape.AT_A_MASS), 4000.0)


def test_synthesis():
    """Both kinds meet the target within 3 dB over the bank, at any scale; wavelets end at rest, damped sines don't;
    the delays follow the seed."""
    for shape in (Shape.SINES, Shape.WAVELETS):
        for knee, seed in ((5.0, 1), (5.0, 2), (300.0, 1)):
            pulse = Pulse(shape, 10.0, knee=knee, seed=seed)
            rec, sp = _spectrum(pulse, 0.05)
            goal = target(pulse, sp.f)
            inside = np.isfinite(goal)
            assert inside.sum() > 50 and not inside[0] and not inside[-1]
            assert np.max(np.abs(20 * np.log10(sp.maximax[inside] / goal[inside]))) < 3.0
            assert rec.synthesis.f.size == 28 and rec.synthesis.worst < 1.5
            v = np.max(np.abs(rec.velocity()))
            if shape is Shape.WAVELETS:
                assert abs(rec.velocity_change) < 1e-6 * v
                assert abs(rec.displacement()[-1]) < 1e-5 * np.max(np.abs(rec.displacement()))
            else:
                assert abs(rec.velocity_change) > 0.1 * v
    one, two = (record(Pulse(Shape.WAVELETS, seed=s), 2000.0) for s in (1, 2))
    assert not np.allclose(one.synthesis.delay, two.synthesis.delay)
    # the record's length comes from the band, not the duration
    assert record(Pulse(Shape.WAVELETS, duration=1.0), 2000.0).a.size == one.a.size


def test_chain_peaks():
    """The modes' terms add up to the exact response with proportional damping (accelerations and spring forces);
    ABS bounds it; without proportional damping the sum is only near; a chain on a free end skips its rigid mode."""
    r = modal_analysis(UNIFORM)
    pulse = Pulse(Shape.HALF_SINE, 10.0, 0.1)
    rec, _ = _spectrum(pulse, 0.05, r, UNIFORM)
    cp = chain_peaks(UNIFORM, rec, r)
    for response in Response:
        exact = cp.exact[response]
        for j in range(UNIFORM.n):
            total = cp.contributions(response, j).sum(axis=0)
            assert total == pytest.approx(exact[:, j], abs=1e-8 * np.abs(exact).max())
        assert np.all(cp.estimate(response, Rule.ABS) >= cp.peak(response))
        assert np.all(cp.estimate(response, Rule.SRSS) <= cp.estimate(response, Rule.NRL))
        assert np.all(cp.estimate(response, Rule.NRL) <= cp.estimate(response, Rule.ABS))
    assert cp.effective.sum() == pytest.approx(1.0)
    # each mode's term peaks where the SRS says, at its own f and ζ
    peaks = np.max(np.abs(cp.modal[Response.ACCELERATION]), axis=1)
    assert peaks == pytest.approx(cp.spectrum[Response.ACCELERATION], rel=2e-3)

    skew = ChainSystem(np.array([1.0, 2.0, 0.5]), np.array([400.0, 300.0, 200.0]), np.array([8.0, 0.0, 0.0]))
    rs = modal_analysis(skew)
    rec, _ = _spectrum(pulse, 0.05, rs, skew)
    cp = chain_peaks(skew, rec, rs)
    total = cp.contributions(Response.ACCELERATION, 2).sum(axis=0)
    exact = cp.exact[Response.ACCELERATION][:, 2]
    assert 1e-4 < np.max(np.abs(total - exact)) / np.max(np.abs(exact)) < 0.3

    free = ChainSystem(np.array([1.0, 1.0]), np.array([0.0, 400.0]), np.array([0.0, 2.0]))
    rf = modal_analysis(free)
    rec, _ = _spectrum(pulse, 0.05, rf, free)
    cp = chain_peaks(free, rec, rf)
    assert cp.f.size == 1 and np.all(np.isfinite(cp.estimate(Response.SPRING, Rule.NRL)))


def test_pseudo_velocity():
    """The undamped pseudo-velocity tends to |Δv| at the left and to A / ω at the right."""
    rec, sp = _spectrum(Pulse(Shape.HALF_SINE, 1.0, 0.1), 0.0)
    pv = sp.pseudo_velocity
    assert pv[0] == pytest.approx(rec.velocity_change, rel=2e-3)
    assert pv[-1] == pytest.approx(G / (2 * math.pi * sp.f[-1]), rel=0.03)


def test_the_trials_notes():
    from vib_tutorial.gui.shock import TRIALS, duration_for

    r = modal_analysis(UNIFORM)
    T = duration_for(r)
    assert T == pytest.approx(0.1)
    note = {name: text for name, _, text in TRIALS}

    def run(name):
        changes = next(c for n, c, _ in TRIALS if n == name)
        pulse = Pulse(changes.get("shape", Shape.HALF_SINE), 10.0, T, -1, changes.get("noise", 0.0),
                      changes.get("offset", 0.0), 1)
        rec, sp = _spectrum(pulse, 1.0 / (2.0 * changes.get("q", 10.0)), r, UNIFORM)
        return rec, sp, float(np.max(np.abs(rec.a)))

    def peak(sp, top):
        k = int(np.argmax(sp.maximax))
        return sp.maximax[k] / top, sp.f[k] * T

    rec, sp, top = run("Half-sine")
    assert peak(sp, top) == pytest.approx((1.65, 0.83), abs=0.005)
    assert "1.65 ×" in note["Half-sine"] and "f T = 0.83" in note["Half-sine"]
    _, sq, top = run("Square")
    assert sq.maximax[-1] / top == pytest.approx(1.85, abs=0.01)
    assert rec.velocity_change < record(Pulse(Shape.SQUARE, 10.0, T), rec.fs).velocity_change
    _, tp, top = run("Terminal-peak sawtooth")
    stiff = tp.f * T > 2
    ratio = tp.negative[stiff] / tp.positive[stiff]
    assert ratio.min() >= 0.845 and ratio.max() < 0.865 and "0.85–0.86 ×" in note["Terminal-peak sawtooth"]
    _, hv, top = run("Haversine")
    assert peak(hv, top) == pytest.approx((1.60, 1.0), abs=0.02) and hv.negative[-1] / top < 0.002
    _, q50, top = run("Light damping, Q = 50")
    assert peak(q50, top)[0] == pytest.approx(1.74, abs=0.005)
    rec, at, top = run("A component on the last mass")
    assert at.at(r.modes[0].fn_hz) / top == pytest.approx(3.3, abs=0.05)
    assert peak(at, top)[1] / T == pytest.approx(r.modes[0].fn_hz, rel=0.03)
    _, pyro, top = run("Pyrotechnic transient")
    assert peak(pyro, top) == pytest.approx((4.5, 38), rel=0.02)
    low = pyro.maximax[0] / top
    assert 0.5e-4 < low < 1.5e-4  # 10⁻⁴
    rec, shift, top = run("Zero shift")
    assert shift.maximax[0] / top / low == pytest.approx(10, rel=0.15)
    dv = abs(rec.velocity_change)
    assert shift.maximax[0] == pytest.approx(2 * math.pi * shift.f[0] * dv, rel=0.1)  # on the line

    rec, sp, top = run("Modes combined")
    cp = chain_peaks(UNIFORM, rec, r)
    a = Response.ACCELERATION
    exact = cp.peak(a)[3] / G
    estimates = {rule: cp.estimate(a, rule)[3] / G for rule in Rule}
    text = note["Modes combined"]
    assert f"exactly {exact:.3g} g" in text
    for rule, value in estimates.items():
        quoted = f"{rule.value} (their sum) " if rule is Rule.ABS else f"{rule.value} "
        assert quoted + f"{value:.3g} g, {100 * (value / exact - 1):+.0f} %".replace("-", "−") in text
    when = cp.t[np.argmax(np.abs(cp.modal[a]), axis=1)]
    assert when.max() < 0.3 and 0.3 == pytest.approx(1 / (3 * r.modes[0].fn_hz), rel=0.02)
    assert "within the first 0.3 s, a third of mode 1's period" in text

    rec, sp, top = run("Four-way paper")
    assert rec.velocity_change == pytest.approx(2 * top * T / math.pi, rel=1e-4)
    assert "6.24 m/s" in note["Four-way paper"]
    assert sp.displacement[0] == pytest.approx(4.6, abs=0.05) and "4.6 m" in note["Four-way paper"]

    from vib_tutorial.gui.shock import shape_spectra

    same_peak, same_dv = shape_spectra(0.05, False), shape_spectra(0.05, True)
    square = same_peak[Shape.SQUARE][2]
    assert square[0] == pytest.approx(0.12, abs=0.005) and square.max() == pytest.approx(1.86, abs=0.005)
    assert all(square[0] >= v[0] and square.max() >= v.max() for _, _, v in same_peak.values())
    assert same_peak[Shape.TERMINAL_PEAK][2].max() == pytest.approx(1.19, abs=0.005)
    lows = [v[0] for _, _, v in same_dv.values()]
    assert max(lows) == pytest.approx(min(lows), rel=2e-3)
    assert same_dv[Shape.INITIAL_PEAK][2].max() == pytest.approx(2.33, abs=0.005)
    assert same_dv[Shape.SQUARE][2].max() == pytest.approx(1.18, abs=0.005)
    assert max(same_dv.values(), key=lambda s: s[2][-1]) is same_dv[Shape.INITIAL_PEAK]
    for text in ("0.12 ×", "1.86 ×", "1.19 ×", "2.33 ×", "1.18 ×"):
        assert text in note["Pulse shapes compared"]

    from vib_tutorial.gui.shock import knee_for

    assert knee_for(r) == 5.0 and "5 Hz knee" in note["Damped sines to a spectrum"]
    rec, sp, top = run("Damped sines to a spectrum")
    assert rec.synthesis.f.size == 28 and "28 damped sines" in note["Damped sines to a spectrum"]
    assert rec.velocity_change == pytest.approx(0.51, abs=0.005) and rec.displacement()[-1] == pytest.approx(6, abs=0.1)
    assert top / G == pytest.approx(4.8, abs=0.05) and "only 4.8 g" in note["Damped sines to a spectrum"]
    rec, sp, top = run("Wavelets to a spectrum")
    assert rec.synthesis.f.size == 28 and abs(rec.displacement()[-1]) < 1e-6  # within a micrometre
    assert np.max(np.abs(rec.displacement())) == pytest.approx(0.54, abs=0.005) and "0.54 m" in note[
        "Wavelets to a spectrum"]
