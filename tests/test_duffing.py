"""The Duffing oscillator: harmonic balance, the time step, sweeps and their jumps, and the presets."""

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from vib_tutorial.core.duffing import (
    PRESETS,
    TWO_PI,
    Drive,
    Duffing,
    DuffingSimulator,
    Sweep,
    amplitudes,
    linear_response,
    measured_jump,
    response_curve,
    steady_phase,
)

PRESET = {p.name: p for p in PRESETS}


def run(sim: DuffingSimulator, until: float, chunk: float = 5.0) -> list:
    """Advance to `until` s (or until the mass escapes); every cycle on the way."""
    cycles = []
    while sim.t < until - 1e-9 and not sim.escaped:
        cycles += sim.advance(min(chunk, until - sim.t)).cycles
    return cycles


def sweep(osc: Duffing, lo: float, hi: float, time: float, mode: Sweep = Sweep.UP_DOWN) -> list:
    sim = DuffingSimulator(osc, Drive(start_hz=lo, end_hz=hi, sweep_time=time))
    sim.drive.start(mode)
    return run(sim, time * (2 if mode is Sweep.UP_DOWN else 1))


def test_curve_solves_the_harmonic_balance():
    """Every point of the curve has G(a^2) = F^2, and the stable flag is the sign of dG/ds."""
    for osc in (Duffing(), Duffing(damping=0.8, cubic=-2e4, force=1.25)):
        curve = response_curve(osc, 6.0)
        for f, a, stable in curve.pieces:
            w = TWO_PI * f
            G = a * a * ((osc.kappa(a) - osc.mass * w * w) ** 2 + (osc.damping * w) ** 2)
            assert G == pytest.approx(np.full_like(a, osc.force**2), rel=1e-6)
        assert [p[2] for p in curve.pieces] == [True, False, True]


def test_without_the_cubic_it_is_the_linear_response():
    osc = Duffing(cubic=0.0)
    curve = response_curve(osc, 6.0)
    assert len(curve.pieces) == 1 and not curve.folds
    f, a, stable = curve.pieces[0]
    assert stable and a == pytest.approx(linear_response(osc, f), rel=1e-6)
    assert curve.tip == pytest.approx(linear_response(osc, np.linspace(3.0, 3.4, 40001)).max(), rel=1e-6)


def test_folds_bound_the_band_of_three_amplitudes():
    osc = Duffing()
    curve = response_curve(osc, 6.0)
    lo, hi = curve.band()
    assert (round(lo, 2), round(hi, 2)) == (3.64, 4.45)
    for f, count in ((lo - 0.01, 1), (lo + 0.01, 3), (hi - 0.01, 3), (hi + 0.01, 1)):
        a, stable = amplitudes(osc, f)
        assert a.size == count
        if count == 3:
            assert list(stable) == [True, False, True]
    down, up = sorted(curve.folds, key=lambda fold: -fold.freq_hz)
    assert down.down and not up.down  # off the upper branch's end, and up from the lower's
    assert down.lands_at == pytest.approx(amplitudes(osc, hi + 1e-6)[0][0], rel=1e-3)
    assert up.lands_at == pytest.approx(amplitudes(osc, lo - 1e-6)[0][-1], rel=1e-3)


def test_backbone_is_the_free_frequency_at_each_amplitude():
    """With no damping and no drive the mass swings at the backbone's frequency (to harmonic balance's
    accuracy: a few tenths of a percent here, where the cubic doubles the stiffness at the peak)."""
    osc = Duffing(damping=0.0, force=0.0)
    a = 0.07
    sim = DuffingSimulator(osc, Drive(freq_hz=0.0))
    sim.set_state(a, 0.0)
    samples = sim.advance(4.0)
    up = np.flatnonzero((samples.x[:-1] < 0) & (samples.x[1:] >= 0))
    period = np.diff(samples.t[up]).mean()
    assert 1 / period == pytest.approx(osc.backbone_hz(a), rel=0.01)
    assert osc.backbone_hz(a) > 1.3 * osc.natural_hz


def test_step_matches_a_fine_integration(monkeypatch):
    """Second order: 0.15 % behind after 13 cycles at the page's step, a quarter of that at half the step."""
    from vib_tutorial.core import duffing

    osc = Duffing()

    def rhs(t, z):
        return [z[1], (osc.force * math.cos(TWO_PI * 4.2 * t) - osc.damping * z[1] - osc.restoring(z[0])) / osc.mass]

    errors = []
    for per_period in (duffing.STEPS_PER_PERIOD, 2 * duffing.STEPS_PER_PERIOD):
        monkeypatch.setattr(duffing, "STEPS_PER_PERIOD", per_period)
        sim = DuffingSimulator(osc, Drive(freq_hz=4.2))
        sim.set_state(0.06, -0.5)
        samples = sim.advance(3.0)
        ref = solve_ivp(rhs, (0.0, samples.t[-1]), [0.06, -0.5], t_eval=samples.t, rtol=1e-11, atol=1e-13)
        errors.append(np.abs(samples.x - ref.y[0]).max() / np.abs(ref.y[0]).max())
    assert errors[0] < 2e-3
    assert errors[1] == pytest.approx(errors[0] / 4, rel=0.1)


def test_undamped_free_swing_keeps_its_energy():
    osc = Duffing(damping=0.0, force=0.0)
    sim = DuffingSimulator(osc)
    sim.set_state(0.07, 0.0)
    e0 = sim.energy
    run(sim, 30.0)
    assert sim.energy == pytest.approx(e0, rel=1e-3)


def test_held_on_each_branch_it_stays_at_harmonic_balance():
    """At 4 Hz the lower and upper amplitudes are both steady states, to about half a percent; the
    middle one is not."""
    osc = Duffing()
    a, stable = amplitudes(osc, 4.0)
    assert list(stable) == [True, False, True]
    ends = []
    for start in a:
        sim = DuffingSimulator(osc, Drive(freq_hz=4.0))
        sim.start_on(start)
        ends.append(np.mean([c.amplitude for c in run(sim, 40.0)[-10:]]))
    assert ends[0] == pytest.approx(a[0], rel=0.01)
    assert ends[2] == pytest.approx(a[2], rel=0.01)
    assert min(abs(ends[1] - a[0]), abs(ends[1] - a[2])) < 0.02 * a[2]  # fell to one of the others
    sim = DuffingSimulator(osc, Drive(freq_hz=4.0))
    assert np.mean([c.amplitude for c in run(sim, 40.0)[-10:]]) == pytest.approx(a[0], rel=0.01)  # from rest


def test_steady_phase_lags_through_the_resonance():
    osc = Duffing(cubic=0.0)
    assert steady_phase(osc, 0.5, float(linear_response(osc, 0.5))) == pytest.approx(0.0, abs=0.02)
    assert steady_phase(osc, osc.natural_hz, 0.1) == pytest.approx(math.pi / 2)
    assert steady_phase(osc, 10.0, float(linear_response(osc, 10.0))) == pytest.approx(math.pi, abs=0.02)


def test_sweeps_jump_just_past_the_folds():
    """Each sweep leaves its branch at the fold, a little late (the slower the sweep, the less): the
    hardening spring falls off going up and jumps up going down; the softening one the other way round."""
    for name in ("Hardening spring: the jump", "Softening spring: the jump"):
        p = PRESET[name]
        cycles = sweep(p.osc, p.start_hz, p.end_hz, p.sweep_time)
        lo, hi = response_curve(p.osc, p.end_hz).band()
        up, down = measured_jump(cycles, "up"), measured_jump(cycles, "down")
        assert hi < up < hi + 0.3, name
        assert lo - 0.3 < down < lo, name
    linear = sweep(PRESET["Linear spring: no jump"].osc, 2.0, 6.0, 120.0)
    tops = [max(c.amplitude for c in linear if c.leg == leg) for leg in ("up", "down")]
    assert tops[0] == pytest.approx(tops[1], rel=0.05)  # no jump: up and down alike,
    assert 0.85 < tops[0] / 0.1 < 0.95  # a little short of the steady F / (c w_n): too quick to build up


def test_drive_legs():
    d = Drive(start_hz=2.0, end_hz=6.0, sweep_time=10.0)
    d.start(Sweep.UP_DOWN)
    assert d.freq_hz == 2.0 and d.leg == "up"
    phase = 0.0
    for _ in range(1000):
        phase += d.advance(0.01)
    assert d.leg == "down" and d.freq_hz == pytest.approx(6.0)
    assert phase == pytest.approx(TWO_PI * 40.0, rel=1e-9)  # the mean frequency, 4 Hz, for 10 s
    for _ in range(1000):
        d.advance(0.01)
    assert d.mode is Sweep.HOLD and d.leg == "hold" and d.freq_hz == pytest.approx(2.0)
    d.start(Sweep.DOWN)
    assert d.freq_hz == 6.0 and d.leg == "down"


def test_presets_show_what_their_notes_say():
    assert len(PRESET) == 6
    hard = response_curve(PRESET["Hardening spring: the jump"].osc, 6.0)
    assert round(PRESET["Hardening spring: the jump"].osc.natural_hz, 2) == 3.18
    assert [round(f, 2) for f in hard.band()] == [3.64, 4.45]
    soft = response_curve(PRESET["Softening spring: the jump"].osc, 4.5)
    assert [round(f, 2) for f in soft.band()] == [2.55, 2.97]

    p = PRESET["Escape from the well"]
    assert p.osc.escape == pytest.approx(0.1)
    assert math.isinf(response_curve(p.osc, p.end_hz).tip)  # the upper branch runs into the escape amplitude
    sim = DuffingSimulator(p.osc, Drive(freq_hz=p.freq_hz, start_hz=p.start_hz, end_hz=p.end_hz,
                                        sweep_time=p.sweep_time))
    sim.drive.start(p.mode)
    run(sim, p.sweep_time)
    assert sim.escaped and p.start_hz < sim.drive.freq_hz < 2.0

    p = PRESET["Chaos (Ueda)"]
    sim = DuffingSimulator(p.osc, Drive(freq_hz=p.freq_hz))
    section = np.array([(c.x, c.v) for c in run(sim, 600.0)[20:]])
    assert len(section) > 60
    assert np.ptp(section[:, 0]) > 0.5 and np.ptp(section[:, 1]) > 1.0  # the dots never settle onto one


def test_hardening_never_escapes_and_softening_needs_a_stiffness():
    sim = DuffingSimulator(Duffing(force=50.0), Drive(freq_hz=3.0))
    run(sim, 10.0)
    assert not sim.escaped and math.isinf(sim.osc.escape)
    with pytest.raises(ValueError):
        Duffing(stiffness=0.0, cubic=-1.0)
