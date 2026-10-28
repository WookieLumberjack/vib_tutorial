"""The Periodic forcing page's model (core/periodic.py): the series in closed form, and the exact steady state."""

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from vib_tutorial.core.periodic import (
    WAVES,
    Forcing,
    Wave,
    from_rest,
    periodic_steady,
    response_coefficients,
    series,
    series_error,
)
from vib_tutorial.core.sdof import Oscillator


@pytest.mark.parametrize("wave", [w for w in WAVES if w is not Wave.IMPULSES])
def test_coefficients_are_the_waves_fourier_series(wave):
    fo = Forcing(wave, 1.3, 2.0, 0.3)
    n = 2 ** 14
    t = np.arange(n) * fo.period / n
    f = fo.force(t + 0.5 * fo.period / n)  # midpoints: no sample on a jump
    fft = np.fft.rfft(f) / n
    C = fo.coefficients(9)
    shift = np.exp(-1j * np.pi * np.arange(10) / n)  # undo the half-sample offset
    expect = np.concatenate([[fft[0].real], 2 * fft[1:10]]) / shift
    assert C == pytest.approx(expect, abs=2e-3)


@pytest.mark.parametrize("wave", WAVES)
def test_the_steady_state_is_where_the_ode_ends_up(wave):
    """From rest, 11 periods (11 time constants) later the ODE runs on the exact steady state, and so does
    `from_rest`."""
    osc = Oscillator()
    fo = Forcing(wave, 0.1 if wave is Wave.IMPULSES else 1.0, 1.0, 0.25)
    t = np.linspace(11.0, 12.0, 501)
    x = periodic_steady(osc, fo, t)
    tt, xr = from_rest(osc, fo, 12.0)
    assert np.interp(t, tt, xr) == pytest.approx(x, abs=1e-3 * np.max(np.abs(x)))
    if wave is Wave.IMPULSES:
        return

    def rhs(s, z):
        return [z[1], (fo.force(np.array([s]))[0] - osc.damping * z[1] - osc.stiffness * z[0]) / osc.mass]

    sol = solve_ivp(rhs, (0, 12.0), [0.0, 0.0], t_eval=t, rtol=1e-10, atol=1e-13, max_step=2e-3)
    assert np.max(np.abs(sol.y[0] - x)) < 1e-3 * np.max(np.abs(x))


def test_series_converges_to_the_exact_steady_state():
    osc = Oscillator()
    for wave in (Wave.SQUARE, Wave.TRIANGLE, Wave.RECTIFIED, Wave.PULSES):
        fo = Forcing(wave)
        errs = [series_error(osc, fo, n) for n in (1, 5, 25, 125)]
        assert errs == sorted(errs, reverse=True) and errs[-1] < 2e-3
    fo = Forcing(Wave.SQUARE)
    t = np.linspace(0, 1, 4001)
    assert series(fo.coefficients(0), 1.0, t) == pytest.approx(0.0)
    X = response_coefficients(osc, Forcing(Wave.RECTIFIED), 3)
    assert X[0].real == pytest.approx(2 / math.pi / osc.stiffness)


def test_no_steady_state_on_an_undamped_resonance():
    osc = Oscillator(1.0, 400.0, 0.0)
    fo = Forcing(Wave.SQUARE, freq_hz=osc.natural_hz / 3)
    assert np.all(np.isnan(periodic_steady(osc, fo, np.array([0.0, 0.1]))))
