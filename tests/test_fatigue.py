"""The Fatigue page's model (core/fatigue.py): rainflow, the moments and the spectral damage estimates."""

import math

import numpy as np
import pytest

from vib_tutorial.core.fatigue import (
    Curve,
    Stress,
    dirlik_damage,
    dirlik_range_pdf,
    miner,
    narrow_band_damage,
    rainflow,
    rates,
    rayleigh_range_pdf,
    synthesize,
)


def test_rainflow_on_the_astm_example():
    # ASTM E1049-85 fig. 6: −2, 1, −3, 5, −1, 3, −4, 4, −2
    ranges, counts = rainflow(np.array([-2, 1, -3, 5, -1, 3, -4, 4, -2], dtype=float))
    got = sorted(zip(ranges, counts))
    want = sorted([(3, 0.5), (4, 0.5), (4, 1.0), (8, 0.5), (9, 0.5), (8, 0.5), (6, 0.5)])
    assert got == want


def test_a_sine_is_one_cycle_a_period():
    t = np.linspace(0, 10, 10001)
    ranges, counts = rainflow(3 * np.sin(2 * np.pi * t))
    assert counts[ranges > 5.99].sum() == pytest.approx(10, abs=0.5) and np.max(ranges) == pytest.approx(6, abs=1e-3)
    c = Curve()
    assert miner(ranges, counts, c) * c.cycles(np.array([6.0]))[0] == pytest.approx(10, abs=0.5)


def test_moments_and_record_agree():
    s = Stress(sigma2=10.0)
    r = rates(s)
    assert r.sigma == pytest.approx(math.sqrt(20 ** 2 + 10 ** 2), rel=1e-3)
    t, x = synthesize(s, 600.0, 3)
    assert np.std(x) == pytest.approx(r.sigma, rel=0.01)
    ups = np.count_nonzero((x[:-1] < 0) & (x[1:] >= 0)) / 600.0
    assert ups == pytest.approx(r.nu0, rel=0.03)
    d = np.diff(x)
    assert np.count_nonzero((d[:-1] > 0) & (d[1:] <= 0)) / 600.0 == pytest.approx(r.nup, rel=0.03)


@pytest.mark.parametrize("stress", [Stress(), Stress(sigma2=10.0), Stress(zeta1=0.3)])
def test_the_range_densities_integrate_to_the_damage(stress):
    c = Curve(b=4.0)
    S = np.linspace(0, 20 * rates(stress).sigma, 200001)
    for pdf in (rayleigh_range_pdf, dirlik_range_pdf):
        assert np.trapezoid(pdf(stress, S), S) == pytest.approx(1.0, abs=1e-6)
    r = rates(stress)
    assert dirlik_damage(stress, c) == pytest.approx(
        r.nup * np.trapezoid(S ** c.b * dirlik_range_pdf(stress, S), S) / c.K, rel=1e-6)
    assert narrow_band_damage(stress, c) == pytest.approx(
        r.nu0 * np.trapezoid(S ** c.b * rayleigh_range_pdf(stress, S), S) / c.K, rel=1e-6)
