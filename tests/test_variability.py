import math

import numpy as np
import pytest

from vib_tutorial.core.model import ChainSystem
from vib_tutorial.core.variability import (
    Ring,
    Scatter,
    amplification,
    forced,
    lognormal,
    monte_carlo,
    natural_hz,
    participation,
    ring_modes,
    sensitivity_sigma,
    sweep_grid,
    tuned_band,
    whitehead,
)


def test_lognormal_has_mean_one_and_the_cov():
    x = lognormal(np.random.default_rng(3), 0.2, 400_000)
    assert x.min() > 0
    assert x.mean() == pytest.approx(1.0, abs=2e-3)
    assert x.std() == pytest.approx(0.2, rel=1e-2)


def test_first_order_sigma_matches_monte_carlo_on_a_small_scatter():
    chain = ChainSystem([1.0, 2.0, 1.5], [300.0, 500.0, 200.0], [1.0, 1.0, 1.0])
    scatter = Scatter(0.01, 0.02, 0.0)
    mc = monte_carlo(chain, scatter, 4000, np.array([1.0]), 0, 0, seed=5)
    assert mc.freqs.std(axis=0) == pytest.approx(sensitivity_sigma(chain, scatter), rel=0.05)


def test_scattered_chains_keep_the_nominal_on_average():
    chain = ChainSystem.uniform(3)
    mc = monte_carlo(chain, Scatter(0.0, 0.03, 0.0), 2000, np.array([1.0]), 2, 2, seed=2)
    assert mc.freqs.mean(axis=0) == pytest.approx(natural_hz(chain)[0], rel=2e-3)


def test_tuned_ring_modes_are_waves_in_the_band():
    ring = Ring(blades=10, coupling=0.08)
    f, phi = ring_modes(ring, np.zeros(10))
    lo, hi = tuned_band(ring)
    assert (f[0], f[-1]) == pytest.approx((lo, hi), rel=1e-9)
    n = np.arange(6)  # nodal diameters
    expected = ring.blade_hz * np.sqrt(1 + 4 * ring.coupling * np.sin(math.pi * n / 10) ** 2)
    assert np.unique(f.round(9)) == pytest.approx(expected, rel=1e-9)
    assert participation(phi[:, :1]) == pytest.approx(10.0)


def test_participation_counts_blades():
    phi = np.zeros((6, 2))
    phi[2, 0] = 1.0
    phi[:3, 1] = 1.0
    assert participation(phi) == pytest.approx([1.0, 3.0])


def test_tuned_ring_answers_every_blade_alike():
    ring = Ring(blades=8)
    x = forced(ring, 2, sweep_grid(ring, 400), np.zeros(8))
    assert np.ptp(x, axis=1) == pytest.approx(0.0, abs=1e-12 * x.max())
    assert amplification(ring, 2) == pytest.approx(1.0)


def test_mistuned_amplification_stays_under_whitehead():
    for seed in range(1, 9):
        ring = Ring(coupling=0.02, mistuning=0.02, seed=seed)
        assert 1.0 <= amplification(ring, 3) < whitehead(ring.blades)


def test_deltas_have_zero_mean():
    d = Ring(mistuning=0.05, seed=4).deltas()
    assert d.mean() == pytest.approx(0.0, abs=1e-15)
