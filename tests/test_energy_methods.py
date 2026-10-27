"""The Energy methods page's model (core/energy_methods.py): the bounds, Rayleigh–Ritz, Holzer and Lagrange."""

import numpy as np
import pytest

from vib_tutorial.core.energy_methods import (
    BASES,
    SHAPES,
    dunkerley,
    exact,
    holzer,
    holzer_residual,
    holzer_roots,
    lagrange_matrices,
    rayleigh,
    rayleigh_ritz,
    ritz_basis,
    trial_shape,
)
from vib_tutorial.core.model import ChainSystem

rng = np.random.default_rng(4)
CHAINS = [ChainSystem.uniform(4)] + [
    ChainSystem(rng.uniform(0.3, 3.0, n), rng.uniform(100.0, 2000.0, n), np.zeros(n)) for n in (2, 3, 5, 7)]


@pytest.mark.parametrize("system", CHAINS)
def test_rayleigh_and_dunkerley_bracket_the_first_frequency(system):
    w, _ = exact(system)
    for shape in SHAPES:
        assert rayleigh(system, trial_shape(system, shape)) >= w[0] ** 2 * (1 - 1e-12)
    for _ in range(20):
        assert rayleigh(system, rng.normal(size=system.n)) >= w[0] ** 2 * (1 - 1e-12)
    assert dunkerley(system) <= w[0] ** 2
    assert 1 / dunkerley(system) == pytest.approx(np.sum(1 / w ** 2))  # the trace of K⁻¹M


@pytest.mark.parametrize("system", CHAINS)
def test_rayleigh_ritz_is_above_each_mode_and_exact_with_every_shape(system):
    w, _ = exact(system)
    for basis in BASES:
        previous = None
        for r in range(1, system.n + 1):
            wr, _shapes = rayleigh_ritz(system, ritz_basis(system, basis, r))
            assert np.all(wr >= w[:r] * (1 - 1e-9))
            if previous is not None:
                assert np.all(wr[:-1] <= previous * (1 + 1e-9))  # adding a shape never raises an estimate
            previous = wr
        assert wr == pytest.approx(w, rel=1e-8)


@pytest.mark.parametrize("system", CHAINS)
def test_holzer_finds_the_natural_frequencies(system):
    w, _ = exact(system)
    f = w / (2 * np.pi)
    assert holzer_roots(system, 1.1 * f[-1], 20000) == pytest.approx(f, rel=1e-7)
    M, _, K = system.matrices()
    for omega in (0.3 * w[0], 0.5 * (w[0] + w[-1]), 1.2 * w[-1]):  # −det(K − ω²M) / (k₂ … k_N)
        det = np.linalg.det(K - omega ** 2 * M)
        assert holzer_residual(system, np.array([omega]))[0] == pytest.approx(
            det / np.prod(system.stiffness[1:]), rel=1e-8)
        rows = holzer(system, omega)
        assert rows[-1].spring == pytest.approx(holzer_residual(system, np.array([omega]))[0])


@pytest.mark.parametrize("system", CHAINS)
def test_lagrange_gives_the_chains_matrices(system):
    M, K = lagrange_matrices(system)
    Mc, _, Kc = system.matrices()
    assert M == pytest.approx(Mc) and K == pytest.approx(Kc)
