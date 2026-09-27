import math

import numpy as np
import pytest

from vib_tutorial.core import ChainSystem, frf, modal_analysis, transmissibility
from vib_tutorial.core.presets import (
    den_hartog,
    modal_mass,
    presets,
    tuned_mass_damper,
    with_absorber,
    without_absorber,
)


def test_undamped_absorber_holds_the_primary_still_at_its_frequency():
    primary = ChainSystem([2.0], [800.0], [1.0])
    fa = 2.5
    s = with_absorber(primary, 0.2, fa)
    assert s.stiffness[1] == pytest.approx(0.2 * (2 * math.pi * fa) ** 2)
    H = frf(s, np.array([fa]), 0)[0]
    assert abs(H[0]) < 1e-12
    # The absorber's spring pushes back with exactly the applied force.
    assert s.stiffness[1] * abs(H[1]) == pytest.approx(1.0)


def test_den_hartog_equal_peaks_at_the_fixed_points():
    mu = 0.05
    ratio, zeta = den_hartog(mu)
    assert ratio == pytest.approx(1 / 1.05)
    s = tuned_mass_damper(ChainSystem([1.0], [400.0], [0.0]), mu)
    assert s.masses[1] == pytest.approx(0.05)
    f = np.linspace(2.0, 4.5, 20001)
    x1 = np.abs(frf(s, f, 0)[:, 0]) * 400.0  # dynamic amplification
    # Den Hartog's peak sqrt(1 + 2/mu) (his damping puts the peaks near, not exactly on, the fixed points).
    assert x1.max() == pytest.approx(math.sqrt(1 + 2 / mu), rel=0.03)
    fn = 400.0**0.5 / (2 * math.pi)
    lo, hi = x1[f < fn].max(), x1[f > fn].max()
    assert lo == pytest.approx(hi, rel=0.05)


def test_modal_mass_at_the_tip_of_a_uniform_chain():
    s = ChainSystem.uniform(4)
    phi = modal_analysis(s).modes[0].shape_mass_normalized
    assert modal_mass(s, 0, 3) == pytest.approx(1 / phi[3] ** 2)
    assert modal_mass(s, 0, 3) < s.masses.sum()  # only part of the chain moves with the tip


def test_presets_are_valid_and_the_absorbers_help():
    for p in presets():
        assert p.force.target < p.system.n
        if not p.absorber:
            continue
        before = without_absorber(p.system)
        assert before.n == p.system.n - 1
        f = np.array([p.force.freq_hz])
        dof = p.system.n - 2  # the mass the absorber hangs on
        if p.force.base:
            with_, without = transmissibility(p.system, f), transmissibility(before, f)
        else:
            with_, without = frf(p.system, f, p.force.target), frf(before, f, p.force.target)
        assert abs(with_[0, dof]) < 0.2 * abs(without[0, dof])


def test_without_absorber_of_one_mass():
    assert without_absorber(ChainSystem.uniform(1)) is None
