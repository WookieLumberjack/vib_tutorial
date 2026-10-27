"""Seismic response spectrum: effective masses, the rules, CQC's correlation, and the records against the spectrum."""

import numpy as np
import pytest

from vib_tutorial.core.random_vibration import G
from vib_tutorial.core.seismic import (
    Design,
    Rule,
    accelerogram,
    building,
    combine,
    correlation,
    design_sa,
    modal,
    time_history,
)


def test_effective_masses_add_up_to_the_mass():
    m = modal(building(6), Design())
    assert m.effective_mass.sum() == pytest.approx(m.total_mass)
    assert np.sum(m.base_shear) == pytest.approx(np.sum(m.effective_mass * m.sa))


def test_spectrum_corners():
    d = Design()
    S, TB, TC, TD = d.corners
    plateau = 2.5 * 0.3 * G * S
    assert design_sa(d, np.array([0.0, TB, TC, 2 * TC, 2 * TD])) == pytest.approx(
        [0.3 * G * S, plateau, plateau, plateau / 2, plateau * TC * TD / (4 * TD * TD)])


def test_rules_agree_for_one_mode_and_order():
    peaks = np.array([[3.0, -4.0, 1.0]])
    rho = correlation(np.array([1.0, 3.0, 9.0]), 0.05)
    assert combine(peaks, Rule.SRSS)[0] == pytest.approx(np.sqrt(26))
    assert combine(peaks, Rule.ABS)[0] == 8
    assert combine(peaks, Rule.NRL)[0] == pytest.approx(4 + np.sqrt(10))
    assert combine(peaks, Rule.CQC, rho)[0] == pytest.approx(np.sqrt(26), rel=0.01)
    assert combine(peaks, Rule.CQC, np.ones((3, 3)))[0] == pytest.approx(0.0)  # fully correlated, signs kept


def test_correlation_is_one_on_the_diagonal_and_symmetric_for_equal_damping():
    rho = correlation(np.array([1.0, 1.1, 2.0]), 0.05)
    assert np.diag(rho) == pytest.approx(1.0)
    assert rho == pytest.approx(rho.T)


def test_the_record_matches_the_spectrum_and_one_mode_is_exact():
    d = Design()
    g = accelerogram(d, 3)
    err = g.spectrum / design_sa(d, 1 / g.f) - 1
    assert np.median(np.abs(err)) < 0.05
    one = building(1)  # one storey: its peak displacement is the record's spectral displacement at T
    h = time_history(one, d, g)
    m = modal(one, d)
    from vib_tutorial.core.shock import srs

    sd = srs(g.record, m.f, d.zeta).displacement[0]
    assert h.peak_displacement[0] == pytest.approx(sd, rel=3e-3)  # sampled peaks; the SRS refines its own
