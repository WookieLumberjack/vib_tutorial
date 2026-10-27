"""Beams & plates: Timoshenko's roots and limits, the plates' closed forms, nodal lines and modal density."""

import math

import numpy as np
import pytest

from vib_tutorial.core.beams_plates import (
    Beam,
    Plate,
    Theory,
    beam_frequencies,
    beam_omega,
    circular_modes,
    circular_shape,
    counting,
    plate_asymptote,
    rect_modes,
    rect_shape,
    timoshenko_determinant,
)


@pytest.mark.parametrize("branch", [0, 1])
def test_timoshenko_roots_zero_the_determinant(branch):
    b = Beam()
    for n in (1, 4, 17):
        k = n * math.pi / b.length
        assert timoshenko_determinant(b, k, float(beam_omega(b, k, branch=branch))) == pytest.approx(0, abs=1e-9)


def test_thin_beams_are_euler_bernoulli():
    b = Beam(depth=1e-4)
    e = beam_frequencies(b, 5, Theory.EULER)
    for th in Theory:
        assert beam_frequencies(b, 5, th) == pytest.approx(e, rel=1e-6)


def test_rayleigh_and_shear_are_timoshenkos_limits():
    stiff_shear = Beam(kappa=1e9)
    assert beam_frequencies(stiff_shear, 8) == pytest.approx(beam_frequencies(stiff_shear, 8, Theory.RAYLEIGH),
                                                             rel=1e-6)
    b = Beam()
    k = np.arange(1, 9) * math.pi
    # each effect lowers the frequencies, and the two together lower them most
    t, s, r, e = (beam_omega(b, k, th) for th in (Theory.TIMOSHENKO, Theory.SHEAR, Theory.RAYLEIGH, Theory.EULER))
    assert np.all(t < s) and np.all(t < r) and np.all(s < e) and np.all(r < e)


def test_plate_frequencies_follow_the_formula():
    p = Plate(0.5, 0.3, 0.002)
    m = rect_modes(p, 1)[0]
    w = math.pi ** 2 * (1 / 0.5 ** 2 + 1 / 0.3 ** 2) * math.sqrt(p.rigidity / p.surface_mass)
    assert m.freq_hz == pytest.approx(w / (2 * math.pi))


def test_nodal_lines():
    p = Plate(0.6, 0.4)
    mode = rect_modes(p, 6)[5]  # (3, 2)
    x = np.linspace(0, p.a, 601)[1:-1]
    w = rect_shape(mode, p, x, np.array([0.13]))[0]
    assert np.count_nonzero(np.diff(np.sign(w))) == mode.m - 1
    c = Plate(0.2, 0.2, circular=True)
    two = circular_modes(c, 4)[3]  # (0, 2): one nodal circle
    r = np.linspace(0, 0.999, 500)
    w = circular_shape(two, r, np.array([0.0]))[0]
    assert np.count_nonzero(np.diff(np.sign(w))) == 1


def test_plate_count_approaches_its_asymptote():
    p = Plate()
    modes = rect_modes(p, 3000)
    f = np.array([modes[-1].freq_hz * 0.8])
    n = counting(np.array([m.freq_hz for m in modes]), f)
    assert n[0] / plate_asymptote(p, f)[0] == pytest.approx(1, abs=0.05)
