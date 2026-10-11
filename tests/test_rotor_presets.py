"""The rotor page's presets: each shows what its note says."""

import math

import pytest

from vib_tutorial.core.rotor import RPM, campbell, whirl_modes
from vib_tutorial.core.rotor_presets import rotor_presets
from vib_tutorial.core.rotor_stability import least_damped, stability_onset

PRESETS = {p.name: p for p in rotor_presets()}


def criticals(name: str, forward: bool | None = None) -> list[int]:
    p = PRESETS[name]
    crits = campbell(p.system, p.top_rpm / RPM, n_modes=8).criticals
    return [round(w * RPM) for w, r, z in crits if z < 0.3 and (forward is None or (r > 0.1) == forward)]


def test_every_preset_is_a_valid_rotor_with_its_target_in_range():
    assert len(PRESETS) == 9
    for p in PRESETS.values():
        assert 0 < p.target_rpm < p.top_rpm and p.note


def test_classic_and_off_centre():
    assert criticals("Classic Jeffcott rotor", forward=True) == [1584]
    assert criticals("Off-centre disc") == [1731, 1737]
    s = PRESETS["Off-centre disc"].system
    assert round(whirl_modes(s, 0.0).freq_hz[-1], -1) == 360
    assert [round(f) for f in whirl_modes(s, 6000.0 / RPM).freq_hz[-2:]] == [289, 465]


def test_anisotropic_supports_split_the_critical():
    assert criticals("Anisotropic supports") == [1365, 1668]


def test_overhung_rotor():
    s = PRESETS["Overhung rotor"].system
    assert s.overhang == pytest.approx(0.1)
    assert round(whirl_modes(s, 0.0).freq_hz[0] * 60) == 870
    assert criticals("Overhung rotor", forward=False)[0] == 664
    assert criticals("Overhung rotor", forward=True) == [1284]
    assert abs(s.mass_unbalance) / s.disc_mass == pytest.approx(20e-6)  # e = 20 µm, as the classic rotor's


def test_fluid_film_bearings_whip_above_twice_the_natural_frequency():
    p = PRESETS["Oil swirl λ = ½ (oil whip)"]
    onset = stability_onset(p.system, p.top_rpm / RPM)
    assert round(onset.omega * RPM) == 3136 and round(onset.freq_hz) == 26
    assert onset.order == pytest.approx(0.5, rel=1e-3)
    assert least_damped(p.system, p.target_rpm / RPM)[0] > 0  # it starts out stable
    zeta, f, _ = least_damped(p.system, 5000.0 / RPM)
    assert f == pytest.approx(26.3, abs=0.05) and 1 / (-zeta * 2 * math.pi * f) == pytest.approx(0.22, abs=0.01)


def test_long_shaft_needs_its_mass():
    s = PRESETS["Long shaft, with its mass"].system
    assert s.shaft_mass == pytest.approx(2.47, abs=0.005) and s.shaft_mass > s.disc_mass
    assert criticals("Long shaft, with its mass", forward=True) == [1589, 6582]
    massless = campbell(s.with_(density=0.0), 10_000.0 / RPM, n_modes=8).criticals
    assert [round(w * RPM) for w, r, z in massless if z < 0.3 and r > 0.1] == [2146]


def test_two_pole_rotor():
    p = PRESETS["Two-pole rotor (asymmetric shaft)"]
    assert p.system.asymmetry == 0.15 and p.system.gravity > 0
    assert criticals("Two-pole rotor (asymmetric shaft)", forward=True) == [1491, 1665]
    band = stability_onset(p.system, p.top_rpm / RPM)
    assert (round(band.omega * RPM), round(band.end * RPM)) == (1527, 1631)
    assert least_damped(p.system, p.target_rpm / RPM)[0] > 0  # it runs stable at the gravity critical


def test_cracked_shaft():
    import numpy as np

    from vib_tutorial.core.rotor_sweep import probe, steady_vectors

    p = PRESETS["Cracked shaft (breathing crack)"]
    assert p.system.crack.depth == 0.4 and p.system.gravity > 0
    assert criticals("Cracked shaft (breathing crack)", forward=True) == [1562, 1574]
    w = np.linspace(300.0, 2000.0, 17001) / RPM
    peaks = []
    for order in (1, 2, 3):
        v = np.abs(probe(steady_vectors(p.system, w, order)[:, 1], "orbit"))
        i = int(np.argmax(v))
        peaks.append((round(v[i] * 1e6, 1), round(w[i] * RPM)))
    assert round(peaks[0][0]) == 164 and peaks[1:] == [(37.6, 783), (18.6, 522)]
    assert p.target_rpm == 783.0
