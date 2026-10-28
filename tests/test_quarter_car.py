"""Quarter-car: the transfer functions against a direct solve, the road's PSD, and the static limits."""

import numpy as np
import pytest

from vib_tutorial.core.quarter_car import (
    BAND,
    Car,
    Output,
    Road,
    road_profile,
    transfer,
)


def test_transfer_matches_a_direct_solve():
    car = Car()
    M, C, K = car.matrices()
    f = np.array([0.3, 1.2, 5.0, 11.8, 30.0])
    H = transfer(car, f)
    for i, fi in enumerate(f):
        w = 2 * np.pi * fi
        x = np.linalg.solve(K - w * w * M + 1j * w * C, [0.0, car.tyre])
        assert H[Output.BODY][i] == pytest.approx(-w * w * x[0])
        assert H[Output.TRAVEL][i] == pytest.approx(x[0] - x[1])
        assert H[Output.TYRE][i] == pytest.approx(car.tyre * (1 - x[1]))


def test_slow_road_is_followed():
    H = transfer(Car(), np.array([1e-4]))
    assert abs(H[Output.TRAVEL][0]) < 1e-6 and abs(H[Output.TYRE][0]) < 1e-3


def test_road_velocity_is_white_and_profile_has_the_psd():
    road = Road("B", 25.0)
    f = np.array([0.5, 2.0, 8.0])
    assert (2 * np.pi * f) ** 2 * road.psd(f) == pytest.approx(np.full(3, (2 * np.pi * f[0]) ** 2 * road.psd(f[:1])[0]))
    _, z = road_profile(road, 200.0, 200.0)
    lines = np.arange(1, 20001) / 200.0  # the record's lines, 1/duration apart
    lines = lines[(lines >= BAND[0]) & (lines <= BAND[1])]
    assert np.var(z) == pytest.approx(np.sum(road.psd(lines)) / 200.0, rel=1e-6)
