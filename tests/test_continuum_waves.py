"""A pulse on the continuum (waveguide, or a beam's modes) and on the mesh: transport, reflections, steps, the presets."""

import itertools

import numpy as np
import pytest

from vib_tutorial.core.continuum import Continuum, End, Kind, Scheme
from vib_tutorial.core.continuum_waves import (
    FRAMES_PER_TRANSIT,
    PLAYBACK,
    WAVE_PRESETS,
    WaveRun,
    Waves,
    end_reflection,
    pulse,
    step_coefficients,
)

PRESET = {p.name: p for p in WAVE_PRESETS}
STRING = Continuum(Kind.STRING, 1.0, End.FIXED, End.FIXED, tension=100.0, line_density=0.01)
BAR = Continuum(Kind.BAR, 2.0, End.FIXED, End.FIXED, modulus=70e9, density=2700.0)


def test_coefficients():
    assert end_reflection(End.FIXED) == -1 and end_reflection(End.FREE) == 1
    assert end_reflection(End.FIXED, dashpot=1.0) == 0 and end_reflection(End.FIXED, dashpot=0.0) == 1
    r, t = step_coefficients(2.0)
    assert (r, t) == pytest.approx((-1 / 3, 2 / 3))
    assert r**2 + t**2 * 2.0 == pytest.approx(1.0)  # the energy: reflected R², transmitted T² Z₂/Z₁


@pytest.mark.parametrize(("left", "right", "sign"), [(End.FIXED, End.FIXED, 1), (End.FIXED, End.FREE, -1),
                                                      (End.FREE, End.FREE, 1)])
def test_the_continuums_pulse_comes_back_whole(left, right, sign):
    """Two crossings: back where it started, upside down once for each fixed end it met."""
    run = WaveRun(BAR.with_(left=left, right=right), 20, Waves(width=0.2, at=0.3))
    x, u0 = run.exact()
    run.advance(2 * FRAMES_PER_TRANSIT)
    assert run.t == pytest.approx(2 * run.transit)
    assert run.exact()[1] == pytest.approx(sign * u0, abs=1e-12)
    # one crossing: at the far end's mirror image, the right end's sign
    run = WaveRun(BAR.with_(left=left, right=right), 20, Waves(width=0.2, at=0.3))
    run.advance(FRAMES_PER_TRANSIT)
    mirror = pulse(x, 0.7 * BAR.length, 0.2 * BAR.length)
    assert run.exact()[1] == pytest.approx(end_reflection(right) * mirror, abs=1e-3)


def test_released_at_rest_it_splits():
    run = WaveRun(STRING, 20, Waves(width=0.1, at=0.5, moving=False))
    run.advance(FRAMES_PER_TRANSIT // 4)  # a quarter of the length each way
    x, u = run.exact()
    assert u == pytest.approx(0.5 * pulse(x, 0.25, 0.1) + 0.5 * pulse(x, 0.75, 0.1), abs=2e-3)


def test_a_matched_dashpot_takes_the_whole_wave():
    run = WaveRun(BAR, 60, Waves(width=0.15, at=0.3, dashpot=1.0))
    run.advance(2 * FRAMES_PER_TRANSIT)
    assert np.max(np.abs(run.exact()[1])) == 0.0
    assert run.energy_left < 3e-3  # the mesh's waves it resolves are matched too


def test_a_step_reflects_and_transmits():
    run = WaveRun(STRING, 100, Waves(width=0.1, at=0.2, step_at=0.5, step_ratio=2.0))
    assert run.transit == pytest.approx(0.5 / 100 + 0.5 / 50)
    run.advance(FRAMES_PER_TRANSIT // 2)
    x, u = run.exact()
    near = x < 0.5
    assert u[near].min() == pytest.approx(-1 / 3, abs=1e-3) and u[near].max() == pytest.approx(0.0, abs=1e-12)
    assert u[~near].max() == pytest.approx(2 / 3, abs=1e-3)
    # the transmitted pulse is half as long: it travels at half the speed
    wide = x[~near][u[~near] > 1 / 3]
    assert wide[-1] - wide[0] == pytest.approx(0.25 * 0.1, abs=0.002)
    # the mesh's step: the same sides, the energy kept
    assert run.energy_left == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("scheme", list(Scheme))
def test_the_mesh_converges_to_the_continuum(scheme):
    errs = []
    for n in (25, 50, 100, 200):
        run = WaveRun(STRING, n, Waves(width=0.2, at=0.3, scheme=scheme))
        run.advance(FRAMES_PER_TRANSIT // 2)
        x, u = run.exact()
        xm, um = run.mesh()
        errs.append(np.max(np.abs(um - np.interp(xm, x, u))))
        assert run.energy_left == pytest.approx(1.0, abs=1e-9)
    assert all(b < a for a, b in itertools.pairwise(errs)) and errs[-1] < 0.02


def test_playback_keeps_the_state():
    run = WaveRun(STRING, 40, Waves())
    other = WaveRun(STRING, 40, Waves())
    run.advance(FRAMES_PER_TRANSIT // 4)
    run.set_playback(2.0)
    run.advance(FRAMES_PER_TRANSIT // 8)
    other.advance(FRAMES_PER_TRANSIT // 2)
    assert run.t == pytest.approx(other.t)
    assert run.exact()[1] == pytest.approx(other.exact()[1])
    assert run.mesh()[1] == pytest.approx(other.mesh()[1], abs=1e-9)
    assert 3.0 not in PLAYBACK
    with pytest.raises(ValueError):
        run.set_playback(3.0)


@pytest.mark.parametrize(("left", "right"), [(End.PINNED, End.PINNED), (End.FREE, End.FREE),
                                             (End.CLAMPED, End.FREE), (End.PINNED, End.FREE)])
def test_the_beams_modes_sum_to_the_pulse(left, right):
    beam = Continuum(Kind.BEAM, 1.0, left, right, width=0.05, depth=0.01)
    run = WaveRun(beam, 30, Waves(width=0.1, at=0.4, moving=False))
    x, u = run.exact()
    assert u == pytest.approx(pulse(x, 0.4, 0.1), abs=3e-3)
    run.advance(FRAMES_PER_TRANSIT)
    assert run.energy_left == pytest.approx(1.0, abs=1e-8)
    assert not run.moving  # a beam's pulse starts at rest
    with pytest.raises(ValueError):
        WaveRun(beam, 30, Waves(dashpot=1.0))


def test_the_presets_notes():
    def after(name, frames, scheme=None):
        p = PRESET[name]
        run = WaveRun(p.continuum, p.elements, p.waves if scheme is None else p.waves.with_(scheme=scheme))
        run.advance(frames)
        return run

    name = "Waves: a pulse on 40 beads"
    assert PRESET[name].continuum.speed == pytest.approx(100.0)
    run = after(name, FRAMES_PER_TRANSIT)
    xm, um = run.mesh()
    assert um.min() == pytest.approx(-0.66, abs=0.005) and xm[np.argmin(um)] - 0.75 == pytest.approx(0.05)
    assert run.exact()[1].min() == pytest.approx(-1.0)
    assert after(name, FRAMES_PER_TRANSIT, Scheme.CONSISTENT).mesh()[1].min() == pytest.approx(-0.53, abs=0.005)
    assert after(name, FRAMES_PER_TRANSIT, Scheme.MIXED).mesh()[1].min() == pytest.approx(-0.81, abs=0.005)
    assert "66 %" in PRESET[name].note and "53 %" in PRESET[name].note and "81 %" in PRESET[name].note

    name = "Waves: a matched end"
    assert after(name, 2 * FRAMES_PER_TRANSIT).energy_left == pytest.approx(0.17, abs=0.005)
    assert after(name, 2 * FRAMES_PER_TRANSIT, Scheme.CONSISTENT).energy_left == pytest.approx(0.05, abs=0.005)

    name = "Waves: a light string tied to a heavy one"
    run = after(name, FRAMES_PER_TRANSIT // 2)
    xm, um = run.mesh()
    assert um[xm > 0.5].max() == pytest.approx(0.49, abs=0.005)

    name = "Waves: a bending pulse"
    run = after(name, FRAMES_PER_TRANSIT // 2)
    assert run.exact()[1].max() == pytest.approx(0.22, abs=0.005)
    assert run.mesh()[1].max() == pytest.approx(0.26, abs=0.005)
    assert after(name, FRAMES_PER_TRANSIT // 2, Scheme.CONSISTENT).mesh()[1].max() == pytest.approx(0.23, abs=0.005)
