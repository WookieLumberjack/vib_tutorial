"""Continuous systems: the exact modes, the meshes' dispersion and modes, their convergence, the presets."""

import math

import numpy as np
import pytest

from vib_tutorial.core.continuum import (
    ENDS,
    PRESETS,
    SCHEMES,
    TWO_PI,
    Continuum,
    End,
    FEModel,
    Kind,
    Scheme,
    convergence,
    dispersion,
    errors,
    flexible_hz,
    order,
    string_dispersion,
)

PRESET = {p.name: p for p in PRESETS}
BAR = Continuum(Kind.BAR, 2.0, End.FIXED, End.FIXED, modulus=70e9, density=2700.0)


def beam(left: End, right: End) -> Continuum:
    return Continuum(Kind.BEAM, 1.5, left, right, modulus=200e9, density=8000.0, width=0.03, depth=0.02)


@pytest.mark.parametrize(("left", "right", "roots"), [
    (End.CLAMPED, End.FREE, (1.875104, 4.694091, 7.854757)),
    (End.CLAMPED, End.CLAMPED, (4.730041, 7.853205, 10.995608)),
    (End.FREE, End.FREE, (4.730041, 7.853205, 10.995608)),
    (End.PINNED, End.PINNED, (math.pi, 2 * math.pi, 3 * math.pi)),
    (End.CLAMPED, End.PINNED, (3.926602, 7.068583, 10.210176)),
    (End.FREE, End.PINNED, (3.926602, 7.068583, 10.210176)),
])
def test_beam_roots_match_the_tables(left, right, roots):
    c = beam(left, right)
    assert c.exact_wavenumbers(3) * c.length == pytest.approx(roots, abs=2e-6)
    # far up, the roots are (n + ½)π apart from their asymptotes, not lost or doubled
    k = c.exact_wavenumbers(300) * c.length
    assert np.diff(k)[-50:] == pytest.approx(math.pi, abs=1e-9)


@pytest.mark.parametrize("left", [End.CLAMPED, End.PINNED, End.FREE])
@pytest.mark.parametrize("right", [End.CLAMPED, End.PINNED, End.FREE])
def test_exact_beam_shapes_meet_their_ends(left, right):
    c = beam(left, right)
    for kappa in c.exact_wavenumbers(4):
        x = np.linspace(0.0, c.length, 4001)
        w = c.exact_shape(kappa, x)
        dx = x[1] - x[0]
        d = [w, np.gradient(w, dx, edge_order=2), np.gradient(np.gradient(w, dx, edge_order=2), dx, edge_order=2)]
        for end, i in ((left, 0), (right, -1)):
            if end in (End.CLAMPED, End.PINNED):
                assert abs(w[i]) < 1e-9
            if end is End.CLAMPED:
                assert abs(d[1][i]) < 2e-3 * kappa
            if end is End.PINNED:
                assert abs(d[2][i + (1 if i == 0 else -1) * 3]) < 2e-2 * kappa**2


@pytest.mark.parametrize("scheme", SCHEMES)
def test_bloch_dispersion_matches_the_closed_form(scheme):
    h = 0.1
    kappa = np.linspace(0.0, math.pi / h, 41)
    assert dispersion(BAR, h, scheme, kappa)[:, 0] == pytest.approx(string_dispersion(BAR, h, scheme, kappa),
                                                                     rel=1e-12, abs=1e-6)
    # long waves travel at c: ω/κ → c, lumped from below, consistent from above
    small = np.array([1e-3 / h])
    assert string_dispersion(BAR, h, scheme, small)[0] / small[0] == pytest.approx(BAR.speed, rel=1e-6)


@pytest.mark.parametrize("left", [End.FIXED, End.FREE])
@pytest.mark.parametrize("right", [End.FIXED, End.FREE])
@pytest.mark.parametrize("scheme", SCHEMES)
def test_the_bars_modes_lie_on_the_meshs_dispersion_curve(left, right, scheme):
    c = BAR.with_(left=left, right=right)
    model = FEModel(c, 9, scheme)
    f = flexible_hz(model)
    kappa = c.exact_wavenumbers(f.size)
    assert f == pytest.approx(string_dispersion(c, model.h, scheme, kappa) / TWO_PI, rel=1e-10)
    # and their shapes are the continuum's, sampled at the nodes
    _, shapes = model.modes()
    for i, k in enumerate(kappa):
        exact = c.exact_shape(k, model.nodes)
        fe = shapes[:, i + c.rigid_modes]
        assert abs(fe @ exact) == pytest.approx(np.linalg.norm(fe) * np.linalg.norm(exact), rel=1e-9)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_a_pinned_beams_modes_lie_on_the_lowest_branch(scheme):
    c = beam(End.PINNED, End.PINNED)
    model = FEModel(c, 7, scheme)
    f = flexible_hz(model)
    n = model.elements - 1  # the sines the mesh resolves; above them the slopes' branch
    kappa = c.exact_wavenumbers(n)
    assert f[:n] == pytest.approx(dispersion(c, model.h, scheme, kappa)[:, 0] / TWO_PI, rel=1e-9)


def test_lumped_is_below_and_consistent_above():
    for c in (BAR, BAR.with_(right=End.FREE), beam(End.CLAMPED, End.FREE), beam(End.FREE, End.FREE)):
        lumped = errors(FEModel(c, 8, Scheme.LUMPED))[1]
        consistent = errors(FEModel(c, 8, Scheme.CONSISTENT))[1]
        assert np.all(lumped < 0) and np.all(consistent > 0)  # consistent: Rayleigh–Ritz, an upper bound
    # the string's errors are equal and opposite to leading order: (κh)²/24
    h = BAR.length / 40
    kh = BAR.exact_wavenumbers(1)[0] * h
    for scheme, sign in ((Scheme.LUMPED, -1), (Scheme.CONSISTENT, 1)):
        assert errors(FEModel(BAR, 40, scheme))[1][0] == pytest.approx(sign * kh**2 / 24, rel=1e-3)


@pytest.mark.parametrize(("c", "rates"), [
    (BAR, {Scheme.LUMPED: 2, Scheme.CONSISTENT: 2, Scheme.MIXED: 4}),
    (beam(End.CLAMPED, End.FREE), {Scheme.LUMPED: 2, Scheme.CONSISTENT: 4, Scheme.MIXED: 2}),
])
def test_convergence_rates(c, rates):
    for scheme, p in rates.items():
        counts, errs = convergence(c, scheme, 2)
        assert order(counts, errs) == pytest.approx(p, abs=0.05)
    # the rate is the same for any size of the same kind: only the mesh matters
    assert convergence(c.with_(length=7.0), Scheme.LUMPED, 2)[1] == pytest.approx(convergence(c, Scheme.LUMPED, 2)[1])


def test_rigid_modes():
    for kind in Kind:
        for left in ENDS[kind]:
            for right in ENDS[kind]:
                c = (beam(left, right) if kind is Kind.BEAM else BAR.with_(kind=kind, left=left, right=right))
                hz, _ = FEModel(c, 5, Scheme.CONSISTENT).modes()
                top = hz[-1]
                assert np.sum(hz < 1e-6 * top) == c.rigid_modes


def test_one_element_between_held_ends_has_no_modes():
    assert flexible_hz(FEModel(BAR, 1, Scheme.LUMPED)).size == 0
    assert flexible_hz(FEModel(BAR, 2, Scheme.LUMPED)) == pytest.approx(
        [2 * BAR.speed / (BAR.length / 2) * math.sin(math.pi / 4) / TWO_PI])  # one bead: Lagrange's formula


def test_a_lumped_string_is_beads_on_a_string():
    """Lagrange: N − 1 beads, ω_n = 2√(T/(m h)) sin(nπ / 2N)."""
    c = PRESET["Guitar string (low E)"].continuum
    n = 12
    model = FEModel(c, n, Scheme.LUMPED)
    m, h = c.line_density * model.h, model.h
    beads = 2 * np.sqrt(c.tension / (m * h)) * np.sin(np.arange(1, n) * math.pi / (2 * n)) / TWO_PI
    assert flexible_hz(model) == pytest.approx(beads, rel=1e-12)


def test_the_presets_notes():
    p = PRESET["Guitar string (low E)"]
    exact, err = errors(FEModel(p.continuum, p.elements, Scheme.LUMPED))
    assert exact[0] == pytest.approx(82.4, abs=0.05)
    assert 100 * err[5] == pytest.approx(-10.0, abs=0.05) and 100 * err[10] == pytest.approx(-31, abs=0.5)
    assert "10.0 % low and the 11th 31 %" in p.note

    p = PRESET["Steel bar, fixed–free"]
    assert p.continuum.speed == pytest.approx(5172, abs=0.5)
    exact = p.continuum.exact_hz(3)
    assert exact[0] == pytest.approx(1293, abs=0.5) and exact / exact[0] == pytest.approx([1, 3, 5])
    lumped = errors(FEModel(p.continuum, p.elements, Scheme.LUMPED))[1]
    mixed = errors(FEModel(p.continuum, p.elements, Scheme.MIXED))[1]
    assert np.all(np.abs(mixed) < np.abs(lumped))

    p = PRESET["Steel ruler, clamped–free"]
    assert p.continuum.exact_hz(3) == pytest.approx([9.28, 58.2, 163], rel=2e-3)
    assert order(*convergence(p.continuum, Scheme.CONSISTENT, 1)) == pytest.approx(4, abs=0.05)
    assert order(*convergence(p.continuum, Scheme.LUMPED, 1)) == pytest.approx(2, abs=0.05)

    p = PRESET["Vibraphone bar, free–free"]
    f = p.continuum.exact_hz(3)
    assert f[0] == pytest.approx(693, abs=0.5) and f[1:] / f[0] == pytest.approx([2.76, 5.40], abs=0.005)
    assert p.continuum.rigid_modes == 2

    p = PRESET["Pinned–pinned beam"]
    model = FEModel(p.continuum, p.elements, Scheme.CONSISTENT)
    f = flexible_hz(model)
    assert f.size == 12
    # modes 6 to 12 on the slopes' branch, drawn in the next zone: at 2π/h − κ_n
    kappa = p.continuum.exact_wavenumbers(12)[5:]
    assert f[5:] == pytest.approx(dispersion(p.continuum, model.h, Scheme.CONSISTENT,
                                             2 * math.pi / model.h - kappa)[:, 1] / TWO_PI, rel=1e-9)
    err = errors(model)[1]
    assert 100 * err[5] == pytest.approx(11, abs=0.5) and 100 * err[11] == pytest.approx(27, abs=0.5)
