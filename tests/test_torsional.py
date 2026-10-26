"""Torsional drive trains: the model, gears and their referred chain, the criticals, the forced response, the presets."""

import math

import numpy as np
import pytest

from vib_tutorial.core.torsional import (
    COMPRESSOR,
    FAN,
    PRESETS,
    TWO_PI,
    Disc,
    DriveTrain,
    Excitation,
    Mesh,
    Shaft,
    criticals,
    forced_response,
    modes,
    natural_hz,
    order_lines,
    response_speeds,
    separation,
    sources,
    two_inertia_hz,
)

PRESET = {p.name: p for p in PRESETS}


def test_two_inertias_match_the_closed_form():
    (mode,) = modes(FAN)
    assert mode.freq_hz == pytest.approx(two_inertia_hz(2.0, 10.0, 1.8e5), rel=1e-12)
    # the node nearer the heavier disc: the fan swings a fifth of the motor's, the other way
    assert mode.shape[1] / mode.shape[0] == pytest.approx(-2.0 / 10.0)
    assert mode.twist == pytest.approx([1.0])
    # zeta from phi'C phi: the coupling's damper and the fan's to ground
    j1, j2, c, cg = 2.0, 10.0, 20.0, 2.0
    w = TWO_PI * mode.freq_hz
    phi = np.array([1.0, -j1 / j2])
    m = j1 + j2 * (j1 / j2) ** 2
    assert mode.zeta == pytest.approx((c * (1 + j1 / j2) ** 2 + cg * (j1 / j2) ** 2) / (2 * w * m), rel=1e-9)
    assert phi @ np.diag([j1, j2]) @ np.ones(2) == pytest.approx(0.0)  # orthogonal to the rigid turn


def test_a_mesh_turns_the_far_shaft_by_the_teeth_and_the_train_still_turns_freely():
    s = COMPRESSOR.speeds
    assert s == pytest.approx([1.0, 1.0, 4.0, 4.0])
    m, c, k = COMPRESSOR.matrices()
    assert k @ s == pytest.approx(np.zeros(4), abs=1e-6 * np.abs(k).max())  # the rigid turn twists nothing
    assert np.allclose(k, k.T) and np.all(np.linalg.eigvalsh(k) > -1e-6 * np.abs(k).max())


def test_the_referred_chain_has_the_same_frequencies_and_rigid_gears_lose_the_mesh_mode():
    f = natural_hz(COMPRESSOR)
    ref = COMPRESSOR.referred()
    assert all(isinstance(link, Shaft) for link in ref.links)
    assert natural_hz(ref) == pytest.approx(f, rel=1e-10)
    rigid = natural_hz(COMPRESSOR.rigid_gears())
    assert rigid.size == f.size - 1 and rigid == pytest.approx(f[:2], rel=1e-3)
    # a stiffer mesh: the lower modes close on the rigid gears', the mesh's mode goes up as sqrt(k)
    stiff = COMPRESSOR.with_link(1, Mesh(120, 30, 0.3, 2e12, 0.0))
    fs = natural_hz(stiff)
    assert fs[:2] == pytest.approx(rigid, rel=1e-6)
    assert fs[2] == pytest.approx(f[2] * math.sqrt(1e3), rel=1e-2)


def test_criticals_are_where_the_orders_meet_the_modes():
    lines = order_lines(COMPRESSOR)
    labels = [label for label, _ in lines]
    assert labels[:2] == ["Compressor 1×", "Tooth pass (Gear mesh)"]
    assert dict(lines)["Tooth pass (Gear mesh)"] == 120  # teeth on the motor's side, at the motor's speed
    assert "1× Compressor" not in labels  # the excitation's 1× is the compressor shaft's
    assert {"2× Compressor", "1× Bull gear", "2× Bull gear"} <= set(labels)
    f = natural_hz(COMPRESSOR)
    for c in criticals(COMPRESSOR, 1800.0):
        assert c.rpm == pytest.approx(60.0 * f[c.mode] / c.order) and c.rpm <= 1800.0
    assert [label for label, _ in order_lines(FAN)] == ["Misalignment 2×", "1× running speed"]
    assert separation(1569.0, 1480.0) == pytest.approx(89 / 1480)


def test_the_forced_response_matches_two_inertias_in_closed_form():
    j1, j2, k, c, t = 2.0, 10.0, 1.8e5, 20.0, 50.0
    train = DriveTrain((Disc("Motor", j1), Disc("Fan", j2)), (Shaft(k, c),), Excitation(1, ((2.0, t),)))
    rpm = np.linspace(100.0, 3000.0, 50)
    fr = forced_response(train, rpm)
    w = 2.0 * rpm * TWO_PI / 60.0
    # theta2 - theta1 from J1 th1'' = -kt (th1 - th2)..., the shaft carries J1 times the motor's acceleration
    z = k + 1j * w * c
    twist = t * j1 / (j1 * j2 * -w**2 + z * (j1 + j2))  # (th1 - th2)
    assert fr.torque[0, 0] == pytest.approx(np.abs(z * twist), rel=1e-9)
    assert fr.total()[0] == pytest.approx(fr.torque[0, 0])


def test_transmission_error_drives_the_mesh_as_an_imposed_stretch():
    """Two gears on a mesh alone: m δ'' + k δ = k e with 1/m = r1²/J1 + r2²/J2, so the force is k e ω²m/(k − ω²m)."""
    j1, j2, r1, km, e = 0.4, 0.1, 0.1, 1e9, 3e-6
    mesh = Mesh(40, 20, r1, km, 0.0, e)
    train = DriveTrain((Disc("Gear", j1), Disc("Pinion", j2)), (mesh,))
    (src,) = sources(train)
    assert src.order == 40 and src.mesh == 0
    rpm = np.array([300.0, 900.0, 3000.0])
    fr = forced_response(train, rpm)
    w = 40 * rpm * TWO_PI / 60.0
    m = 1.0 / (r1**2 / j1 + mesh.radius_out**2 / j2)
    force = km * e * w**2 * m / np.abs(km - w**2 * m)
    assert fr.torque[0, 0] == pytest.approx(force * r1, rel=1e-9)


def test_response_speeds_resolve_each_peak():
    rpm = response_speeds(FAN, 3600.0)
    fr = forced_response(FAN, rpm)
    i = int(np.argmax(fr.total()[0]))
    assert rpm[i] == pytest.approx(1569.1, rel=1e-3)
    # the peak's height is the dense sweep's
    dense = np.linspace(1500.0, 1650.0, 20001)
    assert fr.total()[0, i] == pytest.approx(forced_response(FAN, dense).total()[0].max(), rel=1e-3)


def test_editing_keeps_the_excitation_on_its_disc():
    t = PRESET["Diesel generator set"].train
    assert t.discs[t.excitation.disc].name == "Crank 3"
    more = t.inserted(1)
    assert more.n == t.n + 1 and more.discs[more.excitation.disc].name == "Crank 3"
    assert more.discs[2].name == "Crank 1 (2)"
    fewer = t.removed(0)
    assert fewer.n == t.n - 1 and fewer.discs[fewer.excitation.disc].name == "Crank 3"
    assert t.removed(t.excitation.disc).excitation is None
    assert t.removed(t.n - 1).links[-1].name == "Crank–flywheel"  # the last disc takes the link before it
    with pytest.raises(ValueError):
        FAN.removed(0)
    with pytest.raises(ValueError):
        FAN.with_disc(0, Disc("Motor", 0.0))
    with pytest.raises(ValueError):
        FAN.with_link(0, Mesh(0, 30, 0.1, 1e9))


def test_preset_notes():
    fan = PRESET["Motor and fan: two inertias"]
    assert natural_hz(fan.train) == pytest.approx([52.3], abs=0.05)
    (two,) = [c for c in criticals(fan.train, fan.top_rpm) if c.label == "Misalignment 2×"]
    assert round(two.rpm) == 1569 and round(100 * separation(two.rpm, fan.operating_rpm)) == 6
    softer = fan.train.with_link(0, Shaft(1.2e5, 20.0, "Coupling"))
    assert 60 * natural_hz(softer)[0] / 2 < fan.operating_rpm * 0.9  # "a softer coupling moves it below"

    comp = PRESET["Geared compressor"]
    f = natural_hz(comp.train)
    assert round(f[2]) == 2688 and comp.train.links[1].teeth_in / comp.train.links[1].teeth_out == 4
    ms = modes(comp.train)
    assert [comp.train.links[m.link].name for m in ms] == ["High-speed coupling", "Low-speed coupling", "Gear mesh"]
    assert natural_hz(comp.train.rigid_gears()) == pytest.approx(f[:2], rel=1e-3)
    (tp,) = [c for c in criticals(comp.train, comp.top_rpm) if c.label.startswith("Tooth") and c.mode == 2]
    assert round(tp.rpm) == 1344 and round(100 * separation(tp.rpm, comp.operating_rpm)) == 10

    gen = PRESET["Diesel generator set"]
    f = natural_hz(gen.train)
    assert round(f[0], 1) == 18.9 and round(f[1], 1) == 97.3
    assert modes(gen.train)[0].link == gen.train.links.index(
        next(link for link in gen.train.links if link.name == "Flexible coupling"))
    crits = {(c.label, c.mode): round(c.rpm) for c in criticals(gen.train, gen.top_rpm)}
    assert crits[("Firing 3×", 0)] == 378 and crits[("Firing 6×", 1)] == 973
