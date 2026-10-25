"""The Rust kernels (vib_tutorial._native) against the Python they replace, which stays the reference."""

import math
import os
import sys
import time

import numpy as np
import pytest
import scipy.linalg

from vib_tutorial.core import fluid_bearing, native
from vib_tutorial.core.forcing import ForceKind
from vib_tutorial.core.model import ChainSystem
from vib_tutorial.core.rotor import RPM, RotorSimulator, RotorSystem
from vib_tutorial.core.rotor_presets import JOURNAL_BEARING, rotor_presets
from vib_tutorial.core.simulator import Simulator, foh_discretize

needs_native = pytest.mark.skipif(native.module is None, reason="Rust kernels not loaded")


def test_kernels_are_built():
    # A source install builds them, so a missing module is a broken build, unless asked for the Python.
    if os.environ.get("VIB_NATIVE", "1") == "0":
        assert native.module is None
        assert fluid_bearing.pi_film is fluid_bearing.pi_film_reference
        assert fluid_bearing.full_film_damping is fluid_bearing.full_film_damping_reference
    else:
        assert native.module is not None
        assert fluid_bearing.pi_film is native.module.pi_film
        assert fluid_bearing.full_film_damping is native.module.full_film_damping


@needs_native
def test_constants_agree():
    assert native.module.ECCENTRICITY_MAX == fluid_bearing.ECCENTRICITY_MAX


@needs_native
def test_pi_film_matches_the_reference():
    # Anywhere in the clearance and past the wall, centred, at rest and spinning either way.
    rng = np.random.default_rng(7)
    k, c = 2.4e3, 50e-6
    cases = [(0.0, 0.0, 0.0, 0.0, 300.0), (0.0, 0.0, 1e-3, -2e-3, 300.0), (0.3 * c, 0.0, 0.0, 0.0, 0.0),
             (0.0, -0.99 * c, 0.0, 0.0, 300.0), (2 * c, -c, 0.0, 0.0, 50.0)]
    for _ in range(500):
        eps, angle = rng.choice([rng.uniform(0.0, 0.99), rng.uniform(1.0, 1.5)]), rng.uniform(0.0, 2 * math.pi)
        omega = rng.choice([1.0, -1.0]) * rng.uniform(1.0, 3000.0)
        vx, vy = rng.normal(scale=c * abs(omega), size=2)
        cases.append((eps * c * math.cos(angle), eps * c * math.sin(angle), vx, vy, omega))
    for case in cases:
        want = np.array(fluid_bearing.pi_film_reference(k, c, *case))
        got = native.module.pi_film(k, c, *case)
        assert isinstance(got, tuple) and len(got) == 10
        # Each group (force, stiffness, damping, dissipation) on its own scale. The integrals divide by
        # powers of 1 - ε², up to the stiffness's 3.5, so the libraries' rounding (Rust's sin_cos and
        # powi against Python's math, which differ on macOS) grows near the wall: at ε = 0.99 the
        # stiffness rounds apart by ~1e-9. At the wall (where past it the film is worked out) even the
        # force is good to only ~1e-3 in either, against a 50-digit evaluation, and the stiffness is noise.
        wall = math.hypot(case[0], case[1]) > 0.99 * c
        near = 1e-2 if wall else 1e-10
        for part, tol in ((slice(0, 2), near), (slice(2, 6), None if wall else 1e-8),
                          (slice(6, 9), near), (slice(9, 10), near)):
            if tol is not None:
                scale = np.abs(want[part]).max()
                np.testing.assert_allclose(got[part], want[part], rtol=0, atol=tol * scale, err_msg=str(case))


@needs_native
def test_full_film_damping_matches_the_reference():
    rng = np.random.default_rng(3)
    k, c = 2.4e3, 50e-6
    for eps in [0.0, *rng.uniform(0.0, 1.5, 300)]:
        angle = rng.uniform(0.0, 2 * math.pi)
        x, y = eps * c * math.cos(angle), eps * c * math.sin(angle)
        want = fluid_bearing.full_film_damping_reference(k, c, x, y)
        # At the wall it divides by (1 - ε²)^2.5 ~ 1e-15, and the platforms' pow round apart by ~1e-13 there.
        np.testing.assert_allclose(native.module.full_film_damping(k, c, x, y), want, rtol=0,
                                   atol=1e-12 * max(abs(v) for v in want))


@needs_native
def test_expm_matches_scipy():
    rng = np.random.default_rng(0)
    for n in (1, 5, 32, 48):  # 32 and 48: the film step's augmented matrices, with and without a shaft's mass
        for norm in (1e-3, 0.1, 1.0, 5.0, 40.0):  # each Padé degree, and scaling and squaring
            a = rng.normal(size=(n, n)) * norm / math.sqrt(n)
            want = scipy.linalg.expm(a)
            np.testing.assert_allclose(native.module.expm(a), want, rtol=0, atol=1e-13 * np.abs(want).max())
    # The film step's kind: stiff modes over a short step, with the FOH's input blocks. Its norm is
    # thousands but its powers' are small, so it needs few squarings. (On the Journal bearings preset's
    # own matrix, scaling by the norm alone, as Higham 2005 does, took eleven and Φ came out 3.5e-11 off.)
    n, h = 8, 1.2e-4
    k = np.diag(rng.uniform(1e6, 4e8, n))
    a = np.zeros((4 * n, 4 * n))
    a[:n, n:2 * n] = np.eye(n) * h
    a[n:2 * n, :n], a[n:2 * n, n:2 * n] = -k * h, -1e-3 * k * h
    a[n:2 * n, 2 * n:3 * n] = h * np.eye(n)
    a[2 * n:3 * n, 3 * n:] = np.eye(n)
    assert np.abs(a).sum(axis=0).max() > 1e3
    want = scipy.linalg.expm(a)
    np.testing.assert_allclose(native.module.expm(a), want, rtol=0, atol=1e-13 * np.abs(want).max())


@needs_native
@pytest.mark.parametrize("density", [0.0, 7800.0])
def test_foh_discretize_matches_the_python(density):
    # The film step's update, worked out in the augmented matrix's blocks, on the rotor's own A and B with
    # its films' damping bound: as simulator.foh_discretize's dense expm gives it.
    system = RotorSystem(bearing_a=JOURNAL_BEARING, bearing_b=JOURNAL_BEARING, gravity=9.81, density=density,
                         nonlinear_films=True)
    sim = RotorSimulator(system)
    sim._film_steps = sim._film_steps_reference
    sim.omega = sim.target = 6000.0 / RPM
    sim.advance(0.2)
    films, n, h = sim._films, system.n_dof, sim._films.key[0]
    a = films.A.copy()
    a[n:2 * n, films.rates] -= system.mass_inverse[:, films.dofs] @ (films.Cb - films.Cf)
    b = system.input_matrix()
    # The kernel balances the matrix first (the rotor's A h runs from h to K/M h), so it's the more
    # accurate of the two: against a 40-digit evaluation ~1e-15 at the step, where SciPy's is ~1e-14.
    # Many squarings (64 steps at once) cost both ~1e-12.
    for scale in (0.01, 1.0, 64.0):
        for got, want in zip(native.module.foh_discretize(a, b, h * scale), foh_discretize(a, b, h * scale)):
            np.testing.assert_allclose(got, want, rtol=0, atol=5e-13 * max(scale, 1.0) * np.abs(want).max())


@needs_native
@pytest.mark.parametrize("gravity, density, rpm, drifts", [
    (9.81, 0.0, 9000.0, False),  # Journal bearings (oil whip), past the whip's onset
    (0.0, 0.0, 3500.0, True),  # Oil whirl (vertical rotor): the journals circle past ε = 0.9, the bound retaken often
    (9.81, 7800.0, 6000.0, False),  # a shaft with mass: more states than the stations'
])
def test_film_steps_match_the_reference(gravity, density, rpm, drifts):
    # The kernel steps as _film_steps_reference does, taking the damping bound afresh as the journals
    # drift (with its own expm): the same motion to rounding error, while ramping and at a steady speed.
    # Its expm (balanced) is a little more accurate than SciPy's; the two part by ~1e-12 of the motion.
    system = RotorSystem(bearing_a=JOURNAL_BEARING, bearing_b=JOURNAL_BEARING, gravity=gravity,
                         density=density, nonlinear_films=True)
    sims = [RotorSimulator(system), RotorSimulator(system)]
    sims[1]._film_steps = sims[1]._film_steps_reference
    for sim in sims:
        sim.omega, sim.target = 0.9 * rpm / RPM, rpm / RPM
        sim.accel = 0.1 * rpm / RPM / 0.25  # at speed 0.25 s into the first stretch
    for steady in (False, True):  # ramping, then at speed
        assert (sims[0].omega == sims[0].target) == steady
        takes = sims[0]._films.kernel.takes if steady else 0
        for _ in range(2):
            got, want = sims[0].advance(0.15), sims[1].advance(0.15)
            np.testing.assert_allclose(got.q, want.q, rtol=0, atol=1e-10 * np.abs(want.q).max())
            np.testing.assert_allclose(sims[0].state, sims[1].state, rtol=0,
                                       atol=1e-10 * np.abs(sims[1].state).max())
    # At speed the kernel keeps its bound from one advance to the next, and retakes it as the journals drift.
    assert (sims[0]._films.kernel.takes > takes) == drifts


@needs_native
@pytest.mark.parametrize("name, density, rpm", [
    ("Partial rub (½X)", 0.0, 4400.0),  # under gravity, hitting the ring's bottom now and then
    ("Full rub (backward whirl)", 0.0, 2000.0),  # held against it all round, the friction's drag stiff
    ("Full rub (backward whirl)", 7800.0, 2000.0),  # a shaft with mass: more states than the stations'
    ("Newkirk effect (thermal bow)", 0.0, 1300.0),  # the thermal bow moving the load between chunks
])
def test_rub_steps_match_the_reference(name, density, rpm):
    # The kernel steps as _rub_steps_reference does, Newton and all: the same motion, push and heat to
    # rounding error, while ramping and at a steady speed.
    system = next(p for p in rotor_presets() if p.name == name).system.with_(density=density)
    sims = [RotorSimulator(system), RotorSimulator(system)]
    sims[1]._rub_steps = sims[1]._rub_steps_reference
    for sim in sims:
        sim.omega, sim.target = 0.9 * rpm / RPM, rpm / RPM
        sim.accel = 0.1 * rpm / RPM / 0.25  # at speed 0.25 s into the first stretch
        sim.tap(0.02, 0.0)  # into the ring at once
    for steady in (False, True):
        assert (sims[0].omega == sims[0].target) == steady
        for _ in range(2):
            got, want = sims[0].advance(0.15), sims[1].advance(0.15)
            assert want.rub[:, 2].max() > 0.0
            np.testing.assert_allclose(got.q, want.q, rtol=0, atol=1e-9 * np.abs(want.q).max())
            gap = np.abs(got.rub - want.rub).max(axis=0) / np.maximum(np.abs(want.rub).max(axis=0), 1e-300)
            assert gap.max() < 1e-9, gap  # each column: f_x, f_y, N, heat
            np.testing.assert_allclose(sims[0].state, sims[1].state, rtol=0,
                                       atol=1e-9 * np.abs(sims[1].state).max())
    assert sims[0].thermal_bow == pytest.approx(sims[1].thermal_bow, rel=1e-9)


def _forced(system, freq, amplitude, base=False):
    sim = Simulator(system)
    s = sim.force.settings
    s.kind, s.freq_hz, s.base = ForceKind.HARMONIC, freq, base
    if base:
        s.base_amplitude = amplitude
    else:
        s.amplitude = amplitude
    sim.force.switch_on()
    return sim


FRICTION_CASES = {
    "stick-slip": (ChainSystem([1.0] * 3, [400.0] * 3, [1.0] * 3, [0.3, 0.2, 0.5]), 2.0, 2.0, False),
    "eight masses": (ChainSystem([0.5] * 8, [2000.0] * 8, [0.5] * 8, [1.0] * 8), 5.0, 20.0, False),
    "stiff, 1 kHz": (ChainSystem([0.01] * 4, [1e5] * 4, [0.1] * 4, [0.5] * 4), 1000.0, 50.0, False),
    "ground motion": (ChainSystem([1.0] * 3, [400.0] * 3, [1.0] * 3, [0.3, 0.2, 0.5]), 2.0, 0.01, True),
    "static friction, a free mass": (ChainSystem([1.0] * 3, [400.0] * 3, [1.0] * 3, [0.3, 0.0, 0.5],
                                                 static_ratio=1.5), 2.0, 2.0, False),
}


@needs_native
@pytest.mark.parametrize("case", FRICTION_CASES)
def test_friction_steps_match_the_reference(case, monkeypatch):
    # The kernel steps as _advance_friction_reference does, events and all; its energy ledger comes from
    # the input's work and the energy balance rather than Van Loan's integrals, the same to rounding.
    system, freq, amplitude, base = FRICTION_CASES[case]
    sim, ref = _forced(system, freq, amplitude, base), _forced(system, freq, amplitude, base)
    for _ in range(6):
        got = sim.advance(0.1)
        with monkeypatch.context() as m:
            m.setattr(native, "module", None)
            want = ref.advance(0.1)
        for g, w in zip(got, want):
            np.testing.assert_allclose(g, w, rtol=0, atol=1e-11 * max(np.abs(w).max(), 1e-300))
        np.testing.assert_allclose(sim.ledger, ref.ledger, rtol=0, atol=1e-11 * np.abs(ref.ledger).max())
    assert sim.friction_loss > 0.0
    assert sim.t == ref.t


@needs_native
def test_friction_drive_matches_the_reference(monkeypatch):
    # The virtual modal test's path: a burst, a quiet stretch where everything sticks (skipped), a step, noise.
    rng = np.random.default_rng(1)
    f = np.concatenate([rng.normal(scale=3.0, size=800), np.zeros(1500), np.full(300, 0.4),
                        rng.normal(scale=1.0, size=500)])
    for system in (ChainSystem([1.0] * 3, [400.0] * 3, [1.0] * 3, [0.3, 0.2, 0.5]),
                   ChainSystem([0.5] * 8, [2000.0] * 8, [0.5] * 8, [1.0] * 8, static_ratio=1.3)):
        got = Simulator(system).drive(f, 2.0**-11)
        with monkeypatch.context() as m:
            m.setattr(native, "module", None)
            ref = Simulator(system)
            want = ref.drive(f, 2.0**-11)
        assert (np.abs(want[0][:, system.n:]).max(axis=1) == 0.0).sum() > 1000  # the quiet stretch stuck
        for g, w in zip(got, want):
            np.testing.assert_allclose(g, w, rtol=0, atol=1e-11 * np.abs(w).max())


@needs_native
def test_friction_steps_run_beside_the_caller():
    # Simulator.start leaves the Rust steps on a worker thread, without the GIL: this thread runs meanwhile
    # (with a switch interval of 10 s, a kernel that held the GIL would block it until the steps were done).
    system, freq, amplitude, base = FRICTION_CASES["eight masses"]
    sim, ref = _forced(system, freq, amplitude, base), _forced(system, freq, amplitude, base)
    assert sim.steps_aside
    interval = sys.getswitchinterval()
    sys.setswitchinterval(10.0)
    try:
        sim.start(20_000 * sim.step_size())
        beside = 0
        while not sim.ready:
            time.sleep(1e-3)
            beside += 1
    finally:
        sys.setswitchinterval(interval)
    assert beside >= 3
    np.testing.assert_array_equal(sim.collect()[1], ref.advance(20_000 * ref.step_size())[1])


@needs_native
def test_started_steps_finish_whenever_the_state_is_touched():
    system, freq, amplitude, base = FRICTION_CASES["stick-slip"]
    sim, ref = _forced(system, freq, amplitude, base), _forced(system, freq, amplitude, base)
    sim.start(0.1)
    with pytest.raises(RuntimeError):
        sim.start(0.1)  # one at a time: the first's samples are collected first
    x = sim.displacement.copy()  # waits for the steps
    want = ref.advance(0.1)
    np.testing.assert_array_equal(x, ref.displacement)
    sim.set_state(np.zeros(3), np.zeros(3))  # the samples before it are still there to collect
    got = sim.collect()
    for g, w in zip(got, want):
        np.testing.assert_array_equal(g, w)
    np.testing.assert_array_equal(sim.ledger, ref.ledger)
    with pytest.raises(RuntimeError):
        sim.collect()
    sim.start(0.1)
    sim.reset()  # drops them
    assert sim.t == 0.0 and not sim.displacement.any()
    with pytest.raises(RuntimeError):
        sim.collect()
