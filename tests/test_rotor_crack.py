"""The cracked shaft: its compliance, the hinge's condensation, breathing, and the 1X, 2X and 3X it drives."""

import math

import numpy as np
import pytest

import vib_tutorial.core.rotor as rotor
from vib_tutorial.core.rotor import (
    RPM,
    STEEL,
    XD,
    YD,
    Bearing,
    Crack,
    RotorSamples,
    RotorSimulator,
    RotorSystem,
    beam_stiffness,
    campbell,
    crack_compliance,
    steady_harmonics,
    synchronous_response,
    whirl_modes,
)
from vib_tutorial.core.rotor_presets import rotor_presets
from vib_tutorial.core.rotor_stability import growth_rate, stability_onset
from vib_tutorial.core.rotor_sweep import (
    RevolutionTracker,
    RunUpCoastDown,
    TrackedSweep,
    probe,
    station_motion,
    steady_vectors,
    vibration,
)

CRACKED = next(p for p in rotor_presets() if p.name.startswith("Cracked")).system
ANISO = Bearing(ky=8e4)


def run(sim: RotorSimulator, duration: float) -> RotorSamples:
    """Advance by `duration` s, looping over the simulator's cap on steps per call."""
    parts, end = [], sim.t + duration
    while sim.t < end - 1e-12:
        r = sim.advance(end - sim.t)
        if not r.t.size:
            break
        parts.append(r)
    return RotorSamples(*(np.concatenate([getattr(p, f) for p in parts]) for f in ("t", "q", "omega", "phase")))


def orbit_peak(system: RotorSystem, order: int, w: np.ndarray) -> tuple[float, int]:
    """(the disc's orbit's largest steady amplitude of this order in µm, to 0.1, and its speed in rpm)."""
    v = np.abs(probe(steady_vectors(system, w, order)[:, 1], "orbit"))
    i = int(np.argmax(v))
    return round(v[i] * 1e6, 1), round(w[i] * RPM)


def test_compliance():
    assert crack_compliance(0.0) == (0.0, 0.0)
    depths = np.linspace(0.05, 0.5, 10)
    c = np.array([crack_compliance(d) for d in depths])
    assert np.all(np.diff(c, axis=0) > 0)  # deeper, more compliant both ways
    assert np.all(c[:, 0] > c[:, 1])  # far more to bending that opens it
    assert c[0, 0] / c[0, 1] > 30 and c[-1, 0] / c[-1, 1] < 1.6
    # A shallow crack is an edge crack in a strip at the surface: c ∝ a^{5/2}.
    slope = np.log(crack_compliance(0.004)[0] / crack_compliance(0.002)[0]) / math.log(2.0)
    assert slope == pytest.approx(2.5, abs=0.02)
    s = RotorSystem(crack=Crack(depth=0.4))
    np.testing.assert_allclose(s.crack_compliances, np.array(crack_compliance(0.4)) * 0.91 / (200e9 * 0.005**3))
    assert s.with_(crack=Crack(compliance=(1e-4, 2e-5))).crack_compliances == (1e-4, 2e-5)


def test_validation():
    for crack in (Crack(depth=0.6), Crack(depth=-0.1), Crack(position=1.0 + 1e-9), Crack(position=-0.1)):
        with pytest.raises(ValueError):
            RotorSystem(crack=crack)
    RotorSystem(crack=Crack(position=1.1), position=1.2)  # in the overhang


def test_the_hinge_is_condensed_exactly():
    # Open, along x: the stations' stiffness is a fine model's with a very short segment as compliant as the hinge.
    s = RotorSystem(crack=Crack(position=0.4, depth=0.5, breathing=False))
    c = s.crack_compliances[0]
    eps, L, zc = 1e-5, s.length, 0.4 * s.length
    z = np.unique(np.r_[np.linspace(0, L, 41), zc - eps / 2, zc + eps / 2, s.a])
    n = 2 * z.size
    K = np.zeros((n, n))
    for e, length in enumerate(np.diff(z)):
        ei = eps / c if abs(0.5 * (z[e] + z[e + 1]) - zc) < eps else s.ei
        K[2 * e:2 * e + 4, 2 * e:2 * e + 4] += beam_stiffness(ei, length)
    node = [int(np.argmin(abs(z - x))) for x in (0.0, s.a, L)]
    kept = [2 * node[0], 2 * node[1], 2 * node[1] + 1, 2 * node[2]]
    inner = [i for i in range(n) if i not in kept]
    Kc = K[np.ix_(kept, kept)] - K[np.ix_(kept, inner)] @ np.linalg.solve(K[np.ix_(inner, inner)], K[np.ix_(inner, kept)])
    ix, iy = s.plane_dofs
    K0 = s.stiffness_at(0.0, 0.0)  # mouth along +x
    bearings = np.diag([s.bearing_a.kx, 0.0, 0.0, s.bearing_b.kx])
    np.testing.assert_allclose(K0[np.ix_(ix, ix)] - bearings, Kc, rtol=0, atol=2e-4 * np.abs(Kc).max())
    # Across, y: the smaller compliance.
    cy = s.with_(crack=Crack(position=0.4, compliance=(s.crack_compliances[1],) * 2, breathing=False))
    np.testing.assert_allclose(K0[np.ix_(iy, iy)], cy.stiffness(0.0)[np.ix_(iy, iy)], rtol=1e-12, atol=1e-6)
    # The massless shaft's crack does not depend on the mesh.
    np.testing.assert_allclose(s.with_(elements=2).stiffness_at(0.0, 0.3), s.with_(elements=20).stiffness_at(0.0, 0.3),
                               rtol=1e-9, atol=1e-6)


def test_breathing():
    s = CRACKED
    assert s.crack_sign == 1 and s.crack_opening == 0.5 and s.periodic and s.spacing == 1 and not s.rotating_frame
    open_ = s.with_(crack=Crack(position=0.5, depth=0.4, breathing=False))
    assert open_.spacing == 2 and open_.rotating_frame
    round_ = s.with_(crack=None)
    # Mouth down (φ + β_c = -90°): fully open. Up: shut, the round shaft. Across: half open.
    np.testing.assert_allclose(s.stiffness_at(0.0, -math.pi / 2), open_.stiffness_at(0.0, -math.pi / 2), atol=1e-6)
    np.testing.assert_allclose(s.stiffness_at(0.0, math.pi / 2), round_.stiffness(0.0), atol=1e-6)
    half = 0.5 * (open_.stiffness_at(0.0, 0.0) + round_.stiffness(0.0))
    np.testing.assert_allclose(s.stiffness_at(0.0, 0.0), half, atol=1e-6)
    # Its mean over a turn is the mean stiffness.
    phases = np.linspace(0.0, 2 * math.pi, 64, endpoint=False)
    np.testing.assert_allclose(s.stiffness_at(0.0, phases).mean(axis=0), s.stiffness(0.0), atol=1e-6)
    # An overhang hogs: the weight opens a crack there with its mouth up.
    over = RotorSystem(position=1.2, crack=Crack(position=1.1), gravity=9.81)
    assert over.crack_sign == -1
    np.testing.assert_allclose(over.stiffness_at(0.0, -math.pi / 2), over.with_(crack=None).stiffness(0.0), atol=1e-6)
    # Nothing opens it on a vertical rotor: it is the round shaft.
    vertical = s.with_(gravity=0.0)
    assert vertical.crack_sign == 0 and not vertical.periodic
    np.testing.assert_allclose(vertical.stiffness(0.0), round_.stiffness(0.0))
    w = np.linspace(500.0, 2000.0, 7) / RPM
    np.testing.assert_allclose(synchronous_response(vertical, w), synchronous_response(round_.with_(gravity=0.0), w))


def test_a_vanishing_crack_is_the_round_shaft():
    s = CRACKED.with_(crack=Crack(position=0.4, compliance=(1e-14, 1e-15), angle=0.7))
    r = s.with_(crack=None)
    w = np.linspace(300.0, 2500.0, 9) / RPM
    Q0, V = steady_harmonics(s, w)
    R0, RV = steady_harmonics(r, w)
    np.testing.assert_allclose(Q0, R0, atol=1e-12)
    np.testing.assert_allclose(V, RV, atol=1e-12)
    np.testing.assert_allclose(whirl_modes(s, 1000 / RPM).freq_hz[:4], whirl_modes(r, 1000 / RPM).freq_hz[:4], rtol=1e-8)


def test_hill_is_converged_and_matches_the_rotating_frame(monkeypatch):
    s = CRACKED.with_(crack=Crack(position=0.4, depth=0.5, angle=0.3), bearing_a=ANISO, bearing_b=ANISO)
    for w in (500.0 / RPM, 1560.0 / RPM, 3000.0 / RPM):
        lam3, _ = rotor._hill_modes(s, w)
        monkeypatch.setattr(rotor, "HILL_ORDER", 6)
        lam6, _ = rotor._hill_modes(s, w)
        monkeypatch.setattr(rotor, "HILL_ORDER", 3)
        np.testing.assert_allclose(np.sort_complex(lam3)[:4], np.sort_complex(lam6)[:4], atol=1e-3)
    # An open crack turns with the shaft: on isotropic supports its modes are exact in the rotating frame.
    o = RotorSystem(crack=Crack(position=0.4, depth=0.5, angle=0.3, breathing=False))
    for w in (800.0 / RPM, 1550.0 / RPM):
        exact, _ = rotor._rotating_modes(o, w)
        hill, _ = rotor._hill_modes(o, w)
        np.testing.assert_allclose(np.sort_complex(exact)[:4], np.sort_complex(hill)[:4], atol=1e-6)


@pytest.mark.parametrize("supports", [Bearing(), ANISO])
def test_harmonic_balance_matches_the_time_simulation(supports):
    s = RotorSystem(crack=Crack(position=0.4, depth=0.5, angle=0.3), gravity=9.81, unbalance=2e-6,
                    bearing_a=supports, bearing_b=supports)
    for rpm in (520.0, 780.0):
        w = rpm / RPM
        Q0, V = steady_harmonics(s, [w])
        sim = RotorSimulator(s)
        sim.omega = sim.target = w
        sim.state[:] = 0.0
        sim.state[:s.n_dof] = Q0[0].real
        run(sim, 4.0)
        r = run(sim, 0.5)
        ph = r.phase
        cols = [np.ones_like(ph)] + [f(k * ph) for k in range(1, 5) for f in (np.cos, lambda x: -np.sin(x))]
        for dof in (XD, YD):
            c, *_ = np.linalg.lstsq(np.column_stack(cols), r.q[:, dof], rcond=None)
            assert c[0] == pytest.approx(Q0[0, dof].real, abs=1e-3 * abs(Q0[0, YD]))
            for k in (1, 2, 3):
                assert complex(c[2 * k - 1], c[2 * k]) == pytest.approx(V[0, k - 1, dof], abs=3e-3 * np.abs(V[0]).max())


def test_tracking_takes_the_3x_vector():
    s = CRACKED.with_(unbalance=0.0)
    w = 522.0 / RPM
    sim = RotorSimulator(s)
    sim.omega = sim.target = w
    sim.state[:] = 0.0
    sim.state[:s.n_dof] = s.static_deflection(w)
    run(sim, 4.0)
    r = run(sim, 0.6)
    sweep = TrackedSweep(rate=0.0, up=True)
    RevolutionTracker(sweep).feed(r.phase, r.omega, station_motion(s, vibration(s, r.q, r.omega)))
    assert len(sweep) >= 3
    expected = [steady_vectors(s, [w], order)[0] for order in (1, 2, 3)]
    scale = max(np.abs(e).max() for e in expected)
    for e, got in zip(expected, (sweep.vectors, sweep.vectors2, sweep.vectors3)):
        for v in got:
            np.testing.assert_allclose(v, e, atol=3e-3 * scale)
    _, v3, _ = sweep.trace(1, "orbit", order=3)
    assert np.abs(v3) == pytest.approx(18.6e-6, rel=0.01)


def test_cracked_preset_numbers():
    s = CRACKED
    assert [round(w * RPM) for w, r, z in campbell(s, 3000.0 / RPM).criticals if r > 0.1] == [1562, 1574]
    assert stability_onset(s, 3000.0 / RPM) is None
    assert growth_rate(s, 783.0 / RPM) < 0
    np.testing.assert_allclose(s.crack_compliances, [1.45e-4, 0.60e-4], rtol=0.01)
    r0 = s.with_(crack=None)
    loss = [1.0 - s.axis_system(k).stiffness(0.0)[XD, XD] / r0.stiffness(0.0)[XD, XD] for k in (False, True)]
    assert [round(x * 1000) for x in loss] == [36, 17]
    w = np.linspace(300.0, 2000.0, 17001) / RPM
    assert orbit_peak(s, 2, w) == (37.6, 783)
    assert orbit_peak(s, 3, w) == (18.6, 522)
    assert round(orbit_peak(s, 1, w)[0]) == 164 and round(orbit_peak(r0, 1, w)[0]) == 54
    assert round(orbit_peak(s.with_(unbalance=0.0), 1, w)[0]) == 107
    opposite = s.with_(crack=Crack(position=0.5, depth=0.4, angle=math.pi))
    assert round(orbit_peak(opposite, 1, w)[0]) == 52
    held = s.with_(crack=Crack(position=0.5, depth=0.4, breathing=False))
    assert orbit_peak(held, 2, w) == (75.3, 775) and orbit_peak(held, 3, w)[0] == 0.0
    assert [orbit_peak(s.with_(crack=Crack(position=0.5, depth=d)), 2, w)[0] for d in (0.2, 0.3, 0.5)] == [
        9.5, 21.8, 51.5]
    # The asymmetric shaft's stiffness has no 3X in it, so its weight drives none.
    two_pole = next(p for p in rotor_presets() if p.name.startswith("Two-pole")).system
    assert np.abs(steady_harmonics(two_pole, w[::100])[1][:, 2]).max() == 0.0


def test_run_up_and_coast_down_straddle_the_peaks():
    sim = RotorSimulator(CRACKED)
    sim.accel = 100.0 / RPM
    up, down = RunUpCoastDown(sim, 400.0 / RPM, 1000.0 / RPM).run()
    assert [round(leg.peak(1, "orbit", 3)[1] * RPM) for leg in (up, down)] == [547, 499]
    assert [round(leg.peak(1, "orbit", 2)[1] * RPM) for leg in (up, down)] == [804, 757]


def test_crack_with_shaft_mass():
    s = RotorSystem(crack=Crack(position=0.3, depth=0.4), gravity=9.81, density=STEEL, elements=10)
    assert s.n_dof > 8 and s.periodic
    # Turning slowly it sags as at rest, each instant on the stiffness it has then.
    w = 30.0 / RPM
    Q0, V = steady_harmonics(s.with_(unbalance=0.0), [w])
    q = Q0[0] + np.real(np.einsum("kn,k->n", V[0], np.exp(1j * np.arange(1, 4) * 0.4)))
    np.testing.assert_allclose(q[:8], s.rest_deflection(0.4)[:8], rtol=0, atol=2e-3 * abs(Q0[0, YD]))


def test_axis_systems():
    s = CRACKED
    soft, stiff = s.axis_system(False), s.axis_system(True)
    assert not soft.periodic and not stiff.periodic and soft.crack.compliance[0] > stiff.crack.compliance[0]
    # Along the mouth on average over a turn: the open crack's loss, halved.
    r0 = s.with_(crack=None)
    open_ = s.with_(crack=Crack(position=0.5, depth=0.4, breathing=False))
    ix, _ = s.plane_dofs
    half = 0.5 * (open_.stiffness_at(0.0, 0.0) + r0.stiffness(0.0))
    np.testing.assert_allclose(soft.stiffness(0.0)[np.ix_(ix, ix)], half[np.ix_(ix, ix)], rtol=1e-9, atol=1e-6)
    # With an asymmetry too, the crack's mean loss goes on both axes (approximately).
    both = s.with_(asymmetry=0.1)
    assert both.axis_system(True).youngs == pytest.approx(1.1 * s.youngs)
    losses = [a.with_(crack=None).stiffness(0.0) - a.stiffness(0.0) for a in (both.axis_system(k) for k in (0, 1))]
    np.testing.assert_allclose(losses[0], losses[1], rtol=1e-9, atol=1e-6)
    mean_loss = r0.stiffness(0.0) - s.stiffness(0.0)
    np.testing.assert_allclose(losses[0], mean_loss, rtol=1e-9, atol=1e-6)
