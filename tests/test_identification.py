import numpy as np
import pytest

from vib_tutorial.core import (
    Acquisition,
    ChainSystem,
    Excitation,
    FrfEstimator,
    MeasurementSettings,
    Processing,
    Window,
    frf,
    modal_analysis,
)
from vib_tutorial.core.identification import (
    auto_select,
    circle_fit,
    half_power_damping,
    lscf,
    lsfd,
    match_modes,
    peak_picking,
    pole_from,
)

SYSTEM = ChainSystem.uniform(4)
RESULT = modal_analysis(SYSTEM)
EXACT = [m.damped for m in RESULT.modes]


def measured(system=SYSTEM, window=Window.RECTANGULAR, **settings):
    s = dict(excitation=Excitation.PERIODIC_RANDOM, input_dof=system.n - 1, fs=19.0, averages=3, seed=4)
    s.update(settings)
    acq = Acquisition(system, MeasurementSettings(**s))
    while not acq.done:
        acq.step()
    return acq, FrfEstimator(acq, Processing(window=window)).estimate()


def by_mode(modes, result=RESULT, f_max=10.0):
    return {r.mode: r for r in match_modes(modes, result, f_max) if r.mode and r.identified}


def test_half_power_damping_of_a_single_mode():
    zeta, fn = 0.02, 3.0
    f = np.linspace(2.0, 4.0, 20001)
    w, wn = 2 * np.pi * f, 2 * np.pi * fn
    mag = np.abs(1.0 / (wn**2 - w**2 + 2j * zeta * wn * w))
    assert half_power_damping(f, mag, int(np.argmax(mag))) == pytest.approx(zeta, rel=0.01)


def test_lscf_and_lsfd_recover_every_mode_from_clean_data():
    acq, est = measured()
    band = (0.0, acq.settings.band)
    stab = lscf(est.freqs, est.H, band, 30)
    poles = [stab.pole(o, i) for o, i in auto_select(stab)]
    ident = lsfd(est.freqs, est.H, band, poles)
    found = by_mode(ident.modes)
    assert sorted(found) == [1, 2, 3, 4] and len(ident.modes) == 4
    for r, m in found.items():
        exact = EXACT[r - 1]
        assert m.identified.fn_hz == pytest.approx(exact.fn_hz, rel=1e-4)
        assert m.identified.zeta == pytest.approx(exact.zeta, rel=1e-3)
        assert m.mac > 0.9999
    # The fitted modal model (with residuals) rebuilds the measured FRF.
    keep = (est.freqs > 0) & (est.freqs <= band[1])
    fit = ident.synthesize(est.freqs[keep])
    assert np.abs(fit - est.H[keep]).max() < 1e-3 * np.abs(est.H[keep]).max()


def test_lsfd_with_the_exact_poles_rebuilds_the_exact_frf():
    f = np.linspace(0.05, 8.0, 400)
    H = frf(SYSTEM, f, 3)
    ident = lsfd(f, H, (0.0, 8.0), [m.eigenvalue for m in EXACT])
    np.testing.assert_allclose(ident.synthesize(f), H, atol=1e-9 * np.abs(H).max())
    np.testing.assert_allclose(ident.upper, 0.0, atol=1e-9 * np.abs(H).max())


def test_circle_fit_beats_peak_picking_between_lines():
    # A coarse line spacing: peak picking can only land on a line, the circle fit interpolates.
    acq, est = measured(block=256, averages=2)
    band = (0.0, acq.settings.band)
    peak = by_mode(peak_picking(est.freqs, est.H, band, 4).modes)
    circle = by_mode(circle_fit(est.freqs, est.H, band, 4).modes)
    exact = EXACT[1]  # mode 2: a few lines across its peak
    err = {name: abs(found[2].identified.fn_hz / exact.fn_hz - 1) for name, found in (("peak", peak), ("circle", circle))}
    assert err["circle"] < 0.2 * err["peak"]
    assert abs(circle[2].identified.zeta / exact.zeta - 1) < 0.1  # the neighbouring modes bend the circle a little
    assert circle[2].mac > 0.99


def test_exponential_window_damping_is_removed():
    acq, est = measured(excitation=Excitation.IMPACT, block=256, averages=2, window=Window.RECTANGULAR)
    p = Processing(window=Window.FORCE_EXPONENTIAL, exp_end=0.01)
    est = FrfEstimator(acq, p).estimate()
    band = (0.0, acq.settings.band)
    stab = lscf(est.freqs, est.H, band, 30)
    ident = lsfd(est.freqs, est.H, band, [stab.pole(o, i) for o, i in auto_select(stab)])
    sigma = 1.0 / p.exp_tau(acq.settings)
    raw = by_mode(ident.modes)
    fixed = by_mode(ident.shifted(sigma).modes)
    exact = EXACT[0]
    assert raw[1].identified.zeta > 1.5 * exact.zeta  # the window's damping
    assert fixed[1].identified.zeta == pytest.approx(exact.zeta, rel=0.02)


def test_a_node_at_the_force_hides_its_mode():
    # m3 does not move in mode 2 of the uniform 4-mass chain: with the force there, no method sees it.
    acq, est = measured(input_dof=2)
    band = (0.0, acq.settings.band)
    stab = lscf(est.freqs, est.H, band, 30)
    ident = lsfd(est.freqs, est.H, band, [stab.pole(o, i) for o, i in auto_select(stab)])
    assert 2 not in by_mode(ident.modes)
    assert {1, 3, 4} <= set(by_mode(ident.modes))


def test_match_modes_reports_missed_and_extra():
    ident = lsfd(np.linspace(0.1, 8, 200), frf(SYSTEM, np.linspace(0.1, 8, 200), 3), (0, 8),
                 [EXACT[0].eigenvalue, pole_from(7.0, 0.05)])
    rows = match_modes(ident.modes, RESULT, 8.0)
    assert [(r.mode, r.identified is not None) for r in rows] == [(1, True), (2, False), (3, False), (4, False), (None, True)]


def test_lscf_order_is_limited_by_the_lines_in_the_band():
    f = np.linspace(0.5, 7.0, 12)  # a coarse stepped sine
    stab = lscf(f, frf(SYSTEM, f, 3), (0.0, 7.0), 30)
    assert max(stab.orders) <= 11


def test_circle_fit_starts_inside_its_bounds_for_an_overdamped_noise_peak():
    # Half-power damping above critical (a noise "peak") used to start the fit outside zeta <= 1.
    from vib_tutorial.core.identification import _circle_parameters

    w = np.linspace(1.0, 3.0, 9)
    h = 1.0 / (4.0 - w**2 + 2j * 0.3 * 2.0 * w)
    wr, zeta = _circle_parameters(w, h, 2.0, 2.5)
    assert 1.0 <= wr <= 3.0 and 0.0 < zeta <= 1.0
