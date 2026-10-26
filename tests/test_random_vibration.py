"""Random vibration: the input's synthesis, H S Hᴴ, the RMS, Miles' equation, and the time simulation against them."""

import math

import numpy as np
import pytest

from vib_tutorial.core.modal import TWO_PI, modal_analysis
from vib_tutorial.core.model import ChainSystem
from vib_tutorial.core.random_vibration import (
    G,
    Input,
    Output,
    Quantity,
    RandomInput,
    predict,
    rms_scatter,
    sample_rate,
    simulate,
    suggested_block,
    suggested_duration,
    synthesize,
    transfer,
    welch,
)

CHAIN = ChainSystem.uniform(4)
RESULT = modal_analysis(CHAIN)


def test_synthesized_noise_has_the_psd_asked_for():
    """As a piecewise-linear signal: its samples' PSD is G / sinc⁴, which the linear interpolation takes back."""
    fs, n = 100.0, 1 << 18
    level = 2.5

    def psd(f):
        return np.where((f >= 5.0) & (f <= 30.0), level, 0.0)

    u = synthesize(np.random.default_rng(3), n, fs, psd)[:, 0]
    f, g = welch(u, fs, 1024)
    band = (f > 6) & (f < 29)
    expected = level / np.sinc(f[band] / fs)**4
    assert np.mean(g[band] / expected) == pytest.approx(1.0, abs=0.01)
    assert np.mean(g[(f > 35) | (f < 4)]) < 1e-3 * level
    # its mean square, with the samples' sinc⁴ boost: ∫ G / sinc⁴ df
    fine = np.linspace(5.0, 30.0, 2001)
    assert np.mean(u**2) == pytest.approx(np.trapezoid(level / np.sinc(fine / fs)**4, fine), rel=0.02)


def test_one_mass_matches_the_closed_forms():
    """A single mass on a spring: under white force σ_x² = G / 4kc; on a white base the absolute acceleration's
    σ² = (π/2) f_n Q (1 + 4ζ²) G, Miles' equation and its damping term."""
    m, k, c = 2.0, 800.0, 4.0
    sdof = ChainSystem(np.array([m]), np.array([k]), np.array([c]))
    res = modal_analysis(sdof)
    fn = math.sqrt(k / m) / TWO_PI
    zeta = c / (2 * math.sqrt(k * m))
    force = RandomInput(Input.FORCE_ONE, 3.0, 0.0, 400 * fn)
    p = predict(sdof, force, Output(Quantity.DISPLACEMENT, 0), res)
    assert p.rms == pytest.approx(math.sqrt(3.0 / (4 * k * c)), rel=2e-3)
    assert p.miles_rms == pytest.approx(p.rms, rel=2e-3)  # one mode, white: Miles is exact for the displacement

    base = RandomInput(Input.BASE, 0.04, 0.0, 400 * fn)
    p = predict(sdof, base, Output(Quantity.ACCELERATION, 0), res)
    miles = math.sqrt(0.5 * math.pi * fn / (2 * zeta) * 0.04)  # g
    assert p.miles_rms == pytest.approx(miles, rel=1e-9)
    assert p.rms == pytest.approx(miles * math.sqrt(1 + 4 * zeta**2), rel=2e-3)
    # the spring carries k times the relative displacement
    assert predict(sdof, base, Output(Quantity.SPRING, 0), res).rms == pytest.approx(
        k * predict(sdof, base, Output(Quantity.DISPLACEMENT, 0), res).rms, rel=1e-12)


def test_transfer_limits():
    f = np.array([1e-6, 1e4])
    t = transfer(CHAIN, RandomInput(Input.BASE), Output(Quantity.ACCELERATION, 3), f)[:, 0]
    assert t[0] == pytest.approx(1.0 / G, rel=1e-6)  # slow: the chain moves with the base
    assert abs(t[1]) < 1e-3 / G  # fast: the masses stay put while the base shakes under them
    t = transfer(CHAIN, RandomInput(Input.FORCE_ONE, mass=2), Output(Quantity.ACCELERATION, 2), f)[:, 0]
    assert t[1] == pytest.approx(1.0 / CHAIN.masses[2] / G, rel=1e-3)  # the mass line
    # the static receptance: every spring below a force carries it
    t = transfer(CHAIN, RandomInput(Input.FORCE_ONE, mass=2), Output(Quantity.SPRING, 1), f)[:, 0]
    assert t[0] == pytest.approx(1.0, rel=1e-6)


@pytest.mark.parametrize("inp", [RandomInput(Input.BASE, 0.01, 0.0, 10.0),
                                 RandomInput(Input.FORCE_ONE, 1.0, 0.0, 10.0, mass=3),
                                 RandomInput(Input.FORCE_ALL, 1.0, 2.0, 8.0)])
def test_the_simulation_matches_h_s_hh(inp):
    duration = suggested_duration(inp, RESULT)
    rec = simulate(CHAIN, inp, duration, seed=7, result=RESULT)
    assert rec.fs == sample_rate(inp) and rec.duration == pytest.approx(duration, rel=1e-3)
    block = suggested_block(inp, RESULT, rec.x.shape[0])
    for out in (Output(Quantity.ACCELERATION, 3), Output(Quantity.DISPLACEMENT, 0), Output(Quantity.SPRING, 1)):
        p = predict(CHAIN, inp, out, RESULT)
        y = rec.output(out)
        scatter = rms_scatter(p, rec.duration)
        assert np.std(y) == pytest.approx(p.rms, rel=4 * scatter)
        # Welch's estimate, averaged over the band's lines, against the exact PSD there
        f, g = welch(y, rec.fs, block)
        band = (f > inp.f_lo + 0.2) & (f < inp.f_hi - 0.2)
        exact = np.interp(f[band], p.f, p.psd)
        assert np.trapezoid(g[band], f[band]) == pytest.approx(np.trapezoid(exact, f[band]), rel=4 * scatter)
    # the input's PSD is the level asked for
    f, g = welch(rec.u[:, 0], rec.fs, block)
    band = (f > inp.f_lo + 0.2) & (f < inp.f_hi - 0.2)
    assert np.mean(g[band]) == pytest.approx(inp.si_level, rel=0.03)


def test_miles_per_mode():
    """On a base, the chain's resonant responses come out within a few per cent of the exact RMS by Miles' equation
    mode by mode; the force's mass line it misses."""
    base = RandomInput(Input.BASE, 0.01, 0.0, 10.0)
    for out in (Output(Quantity.ACCELERATION, 3), Output(Quantity.DISPLACEMENT, 3), Output(Quantity.SPRING, 0)):
        p = predict(CHAIN, base, out, RESULT)
        assert [m for m, *_ in p.miles] == [1, 2, 3, 4]
        assert p.miles_rms == pytest.approx(p.rms, rel=0.01)
    # a force's own mass's acceleration has a mass line, f/m above the modes, not resonant: Miles misses it,
    # more as the band goes higher
    short, wide = (predict(CHAIN, RandomInput(Input.FORCE_ONE, 1.0, 0.0, top, mass=3),
                           Output(Quantity.ACCELERATION, 3), RESULT) for top in (10.0, 30.0))
    assert short.miles_rms == pytest.approx(wide.miles_rms)
    assert short.miles_rms < 0.85 * short.rms and wide.miles_rms < 0.65 * wide.rms
    # a band that leaves out modes 3 and 4: Miles counts only the modes inside it
    p = predict(CHAIN, RandomInput(Input.BASE, 0.01, 0.0, 4.0), Output(Quantity.ACCELERATION, 3), RESULT)
    assert [m for m, *_ in p.miles] == [1, 2]


def test_suggested_record():
    inp = RandomInput(Input.BASE, 0.01, 0.0, 10.0)
    fs = sample_rate(inp)
    duration = suggested_duration(inp, RESULT)
    samples = int(round(duration * fs))
    block = suggested_block(inp, RESULT, samples)
    width = 2 * RESULT.modes[0].zeta_modal * RESULT.modes[0].fn_hz
    assert fs / block <= width / 4  # four lines across mode 1's half-power width
    assert samples == 16 * block
    with pytest.raises(ValueError):
        simulate(CHAIN, inp, 1e6)


def test_rice_peak_distribution():
    from vib_tutorial.core.random_vibration import peak_pdf

    eta = np.linspace(-8, 8, 8001)
    for alpha in (0.0, 0.3, 0.7, 0.99, 1.0):
        assert np.trapezoid(peak_pdf(eta, alpha), eta) == pytest.approx(1.0, abs=1e-4)
    assert peak_pdf(eta, 1.0) == pytest.approx(np.where(eta > 0, eta * np.exp(-eta**2 / 2), 0.0))  # Rayleigh
    assert peak_pdf(eta, 0.0) == pytest.approx(np.exp(-eta**2 / 2) / math.sqrt(2 * math.pi))  # Gaussian


def test_davenport_and_vanmarcke():
    from vib_tutorial.core.random_vibration import davenport, vanmarcke

    # Davenport's formula is the Poisson crossings' expected largest value, to its asymptotic accuracy
    r = np.linspace(0, 12, 20001)
    for cycles in (1e2, 1e4, 1e6):
        poisson = np.trapezoid(1 - np.exp(-cycles * np.exp(-r**2 / 2)), r)
        assert davenport(cycles) == pytest.approx(poisson, rel=0.03)
        # clumped crossings (a narrow band) give a lower largest peak; a broad band, nearly Poisson's
        assert vanmarcke(cycles, 0.1) < vanmarcke(cycles, 0.9) < poisson + 0.05
    assert davenport(1e6) == pytest.approx(5.37, abs=0.01) and davenport(1.0) == 0.0


@pytest.mark.parametrize(("inp", "out"), [(RandomInput(Input.BASE, 0.01, 0.0, 10.0), Output(Quantity.SPRING, 0)),
                                          (RandomInput(Input.FORCE_ONE, 1.0, 0.0, 10.0, mass=3),
                                           Output(Quantity.ACCELERATION, 3))])
def test_the_records_statistics_match_rice(inp, out):
    from vib_tutorial.core.random_vibration import (
        LEVELS,
        gaussian_exceedance,
        peak_pdf,
        record_statistics,
        rice,
        segment_maxima,
    )

    rec = simulate(CHAIN, inp, suggested_duration(inp, RESULT), seed=3, result=RESULT)
    p = predict(CHAIN, inp, out, RESULT)
    r = rice(p)
    y = rec.output(out)
    st = record_statistics(y, rec.fs)
    scatter = rms_scatter(p, rec.duration)
    assert st.sigma == pytest.approx(r.sigma, rel=4 * scatter)
    assert st.skewness == pytest.approx(0.0, abs=0.05) and st.kurtosis == pytest.approx(3.0, abs=0.15)
    for k, measured in zip(LEVELS, st.exceedance):
        assert measured == pytest.approx(gaussian_exceedance(k), rel=0.35 if k == 3 else 0.1)
    assert st.nu0 == pytest.approx(r.nu0, rel=0.04) and st.nu_peak == pytest.approx(r.nu_peak, rel=0.04)
    h, edges = np.histogram(st.peaks * st.sigma / r.sigma, bins=32, range=(-4, 4), density=True)
    centres = 0.5 * (edges[1:] + edges[:-1])
    assert np.max(np.abs(h - peak_pdf(centres, r.alpha))) < 0.07
    # the largest value in each 60 s: Vanmarcke within a few per cent, Davenport above it
    largest = np.mean(segment_maxima(y, rec.fs, 60.0)) / r.sigma
    assert r.clumped_peak_factor(60.0) == pytest.approx(largest, rel=0.04)
    assert r.peak_factor(60.0) > r.clumped_peak_factor(60.0)
