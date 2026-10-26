"""Operational modal analysis: the ambient record's spectra against H S Hᴴ, FDD and EFDD against the exact modes
and a closed-form bell, and the pitfalls the page shows (a node at the input, short blocks)."""

import math

import numpy as np
import pytest

from vib_tutorial.core.identification import auto_select, match_modes
from vib_tutorial.core.measurement import AA_CUTOFF, Response
from vib_tutorial.core.modal import TWO_PI, frf, modal_analysis
from vib_tutorial.core.model import ChainSystem
from vib_tutorial.core.oma import (
    AmbientSettings,
    Fdd,
    ambient_record,
    compare,
    cross_spectra,
    efdd,
    correlations,
    fdd,
    hammer_modes,
    harmonic_flags,
    harmonics,
    identify,
    kurtosis,
    oversampling,
    pick_peaks,
    ssi_cov,
    ssi_modes,
)

CHAIN = ChainSystem.uniform(4)
RESULT = modal_analysis(CHAIN)
FS = 30.0  # the top mode, 6 Hz, well inside the passband


def identified(settings, block, noise=0.02, top=AA_CUTOFF * FS / 2):
    rec = ambient_record(CHAIN, settings, RESULT)
    result = fdd(cross_spectra(rec.measured(noise), settings.fs, block))
    modes = identify(result, pick_peaks(result, (0.0, top), CHAIN.n))
    return modes, compare(modes, RESULT, top)


def test_the_record_has_the_spectra_of_h_s_hh():
    """White forces at every mass: G_yy = H S Hᴴ, S = 2 h sinc⁴(f h) I for unit samples h apart, linear between."""
    s = AmbientSettings(fs=FS, duration=4000.0, response=Response.DISPLACEMENT, seed=7)
    rec = ambient_record(CHAIN, s, RESULT)
    sp = cross_spectra(rec.y, s.fs, 1024)
    h = 1.0 / (s.fs * oversampling(RESULT, s.fs))
    band = (sp.freqs > 0.3) & (sp.freqs < 0.7 * s.fs / 2)
    f = sp.freqs[band]
    H = np.stack([frf(CHAIN, f, k) for k in range(CHAIN.n)], axis=-1)  # (lines, response, input)
    theory = np.einsum("fjk,fik->fji", H, H.conj()) * (2.0 * h * np.sinc(f * h) ** 4)[:, None, None]
    measured = sp.G[band]
    for j in range(CHAIN.n):
        ratio = measured[:, j, j].real / theory[:, j, j].real
        # leakage lifts the troughs a little; the averages scatter each line by ~1/√blocks
        assert np.median(ratio) == pytest.approx(1.0, abs=0.05)
    # the cross terms too: the coherence between masses' responses
    coh = measured[:, 0, 3] / np.sqrt(measured[:, 0, 0] * measured[:, 3, 3])
    exact = theory[:, 0, 3] / np.sqrt(theory[:, 0, 0] * theory[:, 3, 3])
    strong = np.abs(exact) > 0.3  # where it is small, the averages' scatter is all there is
    miss = np.abs(coh[strong] - exact[strong])
    assert np.median(miss) < 0.05 and miss.max() < 0.2


def test_fdd_finds_every_mode_and_its_shape():
    modes, rows = identified(AmbientSettings(fs=FS, duration=4000.0, seed=1), 4096)
    assert [r.mode for r in rows] == [1, 2, 3, 4]
    for r in rows:
        fe = abs(r.exact) / TWO_PI
        # modes 3 and 4 are close and heavily damped (7.7 and 9.4 %): their bells overlap
        tol = 0.01 if r.mode <= 2 else 0.06
        assert r.identified.fn_hz == pytest.approx(fe, rel=tol) and r.mac > 0.98
    first = rows[0]
    assert first.identified.fn_hz == pytest.approx(abs(first.exact) / TWO_PI, rel=0.005)
    # damping is the hardest: one record's estimate scatters by ±15 % or so
    assert first.identified.zeta == pytest.approx(-first.exact.real / abs(first.exact), rel=0.2)
    # displacement or acceleration sensors: the same shapes
    _, rows_x = identified(AmbientSettings(fs=FS, duration=4000.0, response=Response.DISPLACEMENT, seed=1), 4096)
    assert [r.mode for r in rows_x][:3] == [1, 2, 3] and all(r.mac > 0.98 for r in rows_x[:3])


def test_efdd_recovers_a_single_mode_from_its_bell():
    """A bell made exactly, |1/(ωₙ² − ω² + 2iζωₙω)|²: the autocorrelation's decay gives ζ, its crossings ωd."""
    fn, zeta = 2.0, 0.02
    freqs = np.linspace(0.0, 10.0, 20001)
    w, wn = TWO_PI * freqs, TWO_PI * fn
    s1 = 1.0 / np.abs(wn**2 - w**2 + 2j * zeta * wn * w) ** 2
    s = np.column_stack([s1, 1e-9 * np.ones_like(s1)])
    u = np.zeros((freqs.size, 2, 2), dtype=complex)
    u[:, 0, 0] = 1.0
    u[:, 1, 1] = 1.0
    mode = efdd(Fdd(freqs, s, u), int(np.argmax(s1)))
    assert mode.bell[1:].all()  # one mode everywhere: the whole band is its bell (but 0 Hz)
    assert mode.fn_hz == pytest.approx(fn, rel=1e-3) and mode.zeta == pytest.approx(zeta, rel=0.02)
    assert mode.extremes.size >= 10 and mode.crossings.size >= 10


def test_a_mode_with_a_node_at_the_input_is_missed():
    """The default chain's mode 2 has a node at m3: forces there alone don't excite it."""
    assert abs(RESULT.modes[1].shape[2]) < 1e-9
    _, rows = identified(AmbientSettings(fs=FS, duration=4000.0, input_dof=2, seed=1), 4096)
    found = {r.mode for r in rows if r.identified is not None}
    assert 1 in found and 3 in found and 2 not in found


def test_short_blocks_leak_and_overstate_the_damping():
    s = AmbientSettings(fs=FS, duration=8000.0, seed=2)
    rec = ambient_record(CHAIN, s, RESULT)
    zeta = RESULT.modes[0].zeta_modal
    err = []
    for block in (512, 8192):
        result = fdd(cross_spectra(rec.measured(0.0), s.fs, block))
        (first, *_) = identify(result, pick_peaks(result, (0.0, 2.0), 1))
        err.append(first.zeta / zeta - 1.0)
    assert err[0] > 0.5 and abs(err[1]) < 0.2


def test_a_peak_inside_a_bell_is_the_same_mode():
    s = AmbientSettings(fs=FS, duration=2000.0, seed=3)
    result = fdd(cross_spectra(ambient_record(CHAIN, s, RESULT).measured(0.02), s.fs, 4096))
    (line,) = pick_peaks(result, (0.0, 2.0), 1)
    modes = identify(result, [line, line + 2, line - 3])
    assert len(modes) == 1 and modes[0].line == line


def test_records_repeat_with_their_seed_and_refuse_too_long():
    s = AmbientSettings(fs=FS, duration=100.0, seed=5)
    a, b = ambient_record(CHAIN, s, RESULT), ambient_record(CHAIN, s, RESULT)
    assert np.array_equal(a.y, b.y) and a.y.shape == (3000, 4)
    assert not np.array_equal(a.y, ambient_record(CHAIN, AmbientSettings(fs=FS, duration=100.0, seed=6)).y)
    assert math.isclose(np.std(a.measured(0.5) - a.y) / np.std(a.y), 0.5, rel_tol=0.1)
    with pytest.raises(ValueError):
        ambient_record(CHAIN, AmbientSettings(fs=1000.0, duration=1e5))


def test_ssi_finds_the_close_damped_modes_efdd_cannot():
    """Modes 3 and 4 (7.7 and 9.4 %) overlap: EFDD's bells cut each other short; SSI fits them together."""
    s = AmbientSettings(fs=19.0, duration=1809.0, seed=2)
    y = ambient_record(CHAIN, s, RESULT).measured(0.02)
    band = (0.0, AA_CUTOFF * s.fs / 2)
    stab = ssi_cov(y, s.fs, 34, 60, band)
    rows = match_modes(ssi_modes(stab, auto_select(stab)), RESULT, band[1])
    assert [r.mode for r in rows] == [1, 2, 3, 4]
    for r in rows:
        assert abs(r.identified.fn_hz / (abs(r.exact) / TWO_PI) - 1) < 0.01 and r.mac > 0.99
        assert r.identified.zeta == pytest.approx(-r.exact.real / abs(r.exact), rel=0.15)
    # the stable column of mode 1 runs up through the orders
    assert len(stab.orders) == 30 and max(max(run, default=0) for run in stab.runs) >= 20


def test_correlations_are_the_lagged_products():
    rng = np.random.default_rng(0)
    y = rng.standard_normal((500, 2))
    R = correlations(y, 3)
    z = y - y.mean(axis=0)
    assert R[2] == pytest.approx(z[2:].T @ z[:-2] / 498)
    assert R[0] == pytest.approx(R[0].T)


def test_a_machine_is_a_harmonic_by_its_kurtosis():
    """A sine's filtered response has a kurtosis of 1.5, a mode's random one 3."""
    f1, f2 = RESULT.modes[0].fn_hz, RESULT.modes[1].fn_hz
    s = AmbientSettings(fs=FS, duration=2000.0, seed=4, harmonic_hz=0.5 * (f1 + f2), harmonic_force=1.0)
    y = ambient_record(CHAIN, s, RESULT).measured(0.02)
    result = fdd(cross_spectra(y, s.fs, 4096))
    modes = identify(result, pick_peaks(result, (0.0, AA_CUTOFF * s.fs / 2), CHAIN.n + 1))
    kurt = harmonic_flags(y, s.fs, result, modes)
    flags = harmonics([m.zeta for m in modes], kurt)
    (sine,) = [m for m, h in zip(modes, flags) if h]
    assert sine.peak_hz == pytest.approx(s.harmonic_hz, abs=2 * result.freqs[1])
    assert sine.zeta < 0.005  # almost no damping: what makes it look like a mode
    assert kurt[modes.index(sine)] == pytest.approx(1.5, abs=0.1)
    assert all(k == pytest.approx(3.0, abs=0.3) for k, h in zip(kurt, flags) if not h)
    # a mode with the sine in its band reads low too, but keeps its damping
    assert harmonics([0.05, 0.001, 0.003], [1.5, 1.5, 3.0]) == [False, True, False]


def test_kurtosis_of_a_sine_and_of_noise():
    rng = np.random.default_rng(1)
    t = np.arange(200_000) / 50.0
    shape = np.array([1.0, 0.5])
    sine = np.outer(np.sin(TWO_PI * 3.0 * t), shape)
    noise = rng.standard_normal((t.size, 2))
    assert kurtosis(sine, 50.0, shape, 3.0, 0.1) == pytest.approx(1.5, abs=0.02)
    assert kurtosis(noise, 50.0, shape, 3.0, 0.3) == pytest.approx(3.0, abs=0.1)


def test_the_hammer_test_for_comparison():
    modes, tau = hammer_modes(CHAIN, RESULT, 19.0, seed=1)
    rows = {r.mode: r for r in match_modes(modes, RESULT, AA_CUTOFF * 19.0 / 2) if r.mode}
    for k in (1, 2):
        r = rows[k]
        assert r.identified.fn_hz == pytest.approx(abs(r.exact) / TWO_PI, rel=0.005)
        assert r.identified.zeta == pytest.approx(-r.exact.real / abs(r.exact), rel=0.05)
    assert tau > 0
