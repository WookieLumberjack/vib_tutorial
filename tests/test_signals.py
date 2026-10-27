"""The Signals & spectra page's model (core/signals.py): scaling, windows, aliasing and transducers."""

import math

import numpy as np
import pytest
from scipy.signal import welch

from vib_tutorial.core.signals import (
    WINDOWS,
    Kind,
    Source,
    Window,
    apparent,
    base_excited,
    mass_loaded,
    nearest_line_reading,
    sample,
    spectrogram,
    spectrum,
    usable_band,
    window_props,
)


def test_a_tone_on_a_line_reads_its_amplitude_with_every_window():
    s = Source(f1=50.0, a1=0.7, a2=0.0)
    _, x = sample(s, 1024.0, 1024)
    for w in WINDOWS:
        sp = spectrum(x, 1024.0, 1024, w)
        assert nearest_line_reading(sp, 50.0) == pytest.approx((50.0, 0.7), rel=1e-9)


def test_window_properties_are_the_textbook_ones():
    rect, hann, flat = (window_props(w) for w in (Window.RECTANGULAR, Window.HANN, Window.FLAT_TOP))
    assert (rect.enbw, hann.enbw) == pytest.approx((1.0, 1.5), rel=1e-9)
    assert rect.scalloping_db == pytest.approx(-3.92, abs=0.01)
    assert hann.scalloping_db == pytest.approx(-1.42, abs=0.01)
    assert abs(flat.scalloping_db) < 0.02
    assert (rect.sidelobe_db, hann.sidelobe_db) == pytest.approx((-13.3, -31.5), abs=0.1)
    assert flat.sidelobe_db < -90
    assert (rect.main_lobe, hann.main_lobe, flat.main_lobe) == (1.0, 2.0, 5.0)


def test_a_tone_between_lines_reads_low_by_the_scalloping_loss():
    s = Source(f1=50.5, a1=1.0, a2=0.0)
    _, x = sample(s, 1024.0, 1024)
    for w in WINDOWS:
        _, reads = nearest_line_reading(spectrum(x, 1024.0, 1024, w), 50.5)
        assert 20 * math.log10(reads) == pytest.approx(window_props(w).scalloping_db, abs=0.05)


@pytest.mark.parametrize("w", list(WINDOWS))
def test_psd_matches_scipy_welch_and_reads_the_noise_density(w):
    s = Source(a1=0.0, a2=0.0, noise=0.5, seed=3)
    fs, n = 1000.0, 256
    _, x = sample(s, fs, 64 * n)
    sp = spectrum(x, fs, n, w, overlap=0.5)
    name = {Window.RECTANGULAR: "boxcar", Window.HANN: "hann", Window.FLAT_TOP: "flattop"}[w]
    f, ref = welch(x, fs, window=name, nperseg=n, noverlap=n // 2, detrend=False)
    assert sp.freqs == pytest.approx(f)
    assert sp.psd == pytest.approx(ref, rel=1e-9)
    assert np.mean(sp.psd[1:-1]) == pytest.approx(0.25 / (fs / 2), rel=0.05)
    assert np.sum(sp.psd) * sp.df == pytest.approx(np.mean(x ** 2), rel=0.05)  # Parseval


def test_aliasing_and_the_filter():
    assert apparent(900.0, 1024.0) == 124.0
    assert apparent(1100.0, 1024.0) == 76.0
    assert apparent(300.0, 1024.0) == 300.0
    s = Source(f1=900.0, a1=1.0, a2=0.0)
    for filtered, expect in ((False, 1.0), (True, 0.00152)):
        _, x = sample(s, 1024.0, 1024, filtered)
        _, reads = nearest_line_reading(spectrum(x, 1024.0, 1024, Window.HANN), 124.0)
        assert reads == pytest.approx(expect, rel=0.01)
    assert s.gain(400.0, 1024.0, True) == pytest.approx(1 / math.sqrt(2))


def test_the_chirp_shows_in_the_spectrogram_where_it_is():
    s = Source(Kind.CHIRP, f1=10.0, a1=1.0, f2=400.0, sweep=4.0)
    fs, n = 1024.0, 256
    _, x = sample(s, fs, 4096)
    t, f, level = spectrogram(x, fs, n, Window.HANN)
    loudest = f[np.argmax(level, axis=1)]
    assert loudest == pytest.approx(10.0 + (400.0 - 10.0) * t / 4.0, abs=2 * fs / n)


def test_base_excited_sensor():
    r = np.array([1e-3, 1e3])
    acc, seis, _ = base_excited(r, 0.01)
    assert acc[0] == pytest.approx(1.0, rel=1e-5) and seis[1] == pytest.approx(1.0, rel=1e-5)
    assert base_excited(np.array([1.0]), 0.05)[0][0] == pytest.approx(10.0)
    assert base_excited(np.array([1.0]), 0.05)[2][0] == pytest.approx(90.0)
    up = usable_band(0.01)
    assert base_excited(np.array([up * 0.999]), 0.01)[0][0] == pytest.approx(1.05, abs=1e-3)
    assert mass_loaded(200.0, 0.5, 0.03) == pytest.approx(200 * math.sqrt(0.5 / 0.53))
