"""Signals and spectra: sampling, aliasing, the DFT's lines, leakage, windows, averaging and transducers.

The *Signals & spectra* page. A `Source` is an analog signal known at every instant (two tones, a linear
chirp, or a struck oscillator's ringing, plus white noise), so sampling it at any rate shows exactly what
aliases. An optional anti-alias filter (an 8-pole Butterworth at f_s/2.56, applied to each tone by its gain,
so it is exact for tones and quasi-static for the chirp) runs before the sampler. `spectrum` takes blocks of
N samples, windows them, and averages them (Welch, with overlap) as a peak amplitude spectrum (corrected by
the window's coherent gain, so a tone on a line reads its amplitude) or a one-sided PSD (corrected by its
noise power, so noise reads its density whatever the window). `window_table` gives each window's coherent
gain, noise bandwidth, scalloping loss and highest sidelobe, worked out from the window itself.

`base_excited` is a transducer: a seismic mass on a spring and damper in a case on the structure, reading its
motion relative to the case. Well below its f_n that is proportional to the case's acceleration (an
accelerometer); well above, to its displacement (a seismometer). `mass_loaded` is what a sensor's own mass
does to the mode it measures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.signal import windows as sw

TWO_PI = 2.0 * math.pi
FILTER_RATIO = 2.56  # the anti-alias filter's corner is f_s/2.56, as analyzers set it
FILTER_POLES = 8


class Kind(Enum):
    TONES = "Two tones"
    CHIRP = "Chirp"
    RINGING = "Ringing (a struck mode)"


class Window(Enum):
    RECTANGULAR = "Rectangular (none)"
    HANN = "Hann"
    FLAT_TOP = "Flat top"


WINDOWS = tuple(Window)


@dataclass(frozen=True)
class Source:
    """An analog signal: two tones (or a chirp from f1 to f2 over `sweep` s, or a mode ringing at f1 with damping
    ratio `zeta`, struck at t = 0 to amplitude a1) plus white noise of RMS `noise` within the Nyquist band."""

    kind: Kind = Kind.TONES
    f1: float = 50.0
    a1: float = 1.0
    f2: float = 120.0
    a2: float = 0.1
    noise: float = 0.0
    zeta: float = 0.01
    sweep: float = 4.0
    seed: int = 1

    def gain(self, freq: np.ndarray | float, fs: float, filtered: bool) -> np.ndarray:
        """The anti-alias filter's gain at a frequency (1 without it)."""
        f = np.asarray(freq, dtype=float)
        if not filtered:
            return np.ones_like(f)
        return 1.0 / np.sqrt(1.0 + (f * FILTER_RATIO / fs) ** (2 * FILTER_POLES))

    def analog(self, t: np.ndarray, fs: float = math.inf, filtered: bool = False) -> np.ndarray:
        """The signal at the instants t, without the noise (which only exists at the samples)."""
        t = np.asarray(t, dtype=float)
        if self.kind is Kind.TONES:
            return (self.a1 * self.gain(self.f1, fs, filtered) * np.sin(TWO_PI * self.f1 * t)
                    + self.a2 * self.gain(self.f2, fs, filtered) * np.sin(TWO_PI * self.f2 * t))
        if self.kind is Kind.CHIRP:
            rate = (self.f2 - self.f1) / self.sweep
            tau = np.mod(t, self.sweep)  # it sweeps again and again
            return self.a1 * self.gain(self.f1 + rate * tau, fs, filtered) * np.sin(
                TWO_PI * (self.f1 * tau + 0.5 * rate * tau * tau))
        wn = TWO_PI * self.f1
        z = min(self.zeta, 0.999)
        wd = wn * math.sqrt(1.0 - z * z)
        x = self.a1 * self.gain(self.f1, fs, filtered) * np.exp(-z * wn * t) * np.sin(wd * t)
        return np.where(t >= 0, x, 0.0)

    def tones(self) -> tuple[tuple[float, float], ...]:
        """(frequency, amplitude) of the steady tones the spectrum should show."""
        if self.kind is not Kind.TONES:
            return ()
        return tuple((f, a) for f, a in ((self.f1, self.a1), (self.f2, self.a2)) if a > 0)


def sample(source: Source, fs: float, n: int, filtered: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """(t, x): n samples at the rate fs, the noise seeded."""
    t = np.arange(n) / fs
    x = source.analog(t, fs, filtered)
    if source.noise > 0:
        x = x + source.noise * np.random.default_rng(source.seed).standard_normal(n)
    return t, x


def apparent(freq: float, fs: float) -> float:
    """Where a tone at `freq` shows after sampling at fs: folded into 0 … f_s/2."""
    f = math.fmod(abs(freq), fs)
    return fs - f if f > fs / 2 else f


def window(kind: Window, n: int) -> np.ndarray:
    """The window's n points, periodic (the DFT's own period), as analyzers use them."""
    if kind is Window.HANN:
        return sw.hann(n, sym=False)
    if kind is Window.FLAT_TOP:
        return sw.flattop(n, sym=False)
    return np.ones(n)


@dataclass(frozen=True)
class WindowProps:
    coherent_gain: float  # mean of w: a tone on a line reads this fraction of its amplitude
    enbw: float  # the equivalent noise bandwidth, in lines: N Σw²/(Σw)²
    scalloping_db: float  # the loss for a tone half-way between lines
    sidelobe_db: float  # the highest sidelobe, below the main lobe
    main_lobe: float  # the main lobe's half-width to its first null, in lines


def window_props(kind: Window, n: int = 1024) -> WindowProps:
    w = window(kind, n)
    cg = float(w.mean())
    enbw = float(n * np.sum(w * w) / np.sum(w) ** 2)
    k = np.arange(n)
    half = abs(np.sum(w * np.exp(-1j * math.pi * k / n))) / np.sum(w)
    bins, level = kernel(kind, 12.0, n=n)
    level_lin = 10 ** (level / 20)
    # first null: the first local minimum away from 0
    right = bins >= 0
    b, lv = bins[right], level_lin[right]
    null = next(i for i in range(1, lv.size - 1) if lv[i] <= lv[i - 1] and lv[i] <= lv[i + 1])
    side = float(np.max(level[right][null:]))
    return WindowProps(cg, enbw, 20 * math.log10(half), side, float(b[null]))


def kernel(kind: Window, span: float, n: int = 256, pad: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """The window's spectrum about a line, |W(f)|/W(0) in dB, against the offset in lines (−span … span)."""
    w = window(kind, n)
    W = np.fft.fft(w, n * pad)
    W = np.abs(np.fft.fftshift(W)) / np.sum(w)
    bins = (np.arange(n * pad) - n * pad // 2) / pad
    keep = np.abs(bins) <= span
    with np.errstate(divide="ignore"):
        return bins[keep], 20 * np.log10(np.maximum(W[keep], 1e-12))


@dataclass(frozen=True)
class Spectrum:
    freqs: np.ndarray
    amplitude: np.ndarray  # peak amplitude, averaged as power: √⟨|X|²⟩ scaled
    psd: np.ndarray  # one-sided, units²/Hz
    blocks: int
    df: float


def blocks_needed(block: int, averages: int, overlap: float) -> int:
    """Samples for `averages` blocks of `block` samples overlapping by `overlap` (0 … <1)."""
    step = max(1, int(round(block * (1.0 - overlap))))
    return block + (averages - 1) * step


def spectrum(x: np.ndarray, fs: float, block: int, kind: Window, overlap: float = 0.0) -> Spectrum:
    """Averaged spectra of x over blocks of `block` samples (Welch): the peak amplitude and the PSD."""
    w = window(kind, block)
    step = max(1, int(round(block * (1.0 - overlap))))
    starts = range(0, x.size - block + 1, step)
    power = np.zeros(block // 2 + 1)
    count = 0
    for s in starts:
        X = np.fft.rfft(w * x[s:s + block])
        power += np.abs(X) ** 2
        count += 1
    if count == 0:
        raise ValueError("the record is shorter than one block")
    power /= count
    one_sided = np.full(power.size, 2.0)
    one_sided[0] = 1.0
    if block % 2 == 0:
        one_sided[-1] = 1.0
    amplitude = np.sqrt(power) * one_sided / np.sum(w)  # 2|X|/Σw: a tone on a line reads its amplitude
    psd = power * one_sided / (fs * np.sum(w * w))
    return Spectrum(np.fft.rfftfreq(block, 1.0 / fs), amplitude, psd, count, fs / block)


def spectrogram(x: np.ndarray, fs: float, block: int, kind: Window,
                overlap: float = 0.75) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(times of the blocks' middles, frequencies, peak amplitude in dB) of each block: a short-time spectrum."""
    w = window(kind, block)
    step = max(1, int(round(block * (1.0 - overlap))))
    starts = np.arange(0, x.size - block + 1, step)
    frames = np.stack([w * x[s:s + block] for s in starts])
    A = 2.0 * np.abs(np.fft.rfft(frames, axis=1)) / np.sum(w)
    with np.errstate(divide="ignore"):
        level = 20 * np.log10(np.maximum(A, 1e-12))
    return (starts + block / 2) / fs, np.fft.rfftfreq(block, 1.0 / fs), level


def nearest_line_reading(spec: Spectrum, freq: float) -> tuple[float, float]:
    """(the line nearest an apparent frequency, the amplitude the largest of it and its neighbours reads)."""
    i = int(round(freq / spec.df))
    i = min(max(i, 0), spec.freqs.size - 1)
    lo, hi = max(i - 1, 0), min(i + 2, spec.freqs.size)
    j = lo + int(np.argmax(spec.amplitude[lo:hi]))
    return float(spec.freqs[j]), float(spec.amplitude[j])


# --------------------------------------------------------------------------------------------- transducers
def base_excited(r: np.ndarray, zeta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A seismic sensor on a moving case at r = f/f_n: (accelerometer, seismometer, phase).

    The mass's motion relative to the case, z, over the case's: z/y = r²/(1 − r² + 2iζr). As an accelerometer
    it reads ω_n² z against the case's acceleration −ω²y, a ratio 1/(1 − r² + 2iζr): flat (1) well below f_n.
    As a seismometer it reads z against y, a ratio r²/(…): flat (1) well above. The phase (degrees) is the
    accelerometer's lag; the seismometer's is 180° less.
    """
    r = np.asarray(r, dtype=float)
    d = 1.0 - r * r + 2j * zeta * r
    acc = 1.0 / d
    return np.abs(acc), np.abs(r * r / d), -np.degrees(np.angle(acc))


def usable_band(zeta: float, tolerance: float = 0.05, accelerometer: bool = True) -> float:
    """The frequency ratio r up to which (accelerometer) or from which (seismometer) the reading stays within
    ± `tolerance` of the truth."""
    r = np.geomspace(1e-3, 1e3, 200001)
    acc, seis, _ = base_excited(r, zeta)
    if accelerometer:
        bad = np.abs(acc - 1.0) > tolerance
        return float(r[np.argmax(bad)]) if bad.any() else math.inf
    bad = np.abs(seis - 1.0) > tolerance
    return float(r[len(r) - np.argmax(bad[::-1])]) if bad.any() else 0.0


def mass_loaded(natural_hz: float, modal_mass: float, sensor_mass: float) -> float:
    """A mode's frequency with a sensor of `sensor_mass` added where the mode's modal mass (scaled to a unit
    motion there) is `modal_mass`: f √(m/(m + m_s)), its stiffness unchanged."""
    return natural_hz * math.sqrt(modal_mass / (modal_mass + sensor_mass))
