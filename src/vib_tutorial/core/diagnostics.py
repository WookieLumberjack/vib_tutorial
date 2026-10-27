"""Machinery diagnostics: order tracking under a run-up, gear-mesh sidebands, bearing defects by envelope
analysis, and the cepstrum.

The *Machinery diagnostics* page. Its own model: an accelerometer on a gearbox's casing, its signal synthesized.

- A steady `Machine`: the shaft's 1X and 2X, the pinion's mesh (`teeth` × the shaft's speed, three harmonics),
  modulated by a gear fault (`GearFault`: an eccentric pinion modulates it once a turn, a cracked tooth gives a
  short kick once a turn), and a rolling bearing's defect (`BearingFault`): an impact each time a ball meets it, at
  the `Bearing`'s defect frequencies, each ringing the casing's resonance (impulse invariance: a sample-aligned
  impulse through the two-pole filter is the resonance's exact response), the spacing jittered by the balls'
  slip, the impacts' size modulated by the load zone for an inner-race or ball defect. Then white noise.
- A run-up (`run_up`): the shaft's speed rising linearly, its unbalance (∝ Ω²) and misalignment (2X) driving one
  casing mode, stepped by the exact FOH update (`foh_discretize`, as a transfer function through `lfilter`); the
  tachometer's angle is exact. `order_map` resamples the record at equal angles (computed order tracking: a
  cubic spline of the record at the angles' times) and takes each block's spectrum, so an order stays on one
  line; `frequency_map` is the plain short-time spectrum against the speed, where it fans out.
- `envelope_spectrum`: band-pass round the resonance, the Hilbert envelope, its spectrum: the impacts' rate shows
  up there though it is buried in the plain spectrum. `cepstrum`: the inverse transform of the log power
  spectrum, where a family of lines spaced Δf becomes one peak at quefrency 1/Δf.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass

import numpy as np
import scipy.interpolate
import scipy.signal
import scipy.stats

from .signals import Window, spectrum
from .simulator import foh_discretize

TWO_PI = 2.0 * math.pi
FS = 20480.0  # Hz, the steady record's sample rate
DURATION = 4.0  # s
BLOCK = 8192  # the spectra's block (2.5 Hz lines at FS)
MESH_HARMONICS = (1.0, 0.4, 0.15)  # m/s², the mesh's harmonics
SHAFT = (0.3, 0.1)  # m/s², 1X and 2X
RUN_FS = 2048.0  # Hz, the run-up's sample rate
PER_REV = 32  # samples per revolution after resampling (orders up to 16)
REVS = 8  # revolutions per order block (1/8-order lines)


class GearFault(enum.Enum):
    NONE = "None"
    ECCENTRIC = "Eccentric pinion"
    CRACK = "Cracked tooth"


class BearingFault(enum.Enum):
    NONE = "None"
    OUTER = "Outer race"
    INNER = "Inner race"
    BALL = "Ball"


@dataclass(frozen=True)
class Bearing:
    """A ball bearing's geometry (the defaults: a 6205, 9 balls)."""

    balls: int = 9
    ball: float = 7.94e-3  # m, the balls' diameter
    pitch: float = 39.04e-3  # m, the pitch circle's
    contact_deg: float = 0.0

    def frequencies(self, shaft_hz: float) -> dict[str, float]:
        """The defect frequencies with the inner race turning at `shaft_hz` and the outer one still: FTF (the
        cage), BPFO and BPFI (a ball over a point on the outer or inner race), BSF (a ball's spin: a point on it
        meets a race once a spin)."""
        r = self.ball / self.pitch * math.cos(math.radians(self.contact_deg))
        return {"FTF": shaft_hz / 2 * (1 - r), "BPFO": self.balls * shaft_hz / 2 * (1 - r),
                "BPFI": self.balls * shaft_hz / 2 * (1 + r),
                "BSF": self.pitch / (2 * self.ball) * shaft_hz * (1 - r * r)}


@dataclass(frozen=True)
class Machine:
    speed: float = 29.95  # Hz, the shaft's (1797 rpm)
    teeth: int = 27
    gear: GearFault = GearFault.NONE
    gear_severity: float = 0.3  # the modulation's depth
    bearing_fault: BearingFault = BearingFault.NONE
    impact: float = 3.0  # m/s², each impact's peak response
    bearing: Bearing = Bearing()
    resonance: float = 3000.0  # Hz, the casing's mode the impacts ring
    zeta: float = 0.05
    slip: float = 0.01  # the impacts' spacing scatters by this share
    noise: float = 0.2  # m/s² RMS
    seed: int = 1

    @property
    def mesh_hz(self) -> float:
        return self.teeth * self.speed

    def defect_hz(self) -> float:
        """The rate the impacts come at: BPFO, BPFI, or twice BSF (a ball defect meets both races)."""
        f = self.bearing.frequencies(self.speed)
        return {BearingFault.OUTER: f["BPFO"], BearingFault.INNER: f["BPFI"],
                BearingFault.BALL: 2 * f["BSF"]}.get(self.bearing_fault, 0.0)

    def modulation_hz(self) -> float:
        """What modulates the impacts: the shaft for an inner-race defect, the cage for a ball's."""
        f = self.bearing.frequencies(self.speed)
        return {BearingFault.INNER: self.speed, BearingFault.BALL: f["FTF"]}.get(self.bearing_fault, 0.0)


def gear_modulation(machine: Machine, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(amplitude factor, phase) of the mesh at the shaft's angles: once a turn for an eccentric pinion, a pulse
    a tooth wide for a cracked tooth (at θ = 0)."""
    s = machine.gear_severity
    if machine.gear is GearFault.ECCENTRIC:
        return 1 + s * np.cos(theta), 0.5 * s * np.sin(theta)
    if machine.gear is GearFault.CRACK:
        width = 0.5 * TWO_PI / machine.teeth
        wrapped = (theta + math.pi) % TWO_PI - math.pi
        pulse = np.exp(-0.5 * (wrapped / width) ** 2)
        return 1 + 3 * s * pulse, s * pulse
    return np.ones_like(theta), np.zeros_like(theta)


def ring_filter(resonance: float, zeta: float, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """(b, a) of the resonance's impulse response e^{−ζωt} sin(ω_d t) sampled (impulse invariance), scaled so
    its largest sample is about 1."""
    wn = TWO_PI * resonance
    wd = wn * math.sqrt(1 - zeta * zeta)
    r = math.exp(-zeta * wn / fs)
    t_peak = math.atan(math.sqrt(1 - zeta * zeta) / zeta) / wd
    peak = math.exp(-zeta * wn * t_peak) * math.sin(wd * t_peak)
    return np.array([0.0, r * math.sin(wd / fs)]) / peak, np.array([1.0, -2 * r * math.cos(wd / fs), r * r])


def impacts(machine: Machine, n: int, fs: float, rng: np.random.Generator) -> np.ndarray:
    """The bearing's impacts as a train of sample-aligned impulses, each scaled by the load it meets."""
    out = np.zeros(n)
    rate = machine.defect_hz()
    if rate <= 0:
        return out
    count = int(n / fs * rate) + 2
    spacing = (1 + machine.slip * rng.standard_normal(count)) / rate
    times = rng.uniform(0, 1 / rate) + np.cumsum(spacing)
    times = times[times < (n - 1) / fs]
    size = np.ones(times.size)
    fm = machine.modulation_hz()
    if fm > 0:  # the defect passes through the load zone once a turn (of the shaft, or of the cage)
        size = 0.2 + 0.8 * 0.5 * (1 + np.cos(TWO_PI * fm * times))
    np.add.at(out, np.rint(times * fs).astype(int), size)
    return out


@dataclass(frozen=True)
class Record:
    t: np.ndarray
    x: np.ndarray  # m/s²
    fs: float
    theta: np.ndarray  # the shaft's angle (rad), from the tachometer


def record(machine: Machine, duration: float = DURATION, fs: float = FS) -> Record:
    n = round(duration * fs)
    t = np.arange(n) / fs
    rng = np.random.default_rng(machine.seed)
    theta = TWO_PI * machine.speed * t
    x = SHAFT[0] * np.cos(theta + 0.3) + SHAFT[1] * np.cos(2 * theta + 1.1)
    amp, phase = gear_modulation(machine, theta)
    for h, a in enumerate(MESH_HARMONICS, start=1):
        x += a * amp * np.cos(h * (machine.teeth * theta + phase) + 0.7 * h)
    b, a = ring_filter(machine.resonance, machine.zeta, fs)
    x += machine.impact * scipy.signal.lfilter(b, a, impacts(machine, n, fs, rng))
    x += machine.noise * rng.standard_normal(n)
    return Record(t, x, fs, theta)


def amplitude_spectrum(x: np.ndarray, fs: float, block: int = BLOCK):
    """Flat-top windowed, averaged with half overlap: a tone reads its amplitude wherever it falls between
    lines."""
    return spectrum(x, fs, block, Window.FLAT_TOP, 0.5)


def envelope(x: np.ndarray, fs: float, band: tuple[float, float]) -> np.ndarray:
    """The Hilbert envelope of x band-passed to `band` (Hz), its mean removed."""
    e = np.abs(scipy.signal.hilbert(band_pass(x, fs, band)))
    return e - e.mean()


def envelope_spectrum(x: np.ndarray, fs: float, band: tuple[float, float], block: int = BLOCK):
    return amplitude_spectrum(envelope(x, fs, band), fs, block)


def cepstrum(x: np.ndarray, fs: float, block: int = BLOCK) -> tuple[np.ndarray, np.ndarray]:
    """(quefrency in s, the real power cepstrum): the inverse FFT of the log of the averaged power spectrum."""
    psd = spectrum(x, fs, block, Window.HANN, 0.5).psd
    c = np.fft.irfft(np.log(np.maximum(psd, 1e-30)), block)
    half = block // 2
    return np.arange(half) / fs, c[:half]


def kurtosis(x: np.ndarray) -> float:
    """3 for Gaussian noise, 1.5 for a sine; impacts raise it."""
    return float(scipy.stats.kurtosis(x, fisher=False))


def band_pass(x: np.ndarray, fs: float, band: tuple[float, float]) -> np.ndarray:
    sos = scipy.signal.butter(4, band, "bandpass", fs=fs, output="sos")
    return scipy.signal.sosfiltfilt(sos, x)


def line_amplitude(freqs: np.ndarray, amplitude: np.ndarray, f: float) -> float:
    """The largest reading on the lines within one line of f (a tone between lines splits over two)."""
    df = freqs[1] - freqs[0]
    i = round(f / df)
    return float(amplitude[max(i - 1, 0):i + 2].max())


def sidebands(freqs: np.ndarray, amplitude: np.ndarray, centre: float, spacing: float, count: int
              ) -> tuple[np.ndarray, np.ndarray]:
    """(the orders −count … count, each line's amplitude at centre + k spacing)."""
    k = np.arange(-count, count + 1)
    return k, np.array([line_amplitude(freqs, amplitude, centre + j * spacing) for j in k])


# ------------------------------------------------------------- the run-up
@dataclass(frozen=True)
class RunUp:
    start: float = 10.0  # Hz
    end: float = 70.0  # Hz
    duration: float = 20.0  # s
    mode: float = 40.0  # Hz, the casing's mode
    zeta: float = 0.03
    unbalance: float = 2e-4  # m: the 1X force per unit mass is this × Ω²
    misalignment: float = 2.0  # m/s², the 2X force per unit mass
    noise: float = 0.05  # m/s² RMS
    seed: int = 1

    def speed(self, t: np.ndarray) -> np.ndarray:
        return self.start + (self.end - self.start) * np.asarray(t) / self.duration

    def angle(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t)
        return TWO_PI * (self.start * t + (self.end - self.start) * t * t / (2 * self.duration))

    def steady(self, order: int, speed: np.ndarray) -> np.ndarray:
        """The order's steady-state acceleration amplitude at each shaft speed (Hz): what an infinitely slow
        run-up would track."""
        w = TWO_PI * order * np.asarray(speed, dtype=float)
        wn = TWO_PI * self.mode
        force = self.unbalance * (TWO_PI * np.asarray(speed)) ** 2 if order == 1 else self.misalignment
        return force * w * w / np.abs(wn * wn - w * w + 2j * self.zeta * wn * w)


def mode_filter(mode: float, zeta: float, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """(b, a): the casing's acceleration per unit force/mass, the exact FOH update (the force linear between
    samples) as a transfer function. With s' = A s + B u, y = C s + u, and v = s − Γ1 u:
    v_{k+1} = Φ v_k + (ΦΓ1 + Γ0) u_k, y_k = C v_k + (CΓ1 + 1) u_k."""
    wn = TWO_PI * mode
    A = np.array([[0.0, 1.0], [-wn * wn, -2 * zeta * wn]])
    B = np.array([[0.0], [1.0]])
    C = A[1:2]
    Phi, G0, G1 = foh_discretize(A, B, 1.0 / fs)
    b, a = scipy.signal.ss2tf(Phi, Phi @ G1 + G0, C, C @ G1 + 1.0)
    return b[0], a


@dataclass(frozen=True)
class RunRecord:
    t: np.ndarray
    x: np.ndarray
    theta: np.ndarray
    speed: np.ndarray
    fs: float


def run_up(run: RunUp, fs: float = RUN_FS) -> RunRecord:
    n = round(run.duration * fs) + 1
    t = np.arange(n) / fs
    theta = run.angle(t)
    w = TWO_PI * run.speed(t)
    u = run.unbalance * w * w * np.cos(theta) + run.misalignment * np.cos(2 * theta + 0.8)
    b, a = mode_filter(run.mode, run.zeta, fs)
    x = scipy.signal.lfilter(b, a, u)
    x += run.noise * np.random.default_rng(run.seed).standard_normal(n)
    return RunRecord(t, x, theta, run.speed(t), fs)


@dataclass(frozen=True)
class OrderMap:
    speed: np.ndarray  # Hz, each block's mean
    axis: np.ndarray  # orders (or Hz)
    amplitude: np.ndarray  # (blocks, lines), peak amplitude


def resample_by_angle(rec: RunRecord, per_rev: int = PER_REV) -> tuple[np.ndarray, np.ndarray]:
    """(angles, the record at them): equal steps of 2π/per_rev, the record's cubic spline at each angle's time
    (the angle is monotonic, so its time comes by interpolation)."""
    angles = np.arange(rec.theta[0], rec.theta[-1], TWO_PI / per_rev)
    times = np.interp(angles, rec.theta, rec.t)
    return angles, scipy.interpolate.CubicSpline(rec.t, rec.x)(times)


def order_map(rec: RunRecord, per_rev: int = PER_REV, revs: int = REVS) -> OrderMap:
    angles, y = resample_by_angle(rec, per_rev)
    block = per_rev * revs
    w = np.hanning(block)
    starts = np.arange(0, y.size - block + 1, block // 2)
    frames = np.stack([w * y[s:s + block] for s in starts])
    amp = 2 * np.abs(np.fft.rfft(frames, axis=1)) / w.sum()
    mid = np.interp(angles[starts + block // 2], rec.theta, rec.t)
    return OrderMap(np.interp(mid, rec.t, rec.speed), np.fft.rfftfreq(block, 1.0 / per_rev), amp)


def frequency_map(rec: RunRecord, block: int = 512) -> OrderMap:
    """The plain short-time spectrum (Hann, blocks of `block` samples, half overlap) against the speed."""
    w = np.hanning(block)
    starts = np.arange(0, rec.x.size - block + 1, block // 2)
    frames = np.stack([w * rec.x[s:s + block] for s in starts])
    amp = 2 * np.abs(np.fft.rfft(frames, axis=1)) / w.sum()
    return OrderMap(rec.speed[starts + block // 2], np.fft.rfftfreq(block, 1.0 / rec.fs), amp)


def order_track(m: OrderMap, order: float) -> np.ndarray:
    """Each block's amplitude of an order (on its line: a whole number of revolutions per block)."""
    j = round(order / (m.axis[1] - m.axis[0]))
    return m.amplitude[:, j]


def frequency_track(m: OrderMap, order: float) -> np.ndarray:
    """Each block's amplitude read at the order's frequency at the block's mean speed, from the plain spectrum:
    the largest of the nearest line and its neighbours."""
    df = m.axis[1] - m.axis[0]
    out = np.empty(m.speed.size)
    for i, s in enumerate(m.speed):
        j = round(order * s / df)
        out[i] = m.amplitude[i, max(j - 1, 0):j + 2].max()
    return out
