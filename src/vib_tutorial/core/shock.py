"""The shock response spectrum: a pulse on a bank of single-degree-of-freedom oscillators.

The spectrum
------------
A shock is a short, strong acceleration of a structure's base: a drop, a collision, a
pyrotechnic separation. Rather than the pulse itself, designers use what it does to a
single-degree-of-freedom oscillator on that base, a mass on a spring and damper with natural
frequency f and damping ζ (Q = 1/2ζ): in the coordinate z relative to the base,

    z'' + 2ζω z' + ω² z = −a(t),

and the mass's absolute acceleration is ÿ = −2ζω z' − ω² z (what an accelerometer on it would
read, and the spring's and damper's force over the mass). The largest |ÿ| over the oscillator's
whole response, against f, is the shock response spectrum (SRS), at that Q (10 by custom). A
component whose natural frequency is f, mounted on that base, sees that peak acceleration.

The response splits in two: the primary (while the pulse lasts) and the residual (after it, a
free decay). Each has a positive and a negative peak; the largest of the four is the maximax.
Three regions follow from the pulse's duration T:

* f T ≫ 1: a stiff oscillator follows the pulse, so its peak is the pulse's own (×2 at most if
  the pulse jumps: a step's overshoot).
* f T ≈ 1: the oscillator amplifies, up to about 1.7× for a half-sine, on its primary.
* f T ≪ 1: a soft oscillator barely moves while the pulse lasts; it takes the pulse as an
  impulse, a change of velocity Δv, and then rings with residual amplitude ω Δv. The SRS rises as
  2π f Δv: the pulse's area, not its shape, decides it. A pulse with no net Δv (a pyrotechnic
  transient) is far lower there: the soft oscillator feels only its damper's push, 2ζω times
  the base's velocity.

Computing it
------------
Each oscillator is stepped by the same exact first-order-hold update as the chain (the input
linear between samples, a "ramp-invariant" filter, as Smallwood's): one complex pole λ = −ζω +
iω√(1 − ζ²) per oscillator, z = 2 Re η with η' = λη + βu, β = −1/(λ − λ̄), so ÿ = 2 Re(λ² η)
and z' = 2 Re(λη). The update is exact for a piecewise-linear pulse; a sine pulse is sampled
finely (`SAMPLES_PER_PULSE`), and the sample rate keeps `PER_PERIOD` samples to the bank's
fastest oscillator, each peak refined by a parabola through the samples round it. The residual
is the free decay from the record's last state, in closed form: its largest value is within
one damped period, since the envelope only shrinks.

The pulses
----------
The classical ones (half-sine, haversine, the two sawtooths, a trapezoid, a square) and two
"measured" records:

* the absolute acceleration of a mass on the chain, its base given a half-sine: the shock a
  component mounted there sees, ringing at the chain's modes (an in-structure shock);
* a transient: damped sines at scattered frequencies, decaying well within the record, as a
  pyrotechnic shock is. Each is the derivative of a damped sine in velocity, so the record
  starts and ends at rest (no net Δv), as a real one does.

Either can have noise (white, seeded) and an offset added, as a measurement does. A small
offset is an accelerometer's zero shift: integrated over the record it is a velocity change the
real shock didn't have, and the SRS's low end rises with it.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass

import numpy as np

from .frf_matrix import MAX_EIGVEC_CONDITION
from .modal import TWO_PI, ModalResult, modal_analysis
from .model import ChainSystem, state_space
from .random_vibration import G, linear
from .simulator import foh_discretize

SAMPLES_PER_PULSE = 400  # samples across a classical pulse at least
PER_PERIOD = 20  # samples per period of the bank's fastest oscillator at least
PER_OCTAVE = 24  # oscillators per octave in the bank
LOW = 0.02  # the bank's lowest f T
HIGH = 20.0  # the bank's highest f T (a transient's: TRANSIENT_HIGH)
RING = 4.0  # periods of the chain's mode 1 recorded after its base's pulse
RESIDUAL_POINTS = 256  # per damped period, where the residual's peak is looked for
MAX_SAMPLES = 400_000
CHUNK = 32  # oscillators worked out at once
TRAPEZOID_RISE = 0.1  # of the duration, each way

# The transient: TRANSIENT_SINES damped sines in velocity, log-uniform over TRANSIENT_BAND / T, each decaying at
# TRANSIENT_DECAY / T (so the record, T long, ends at rest), starting within its first TRANSIENT_START.
TRANSIENT_SINES = 12
TRANSIENT_BAND = (5.0, 40.0)
TRANSIENT_DECAY = 10.0
TRANSIENT_START = 0.1
TRANSIENT_RISE = 3.0  # each sine's rise rate, over ω
TRANSIENT_HIGH = 400.0


class Shape(enum.Enum):
    HALF_SINE = "Half-sine"
    HAVERSINE = "Haversine"
    TERMINAL_PEAK = "Terminal-peak sawtooth"
    INITIAL_PEAK = "Initial-peak sawtooth"
    TRAPEZOID = "Trapezoid"
    SQUARE = "Square"
    AT_A_MASS = "Measured at a mass"
    TRANSIENT = "Measured transient"

    @property
    def measured(self) -> bool:
        return self in (Shape.AT_A_MASS, Shape.TRANSIENT)


CLASSICAL = tuple(s for s in Shape if not s.measured)


@dataclass(frozen=True)
class Pulse:
    """A base acceleration: `peak` (g) and `duration` (s) set a classical pulse; AT_A_MASS's base half-sine and the
    TRANSIENT's scale. `mass` is AT_A_MASS's; `noise` (RMS) and `offset` are fractions of the peak, added to a
    measured record (`seed`'s noise)."""

    shape: Shape = Shape.HALF_SINE
    peak: float = 10.0
    duration: float = 0.1
    mass: int = -1
    noise: float = 0.0
    offset: float = 0.0
    seed: int = 1

    def __post_init__(self) -> None:
        if not self.duration > 0:
            raise ValueError("the pulse's duration must be positive")

    @property
    def high(self) -> float:
        """The bank's highest f T."""
        return TRANSIENT_HIGH if self.shape is Shape.TRANSIENT else HIGH


def classical(shape: Shape, s: np.ndarray) -> np.ndarray:
    """A classical pulse's shape at s = t / T in [0, 1], peak 1."""
    s = np.asarray(s, dtype=float)
    if shape is Shape.HALF_SINE:
        return np.sin(np.pi * s)
    if shape is Shape.HAVERSINE:
        return np.sin(np.pi * s) ** 2
    if shape is Shape.TERMINAL_PEAK:
        return s.copy()
    if shape is Shape.INITIAL_PEAK:
        return 1.0 - s
    if shape is Shape.TRAPEZOID:
        r = TRAPEZOID_RISE
        return np.minimum(1.0, np.minimum(s, 1.0 - s) / r)
    if shape is Shape.SQUARE:
        return np.ones_like(s)
    raise ValueError(f"{shape.value} isn't a classical pulse")


@dataclass(frozen=True)
class Record:
    """A pulse sampled: a (m/s²) at t = k h, k = 0 … The input is linear between samples, zero before the first
    (a jump there a ramp over one step) and after the last. `end` is a classical pulse's end; None for a measured
    record (the whole record is "primary")."""

    pulse: Pulse
    fs: float
    a: np.ndarray
    end: float | None

    @property
    def h(self) -> float:
        return 1.0 / self.fs

    @property
    def t(self) -> np.ndarray:
        return np.arange(self.a.size) * self.h

    @property
    def duration(self) -> float:
        return (self.a.size - 1) * self.h

    def velocity(self) -> np.ndarray:
        """∫a dt (m/s) at each sample, from rest (the ramp from zero before the first sample included)."""
        a = np.concatenate([[0.0], self.a])
        return np.cumsum(0.5 * (a[1:] + a[:-1])) * self.h

    def displacement(self) -> np.ndarray:
        """∫∫a dt (m), from rest, exact for the linear segments."""
        v = np.concatenate([[0.0], self.velocity()])
        a = np.concatenate([[0.0], self.a])
        h = self.h
        return np.cumsum(v[:-1] * h + h * h * (a[:-1] / 3.0 + a[1:] / 6.0))

    @property
    def velocity_change(self) -> float:
        return float(self.velocity()[-1])


def bank(pulse: Pulse, result: ModalResult | None = None, per_octave: int = PER_OCTAVE) -> np.ndarray:
    """The oscillators' frequencies (Hz): LOW / T to the pulse's `high` / T, widened to a quarter of the chain's
    lowest mode and four times its highest, `per_octave` to the octave."""
    lo, hi = LOW / pulse.duration, pulse.high / pulse.duration
    if result is not None:
        fn = [m.fn_hz for m in result.modes if m.fn_hz > 0]
        if fn:
            lo, hi = min(lo, min(fn) / 4.0), max(hi, 4.0 * max(fn))
    count = int(math.ceil(per_octave * math.log2(hi / lo))) + 1
    return lo * 2.0 ** (np.arange(count) / per_octave)


def sample_rate(pulse: Pulse, f_top: float) -> float:
    """SAMPLES_PER_PULSE across the pulse and PER_PERIOD to the fastest oscillator, whichever is more."""
    return max(SAMPLES_PER_PULSE / pulse.duration, PER_PERIOD * f_top)


def record(pulse: Pulse, fs: float, system: ChainSystem | None = None, result: ModalResult | None = None) -> Record:
    """Sample the pulse at fs, raised to a whole number of samples across it (AT_A_MASS runs the chain, `system`)."""
    T = pulse.duration
    A = pulse.peak * G
    k = max(2, int(math.ceil(T * fs - 1e-9)))
    fs = k / T
    h = 1.0 / fs
    if not pulse.shape.measured:
        a = A * classical(pulse.shape, np.arange(k + 1) / k)
        return Record(pulse, fs, a, k * h)
    rng = np.random.default_rng(pulse.seed)
    if pulse.shape is Shape.AT_A_MASS:
        if system is None:
            raise ValueError("a measured response needs the chain")
        lin = linear(system)
        result = result or modal_analysis(lin)
        f1 = min((m.fn_hz for m in result.modes if m.fn_hz > 0), default=1.0 / T)
        n = int(round((T + RING / f1) * fs)) + 1
        if n > MAX_SAMPLES:
            raise ValueError(f"the record is too long: {n:,} samples, at most {MAX_SAMPLES:,}")
        base = np.zeros(n)
        base[:k + 1] = A * classical(Shape.HALF_SINE, np.arange(k + 1) / k)
        mass = pulse.mass % lin.n
        a = chain_response(lin, base, fs, result).acceleration[:, mass]
    else:
        t = np.arange(k + 1) * h
        lo, hi = TRANSIENT_BAND
        w = TWO_PI * np.exp(rng.uniform(math.log(lo), math.log(hi), TRANSIENT_SINES)) / T
        start = rng.uniform(0.0, TRANSIENT_START * T, TRANSIENT_SINES)
        amp = rng.uniform(0.3, 1.0, TRANSIENT_SINES) * rng.choice([-1.0, 1.0], TRANSIENT_SINES)
        sigma = TRANSIENT_DECAY / T
        s = np.maximum(t[:, None] - start[None, :], 0.0)
        # each a velocity (e^{−σs} − e^{−(σ + ρ)s}) sin ωs, which starts and ends at rest, its rise ρ = 3ω quick but
        # smooth: a is its derivative
        rho = TRANSIENT_RISE * w
        g = np.exp(-sigma * s) - np.exp(-(sigma + rho) * s)
        dg = -sigma * np.exp(-sigma * s) + (sigma + rho) * np.exp(-(sigma + rho) * s)
        a = np.sum(amp * (dg * np.sin(w * s) + g * w * np.cos(w * s)), axis=1)
        a *= A / np.max(np.abs(a))
    if pulse.noise > 0:
        a = a + pulse.noise * A * rng.standard_normal(a.size)
    if pulse.offset:
        a = a + pulse.offset * A
    return Record(pulse, fs, a, None)


# ------------------------------------------------------------- the oscillators
def poles(f: np.ndarray, zeta: float) -> np.ndarray:
    w = TWO_PI * np.asarray(f, dtype=float)
    return w * (-zeta + 1j * math.sqrt(1.0 - zeta * zeta))


def _foh(lam: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """η_{k+1} = μ η_k + c0 u_k + c1 u_{k+1} for η' = λη + βu, the input linear over the step."""
    beta = -1.0 / (2j * lam.imag)
    mu = np.exp(lam * h)
    i0 = (mu - 1.0) / lam
    i1 = (mu - 1.0 - lam * h) / (lam * lam * h)
    return mu, beta * (i0 - i1), beta * i1


def _states(rec: Record, f: np.ndarray, zeta: float) -> tuple[np.ndarray, np.ndarray]:
    """Each oscillator's η at every sample, (f, samples), and its poles."""
    import scipy.signal  # deferred: scipy.signal adds ~0.9 s to startup

    lam = poles(f, zeta)
    mu, c0, c1 = _foh(lam, rec.h)
    eta = np.empty((f.size, rec.a.size), dtype=complex)
    for i in range(f.size):
        eta[i] = scipy.signal.lfilter([c1[i], c0[i]], [1.0, -mu[i]], rec.a)
    return eta, lam


def _refined(y: np.ndarray) -> np.ndarray:
    """Each row's largest value, refined by a parabola through the samples round it."""
    k = np.argmax(y, axis=1)
    rows = np.arange(y.shape[0])
    peak = y[rows, k]
    inside = (k > 0) & (k < y.shape[1] - 1)
    a = y[rows, np.maximum(k - 1, 0)]
    c = y[rows, np.minimum(k + 1, y.shape[1] - 1)]
    curve = a - 2.0 * peak + c
    ok = inside & (curve < 0)
    p = np.where(ok, 0.5 * (a - c) / np.where(ok, curve, 1.0), 0.0)
    return np.where(ok, peak - 0.25 * (a - c) * p, peak)


def _free(eta_end: np.ndarray, lam: np.ndarray, power: int) -> np.ndarray:
    """2 Re(λ^power η e^{λτ}) over one damped period, RESIDUAL_POINTS a period: (f, points)."""
    period = TWO_PI / lam.imag
    tau = np.linspace(0.0, 1.0, RESIDUAL_POINTS + 1)[None, :] * period[:, None]
    return 2.0 * np.real((lam ** power * eta_end)[:, None] * np.exp(lam[:, None] * tau))


@dataclass(frozen=True)
class Spectrum:
    """The SRS at damping ζ: the absolute acceleration's largest positive and negative values (both as positive
    numbers, m/s²), `primary` while the record lasts, `residual` after it, (2, f); and the relative displacement's
    largest |z| (m), the whole response."""

    f: np.ndarray
    zeta: float
    primary: np.ndarray
    residual: np.ndarray
    displacement: np.ndarray

    @property
    def q(self) -> float:
        return 1.0 / (2.0 * self.zeta) if self.zeta > 0 else math.inf

    @property
    def positive(self) -> np.ndarray:
        return np.maximum(self.primary[0], self.residual[0])

    @property
    def negative(self) -> np.ndarray:
        return np.maximum(self.primary[1], self.residual[1])

    @property
    def maximax(self) -> np.ndarray:
        return np.maximum(self.positive, self.negative)

    @property
    def pseudo_velocity(self) -> np.ndarray:
        """ω |z|max (m/s), the tripartite plot's."""
        return TWO_PI * self.f * self.displacement

    def at(self, f: float) -> float:
        """The maximax at f, interpolated in log–log."""
        return float(np.exp(np.interp(math.log(f), np.log(self.f), np.log(self.maximax))))


def srs(rec: Record, f: np.ndarray, zeta: float) -> Spectrum:
    """The shock response spectrum of the record over the oscillators at f (Hz), damping ζ < 1."""
    if not 0 <= zeta < 1:
        raise ValueError("the oscillators' damping must be below critical")
    f = np.asarray(f, dtype=float)
    primary, residual = np.empty((2, f.size)), np.empty((2, f.size))
    zmax = np.empty(f.size)
    for lo in range(0, f.size, CHUNK):
        part = slice(lo, lo + CHUNK)
        eta, lam = _states(rec, f[part], zeta)
        y = 2.0 * np.real(lam[:, None] ** 2 * eta)
        primary[:, part] = _refined(y), _refined(-y)
        end = eta[:, -1]
        yr = _free(end, lam, 2)
        residual[:, part] = np.max(yr, axis=1), np.max(-yr, axis=1)
        zmax[part] = np.maximum(_refined(np.abs(2.0 * np.real(eta))), np.max(np.abs(_free(end, lam, 0)), axis=1))
    return Spectrum(f, zeta, np.maximum(primary, 0.0), np.maximum(residual, 0.0), zmax)


@dataclass(frozen=True)
class Oscillator:
    """One oscillator's response, sampled at the record's h: through the record and `tail` s after, the absolute
    acceleration (m/s²) and the relative displacement (m)."""

    f: float
    zeta: float
    t: np.ndarray
    acceleration: np.ndarray
    displacement: np.ndarray
    end: float


def oscillator(rec: Record, f: float, zeta: float, tail: float) -> Oscillator:
    eta, lam = _states(rec, np.array([f]), zeta)
    n = int(round(tail * rec.fs))
    tau = np.arange(1, n + 1) * rec.h
    eta = np.concatenate([eta[0], eta[0, -1] * np.exp(lam[0] * tau)])
    t = np.arange(eta.size) * rec.h
    return Oscillator(f, zeta, t, 2.0 * np.real(lam[0] ** 2 * eta), 2.0 * np.real(eta), rec.duration)


# ------------------------------------------------------------- the chain on a shaken base
@dataclass(frozen=True)
class ChainResponse:
    """The chain under its base's acceleration: displacements relative to the base (m) and absolute accelerations
    (m/s²), a row per sample."""

    displacement: np.ndarray
    acceleration: np.ndarray


def chain_response(system: ChainSystem, a: np.ndarray, fs: float, result: ModalResult | None = None
                   ) -> ChainResponse:
    """The chain from rest, its base accelerated by a (m/s², linear between samples 1/fs apart): in coordinates
    relative to the base each mass is pushed by −m a, by the exact FOH update, diagonalized (as core.oma's)."""
    import scipy.signal

    lin = linear(system)
    n = lin.n
    A, B = state_space(lin)
    b = B @ -lin.masses
    Phi, G0, G1 = foh_discretize(A, b[:, None], 1.0 / fs)
    mu, V = np.linalg.eig(Phi)
    if np.linalg.cond(V) < MAX_EIGVEC_CONDITION:
        Vinv = np.linalg.inv(V)
        g0, g1 = (Vinv @ G0)[:, 0], (Vinv @ G1)[:, 0]
        eta = np.empty((a.size, 2 * n), dtype=complex)
        for r in range(2 * n):
            eta[:, r] = scipy.signal.lfilter([g1[r], g0[r]], [1.0, -mu[r]], a)
        z = (eta @ V.T).real
    else:
        z = np.empty((a.size, 2 * n))
        state = np.zeros(2 * n)
        prev = 0.0
        for k, ak in enumerate(a):
            state = Phi @ state + G0[:, 0] * prev + G1[:, 0] * ak
            z[k] = state
            prev = ak
    acc = z @ A[n:].T + np.outer(a, b[n:]) + a[:, None]
    return ChainResponse(z[:, :n], acc)
