"""Random vibration of the chain: a random input given by its spectrum, and the response's spectrum and RMS.

The input
---------
A stationary Gaussian random process is described by its one-sided power spectral density
(PSD) G(f): its mean square per hertz, so the RMS² is the area under it. Here G is flat at a
level between two frequencies (band-limited white noise), and acts either as

* a force on one mass, or on every mass (independent forces, each with that PSD), in N²/Hz;
* the base's acceleration, as a shaker table or a vehicle's floor drives it, in g²/Hz. The
  chain then moves relative to the base: in z = x − x_g the base's acceleration a_g is a
  force −m_i a_g on each mass, M z'' + C z' + K z = −M 1 a_g, all the forces one signal.

The response
------------
A linear system's response to a random input is random too, with PSD

    G_y(f) = H(f) G_u(f) H(f)ᴴ,

H the transfer from the inputs to the response, (K − ω²M + iωC)⁻¹ applied to the input's
force pattern and read off as a mass's displacement (relative to the base for a base input),
its acceleration (absolute: + a_g for a base input), or a spring's force k_i (x_i − x_{i−1}).
Independent inputs add their PSDs. The RMS is √∫G_y df, worked out on a grid packed round
each mode's half-power band (`frequency_grid`), and its running value, the cumulative RMS,
shows which modes make it up.

Miles' equation
---------------
One lightly damped mode under an input that is flat across its resonance responds as if to
white noise, and the area under its peak is (π/2) f_n Q times the peak's input, with Q =
1/2ζ: a single-degree-of-freedom system on a base shaken at G_a g²/Hz has an absolute
acceleration of RMS √((π/2) f_n Q G_a(f_n)), Miles' equation. On the chain each mode r is
one such oscillator in its modal coordinate q_r (mass-normalized φ_r), with modal force
φ_rᵀ f: σ_q² = (π/2) f_r Q_r G_q(f_r) / ω_r⁴, with G_q = Σ (φ_rᵀ b)² G over the inputs' force
patterns b (−M 1 for the base). Each response is then φ_r's view of σ_q (× ω_r² for an
acceleration), and the modes combine by the square root of the sum of their squares (SRSS,
right when they are well apart). It misses what isn't resonant: a mass line under a force
(acceleration = force / mass far above the modes), modes outside the band, and the cross
terms of close modes.

Statistics
----------
A Gaussian response is described by its RMS σ and, for how often it does things, by its
PSD's spectral moments m_k = ∫ f^k G df (m₀ = σ²). Rice's formulas give the rate at which it
crosses zero upwards, ν₀ = √(m₂/m₀), and the rate of its peaks (maxima), ν_p = √(m₄/m₂); a
level a is crossed upwards ν₀ e^{−a²/2σ²} times a second. Their ratio α = ν₀/ν_p is the
irregularity factor: 1 for a narrow-band response (one peak per cycle, the peaks' heights
Rayleigh distributed), towards 0 for a broad one (many small wiggles, the peaks' heights
nearly Gaussian, many below zero). Rice's peak distribution covers everything between
(`peak_pdf`). The largest peak in a time T is, on average (Davenport), σ times
√(2 ln ν₀T) + 0.5772/√(2 ln ν₀T): about 3σ for a few hundred cycles, 4–5σ for a long
service life, which is why "3σ" design is not a maximum. Davenport counts each upward
crossing as independent; a narrow-band response's come in clumps, on the swells of its slowly
varying envelope, so it overestimates the largest peak in a short time. Vanmarcke's
correction (`vanmarcke`) counts the clumps instead, through the bandwidth δ = √(1 − m₁²/m₀m₂).

The record's statistics count the same things (`record_statistics`): the fraction of the
time beyond ±kσ, the crossings and the peaks, and its skewness and kurtosis (0 and 3 for a
Gaussian). The record holds 2.5 samples per period of the band's top, enough to describe it
(it has nothing above the band) but not to see its peaks, so it is interpolated eightfold
first (a polyphase low-pass, which restores a band-limited signal between its samples).

The time simulation
-------------------
The record's input is synthesized from its PSD (random phases and Gaussian amplitudes
on every FFT line, `synthesize`) at an analog rate ten times the band's top, divided by
sinc⁴(f h): the exact first-order-hold update takes the input as linear between the samples
h apart, which filters it by sinc⁴. It then has the PSD asked for, and the chain's response
comes from the same exact FOH update as everywhere in the app, diagonalized into a
first-order filter per eigenvalue (as in core.oma), run for its settling time before the
record starts. A linear system responds only where it is driven, so the response has nothing
above the band either, and every fourth sample is kept: 2.5 samples per period of the band's
top, clear of aliasing. Friction is left out: all of this needs the chain linear. Welch's
method estimates the record's PSD, and its RMS is its standard deviation, to compare with the
exact ones.

A lightly damped mode's peak is narrow, 2ζf_n between its half-power points: Welch's lines
must be finer than that or the peak comes out low (the window smears it), and each line's
estimate scatters by 1/√(averages). Both ask for a long record: `suggested_duration` puts
four lines across the narrowest peak and sixteen blocks in the record.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass

import numpy as np

from .frf_matrix import MAX_EIGVEC_CONDITION
from .measurement import settle_time
from .modal import TWO_PI, ModalResult, modal_analysis
from .model import ChainSystem, state_space
from .simulator import foh_discretize

G = 9.80665  # m/s² per g
GRID_POINTS = 4000  # uniform lines across the band
MODE_POINTS = 401  # more across each mode's ±MODE_REACH half-power widths
MODE_REACH = 25.0
OVERSAMPLE = 10  # analog samples per period of the band's top
KEEP = 4  # analog samples per sample kept: 2.5 per period of the band's top
LINES = 4  # Welch's lines across the narrowest half-power width
BLOCKS_IN_RECORD = 16  # Welch's blocks in a suggested record (31 half-overlapping averages)
MAX_SAMPLES = 400_000  # kept, per record
CHUNK = 1 << 15
BLOCKS = tuple(2**k for k in range(8, 18))  # Welch's block lengths


class Input(enum.Enum):
    FORCE_ONE = "Force on one mass"
    FORCE_ALL = "Force on every mass"
    BASE = "Base acceleration"


class Quantity(enum.Enum):
    DISPLACEMENT = "Displacement"
    ACCELERATION = "Acceleration"
    SPRING = "Spring force"


@dataclass(frozen=True)
class RandomInput:
    """Flat at `level` (N²/Hz for a force, g²/Hz for the base) from f_lo to f_hi (Hz); `mass` is FORCE_ONE's."""

    kind: Input = Input.BASE
    level: float = 0.01
    f_lo: float = 0.0
    f_hi: float = 10.0
    mass: int = 0

    def __post_init__(self) -> None:
        if not self.level > 0:
            raise ValueError("the level must be positive")
        if not 0 <= self.f_lo < self.f_hi:
            raise ValueError("the band needs 0 ≤ f_lo < f_hi")

    @property
    def si_level(self) -> float:
        """The level in SI: N²/Hz, or (m/s²)²/Hz."""
        return self.level * G * G if self.kind is Input.BASE else self.level

    def psd(self, f) -> np.ndarray:
        """The one-sided PSD in SI at f."""
        f = np.asarray(f, dtype=float)
        return np.where((f >= self.f_lo) & (f <= self.f_hi), self.si_level, 0.0)

    @property
    def rms(self) -> float:
        """The input's RMS: N, or g for the base."""
        return math.sqrt(self.level * (self.f_hi - self.f_lo))

    def patterns(self, system: ChainSystem) -> np.ndarray:
        """Each independent input's force on the masses per unit of it, a column each."""
        n = system.n
        if self.kind is Input.FORCE_ONE:
            if not 0 <= self.mass < n:
                raise ValueError("the force's mass isn't on the chain")
            b = np.zeros((n, 1))
            b[self.mass, 0] = 1.0
            return b
        if self.kind is Input.FORCE_ALL:
            return np.eye(n)
        return -system.masses[:, None]


@dataclass(frozen=True)
class Output:
    """A response: a mass's displacement or acceleration, or a spring's force (element `index`, 0 the ground's)."""

    quantity: Quantity = Quantity.ACCELERATION
    index: int = 0

    def unit(self) -> str:
        return {Quantity.DISPLACEMENT: "m", Quantity.ACCELERATION: "g", Quantity.SPRING: "N"}[self.quantity]

    def scale(self) -> float:
        """SI to the output's unit."""
        return 1.0 / G if self.quantity is Quantity.ACCELERATION else 1.0

    def row(self, system: ChainSystem) -> np.ndarray:
        """The displacements' (or accelerations') combination this output reads."""
        r = np.zeros(system.n)
        i = self.index
        if not 0 <= i < system.n:
            raise ValueError("the output isn't on the chain")
        if self.quantity is Quantity.SPRING:
            r[i] = system.stiffness[i]
            if i > 0:
                r[i - 1] = -system.stiffness[i]
        else:
            r[i] = 1.0
        return r


def linear(system: ChainSystem) -> ChainSystem:
    """The chain without its friction: random vibration needs it linear."""
    return ChainSystem(system.masses, system.stiffness, system.damping)


def transfer(system: ChainSystem, inp: RandomInput, out: Output, f: np.ndarray) -> np.ndarray:
    """The output (in its unit) per unit of each input (N, or m/s² of the base), at each f: (f, inputs)."""
    M, C, K = linear(system).matrices()
    b = inp.patterns(system)
    row = out.row(system)
    w = TWO_PI * np.asarray(f, dtype=float)
    t = np.empty((w.size, b.shape[1]), dtype=complex)
    for k, wk in enumerate(w):
        x = np.linalg.solve(K - wk * wk * M + 1j * wk * C, b)
        t[k] = row @ x
    if out.quantity is Quantity.ACCELERATION:
        t *= -(w * w)[:, None]
        if inp.kind is Input.BASE:
            t += 1.0  # the absolute acceleration: the base's own on top of the motion relative to it
    return t * out.scale()


def response_psd(system: ChainSystem, inp: RandomInput, out: Output, f: np.ndarray) -> np.ndarray:
    """G_y = H G_u Hᴴ, the inputs independent: Σ |H_c|² G (the output's unit squared per Hz)."""
    return np.sum(np.abs(transfer(system, inp, out, f))**2, axis=1) * inp.psd(f)


def frequency_grid(inp: RandomInput, result: ModalResult) -> np.ndarray:
    """The band's lines: uniform, with more packed across each mode's half-power width inside it."""
    lo = inp.f_lo if inp.f_lo > 0 else min(1e-3 * inp.f_hi, 1e-3)
    parts = [np.linspace(lo, inp.f_hi, GRID_POINTS)]
    for m in result.modes:
        z = m.zeta_modal
        if m.fn_hz > 0 and math.isfinite(z) and z > 0:
            f = m.fn_hz * (1.0 + max(z, 1e-4) * np.linspace(-MODE_REACH, MODE_REACH, MODE_POINTS))
            parts.append(f[(f >= lo) & (f <= inp.f_hi)])
    return np.unique(np.concatenate(parts))


def cumulative_rms(f: np.ndarray, psd: np.ndarray) -> np.ndarray:
    """√∫₀^f G df, by the trapezium rule on the grid."""
    area = np.concatenate([[0.0], np.cumsum(0.5 * (psd[1:] + psd[:-1]) * np.diff(f))])
    return np.sqrt(area)


@dataclass(frozen=True)
class Prediction:
    """The exact response PSD on its grid, its RMS, and Miles' estimate mode by mode."""

    f: np.ndarray
    psd: np.ndarray
    cumulative: np.ndarray
    miles: list[tuple[int, float, float, float]]  # (mode, fn Hz, ζ, its RMS), the modes in the band

    @property
    def rms(self) -> float:
        return float(self.cumulative[-1])

    @property
    def miles_rms(self) -> float:
        """The modes' Miles estimates combined by SRSS."""
        return math.sqrt(sum(s * s for *_, s in self.miles))


def miles(system: ChainSystem, inp: RandomInput, out: Output, result: ModalResult) -> list[tuple[int, float, float, float]]:
    """Each mode in the band as a single-degree-of-freedom system under white input at its frequency (see the
    module's notes): (mode, fn, ζ, the output's RMS from it)."""
    b = inp.patterns(system)
    row = out.row(system)
    rows = []
    for m in result.modes:
        z, fn = m.zeta_modal, m.fn_hz
        if not (fn > 0 and math.isfinite(z) and z > 0 and inp.f_lo <= fn <= inp.f_hi):
            continue
        phi = m.shape_mass_normalized
        g_q = float(np.sum((phi @ b)**2)) * inp.si_level
        w = m.omega_n
        sigma_q = math.sqrt(0.5 * math.pi * fn * g_q / (2.0 * z)) / (w * w)
        sigma = abs(row @ phi) * sigma_q
        if out.quantity is Quantity.ACCELERATION:
            sigma *= w * w
        rows.append((m.index, fn, z, sigma * out.scale()))
    return rows


def predict(system: ChainSystem, inp: RandomInput, out: Output, result: ModalResult | None = None) -> Prediction:
    result = result or modal_analysis(linear(system))
    f = frequency_grid(inp, result)
    psd = response_psd(system, inp, out, f)
    return Prediction(f, psd, cumulative_rms(f, psd), miles(system, inp, out, result))


# ------------------------------------------------------------- the time simulation
def sample_rate(inp: RandomInput) -> float:
    """The record's samples per second (the analog rate is KEEP times as many)."""
    return OVERSAMPLE * inp.f_hi / KEEP


def synthesize(rng: np.random.Generator, samples: int, fs: float, psd, columns: int = 1) -> np.ndarray:
    """Gaussian noise with the one-sided PSD psd(f) (per Hz) as a piecewise-linear signal through its samples: each
    FFT line a complex Gaussian of mean square G N fs / 2, divided by sinc²(f/fs) for the linear interpolation's
    sinc⁴. (samples, columns)."""
    f = np.fft.rfftfreq(samples, 1.0 / fs)
    g = psd(f) / np.sinc(f / fs)**4
    g[0] = 0.0
    if samples % 2 == 0:
        g[-1] = 0.0
    s = np.sqrt(g * samples * fs / 4.0)
    spec = s[:, None] * (rng.standard_normal((f.size, columns)) + 1j * rng.standard_normal((f.size, columns)))
    return np.fft.irfft(spec, n=samples, axis=0)


@dataclass
class RandomRecord:
    """The simulated record: displacements (relative to the base for a base input) and accelerations (absolute),
    SI, a row per sample, and the input (N, or m/s²: a column per independent input)."""

    system: ChainSystem
    input: RandomInput
    fs: float
    x: np.ndarray
    a: np.ndarray
    u: np.ndarray

    @property
    def duration(self) -> float:
        return self.x.shape[0] / self.fs

    def output(self, out: Output) -> np.ndarray:
        """An output's record, in its unit."""
        if out.quantity is Quantity.ACCELERATION:
            return self.a[:, out.index] * out.scale()
        return self.x @ out.row(self.system)


def simulate(system: ChainSystem, inp: RandomInput, duration: float, seed: int | None = None,
             result: ModalResult | None = None) -> RandomRecord:
    """Run the chain under the random input for `duration` s after it has settled (see the module's notes)."""
    import scipy.signal  # deferred: scipy.signal adds ~0.9 s to startup

    lin = linear(system)
    result = result or modal_analysis(lin)
    n = lin.n
    fs = sample_rate(inp)
    fs_a = KEEP * fs
    h = 1.0 / fs_a
    samples = int(round(duration * fs))
    if samples > MAX_SAMPLES:
        raise ValueError(f"the record is too long: at most {MAX_SAMPLES:,} samples ({MAX_SAMPLES / fs:.4g} s here)")
    if samples < 2:
        raise ValueError("the record is too short")
    skip = KEEP * int(math.ceil(settle_time(result) * fs))  # analog samples before the record, whole samples
    total = skip + KEEP * samples
    rng = np.random.default_rng(seed)
    b = inp.patterns(lin)
    u = synthesize(rng, total, fs_a, inp.psd, b.shape[1])
    A, B = state_space(lin)
    Phi, G0, G1 = foh_discretize(A, B, h)
    mu, V = np.linalg.eig(Phi)
    modal = np.linalg.cond(V) < MAX_EIGVEC_CONDITION
    if modal:
        Vinv = np.linalg.inv(V)
        b0, b1 = Vinv @ G0, Vinv @ G1
        eta = np.zeros(2 * n, dtype=complex)
    z = np.zeros(2 * n)
    f_prev = np.zeros(n)
    xs, accs = [], []
    for start in range(0, total, CHUNK):
        m = min(CHUNK, total - start)
        fk = u[start:start + m] @ b.T  # the forces on the masses (−m a_g for the base)
        if modal:
            drive = np.vstack([f_prev[None, :], fk[:-1]]) @ b0.T + fk @ b1.T
            states = np.empty((m, 2 * n), dtype=complex)
            for r in range(2 * n):
                states[:, r], _ = scipy.signal.lfilter([1.0], [1.0, -mu[r]], drive[:, r], zi=[mu[r] * eta[r]])
                eta[r] = states[-1, r]
            zs = (states @ V.T).real
        else:
            zs = np.empty((m, 2 * n))
            f0 = f_prev
            for k in range(m):
                z = Phi @ z + G0 @ f0 + G1 @ fk[k]
                zs[k] = z
                f0 = fk[k]
        f_prev = fk[-1].copy()
        acc = zs @ A[n:].T + fk @ B[n:].T
        if inp.kind is Input.BASE:
            acc = acc + u[start:start + m, :1]  # absolute: the base's own on top
        keep = np.arange(start, start + m)
        keep = keep[(keep >= skip) & ((keep - skip) % KEEP == 0)] - start
        xs.append(zs[keep, :n])
        accs.append(acc[keep])
    return RandomRecord(lin, inp, fs, np.concatenate(xs), np.concatenate(accs), u[skip::KEEP])


def welch(y: np.ndarray, fs: float, block: int) -> tuple[np.ndarray, np.ndarray]:
    """Welch's one-sided PSD estimate: Hann blocks, half overlapping."""
    import scipy.signal

    block = min(block, y.size)
    return scipy.signal.welch(y, fs, window="hann", nperseg=block, noverlap=block // 2, axis=0)


def narrowest(inp: RandomInput, result: ModalResult) -> float | None:
    """The narrowest half-power width 2ζf_n of the modes in the band (Hz), None if none is."""
    widths = [2.0 * m.zeta_modal * m.fn_hz for m in result.modes
              if inp.f_lo <= m.fn_hz <= inp.f_hi and math.isfinite(m.zeta_modal) and m.zeta_modal > 0]
    return min(widths) if widths else None


def suggested_block(inp: RandomInput, result: ModalResult, samples: int) -> int:
    """LINES lines across the narrowest peak, if the record holds BLOCKS_IN_RECORD such blocks; else the longest
    block that many fit."""
    fs = sample_rate(inp)
    width = narrowest(inp, result)
    want = LINES * fs / width if width else BLOCKS[0]
    fits = [blk for blk in BLOCKS if blk * BLOCKS_IN_RECORD <= samples] or [BLOCKS[0]]
    return min((blk for blk in fits if blk >= want), default=fits[-1])


def suggested_duration(inp: RandomInput, result: ModalResult) -> float:
    """BLOCKS_IN_RECORD blocks of LINES lines across the narrowest peak, within MAX_SAMPLES."""
    fs = sample_rate(inp)
    width = narrowest(inp, result)
    block = min((blk for blk in BLOCKS if blk >= LINES * fs / width), default=BLOCKS[-1]) if width else BLOCKS[0]
    return min(BLOCKS_IN_RECORD * block, MAX_SAMPLES) / fs


def rms_scatter(prediction: Prediction, duration: float) -> float:
    """The record's RMS's expected relative scatter, 1 / (2√(B T)), with B the response's statistical bandwidth
    (∫G)² / ∫G²."""
    f, g = prediction.f, prediction.psd
    area = np.trapezoid(g, f)
    if area <= 0:
        return 0.0
    bandwidth = area * area / np.trapezoid(g * g, f)
    return 1.0 / (2.0 * math.sqrt(bandwidth * duration))



# ------------------------------------------------------------- statistics
UPSAMPLE = 8  # the record's samples interpolated this many times over before peaks and crossings are counted
EULER = 0.5772156649015329
LEVELS = (1.0, 2.0, 3.0)  # σ levels whose exceedances are compared


@dataclass(frozen=True)
class Rice:
    """A Gaussian response's rates from its PSD's moments: RMS, upward zero crossings and peaks a second (Hz), and
    Vanmarcke's bandwidth δ."""

    sigma: float
    nu0: float
    nu_peak: float
    delta: float = 1.0

    @property
    def alpha(self) -> float:
        """The irregularity factor ν₀/ν_p: 1 narrow-band, towards 0 broad."""
        return self.nu0 / self.nu_peak if self.nu_peak > 0 else 1.0

    def crossings(self, level: float) -> float:
        """Upward crossings of `level` a second."""
        return self.nu0 * math.exp(-0.5 * (level / self.sigma)**2)

    def peak_factor(self, duration: float) -> float:
        """Davenport's expected largest peak over `duration`, over σ."""
        return davenport(self.nu0 * duration)

    def clumped_peak_factor(self, duration: float) -> float:
        """Vanmarcke's expected largest peak over `duration`, over σ."""
        return vanmarcke(self.nu0 * duration, self.delta)


def spectral_moments(f: np.ndarray, psd: np.ndarray) -> tuple[float, float, float, float]:
    """m₀, m₁, m₂, m₄ = ∫ f^k G df (f in Hz)."""
    return tuple(float(np.trapezoid(f**k * psd, f)) for k in (0, 1, 2, 4))


def rice(prediction: Prediction) -> Rice:
    m0, m1, m2, m4 = spectral_moments(prediction.f, prediction.psd)
    if m0 <= 0 or m2 <= 0:
        return Rice(0.0, 0.0, 0.0)
    return Rice(math.sqrt(m0), math.sqrt(m2 / m0), math.sqrt(m4 / m2),
                math.sqrt(max(1.0 - m1 * m1 / (m0 * m2), 0.0)))


def davenport(cycles: float) -> float:
    """The expected largest of a Gaussian process's peaks over ν₀T zero crossings, over σ."""
    if cycles <= 1.0:
        return 0.0
    r = math.sqrt(2.0 * math.log(cycles))
    return r + EULER / r


def vanmarcke(cycles: float, delta: float) -> float:
    """The expected largest value over ν₀T zero crossings, over σ, the crossings clumped (Vanmarcke): the chance
    of staying below r is (1 − e^{−r²/2}) exp(−ν₀T (1 − e^{−√(π/2) δ^1.2 r}) / (e^{r²/2} − 1)), integrated."""
    if cycles <= 1.0:
        return 0.0
    r = np.linspace(1e-6, 12.0, 6001)
    x = np.exp(-0.5 * r * r)
    clumps = 1.0 - np.exp(-math.sqrt(0.5 * math.pi) * max(delta, 1e-6)**1.2 * r)
    below = (1.0 - x) * np.exp(-cycles * clumps * x / (1.0 - x))
    return float(np.trapezoid(1.0 - below, r))


def gaussian_exceedance(k: float) -> float:
    """The fraction of the time a Gaussian signal is beyond ±kσ."""
    return math.erfc(k / math.sqrt(2.0))


def peak_pdf(eta: np.ndarray, alpha: float) -> np.ndarray:
    """Rice's distribution of a Gaussian process's peaks (maxima), η = height / σ: Rayleigh for α = 1, Gaussian as
    α → 0."""
    eta = np.asarray(eta, dtype=float)
    alpha = min(max(alpha, 0.0), 1.0)
    eps = math.sqrt(max(1.0 - alpha * alpha, 0.0))
    rayleigh = alpha * eta * np.exp(-0.5 * eta * eta)
    if eps < 1e-9:
        return np.where(eta > 0, rayleigh, 0.0)
    phi = 0.5 * (1.0 + _erf(alpha * eta / (eps * math.sqrt(2.0))))
    return eps / math.sqrt(2.0 * math.pi) * np.exp(-0.5 * (eta / eps)**2) + rayleigh * phi


def _erf(x: np.ndarray) -> np.ndarray:
    import scipy.special

    return scipy.special.erf(x)


@dataclass(frozen=True)
class RecordStatistics:
    """What a record does, counted: its σ, skewness, kurtosis, the fraction of the time beyond ±kσ (LEVELS), upward
    zero crossings and peaks a second, and the peaks' heights over σ."""

    sigma: float
    skewness: float
    kurtosis: float
    exceedance: tuple[float, ...]
    nu0: float
    nu_peak: float
    peaks: np.ndarray

    def crossings(self, y: np.ndarray, fs: float, level: float) -> float:
        return _upcrossings(y - np.mean(y), level) * fs / y.size


def _upcrossings(y: np.ndarray, level: float) -> int:
    return int(np.count_nonzero((y[:-1] < level) & (y[1:] >= level)))


def upsampled(y: np.ndarray) -> np.ndarray:
    """The record interpolated UPSAMPLE times over (it has nothing above its Nyquist frequency)."""
    import scipy.signal

    return scipy.signal.resample_poly(y, UPSAMPLE, 1)


def record_statistics(y: np.ndarray, fs: float) -> RecordStatistics:
    fine = upsampled(y - np.mean(y))
    duration = y.size / fs
    sigma = float(np.std(fine))
    z = fine / sigma if sigma > 0 else fine
    skew = float(np.mean(z**3))
    kurt = float(np.mean(z**4))
    exceed = tuple(float(np.mean(np.abs(z) > k)) for k in LEVELS)
    up = _upcrossings(fine, 0.0) / duration
    mid = fine[1:-1]
    maxima = (mid > fine[:-2]) & (mid >= fine[2:])
    peaks = mid[maxima] / sigma if sigma > 0 else mid[maxima]
    return RecordStatistics(sigma, skew, kurt, exceed, up, peaks.size / duration, peaks)


def segment_maxima(y: np.ndarray, fs: float, duration: float) -> np.ndarray:
    """The largest value in each whole stretch `duration` long, the record interpolated first."""
    fine = upsampled(y - np.mean(y))
    per = int(round(duration * fs * UPSAMPLE))
    count = fine.size // per if per > 0 else 0
    if count == 0:
        return np.zeros(0)
    return fine[:count * per].reshape(count, per).max(axis=1)
