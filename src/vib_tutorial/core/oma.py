"""Operational (output-only) modal analysis: modes from the chain's response to ambient forces alone.

In service a structure is shaken by what it is used for, by wind or traffic: forces nobody
measures. Operational modal analysis identifies its modes from the responses alone, on the
assumption that those forces are broadband and random, so that the responses' spectra take
their shape from the structure, not from the input.

Ambient record
--------------
Independent Gaussian forces drive every mass (or one), white at the analog rate fs_sim =
R fs, optionally coloured by a second-order low-pass. The chain's response comes from the
same exact first-order-hold update as everywhere in the app, diagonalized into one
first-order recursive filter per eigenvalue (as in measurement.ChainResponse, here with an
input at every mass), run for the slowest transient's settling time before the record
starts so the record is stationary. Then the anti-alias filter, sampling every R-th sample,
and sensor noise sized to each channel's RMS, as in the virtual modal test.

Frequency domain decomposition (FDD)
------------------------------------
Welch's method gives the outputs' cross-spectral matrix G_yy(f), n × n at each line. For a
lightly damped structure under white input, near a mode's frequency the matrix is
dominated by that mode: G_yy ≈ d φ φᴴ. Its singular value decomposition at each line,
G_yy = U S Uᴴ, then has a first singular value that peaks at each mode, with the first
singular vector its shape. Peak picking on s₁ gives the frequencies (to the line) and the
shapes.

Enhanced FDD (EFDD) adds the damping: the lines round a peak whose first singular vector
still looks like the peak's (MAC above a threshold) are that mode's single-degree-of-freedom
"bell". Its inverse Fourier transform is the mode's autocorrelation function, a decaying
cosine: the zero crossings give the damped frequency and the extremes' logarithmic decay
the damping. Leakage (the window, the block length) and the bell's cut-off edges bias it.

Stochastic subspace identification (SSI-cov)
--------------------------------------------
The outputs' correlations R_i = E[y_{k+i} y_kᵀ] of a linear system driven by white noise are
an impulse response in disguise: R_i = C Aⁱ⁻¹ G, with A and C the sampled system's state and
output matrices. Stacked into a block Toeplitz matrix of lags 1 … 2s − 1, they factor as the
observability matrix times a controllability-like one, so the Toeplitz matrix has rank 2N
(two states per mode). Its SVD's leading p singular vectors give the observability matrix
O_p = U_p S_p^½ at model order p; C is its first block row, and A follows from its shift
structure, O↑ A = O↓ by least squares. A's eigenvalues μ are the poles, λ = fs ln μ, and
C's view of its eigenvectors the shapes. As with LSCF, the order is unknown: the poles are
found at every order and the stable ones picked from a stabilization diagram.

Harmonics
---------
A machine running on the structure adds a sine: a sharp peak in s₁, a pole with almost no
damping. It looks like a lightly damped mode. A mode's response, band-pass filtered round its
frequency, is Gaussian (kurtosis E[x⁴]/E[x²]² = 3); a sine's is 1.5. The kurtosis of the
response projected on a peak's shape and filtered round it tells them apart (`kurtosis`); a
mode with a sine in its band reads low too, but keeps its damping (`harmonics`).

Nothing here scales the shapes: without the force, the modal masses are unknown.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass

import numpy as np

from .frf_matrix import MAX_EIGVEC_CONDITION
from .identification import (
    MAX_ZETA,
    IdentifiedMode,
    ModeMatch,
    Stabilization,
    _classify,
    auto_select,
    lscf,
    lsfd,
    mac,
    match_modes,
    pole_from,
)
from .measurement import (
    AA_CUTOFF,
    MAX_OVERSAMPLE,
    MIN_OVERSAMPLE,
    Acquisition,
    Excitation,
    FrfEstimator,
    MeasurementSettings,
    Processing,
    Response,
    Window,
    settle_time,
    to_receptance,
)
from .modal import TWO_PI, ModalResult, modal_analysis
from .model import ChainSystem, state_space
from .simulator import foh_discretize

CHUNK = 1 << 15  # analog samples run at a time
PEAK_PROMINENCE = 0.3  # decades s₁ must rise above its surroundings to be a peak
MAC_MIN = 0.8  # EFDD: lines whose first singular vector has a MAC with the peak's above this form the bell
DECAY_FROM = 0.9  # EFDD: the extremes and zero crossings used, from where the envelope is below this...
DECAY_TO = 0.1  # ... to where it falls below this
MIN_EXTREMES = 3
MAX_SAMPLES = 2_000_000  # in the record, per channel
HARMONIC_KURTOSIS = 2.0  # below this a peak's filtered response is more like a sine (1.5) than a mode (3)
HARMONIC_ZETA = 0.01  # ... and its damping (a sine's pole has next to none) below this
KURTOSIS_BAND = 0.03  # the band-pass round a peak: ± this fraction of its frequency, at least the bell


class Colour(enum.Enum):
    WHITE = "White"
    LOW_PASS = "Low-pass"


@dataclass(frozen=True)
class AmbientSettings:
    fs: float  # Hz, the sample rate
    duration: float  # s, the record's length
    input_dof: int | None = None  # the mass the ambient forces act on; None: every mass
    colour: Colour = Colour.WHITE
    corner_hz: float = 5.0  # the low-pass's corner, if coloured
    response: Response = Response.ACCELERATION
    harmonic_hz: float = 0.0  # a machine running: a sine force on the first mass (0: none)
    harmonic_force: float = 0.0  # its amplitude, N (the random forces' RMS is 1 N at each analog sample)
    seed: int | None = None

    @property
    def samples(self) -> int:
        return int(round(self.duration * self.fs))


@dataclass
class AmbientRecord:
    """The sampled responses (noise-free) and a unit noise for each channel, added at processing."""

    settings: AmbientSettings
    y: np.ndarray  # (samples, n)
    unit_noise: np.ndarray  # (samples, n), standard normal

    @property
    def t(self) -> np.ndarray:
        return np.arange(self.y.shape[0]) / self.settings.fs

    def measured(self, noise: float) -> np.ndarray:
        """The responses with sensor noise of `noise` × each channel's RMS."""
        rms = np.sqrt(np.mean(self.y**2, axis=0))
        return self.y + noise * rms[None, :] * self.unit_noise


def oversampling(result: ModalResult, fs: float) -> int:
    """Analog samples per sample: enough to resolve every mode."""
    f_top = max((m.fn_hz for m in result.complex_modes), default=0.0)
    return int(np.clip(math.ceil(20.0 * f_top / fs), MIN_OVERSAMPLE, MAX_OVERSAMPLE))


def ambient_record(system: ChainSystem, settings: AmbientSettings, result: ModalResult | None = None) -> AmbientRecord:
    """Run the chain under the ambient forces (see the module's notes); friction is left out (OMA needs it linear)."""
    import scipy.signal  # deferred: scipy.signal adds ~0.9 s to startup

    s = settings
    if s.samples > MAX_SAMPLES:
        raise ValueError(f"the record is too long: at most {MAX_SAMPLES} samples")
    linear = ChainSystem(system.masses, system.stiffness, system.damping)
    result = result or modal_analysis(linear)
    n = linear.n
    R = oversampling(result, s.fs)
    fs_sim = s.fs * R
    h = 1.0 / fs_sim
    A, B = state_space(linear)
    Phi, G0, G1 = foh_discretize(A, B, h)
    inputs = list(range(n)) if s.input_dof is None else [s.input_dof]  # the random forces'
    harmonic = s.harmonic_hz > 0 and s.harmonic_force > 0
    cols = inputs + ([0] if harmonic and 0 not in inputs else [])  # and the machine's, on m1
    G0, G1 = G0[:, cols], G1[:, cols]
    rng = np.random.default_rng(s.seed)
    mu, V = np.linalg.eig(Phi)
    modal = np.linalg.cond(V) < MAX_EIGVEC_CONDITION
    if modal:
        Vinv = np.linalg.inv(V)
        b0, b1 = Vinv @ G0, Vinv @ G1
        eta = np.zeros(2 * n, dtype=complex)
    z = np.zeros(2 * n)
    f_prev = np.zeros(len(cols))
    colour = None
    if s.colour is Colour.LOW_PASS:
        colour = scipy.signal.butter(2, min(s.corner_hz, 0.45 * fs_sim), fs=fs_sim, output="sos")
        colour_zi = np.zeros((colour.shape[0], 2, len(inputs)))
    aa = scipy.signal.ellip(8, 0.05, 90.0, AA_CUTOFF * s.fs / 2.0, fs=fs_sim, output="sos")
    aa_zi = np.zeros((aa.shape[0], 2, n))
    out_a, out_b = A[n:], B[n:][:, cols]  # accelerations: a = A[n:] z + B[n:] f
    settle = settle_time(result)
    skip = int(math.ceil(settle * fs_sim / R)) * R  # analog samples before the record, whole samples
    total = skip + s.samples * R
    kept = []
    for start in range(0, total, CHUNK):
        m = min(CHUNK, total - start)
        f = np.zeros((m, len(cols)))
        f[:, :len(inputs)] = rng.standard_normal((m, len(inputs)))
        if colour is not None:
            f[:, :len(inputs)], colour_zi = scipy.signal.sosfilt(colour, f[:, :len(inputs)], axis=0, zi=colour_zi)
        if harmonic:
            t = (start + 1 + np.arange(m)) * h
            f[:, cols.index(0)] += s.harmonic_force * np.sin(TWO_PI * s.harmonic_hz * t)
        if modal:
            u = np.vstack([f_prev[None, :], f[:-1]]) @ b0.T + f @ b1.T  # (m, 2n)
            states = np.empty((m, 2 * n), dtype=complex)
            for r in range(2 * n):
                states[:, r], zf = scipy.signal.lfilter([1.0], [1.0, -mu[r]], u[:, r], zi=[mu[r] * eta[r]])
                eta[r] = states[-1, r]
            zs = (states @ V.T).real
        else:
            zs = np.empty((m, 2 * n))
            f0 = f_prev
            for k in range(m):
                z = Phi @ z + G0 @ f0 + G1 @ f[k]
                zs[k] = z
                f0 = f[k]
        f_prev = f[-1].copy()
        y = zs[:, :n] if s.response is Response.DISPLACEMENT else zs @ out_a.T + f @ out_b.T
        y, aa_zi = scipy.signal.sosfilt(aa, y, axis=0, zi=aa_zi)
        first = max(0, skip - start)  # the analog samples of this chunk that belong to the record
        sel = np.arange(start + first, start + m)
        sel = sel[(sel - skip) % R == R - 1]
        kept.append(y[sel - start])
    y = np.concatenate(kept)[: s.samples]
    noise_rng = np.random.default_rng(None if s.seed is None else s.seed + 1)
    return AmbientRecord(s, y, noise_rng.standard_normal(y.shape))


# ------------------------------------------------------------------ spectra
@dataclass
class Spectra:
    """The outputs' cross-spectral matrix, one-sided, by Welch's method (Hann window)."""

    freqs: np.ndarray  # (lines,)
    G: np.ndarray  # (lines, n, n), units²/Hz
    blocks: int
    block: int  # samples

    @property
    def resolution(self) -> float:
        return float(self.freqs[1] - self.freqs[0])


def cross_spectra(y: np.ndarray, fs: float, block: int, overlap: float = 0.5) -> Spectra:
    """G_yy(f) = 2 ⟨Y Yᴴ⟩ / (fs Σw²), averaged over Hann-windowed blocks."""
    import scipy.signal  # deferred: scipy.signal adds ~0.9 s to startup

    N = y.shape[0]
    block = min(block, N)
    hop = max(1, int(round(block * (1.0 - overlap))))
    w = scipy.signal.windows.hann(block, sym=False)
    starts = range(0, N - block + 1, hop)
    G = 0.0
    count = 0
    for i in starts:
        seg = y[i:i + block] - y[i:i + block].mean(axis=0)
        Y = np.fft.rfft(seg * w[:, None], axis=0)  # (lines, n)
        G = G + Y[:, :, None] * Y[:, None, :].conj()
        count += 1
    scale = 2.0 / (fs * np.sum(w**2) * count)
    G = G * scale
    G[0] /= 2.0
    if block % 2 == 0:
        G[-1] /= 2.0
    return Spectra(np.fft.rfftfreq(block, 1.0 / fs), G, count, block)


# ------------------------------------------------------------------ FDD
@dataclass
class Fdd:
    freqs: np.ndarray
    s: np.ndarray  # (lines, n) singular values, largest first
    u: np.ndarray  # (lines, n, n) singular vectors in the columns

    @property
    def first(self) -> np.ndarray:
        """(lines, n): the first singular vector at each line."""
        return self.u[:, :, 0]


def fdd(spectra: Spectra) -> Fdd:
    """The SVD of the cross-spectral matrix at every line (it is Hermitian: an eigendecomposition)."""
    vals, vecs = np.linalg.eigh(spectra.G)
    return Fdd(spectra.freqs, np.clip(vals[:, ::-1], 0.0, None), vecs[:, :, ::-1])


def pick_peaks(result: Fdd, band: tuple[float, float], max_modes: int) -> list[int]:
    """Lines where s₁ peaks in the band, by prominence (decades), at most max_modes, lowest first."""
    import scipy.signal  # deferred: scipy.signal adds ~0.9 s to startup

    lo, hi = band
    idx = np.flatnonzero((result.freqs > 0) & (result.freqs >= lo) & (result.freqs <= hi))
    if idx.size < 3:
        return []
    s1 = np.log10(np.maximum(result.s[idx, 0], 1e-300))
    found, props = scipy.signal.find_peaks(s1, prominence=PEAK_PROMINENCE)
    strongest = np.argsort(props["prominences"])[::-1][:max_modes]
    return sorted(int(idx[p]) for p in found[strongest])


def real_shape(u: np.ndarray) -> np.ndarray:
    """A complex shape turned so its largest entry is real and positive, scaled to 1."""
    return u / u[np.argmax(np.abs(u))]


@dataclass
class FddMode:
    line: int  # the peak's line
    peak_hz: float  # FDD: the peak's frequency, to the line
    shape: np.ndarray  # the first singular vector there, largest entry 1
    bell: np.ndarray  # (lines,) bool: the EFDD bell's lines
    fn_hz: float  # EFDD (nan if it couldn't fit the decay)
    zeta: float
    lag: np.ndarray  # s, the autocorrelation's time axis
    correlation: np.ndarray  # the bell's autocorrelation, normalized to 1 at zero lag
    extremes: np.ndarray  # indices into lag of the extremes fitted
    crossings: np.ndarray  # s, the zero crossings fitted

    @property
    def identified(self) -> IdentifiedMode:
        """As identification's IdentifiedMode (EFDD's pole, or the peak's undamped one), to match with the exact modes."""
        if math.isfinite(self.fn_hz):
            pole = pole_from(self.fn_hz, self.zeta)
        else:
            pole = complex(0.0, TWO_PI * self.peak_hz)
        return IdentifiedMode(pole, self.shape.copy(), self.shape.conj())


def efdd(result: Fdd, line: int, mac_min: float = MAC_MIN) -> FddMode:
    """The mode at a peak line: its shape, its SDOF bell, and the frequency and damping from the bell's
    autocorrelation (see the module's notes)."""
    u0 = result.first[line]
    lines = result.freqs.size
    bell = np.zeros(lines, dtype=bool)
    bell[line] = True
    for step in (-1, 1):
        k = line + step
        while 0 < k < lines and mac(result.first[k], u0) >= mac_min:
            bell[k] = True
            k += step
    spectrum = np.where(bell, result.s[:, 0], 0.0)
    r = np.fft.irfft(spectrum)
    dt = 1.0 / (2.0 * result.freqs[-1])
    r = r[: r.size // 2]
    r = r / r[0]
    lag = np.arange(r.size) * dt
    peak = result.freqs[line]
    shape = real_shape(u0)
    # the extremes of |r|, alternate signs, between DECAY_FROM and DECAY_TO of the start
    mag = np.abs(r)
    ext = np.flatnonzero((mag[1:-1] >= mag[:-2]) & (mag[1:-1] > mag[2:])) + 1
    end = np.flatnonzero(mag[ext] < DECAY_TO)
    ext = ext[: end[0]] if end.size else ext
    ext = ext[mag[ext] <= DECAY_FROM]
    nan = math.nan
    if ext.size < MIN_EXTREMES:
        return FddMode(line, peak, shape, bell, nan, nan, lag, r, ext, np.empty(0))
    lo, hi = lag[ext[0]], lag[ext[-1]]
    sign = np.flatnonzero(np.signbit(r[:-1]) != np.signbit(r[1:]))
    cross = lag[sign] + dt * r[sign] / (r[sign] - r[sign + 1])
    cross = cross[(cross >= lo - 0.5 * (hi - lo) / ext.size) & (cross <= hi)]
    if cross.size < 2:
        return FddMode(line, peak, shape, bell, nan, nan, lag, r, ext, cross)
    half_period = np.polyfit(np.arange(cross.size), cross, 1)[0]
    sigma = -np.polyfit(lag[ext], np.log(mag[ext]), 1)[0]
    wd = math.pi / half_period
    wn = math.hypot(wd, sigma)
    return FddMode(line, peak, shape, bell, wn / TWO_PI, sigma / wn, lag, r, ext, cross)


def identify(result: Fdd, lines: list[int], mac_min: float = MAC_MIN) -> list[FddMode]:
    """EFDD at each peak line, the highest first; a peak inside a higher one's bell is that mode again
    (a ripple on its s₁) and is dropped. Lowest frequency first."""
    modes: list[FddMode] = []
    for line in sorted(lines, key=lambda k: -result.s[k, 0]):
        if not any(m.bell[line] for m in modes):
            modes.append(efdd(result, line, mac_min))
    return sorted(modes, key=lambda m: m.line)


def compare(modes: list[FddMode], result: ModalResult, f_max: float) -> list[ModeMatch]:
    """The identified modes paired with the exact ones below f_max (identification.match_modes)."""
    return match_modes([m.identified for m in modes], result, f_max)


# ------------------------------------------------------------------ harmonics
def kurtosis(y: np.ndarray, fs: float, shape: np.ndarray, freq_hz: float, half_width: float) -> float:
    """E[q⁴]/E[q²]² of the response projected on `shape` and band-passed over freq_hz ± half_width
    (at least KURTOSIS_BAND of it): 3 for a mode's random response, 1.5 for a sine."""
    import scipy.signal  # deferred: scipy.signal adds ~0.9 s to startup

    half = max(half_width, KURTOSIS_BAND * freq_hz)
    lo, hi = max(freq_hz - half, 1e-3 * fs), min(freq_hz + half, 0.49 * fs)
    if not lo < hi:
        return math.nan
    q = y @ shape.conj()
    q = (q * np.exp(-1j * np.angle(q @ q) / 2)).real  # turned to its real direction
    sos = scipy.signal.butter(4, (lo, hi), btype="bandpass", fs=fs, output="sos")
    x = scipy.signal.sosfiltfilt(sos, q)
    x = x[x.size // 20: -x.size // 20 or None]  # the filter's ends
    m2 = np.mean(x**2)
    return float(np.mean(x**4) / m2**2) if m2 > 0 else math.nan


def harmonic_flags(y: np.ndarray, fs: float, result: Fdd, modes: list[FddMode]) -> list[float]:
    """The kurtosis at each FDD mode: below HARMONIC_KURTOSIS, a harmonic rather than a mode."""
    df = float(result.freqs[1] - result.freqs[0])
    out = []
    for m in modes:
        span = np.flatnonzero(m.bell)
        half = 0.5 * (span[-1] - span[0] + 1) * df
        out.append(kurtosis(y, fs, m.shape, m.peak_hz, half))
    return out


# ------------------------------------------------------------------ SSI-cov
@dataclass
class SsiStabilization(Stabilization):
    """The SSI poles at every order (as LSCF's), with each pole's shape."""

    shapes: list[np.ndarray] = None  # per order, (n, poles) complex


def correlations(y: np.ndarray, lags: int) -> np.ndarray:
    """R_i = mean of y_{k+i} y_kᵀ for i = 0 … lags − 1: (lags, n, n)."""
    N = y.shape[0]
    y = y - y.mean(axis=0)
    return np.stack([y[i:].T @ y[:N - i] / (N - i) for i in range(lags)])


def ssi_cov(y: np.ndarray, fs: float, rows: int, max_order: int, band: tuple[float, float]) -> SsiStabilization:
    """Covariance-driven SSI (see the module's notes) at every even order 2 … max_order, `rows` block rows."""
    n = y.shape[1]
    R = correlations(y, 2 * rows)
    # T[a, b] = R_{rows + a − b}: lags 1 … 2 rows − 1
    T = np.block([[R[rows + a - b] for b in range(rows)] for a in range(rows)])
    U, S, _ = np.linalg.svd(T)
    top = min(max_order, n * (rows - 1))
    lo, hi = band
    orders, poles, status, runs, shapes = [], [], [], [], []
    for p in range(2, top + 1, 2):
        O = U[:, :p] * np.sqrt(S[:p])
        A = np.linalg.lstsq(O[:-n], O[n:], rcond=None)[0]
        mu, psi = np.linalg.eig(A)
        keep = np.abs(mu) > 0
        lam = np.log(mu[keep].astype(complex)) * fs
        phi = (O[:n] @ psi)[:, keep]
        fn = np.abs(lam) / TWO_PI
        ok = (lam.imag > 0) & (lam.real < 0) & (fn >= lo) & (fn <= hi) & (-lam.real / np.abs(lam) <= MAX_ZETA)
        lam, phi = lam[ok], phi[:, ok]
        order = np.argsort(np.abs(lam))
        lam, phi = lam[order], phi[:, order]
        st, run = _classify(lam, poles[-1] if poles else None, runs[-1] if runs else None)
        orders.append(p)
        poles.append(lam)
        status.append(st)
        runs.append(run)
        shapes.append(phi)
    return SsiStabilization(orders, poles, status, runs, band, shapes)


def ssi_modes(stab: SsiStabilization, picked: list[tuple[int, int]]) -> list[IdentifiedMode]:
    """The picked poles (order, index) as modes, lowest first, each shape scaled to its largest entry."""
    out = []
    for order, i in picked:
        k = stab.orders.index(order)
        phi = real_shape(stab.shapes[k][:, i])
        out.append(IdentifiedMode(complex(stab.poles[k][i]), phi, phi.conj()))
    return sorted(out, key=lambda m: m.fn_hz)


# ------------------------------------------------------------------ the hammer test
HAMMER_BLOCK = 2048
HAMMER_AVERAGES = 5
HAMMER_NOISE = 0.01  # response noise, a fraction of each channel's peak
HAMMER_ORDER = 30


def hammer_modes(system: ChainSystem, result: ModalResult, fs: float, seed: int | None = None
                 ) -> tuple[list[IdentifiedMode], float]:
    """The virtual modal test's way to the modes, for comparison: hammer hits on m1, the force and the
    displacements measured, H1 averaged under a force-exponential window, LSCF's stable poles picked
    automatically and LSFD's residues, the window's damping taken off. The modes and the force window's
    time constant (s)."""
    linear = ChainSystem(system.masses, system.stiffness, system.damping)
    settings = MeasurementSettings(excitation=Excitation.IMPACT, input_dof=0, response=Response.DISPLACEMENT,
                                   fs=fs, block=HAMMER_BLOCK, averages=HAMMER_AVERAGES, seed=seed)
    processing = Processing(window=Window.FORCE_EXPONENTIAL, exp_end=0.05, response_noise=HAMMER_NOISE)
    acq = Acquisition(linear, settings, result)
    while not acq.done:
        acq.step()
    est = FrfEstimator(acq, processing).estimate()
    H = to_receptance(est.freqs, est.H, settings.response)
    band = (0.0, settings.band)
    stab = lscf(est.freqs, H, band, HAMMER_ORDER)
    ident = lsfd(est.freqs, H, band, [stab.pole(o, i) for o, i in auto_select(stab)])
    tau = processing.exp_tau(settings)
    return sorted(ident.shifted(1.0 / tau).modes, key=lambda m: m.fn_hz), tau


def harmonics(zetas: list[float], kurt: list[float]) -> list[bool]:
    """Which peaks (or poles) are harmonics: a kurtosis below HARMONIC_KURTOSIS and almost no damping. A sine
    inside a mode's filtered band lowers the mode's kurtosis too, but the mode keeps its damping."""
    return [math.isfinite(k) and k < HARMONIC_KURTOSIS and not (math.isfinite(z) and z >= HARMONIC_ZETA)
            for z, k in zip(zetas, kurt)]
