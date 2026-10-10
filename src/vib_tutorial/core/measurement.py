"""Virtual modal test: excite the chain, record force and response, estimate the FRF.

This mimics a laboratory test. A force is applied at one mass (impact hammer,
shaker driven by random or chirp signals, or a stepped sine), and a data
acquisition system samples the force and the response of every mass, measured
as displacement or (as with accelerometers) acceleration. The FRF, receptance
or accelerance, is then estimated from those sampled signals alone, as it would be from
real measurements, so it can be compared with the exact H(w).

Signal chain
------------
1. "Analog" signals. The excitation is generated at fs_sim = R fs (R >= 4
   samples per acquired sample), and the chain's response is found with the
   same exact first-order-hold discretization as the simulator. The discrete
   system is diagonalized, so each mode is a first-order recursive filter and
   long records are fast. With friction on any mass the chain is nonlinear and
   is stepped by the simulator instead (stick-slip included), which is slower.
2. Anti-alias filter (optional): an elliptic low-pass with its passband edge
   at AA_CUTOFF x fs/2, on every channel. The same filter on force and
   response cancels in their ratio.
3. Sampling: every R-th analog sample is kept. Without the filter, response
   above fs/2 folds back (aliases) into the measured band.
4. Noise: white Gaussian noise on each channel, sized as a fraction of that
   channel's peak in the block (the range an auto-ranging analyzer would pick).
   Noise is added when the data are processed, so its level can be changed
   without measuring again.

Estimation
----------
Each block of Nb samples is windowed and Fourier transformed. Averaged over
blocks, G_ff = <|F|^2>, G_xx = <|X|^2> and G_xf = <X F*> give

    H1 = G_xf / G_ff     unbiased by response noise, biased low by force noise
    H2 = G_xx / G_fx     unbiased by force noise, biased high by response noise
    coherence = |G_xf|^2 / (G_ff G_xx)

A stepped sine instead fits a sine to force and response at each frequency.

Force level
-----------
The force is scaled by ``force_level``. A linear chain's FRF does not depend on
it (the noise is sized to each channel, so it scales too). With friction it
does, which is how a lab checks linearity: measure at two levels and overlay.
"""

from __future__ import annotations

import enum
import math
import time
from dataclasses import dataclass

import numpy as np

from .frf_matrix import MAX_EIGVEC_CONDITION
from .modal import TWO_PI, ModalResult, modal_analysis
from .forcing import ForceController, ForceSettings
from .model import ChainSystem, state_space
from .simulator import Simulator, foh_discretize

# At force level 1:
FORCE_RMS = 10.0  # N, random and chirp excitation
IMPACT_PEAK = 100.0  # N
SINE_AMPLITUDE = 10.0  # N
AA_CUTOFF = 0.8  # anti-alias passband edge, and excitation band, as a fraction of fs/2
MIN_OVERSAMPLE = 4
MAX_OVERSAMPLE = 64
PRETRIGGER = 4  # samples recorded before the hammer hit
FORCE_WINDOW_MARGIN = 16  # samples kept after the hit by the force window (filter ringing)
FORCE_WINDOW_TAPER = 4  # samples
SETTLE_TOL = 1e-3  # transient left when a periodic or stepped-sine measurement starts
MAX_SETTLE = 300.0  # s
SINE_CYCLES = 8  # measured per stepped-sine frequency
MIN_SINE_SAMPLES = 64
FIT_POINTS_PER_CYCLE = 32  # drawing the fitted stepped sine
CHUNK = 1 << 16  # analog samples simulated at a time
FRICTION_PIECE = 2048  # analog samples between pauses (Acquisition.work) when friction makes them slow


class Excitation(enum.Enum):
    IMPACT = "Impact hammer"
    RANDOM = "Random (continuous)"
    BURST_RANDOM = "Burst random"
    PERIODIC_RANDOM = "Periodic random"
    CHIRP = "Periodic chirp"
    STEPPED_SINE = "Stepped sine"

    @property
    def settles(self) -> bool:
        """Is the transient waited out before measuring (steady-state excitations)?"""
        return self in (Excitation.RANDOM, Excitation.PERIODIC_RANDOM, Excitation.CHIRP, Excitation.STEPPED_SINE)

    @property
    def waits(self) -> bool:
        """Does the response die away before each block (so it starts from rest)?"""
        return self is Excitation.IMPACT


class Window(enum.Enum):
    RECTANGULAR = "Rectangular (none)"
    HANN = "Hann"
    FLAT_TOP = "Flat top"
    FORCE_EXPONENTIAL = "Force + exponential"


class Response(enum.Enum):
    """What the response sensors measure, and so which form of the FRF is estimated."""

    DISPLACEMENT = "Displacement"
    ACCELERATION = "Acceleration"

    @property
    def symbol(self) -> str:
        return "x" if self is Response.DISPLACEMENT else "a"

    @property
    def unit(self) -> str:
        return "m" if self is Response.DISPLACEMENT else "m/s²"

    @property
    def frf_name(self) -> str:
        return "receptance" if self is Response.DISPLACEMENT else "accelerance"

    @property
    def power(self) -> int:
        """p in FRF = (s)^p x receptance: each derivative multiplies by s = iw."""
        return 0 if self is Response.DISPLACEMENT else 2


class Estimator(enum.Enum):
    H1 = "H1 = G_xf / G_ff"
    H2 = "H2 = G_xx / G_fx"


@dataclass
class MeasurementSettings:
    """What is measured and how; changing any of these needs a new measurement."""

    excitation: Excitation = Excitation.IMPACT
    input_dof: int = 0
    response: Response = Response.DISPLACEMENT
    fs: float = 20.0  # Hz, acquisition sample rate
    block: int = 1024  # samples per block (Nb)
    averages: int = 10  # blocks (stepped sine: ignored)
    overlap: float = 0.0  # continuous random only, fraction of a block
    anti_alias: bool = True
    tip_width: float = 0.05  # s, duration of the hammer's half-sine pulse
    burst: float = 0.5  # burst random: fraction of the block the shaker is on
    sine_points: int = 100  # stepped sine
    force_level: float = 1.0  # multiplies FORCE_RMS, IMPACT_PEAK and SINE_AMPLITUDE
    seed: int | None = None

    @property
    def duration(self) -> float:
        """Block length T = Nb / fs (s)."""
        return self.block / self.fs

    @property
    def resolution(self) -> float:
        """Frequency resolution df = 1 / T (Hz)."""
        return self.fs / self.block

    @property
    def band(self) -> float:
        """Excitation band and anti-alias passband (Hz)."""
        return AA_CUTOFF * self.fs / 2


@dataclass
class Processing:
    """How the recorded data are turned into an FRF; can change without measuring again."""

    window: Window = Window.HANN
    exp_end: float = 0.05  # exponential window's value at the end of the block
    estimator: Estimator = Estimator.H1
    force_noise: float = 0.0  # noise std as a fraction of the channel's peak
    response_noise: float = 0.0

    def exp_tau(self, settings: MeasurementSettings) -> float:
        """Time constant of the exponential window (s); infinite when it is off (exp_end = 1)."""
        if self.exp_end >= 1.0:
            return math.inf
        return settings.duration / math.log(1.0 / self.exp_end)


# ------------------------------------------------------------------ helpers
def transfer(system: ChainSystem, s: np.ndarray, input_dof: int) -> np.ndarray:
    """H(s) = (Ms^2 + Cs + K)^-1 e_input at complex frequencies s, shape (len(s), n).

    On the imaginary axis (s = iw) this is the receptance. A block multiplied
    by the exponential window e^{-t/tau} measures H(iw + 1/tau): every pole
    moves 1/tau further into the left half plane.
    """
    M, C, K = system.matrices()
    s = np.asarray(s, dtype=complex)
    Z = K[None] + s[:, None, None] * C[None] + (s**2)[:, None, None] * M[None]
    e = np.zeros((s.size, system.n, 1), dtype=complex)
    e[:, input_dof] = 1.0
    return np.linalg.solve(Z, e)[:, :, 0]


def from_receptance(freqs_hz: np.ndarray, H: np.ndarray, response: Response) -> np.ndarray:
    """The FRF the sensors measure, (iw)^p H, from receptance H (rows: frequencies)."""
    iw = 1j * TWO_PI * np.asarray(freqs_hz, dtype=float)
    factor = iw**response.power
    return H * (factor[:, None] if np.ndim(H) > 1 else factor)


def to_receptance(freqs_hz: np.ndarray, H: np.ndarray, response: Response) -> np.ndarray:
    """Receptance from a measured FRF: accelerance divided by -w^2 (NaN at 0 Hz)."""
    if response.power == 0:
        return H
    iw = 1j * TWO_PI * np.asarray(freqs_hz, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where(iw != 0, 1.0 / np.where(iw != 0, iw, 1.0) ** response.power, np.nan)
    return H * (factor[:, None] if np.ndim(H) > 1 else factor)


def impact_spectrum(width: float, freqs_hz: np.ndarray) -> np.ndarray:
    """|F(f)| / F(0) of a half-sine pulse lasting `width` s.

    It rolls off above about 1 / width and first reaches zero at 1.5 / width:
    a harder hammer tip gives a shorter pulse, which excites higher frequencies.
    """
    u = 2.0 * np.asarray(freqs_hz, dtype=float) * width
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.abs(np.cos(0.5 * np.pi * u) / (1.0 - u**2))
    return np.where(np.abs(u - 1.0) < 1e-9, np.pi / 4, r)


def settle_time(result: ModalResult) -> float:
    """Time for the slowest transient to decay to SETTLE_TOL (s), at most MAX_SETTLE."""
    sigma = min((-m.eigenvalue.real for m in result.complex_modes), default=0.0)
    if sigma <= 1e-9:
        return MAX_SETTLE
    return min(MAX_SETTLE, math.log(1.0 / SETTLE_TOL) / sigma)


def oversampling(result: ModalResult, settings: MeasurementSettings) -> int:
    """Analog samples per acquired sample: enough to resolve every mode and the hammer pulse."""
    f_top = max((m.fn_hz for m in result.complex_modes), default=0.0)
    need = 20.0 * f_top
    if settings.excitation is Excitation.IMPACT:
        need = max(need, 10.0 / settings.tip_width)
    return int(np.clip(math.ceil(need / settings.fs), MIN_OVERSAMPLE, MAX_OVERSAMPLE))


def block_windows(settings: MeasurementSettings, processing: Processing) -> tuple[np.ndarray, np.ndarray]:
    """(force window, response window), each of length Nb."""
    import scipy.signal  # deferred: scipy.signal adds ~0.9 s to startup

    nb = settings.block
    w = processing.window
    if w is Window.RECTANGULAR:
        ones = np.ones(nb)
        return ones, ones
    if w is Window.HANN:
        win = scipy.signal.windows.hann(nb, sym=False)
        return win, win
    if w is Window.FLAT_TOP:
        win = scipy.signal.windows.flattop(nb, sym=False)
        return win, win
    # Force window: keep the hit (and the anti-alias filter's ringing), drop the
    # noise after it. Exponential window on both channels.
    t = np.arange(nb) / settings.fs
    decay = np.exp(-t / processing.exp_tau(settings))
    end = PRETRIGGER + math.ceil(settings.tip_width * settings.fs) + FORCE_WINDOW_MARGIN
    force = np.zeros(nb)
    force[: min(end, nb)] = 1.0
    taper = 0.5 * (1.0 + np.cos(np.pi * np.arange(1, FORCE_WINDOW_TAPER + 1) / (FORCE_WINDOW_TAPER + 1)))
    stop = min(end + FORCE_WINDOW_TAPER, nb)
    force[end:stop] = taper[: max(0, stop - end)]
    return force * decay, decay


# ---------------------------------------------------------- the "analog" chain
class ChainResponse:
    """Exact FOH response of the chain to a force at one mass, sampled every h seconds.

    The response is the displacements, or the accelerations
    a = M^-1 (f - C v - K x), which the state and the force give exactly.

    Diagonalizing Phi = V diag(mu) V^-1 turns the update into one first-order
    filter per eigenvalue, eta_r[k] = mu_r eta_r[k-1] + b0_r f[k-1] + b1_r f[k],
    which scipy runs in C. A defective eigenvalue (a free chain's rigid-body
    lambda = 0) has no diagonal form, so that case steps the state directly.
    With friction the simulator steps it (Simulator.drive), stick-slip and all.
    """

    def __init__(
        self, system: ChainSystem, h: float, input_dof: int, response: Response = Response.DISPLACEMENT
    ) -> None:
        A, B = state_space(system)
        Phi, G0, G1 = foh_discretize(A, B, h)
        self.n = n = system.n
        if response is Response.DISPLACEMENT:  # y = C_out z + d f
            self._C, self._d = np.eye(2 * n)[:n], np.zeros(n)
        else:
            self._C, self._d = A[n:].copy(), B[n:, input_dof].copy()
        self.state = np.zeros(2 * system.n)
        self.f_prev = 0.0
        self._Phi, self._g0, self._g1 = Phi, G0[:, input_dof].copy(), G1[:, input_dof].copy()
        self.h = h
        self.displacement = response is Response.DISPLACEMENT
        self._sim = None
        if np.any(system.friction > 0):
            self._sim = Simulator(system, ForceController(ForceSettings(target=input_dof)))
        mu, V = np.linalg.eig(Phi)
        self._modal = None
        if self._sim is None and np.linalg.cond(V) < MAX_EIGVEC_CONDITION:
            Vinv = np.linalg.inv(V)
            self._modal = (mu, V, Vinv, Vinv @ self._g0, Vinv @ self._g1)

    def run(self, f: np.ndarray) -> np.ndarray:
        """Responses (len(f), n) at the instants of the force samples f.

        f[k] is the force at the end of step k; the force before the first
        step is the last sample of the previous call.
        """
        f = np.asarray(f, dtype=float)
        out = np.empty((f.size, self.n))
        for start in range(0, f.size, CHUNK):
            seg = f[start : start + CHUNK]
            out[start : start + seg.size] = self._run(seg)
        return out

    def _run(self, f: np.ndarray) -> np.ndarray:
        import scipy.signal

        if self._sim is not None:
            z, a = self._sim.drive(f, self.h)
            self.state, self.f_prev = z[-1].copy(), float(f[-1])
            return z[:, : self.n] if self.displacement else a
        if self._modal is not None:
            mu, V, Vinv, b0, b1 = self._modal
            eta0 = Vinv @ self.state
            eta = np.empty((f.size, mu.size), dtype=complex)
            for r in range(mu.size):
                zi = [b0[r] * self.f_prev + mu[r] * eta0[r]]
                eta[:, r], _ = scipy.signal.lfilter([b1[r], b0[r]], [1.0, -mu[r]], f, zi=zi)
            z = (eta @ V.T).real
        else:
            z = np.empty((f.size, 2 * self.n))
            zk, f0 = self.state, self.f_prev
            for k, f1 in enumerate(f):
                zk = self._Phi @ zk + self._g0 * f0 + self._g1 * f1
                z[k] = zk
                f0 = f1
        self.state = z[-1].copy()
        self.f_prev = float(f[-1])
        return z @ self._C.T + f[:, None] * self._d[None, :]


# --------------------------------------------------------------- acquisition
@dataclass
class SinePoint:
    """One stepped-sine frequency: fitted complex amplitudes, x(t) = Re(X e^{iwt})."""

    freq_hz: float
    force: complex
    response: np.ndarray  # (n,) complex
    force_noise: complex  # fit of the unit noise record, scaled when processed
    response_noise: np.ndarray  # (n,) complex


class Acquisition:
    """Runs the test one average (or one stepped-sine frequency) at a time.

    The records are kept noise-free; each chunk of samples has its own seeded
    unit noise, so processing can add any noise level to the same data.

    ``step`` measures the next record at once. ``work`` measures for a given
    time and can stop part-way through a record, so a slow (friction) chain
    does not hold up the GUI; the record carries on with the next call.
    """

    def __init__(self, system: ChainSystem, settings: MeasurementSettings, result: ModalResult | None = None) -> None:
        import scipy.signal

        self.system = system
        self.settings = s = settings
        self.result = result or modal_analysis(system)
        self.n = system.n
        self.rng = np.random.default_rng(s.seed)
        self.noise_seed = int(self.rng.integers(2**62))
        self.oversample = R = oversampling(self.result, s)
        self.fs_sim = s.fs * R
        self.settle = settle_time(self.result)
        self._chain = ChainResponse(system, 1.0 / self.fs_sim, s.input_dof, s.response)
        self._aa = None
        if s.anti_alias:
            self._aa = scipy.signal.ellip(8, 0.05, 90.0, s.band, fs=self.fs_sim, output="sos")
            self._aa_zi = np.zeros((self._aa.shape[0], 2, self.n + 1))
        self._shaker = scipy.signal.butter(8, s.band, fs=self.fs_sim, output="sos")
        self._shaker_zi = np.zeros((self._shaker.shape[0], 2))
        self.t_analog = 0.0  # s of analog signal generated so far
        # Recorded (sampled, noise-free) data, one chunk per step.
        self._chunks: list[tuple[np.ndarray, np.ndarray]] = []
        self._starts: list[int] = []
        self.samples = 0
        self.points: list[SinePoint] = []
        self.last_sine: tuple[np.ndarray, ...] | None = None  # (t, f, x, unit noise f, x) of the last point
        self._settled = False
        self._chirp = self._chirp_period() if s.excitation is Excitation.CHIRP else None
        self._piece = FRICTION_PIECE if np.any(system.friction > 0) else CHUNK
        self._pending = None  # the record being measured (a generator), if work stopped part-way

    # ---------------------------------------------------------- properties
    @property
    def stepped(self) -> bool:
        return self.settings.excitation is Excitation.STEPPED_SINE

    @property
    def hop(self) -> int:
        """Samples between the starts of consecutive blocks."""
        s = self.settings
        if s.excitation is Excitation.RANDOM:
            return max(1, round(s.block * (1.0 - s.overlap)))
        return s.block

    @property
    def blocks(self) -> int:
        """Complete blocks recorded."""
        nb = self.settings.block
        return 0 if self.samples < nb else (self.samples - nb) // self.hop + 1

    @property
    def progress(self) -> tuple[int, int]:
        """(done, wanted): blocks, or stepped-sine frequencies."""
        s = self.settings
        if self.stepped:
            return len(self.points), s.sine_points
        return min(self.blocks, s.averages), s.averages

    @property
    def done(self) -> bool:
        done, wanted = self.progress
        return done >= wanted

    def sine_frequencies(self) -> np.ndarray:
        s = self.settings
        return s.band * np.arange(1, s.sine_points + 1) / s.sine_points

    # --------------------------------------------------------------- steps
    def step(self) -> None:
        """Measure the next block (or stepped-sine frequency)."""
        self.work(math.inf)

    def work(self, budget: float) -> bool:
        """Measure for about `budget` seconds; True once the record in progress is complete."""
        if self._pending is None:
            if self.done:
                return True
            self._pending = self._step()
        end = time.perf_counter() + budget
        for _ in self._pending:
            if time.perf_counter() >= end:
                return False
        self._pending = None
        return True

    def _step(self):
        """Measure the next record, pausing (yield) between pieces of analog signal."""
        s = self.settings
        nb, R = s.block, self.oversample
        kind = s.excitation
        if kind is Excitation.STEPPED_SINE:
            yield from self._sine_point(self.sine_frequencies()[len(self.points)])
            return
        if kind is Excitation.IMPACT:
            if self._chunks:  # wait for the last hit to die away before hitting again
                yield from self._settle_with(np.zeros)
            f = np.zeros(nb * R)
            k = np.arange(math.ceil(s.tip_width * self.fs_sim) + 1)
            pulse = s.force_level * IMPACT_PEAK * np.sin(np.pi * k / (s.tip_width * self.fs_sim))
            start = PRETRIGGER * R
            f[start : start + k.size] = np.clip(pulse, 0.0, None)[: max(0, f.size - start)]
            yield from self._record(f)
        elif kind is Excitation.BURST_RANDOM:
            f = self._shaker_noise(nb * R)
            f[round(s.burst * nb) * R :] = 0.0
            yield from self._record(f)
        elif kind is Excitation.RANDOM:
            if not self._settled:
                yield from self._settle_with(self._shaker_noise)
            yield from self._record(self._shaker_noise((nb if not self._chunks else self.hop) * R))
        else:  # periodic random, chirp: steady state of a signal repeated every block
            period = self._chirp if kind is Excitation.CHIRP else self._random_period()
            if kind is Excitation.PERIODIC_RANDOM or not self._settled:
                repeats = max(1, math.ceil(self.settle / s.duration))
                yield from self._run_analog(np.tile(period, repeats), keep=False)
                self._settled = True
            yield from self._record(period)

    def _settle_with(self, source):
        """Run the chain for the settling time on force samples source(m), recording nothing."""
        remaining = round(self.settle * self.fs_sim)
        while remaining > 0:
            m = min(remaining, CHUNK)
            yield from self._run_analog(source(m), keep=False)
            remaining -= m
        self._settled = True

    def _shaker_noise(self, m: int) -> np.ndarray:
        """Band-limited Gaussian noise with RMS force_level x FORCE_RMS, continuous across calls."""
        import scipy.signal

        s = self.settings
        white = self.rng.standard_normal(m) * s.force_level * FORCE_RMS * math.sqrt(self.fs_sim / (2.0 * s.band))
        out, self._shaker_zi = scipy.signal.sosfilt(self._shaker, white, zi=self._shaker_zi)
        return out

    def _random_period(self) -> np.ndarray:
        """One period of periodic random: flat magnitude, random phase, up to the band."""
        s = self.settings
        m = s.block * self.oversample
        spectrum = np.zeros(m // 2 + 1, dtype=complex)
        top = int(s.band * s.duration)
        spectrum[1 : top + 1] = np.exp(1j * self.rng.uniform(0.0, TWO_PI, top))
        p = np.fft.irfft(spectrum, m)
        return p * s.force_level * FORCE_RMS / np.sqrt(np.mean(p**2))

    def _chirp_period(self) -> np.ndarray:
        """Linear sweep from 0 to the band edge over one block, repeated."""
        s = self.settings
        t = np.arange(1, s.block * self.oversample + 1) / self.fs_sim
        return math.sqrt(2.0) * s.force_level * FORCE_RMS * np.sin(np.pi * s.band * t**2 / s.duration)

    def _run_analog(self, f: np.ndarray, keep: bool = True):
        """Drive the chain with analog force samples; return them sampled at fs (f, x) if `keep`.

        A generator (use ``yield from``): it pauses after each piece of the signal.
        """
        import scipy.signal

        kept = []
        for start in range(0, f.size, self._piece):
            seg = f[start : start + self._piece]
            data = np.column_stack([seg, self._chain.run(seg)])
            self.t_analog += seg.size / self.fs_sim
            if self._aa is not None:
                data, self._aa_zi = scipy.signal.sosfilt(self._aa, data, axis=0, zi=self._aa_zi)
            if keep:
                kept.append(data)
            yield
        if not keep:
            return None
        R = self.oversample
        sampled = np.concatenate(kept)[R - 1 :: R]
        return sampled[:, 0].copy(), sampled[:, 1:].copy()

    def _record(self, f_analog: np.ndarray):
        f, x = yield from self._run_analog(f_analog)
        self._starts.append(self.samples)
        self._chunks.append((f, x))
        self.samples += f.size

    def _sine_point(self, freq: float):
        s = self.settings
        w = TWO_PI * freq
        settle = max(self.settle, 5.0 / freq)
        n_meas = max(MIN_SINE_SAMPLES, math.ceil(SINE_CYCLES * s.fs / freq))
        m_settle = round(settle * self.fs_sim)
        t = np.arange(1, m_settle + n_meas * self.oversample + 1) / self.fs_sim
        f_analog = s.force_level * SINE_AMPLITUDE * np.sin(w * t)
        yield from self._run_analog(f_analog[:m_settle], keep=False)
        f, x = yield from self._run_analog(f_analog[m_settle:])
        tm = np.arange(n_meas) / s.fs
        rng = np.random.default_rng([self.noise_seed, len(self.points)])
        nf = rng.standard_normal(n_meas)
        nx = rng.standard_normal((n_meas, self.n))
        # Least-squares fit of c cos + s sin + offset: amplitude c - i s.
        basis = np.column_stack([np.cos(w * tm), np.sin(w * tm), np.ones(n_meas)])
        coef = np.linalg.lstsq(basis, np.column_stack([f, x, nf, nx]), rcond=None)[0]
        amp = coef[0] - 1j * coef[1]
        n = self.n
        self.points.append(SinePoint(freq, amp[0], amp[1 : n + 1], amp[n + 1], amp[n + 2 :]))
        self.last_sine = (tm, f, x, nf, nx)

    # ------------------------------------------------------------ records
    def block(self, b: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Noise-free force and response of block b, and their unit noise: (f, x, nf, nx)."""
        nb = self.settings.block
        a = b * self.hop
        fs, xs, nfs, nxs = [], [], [], []
        c = max(0, int(np.searchsorted(self._starts, a, side="right")) - 1)
        while c < len(self._chunks) and self._starts[c] < a + nb:
            f, x = self._chunks[c]
            lo = max(0, a - self._starts[c])
            hi = min(f.size, a + nb - self._starts[c])
            rng = np.random.default_rng([self.noise_seed, c])
            nf = rng.standard_normal(f.size)
            nx = rng.standard_normal(x.shape)
            fs.append(f[lo:hi])
            xs.append(x[lo:hi])
            nfs.append(nf[lo:hi])
            nxs.append(nx[lo:hi])
            c += 1
        return np.concatenate(fs), np.concatenate(xs), np.concatenate(nfs), np.concatenate(nxs)


def add_noise(
    f: np.ndarray, x: np.ndarray, nf: np.ndarray, nx: np.ndarray, processing: Processing
) -> tuple[np.ndarray, np.ndarray]:
    """Measured signals: clean plus unit noise scaled to a fraction of each channel's peak."""
    f_peak = float(np.abs(f).max()) if f.size else 0.0
    x_peak = np.abs(x).max(axis=0) if x.size else np.zeros(x.shape[1])
    return f + processing.force_noise * f_peak * nf, x + processing.response_noise * x_peak * nx


# --------------------------------------------------------------- estimation
@dataclass
class Estimate:
    """An estimated FRF column and what went into it."""

    freqs: np.ndarray  # Hz
    H: np.ndarray  # (F, n) complex estimate of the FRF the sensors measure (receptance or accelerance)
    coherence: np.ndarray | None  # (F, n); None for a stepped sine
    force_spectrum: np.ndarray  # (F,) averaged |F|^2 (N^2), or |F|^2 per stepped-sine point
    count: int  # blocks averaged (or stepped-sine frequencies measured)
    # The latest block as measured (with noise), for display.
    t: np.ndarray
    f: np.ndarray
    x: np.ndarray  # (len(t), n) responses (displacement or acceleration)
    force_window: np.ndarray | None
    response_window: np.ndarray | None
    # Stepped sine: the sines fitted to the latest frequency, finely sampled (t, f, x).
    fit: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    response: Response = Response.DISPLACEMENT  # what x is, and so whether H is receptance or accelerance


class FrfEstimator:
    """Averages the spectra of an Acquisition's blocks as they arrive.

    Holds running sums for one Processing; make a new one when it changes.
    """

    def __init__(self, acquisition: Acquisition, processing: Processing) -> None:
        self.acq = acquisition
        self.processing = processing
        s = acquisition.settings
        self.w_f, self.w_x = block_windows(s, processing)
        nbins = s.block // 2 + 1
        n = acquisition.n
        self.freqs = np.fft.rfftfreq(s.block, 1.0 / s.fs)
        self._Sff = np.zeros(nbins)
        self._Sxf = np.zeros((nbins, n), dtype=complex)
        self._Sxx = np.zeros((nbins, n))
        self.count = 0

    def estimate(self) -> Estimate | None:
        """The FRF from every block recorded so far (up to the averages wanted); None before the first."""
        if self.acq.stepped:
            return self._sine()
        wanted = min(self.acq.blocks, self.acq.settings.averages)
        if wanted < self.count:  # fewer averages asked for: start the sums again
            self._Sff[:] = 0.0
            self._Sxf[:] = 0.0
            self._Sxx[:] = 0.0
            self.count = 0
        last = None
        for b in range(self.count, wanted):
            f, x = add_noise(*self.acq.block(b), self.processing)
            F = np.fft.rfft(self.w_f * f)
            X = np.fft.rfft(self.w_x[:, None] * x, axis=0)
            self._Sff += np.abs(F) ** 2
            self._Sxf += X * F.conj()[:, None]
            self._Sxx += np.abs(X) ** 2
            last = (f, x)
        self.count = wanted
        if wanted == 0:
            return None
        if last is None:
            last = add_noise(*self.acq.block(wanted - 1), self.processing)
        with np.errstate(divide="ignore", invalid="ignore"):
            if self.processing.estimator is Estimator.H1:
                H = self._Sxf / self._Sff[:, None]
            else:
                H = self._Sxx / self._Sxf.conj()
            coh = np.abs(self._Sxf) ** 2 / (self._Sff[:, None] * self._Sxx)
        s = self.acq.settings
        return Estimate(
            freqs=self.freqs,
            H=H,
            coherence=coh,
            force_spectrum=self._Sff / wanted,
            count=wanted,
            t=np.arange(s.block) / s.fs,
            f=last[0],
            x=last[1],
            force_window=self.w_f,
            response_window=self.w_x,
            response=s.response,
        )

    def _sine(self) -> Estimate | None:
        pts = self.acq.points
        if not pts:
            return None
        p = self.processing
        F = np.array([q.force + p.force_noise * abs(q.force) * q.force_noise for q in pts])
        X = np.array([q.response + p.response_noise * np.abs(q.response) * q.response_noise for q in pts])
        t, f, x, nf, nx = self.acq.last_sine
        f, x = add_noise(f, x, nf, nx, p)
        # The fitted sines, drawn smoothly (the samples can be as few as 2.5 per cycle).
        w = TWO_PI * pts[-1].freq_hz
        tf = np.linspace(0.0, t[-1], max(t.size, math.ceil(FIT_POINTS_PER_CYCLE * pts[-1].freq_hz * t[-1]) + 1))
        e = np.exp(1j * w * tf)
        fit = (tf, (F[-1] * e).real, (X[-1][None, :] * e[:, None]).real)
        return Estimate(
            freqs=np.array([q.freq_hz for q in pts]),
            H=X / F[:, None],
            coherence=None,
            force_spectrum=np.abs(F) ** 2,
            count=len(pts),
            t=t,
            f=f,
            x=x,
            force_window=None,
            response_window=None,
            fit=fit,
            response=self.acq.settings.response,
        )
