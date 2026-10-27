"""Fatigue under random vibration: rainflow counting, Basquin's S–N curve, Miner's sum, and the spectral methods.

The *Fatigue* page. A `Stress` is a Gaussian stress process whose PSD is one or two resonant peaks (the shape a
structure's response to broadband forcing has), each scaled to its own RMS. From its spectral moments come the
rates of zero crossings and peaks and the irregularity factor α₂ = m₂/√(m₀m₄) (1 for a narrow band). The damage
rate under a Basquin curve N = N_ref (S/S_ref)^(−b) (S the stress range) is worked out three ways: the narrow-band
(Rayleigh) formula, Dirlik's empirical range distribution, and Miner's sum over a synthesized record's rainflow
cycles (`rainflow`, the ASTM E1049 three-point algorithm).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

TWO_PI = 2.0 * math.pi
GRID = 4000  # PSD lines for the spectral moments


@dataclass(frozen=True)
class Stress:
    """Two resonant peaks at f₁, f₂ (Hz) with damping ζ₁, ζ₂ and RMS σ₁, σ₂ (MPa); σ₂ = 0 leaves one."""

    f1: float = 2.0
    zeta1: float = 0.02
    sigma1: float = 20.0
    f2: float = 15.0
    zeta2: float = 0.02
    sigma2: float = 0.0

    def peaks(self) -> list[tuple[float, float, float]]:
        return [p for p in ((self.f1, self.zeta1, self.sigma1), (self.f2, self.zeta2, self.sigma2)) if p[2] > 0]

    @property
    def top_hz(self) -> float:
        return 4.0 * max(f for f, _, _ in self.peaks())

    def psd(self, f: np.ndarray) -> np.ndarray:
        """One-sided PSD (MPa²/Hz): each peak an oscillator's |H|² under white noise, scaled so its area is σ²; nothing above `top_hz`."""
        f = np.asarray(f, dtype=float)
        out = np.zeros_like(f)
        grid = self.grid()
        for fn, z, s in self.peaks():
            shape = _peak(fn, z, f)
            area = np.trapezoid(_peak(fn, z, grid), grid)
            out += s * s * shape / area
        return np.where(f <= self.top_hz, out, 0.0)  # band-limited, so the moments are finite

    def grid(self) -> np.ndarray:
        """Frequencies packed round the peaks, up to `top_hz`."""
        lines = [np.linspace(0.0, self.top_hz, GRID)]
        for fn, z, _ in self.peaks():
            lines.append(fn * (1 + z * np.linspace(-12, 12, 401)))
        g = np.unique(np.concatenate(lines))
        return g[(g >= 0) & (g <= self.top_hz)]

    def moments(self) -> np.ndarray:
        """m₀, m₁, m₂, m₄: ∫ fⁿ G(f) df (with f in Hz)."""
        f = self.grid()
        G = self.psd(f)
        return np.array([np.trapezoid(f ** n * G, f) for n in (0, 1, 2, 4)])


def _peak(fn: float, zeta: float, f: np.ndarray) -> np.ndarray:
    r = f / fn
    return 1.0 / ((1 - r * r) ** 2 + (2 * zeta * r) ** 2)


@dataclass(frozen=True)
class Rates:
    """What the moments give: RMS σ, zero up-crossings ν₀ and peaks ν_p per second, α₂ = ν₀/ν_p, and Dirlik's
    x_m = (m₁/m₀)√(m₂/m₄)."""

    sigma: float
    nu0: float
    nup: float
    alpha2: float
    xm: float


def rates(stress: Stress) -> Rates:
    m0, m1, m2, m4 = stress.moments()
    nu0, nup = math.sqrt(m2 / m0), math.sqrt(m4 / m2)
    return Rates(math.sqrt(m0), nu0, nup, nu0 / nup, (m1 / m0) * math.sqrt(m2 / m4))


@dataclass(frozen=True)
class Curve:
    """Basquin's S–N curve on stress ranges: N = n_ref (S/s_ref)^(−b) cycles to failure."""

    b: float = 3.0
    s_ref: float = 80.0  # MPa range at n_ref cycles
    n_ref: float = 2e6

    @property
    def K(self) -> float:
        """N S^b."""
        return self.n_ref * self.s_ref ** self.b

    def cycles(self, S: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore"):
            return self.K / np.asarray(S, dtype=float) ** self.b


def narrow_band_damage(stress: Stress, curve: Curve) -> float:
    """Damage per second if every up-crossing is one cycle of a Rayleigh amplitude (range 2a):
    ν₀ (2√2 σ)^b Γ(1 + b/2) / K."""
    r = rates(stress)
    return r.nu0 * (2 * math.sqrt(2) * r.sigma) ** curve.b * math.gamma(1 + curve.b / 2) / curve.K


def rayleigh_range_pdf(stress: Stress, S: np.ndarray) -> np.ndarray:
    """Narrow-band ranges: S = 2a with a Rayleigh in σ."""
    s = rates(stress).sigma
    S = np.asarray(S, dtype=float)
    return S / (4 * s * s) * np.exp(-S * S / (8 * s * s))


def dirlik_parameters(stress: Stress) -> tuple[float, float, float, float, float]:
    """(D₁, D₂, D₃, Q, R) of Dirlik's range distribution."""
    r = rates(stress)
    g, xm = r.alpha2, r.xm
    D1 = 2 * (xm - g * g) / (1 + g * g)
    R = (g - xm - D1 * D1) / (1 - g - D1 + D1 * D1)
    D2 = (1 - g - D1 + D1 * D1) / (1 - R)
    D3 = 1 - D1 - D2
    Q = 1.25 * (g - D3 - D2 * R) / D1
    return D1, D2, D3, Q, R


def dirlik_range_pdf(stress: Stress, S: np.ndarray) -> np.ndarray:
    """Dirlik's density of rainflow ranges (per MPa): an exponential and two Rayleighs in Z = S/2√m₀."""
    D1, D2, D3, Q, R = dirlik_parameters(stress)
    s = rates(stress).sigma
    Z = np.asarray(S, dtype=float) / (2 * s)
    p = D1 / Q * np.exp(-Z / Q) + D2 * Z / R ** 2 * np.exp(-Z * Z / (2 * R * R)) + D3 * Z * np.exp(-Z * Z / 2)
    return p / (2 * s)


def dirlik_damage(stress: Stress, curve: Curve) -> float:
    """Damage per second: ν_p ∫ S^b p(S) dS / K, in closed form term by term."""
    D1, D2, D3, Q, R = dirlik_parameters(stress)
    r = rates(stress)
    b = curve.b
    EZb = D1 * Q ** b * math.gamma(1 + b) + (math.sqrt(2) ** b) * math.gamma(1 + b / 2) * (D2 * abs(R) ** b + D3)
    return r.nup * (2 * r.sigma) ** b * EZb / curve.K


def synthesize(stress: Stress, duration: float, seed: int = 1, per_top: float = 8.0) -> tuple[np.ndarray, np.ndarray]:
    """(t, x): a Gaussian record with the PSD, `duration` long, sampled at `per_top` × `top_hz` (random phases on
    lines 1/duration apart; it repeats after `duration`)."""
    fs = per_top * stress.top_hz
    n = int(round(duration * fs))
    n += n % 2
    df = fs / n
    f = np.arange(n // 2 + 1) * df
    amp = np.sqrt(2 * stress.psd(f) * df)
    amp[0] = 0.0
    rng = np.random.default_rng(seed)
    spec = amp * np.exp(1j * rng.uniform(0, TWO_PI, f.size)) * n / 2
    x = np.fft.irfft(spec, n)
    return np.arange(n) / fs, x


def reversals(x: np.ndarray) -> np.ndarray:
    """The turning points of x (its first and last samples kept)."""
    x = np.asarray(x, dtype=float)
    d = np.diff(x)
    keep = np.nonzero(d[:-1] * d[1:] < 0)[0] + 1
    return np.concatenate([[x[0]], x[keep], [x[-1]]])


def rainflow(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(ranges, counts): ASTM E1049's three-point rainflow count, a full cycle 1 and a half cycle 0.5."""
    ranges, counts = [], []
    stack: list[float] = []
    for p in reversals(x):
        stack.append(float(p))
        while len(stack) >= 3:
            X = abs(stack[-1] - stack[-2])
            Y = abs(stack[-2] - stack[-3])
            if X < Y:
                break
            if len(stack) == 3:  # Y holds the start: half a cycle, and the start goes
                ranges.append(Y)
                counts.append(0.5)
                stack.pop(0)
            else:
                ranges.append(Y)
                counts.append(1.0)
                last = stack.pop()
                del stack[-2:]
                stack.append(last)
    for a, b in zip(stack[:-1], stack[1:]):
        ranges.append(abs(b - a))
        counts.append(0.5)
    return np.array(ranges), np.array(counts)


def miner(ranges: np.ndarray, counts: np.ndarray, curve: Curve) -> float:
    """Σ nᵢ / Nᵢ."""
    keep = ranges > 0
    return float(np.sum(counts[keep] / curve.cycles(ranges[keep])))


@dataclass(frozen=True)
class Record:
    """A synthesized record and its rainflow count."""

    t: np.ndarray
    x: np.ndarray
    ranges: np.ndarray
    counts: np.ndarray

    @property
    def duration(self) -> float:
        return float(self.t[-1] + self.t[1])

    def damage_rate(self, curve: Curve) -> float:
        return miner(self.ranges, self.counts, curve) / self.duration


def record(stress: Stress, duration: float, seed: int = 1) -> Record:
    t, x = synthesize(stress, duration, seed)
    ranges, counts = rainflow(x)
    return Record(t, x, ranges, counts)


def life_text(rate: float) -> str:
    """A damage rate per second as a life, in the largest sensible unit."""
    if rate <= 0:
        return "no damage"
    life = 1.0 / rate
    for unit, seconds in (("years", 365.25 * 86400), ("days", 86400), ("hours", 3600), ("minutes", 60)):
        if life >= seconds:
            return f"{life / seconds:.3g} {unit}"
    return f"{life:.3g} s"
