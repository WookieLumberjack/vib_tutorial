"""Self-excited vibration: motion that feeds itself, four ways.

The *Self-excited vibration* page. Each model is linear damping made negative by something that draws on a steady
source (a power supply, the wind, a current), so the rest state loses stability as a parameter passes a critical
value (a Hopf bifurcation: a pair of poles crosses the imaginary axis), and a nonlinearity sets the size of the
motion that follows (a limit cycle).

* `VanDerPol`: ẍ + ω²x = ω(ε − x²/x_r²)ẋ, the classic: negative damping ε at small amplitude, positive at large.
* `Galloping`: a square section across a wind, quasi-steady (Parkinson and Smith's polynomial for the force
  coefficient). Den Hartog: unstable when dC_L/dα + C_D < 0; onset U_g = 4mζω/(ρDA₁). Amplitudes by averaging
  (`galloping_branches`), stable and unstable.
* `Viv`: a cylinder in a current, with Facchinetti, de Langre and Biolley's wake oscillator: lock-in, where the
  shedding follows the structure.
* `Flutter`: a typical section in plunge and pitch with quasi-steady lift; the two modes' frequencies approach as
  the speed grows and one's damping crosses zero (`flutter_speed`).

The time responses are fixed-step RK4 over many parameters at once (`rk4`): they are checked against solve_ivp.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

TWO_PI = 2.0 * math.pi


def rk4(rhs: Callable[[np.ndarray], np.ndarray], z0: np.ndarray, h: float, steps: int,
        keep: int = 1) -> np.ndarray:
    """Classical RK4 of z' = rhs(z) (autonomous; z may hold many systems side by side), every `keep`-th state."""
    z = np.array(z0, dtype=float)
    out = [z.copy()]
    for i in range(1, steps + 1):
        k1 = rhs(z)
        k2 = rhs(z + 0.5 * h * k1)
        k3 = rhs(z + 0.5 * h * k2)
        k4 = rhs(z + h * k3)
        z = z + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        if i % keep == 0:
            out.append(z.copy())
    return np.array(out)


def settled_amplitude(x: np.ndarray, tail: float = 0.25) -> np.ndarray:
    """Half the peak-to-peak over the last `tail` of each history (axis 0 is time)."""
    n = x.shape[0]
    end = x[int((1 - tail) * n):]
    return 0.5 * (end.max(axis=0) - end.min(axis=0))


def dominant_hz(x: np.ndarray, dt: float, tail: float = 0.5) -> np.ndarray:
    """The frequency of each history's largest spectral line over its last `tail` (zero-padded ×8, Hann)."""
    n = x.shape[0]
    end = x[int((1 - tail) * n):]
    end = end - end.mean(axis=0)
    w = np.hanning(end.shape[0])[:, None] if end.ndim == 2 else np.hanning(end.shape[0])
    spec = np.abs(np.fft.rfft(end * w, 8 * end.shape[0], axis=0))
    f = np.fft.rfftfreq(8 * end.shape[0], dt)
    return f[np.argmax(spec, axis=0)]


# ------------------------------------------------------------------ van der Pol
@dataclass(frozen=True)
class VanDerPol:
    """ẍ + ω²x = ω(ε − x²/x_r²)ẋ: net negative damping ε (ζ = −ε/2) at small x, positive past x_r√ε."""

    freq_hz: float = 1.0
    eps: float = 0.2
    x_ref: float = 0.01
    start: float = 0.0005

    @property
    def omega(self) -> float:
        return TWO_PI * self.freq_hz

    def rhs(self, eps: np.ndarray | float) -> Callable[[np.ndarray], np.ndarray]:
        w, xr = self.omega, self.x_ref

        def f(z: np.ndarray) -> np.ndarray:
            x, v = z[0], z[1]
            return np.array([v, -w * w * x + w * (eps - (x / xr) ** 2) * v])

        return f

    def simulate(self, periods: float, per_period: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(t, x, v) from x = `start` at rest."""
        h = 1.0 / (self.freq_hz * per_period)
        n = round(periods * per_period)
        z = rk4(self.rhs(self.eps), np.array([self.start, 0.0]), h, n)
        return np.arange(n + 1) * h, z[:, 0], z[:, 1]

    def theory_amplitude(self, eps: np.ndarray | float) -> np.ndarray:
        """Averaging, for small ε: 2x_r√ε above the Hopf point, 0 below."""
        e = np.asarray(eps, dtype=float)
        return 2 * self.x_ref * np.sqrt(np.clip(e, 0.0, None))

    def sweep(self, eps: np.ndarray, periods: float = 60.0, per_period: int = 100) -> np.ndarray:
        """The settled amplitude at each ε (all stepped together, from x = x_r√|ε| or `start`)."""
        eps = np.asarray(eps, dtype=float)
        h = 1.0 / (self.freq_hz * per_period)
        n = round(periods * per_period)
        x0 = np.maximum(self.x_ref * np.sqrt(np.abs(eps)), self.start)
        z = rk4(self.rhs(eps), np.array([x0, np.zeros_like(eps)]), h, n, keep=4)
        return settled_amplitude(z[:, 0, :])


# ------------------------------------------------------------------ galloping
# Parkinson and Smith (1964): a square section's C_Fy(α) = A₁α − A₃α³ + A₅α⁵ − A₇α⁷ (tan α ≈ α)
PARKINSON = (2.69, 168.0, 6270.0, 59900.0)


def force_coefficient(alpha: np.ndarray) -> np.ndarray:
    a1, a3, a5, a7 = PARKINSON
    a = np.asarray(alpha, dtype=float)
    return a1 * a - a3 * a ** 3 + a5 * a ** 5 - a7 * a ** 7


@dataclass(frozen=True)
class Galloping:
    """A square section of side D (m), mass m (kg per m of length) on a spring (f_n, ζ), across a wind U (m/s)."""

    side: float = 0.1
    mass: float = 5.0
    freq_hz: float = 2.0
    zeta: float = 0.005
    speed: float = 6.0
    rho: float = 1.225
    start: float = 0.001

    @property
    def omega(self) -> float:
        return TWO_PI * self.freq_hz

    @property
    def onset(self) -> float:
        """U_g = 4mζω/(ρDA₁): where the wind's negative damping ½ρUDA₁ beats the structure's 2mζω."""
        return 4 * self.mass * self.zeta * self.omega / (self.rho * self.side * PARKINSON[0])

    def rhs(self, speed: np.ndarray | float) -> Callable[[np.ndarray], np.ndarray]:
        m, w, z_, D, rho = self.mass, self.omega, self.zeta, self.side, self.rho
        U = np.asarray(speed, dtype=float)

        def f(z: np.ndarray) -> np.ndarray:
            y, v = z[0], z[1]
            F = 0.5 * rho * U * U * D * force_coefficient(-v / np.maximum(U, 1e-9))  # α = −ẏ/U (the wind seen)
            return np.array([v, -w * w * y - 2 * z_ * w * v - F / m])

        return f

    def simulate(self, periods: float, per_period: int = 50) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h = 1.0 / (self.freq_hz * per_period)
        n = round(periods * per_period)
        z = rk4(self.rhs(self.speed), np.array([self.start, 0.0]), h, n)
        return np.arange(n + 1) * h, z[:, 0], z[:, 1]

    def net_power(self, amplitude: np.ndarray, speed: float) -> np.ndarray:
        """The wind's work per cycle less the damper's, for y = a cos ωt (averaging; J per m per cycle). The force
        coefficient is a polynomial, so the cycle's averages are ⟨sin^(k+1)⟩ = k!!/(k+1)!!."""
        a = np.atleast_1d(np.asarray(amplitude, dtype=float))
        damper = math.pi * 2 * self.mass * self.zeta * self.omega * self.omega * a * a
        if speed <= 0:
            return -damper
        beta = a * self.omega / speed  # the largest angle of attack
        a1, a3, a5, a7 = PARKINSON
        avg = 0.5 * a1 * beta - 3 / 8 * a3 * beta ** 3 + 5 / 16 * a5 * beta ** 5 - 35 / 128 * a7 * beta ** 7
        wind = TWO_PI / self.omega * 0.5 * self.rho * speed * speed * self.side * a * self.omega * avg
        return wind - damper


@dataclass(frozen=True)
class Branch:
    """Steady galloping amplitudes against the wind speed: `stable` or not (NaN where there's none)."""

    speed: np.ndarray
    amplitude: np.ndarray
    stable: bool


def galloping_branches(model: Galloping, speeds: np.ndarray, top: float | None = None) -> list[Branch]:
    """Every root of `net_power` / a² in a > 0 at each speed, joined into branches by their order."""
    top = top or 0.45 * float(np.max(speeds)) / model.omega  # the polynomial holds to an angle of ~0.4 rad
    a = np.linspace(1e-6 * model.side, top, 3000)
    roots: list[list[tuple[float, bool]]] = []
    for U in speeds:
        g = model.net_power(a, float(U)) / (a * a)
        s = np.sign(g)
        idx = np.nonzero(s[:-1] * s[1:] < 0)[0]
        found = []
        for i in idx:
            r = a[i] - g[i] * (a[i + 1] - a[i]) / (g[i + 1] - g[i])
            found.append((float(r), bool(g[i] > 0)))  # gaining below and losing above: stable
        roots.append(found)
    most = max((len(r) for r in roots), default=0)
    branches = []
    for j in range(most):
        for want in (True, False):
            amp = np.array([r[j][0] if len(r) > j and r[j][1] == want else np.nan for r in roots])
            if np.isfinite(amp).any():
                branches.append(Branch(np.asarray(speeds, dtype=float), amp, want))
    return branches


# ------------------------------------------------------------------ vortex-induced vibration
@dataclass(frozen=True)
class Viv:
    """A cylinder of diameter D (m), mass m (kg/m) on a spring (f_n, ζ), in a current U (m/s) of density ρ,
    coupled to Facchinetti's wake oscillator q̈ + εΩ_f(q² − 1)q̇ + Ω_f²q = (A/D)ÿ, Ω_f = 2π St U/D. f_n is the
    natural frequency in still fluid (with the added mass), and the reduced velocity is U / f_n D."""

    diameter: float = 0.05
    mass: float = 20.0
    freq_hz: float = 1.0
    zeta: float = 0.005
    reduced: float = 5.0  # U / f_n D
    rho: float = 1000.0
    strouhal: float = 0.2
    lift: float = 0.3  # C_L0
    eps: float = 0.3
    A: float = 12.0
    gamma: float = 0.8  # fluid damping coefficient
    added: float = 1.0  # C_M

    @property
    def omega(self) -> float:
        return TWO_PI * self.freq_hz

    @property
    def mass_ratio(self) -> float:
        """m* = m / (ρπD²/4)."""
        return self.mass / (self.rho * math.pi * self.diameter ** 2 / 4)

    @property
    def added_mass(self) -> float:
        return self.added * self.rho * math.pi * self.diameter ** 2 / 4

    @property
    def lock_in_at(self) -> float:
        """The reduced velocity at which the shedding frequency St U/D meets f_n: 1/St."""
        return 1.0 / self.strouhal

    def rhs(self, reduced: np.ndarray | float) -> Callable[[np.ndarray], np.ndarray]:
        D, rho = self.diameter, self.rho
        U = np.asarray(reduced, dtype=float) * self.freq_hz * D
        Wf = TWO_PI * self.strouhal * U / D
        mt = self.mass + self.added_mass
        r = 2 * mt * self.zeta * self.omega + self.gamma * Wf * rho * D * D
        k = mt * self.omega ** 2

        def f(z: np.ndarray) -> np.ndarray:
            y, v, q, p = z
            S = 0.25 * rho * U * U * D * self.lift * q
            acc = (S - r * v - k * y) / mt
            dp = self.A / D * acc - self.eps * Wf * (q * q - 1) * p - Wf * Wf * q
            return np.array([v, acc, p, dp])

        return f

    def run(self, reduced: np.ndarray, periods: float = 150.0, per_period: int = 60,
            keep: int = 2) -> tuple[np.ndarray, np.ndarray]:
        """(t, z[time, state, case]) at each reduced velocity, from rest with the wake at q = 2 (all at once)."""
        reduced = np.atleast_1d(np.asarray(reduced, dtype=float))
        h = 1.0 / (self.freq_hz * max(per_period, 1) * max(1.0, float(np.max(reduced)) * self.strouhal))
        n = round(periods / self.freq_hz / h)
        z0 = np.array([np.zeros_like(reduced), np.zeros_like(reduced), np.full_like(reduced, 2.0),
                       np.zeros_like(reduced)])
        z = rk4(self.rhs(reduced), z0, h, n, keep=keep)
        return np.arange(z.shape[0]) * h * keep, z

    def sweep(self, reduced: np.ndarray, periods: float = 150.0) -> tuple[np.ndarray, np.ndarray]:
        """(amplitude / D, frequency / f_n) of the settled motion at each reduced velocity."""
        t, z = self.run(reduced, periods)
        dt = t[1] - t[0]
        return settled_amplitude(z[:, 0, :]) / self.diameter, dominant_hz(z[:, 0, :], dt) / self.freq_hz


# ------------------------------------------------------------------ flutter
@dataclass(frozen=True)
class Flutter:
    """A typical section: plunge h and pitch θ about an elastic axis `e` semichords behind the quarter chord,
    the centre of mass `x_cg` semichords behind the elastic axis, radius of gyration `r` semichords; quasi-steady
    lift L = ½ρU²(2b)2π(θ + ḣ/U) at the quarter chord."""

    semichord: float = 0.15
    mass: float = 10.0  # kg per m of span
    x_cg: float = 0.2
    radius: float = 0.5
    e: float = 0.3
    plunge_hz: float = 3.0
    pitch_hz: float = 6.0
    zeta: float = 0.01
    rho: float = 1.225

    def matrices(self, speed: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        b, m = self.semichord, self.mass
        S = m * self.x_cg * b
        I = m * (self.radius * b) ** 2
        M = np.array([[m, S], [S, I]])
        kh, kt = m * (TWO_PI * self.plunge_hz) ** 2, I * (TWO_PI * self.pitch_hz) ** 2
        ch, ct = 2 * self.zeta * math.sqrt(kh * m), 2 * self.zeta * math.sqrt(kt * I)
        q = 0.5 * self.rho * speed * speed
        lift = q * 2 * b * TWO_PI  # dL/dθ
        arm = self.e * b  # the lift acts this far ahead of the elastic axis
        # m ḧ + S θ̈ + k_h h = −L;  S ḧ + I θ̈ + k_θ θ = L·arm;  L = lift (θ + ḣ/U)
        K = np.array([[kh, lift], [0.0, kt - lift * arm]])
        U = max(speed, 1e-9)
        C = np.array([[ch + lift / U, 0.0], [-lift * arm / U, ct]])
        return M, C, K

    def poles(self, speed: float) -> np.ndarray:
        """The four eigenvalues of the section at this speed, the two with Im ≥ 0 first by frequency."""
        M, C, K = self.matrices(speed)
        Mi = np.linalg.inv(M)
        A = np.block([[np.zeros((2, 2)), np.eye(2)], [-Mi @ K, -Mi @ C]])
        lam = np.linalg.eigvals(A)
        return lam[np.argsort(-lam.imag)]

    def modes(self, speeds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(f, ζ), each (len(speeds), 2): the two oscillating modes by frequency (NaN once a mode stops
        oscillating), followed by continuity."""
        f = np.full((len(speeds), 2), np.nan)
        z = np.full((len(speeds), 2), np.nan)
        prev = None
        for i, U in enumerate(speeds):
            lam = self.poles(float(U))
            up = lam[lam.imag > 1e-9]
            up = up[np.argsort(up.imag)]
            if prev is not None and up.size == 2 and (
                    abs(up[0] - prev[1]) + abs(up[1] - prev[0]) < abs(up[0] - prev[0]) + abs(up[1] - prev[1])):
                up = up[::-1]
            for j, p in enumerate(up[:2]):
                f[i, j] = abs(p) / TWO_PI
                z[i, j] = -p.real / abs(p)
            if up.size == 2:
                prev = up
        return f, z

    def flutter_speed(self, top: float = 200.0) -> float:
        """The lowest speed at which a pole crosses into the right half plane (bisection), NaN if none to `top`."""
        def growing(U: float) -> bool:
            return bool(np.max(self.poles(U).real) > 0)

        U = np.linspace(0.0, top, 801)
        for lo, hi in itertools.pairwise(U):
            if growing(hi):
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    if growing(mid):
                        hi = mid
                    else:
                        lo = mid
                return hi
        return math.nan

    def divergence_speed(self) -> float:
        """Where the pitch spring is used up by the lift's moment: k_θ = q 2b 2π e b."""
        if self.e <= 0:
            return math.inf
        b = self.semichord
        I = self.mass * (self.radius * b) ** 2
        kt = I * (TWO_PI * self.pitch_hz) ** 2
        return math.sqrt(kt / (0.5 * self.rho * 2 * b * TWO_PI * self.e * b))

    def at(self, **changes) -> Flutter:
        return replace(self, **changes)
