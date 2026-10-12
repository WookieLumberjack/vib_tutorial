"""Active magnetic bearing: an electromagnet pair under PID control, linearized about its bias.

Each axis (x or y) of a journal sits between two electromagnets facing each
other across it, each with an air gap s0 when the journal is centred. A magnet
pulls the journal toward itself with F = k i² / s², k = μ0 N² A / 4 (N turns,
A the area of each of its two pole faces, the iron's reluctance and the poles'
angle left out). Both carry a bias current i0, and the control current i is
added to one and taken from the other (differential driving):

    f = k [(i0 + i)² / (s0 - x)² - (i0 - i)² / (s0 + x)²] ≈ k_i i + k_m x,

    k_i = 4 k i0 / s0²  (N/A),   k_m = 4 k i0² / s0³  (N/m),

for small x and i. The magnets' pull grows as the journal comes nearer, so k_m
is a negative stiffness: left alone the journal falls onto one of them. The
linear force reaches its greatest, k_i i0 = k_m s0, where one magnet's current
runs out (i = i0): the load capacity. So for the same negative stiffness, only a
wider gap carries more. Its weight is held by a steady current of
W / k_i on top of the bias; the model stays linearized about the centred
journal and the bias, so that should stay well below i0.

A sensor at the journal measures x, and a controller sets the current from it,
i = -G(s) x (against the motion), through an amplifier of bandwidth ω_a:

    G(s) = (k_p + k_I / s + k_d s / (1 + s / ω_d)) · 1 / (1 + s / ω_a),

a PID whose derivative is filtered at ω_d. The bearing then pushes the journal
back with -K_b(s) x, its dynamic stiffness

    K_b(s) = k_i G(s) - k_m.

At a frequency ω its real part is the bearing's equivalent stiffness k_b(ω)
and its imaginary part over ω the equivalent damping c_b(ω). The
proportional gain has to beat the magnets first (k_i k_p > k_m), the derivative
gives the damping, and the integral pulls the journal back to the centre
under a steady load (at ω = 0 the stiffness is infinite). Up the frequency the
derivative's filter and the amplifier lag the force behind the motion: past
some frequency c_b turns negative, and a mode there is driven, not damped.
That is how a controller's phase lag destabilizes a rotor.

Per axis the controller has three states, xc = [s, d, i]: the integral of x,
the filter's lagged x (k_d s/(1 + s/ω_d) x = k_d ω_d (x - d)), and the
amplifier's current, i' = ω_a (u - i) with u = k_p x + k_I s + k_d ω_d (x - d).
The integral's state is left out when k_I = 0, and the filter's when k_d = 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

MU0 = 4e-7 * math.pi  # H/m, the permeability of free space


@dataclass(frozen=True)
class MagneticBearing:
    """A radial active magnetic bearing: two magnet pairs (x and y), each under its own PID. SI units."""

    gap: float = 1e-3  # m, s0: each magnet's air gap with the journal centred
    bias: float = 1.0  # A, i0: the bias current in every coil
    area: float = 4e-4  # m², A: each of a magnet's two pole faces
    turns: int = 200  # N, of each magnet's coil
    kp: float = 3500.0  # A/m, proportional gain
    ki: float = 35000.0  # A/(m·s), integral gain
    kd: float = 5.0  # A·s/m, derivative gain
    filter_hz: float = 2000.0  # Hz, ω_d/2π: the derivative's filter
    amplifier_hz: float = 1000.0  # Hz, ω_a/2π: the amplifier's bandwidth

    def __post_init__(self) -> None:
        if min(self.gap, self.bias, self.area, self.turns, self.filter_hz, self.amplifier_hz) <= 0.0:
            raise ValueError("the gap, bias, pole area, turns, filter and bandwidth must be positive")
        if min(self.kp, self.ki, self.kd) < 0.0:
            raise ValueError("the gains cannot be negative")

    @property
    def force_constant(self) -> float:
        """k = μ0 N² A / 4 (N·m²/A²): each magnet pulls with k i² / s²."""
        return MU0 * self.turns**2 * self.area / 4.0

    @property
    def current_gain(self) -> float:
        """k_i = 4 k i0 / s0² (N/A): the force per ampere of control current."""
        return 4.0 * self.force_constant * self.bias / self.gap**2

    @property
    def negative_stiffness(self) -> float:
        """k_m = 4 k i0² / s0³ (N/m): the magnets' pull, growing as the journal comes nearer to one."""
        return 4.0 * self.force_constant * self.bias**2 / self.gap**3

    @property
    def capacity(self) -> float:
        """k_i i0 (N): the most the linearized pair can push, one magnet's current run out."""
        return self.current_gain * self.bias

    @property
    def touchdown(self) -> float:
        """The backup bearing's clearance (m), half the air gap: the journal lands on it before a magnet."""
        return 0.5 * self.gap

    @property
    def omega_d(self) -> float:
        return 2.0 * math.pi * self.filter_hz

    @property
    def omega_a(self) -> float:
        return 2.0 * math.pi * self.amplifier_hz

    @property
    def integral(self) -> bool:
        return self.ki > 0.0

    def gain(self, s: complex | np.ndarray) -> complex | np.ndarray:
        """G(s) (A/m), the controller and amplifier: i = -G(s) x. Infinite at s = 0 with integral action."""
        s = np.asarray(s, dtype=complex)
        with np.errstate(divide="ignore", invalid="ignore"):
            pid = self.kp + self.kd * s / (1.0 + s / self.omega_d)
            pid = pid + np.where(s == 0, np.inf if self.integral else 0.0, self.ki / np.where(s == 0, 1.0, s))
        out = pid / (1.0 + s / self.omega_a)
        return out[()] if out.ndim == 0 else out

    def dynamic_stiffness(self, s: complex | np.ndarray) -> complex | np.ndarray:
        """K_b(s) = k_i G(s) - k_m (N/m): the bearing's push back is -K_b(s) x."""
        return self.current_gain * self.gain(s) - self.negative_stiffness

    def equivalent(self, freq_hz: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(k_b, c_b) at frequencies f > 0 (Hz): the spring and damper with the same push at that frequency."""
        w = 2.0 * math.pi * np.asarray(freq_hz, dtype=float)
        kb = self.dynamic_stiffness(1j * w)
        return np.real(kb), np.imag(kb) / w

    @property
    def n_states(self) -> int:
        """The controller's states per axis: the integral (if k_I > 0), the filter (if k_d > 0), the current."""
        return 1 + self.integral + (self.kd > 0.0)

    def state_space(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(a, b, c) for one axis: xc' = a xc + b x, and the amplifier's current i = c xc, i = G(s) x.

        The current goes to the coils against x (i = -G x in the notes), so the force
        on the journal is k_m x - k_i c xc. xc is [s, d, i], the integral's and the
        filter's states left out where their gains are zero.
        """
        wd, wa = self.omega_d, self.omega_a
        names = (["s"] if self.integral else []) + (["d"] if self.kd > 0.0 else []) + ["i"]
        m = len(names)
        a, b, c = np.zeros((m, m)), np.zeros(m), np.zeros(m)
        at = {name: j for j, name in enumerate(names)}
        i = at["i"]
        # u = k_p x + k_I s + k_d ω_d (x - d); i' = ω_a (u - i).
        a[i, i] = -wa
        b[i] = wa * (self.kp + self.kd * wd * (self.kd > 0.0))
        if "s" in at:
            a[at["s"], :], b[at["s"]] = 0.0, 1.0
            a[i, at["s"]] = wa * self.ki
        if "d" in at:
            a[at["d"], at["d"]], b[at["d"]] = -wd, wd
            a[i, at["d"]] = -wa * self.kd * wd
        c[i] = 1.0
        return a, b, c

    def damped_band(self, f_max: float = 1e6) -> tuple[float, float]:
        """(f_lo, f_hi) (Hz): where c_b > 0, the bearing damps; outside it, it feeds a mode.

        Below f_lo the integral's lag (0 without it), above f_hi the filter's and
        the amplifier's. (nan, nan) if it never damps.
        """
        f = np.geomspace(1e-3, f_max, 3000)
        pos = self.equivalent(f)[1] > 0.0
        if not pos.any():
            return math.nan, math.nan
        j = int(np.argmax(pos))
        k = f.size - 1 - int(np.argmax(pos[::-1]))
        lo = 0.0 if j == 0 else self._edge(f[j - 1], f[j])
        hi = math.inf if k == f.size - 1 else self._edge(f[k + 1], f[k])
        return lo, hi

    def _edge(self, out: float, inside: float) -> float:
        """Where c_b changes sign between `out` (c_b ≤ 0) and `inside` (c_b > 0), by bisection in log f."""
        for _ in range(60):
            mid = math.sqrt(out * inside)
            if self.equivalent(mid)[1] > 0.0:
                inside = mid
            else:
                out = mid
        return math.sqrt(out * inside)
