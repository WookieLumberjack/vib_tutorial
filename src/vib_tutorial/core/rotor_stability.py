"""Stability of the Jeffcott rotor: the onset speed, log decrements and the full spectrum.

Cross-coupled bearing stiffness and the shaft's internal damping (see rotor.py)
make the stiffness non-symmetric. A skew stiffness pushes a whirling journal
along its path, so it feeds energy into one whirl direction (forward, for
k_xy > 0 and for internal damping) and takes it from the other. Where the energy
fed in outweighs the damping, the mode's damping ratio goes negative and any
disturbance grows: the rotor is unstable. Unlike a resonance, the whirl grows at
the mode's own natural frequency, not at the speed, and no balancing helps.

An asymmetric shaft (stiffer about one axis) is different again: between its two
critical speeds, turning with the shaft, the centrifugal force outweighs the
soft axis's stiffness but not the stiff one's, and the whirl grows locked to
1X, unless the damping holds it. Above the band the rotor is stable again: a
parametric instability, mapped against the asymmetry and the speed by
AsymmetryChart.

The full spectrum tells the two apart on a measured orbit: the Fourier transform
of x + iy separates forward (positive frequencies) from backward whirl
(negative), so a forward subsynchronous whirl shows as a line below +1X.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from .rotor import RotorSystem, WhirlModes, campbell, rotating_state_matrix, whirl_modes

UNSTABLE = -1e-9  # a damping ratio below this is growing, not round-off


def log_decrement(zeta: np.ndarray) -> np.ndarray:
    """δ = 2πζ/√(1 − ζ²): the log of the ratio of successive peaks of a decaying mode."""
    zeta = np.asarray(zeta, dtype=float)
    return 2.0 * math.pi * zeta / np.sqrt(np.maximum(1.0 - zeta**2, 1e-12))


@dataclass(frozen=True)
class Onset:
    """Where the rotor first goes unstable as it speeds up."""

    omega: float  # spin speed (rad/s); 0 if it is unstable even at rest
    freq_hz: float  # the whirl frequency of the mode that goes unstable
    whirl: float  # its whirl ratio, > 0 forward
    end: float = math.inf  # the speed it is stable again (rad/s), if it is below the top speed

    @property
    def order(self) -> float:
        """Its whirl frequency over the spin speed (the ×Ω of a spectrum); inf at rest."""
        return 2.0 * math.pi * self.freq_hz / self.omega if self.omega > 0 else math.inf


def least_damped(system: RotorSystem, omega: float) -> tuple[float, float, float]:
    """(ζ, frequency in Hz, whirl ratio) of the least damped mode at speed Ω.

    A root that grows without oscillating (a magnetic bearing too weak for its
    magnets) is ζ = -1 at 0 Hz.
    """
    m = whirl_modes(system, omega)
    if m.drift > 0.0:
        return -1.0, 0.0, 0.0
    if not m.zeta.size:
        return math.inf, 0.0, 0.0
    i = int(np.argmin(m.zeta))
    return float(m.zeta[i]), float(m.freq_hz[i]), float(m.whirl[i])


def stability_onset(system: RotorSystem, omega_max: float, n_speeds: int = 121) -> Onset | None:
    """The lowest speed in 0..omega_max at which a mode's damping goes negative, or None.

    Swept on a grid, then bisected to about 0.01 rpm; so is the end of the band,
    where it is stable again (an asymmetric shaft's), if that is below omega_max.
    """
    def unstable(w: float) -> bool:
        return least_damped(system, w)[0] < UNSTABLE

    omegas = np.linspace(0.0, omega_max, n_speeds)
    zeta0, f0, r0 = least_damped(system, 0.0)
    if zeta0 < UNSTABLE:
        start, f, r = 0.0, f0, r0
    else:
        hit = next((i for i in range(1, n_speeds) if unstable(omegas[i])), None)
        if hit is None:
            return None
        start = _bisect(unstable, omegas[hit - 1], omegas[hit])
        _, f, r = least_damped(system, start)
    after = np.flatnonzero(omegas > start)
    back = next((i for i in after if not unstable(omegas[i])), None)
    end = math.inf if back is None else _bisect(lambda w: not unstable(w), max(omegas[back - 1], start), omegas[back])
    return Onset(start, f, r, end)


def _bisect(changed, lo: float, hi: float) -> float:
    """The speed (to 1e-3 rad/s) in [lo, hi] where changed(·) turns True; it is False at lo, True at hi."""
    while hi - lo > 1e-3:
        mid = 0.5 * (lo + hi)
        if changed(mid):
            hi = mid
        else:
            lo = mid
    return hi


def growth_rate(system: RotorSystem, omega: float) -> float:
    """The fastest growth (or slowest decay) rate max Re λ of any mode at speed Ω (1/s): > 0 unstable.

    For an asymmetric shaft on isotropic supports, the rotating frame's eigenvalues
    (the growth rates do not change with the frame), so no modes need sorting.
    """
    if system.periodic and system.rotating_frame:
        return float(np.linalg.eigvals(rotating_state_matrix(system, omega)).real.max())
    m = whirl_modes(system, omega)
    lam = -m.zeta * 2.0 * math.pi * m.freq_hz / np.sqrt(np.maximum(1.0 - m.zeta**2, 1e-12))
    return max(float(lam.max()) if lam.size else -math.inf, m.drift)


class AsymmetryChart:
    """Stability against the shaft's asymmetry Δ and the speed: an Ince–Strutt chart, filled in a bit at a time.

    growth[i, j] is growth_rate at asymmetries[i] and omegas[j], NaN until worked
    out; soft[i] and stiff[i] are the lowest forward critical speeds of a round
    shaft as stiff as each axis (NaN if above the range), between which an
    undamped rotor is unstable. work(budget) fills cells until `budget` seconds
    have passed, so a page can call it each frame; done tells when all are in.
    """

    def __init__(self, system: RotorSystem, omega_max: float, delta_max: float = 0.5,
                 n_deltas: int = 41, n_speeds: int = 121) -> None:
        self.system = system
        self.asymmetries = np.linspace(0.0, delta_max, n_deltas)
        self.omegas = np.linspace(0.0, omega_max, n_speeds)
        self.growth = np.full((n_deltas, n_speeds), np.nan)
        self.soft = np.full(n_deltas, np.nan)
        self.stiff = np.full(n_deltas, np.nan)
        self._next = 0  # the next cell, row by row
        self._row: RotorSystem | None = None

    @property
    def done(self) -> bool:
        return self._next >= self.growth.size

    def work(self, budget: float = math.inf) -> bool:
        """Fill cells for about `budget` seconds (at least one); True if any came in."""
        start = time.perf_counter()
        did = False
        n = self.omegas.size
        while not self.done and (not did or time.perf_counter() - start < budget):
            i, j = divmod(self._next, n)
            if j == 0:
                self._row = self.system.with_(asymmetry=float(self.asymmetries[i]))
                self.soft[i], self.stiff[i] = (_first_forward_critical(self._row.axis_system(stiff), self.omegas[-1])
                                               for stiff in (False, True))
            self.growth[i, j] = growth_rate(self._row, float(self.omegas[j]))
            self._next += 1
            did = True
        return did

    def run(self) -> AsymmetryChart:
        self.work()
        return self


def _first_forward_critical(system: RotorSystem, omega_max: float) -> float:
    """The lowest critical speed with a forward whirl up to omega_max (rad/s), or NaN."""
    found = [w for w, r, _ in campbell(system, omega_max, n_speeds=61, n_modes=2).criticals if r > 0.1]
    return found[0] if found else math.nan


def full_spectrum(t: np.ndarray, xy: np.ndarray, points: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Full spectrum of an orbit: (frequency in Hz, amplitude in m), negative frequencies backward.

    xy (k, 2) is sampled at times t (not necessarily evenly; it is resampled to
    `points` even samples). A Hann window, scaled so that a circular whirl of
    radius R at frequency f shows as a line of height R at +f (forward) or -f
    (backward); a straight-line vibration of amplitude A shows as A/2 at both.
    """
    t = np.asarray(t, dtype=float)
    if t.size < 8 or t[-1] <= t[0]:
        return np.empty(0), np.empty(0)
    n = min(points, t.size)
    even = np.linspace(t[0], t[-1], n)
    z = np.interp(even, t, xy[:, 0]) + 1j * np.interp(even, t, xy[:, 1])
    return whirl_spectrum(z, even[1] - even[0])


def whirl_spectrum(z: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Full spectrum of whirls z = x + iy (n, ...) sampled evenly dt apart, along the first axis.

    (frequency in Hz (n,), amplitude in m (n, ...)), scaled as full_spectrum's. The
    mean is taken out first, so a static offset leaves no line at 0 Hz.
    """
    n = z.shape[0]
    window = np.hanning(n).reshape((n,) + (1,) * (z.ndim - 1))
    spectrum = np.fft.fft((z - z.mean(axis=0)) * window, axis=0) / window.sum()
    return np.fft.fftshift(np.fft.fftfreq(n, d=dt)), np.abs(np.fft.fftshift(spectrum, axes=0))


def whirl_lines(modes: WhirlModes, damped_limit: float) -> np.ndarray:
    """Where the lightly damped modes show on a full spectrum (Hz): +f forward, −f backward.

    A straight-line whirl (whirl ratio within ±0.1) shows on both sides.
    """
    lines = []
    for f, z, w in zip(modes.freq_hz, modes.zeta, modes.whirl):
        if z <= damped_limit:
            if w >= -0.1:
                lines.append(f)
            if w <= 0.1:
                lines.append(-f)
    return np.array(lines)
