"""Stability of the Jeffcott rotor: the onset speed, log decrements and the full spectrum.

Cross-coupled bearing stiffness and the shaft's internal damping (see rotor.py)
make the stiffness non-symmetric. A skew stiffness pushes a whirling journal
along its path, so it feeds energy into one whirl direction (forward, for
k_xy > 0 and for internal damping) and takes it from the other. Where the energy
fed in outweighs the damping, the mode's damping ratio goes negative and any
disturbance grows: the rotor is unstable. Unlike a resonance, the whirl grows at
the mode's own natural frequency, not at the speed, and no balancing helps.

The full spectrum tells the two apart on a measured orbit: the Fourier transform
of x + iy separates forward (positive frequencies) from backward whirl
(negative), so a forward subsynchronous whirl shows as a line below +1X.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .rotor import RotorSystem, whirl_modes

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

    @property
    def order(self) -> float:
        """Its whirl frequency over the spin speed (the ×Ω of a spectrum); inf at rest."""
        return 2.0 * math.pi * self.freq_hz / self.omega if self.omega > 0 else math.inf


def least_damped(system: RotorSystem, omega: float) -> tuple[float, float, float]:
    """(ζ, frequency in Hz, whirl ratio) of the least damped mode at speed Ω."""
    m = whirl_modes(system, omega)
    if not m.zeta.size:
        return math.inf, 0.0, 0.0
    i = int(np.argmin(m.zeta))
    return float(m.zeta[i]), float(m.freq_hz[i]), float(m.whirl[i])


def stability_onset(system: RotorSystem, omega_max: float, n_speeds: int = 121) -> Onset | None:
    """The lowest speed in 0..omega_max at which a mode's damping goes negative, or None.

    Swept on a grid, then bisected to about 0.01 rpm.
    """
    zeta0, f0, r0 = least_damped(system, 0.0)
    if zeta0 < UNSTABLE:
        return Onset(0.0, f0, r0)
    omegas = np.linspace(0.0, omega_max, n_speeds)
    lo = 0.0
    for w in omegas[1:]:
        if least_damped(system, w)[0] < UNSTABLE:
            hi = w
            break
        lo = w
    else:
        return None
    while hi - lo > 1e-3:
        mid = 0.5 * (lo + hi)
        if least_damped(system, mid)[0] < UNSTABLE:
            hi = mid
        else:
            lo = mid
    _, f, r = least_damped(system, hi)
    return Onset(hi, f, r)


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
    z -= z.mean()
    window = np.hanning(n)
    spectrum = np.fft.fftshift(np.fft.fft(z * window)) / window.sum()
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=even[1] - even[0]))
    return freqs, np.abs(spectrum)
