"""Waterfall and cascade plots of the Jeffcott rotor: full spectra taken block by block as it runs.

A spectrum analyser on a machine takes the spectrum of a block of signal (here
the full spectrum of the orbit x + iy at each station, see rotor_stability),
then the next block, overlapping it, and so on. Stacked against time they make
a waterfall plot; against the speed during a run-up or coast-down, a cascade.
The unbalance shows as a line at +1X that moves with the speed; whatever sets
the rotor ringing (a tap, the passage through a critical, an instability) shows
at the free whirl frequencies, forward at +f and backward at −f. A cascade with
taps all through a run-up is a measured Campbell diagram.

The block length sets the trade-off every analyser has: the frequency
resolution is 1/T, but over a ramp the speed changes by αT while a block is
taken, smearing the 1X line.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .rotor import RotorSystem, whirl_modes
from .rotor_stability import whirl_lines, whirl_spectrum

RATE = 4000.0  # Hz, the default resampling rate: at least the input's, so nothing aliases
OVERLAP = 0.5  # each block starts this fraction of a block before the last one ends
MAX_BLOCKS = 400  # the oldest blocks are dropped past this
DAMPED = 0.3  # modes damped more than this are not listed as free whirl frequencies


@dataclass
class SpectrumBlock:
    """The full spectra at every station over one block of time."""

    t: float  # s, the block's middle
    omega: float  # rad/s, the mean speed over it
    rise: float  # rad/s, the speed at its end less that at its start
    amps: np.ndarray  # (stations, len(freqs)) amplitudes (m), float32
    natural: np.ndarray  # the free whirl frequencies at the mean speed (Hz): +f forward, −f backward


class Waterfall:
    """Full spectra of the orbits at a few stations, one per block of `block` s, overlapping.

    feed() takes the samples in order (any spacing); each time a block's worth has
    come in, they are resampled evenly at `rate` and transformed with a Hann
    window. Only |f| ≤ f_max is kept. Given the rotor, each block also lists its
    free whirl frequencies at the block's mean speed, for the plot to mark.
    """

    def __init__(self, block: float, f_max: float, system: RotorSystem | None = None,
                 rate: float = RATE, overlap: float = OVERLAP, max_blocks: int = MAX_BLOCKS) -> None:
        if block <= 0.0 or not 0.0 <= overlap < 1.0:
            raise ValueError("the block must be positive and the overlap in [0, 1)")
        self.block = block
        self.hop = block * (1.0 - overlap)
        self.system = system
        self.max_blocks = max_blocks
        self.points = 2 ** max(6, math.ceil(math.log2(block * rate)))
        freqs = np.fft.fftshift(np.fft.fftfreq(self.points, d=block / self.points))
        self._keep = np.abs(freqs) <= f_max
        self.freqs = freqs[self._keep]
        self.blocks: list[SpectrumBlock] = []
        self._t = np.empty(0)
        self._z = np.empty((0, 0), dtype=complex)
        self._w = np.empty(0)
        self._next: float | None = None  # start of the next block

    def clear(self) -> None:
        self.blocks = []
        self._t, self._z, self._w = np.empty(0), np.empty((0, 0), dtype=complex), np.empty(0)
        self._next = None

    def feed(self, t: np.ndarray, z: np.ndarray, omega: np.ndarray) -> int:
        """Add samples: times t (k,), whirls z = x + iy (k, stations) or (k,), speeds Ω (k,) rad/s.

        Returns how many blocks were completed.
        """
        if not t.size:
            return 0
        z = np.asarray(z).reshape(t.size, -1)
        if self._next is None:
            self._next = float(t[0])
            self._z = np.empty((0, z.shape[1]), dtype=complex)
        self._t = np.concatenate([self._t, t])
        self._z = np.concatenate([self._z, z])
        self._w = np.concatenate([self._w, omega])
        done = 0
        while self._t[-1] >= self._next + self.block:
            self.blocks.append(self._transform(self._next))
            self._next += self.hop
            done += 1
        if len(self.blocks) > self.max_blocks:
            del self.blocks[: len(self.blocks) - self.max_blocks]
        keep = max(int(np.searchsorted(self._t, self._next)) - 1, 0)  # the last sample before the next block
        self._t, self._z, self._w = self._t[keep:], self._z[keep:], self._w[keep:]
        return done

    def _transform(self, start: float) -> SpectrumBlock:
        n = self.points
        dt = self.block / n
        even = start + dt * np.arange(n)
        t = self._t
        z = np.column_stack([np.interp(even, t, c.real) + 1j * np.interp(even, t, c.imag) for c in self._z.T])
        _, amps = whirl_spectrum(z, dt)
        w = np.interp(even, t, self._w)
        end = float(np.interp(start + self.block, t, self._w))
        omega = float(w.mean())
        if self.system is None:
            natural = np.empty(0)
        elif self.blocks and self.blocks[-1].omega == omega:  # running steady: the same modes again
            natural = self.blocks[-1].natural
        else:
            natural = whirl_lines(whirl_modes(self.system, omega), DAMPED)
        return SpectrumBlock(t=start + 0.5 * self.block, omega=omega, rise=end - float(w[0]),
                             amps=amps[self._keep].T.astype(np.float32), natural=natural)

    def latest_ramp(self, up: bool, threshold: float = 0.05) -> list[SpectrumBlock]:
        """The blocks of the latest run-up (or coast-down): the last unbroken run of blocks over
        which the speed rose (fell) by more than `threshold` rad/s, in time order."""
        def ramping(b: SpectrumBlock) -> bool:
            return b.rise > threshold if up else b.rise < -threshold

        end = len(self.blocks)
        while end and not ramping(self.blocks[end - 1]):
            end -= 1
        start = end
        while start and ramping(self.blocks[start - 1]):
            start -= 1
        return self.blocks[start:end]
