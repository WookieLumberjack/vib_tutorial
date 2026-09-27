"""Excitation of the chain, switched on and off by the user.

Either a force on one mass, or a prescribed motion x_g(t) of the ground that
element 1 is attached to (base excitation). Both follow the same waveforms:
a step, a harmonic, a rectangular pulse, or a chirp (a sine whose frequency
sweeps from f_start to f_end over the sweep time, linearly or logarithmically).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class ForceKind(Enum):
    STEP = "Step (constant)"
    HARMONIC = "Harmonic  F sin(2πft)"
    PULSE = "Rectangular pulse"
    CHIRP = "Chirp (frequency sweep)"


@dataclass
class ForceSettings:
    target: int = 0  # 0-based mass index (force only)
    kind: ForceKind = ForceKind.HARMONIC
    amplitude: float = 10.0  # N
    freq_hz: float = 1.0  # harmonic only
    pulse_duration: float = 0.05  # s, pulse only
    sweep_start_hz: float = 0.5  # chirp only
    sweep_end_hz: float = 8.0  # chirp only
    sweep_time: float = 60.0  # s, chirp only
    sweep_log: bool = False  # chirp only: equal time per octave instead of per Hz
    base: bool = False  # move the ground instead of pushing a mass
    base_amplitude: float = 0.01  # m, ground motion only

    @property
    def max_freq_hz(self) -> float:
        """Highest frequency the waveform reaches (0 for step and pulse)."""
        if self.kind is ForceKind.HARMONIC:
            return self.freq_hz
        if self.kind is ForceKind.CHIRP:
            return max(self.sweep_start_hz, self.sweep_end_hz)
        return 0.0


class ForceController:
    """Evaluates the excitation as the simulation advances.

    The harmonic phase is integrated (phase += 2*pi*f*dt) rather than computed
    as 2*pi*f*t, so the frequency can be changed mid-run without a phase jump.
    A chirp integrates its phase the same way, at the frequency of the middle
    of each step (exact for a linear sweep), and switches off at the end.
    """

    def __init__(self, settings: ForceSettings | None = None) -> None:
        self.settings = settings or ForceSettings()
        self.on = False
        self.phase = 0.0
        self.pulse_remaining = 0.0
        self.sweep_elapsed = 0.0  # s since the chirp started

    def switch_on(self) -> None:
        if self.settings.kind is ForceKind.PULSE:
            self.fire_pulse()
            return
        if not self.on:
            self.phase = 0.0
            self.sweep_elapsed = 0.0
        self.on = True

    def switch_off(self) -> None:
        self.on = False
        self.pulse_remaining = 0.0

    def fire_pulse(self) -> None:
        self.pulse_remaining = self.settings.pulse_duration

    def reset(self) -> None:
        self.switch_off()
        self.phase = 0.0

    @property
    def active(self) -> bool:
        return self.on or self.pulse_remaining > 0.0

    @property
    def amplitude(self) -> float:
        """Peak of the waveform: N for a force, m for ground motion."""
        s = self.settings
        return s.base_amplitude if s.base else s.amplitude

    def frequency(self, elapsed: float | None = None) -> float | None:
        """Frequency (Hz) of a harmonic or chirp now (or `elapsed` s into the sweep), else None."""
        s = self.settings
        if s.kind is ForceKind.HARMONIC:
            return s.freq_hz
        if s.kind is not ForceKind.CHIRP:
            return None
        frac = min((self.sweep_elapsed if elapsed is None else elapsed) / s.sweep_time, 1.0)
        f0, f1 = s.sweep_start_hz, s.sweep_end_hz
        if s.sweep_log:
            return f0 * (f1 / f0) ** frac
        return f0 + (f1 - f0) * frac

    def value(self) -> float:
        """Force (N), or ground displacement (m), at the current instant."""
        s = self.settings
        if self.pulse_remaining > 0.0:
            return self.amplitude
        if not self.on:
            return 0.0
        if s.kind is ForceKind.STEP:
            return self.amplitude
        if s.kind in (ForceKind.HARMONIC, ForceKind.CHIRP):
            return self.amplitude * math.sin(self.phase)
        return 0.0

    def advance(self, dt: float) -> None:
        kind = self.settings.kind
        if self.on and kind in (ForceKind.HARMONIC, ForceKind.CHIRP):
            f = self.frequency(self.sweep_elapsed + 0.5 * dt)
            self.phase = (self.phase + 2.0 * math.pi * f * dt) % (2.0 * math.pi)
        if self.on and kind is ForceKind.CHIRP:
            self.sweep_elapsed += dt
            if self.sweep_elapsed >= self.settings.sweep_time - 1e-12:
                self.switch_off()  # the sweep is over
        if self.pulse_remaining > 0.0:
            # Snap tiny remainders to zero so the pulse ends on a step boundary.
            self.pulse_remaining = max(0.0, self.pulse_remaining - dt)
            if self.pulse_remaining < 1e-12:
                self.pulse_remaining = 0.0
