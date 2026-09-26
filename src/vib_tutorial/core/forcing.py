"""External force applied to a single mass, switched on and off by the user."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class ForceKind(Enum):
    STEP = "Step (constant)"
    HARMONIC = "Harmonic  F sin(2πft)"
    PULSE = "Rectangular pulse"


@dataclass
class ForceSettings:
    target: int = 0  # 0-based mass index
    kind: ForceKind = ForceKind.HARMONIC
    amplitude: float = 10.0  # N
    freq_hz: float = 1.0  # harmonic only
    pulse_duration: float = 0.05  # s, pulse only


class ForceController:
    """Evaluates the force as the simulation advances.

    The harmonic phase is integrated (phase += 2*pi*f*dt) rather than computed
    as 2*pi*f*t, so the frequency can be changed mid-run without a phase jump.
    """

    def __init__(self, settings: ForceSettings | None = None) -> None:
        self.settings = settings or ForceSettings()
        self.on = False
        self.phase = 0.0
        self.pulse_remaining = 0.0

    def switch_on(self) -> None:
        if self.settings.kind is ForceKind.PULSE:
            self.fire_pulse()
            return
        if not self.on:
            self.phase = 0.0
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

    def value(self) -> float:
        """Force (N) at the current instant."""
        s = self.settings
        if self.pulse_remaining > 0.0:
            return s.amplitude
        if not self.on:
            return 0.0
        if s.kind is ForceKind.STEP:
            return s.amplitude
        if s.kind is ForceKind.HARMONIC:
            return s.amplitude * math.sin(self.phase)
        return 0.0

    def advance(self, dt: float) -> None:
        if self.on and self.settings.kind is ForceKind.HARMONIC:
            self.phase = (self.phase + 2.0 * math.pi * self.settings.freq_hz * dt) % (2.0 * math.pi)
        if self.pulse_remaining > 0.0:
            # Snap tiny remainders to zero so the pulse ends on a step boundary.
            self.pulse_remaining = max(0.0, self.pulse_remaining - dt)
            if self.pulse_remaining < 1e-12:
                self.pulse_remaining = 0.0
