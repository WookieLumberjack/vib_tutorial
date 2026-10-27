"""Transient response of one oscillator: pulses, the impulse response, convolution and the pulse spectrum.

The *Transient response* page. Each input is a `Pulse` (an impulse, a step, a ramp up to a step, a
rectangular or a half-sine pulse), applied from rest. The response is exact: an impulse is the free
motion from v = I/m, a step and a rectangle are sums of step responses, and a ramp and a half-sine run
through the first-order-hold update (exact for the ramp, whose corners fall on samples; second order for
the half-sine, on a step of a few hundredths of the period). `convolution` works out Duhamel's integral
at one instant numerically, to set beside it. `peak_ratio` gives each pulse's largest response over the
static one against its length over the period, for every length at once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .sdof import (
    Oscillator,
    free_response,
    impulse_response,
    respond,
    response,
    step_overshoot,
    step_response,
)

STEPS_PER_PERIOD = 200  # FOH steps per natural period (and per pulse, if it's shorter)
RATIO_STEPS = 60  # the pulse spectrum's steps per period of its fastest oscillator


class PulseKind(Enum):
    IMPULSE = "Impulse"
    STEP = "Step"
    RAMP = "Ramp, then hold"
    RECTANGLE = "Rectangular pulse"
    HALF_SINE = "Half-sine pulse"


@dataclass(frozen=True)
class Pulse:
    """F(t) from t = 0: `amplitude` is the force (N), or for an impulse the impulse I (N·s).

    `duration` is the ramp's rise time or the pulse's length (s); a step and an impulse ignore it.
    """

    kind: PulseKind = PulseKind.STEP
    amplitude: float = 1.0
    duration: float = 0.1

    @property
    def has_duration(self) -> bool:
        return self.kind in (PulseKind.RAMP, PulseKind.RECTANGLE, PulseKind.HALF_SINE)

    def force(self, t: np.ndarray) -> np.ndarray:
        """The force at the times t (an impulse is drawn as nothing: it has no finite value)."""
        t = np.asarray(t, dtype=float)
        a, T = self.amplitude, self.duration
        k = self.kind
        if k is PulseKind.IMPULSE:
            return np.zeros_like(t)
        if k is PulseKind.STEP:
            return np.where(t >= 0, a, 0.0)
        if k is PulseKind.RAMP:
            return a * np.clip(t / T, 0.0, 1.0)
        if k is PulseKind.RECTANGLE:
            return np.where((t >= 0) & (t < T), a, 0.0)
        return np.where((t >= 0) & (t <= T), a * np.sin(math.pi * np.clip(t, 0, T) / T), 0.0)

    def impulse(self) -> float:
        """∫F dt over the whole pulse (N·s); infinite for a step and a ramp."""
        a, T = self.amplitude, self.duration
        return {PulseKind.IMPULSE: a, PulseKind.RECTANGLE: a * T, PulseKind.HALF_SINE: 2.0 * a * T / math.pi
                }.get(self.kind, math.inf)


def pulse_response(osc: Oscillator, pulse: Pulse, t: np.ndarray) -> np.ndarray:
    """The displacement at the times t (uniform from 0) under the pulse, from rest."""
    t = np.asarray(t, dtype=float)
    a, T = pulse.amplitude, pulse.duration
    k = pulse.kind
    if k is PulseKind.IMPULSE:
        return a * impulse_response(osc, t)
    if k is PulseKind.STEP:
        return a * step_response(osc, t)
    if k is PulseKind.RECTANGLE:
        return a * (step_response(osc, t) - step_response(osc, t - T))
    # ramp and half-sine: the FOH update on a fine grid with the pulse's corners on it, then sampled at t
    h = min(osc.period, T) / STEPS_PER_PERIOD
    n_t = max(int(math.ceil(T / h)), 1)
    h = T / n_t  # T on a sample
    fine = np.arange(0.0, t[-1] + h, h) if t[-1] > 0 else np.zeros(1)
    x = response(osc, h, pulse.force(fine))
    return np.interp(t, fine, x)


def convolution(osc: Oscillator, pulse: Pulse, at: float, points: int = 2001) -> tuple[np.ndarray, np.ndarray,
                                                                                          np.ndarray, float]:
    """Duhamel's integral x(t) = ∫₀ᵗ F(τ) h(t − τ) dτ at t = `at`, by the trapezium rule on `points` τs.

    Returns (τ, F(τ), h(t − τ), x(t)). An impulse has no samples to integrate: x = I h(t).
    """
    at = max(at, 0.0)
    if pulse.has_duration and pulse.duration < at:  # two pieces, so the trapezium doesn't straddle the end
        n1 = max(int(points * pulse.duration / at), 2)
        t1 = np.linspace(0.0, pulse.duration, n1)
        t2 = np.linspace(pulse.duration, at, max(points - n1, 2))
        tau = np.concatenate([t1, t2])
        f = np.concatenate([pulse.force(np.minimum(t1, pulse.duration * (1 - 1e-12))), pulse.force(t2)])
    else:
        tau = np.linspace(0.0, at, points)
        f = pulse.force(tau)
    h = impulse_response(osc, at - tau)
    if pulse.kind is PulseKind.IMPULSE:
        return tau, f, h, pulse.amplitude * float(impulse_response(osc, np.array([at]))[0])
    y = f * h
    return tau, f, h, float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(tau)))


@dataclass(frozen=True)
class StepMetrics:
    """A step response's shape: its overshoot (fraction of the final value), when it peaks, and when it
    stays within 2 % (the classic 4/ζω_n estimate and the exact time)."""

    overshoot: float
    peak_time: float  # s; inf with no overshoot
    settling: float  # s, exact: the last time it is 2 % or more off its final value
    settling_estimate: float  # 4/ζω_n


def step_metrics(osc: Oscillator) -> StepMetrics:
    z = osc.zeta
    est = 4.0 / (z * osc.omega_n) if z > 0 else math.inf
    if z == 0:
        return StepMetrics(1.0, math.pi / osc.omega_n, math.inf, math.inf)
    span = 3.0 * max(osc.time_constant, est, osc.period) + osc.period
    t = np.linspace(0.0, span, 20001)
    x = step_response(osc, t) * osc.stiffness
    off = np.nonzero(np.abs(x - 1.0) >= 0.02)[0]
    settle = float(t[off[-1]]) if off.size else 0.0
    return StepMetrics(step_overshoot(osc), math.pi / osc.omega_d if z < 1 else math.inf, settle, est)


def impulse_spectrum(osc: Oscillator, span: float, points: int) -> tuple[np.ndarray, np.ndarray]:
    """The Fourier transform of h(t) by the FFT of its samples over `span` s: (f, H(f)).

    h dies away within the span, so the transform is the receptance (to the sampling's accuracy): h and H
    are a Fourier pair.
    """
    t = np.linspace(0.0, span, points, endpoint=False)
    dt = t[1] - t[0]
    h = impulse_response(osc, t)
    h[0] *= 0.5  # h jumps at 0 (its slope does): the trapezium's half weight
    H = np.fft.rfft(h) * dt
    f = np.fft.rfftfreq(points, dt)
    return f, H


def peak_ratio(zeta: float, kind: PulseKind, ratios: np.ndarray) -> np.ndarray:
    """max |x| k / F over all time for a pulse of length T = ratio × T_n, for each ratio.

    An oscillator of unit natural frequency under each pulse, by the FOH update through the pulse (the
    rectangle's force held through it), then the free vibration from where the pulse leaves it, in closed form,
    over the next period: its largest swing lies within it.
    """
    ratios = np.asarray(ratios, dtype=float)
    out = np.empty(ratios.size)
    osc = Oscillator.from_modal(1.0, 1.0, zeta)
    A, B = osc.matrices()
    after = np.linspace(0.0, 1.05 * osc.period, 400)
    for i, r in enumerate(ratios):
        T = r * osc.period
        n = max(int(math.ceil(RATIO_STEPS * max(r, 0.2))), 40)
        h = T / n
        t = np.arange(n + 1) * h
        # a rectangle is 1 all through [0, T] (its drop at T is the free vibration's start)
        f = np.ones_like(t) if kind is PulseKind.RECTANGLE else Pulse(kind, 1.0, T).force(t)
        z = respond(A, B, h, f)
        x = z[:, 0]
        hold = 1.0 if kind is PulseKind.RAMP else 0.0  # the force left on after the pulse (static k x)
        free, _ = free_response(osc, z[-1, 0] - hold / osc.stiffness, z[-1, 1], after)
        out[i] = max(float(np.max(np.abs(x))), float(np.max(np.abs(free + hold / osc.stiffness)))) * osc.stiffness
    return out


def peak_ratio_undamped(kind: PulseKind, ratio: float) -> float:
    """The undamped closed forms: a rectangle's 2 sin(πr) below r = ½ and 2 above, a ramp's 1 + |sin πr|/πr."""
    r = ratio
    if kind is PulseKind.RECTANGLE:
        return 2.0 * math.sin(math.pi * r) if r < 0.5 else 2.0
    if kind is PulseKind.RAMP:
        return 1.0 + abs(math.sin(math.pi * r)) / (math.pi * r)
    raise ValueError(kind)
