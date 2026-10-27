"""Vibration isolation: a machine with a rotating unbalance on mounts, and what reaches the floor.

The *Vibration isolation* page. The mounts are sized by their static deflection δ under the machine's
weight (k = Mg/δ, so f_n = √(g/δ)/2π whatever M is) and a damping ratio, as one of three kinds:

- `VISCOUS`: a spring and a damper side by side;
- `RELAXATION`: the damper in series with a spring N k (an elastically supported damper), so the damper can't
  stiffen the mount at high frequency;
- `TWO_STAGE`: the machine on mounts on an intermediate mass (a fraction of the machine's), itself on the same
  mounts again.

The transmissibility (the force on the floor over the force the machine makes, which is also the motion that
reaches the machine over the floor's) is worked out exactly from each kind's dynamic stiffness. `start_up`
runs the machine up to speed through the mount's resonance by the exact first-order-hold update, the
unbalance's force including its tangential part as the speed changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .sdof import TWO_PI, respond

G = 9.80665  # m/s²


class Mount(Enum):
    VISCOUS = "Spring + damper"
    RELAXATION = "Relaxation"
    TWO_STAGE = "Two stages"


@dataclass(frozen=True)
class Isolator:
    """A machine of mass M (kg) with an unbalance m_e e (kg·m), running at `rpm`, on mounts."""

    mass: float = 100.0
    rpm: float = 1500.0
    unbalance: float = 0.01  # kg·m
    deflection: float = 0.005  # m, the mounts' static deflection under the machine (each stage's, for two)
    zeta: float = 0.05
    mount: Mount = Mount.VISCOUS
    stiffness_ratio: float = 3.0  # N: the relaxation damper's spring over the mount's
    stage_mass: float = 0.5  # the intermediate mass over the machine's
    block: float = 0.0  # kg of inertia block bolted to the machine (the mounts keep their deflection)

    def __post_init__(self) -> None:
        if not (self.mass > 0 and self.deflection > 0 and self.zeta >= 0 and self.rpm >= 0 and self.block >= 0):
            raise ValueError("the mass and the deflection must be positive, and the rest not negative")
        if self.mount is Mount.RELAXATION and not self.stiffness_ratio > 0:
            raise ValueError("the relaxation spring must be positive")
        if self.mount is Mount.TWO_STAGE and not self.stage_mass > 0:
            raise ValueError("the intermediate mass must be positive")

    @property
    def total(self) -> float:
        """The mass on the mounts: the machine and its inertia block."""
        return self.mass + self.block

    @property
    def stiffness(self) -> float:
        """k = Mg/δ: the mounts under the machine (and block)."""
        return self.total * G / self.deflection

    @property
    def damping(self) -> float:
        return 2.0 * self.zeta * math.sqrt(self.stiffness * self.total)

    @property
    def natural_hz(self) -> float:
        """√(k/M)/2π = √(g/δ)/2π: the single-stage mount's natural frequency."""
        return math.sqrt(G / self.deflection) / TWO_PI

    @property
    def speed_hz(self) -> float:
        return self.rpm / 60.0

    def force_at(self, freq_hz: np.ndarray | float) -> np.ndarray:
        """The unbalance's force amplitude m_e e Ω² at a steady speed."""
        return self.unbalance * (TWO_PI * np.asarray(freq_hz, dtype=float)) ** 2

    def _two_stage(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(M, C, K) of [machine, intermediate]: the lower mounts carry both, at the same deflection."""
        m1, m2 = self.total, self.stage_mass * self.total
        k1, c1 = self.stiffness, self.damping
        k2 = (m1 + m2) * G / self.deflection
        c2 = 2.0 * self.zeta * math.sqrt(k2 * (m1 + m2))
        M = np.diag([m1, m2])
        K = np.array([[k1, -k1], [-k1, k1 + k2]])
        C = np.array([[c1, -c1], [-c1, c1 + c2]])
        return M, C, K

    def mount_stiffness(self, omega: np.ndarray) -> np.ndarray:
        """A single stage's dynamic stiffness K*(ω): the force on the floor per metre of the machine's motion."""
        w = np.asarray(omega, dtype=float)
        k, c = self.stiffness, self.damping
        if self.mount is Mount.RELAXATION:
            k1 = self.stiffness_ratio * k
            return k + k1 * (1j * w * c) / (k1 + 1j * w * c)
        return k + 1j * w * c

    def response(self, freqs_hz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(machine's receptance X/F, transmissibility F_floor/F) under a harmonic force F on the machine."""
        w = TWO_PI * np.asarray(freqs_hz, dtype=float)
        if self.mount is Mount.TWO_STAGE:
            M, C, K = self._two_stage()
            Z = K[None] - (w * w)[:, None, None] * M[None] + 1j * w[:, None, None] * C[None]
            H = np.linalg.solve(Z, np.broadcast_to(np.array([[1.0], [0.0]], dtype=complex), (w.size, 2, 1)))[:, :, 0]
            lower = K[1, 1] - K[0, 0] + 1j * w * (C[1, 1] - C[0, 0])  # k2 + iωc2
            return H[:, 0], lower * H[:, 1]
        Ks = self.mount_stiffness(w)
        H = 1.0 / (Ks - w * w * self.total)
        return H, Ks * H

    def transmissibility(self, freqs_hz: np.ndarray) -> np.ndarray:
        return np.abs(self.response(freqs_hz)[1])

    def at_speed(self) -> tuple[float, float, float]:
        """(the machine's amplitude, the force on the floor, the transmissibility) at its running speed."""
        f = np.array([self.speed_hz])
        H, T = self.response(f)
        F = float(self.force_at(f)[0])
        return abs(complex(H[0])) * F, abs(complex(T[0])) * F, abs(complex(T[0]))

    def state_space(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(A, B, D) with z' = A z + B f and the force on the floor D z: z = [x, v] (viscous), [x, v, y] with
        y the relaxation spring's end on the damper, or [x1, x2, v1, v2] for two stages."""
        k, c, m = self.stiffness, self.damping, self.total
        if self.mount is Mount.VISCOUS:
            A = np.array([[0.0, 1.0], [-k / m, -c / m]])
            return A, np.array([[0.0], [1.0 / m]]), np.array([k, c])
        if self.mount is Mount.RELAXATION:
            k1 = self.stiffness_ratio * k
            if c == 0:  # no damper: the spring k1 hangs free
                A = np.array([[0.0, 1.0, 0.0], [-k / m, 0.0, 0.0], [0.0, 0.0, 0.0]])
                return A, np.array([[0.0], [1.0 / m], [0.0]]), np.array([k, 0.0, 0.0])
            # machine — k1 — y — c — floor: k1 (x − y) = c y'
            A = np.array([[0.0, 1.0, 0.0], [-(k + k1) / m, 0.0, k1 / m], [k1 / c, 0.0, -k1 / c]])
            return A, np.array([[0.0], [1.0 / m], [0.0]]), np.array([k + k1, 0.0, -k1])
        M, C, K = self._two_stage()
        Minv = np.linalg.inv(M)
        A = np.block([[np.zeros((2, 2)), np.eye(2)], [-Minv @ K, -Minv @ C]])
        B = np.concatenate([np.zeros(2), Minv[:, 0]])[:, None]
        k2, c2 = K[1, 1] - K[0, 0], C[1, 1] - C[0, 0]
        return A, B, np.array([0.0, k2, 0.0, c2])


@dataclass(frozen=True)
class StartUp:
    """A run up from rest: time, speed (Hz), the unbalance's force, the machine's displacement and the floor's
    force."""

    t: np.ndarray
    speed_hz: np.ndarray
    force: np.ndarray
    x: np.ndarray
    floor: np.ndarray


def start_up(iso: Isolator, ramp: float, hold: float, per_period: int = 40) -> StartUp:
    """Run the machine up from rest to its speed in `ramp` s at a constant acceleration, then hold for `hold` s.

    The unbalance turns through φ(t) = ∫Ω dt; its force along the mount is m_e e (Ω² cos φ + Ω' sin φ). The
    step resolves the running speed and the mount's modes `per_period` times a period.
    """
    w_end = TWO_PI * iso.speed_hz
    A, B, D = iso.state_space()
    fastest = max(iso.speed_hz, float(np.max(np.abs(np.linalg.eigvals(A)))) / TWO_PI, 1e-3)
    h = 1.0 / (per_period * fastest)
    n = int(math.ceil((ramp + hold) / h)) + 1
    t = np.arange(n) * h
    alpha = w_end / ramp if ramp > 0 else 0.0
    w = np.where(t < ramp, alpha * t, w_end)
    phi = np.where(t < ramp, 0.5 * alpha * t * t, 0.5 * alpha * ramp * ramp + w_end * (t - ramp))
    dw = np.where(t < ramp, alpha, 0.0)
    f = iso.unbalance * (w * w * np.cos(phi) + dw * np.sin(phi))
    z = respond(A, B, h, f)
    return StartUp(t, w / TWO_PI, f, z[:, 0], z @ D)


def deflection_for(speed_hz: float, isolation: float) -> float:
    """The static deflection (m) undamped mounts need for a fraction `isolation` (0.9 = 90 %) at a speed.

    T = 1/(r² − 1) = 1 − isolation, so r² = 1 + 1/T, f_n = f/r and δ = g/ω_n².
    """
    T = 1.0 - isolation
    if not 0 < T < 1:
        raise ValueError("the isolation must be between 0 and 1")
    fn = speed_hz / math.sqrt(1.0 + 1.0 / T)
    return G / (TWO_PI * fn) ** 2
