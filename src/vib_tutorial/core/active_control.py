"""Active vibration control on the chain: velocity feedback, collocation, spillover, actuator lag, against a TMD.

The *Active control* page. A `Controller` measures one mass's velocity and pushes on another (or the same) with
u = −g ẋ_s through two first-order lags of time constant τ (a sensor filter and the actuator, a second-order roll-off:
with one lag alone the feedback turns into added stiffness at high frequency, never into negative damping). `closed_loop` is the chain's state space with the
loop closed (the two lags' states appended when τ > 0), so its poles give each mode's damping exactly;
`modal_damping_estimate` is the light-feedback first-order estimate, Δζ_r = g φ_r(a) φ_r(s) / 2ω_r, which is
positive for every mode only when the sensor and actuator are collocated. `with_tmd` adds a tuned mass damper
(Den Hartog's tuning on mode 1's modal mass at its mass) for comparison, and `skyhook` is the single-mass isolator
with a damper to the sky against one to the base.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.linalg
import scipy.optimize

from .model import ChainSystem

TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class Controller:
    sensor: int = 3  # mass index whose velocity is measured
    actuator: int = 3  # mass index pushed on
    gain: float = 20.0  # N·s/m
    lag: float = 0.0  # s, each of the two lags' time constant (0: ideal)


def undamped_modes(system: ChainSystem) -> tuple[np.ndarray, np.ndarray]:
    """(ω, Φ) mass-normalized."""
    M, _, K = system.matrices()
    w2, phi = scipy.linalg.eigh(K, M)
    return np.sqrt(np.clip(w2, 0.0, None)), phi


def closed_loop(system: ChainSystem, ctrl: Controller | None,
                tmd: tuple[int, float, float, float] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(A, B, C_x) of the chain with the loop closed: z' = A z + B f (f on each mass), x = C_x z.

    `tmd` is (mass index, m, k, c) of an absorber hung from that mass.
    """
    M, C, K = system.matrices()
    n = system.n
    if tmd is not None:
        j, mt, kt, ct = tmd
        M = np.pad(M, ((0, 1), (0, 1)))
        C = np.pad(C, ((0, 1), (0, 1)))
        K = np.pad(K, ((0, 1), (0, 1)))
        M[n, n] = mt
        for mat, v in ((K, kt), (C, ct)):
            mat[j, j] += v
            mat[n, n] += v
            mat[j, n] -= v
            mat[n, j] -= v
    m = M.shape[0]
    Mi = np.linalg.inv(M)
    lagged = ctrl is not None and ctrl.lag > 0
    size = 2 * m + (2 if lagged else 0)
    A = np.zeros((size, size))
    A[:m, m:2 * m] = np.eye(m)
    A[m:2 * m, :m] = -Mi @ K
    A[m:2 * m, m:2 * m] = -Mi @ C
    if ctrl is not None:
        s, a = ctrl.sensor, ctrl.actuator
        if lagged:  # τu̇₁ = −u₁ − g ẋ_s, τu̇₂ = u₁ − u₂; the force u₂ on mass a
            t = ctrl.lag
            A[m:2 * m, 2 * m + 1] = Mi[:, a]
            A[2 * m, m + s] = -ctrl.gain / t
            A[2 * m, 2 * m] = -1.0 / t
            A[2 * m + 1, 2 * m] = 1.0 / t
            A[2 * m + 1, 2 * m + 1] = -1.0 / t
        else:
            A[m:2 * m, m + s] -= Mi[:, a] * ctrl.gain
    B = np.zeros((size, n))
    B[m:2 * m, :] = Mi[:, :n]
    Cx = np.zeros((n, size))
    Cx[:, :n] = np.eye(n)
    return A, B, Cx


def frf(A: np.ndarray, B: np.ndarray, Cx: np.ndarray, out: int, inp: int, f: np.ndarray) -> np.ndarray:
    """Receptance x_out / f_inp at the frequencies f (Hz)."""
    I = np.eye(A.shape[0])
    return np.array([(Cx[out] @ np.linalg.solve(2j * math.pi * fk * I - A, B[:, inp])) for fk in f])


@dataclass(frozen=True)
class Pole:
    freq_hz: float
    zeta: float


def oscillating_poles(A: np.ndarray) -> list[Pole]:
    """The poles with Im > 0 (one per oscillating mode), by frequency."""
    lam = np.linalg.eigvals(A)
    up = sorted((p for p in lam if p.imag > 1e-9), key=lambda p: abs(p))
    return [Pole(abs(p) / TWO_PI, -p.real / abs(p)) for p in up]


def max_real(A: np.ndarray) -> float:
    return float(np.max(np.linalg.eigvals(A).real))


def modal_damping_estimate(system: ChainSystem, ctrl: Controller) -> np.ndarray:
    """Δζ_r = g φ_r(a) φ_r(s) / 2ω_r for each mode: light, ideal feedback, the modes left uncoupled."""
    w, phi = undamped_modes(system)
    with np.errstate(divide="ignore", invalid="ignore"):
        return ctrl.gain * phi[ctrl.actuator] * phi[ctrl.sensor] / (2 * w)


def tracked_damping(system: ChainSystem, ctrl: Controller, gains: np.ndarray) -> np.ndarray:
    """Each mode's damping ratio (its pole followed from the first gain by continuity) at each gain; NaN once the
    pole has reached the real axis (no longer oscillating). Shape (len(gains), N).

    The poles in the upper half plane, real ones included, are paired with the last step's by least total
    distance (`linear_sum_assignment`), so a mode that turns overdamped keeps its own pole on the real axis
    instead of taking its neighbour's."""
    n = system.n
    out = np.full((len(gains), n), np.nan)
    prev = None
    for i, g in enumerate(gains):
        lam = np.linalg.eigvals(closed_loop(system, Controller(ctrl.sensor, ctrl.actuator, float(g), ctrl.lag))[0])
        if prev is None:
            up = lam[lam.imag > 1e-9]
            prev = up[np.argsort(np.abs(up))][:n]
        else:
            pool = lam[lam.imag > -1e-9]
            rows, cols = scipy.optimize.linear_sum_assignment(np.abs(prev[:, None] - pool[None, :]))
            nxt = prev.copy()
            nxt[rows] = pool[cols]
            prev = nxt
        oscillating = prev.imag > 1e-9 * np.maximum(1.0, np.abs(prev))
        out[i, :len(prev)] = np.where(oscillating, -prev.real / np.abs(prev), np.nan)
    return out


def best_gain(system: ChainSystem, ctrl: Controller, mode: int = 0, top: float | None = None) -> tuple[float, float]:
    """(g, ζ): the gain that damps `mode` most, collocated or not, and the damping it reaches."""
    w, _ = undamped_modes(system)
    top = top or 20 * float(np.max(system.masses)) * float(w[-1])
    gains = np.concatenate([[0.0], np.geomspace(1e-3 * top, top, 400)])
    z = tracked_damping(system, ctrl, gains)[:, mode]
    i = int(np.nanargmax(z))
    return float(gains[i]), float(z[i])


def tmd_for(system: ChainSystem, mass: int, ratio: float) -> tuple[int, float, float, float]:
    """Den Hartog's absorber on mode 1 at `mass`: m_t = μ × the mode's modal mass there (1/φ²), tuned to
    f = 1/(1 + μ) and ζ = √(3μ / 8(1 + μ)³)."""
    w, phi = undamped_modes(system)
    modal_mass = 1.0 / phi[mass, 0] ** 2
    mt = ratio * modal_mass
    wt = w[0] / (1 + ratio)
    zt = math.sqrt(3 * ratio / (8 * (1 + ratio) ** 3))
    return mass, mt, mt * wt * wt, 2 * zt * mt * wt


def skyhook(r: np.ndarray, zeta_passive: float, zeta_sky: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transmissibility |X/Y| of a mass on a spring over a moving base, against r = ω/ω_n: a passive damper
    (to the base), a skyhook damper (to an inertial frame) and both together."""
    r = np.asarray(r, dtype=float)
    passive = np.abs((1 + 2j * zeta_passive * r) / (1 - r * r + 2j * zeta_passive * r))
    sky = np.abs(1 / (1 - r * r + 2j * zeta_sky * r))
    both = np.abs((1 + 2j * zeta_passive * r) / (1 - r * r + 2j * (zeta_passive + zeta_sky) * r))
    return passive, sky, both
