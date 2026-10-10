"""Approximate frequency response of the chain with Coulomb friction (describing function).

Driven by F cos(w t), a chain with friction does not respond in proportion to
F, so it has no FRF. The describing-function (harmonic balance) approximation
assumes every mass still moves as x_i = Re(X_i e^{iwt}) and keeps only the
first harmonic of each friction force. A mass sliding against F_f is pushed
back by a square wave in phase with its velocity, whose first harmonic is a
phasor P_i of size 4 F_f / pi along iwX_i: an amplitude-dependent damper,
c_eq = 4 F_f / (pi w |X_i|). A mass the rest of the chain can't drag past
friction sticks: X_i = 0 and |P_i| <= 4 F_f / pi.

Both cases are the condition P = proj(P + rho V), V_i = iwX_i the velocity
phasors and proj the projection of each P_i onto the disk |P_i| <= 4 F_f / pi
(any rho > 0). With the linear chain's receptance this is a small problem per
frequency, in the friction forces only:

    X = H b - H_f P,   V_f = iw X_f,

H = (K - w^2 M + i w C)^-1, b the input and H_f the columns of H at the
masses with friction. It is solved by semismooth Newton with a line search,
retried with other rho where it stalls; the few frequencies that still don't
converge are NaN.

For a single mass it gives the classic result |X| = sqrt(F^2 - (4F_f/pi)^2) / |k - w^2 m|
(with no viscous damping): unbounded at resonance unless F_f > pi F / 4.
Between F_f and 4 F_f / pi of drive it predicts a stuck mass where the real
one stick-slips, and the response is never purely harmonic, so treat the
curve as an approximation, best where every mass slides through the cycle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .modal import TWO_PI
from .model import ChainSystem

# Penalties tried in turn, as multiples of 1/|Y_ii| (Y the mobility of the friction
# masses): no single one converges everywhere, but between them nearly every frequency does.
RHO_FACTORS = (3.0, 1.0, 0.3, 10.0, 0.1)
MAX_ITERATIONS = 30
TOLERANCE = 1e-9  # on |P - proj(...)| relative to 4 F_f / pi


@dataclass
class FrictionResponse:
    """Describing-function response at each frequency: shapes (len(freqs), n)."""

    X: np.ndarray  # complex displacement amplitudes (m); NaN where not converged
    stuck: np.ndarray  # bool: the mass sticks through the whole cycle
    converged: np.ndarray  # bool, shape (len(freqs),)


def friction_frf(system: ChainSystem, freqs_hz: np.ndarray, input_dof: int, amplitude: float) -> FrictionResponse:
    """Response to a force `amplitude` cos(w t) (N) at `input_dof`; X / amplitude is the receptance."""
    b = np.zeros(system.n, dtype=complex)
    b[input_dof] = amplitude
    return friction_response(system, freqs_hz, np.broadcast_to(b, (np.size(freqs_hz), system.n)))


def friction_transmissibility(system: ChainSystem, freqs_hz: np.ndarray, amplitude: float) -> FrictionResponse:
    """Response to a ground motion `amplitude` cos(w t) (m); X / amplitude is the transmissibility.

    The ground pulls mass 1 through spring and damper 1; friction acts against the
    masses' own velocities (the floor stays put).
    """
    w = TWO_PI * np.asarray(freqs_hz, dtype=float)
    b = np.zeros((w.size, system.n), dtype=complex)
    b[:, 0] = (system.stiffness[0] + 1j * w * system.damping[0]) * amplitude
    return friction_response(system, freqs_hz, b)


def friction_response(system: ChainSystem, freqs_hz: np.ndarray, b: np.ndarray) -> FrictionResponse:
    """Describing-function response to the force phasors b, shape (len(freqs), n)."""
    M, C, K = system.matrices()
    n = system.n
    w = TWO_PI * np.asarray(freqs_hz, dtype=float)
    Z = K[None] - (w**2)[:, None, None] * M[None] + 1j * w[:, None, None] * C[None]
    # A whisker of damping keeps Z invertible exactly at an undamped resonance.
    idx = np.arange(n)
    scale = np.maximum(np.abs(np.diag(K)).max(), (w**2) * np.diag(M).max())
    Z[:, idx, idx] += 1j * 1e-12 * scale[:, None]
    fi = np.flatnonzero(system.friction > 0)
    m = fi.size
    unit = np.zeros((n, m), dtype=complex)
    unit[fi, np.arange(m)] = 1.0
    rhs = np.concatenate([np.asarray(b, dtype=complex)[:, :, None], np.broadcast_to(unit, (w.size, n, m))], axis=2)
    H = np.linalg.solve(Z, rhs)
    Xb, Hf = H[:, :, 0], H[:, :, 1:]  # X = Xb - Hf P
    if m == 0:
        return FrictionResponse(Xb, np.zeros((w.size, n), bool), np.ones(w.size, bool))

    r = 4.0 * system.friction[fi] / np.pi
    Y = 1j * w[:, None, None] * Hf[:, fi, :]  # V_f = Vb - Y P
    Vb = 1j * w[:, None] * Xb[:, fi]
    y_diag = np.maximum(np.abs(Y[:, np.arange(m), np.arange(m)]), 1e-300)

    P = np.full((w.size, m), np.nan + 0j)
    z = np.zeros((w.size, m), dtype=complex)
    converged = np.zeros(w.size, bool)
    for factor in RHO_FACTORS:
        left = np.flatnonzero(~converged)
        if not left.size:
            break
        p, zl, ok = _solve(Y[left], Vb[left], r, factor / y_diag[left])
        P[left[ok]], z[left[ok]], converged[left[ok]] = p[ok], zl[ok], True

    X = Xb - np.einsum("fij,fj->fi", Hf, np.nan_to_num(P))
    X[~converged] = np.nan
    stuck = np.zeros((w.size, n), bool)
    stuck[:, fi] = converged[:, None] & (np.abs(z) <= r)
    X[stuck] = 0.0  # its velocity is zero at the solution; clear the round-off
    return FrictionResponse(X, stuck, converged)


def _solve(Y: np.ndarray, Vb: np.ndarray, r: np.ndarray, rho: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Semismooth Newton on G(P) = P - proj(P + rho (Vb - Y P)) = 0 at each frequency.

    Returns (P, z, converged), z = P + rho V at the solution (|z_i| <= r_i: stuck).
    Works in real coordinates [Re P, Im P]; only unconverged frequencies are iterated.
    """
    nf, m = Vb.shape
    Yr = np.block([[Y.real, -Y.imag], [Y.imag, Y.real]])
    rho2 = np.concatenate([rho, rho], axis=1)
    eye = np.eye(2 * m)
    k = np.arange(m)

    def residual(P, Y, Vb, rho):
        z = P + rho * (Vb - np.einsum("fij,fj->fi", Y, P))
        az = np.abs(z)
        return P - np.where(az > r, z * r / np.maximum(az, 1e-300), z), z

    P = np.zeros((nf, m), dtype=complex)
    G, z = residual(P, Y, Vb, rho)
    done = np.zeros(nf, bool)
    for _ in range(MAX_ITERATIONS):
        done |= np.max(np.abs(G) / r, axis=1) < TOLERANCE
        a = np.flatnonzero(~done)
        if not a.size:
            break
        Pa, Ga, za = P[a], G[a], z[a]
        # Derivative of the projection: identity inside the disk, (r/|z|)(I - zhat zhat^T) outside.
        az = np.abs(za)
        out = az > r
        s = np.where(out, r / np.maximum(az, 1e-300), 1.0)
        zh = np.where(out, za / np.maximum(az, 1e-300), 0.0)
        D = np.zeros((a.size, 2 * m, 2 * m))
        D[:, k, k] = s * (1 - zh.real**2)
        D[:, k, m + k] = D[:, m + k, k] = -s * zh.real * zh.imag
        D[:, m + k, m + k] = s * (1 - zh.imag**2)
        J = eye - D @ (eye - rho2[a][:, :, None] * Yr[a])
        step_r = -np.linalg.solve(J, np.concatenate([Ga.real, Ga.imag], axis=1)[:, :, None])[:, :, 0]
        dP = step_r[:, :m] + 1j * step_r[:, m:]
        # Backtrack until the residual falls.
        merit = np.sum(np.abs(Ga) ** 2, axis=1)
        t = np.ones(a.size)
        for _ in range(12):
            Pt = Pa + t[:, None] * dP
            Gt, zt = residual(Pt, Y[a], Vb[a], rho[a])
            worse = np.sum(np.abs(Gt) ** 2, axis=1) > (1 - 1e-4 * t) * merit
            if not worse.any():
                break
            t = np.where(worse, t / 2, t)
        P[a], G[a], z[a] = Pt, Gt, zt
    done |= np.max(np.abs(G) / r, axis=1) < TOLERANCE
    return P, z, done
