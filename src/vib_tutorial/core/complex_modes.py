"""Non-proportional damping and complex modes: how complex the chain's modes are, and what ignoring it costs.

Why modes go complex
--------------------
The undamped modes φ_r diagonalize M and K. Damping is *proportional* (classical) when they
diagonalize C too, ΦᵀCΦ diagonal: then each mode keeps its real shape with damping, every mass
passes through zero at the same instant, and the damped modes are the undamped ones with
poles −ζ_r ω_r ± iω_r√(1 − ζ_r²). Otherwise the off-diagonal terms c_rs couple the modes, and
the exact modes, the eigenvectors of the 2N × 2N state matrix, have complex shapes ψ: each mass
has its own phase, so the masses peak one after another and the nodes move, a wave travelling
along the chain rather than a standing one. To first order in the damping,

    ψ_r ≈ φ_r + Σ_s i ω_r c_sr / (ω_r² − ω_s²) φ_s,

so ψ's imaginary part grows with the coupling and with how close the other modes are: close
modes go complex first. A chain with a damper per spring, each in proportion to its spring
(c_i = β k_i), has C = βK, which is proportional; a damper on its own somewhere is not.

How complex
-----------
* MPC, the modal phase collinearity (Pappa): from the scatter of ψ's entries in the complex
  plane, 1 when they lie on a line (a real mode, rotated) and 0 when they spread evenly round it.
* MPD, the mean phase deviation: each entry's angle from the best-fit line, weighted by its
  size, in degrees; 0 for a real mode, up to 90.
* MAC(φ, ψ): how much of ψ the undamped shape explains.

What ignoring it costs
----------------------
The usual approximation keeps only ΦᵀCΦ's diagonal, ζ_r = φ_rᵀCφ_r / 2ω_r: in effect the chain
with C_p = MΦ diag(ΦᵀCΦ) ΦᵀM, which is proportional. `compare` sets its poles against the exact
ones, `frf_pair` its FRF against the exact one, and `free_decay` releases the chain in an
undamped mode's shape: with proportional damping it stays in that mode, otherwise the damping
passes some of the motion to the others.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg

from .modal import TWO_PI, ModalResult, _mac, modal_analysis, receptance
from .model import ChainSystem, state_space


@dataclass(frozen=True)
class Complexity:
    """How far a shape's entries spread from a line through the origin."""

    mpc: float  # modal phase collinearity: 1 real, 0 spread evenly
    mpd: float  # mean phase deviation from the best-fit line, degrees, weighted by |ψ_i|
    angle: float  # the best-fit line's angle, rad
    deviation: np.ndarray  # each entry's angle from the line, degrees, in (−90, 90]


def complexity(psi: np.ndarray) -> Complexity:
    """MPC and MPD of a complex shape; both the same for ψ times any complex number."""
    psi = np.asarray(psi, dtype=complex)
    x, y = psi.real, psi.imag
    sxx, syy, sxy = float(x @ x), float(y @ y), float(x @ y)
    total = sxx + syy
    if total <= 0:
        return Complexity(1.0, 0.0, 0.0, np.zeros(psi.size))
    mpc = ((sxx - syy) ** 2 + 4 * sxy**2) / total**2
    angle = 0.5 * np.arctan2(2 * sxy, sxx - syy)  # the major axis of the entries' scatter
    turned = psi * np.exp(-1j * angle)
    dev = np.degrees(np.arctan2(turned.imag, turned.real))
    dev = (dev + 90.0) % 180.0 - 90.0  # an entry pointing back along the line is on it too
    dev[dev == -90.0] = 90.0
    size = np.abs(psi)
    mpd = float(np.sum(size * np.abs(dev)) / np.sum(size))
    return Complexity(float(mpc), mpd, float(angle), dev)


def proportional_damping(system: ChainSystem, result: ModalResult) -> np.ndarray:
    """C_p = MΦ diag(ΦᵀCΦ) ΦᵀM: the damping the usual approximation keeps (Φ mass-normalized)."""
    M, C, _ = system.matrices()
    phi = np.column_stack([m.shape_mass_normalized for m in result.modes])
    return M @ phi @ np.diag(np.diag(phi.T @ C @ phi)) @ phi.T @ M


def made_proportional(system: ChainSystem, result: ModalResult | None = None) -> ChainSystem:
    """The chain with c_i = β k_i (so C = βK, proportional), β keeping mode 1's ζ = φᵀCφ/2ω.

    Without springs or modes to keep, the dampers are left as they are.
    """
    result = result or modal_analysis(system)
    mode = next((m for m in result.modes if m.omega_n > 1e-9), None)
    if mode is None or not np.isfinite(mode.zeta_modal):
        return system.copy()
    beta = 2 * mode.zeta_modal / mode.omega_n
    return ChainSystem(system.masses, system.stiffness, beta * system.stiffness, system.friction,
                       system.static_ratio)


@dataclass(frozen=True)
class ModeComparison:
    """One mode: the exact pole and shape against the proportional approximation's."""

    index: int  # 1-based, as the undamped modes
    fn_hz: float  # undamped natural frequency
    zeta_approx: float  # φᵀCφ / 2ω
    exact: complex | None  # the matched exact pole (Im > 0); None when the mode is overdamped
    shape: np.ndarray | None  # its ψ, largest entry 1
    mac: float  # MAC(φ, ψ)
    measures: Complexity | None

    @property
    def approx(self) -> complex:
        """The approximation's pole, −ζω + iω√(1 − ζ²) (real when ζ ≥ 1)."""
        w, z = TWO_PI * self.fn_hz, self.zeta_approx
        return complex(-z * w, w * np.sqrt(max(1.0 - z * z, 0.0)))

    @property
    def fn_exact(self) -> float:
        """|λ| / 2π, Hz (NaN if overdamped)."""
        return abs(self.exact) / TWO_PI if self.exact is not None else float("nan")

    @property
    def zeta_exact(self) -> float:
        return -self.exact.real / abs(self.exact) if self.exact is not None else float("nan")


def compare(system: ChainSystem, result: ModalResult) -> list[ModeComparison]:
    """Each undamped mode with its exact damped mode, and how complex that is."""
    out = []
    for m in result.modes:
        d = m.damped
        out.append(ModeComparison(
            m.index, m.fn_hz, m.zeta_modal, d.eigenvalue if d is not None else None,
            d.shape if d is not None else None, _mac(m.shape, d.shape) if d is not None else float("nan"),
            complexity(d.shape) if d is not None else None))
    return out


def frf_pair(system: ChainSystem, result: ModalResult, freqs_hz: np.ndarray, inp: int, out: int
             ) -> tuple[np.ndarray, np.ndarray]:
    """Receptance x_out / f_inp at each frequency: exact, and with the proportional approximation's C_p."""
    M, C, K = system.matrices()
    f = np.zeros(system.n)
    f[inp] = 1.0
    exact = receptance(M, C, K, freqs_hz, f)[:, out]
    approx = receptance(M, proportional_damping(system, result), K, freqs_hz, f)[:, out]
    return exact, approx


def frf_grid(result: ModalResult, points: int = 1500) -> np.ndarray:
    """Frequencies (Hz) for the FRFs: log-spaced over the modes, with extra points over each peak.

    None falls on a natural frequency, where an undamped mode's FRF is infinite.
    """
    fn = np.array([m.fn_hz for m in result.modes if m.fn_hz > 0])
    if fn.size == 0:
        return np.geomspace(0.01, 10.0, points)
    base = np.geomspace(0.2 * fn.min(), 2.0 * fn.max(), points)
    peaks = [f * (1 + np.linspace(-0.2, 0.2, 200)) for f in fn]  # an even count: f itself left out
    return np.unique(np.concatenate([base, *peaks]))


@dataclass(frozen=True)
class FreeDecay:
    """The chain released at rest in undamped mode `mode`'s shape: its modal coordinates over time.

    q[k, s] = φ_sᵀ M x(t_k) (mass-normalized φ_s), over mode `mode`'s value at t = 0, so the
    released mode starts at 1 and the rest at 0. `approx` is the proportional approximation's:
    mode `mode` alone, decaying at its ζ.
    """

    mode: int  # 0-based
    t: np.ndarray
    q: np.ndarray  # (len(t), N), exact
    approx: np.ndarray  # (len(t),), the released mode's, approximated

    @property
    def leak(self) -> np.ndarray:
        """Each mode's largest |q| over the run, the released one's set to 0: how much reached it."""
        out = np.max(np.abs(self.q), axis=0)
        out[self.mode] = 0.0
        return out


def free_decay(system: ChainSystem, result: ModalResult, mode: int, periods: float = 12.0, per_period: int = 40
               ) -> FreeDecay:
    """Release from x = φ_mode, at rest, for `periods` of that mode, stepped exactly by expm(AΔ)."""
    m = result.modes[mode]
    n = system.n
    M = system.matrices()[0]
    phi = np.column_stack([r.shape_mass_normalized for r in result.modes])
    w = m.omega_n
    if w <= 1e-9:
        raise ValueError("a rigid-body mode doesn't oscillate")
    dt = TWO_PI / w / per_period
    steps = int(round(periods * per_period))
    A, _ = state_space(system)
    step = scipy.linalg.expm(A * dt)
    z = np.zeros((steps + 1, 2 * n))
    z[0, :n] = phi[:, mode]
    for k in range(steps):
        z[k + 1] = step @ z[k]
    q = z[:, :n] @ M @ phi  # φ_sᵀ M x for each s
    t = np.arange(steps + 1) * dt
    zeta = m.zeta_modal
    if zeta < 1:
        wd = w * np.sqrt(1 - zeta**2)
        approx = np.exp(-zeta * w * t) * (np.cos(wd * t) + zeta * w / wd * np.sin(wd * t))
    else:  # the approximation's own mode overdamped: its two real roots, released at rest
        s1, s2 = (-zeta + np.array([1.0, -1.0]) * np.sqrt(zeta**2 - 1)) * w
        approx = (s1 * np.exp(s2 * t) - s2 * np.exp(s1 * t)) / (s1 - s2) if s1 != s2 else (
            (1 - s1 * t) * np.exp(s1 * t))
    return FreeDecay(mode, t, q, approx)


def strobe(psi: np.ndarray, count: int = 8) -> np.ndarray:
    """Re(ψ e^{iθ}) at `count` phases θ over half a cycle, (count, N): the shape's snapshots.

    A real mode's are all one shape, scaled; a complex mode's differ, and their zeros move.
    """
    theta = np.pi * np.arange(count) / count
    return (np.asarray(psi, dtype=complex)[None, :] * np.exp(1j * theta)[:, None]).real
