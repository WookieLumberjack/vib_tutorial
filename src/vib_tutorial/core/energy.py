"""Energy of the chain: kinetic, potential, and how it splits over the modes.

Kinetic energy T = 1/2 v^T M v (one term per mass) and potential energy
V = 1/2 x^T K x (one term per spring, 1/2 k_i (x_i - x_{i-1})^2).

With mass-normalized undamped modes, q = Phi^T M x diagonalizes both M and K,
so the stored energy splits exactly into one term per mode,

    T + V = sum_r 1/2 (qdot_r^2 + w_r^2 q_r^2),

whatever the damping. Damping does not enter the split; with non-proportional
damping it moves energy from one modal term to another.

The energy put in by the force and taken out by the dampers is integrated
exactly by the simulator (see ``Simulator.work`` and ``Simulator.dissipated``).
"""

from __future__ import annotations

import numpy as np

from .modal import ModalResult
from .model import ChainSystem


def kinetic_energy(system: ChainSystem, v: np.ndarray) -> np.ndarray:
    """Per-mass kinetic energy 1/2 m_i v_i^2 (J), shape (n,)."""
    return 0.5 * system.masses * v**2


def potential_energy(system: ChainSystem, x: np.ndarray) -> np.ndarray:
    """Per-spring potential energy 1/2 k_i (x_i - x_{i-1})^2 (J), shape (n,); x_0 = ground."""
    stretch = np.diff(x, prepend=0.0)
    return 0.5 * system.stiffness * stretch**2


def stored_energy(system: ChainSystem, x: np.ndarray, v: np.ndarray) -> float:
    """Total kinetic plus potential energy (J)."""
    return float(kinetic_energy(system, v).sum() + potential_energy(system, x).sum())


def modal_energies(system: ChainSystem, result: ModalResult, x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Energy of each undamped mode, 1/2 (qdot_r^2 + w_r^2 q_r^2) (J), shape (n,).

    Sums to ``stored_energy`` for any damping, since Phi^T M Phi = I and
    Phi^T K Phi = diag(w_r^2).
    """
    Phi = np.column_stack([m.shape_mass_normalized for m in result.modes])
    omega = np.array([m.omega_n for m in result.modes])
    MPhi = system.matrices()[0] @ Phi
    q, qdot = x @ MPhi, v @ MPhi
    return 0.5 * (qdot**2 + (omega * q) ** 2)
