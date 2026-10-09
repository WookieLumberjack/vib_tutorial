"""Ready-made chains for class: a plain chain, vibration absorbers and tuned mass dampers.

An absorber is one more mass on the end of the chain, joined to the last mass by
its own spring and damper, so it fits the chain topology unchanged::

    ground --[k1,c1]-- m1 ... mN --[ka,ca]-- ma

Tuned to one mode r of the chain it is attached to (the *primary* system), it
sees that mode as a single-DOF oscillator of modal mass 1/phi_N^2 at the
attachment (phi mass-normalized), which is what the mass ratio mu refers to.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .forcing import ForceSettings
from .modal import modal_analysis
from .model import ChainSystem


def den_hartog(mu: float) -> tuple[float, float]:
    """Den Hartog's optimum tuning of a damped absorber of mass ratio mu on an undamped SDOF.

    Returns (frequency ratio f_a / f_primary, damping ratio zeta_a), which make the
    two fixed points of the primary's receptance equally high and put the peaks on them.
    As in Den Hartog's own derivation, zeta_a = c_a / (2 m_a omega_primary): it is
    referenced to the primary's natural frequency, not the absorber's.
    """
    return 1.0 / (1.0 + mu), math.sqrt(3.0 * mu / (8.0 * (1.0 + mu) ** 3))


def modal_mass(system: ChainSystem, mode: int, dof: int) -> float:
    """Effective mass (kg) of undamped mode `mode` (0-based) seen at mass `dof`: 1 / phi_dof^2."""
    phi = modal_analysis(system).modes[mode].shape_mass_normalized
    return 1.0 / phi[dof] ** 2


def with_absorber(primary: ChainSystem, mass: float, freq_hz: float, zeta: float = 0.0) -> ChainSystem:
    """The primary chain with an absorber of `mass` kg on its last mass.

    The absorber on its own (last mass held) has natural frequency freq_hz and damping ratio zeta.
    """
    omega = 2.0 * math.pi * freq_hz
    return ChainSystem(
        np.append(primary.masses, mass),
        np.append(primary.stiffness, mass * omega**2),
        np.append(primary.damping, 2.0 * zeta * mass * omega),
    )


def tuned_mass_damper(primary: ChainSystem, mu: float, mode: int = 0) -> ChainSystem:
    """The primary with a Den Hartog-tuned damper on its last mass, mu times mode `mode`'s modal mass there."""
    m_eff = modal_mass(primary, mode, primary.n - 1)
    ratio, zeta = den_hartog(mu)
    fn = modal_analysis(primary).modes[mode].fn_hz
    # with_absorber takes the damping ratio on the absorber's own frequency ratio * fn.
    return with_absorber(primary, mu * m_eff, ratio * fn, zeta / ratio)


@dataclass(frozen=True)
class Preset:
    name: str
    summary: str  # one or two sentences (HTML) shown under the parameters once loaded
    system: ChainSystem
    force: ForceSettings = field(default_factory=ForceSettings)
    absorber: bool = False  # the last mass is an absorber: compare with the chain without it


def presets() -> list[Preset]:
    uniform = ChainSystem.uniform(4)

    sdof = ChainSystem([1.0], [400.0], [1.0])
    f1 = modal_analysis(sdof).modes[0].fn_hz
    absorber = with_absorber(sdof, 0.1, f1)

    tmd_mu = 0.05
    tmd_primary = ChainSystem([1.0], [400.0], [0.5])
    tmd = tuned_mass_damper(tmd_primary, tmd_mu)
    tmd_ratio, tmd_zeta = den_hartog(tmd_mu)
    f_tmd = modal_analysis(tmd_primary).modes[0].fn_hz

    building = ChainSystem.uniform(4)
    b_mu = 0.05
    b_fn = modal_analysis(building).modes[0].fn_hz
    b_meff = modal_mass(building, 0, building.n - 1)
    building_tmd = tuned_mass_damper(building, b_mu)
    b_ratio, b_zeta = den_hartog(b_mu)

    return [
        Preset(
            "Uniform chain (4 masses)",
            "The default chain: four equal masses, springs and dampers. Force at m4.",
            uniform,
            ForceSettings(target=3),
        ),
        Preset(
            "Vibration absorber (undamped)",
            f"m1 on k1 is the machine ({f1:.3g} Hz). m2 is an absorber, a tenth of its mass, "
            f"tuned to the same {f1:.3g} Hz with no damper. Driven at that frequency, m1 stands "
            "still: the absorber's spring pushes back with exactly the applied force.",
            absorber,
            ForceSettings(target=0, freq_hz=f1),
            absorber=True,
        ),
        Preset(
            "Tuned mass damper (Den Hartog)",
            f"m1 on k1 is the structure ({f_tmd:.3g} Hz). m2 is a damper of {tmd_mu:.0%} of its "
            f"mass, tuned to {tmd_ratio:.3f} f₁ with ζ = c₂/2m₂ω₁ = {tmd_zeta:.3f} (Den Hartog's optimum). "
            "The resonance splits into two low, flat peaks. Force at m1 at the old resonance.",
            tmd,
            ForceSettings(target=0, freq_hz=f_tmd),
            absorber=True,
        ),
        Preset(
            "TMD on a 4-storey building",
            f"The default chain as a building on shaking ground, with a damper on the roof (m5): "
            f"{b_mu:.0%} of mode 1's modal mass there ({b_meff:.3g} kg), tuned to "
            f"{b_ratio:.3f} × {b_fn:.3g} Hz with ζ = {b_zeta:.3f} (relative to {b_fn:.3g} Hz). The ground shakes at mode 1.",
            building_tmd,
            ForceSettings(target=3, freq_hz=b_fn, base=True, base_amplitude=0.005),
            absorber=True,
        ),
    ]


def without_absorber(system: ChainSystem) -> ChainSystem | None:
    """The chain with its last mass (an absorber) removed, or None for a single mass."""
    return system.resized(system.n - 1) if system.n > 1 else None

