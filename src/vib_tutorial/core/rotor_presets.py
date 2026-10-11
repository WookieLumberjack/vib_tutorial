"""Ready-made rotors for the Jeffcott rotor page, each showing one effect at its clearest.

Each preset is a whole RotorSystem, plus the speed range its Campbell diagram
wants and a speed to run at. The numbers in the notes are checked by tests.
"""

from __future__ import annotations

from dataclasses import dataclass

from .rotor import STEEL, Bearing, RotorSystem

# The overhung rotor: a 3 kg disc of radius 150 mm (I_p = m R²/2, I_d = m R²/4), 100 mm past
# bearing B, on stiffer supports so that the shaft, not the supports, does the bending.
OVERHUNG_MASS = 3.0  # kg
OVERHUNG_RADIUS = 0.15  # m
OVERHUNG_BEARING = Bearing(kx=2e5, ky=2e5, cx=400.0, cy=400.0)
# The long shaft: a 2 kg disc of radius 80 mm on a 1 m steel shaft of 20 mm (2.47 kg), on stiff supports.
LONG_MASS = 2.0  # kg
LONG_RADIUS = 0.08  # m
LONG_BEARING = Bearing(kx=1e6, ky=1e6, cx=200.0, cy=200.0, mass=0.5)


@dataclass(frozen=True)
class RotorPreset:
    name: str
    system: RotorSystem
    top_rpm: float  # the speed range shown (Campbell diagram, stability map, Bode plot)
    target_rpm: float  # the speed it runs up to when loaded
    note: str  # one or two sentences: what to look for


def rotor_presets() -> list[RotorPreset]:
    m, r = OVERHUNG_MASS, OVERHUNG_RADIUS
    m2, r2 = LONG_MASS, LONG_RADIUS
    return [
        RotorPreset(
            "Classic Jeffcott rotor", RotorSystem(), 10_000.0, 1000.0,
            "Disc at midspan on identical bearings: the disc whirls without tilting, so the spin's "
            "gyroscopic moment never shows. One critical speed, 1584 rpm.",
        ),
        RotorPreset(
            "Off-centre disc", RotorSystem(position=0.3), 10_000.0, 1000.0,
            "At a/L = 0.3 the disc tilts as it whirls. The first critical barely splits (1731 and 1737 rpm), "
            "but the disc's tilting mode does, from 360 Hz at rest to 289 and 465 Hz at 6000 rpm.",
        ),
        RotorPreset(
            "Anisotropic supports",
            RotorSystem(bearing_a=Bearing(kx=2e4, ky=8e4), bearing_b=Bearing(kx=2e4, ky=8e4)),
            5000.0, 1500.0,
            "Supports four times stiffer vertically: the critical splits into 1365 rpm (horizontal) and "
            "1668 rpm (vertical), and between them the shaft whirls backward on an ellipse.",
        ),
        RotorPreset(
            "Overhung rotor",
            RotorSystem(position=1.2, disc_mass=m, ip=m * r * r / 2, id=m * r * r / 4,
                        bearing_a=OVERHUNG_BEARING, bearing_b=OVERHUNG_BEARING, unbalance=60e-6),
            5000.0, 1000.0,
            "A large disc 100 mm past bearing B tilts as much as it moves, so the gyroscopic moment "
            "dominates: the first mode, 870 rpm at rest, meets 1X backward at 664 rpm and forward at "
            "1284 rpm, the only one the unbalance drives.",
        ),
        RotorPreset(
            "Long shaft, with its mass",
            RotorSystem(length=1.0, diameter=0.02, position=0.3, disc_mass=m2, ip=m2 * r2 * r2 / 2,
                        id=m2 * r2 * r2 / 4, bearing_a=LONG_BEARING, bearing_b=LONG_BEARING, density=STEEL),
            10_000.0, 2500.0,
            "A 2 kg disc on a 1 m shaft that weighs more than it does (2.47 kg). With the shaft's mass the "
            "critical speeds are 1589 and 6582 rpm. The Jeffcott model, its shaft massless (dashed on the "
            "Campbell diagram), puts the first at 2146 rpm and misses the second, the shaft's own bending.",
        ),
        RotorPreset(
            "Fluid-film bearings (oil whip)",
            RotorSystem(bearing_a=Bearing(swirl=0.5), bearing_b=Bearing(swirl=0.5)),
            10_000.0, 2500.0,
            "The oil swirls round at half the shaft's speed, so the cross-coupling grows as cΩ/2. Past "
            "twice the first natural frequency, 3136 rpm, the forward whirl grows: oil whip, locked at "
            "26 Hz whatever the speed.",
        ),
    ]
