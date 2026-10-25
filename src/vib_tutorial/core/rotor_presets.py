"""Ready-made rotors for the Jeffcott rotor page, each showing one effect at its clearest.

Each preset is a whole RotorSystem, plus the speed range its Campbell diagram
wants and a speed to run at. The numbers in the notes are checked by tests.
"""

from __future__ import annotations

from dataclasses import dataclass

from .fluid_bearing import JournalBearing
from .magnetic_bearing import MagneticBearing
from .rotor import STEEL, Bearing, Crack, RotorSystem, Stator

# The overhung rotor: a 3 kg disc of radius 150 mm (I_p = m R²/2, I_d = m R²/4), 100 mm past
# bearing B, on stiffer supports so that the shaft, not the supports, does the bending.
OVERHUNG_MASS = 3.0  # kg
OVERHUNG_RADIUS = 0.15  # m
OVERHUNG_BEARING = Bearing(kx=2e5, ky=2e5, cx=400.0, cy=400.0)
# The long shaft: a 2 kg disc of radius 80 mm on a 1 m steel shaft of 20 mm (2.47 kg), on stiff supports.
LONG_MASS = 2.0  # kg
LONG_RADIUS = 0.08  # m
LONG_BEARING = Bearing(kx=1e6, ky=1e6, cx=200.0, cy=200.0, mass=0.5)
# Magnetic bearings: the Magnetic bearing tab's default, with a slower amplifier.
MAGNETIC_BEARING = Bearing(magnet=MagneticBearing(amplifier_hz=400.0))
# Journal bearings: the Fluid-film bearing tab's default, D = 10 mm, L = 5 mm, c = 15 µm, 30 mPa·s oil.
JOURNAL_BEARING = Bearing(film=JournalBearing())


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
            "At a/L = 0.3 the disc tilts as it whirls. The first critical splits only slightly (1731 and "
            "1737 rpm), but the disc's tilting mode splits strongly, from 360 Hz at rest to 289 and 465 Hz at "
            "6000 rpm.",
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
            "Two-pole rotor (asymmetric shaft)",
            RotorSystem(asymmetry=0.15, gravity=9.81, unbalance=5e-6), 3000.0, 785.0,
            "A shaft 15% stiffer about one axis than its mean, 15% softer across it, as a two-pole "
            "generator's slots make it. Laid horizontal, its weight drives a 2X whirl, 447 µm at 785 rpm: "
            "the gravity critical, half the axes' critical speeds (1491 and 1665 rpm). From 1527 to 1631 rpm "
            "it is unstable.",
        ),
        RotorPreset(
            "Cracked shaft (breathing crack)",
            RotorSystem(crack=Crack(position=0.5, depth=0.4), gravity=9.81, unbalance=5e-6), 3000.0, 783.0,
            "A crack 40% through the shaft at the disc, laid horizontal: the weight opens it as its mouth "
            "turns down and shuts it as it turns up, so the stiffness changes once a turn, with 2X and 3X in "
            "it. The weight drives 2X, 37.6 µm at 783 rpm, and 3X, 18.6 µm at 522 rpm: a half and a third of "
            "the critical speeds (1562 and 1574 rpm). Its 1X adds to the unbalance's: 164 µm at the critical, "
            "against 54 µm uncracked.",
        ),
        RotorPreset(
            "Oil swirl λ = ½ (oil whip)",
            RotorSystem(bearing_a=Bearing(swirl=0.5), bearing_b=Bearing(swirl=0.5)),
            10_000.0, 2500.0,
            "The oil swirls round at half the shaft's speed, so the cross-coupling grows as cΩ/2. Past "
            "twice the first natural frequency, 3136 rpm, the forward whirl grows: oil whip, locked at "
            "26 Hz whatever the speed.",
        ),
        RotorPreset(
            "Journal bearings (oil whip)",
            RotorSystem(bearing_a=JOURNAL_BEARING, bearing_b=JOURNAL_BEARING, gravity=9.81, nonlinear_films=True),
            10_000.0, 2500.0,
            "Horizontal, in plain journal bearings: the oil film is stiff, so the critical, 1845 rpm, is "
            "close to the rigid-bearing one (1854 rpm), and the journals rise along the bearing's locus as "
            "the speed rises. A little below twice the critical, at 3576 rpm, the film's own cross-coupling "
            "drives oil whip at 30.5 Hz (0.51×). The films hold the journals inside their clearance, but "
            "not the shaft's bending: the whip grows on until it would rub.",
        ),
        RotorPreset(
            "Oil whirl (vertical rotor)",
            RotorSystem(bearing_a=JOURNAL_BEARING, bearing_b=JOURNAL_BEARING, nonlinear_films=True),
            10_000.0, 3500.0,
            "Upright in plain journal bearings: unloaded, the films have no direct stiffness, so the rotor "
            "whirls at any speed. At 3500 rpm it is oil whirl, at half the speed (29.2 Hz), until the films, "
            "stiffening toward the wall, hold it to a limit cycle with the journals circling past ε = 0.9. "
            "Past twice the first natural frequency, about 3700 rpm, the whirl locks at 30.8 Hz: oil whip, "
            "growing on.",
        ),
        RotorPreset(
            "Partial rub (½X)",
            RotorSystem(unbalance=10e-6, gravity=9.81, stator=Stator(clearance=385e-6)),
            10_000.0, 4400.0,
            "Horizontal, in a seal ring concentric with the bearings, 385 µm round the disc. The disc sags "
            "378 µm under its weight, so above the critical its whirl brushes the bottom of the ring, once "
            "a turn at first. Past about 4000 rpm it hits only every other turn: the orbit closes every two "
            "revolutions, and the spectrum shows ½X (36.7 Hz at 4400 rpm) and its harmonics.",
        ),
        RotorPreset(
            "Full rub (backward whirl)",
            RotorSystem(stator=Stator(clearance=30e-6, friction=0.3)),
            10_000.0, 2000.0,
            "A tight ring, 30 µm round the disc, with dry friction μ = 0.3. Through the critical the whirl "
            "reaches the ring, and the friction, dragging the disc's rim backward, flips it at 1545 rpm "
            "into backward whirl at 227 Hz, the disc's frequency on the ring, far above the speed. The "
            "rim rolls round the ring, so the orbit's radius is R Ω / ω, 7.3 mm at 2000 rpm, pushing the "
            "ring back far past its clearance: dry whirl, which would wreck a real machine.",
        ),
        RotorPreset(
            "Newkirk effect (thermal bow)",
            RotorSystem(unbalance=5e-6, stator=Stator(clearance=15e-6, stiffness=2e4, friction=0.2, offset=10e-6,
                                                      thermal_gain=120e-6, thermal_lag=5.0)),
            10_000.0, 1300.0,
            "A seal 15 µm round the disc, set 10 µm low, so at 1300 rpm, below the 1584 rpm critical, the disc "
            "brushes its bottom. The rub heats the shaft at its high spot and bows it that way, which moves the "
            "high spot further out: the 1X vector spirals out, past 100 µm by 18 s, its lag growing from 13° to "
            "40°, turning against the spin (the Rub tab's polar plot). Above the critical the same rub heals.",
        ),
        RotorPreset(
            "Magnetic bearings",
            RotorSystem(position=1.15, bearing_a=MAGNETIC_BEARING, bearing_b=MAGNETIC_BEARING, gravity=9.81),
            20_000.0, 6000.0,
            "The disc overhung past B, horizontal, floating in active magnetic bearings: the integral holds "
            "both journals at the centre, and the shaft alone sags. The forward critical, 1542 rpm, is well "
            "damped (ζ = 0.054). The disc's forward tilting mode rises with the speed, and at 8894 rpm it "
            "reaches 739 Hz, where the amplifier's lag turns the bearings' damping negative: past it the "
            "whirl grows.",
        ),
    ]
