"""Things to try on the Simulation page: each sets up the chain and the controls, and says what to look for.

`MainWindow._on_trial` applies a trial's settings; the Theory tab lists the same notes.
`tests/test_tutorial_numbers.py` checks their numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core import ForceKind


@dataclass(frozen=True)
class SimTrial:
    name: str
    note: str
    preset: str = ""  # a preset's name (or the start of it); else the uniform 4-mass chain, changed by:
    n: int = 4
    damping: dict[int, float] = field(default_factory=dict)  # element index -> c
    friction: float = 0.0  # N on every mass
    state_space: bool = False
    coords: str = "Physical"  # the Plot coordinates entry it starts with
    by_mode: bool = False  # the energy panel by mode
    speed: float = 1.0
    window: float = 10.0  # s of plot
    release: int | None = None  # the modal table's row to release
    amplitude: float = 20.0  # mm, the release's initial displacement
    animate: bool = False
    tune: int = 0  # mode to tune the harmonic force to (1-based), 0 to leave it
    kind: ForceKind | None = None
    sweep: tuple[float, float, float] | None = None  # chirp start, end (Hz) and time (s)
    base: tuple[float, float] | None = None  # ground motion: amplitude (mm) and frequency (Hz)
    apply: bool = False  # switch the force on
    frf_tab: bool = False  # show the Frequency response tab


TRIALS = (
    SimTrial(
        "Resonance building up",
        "A harmonic force on m4, tuned to mode 1 (1.11 Hz). The swing grows over a few of the mode's time "
        "constants, 1/ζω = 8.3 s, to a steady amplitude where the dampers take out what the force puts in. "
        "The Frequency response tab marks the drive (dashed) on the peak.",
        tune=1, apply=True, window=30.0),
    SimTrial(
        "Proportional damping: one mode stays one mode",
        "Every damper is 0.005 times its spring (C = βK), the default chain. Mode 3 is released at 0.25× "
        "speed: it decays in one clean shape, only its curve moves in modal coordinates, and it keeps 100% "
        "of the energy (Energy: by mode) while the total decays.",
        coords="Modal", by_mode=True, speed=0.25, window=2.05, release=2),
    SimTrial(
        "Non-proportional damping: modes trade energy",
        "c1 = 15 N·s/m, the other dampers at 2. The note under the modal table says non-proportional, and ζ "
        "modal and ζ exact separate. Mode 3's real shape is no longer an exact mode of the damped chain: "
        "released, it feeds modes 1 and 2, and within a second most of the energy left is in mode 1, which "
        "keeps ringing after mode 3 has died away.",
        damping={0: 15.0}, coords="Modal", by_mode=True, speed=0.25, window=2.05, release=2),
    SimTrial(
        "A complex mode, released",
        "c1 = 15 again, with Method → State-space: the table lists 2N eigenvalues in conjugate pairs. Mode 3's "
        "λ is released with its velocities as well as its displacements, so it stays in one shape as it "
        "decays. The arrows in the complex plane fan out (phases other than 0° and 180°), and the shape never "
        "passes through zero everywhere at once. Release λ* instead: the motion is the same.",
        damping={0: 15.0}, state_space=True, speed=0.25, window=2.12, release=4, animate=True),
    SimTrial(
        "An overdamped pair (c1 = 200)",
        "A damper so strong that one pair of eigenvalues turns into two real ones, one slow and one very "
        "fast, listed with their time constants instead of a frequency. There are still 2N in all.",
        damping={0: 200.0}, state_space=True),
    SimTrial(
        "Friction: a straight-line decay",
        "One mass with 0.5 N of friction and a damper too weak to matter (0.01 N·s/m), released from 20 mm. "
        "Each cycle it loses the same "
        "4F<sub>f</sub>/k = 5 mm, not the same fraction: after four cycles (1.3 s) it sticks, at rest. The "
        "modal table's ζ friction column is the equivalent viscous damping at the release amplitude.",
        n=1, damping={0: 0.01}, friction=0.5, window=2.0, release=0),
    SimTrial(
        "A chirp through the modes",
        "A sweep from 2 to 7 Hz over 40 s at m4, in modal coordinates: mode 1 rings from the switch-on and "
        "decays, then modes 2, 3 and 4 swell in turn as the sweep passes them. Sweep faster (Sweep time) and "
        "the peaks come later and lower.",
        coords="Modal", kind=ForceKind.CHIRP, sweep=(2.0, 7.0, 40.0), apply=True, window=40.0, frf_tab=True),
    SimTrial(
        "Ground motion: isolation",
        "The ground moves 5 mm at 1.6 Hz instead of a force pushing m4. The Frequency response tab shows the "
        "transmissibility: 1 at low frequency, peaks at the modes, and below 1 well above them. At 1.6 Hz m3 "
        "already moves less than the ground.",
        base=(5.0, 1.6), apply=True, window=4.0, frf_tab=True),
    SimTrial(
        "Vibration absorber (undamped)",
        "A small mass tuned to the machine's frequency, driven there. After the start-up transient m1 stands "
        "still and only the absorber moves: in Element forces its spring carries the applied force. Change "
        "the drive frequency a little and m1 starts moving again.",
        preset="Vibration absorber", coords="Element forces", apply=True),
    SimTrial(
        "Tuned mass damper (Den Hartog)",
        "The Frequency response tab: the dashed primary alone peaks at 40 times its static deflection, the "
        "damped system at under 6. Set c2 to 0: two sharp peaks. Set it to 5: the damper locks the absorber "
        "to m1 and one tall peak comes back. Every curve passes close to the same two fixed points.",
        preset="Tuned mass damper", frf_tab=True, apply=True),
)
