# Vibration Tutorial

An interactive, real-time teaching tool for multi-degree-of-freedom mechanical vibration.
It simulates a chain of lumped masses connected by springs and dampers, animates the
motion, and shows the modal-analysis solution for reference, all updating live while
you change parameters.

```
ground ──[k1,c1]── m1 ──[k2,c2]── m2 ──[k3,c3]── m3 ──[k4,c4]── m4
```

## Quick start

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync               # create .venv and install dependencies
uv run vib-tutorial   # launch the app (or: uv run python -m vib_tutorial)
uv run pytest         # run the tests
```

## What you can do

- **Edit any mass, stiffness, or damping value while the simulation runs.** The state is
  kept, so you see the system respond to the change. You can use 1 to 8 masses.
- **Apply a force to any mass**: a step, a harmonic `F sin(2πft)`, or a rectangular pulse.
  Press **Space** (or the button) to switch it on and off, and watch the transients as it
  starts and stops.
- **Tune the drive frequency to a natural frequency** from the "Tune to…" menu to see
  resonance build up.
- **Release a mode**: select a mode in the table and click *Release selected mode*. The
  masses start from that mode shape and oscillate at its frequency. With proportional
  damping, only that mode responds.
- **Slow motion** (0.05× to 2×) for the higher modes, and auto-scaled animation so small
  motions stay visible. The scale bar shows the real displacement.

## Reference panels

| Panel | What it shows |
|---|---|
| Modal table | Undamped natural frequency fₙ, modal damping ratio ζ = φᵀCφ/(2ωₙ), exact ζ and damped frequency f_d from the complex eigenvalues |
| Mode shapes | Undamped mode shapes vs. position (can be animated) |
| Frequency response | Receptance magnitude and phase \|X_i/F\| for the forced mass, with natural frequencies and the drive frequency marked |

The note under the modal table tells you whether the damping is **proportional**. If it
is, ζ modal equals ζ exact and the modes are real. Try changing only `c1` to see the
difference: the modes become complex and ζ modal is only an approximation.

## How it works

- `core/model.py` assembles the diagonal **M** and the tridiagonal **C** and **K**
  element by element.
- `core/modal.py` solves `Kφ = ω²Mφ` (`scipy.linalg.eigh`) for the undamped modes. It
  solves the eigenvalues of the 2N×2N state matrix for the exact damped poles
  `λ = −ζωₙ ± iω_d`, and matches the two sets using MAC.
- `core/simulator.py` integrates `z' = Az + Bu` using the **exact matrix-exponential
  transition** with a first-order hold on the force. It is unconditionally stable and adds
  no numerical damping, so any damping you see is physical. The step size adapts to the
  fastest mode and the drive frequency.
- `gui/` is the PySide6 and pyqtgraph interface. It runs on a ~60 fps timer that advances
  the simulator by wall-clock time × speed.

The `core` package has no Qt dependency, so you can use it from scripts or notebooks:

```python
from vib_tutorial.core import ChainSystem, modal_analysis
res = modal_analysis(ChainSystem.uniform(4, mass=1.0, stiffness=400.0, damping=2.0))
[(m.fn_hz, m.damped.zeta) for m in res.modes]
```

## Ideas for extension

- Drag a mass with the mouse and let go (an initial-condition "pluck")
- Frequency sweep (chirp) forcing, and base excitation instead of an applied force
- Energy bars (kinetic, potential, dissipated) and modal-coordinate time histories
- Tuned mass damper and vibration absorber presets
- Save and load parameter presets for classroom exercises
