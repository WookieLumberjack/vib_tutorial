# Vibration Tutorial

An interactive, real-time teaching tool for multi-degree-of-freedom mechanical vibration.
It simulates a chain of lumped masses connected by springs and dampers, animates the
motion, and shows the modal-analysis solution for reference, all updating live while
you change parameters.

```
ground ──[k1,c1]── m1 ──[k2,c2]── m2 ──[k3,c3]── m3 ──[k4,c4]── m4
```

![Main window: a harmonic force tuned to mode 2 drives the chain; the modal table and mode shapes are on the right](docs/images/main_window.png)

## Quick start

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync               # create .venv and install dependencies
uv run vib-tutorial   # launch the app (or: uv run python -m vib_tutorial)
uv run pytest         # run the tests
```

## The window

| Area | What it holds |
|---|---|
| **Left** | System parameters (m, k, c for each element, 1 to 8 masses), the applied force, and simulation controls (run/pause, speed, plot window, auto-scale) |
| **Centre** | Animation of the chain (with a scale bar for the real displacement) above time histories of the applied force and of the motion, in physical or modal coordinates |
| **Right** | Tabs: *Modal analysis* (table, mode shapes, release), *Frequency response*, and *Background* (theory notes written for students) |

Two more pages work on the same chain: **FRF matrix** shows every term of the receptance
matrix and how the modes build it up, and **Substructuring (Craig–Bampton)** reduces the
chain by component mode synthesis (both below).

## What you can do

- **Edit any mass, stiffness, or damping value while the simulation runs.** The state is
  kept, so you see the system respond to the change.
- **Apply a force to any mass**: a step, a harmonic `F sin(2πft)`, or a rectangular pulse.
  Press **Space** (or the button) to switch it on and off, and watch the transients as it
  starts and stops.
- **Tune the drive frequency to a natural frequency** from the "Tune to…" menu to see
  resonance build up.
- **Release a mode**: select a mode in the table and click *Release selected mode*. The
  masses start from that mode shape and vibrate freely. The plot window can be fitted to
  a number of cycles of any mode.
- **Switch between two modal-analysis methods** (below), and compare them on the same
  system.
- **Plot the motion in modal coordinates** (*Plot coordinates → Modal*, above the time
  histories): one curve per mode instead of one per mass (below).
- **Slow motion** (0.05× to 2×) for the higher modes, and auto-scaled animation and plots
  so small motions stay visible.

### Two modal-analysis methods

The *Method* selector on the Modal analysis tab switches between two ways of solving for
the modes.

**Classical: N real modes.** The undamped eigenproblem gives N natural frequencies and
real mode shapes; damping is estimated per mode. The table compares this estimate
(ζ modal) with the exact value (ζ exact). The note under the table says whether the
damping is *proportional*. If it is not, the estimate is only an approximation, and
releasing a real mode shape excites other modes too. Below, with only `c1` raised to 15,
the release of mode 3 leaves slow mode-1 motion behind:

![Classical method with non-proportional damping: releasing mode 3 also excites mode 1](docs/images/classical_release_nonproportional.png)

**State-space: 2N complex modes.** The full damped problem is solved in first-order form,
which gives all 2N eigenvalues: complex-conjugate pairs λ, λ* for oscillatory modes (and
real eigenvalues for overdamped or rigid-body motion). Selecting a row shows its complex
mode shape in a complex-plane (phasor) plot, with the amplitude and phase of each mass.
Phases other than 0° or 180° mean the masses peak at different times. Releasing a
complex mode sets both displacement and velocity, so on the same system only that mode
responds and decays cleanly:

![State-space method on the same system: 2N eigenvalues, the complex-plane plot of λ5, and a clean single-mode release](docs/images/state_space_release.png)

### Modal coordinates

*Plot coordinates → Modal* splits the motion into one curve per mode. Each curve is that
mode's share of the displacement of the mass it moves most, so its size compares
directly with the physical plot. The curves follow the *Method* selector:

- **Classical**: q = Φ<sup>T</sup>Mx with the real mode shapes, so x = Σ φ<sub>r</sub>q<sub>r</sub>
  exactly. With non-proportional damping these coordinates are coupled. Release mode 3 with
  `c1` = 15 and modes 1 and 2 pick up motion.
- **State-space**: η = V<sup>−1</sup>z with the complex eigenvectors, one curve 2 Re(η) per
  conjugate pair. These decouple for any damping. A released complex mode moves only its
  own curve.

![Modal coordinates, classical method, c1 = 15: releasing mode 3 also drives modes 1 and 2](docs/images/modal_coordinates.png)

### Frequency response

The receptance |X_i / F| and phase of every mass for a force at the selected mass. The
natural frequencies are dotted and the current drive frequency is dashed.

<img src="docs/images/frequency_response.png" alt="Frequency response magnitude and phase" width="420">

### FRF matrix

The *Frequency response* tab shows one column of the receptance matrix. The **FRF matrix**
page shows all N × N terms H<sub>jk</sub> = x<sub>j</sub>/F<sub>k</sub> as a grid of small
plots (row: response, column: force), on shared axes so their sizes compare directly.

- **Show** magnitude, phase (unwrapped), real part or imaginary part. The large plot on the right
  shows the selected term as magnitude with phase, or real with imaginary part.
- **Modes included in the sum**: each term is a sum of one term per mode. Untick modes to see
  the truncated modal sum (dashed red) against the full solution (black), and each mode's term
  in its own colour. The real part shows best how the terms add. The large plot shades the
  frequency band where the sum differs most from the full solution, and lists the **static
  compliance** (ω = 0): the ticked modes' sum against K⁻¹, and each mode's share of it.
- **Exact or classical terms**: the exact expansion uses the complex (state-space) modes and
  always sums to the full solution. The classical one uses the real modes with ζ modal, and
  misses even with every mode when damping is non-proportional.
- **Several forces at once**: Ctrl+click more plots in a row, or tick more forces, to see
  x<sub>j</sub> = Σ<sub>k</sub> H<sub>jk</sub>F<sub>k</sub> built from its terms.
- **Theory**: reciprocity, modal constants, why antiresonances appear (and how many), and
  truncation.

![FRF matrix page: mode 4 left out of the sum; the end-to-end term H14 misses above 5 Hz](docs/images/frf_matrix.png)

### Substructuring (Craig–Bampton)

The second page cuts the chain at one *interface* mass into substructure A (grounded) and
B (free end). The boundary (master) DOFs are the interface and the last mass, which is
where the force is applied, so the loaded DOF stays physical. Each substructure keeps a
chosen number of its fixed-interface normal modes plus one constraint mode per boundary
DOF, and the reduced substructures are assembled on the shared interface DOF.

- **Comparison table**: true vs reduced natural frequencies, the error (never negative:
  CB frequencies are upper bounds), and the MAC of the recovered shape.
- **Substructures on their own**: each substructure's fixed-interface modes (natural
  frequency and exact damping ratio with its boundary held), whether each is kept, and the
  coupled mode it becomes (largest share of that mode's strain energy).
- **Component shape overlay**: click a row of that table to draw the clamped component
  shape over the coupled mode it becomes.
- **Basis (T)**: every column of T drawn as a shape along the chain, the set of shapes every
  reduced-model motion is built from.
- **Modes & FRF**: the selected mode shape, true vs CB, and the tip and interface
  receptance of both models for the force at the tip.
- **Matrices (step by step)**: every stage with the current numbers: the full K and M,
  each substructure's partitioned matrices, the fixed-interface modes (kept and discarded),
  the constraint modes Ψ, the transformation T, the reduced matrices, and the assembled
  reduced model.
- **Theory**: what a basis is (with a 2-mass worked example) and why Craig–Bampton chooses
  its shapes, original vs substructured formulation side by side, Hurty vs Craig–Bampton,
  Guyan reduction as the zero-mode case, and a "Try it" walkthrough, and a notation table.
- **Presets**: *Guyan (0 modes)* and *All modes (exact)*.

For each substructure, partitioned into interior $i$ and boundary $b$ DOFs,

$$\begin{bmatrix}x_i\\ x_b\end{bmatrix} = \underbrace{\begin{bmatrix}\Phi_k & \Psi\\ 0 & I\end{bmatrix}}_{T}\begin{bmatrix}q\\ x_b\end{bmatrix},
\qquad \Psi = -K_{ii}^{-1}K_{ib},\qquad \hat M = T^TMT,\ \hat K = T^TKT$$

where $\Phi_k$ are the $k$ lowest mass-normalized modes of $K_{ii}\phi = \omega^2 M_{ii}\phi$.
$\hat K$ is block diagonal, $\mathrm{diag}(\omega_k^2)$ and $K_{bb} - K_{bi}K_{ii}^{-1}K_{ib}$, while
$\hat M$ couples $q$ to $x_b$.

## The math, briefly

The *Background* tab in the app explains this in more depth; this is the outline.

**Model.** Newton's second law for each mass gives

$$M\ddot{x} + C\dot{x} + Kx = f(t)$$

where $M$ is diagonal (the masses) and $C$, $K$ are tridiagonal, assembled element by
element: each spring or damper between two neighbours adds $\begin{bmatrix}v & -v\\ -v & v\end{bmatrix}$
to their rows and columns (element 1 connects to ground). Everything is SI.

**Classical modal analysis.** Ignoring damping, $x = \phi\cos\omega t$ gives the symmetric
generalized eigenproblem $K\phi = \omega^2 M\phi$, which has N real natural frequencies
$\omega_r$ and real mode shapes $\phi_r$. With mass-normalized shapes collected in $\Phi$,
$\Phi^T M\Phi = I$ and $\Phi^T K\Phi = \mathrm{diag}(\omega_r^2)$. Damping is estimated
from the diagonal of $\Phi^T C\Phi$:

$$\zeta_r = \frac{\phi_r^T C\,\phi_r}{2\omega_r}$$

This is exact only if $\Phi^T C\Phi$ is diagonal (*proportional* damping, e.g.
$C = \alpha M + \beta K$). The app reports a coupling index, the largest off-diagonal term
relative to the diagonal, to show how far from proportional the damping is.

**State-space (complex) modal analysis.** With $z = [x,\ \dot{x}]$ the N second-order
equations become 2N first-order ones:

$$\dot{z} = Az + Bf, \qquad A = \begin{bmatrix} 0 & I \\ -M^{-1}K & -M^{-1}C \end{bmatrix}$$

The eigenproblem $A\psi = \lambda\psi$ is exact for any damping. It has 2N eigenvalues,
$\lambda = -\zeta\omega_n \pm i\omega_d$ in conjugate pairs, and eigenvectors of the form
$[\psi_x,\ \lambda\psi_x]$. One pair together gives a real motion
$x(t) = 2\,\mathrm{Re}(\psi_x e^{\lambda t})$. With non-proportional damping $\psi_x$ is
complex, so each mass has its own phase. The two methods are linked by matching each
damped pair to the undamped mode it most resembles (the modal assurance criterion, MAC).

**Frequency response.** The receptance is solved directly at each frequency:
$H(\omega) = (K - \omega^2 M + i\omega C)^{-1}$. It is also a sum of modal terms. With the
state-space eigenvectors $V$ (and $V^{-1}$), every eigenvalue contributes a residue matrix,

$$H(\omega) = \sum_{r=1}^{2N} \frac{R_r}{i\omega - \lambda_r}, \qquad R_r = (V)_{x,r}\,(V^{-1})_{r,\dot{x}}\,M^{-1}$$

which is exact for any damping. For proportional damping it reduces to the classical
$H = \sum_r \phi_r\phi_r^T / (\omega_r^2 - \omega^2 + 2i\zeta_r\omega_r\omega)$.

**Time simulation.** The simulator advances $\dot{z} = Az + Bf$ with the exact
discrete-time solution. Over a step $h$, with the force varying linearly across the step
(first-order hold),

$$z_{k+1} = e^{Ah}z_k + \Gamma_0 f_k + \Gamma_1 f_{k+1}$$

where $e^{Ah}$, $\Gamma_0$ and $\Gamma_1$ come from a single matrix exponential of an augmented
matrix. The free response is exact, so the scheme is unconditionally stable and adds no
numerical damping, however stiff the springs are made: any decay you see is physical. The
step size adapts to the fastest mode, the drive frequency and the pulse length.

## Code layout

- `core/model.py`: assembles $M$, $C$, $K$ and the state-space matrices.
- `core/modal.py`: classical modes (`scipy.linalg.eigh`), all 2N state-space eigenpairs
  (`scipy.linalg.eig`), the MAC pairing, the modal-coordinate map, and the frequency response.
- `core/frf_matrix.py`: the full receptance matrix and its modal (pole–residue) terms.
- `core/simulator.py`: the exact first-order-hold time stepper.
- `core/substructure.py`: Craig–Bampton substructuring, mode comparison (frequency error,
  MAC) and the reduced-model FRF.
- `gui/`: the PySide6 and pyqtgraph interface. It runs on a ~60 fps timer that advances
  the simulator by wall-clock time × speed.

The `core` package has no Qt dependency, so you can use it from scripts or notebooks:

```python
from vib_tutorial.core import ChainSystem, modal_analysis

res = modal_analysis(ChainSystem([1.0] * 4, [400.0] * 4, [15.0, 2.0, 2.0, 2.0]))
[(m.fn_hz, m.zeta_modal, m.damped.zeta) for m in res.modes]    # classical, N modes
[(m.eigenvalue, m.shape) for m in res.complex_modes]           # state-space, 2N modes

from vib_tutorial.core import compare_modes, craig_bampton

s = ChainSystem.uniform(8)
cb = craig_bampton(s, interfaces=[3], n_kept=[1, 1])          # cut at m4, 1 mode each
[(c.fn_true, c.fn_cb, c.mac) for c in compare_modes(cb, modal_analysis(s))]
```

## Ideas for extension

- Drag a mass with the mouse and let go (an initial-condition "pluck")
- Frequency sweep (chirp) forcing, and base excitation instead of an applied force
- Energy bars (kinetic, potential, dissipated), including each mode's energy
  ½(q̇<sub>r</sub>² + ω<sub>r</sub>²q<sub>r</sub>²), to show energy moving between modes under
  non-proportional damping
- Tuned mass damper and vibration absorber presets
- Save and load parameter presets for classroom exercises
- Substructuring: time-simulate the Craig–Bampton reduced model alongside the full one
  under the same tip force, so the reduction error shows up in the animation and time
  histories
- Substructuring: more than one interface (three or more substructures); the core
  (`craig_bampton`) already accepts several cuts, so only the controls and schematic need
  extending
- Substructuring: free-interface component mode synthesis (MacNeal, Rubin) as a comparison
  with the fixed-interface Craig–Bampton method
