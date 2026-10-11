"""Single- and two-plane balancing of the Jeffcott rotor by influence coefficients.

As on a real machine: run at the balancing speed and measure the 1X vector at
a probe (V0, "as found"); stop, add a known trial weight T on the disc, run
again and measure V1. The rotor is linear, so the weight's effect is V1 - V0,
and per unit of weight the influence coefficient is

    α = (V1 - V0) / T.

The weight that cancels the reading is then W = -V0 / α. Weights are complex
numbers m r e^{iθ} (kg·m): mass times radius, at the angle θ from the
keyphasor mark in the direction of spin.

A reading at slow roll (S), turning too slowly for any unbalance to matter, is
runout: a bowed shaft, or on a real machine a scratch or a magnetic spot on the
probe's track. It does not come from unbalance, so it is subtracted first:
W = -(V0 - S) / α. A bow does respond to speed, though, so a bowed rotor
balanced at one speed is left out of balance at others; and a weight in one
plane cannot cancel a couple unbalance (a skewed disc) at every speed either.

Two planes and two probes (two-plane balancing) work the same way with
vectors: a trial run for each plane in turn gives a column of the influence
matrix,

    α[:, j] = (V_j - V0) / T_j,

and the weights W = -α⁻¹ (V0 - S) cancel both readings. A disc's unbalance is
a force and a couple, so two weights in planes on the disc can cancel both,
and then the rotor is balanced at every speed. Planes elsewhere on a flexible
shaft cancel the readings only at the balancing speed. Planes close together
make α nearly singular: the weights grow large and small reading errors
swing them (its condition number says how much).

SteadyMeasurement drives a RotorSimulator to a speed, waits for the transient
to die away and averages the 1X vectors over a number of revolutions.
"""

from __future__ import annotations

import math

import numpy as np

from .rotor import RotorSimulator
from .rotor_sweep import RevolutionTracker, StagedRun, TrackedSweep, settle_time

MEASURE_REVS = 10  # revolutions averaged per measurement
SETTLE_CONSTANTS = 6.0  # time constants of the least damped mode to wait first (e^-6: 0.25% left)
SETTLE_LONGEST = 4.0  # s, at most
PROBES = ("x", "y")  # the probe directions balancing works with: linear in the response


def weight(mass: float, radius: float, angle: float) -> complex:
    """A weight of `mass` kg at `radius` m and `angle` rad from the keyphasor mark, as m r e^{iθ} (kg·m)."""
    return mass * radius * complex(math.cos(angle), math.sin(angle))


def weight_parts(w: complex, radius: float) -> tuple[float, float]:
    """(mass in kg, angle in [0, 2π)) of the weight w (kg·m) placed at `radius` m."""
    return abs(w) / radius, math.atan2(w.imag, w.real) % (2.0 * math.pi)


def influence(as_found: complex, trial_run: complex, trial: complex) -> complex:
    """α = (V1 - V0) / T: the reading's change per unit of weight (m per kg·m)."""
    if trial == 0:
        raise ValueError("the trial weight must not be zero")
    return (trial_run - as_found) / trial


def correction(as_found: complex, alpha: complex, slow_roll: complex = 0j) -> complex:
    """W = -(V0 - S) / α: the weight (kg·m) that cancels the reading less its slow-roll runout."""
    if alpha == 0:
        raise ValueError("the trial weight changed nothing: no influence coefficient")
    return -(as_found - slow_roll) / alpha


def influence_matrix(as_found: np.ndarray, trial_runs: np.ndarray, trials: np.ndarray) -> np.ndarray:
    """α (probes x planes): column j is (V_j - V0) / T_j, from a trial run with weight T_j in plane j alone.

    as_found (probes,), trial_runs (planes, probes), trials (planes,) complex.
    """
    as_found, trial_runs, trials = (np.asarray(a, dtype=complex) for a in (as_found, trial_runs, trials))
    if not np.all(trials):
        raise ValueError("the trial weights must not be zero")
    return (trial_runs - as_found).T / trials


def corrections(as_found: np.ndarray, alpha: np.ndarray, slow_roll: np.ndarray | complex = 0j) -> np.ndarray:
    """W = -α⁻¹ (V0 - S): the weights (kg·m), one per plane, that cancel the readings less their runout.

    With more probes than planes, the least-squares weights.
    """
    v = np.asarray(as_found, dtype=complex) - slow_roll
    alpha = np.asarray(alpha, dtype=complex)
    if np.linalg.matrix_rank(alpha) < alpha.shape[1]:
        raise ValueError("the trial runs cannot tell the planes apart: no influence matrix")
    return -np.linalg.lstsq(alpha, v, rcond=None)[0]


class SteadyMeasurement(StagedRun):
    """Run a RotorSimulator at `omega` (rad/s), let it settle, and average the 1X vectors over `revs` turns.

    Stages: "approach" (ramping at the simulator's rate), "settle", "measure",
    then "done", when `result` holds the mean 1X vectors at the four stations,
    (4, 2) complex, as steady_vectors gives them.
    """

    def __init__(self, sim: RotorSimulator, omega: float, revs: int = MEASURE_REVS,
                 settle: float | None = None) -> None:
        if omega <= 0.0 or revs < 1:
            raise ValueError("measure at a positive speed, over at least one revolution")
        super().__init__(sim)
        self.omega, self.revs = omega, revs
        self.settle = settle_time(sim.system, omega, SETTLE_CONSTANTS, SETTLE_LONGEST) if settle is None else settle
        self.sweep = TrackedSweep(rate=0.0, up=True)
        self.result: np.ndarray | None = None
        self._settle_end = 0.0
        sim.target = omega

    @property
    def tracked(self) -> int:
        """Revolutions measured so far."""
        return len(self.sweep)

    def _until_next_stage(self) -> float:
        sim = self.sim
        if self.stage == "approach":
            return abs(sim.target - sim.omega) / sim.accel
        if self.stage == "settle":
            return self._settle_end - sim.t
        if self.stage == "measure":
            if self.tracked >= self.revs:
                return 0.0
            # The last revolution is recorded once a sample past its end comes in.
            return (self.revs - self.tracked + 0.05) * 2.0 * math.pi / self.omega
        return math.inf

    def _next_stage(self) -> None:
        if self.stage == "approach":
            self.stage, self._settle_end = "settle", self.sim.t + self.settle
        elif self.stage == "settle":
            self.stage = "measure"
            self._tracker = RevolutionTracker(self.sweep)
        elif self.stage == "measure":
            self.stage, self._tracker = "done", None
            self.result = np.mean(np.array(self.sweep.vectors[: self.revs]), axis=0)

    def run(self, chunk: float = 0.25) -> np.ndarray:
        """Measure to the end (for scripts and tests); return the mean 1X vectors (4, 2)."""
        super().run(chunk)
        return self.result
