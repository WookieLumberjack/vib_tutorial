"""The rotor page's analysis: the Campbell diagram, the stability onset and the steady states.

`analyse` works them out for a rotor, which takes 0.2–5 s (a turning shaft's
modes come from Hill's method, an eigenproblem per speed). The page runs it in
`AnalysisWorker`, a process of its own: on a thread it would share the GIL with
the page's frames, which hold it for most of each frame, and take several times
as long.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
import traceback
import types
from collections import OrderedDict
from dataclasses import dataclass, replace

import numpy as np

from .. import ONE_THREAD
from .rotor import RPM, Campbell, RotorSystem, campbell, steady_harmonics
from .rotor_stability import Onset, stability_onset
from .rotor_sweep import station_motion

STEADY_POINTS = 2000  # speeds on the steady-state curve, plus as many again around the criticals
ORDERS = 3  # the steady states' harmonics: 1X, 2X and 3X
KEEP_RESULTS = 8  # results the worker holds for collecting, the oldest dropped first
# The worker's BLAS runs on one thread (as the app's, see vib_tutorial.main): more barely speed its
# eigenproblems up (5.8 s to 5.2 s on six cores), and their pools, spinning beside the app's, would
# slow both down several times.


@dataclass
class Analysis:
    """What the page shows of a rotor apart from its motion: worked out once its parameters settle."""

    campbell: Campbell
    massless: Campbell | None  # with the shaft's mass: the same rotor's Jeffcott model, for comparison
    onset: Onset | None  # where the rotor goes unstable, if it does up to the top speed
    steady_rpm: np.ndarray  # the steady-state curve's speeds (rpm)
    steady_orders: list[np.ndarray]  # and its 1X, 2X and 3X vectors there, (speeds, stations, 2) each


def analyse(system: RotorSystem, top_rpm: float) -> Analysis:
    """The Campbell diagram, the stability onset and the steady states, up to `top_rpm`."""
    top = top_rpm / RPM
    result = campbell(system, top, n_modes=8)
    massless = campbell(system.with_(density=0.0), top, n_modes=8) if system.density else None
    onset = stability_onset(system, top)
    # The steady-state vectors at every station, finely around each critical speed.
    w = [np.linspace(top / STEADY_POINTS, top, STEADY_POINTS)]
    crits = [c for c, _, _ in result.criticals]
    if system.periodic and system.gravity:
        # Gravity's 2X peaks near half of each (and a breathing crack's 3X near a third).
        crits += [c / k for c in crits for k in ((2, 3) if system.spacing == 1 else (2,))]
    w += [np.linspace(0.8 * c, 1.2 * c, STEADY_POINTS // max(len(crits), 1)) for c in crits]
    w = np.unique(np.concatenate(w))
    V = steady_harmonics(system, w, ORDERS)[1]
    return Analysis(result, massless, onset, w * RPM, [station_motion(system, V[:, k]) for k in range(ORDERS)])


def _serve(conn) -> None:
    """The worker process: analyse each (ticket, system, top_rpm) received, in turn, and send back the result."""
    while True:
        try:
            ticket, system, top_rpm = conn.recv()
        except EOFError:  # the app is gone
            return
        try:
            reply = (ticket, analyse(system, top_rpm), None)
        except Exception:  # noqa: BLE001 (reported on the page's side)
            reply = (ticket, None, traceback.format_exc())
        try:
            conn.send(reply)
        except OSError:  # the app is gone
            return


class AnalysisWorker:
    """`analyse` in a process of its own, started on first use; jobs are run in the order submitted.

    `submit` hands it a rotor and returns a ticket, `result(ticket)` the
    Analysis once it's done (None until then). `restart` drops the jobs
    underway (a fresh process starts in ~0.3 s, quicker than seeing out a
    turning shaft's analysis no one wants any more); their tickets are then
    `lost`. Should the process fail to start, it works them out at once instead.
    """

    def __init__(self) -> None:
        self._process: multiprocessing.process.BaseProcess | None = None
        self._conn = None
        self._next = 0
        self._first = 1  # the first ticket the running process was given
        self._results: OrderedDict[int, Analysis | str] = OrderedDict()  # done: the analysis, or its error

    @property
    def running(self) -> bool:
        """True if the process is up (False before the first job, or if it couldn't start)."""
        return self._process is not None and self._process.is_alive()

    def start(self) -> None:
        """Start the process now, so the first job doesn't wait for it to import NumPy and SciPy."""
        if self.running:
            return
        ctx = multiprocessing.get_context("spawn")  # never fork a process with Qt's threads in it
        self._conn, child = ctx.Pipe()
        self._process = ctx.Process(target=_serve, args=(child,), name="rotor-analysis", daemon=True)
        # A spawned process imports the parent's main script again, unless it's a module: any script that
        # builds the page without an `if __name__ == "__main__"` guard would then build it in the worker
        # too. The worker needs nothing from it, so hide it while the process starts.
        main = sys.modules["__main__"]
        sys.modules["__main__"] = types.ModuleType("__main__")
        env = {name: os.environ.get(name) for name in ONE_THREAD}
        os.environ.update(dict.fromkeys(ONE_THREAD, "1"))  # the process starts with a copy of these
        try:
            self._process.start()
        except OSError:  # no processes here: work the jobs out on the spot
            self._process = self._conn = None
            return
        finally:
            sys.modules["__main__"] = main
            for name, value in env.items():
                if value is None:
                    os.environ.pop(name)
                else:
                    os.environ[name] = value
        child.close()

    def restart(self) -> None:
        """Drop the jobs underway, and start afresh."""
        self.close()
        self._first = self._next + 1
        self.start()

    def lost(self, ticket: int) -> bool:
        """True if `ticket`'s job was dropped by a restart: submit it again."""
        return ticket < self._first and ticket not in self._results

    def submit(self, system: RotorSystem, top_rpm: float) -> int:
        self._next += 1
        ticket = self._next
        self.start()
        if self.running:
            # A copy without the caches the page's simulation filled (it sends faster, and the caches stay here).
            self._conn.send((ticket, replace(system), top_rpm))
        else:
            self._keep(ticket, analyse(system, top_rpm))
        return ticket

    def result(self, ticket: int) -> Analysis | None:
        """The analysis for `ticket` once it's done, else None. Raises if it failed, or the process stopped."""
        while ticket not in self._results and self._conn is not None and self._conn.poll():
            done, analysis, error = self._conn.recv()
            self._keep(done, analysis if error is None else error)
        if ticket not in self._results:
            if self._process is not None and not self._process.is_alive():
                self._process = self._conn = None  # the next job starts another
                raise RuntimeError("the rotor analysis process stopped")
            return None
        analysis = self._results.pop(ticket)
        if type(analysis) is str:  # its traceback
            raise RuntimeError(f"rotor analysis failed:\n{analysis}")
        return analysis

    def _keep(self, ticket: int, analysis: Analysis | str) -> None:
        self._results[ticket] = analysis
        while len(self._results) > KEEP_RESULTS:
            self._results.popitem(last=False)

    def close(self) -> None:
        if self._process is not None:
            self._conn.close()
            self._process.terminate()
            self._process.join(timeout=1.0)
        self._process = self._conn = None
