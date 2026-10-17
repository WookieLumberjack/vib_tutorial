"""The rotor page's analysis, worked out in a process of its own."""

import subprocess
import sys
import textwrap
import time

import numpy as np
import pytest

from vib_tutorial.core.rotor import Bearing, RotorSystem
from vib_tutorial.core.rotor_analysis import AnalysisWorker, analyse


def _wait(worker, ticket, timeout=60.0):
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        if (result := worker.result(ticket)) is not None:
            return result
        time.sleep(0.01)
    raise AssertionError("the worker never answered")


def _same(a, b):
    assert np.array_equal(a.campbell.freq_hz, b.campbell.freq_hz, equal_nan=True)
    assert a.campbell.criticals == b.campbell.criticals and a.onset == b.onset
    assert np.array_equal(a.steady_rpm, b.steady_rpm)
    for x, y in zip(a.steady_orders, b.steady_orders, strict=True):
        assert np.allclose(x, y, rtol=1e-12, atol=1e-18)


@pytest.fixture
def worker():
    w = AnalysisWorker()
    yield w
    w.close()


def test_worker_matches_analyse(worker):
    system = RotorSystem(bearing_a=Bearing(kx=2e4, ky=8e4, cx=20.0, cy=20.0), gravity=9.81)
    ticket = worker.submit(system, 8000.0)
    assert worker.running
    _same(_wait(worker, ticket), analyse(system, 8000.0))


def test_restart_drops_the_jobs_underway(worker):
    first = worker.submit(RotorSystem(), 6000.0)
    worker.restart()
    assert worker.lost(first) and worker.result(first) is None
    second = worker.submit(RotorSystem(unbalance=2e-4), 6000.0)
    assert not worker.lost(second)
    assert _wait(worker, second).steady_rpm.size


def test_a_script_without_a_main_guard_runs_once(tmp_path):
    """A spawned process imports the parent's main script again: the worker mustn't."""
    log = tmp_path / "runs.txt"
    script = tmp_path / "unguarded.py"
    script.write_text(textwrap.dedent(f"""
        import time
        from vib_tutorial.core.rotor import RotorSystem
        from vib_tutorial.core.rotor_analysis import AnalysisWorker
        with open({str(log)!r}, "a") as f:
            f.write("run\\n")
        w = AnalysisWorker()
        ticket = w.submit(RotorSystem(), 6000.0)
        while w.result(ticket) is None:
            time.sleep(0.01)
        assert w.running
        w.close()
    """))
    subprocess.run([sys.executable, str(script)], check=True, timeout=120)
    assert log.read_text() == "run\n"
