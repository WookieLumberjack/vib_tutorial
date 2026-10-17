"""Interactive teaching tool for lumped mass-spring-damper vibration."""

import time

# When the interpreter reached this package, for the startup timings --smoke-test records.
STARTED = time.perf_counter()


# BLAS on one thread: the app's matrices are small, and a pool of threads spinning up for each
# solve costs far more than it saves (the cracked shaft's frames: 91 ms on six threads, 14 ms on one).
# Read when NumPy loads, so set before anything imports it; a value set already is kept.
ONE_THREAD = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def main() -> None:
    import multiprocessing
    import os

    multiprocessing.freeze_support()  # a packaged app's worker processes start here: run theirs, not the app
    for name in ONE_THREAD:
        os.environ.setdefault(name, "1")
    from .gui import run

    raise SystemExit(run())
