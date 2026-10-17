"""Test-wide setup, run before any test module imports NumPy."""

import os

from vib_tutorial import ONE_THREAD

# BLAS on one thread, as the app runs it (vib_tutorial.main): the matrices are small, and
# OpenBLAS's pool makes their solves several times slower (worst on Windows CI).
for name in ONE_THREAD:
    os.environ.setdefault(name, "1")
