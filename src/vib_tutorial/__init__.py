"""Interactive teaching tool for lumped mass-spring-damper vibration."""

import time

# When the interpreter reached this package, for the startup timings --smoke-test records.
STARTED = time.perf_counter()


def main() -> None:
    from .gui import run

    raise SystemExit(run())
