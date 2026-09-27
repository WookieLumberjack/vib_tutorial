# vib_tutorial

Real-time teaching app for an N-mass spring-damper chain (PySide6 + pyqtgraph + NumPy/SciPy), managed with uv.

- Run: `uv run vib-tutorial`. Test: `uv run pytest` (GUI smoke test runs headless via `QT_QPA_PLATFORM=offscreen`).
- `src/vib_tutorial/core/` is pure NumPy/SciPy with no Qt imports; keep it that way so it stays testable and scriptable.
- `src/vib_tutorial/gui/` holds the Qt widgets. `main_window.py` owns the timer loop and wires the panels to `Simulator`.
- Units are SI throughout (kg, N/m, N·s/m, m, N, Hz in UI / rad/s internally where named `omega`).
- The simulator uses an exact FOH matrix-exponential discretization. Don't replace it with an explicit integrator, because stiff user inputs must stay stable.
- `core/measurement.py` (virtual modal test) uses the same FOH update, diagonalized into one `lfilter` per eigenvalue (`ChainResponse`) so long records are fast; it steps directly when the eigenvectors are singular.
- `Simulator.advance` caps the number of steps per call (`MAX_STEPS_PER_ADVANCE`). Loop over it for long runs in tests.
- To check UI changes, grab a screenshot offscreen: run the script with `QT_QPA_PLATFORM=offscreen` (only the tests set it; without it the window opens on the real display, picks up a dark desktop scheme, and gets resized by the compositor). Create the app with `gui.make_app()` as `gui.run()` does, which sets pyqtgraph's white background (the default is black) and Qt's light colour scheme. Then build `MainWindow`, resize it to 1700×900 (the README screenshots' size), run the event loop briefly, and `window.grab().save(path)`.
