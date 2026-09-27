# vib_tutorial

Real-time teaching app for an N-mass spring-damper chain (PySide6 + pyqtgraph + NumPy/SciPy), managed with uv.

- Run: `uv run vib-tutorial`. Test: `uv run pytest` (GUI smoke test runs headless via `QT_QPA_PLATFORM=offscreen`).
- `src/vib_tutorial/core/` is pure NumPy/SciPy with no Qt imports; keep it that way so it stays testable and scriptable.
- `src/vib_tutorial/gui/` holds the Qt widgets. `main_window.py` owns the timer loop and wires the panels to `Simulator`.
- Units are SI throughout (kg, N/m, N·s/m, m, N, Hz in UI / rad/s internally where named `omega`).
- The simulator uses an exact FOH matrix-exponential discretization. Don't replace it with an explicit integrator, because stiff user inputs must stay stable.
- `core/measurement.py` (virtual modal test) uses the same FOH update, diagonalized into one `lfilter` per eigenvalue (`ChainResponse`) so long records are fast; it steps directly when the eigenvectors are singular.
- `core/back_expansion.py` recovers the interior DOFs from the coupled CMS solution; its enhanced recovery adds the interior states to the reduced model's state vector (`RecoveryResponse`), so it shares the same exact FOH step.
- `Simulator.advance` caps the number of steps per call (`MAX_STEPS_PER_ADVANCE`). Loop over it for long runs in tests.
- Colours come from the theme in `gui/style.py`, read at draw time through `colors` (`colors.mass[i]`, `colors.force`, `colors.muted`). Don't hard-code a colour: it has to be legible in every theme (`style.contrast` gives the WCAG ratio; aim for 3:1 for curves, 4.5:1 for text). A widget that keeps pens or brushes across redraws needs an `apply_theme()`, called from `MainWindow._restyle`. Mute secondary labels with `theming.mute`, not a stylesheet colour.
- To check UI changes, grab a screenshot offscreen: run the script with `QT_QPA_PLATFORM=offscreen` (only the tests set it; without it the window opens on the real display and gets resized by the compositor). Create the app with `gui.make_app("Light")` (or another theme name); `gui.run()` uses the theme last picked, which is stored in the user's settings. This sets Qt's palette and pyqtgraph's background (the default is black). Then build `MainWindow`, resize it to 1700×900 (the README screenshots' size), run the event loop briefly, and `window.grab().save(path)`. Check a change in Light and Dark at least; `MainWindow.set_theme(name)` switches live.
- pyqtgraph views left in a scene until interpreter exit can segfault in PySide's teardown (exit 139 *after* pytest prints "passed"). `gui.close_graphics(widget)` tears them down; `gui.run()` and the GUI tests' autouse fixture call it. Check the exit code, not just the summary line.
