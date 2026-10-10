"""Regenerate the README screenshots in docs/images, each at 1700x900 in the Light theme.

    uv run python scripts/readme_screenshots.py              # all of them
    uv run python scripts/readme_screenshots.py chirp pluck  # just these (the function names below)

Each shot builds a fresh MainWindow, sets up the state its README caption
describes and saves window.grab(). The window's frame timer is stopped and the
script advances simulated time itself, so every image is taken at a fixed
moment and reruns give the same pictures. Run it after a UI change, then check
that the captions' numbers still match the images.

It runs offscreen with a temporary settings folder, so it neither opens a
window nor touches the user's saved theme or playback speed. It drives the
widgets directly (some private attributes too, such as MainWindow._tick), so a
UI refactor may need a matching change here.
"""

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["XDG_CONFIG_HOME"] = tempfile.mkdtemp(prefix="vib-screenshots-")  # before Qt reads settings

from PySide6 import QtCore
from PySide6.QtTest import QTest

from vib_tutorial import gui
from vib_tutorial.core import ForceKind
from vib_tutorial.gui.main_window import MainWindow

OUT = Path(__file__).resolve().parent.parent / "docs" / "images"
app = gui.make_app("Light")


def window(theme="Light"):
    w = MainWindow()
    # The script drives time (run), so every shot is at a known moment. Stop the timer before
    # the window shows: frames run while it settles would advance History.offset by a
    # wall-clock-dependent count, and the strip chart's decimation buckets are aligned to it.
    w._timer.stop()
    w.set_theme(theme)
    w.resize(1700, 900)
    w.show()
    QTest.qWait(200)
    running, w.running = w.running, False
    w._tick()  # draw the chain (its masses' positions) without advancing time
    w.running = running
    w.controls.reset.click()
    return w


def run(w, seconds):
    """Advance the Simulation page by `seconds` of simulated time (at the chosen speed) through its tick."""
    speed = w.controls.speed_factor
    for _ in range(round(seconds / (0.1 * speed))):
        w._last_wall = w._clock.elapsed() / 1000.0 - 0.1
        w._tick()
    app.processEvents()


def settle(ms=300):
    QTest.qWait(ms)


def save(w, name):
    settle()
    w.grab().save(str(OUT / f"{name}.png"))
    print("saved", name)
    done(w)


def done(w):
    w._timer.stop()
    w.close()
    gui.close_graphics(w)
    w.deleteLater()
    app.processEvents()


def set_c1(w, c):
    w.params.rows[0][3].setValue(c)


def kind(panel, k):
    panel.kind.setCurrentIndex(panel.kind.findData(k))


def tune(panel, mode):
    """Pick mode `mode` (1-based) in a 'Tune to…' combo."""
    panel.tune.activated.emit(mode)


def coords(w, text):
    w.coords_combo.setCurrentIndex(w.coords_combo.findText(text, QtCore.Qt.MatchFlag.MatchStartsWith))


def energy_view(w, by_mode):
    w.energy.view.setCurrentIndex(1 if by_mode else 0)


def release(w, row):
    w.table.selectRow(row)
    w._release_mode()


def method(w, state_space):
    w.method_combo.setCurrentIndex(1 if state_space else 0)


# ------------------------------------------------------------------ Simulation page
def main_window():
    w = window()
    tune(w.force_panel, 2)
    w.force_panel.button.click()
    run(w, 6.0)
    w.table.selectRow(1)
    save(w, "main_window")


def dark_theme():
    w = window("Dark")
    set_c1(w, 15.0)
    method(w, True)
    coords(w, "Modal")
    tune(w.force_panel, 2)
    w.force_panel.button.click()
    run(w, 11.0)
    w.table.selectRow(2)
    save(w, "dark_theme")


def pluck():
    from vib_tutorial.gui.animation import MAX_DRAG, SPACING

    w = window()
    energy_view(w, True)
    w.controls.window.setValue(4.0)
    view = w.chain

    def pixel(x):
        return view.mapFromScene(view.plotItem.vb.mapViewToScene(QtCore.QPointF(x, 0.0)))

    left = QtCore.Qt.MouseButton.LeftButton
    QTest.mousePress(view.viewport(), left, pos=pixel(view._centers[1]))
    QTest.mouseMove(view.viewport(), pixel(2 * SPACING + 0.8 * MAX_DRAG))
    QTest.mouseRelease(view.viewport(), left, pos=pixel(2 * SPACING + 0.8 * MAX_DRAG))
    run(w, 3.1)
    QTest.mousePress(view.viewport(), left, pos=pixel(view._centers[1]))
    QTest.mouseMove(view.viewport(), pixel(2 * SPACING - 0.5 * MAX_DRAG))
    for _ in range(5):
        w._tick()
    save(w, "pluck")


def chirp():
    w = window()
    p = w.force_panel
    kind(p, ForceKind.CHIRP)
    p.sweep_start.setValue(2.0)
    p.sweep_end.setValue(7.0)
    p.sweep_time.setValue(40.0)
    w.controls.window.setValue(40.0)
    coords(w, "Modal")
    w.tabs.setCurrentWidget(w.frf_tab)
    p.button.click()
    run(w, 36.3)
    save(w, "chirp")


def base_excitation():
    w = window()
    p = w.force_panel
    p.input.setCurrentIndex(1)
    p.base_amplitude.setValue(5.0)
    p.freq.setValue(1.6)
    w.controls.window.setValue(4.0)
    w.tabs.setCurrentWidget(w.frf_tab)
    p.button.click()
    run(w, 30.3)
    save(w, "base_excitation")


def tuned_mass_damper():
    w = window()
    w.params.load_preset(next(pr for pr in w.params.presets if "Den Hartog" in pr.name))
    settle()
    w.force_panel.button.click()
    run(w, 12.7)
    save(w, "tuned_mass_damper")


def classical_release_nonproportional():
    w = window()
    set_c1(w, 15.0)
    w.controls.window.setValue(2.05)
    release(w, 2)
    run(w, 1.96)
    save(w, "classical_release_nonproportional")


def state_space_release():
    w = window()
    set_c1(w, 15.0)
    method(w, True)
    w.controls.window.setValue(2.12)
    release(w, 4)
    run(w, 1.97)
    save(w, "state_space_release")


def modal_coordinates():
    w = window()
    set_c1(w, 15.0)
    coords(w, "Modal")
    w.controls.window.setValue(2.05)
    release(w, 2)
    run(w, 2.07)
    save(w, "modal_coordinates")


def energy_by_mode():
    w = window()
    set_c1(w, 15.0)
    coords(w, "Modal")
    energy_view(w, True)
    w.controls.window.setValue(2.05)
    release(w, 2)
    run(w, 0.8)
    save(w, "energy_by_mode")


def energy_history():
    w = window()
    coords(w, "Energy")
    w.force_panel.freq.setValue(1.0)
    w.force_panel.button.click()
    run(w, 8.3)
    w.table.selectRow(0)
    save(w, "energy_history")


def element_forces():
    w = window()
    kind(w.force_panel, ForceKind.STEP)
    coords(w, "Element forces")
    w.force_panel.button.click()
    run(w, 9.3)
    save(w, "element_forces")


def frequency_response():
    w = window()
    w.tabs.setCurrentWidget(w.frf_tab)
    tune(w.force_panel, 2)
    settle(500)
    w.frf_plot.grab().save(str(OUT / "frequency_response.png"))
    print("saved frequency_response")
    done(w)


# ------------------------------------------------------------------ other pages
def frf_matrix():
    w = window()
    page = w.frf_page
    w.pages.setCurrentWidget(page)
    settle()
    page.mode_checks[3].setChecked(False)
    page._on_cell(0, 3, False)
    save(w, "frf_matrix")


def modal_coupling():
    w = window()
    for i in (2, 3):
        w.params.rows[i][1].setValue(0.1)
        w.params.rows[i][2].setValue(40.0)
        w.params.rows[i][3].setValue(0.2)
    w.pages.setCurrentWidget(w.coupling_page)
    save(w, "modal_coupling")


def cms(w, n, interfaces, kept, method_name="Craig"):
    w.params.dof.setValue(n)
    page = w.cms_page
    w.pages.setCurrentWidget(page)
    settle()
    page.method.setCurrentIndex(page.method.findText(method_name, QtCore.Qt.MatchFlag.MatchStartsWith))
    page.set_interfaces(interfaces)
    for spin, k in zip(page.kept, kept):
        spin.setValue(k)
    settle()
    return page


def substructuring_three():
    w = window()
    page = cms(w, 8, [2, 5], [1, 1, 1])
    page.tabs.setCurrentWidget(page.basis)
    page.table.selectRow(3)
    save(w, "substructuring_three")


def drive_cms(view, w, seconds):
    view._timer.stop()
    view.reset()
    view.kind.setCurrentIndex(view.kind.findData(ForceKind.HARMONIC))
    view.freq.setValue(w.modal.modes[3].fn_hz)
    view.button.click()
    for _ in range(round(seconds / 0.05)):
        view.step(0.05)
    view._timer.stop()


def substructuring_time():
    w = window()
    page = cms(w, 8, [3], [1, 1])
    page.tabs.setCurrentWidget(page.time)
    settle()
    drive_cms(page.time, w, 9.36)
    save(w, "substructuring_time")


def substructuring_recovery():
    w = window()
    page = cms(w, 8, [3], [2, 2])
    page.tabs.setCurrentWidget(page.recovery)
    settle()
    r = page.recovery
    r.mass_box.setCurrentIndex(r.mass_box.findText("m2", QtCore.Qt.MatchFlag.MatchStartsWith))
    r.spring_box.setCurrentIndex(r.spring_box.findText("Spring k2", QtCore.Qt.MatchFlag.MatchStartsWith))
    drive_cms(r, w, 10.06)
    save(w, "substructuring_recovery")


def substructuring_compare():
    w = window()
    page = cms(w, 8, [3], [1, 1], "Rubin")
    page.tabs.setCurrentWidget(page.compare)
    page.table.selectRow(2)
    save(w, "substructuring_compare")


def test_page(block, force_noise, response_noise, exp_end, tab, poles=None):
    w = window()
    p = w.test_page
    w.pages.setCurrentWidget(p)
    settle()
    p.block.setCurrentIndex(p.block.findText(str(block)))
    p.force_noise.setValue(force_noise)
    p.response_noise.setValue(response_noise)
    p.exp_end.setValue(exp_end)
    # Random noise: take the first seed whose automatic pick finds `poles` modes, as the caption says.
    for seed in range(40):
        p.seed = seed
        p.restart()
        while not p.acq.done:
            p._measure_some()
        app.processEvents()
        if poles is None or len(p.poles) == poles:
            break
    else:
        print(f"  no seed gave {poles} poles; the image shows {len(p.poles)}")
    p.tabs.setCurrentWidget(getattr(p, tab))
    return w


def virtual_modal_test():
    save(test_page(256, 1.0, 1.0, 2.0, "check"), "virtual_modal_test")


def modal_extraction():
    save(test_page(1024, 0.0, 1.0, 100.0, "stab_plot", poles=3), "modal_extraction")


# ------------------------------------------------------------------ Jeffcott rotor
def jeffcott_rotor():
    w = window()
    page = w.rotor_page
    w.pages.setCurrentWidget(page)
    settle()
    page._timer.stop()
    page.running = False  # the script drives time, as on the Simulation page
    page.ramp.setValue(5000.0)
    page.set_target_rpm(1400.0)
    for _ in range(150):
        page.step(0.02)
    save(w, "jeffcott_rotor")


def jeffcott_run_up():
    w = window()
    page = w.rotor_page
    w.pages.setCurrentWidget(page)
    settle()
    page._timer.stop()
    page.running = False
    page.sweep_from.setValue(800.0)
    page.sweep_to.setValue(2400.0)
    for rate in (500.0, 5000.0):  # held side by side
        page.ramp.setValue(rate)
        page.start_sweep()
        while page.run_up is not None:
            page.step(0.02)
    save(w, "jeffcott_run_up")


def jeffcott_stability():
    w = window()
    page = w.rotor_page
    w.pages.setCurrentWidget(page)
    settle()
    page._timer.stop()
    page.running = False
    page.internal_damping.setValue(20.0)
    page._refresh_analysis()
    page.ramp.setValue(5000.0)
    page.set_target_rpm(5000.0)
    for _ in range(60):
        page.step(0.02)
    page.tap()
    for _ in range(60):
        page.step(0.02)
    page.tabs.setCurrentWidget(page.spectrum.parentWidget())
    page._draw()
    save(w, "jeffcott_stability")


SHOTS = [main_window, dark_theme, pluck, chirp, base_excitation, tuned_mass_damper,
         classical_release_nonproportional, state_space_release, modal_coordinates, energy_by_mode,
         energy_history, element_forces, frequency_response, frf_matrix, modal_coupling,
         substructuring_three, substructuring_time, substructuring_recovery, substructuring_compare,
         virtual_modal_test, modal_extraction, jeffcott_rotor, jeffcott_run_up,
         jeffcott_stability]


def main(names: list[str]) -> None:
    unknown = set(names) - {shot.__name__ for shot in SHOTS}
    if unknown:
        sys.exit(f"unknown shot(s): {', '.join(sorted(unknown))}; choose from: "
                 + ", ".join(shot.__name__ for shot in SHOTS))
    for shot in SHOTS:
        if not names or shot.__name__ in names:
            shot()


if __name__ == "__main__":
    main(sys.argv[1:])
