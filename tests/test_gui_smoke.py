"""Headless smoke test: build the window and exercise the main interactions."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from vib_tutorial.core import ForceKind  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_window_interactions(app):
    from vib_tutorial.gui.main_window import MainWindow

    w = MainWindow()
    w.force_panel.button.click()  # harmonic force on
    assert w.force.on
    w._sim_target = 1.0
    w._tick()
    assert w.sim.t > 0.1 and w.history.size > 0  # one frame advances at most ~0.3 s

    # Live parameter edit keeps running and updates the modal table.
    w.params.rows[0][2].setValue(800.0)
    assert w.sim.system.stiffness[0] == 800.0
    assert w.table.item(0, 1).text() != ""

    # Fit the plot window to N cycles of mode 2; changing N re-fits.
    w.controls.fit_window.activated.emit(2)
    assert w.controls.window.value() == pytest.approx(10 / w.modal.modes[1].fn_hz, abs=0.005)
    assert w.controls.fit_window.currentIndex() == 0
    w.controls.cycles.setValue(4)
    assert w.controls.window.value() == pytest.approx(4 / w.modal.modes[1].fn_hz, abs=0.005)

    # Non-proportional damping: note explains coupling; its link opens the Background tab.
    w.params.rows[0][3].setValue(15.0)
    assert "non-proportional" in w.modal_note.text()
    w.modal_note.linkActivated.emit("#background")
    assert w.tabs.currentWidget() is w.background

    # Change number of masses.
    w.params.dof.setValue(6)
    assert w.sim.system.n == 6 and w.table.rowCount() == 6
    w._tick()

    # Pulse and mode release.
    w.force_panel.kind.setCurrentIndex(list(ForceKind).index(ForceKind.PULSE))
    w.force_panel.button.click()
    assert w.force.active
    w.table.selectRow(2)
    w._release_mode()
    assert abs(w.sim.displacement).max() == pytest.approx(0.02)
    # Releasing a mode fits the plot window to it (auto-scale on by default)...
    assert w.controls.window.value() == pytest.approx(4 / w.modal.modes[2].fn_hz, abs=0.005)
    # ...but not when auto-scale is off.
    w.controls.auto_scale.setChecked(False)
    w.table.selectRow(5)
    w._release_mode()
    assert w.controls.window.value() == pytest.approx(4 / w.modal.modes[2].fn_hz, abs=0.005)
    w.close()


def test_state_space_method(app):
    import numpy as np

    from vib_tutorial.gui.main_window import MainWindow
    from vib_tutorial.gui.modes import Method

    w = MainWindow()
    w.params.rows[0][3].setValue(15.0)  # non-proportional
    n = w.sim.system.n
    w.method_combo.setCurrentIndex(list(Method).index(Method.STATE_SPACE))
    assert w.table.rowCount() == 2 * n
    assert w.phasor_plot.isVisibleTo(w) and "2N" in w.modal_note.text()
    assert w.table.item(1, 0).text() == "2 = λ1*"
    # Combos list one entry per conjugate pair, at the damped frequency.
    assert w.force_panel.tune.count() == 1 + n
    assert w.force_panel.tune.itemData(1) == pytest.approx(w.modal.complex_modes[0].fd_hz)

    # Releasing a complex mode sets displacement Re(psi) and velocity Re(lambda psi);
    # its conjugate releases the same motion.
    w.table.selectRow(2)
    w._release_mode()
    m = w.modal.complex_modes[2]
    amp = w.release_amp.value() * 1e-3
    np.testing.assert_allclose(w.sim.state, amp * m.state_vector.real)
    assert w.controls.window.value() == pytest.approx(10 / m.fd_hz, abs=0.005)
    state = w.sim.state.copy()
    w.table.selectRow(3)
    w._release_mode()
    np.testing.assert_allclose(w.sim.state, state)
    assert w.phasor_plot.table.rowCount() == n

    w.animate_modes.setChecked(True)
    w._tick()

    # Overdamped and rigid-body systems produce real eigenvalues; they must display.
    w.params.rows[0][3].setValue(200.0)
    w.params.rows[0][2].setValue(0.0)
    assert w.table.rowCount() == 2 * n
    assert any("τ" in w.table.item(r, 4).text() for r in range(2 * n))
    w.table.selectRow(0)
    w._release_mode()
    w._tick()

    # Switching back restores the classical table.
    w.method_combo.setCurrentIndex(list(Method).index(Method.CLASSICAL))
    assert w.table.rowCount() == n and not w.phasor_plot.isVisibleTo(w)
    assert w.table.horizontalHeaderItem(2).text() == "ζ modal"
    w.close()


def test_damper_stays_connected():
    import numpy as np

    from vib_tutorial.gui.animation import damper_path

    rest = 0.6
    for x1 in np.linspace(-0.2, 2.0, 23):  # crushed ... stretched far past the cylinder
        xs, _ = damper_path(0.0, x1, 0.0, rest)
        segments = np.split(xs, np.where(np.isnan(xs))[0])
        piston_x = segments[2][~np.isnan(segments[2])][0]
        rod = segments[3][~np.isnan(segments[3])]
        assert rod[0] == piston_x and rod[-1] == x1  # rod joins piston to the mass


def test_decimation_keeps_extremes():
    import numpy as np

    from vib_tutorial.gui.plots import decimation_index

    t = np.linspace(0, 10, 50_000)
    y = np.column_stack([np.sin(40 * t), np.cos(7 * t), np.where(np.abs(t - 5) < 0.01, 1.0, 0.0)])
    idx = decimation_index(y, 3000)
    assert idx.size <= 3000
    np.testing.assert_array_equal(y[idx].max(axis=0), y.max(axis=0))
    np.testing.assert_array_equal(y[idx].min(axis=0), y.min(axis=0))


def test_auto_scale_gain_is_bounded(app):
    import numpy as np

    from vib_tutorial.gui.animation import MAX_SWING, MIN_AUTO_PEAK, ChainView

    view = ChainView()
    view.set_masses(np.ones(4))
    for _ in range(2000):  # vibration decayed to numerical noise
        view.update_state(np.full(4, 1e-20), 1e-20, 0.0, 0, 1.0)
    assert view.gain <= MAX_SWING / MIN_AUTO_PEAK * (1 + 1e-9)


def test_chain_view_fits_all_masses(app):
    import numpy as np

    from vib_tutorial.gui.animation import SPACING, ChainView

    view = ChainView()
    view.resize(700, 300)  # wide-but-short, like the app's animation pane
    for n in (4, 8, 2, 6):
        view.set_masses(np.ones(n))
        (x0, x1), _ = view.viewRange()
        assert x0 <= 0 and x1 >= n * SPACING + 0.2, f"n={n}: x-range {x0:.2f}..{x1:.2f} crops the chain"


def test_substructuring_page(app):
    from vib_tutorial.gui.main_window import MainWindow

    w = MainWindow()
    page = w.cms_page
    w.show()
    w.pages.setCurrentWidget(page)
    app.processEvents()

    # N set on this page drives the shared chain; the interface starts in the middle.
    page.dof.setValue(8)
    assert w.sim.system.n == 8 and w.params.dof.value() == 8
    assert page.interface.currentText() == "m4"
    assert page.model.labels == ["q_A1", "q_B1", "x4", "x8"]
    assert page.table.rowCount() == 8 and page.table.item(4, 3).text() == "not in model"
    assert page.table.horizontalHeaderItem(5).text() == "ζ CB" and page.table.item(0, 5).text() != "—"
    assert page.component_table.rowCount() == 6 and page.component_table.item(0, 0).text() == "A1"
    assert len(page.basis.plots.ci.items) == 4  # one shape per reduced coordinate
    page.component_table.selectRow(3)  # B1 alone, overlaid on the coupled mode it becomes
    assert page.plots.component.substructure == "B" and page.table.currentRow() == page.plots.component.closest - 1
    page.table.selectRow(0)
    assert page.plots.component is None
    assert "4 DOFs" in page.summary.text()
    text = page.matrices.toPlainText()
    assert "Assemble the reduced model" in text and "Damping in the reduced model" in text

    # Guyan keeps only the boundary; "All modes" is exact.
    page.guyan_button.click()
    assert page.model.n_red == 2 and "Guyan" in page.summary.text()
    page.exact_button.click()
    assert page.model.n_red == 8
    assert all(c.error == pytest.approx(0.0, abs=1e-9) for c in page.comparisons)

    # Interface at the last possible mass: B has no interior DOFs, its spin is disabled.
    page.interface.setCurrentIndex(page.interface.count() - 1)
    assert not page.kept[1].isEnabled() and page.model.substructures[1].ni == 0

    # A parameter edit on the Simulation page reaches this page; a floating interior is reported.
    w.params.rows[0][2].setValue(0.0)
    w.params.rows[1][2].setValue(0.0)
    assert page.model is None and "K_ii" in page.summary.text()
    w.params.rows[1][2].setValue(400.0)
    assert page.model is not None

    # The simulation keeps running behind this page without drawing.
    w._sim_target = 0.5
    w._tick()
    assert w.sim.t > 0.1
    w.close()
