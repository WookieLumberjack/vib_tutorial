"""Headless smoke test: build the window and exercise the main interactions."""

import os
import tempfile

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
from PySide6 import QtCore  # noqa: E402

# Keep the theme the tests pick out of the user's own settings.
QtCore.QSettings.setPath(QtCore.QSettings.Format.NativeFormat, QtCore.QSettings.Scope.UserScope, tempfile.mkdtemp())

from vib_tutorial.core import ForceKind  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def _close_graphics(app):
    """Tear down each test's plots now. Signal closures keep the windows alive until interpreter
    exit, where PySide deleting pyqtgraph items still in a scene can segfault.

    (Don't deleteLater() the windows here: collecting pyqtgraph's Python objects after their C++
    side is gone segfaults too.)"""
    from shiboken6 import isValid

    from vib_tutorial.gui import close_graphics

    before = set(QtWidgets.QApplication.topLevelWidgets())
    yield
    for w in QtWidgets.QApplication.topLevelWidgets():
        if w not in before and isValid(w):
            close_graphics(w)


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


def test_secondary_pages_build_when_opened(app):
    from vib_tutorial.gui.main_window import LAZY_PAGES, MainWindow
    from vib_tutorial.gui.modal_coupling import ModalCouplingPage

    w = MainWindow()
    assert not w._built  # startup builds only the Simulation page
    titles = [w.pages.tabText(i) for i in range(w.pages.count())]
    w.params.dof.setValue(5)  # a page built later gets the current system

    w.pages.setCurrentIndex(2)  # as a click on the tab does
    page = w.pages.currentWidget()
    assert isinstance(page, ModalCouplingPage) and page is w.coupling_page
    assert list(w._built) == ["coupling_page"]
    assert page.system.n == 5
    assert [w.pages.tabText(i) for i in range(w.pages.count())] == titles

    for name, _ in LAZY_PAGES:  # reading one builds it, without switching to it
        assert w.pages.indexOf(getattr(w, name)) >= 0
    assert w.pages.currentWidget() is page and w.pages.count() == 1 + len(LAZY_PAGES)
    page.edit_parameters.emit()
    assert w.pages.currentWidget() is w.sim_page


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
    assert page.interfaces == [] and len(page.cut_buttons) == 7  # starts as one substructure
    assert page.model.labels == ["q_A1", "x8"]
    page.cut_buttons[3].click()  # cut at m4
    assert page.interfaces == [3] and page.cut_buttons[3].isChecked()
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
    page.set_interfaces([len(page.cut_buttons) - 1])
    assert not page.kept[1].isEnabled() and page.model.substructures[1].ni == 0

    # A parameter edit on the Simulation page reaches this page; a floating interior is reported.
    w.params.rows[0][2].setValue(0.0)
    w.params.rows[1][2].setValue(0.0)
    assert page.model is None and "K_ii" in page.summary.text()
    w.params.rows[1][2].setValue(400.0)
    assert page.model is not None

    # The simulation pauses behind this page and resumes on its own.
    t0 = w.sim.t
    w._sim_target = t0 + 0.5
    w._tick()
    assert w.sim.t == t0
    w.pages.setCurrentWidget(w.sim_page)
    w._tick()
    assert w.sim.t > t0 + 0.1
    w.close()


def test_substructuring_time_response(app):
    from vib_tutorial.gui.main_window import MainWindow

    w = MainWindow()
    page = w.cms_page
    w.show()
    w.pages.setCurrentWidget(page)
    page.dof.setValue(6)
    page.set_interfaces([2])
    page.tabs.setCurrentWidget(page.time)
    app.processEvents()
    tv = page.time
    assert tv._timer.isActive()  # runs while the tab shows

    # A step at the tip: both models move, and 1 mode each leaves a reduction error.
    tv.button.click()
    assert tv.force.on and tv.button.text() == "Stop force"
    for _ in range(5):
        tv.step(0.2)
    assert tv.sim.t >= 0.99  # (the tab's own timer may add a frame or two)
    t, y = tv.history.window(tv.window.value())
    assert t.size and np.abs(y[:, 0]).max() > 0.01  # the tip moves (m)
    assert 0 < np.abs(y[:, 2] - y[:, 0]).max() < 0.2 * np.abs(y[:, 0]).max()
    assert "tip error" in tv.error_note.text()
    assert "4 DOFs" in tv.red_view.plotItem.titleLabel.text

    # A theme change keeps the motion; a new model restarts from rest with the force still on.
    before = tv.sim.t
    w.set_theme("Dark")
    assert tv.sim.t == before
    page.exact_button.click()
    assert tv.sim.t == 0.0 and tv.force.on and tv.history.size == 0
    for _ in range(3):
        tv.step(0.2)
    t, y = tv.history.window(10.0)
    np.testing.assert_allclose(y[:, 2:], y[:, :2], atol=1e-9)  # nothing reduced: exact

    # Harmonic, tuned to a mode; then a pulse.
    tv.kind.setCurrentIndex(1)
    assert tv.freq.isVisibleTo(tv) and not tv.force.on
    tv.tune.activated.emit(2)
    assert tv.freq.value() == pytest.approx(w.modal.modes[1].fn_hz, abs=1e-3)
    tv.kind.setCurrentIndex(2)
    tv.button.click()
    assert tv.force.active
    tv.step(0.2)
    tv.reset_button.click()
    assert tv.sim.t == 0.0 and not tv.force.active

    # One mass is one substructure of just the tip: exact, 1 DOF.
    page.dof.setValue(1)
    assert page.interfaces == [] and page.model.labels == ["x1"] and tv.sim is not None
    # No model at all (Craig-Bampton with a floating interior): the tab says why.
    page.dof.setValue(4)
    w.params.rows[0][2].setValue(0.0)
    w.params.rows[1][2].setValue(0.0)
    assert tv.sim is None and "K_ii" in tv.header.text()
    w.pages.setCurrentWidget(w.sim_page)
    assert not tv._timer.isActive()
    w.set_theme("Light")
    w.close()


def test_substructuring_several_interfaces(app):
    import pyqtgraph as pg

    from vib_tutorial.gui.main_window import MainWindow

    w = MainWindow()
    page = w.cms_page
    w.show()
    w.pages.setCurrentWidget(page)
    page.dof.setValue(8)
    page.set_interfaces([3])
    app.processEvents()

    # Click m2 and m6 on as well: four substructures, one "modes kept" row each.
    page.cut_buttons[1].click()
    page.cut_buttons[5].click()
    assert page.interfaces == [1, 3, 5]
    assert [sub.name for sub in page.model.substructures] == ["A", "B", "C", "D"]
    assert len(page.kept) == 4 and page.kept_labels[3].text().endswith("Modes kept in D:</b>")
    assert page.model.labels == ["q_A1", "q_B1", "q_C1", "q_D1", "x2", "x4", "x6", "x8"]
    assert page.kept[3].maximum() == 1  # D is m7 (interior) and m8
    assert "A: keep 1 of 1" in [i.textItem.toPlainText() for i in page.schematic._items
                                 if isinstance(i, pg.TextItem)]
    assert page.compare.current["rubin"] is not None and page.compare.current["rubin"].n_red == 8

    # Fewer modes in C only; the other rows keep theirs.
    page.kept[2].setValue(0)
    assert [sub.n_kept for sub in page.model.substructures] == [1, 1, 0, 1]
    page.guyan_button.click()
    assert page.model.n_red == 4 and "Guyan" in page.summary.text()
    page.exact_button.click()
    assert all(c.error == pytest.approx(0.0, abs=1e-9) for c in page.comparisons)

    # The component table names every substructure; the FRF shows the tip and each interface.
    names = {page.component_table.item(r, 0).text()[0] for r in range(page.component_table.rowCount())}
    assert names == {"A", "B", "C", "D"}
    assert len(page.plots.frf_legend.items) == 8  # 4 masses, true and reduced

    # The time response follows the tip and all three interfaces.
    page.tabs.setCurrentWidget(page.time)
    page.guyan_button.click()
    tv = page.time
    assert tv._dofs == [7, 1, 3, 5] and len(tv.e_curves) == 4
    tv.button.click()
    tv.step(0.3)
    t, y = tv.history.window(10.0)
    assert y.shape[1] == 8 and "interfaces" in tv.error_note.text()

    # No cut: the whole chain is one substructure, with the tip its only boundary DOF.
    page.set_interfaces([2])
    assert len(page.kept) == 2
    page.cut_buttons[2].click()
    assert page.interfaces == [] and not page.cut_buttons[2].isChecked()
    assert len(page.kept) == 1 and page.kept[0].maximum() == 7
    assert [sub.name for sub in page.model.substructures] == ["A"]
    assert "one substructure" in page.summary.text()
    assert "nothing to join" in page.matrices.toPlainText()
    page.guyan_button.click()
    assert page.model.labels == ["x8"] and page.comparisons[0].error > 0.05  # Guyan onto the tip alone
    assert tv._dofs == [7] and len(tv.e_curves) == 1
    tv.step(0.3)
    assert "interface" not in tv.error_note.text()
    page.method.setCurrentIndex(2)  # MacNeal keeps at least one mode, or it would have no mass
    assert page.kept[0].minimum() == 1 and page.model is not None
    page.method.setCurrentIndex(0)
    with pytest.raises(ValueError):
        page.set_interfaces([7])

    # A new N starts again from one cut in the middle.
    page.dof.setValue(6)
    assert page.interfaces == [] and len(page.cut_buttons) == 5
    w.close()


def test_substructuring_free_interface(app):
    from vib_tutorial.gui.main_window import MainWindow

    w = MainWindow()
    page = w.cms_page
    w.show()
    w.pages.setCurrentWidget(page)
    page.dof.setValue(8)
    page.set_interfaces([3])
    app.processEvents()
    assert page.compare.table.rowCount() == 8
    cb_error = page.comparisons[0].error

    # Rubin: same size, B's rigid-body mode always kept, more accurate on mode 1.
    page.method.setCurrentIndex(page.method.findData("rubin"))
    assert page.model.method == "rubin" and page.model.labels == ["q_A1", "q_B1", "x4", "x8"]
    assert page.kept[1].minimum() == 1 and page.model.substructures[1].n_rigid == 1
    assert page.table.horizontalHeaderItem(5).text() == "ζ Rubin"
    assert 0 < page.comparisons[0].error < cb_error
    assert page.component_table.rowCount() == 9 and page.component_table.item(4, 1).text() == "0 (rigid)"
    assert "residual attachment" in page.basis.plots.ci.getItem(1, 0).titleLabel.text
    text = page.matrices.toPlainText()
    assert "Residual flexibility" in text and "split half and half" in text
    page.guyan_button.click()
    assert [s.value() for s in page.kept] == [0, 1] and page.guyan_button.text() == "Fewest modes"
    page.exact_button.click()
    assert all(c.error == pytest.approx(0.0, abs=1e-9) for c in page.comparisons)

    # MacNeal: massless boundary, so fewer modes than coordinates, and never exact.
    page.method.setCurrentIndex(page.method.findData("macneal"))
    assert page.model.n_red == 8 and page.model.omegas.size == 6 and "6 modes" in page.summary.text()
    assert page.exact_button.text() == "All modes"
    assert max(abs(c.error) for c in page.comparisons if c.error is not None) > 0.01

    # Compare methods: every method in the table, and a convergence curve for the selected mode.
    page.table.selectRow(2)
    assert all("%" in page.compare.table.item(2, col).text() for col in (2, 3, 4))
    assert len(page.compare.plot.getPlotItem().listDataItems()) >= 3
    assert "Mode 3" in page.compare.plot.getPlotItem().titleLabel.text

    # A floating interior (k1 = k2 = 0): Craig-Bampton fails, the free-interface methods do not.
    w.params.rows[0][2].setValue(0.0)
    w.params.rows[1][2].setValue(0.0)
    assert page.model is not None and page.model.substructures[0].n_rigid == 2
    assert page.compare.current["craig-bampton"] is None and "Craig–Bampton" in page.compare.header.text()
    page.method.setCurrentIndex(page.method.findData("craig-bampton"))
    assert page.model is None and "K_ii" in page.summary.text()
    w.close()


def test_frf_matrix_page(app):
    import numpy as np

    from vib_tutorial.gui.frf_matrix import Expansion, Quantity
    from vib_tutorial.gui.main_window import MainWindow

    w = MainWindow()
    page = w.frf_page
    w.show()
    w.pages.setCurrentWidget(page)
    app.processEvents()
    n = w.sim.system.n
    assert len(page.grid.cells) == n and len(page.mode_checks) == n
    assert (page.output, page.inputs) == (n - 1, {n - 1})
    assert "none" in page.detail.header.text()  # all modes: the sum is the full solution
    assert "Shaded" not in page.detail.header.text() and "0% (exact)" in page.detail.header.text()
    assert f"({n},{n}) entry only" in page.detail.header.text()

    # Truncate to mode 1: the driving-point sum now misses the full solution.
    page._check_all(False)
    page.mode_checks[0].setChecked(True)
    assert page.selected_terms().tolist() == [True] + [False] * (n - 1)
    assert "%" in page.detail.header.text() and "Shaded" in page.detail.header.text()
    # Static compliance of mode 1 alone, against K^-1.
    K = w.sim.system.matrices()[2]
    phi = w.modal.modes[0].shape_mass_normalized
    static = np.linalg.inv(K)[n - 1, n - 1]
    one = phi[n - 1] ** 2 / w.modal.modes[0].omega_n ** 2
    assert f"{100 * (one - static) / static:+.3g}%" in page.detail.header.text()

    # Clicking selects a cell; Ctrl+click adds another force in the same row.
    page._on_cell(0, 1, False)
    page._on_cell(0, 2, True)
    assert page.output == 0 and page.inputs == {1, 2}
    assert [c.isChecked() for c in page.input_checks] == [False, True, True, False]
    assert "H<sub>12</sub> + H<sub>13</sub>" in page.detail.header.text()
    assert "entries (1,2), (1,3)" in page.detail.header.text()
    page._on_cell(0, 2, True)  # Ctrl+click again removes it
    assert page.inputs == {1}

    for q in Quantity:
        page.quantity.setCurrentIndex(list(Quantity).index(q))
    page.show_terms.setChecked(True)

    # Non-proportional damping: classical misses even with every mode; exact does not.
    page._check_all(True)
    w.params.rows[0][3].setValue(15.0)
    page.expansion.setCurrentIndex(list(Expansion).index(Expansion.CLASSICAL))
    assert "none" not in page.detail.header.text()
    page.expansion.setCurrentIndex(list(Expansion).index(Expansion.EXACT))
    assert "none" in page.detail.header.text()

    # N set on this page drives the shared chain; the grid follows.
    page.dof.setValue(6)
    assert w.sim.system.n == 6 and len(page.grid.cells) == 6 and len(page.input_checks) == 6
    np.testing.assert_allclose(page.H[:, 2, 4], page.H[:, 4, 2])

    # A free (rigid-body) chain falls back to the classical expansion.
    w.params.rows[0][2].setValue(0.0)
    w.params.rows[0][3].setValue(0.0)
    assert "not available" in page.detail.header.text() and "rigid body" in page.detail.header.text()
    app.processEvents()  # let queued axis relayouts run before teardown
    w.close()


def test_worst_band_brackets_the_largest_difference():
    import numpy as np

    from vib_tutorial.gui.frf_matrix import worst_band

    f = np.linspace(1.0, 10.0, 901)
    full = np.ones(f.size, dtype=complex)
    partial = full + np.exp(-(((f - 6.0) / 0.5) ** 2))  # difference peaks at 6 Hz, half-height at +/-0.42
    lo, hi, peak = worst_band(f, full, partial)
    assert peak == pytest.approx(6.0)
    assert lo == pytest.approx(6.0 - 0.5 * np.sqrt(np.log(2)), abs=0.02)
    assert hi == pytest.approx(6.0 + 0.5 * np.sqrt(np.log(2)), abs=0.02)
    assert worst_band(f, full, full.copy()) is None


def test_modal_coordinates_view(app):
    import numpy as np

    from vib_tutorial.gui.main_window import MainWindow
    from vib_tutorial.gui.modes import Method

    w = MainWindow()
    w.params.rows[0][3].setValue(15.0)  # non-proportional: classical coordinates couple
    n = w.sim.system.n
    w.coords_combo.setCurrentIndex(1)
    assert w.modal_view and len(w.time_plot.x_curves) == n
    assert w.time_plot.x_curves[0].name() == "Mode 1"

    def release_and_run(row):
        w._reset()
        w.table.selectRow(row)
        w._release_mode()
        for _ in range(5):
            w._sim_target = w.sim.t + 0.2
            w._tick()
        return np.array([np.abs(c.getData()[1]).max() for c in w.time_plot.x_curves])

    # Classical release of mode 3 with coupled damping: the other coordinates pick up motion.
    amp = release_and_run(2)
    assert amp[2] > 0.015
    assert np.delete(amp, 2).max() > 1e-4

    # State-space: one curve per conjugate pair, and a released complex mode stays in its own.
    w.method_combo.setCurrentIndex(list(Method).index(Method.STATE_SPACE))
    assert len(w.time_plot.x_curves) == n
    assert w.time_plot.x_curves[2].name() == "λ5,6"
    amp = release_and_run(4)
    assert amp[2] > 0.015 and np.delete(amp, 2).max() < 1e-9

    # Changing the number of masses rebuilds the modal curves.
    w.params.dof.setValue(6)
    w._tick()
    assert len(w.time_plot.x_curves) == 6
    w.coords_combo.setCurrentIndex(0)
    assert w.time_plot.x_curves[0].name() == "x1"
    w.close()


def test_energy_time_history(app):
    import numpy as np

    from vib_tutorial.gui.main_window import ENERGY_CURVES, MainWindow

    w = MainWindow()
    w.coords_combo.setCurrentIndex(2)
    assert w.energy_view and not w.modal_view
    assert [c.name() for c in w.time_plot.x_curves] == [name for name, _ in ENERGY_CURVES]
    w.table.selectRow(0)
    w._release_mode()
    for _ in range(5):
        w._sim_target = w.sim.t + 0.2
        w._tick()
    T, V, stored, added, work, dissipated = (c.getData()[1] for c in w.time_plot.x_curves)
    assert added.max() > 0 and np.abs(work).max() == 0.0 and dissipated[-1] > 0
    np.testing.assert_allclose(added + work, stored + dissipated, rtol=0, atol=1e-12)
    # The balance still closes across a parameter edit (T, V use the parameters of the time).
    w.params.rows[0][2].setValue(800.0)
    w._sim_target = w.sim.t + 0.2
    w._tick()
    T, V, stored, added, work, dissipated = (c.getData()[1] for c in w.time_plot.x_curves)
    np.testing.assert_allclose(added + work, stored + dissipated, rtol=0, atol=1e-12)
    # Changing the number of masses keeps the energy curves.
    w.params.dof.setValue(6)
    w._tick()
    assert len(w.time_plot.x_curves) == len(ENERGY_CURVES)
    w.close()


def test_element_forces_view(app):
    import numpy as np

    from vib_tutorial.core import ForceKind
    from vib_tutorial.gui.main_window import MainWindow

    w = MainWindow()
    assert not w.element_combo.isVisibleTo(w)
    w.coords_combo.setCurrentIndex(3)
    assert w.forces_view and w.element_combo.isVisibleTo(w)
    n = w.sim.system.n
    assert [c.name() for c in w.time_plot.x_curves] == [f"k{i + 1}" for i in range(n)]
    # A step force at the tip: once the motion settles, every spring carries it.
    w.force.settings.kind = ForceKind.STEP
    w.force.switch_on()
    while w.sim.t < 150.0:  # mode 1 decays at zeta * omega_n = 0.12 1/s
        w._sim_target = w.sim.t + 2.0
        w._tick()
    springs = np.array([c.getData()[1][-1] for c in w.time_plot.x_curves])
    np.testing.assert_allclose(springs, w.force.value(), rtol=1e-6)
    w.element_combo.setCurrentIndex(1)
    w._tick()  # new curves fill on the next frame
    assert w.time_plot.x_curves[0].name() == "c1"
    assert max(abs(c.getData()[1][-1]) for c in w.time_plot.x_curves) < 1e-6
    w.element_combo.setCurrentIndex(2)
    assert w.time_plot.x_curves[0].name() == "k1 + c1"
    # Changing the number of masses gives one curve per element.
    w.params.dof.setValue(6)
    w._tick()
    assert len(w.time_plot.x_curves) == 6
    w.coords_combo.setCurrentIndex(0)
    assert not w.element_combo.isVisibleTo(w)
    w.close()


def test_energy_bars(app):
    import pytest

    from vib_tutorial.gui.main_window import MainWindow

    w = MainWindow()
    w.show()
    w.force_panel.button.click()  # harmonic force on
    for _ in range(5):
        w._sim_target = w.sim.t + 0.2
        w._tick()
    s = w.energy.bars.state
    assert s.work > 0 and s.dissipated > 0 and s.stored > 0
    assert s.added + s.work == pytest.approx(s.stored + s.dissipated, rel=1e-9)
    assert s.modal.sum() == pytest.approx(s.stored, rel=1e-9)
    w.energy.bars.grab()  # paints without error

    # Release mode 3 with coupled damping: energy moves to other modes.
    w.params.rows[0][3].setValue(15.0)
    w.energy.view.setCurrentIndex(1)
    assert w.energy.bars.by_mode
    w._reset()
    w.table.selectRow(2)
    w._release_mode()
    w._sim_target = w.sim.t + 0.02
    w._tick()
    early = w.energy.bars.state.modal / w.energy.bars.state.stored
    for _ in range(4):
        w._sim_target = w.sim.t + 0.2
        w._tick()
    late = w.energy.bars.state.modal / w.energy.bars.state.stored
    assert early[2] > 0.85 and late[0] > 0.5  # mode 3 loses ~10% to the others within 0.02 s
    w.energy.bars.grab()
    w.close()


def test_basis_plots_survive_changes_of_n(app):
    # Rebuilding the basis PlotItems on every change of N segfaulted after ~13 changes
    # (freed C++ objects still painted); the plots are now pooled and reused.
    from vib_tutorial.gui.main_window import MainWindow

    w = MainWindow()
    page = w.cms_page
    w.show()
    w.pages.setCurrentWidget(page)
    page.tabs.setCurrentWidget(page.basis)
    for n in [5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 8, 4, 8]:
        page.dof.setValue(n)
        app.processEvents()
        page.basis.plots.grab()  # paint
        expected = len(page.model.labels) if page.model else 0
        assert len(page.basis.plots.ci.items) == expected
    assert page.basis.shown == expected and len(page.basis._slots) <= 8  # reused, not rebuilt
    w.close()


def test_rigid_body_mode_title(app):
    import numpy as np

    from vib_tutorial.core import ModeComparison
    from vib_tutorial.gui.substructuring import ComparisonPlots

    shape = np.ones(3)
    plots = ComparisonPlots()
    plots.set_comparisons([ModeComparison(1, 0.0, 0.0, 1.0, shape, shape)], None)
    plots.set_highlight(0)  # error is None for f_true = 0: used to raise TypeError
    assert "error —" in plots.shape.titleLabel.text


def test_modal_test_page(app):
    from vib_tutorial.core import Excitation, Window
    from vib_tutorial.gui.main_window import MainWindow

    w = MainWindow()
    p = w.test_page
    assert p.acq is None  # nothing is measured until the page is shown
    w.show()
    w.pages.setCurrentWidget(p)
    assert p.acq is not None and p.fs_auto.isChecked()
    while not p.acq.done:
        p._measure_some()
    assert p.progress.text().startswith("10 / 10")
    assert p.window.currentData() is Window.FORCE_EXPONENTIAL  # picked for the impact test
    first = p.acq

    # Noise and processing reuse the same measurement.
    p.force_noise.setValue(5.0)
    p.window.setCurrentIndex(list(Window).index(Window.RECTANGULAR))
    assert p.acq is first
    # Test settings measure again; the excitation picks its usual window.
    p.excitation.setCurrentIndex(list(Excitation).index(Excitation.RANDOM))
    assert p.acq is not first and p.window.currentData() is Window.HANN
    assert p.overlap.isVisibleTo(p) and not p.tip.isVisibleTo(p)

    # Fewer averages are processed at once, from the data already measured.
    while not p.acq.done:
        p._measure_some()
    p.averages.setValue(3)
    assert p.estimator.estimate().count == 3

    p.excitation.setCurrentIndex(list(Excitation).index(Excitation.STEPPED_SINE))
    p.points.setValue(12)
    while not p.acq.done:
        p._measure_some()
    assert p.progress.text().startswith("12 / 12") and not p.window.isEnabled()
    assert "Points in it" in p.check.toHtml()
    # Samples as dots, the fitted sine drawn smoothly through them.
    sig = p.signals
    assert sig.f_curve.opts["symbol"] == "o" and sig.f_curve.opts["pen"] is None
    assert sig.f_fit.getData()[0].size > sig.f_curve.getData()[0].size

    # The chain changes on the Simulation page: the test follows, with fs re-chosen.
    fs = p.fs.value()
    w.params.dof.setValue(6)
    assert p.acq.n == 6 and p.output.count() == 6 and p.fs.value() > fs
    p.output.setCurrentIndex(0)
    p.anti_alias.setChecked(False)
    assert not p.acq.settings.anti_alias
    w.close()


def test_modal_test_playback_draws_each_record_as_it_is_recorded(app):
    from vib_tutorial.gui.main_window import MainWindow

    def size(curve):
        x = curve.getData()[0]
        return 0 if x is None else x.size

    w = MainWindow()
    w.show()
    p = w.test_page
    w.pages.setCurrentWidget(p)
    assert p.speed.currentData() is None  # instant by default
    p.speed.setCurrentIndex(p.speed.findText("10× real time"))
    p.restart()
    # One hit recorded; its time plots are drawn from the start, the FRF waits for it.
    assert p.acq.progress[0] == 1 and p._shown is not None
    full = p._live.t.size
    assert size(p.signals.f_curve) < full and size(p.frf_view.measured[0]) == 0
    assert "recording 1" in p.progress.text()
    p._shown = 0.5 * p._live.t[-1]
    p._measure_some()
    assert full // 3 < size(p.signals.f_curve) < full
    p._shown = p._live.t[-1]  # the record is complete: spectrum and FRF follow
    p._measure_some()
    assert p._shown is None and size(p.signals.f_curve) == full and size(p.frf_view.measured[0]) > 0
    # Instant finishes the rest at once.
    p.speed.setCurrentIndex(0)
    while not p.acq.done:
        p._measure_some()
    assert p.progress.text().startswith("10 / 10") and not p.running

    # The choice is remembered for the next launch.
    p.speed.setCurrentIndex(p.speed.findText("3× real time"))
    w.close()
    w = MainWindow()
    assert w.test_page.speed.currentText() == "3× real time"
    w.test_page.speed.setCurrentIndex(0)  # back to Instant for the other tests
    w.close()


def test_modal_extraction_on_the_test_page(app):
    import numpy as np
    from PySide6 import QtTest

    from vib_tutorial.core import Excitation, Window
    from vib_tutorial.core.identification import Method, auto_select
    from vib_tutorial.gui.main_window import MainWindow

    w = MainWindow()
    w.show()
    p = w.test_page
    # Fixed random signals and noise: with fresh ones, about 1 run in 8 fits an
    # extra mode inside the 0.5-4 Hz band below.
    p.seed = 0
    w.pages.setCurrentWidget(p)
    assert "complete" in p.results.notes.text()  # nothing extracted while measuring
    p.excitation.setCurrentIndex(list(Excitation).index(Excitation.PERIODIC_RANDOM))
    while not p.acq.done:
        p._measure_some()
    # LSCF by default, poles picked automatically: all four modes.
    assert p.extract.current is Method.LSCF and len(p.poles) == 4
    assert p.results.table.rowCount() == 4 and p.results.table.item(3, 7).text() == "1.000"
    assert p.frf_view.fit_curves[0].getData()[0] is not None

    # Clicking a selected pole (through its ring) removes it; clicking it again brings it back.
    p.tabs.setCurrentWidget(p.stab_plot)
    app.processEvents()

    def click(order, i):
        sp = p.stab_plot
        at = QtCore.QPointF(abs(p.stab.pole(order, i)) / (2 * np.pi), order)
        pos = sp.mapFromScene(sp.getViewBox().mapViewToScene(at))
        QtTest.QTest.mouseClick(sp.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=pos)
        app.processEvents()

    order, i = p.poles[1]
    click(order, i)
    assert len(p.poles) == 3 and p.results.table.item(1, 2).text() == "missed"
    click(order, i)
    assert len(p.poles) == 4
    # A new noise level keeps the hand-picked poles (found again by frequency).
    p.response_noise.setValue(0.5)
    assert len(p.poles) == 4 and p.picked is not None
    # Clear, then build the model up one pole at a time.
    p.extract.clear.click()
    assert p.poles == [] and p.frf_view.fit_curves[0].getData()[0] is None
    first = auto_select(p.stab)[0]
    click(*first)
    assert p.poles == [first] and len(p.ident.modes) == 1
    p.extract.auto.click()
    assert p.picked is None

    # The fit band follows the draggable region and limits the modes found.
    p.frf_view.fit_band_changed.emit(0.5, 4.0)
    assert p.extract.band == (0.5, 4.0)
    assert {int(p.results.table.item(r, 0).text()) for r in range(p.results.table.rowCount())} == {1, 2}

    for m in (Method.PEAK, Method.CIRCLE):
        p.extract.method.setCurrentIndex(list(Method).index(m))
        assert p.stab is None and p.ident.method is m and p.ident.modes
    assert not p.extract.order.isVisibleTo(p)

    # The exponential-window correction is offered only with that window.
    assert not p.extract.correct.isVisibleTo(p)
    p.excitation.setCurrentIndex(list(Excitation).index(Excitation.IMPACT))
    p.exp_end.setValue(5.0)
    while not p.acq.done:
        p._measure_some()
    assert p.window.currentData() is Window.FORCE_EXPONENTIAL and p.extract.correct.isVisibleTo(p)
    assert "moved" in p.results.notes.text()
    w.close()


def test_drag_a_mass_and_let_go(app):
    import numpy as np
    from PySide6 import QtCore
    from PySide6.QtTest import QTest

    from vib_tutorial.gui.animation import MAX_DRAG, SPACING
    from vib_tutorial.gui.main_window import MainWindow

    w = MainWindow()
    w.show()
    w.force_panel.button.click()  # a running force is switched off by the grab
    w._sim_target = 0.5
    w._tick()
    app.processEvents()
    view = w.chain

    def pixel(x: float, y: float = 0.0) -> QtCore.QPoint:
        return view.mapFromScene(view.plotItem.vb.mapViewToScene(QtCore.QPointF(x, y)))

    # Grab m2 and pull it right, past the limit: it stops at MAX_DRAG.
    QTest.mousePress(view.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=pixel(view._centers[1]))
    assert view.held == 1 and not w.force.active
    QTest.mouseMove(view.viewport(), pixel(2 * SPACING + 2 * MAX_DRAG))
    x = w.sim.displacement.copy()
    assert x[1] == pytest.approx(MAX_DRAG / view.gain) and np.all(w.sim.velocity == 0)
    np.testing.assert_allclose(x[0], x[1] / 2)  # uniform chain: m1 halfway
    t_held = w.sim.t
    w._tick()
    assert w.sim.t == t_held  # time stands still while the mass is held
    assert view._force_text.isVisible() and view._force_text.toPlainText().startswith("F = +")

    QTest.mouseRelease(view.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=pixel(2 * SPACING + MAX_DRAG))
    assert view.held is None
    w._sim_target = w.sim.t + 0.2
    w._tick()
    assert w.sim.t > t_held and np.any(w.sim.velocity != 0)
    assert w.sim.energy_added + w.sim.work == pytest.approx(w.sim.stored_energy + w.sim.dissipated)

    # A press away from the masses is not a grab.
    QTest.mousePress(view.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=pixel(0.5 * SPACING, 0.4))
    assert view.held is None
    w.close()


def test_base_excitation_and_chirp(app):
    import math

    import numpy as np

    from vib_tutorial.gui.main_window import MainWindow

    w = MainWindow()
    w.show()
    p = w.force_panel
    p.button.click()  # a running force stops when the input changes
    p.input.setCurrentIndex(1)
    assert w.force.settings.base and not w.force.on
    assert not p.target.isVisibleTo(p) and p.base_amplitude.isVisibleTo(p) and not p.amplitude.isVisibleTo(p)
    assert p.button.text().startswith("Move ground")
    assert w.frf_plot.mag.titleLabel.text.startswith("Ground motion")
    assert "Ground" in w.time_plot.f_plot.getAxis("left").labelText

    p.base_amplitude.setValue(5.0)
    p.button.click()
    for _ in range(4):
        w._sim_target = w.sim.t + 0.1
        w._tick()
    assert w.force.settings.base_amplitude == 0.005 and w.sim.ground != 0.0
    assert w.chain._wall[0].pos().x() == pytest.approx(w.chain.gain * w.sim.ground)
    assert w.sim.work != 0.0
    assert w.sim.energy_added + w.sim.work == pytest.approx(w.sim.stored_energy + w.sim.dissipated)
    for key in ("modal", "forces", "energy"):
        w.coords_combo.setCurrentIndex(w.coords_combo.findData(key))
        w._tick()

    # A short chirp: the drive line follows it, and the panel notices when it ends.
    p.sweep_time.setValue(0.5)
    p.kind.setCurrentIndex(list(ForceKind).index(ForceKind.CHIRP))
    assert p.sweep_start.isVisibleTo(p) and not p.freq.isVisibleTo(p)
    p.button.click()
    assert "sweeping" in p.button.text()
    drive = w.frf_plot.drive_lines[0]
    w._sim_target = w.sim.t + 0.25
    w._tick()
    assert drive.isVisible()
    assert 10 ** drive.value() == pytest.approx(w.force.frequency(), rel=1e-6) and w.force.frequency() > 1.0
    while w.force.on:
        w._sim_target = w.sim.t + 0.1
        w._tick()
    assert p.button.text().startswith("Start sweep") and not drive.isVisible()
    assert not math.isnan(w.sim.stored_energy)

    # Back to a force: the ground returns and the FRF is a receptance again.
    p.input.setCurrentIndex(0)
    w._sim_target = w.sim.t + 0.1
    w._tick()
    assert w.sim.ground == 0.0 and w.frf_plot.mag.titleLabel.text.startswith("Force at")
    assert np.isclose(w.sim.energy_added + w.sim.work, w.sim.stored_energy + w.sim.dissipated)
    w.close()


def test_make_app_themes(app):
    import pyqtgraph as pg

    from vib_tutorial.gui import make_app, theming
    from vib_tutorial.gui.style import colors

    try:
        assert make_app("Light") is app
        assert app.palette().window().color().lightness() > 200
        assert pg.getConfigOption("background") == "#ffffff" and colors.name == "Light"
        make_app("Dark")
        assert app.palette().window().color().lightness() < 60
        assert pg.getConfigOption("background") == colors.background == "#1c1e21"
        # The offscreen platform reports no desktop scheme, so System means Light.
        make_app("System")
        assert colors.name == "Light" and theming.current_name() == "System"
    finally:
        theming.apply("Light")


def test_switch_themes_live(app):
    import numpy as np
    import pyqtgraph as pg

    from vib_tutorial.gui import theming
    from vib_tutorial.gui.main_window import MainWindow
    from vib_tutorial.gui.style import THEMES, colors

    w = MainWindow()
    w.show()
    w.force_panel.button.click()
    w._sim_target = 0.5
    w._tick()
    try:
        for name in THEMES:
            w.theme_combo.textActivated.emit(name)
            assert theming.saved_name() == name and colors.name == name
            # Plots, the animation and the curves pick up the theme.
            assert w.chain.backgroundBrush().color().name() == colors.background
            assert w.time_plot.x_plot.getAxis("left").pen().color().name() == colors.foreground
            assert w.time_plot.x_curves[0].opts["pen"].color().name() == colors.mass[0]
            assert w.chain._rects[0].brush().color().name() == colors.mass[0]
            assert w.table.item(0, 0).foreground().color().name() == colors.mode[0]
            assert w.energy.bars.grab().toImage().pixelColor(2, 2).name() == colors.background
            # Every page redraws in it when shown.
            for page in (w.frf_page, w.cms_page, w.coupling_page, w.test_page):
                w.pages.setCurrentWidget(page)
                app.processEvents()
            assert w.frf_page.grid.full[0][0].opts["pen"].color().name() == colors.strong
            assert w.cms_page.plots.frf.getAxis("bottom").textPen().color().name() == colors.foreground
            assert all(s.plot.getAxis("left").pen().color().name() == colors.foreground
                       for s in w.cms_page.basis._slots)
            assert w.coupling_page.veering.plot.getAxis("left").pen().color().name() == colors.foreground
            while not w.test_page.acq.done:
                w.test_page._measure_some()
            assert w.test_page.frf_view.measured[0].opts["pen"].color().name() == colors.mass[w.sim.system.n - 1]
            w.pages.setCurrentWidget(w.sim_page)
            w._tick()
            assert np.isfinite(w.sim.displacement).all()
            for view in w.findChildren(pg.GraphicsView):
                assert view.backgroundBrush().color().name() == colors.background
    finally:
        w.set_theme("Light")
        w.close()


def test_load_presets(app):
    from vib_tutorial.gui.main_window import MainWindow

    w = MainWindow()
    names = [w.params.preset.itemText(i) for i in range(w.params.preset.count())]
    tmd = names.index("Tuned mass damper (Den Hartog)")
    w.force_panel.button.click()
    w._sim_target = 0.5
    w._tick()
    w.params.preset.activated.emit(tmd)
    assert w.params.preset.currentIndex() == 0  # a one-shot menu
    preset = w.params.presets[tmd - 1]
    assert w.sim.system.n == 2 and len(w.params.rows) == 2
    assert list(w.sim.system.stiffness) == pytest.approx(list(preset.system.stiffness), rel=1e-3)
    assert w.sim.t == 0.0 and not w.force.on  # starts from rest, force off
    assert w.force.settings.target == 0 and w.force.settings.freq_hz == pytest.approx(preset.force.freq_hz, abs=1e-3)
    assert w.frf_compare.isChecked() and w.tabs.currentWidget() is w.frf_tab
    assert "without m2" in w.frf_plot.mag.titleLabel.text
    assert len(w.frf_plot.mag_curves) == 3  # x1, x2 and x1 without the absorber
    assert "Den Hartog" in w.params.preset_note.text()

    # The building: ground motion, five masses.
    w.params.preset.activated.emit(names.index("TMD on a 4-storey building"))
    assert w.sim.system.n == 5 and w.force.settings.base
    assert w.force_panel.target.isHidden()  # no mass to pick for ground motion
    w.force_panel.button.click()
    w._sim_target = 0.5
    w._tick()
    assert w.sim.t > 0.1

    # Back to the plain chain: no comparison.
    w.params.preset.activated.emit(names.index("Uniform chain (4 masses)"))
    assert w.sim.system.n == 4 and not w.force.settings.base and not w.frf_compare.isChecked()
    assert len(w.frf_plot.mag_curves) == 4


def test_substructuring_back_expansion(app):
    from vib_tutorial.gui.main_window import MainWindow

    w = MainWindow()
    page = w.cms_page
    w.show()
    w.pages.setCurrentWidget(page)
    page.dof.setValue(6)
    page.set_interfaces([2])
    page.tabs.setCurrentWidget(page.recovery)
    app.processEvents()
    v = page.recovery
    assert v._timer.isActive() and not page.time._timer.isActive()  # only the tab on show runs
    assert [v.mass_box.itemData(i) for i in range(v.mass_box.count())] == [0, 1, 3, 4]
    assert v.rec_view._recovered == {0, 1, 3, 4}
    assert "4 interior masses" in v.header.text()

    # Choosing a mass picks the spring on its left, among its substructure's springs.
    v.select(1)
    assert v.spring == 1 and v.spring_box.count() == 3  # A: k1, k2, k3
    v.select(4)
    assert v.spring == 4 and v.spring_box.count() == 3  # B: k4, k5, k6
    v.select(1, 2)
    assert v.spring == 2

    v.button.click()
    for _ in range(5):
        v.step(0.2)
    assert v.sim.t >= 0.99
    x_true = v.x_curves["true"].yData
    assert x_true.size and np.abs(x_true).max() > 1e-3
    for curve in (*v.e_curves.values(), *v.f_curves.values()):
        assert curve.yData.size == x_true.size
    np.testing.assert_allclose(v.x_curves["from_boundary"].yData + v.x_curves["from_modes"].yData,
                               v.x_curves["coupled"].yData, atol=1e-12)
    assert "RMS error" in v.error_note.text() and "enhanced" in v.error_note.text()
    for i in range(v.chain_box.count()):
        v.chain_box.setCurrentIndex(i)
    assert "enhanced" in v.rec_view.plotItem.titleLabel.text

    # A theme change keeps the motion; Guyan restarts, and then the coupled recovery is the
    # boundary-only one.
    before = v.sim.t
    w.set_theme("Dark")
    assert v.sim.t == before
    page.guyan_button.click()
    assert v.sim.t == 0.0 and v.force.on and v.mass_dof == 1 and v.spring == 2
    for _ in range(3):
        v.step(0.2)
    np.testing.assert_allclose(v.e_curves["coupled"].yData, v.e_curves["boundary"].yData, atol=1e-12)
    assert not np.any(v.x_curves["from_modes"].yData)

    # Every mass a boundary DOF: nothing to recover.
    page.set_interfaces([0, 1, 2, 3, 4])
    assert v.mass_box.count() == 0 and "no interior" in v.header.text()
    v.step(0.2)
    # No model at all: the tab says why.
    page.set_interfaces([2])
    w.params.rows[0][2].setValue(0.0)
    w.params.rows[1][2].setValue(0.0)
    assert v.sim is None and "K_ii" in v.header.text()
    w.pages.setCurrentWidget(w.sim_page)
    assert not v._timer.isActive()
    w.set_theme("Light")
    w.close()


def test_modal_coupling_page(app):
    from vib_tutorial.gui.main_window import MainWindow

    w = MainWindow()
    page = w.coupling_page
    w.show()
    w.pages.setCurrentWidget(page)
    app.processEvents()

    # Four masses: split in the middle, the first modes of A and B coupled.
    assert page.split == 2 and len(page.split_buttons) == 3 and page.split_buttons[1].isChecked()
    assert page.model.split == 2 and (page.model.mode_a, page.model.mode_b) == (0, 0)
    assert page.sub_table.rowCount() == 4 and page.sub_table.item(0, 0).text().startswith("A1")
    assert page.result_table.rowCount() == 2 and page.result_table.item(0, 3).text() == "mode 1"
    assert "Rayleigh–Ritz" in page.result_note.text() and "strongly coupled" in page.summary.text()
    assert page.veering.data is not None and page.veering.data.full.shape[1] == 4
    # The result table is laid out in full, with no scroll bar.
    assert page.result_table.height() >= page.result_table.horizontalHeader().height() + 2 * page.result_table.rowHeight(0)

    # The step-by-step matrices follow the model: the tip modal mass is the generalized mass of
    # the tip-scaled shape, and the influence vector r comes out as ones.
    text = page.matrices.toPlainText()
    assert "Participation" in text and "r = [1, 1]" in text and "Notation" in text
    assert f"{page.model.osc_a.m:.4g} kg" in text and "These are the 2-DOF K and M" in text
    w.params.rows[0][1].setValue(2.0)  # m1 = 2 kg
    assert page.matrices.toPlainText() != text

    # Split after m1, couple B's mode 2 by clicking its row.
    page.split_buttons[0].click()
    assert page.model.split == 1 and page.sub_table.rowCount() == 4
    page.sub_table.cellClicked.emit(2, 0)  # rows: A1, B1, B2, B3
    assert page.model.mode_b == 1 and page.mode_b.currentIndex() == 1
    page.set_modes(0, 0)

    # A's effective mass instead, and no residual mass: no longer Rayleigh-Ritz.
    page.a_mass.setCurrentIndex(1)
    assert page.model.a_mass == "effective" and page.model.rayleigh_ritz  # A is one mass: the same mass
    assert "A is one mass" in page.result_note.text()
    page.set_split(2)
    assert not page.model.rayleigh_ritz and "no longer a Rayleigh" in page.result_note.text()
    assert "These differ" in page.matrices.toPlainText()
    page.set_split(1)
    page.a_mass.setCurrentIndex(0)
    page.residual.setChecked(False)
    assert not page.model.residual and "no longer a Rayleigh" in page.result_note.text()
    page.residual.setChecked(True)

    # Sweep the mass ratio instead of the frequency.
    page.veering.kind.setCurrentIndex(1)
    assert page.veering.data.kind == "mass"
    for tab in range(page.tabs.count()):
        page.tabs.setCurrentIndex(tab)
        app.processEvents()

    # N changed on the Simulation page: the split follows, kept where it was.
    w.params.dof.setValue(8)
    assert len(page.split_buttons) == 7 and page.split == 1 and page.model.B.dofs.size == 7
    page.set_split(6)
    page.set_modes(1, 0)
    assert page.model.split == 6 and page.model.mode_a == 1
    assert page.result_table.item(0, 3).text() != "mode 1"  # paired by shape with a higher mode

    # A parameter edit on the Simulation page reaches this page.
    k_before = page.model.osc_b.k
    w.params.rows[6][2].setValue(800.0)
    assert page.model.osc_b.k != k_before

    # One mass cannot be split.
    w.params.dof.setValue(1)
    assert page.model is None and "at least 2 masses" in page.summary.text()
    w.params.dof.setValue(4)
    assert page.model is not None
    w.pages.setCurrentWidget(w.sim_page)
    w.close()


def test_log_axis_labels(app):
    import math

    from vib_tutorial.gui.axes import LogAxis, log_label

    assert [log_label(v) for v in (1e-5, 3e-5, 1e-4, 0.05, 1.0, 20.0, 1000.0, 1e4, 2e6)] == [
        "10⁻⁵", "3·10⁻⁵", "0.0001", "0.05", "1", "20", "1000", "10⁴", "2·10⁶"]
    values = [math.log10(m * 0.1) for m in range(1, 10)] + [0.0]
    for orientation in ("left", "bottom"):
        axis = LogAxis(orientation)
        axis.setLogMode(True)
        assert not axis.autoSIPrefix
        labelled = {}
        for px in (2000.0, 250.0, 20.0):  # pixels per decade: room for all, for 1-2-5, for the decades only
            axis._px_per_decade = px
            labelled[px] = [s for s in axis.logTickStrings(values, 1.0, None) if s]
        assert labelled[2000.0] == ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1"]
        assert labelled[250.0] == ["0.1", "0.2", "0.5", "1"]
        assert labelled[20.0] == ["0.1", "1"]
    # The minor level does not repeat the decades of a major level.
    axis = LogAxis("left")
    axis.setLogMode(True)
    levels = axis.logTickValues(-2.2, 0.3, 300, [(1.0, [-2.0, -1.0, 0.0])])
    minor = next(v for s, v in levels if s is None)
    assert not {round(v, 9) for v in minor} & {-2.0, -1.0, 0.0}
