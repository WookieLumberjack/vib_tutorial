"""Headless smoke test: build the window and exercise the main interactions."""

import math
import os
import sys
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
    picks = decimation_index(y, 3000)
    for c, idx in enumerate(picks):
        assert idx.size <= 3000
        assert y[idx, c].max() == y[:, c].max() and y[idx, c].min() == y[:, c].min()


def test_decimation_keeps_each_curves_own_extremes_per_bucket():
    """A fast sine beside a slow one: every bucket's min and max of each curve are among its own points."""
    import numpy as np

    from vib_tutorial.gui.plots import decimation_index

    t = np.linspace(0, 10, 100_000)
    y = np.column_stack([np.sin(2 * np.pi * 12 * t), np.sin(2 * np.pi * 0.5 * t)])
    picks = decimation_index(y, 1500)  # ~750 buckets of 133 samples, ~1.6 per cycle of the fast one
    b = math.ceil(1.2 * math.ceil(2 * len(t) / 1500))
    for c, idx in enumerate(picks):
        bucket = idx // b
        for j in range(len(t) // b):
            mine = y[idx[bucket == j], c]
            whole = y[j * b : (j + 1) * b, c]
            assert mine.max() == whole.max() and mine.min() == whole.min()


def test_decimator_cache_matches_a_fresh_pass():
    import numpy as np

    from vib_tutorial.gui.plots import Decimator

    rng = np.random.default_rng(0)
    y = rng.standard_normal((200_000, 3)).cumsum(axis=0)
    cached = Decimator(3000)
    start, end = 0, 2000
    for _ in range(200):  # a window that grows, scrolls, and changes length
        end = min(end + int(rng.integers(0, 3000)), len(y))
        start = max(start, end - int(rng.integers(20_000, 60_000)))
        w = y[start:end]
        picks = cached.index(start, len(w), w.__getitem__, 3)
        if len(w) > 3000:
            fresh = Decimator(3000)  # same buckets, nothing cached
            fresh.bucket, fresh.picks = cached.bucket, np.empty((0, 6), dtype=np.intp)
            for a, b in zip(picks, fresh.index(start, len(w), w.__getitem__, 3), strict=True):
                np.testing.assert_array_equal(a, b)
        for c, idx in enumerate(picks):
            assert idx[0] == 0 and idx[-1] == len(w) - 1 and (len(w) <= 3000 or idx.size <= 3000 + 6)  # the edges add a few
            assert w[idx, c].max() == w[:, c].max() and w[idx, c].min() == w[:, c].min()


def test_history_offset_numbers_samples():
    import numpy as np

    from vib_tutorial.gui.history import ENERGY_COLUMNS, History

    h = History(1, capacity=100)
    count = 0

    def extend(k):
        nonlocal count
        ts = np.arange(count, count + k, dtype=float)  # t is the sample's number
        e = np.zeros((k, len(ENERGY_COLUMNS)))
        h.extend(ts, np.zeros((k, 1)), np.zeros((k, 1)), np.zeros(k), e, np.zeros((k, 2)))
        count += k
        assert h.t[0] == h.offset and h.offset + h.size == count

    for k in (30, 40, 50, 7, 250, 60):  # compactions, and a batch larger than the buffer
        extend(k)
    h.reset(1)
    assert h.offset == count  # numbering continues after a reset
    extend(10)


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


def _energy_rows(w):
    """The energy curves' whole rows (not the decimated points each curve shows), checked against the curves."""
    import numpy as np

    t, *rows = w.history.window(w.controls.window.value())
    y = w._plot_values(*rows)(slice(None))
    for i, c in enumerate(w.time_plot.x_curves):
        tc, yc = c.getData()
        np.testing.assert_array_equal(yc, y[np.searchsorted(t, tc), i])
    return y[:, :-1].T


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
    T, V, stored, added, work, dissipated, rubbed = _energy_rows(w)
    assert added.max() > 0 and np.abs(work).max() == 0.0 and dissipated[-1] > 0 and rubbed.max() == 0.0
    np.testing.assert_allclose(added + work, stored + dissipated, rtol=0, atol=1e-12)
    # The balance still closes across a parameter edit (T, V use the parameters of the time),
    # and with friction.
    w.params.rows[0][2].setValue(800.0)
    w.params.rows[1][4].setValue(0.05)
    for _ in range(3):
        w._sim_target = w.sim.t + 0.2
        w._tick()
        w._collect()  # with friction the steps run beside the frame: wait for them
    assert w.sim.system.friction[1] == 0.05
    T, V, stored, added, work, dissipated, rubbed = _energy_rows(w)
    assert rubbed[-1] > 0
    np.testing.assert_allclose(added + work, stored + dissipated + rubbed, rtol=0, atol=1e-12)
    # Changing the number of masses keeps the energy curves.
    w.params.dof.setValue(6)
    w._tick()
    assert len(w.time_plot.x_curves) == len(ENERGY_CURVES)
    w.close()


def test_static_friction_ratio_reaches_the_simulator(app):
    from vib_tutorial.gui.main_window import MainWindow

    w = MainWindow()
    params = w.params
    params.static_ratio.setValue(1.8)
    assert w.sim.system.static_ratio == 1.8
    params.dof.setValue(3)  # kept when the chain is resized
    assert w.sim.system.static_ratio == 1.8 and w.sim.system.n == 3
    params.rows[0][4].setValue(0.4)
    params._copy_first_row()
    assert list(w.sim.system.friction) == [0.4] * 3 and w.sim.system.static_ratio == 1.8
    params._reset_defaults()
    assert w.sim.system.static_ratio == 1.0 and not w.sim.system.friction.any()
    w.close()


def test_modal_table_friction_column(app):
    from vib_tutorial.gui.main_window import MainWindow
    from vib_tutorial.gui.modes import Method

    w = MainWindow()
    table, column = w.table, w.table.FRICTION_COLUMN
    assert table.isColumnHidden(column)  # no friction by default
    w.params.rows[0][4].setValue(0.5)
    w.params._copy_first_row()
    assert not table.isColumnHidden(column)
    zeta = float(table.item(0, column).text())
    w.release_amp.setValue(2 * w.release_amp.value())  # twice the amplitude, half the damping
    assert float(table.item(0, column).text()) == pytest.approx(zeta / 2, abs=1e-4)
    # The state-space table has one column fewer; coming back restores the friction column.
    w.method_combo.setCurrentIndex(list(Method).index(Method.STATE_SPACE))
    assert table.columnCount() == len(table.STATE_SPACE_HEADERS)
    w.method_combo.setCurrentIndex(list(Method).index(Method.CLASSICAL))
    assert table.columnCount() == len(table.CLASSICAL_HEADERS) and not table.isColumnHidden(column)
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


def test_modal_test_with_friction_holds_one_level_for_comparison(app):
    from vib_tutorial.gui.main_window import MainWindow

    w = MainWindow()
    w.show()
    p = w.test_page
    p.seed = 1
    w.pages.setCurrentWidget(p)
    for row in w.params.rows:
        row[4].setValue(0.5)
    assert "Friction on m1, m2, m3, m4" in p.check.toPlainText()
    legend = p.frf_view.mag.legend
    labels = lambda: [label.text for _, label in legend.items]  # noqa: E731
    assert p.force_level_info.text() == "100 N peak"
    p.hold.setChecked(True)  # nothing finished yet: nothing to hold
    assert not p.hold.isChecked() and p.held is None
    while not p.acq.done:
        p._measure_some()
    assert "Exact, without friction" in labels()
    p.hold.setChecked(True)
    first = p.estimate.H[:, 3].copy()
    p.force_level.setValue(1000.0)
    assert p.force_level_info.text() == "1000 N peak" and p.acq.settings.force_level == 10.0
    while not p.acq.done:
        p._measure_some()
    assert "Held: impact hammer, force level 100%" in labels()
    held = p.frf_view.held_curves[0].getData()[1]
    assert held.size and 10 ** np.nanmax(held) == pytest.approx(np.abs(first[1:]).max())  # log axis
    # Friction: the stronger hit gives a taller first peak.
    assert np.abs(p.estimate.H[:, 3]).max() > 1.5 * np.abs(first).max()
    # Measuring another quantity hides the held curve; letting go of it clears it.
    p.response.setCurrentIndex(1)
    while not p.acq.done:
        p._measure_some()
    hidden = p.frf_view.held_curves[0].getData()[0]
    assert hidden is None or hidden.size == 0
    p.hold.setChecked(False)
    assert p.held is None
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

    # Accelerometers: a new measurement of the accelerance, fitted as receptance after dividing by -w^2.
    from vib_tutorial.core import Response

    p.excitation.setCurrentIndex(list(Excitation).index(Excitation.PERIODIC_RANDOM))
    p.response.setCurrentIndex(list(Response).index(Response.ACCELERATION))
    while not p.acq.done:
        p._measure_some()
    assert p.acq.settings.response is Response.ACCELERATION and p.estimate.response is Response.ACCELERATION
    assert "accelerance" in p.frf_view.mag.titleLabel.text and "a4" in p.signals.response.getAxis("left").labelText
    assert "−ω²" in p.results.notes.text() and "acceleration measured" in p.check.toHtml()
    p.frf_view.fit_band_changed.emit(0.0, p.acq.settings.band)
    p.extract.method.setCurrentIndex(list(Method).index(Method.LSCF))
    rows = [p.results.table.item(r, 0).text() for r in range(p.results.table.rowCount())]
    assert rows == ["1", "2", "3", "4"]
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


def test_dragging_with_friction_leaves_the_chain_where_friction_holds_it(app):
    import numpy as np

    from vib_tutorial.gui.main_window import MainWindow

    w = MainWindow()
    w.show()
    for row in w.params.rows:
        row[4].setValue(2.0)
    w.params.static_ratio.setValue(1.5)
    w._on_mass_grabbed(3)
    for x in np.linspace(0.0, 0.05, 11):  # out and back, as a hand would
        w._on_mass_dragged(3, x)
    for x in np.linspace(0.05, 0.0, 11):
        w._on_mass_dragged(3, x)
    w._on_mass_released(3)
    held = w.sim.displacement.copy()
    assert held[3] == 0.0 and np.all(held[:3] > 0.0)  # the others stayed out
    # Let go: friction holds every mass, so nothing moves.
    w._sim_target = w.sim.t + 0.5
    w._tick()
    np.testing.assert_array_equal(w.sim.displacement, held)
    assert not w.sim.velocity.any()
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


def test_startup_skips_scipy_signal():
    """scipy.signal (and the scipy.stats it pulls in) costs ~0.9 s; only the modal test needs it."""
    import subprocess
    import sys

    code = (
        "import sys; import vib_tutorial.gui.main_window; "
        "print([m for m in ('scipy.signal', 'scipy.stats', 'scipy.optimize') if m in sys.modules])"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_record_startup(app, tmp_path):
    """The release builds' smoke test records the time to the first painted window."""
    import json
    import time

    from vib_tutorial.gui import record_startup
    from vib_tutorial.gui.main_window import MainWindow

    path = tmp_path / "startup.json"
    w = MainWindow()
    record_startup(w, str(path), launched_at=time.time() - 1.0)
    w.show()
    for _ in range(50):
        app.processEvents()
        if path.exists():
            break
        time.sleep(0.01)
    t = json.loads(path.read_text())
    assert 0 < t["window"] <= t["python"] and t["launch"] >= 1.0


def test_frequency_response_with_friction(app):
    from vib_tutorial.gui.main_window import MainWindow

    w = MainWindow()
    w.show()
    plot = w.frf_plot
    w.params.rows[0][4].setValue(0.5)
    w.params._copy_first_row()
    assert not plot.friction_curves  # the Modal analysis tab is in front: not worked out yet
    w.tabs.setCurrentWidget(w.frf_tab)
    assert len(plot.friction_curves) == 2 * w.sim.system.n and w.frf_friction_note.isVisible()
    assert "friction at 10 N" in plot.mag.titleLabel.text
    w.force_panel.amplitude.setValue(2.0)
    assert "friction at 2 N" in plot.mag.titleLabel.text and len(plot.friction_curves) == 2 * w.sim.system.n
    drawn = list(plot.friction_curves)
    w.tabs.setCurrentIndex(0)
    w.tabs.setCurrentWidget(w.frf_tab)
    assert plot.friction_curves == drawn  # nothing changed while hidden: kept, not worked out again
    w.tabs.setCurrentIndex(0)
    w.force_panel.amplitude.setValue(3.0)
    assert plot.friction_curves == drawn  # changed while hidden: drawn when shown
    w.tabs.setCurrentWidget(w.frf_tab)
    assert plot.friction_curves != drawn and len(plot.friction_curves) == 2 * w.sim.system.n
    w.force_panel.input.setCurrentIndex(1)  # ground motion
    w.force_panel.base_amplitude.setValue(5.0)
    assert "friction at 5 mm" in plot.mag.titleLabel.text
    w.params._reset_defaults()
    assert not plot.friction_curves and "friction" not in plot.mag.titleLabel.text
    assert not w.frf_friction_note.isVisible()
    w.close()


def test_every_legend_has_the_theme_background(app):
    # restyle() only reaches the legends that exist when the theme is applied; one made later
    # (a page built on first use, a rebuilt plot) must get its background when it's created.
    import pyqtgraph as pg

    from vib_tutorial.gui.main_window import MainWindow

    w = MainWindow()
    w.show()
    for i in range(w.pages.count()):
        w.pages.setCurrentIndex(i)
        app.processEvents()
    w.params.dof.setValue(6)  # rebuilds the plots that have one panel per mass
    app.processEvents()
    legends = [
        item
        for view in w.findChildren(pg.GraphicsView)
        for item in view.scene().items()
        if isinstance(item, pg.LegendItem)
    ]
    assert len(legends) > 10
    bare = [legend for legend in legends if legend.opts["brush"].style() == QtCore.Qt.BrushStyle.NoBrush]
    assert not bare
    w.close()


def test_modal_test_draws_the_describing_function_for_a_stepped_sine(app):
    from vib_tutorial.core import Excitation
    from vib_tutorial.gui.main_window import MainWindow

    w = MainWindow()
    w.show()
    p = w.test_page
    p.seed = 1
    w.pages.setCurrentWidget(p)
    legend = p.frf_view.mag.legend
    labels = lambda: [label.text for _, label in legend.items]  # noqa: E731

    def first_frequency():
        while p.acq.progress[0] < 1:
            p._measure_some()

    p.excitation.setCurrentIndex(p.excitation.findData(Excitation.STEPPED_SINE))
    first_frequency()
    assert not any(t.startswith("Describing function") for t in labels())  # no friction
    for row in w.params.rows:
        row[4].setValue(0.5)
    first_frequency()
    assert "Describing function at 10 N" in labels()
    p.force_level.setValue(30.0)
    first_frequency()
    assert "Describing function at 3 N" in labels()
    p.excitation.setCurrentIndex(p.excitation.findData(Excitation.IMPACT))  # no single amplitude
    first_frequency()
    assert not any(t.startswith("Describing function") for t in labels())
    w.close()


def test_rotor_page(app):
    from vib_tutorial.core.rotor import RPM, Bearing, RotorSystem
    from vib_tutorial.gui.main_window import MainWindow
    from vib_tutorial.gui.style import colors

    w = MainWindow()
    p = w.rotor_page
    w.pages.setCurrentWidget(p)
    p._timer.stop()
    w.params.dof.setValue(3)  # the chain's edits leave the rotor alone
    assert p.sim.system == RotorSystem()

    # It ramps toward the target and the plots fill.
    p.set_target_rpm(1500.0)
    assert p.sim.target == pytest.approx(1500.0 / RPM)
    for _ in range(40):
        p.step(0.05)
    assert 0 < p.sim.omega <= p.sim.target and p.history.size > 0
    assert p.orbits.trails[1].getData()[0].size > 0
    assert "rpm" in p.readout.text()
    assert "1,584 rpm" in p.crit_label.text() and "forward" in p.crit_label.text()
    assert not p.orbits.plots[2].isVisible()  # midspan is the disc

    # A run-up and coast-down, tracked once per revolution onto the Bode and polar plots.
    p.ramp.setValue(5000.0)
    p.sweep_from.setValue(1200.0)
    p.sweep_to.setValue(2000.0)
    p.start_sweep()
    assert p.tabs.current_page() is p.bode.parentWidget() and not p.ramp.isEnabled()
    assert p.sweep_button.text() == "Stop sweep"
    for _ in range(100):
        if p.run_up is None:
            break
        p.step(0.05)
    assert p.run_up is None and len(p.sweeps) == 2 and p.ramp.isEnabled()
    assert p.target_rpm.value() == pytest.approx(1200.0) and p.sim.omega == pytest.approx(1200.0 / RPM)
    assert p.sweep_plots.curves[0].amp.getData()[0].size > 3
    summary = p.sweep_summary.text()
    assert "Run-up 5,000 rpm/s" in summary and "Coast-down 5,000 rpm/s" in summary and "1,590 rpm" in summary
    # Another station and direction redraw the same sweeps.
    p.probe_station.setCurrentIndex(0)
    p.probe_direction.setCurrentIndex(2)
    assert "bearing a, orbit" in p.sweep_summary.text()
    assert p.polar.steady.getData()[0].size > 0 and p.polar.mark_labels
    # Held, a second sweep adds to the first; taking over the speed stops it, its legs so far kept.
    p.start_sweep()
    for _ in range(30):
        if len(p.sweeps) == 3:
            break
        p.step(0.02)
    assert p.run_up is not None and len(p.sweeps) == 3
    p.set_target_rpm(1500.0)
    assert p.run_up is None and len(p.sweeps) == 3 and p.sim.target == pytest.approx(1500.0 / RPM)
    assert p.target_rpm.value() == 1500.0
    # Not held, a new sweep replaces them; Clear takes them all off.
    p.hold_sweeps.setChecked(False)
    p.start_sweep()
    assert p.sweeps == [] and len(p.sweep_plots.curves) == 0
    p.step(0.05)
    p.clear_sweeps()
    assert p.run_up is None and not p.sweeps and "sweep" not in p.sweep_status.text()
    p.ramp.setValue(500.0)

    # Off centre, anisotropic, different bearings: the panels describe it and the analysis follows.
    p.position.setValue(0.3)
    p.isotropic.setChecked(False)
    p.bearing_rows[0][1].setValue(8e4)
    p.same_bearings.setChecked(False)
    p.bearing_rows[1][0].setValue(3e4)
    s = p.sim.system
    assert s.position == 0.3 and s.bearing_a.ky == 8e4 and s.bearing_b.kx == 3e4 and s.bearing_b.ky == 8e4
    assert p.orbits.plots[2].isVisible()
    p._refresh_analysis()
    assert len(p.campbell.criticals) >= 2
    p.tap()
    for _ in range(5):
        p.step(0.05)
    assert np.isfinite(p.sim.state).all()

    # The matrices at the current speed.
    p.tabs.show_page(p.matrices)
    html = p.matrices.toHtml()
    assert "Ω G" in html and "backward" in html

    # Dragging the speed line on the Campbell diagram sets the target.
    p.campbell_plot.speed_dragged.emit(2500.0)
    assert p.target_rpm.value() == 2500.0 and p.speed_slider.value() == 2500

    # Themes reach the view, and set_system round-trips through the panels.
    w.set_theme("Dark")
    assert p.view.shaft.opts["pen"].color().name() == colors.structure
    bearing = Bearing(kx=2e4, ky=8e4, cx=20.0, cy=20.0)
    system = RotorSystem(position=0.4, bearing_a=bearing, bearing_b=bearing, unbalance=5e-5, bow=30e-6,
                         bow_angle=math.radians(45.0), skew=math.radians(0.2), skew_angle=math.radians(-90.0))
    p.set_system(system)
    assert p.sim.system == system and not p.isotropic.isChecked() and p.same_bearings.isChecked()
    p.reset()
    assert p.sim.t == 0.0 and p.history.size == 0
    assert p.sim.q[1] == pytest.approx(30e-6 * math.cos(math.radians(45.0)))  # at rest in its bowed shape
    p._refresh_analysis()
    assert abs(p._steady_vectors[0, 1, 0]) == pytest.approx(30e-6, rel=1e-2)  # the slow-roll runout
    w.set_theme("Light")

    # Gravity: the rotor rests sagged and whirls about the sag; unticking Static sag draws the vibration.
    p.set_system(RotorSystem())
    assert not p.show_sag.isEnabled()
    p.gravity.setChecked(True)
    assert p.sim.system.gravity == pytest.approx(9.81) and p.show_sag.isEnabled()
    assert "Static sag at rest: disc 378 µm" in p.shaft_note.text()
    p.reset()
    sag = p.sim.system.static_deflection()
    np.testing.assert_allclose(p.sim.q, sag)
    p.ramp.setValue(5000.0)
    p.set_target_rpm(1200.0)
    for _ in range(30):
        p.step(0.05)
    disc = p.orbits.trails[1].getData()
    assert np.mean(disc[1]) == pytest.approx(sag[5], rel=0.05)  # below the centre
    p.show_sag.setChecked(False)
    disc = p.orbits.trails[1].getData()
    assert abs(np.mean(disc[1])) < 0.1 * abs(sag[5])
    assert "about its sag" in p.readout.text()
    p.gravity.setChecked(False)
    assert not p.show_sag.isEnabled()

    # Balancing: a hidden unbalance, found in four measured steps and cancelled by one weight.
    p.set_system(RotorSystem())
    b = p.balancing
    b._rng = np.random.default_rng(3)
    b.new_mystery()
    hidden_u = p.sim.system.mass_unbalance
    assert p.unbalance.text() == "hidden" and not p.unbalance.isEnabled() and not p.view.show_heavy
    p.tabs.show_page(p.matrices)
    p._refresh_matrices()
    assert "hidden" in p.matrices.toPlainText() and f"{abs(hidden_u) * 1e6:.4g} g·mm" not in p.matrices.toPlainText()
    p.tabs.show_page(b)
    p.ramp.setValue(5000.0)
    for i, button in enumerate(b.step_buttons):
        assert button.isEnabled()
        button.click()
        assert p.measurement is not None and not button.isEnabled()
        for _ in range(400):
            p.step(0.05)
            if p.measurement is None:
                break
        assert p.measurement is None
    assert len(p.sim.system.weights) == 1  # the trial is off, the correction on
    assert p.sim.system.mass_unbalance == pytest.approx(0.0, abs=0.02 * abs(hidden_u))
    assert abs(b.reading("verify")) < 0.02 * abs(b.reading("found"))
    assert "Correction" in b.results.text() and "Check" in b.results.text()
    b.station.setCurrentIndex(1)  # the readings are kept at every station: switch the probe after
    assert abs(b.reading("verify")) < 0.02 * abs(b.reading("found"))
    p.reveal_unbalance()
    assert p.unbalance.isEnabled() and p.unbalance.value() == pytest.approx(abs(hidden_u) * 1e6, abs=0.05)
    assert p.view.show_heavy and b.show_heavy.isChecked()
    b.remove_weights()
    assert p.sim.system.weights == () and b.reading("found") is None
    # Two planes on the disc's faces: a hidden unbalance and skew, cancelled at every speed in five steps.
    p.set_system(RotorSystem(position=0.3))
    b.mode.setCurrentIndex(1)
    assert len(b.step_buttons) == 5 and b.station2.isVisible() == b.station.isVisible()
    b.new_mystery()
    hidden = p.sim.system
    assert hidden.skew > 0 and p.skew.text() == "hidden" and not p.skew.isEnabled()
    p.tabs.show_page(p.matrices)
    p._refresh_matrices()
    assert "skew τ (hidden)" in p.matrices.toPlainText()
    p.tabs.show_page(b)
    for button in b.step_buttons:
        button.click()
        for _ in range(400):
            p.step(0.05)
            if p.measurement is None:
                break
        assert p.measurement is None
    s = p.sim.system
    assert s.weights == () and len(s.plane_weights) == 2 and all(w.disc for w in s.plane_weights)
    assert abs(s.mass_unbalance) < 0.02 * abs(hidden.mass_unbalance)
    # The faces are close, cond(α) ≈ 28, so the measured readings' small errors grow in the couple.
    assert abs(s.couple_unbalance) < 0.05 * abs(hidden.couple_unbalance)
    slow = b.readings("slow")
    assert np.all(np.abs(b.readings("verify") - slow) < 0.03 * np.abs(b.readings("found") - slow))
    assert "influence matrix" in b.results.text().lower() and "W<sub>c2</sub>" in b.results.text()
    p.reveal_unbalance()
    assert p.skew.isEnabled() and p.skew.value() == pytest.approx(math.degrees(hidden.skew), abs=1e-3)
    # The planes changed: the trial runs no longer apply.
    b.mode.setCurrentIndex(2)
    assert b.reading("trial") is None and b.reading("found") is not None
    assert not b._buttons["verify"].isEnabled()
    b.remove_weights()
    assert p.sim.system.plane_weights == ()
    b.mode.setCurrentIndex(0)

    # Taking over the speed cancels a measurement.
    b.step_buttons[1].click()
    p.set_target_rpm(700.0)
    assert p.measurement is None and b.step_buttons[1].isEnabled()

    # Stability: internal damping makes it unstable above an onset speed, marked on the Campbell
    # diagram and the stability map; past it a tap grows at the natural frequency until it pauses.
    p.set_system(RotorSystem())
    assert "stable up to 10,000 rpm" in p.crit_label.text() and not p.campbell_plot.onset_line.isVisible()
    p.internal_damping.setValue(20.0)
    p._refresh_analysis()
    assert p.sim.system.internal_damping == 20.0
    assert "unstable above 3,910 rpm" in p.crit_label.text() and "3,910 rpm" in p.stability_note.text()
    assert p.campbell_plot.onset_line.isVisible() and p.campbell_plot.unstable.getData()[0].size > 0
    assert p.stability_map.onset_line.value() == pytest.approx(3910, abs=5)
    p.measure.setCurrentIndex(1)
    assert "Log decrement" in p.stability_map.plot.getAxis("left").labelText
    p.tabs.show_page(p.spectrum.parentWidget())
    p.ramp.setValue(20_000.0)
    p.set_target_rpm(5000.0)
    p.running = True
    for _ in range(20):
        p.step(0.05)
    p.tap()
    for _ in range(200):
        p.step(0.05)
        if p.rubbed:
            break
    assert p.rubbed and not p.running and p.run_button.text() == "Run" and "Paused" in p.readout.text()
    assert "unstable" in p.readout.text() and "heavy spot leads" not in p.readout.text()
    freqs, amps = p.spectrum.curve.getData()
    assert freqs.size and 20 < freqs[np.argmax(amps)] < 35  # forward, at the natural frequency
    p.reset()
    assert not p.rubbed
    # The cross-coupling goes through the panels and round-trips.
    p.bearing_rows[0][5].setValue(1.5e4)
    assert p.sim.system.bearing_a.kxy == 1.5e4 and p.sim.system.bearing_b.kxy == 1.5e4  # B same as A
    system = RotorSystem(bearing_a=Bearing(kxy=-2e3), bearing_b=Bearing(kxy=5e3), internal_damping=7.5)
    p.set_system(system)
    assert p.sim.system == system
    p.tabs.show_page(p.matrices)
    assert "Ω H" in p.matrices.toHtml()


def test_rotor_presets_speed_range_and_overhung_disc(app):
    from vib_tutorial.core.rotor import RPM, RotorSystem
    from vib_tutorial.gui.main_window import MainWindow
    from vib_tutorial.gui.rotor import CUSTOM

    w = MainWindow()
    p = w.rotor_page
    w.pages.setCurrentWidget(p)
    p._timer.stop()
    # The defaults are the classic preset.
    assert p.preset.currentText() == "Classic Jeffcott rotor" and "1584 rpm" in p.preset_note.text()

    # Each preset loads its rotor, speed range and target, from rest, and the panels show it exactly.
    for preset in p.presets:
        p.load_preset(preset.name)
        p._refresh_analysis()
        assert p.preset.currentText() == preset.name
        assert p.top_rpm == preset.top_rpm and p.speed_slider.maximum() == int(preset.top_rpm)
        assert p.sim.target == pytest.approx(preset.target_rpm / RPM) and p.sim.t == 0.0
        assert p.campbell.omega[-1] == pytest.approx(preset.top_rpm / RPM)
        for _ in range(5):
            p.step(0.02)

    # The overhung rotor: the disc past bearing B, drawn beyond it; one forward critical.
    p.load_preset("Overhung rotor")
    p._refresh_analysis()
    assert p.sim.system.overhung and "3EI/(c²(L + c))" in p.shaft_note.text()
    assert "1,284 rpm" in p.crit_label.text()
    disc, bearing_b = p.view._z(p.sim.system.a), p.view._z(p.sim.system.length)
    assert disc > bearing_b and disc == pytest.approx(2.0)  # the overhang's end is the shaft's end
    p.tabs.show_page(p.matrices)
    assert "overhang" in p.matrices.toHtml()
    # Editing a parameter makes it the user's own rotor.
    p.disc_mass.setValue(2.0)
    assert p.preset.currentText() == CUSTOM

    # The position steps over the bearings, and a typed value on one is moved off it.
    p.set_system(RotorSystem(position=0.95))
    p.position.stepBy(1)
    assert p.position.value() == pytest.approx(1.05) and p.sim.system.position == pytest.approx(1.05)
    p.position.setValue(0.0)
    assert p.sim.system.position == pytest.approx(0.02)

    # The speed range: the controls and the analysis follow it, and a target above it comes down.
    p.set_target_rpm(8000.0)
    p.top_speed.setValue(4000.0)
    p._refresh_analysis()
    assert p.target_rpm.value() == 4000.0 and p.sim.target == pytest.approx(4000.0 / RPM)
    assert p.sweep_to.maximum() == 4000.0 and p.campbell.omega[-1] == pytest.approx(4000.0 / RPM)
    p.top_speed.setValue(30_000.0)
    p._refresh_analysis()
    assert p.speed_slider.maximum() == 30_000 and "up to 30,000 rpm" in p.crit_label.text()
    # The order lines are drawn on request.
    assert not p.campbell_plot.orders[2.0].isVisible()
    p.order_boxes[2.0].setChecked(True)
    p.order_boxes[0.5].setChecked(True)
    assert p.campbell_plot.orders[2.0].isVisible() and p.campbell_plot.order_labels[0.5].isVisible()
    x, y = p.campbell_plot.orders[0.5].getData()
    assert x[-1] == 30_000.0 and y[-1] == pytest.approx(0.5 * 30_000.0 / 60.0)

    # The swirl ratio goes through the panel (B follows A) and onto the stability note.
    p.bearing_rows[0][6].setValue(0.5)
    assert p.sim.system.bearing_a.swirl == 0.5 == p.sim.system.bearing_b.swirl
    p._refresh_analysis()
    assert "swirl λ = 0.5" in p.stability_note.text() and "unstable above" in p.stability_note.text()


def test_rotor_waterfall(app):
    from vib_tutorial.core.rotor import RPM
    from vib_tutorial.gui.main_window import MainWindow

    w = MainWindow()
    p = w.rotor_page
    w.pages.setCurrentWidget(p)
    p._timer.stop()
    p.running = False
    p.load_preset("Overhung rotor")
    p._refresh_analysis()
    plot = p.waterfall_plot
    p.tabs.show_page(plot.parentWidget())
    assert "No run-up" in p.waterfall_note.text()
    # Repeated taps through a run-up: a cascade of spectra, one per half block, at rising speeds.
    p.ramp.setValue(1000.0)
    p.repeat_taps.setChecked(True)
    p.set_target_rpm(2000.0)
    while p.sim.omega < p.sim.target:
        p.step(0.05)
    blocks = p.waterfall.latest_ramp(up=True)
    shown = [line for line in plot.lines if line.isVisible()]
    assert len(blocks) >= 6 and len(shown) == len(blocks)
    assert f"{len(blocks)} full spectra of the disc orbit" in p.waterfall_note.text()
    assert "Tap the disc every" not in p.waterfall_note.text()  # already ticked
    speeds = [b.omega * RPM for b in blocks]
    assert speeds == sorted(speeds) and plot.plot.getAxis("left").labelText == "Spin speed"
    x, y = plot.sync.getData()
    assert x == pytest.approx(np.array(speeds) / 60.0) and y == pytest.approx(speeds)
    assert plot.forward.getData()[0].size and plot.backward.getData()[0].size
    # Unticked, the marks go; against time every block is drawn.
    p.wf_marks.setChecked(False)
    assert not plot.sync.getData()[0].size and not plot.forward.getData()[0].size
    p.wf_against.setCurrentIndex(2)
    assert plot.plot.getAxis("left").labelText == "Time"
    assert sum(line.isVisible() for line in plot.lines) == len(p.waterfall.blocks)
    p.wf_scale.setCurrentIndex(1)
    w.set_theme("Dark")
    # New blocks are drawn as they come; a new rotor, block length or Reset starts again.
    n = len(p.waterfall.blocks)
    for _ in range(10):
        p.step(0.05)
    assert len(p.waterfall.blocks) > n
    assert sum(line.isVisible() for line in plot.lines) == len(p.waterfall.blocks)
    p.wf_block.setValue(0.25)
    assert not p.waterfall.blocks and p.waterfall.block == 0.25
    p.step(0.5)
    assert p.waterfall.blocks
    p.disc_mass.setValue(2.0)
    assert not p.waterfall.blocks and p.waterfall.system.disc_mass == 2.0
    p.step(0.5)
    p.reset()
    assert not p.waterfall.blocks and "first 0.25 s block" in p.waterfall_note.text()


def test_rotor_shaft_mass(app):
    from vib_tutorial.core.rotor import RotorSystem
    from vib_tutorial.gui.main_window import MainWindow

    w = MainWindow()
    p = w.rotor_page
    w.pages.setCurrentWidget(p)
    p._timer.stop()
    p.running = False
    p._refresh_analysis()
    assert not p.elements.isEnabled() and not p.campbell_plot.reference[0:1]
    p.set_target_rpm(1500.0)
    p.step(0.2)
    # Ticked: the shaft has mass, the stations keep their motion and the massless model is drawn dashed.
    p.shaft_mass.setChecked(True)
    p._refresh_analysis()
    s = p.sim.system
    assert s.density == 7850.0 and s.n_dof > 8 and p.sim.q.size == s.n_dof
    assert p.elements.isEnabled() and "Shaft 0.308 kg" in p.shaft_note.text()
    assert any(line.isVisible() for line in p.campbell_plot.reference)
    assert "With a massless shaft (the Jeffcott model, dashed): 1,584 rpm" in p.crit_label.text()
    assert "1,458 rpm" in p.crit_label.text()
    p.step(0.2)
    p.tabs.show_page(p.matrices)
    p._refresh_matrices()
    html = p.matrices.toHtml()
    text = p.matrices.toPlainText()
    assert "ξ" in html and "Craig–Bampton" in text and f"A is {2 * s.n_dof} × {2 * s.n_dof}" in text
    assert f"modal coordinates ξ, {s.modes_kept} in each plane" in text
    p.mass_matrix.setCurrentIndex(1)
    assert p.sim.system.lumped
    p.elements.setValue(4)
    assert p.sim.system.nodes.size == 5
    w.set_theme("Dark")
    # The preset, and back to a massless shaft: whatever the mesh, the classic preset again.
    p.load_preset("Long shaft, with its mass")
    assert p.shaft_mass.isChecked() and p.elements.value() == 8 and p.mass_matrix.currentIndex() == 0
    p.step(0.2)
    p.load_preset("Classic Jeffcott rotor")
    assert not p.shaft_mass.isChecked() and p.sim.q.size == 8
    p.elements.setValue(12)
    assert p.preset.currentText() == "Classic Jeffcott rotor"
    p._refresh_analysis()
    assert not any(line.isVisible() for line in p.campbell_plot.reference)
    p.set_system(RotorSystem(density=2700.0, elements=6, lumped=True))
    assert p.density.value() == 2700.0 and p.elements.value() == 6 and p.mass_matrix.currentIndex() == 1


def test_rotor_asymmetric_shaft(app):
    from vib_tutorial.core.rotor import RotorSystem
    from vib_tutorial.gui.main_window import MainWindow

    w = MainWindow()
    p = w.rotor_page
    w.pages.setCurrentWidget(p)
    p._timer.stop()
    p.running = False
    p.load_preset("Two-pole rotor (asymmetric shaft)")
    p._refresh_analysis()
    s = p.sim.system
    assert s.asymmetry == pytest.approx(0.15) and p.asymmetry.value() == pytest.approx(15.0)
    assert "Asymmetric: EI" in p.shaft_note.text()
    assert "unstable from 1,527 to 1,631 rpm" in p.crit_label.text()
    assert "1,491 rpm" in p.crit_label.text() and "1,665 rpm" in p.crit_label.text()
    assert p.campbell_plot.end_line.isVisible() and p.stability_map.end_line.isVisible()
    assert p.stability_map.end_line.value() == pytest.approx(1631, abs=1)
    # The Bode plot tracks 2X: gravity's steady state peaks at the gravity critical.
    p.probe_direction.setCurrentIndex(2)
    p.probe_order.setCurrentIndex(1)
    p.sweep_from.setValue(700.0)
    p.sweep_to.setValue(850.0)
    assert "2X amplitude" in p.bode.amp.getAxis("left").labelText
    text = p.sweep_summary.text()
    assert "peak 447 µm at 785 rpm" in text and "critical speed 1,491 rpm" in text and "gravity critical" in text
    p.ramp.setValue(2000.0)
    p.start_sweep()
    for _ in range(400):
        if p.run_up is None:
            break
        p.step(0.05)
    assert p.run_up is None and len(p.sweeps) == 2 and len(p.sweeps[0].vectors2) == len(p.sweeps[0])
    assert "2X peak" in p.sweep_summary.text()
    # The 3D view draws the stiff axis; the readout drops the heavy spot's lag, which 2X swamps.
    assert p.view.stiff_axis.getData()[0].size == 2
    assert "heavy spot leads" not in p.readout.text()
    # The asymmetry chart fills in while the stability tab shows it.
    p.tabs.show_page(p.spectrum.parentWidget())
    p.measure.setCurrentIndex(2)
    assert p.stability_stack.currentWidget() is p.asymmetry_plot and p.asym_chart is not None
    chart = p.asym_chart
    for _ in range(5):
        p._draw()
    assert chart._next > 0 and "Working it out" in p.stability_note.text()
    p.asymmetry.setValue(10.0)  # only the asymmetry: the same chart
    p._refresh_analysis()
    p._draw()
    assert p.asym_chart is chart and p.asymmetry_plot.here[1] == pytest.approx(0.1)
    chart.run()
    p._draw()
    assert p.asymmetry_plot.drawn == chart.growth.size and "Working it out" not in p.stability_note.text()
    p.disc_mass.setValue(1.2)  # a new rotor: a new chart
    p._refresh_analysis()
    assert p.asym_chart is not chart
    w.set_theme("Dark")
    p.measure.setCurrentIndex(0)
    assert p.stability_stack.currentWidget() is p.stability_map
    # The matrices tab shows the turning stiffness.
    p.tabs.show_page(p.matrices)
    p._refresh_matrices()
    text = p.matrices.toPlainText()
    assert "ΔK(φ)" in text and "Floquet" in text
    p.set_system(RotorSystem())
    p._refresh_analysis()
    p._draw()
    assert p.asymmetry.value() == 0.0 and not len(p.view.stiff_axis.getData()[0])
    assert p.probe_order.currentIndex() == 1
    assert "nothing drives 2X on a round shaft" in p.sweep_summary.text()


def test_rotor_cracked_shaft(app):
    from vib_tutorial.core.rotor import RotorSystem
    from vib_tutorial.gui.main_window import MainWindow

    w = MainWindow()
    p = w.rotor_page
    w.pages.setCurrentWidget(p)
    p._timer.stop()
    p.running = False
    p.load_preset("Cracked shaft (breathing crack)")
    p._refresh_analysis()
    s = p.sim.system
    assert p.preset.currentText() == "Cracked shaft (breathing crack)"
    assert s.crack.depth == pytest.approx(0.4) and s.crack.breathing and p.crack_depth.value() == pytest.approx(40.0)
    assert "Crack 40% deep" in p.shaft_note.text() and "opened by the weight" in p.shaft_note.text()
    assert "1,562 rpm" in p.crit_label.text() and "1,574 rpm" in p.crit_label.text()
    # The Bode plot tracks 3X: the weight drives it most near a third of the critical speed.
    p.probe_direction.setCurrentIndex(2)
    p.probe_order.setCurrentIndex(2)
    p.sweep_from.setValue(450.0)
    p.sweep_to.setValue(600.0)
    assert "3X amplitude" in p.bode.amp.getAxis("left").labelText
    text = p.sweep_summary.text()
    assert "peak 18.6 µm at 522 rpm" in text and "a third of the critical speed 1,562 rpm" in text
    p.ramp.setValue(2000.0)
    p.start_sweep()
    for _ in range(400):
        if p.run_up is None:
            break
        p.step(0.05)
    assert p.run_up is None and len(p.sweeps) == 2 and len(p.sweeps[0].vectors3) == len(p.sweeps[0])
    assert "3X peak" in p.sweep_summary.text()
    # The 3D view draws the crack; the spectrum marks +2X and +3X.
    p._draw()
    assert p.view.crack_mark.getData()[0].size == 2
    assert "heavy spot leads" not in p.readout.text()
    p.tabs.show_page(p.spectrum.parentWidget())
    p._draw_spectrum()
    assert all(line.isVisible() for line in p.spectrum.sync)
    w.set_theme("Dark")
    # The matrices tab explains the crack.
    p.tabs.show_page(p.matrices)
    p._refresh_matrices()
    text = p.matrices.toPlainText()
    assert "The crack" in text and "It breathes" in text and "Floquet" in text
    # Held open: a softened asymmetric shaft, no 3X.
    p.crack_breathing.setChecked(False)
    p._refresh_analysis()
    assert not p.sim.system.crack.breathing and p.sim.system.spacing == 2
    assert "nothing drives 3X on an asymmetric shaft" in p.sweep_summary.text()
    # A vertical rotor: nothing opens a breathing crack.
    p.crack_breathing.setChecked(True)
    p.gravity.setChecked(False)
    p._refresh_analysis()
    assert not p.sim.system.periodic and "but shut" in p.shaft_note.text()
    # A crack is kept inside the shaft, and depth 0 takes it away.
    p.crack_position.setValue(1.5)
    assert p.sim.system.crack.position < 1.0
    p.crack_depth.setValue(0.0)
    assert p.sim.system.crack is None and not p.crack_angle.isEnabled()
    p.set_system(RotorSystem())
    p._refresh_analysis()
    p._draw()
    assert not len(p.view.crack_mark.getData()[0])


def test_rotor_hidden_tabs_draw_when_shown(app, monkeypatch):
    """The Balancing and bearing tabs draw only while they show: a preset loaded meanwhile is drawn when
    the tab shows, once, while the state they keep (the load box, the buttons) follows at once."""
    from vib_tutorial.gui.main_window import MainWindow
    from vib_tutorial.gui.rotor_balance import BalancingPanel
    from vib_tutorial.gui.rotor_bearing import BearingPanel

    w = MainWindow()
    p = w.rotor_page
    w.pages.setCurrentWidget(p)
    p._timer.stop()
    p.running = False
    drawn = []
    for cls, name in ((BearingPanel, "_draw_locus"), (BalancingPanel, "_draw_curves")):
        draw = getattr(cls, name)
        monkeypatch.setattr(cls, name, lambda self, *a, draw=draw, name=name: (drawn.append(name), draw(self, *a)))
    p.load_preset("Journal bearings (oil whip)")
    p._refresh_analysis()
    s = p.sim.system
    assert not drawn and p.bearing_panel._stale and p.balancing._stale
    assert p.bearing_panel.load.value() == pytest.approx(abs(s.bearing_loads[0]), abs=0.01)
    p.tabs.show_page(p.bearing_panel)
    assert drawn == ["_draw_locus"] and not p.bearing_panel._stale and "At rest" in p.bearing_panel.readout.text()
    p.tabs.show_page(p.balancing)
    assert drawn == ["_draw_locus", "_draw_curves"] and not p.balancing._stale
    p.tabs.show_page(p.bearing_panel)  # nothing changed meanwhile
    assert drawn == ["_draw_locus", "_draw_curves"] and p.tabs.current_page() is p.bearing_panel
    # Another group in front: the bearing tab is still its own group's current one, so not isHidden().
    p.tabs.show_page(p.campbell_plot.parentWidget())
    assert not p.bearing_panel.isHidden() and p.tabs.current_page() is not p.bearing_panel
    p.load_preset("Classic Jeffcott rotor")
    p._refresh_analysis()
    assert drawn == ["_draw_locus", "_draw_curves"] and p.bearing_panel._stale
    p.tabs.show_page(p.bearing_panel)
    assert drawn[-1] == "_draw_locus" and not p.bearing_panel._stale
    w.close()


def test_window_fits_a_1280_px_screen(app, request):
    """No page holds the window wider than 1280 px or taller than 720, even with its widest controls showing."""
    from vib_tutorial.gui.main_window import MainWindow
    from vib_tutorial.gui.modes import Method

    request.addfinalizer(lambda font=app.font(): app.setFont(font))
    _windows_font(app)
    w = MainWindow()
    for i in range(w.pages.count()):
        w.pages.setCurrentIndex(i)  # builds each page
    w.method_combo.setCurrentIndex(list(Method).index(Method.STATE_SPACE))  # the complex plane beside the shapes
    p = w.rotor_page
    p._timer.stop()
    p.balancing.mode.setCurrentIndex(2)  # two planes on the shaft: two probes, five steps
    p.orbits.set_midspan_shown(True)  # four orbits
    app.processEvents()
    size = w.minimumSizeHint()
    assert size.width() <= 1280 and size.height() <= 720, _widest(w.pages, size.width() // 3) + "\n" + _font_report(app)
    names = [p.tabs.tabText(i) for i in range(p.tabs.count())]
    bar = p.tabs.tabBar()
    assert sum(bar.tabSizeHint(i).width() for i in range(bar.count())) < p.tabs.minimumSizeHint().width(), names
    w.close()


def _windows_font(app) -> None:
    """Measure in Segoe UI 9 pt, as on a Windows desktop, where the offscreen platform has no fonts.

    It finds none on Windows and draws boxes ~1.8x as wide as the text, so every row of
    controls would measure far wider than the app's on screen. Qt keeps 96 logical dpi
    under Windows' display scaling (the scale goes in the pixel ratio), so 1280 here is
    1280 logical pixels there: a 1920 px screen at 150%.
    """
    from PySide6 import QtGui

    if QtGui.QFontInfo(app.font()).family() or sys.platform != "win32":
        return
    QtGui.QFontDatabase.addApplicationFont(os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "segoeui.ttf"))
    app.setFont(QtGui.QFont("Segoe UI", 9))


def _font_report(app) -> str:
    from PySide6 import QtGui

    font = app.font()
    screen = app.primaryScreen()
    sample = QtGui.QFontMetrics(font).horizontalAdvance("Number of masses:")
    return (f"font {font.family()!r} {font.pointSizeF()} pt / {font.pixelSize()} px, resolved "
            f"{QtGui.QFontInfo(font).family()!r}; screen dpi {screen.logicalDotsPerInch()} "
            f"(physical {screen.physicalDotsPerInch()}), ratio {screen.devicePixelRatio()}; "
            f"'Number of masses:' {sample} px; platform {app.platformName()}, style {app.style().name()}")


def _least_width(widget) -> int:
    return max(widget.minimumSizeHint().width(), widget.minimumWidth())


def _widest(widget, floor: int, depth: int = 0) -> str:
    """What holds `widget` wide: each child at least `floor` px wide at its least, down the tree."""
    lines = []
    for child in widget.findChildren(QtWidgets.QWidget, options=QtCore.Qt.FindChildOption.FindDirectChildrenOnly):
        least = _least_width(child)
        if least >= floor and not child.isWindow():
            text = getattr(child, "text", None)
            label = f" {text()[:30]!r}" if callable(text) and isinstance(text(), str) else ""
            lines.append(f"{'  ' * depth}{type(child).__name__}{label}: {least} (hint {child.sizeHint().width()})")
            if depth < 12:
                lines.append(_widest(child, floor, depth + 1))
    return "\n".join(line for line in lines if line)


def test_flow_layout_wraps_when_narrow(app):
    from vib_tutorial.gui.layouts import FlowLayout, group

    holder = QtWidgets.QWidget()
    row = FlowLayout(holder)
    row.add(*(QtWidgets.QPushButton(f"Button number {i}") for i in range(3)), group("Label:", QtWidgets.QSpinBox()))
    items = [row.itemAt(i) for i in range(row.count())]
    widths = [i.sizeHint().width() for i in items]
    line = max(i.sizeHint().height() for i in items)
    assert row.minimumSize().width() == max(i.minimumSize().width() for i in items) < sum(widths)
    assert row.heightForWidth(row.sizeHint().width()) == line  # room for all: one line
    assert row.heightForWidth(max(widths)) > 3 * line  # room for one: one each
    holder.resize(max(widths), row.heightForWidth(max(widths)))
    row.setGeometry(holder.rect())
    assert all(i.geometry().right() <= holder.width() for i in items)
    assert len({i.geometry().y() for i in items}) == 4


def test_rotor_fluid_film_bearing(app):
    from vib_tutorial.core.fluid_bearing import JournalBearing
    from vib_tutorial.gui.main_window import MainWindow

    w = MainWindow()
    p = w.rotor_page
    w.pages.setCurrentWidget(p)
    p._timer.stop()
    p.running = False
    panel = p.bearing_panel
    p.tabs.show_page(panel)
    app.processEvents()
    assert panel.bearing() == JournalBearing()
    assert "At rest" in panel.readout.text() and "21,3" in panel.readout.text()  # stable over the range
    assert not panel.unstable_path.getData()[0].size
    # At speed: the readings follow the rotor, the dot moves off the bottom, the speed line is drawn.
    p.set_target_rpm(3000.0)
    while p.sim.omega < p.sim.target:
        p.step(0.5)
    p._draw()
    assert "At 3,000 rpm" in panel.readout.text() and "= 0.50: the swirl ratio λ" in panel.readout.text()
    x, y = panel.dot.getData()
    assert x[0] > 0.0 and -1.0 < y[0] < 0.0
    assert panel.speed_lines[0].value() == pytest.approx(3000.0)
    # A looser bearing on a wider range: a rigid rotor whirls inside it, drawn red on the locus.
    panel.set_bearing(JournalBearing(clearance=60e-6))
    p.set_top_rpm(20_000.0)
    assert panel.unstable_path.getData()[0].size and "whirls unstably" in panel.readout.text()
    # Against S(L/D)²: log scales, dashed negative parts, the speed line at log10(S(L/D)²).
    panel.against.setCurrentIndex(1)
    assert panel.k_plot.ctrl.logXCheck.isChecked()
    s = panel.bearing().sommerfeld(3000.0 / 60.0 * 2.0 * 3.141592653589793) * 0.5**2
    assert panel.speed_lines[1].value() == pytest.approx(math.log10(s))
    pos, neg = panel.k_curves[2]  # k_yx: negative throughout
    assert np.isnan(pos.getData()[1]).all() and np.isfinite(neg.getData()[1]).all()
    panel.against.setCurrentIndex(0)
    assert not panel.k_plot.ctrl.logXCheck.isChecked()
    # A long bearing is flagged.
    panel.length.setValue(20.0)
    assert "L<sub>b</sub>/D<sub>b</sub> = 2" in panel.note.text()
    w.set_theme("Dark")


def test_rotor_on_oil_film_bearings(app):
    from vib_tutorial.core.fluid_bearing import JournalBearing
    from vib_tutorial.core.rotor import RPM
    from vib_tutorial.gui.main_window import MainWindow
    from vib_tutorial.gui.rotor import CUSTOM, FILMS, SPRINGS

    w = MainWindow()
    p = w.rotor_page
    w.pages.setCurrentWidget(p)
    p._timer.stop()
    p.running = False
    panel = p.bearing_panel
    # The preset runs on the tab's bearing, each journal under its share of the weight.
    p.load_preset("Journal bearings (oil whip)")
    p._refresh_analysis()
    s = p.sim.system
    assert s.films and p.supports.currentIndex() == FILMS and p.preset.currentText() == "Journal bearings (oil whip)"
    assert s.whole_films and p.nonlinear_film.isChecked() and p.nonlinear_film.isEnabled()
    row = p.bearing_rows[0]
    assert row[4].isEnabled() and not any(row[c].isEnabled() for c in (0, 1, 2, 3, 5, 6))
    assert panel.loads == s.bearing_loads and not panel.load.isEnabled() and panel.which.isVisibleTo(panel)
    assert panel.load.value() == pytest.approx(abs(s.bearing_loads[0]), abs=0.01)
    assert panel.onset is not None and round(panel.onset.omega * RPM) == 3576
    p.tabs.show_page(panel)
    p._draw()
    assert "goes unstable at <b>3,576 rpm</b>" in panel.readout.text()
    assert "oil-film bearings" in p._stability_note()
    p.tabs.show_page(p.matrices)
    assert "The oil films" in p.matrices.toHtml()
    # Changing the bearing on the tab changes the rotor's (and makes it the user's own rotor).
    panel.clearance.setValue(20.0)
    assert p.sim.system.bearing_a.film.clearance == pytest.approx(20e-6)
    assert p.preset.currentText() == CUSTOM
    panel.set_bearing(JournalBearing())
    assert p.preset.currentText() == "Journal bearings (oil whip)"
    # A frame's budget caps how far a slow simulation advances: the page falls behind instead.
    p._cost = 1.0  # s of computing per simulated s
    t0 = p.sim.t
    p.step(0.1, budget=0.004)
    assert 0.0 < p.sim.t - t0 <= 0.004 + 1e-3
    # Linearized, past the onset the whirl grows until a journal reaches its clearance, and the page pauses.
    p.nonlinear_film.setChecked(False)
    assert not p.sim.system.whole_films and p.sim.system.films
    p.reset()
    p.running = True
    p.ramp.setValue(5000.0)
    p.set_target_rpm(5000.0)
    for _ in range(400):
        p.step(0.1)
        if not p.running:
            break
    assert p.rubbed and p.touched is not None and p.touched[1] == pytest.approx(15e-6)
    assert "reached the wall of its bearing" in p.readout.text() and "Nonlinear film" in p.readout.text()
    # A turning shaft keeps the films linear.
    p.nonlinear_film.setChecked(True)
    p.asymmetry.setValue(5.0)
    assert not p.nonlinear_film.isEnabled() and not p.sim.system.whole_films
    p.asymmetry.setValue(0.0)
    assert p.nonlinear_film.isEnabled() and p.sim.system.whole_films
    # Unticked, the rotor is back on its springs and the tab has its own load again.
    p.supports.setCurrentIndex(SPRINGS)
    assert not p.sim.system.films and panel.loads is None and panel.load.isEnabled()
    assert not p.nonlinear_film.isEnabled() and not p.nonlinear_film.isVisibleTo(p)
    assert all(row[c].isEnabled() for c in (0, 2, 4, 5, 6))  # (k_y and c_y follow k_x and c_x: isotropic)
    w.set_theme("Dark")


def test_rotor_on_magnetic_bearings(app):
    from dataclasses import replace

    from vib_tutorial.core.magnetic_bearing import MagneticBearing
    from vib_tutorial.core.rotor import RPM, XA, XB, YA, YB
    from vib_tutorial.gui.main_window import MainWindow
    from vib_tutorial.gui.rotor import CUSTOM, MAGNETS, SPRINGS
    from vib_tutorial.gui.rotor_notes import matrices_html

    w = MainWindow()
    p = w.rotor_page
    w.pages.setCurrentWidget(p)
    p._timer.stop()
    p.running = False
    panel = p.magnet_panel
    assert panel.bearing() == MagneticBearing() and panel.system is None
    assert panel.note.text() == ""  # a tab not showing draws when it shows
    p.tabs.show_page(panel)
    assert "Choose <i>Magnetic bearings</i>" in panel.note.text()
    p.tabs.setCurrentIndex(0)  # hidden again while the preset loads
    # The preset floats on the tab's bearing (its amplifier slowed), the journals held at the centre.
    p.load_preset("Magnetic bearings")
    p._refresh_analysis()
    s = p.sim.system
    assert s.magnets and p.supports.currentIndex() == MAGNETS and p.preset.currentText() == "Magnetic bearings"
    assert panel.bearing() == s.bearing_a.magnet and panel.amplifier.value() == 400.0
    assert not p.nonlinear_film.isVisibleTo(p)
    row = p.bearing_rows[0]
    assert row[4].isEnabled() and not any(row[c].isEnabled() for c in (0, 1, 2, 3, 5, 6))
    assert abs(p.sim.q[[XA, YA, XB, YB]]).max() < 1e-12 and p.sim.control.size == s.n_control
    assert panel.onset is not None and round(panel.onset.omega * RPM) == 8894
    p.tabs.show_page(panel)
    p._draw()
    text = panel.readout.text()
    assert "goes unstable at <b>8,894 rpm</b>" in text and "0.61 A at B" in text and "61% of the bias" in text
    assert "It damps from 15.7 to 739.2 Hz" in text
    assert len(panel.mode_lines[0]) == len(panel.mode_lines[1]) >= 6  # the rotor's modes at rest
    assert sum(b.isVisible() for b in panel.bands) == 4  # below and above the band, on both plots
    assert "magnetic bearings" in p._stability_note()
    html = matrices_html(p.sim.system, p.sim.omega)
    assert "The magnetic bearings" in html and "F<sub>c</sub>" in html and "journals at the centre" in html
    assert "the journals at (0, 0) µm vertically" in html
    # Changing the bearing on the tab changes the rotor's (and makes it the user's own rotor).
    panel.amplifier.setValue(1000.0)
    assert p.sim.system.bearing_a.magnet.amplifier_hz == 1000.0 and p.preset.currentText() == CUSTOM
    p._refresh_analysis()
    assert panel.onset is None and "stays stable up to 20,000 rpm" in panel.readout.text()
    panel.set_bearing(replace(MagneticBearing(), amplifier_hz=400.0))
    assert p.preset.currentText() == "Magnetic bearings"
    # Too weak a magnet: flagged on the tab.
    panel.kp.setValue(900.0)
    assert "the magnets outpull the controller" in panel.readout.text()
    panel.kp.setValue(MagneticBearing().kp)
    # Past the onset the forward tilting mode grows until a journal lands on its backup bearing, and the page pauses.
    p._refresh_analysis()
    p.reset()
    p.running = True
    p.ramp.setValue(5000.0)
    p.set_target_rpm(12_000.0)
    p.tap()
    for _ in range(600):
        p.step(0.1)
        if not p.running:
            break
    assert p.rubbed and p.touched is not None and p.touched[1] == pytest.approx(0.5e-3)
    assert "landed on its backup bearing" in p.readout.text()
    # Back on springs: the tab forgets the rotor.
    p.supports.setCurrentIndex(SPRINGS)
    assert not p.sim.system.magnets and panel.system is None and p.sim.control.size == 0
    assert all(row[c].isEnabled() for c in (0, 2, 4, 5, 6))
    w.set_theme("Dark")


def test_rotor_rub(app):
    from vib_tutorial.core.rotor import Stator
    from vib_tutorial.gui.main_window import MainWindow
    from vib_tutorial.gui.rotor import CUSTOM, FILMS, PUSH

    w = MainWindow()
    p = w.rotor_page
    w.pages.setCurrentWidget(p)
    p._timer.stop()
    panel = p.rub_panel
    assert p.sim.system.stator is None and not panel.clearance.isEnabled()
    # Ticking the ring puts it on the rotor (the user's own rotor), drawn round the disc.
    panel.enabled.setChecked(True)
    assert p.sim.system.stator == Stator() and p.preset.currentText() == CUSTOM
    assert "At rest it has 200 µm to the ring" in panel.readout.text() and "227 Hz" in panel.readout.text()
    panel.friction.setValue(0.2)
    assert p.sim.system.stator.friction == 0.2
    # A turning shaft, or the oil films' whole force, can't have one.
    p.asymmetry.setValue(10.0)
    assert p.sim.system.stator is None and not panel.enabled.isEnabled()
    p.asymmetry.setValue(0.0)
    assert p.sim.system.stator is not None
    p.supports.setCurrentIndex(FILMS)
    assert p.nonlinear_film.isChecked() and p.sim.system.stator is None
    p.nonlinear_film.setChecked(False)
    assert p.sim.system.stator is not None and p.sim.system.films
    # The preset: rubbing under the sag; the push plotted while the tab shows.
    p.load_preset("Partial rub (½X)")
    assert panel.enabled.isChecked() and panel.clearance.value() == 385.0 and p.preset.currentText() == "Partial rub (½X)"
    assert "sags 378 µm" in panel.readout.text() and "leaving 7.06 µm" in panel.readout.text()
    p.tabs.show_page(panel)
    p._catch_up_tab()
    while p.sim.t < 6.0:
        p.step(0.25)
    t, hist = p.history.window(p.window.value())
    assert hist[:, PUSH].max() > 0.0
    assert panel.curve.xData is not None and panel.curve.xData.size == t.size
    assert "In contact" in panel.contact.text()
    assert "rubbing the stator ring" in p.readout.text()
    assert p.view.ring is not None and p.view.stator_ring.xData.size and p.orbits.ring.xData.size
    p.show_sag.setChecked(False)  # drawn about the sag: the ring moves up by it
    p._draw()
    assert p.view.ring[1] > 300e-6
    assert not panel.spiral.isVisibleTo(panel)  # no thermal bow
    # Off again: nothing drawn.
    panel.enabled.setChecked(False)
    p._draw()
    assert p.view.ring is None and (p.orbits.ring.xData is None or not p.orbits.ring.xData.size)
    # The Newkirk preset: the spiral is recorded once the speed settles, and grows.
    p.load_preset("Newkirk effect (thermal bow)")
    assert panel.thermal_gain.value() == 120.0 and panel.stiffness.value() == 20.0 and p.sim.thermal
    assert panel.spiral.isVisibleTo(panel)
    while p.sim.t < 2.0:
        p.step(0.25)
    assert p._spiral == ([], [])  # still ramping
    while p.sim.t < 12.0:
        p.step(0.25)
    vectors, bows = (np.array(a) for a in p._spiral)
    assert len(vectors) > 30 and abs(vectors[-1]) > abs(vectors[0]) and abs(bows[-1]) > 1e-6
    p._draw()
    assert panel.vector_curve.xData.size == len(vectors) and "thermal bow" in panel.spiral_note.text()
    p.set_target_rpm(1400.0)
    p.step(0.05)
    assert p._spiral == ([], [])  # a new speed starts it afresh
    p.reset()
    assert p.sim.thermal_bow == 0
    panel.thermal_gain.setValue(0.0)
    assert not p.sim.thermal and not panel.spiral.isVisibleTo(panel)
    w.set_theme("Dark")


def test_rotor_theory_figures_follow_the_theme(app):
    # The Theory's figures are SVG drawn from the theme's colours, rendered as the notes are laid out.
    from PySide6 import QtCore, QtGui

    from vib_tutorial.gui import rotor_figures, style
    from vib_tutorial.gui.rotor_notes import THEORY_HTML

    assert all(f'src="{name}"' in THEORY_HTML for name in rotor_figures.FIGURES)
    browser = rotor_figures.FigureBrowser()
    before = style.THEMES[style.colors.name]
    try:
        for theme in style.THEMES.values():
            style.set_theme(theme)
            for name, build in rotor_figures.FIGURES.items():
                svg = build()
                assert theme.background in svg and theme.foreground in svg and theme.mass[0] in svg
                image = browser.loadResource(QtGui.QTextDocument.ResourceType.ImageResource, QtCore.QUrl(name))
                assert isinstance(image, QtGui.QImage) and not image.isNull()
                assert image.deviceIndependentSize().width() > 400
                # Drawn, not blank: the background's colour and others.
                assert len({image.pixel(x, y) for x in range(0, image.width(), 7) for y in range(0, image.height(), 7)}) > 5
    finally:
        style.set_theme(before)
        browser.deleteLater()


def test_frame_stats_summarize_the_frames():
    from vib_tutorial.gui.frame_stats import FrameStats

    stats = FrameStats()
    for i in range(11):  # every 20 ms, 4 ms of computing; the 5th starts 40 ms late
        start = 0.02 * i + (0.04 if i >= 5 else 0.0)
        stats.record("Page", start, 0.004, sim_t=0.01 * i, target=0.016)
        stats.paint(0.001)
        stats.paint(0.002)  # every repaint until the next frame counts
    s = stats.summary()
    assert s["frames"] == 10  # the last frame's paint is still to come
    assert s["work_med"] == pytest.approx(4.0) and s["paint_med"] == pytest.approx(3.0)
    assert s["late"] == pytest.approx(1 / 9)  # 60 ms > 1.5 x 16 ms; the first has no interval
    assert s["fps"] == pytest.approx(9 / 0.22)
    assert s["sim_speed"] == pytest.approx(0.09 / 0.22)
    stats.record("Page", 10.0, 0.004, None, 0.016)  # after a pause: no interval
    assert np.isnan(stats.frames["Page"][-1].interval)


def test_frame_stats_overlay_is_off_unless_asked_for(app, monkeypatch, tmp_path):
    import csv
    import time

    from vib_tutorial.gui import frame_stats
    from vib_tutorial.gui.main_window import MainWindow

    monkeypatch.delenv(frame_stats.ENV, raising=False)
    w = MainWindow()
    assert w.frame_overlay is None and frame_stats.STATS is None
    w._timer.stop()  # or its frames would count with the next window's
    w.close()

    log = tmp_path / "frames.csv"
    monkeypatch.setenv(frame_stats.ENV, str(log))
    w = MainWindow()
    try:
        w._timer.stop()
        w.show()
        for _ in range(6):
            time.sleep(0.02)  # a frame's worth of time, so there's something new to draw
            w._frame()
            w._timer.stop()
            for _ in range(2):  # the repaint the frame asked for (posted while the first pass runs)
                app.processEvents()
        w.frame_overlay.refresh()
        text = w.frame_overlay.text()
        assert text.startswith("Simulation") and "fps" in text and "sim speed" in text
        rows = list(csv.DictReader(log.open(encoding="utf-8")))
        frames = [r for r in rows if r["kind"] == "frame"]
        assert len(frames) >= 4
        assert all(float(r["paint_ms"]) > 0 for r in frames[1:])  # each frame's repaint was timed
        view = next(r for r in rows if r["kind"] == "view")
        assert view["detail"].startswith("Simulation & modal analysis > Modal analysis | ")

        # What the user does is logged between the frames, with what the control is called.
        from PySide6.QtTest import QTest

        box = next(b for b in w.params.rows[0] if isinstance(b, QtWidgets.QAbstractSpinBox))
        box.setFocus()
        QTest.keyClick(box, QtCore.Qt.Key.Key_Up)
        bar = w.pages.tabBar()
        rotor = next(i for i in range(w.pages.count()) if "rotor" in w.pages.tabText(i).lower())
        QTest.mouseClick(bar, QtCore.Qt.MouseButton.LeftButton, pos=bar.tabRect(rotor).center())
        for _ in range(3):
            app.processEvents()
        page = w.pages.currentWidget()
        QTest.mouseClick(page.gravity, QtCore.Qt.MouseButton.LeftButton, pos=QtCore.QPoint(8, 8))
        app.processEvents()
        w.frame_overlay.refresh()
        rows = list(csv.DictReader(log.open(encoding="utf-8")))
        inputs = [r["detail"] for r in rows if r["kind"] == "input"]
        assert inputs[0].startswith("1 m [kg]: ") and inputs[1:] == ["tab: Jeffcott rotor", "Gravity: on"]
        assert rows[-1]["kind"] == "view" and rows[-1]["detail"].startswith("Jeffcott rotor > [Custom] > ")
        # A combo alone in its group is named by its tooltip, not by a label outside the group;
        # scrolling a panel isn't an input.
        assert frame_stats.describe_input(page.supports, "click") == ("What holds the journals",
                                                                     "Springs and dampers")
        bars = w.findChildren(QtWidgets.QScrollBar)
        assert bars and all(w.frame_overlay.inputs._control(b) is None for b in bars)
    finally:
        frame_stats.uninstall()
        w.close()


def test_rotor_plots_paint_once_per_frame(app):
    """The orbits rescale and the strip charts scroll every frame: each paints once per frame, not twice
    (a range applied at the start of the paint marked the items dirty again, and a narrower tick label
    re-laid the orbits out). And the Bode plot, whose marker stands still at a steady speed, doesn't
    repaint."""
    import time

    import pyqtgraph as pg

    from vib_tutorial.core.rotor import RPM
    from vib_tutorial.gui.main_window import MainWindow

    class Paints(QtCore.QObject):
        n = 0

        def eventFilter(self, obj, event):  # noqa: N802 (Qt override)
            if event.type() == QtCore.QEvent.Type.Paint:
                self.n += 1
            return False

    def run(seconds):
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            app.processEvents()
            time.sleep(0.001)

    w = MainWindow()
    w.resize(1700, 900)
    w.show()
    page = w.rotor_page
    w.pages.setCurrentWidget(page)
    page._refresh_analysis()
    page.tabs.show_page(page.bode.parentWidget())
    page.sim.omega = page.sim.target = page.target_rpm.value() / RPM  # at speed: the marker stands still
    run(1.0)
    # Building the page also handed the analysis to the worker, which can take seconds to spawn (Windows):
    # let its result arrive (and redraw the Bode plot) before counting.
    end = time.perf_counter() + 60.0
    while (page._job is not None or page._analysis_timer.isActive()) and time.perf_counter() < end:
        run(0.05)
    run(0.2)
    views = {type(v).__name__: v for v in page.findChildren(pg.GraphicsView) if v.isVisible()}
    counts = {name: Paints() for name in ("OrbitPlots", "TimePlots", "BodePlots")}
    for name, c in counts.items():
        views[name].viewport().installEventFilter(c)
    frames = Paints()
    page._timer.timeout.connect(lambda: setattr(frames, "n", frames.n + 1))
    run(1.5)
    page._timer.stop()
    assert frames.n > 10
    for name in ("OrbitPlots", "TimePlots"):
        assert counts[name].n <= frames.n + 1, name
    assert counts["BodePlots"].n == 0
    w.close()


def test_frame_timers_keep_the_frame_rate(app, monkeypatch):
    """The rotor and substructuring pages start their next frame FRAME_MS after the last one began: a
    frame's computing comes out of the wait, not on top of it (16 ms + its work held the rotor at 47 fps)."""
    import time

    from vib_tutorial.gui import cms_time, continuum, duffing, rotor, torsional
    from vib_tutorial.gui.main_window import MainWindow

    w = MainWindow()
    w.show()
    w.pages.setCurrentWidget(w.cms_page)
    w.cms_page.tabs.setCurrentWidget(w.cms_page.time)
    app.processEvents()
    for p, frame_ms in ((w.cms_page.time, cms_time.FRAME_MS), (w.rotor_page, rotor.FRAME_MS),
                        (w.duffing_page, duffing.FRAME_MS), (w.torsional_page, torsional.FRAME_MS),
                        (w.continuum_page, continuum.FRAME_MS)):
        w.pages.setCurrentWidget(w.cms_page if p is w.cms_page.time else p)
        app.processEvents()
        if p is w.torsional_page:
            p.start()  # it plays only once started
        if p is w.continuum_page:
            p.views.setCurrentWidget(p.waves_tab)  # it plays only on its Waves tab
        monkeypatch.setattr(p, "step", lambda *_, **__: time.sleep(0.008))
        p._frame()
        assert p._timer.isActive()
        assert p._timer.remainingTime() <= frame_ms - 7
    w.close()


def test_rotor_analysis_runs_in_the_worker(app, monkeypatch):
    """A parameter change hands the analysis to the worker process, and the page shows it when it's done;
    a change while one runs supersedes it."""
    import time

    from vib_tutorial.core.rotor_analysis import analyse
    from vib_tutorial.gui import rotor as rotor_page

    p = rotor_page.RotorPage()

    def settle():
        p._analysis_timer.stop()
        p._start_analysis()
        end = time.perf_counter() + 60.0
        while p._job is not None and time.perf_counter() < end:
            app.processEvents()
            time.sleep(0.01)
        assert p._job is None

    p.gravity.setChecked(not p.gravity.isChecked())
    shown = p.campbell
    settle()
    assert p.campbell is not shown
    expected = analyse(p.sim.system, p.top_rpm)
    assert p.campbell.criticals == expected.campbell.criticals
    assert np.allclose(p._steady_vectors, expected.steady_orders[0], rtol=1e-12, atol=1e-18)

    # Superseded: the worker drops the old job, and the page shows the rotor it has now.
    monkeypatch.setattr(rotor_page, "RESTART_S", 0.0)
    p.unbalance.setValue(p.unbalance.value() * 2.0)
    p._analysis_timer.stop()
    p._start_analysis()
    first = p._job[2]
    p.gravity.toggle()
    settle()
    assert rotor_page.WORKER.lost(first)
    assert p.campbell.criticals == analyse(p.sim.system, p.top_rpm).campbell.criticals


def test_simulation_page_steps_friction_beside_the_frame(app):
    """With friction the steps run in Rust on a worker thread: a frame starts them as it ends (so they run
    while it paints) and the next records them. A parameter edit records the steps in flight first, under
    the parameters they ran with; a reset drops them."""
    import time

    from vib_tutorial.core import native
    from vib_tutorial.core.forcing import ForceController
    from vib_tutorial.core.simulator import Simulator
    from vib_tutorial.gui.main_window import MainWindow

    if native.module is None:
        pytest.skip("Rust kernels not loaded")
    w = MainWindow()
    w.pages.setCurrentWidget(w.sim_page)
    w.params.rows[0][4].setValue(0.3)
    w.params._copy_first_row()
    system = w.sim.system
    w.force_panel.button.click()
    w.running = True
    w._sim_target = 0.1
    w._tick()
    assert w._started is not None and w.history.size == 0
    t = w.sim.t
    while not w.sim.ready:
        time.sleep(1e-3)
    w.running = False
    w._tick()  # records them, and starts no more
    w.running = True
    assert w._started is None and w.history.size == round(t / w.sim.step_size())

    ref = Simulator(system, ForceController(w.force.settings))
    ref.force.switch_on()
    want = ref.advance(t)
    np.testing.assert_allclose(w.history.window(1.0)[1], want[1], rtol=0, atol=1e-12)

    w._sim_target = t + 0.05
    w._tick()
    assert w._started is not None
    w.params.rows[0][1].setValue(2.0)  # a heavier first mass: the steps before it are recorded first
    assert w._started is None and w.history.size > round(t / w.sim.step_size())
    w._sim_target = w.sim.t + 0.05
    w._tick()
    w._reset()
    assert w._started is None and w.history.size == 0 and w.sim.t == 0.0
    w.close()


def test_history_keeps_every_stride_th_step_of_a_fast_chain(app):
    """A stiff chain's steps are short: the history keeps STORE_PER_PERIOD samples to its fastest period,
    evenly across advances, so it spans the plot window; the energy balance still closes on them."""
    import numpy as np

    from vib_tutorial.core.model import ChainSystem
    from vib_tutorial.gui.main_window import STORE_PER_PERIOD, MainWindow

    w = MainWindow()
    w.pages.setCurrentWidget(w.sim_page)
    w.coords_combo.setCurrentIndex(2)  # energy
    w._on_system_changed(ChainSystem([0.01] * 4, [3.6e5] * 4, [0.1] * 4, [0.5] * 4))
    w.force.settings.freq_hz, w.force.settings.amplitude = 1000.0, 50.0
    w.force_panel.button.click()
    h, stride = w.sim.step_size(), w._store_stride()
    assert stride > 1 and 1.0 / (stride * h * w.sim.fastest_hz()) >= STORE_PER_PERIOD
    assert w.history.capacity * stride * h > 9.0  # most of the 10 s window, not 2.3 s
    for steps in (7, 1, 3000, 2):  # advances shorter and longer than the stride
        w._sim_target = w.sim.t + (steps + 0.4) * h
        w._start()
        w._collect()
    t = w.history.t[: w.history.size]
    np.testing.assert_allclose(np.diff(t), stride * h, rtol=1e-9)
    assert t[-1] <= w.sim.t < t[-1] + stride * h
    w.running = False
    w._tick()  # draws them
    T, V, stored, added, work, dissipated, rubbed = _energy_rows(w)
    assert rubbed[-1] > 0
    np.testing.assert_allclose(added + work, stored + dissipated + rubbed, rtol=0, atol=1e-12 * work.max())
    w.close()


def test_simulation_page_keeps_to_its_compute_budget(app, monkeypatch):
    """A frame takes no more simulator steps than FRAME_BUDGET allows at the cost of a step seen so far,
    so a slow simulation (friction on a stiff chain) falls behind real time instead of stalling the
    frames. A new system is measured afresh, with PROBE_STEPS; a dearer step counts at once."""
    import time

    from vib_tutorial.gui.main_window import FRAME_BUDGET, PROBE_STEPS, MainWindow

    w = MainWindow()
    w.pages.setCurrentWidget(w.sim_page)
    w.running = True

    def tick():
        w._last_wall = w._clock.elapsed() / 1000.0 - 0.1
        w._sim_target = w.sim.t + 10.0  # far behind: only the caps hold it
        before = w.sim.t
        w._tick()
        return round((w.sim.t - before) / w.sim.step_size())

    w._on_system_changed(w.sim.system)
    assert w._step_cost == 0.0
    assert tick() <= PROBE_STEPS
    assert w._step_cost > 0.0

    collect = w.sim.collect

    def slow():  # 1 ms per step
        out = collect()
        time.sleep(1e-3 * out[0].size)
        return out

    monkeypatch.setattr(w.sim, "collect", slow)
    tick()
    assert w._step_cost >= 1e-3  # dearer: taken at once
    assert tick() <= FRAME_BUDGET / 1e-3 + 1
    slow_cost = w._step_cost  # ≥ 1 ms: more where sleeps overshoot (macOS runners)
    monkeypatch.setattr(w.sim, "collect", collect)
    tick()
    assert 0.0 < w._step_cost < slow_cost  # cheaper: gradually
    w.close()


def test_duffing_page(app):
    """The Duffing page sweeps up and down over harmonic balance's curve, measures its jumps, starts the
    mass on a branch, pauses when a softening spring's mass escapes, and keeps going across a theme switch."""
    from vib_tutorial.core.duffing import Sweep
    from vib_tutorial.gui.main_window import MainWindow

    w = MainWindow()
    w.show()
    p = w.duffing_page
    w.pages.setCurrentWidget(p)
    app.processEvents()
    assert p.running and p._timer.isActive() and p.sim.drive.mode is Sweep.UP_DOWN
    p._timer.stop()
    assert p.preset.currentText() == "Hardening spring: the jump" and p.frame_context() == p.preset.currentText()
    assert "3.64 and 4.45 Hz" in p.readout.text()
    while p.sim.drive.mode is not Sweep.HOLD:
        p.step(5.0)
    legs = {c.leg for c in p.cycles}
    assert {"up", "down"} <= legs and len(p.up_item.data) > 100 and len(p.down_item.data) > 100
    assert "Measured: the jump at 4." in p.readout.text() and "and 3.5" in p.readout.text()
    assert p.x_item.getData()[0].size <= 2 * 1200 + 2
    assert len(p.section) == sum(c.leg == "hold" for c in p.cycles)  # Poincaré dots only once it holds

    w.set_theme("Dark")
    p.step(1.0)
    w.set_theme("Light")

    p.preset.setCurrentText("Two stable states")
    assert p.sim.drive.mode is Sweep.HOLD and p.freq.value() == pytest.approx(4.0)
    assert all(b.isEnabled() for b in p.branch_buttons)
    p.start_on_branch(2)
    for _ in range(20):
        p.step(1.0)
    assert p.cycles[-1].amplitude == pytest.approx(0.0583, rel=0.02)  # stays on the upper branch
    assert len(p.section) >= 60 and np.ptp(np.array(p.section[-10:]), axis=0).max() < 1e-3  # one dot
    p.freq.setValue(5.0)  # outside the band: one amplitude
    assert not p.branch_buttons[1].isEnabled() and p.branch_buttons[0].isEnabled()

    p.preset.setCurrentText("Escape from the well")
    while p.running:
        p.step(5.0)
    assert p.sim.escaped and "ran away" in p.readout.text() and p.run_button.text() == "Run"
    p.reset()
    assert p.running and not p.sim.escaped and p.sim.t == 0 and not p.cycles

    p.stiffness.setValue(500.0)  # an edit: Custom, the points cleared, the curve redrawn
    assert p.preset.currentText() == "Custom" and p.sim.osc.stiffness == 500.0
    p.stiffness.setValue(0.0)  # a softening spring with nothing to soften: kept as it was
    assert p.sim.osc.stiffness == 500.0 and "keeps the last" in p.osc_info.text()
    w.close()


def test_mathieu_on_the_duffing_page(app):
    """Pumped, the page draws the half-harmonic branches and measures the ½X swing, its branch buttons start the
    pump's swings, the Ince–Strutt chart shows the tongues and a click pumps there, and a linear spring's swing
    grows until the page pauses."""
    from PySide6.QtCore import QPointF

    from vib_tutorial.core.duffing import Sweep
    from vib_tutorial.gui.main_window import MainWindow

    w = MainWindow()
    w.show()
    p = w.duffing_page
    w.pages.setCurrentWidget(p)
    app.processEvents()
    p._timer.stop()
    assert w.pages.tabText(w.pages.currentIndex()) == "Duffing && Mathieu"

    p.preset.setCurrentText("Parametric resonance, hardening")
    assert p.pumped_only and p.sim.drive.mode is Sweep.HOLD
    names = [label.text for _, label in p.legend.items]
    assert "Pumped, ½X" in names and "Sweeping up" not in names
    assert [b.isEnabled() for b in p.branch_buttons] == [True, False, True]  # rest, the swing
    assert "rest (unstable)" in p.branch_note.text()
    for _ in range(30):
        p.step(2.0)
    assert p.cycles[-1].half == pytest.approx(0.0275, rel=0.02)
    assert len(p.half_items["hold"].data) > 50 and not len(p.hold_item.data)
    p.start_on_branch(0)  # back to rest: unstable, so the noise grows again
    assert p.sim.x == 0 and p.sim.v == 0

    p.tabs.setCurrentWidget(p.chart_tab)
    assert p.chart_image.image is not None and p.chart_image.image.shape == (186, 61)
    assert "rest is unstable" in p.chart_note.text()
    vb = p.chart.getPlotItem().getViewBox()

    class Click:
        def scenePos(self):  # noqa: N802 (Qt's name)
            return vb.mapViewToScene(QPointF(1.0, 0.8))

    p._on_chart_click(Click())  # the second tongue, at the natural frequency
    assert p.pump.value() == pytest.approx(0.8) and p.freq.value() == pytest.approx(3.183, abs=1e-3)
    assert p.preset.currentText() == "Custom" and "rest is unstable" in p.chart_note.text()
    p.damping.setValue(4.0)  # more damping lifts the tongue off the pump
    assert "rest is stable" in p.chart_note.text()

    p.preset.setCurrentText("Parametric resonance")
    while p.running:
        p.step(5.0)
    assert p.sim.escaped and "without bound" in p.readout.text() and "km" in p.readout.text()

    p.preset.setCurrentText("Parametric sweep: the jump")
    assert p.sim.drive.mode is Sweep.DOWN_UP and p.sim.drive.leg == "down" and p.freq.value() == 8.0
    w.set_theme("Dark")
    p.step(1.0)
    w.close()


def test_torsional_page(app):
    """The torsional page loads each preset's criticals and response, edits the train (a disc, a shaft made a
    mesh and back, a disc added and removed, the excitation), picks elements by clicking the drawing, switches
    log scales, and survives a theme switch."""
    from PySide6.QtCore import QPointF

    from vib_tutorial.core.torsional import Mesh, Shaft
    from vib_tutorial.gui.main_window import MainWindow

    w = MainWindow()
    w.show()
    p = w.torsional_page
    w.pages.setCurrentWidget(p)
    app.processEvents()
    assert w.pages.tabText(w.pages.currentIndex()) == "Torsional vibration"
    assert p.preset.currentText() == "Motor and fan: two inertias" and p.frame_context() == p.preset.currentText()
    assert "Misalignment 2× meets mode 1 (52.3 Hz) at 1569 rpm" in p.readout.text()
    assert "6.0 % above" in p.readout.text() and not p.log_scales.isChecked()
    assert "Coupling: ±72.1 N·m at 1480 rpm" in p.response_note.text()

    p.preset.setCurrentText("Geared compressor")
    assert p.log_scales.isChecked() and "Tooth pass (Gear mesh) meets mode 3 (2688 Hz) at 1344 rpm" in p.readout.text()
    assert p.response_link.currentText() == "High-speed coupling"  # the slowest mode's
    assert "38.92, 83.09, 2688 Hz, the same" in p.check.text() and "1 mode lost" in p.check.text()
    p.log_scales.setChecked(False)
    p.log_scales.setChecked(True)

    p.preset.setCurrentText("Diesel generator set")
    assert "No torsional critical within ±10 % of 1500 rpm" in p.readout.text()
    assert "6 faster modes no order reaches" in p.readout.text()
    assert p.response_link.currentText() == "Flexible coupling"

    # edit the fan's coupling: Custom, and the mode moves as sqrt(k)
    p.preset.setCurrentText("Motor and fan: two inertias")
    p.select(1)
    assert p.editor.currentIndex() == 1 and p.shaft_stiffness.value() == pytest.approx(180.0)
    p.shaft_stiffness.setValue(120.0)
    assert p.preset.currentText() == "Custom" and p.train.links[0].stiffness == pytest.approx(1.2e5)
    assert "No torsional critical within ±10 %" in p.readout.text()
    f = p._modes[0].freq_hz
    p.to_mesh.click()  # the same referred stiffness: the same mode
    assert isinstance(p.train.links[0], Mesh) and p.editor.currentIndex() == 2
    assert p._modes[0].freq_hz == pytest.approx(f, rel=1e-9)
    p.teeth_out.setValue(80)  # a 2:1 reduction: the fan's inertia counts a quarter
    assert p.train.speeds[1] == pytest.approx(0.5) and len(p._modes) == 1
    p.to_shaft.click()
    assert isinstance(p.train.links[0], Shaft)

    p.select(2)  # the fan
    p.add_button.click()
    assert p.train.n == 3 and p.selected == 4 and p.element.count() == 5
    p.remove_button.click()
    assert p.train.n == 2 and not p.remove_button.isEnabled()

    p.excite_disc.setCurrentIndex(0)  # None: no sources
    assert p.train.excitation is None and "No excitation" in p.response_note.text()
    p.excite_disc.setCurrentIndex(1)  # on the motor, a fresh 1×
    assert p.train.excitation.disc == 0 and p.train.excitation.harmonics == ((1.0, 100.0),)

    p.tabs.setCurrentWidget(p.drawing_tab)
    app.processEvents()
    vb = p.drawing.getPlotItem().getViewBox()
    spots, _ = p._layout()

    class Click:
        def __init__(self, x, y):
            self.at = vb.mapViewToScene(QPointF(x, y))

        def scenePos(self):  # noqa: N802 (Qt's name)
            return self.at

    p._on_drawing_click(Click(*spots[1][:2]))
    assert p.selected == 2 and p.editor.currentIndex() == 0
    p._on_drawing_click(Click(0.5 * (spots[0][0] + spots[1][0]), spots[0][1]))
    assert p.selected == 1

    w.set_theme("Dark")
    p.tabs.setCurrentWidget(p.shapes_tab)
    w.set_theme("Light")
    w.close()


def test_torsional_start_up(app):
    """The torsional page's start-ups: the fan's induction motor and the compressor's synchronous one run to the
    end with their peaks read out, pause and resume, the diesel set has no motor until one is picked, and
    editing the train or the motor starts afresh."""
    from vib_tutorial.core.torsional_startup import MotorKind
    from vib_tutorial.gui.main_window import MainWindow

    w = MainWindow()
    w.show()
    p = w.torsional_page
    w.pages.setCurrentWidget(p)
    app.processEvents()
    assert p.sim is not None and not p.running and p.views.currentWidget() is p.criticals_tab
    assert "At rest" in p.startup_status.text() and "mode 1 (52.3 Hz) is 5 % from it" in p.startup_info.text()
    p.start_button.click()
    assert p.running and p.views.currentWidget() is p.startup_tab and p.pause_button.text() == "Pause"
    p.step(0.5)
    assert p.sim.t == pytest.approx(0.5) and p._k == round(0.5 / p.sim.h)
    p.pause_button.click()
    assert not p.running and p.pause_button.text() == "Run"
    p.step(10.0)  # a test steps it whether it plays or not: to the end, and no further
    assert p.sim.done and not p.pause_button.isEnabled()
    assert "Done." in p.startup_status.text()
    assert "<b>Coupling: 6 kN·m at 0.068 s (38 rpm), 10.3× its running 580 N·m</b>" in p.startup_note.text()
    assert p.link_curve.xData.size == p._k  # (getData is what shows, decimated)

    p.preset.setCurrentText("Geared compressor")
    assert p.motor_kind.currentText() == "Synchronous" and p._k == 0 and not p.running
    assert "meets mode 2 (83.1 Hz) at 254 rpm, mode 1 (38.9 Hz) at 916 rpm" in p.startup_info.text()
    assert len(p._marks) == 2  # the two crossings' speeds
    p.start()
    p.step(3.0)
    assert len(p._marks) == 4 and p.sim.field_on  # and when the run passed them
    assert "in step, the field on. Done." in p.startup_status.text()
    assert "<b>Low-speed coupling: 11.3 kN·m" in p.startup_note.text()
    p.startup_link.setCurrentText("High-speed coupling")
    assert np.abs(p.link_curve.yData).max() == pytest.approx(p.sim.peaks[2])

    p.start()
    p.step(0.2)
    p.select(3)  # the mesh: a stiffer one starts the run again from rest, still playing
    p.mesh_stiffness.setValue(3000.0)
    assert p.preset.currentText() == "Custom" and p.running and p.sim.t == 0.0 and p._k == 0
    p.breakdown.setValue(7000.0)
    assert p.startup.motor.breakdown == 7000.0 and p.sim.t == 0.0

    p.preset.setCurrentText("Diesel generator set")
    assert p.sim is None and not p.start_button.isEnabled() and "engine drives this train" in p.startup_info.text()
    p.start()
    assert not p.running
    p.motor_kind.setCurrentIndex(1)  # the last motor's numbers, made an induction motor
    assert p.startup.motor.kind is MotorKind.INDUCTION and p.sim is not None and p.start_button.isEnabled()
    assert p.preset.currentText() == "Custom"
    p.start()
    p.step(0.1)
    assert p._k > 0
    w.set_theme("Dark")
    w.set_theme("Light")
    w.close()


def test_oma_page(app):
    """The operational modal analysis page follows the chain, finds its modes from the ambient record, picks
    and drops peaks (and SSI's poles) by clicking, reprocesses without recording again, runs each trial, crosses
    out a machine's harmonic, compares the methods with a hammer test, and survives a theme switch."""
    from PySide6.QtCore import QPointF

    from vib_tutorial.gui.main_window import MainWindow

    w = MainWindow()
    w.show()
    p = w.oma_page
    w.pages.setCurrentWidget(p)
    app.processEvents()
    assert w.pages.tabText(w.pages.currentIndex()) == "Operational modal analysis"
    assert p.record is not None and p.dof.value() == 4 and p.input.count() == 5
    assert p.readout.text().startswith("FDD + EFDD found: mode 1 ") and "mode 4" in p.readout.text()
    assert "2,000 periods of mode 1" in p.daq_info.text() and p.block.currentData() == 2048
    assert p.results.table.rowCount() == 4 and p.mode_pick.count() == 4

    record = p.record
    p.noise.setValue(10.0)
    p.recompute()
    assert p.record is record  # processing only: the same record
    p.new_record.click()
    assert p.record is not record and p.seed == 2

    # click on mode 1's peak: dropped; click there again: picked again
    vb = p.sv.getPlotItem().getViewBox()

    class Click:
        def __init__(self, x):
            lo, hi = vb.viewRange()[1]
            self.at = vb.mapViewToScene(QPointF(x, 0.5 * (lo + hi)))

        def scenePos(self):  # noqa: N802 (Qt's name)
            return self.at

    first = p.fdd.freqs[p.modes[0].line]
    p._on_sv_click(Click(first))
    assert len(p.modes) == 3 and p.peaks is not None
    p._on_sv_click(Click(first))
    assert len(p.modes) == 4
    p.auto_peaks.click()
    assert p.peaks is None

    for i in range(p.trial.count()):
        p.trial.setCurrentIndex(i)
        p.recompute()
        assert p.trial_note.text()
    p.trial.setCurrentText("Forces at m3 only")
    p.recompute()
    assert p.input.currentText() == "m3" and "missed: mode 2" in p.results.notes.text()

    p.tabs.setCurrentWidget(p.efdd_tab)
    p.mode_pick.setCurrentIndex(1)
    assert p.selected == 1 and "Bell of" in p.efdd_note.text()

    # SSI, its stabilization diagram (click a pole to drop it, again to take it back), the machine
    p.trial.setCurrentText("SSI on close, damped modes")
    p.recompute()
    assert p.method.currentText() == "SSI-cov" and p.rows.isVisible() and not p.mac_min.isVisible()
    assert p.readout.text().startswith("SSI-cov found: mode 1") and "mode 4" in p.readout.text()
    order, i = p.ssi_poles[0]
    p._on_pole_clicked(order, i)
    assert len(p.ssi) == 3 and p.ssi_picked is not None
    p._on_pole_clicked(order, i)
    assert len(p.ssi) == 4
    p.auto_peaks.click()
    assert p.ssi_picked is None
    p.tabs.setCurrentWidget(p.methods_tab)
    assert p.methods.rowCount() == 4 and p._hammer is not None
    assert p.methods.item(0, 4).text().startswith("1.1")  # the hammer test's mode 1
    p.trial.setCurrentText("A machine near a mode")
    p.recompute()
    assert p.machine_hz.value() > 0 and p.method.currentText() == "FDD + EFDD"
    assert "<b>Harmonic</b> at" in p.readout.text() and "mode 2" not in p.readout.text()
    p.method.setCurrentText("SSI-cov")
    assert "mode 2" in p.readout.text()  # SSI fits mode 2 beside the sine

    w.params.dof.setValue(3)  # the chain changes: the page follows
    app.processEvents()
    p.recompute()
    assert p.system.n == 3 and p.input.count() == 4 and p.record.y.shape[1] == 3
    w.set_theme("Dark")
    w.set_theme("Light")
    w.close()


def test_continuum_page(app):
    """The continuous systems page loads each preset, picks a mode (spin box, table, a click on the error plot),
    switches the kind (the ends refill), meshes it finer and coarser, hides a scheme, and survives a theme switch."""
    from PySide6.QtCore import QPointF

    from vib_tutorial.core.continuum import PRESETS, End, Kind, Scheme
    from vib_tutorial.gui.main_window import MainWindow

    w = MainWindow()
    w.show()
    p = w.continuum_page
    w.pages.setCurrentWidget(p)
    app.processEvents()
    assert w.pages.tabText(w.pages.currentIndex()) == "Continuous systems"
    assert p.preset.currentText() == "Guitar string (low E)" and p.frame_context() == p.preset.currentText()
    assert "wave speed c = 106.8 m/s" in p.medium_info.text()
    p.mode.setValue(6)
    assert "<b>Mode 6</b>" in p.readout.text() and "4 elements</b>" in p.readout.text()
    assert "lumped 445.2 Hz (-9.97 %)" in p.readout.text()
    assert "lumped 1.000000" in p.shape_note.text()  # the nodes are the exact shape's, only f is off
    assert p.table.currentRow() == 5 and p.table.item(5, 3).text() == "-9.97 %"
    p.views.setCurrentWidget(p.errors_tab)
    app.processEvents()
    assert "lumped N<sup>−2</sup>" in p.convergence_note.text() and "mixed N<sup>−4</sup>" in p.convergence_note.text()

    vb = p.errors.getPlotItem().getViewBox()

    class Click:
        def __init__(self, x, y):
            self.at = vb.mapViewToScene(QPointF(x, y))

        def scenePos(self):  # noqa: N802 (Qt's name)
            return self.at

    p._on_errors_click(Click(3.2, 0.0))
    assert p.mode.value() == 3
    p.table.selectRow(8)
    assert p.mode.value() == 9

    for preset in (*PRESETS[1:], PRESETS[0]):
        p.preset.setCurrentText(preset.name)
        assert p.continuum == preset.continuum and p.elements.value() == preset.elements
        assert p.preset_note.text() == preset.note and p.mode.value() == 1

    p.kind.setCurrentText("Bar")
    assert p.preset.currentText() == "Custom" and p.continuum.kind is Kind.BAR
    assert [p.left_end.itemText(i) for i in range(p.left_end.count())] == ["Fixed", "Free"]
    assert p.continuum.left is End.FIXED and p.material.currentIndex() == 1
    p.kind.setCurrentText("String")
    assert p.material.currentIndex() == 0 and p.continuum.kind is Kind.STRING
    p.elements.setValue(1)  # fixed–fixed, one element: nothing free
    assert not p.mode.isEnabled() and "No flexible mode" in p.readout.text()
    p.elements.setValue(40)
    assert p.mode.isEnabled() and p.mode.maximum() == 39
    p.shown[Scheme.LUMPED].setChecked(False)
    assert "lumped" not in p.readout.text() and "consistent" in p.readout.text()

    w.set_theme("Dark")
    p.tabs.setCurrentWidget(p.table)
    w.set_theme("Light")
    w.close()


def test_continuum_waves_tab(app):
    """The Waves tab: a wave preset opens it and plays; the pulse runs, pauses and restarts; the dashpot and the
    step reach the run; a beam turns them off; leaving the tab stops the frames."""
    from vib_tutorial.core.continuum import Kind, Scheme
    from vib_tutorial.core.continuum_waves import FRAMES_PER_TRANSIT
    from vib_tutorial.gui.main_window import MainWindow

    w = MainWindow()
    w.show()
    p = w.continuum_page
    w.pages.setCurrentWidget(p)
    app.processEvents()
    assert p.run is None  # not built until the tab shows
    p.preset.setCurrentText("Waves: a pulse on 40 beads")
    assert p.views.currentWidget() is p.waves_tab and p.run is not None and p._timer.isActive()
    assert p.run.elements == 40 and p.run.waves.moving and p.play_button.text() == "Pause"
    p._timer.stop()
    p.step(FRAMES_PER_TRANSIT)
    assert "t = 1.00 crossings (10 ms" in p.wave_status.text() and "keeps <b>100 %</b>" in p.wave_status.text()
    assert p.mesh_nodes.getData()[0].size == 41
    p.play_button.click()
    assert not p.running and not p._timer.isActive() and p.play_button.text() == "Run"
    p.restart_button.click()
    assert p.run.t == 0.0 and not p._timer.isActive()

    p.preset.setCurrentText("Waves: a matched end")
    assert p.run.waves.dashpot == 1.0 and p.end_line.isVisible() and "reflects +0 of a wave" in p.wave_info.text()
    p._timer.stop()
    p.step(2 * FRAMES_PER_TRANSIT)
    assert "keeps <b>17.3 %</b>" in p.wave_status.text()
    p.wave_scheme.setCurrentText("Consistent")
    assert p.run.waves.scheme is Scheme.CONSISTENT and p.run.t == 0.0 and p.preset.currentText() == "Waves: a matched end"
    p.playback.setCurrentIndex(0)
    assert p.run.playback == 0.25

    p.preset.setCurrentText("Waves: a light string tied to a heavy one")
    assert p.step_region.isVisible() and p.run.step_x == pytest.approx(0.5)
    assert "reflects -0.333 and passes 0.667" in p.wave_info.text()
    p.step_on.setChecked(False)
    assert p.run.step_x is None and not p.step_region.isVisible() and not p.step_at.isEnabled()

    p.kind.setCurrentText("Beam")
    assert p.continuum.kind is Kind.BEAM and not p.moving.isEnabled() and not p.dashpot_on.isEnabled()
    assert p.run.waves.dashpot is None and not p.run.moving

    p.views.setCurrentWidget(p.errors_tab)
    assert not p._timer.isActive()
    p.views.setCurrentWidget(p.waves_tab)
    w.set_theme("Dark")
    w.set_theme("Light")
    w.close()


def test_random_vibration_page(app):
    """The random vibration page follows the chain, predicts the response's PSD and RMS and checks them against a
    simulated record and Miles' equation, re-records only when the input changes, runs each trial, and survives a
    theme switch."""
    from vib_tutorial.core.random_vibration import Input, Quantity
    from vib_tutorial.gui.main_window import MainWindow
    from vib_tutorial.gui.random_vibration import TRIALS

    w = MainWindow()
    w.show()
    p = w.random_page
    w.pages.setCurrentWidget(p)
    app.processEvents()
    p.recompute()
    assert w.pages.tabText(w.pages.currentIndex()) == "Random vibration"
    assert p.dof.value() == 4 and p.mass.count() == 4 and p.index.currentText() == "m4"
    assert p.kind.currentData() is Input.BASE and p.level.suffix() == " g²/Hz" and not p.mass.isEnabled()
    assert "exact 0.9109 g" in p.readout.text() and "Miles 0.9129 g (+0.22 %)" in p.readout.text()
    assert "lines 0.0061 Hz apart" in p.record_info.text() and "0.3162 g" in p.input_info.text()
    assert p.miles_table.rowCount() == 5 and p.miles_table.item(4, 4).text() == "SRSS 0.9129"

    record = p.record
    p.quantity.setCurrentIndex(p.quantity.findData(Quantity.SPRING))
    p.recompute()
    assert p.record is record and p.index.currentText() == "k4" and "N" in p.readout.text()
    p.block.setCurrentIndex(0)
    p.recompute()
    assert p.record is record and "lines 0.0977 Hz apart" in p.record_info.text()
    p.new_record.click()
    assert p.record is not record and p.seed == 2

    for i, (name, _, note) in enumerate(TRIALS):
        p.trial.setCurrentIndex(i)
        p.recompute()
        assert p.trial_note.text() == note and p.prediction is not None and "RMS" in p.readout.text()
    p.trial.setCurrentIndex(1)  # a force on the last mass: its own acceleration's mass line
    p.recompute()
    assert p.mass.isEnabled() and p.mass.currentText() == "m4" and p.level.suffix() == " N²/Hz"
    assert "Miles 0.4163 g (-18 %)" in p.readout.text()

    p.dof.setValue(3)  # the Simulation page's chain follows, and the page with it
    app.processEvents()
    p.recompute()
    assert w.sim.system.n == 3 and p.mass.count() == 3 and p.record.x.shape[1] == 3
    p.f_hi.setValue(0.0)  # an empty band
    p.f_lo.setValue(1.0)
    p.recompute()
    assert "band" in p.readout.text()
    w.set_theme("Dark")
    p.tabs.setCurrentWidget(p.record_tab)
    w.set_theme("Light")
    w.close()


def test_random_vibration_statistics(app):
    """The random vibration page's Statistics tab: the record's crossings, peaks and exceedances beside Rice's and the
    Gaussian's, and the largest peak over a stretch beside Davenport's and Vanmarcke's."""
    from vib_tutorial.gui.main_window import MainWindow

    w = MainWindow()
    w.show()
    p = w.random_page
    w.pages.setCurrentWidget(p)
    p.tabs.setCurrentWidget(p.stats_tab)
    app.processEvents()
    p.trial.setCurrentText("The base spring's force")
    p.recompute()
    table = p.stats_table.text()
    assert "Upward zero crossings" in table and "31.7 %" in table and "Rice, α = 0.66" in [
        p.peaks_legend.items[1][1].text]
    assert "Davenport 3.11σ" in p.largest_note.text() and "Vanmarcke (δ = 0.25) 2.84σ" in p.largest_note.text()
    assert "43 stretches of 60 s" in p.largest_note.text()
    assert not p.cumulative.getAxis("left").autoSIPrefix
    p.largest_over.setValue(1000.0)
    assert "too short for four stretches" in p.largest_note.text()
    w.set_theme("Dark")
    w.set_theme("Light")
    w.close()


def test_complex_modes_page(app):
    """The complex modes page follows the chain, loads its trial chains into the Simulation page, makes the damping
    proportional, and draws every tab for every mode, an overdamped one included, through a theme switch."""
    from vib_tutorial.gui.complex_modes import TRIALS
    from vib_tutorial.gui.main_window import MainWindow

    w = MainWindow()
    w.show()
    p = w.complex_page
    w.pages.setCurrentWidget(p)
    app.processEvents()
    assert w.pages.tabText(w.pages.currentIndex()) == "Complex modes"
    assert "proportional" in p.coupling.text() and not p.make_proportional.isEnabled()
    assert p.table.rowCount() == 4 and "Every mode is real" in p.readout.text()
    assert "stays there" in p.decay_note.text() and "the same" in p.frf_note.text()

    names = [name for name, _, _ in TRIALS]
    p.trial.setCurrentIndex(1 + names.index("One damper, at the base"))
    app.processEvents()
    assert p.trial.currentIndex() == 0 and p.trial_note.isVisible()
    assert list(w.sim.system.damping) == [20.0, 0, 0, 0]
    assert "not proportional" in p.coupling.text() and "mode 3, MPC 0.20" in p.readout.text()
    p.mode.setCurrentIndex(1)
    assert p.table.selectedItems()[0].row() == 1 and "45 %</b> of it reaches mode 1" in p.decay_note.text()
    p.table.selectRow(2)
    assert p.mode.currentIndex() == 2 and "MPC 0.200" in p.argand.getPlotItem().titleLabel.text
    for tab in (p.frf_tab, p.decay_tab, p.scatter_tab):
        p.tabs.setCurrentWidget(tab)
        app.processEvents()
    p.frf_in.setCurrentIndex(0)
    assert "%</b> of the exact FRF's peak" in p.frf_note.text()

    p.make_proportional.click()
    app.processEvents()
    assert w.sim.system.damping == pytest.approx(w.sim.system.damping[0] / 400.0 * w.sim.system.stiffness)
    assert "proportional" in p.coupling.text() and not p.make_proportional.isEnabled()

    p.trial.setCurrentIndex(1 + names.index("A strong damper locks a joint"))
    app.processEvents()
    assert "Overdamped (no oscillating pole): mode 4." in p.readout.text()
    assert p.table.item(3, 1).text() == "overdamped"
    p.mode.setCurrentIndex(3)
    assert "real roots" in p.mode_info.text()
    p.trial.setCurrentIndex(1 + names.index("Tuned mass damper"))
    app.processEvents()
    assert p.dof.value() == 5 and p.mode.count() == 5 and p.table.rowCount() == 5
    p.dof.setValue(2)
    app.processEvents()
    assert w.sim.system.n == 2 and p.table.rowCount() == 2 and p.frf_in.count() == 2
    w.set_theme("Dark")
    w.set_theme("Light")
    w.close()


def test_complex_modes_sweep_and_identification(app):
    """The complex modes page's Sweep tab follows the chain's damper through its root locus, and its
    Identification tab fits the FRF with real and complex modes; both work out only while they show."""
    from vib_tutorial.core.complex_modes import Parameter
    from vib_tutorial.gui.main_window import MainWindow

    w = MainWindow()
    w.show()
    p = w.complex_page
    w.pages.setCurrentWidget(p)
    p.trial.setCurrentText("One damper, at the top")
    app.processEvents()
    assert p.sweep_result is None and p.ident_result is None  # not worked out while hidden
    p.tabs.setCurrentWidget(p.sweep_tab)
    app.processEvents()
    sw = p.sweep_result
    assert sw.parameter is Parameter.DAMPER and sw.element == 3 and p.sweep_element.currentText() == "c4"
    assert "as c4 locks its joint" in p.sweep_note.text() and "c4 is 20 N·s/m" in p.sweep_note.text()
    p.sweep_parameter.setCurrentIndex(p.sweep_parameter.findData(Parameter.SPRING))
    assert p.sweep_result.parameter is Parameter.SPRING and p.sweep_element.currentText() == "k4"
    assert "most complex" in p.sweep_note.text()

    p.trial.setCurrentText("One damper, at the base")
    app.processEvents()
    assert p.sweep_result.parameter is Parameter.SPRING  # worked out again for the new chain, as it shows
    p.tabs.setCurrentWidget(p.ident_tab)
    app.processEvents()
    assert p.ident_table.rowCount() == 4 and p.ident_table.item(2, 2).text() == "0.506"
    assert "<b>29 %</b> (real modes)" in p.ident_note.text()
    p.ident_noise.setValue(0.0)
    assert "<b>0 %</b> (complex modes)" in p.ident_note.text()
    p.ident_in.setCurrentIndex(3)
    p.ident_out.setCurrentIndex(0)
    assert p.ident_result.H.shape == (2000, 4)
    p.make_proportional.click()
    app.processEvents()
    assert "<b>0 %</b> (real modes)" in p.ident_note.text()
    w.set_theme("Dark")
    w.set_theme("Light")
    w.close()


def test_model_updating_page(app):
    """The model updating page measures the Simulation page's chain, correlates its own model with errors in it,
    re-correlates without measuring again when the model or the sensors change, runs each trial, and follows the
    chain's size."""
    from vib_tutorial.core.model_updating import Kind, Source
    from vib_tutorial.gui.main_window import MainWindow
    from vib_tutorial.gui.model_updating import TRIALS

    w = MainWindow()
    w.show()
    p = w.updating_page
    w.pages.setCurrentWidget(p)
    app.processEvents()
    p.recompute()
    assert w.pages.tabText(w.pages.currentIndex()) == "Model updating"
    assert p.source.currentData() is Source.HAMMER and len(p.sensors) == 4 and len(p.error_rows) == 4
    assert p.error_rows[2][2].value() == pytest.approx(1.3)  # the first trial: k3 30 % too stiff
    assert "found 3 of the structure's 4 modes" in p.test_info.text()
    assert p.table.rowCount() == 3 and "3 pairs" in p.readout.text() and "+4.09 %" in p.readout.text()

    test = p.test
    p.error_rows[2][2].setValue(1.0)
    p.recompute()
    assert p.test is test and max(abs(x.error) for x in p.corr.pairs) < 0.003
    p.sensors[1].setChecked(False)
    p.recompute()
    assert p.test is test and p.corr.test.sensors.sum() == 3 and p.corr.comac.size == 3
    p.again.click()
    assert p.test is not test and p.seed == 2
    for tab in (p.shape_tab, p.auto_tab, p.comac_tab):
        p.tabs.setCurrentWidget(tab)
        app.processEvents()
    p.table.selectRow(1)
    assert "Test mode 2" in p.shape_note.text()

    for i, (_, _, note) in enumerate(TRIALS):
        p.trial.setCurrentIndex(i)
        p.recompute()
        assert p.trial_note.text() == note and p.corr.pairs
    p.trial.setCurrentText("Two sensors")
    p.recompute()
    assert p.source.currentData() is Source.EXACT and [b.isChecked() for b in p.sensors] == [True, False, False, True]
    assert "test modes 2 and 4. At 2 sensors" in p.auto_note.text()
    p.clear_errors.click()
    p.recompute()
    assert all(box.value() == 1.0 for row in p.error_rows for box in row[1:])

    p.trial.setCurrentText("Update: find the stiff spring")
    p.recompute()
    assert p.update_result is None  # worked out only while its tab shows
    p.tabs.setCurrentWidget(p.update_tab)
    u = p.update_result
    assert u is not None and "k3, 1.300 → <b>1.023</b>" in p.update_note.text()
    p.apply_update.click()
    assert p.error_rows[2][2].value() == pytest.approx(1.023, abs=0.001) and p.update_result is not u
    assert max(abs(x.error) for x in p.corr.pairs) < 0.01
    p.trial.setCurrentText("Update: masses too, not unique")
    p.recompute()
    assert p.update_kind[Kind.MASS].isChecked() and "1 of the 8 directions" in p.update_note.text()
    p.regularization.setValue(1.0)
    assert p.update_result.history  # worked out again for the new setting
    p.tabs.setCurrentWidget(p.sens_tab)
    assert len(p.sens_plot.labels) == 4 * 8

    p.dof.setValue(6)
    app.processEvents()
    p.recompute()
    assert w.sim.system.n == 6 and len(p.sensors) == 6 and len(p.error_rows) == 6 and len(p.corr.result.modes) == 6
    for b in p.sensors:
        b.setChecked(False)
    p.recompute()
    assert "No sensors" in p.readout.text()
    w.set_theme("Dark")
    w.set_theme("Light")
    w.close()
