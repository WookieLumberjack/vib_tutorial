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
