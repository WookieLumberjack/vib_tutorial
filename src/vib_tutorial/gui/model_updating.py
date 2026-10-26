"""Model updating page: a model of the chain, with errors in it, correlated against a modal test of the chain.

Left: the test (the virtual modal test, or the exact modes with noise; which masses have sensors),
the model's errors (factors on each mass, spring and damper), and errors to try. Centre: the MAC
of every test shape with every model shape, and the pairs it makes, with their frequency errors.
Right: the frequencies against each other, a pair's shapes, the test's auto-MAC, the COMAC, notes.
Follows the chain from the Simulation page, which is the structure tested.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.complex_modes import complexity
from ..core.modal import ModalResult, modal_analysis
from ..core.model import ChainSystem
from ..core.model_updating import Correlation, Errors, Source, Test, correlate, measure
from .layouts import ColumnScroll, FlowLayout, plot_height
from .modal_extraction import MacPlot, colored
from .panels import spin
from .style import MAX_DOF, colors
from .theming import add_legend, mute

RECOMPUTE_MS = 150  # the model's errors settle this long before it is correlated again

TIPS = {
    "source": "<p><b>Virtual modal test</b>: a hammer on m1, every mass's displacement measured with a little "
              "noise, H1 averaged, LSCF's stable poles and LSFD's shapes, as on the <i>Virtual modal test</i> page: "
              "its modes carry its errors, and the weakest may be missed. <b>Exact modes + noise</b>: every mode, "
              "with a little noise on its frequency, damping and shape.</p>",
    "sensors": "<p>The masses with a sensor. The test's shapes, and the model's, are compared there only.</p>",
    "again": "<p>Measure again: the same structure, other noise.</p>",
    "errors": "<p>The model's errors, as factors on the structure's values: 1.3 under k3 makes the model's "
              "spring 3 30 % stiffer than the structure's. In practice they are unknown; here they are set, so "
              "the correlation's verdict can be checked.</p>",
}
MAC_TIP = ("<p>MAC of each test shape (rows) with each model shape (columns), at the sensors: "
           "|ψ<sup>H</sup>φ|²/(ψ<sup>H</sup>ψ φ<sup>T</sup>φ). The pairs are outlined.</p>")
FREQ_TIP = ("<p>Each pair's model frequency against its test frequency: on the diagonal, they agree. A model "
            "uniformly too stiff or too heavy shifts every point the same way; a local error moves some modes more "
            "than others.</p>")
SHAPE_TIP = ("<p>A pair's shapes at the sensors: the test's (rotated to its best-fit line and scaled onto the "
             "model's) and the model's.</p>")
AUTO_TIP = ("<p>The test's shapes against each other. Distinct modes should be orthogonal-looking (small "
            "off-diagonal MAC); with too few or badly placed sensors different modes look alike: spatial "
            "aliasing.</p>")
COMAC_TIP = ("<p>COMAC per sensor: how well the paired shapes agree there, over all the pairs, "
             "(Σ|ψ<sub>i</sub>φ<sub>i</sub>|)²/(Σ|ψ<sub>i</sub>|² Σφ<sub>i</sub>²), each test shape scaled onto its "
             "model shape first. Low where the shapes disagree.</p>")

# Errors to try: (name, {"k": {element: factor}, "m": ..., "c": ..., "all_k": factor, "sensors": [...],
# "source": Source}, note). Elements count from 0; -1 is the last.
TRIALS = (
    ("One spring 30 % too stiff", {"k": {2: 1.3}},
     "The model's k3 is 1.3 times the structure's. Its frequencies are all too high, by 2.2, 4.1, 0.6 and 6.6 % "
     "(the hammer test finds the first three): mode 3's least, since k3 stretches little in it. The shapes "
     "barely change (MACs of 0.99 and up), so they are little help in finding it."),
    ("One mass 20 % too heavy", {"m": {1: 1.2}},
     "The model's m2 is 1.2 times the structure's: its frequencies are too low, by 1.8, 3.2, 0.5 and 3.5 %. A "
     "mass error and a spring error both just move the frequencies; which one it is, and where, takes the "
     "pattern of the errors over the modes."),
    ("Every spring 10 % too soft", {"all_k": 0.9},
     "Every frequency is √0.9 − 1 = −5.1 % off, and every shape exactly right: a uniform stiffness (or mass) "
     "error scales the frequencies and leaves the shapes alone, so the MACs see nothing."),
    ("Dampers twice too strong", {"all_c": 2.0},
     "The model's dampers are twice the structure's: its damping ratios are double the test's, and its "
     "frequencies and shapes don't move. Damping can only be checked against the measured ζ."),
    ("Two sensors", {"k": {2: 1.3}, "sensors": [0, -1], "source": Source.EXACT},
     "The same spring error, every mode measured, but on m1 and the last mass only. Each shape is now two "
     "numbers, and four modes can't all look different in two: modes 2 and 4 have an auto-MAC of 0.91 without noise "
     "(spatial aliasing), and only their frequencies tell them apart."),
    ("No error: the test's own noise", {"source": Source.HAMMER},
     "The model is exactly the structure. Every difference left is the test's: its frequencies within "
     "0.3 %, and the highest mode it finds (mode 3) at a MAC of only 0.92, its curve fit's error. The model "
     "can't be correlated more finely than that."),
)

NOTES_HTML = """
<h3>Correlation</h3>
<p>A model of a structure (finite elements, or here a chain of masses and springs) is never quite right: a
joint stiffer than drawn, a mass left out. A modal test measures the real thing, and <b>correlation</b>
compares the two mode by mode. Here the Simulation page's chain is the structure, and the model is a copy
of it with errors set on the left: since both are known, what the correlation says can be checked against
what is wrong.</p>

<h3>The modal assurance criterion</h3>
<p>The MAC compares two shapes, whatever their scale:</p>
<p align="center">MAC(ψ, φ) = |ψ<sup>H</sup>φ|² / (ψ<sup>H</sup>ψ φ<sup>T</sup>φ),</p>
<p>1 when they are the same shape, 0 when they are orthogonal. It is worked out at the sensors only: the test
knows nothing else. Every test mode is <b>paired</b> with the model mode it matches best, best pair first,
and only above a MAC of 0.5. Pairing by frequency alone goes wrong once the model's errors move modes past
each other; the shapes don't. A good correlation has MACs above about 0.9 on the diagonal, small ones off
it, and frequency errors of a few per cent.</p>

<h3>What the shapes don't see</h3>
<p>A model uniformly too stiff, or too heavy, has exactly the right shapes and wrong frequencies. A single
spring 30 % too stiff barely changes the shapes either: on a chain the MAC stays near 0.99. Frequency errors
are the sharper evidence, and the pattern of them over the modes says where the error is: a spring
changes a mode's frequency in proportion to how much it stretches in it, a mass in proportion to how much it
moves. That is what sensitivity-based updating uses.</p>

<h3>Sensors</h3>
<p>With too few sensors, or badly placed ones, different modes look alike: their shapes, seen at those points
only, are nearly parallel. The test's <b>auto-MAC</b> (its shapes against each other) shows it: its
off-diagonal terms rise. Sensors are placed to keep them small. A sensor at a node of a mode sees nothing of
it.</p>

<h3>COMAC</h3>
<p>The coordinate MAC turns the comparison round: per sensor, over all the pairs, how well the test's and the
model's shapes agree there. Where it drops the shapes disagree, which can point to a region of the model to
look at. It is a weak locator: a stiffness error changes the shapes everywhere a little, and the test's own
noise is spread over every sensor.</p>

<h3>The test's own errors</h3>
<p>The test is not exact either: its noise and its curve fit give each mode an error of its own, and weakly
excited modes may be missed. The correlation can't be finer than that: with no error at all in the model, the
MACs and frequency errors left are the test's.</p>
"""


def _plot(left: str, bottom: str) -> pg.PlotWidget:
    p = pg.PlotWidget()
    p.setMenuEnabled(False)
    p.hideButtons()
    p.showGrid(x=True, y=True, alpha=0.15)
    p.setLabel("left", left)
    p.setLabel("bottom", bottom)
    p.getAxis("left").enableAutoSIPrefix(False)
    p.getAxis("bottom").enableAutoSIPrefix(False)
    p.setMouseEnabled(x=False, y=False)
    return p


def _cell(text: str, colour: str | None = None) -> QtWidgets.QTableWidgetItem:
    item = QtWidgets.QTableWidgetItem(text)
    item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
    if colour:
        item.setForeground(pg.mkColor(colour))
    return item


class ModelUpdatingPage(QtWidgets.QWidget):
    """The test and the model's errors on the left, the MAC and the pairs in the centre, the rest on the right."""

    dof_requested = QtCore.Signal(int)  # the user changed N here; the main window owns the system
    edit_parameters = QtCore.Signal()  # the user wants the Simulation page's parameter panel

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.system: ChainSystem | None = None
        self.result: ModalResult | None = None
        self.errors: Errors | None = None
        self.test: Test | None = None
        self.corr: Correlation | None = None
        self.seed = 1
        self._remeasure = True
        self._dirty = False
        self._loading = False
        self._recompute = QtCore.QTimer(self)
        self._recompute.setSingleShot(True)
        self._recompute.setInterval(RECOMPUTE_MS)
        self._recompute.timeout.connect(self.recompute)

        # --- left: the structure, the test, the model's errors, errors to try
        box = QtWidgets.QGroupBox("Test")
        form = QtWidgets.QFormLayout(box)
        self.dof = QtWidgets.QSpinBox()
        self.dof.setRange(1, MAX_DOF)
        self.dof.setToolTip("Same chain as the Simulation page, the structure tested; changing it here changes it "
                            "there too.")
        self.dof.valueChanged.connect(self.dof_requested)
        form.addRow("Number of masses:", self.dof)
        self.source = QtWidgets.QComboBox()
        for s in Source:
            self.source.addItem(s.value, s)
        self.source.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.source.setToolTip(TIPS["source"])
        form.addRow("Modes from:", self.source)
        self.sensor_row = QtWidgets.QWidget()
        self.sensor_layout = FlowLayout(self.sensor_row)
        self.sensor_layout.setContentsMargins(0, 0, 0, 0)
        self.sensor_row.setToolTip(TIPS["sensors"])
        self.sensors: list[QtWidgets.QCheckBox] = []
        label = QtWidgets.QLabel("Sensors:")
        label.setToolTip(TIPS["sensors"])
        form.addRow(label, self.sensor_row)
        self.again = QtWidgets.QPushButton("Measure again")
        self.again.setToolTip(TIPS["again"])
        form.addRow(self.again)
        self.test_info = QtWidgets.QLabel()
        self.test_info.setWordWrap(True)
        mute(self.test_info)
        form.addRow(self.test_info)
        link = QtWidgets.QLabel("The structure is the chain on the <a href='#sim'>Simulation page</a>.")
        mute(link)
        link.setWordWrap(True)
        link.linkActivated.connect(lambda _: self.edit_parameters.emit())
        form.addRow(link)

        model_box = QtWidgets.QGroupBox("The model's errors")
        mv = QtWidgets.QVBoxLayout(model_box)
        self.error_grid = QtWidgets.QGridLayout()
        self.error_grid.setHorizontalSpacing(4)
        for col, head in enumerate(("", "× mass", "× spring", "× damper")):
            h = QtWidgets.QLabel(head)
            h.setToolTip(TIPS["errors"])
            self.error_grid.addWidget(h, 0, col)
        self.error_rows: list[tuple[QtWidgets.QLabel, QtWidgets.QDoubleSpinBox, QtWidgets.QDoubleSpinBox,
                                    QtWidgets.QDoubleSpinBox]] = []
        mv.addLayout(self.error_grid)
        self.clear_errors = QtWidgets.QPushButton("No errors")
        self.clear_errors.setToolTip("<p>Set every factor to 1: the model is then the structure.</p>")
        mv.addWidget(self.clear_errors)

        trials = QtWidgets.QGroupBox("Things to try")
        tv = QtWidgets.QVBoxLayout(trials)
        self.trial = QtWidgets.QComboBox()
        self.trial.addItems([name for name, _, _ in TRIALS])
        self.trial.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.trial.setToolTip("<p>Model errors (and tests) that show something about correlation, set on the chain "
                              "as it is now.</p>")
        tv.addWidget(self.trial)
        self.trial_note = QtWidgets.QLabel()
        self.trial_note.setWordWrap(True)
        mute(self.trial_note)
        tv.addWidget(self.trial_note)

        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        for b in (box, trials, model_box):
            lv.addWidget(b)
        lv.addStretch(1)
        left_scroll = ColumnScroll(left)

        # --- centre: the MAC matrix and the pairs
        self.mac_plot = MacPlot()
        self.mac_plot.setLabel("bottom", "Model mode")
        self.mac_plot.setLabel("left", "Test mode")
        self.mac_plot.setTitle("MAC: test against model shapes", size="9pt")
        self.mac_plot.setToolTip(MAC_TIP)
        plot_height(self.mac_plot, 220)
        self.pair_boxes: list[pg.PlotDataItem] = []
        self.table = QtWidgets.QTableWidget()
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.setToolTip("<p>Each test mode, the model mode it pairs with, their frequencies and damping ratios "
                              "and the MAC between their shapes. Select one to see its shapes.</p>")
        self.readout = QtWidgets.QLabel()
        self.readout.setWordWrap(True)
        self.readout.setTextFormat(QtCore.Qt.TextFormat.RichText)
        center = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(center)
        cv.addWidget(self.mac_plot, 3)
        cv.addWidget(self.table, 2)
        cv.addWidget(self.readout)

        # --- right: frequencies, shapes, auto-MAC, COMAC, notes
        self.freq_plot = _plot("Model frequency (Hz)", "Test frequency (Hz)")
        self.freq_plot.setToolTip(FREQ_TIP)
        self.error_plot = _plot("Frequency error (%)", "Test mode")
        self.error_plot.setToolTip(FREQ_TIP)
        plot_height(self.freq_plot, 180)
        plot_height(self.error_plot, 130)
        freq_tab = QtWidgets.QWidget()
        fv = QtWidgets.QVBoxLayout(freq_tab)
        fv.addWidget(self.freq_plot, 3)
        fv.addWidget(self.error_plot, 2)

        self.shape_plot = _plot("Shape", "Mass")
        self.shape_plot.setToolTip(SHAPE_TIP)
        self.shape_legend = add_legend(self.shape_plot.getPlotItem(), offset=(-10, 10))
        plot_height(self.shape_plot, 200)
        self.shape_note = QtWidgets.QLabel()
        self.shape_note.setWordWrap(True)
        mute(self.shape_note)
        shape_tab = QtWidgets.QWidget()
        sv = QtWidgets.QVBoxLayout(shape_tab)
        sv.addWidget(self.shape_plot, 1)
        sv.addWidget(self.shape_note)

        self.auto_plot = MacPlot()
        self.auto_plot.setLabel("bottom", "Test mode")
        self.auto_plot.setLabel("left", "Test mode")
        self.auto_plot.setTitle("Auto-MAC: the test's shapes against each other", size="9pt")
        self.auto_plot.setToolTip(AUTO_TIP)
        plot_height(self.auto_plot, 200)
        self.auto_note = QtWidgets.QLabel()
        self.auto_note.setWordWrap(True)
        self.auto_note.setTextFormat(QtCore.Qt.TextFormat.RichText)
        auto_tab = QtWidgets.QWidget()
        av = QtWidgets.QVBoxLayout(auto_tab)
        av.addWidget(self.auto_plot, 1)
        av.addWidget(self.auto_note)

        self.comac_plot = _plot("COMAC", "Sensor")
        self.comac_plot.setToolTip(COMAC_TIP)
        plot_height(self.comac_plot, 200)
        self.comac_note = QtWidgets.QLabel()
        self.comac_note.setWordWrap(True)
        self.comac_note.setTextFormat(QtCore.Qt.TextFormat.RichText)
        comac_tab = QtWidgets.QWidget()
        ov = QtWidgets.QVBoxLayout(comac_tab)
        ov.addWidget(self.comac_plot, 1)
        ov.addWidget(self.comac_note)

        self.notes = QtWidgets.QTextBrowser()
        self.notes.setHtml(NOTES_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.freq_tab, self.shape_tab, self.auto_tab, self.comac_tab = freq_tab, shape_tab, auto_tab, comac_tab
        self.tabs.addTab(freq_tab, "Frequencies")
        self.tabs.addTab(shape_tab, "Shapes")
        self.tabs.addTab(auto_tab, "Auto-MAC")
        self.tabs.addTab(comac_tab, "COMAC")
        self.tabs.addTab(self.notes, "Notes")

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(left_scroll)
        splitter.addWidget(center)
        splitter.addWidget(self.tabs)
        splitter.setSizes([360, 700, 640])
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        # --- wiring
        self.source.currentIndexChanged.connect(lambda _: self._schedule(measure=True))
        self.again.clicked.connect(self._on_again)
        self.clear_errors.clicked.connect(self._on_clear)
        self.trial.currentIndexChanged.connect(lambda _: self.apply_trial())
        self.table.itemSelectionChanged.connect(self._draw_shapes)

    # ------------------------------------------------------------- inputs
    def set_system(self, system: ChainSystem, result: ModalResult | None = None) -> None:
        """The structure changed: measured again (now if shown); the model's errors stay as they are."""
        n_changed = self.system is None or system.n != self.system.n
        self.system = system
        self.result = modal_analysis(ChainSystem(system.masses, system.stiffness, system.damping)) \
            if result is None or system.friction.any() else result
        self._loading = True
        self.dof.blockSignals(True)
        self.dof.setValue(system.n)
        self.dof.blockSignals(False)
        if n_changed:
            self._build_rows(system.n)
        self._loading = False
        if self.errors is None:
            self.apply_trial()
        else:
            self._schedule(measure=True)

    def _build_rows(self, n: int) -> None:
        kept = [box.isChecked() for box in self.sensors]
        for box in self.sensors:
            self.sensor_layout.removeWidget(box)
            box.deleteLater()
        self.sensors = []
        for i in range(n):
            box = QtWidgets.QCheckBox(f"m{i + 1}")
            box.setChecked(kept[i] if i < len(kept) else True)
            box.setToolTip(TIPS["sensors"])
            box.toggled.connect(lambda _: self._schedule(measure=False))
            self.sensor_layout.addWidget(box)
            self.sensors.append(box)
        values = [[s.value() for s in row[1:]] for row in self.error_rows]
        while self.error_rows:
            for w in self.error_rows.pop():
                self.error_grid.removeWidget(w)
                w.deleteLater()
        for i in range(n):
            label = QtWidgets.QLabel(f"{i + 1}")
            row = [label]
            for col in range(3):
                box = spin(0.05, 20.0, values[i][col] if i < len(values) else 1.0, 3)
                box.setToolTip(TIPS["errors"])
                box.valueChanged.connect(self._on_errors)
                row.append(box)
            for col, w in enumerate(row):
                self.error_grid.addWidget(w, i + 1, col)
            self.error_rows.append(tuple(row))
        self.errors = self._read_errors() if self.errors is not None else None

    def _read_errors(self) -> Errors:
        cols = np.array([[box.value() for box in row[1:]] for row in self.error_rows]).reshape(-1, 3)
        return Errors(cols[:, 0], cols[:, 1], cols[:, 2])

    def _set_errors(self, errors: Errors) -> None:
        self._loading = True
        for row, m, k, c in zip(self.error_rows, errors.mass, errors.stiffness, errors.damping):
            for box, v in zip(row[1:], (m, k, c)):
                box.setValue(float(v))
        self._loading = False
        self.errors = errors

    def apply_trial(self) -> None:
        """Set the errors (and the test) the chosen trial asks for, on the chain as it is."""
        if self.system is None:
            return
        _, changes, note = TRIALS[self.trial.currentIndex()]
        n = self.system.n
        e = Errors.none(n)
        for key, arr in (("m", e.mass), ("k", e.stiffness), ("c", e.damping)):
            for i, f in changes.get(key, {}).items():
                arr[min(i % n, n - 1)] = f
        e.stiffness[:] *= changes.get("all_k", 1.0)
        e.damping[:] *= changes.get("all_c", 1.0)
        self._set_errors(e)
        want = {i % n for i in changes.get("sensors", range(n))}
        self._loading = True
        for i, box in enumerate(self.sensors):
            box.setChecked(i in want)
        if "source" in changes:
            self.source.setCurrentIndex(self.source.findData(changes["source"]))
        self._loading = False
        self.trial_note.setText(note)
        self._schedule(measure=self.test is None or self.test.source is not self.source.currentData())

    def _on_errors(self, *_args) -> None:
        if not self._loading:
            self.errors = self._read_errors()
            self._schedule(measure=False)

    def _on_clear(self) -> None:
        if self.system is not None:
            self._set_errors(Errors.none(self.system.n))
            self._schedule(measure=False)

    def _on_again(self) -> None:
        self.seed += 1
        self._schedule(measure=True, now=True)

    def sensor_mask(self) -> np.ndarray:
        return np.array([box.isChecked() for box in self.sensors], dtype=bool)

    def _schedule(self, measure: bool, now: bool = False) -> None:
        if self._loading:
            self._remeasure |= measure
            return
        self._remeasure |= measure
        if not self.isVisible():
            self._dirty = True
            return
        if now:
            self.recompute()
        else:
            self._recompute.start()

    def showEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().showEvent(event)
        if self._dirty:
            self.recompute()

    # ------------------------------------------------------------- computing
    def recompute(self) -> None:
        """Measure again if asked (or the structure changed), then correlate and draw."""
        self._recompute.stop()
        self._dirty = False
        if self.system is None or self.errors is None:
            return
        mask = self.sensor_mask()
        if mask.sum() == 0:
            self.readout.setText("<b>No sensors</b>: tick at least one mass.")
            return
        if self._remeasure or self.test is None:
            self.test = measure(self.system, self.source.currentData(), mask, self.seed, self.result)
            self._remeasure = False
        self.corr = correlate(self.system, self.errors, self.test.with_sensors(mask))
        self.redraw()

    # ------------------------------------------------------------- drawing
    def apply_theme(self) -> None:
        for p in (self.mac_plot, self.auto_plot):
            p.apply_theme()
        if self.corr is not None:
            self.redraw()

    def redraw(self) -> None:
        c = self.corr
        test = c.test
        n_model = len(c.result.modes)
        found = len(test.modes)
        self.test_info.setText(
            f"The test found {found} of the structure's {n_model} modes, measured at {int(test.sensors.sum())} "
            f"sensor{'s' if test.sensors.sum() != 1 else ''}.")
        rows = [f"T{i + 1}" for i in range(found)]
        cols = [f"M{j + 1}" for j in range(n_model)]
        self.mac_plot.set_matrix(c.macs, rows, cols)
        for item in self.pair_boxes:
            self.mac_plot.removeItem(item)
        self.pair_boxes = []
        for p in c.pairs:
            x, y = p.model, p.test
            box = pg.PlotDataItem([x, x + 1, x + 1, x, x], [y, y, y + 1, y + 1, y],
                                  pen=pg.mkPen(colors.force, width=2))
            self.mac_plot.addItem(box)
            self.pair_boxes.append(box)
        self._fill_table()
        self._draw_frequencies()
        self._draw_shapes()
        self._draw_auto()
        self._draw_comac()
        if c.pairs:
            worst = max(c.pairs, key=lambda p: abs(p.error))
            low = min(c.pairs, key=lambda p: p.mac)
            unpaired = found - len(c.pairs)
            self.readout.setText(
                f"{len(c.pairs)} pairs. Largest frequency error: <b>{100 * worst.error:+.2f} %</b> (test mode "
                f"{worst.test + 1}, model mode {worst.model + 1}); lowest MAC <b>{low.mac:.3f}</b> (test mode "
                f"{low.test + 1})." + (f" {unpaired} test mode{'s' if unpaired > 1 else ''} unpaired (MAC under "
                                       "0.5 with every model mode)." if unpaired else ""))
        else:
            self.readout.setText("No pairs: no test shape matches a model shape with a MAC of 0.5 or more.")

    def _fill_table(self) -> None:
        c = self.corr
        t = self.table
        current = t.currentRow()
        t.blockSignals(True)
        t.clear()
        heads = ["Test", "f test (Hz)", "ζ test", "Model", "f model (Hz)", "ζ model", "Δf", "MAC"]
        t.setColumnCount(len(heads))
        t.setHorizontalHeaderLabels(heads)
        t.setRowCount(len(c.test.modes))
        by_test = {p.test: p for p in c.pairs}
        for i, m in enumerate(c.test.modes):
            cells = [_cell(f"T{i + 1}"), _cell(f"{m.fn_hz:.4g}"), _cell(f"{m.zeta:.3g}")]
            p = by_test.get(i)
            if p is None:
                cells += [_cell("unpaired", colors.poor)] + [_cell("") for _ in range(4)]
            else:
                mode = c.result.modes[p.model]
                zeta = mode.damped.zeta if mode.damped is not None else float("nan")
                text, colour = colored(100 * p.error, 1.0, 3.0, f"{100 * p.error:+.2f} %")
                mtext, mcolour = colored(1 - p.mac, 0.02, 0.1, f"{p.mac:.3f}")
                cells += [_cell(f"M{p.model + 1}", colors.mode[p.model % len(colors.mode)]),
                          _cell(f"{p.f_model:.4g}"), _cell(f"{zeta:.3g}"), _cell(text, colour), _cell(mtext, mcolour)]
            for j, item in enumerate(cells):
                t.setItem(i, j, item)
        t.resizeColumnsToContents()
        t.selectRow(current if 0 <= current < t.rowCount() else 0)
        t.blockSignals(False)

    def _draw_frequencies(self) -> None:
        c = self.corr
        p, q = self.freq_plot, self.error_plot
        p.clear()
        q.clear()
        if not c.pairs:
            return
        ft = np.array([pr.f_test for pr in c.pairs])
        fm = np.array([pr.f_model for pr in c.pairs])
        top = 1.1 * max(ft.max(), fm.max())
        p.plot([0, top], [0, top], pen=pg.mkPen(colors.faint, width=1))
        brushes = [pg.mkBrush(colors.mode[pr.model % len(colors.mode)]) for pr in c.pairs]
        p.addItem(pg.ScatterPlotItem(ft, fm, size=10, pen=None, brush=brushes))
        p.setRange(xRange=(0, top), yRange=(0, top), padding=0)
        err = np.array([100 * pr.error for pr in c.pairs])
        x = np.array([pr.test + 1 for pr in c.pairs], dtype=float)
        q.addItem(pg.BarGraphItem(x=x, height=err, width=0.6, brushes=brushes, pen=None))
        q.addItem(pg.InfiniteLine(0.0, angle=0, pen=pg.mkPen(colors.faint, width=1)))
        span = max(1.0, 1.2 * float(np.abs(err).max()))
        q.setRange(xRange=(0.4, len(c.test.modes) + 0.6), yRange=(-span, span), padding=0)
        q.getAxis("bottom").setTicks([[(i + 1, f"T{i + 1}") for i in range(len(c.test.modes))]])

    def _draw_shapes(self) -> None:
        c = self.corr
        if c is None:
            return
        p = self.shape_plot
        p.clear()
        self.shape_legend.clear()
        row = self.table.currentRow()
        pr = next((x for x in c.pairs if x.test == row), None)
        sensors = np.flatnonzero(c.test.sensors)
        if pr is None:
            self.shape_note.setText("Select a paired test mode in the table.")
            return
        psi = c.test.modes[pr.test].shape[sensors]
        phi = c.result.modes[pr.model].shape[sensors]
        psi = psi * np.exp(-1j * complexity(psi).angle)  # onto its best-fit line
        real = psi.real
        scale = float(real @ phi / (real @ real)) if real @ real > 0 else 1.0
        n = self.system.n
        full_phi = np.concatenate([[0.0], c.result.modes[pr.model].shape])
        colour = colors.mode[pr.model % len(colors.mode)]
        model = p.plot(np.arange(n + 1), full_phi, pen=pg.mkPen(colour, width=2))
        test = p.plot(sensors + 1, scale * real, pen=None, symbol="o", symbolSize=9, symbolBrush=pg.mkBrush(colors.strong),
                      symbolPen=None)
        self.shape_legend.addItem(model, f"Model mode {pr.model + 1}")
        self.shape_legend.addItem(test, f"Test mode {pr.test + 1}")
        p.addItem(pg.InfiniteLine(0.0, angle=0, pen=pg.mkPen(colors.faint, width=1)))
        p.setXRange(-0.25, n + 0.25, padding=0)
        p.setYRange(-1.15, 1.15, padding=0)
        p.getAxis("bottom").setTicks([[(i + 1, f"m{i + 1}") for i in range(n)]])
        self.shape_note.setText(
            f"Test mode {pr.test + 1} ({pr.f_test:.4g} Hz, dots at the sensors) against model mode {pr.model + 1} "
            f"({pr.f_model:.4g} Hz, the line): MAC {pr.mac:.3f}.")

    def _draw_auto(self) -> None:
        c = self.corr
        k = len(c.test.modes)
        names = [f"T{i + 1}" for i in range(k)]
        self.auto_plot.set_matrix(c.auto_mac, names, names)
        if k < 2:
            self.auto_note.setText("")
            return
        off = c.auto_mac - np.diag(np.diag(c.auto_mac))
        i, j = np.unravel_index(int(np.argmax(off)), off.shape)
        sensors = int(c.test.sensors.sum())
        text = f"Largest off-diagonal term: <b>{off[i, j]:.2f}</b>, test modes {i + 1} and {j + 1}."
        if off[i, j] > 0.5:
            text += (f" At {sensors} sensor{'s' if sensors > 1 else ''} these modes look alike: add a sensor or "
                     "move one.")
        self.auto_note.setText(text)

    def _draw_comac(self) -> None:
        c = self.corr
        p = self.comac_plot
        p.clear()
        sensors = np.flatnonzero(c.test.sensors)
        values = np.nan_to_num(c.comac, nan=0.0)
        p.addItem(pg.BarGraphItem(x=sensors + 1, height=values, width=0.6,
                                  brushes=[pg.mkBrush(colors.mass[i]) for i in sensors], pen=None))
        p.setRange(xRange=(0.4, self.system.n + 0.6), yRange=(min(0.8, float(values.min()) - 0.02), 1.005), padding=0)
        p.getAxis("bottom").setTicks([[(i + 1, f"m{i + 1}") for i in sensors]])
        if c.pairs and sensors.size:
            low = int(np.argmin(values))
            self.comac_note.setText(f"Lowest at m{sensors[low] + 1}: <b>{values[low]:.3f}</b>, over {len(c.pairs)} "
                                    "pairs. A weak locator: a stiffness error changes every shape a little.")
        else:
            self.comac_note.setText("")
