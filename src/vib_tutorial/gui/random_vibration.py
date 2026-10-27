"""Random vibration page: the chain under a random force or a shaken base, its response's PSD and RMS.

Left: the input (a force on one mass or every mass, or the base's acceleration, flat over a
band), the response looked at (a mass's displacement or acceleration, or a spring's force),
the time record (its length, Welch's block) and things to try. Centre: the input's PSD, the
response's PSD (exact H G Hᴴ and the record's Welch estimate) and its cumulative RMS, with
Miles' estimate. Right: Miles' equation mode by mode, the record itself, its statistics against
Rice's and the largest peak's (Davenport's, Vanmarcke's), and the notes.
Follows the chain from the Simulation page (friction left out: all of this is linear).
"""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.modal import ModalResult, modal_analysis
from ..core.model import ChainSystem
from ..core.random_vibration import (
    BLOCKS,
    LEVELS,
    Input,
    Output,
    Prediction,
    Quantity,
    RandomInput,
    RandomRecord,
    cumulative_rms,
    gaussian_exceedance,
    linear,
    peak_pdf,
    predict,
    record_statistics,
    rice,
    rms_scatter,
    sample_rate,
    segment_maxima,
    simulate,
    suggested_block,
    suggested_duration,
    welch,
)
from .axes import log_axes
from .layouts import ColumnScroll, group, plot_height
from .links import before, takeaways
from .panels import spin
from .style import MAX_DOF, colors
from .theming import add_legend, mute

RECOMPUTE_MS = 150  # the parameters settle this long before the record is made again
SHOWN_PERIODS = 30  # of mode 1, on the record's plot
PSD_UNITS = {Input.BASE: "g²/Hz", Input.FORCE_ONE: "N²/Hz", Input.FORCE_ALL: "N²/Hz"}

TIPS = {
    "input": "<p><b>Force on one mass</b> or <b>on every mass</b> (independent forces, each with the PSD), or "
             "the <b>base's acceleration</b>, as a shaker table or a vehicle's floor drives it: the chain then "
             "moves relative to the base, each mass pushed by −m a<sub>g</sub>.</p>",
    "level": "<p>The input's power spectral density: its mean square per hertz, flat across the band. A force's in "
             "N²/Hz, the base's in g²/Hz, as test specifications give it. Its RMS is √(level × bandwidth).</p>",
    "band": "<p>The band the input covers: nothing outside it. The response has nothing outside it either.</p>",
    "output": "<p>The response to look at: a mass's displacement (relative to the base, for a base input), its "
              "acceleration (absolute, in g), or a spring's force (spring 1 ties m1 to the base).</p>",
    "duration": "<p>The simulated record's length. A lightly damped mode needs a long one: Welch's lines must be "
                "finer than its peak, and each line needs many averages.</p>",
    "block": "<p>Welch's block: its length sets the lines' spacing, f<sub>s</sub>/block, and the record's length "
             "over it the number of averages. Long blocks resolve the peaks; short ones average more.</p>",
    "new": "<p>A new record: the same spectrum, other random numbers.</p>",
}
INPUT_TIP = "<p>The input's PSD: the level asked for (flat over the band), and the record's input by Welch.</p>"
PSD_TIP = (
    "<p>The response's power spectral density: exact, G<sub>y</sub> = H G<sub>u</sub> H<sup>H</sup>, and the "
    "simulated record's by Welch's method. Each mode is a peak; the dotted lines mark them.</p>")
CUMULATIVE_TIP = (
    "<p>The RMS so far, √∫<sub>0</sub><sup>f</sup> G df: each mode a step, its height its share. Dashed: the "
    "Miles estimate (the modes' together) and the record's standard deviation.</p>")
RECORD_TIP = "<p>The start of the simulated response, with ±1σ and ±3σ of the exact RMS.</p>"
AMPLITUDE_TIP = (
    "<p>How the record's values are distributed, over σ (the exact RMS), against the Gaussian: a linear system's "
    "response to a Gaussian input is Gaussian. Dashed: ±1, 2 and 3σ.</p>")
PEAKS_TIP = (
    "<p>The heights of the record's peaks (its local maxima), over σ, against Rice's distribution for the "
    "response's irregularity α = ν₀/ν<sub>p</sub>: Rayleigh for a narrow band (α = 1: one peak a cycle, all "
    "above zero), towards Gaussian for a broad one (many small wiggles, some below zero).</p>")
LARGEST_TIP = (
    "<p>How long a stretch to ask the largest peak of. The record is cut into stretches this long, and their "
    "largest values averaged, to compare with Davenport's and Vanmarcke's expected largest peak.</p>")
AMPLITUDE_BINS = 60
PEAK_BINS = 40

# Things to try: (name, what it sets, note)
TRIALS = (
    ("Base shake", {},
     "The base shaken flat at 0.01 g²/Hz over the chain's modes, the last mass's acceleration: each mode a "
     "peak, and Miles' equation, mode by mode, within a few per cent of the exact RMS. The record's RMS "
     "agrees to its scatter."),
    ("Force on the last mass", {"input": Input.FORCE_ONE, "mass": -1},
     "A random force on the last mass, its own acceleration: above the modes it is the force over the mass "
     "(the mass line), not a resonance, and Miles' equation misses it: the wider the band, the more."),
    ("Forces on every mass", {"input": Input.FORCE_ALL, "output": Quantity.DISPLACEMENT},
     "Independent forces on every mass (wind, say), the last mass's displacement: dominated by mode 1, the "
     "most flexible."),
    ("The base spring's force", {"output": Quantity.SPRING, "index": 0},
     "Spring 1, which carries the whole chain off the base: the force a fatigue check would need. Its RMS is "
     "almost all mode 1."),
    ("Short record", {"duration": 1 / 16},
     "A sixteenth of the record: few blocks to average, and the Welch estimate is rough; its RMS scatters "
     "round the exact one (press New record a few times)."),
    ("Coarse lines", {"block": 1 / 16},
     "Blocks a sixteenth as long: lines coarser than mode 1's peak, which comes out low and wide (the window "
     "smears it). The RMS is still right: the area is kept."),
    ("Narrow band on mode 2", {"band": (0.9, 1.1)},
     "Only ±10 % round mode 2: one peak, its skirts cut off. Miles' equation, which counts the whole of a "
     "mode's peak, now overestimates."),
)

NOTES_HTML = """
<!--BEFORE-->
<h3>Random vibration</h3>
<p>Wind, a road, a rocket's engines: many inputs are random, never the same twice, but their statistics are
steady. A stationary random signal is described by its <b>power spectral density</b> (PSD) G(f), its mean square
per hertz: the RMS² is the area under it. A flat PSD across a band is band-limited white noise. A random
shaker test is usually specified as such a band, giving the base's acceleration PSD in g²/Hz. For example,
0.01 g²/Hz from 20 to 2000 Hz has an area of 0.01 × 1980 = 19.8 g², so its RMS is √19.8 = 4.4 g (written
4.4 Grms).</p>

<h3>The response</h3>
<p>A linear system turns a random input into a random output, Gaussian if the input is, with PSD</p>
<p align="center">G<sub>y</sub>(f) = H(f) G<sub>u</sub>(f) H(f)<sup>H</sup>,</p>
<p>H the frequency response from the input to the output: here (K − ω²M + iωC)<sup>−1</sup> applied to the
input's forces, read off as a displacement, an acceleration (−ω² × displacement) or a spring's force. Inputs
that are independent add their PSDs. The response's RMS is √∫G<sub>y</sub> df, and its running value, the
<b>cumulative RMS</b>, climbs a step at each mode: how high each step is says how much that mode contributes.
On a shaken base the chain moves relative to the base: in those coordinates the base's acceleration
a<sub>g</sub> pushes each mass with −m a<sub>g</sub>, and the absolute acceleration is the relative one plus
a<sub>g</sub>.</p>

<h3><a name="miles"></a>Miles' equation</h3>
<p>A lightly damped mode amplifies its input by Q = 1/2ζ over a band about 2ζ f<sub>n</sub> wide. If the input
is flat across that band, the area under the peak is (π/2) f<sub>n</sub> Q times the input there, and a single
mass on a shaken base has an absolute acceleration of</p>
<p align="center">a<sub>RMS</sub> = √((π/2) f<sub>n</sub> Q G<sub>a</sub>(f<sub>n</sub>)),</p>
<p><b>Miles' equation</b>, the most used formula in random vibration. On the chain each mode is such a
system in its modal coordinate, driven by the modal force φ<sup>T</sup>f: its RMS follows, each response
takes the mode shape's share of it, and the modes combine by the square root of the sum of their squares.
The <i>Miles</i> tab lists them. It is good where the response is resonant and the modes are well apart. It
misses what isn't resonant: a forced mass's acceleration far above the modes is the force over the mass (the
<b>mass line</b>), modes outside the band, and a band edge that cuts across a peak.</p>

<h3>How often, how high</h3>
<p>A linear system's response to a Gaussian input is Gaussian: 68.3 % of the time within ±1σ, 95.4 % within
±2σ, 99.73 % within ±3σ, its skewness 0 and kurtosis 3. How often it does things comes from the PSD's
<b>spectral moments</b> m<sub>k</sub> = ∫ f<sup>k</sup> G df. <b>Rice</b>: it crosses zero upwards
ν₀ = √(m₂/m₀) times a second, a level a ν₀ e<sup>−a²/2σ²</sup> times, and has ν<sub>p</sub> = √(m₄/m₂) peaks a
second. Their ratio, the <b>irregularity factor</b> α = ν₀/ν<sub>p</sub>, says how narrow its band is: at 1, one
peak per cycle, their heights Rayleigh distributed; towards 0, many small peaks riding on the big swings, some
below zero, their heights nearly Gaussian.</p>
<p>The <b>largest peak</b> in a time T grows only slowly with T. Counting each upward crossing as independent,
<b>Davenport</b> finds its expected value σ(√(2 ln ν₀T) + 0.577/√(2 ln ν₀T)): about 3.2σ over a hundred cycles,
4.1σ over three thousand, 5.1σ over 270,000. "3σ" is not a maximum: a long life sees more. A narrow-band response's
crossings come in clumps, on the swells of its slowly varying envelope, so there are fewer independent chances
than ν₀T, and Davenport overestimates the largest peak, most over short times. <b>Vanmarcke</b> counts the clumps,
through the band's width δ = √(1 − m₁²/m₀m₂). The <i>Statistics</i> tab compares both with the record's own
largest values, stretch by stretch.</p>

<h3>Estimating it from a record</h3>
<p><b>Welch's method</b> cuts the record into blocks, windows each (Hann), Fourier transforms them and averages
|Y|². Two errors trade against each other:</p>
<ul>
<li><b>Bias</b>: the lines are f<sub>s</sub>/block apart, and the window smears each over a few. A peak
narrower than that comes out low and wide. Put four or more lines across the half-power width
2ζf<sub>n</sub>.</li>
<li><b>Random error</b>: one block's estimate scatters by 100 %; averaging n blocks cuts it to 1/√n.</li>
</ul>
<p>Fine lines need long blocks and many averages need many blocks: a lightly damped mode needs a long record.
The RMS itself is less fussy, a whole area: its scatter is about 1/(2√(B T)) for a record T long, B the
response's bandwidth.</p>

<h3><a name="computing"></a>How this page computes it</h3>
<p>The simulated record comes from the exact update, its input synthesized to have the PSD asked for, so the
differences between the record's estimates and the exact PSD are the estimates', not the simulation's. The
record holds 2.5 samples per period of the band's top: enough to describe it, since it has nothing above the
band, but not to see its peaks. It is interpolated eightfold before they are counted.</p>
<!--TAKEAWAYS-->
"""
NOTES_HTML = NOTES_HTML.replace("<!--BEFORE-->", before(
    ("sim_page", "sweeps-and-ground-motion", "Sweeps and ground motion"),
    ("frf_page", "sum-of-modes", "Every term is a sum of modes"),
)).replace("<!--TAKEAWAYS-->", takeaways(
    "A random input is described by its PSD; a linear system's response PSD is H G Hᴴ, and its RMS is the "
    "root of the area under it.",
    "The cumulative RMS steps up at each mode: the steps say which modes matter.",
    "Miles' equation, √((π/2) f<sub>n</sub> Q G), gives a lightly damped mode's RMS from the input at its own"
    " frequency.",
    "The response is Gaussian, but the largest peak in a record grows with its length: 3σ is not a maximum.",
    "A PSD estimate trades resolution (long blocks) against scatter (many averages).",
))



def _plot(left: str, left_units: str | None, bottom: str, bottom_units: str | None, log: bool = True
          ) -> pg.PlotWidget:
    p = pg.PlotWidget(axisItems=log_axes()) if log else pg.PlotWidget()
    p.setMenuEnabled(False)
    p.hideButtons()
    p.showGrid(x=True, y=True, alpha=0.15)
    p.setLabel("left", left, units=left_units)
    p.setLabel("bottom", bottom, units=bottom_units)
    p.getAxis("left").enableAutoSIPrefix(False)  # the unit is in the label: no "(x0.001)"
    return p


def _form_row(form: QtWidgets.QFormLayout, text: str, box: QtWidgets.QWidget, tip: str) -> None:
    box.setToolTip(tip)
    label = QtWidgets.QLabel(text)
    label.setToolTip(tip)
    form.addRow(label, box)


def fmt(value: float, unit: str) -> str:
    return f"{value:.4g} {unit}"


class RandomVibrationPage(QtWidgets.QWidget):
    """Controls on the left, the spectra and the cumulative RMS in the centre, Miles, the record and notes on the right."""

    dof_requested = QtCore.Signal(int)  # the user changed N here; the main window owns the system
    edit_parameters = QtCore.Signal()  # the user wants the Simulation page's parameter panel

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.system: ChainSystem | None = None
        self.result: ModalResult | None = None
        self.record: RandomRecord | None = None
        self.prediction: Prediction | None = None
        self.seed = 1
        self._loading = False
        self._dirty = False
        self._remeasure = True
        self._recompute = QtCore.QTimer(self)
        self._recompute.setSingleShot(True)
        self._recompute.setInterval(RECOMPUTE_MS)
        self._recompute.timeout.connect(self.recompute)

        # --- left: the input, the response, the record, things to try
        box = QtWidgets.QGroupBox("Input")
        form = QtWidgets.QFormLayout(box)
        self.dof = QtWidgets.QSpinBox()
        self.dof.setRange(1, MAX_DOF)
        self.dof.setToolTip("Same chain as the Simulation page; changing it here changes it there too.")
        self.dof.valueChanged.connect(self.dof_requested)
        form.addRow("Number of masses:", self.dof)
        self.kind = QtWidgets.QComboBox()
        for k in Input:
            self.kind.addItem(k.value, k)
        self.kind.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        _form_row(form, "Random:", self.kind, TIPS["input"])
        self.mass = QtWidgets.QComboBox()
        _form_row(form, "On:", self.mass, TIPS["input"])
        self.level = spin(1e-6, 1e6, 0.01, 6, " g²/Hz")
        _form_row(form, "Level:", self.level, TIPS["level"])
        self.f_lo = spin(0.0, 1e5, 0.0, 2, " Hz")
        self.f_hi = spin(0.01, 1e5, 10.0, 2, " Hz")
        _form_row(form, "Band:", group(self.f_lo, "to", self.f_hi), TIPS["band"])
        self.input_info = QtWidgets.QLabel()
        self.input_info.setWordWrap(True)
        mute(self.input_info)
        form.addRow(self.input_info)
        link = QtWidgets.QLabel("Masses, springs and dampers are edited on the <a href='#sim'>Simulation page</a>.")
        mute(link)
        link.setWordWrap(True)
        link.linkActivated.connect(lambda _: self.edit_parameters.emit())
        form.addRow(link)

        out_box = QtWidgets.QGroupBox("Response")
        form = QtWidgets.QFormLayout(out_box)
        self.quantity = QtWidgets.QComboBox()
        for q in Quantity:
            self.quantity.addItem(q.value, q)
        self.index = QtWidgets.QComboBox()
        _form_row(form, "Look at:", group(self.quantity, "of", self.index), TIPS["output"])

        rec_box = QtWidgets.QGroupBox("Record")
        form = QtWidgets.QFormLayout(rec_box)
        self.duration = spin(1.0, 1e6, 600.0, 0, " s")
        _form_row(form, "Length:", self.duration, TIPS["duration"])
        self.block = QtWidgets.QComboBox()
        for b in BLOCKS:
            self.block.addItem(f"{b:,}", b)
        _form_row(form, "Welch block:", self.block, TIPS["block"])
        self.new_record = QtWidgets.QPushButton("New record")
        self.new_record.setToolTip(TIPS["new"])
        form.addRow(self.new_record)
        self.record_info = QtWidgets.QLabel()
        self.record_info.setWordWrap(True)
        mute(self.record_info)
        form.addRow(self.record_info)

        trials = QtWidgets.QGroupBox("Things to try")
        tv = QtWidgets.QVBoxLayout(trials)
        self.trial = QtWidgets.QComboBox()
        self.trial.addItems([name for name, _, _ in TRIALS])
        self.trial.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.trial.setToolTip("<p>Settings that show something about random vibration, set from the chain as it "
                              "is now.</p>")
        tv.addWidget(self.trial)
        self.trial_note = QtWidgets.QLabel()
        self.trial_note.setWordWrap(True)
        mute(self.trial_note)
        tv.addWidget(self.trial_note)

        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        for b in (box, out_box, rec_box, trials):
            lv.addWidget(b)
        lv.addStretch(1)
        left_scroll = ColumnScroll(left)

        # --- centre: the input's PSD, the response's, the cumulative RMS
        self.input_plot = _plot("Input PSD", None, "Frequency", "Hz")
        self.input_plot.setLogMode(x=True, y=True)
        self.input_plot.setToolTip(INPUT_TIP)
        plot_height(self.input_plot, 110)
        self.psd_plot = _plot("Response PSD", None, "Frequency", "Hz")
        self.psd_plot.setLogMode(x=True, y=True)
        self.psd_plot.setToolTip(PSD_TIP)
        plot_height(self.psd_plot, 200)
        self.psd_legend = add_legend(self.psd_plot.getPlotItem(), offset=(-10, 10))
        self.cumulative = _plot("Cumulative RMS", None, "Frequency", "Hz")
        self.cumulative.setLogMode(x=True, y=False)
        self.cumulative.setToolTip(CUMULATIVE_TIP)
        plot_height(self.cumulative, 140)
        self.cumulative_legend = add_legend(self.cumulative.getPlotItem(), offset=(10, 10))
        for p in (self.psd_plot, self.cumulative):
            p.setXLink(self.input_plot)
        for p in (self.input_plot, self.psd_plot, self.cumulative):  # setLogMode turned it back on where linear
            p.getAxis("left").enableAutoSIPrefix(False)
        self.readout = QtWidgets.QLabel()
        self.readout.setWordWrap(True)
        self.readout.setTextFormat(QtCore.Qt.TextFormat.RichText)
        center = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(center)
        cv.addWidget(self.input_plot, 2)
        cv.addWidget(self.psd_plot, 5)
        cv.addWidget(self.cumulative, 3)
        cv.addWidget(self.readout)

        # --- right: Miles mode by mode, the record, the notes
        self.miles_table = QtWidgets.QTableWidget()
        self.miles_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.miles_table.verticalHeader().setVisible(False)
        self.miles_table.setToolTip("<p>Each mode in the band as a single-degree-of-freedom system under the input "
                                    "at its frequency: its Q = 1/2ζ, and the RMS Miles' equation gives it. Their "
                                    "SRSS is the Miles estimate; the share is each mode's of its square.</p>")
        self.miles_note = QtWidgets.QLabel()
        self.miles_note.setWordWrap(True)
        self.miles_note.setTextFormat(QtCore.Qt.TextFormat.RichText)
        mute(self.miles_note)
        miles_tab = QtWidgets.QWidget()
        mv = QtWidgets.QVBoxLayout(miles_tab)
        mv.addWidget(self.miles_table, 1)
        mv.addWidget(self.miles_note)

        self.record_plot = _plot("Response", None, "Time", "s", log=False)
        self.record_plot.setToolTip(RECORD_TIP)
        plot_height(self.record_plot, 200)
        self.record_plot.setClipToView(True)
        self.record_plot.setDownsampling(auto=True, mode="peak")
        record_tab = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(record_tab)
        rv.addWidget(self.record_plot, 1)
        self.record_note = QtWidgets.QLabel()
        self.record_note.setWordWrap(True)
        mute(self.record_note)
        rv.addWidget(self.record_note)

        self.amplitude_plot = _plot("Probability density", None, "Value / σ", None, log=False)
        self.amplitude_plot.setToolTip(AMPLITUDE_TIP)
        plot_height(self.amplitude_plot, 130)
        self.amplitude_legend = add_legend(self.amplitude_plot.getPlotItem(), offset=(-10, 10))
        self.peaks_plot = _plot("Probability density", None, "Peak height / σ", None, log=False)
        self.peaks_plot.setToolTip(PEAKS_TIP)
        plot_height(self.peaks_plot, 130)
        self.peaks_legend = add_legend(self.peaks_plot.getPlotItem(), offset=(-10, 10))
        for p in (self.amplitude_plot, self.peaks_plot):
            p.setMouseEnabled(x=False, y=False)
        self.stats_table = QtWidgets.QLabel()
        self.stats_table.setWordWrap(True)
        self.stats_table.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.largest_over = spin(0.1, 1e6, 60.0, 1, " s")
        self.largest_over.setToolTip(LARGEST_TIP)
        self.largest_note = QtWidgets.QLabel()
        self.largest_note.setWordWrap(True)
        self.largest_note.setTextFormat(QtCore.Qt.TextFormat.RichText)
        stats_tab = QtWidgets.QWidget()
        sv = QtWidgets.QVBoxLayout(stats_tab)
        sv.addWidget(self.amplitude_plot, 1)
        sv.addWidget(self.peaks_plot, 1)
        sv.addWidget(self.stats_table)
        row = QtWidgets.QFormLayout()
        label = QtWidgets.QLabel("Largest peak over:")
        label.setToolTip(LARGEST_TIP)
        row.addRow(label, self.largest_over)
        sv.addLayout(row)
        sv.addWidget(self.largest_note)
        stats_scroll = ColumnScroll(stats_tab)

        self.notes = QtWidgets.QTextBrowser()
        self.notes.setHtml(NOTES_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.miles_tab = miles_tab
        self.record_tab = record_tab
        self.tabs.addTab(miles_tab, "Miles")
        self.tabs.addTab(record_tab, "Record")
        self.stats_tab = stats_scroll
        self.tabs.addTab(stats_scroll, "Statistics")
        self.tabs.addTab(self.notes, "Theory")

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(left_scroll)
        splitter.addWidget(center)
        splitter.addWidget(self.tabs)
        splitter.setSizes([340, 810, 550])
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        # --- wiring
        self.kind.currentIndexChanged.connect(self._on_input)
        self.mass.currentIndexChanged.connect(self._on_input)
        for b in (self.level, self.f_lo, self.f_hi, self.duration):
            b.valueChanged.connect(self._on_input)
        self.quantity.currentIndexChanged.connect(self._on_output)
        self.index.currentIndexChanged.connect(self._on_output)
        self.block.currentIndexChanged.connect(lambda _: self._on_output())
        self.largest_over.valueChanged.connect(lambda _: self._show_largest())
        self.new_record.clicked.connect(self._on_new_record)
        self.trial.currentIndexChanged.connect(lambda _: self.apply_trial())

    # ------------------------------------------------------------- inputs
    def set_system(self, system: ChainSystem, result: ModalResult | None = None) -> None:
        """New chain parameters: the settings follow its modes, and it's recorded again (now if shown)."""
        n_changed = self.system is None or system.n != self.system.n
        self.system = linear(system)
        self.result = modal_analysis(self.system) if result is None or system.friction.any() else result
        self._loading = True
        self.dof.blockSignals(True)
        self.dof.setValue(system.n)
        self.dof.blockSignals(False)
        if n_changed:
            mass = self.mass.currentIndex()
            self.mass.clear()
            self.mass.addItems([f"m{i + 1}" for i in range(system.n)])
            self.mass.setCurrentIndex(mass if 0 <= mass < system.n else system.n - 1)
            self._fill_index()
        self._loading = False
        self.apply_trial()

    def _fill_index(self) -> None:
        n = self.system.n
        current = self.index.currentIndex()
        spring = self.quantity.currentData() is Quantity.SPRING
        self.index.blockSignals(True)
        self.index.clear()
        self.index.addItems([f"k{i + 1}" if spring else f"m{i + 1}" for i in range(n)])
        self.index.setCurrentIndex(current if 0 <= current < n else n - 1)
        self.index.blockSignals(False)

    def apply_trial(self) -> None:
        """Set the settings the chosen trial asks for, from the chain as it is."""
        if self.result is None:
            return
        _, changes, note = TRIALS[self.trial.currentIndex()]
        n = self.system.n
        freqs = sorted(m.fn_hz for m in self.result.modes if m.fn_hz > 0) or [1.0]
        kind = changes.get("input", Input.BASE)
        lo, hi = 0.0, _nice(1.5 * freqs[-1])
        if "band" in changes:
            f2 = freqs[1] if len(freqs) > 1 else freqs[0]
            lo, hi = changes["band"][0] * f2, changes["band"][1] * f2
        inp = RandomInput(kind, 0.01 if kind is Input.BASE else 1.0, lo, hi)
        duration = suggested_duration(inp, self.result)
        samples = int(round(duration * sample_rate(inp)))
        block = suggested_block(inp, self.result, samples)
        if "duration" in changes:
            duration *= changes["duration"]
        if "block" in changes:
            block = max(BLOCKS[0], int(block * changes["block"]))
        self._loading = True
        self.kind.setCurrentIndex(self.kind.findData(kind))
        self.mass.setCurrentIndex(changes.get("mass", -1) % n)
        self.level.setValue(inp.level)
        self.f_lo.setValue(lo)
        self.f_hi.setValue(hi)
        self.duration.setValue(round(duration))
        self.block.setCurrentIndex(BLOCKS.index(block))
        self.quantity.setCurrentIndex(self.quantity.findData(changes.get("output", Quantity.ACCELERATION)))
        self._fill_index()
        self.index.setCurrentIndex(changes.get("index", n - 1))
        self._loading = False
        self._show_controls()
        self.trial_note.setText(note)
        self._schedule(measure=True)

    def random_input(self) -> RandomInput:
        return RandomInput(self.kind.currentData(), self.level.value(), self.f_lo.value(), self.f_hi.value(),
                           max(self.mass.currentIndex(), 0))

    def output(self) -> Output:
        return Output(self.quantity.currentData(), max(self.index.currentIndex(), 0))

    def _show_controls(self) -> None:
        kind = self.kind.currentData()
        self.mass.setEnabled(kind is Input.FORCE_ONE)
        self.level.setSuffix(" " + PSD_UNITS[kind])

    def _on_input(self, *_args) -> None:
        if self._loading:
            return
        self._show_controls()
        self._schedule(measure=True)

    def _on_output(self, *_args) -> None:
        if self._loading:
            return
        if self.sender() is self.quantity:
            self._fill_index()
        self._schedule(measure=False)

    def _on_new_record(self) -> None:
        self.seed += 1
        self._schedule(measure=True, now=True)

    def _schedule(self, measure: bool, now: bool = False) -> None:
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
        """Record again if asked for (or stale), then predict and draw."""
        self._recompute.stop()
        self._dirty = False
        if self.system is None:
            return
        try:
            inp = self.random_input()
        except ValueError as e:
            self.readout.setText(f"<b>{e}</b>.")
            return
        if self._remeasure or self.record is None:
            try:
                self.record = simulate(self.system, inp, self.duration.value(), self.seed, self.result)
            except ValueError as e:
                self.record = None
                self.readout.setText(f"<b>{e}</b>.")
                return
            self._remeasure = False
        self.prediction = predict(self.system, inp, self.output(), self.result)
        self.redraw()

    # ------------------------------------------------------------- drawing
    def apply_theme(self) -> None:
        if self.prediction is not None:
            self.redraw()

    def redraw(self) -> None:
        rec, pred, out = self.record, self.prediction, self.output()
        inp = rec.input
        unit = out.unit()
        block = int(self.block.currentData())
        y = rec.output(out)
        f, g = welch(y, rec.fs, block)
        keep = f > 0
        f, g = f[keep], g[keep]
        fu, gu = welch(rec.u[:, 0], rec.fs, block)
        gu = gu[fu > 0] * (1.0 / 9.80665**2 if inp.kind is Input.BASE else 1.0)
        fu = fu[fu > 0]
        slow = min((m.fn_hz for m in self.result.modes if m.fn_hz > 0), default=inp.f_hi)
        lo = max(inp.f_lo, 0.2 * slow, pred.f[0]) if inp.f_lo < slow else max(pred.f[0], f[0])
        hi = inp.f_hi

        # the input
        p = self.input_plot
        p.clear()
        p.setLabel("left", f"Input PSD ({PSD_UNITS[inp.kind]})")
        p.plot(fu, np.maximum(gu, 1e-300), pen=pg.mkPen(colors.grey, width=1))
        band = np.array([max(inp.f_lo, lo), inp.f_hi])
        p.plot(band, np.full(2, inp.level), pen=pg.mkPen(colors.force, width=2))
        p.setXRange(math.log10(lo), math.log10(hi), padding=0.02)
        p.setYRange(math.log10(inp.level) - 1.5, math.log10(inp.level) + 0.7, padding=0.0)

        # the response
        p = self.psd_plot
        p.clear()
        self.psd_legend.clear()
        p.setLabel("left", f"Response PSD ({unit}²/Hz)")
        welch_curve = p.plot(f, np.maximum(g, 1e-300), pen=pg.mkPen(colors.grey, width=1))
        exact = p.plot(pred.f, np.maximum(pred.psd, 1e-300), pen=pg.mkPen(colors.strong, width=2))
        self.psd_legend.addItem(exact, "Exact, H G Hᴴ")
        self.psd_legend.addItem(welch_curve, "Record (Welch)")
        for i, m in enumerate(self.result.modes):
            if m.fn_hz > 0:
                p.addItem(pg.InfiniteLine(math.log10(m.fn_hz), angle=90, pen=pg.mkPen(
                    colors.mode[i % len(colors.mode)], width=1, style=QtCore.Qt.PenStyle.DotLine)))
        top = float(np.max(pred.psd)) if pred.psd.size else 1.0
        if top > 0:
            p.setYRange(math.log10(top) - 6, math.log10(top) + 0.5, padding=0.0)

        # the cumulative RMS
        p = self.cumulative
        p.clear()
        self.cumulative_legend.clear()
        p.setLabel("left", f"Cumulative RMS ({unit})")
        c = p.plot(pred.f, pred.cumulative, pen=pg.mkPen(colors.strong, width=2))
        self.cumulative_legend.addItem(c, "Exact")
        c = p.plot(f, cumulative_rms(f, g), pen=pg.mkPen(colors.grey, width=1))
        self.cumulative_legend.addItem(c, "Record (Welch)")
        dash = QtCore.Qt.PenStyle.DashLine
        miles_line = pg.InfiniteLine(pred.miles_rms, angle=0, pen=pg.mkPen(colors.fit, width=1.5, style=dash))
        p.addItem(miles_line)
        sim = float(np.std(y))
        record_pen = pg.mkPen(colors.grey, width=1, style=dash)
        p.addItem(pg.InfiniteLine(sim, angle=0, pen=record_pen))
        self.cumulative_legend.addItem(pg.PlotDataItem(pen=pg.mkPen(colors.fit, width=1.5, style=dash)), "Miles")
        self.cumulative_legend.addItem(pg.PlotDataItem(pen=record_pen), "The record's RMS")
        p.setYRange(0.0, 1.15 * max(pred.rms, pred.miles_rms, sim, 1e-300), padding=0.0)

        scatter = rms_scatter(pred, rec.duration)
        self.readout.setText(
            f"<b>RMS</b>: exact {fmt(pred.rms, unit)}; the record {fmt(sim, unit)} "
            f"({100 * (sim / pred.rms - 1):+.2g} %, expected scatter ±{100 * scatter:.2g} %); Miles "
            f"{fmt(pred.miles_rms, unit)} ({100 * (pred.miles_rms / pred.rms - 1):+.2g} %)."
            if pred.rms > 0 else "Nothing in the band reaches this response.")
        unit_in = "g" if inp.kind is Input.BASE else "N"
        self.input_info.setText(f"The input's RMS: {inp.rms:.4g} {unit_in}"
                                + (" on each mass." if inp.kind is Input.FORCE_ALL else "."))
        blocks = max(1, (y.size - block) // (block // 2) + 1) if y.size >= block else 1
        self.record_info.setText(
            f"{y.size:,} samples at {rec.fs:.4g} Hz; {blocks} blocks averaged (half overlapping), lines "
            f"{rec.fs / block:.3g} Hz apart.")
        self._fill_miles(pred, unit)
        self._draw_record(y, rec.fs, pred.rms, unit)
        self._draw_statistics(y, rec.fs, pred, unit)

    def _draw_statistics(self, y: np.ndarray, fs: float, pred: Prediction, unit: str) -> None:
        r = rice(pred)
        self._rice = r
        if r.sigma <= 0:
            self.stats_table.setText("")
            self.largest_note.setText("")
            return
        st = record_statistics(y, fs)
        self._stats = st
        dash = QtCore.Qt.PenStyle.DashLine

        p = self.amplitude_plot
        p.clear()
        self.amplitude_legend.clear()
        h, edges = np.histogram((y - np.mean(y)) / r.sigma, bins=AMPLITUDE_BINS, range=(-5.0, 5.0), density=True)
        bars = p.plot(edges, h, stepMode="center", fillLevel=0, pen=pg.mkPen(colors.mass[0], width=1),
                      brush=pg.mkBrush(_alpha(colors.mass[0], 90)))
        eta = np.linspace(-5.0, 5.0, 401)
        gauss = p.plot(eta, np.exp(-0.5 * eta * eta) / math.sqrt(2 * math.pi), pen=pg.mkPen(colors.strong, width=2))
        self.amplitude_legend.addItem(bars, "The record")
        self.amplitude_legend.addItem(gauss, "Gaussian")
        for k in LEVELS:
            for sign in (1, -1):
                p.addItem(pg.InfiniteLine(sign * k, angle=90, pen=pg.mkPen(colors.faint, width=1, style=dash)))
        p.setXRange(-5.0, 5.0, padding=0.0)

        p = self.peaks_plot
        p.clear()
        self.peaks_legend.clear()
        h, edges = np.histogram(st.peaks * st.sigma / r.sigma, bins=PEAK_BINS, range=(-3.0, 5.0), density=True)
        bars = p.plot(edges, h, stepMode="center", fillLevel=0, pen=pg.mkPen(colors.mass[1], width=1),
                      brush=pg.mkBrush(_alpha(colors.mass[1], 90)))
        eta = np.linspace(-3.0, 5.0, 401)
        theory = p.plot(eta, peak_pdf(eta, r.alpha), pen=pg.mkPen(colors.strong, width=2))
        self.peaks_legend.addItem(bars, "The record's peaks")
        self.peaks_legend.addItem(theory, f"Rice, α = {r.alpha:.2f}")
        p.setXRange(-3.0, 5.0, padding=0.0)

        def row(name: str, exact: str, record: str) -> str:
            return f"<tr><td>{name}</td><td align='right'>{exact}</td><td align='right'>{record}</td></tr>"

        rows = [row("σ", fmt(r.sigma, unit), fmt(st.sigma, unit)),
                row("Upward zero crossings", f"{r.nu0:.4g} /s", f"{st.nu0:.4g} /s"),
                row("Peaks", f"{r.nu_peak:.4g} /s", f"{st.nu_peak:.4g} /s"),
                row("Irregularity α", f"{r.alpha:.3f}", f"{st.nu0 / st.nu_peak:.3f}" if st.nu_peak else "–")]
        rows += [row(f"Beyond ±{k:g}σ", f"{100 * gaussian_exceedance(k):.3g} %", f"{100 * e:.3g} %")
                 for k, e in zip(LEVELS, st.exceedance)]
        rows += [row("Skewness", "0", f"{st.skewness:+.3f}"), row("Kurtosis", "3", f"{st.kurtosis:.3f}")]
        self.stats_table.setText("<table cellspacing='6'><tr><th></th><th align='right'>Rice, Gaussian</th>"
                                 "<th align='right'>The record</th></tr>" + "".join(rows) + "</table>")
        self._show_largest()

    def _show_largest(self) -> None:
        r, rec = getattr(self, "_rice", None), self.record
        if r is None or rec is None or r.sigma <= 0:
            return
        unit = self.output().unit()
        t = self.largest_over.value()
        dav, van = r.peak_factor(t), r.clumped_peak_factor(t)
        y = rec.output(self.output())
        maxima = segment_maxima(y, rec.fs, t)
        text = (f"Over {t:g} s ({r.nu0 * t:.3g} cycles), the expected largest peak: Davenport {dav:.3g}σ = "
                f"{fmt(dav * r.sigma, unit)}; Vanmarcke (δ = {r.delta:.2f}) {van:.3g}σ = {fmt(van * r.sigma, unit)}.")
        if maxima.size >= 4:
            mean = float(np.mean(maxima)) / r.sigma
            text += (f" The record's {maxima.size} stretches of {t:g} s: their largest values average "
                     f"<b>{mean:.3g}σ</b>.")
        else:
            text += f" (The record is too short for four stretches of {t:g} s to compare.)"
        self.largest_note.setText(text)

    def _fill_miles(self, pred: Prediction, unit: str) -> None:
        t = self.miles_table
        t.clear()
        t.setColumnCount(5)
        t.setHorizontalHeaderLabels(["Mode", "f (Hz)", "ζ", "Q", f"RMS ({unit})"])
        rows = pred.miles
        t.setRowCount(len(rows) + 1)
        total = pred.miles_rms
        for r, (mode, fn, zeta, sigma) in enumerate(rows):
            for c, text in enumerate((f"{mode}", f"{fn:.4g}", f"{zeta:.3g}", f"{1 / (2 * zeta):.3g}",
                                      f"{sigma:.4g} ({100 * sigma**2 / total**2:.0f} %)" if total > 0 else "0")):
                item = QtWidgets.QTableWidgetItem(text)
                item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
                t.setItem(r, c, item)
        item = QtWidgets.QTableWidgetItem(f"SRSS {total:.4g}")
        item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        t.setItem(len(rows), 4, item)
        t.resizeColumnsToContents()
        self.miles_note.setText(
            f"Exact RMS {fmt(pred.rms, unit)}: Miles' estimate is {100 * (total / pred.rms - 1):+.2g} % off."
            if pred.rms > 0 else "")

    def _draw_record(self, y: np.ndarray, fs: float, sigma: float, unit: str) -> None:
        p = self.record_plot
        p.clear()
        p.setLabel("left", f"Response ({unit})")
        slow = min((m.fn_hz for m in self.result.modes if m.fn_hz > 0), default=1.0)
        n = min(y.size, int(SHOWN_PERIODS / slow * fs))
        t = np.arange(n) / fs
        p.plot(t, y[:n], pen=pg.mkPen(colors.mass[0], width=1))
        for k, style in ((1, QtCore.Qt.PenStyle.DashLine), (3, QtCore.Qt.PenStyle.DotLine)):
            for sign in (1, -1):
                p.addItem(pg.InfiniteLine(sign * k * sigma, angle=0, pen=pg.mkPen(colors.muted, width=1, style=style)))
        p.setXRange(0.0, n / fs, padding=0.0)
        p.setYRange(-4.0 * sigma, 4.0 * sigma, padding=0.02)
        self.record_note.setText(f"The first {n / fs:.3g} s ({SHOWN_PERIODS} periods of mode 1); dashed ±σ, "
                                 f"dotted ±3σ, σ the exact RMS.")


def _alpha(color: str, alpha: int):
    c = pg.mkColor(color)
    c.setAlpha(alpha)
    return c


def _nice(x: float) -> float:
    """x rounded up to 1, 2 or 5 times a power of ten."""
    e = 10.0 ** math.floor(math.log10(x))
    return next(m * e for m in (1, 2, 5, 10) if m * e >= x)
