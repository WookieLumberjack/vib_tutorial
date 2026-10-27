"""Periodic forcing page: a repeating force as a Fourier series, and each harmonic through the oscillator.

Left: the oscillator, the waveform, its frequency and how many harmonics the series keeps. Centre: two periods
of the force (the true wave and the series) and of the steady response (exact, and the series). Right: the
force's and the response's line spectra, the response from rest, the series' convergence, and the Theory.
The page has its own model (core/periodic.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.periodic import (
    WAVES,
    Forcing,
    Wave,
    from_rest,
    periodic_steady,
    response_coefficients,
    series,
    series_error,
)
from ..core.sdof import Oscillator, frequency_grid, receptance
from . import settle_plots
from .lesson import (
    Form,
    TrialBox,
    column,
    combo,
    connect_all,
    fmt,
    mark,
    marker,
    note_label,
    plot,
    readout_label,
    set_quietly,
    stack,
    theory,
    three_panes,
    trials_html,
)
from .links import before, takeaways
from .style import colors
from .theming import add_legend

TAB_LINES, TAB_REST, TAB_CONVERGENCE = range(3)
POINTS = 1200  # per period drawn
MOST = 200  # harmonics the series can keep
CONVERGENCE = (1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200)


@dataclass(frozen=True)
class PeriodicTrial:
    name: str
    note: str
    wave: Wave = Wave.SQUARE
    amplitude: float = 1.0
    freq_hz: float = 1.0
    duty: float = 0.25
    harmonics: int = 5
    damping: float = 2.0
    tab: int = TAB_LINES
    tune: int = 0  # set f₀ to f_n / tune when not 0


TRIALS = (
    PeriodicTrial(
        "A square wave, harmonic by harmonic",
        "A ±1 N square wave at 1 Hz is (4/π)(sin ω₀t + sin 3ω₀t/3 + sin 5ω₀t/5 + …): odd harmonics only, falling as "
        "1/k. With 5 kept, the series already looks square, but it overshoots each jump by 19 % of F. Keeping more terms "
        "only narrows the overshoot; its height settles at 18 % of F, 9 % of the jump (Gibbs). The response is "
        "smoother: the oscillator divides the k-th harmonic by about k² above f<sub>n</sub>.",
        harmonics=5),
    PeriodicTrial(
        "Resonance through the third harmonic",
        "The square wave at f<sub>n</sub>/3 = 1.06 Hz. Its fundamental is far below resonance, but its third "
        "harmonic, 4/3π = 0.42 N, sits on f<sub>n</sub>, where |H| = 1/cω<sub>n</sub> = 25 mm/N: it alone swings "
        "the mass 10.6 mm, three times the fundamental's 3.6 mm. The response rings at three times the drive: "
        "slow machines excite fast modes this way.",
        tune=3, harmonics=7),
    PeriodicTrial(
        "No even harmonics, no resonance",
        "The same square wave at f<sub>n</sub>/2. A square wave has no second harmonic, so nothing reaches "
        "f<sub>n</sub>: the response is 5.0 mm, a third of the 13.9 mm at f<sub>n</sub>/3, though the drive is "
        "closer to resonance.",
        tune=2, harmonics=7),
    PeriodicTrial(
        "Hammer blows",
        "A 0.1 N·s blow every second. An impulse train has every harmonic at the same height, 2I/T = 0.2 N, so "
        "the line spectra show H itself, sampled at 1 Hz steps. In time, each blow sets the mass ringing and "
        "the next arrives before it has died away (τ = 1 s): the steady state swings from −4.5 to 5.2 mm about the "
        "mean I/kT = 0.25 mm.",
        wave=Wave.IMPULSES, amplitude=0.1, harmonics=20),
    PeriodicTrial(
        "A rectified sine",
        "F|sin πt/T| with F = 1 N: a mean of 2F/π = 0.64 N, which only sags the spring, 1.59 mm, and harmonics "
        "−4F/π(4k² − 1) that fall as 1/k². What a full-wave rectifier, or a cam, puts on its mounts.",
        wave=Wave.RECTIFIED, harmonics=5),
    PeriodicTrial(
        "Smooth waves converge faster",
        "A triangle's harmonics fall as 1/k², a square's as 1/k: with 3 harmonics the triangle's response series "
        "is within 1.9 % of the exact steady state, the square's 5.7 %; with 9, 0.10 % against 0.65 %. A jump in "
        "the force costs 1/k, a kink 1/k². The Convergence tab draws it.",
        wave=Wave.TRIANGLE, harmonics=3, tab=TAB_CONVERGENCE),
    PeriodicTrial(
        "Starting from rest",
        "The square wave at f<sub>n</sub>/3, switched on from rest. The resonant third harmonic builds up over a "
        "few time constants (1 s), as a sine at f<sub>n</sub> would: the steady state only arrives once the start "
        "has died away.",
        tune=3, harmonics=7, tab=TAB_REST),
)

TIPS = {
    "mass": "<p>The mass m.</p>",
    "stiffness": "<p>The spring k.</p>",
    "damping": "<p>The damper c.</p>",
    "wave": "<p>The waveform repeated every period: each has its Fourier series in closed form.</p>",
    "amplitude": "<p>The waveform's height F (for the impulse train, each blow's impulse, N·s).</p>",
    "freq": "<p>How often it repeats, f₀ = 1/T.</p>",
    "duty": "<p>The fraction of each period the rectangular pulses are on.</p>",
    "harmonics": "<p>How many harmonics the series keeps.</p>",
    "tune": "<p>Set f₀ to f<sub>n</sub> divided by this, so that harmonic sits on resonance.</p>",
}
TIME_TIP = ("<p>Two periods. Grey: the true waveform; colour: its Fourier series with the harmonics kept.</p>")
RESPONSE_TIP = ("<p>The steady response over two periods: solid, exact (one period stepped from the state that "
                "repeats); dashed, the series Σ H(kω₀) C<sub>k</sub> e<sup>ikω₀t</sup>.</p>")
LINES_TIP = ("<p>Each harmonic's amplitude at its frequency: the force's (top) and the response's (bottom), with "
             "the receptance |H| (dashed, scaled to the force's largest harmonic) and f<sub>n</sub> (dotted).</p>")
REST_TIP = "<p>Switched on from rest, stepped exactly; the steady state's band dotted.</p>"
CONVERGENCE_TIP = ("<p>The series' largest error over a period against the exact steady state, over the steady "
                   "swing, against the number of harmonics kept. Dotted: 1/N and 1/N².</p>")

NOTES_HTML = """
<!--BEFORE-->
<h3><a name="fourier-series"></a>A periodic force is a sum of harmonics</h3>
<p>A force that repeats every T = 1/f₀, however it is shaped, is a <b>Fourier series</b>:</p>
<p align="center">f(t) = a₀ + Σ<sub>k≥1</sub> (a<sub>k</sub> cos kω₀t + b<sub>k</sub> sin kω₀t), &nbsp; ω₀ = 2πf₀,</p>
<p>a₀ its mean, a<sub>k</sub> = (2/T)∫f cos kω₀t dt and b<sub>k</sub> = (2/T)∫f sin kω₀t dt. Only multiples of
f₀ appear: a periodic force has a <b>line spectrum</b>. Written with complex amplitudes,
f = Re Σ C<sub>k</sub> e<sup>ikω₀t</sup>, C<sub>k</sub> = a<sub>k</sub> − i b<sub>k</sub>.</p>

<h3><a name="superposition"></a>Each harmonic through H</h3>
<p>The oscillator is linear, so each harmonic is a harmonic force on its own (the Single oscillator page's
forced response), and the steady response is the sum of their responses:</p>
<p align="center">x(t) = Re Σ H(kω₀) C<sub>k</sub> e<sup>ikω₀t</sup>, &nbsp; H(ω) = 1/(k − ω²m + iωc).</p>
<p>The mean only sags the spring (H(0) = 1/k). A harmonic that lands near f<sub>n</sub> is amplified by Q; the
rest are not. So a force at f₀ can resonate the oscillator at <i>any</i> multiple k f₀ = f<sub>n</sub> it
contains: a machine running at a third of a mode's frequency can still drive it, through its third harmonic.</p>

<h3><a name="shapes"></a>Shapes and their harmonics</h3>
<ul>
<li><b>Square wave</b> (±F): (4F/π) Σ sin kω₀t / k over odd k. A wave with the half-wave symmetry
f(t + T/2) = −f(t) has odd harmonics only.</li>
<li><b>Sawtooth</b>: (2F/π) Σ (−1)<sup>k+1</sup> sin kω₀t / k, every harmonic.</li>
<li><b>Triangle</b>: (8F/π²) Σ ± sin kω₀t / k², odd only.</li>
<li><b>Rectified sine</b> F|sin πt/T|: 2F/π − (4F/π) Σ cos kω₀t / (4k² − 1).</li>
<li><b>Rectangular pulses</b> (F for a fraction d of each period): mean Fd; |C<sub>k</sub>| = (2F/πk)|sin πkd|,
zero at every multiple of 1/d.</li>
<li><b>Impulse train</b> (a blow I every T): I/T + (2I/T) Σ cos kω₀t: every harmonic the same height.</li>
</ul>
<p>How fast the harmonics fall depends on how smooth the wave is: as 1/k with a jump, 1/k² with a kink (a jump
in slope), and not at all for an impulse. A jump also brings the <b>Gibbs phenomenon</b>: the series overshoots
it by 9 % of the jump however many terms it keeps, the overshoot only getting narrower.</p>

<h3><a name="exact"></a>The steady state without the series</h3>
<p>The page also works the steady state out directly. One period from rest gives z<sub>T</sub> = Φ<sub>T</sub>z₀ +
z<sub>p</sub> (Φ<sub>T</sub> = e<sup>AT</sup>, the force stepped exactly by the first-order hold); a periodic state
has z<sub>T</sub> = z₀, so z₀ = (I − Φ<sub>T</sub>)<sup>−1</sup>z<sub>p</sub>. That is exact, and it is what the series
converges to. Without damping, a harmonic exactly on f<sub>n</sub> leaves I − Φ<sub>T</sub> singular: no steady
state, the swing grows forever.</p>

<h3><a name="from-rest"></a>Starting from rest</h3>
<p>Switched on from rest, the response is the steady state plus a free vibration that makes up the start, as for a
single sine; it dies away in a few time constants 1/ζω<sub>n</sub>. The <i>From rest</i> tab steps it.</p>

<h3><a name="try-it"></a>Things to try</h3>
<p>Each of these sets up the oscillator, the force and the plots (Things to try, top left).</p>
<!--TRIALS-->
<!--TAKEAWAYS-->
"""
NOTES_HTML = NOTES_HTML.replace("<!--BEFORE-->", before(
    ("sdof_page", "forced-response", "Forced response"),
    ("transient_page", "fourier-pair", "The impulse response and the FRF"),
)).replace("<!--TRIALS-->", trials_html(TRIALS)).replace("<!--TAKEAWAYS-->", takeaways(
    "A periodic force is a line spectrum: its mean and harmonics at whole multiples of f₀.",
    "Each harmonic goes through H(kω₀) on its own; a mode at any k f₀ resonates.",
    "The harmonics fall as 1/k after a jump, 1/k² after a kink; a jump's series overshoots it by 9 % (Gibbs).",
    "The steady state can be found exactly from one period: z₀ = (I − Φ<sub>T</sub>)<sup>−1</sup>z<sub>p</sub>.",
))


def stems(x: np.ndarray, y: np.ndarray, floor: float) -> tuple[np.ndarray, np.ndarray]:
    """Vertical lines from `floor` up to each y, as pairs for connect="pairs"."""
    xs = np.repeat(x, 2)
    ys = np.column_stack([np.full(y.size, floor), y]).ravel()
    return xs, ys


class PeriodicPage(QtWidgets.QWidget):
    """Controls on the left, the force and the steady response in the centre, spectra and convergence on the
    right."""

    follows_chain = False  # MainWindow: this page has its own model, not the chain

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.osc = Oscillator()
        self.forcing = Forcing()

        self.trials = TrialBox([t.name for t in TRIALS],
                               "<p>Each sets up the force and the oscillator to show one idea from the Theory "
                               "tab.</p>")
        self.trials.picked.connect(self.load_trial)

        osc = Form("Oscillator")
        self.mass = osc.number("Mass m:", 0.001, 1e4, 1.0, 3, " kg", TIPS["mass"])
        self.stiffness = osc.number("Stiffness k:", 0.01, 1e8, 400.0, 2, " N/m", TIPS["stiffness"])
        self.damping = osc.number("Damping c:", 0.0, 1e6, 2.0, 3, " N·s/m", TIPS["damping"])
        self.osc_info = note_label()
        osc.row(self.osc_info)

        force = Form("Force")
        self.wave = combo([w.value for w in WAVES], TIPS["wave"])
        force.add("Waveform:", self.wave, TIPS["wave"])
        self.amplitude = force.number("Height F:", 0.0, 1e4, 1.0, 3, "", TIPS["amplitude"])
        self.freq = force.number("Frequency f₀:", 0.01, 1e4, 1.0, 3, " Hz", TIPS["freq"])
        self.duty = force.number("Duty:", 0.01, 0.99, 0.25, 2, "", TIPS["duty"])
        self.harmonics = force.number("Harmonics kept:", 1, MOST, 5, 0, "", TIPS["harmonics"])
        tune_row = QtWidgets.QHBoxLayout()
        self.tune_k = QtWidgets.QSpinBox()
        self.tune_k.setRange(1, 20)
        self.tune_k.setValue(3)
        self.tune_k.setToolTip(TIPS["tune"])
        self.tune = QtWidgets.QPushButton("Set f₀ = fₙ /")
        self.tune.setToolTip(TIPS["tune"])
        self.tune.clicked.connect(self._on_tune)
        tune_row.addWidget(self.tune)
        tune_row.addWidget(self.tune_k)
        force.row(tune_row)

        connect_all((self.mass, self.stiffness, self.damping, self.wave, self.amplitude, self.freq, self.duty,
                     self.harmonics), self._on_inputs)

        # --- centre
        self.force_plot = plot("Force", "N", "Time", "s", height=150)
        self.force_plot.setToolTip(TIME_TIP)
        add_legend(self.force_plot.getPlotItem(), offset=(-10, 10))
        self.wave_item = pg.PlotDataItem(name="the wave")
        self.series_item = pg.PlotDataItem(name="its series")
        self.blow_items = pg.PlotDataItem()
        for item in (self.wave_item, self.blow_items, self.series_item):
            self.force_plot.addItem(item)
        self.x_plot = plot("Displacement x", "m", "Time", "s", height=150)
        self.x_plot.setToolTip(RESPONSE_TIP)
        self.x_plot.setXLink(self.force_plot)
        add_legend(self.x_plot.getPlotItem(), offset=(-10, 10))
        self.exact_item = pg.PlotDataItem(name="exact steady state")
        self.xseries_item = pg.PlotDataItem(name="series")
        for item in (self.exact_item, self.xseries_item):
            self.x_plot.addItem(item)
        self.readout = readout_label()

        # --- right
        lines = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(lines)
        self.fline_plot = plot("Force harmonic |C<sub>k</sub>|", "N", "Frequency", "Hz", log_y=True, height=130)
        self.xline_plot = plot("Response harmonic |X<sub>k</sub>|", "m", "Frequency", "Hz", log_y=True, height=130)
        self.xline_plot.setXLink(self.fline_plot)
        self.fstems = pg.PlotDataItem(connect="pairs")
        self.fdots = marker(6)
        self.H_item = pg.PlotDataItem()
        self.xstems = pg.PlotDataItem(connect="pairs")
        self.xdots = marker(6)
        self.fn_lines = [pg.InfiniteLine(angle=90), pg.InfiniteLine(angle=90)]
        for p, items in ((self.fline_plot, (self.fn_lines[0], self.H_item, self.fstems, self.fdots)),
                         (self.xline_plot, (self.fn_lines[1], self.xstems, self.xdots))):
            p.setToolTip(LINES_TIP)
            p.setMouseEnabled(x=True, y=False)
            for item in items:
                p.addItem(item)
            lv.addWidget(p, 1)
        self.lines_note = note_label()
        lv.addWidget(self.lines_note)

        self.rest_plot = plot("Displacement x", "m", "Time", "s", height=200)
        self.rest_plot.setToolTip(REST_TIP)
        self.rest_item = pg.PlotDataItem()
        self.band_items = [pg.InfiniteLine(angle=0), pg.InfiniteLine(angle=0)]
        for item in (*self.band_items, self.rest_item):
            self.rest_plot.addItem(item)
        self.rest_note = note_label()

        self.conv_plot = plot("Largest error / steady swing", None, "Harmonics kept N", None, log_x=True,
                              log_y=True, height=200)
        self.conv_plot.setToolTip(CONVERGENCE_TIP)
        self.conv_plot.setMouseEnabled(x=False, y=False)
        add_legend(self.conv_plot.getPlotItem(), offset=(-10, 10))
        self.conv_item = pg.PlotDataItem(name="this wave's response series", symbol="o", symbolSize=6)
        self.slope_items = [pg.PlotDataItem(name="∝ 1/N"), pg.PlotDataItem(name="∝ 1/N²")]
        self.conv_now = marker(12)
        for item in (*self.slope_items, self.conv_item, self.conv_now):
            self.conv_plot.addItem(item)
        self.conv_note = note_label()

        self.notes = theory(NOTES_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(lines, "Line spectra")
        self.tabs.addTab(column(self.rest_plot, self.rest_note, stretch=False), "From rest")
        self.tabs.addTab(column(self.conv_plot, self.conv_note, stretch=False), "Convergence")
        self.tabs.addTab(self.notes, "Theory")
        self.tabs.currentChanged.connect(lambda _: self._draw_tab())

        three_panes(self, column(self.trials, osc, force),
                    stack((self.force_plot, 2), (self.x_plot, 3), (self.readout, 0)), self.tabs)
        self._on_inputs()
        self.apply_theme()

    # ------------------------------------------------------------- inputs
    def load_trial(self, index: int) -> None:
        t = TRIALS[index]
        osc = Oscillator(1.0, 400.0, t.damping)
        freq = osc.natural_hz / t.tune if t.tune else t.freq_hz
        for box, value in ((self.mass, 1.0), (self.stiffness, 400.0), (self.damping, t.damping),
                           (self.wave, WAVES.index(t.wave)), (self.amplitude, t.amplitude), (self.freq, freq),
                           (self.duty, t.duty), (self.harmonics, t.harmonics)):
            set_quietly(box, value)
        self.trials.show_note(t.note)
        self.tabs.setCurrentIndex(t.tab)
        self._on_inputs()

    def _on_tune(self) -> None:
        self.freq.setValue(self.osc.natural_hz / self.tune_k.value())

    def _on_inputs(self) -> None:
        self.osc = Oscillator(self.mass.value(), self.stiffness.value(), self.damping.value())
        wave = WAVES[self.wave.currentIndex()]
        self.forcing = Forcing(wave, self.amplitude.value(), self.freq.value(), self.duty.value())
        self.duty.setEnabled(wave is Wave.PULSES)
        self.amplitude.setSuffix(" N·s" if wave is Wave.IMPULSES else " N")
        o = self.osc
        self.osc_info.setText(f"f<sub>n</sub> = {o.natural_hz:.4g} Hz, ζ = {o.zeta:.3g}; f<sub>n</sub>/f₀ = "
                              f"{o.natural_hz / self.forcing.freq_hz:.3g}.")
        self.redraw()

    # ------------------------------------------------------------- drawing
    def apply_theme(self) -> None:
        c = colors
        self.wave_item.setPen(pg.mkPen(c.grey, width=3))
        self.blow_items.setPen(pg.mkPen(c.grey, width=3))
        self.series_item.setPen(pg.mkPen(c.force, width=1.5))
        self.exact_item.setPen(pg.mkPen(c.strong, width=2))
        self.xseries_item.setPen(pg.mkPen(c.mass[0], width=2, style=QtCore.Qt.PenStyle.DashLine))
        self.fstems.setPen(pg.mkPen(c.force, width=2))
        mark(self.fdots, c.force, filled=True)
        self.xstems.setPen(pg.mkPen(c.mass[0], width=2))
        mark(self.xdots, c.mass[0], filled=True)
        self.H_item.setPen(pg.mkPen(c.grey, width=1, style=QtCore.Qt.PenStyle.DashLine))
        for line in (*self.fn_lines, *self.band_items):
            line.setPen(pg.mkPen(c.fit, width=1, style=QtCore.Qt.PenStyle.DotLine))
        self.rest_item.setPen(pg.mkPen(c.mass[0], width=1.5))
        self.conv_item.setPen(pg.mkPen(c.mass[0], width=2))
        self.conv_item.setSymbolBrush(pg.mkBrush(c.mass[0]))
        self.conv_item.setSymbolPen(None)
        for item, style in zip(self.slope_items, (QtCore.Qt.PenStyle.DashLine, QtCore.Qt.PenStyle.DotLine)):
            item.setPen(pg.mkPen(c.grey, width=1, style=style))
        mark(self.conv_now, c.force)
        self.redraw()

    def redraw(self) -> None:
        self._draw_time()
        self._draw_tab()
        settle_plots(self)

    @property
    def n_harmonics(self) -> int:
        return int(self.harmonics.value())

    def _draw_time(self) -> None:
        fo, osc, n = self.forcing, self.osc, self.n_harmonics
        T = fo.period
        t = np.linspace(0.0, 2 * T, 2 * POINTS + 1)
        C = fo.coefficients(n)
        if fo.wave is Wave.IMPULSES:
            self.wave_item.setData([], [])
            peak = 2 * fo.amplitude / T * max(n, 1)
            self.blow_items.setData([0, 0, T, T, 2 * T, 2 * T], [0, peak, 0, peak, 0, peak], connect="pairs")
        else:
            self.blow_items.setData([], [])
            self.wave_item.setData(t, fo.force(t))
        self.series_item.setData(t, series(C, fo.freq_hz, t))
        exact = periodic_steady(osc, fo, t)
        X = response_coefficients(osc, fo, n)
        approx = series(X, fo.freq_hz, t)
        if np.all(np.isfinite(exact)):
            self.exact_item.setData(t, exact)
            mean = float(fo.coefficients(0)[0].real) / osc.stiffness  # C₀ H(0), exactly
            swing = float(np.max(np.abs(exact - mean))) or 1.0
            err = float(np.max(np.abs(approx - exact))) / swing
            self.readout.setText(
                f"Steady state from {fmt(float(exact.min()), 'm')} to {fmt(float(exact.max()), 'm')} (mean "
                f"{fmt(mean, 'm')}); with {n} harmonic{'s' if n > 1 else ''} the series is within "
                f"{100 * err:.3g} % of its swing.")
        else:
            self.exact_item.setData([], [])
            self.readout.setText("A harmonic sits exactly on f<sub>n</sub> with no damping: there is no steady "
                                 "state (the swing grows for ever; see From rest).")
        self.xseries_item.setData(t, np.where(np.isfinite(approx), approx, np.nan), connect="finite")
        self.force_plot.setXRange(0.0, 2 * T, padding=0)

    def _draw_tab(self) -> None:
        i = self.tabs.currentIndex()
        if i == TAB_LINES:
            self._draw_lines()
        elif i == TAB_REST:
            self._draw_rest()
        elif i == TAB_CONVERGENCE:
            self._draw_convergence()

    def _draw_lines(self) -> None:
        fo, osc, n = self.forcing, self.osc, self.n_harmonics
        C = fo.coefficients(n)
        X = response_coefficients(osc, fo, n)
        f = fo.freq_hz * np.arange(n + 1)
        keep = np.abs(C) > 1e-12 * max(float(np.max(np.abs(C))), 1e-300)
        fa, Xa = np.abs(C[keep]), np.abs(X[keep])
        top = float(fa.max()) if fa.size else 1.0
        self.fstems.setData(*stems(f[keep], fa, top * 1e-4))
        self.fdots.setData(f[keep], fa)
        finite = np.isfinite(Xa)
        xtop = float(Xa[finite].max()) if finite.any() else 1.0
        self.xstems.setData(*stems(f[keep][finite], Xa[finite], xtop * 1e-4))
        self.xdots.setData(f[keep][finite], Xa[finite])
        grid = frequency_grid(osc, 1e-3, max(1.2 * f[-1] / osc.natural_hz, 2.0), 1500)
        H = np.abs(receptance(osc, grid))
        self.H_item.setData(grid, H / H[0] * top, connect="finite")
        for line in self.fn_lines:
            line.setPos(osc.natural_hz)
        hi = max(f[-1], osc.natural_hz) * 1.05
        self.fline_plot.setXRange(0.0, hi, padding=0)
        self.fline_plot.setYRange(math.log10(top) - 4.2, math.log10(top) + 0.5, padding=0)
        self.xline_plot.setYRange(math.log10(xtop) - 4.2, math.log10(xtop) + 0.5, padding=0)
        order = np.argsort(Xa[finite])[::-1][:3]
        ks = np.arange(n + 1)[keep][finite]
        total = float(np.sum(Xa[finite][ks > 0] ** 2)) or 1.0
        parts = [f"k = {ks[i]} ({fmt(f[keep][finite][i], 'Hz')}): {fmt(float(Xa[finite][i]), 'm')}"
                 + (f", {100 * Xa[finite][i] ** 2 / total:.3g} % of the swing's power" if ks[i] > 0 else " (mean)")
                 for i in order]
        self.lines_note.setText("Largest response harmonics: " + "; ".join(parts) + ".")

    def _draw_rest(self) -> None:
        fo, osc = self.forcing, self.osc
        periods = max(6.0, min(400.0, 5.0 * min(osc.time_constant, 60.0) / fo.period))
        t, x = from_rest(osc, fo, periods, per_period=200 if periods > 60 else 400)
        self.rest_item.setData(t, x)
        tt = np.linspace(0.0, fo.period, 2000, endpoint=False)
        steady = periodic_steady(osc, fo, tt)
        if np.all(np.isfinite(steady)):
            for line, v in zip(self.band_items, (steady.min(), steady.max())):
                line.setPos(float(v))
                line.setVisible(True)
            self.rest_note.setText(f"{periods:.0f} periods from rest. The steady state (dotted) swings from "
                                   f"{fmt(float(steady.min()), 'm')} to {fmt(float(steady.max()), 'm')}; the largest "
                                   f"swing on the way is {fmt(float(np.max(np.abs(x))), 'm')}.")
        else:
            for line in self.band_items:
                line.setVisible(False)
            self.rest_note.setText(f"{periods:.0f} periods from rest: with no damping and a harmonic on f<sub>n</sub> "
                                   "the swing grows without bound.")
        self.rest_plot.setXRange(0.0, t[-1], padding=0)

    def _draw_convergence(self) -> None:
        fo, osc = self.forcing, self.osc
        ns = np.array(CONVERGENCE)
        errs = np.array([series_error(osc, fo, int(k), 800) for k in ns])
        ok = np.isfinite(errs) & (errs > 0)
        self.conv_item.setData(ns[ok], errs[ok])
        if ok.any():
            e0 = float(errs[ok][0]) * ns[ok][0]
            self.slope_items[0].setData(ns, e0 / ns)
            self.slope_items[1].setData(ns, e0 / ns ** 2)
        n = self.n_harmonics
        now = series_error(osc, fo, n, 800)
        if np.isfinite(now) and now > 0:
            self.conv_now.setData([n], [now])
        else:
            self.conv_now.setData([], [])
        low = math.log10(float(errs[ok].min())) if ok.any() else -6.0
        self.conv_plot.setXRange(0.0, math.log10(MOST), padding=0.02)
        self.conv_plot.setYRange(min(-4.0, math.floor(low - 0.3)), 0.5, padding=0)
        self.conv_note.setText(f"{fo.wave.value}: with {n} harmonics kept the series is within {100 * now:.3g} % "
                               "of the exact steady state's swing.")
