"""Signals & spectra page: sampling, aliasing, the DFT's lines, leakage, windows, averaging and transducers.

Left: the signal (two tones, a chirp or a struck mode, plus noise) and how it is acquired (the sample rate,
the anti-alias filter, the block, the window, the averages). Centre: the first block, the analog signal behind
its samples and the window over them, and the averaged spectrum, as a peak amplitude or a PSD, with each
tone's true amplitude where it should show. Right: the windows compared, a spectrogram of the record, the
transducers (a seismic mass in a case), and the Theory. The page has its own model (core/signals.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.signals import (
    FILTER_RATIO,
    WINDOWS,
    Kind,
    Source,
    Window,
    apparent,
    base_excited,
    blocks_needed,
    kernel,
    mass_loaded,
    nearest_line_reading,
    sample,
    spectrogram,
    spectrum,
    usable_band,
    window,
    window_props,
)
from . import settle_plots
from .layouts import ColumnScroll
from .lesson import (
    Form,
    TrialBox,
    connect_all,
    db,
    fmt,
    note_label,
    plot,
    readout_label,
    set_quietly,
    trials_html,
)
from .links import before, takeaways
from .style import colors
from .theming import add_legend

KINDS = list(Kind)
BLOCKS = tuple(2 ** k for k in range(6, 15))  # 64 … 16384 samples
SCALES = ("Peak amplitude", "PSD")
TAB_WINDOWS, TAB_SPECTROGRAM, TAB_SENSORS = range(3)
ANALOG_POINTS = 20000  # most points drawn for the analog signal behind a block
ZOOM_SAMPLES = 64  # the time plot's close-up: the block's first samples
SPAN = 12.0  # lines either side drawn on the Windows tab
SENSOR_RATIOS = np.geomspace(0.01, 100.0, 1200)


@dataclass(frozen=True)
class SignalsTrial:
    name: str
    note: str
    kind: Kind = Kind.TONES
    f1: float = 50.0
    a1: float = 1.0
    f2: float = 120.0
    a2: float = 0.1
    noise: float = 0.0
    zeta: float = 0.01
    sweep: float = 4.0
    fs: float = 1024.0
    block: int = 1024
    window: Window = Window.HANN
    averages: int = 1
    overlap: float = 50.0
    filtered: bool = True
    scale: int = 0
    zoom: bool = False
    tab: int = TAB_WINDOWS
    sensor_hz: float = 30000.0
    sensor_zeta: float = 0.01


TRIALS = (
    SignalsTrial(
        "A tone on a line",
        "1024 samples at 1024 Hz: a 1 s block, lines Δf = 1 Hz apart, the highest at the Nyquist frequency, "
        "512 Hz. A 50 Hz tone fits 50 whole cycles in the block, so it lands on a line and, without a window, "
        "reads its amplitude exactly, 1.000, with nothing on the other lines: no leakage.",
        f2=120.0, a2=0.0, window=Window.RECTANGULAR),
    SignalsTrial(
        "Between lines: leakage",
        "The tone moved to 50.5 Hz, half-way between lines, still unwindowed. It reads 0.64 (−3.9 dB, the "
        "rectangular window's scalloping loss), and its energy leaks across the whole spectrum, falling only as "
        "1/f: at 70 Hz the leak reads 0.015, fifteen times a 0.001 tone that is there and can't be seen.",
        f1=50.5, f2=70.0, a2=0.001, window=Window.RECTANGULAR),
    SignalsTrial(
        "A Hann window",
        "The same two tones through a Hann window. The big one reads 0.85 (−1.4 dB) and spreads over a few "
        "lines, but its skirts fall as 1/f³: the 0.001 tone at 70 Hz now stands clear, reading 0.0010.",
        f1=50.5, f2=70.0, a2=0.001, window=Window.HANN),
    SignalsTrial(
        "A flat-top window",
        "A flat-top window reads 0.999 for the 50.5 Hz tone: its scalloping loss is 0.01 dB, so a tone's "
        "amplitude is right wherever it falls. The price is a main lobe 5 lines wide each side, and a noise "
        "bandwidth of 3.8 lines. Calibrations use it; separating close tones doesn't.",
        f1=50.5, f2=70.0, a2=0.001, window=Window.FLAT_TOP),
    SignalsTrial(
        "Aliasing",
        "A 900 Hz tone sampled at 1024 Hz with the anti-alias filter off: above Nyquist (512 Hz), it folds back "
        "and shows at 1024 − 900 = 124 Hz with its whole amplitude, indistinguishable from a real 124 Hz tone. "
        "The close-up of the first 64 samples shows why: the dots trace a slow sine through the fast one.",
        f1=900.0, a1=1.0, a2=0.0, filtered=False, zoom=True),
    SignalsTrial(
        "The anti-alias filter",
        "The same 900 Hz tone with the filter on: an 8-pole low-pass at f<sub>s</sub>/2.56 = 400 Hz cuts it to "
        "0.0015 (−56 dB) before it is sampled. The filter also takes 3 dB off a tone at 400 Hz, which is why only "
        "the lines up to f<sub>s</sub>/2.56 are kept (the shaded band is not to be trusted).",
        f1=900.0, a1=1.0, a2=0.0, filtered=True, zoom=True),
    SignalsTrial(
        "A tone in noise",
        "A 0.05 tone at 100 Hz in noise of RMS 0.5 (ten times its amplitude). As a PSD the noise reads its "
        "density, 0.5²/512 = 4.9 × 10⁻⁴ per Hz, whatever the block. As an amplitude, each line holds the noise "
        "in a band about Δf wide: in blocks of 4096 the noise sits at 0.019 and the tone stands 2.6 times above it. "
        "It falls as 1/√N: in blocks of 256 it is at 0.077 and the tone drowns.",
        f1=100.0, a1=0.05, a2=0.0, noise=0.5, block=4096, averages=8, scale=0),
    SignalsTrial(
        "Averaging smooths",
        "Noise alone, of RMS 0.5, as a PSD from one block of 256: each line scatters by 100 %. Raise the averages "
        "to 64 and the scatter falls to an eighth (1/√n), but the level stays at 4.9 × 10⁻⁴: averaging doesn't "
        "lower the noise, it makes its level certain. The PSD's area, 0.25, is the RMS².",
        a1=0.0, a2=0.0, noise=0.5, block=256, averages=1, scale=1),
    SignalsTrial(
        "A chirp's spectrogram",
        "A sine swept from 10 to 400 Hz in 4 s, in blocks of 256 (0.25 s). The averaged spectrum is a flat "
        "smear: it can't say when each frequency was there. The spectrogram, one spectrum per block, shows the "
        "sweep as a line. A shorter block places it better in time and worse in frequency: Δf Δt ≥ 1.",
        kind=Kind.CHIRP, f1=10.0, a1=1.0, f2=400.0, sweep=4.0, block=256, averages=31, tab=TAB_SPECTROGRAM),
    SignalsTrial(
        "A ringing mode",
        "A mode at 50 Hz with ζ = 0.01 struck once: its half-power width is 2ζf<sub>n</sub> = 1 Hz. In a 4 s "
        "block (Δf = 0.25 Hz) it has four lines across it and its peak shows, without a window since it dies "
        "away inside the block. A Hann window here would throw away its start, where most of it is.",
        kind=Kind.RINGING, f1=50.0, a1=1.0, zeta=0.01, block=4096, window=Window.RECTANGULAR),
    SignalsTrial(
        "An accelerometer",
        "A piezoelectric accelerometer is a seismic mass on a stiff crystal: f<sub>n</sub> = 30 kHz, ζ ≈ 0.01. "
        "Below its resonance its output follows the acceleration, within 5 % up to 0.22 f<sub>n</sub>, 6.5 kHz. "
        "A geophone is the other way round: soft, f<sub>n</sub> = 4.5 Hz, and with ζ = 0.7 it follows the "
        "displacement within 5 % from 1.7 f<sub>n</sub>, 7.6 Hz, up.",
        tab=TAB_SENSORS),
)

TIPS = {
    "kind": "<p><b>Two tones:</b> two sines. <b>Chirp:</b> a sine swept from f₁ to f₂, again and again. "
            "<b>Ringing:</b> a mode at f₁ with damping ζ, struck once at t = 0.</p>",
    "f1": "<p>The first tone's frequency (the chirp's start, the mode's frequency).</p>",
    "a1": "<p>Its amplitude (an acceleration, m/s²).</p>",
    "f2": "<p>The second tone's frequency (the chirp's end).</p>",
    "a2": "<p>The second tone's amplitude.</p>",
    "noise": "<p>White noise added at the samples, by its RMS (spread evenly up to the Nyquist frequency).</p>",
    "zeta": "<p>The struck mode's damping ratio.</p>",
    "sweep": "<p>How long each sweep of the chirp takes.</p>",
    "fs": "<p>The sample rate f<sub>s</sub>. Only frequencies up to f<sub>s</sub>/2 (Nyquist) can be "
          "represented.</p>",
    "filter": "<p>An 8-pole Butterworth low-pass at f<sub>s</sub>/2.56 before the sampler, as analyzers have, so "
              "nothing above Nyquist folds back.</p>",
    "block": "<p>Samples per block N: the block lasts N/f<sub>s</sub> and its lines are f<sub>s</sub>/N apart.</p>",
    "window": "<p>The taper each block is multiplied by before its DFT.</p>",
    "averages": "<p>How many blocks are averaged (as power).</p>",
    "overlap": "<p>How much each block overlaps the one before.</p>",
    "scale": "<p><b>Peak amplitude:</b> scaled so a tone reads its amplitude. <b>PSD:</b> scaled so noise reads "
             "its density (units²/Hz), whatever the block or the window.</p>",
    "sensor_hz": "<p>The transducer's own natural frequency: its seismic mass on its spring.</p>",
    "sensor_zeta": "<p>Its damping ratio.</p>",
    "structure_hz": "<p>A mode of the structure being measured.</p>",
    "modal_mass": "<p>The mode's modal mass at the measuring point (scaled to a unit motion there): for a small "
                  "part, a few grams to a kilogram.</p>",
    "sensor_mass": "<p>The sensor's mass, mounting included.</p>",
}
TIME_TIP = (
    "<p>The first block: the analog signal (thin), its samples (dots, joined), and the window it is multiplied by "
    "(dashed, scaled to the block's largest value). Drag or wheel to zoom in time.</p>")
SPECTRUM_TIP = (
    "<p>The averaged spectrum. Circles: each tone's true amplitude, where it should show (folded back if it is "
    "above Nyquist). Dashed: the noise's expected level. Shaded: above f<sub>s</sub>/2.56, where the anti-alias "
    "filter cuts in.</p>")
WINDOWS_TIP = (
    "<p>Each window's spectrum about a tone's line, against the offset in lines: the main lobe, how wide it is, "
    "and the sidelobes, how far they leak. A tone between lines is read at ±0.5.</p>")
SPECTROGRAM_TIP = "<p>One spectrum per block (the acquisition's block and window, 75 % overlap), in dB.</p>"
SENSOR_TIP = (
    "<p>A seismic mass in a case on the structure, read by its motion relative to the case. Blue: its reading "
    "over the acceleration it should read (an accelerometer). Orange: over the displacement (a seismometer). "
    "The band is ±5 %.</p>")


NOTES_HTML = """
<!--BEFORE-->
<h3><a name="sampling"></a>Sampling</h3>
<p>A transducer's signal is continuous; the analyzer keeps its value f<sub>s</sub> times a second. A sine of
frequency f is pinned down by its samples only if there are more than two per cycle: up to the <b>Nyquist
frequency</b> f<sub>s</sub>/2.</p>

<h3><a name="aliasing"></a>Aliasing</h3>
<p>Above Nyquist the samples can't tell a sine at f from one at |f − kf<sub>s</sub>|: they lie on both. So a
tone at f shows at its <b>alias</b>, folded back into 0 … f<sub>s</sub>/2, with its whole amplitude: 900 Hz
sampled at 1024 Hz shows at 124 Hz. Nothing after the sampler can undo this; the data are simply wrong.</p>

<h3><a name="anti-alias"></a>The anti-alias filter</h3>
<p>So an analog low-pass filter runs <i>before</i> the sampler. A real filter can't cut off at Nyquist like a
wall: it rolls off over a band. Analyzers set it to keep f<sub>s</sub>/2.56 and below (here an 8-pole
Butterworth, 3 dB down there and 48 dB per octave beyond), so a tone that could fold back into the kept lines (above
f<sub>s</sub> − f<sub>s</sub>/2.56) is cut by 31 dB or more; real analyzers use steeper (elliptic) filters, 80 dB
or more. The lines above f<sub>s</sub>/2.56 are thrown away: 1024 samples give 400 good
lines, not 512.</p>

<h3><a name="dft-lines"></a>Blocks and lines</h3>
<p>The DFT of a block of N samples, T = N/f<sub>s</sub> long, gives lines</p>
<p align="center">Δf = 1/T = f<sub>s</sub>/N</p>
<p>apart, from 0 to f<sub>s</sub>/2. The resolution is set by the block's length in time, nothing else: a finer
Δf needs a longer block. More samples in the same time (a higher f<sub>s</sub>) only add lines at the top.</p>

<h3><a name="leakage"></a>Leakage</h3>
<p>The DFT treats the block as one period of a signal that repeats forever. A tone with a whole number of cycles
in the block lands on a line and reads exactly. Any other tone, joined end to end, has a jump at each join, and
its energy spreads over every line: <b>leakage</b>. It reads low (by up to 3.9 dB half-way between lines, the
<b>scalloping loss</b>), and its skirts, falling only as 1/f, can bury a small tone tens of lines away. A peak
in an FRF comes out too low and too wide, which reads as too much damping.</p>

<h3><a name="windows"></a>Windows</h3>
<p>Multiplying the block by a taper that goes smoothly to zero at both ends removes the jump. Each window is a
trade, between how wide its main lobe is (how close two tones can be told apart), how low its sidelobes are
(how far a big tone leaks), and its scalloping loss (how wrong an amplitude can read):</p>
<!--WINDOW_TABLE-->
<p><b>Hann</b> is the usual choice for random and continuous signals; <b>flat top</b> for reading amplitudes
(calibration); <b>none</b> (rectangular) for signals that are periodic in the block, or that die away inside it
(a hammer hit, a burst), where a window would cut away the signal itself. The <i>Windows</i> tab draws them.</p>

<h3><a name="scaling"></a>Amplitude or density</h3>
<p>A window changes how much a tone and how much noise come through, differently. Scaled by its coherent gain
(the mean of w), the spectrum reads a tone's amplitude: a <b>peak amplitude</b> (or RMS) spectrum, for tones.
Scaled by its noise power (Σw²) and f<sub>s</sub>, it reads noise's density: the <b>PSD</b>, in units²/Hz, the
same whatever the block and the window, and its area is the mean square (Parseval). Noise in an amplitude
spectrum depends on Δf: each line holds the noise in a band Δf × ENBW wide, so a longer block lowers it, as
1/√N, while a tone stays put. That is how a tone is dug out of noise. A tone in a PSD grows with N instead:
read tones as amplitudes and noise as a density.</p>

<h3><a name="averaging"></a>Averaging</h3>
<p>One block's spectrum of noise is itself random: each line's power scatters by 100 %. Averaging n blocks (as
power) cuts the scatter to 1/√n, without lowering the level. Blocks may <b>overlap</b> (50 % with Hann): the
window throws away the ends of each block, and the overlap uses them in the next, so a record gives nearly
twice the averages. Averaging only makes sense for a steady signal; a transient wants one block that holds
all of it.</p>

<h3><a name="spectrogram"></a>Time and frequency</h3>
<p>An averaged spectrum says what frequencies there are, not when. A <b>spectrogram</b> (short-time Fourier
transform) takes one spectrum per block as the record goes by: a run-up's orders, a chirp's sweep. A block T
long resolves 1/T in frequency and T in time; their product can't go below about 1. The rotor page's waterfall
is one.</p>

<h3><a name="transducers"></a>Transducers</h3>
<p>Most vibration sensors are themselves a single oscillator: a seismic mass m on a spring k and a damper c in a
case fixed to the structure, read by the mass's motion z relative to the case. The case moving as y e<sup>iωt</sup>,</p>
<p align="center">z/y = r² / (1 − r² + 2iζr), &nbsp; r = f/f<sub>n</sub>,</p>
<p>the Single oscillator page's equation with the case's acceleration as the force. Two regimes:</p>
<ul>
<li><b>Accelerometer</b> (r ≪ 1, a stiff spring): z ≈ −ÿ/ω<sub>n</sub>², the mass follows the case and the
spring's stretch is proportional to the acceleration. A piezoelectric crystal is the spring, f<sub>n</sub> tens of
kHz, almost undamped: it reads within 5 % up to 0.22 f<sub>n</sub>. With ζ ≈ 0.7 the flat band would reach
0.6 f<sub>n</sub>, with the phase nearly proportional to frequency (a pure delay), but piezo crystals can't be
damped that much.</li>
<li><b>Seismometer</b> (r ≫ 1, a soft spring): the mass stays still in space, so z ≈ −y, the displacement. A
geophone (f<sub>n</sub> ≈ 4.5 Hz, ζ ≈ 0.7) reads ground displacement from about 1.7 f<sub>n</sub> up.</li>
</ul>
<p>Between them, near r = 1, the sensor resonates: a sensor is only good away from its own f<sub>n</sub>.</p>

<h3><a name="mass-loading"></a>Mass loading</h3>
<p>A sensor's mass m<sub>s</sub> joins the structure where it is fixed. A mode whose modal mass there (scaled to
a unit motion at that point) is m drops to f √(m/(m + m<sub>s</sub>)): a 30 g accelerometer on a panel mode of
0.5 kg modal mass lowers it by 2.9 %. The same sensor roved from point to point loads each mode differently,
and the modes seem to shift between measurements. Light sensors, a fixed reference, or a test that adds a
dummy mass and sees how much the frequency moves, keep it in check.</p>

<h3><a name="try-it"></a>Things to try</h3>
<p>Each of these sets up the signal, the acquisition and the plots (Things to try, top left).</p>
<!--TRIALS-->
<!--TAKEAWAYS-->
"""


def window_table() -> str:
    rows = "".join(
        f"<tr><td>{w.value}</td><td align='right'>{p.main_lobe:.0f}</td><td align='right'>{p.sidelobe_db:.0f}</td>"
        f"<td align='right'>{p.scalloping_db:.2f}</td><td align='right'>{p.enbw:.2f}</td>"
        f"<td align='right'>{p.coherent_gain:.3f}</td></tr>"
        for w, p in ((w, window_props(w)) for w in WINDOWS))
    return ("<table cellspacing='0' cellpadding='3' border='1'><tr><th>Window</th><th>main lobe (lines each "
            "side)</th><th>highest sidelobe (dB)</th><th>scalloping loss (dB)</th><th>noise bandwidth (lines)</th>"
            f"<th>coherent gain</th></tr>{rows}</table>")


NOTES_HTML = NOTES_HTML.replace("<!--BEFORE-->", before(
    ("transient_page", "fourier-pair", "The impulse response and the FRF"),
    ("sdof_page", "half-power", "Half-power bandwidth"),
    ("random_page", "psd", "The power spectral density"),
    ("random_page", "welch", "Estimating it from a record"),
)).replace("<!--WINDOW_TABLE-->", window_table()).replace("<!--TRIALS-->", trials_html(TRIALS)).replace(
    "<!--TAKEAWAYS-->", takeaways(
        "Sample faster than twice the highest frequency present, and filter before sampling: an alias can't be "
        "removed afterwards.",
        "The lines are 1/T apart: resolution comes from the block's length in time.",
        "A tone that isn't periodic in the block leaks; a window trades a wider peak for lower leakage (Hann), or "
        "for a true amplitude (flat top).",
        "Read tones as amplitudes and noise as a PSD; averaging steadies a spectrum but doesn't lower its noise.",
        "A seismic sensor reads acceleration below its own f<sub>n</sub> and displacement above it, and its mass "
        "lowers the modes it measures.",
    ))


class SignalsPage(QtWidgets.QWidget):
    """Controls on the left, the block and its spectrum in the centre, windows, spectrogram and sensors on the
    right."""

    follows_chain = False  # MainWindow: this page has its own model, not the chain

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.source = Source()
        self.x = np.zeros(0)

        self.trials = TrialBox([t.name for t in TRIALS],
                               "<p>Each sets up the signal and its acquisition to show one idea from the Theory "
                               "tab.</p>")
        self.trials.picked.connect(self.load_trial)

        sig = Form("Signal")
        self.kind = QtWidgets.QComboBox()
        self.kind.addItems([k.value for k in KINDS])
        self.kind.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        sig.add("Kind:", self.kind, TIPS["kind"])
        self.f1 = sig.number("f₁:", 0.0, 1e5, 50.0, 2, " Hz", TIPS["f1"])
        self.a1 = sig.number("Amplitude a₁:", 0.0, 1e3, 1.0, 4, " m/s²", TIPS["a1"])
        self.f2 = sig.number("f₂:", 0.0, 1e5, 120.0, 2, " Hz", TIPS["f2"])
        self.a2 = sig.number("Amplitude a₂:", 0.0, 1e3, 0.1, 4, " m/s²", TIPS["a2"])
        self.zeta = sig.number("Damping ζ:", 0.0001, 0.5, 0.01, 4, "", TIPS["zeta"])
        self.sweep = sig.number("Sweep time:", 0.01, 100.0, 4.0, 2, " s", TIPS["sweep"])
        self.noise = sig.number("Noise RMS:", 0.0, 1e3, 0.0, 4, " m/s²", TIPS["noise"])

        acq = Form("Acquisition")
        self.fs = acq.number("Sample rate fₛ:", 1.0, 1e6, 1024.0, 1, " Hz", TIPS["fs"])
        self.filtered = QtWidgets.QCheckBox("Anti-alias filter")
        self.filtered.setChecked(True)
        self.filtered.setToolTip(TIPS["filter"])
        acq.row(self.filtered)
        self.block = QtWidgets.QComboBox()
        self.block.addItems([str(b) for b in BLOCKS])
        self.block.setCurrentIndex(BLOCKS.index(1024))
        acq.add("Block N:", self.block, TIPS["block"])
        self.window = QtWidgets.QComboBox()
        self.window.addItems([w.value for w in WINDOWS])
        self.window.setCurrentIndex(WINDOWS.index(Window.HANN))
        self.window.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        acq.add("Window:", self.window, TIPS["window"])
        self.averages = acq.number("Averages:", 1, 1000, 1, 0, "", TIPS["averages"])
        self.overlap = acq.number("Overlap:", 0.0, 90.0, 50.0, 0, " %", TIPS["overlap"])
        self.acq_info = note_label()
        acq.row(self.acq_info)

        connect_all((self.kind, self.f1, self.a1, self.f2, self.a2, self.zeta, self.sweep, self.noise, self.fs,
                     self.filtered, self.block, self.window, self.averages, self.overlap), self._on_inputs)

        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        for w in (self.trials, sig, acq):
            lv.addWidget(w)
        lv.addStretch(1)

        # --- centre
        self.time_plot = plot("Signal", "m/s²", "Time", "s", height=150)
        self.time_plot.setToolTip(TIME_TIP)
        self.time_plot.setMouseEnabled(x=True, y=False)
        self.analog_item = pg.PlotDataItem()
        self.samples_item = pg.PlotDataItem(symbol="o", symbolSize=4)
        self.window_items = (pg.PlotDataItem(), pg.PlotDataItem())
        for item in (self.analog_item, *self.window_items, self.samples_item):
            self.time_plot.addItem(item)

        self.scale = QtWidgets.QComboBox()
        self.scale.addItems(SCALES)
        self.scale.setToolTip(TIPS["scale"])
        self.scale.currentIndexChanged.connect(lambda _: self._draw_spectrum())
        self.spec_plot = plot("Amplitude", "m/s²", "Frequency", "Hz", log_y=True, height=200)
        self.spec_plot.setToolTip(SPECTRUM_TIP)
        self.spec_plot.setMouseEnabled(x=True, y=False)
        self.spec_legend = add_legend(self.spec_plot.getPlotItem(), offset=(-10, 10))
        self.band = pg.LinearRegionItem((0, 1), movable=False)
        self.band.setZValue(-10)
        for line in self.band.lines:
            line.setPen(pg.mkPen(None))
        self.spec_item = pg.PlotDataItem(name="spectrum")
        self.noise_line = pg.PlotDataItem()
        self.noise_line.setZValue(5)
        self.tone_items = pg.PlotDataItem(pen=None, symbol="o", symbolSize=11)
        self.tone_items.setZValue(6)
        self.legend_shown = ()
        for item in (self.band, self.spec_item, self.noise_line, self.tone_items):
            self.spec_plot.addItem(item)
        self.spec_note = readout_label()

        center = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(center)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.addWidget(self.time_plot, 2)
        self.zoom = QtWidgets.QCheckBox(f"First {ZOOM_SAMPLES} samples only")
        self.zoom.setToolTip("<p>Show the block's first samples close up, to see what the sampler sees, rather "
                             "than the whole block and its window.</p>")
        self.zoom.toggled.connect(lambda _: self._zoom_time())
        scale_row = QtWidgets.QHBoxLayout()
        scale_row.addWidget(self.zoom)
        scale_row.addStretch(1)
        scale_row.addWidget(QtWidgets.QLabel("Spectrum as:"))
        scale_row.addWidget(self.scale)
        cv.addLayout(scale_row)
        cv.addWidget(self.spec_plot, 3)
        cv.addWidget(self.spec_note)

        # --- right: windows
        wins = QtWidgets.QWidget()
        wv = QtWidgets.QVBoxLayout(wins)
        self.kernel_plot = plot("Level", "dB", "Offset from the tone", "lines", height=200)
        self.kernel_plot.setToolTip(WINDOWS_TIP)
        self.kernel_plot.setMouseEnabled(x=False, y=False)
        add_legend(self.kernel_plot.getPlotItem(), offset=(-10, 10))
        self.kernel_items = [pg.PlotDataItem(name=w.value) for w in WINDOWS]
        self.half_lines = [pg.InfiniteLine(angle=90, pos=p) for p in (-0.5, 0.5)]
        for item in (*self.half_lines, *self.kernel_items):
            self.kernel_plot.addItem(item)
        wv.addWidget(self.kernel_plot, 1)
        self.shape_plot = plot("w", None, "Sample / N", None, height=110)
        self.shape_plot.setMouseEnabled(x=False, y=False)
        self.shape_items = [pg.PlotDataItem() for _ in WINDOWS]
        for item in self.shape_items:
            self.shape_plot.addItem(item)
        wv.addWidget(self.shape_plot)
        self.window_note = note_label()
        wv.addWidget(self.window_note)

        # --- right: spectrogram
        sgram = QtWidgets.QWidget()
        sgv = QtWidgets.QVBoxLayout(sgram)
        self.sgram_plot = plot("Frequency", "Hz", "Time", "s", height=200)
        self.sgram_plot.setToolTip(SPECTROGRAM_TIP)
        self.sgram_plot.setMouseEnabled(x=False, y=False)
        self.sgram_image = pg.ImageItem(axisOrder="row-major")
        self.sgram_plot.addItem(self.sgram_image)
        sgv.addWidget(self.sgram_plot, 1)
        self.sgram_note = note_label()
        sgv.addWidget(self.sgram_note)

        # --- right: transducers
        sensors = QtWidgets.QWidget()
        sv = QtWidgets.QVBoxLayout(sensors)
        sensor = Form("The sensor")
        self.sensor_hz = sensor.number("Its fₙ:", 0.01, 1e6, 30000.0, 2, " Hz", TIPS["sensor_hz"])
        self.sensor_zeta = sensor.number("Its damping ζ:", 0.0, 2.0, 0.01, 3, "", TIPS["sensor_zeta"])
        sensor.changed.connect(self._draw_tab)
        sv.addWidget(sensor)
        self.ratio_plot = plot("Reading / truth", None, "f / fₙ (the sensor's)", None, log_x=True, log_y=True,
                               height=150)
        self.ratio_plot.setToolTip(SENSOR_TIP)
        self.ratio_plot.setMouseEnabled(x=False, y=False)
        add_legend(self.ratio_plot.getPlotItem(), offset=(10, 10))
        self.flat_band = pg.LinearRegionItem((math.log10(0.95), math.log10(1.05)), orientation="horizontal",
                                             movable=False)
        for line in self.flat_band.lines:
            line.setPen(pg.mkPen(None))
        self.acc_item = pg.PlotDataItem(name="accelerometer")
        self.seis_item = pg.PlotDataItem(name="seismometer")
        for item in (self.flat_band, self.acc_item, self.seis_item):
            self.ratio_plot.addItem(item)
        sv.addWidget(self.ratio_plot, 2)
        self.phase_plot = plot("Accelerometer's lag", "°", "f / fₙ (the sensor's)", None, log_x=True, height=100)
        self.phase_plot.setMouseEnabled(x=False, y=False)
        self.phase_item = pg.PlotDataItem()
        self.phase_plot.addItem(self.phase_item)
        self.phase_plot.setXLink(self.ratio_plot)
        sv.addWidget(self.phase_plot, 1)
        self.sensor_note = note_label()
        sv.addWidget(self.sensor_note)
        loading = Form("Mass loading")
        self.structure_hz = loading.number("A mode at:", 0.1, 1e5, 200.0, 1, " Hz", TIPS["structure_hz"])
        self.modal_mass = loading.number("Its modal mass:", 0.001, 1e4, 0.5, 3, " kg", TIPS["modal_mass"])
        self.sensor_mass = loading.number("Sensor's mass:", 0.0, 100.0, 0.03, 4, " kg", TIPS["sensor_mass"])
        self.loading_note = note_label()
        loading.row(self.loading_note)
        loading.changed.connect(self._show_loading)
        sv.addWidget(loading)

        self.notes = QtWidgets.QTextBrowser()
        self.notes.setHtml(NOTES_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(wins, "Windows")
        self.tabs.addTab(sgram, "Spectrogram")
        self.tabs.addTab(ColumnScroll(sensors), "Transducers")
        self.tabs.addTab(self.notes, "Theory")
        self.tabs.currentChanged.connect(lambda _: self._draw_tab())

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(ColumnScroll(left))
        splitter.addWidget(center)
        splitter.addWidget(self.tabs)
        splitter.setSizes([320, 760, 560])
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        self._on_inputs()
        self.apply_theme()

    # ------------------------------------------------------------- inputs
    def load_trial(self, index: int) -> None:
        t = TRIALS[index]
        for box, value in ((self.kind, KINDS.index(t.kind)), (self.f1, t.f1), (self.a1, t.a1), (self.f2, t.f2),
                           (self.a2, t.a2), (self.noise, t.noise), (self.zeta, t.zeta), (self.sweep, t.sweep),
                           (self.fs, t.fs), (self.block, BLOCKS.index(t.block)),
                           (self.window, WINDOWS.index(t.window)), (self.averages, t.averages),
                           (self.overlap, t.overlap), (self.filtered, t.filtered), (self.scale, t.scale), (self.zoom, t.zoom),
                           (self.sensor_hz, t.sensor_hz), (self.sensor_zeta, t.sensor_zeta)):
            set_quietly(box, value)
        self.trials.show_note(t.note)
        self.tabs.setCurrentIndex(t.tab)
        self._on_inputs()

    @property
    def n_block(self) -> int:
        return BLOCKS[self.block.currentIndex()]

    @property
    def window_kind(self) -> Window:
        return WINDOWS[self.window.currentIndex()]

    def _on_inputs(self) -> None:
        kind = KINDS[self.kind.currentIndex()]
        self.source = Source(kind, self.f1.value(), self.a1.value(), self.f2.value(), self.a2.value(),
                             self.noise.value(), self.zeta.value(), self.sweep.value())
        self.zeta.setEnabled(kind is Kind.RINGING)
        self.sweep.setEnabled(kind is Kind.CHIRP)
        self.f2.setEnabled(kind is not Kind.RINGING)
        self.a2.setEnabled(kind is Kind.TONES)
        fs, n = self.fs.value(), self.n_block
        self.n_record = blocks_needed(n, int(self.averages.value()), self.overlap.value() / 100.0)
        _, self.x = sample(self.source, fs, self.n_record, self.filtered.isChecked())
        self.spec = spectrum(self.x, fs, n, self.window_kind, self.overlap.value() / 100.0)
        self.acq_info.setText(
            f"Block T = {fmt(n / fs, 's')}, lines Δf = {fmt(fs / n, 'Hz')} apart up to Nyquist, "
            f"{fmt(fs / 2, 'Hz')}" + (f"; good to f<sub>s</sub>/2.56 = {fmt(fs / FILTER_RATIO, 'Hz')}"
                                      if self.filtered.isChecked() else "")
            + f". The record: {self.n_record} samples, {fmt(self.n_record / fs, 's')}, {self.spec.blocks} "
            f"block{'s' if self.spec.blocks > 1 else ''}.")
        self.redraw()

    # ------------------------------------------------------------- drawing
    def apply_theme(self) -> None:
        c = colors
        self.analog_item.setPen(pg.mkPen(c.grey, width=1))
        self.samples_item.setPen(pg.mkPen(c.mass[0], width=1.5))
        self.samples_item.setSymbolPen(pg.mkPen(c.mass[0], width=1))
        self.samples_item.setSymbolBrush(pg.mkBrush(c.mass[0]))
        for item in self.window_items:
            item.setPen(pg.mkPen(c.fit, width=1.2, style=QtCore.Qt.PenStyle.DashLine))
        self.band.setBrush(pg.mkBrush(*c.band_shade))
        self.spec_item.setPen(pg.mkPen(c.mass[0], width=1.5))
        self.noise_line.setPen(pg.mkPen(c.strong, width=1.5, style=QtCore.Qt.PenStyle.DashLine))
        self.tone_items.setSymbolPen(pg.mkPen(c.force, width=2))
        self.tone_items.setSymbolBrush(None)
        for item, color in zip(self.kernel_items, (c.grey, c.mass[0], c.mode[3])):
            item.setPen(pg.mkPen(color, width=1.5))
        for item, color in zip(self.shape_items, (c.grey, c.mass[0], c.mode[3])):
            item.setPen(pg.mkPen(color, width=1.5))
        for line in self.half_lines:
            line.setPen(pg.mkPen(c.faint, width=1, style=QtCore.Qt.PenStyle.DotLine))
        self.sgram_image.setLookupTable(pg.ColorMap([0.0, 0.5, 1.0], [
            pg.mkColor(c.background), pg.mkColor(c.mode[1]), pg.mkColor(c.force)]).getLookupTable(nPts=256))
        self.flat_band.setBrush(pg.mkBrush(*c.band_shade))
        self.acc_item.setPen(pg.mkPen(c.mass[0], width=2))
        self.seis_item.setPen(pg.mkPen(c.mass[1], width=2))
        self.phase_item.setPen(pg.mkPen(c.mass[0], width=1.5))
        self.redraw()

    def redraw(self) -> None:
        self._draw_time()
        self._draw_spectrum()
        self._draw_tab()
        self._show_loading()
        settle_plots(self)

    def _draw_time(self) -> None:
        fs, n = self.fs.value(), self.n_block
        t = np.arange(n) / fs
        x = self.x[:n]
        fine = np.linspace(0.0, t[-1], min(ANALOG_POINTS, 20 * n))
        self.analog_item.setData(fine, self.source.analog(fine, fs, self.filtered.isChecked()))
        self.samples_item.setData(t, x)
        self.samples_item.setSymbolSize(4 if n <= 4096 else 0)
        top = float(np.max(np.abs(x))) if x.size else 1.0
        w = window(self.window_kind, n) * (top or 1.0)
        self.window_items[0].setData(t, w)
        self.window_items[1].setData(t, -w)
        self._zoom_time()

    def _zoom_time(self) -> None:
        fs, n = self.fs.value(), self.n_block
        shown = min(n, ZOOM_SAMPLES) if self.zoom.isChecked() else n
        self.time_plot.setXRange(0.0, (shown - 1) / fs, padding=0.02)
        settle_plots(self)

    def _draw_spectrum(self) -> None:
        fs, n, spec = self.fs.value(), self.n_block, self.spec
        psd = self.scale.currentIndex() == 1
        values = spec.psd if psd else spec.amplitude
        self.spec_plot.setLabel("left", "PSD" if psd else "Amplitude", units="(m/s²)²/Hz" if psd else "m/s²")
        floor = values[values > 0]
        low = float(np.min(floor)) if floor.size else 1e-12
        self.spec_item.setData(spec.freqs, np.maximum(values, 1e-300))
        nyquist = fs / 2
        self.band.setRegion((fs / FILTER_RATIO, nyquist))
        self.band.setVisible(self.filtered.isChecked())
        noise = self.source.noise
        props = window_props(self.window_kind, n)
        if noise > 0:
            level = noise ** 2 / nyquist if psd else 2 * noise * math.sqrt(props.enbw / n)
            self.noise_line.setData([0.0, nyquist], [level, level])
        else:
            self.noise_line.setData([], [])
        lines, texts = [], []
        for f, a in self.source.tones():
            shown = apparent(f, fs)
            # in a PSD a tone's power a²/2 shows spread over the window's noise bandwidth
            true = a * a / 2 / (props.enbw * spec.df) if psd else a
            lines.append((shown, true))
            at, reads = nearest_line_reading(spec, shown)
            where = f"{fmt(f, 'Hz')}" + (f" shows at {fmt(shown, 'Hz')}" if abs(shown - f) > 1e-9 else "")
            if psd:
                texts.append(f"{where}: its power {a * a / 2:.3g} (m/s²)² over {fmt(spec.df, 'Hz')} lines")
            else:
                texts.append(f"{where} (amplitude {a:.4g}): reads {reads:.4g} at {fmt(at, 'Hz')} "
                             f"({round(db(reads / a), 2) + 0.0 if a > 0 else 0.0:+.2f} dB)")
        self.tone_items.setData([p[0] for p in lines], [p[1] for p in lines])
        self._legend(noise > 0, bool(lines))
        mean_square = float(np.mean(self.x ** 2)) if self.x.size else 0.0
        area = float(np.sum(spec.psd) * spec.df)
        texts.append(f"the record's mean square {mean_square:.4g}, the PSD's area {area:.4g}")
        text = "; ".join(texts)
        self.spec_note.setText(text[0].upper() + text[1:] + ".")
        top = max(float(np.max(values)) if values.size else 1.0, max((p[1] for p in lines), default=0.0))
        bottom = max(low, top * 1e-7)
        self.spec_plot.setXRange(0.0, nyquist, padding=0)
        self.spec_plot.setYRange(math.log10(bottom) - 0.3, math.log10(top) + 0.5, padding=0)

    def _legend(self, noise: bool, tones: bool) -> None:
        """List the noise's level and the true tones only when there are some."""
        shown = (noise, tones)
        if shown == self.legend_shown:
            return
        self.legend_shown = shown
        for item in (self.noise_line, self.tone_items):
            self.spec_legend.removeItem(item)
        if noise:
            self.spec_legend.addItem(self.noise_line, "noise's expected level")
        if tones:
            self.spec_legend.addItem(self.tone_items, "true tones")

    def _draw_tab(self) -> None:
        i = self.tabs.currentIndex()
        if i == TAB_WINDOWS:
            self._draw_windows()
        elif i == TAB_SPECTROGRAM:
            self._draw_spectrogram()
        elif i == TAB_SENSORS:
            self._draw_sensors()

    def _draw_windows(self) -> None:
        current = self.window_kind
        for item, shape, w in zip(self.kernel_items, self.shape_items, WINDOWS):
            bins, level = kernel(w, SPAN)
            item.setData(bins, np.maximum(level, -140.0))
            u = np.linspace(0.0, 1.0, 257)
            shape.setData(u, window(w, 257) if w is not Window.RECTANGULAR else np.ones(257))
            width = 2.5 if w is current else 1.0
            for it in (item, shape):
                pen = pg.mkPen(it.opts["pen"])
                pen.setWidthF(width)
                it.setPen(pen)
        self.kernel_plot.setXRange(-SPAN, SPAN, padding=0)
        self.kernel_plot.setYRange(-120.0, 5.0, padding=0)
        p = window_props(current, self.n_block)
        self.window_note.setText(
            f"{current.value}: main lobe {p.main_lobe:.0f} lines each side, highest sidelobe {p.sidelobe_db:.0f} dB, "
            f"scalloping loss {p.scalloping_db:.2f} dB, noise bandwidth {p.enbw:.2f} lines "
            f"({fmt(p.enbw * self.spec.df, 'Hz')} here).")

    def _draw_spectrogram(self) -> None:
        fs, n = self.fs.value(), self.n_block
        if self.x.size < 2 * n:
            self.sgram_image.clear()
            self.sgram_note.setText("The record holds one block: raise the averages to see it change in time.")
            return
        t, f, level = spectrogram(self.x, fs, n, self.window_kind)
        top = float(np.max(level))
        image = np.clip((level.T - (top - 60.0)) / 60.0, 0.0, 1.0)
        self.sgram_image.setImage(image, autoLevels=False, levels=(0.0, 1.0))
        dt = t[1] - t[0] if t.size > 1 else n / fs
        self.sgram_image.setRect(QtCore.QRectF(t[0] - dt / 2, 0.0, dt * t.size, f[-1] + fs / n))
        self.sgram_plot.setXRange(0.0, self.x.size / fs, padding=0)
        self.sgram_plot.setYRange(0.0, fs / 2, padding=0)
        self.sgram_note.setText(
            f"{t.size} blocks of {fmt(n / fs, 's')}, {fmt(dt, 's')} apart (75 % overlap), lines {fmt(fs / n, 'Hz')} "
            "apart; the colour spans 60 dB below the strongest.")

    def _draw_sensors(self) -> None:
        z = self.sensor_zeta.value()
        acc, seis, lag = base_excited(SENSOR_RATIOS, z)
        self.acc_item.setData(SENSOR_RATIOS, acc)
        self.seis_item.setData(SENSOR_RATIOS, seis)
        self.phase_item.setData(SENSOR_RATIOS, lag)
        self.ratio_plot.setXRange(-2.0, 2.0, padding=0)
        self.ratio_plot.setYRange(-2.0, min(math.log10(float(max(acc.max(), seis.max()))) + 0.2, 3.0), padding=0)
        self.phase_plot.setYRange(0.0, 180.0, padding=0.05)
        fn = self.sensor_hz.value()
        up, down = usable_band(z), usable_band(z, accelerometer=False)
        self.sensor_note.setText(
            f"Within ±5 %: as an accelerometer up to r = {up:.3g} ({fmt(up * fn, 'Hz')}); as a seismometer from "
            f"r = {down:.3g} ({fmt(down * fn, 'Hz')}) up.")

    def _show_loading(self) -> None:
        f = self.structure_hz.value()
        loaded = mass_loaded(f, self.modal_mass.value(), self.sensor_mass.value())
        self.loading_note.setText(f"With the sensor on, the mode is at {loaded:.4g} Hz, "
                                  f"{100 * (loaded / f - 1):+.2f} %.")
