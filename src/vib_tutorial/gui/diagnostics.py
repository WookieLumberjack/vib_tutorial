"""Machinery diagnostics page: order tracking under a run-up, gear-mesh sidebands, bearing defects by envelope
analysis, and the cepstrum.

Left: the machine (speed, gear and its fault, bearing and its fault, the casing's resonance, noise), the bearing's
geometry, and the run-up. Centre: the casing's acceleration and its spectrum. Right: the run-up's frequency and
order maps, the orders tracked against their steady state, the spectrum round the mesh, the envelope spectrum,
the cepstrum, and the Theory. The page has its own model (core/diagnostics.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
import scipy.signal
from PySide6 import QtCore, QtWidgets

from ..core.diagnostics import (
    Bearing,
    BearingFault,
    GearFault,
    Machine,
    RunUp,
    amplitude_spectrum,
    band_pass,
    cepstrum,
    envelope,
    envelope_spectrum,
    frequency_map,
    kurtosis,
    line_amplitude,
    order_map,
    order_track,
    record,
    run_up,
    sidebands,
)
from . import settle_plots
from .lesson import (
    Form,
    TrialBox,
    column,
    combo,
    connect_all,
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

TAB_RUNUP, TAB_ORDERS, TAB_SIDEBANDS, TAB_ENVELOPE, TAB_CEPSTRUM = range(5)
GEAR_FAULTS = tuple(GearFault)
BEARING_FAULTS = tuple(BearingFault)
SHOW_S = 0.1  # s of the record drawn
SIDEBANDS = 6  # each side of the mesh
TOP_HZ = 5000.0  # the spectrum's range
MAP_DB = 50.0  # the maps' colour range below their peak
MAP_TOP_HZ = 300.0  # the frequency map's
MAP_TOP_ORDER = 6.0  # the order map's
ENVELOPE_TOP_HZ = 500.0
CEPSTRUM_TOP_S = 0.1


@dataclass(frozen=True)
class DiagnosticsTrial:
    name: str
    note: str
    rpm: float = 1797.0
    gear: GearFault = GearFault.NONE
    bearing: BearingFault = BearingFault.NONE
    noise: float = 0.2  # m/s² RMS
    run_from: float = 600.0  # rpm
    run_to: float = 4200.0
    run_s: float = 20.0
    tab: int = TAB_SIDEBANDS


TRIALS = (
    DiagnosticsTrial(
        "A healthy gearbox",
        "A shaft at 1797 rpm (29.95 Hz) drives a 27-tooth pinion: the mesh is at 27 × 29.95 = 808.65 Hz, with "
        "harmonics at 1617 and 2426 Hz. The spectrum has those, the shaft's 1X and 2X, and noise; round the "
        "mesh nothing stands above the floor. The record's kurtosis is 2.4, below a Gaussian's 3: a sum of "
        "steady sines is flatter-topped than noise."),
    DiagnosticsTrial(
        "An eccentric pinion",
        "The pinion runs out of true: once a turn the mesh grows and shrinks, and its phase swings, so lines "
        "appear at 808.65 ± 29.95 Hz, 0.23 and 0.08 m/s² beside the mesh's 1.0. Amplitude and phase "
        "modulation together make the pair unequal. One pair of sidebands is a smooth, once-a-turn change.",
        gear=GearFault.ECCENTRIC),
    DiagnosticsTrial(
        "A cracked tooth",
        "One tooth is cracked: once a turn, for about one tooth's mesh, the mesh kicks. A short event once a turn "
        "has a long family of sidebands, all about 0.04 m/s², spaced by the shaft's 29.95 Hz. The cepstrum "
        "(Cepstrum tab) gathers the whole family into one peak at 1/29.95 Hz = 33.4 ms, 0.12 against 0.027 "
        "for the healthy gear.",
        gear=GearFault.CRACK, tab=TAB_CEPSTRUM),
    DiagnosticsTrial(
        "Outer race defect",
        "A spall on the bearing's outer race (a 6205: 9 balls) in a noisier machine (1 m/s² RMS): a ball "
        "strikes it at BPFO = 3.585 × the shaft's speed = 107.4 Hz, and each strike rings the casing's 3 kHz "
        "resonance. The plain spectrum reads 0.049 m/s² at 107 Hz, where its floor is 0.043: lost. The "
        "record's kurtosis is 3.1, like noise, but the band round 3 kHz has 5.6, and its envelope's spectrum "
        "(Envelope tab) has 0.45 m/s² at BPFO, then 0.34 and 0.23 at its harmonics, ten times its floor.",
        bearing=BearingFault.OUTER, noise=1.0, tab=TAB_ENVELOPE),
    DiagnosticsTrial(
        "Inner race defect",
        "The defect on the inner race turns with the shaft, through the load zone once a turn, so its strikes "
        "(at BPFI = 162.2 Hz) are hard, then soft: the envelope spectrum has BPFI with sidebands at ± 1X "
        "(0.16 and 0.13 m/s² beside 0.31), the mark of a defect that turns.",
        bearing=BearingFault.INNER, noise=1.0, tab=TAB_ENVELOPE),
    DiagnosticsTrial(
        "Ball defect",
        "A ball's defect strikes both races once each spin: at 2 × BSF = 141.2 Hz. The ball rides round with "
        "the cage, so its strikes swell and fade at the cage's FTF = 11.9 Hz: the envelope spectrum has 0.29 "
        "m/s² at 141 Hz, with sidebands of 0.14 and 0.135 at ± FTF.",
        bearing=BearingFault.BALL, noise=1.0, tab=TAB_ENVELOPE),
    DiagnosticsTrial(
        "A run-up through a resonance",
        "The shaft runs up from 600 to 4200 rpm in 20 s, its unbalance (∝ speed²) and a misalignment (2X) "
        "driving a casing mode at 40 Hz (Run-up tab). Against frequency the orders fan out as rays and the mode "
        "is a horizontal band they light up as they cross it; against order they stand still. Tracked (Orders "
        "tab), the 1X peaks at 209 m/s² at 2445 rpm, close to the steady curve's 211 at 2406 rpm. The 2X peaks "
        "near 1220 rpm, half the speed, but at 28.5 m/s² against its steady 33.3: it sweeps through the mode "
        "twice as fast.",
        tab=TAB_RUNUP),
    DiagnosticsTrial(
        "A fast run-up",
        "The same run-up in 4 s: the 1X sweeps through the mode's half-power band (2.4 Hz) in about 0.16 s, "
        "only six cycles of the mode, so the response can't build up. It peaks lower (179 m/s², 15 % under the "
        "steady 211) and later (2531 rpm), and rings after (Orders tab). Run up slowly to read a resonance.",
        run_s=4.0, tab=TAB_ORDERS),
)

TIPS = {
    "rpm": "<p>The shaft's speed; the gear and the bearing's inner race turn with it.</p>",
    "teeth": "<p>The pinion's teeth: the mesh is at teeth × the shaft's speed.</p>",
    "gear": "<p>An eccentric pinion modulates the mesh once a turn; a cracked tooth kicks it once a turn.</p>",
    "severity": "<p>The gear fault's depth: the modulation's share of the mesh's amplitude.</p>",
    "bearing": "<p>Where the bearing's defect is: each ball that meets it strikes it.</p>",
    "impact": "<p>Each strike's peak response at the casing.</p>",
    "resonance": "<p>The casing's mode the strikes ring (structures have them in the kilohertz).</p>",
    "noise": "<p>White noise at the accelerometer, RMS.</p>",
    "balls": "<p>The number of balls.</p>",
    "ball": "<p>The balls' diameter.</p>",
    "pitch": "<p>The pitch circle's diameter: through the balls' centres.</p>",
    "contact": "<p>The contact angle: 0 for a deep-groove bearing under a radial load.</p>",
    "band": "<p>The band the envelope is taken in: round a resonance the strikes ring, clear of the mesh.</p>",
    "run_from": "<p>The run-up's starting speed.</p>",
    "run_to": "<p>The run-up's final speed.</p>",
    "run_s": "<p>How long the run-up takes (the speed rises linearly).</p>",
    "mode": "<p>The casing's mode the run-up passes through.</p>",
}
TIME_TIP = "<p>The casing's acceleration over the first 0.1 s.</p>"
SPECTRUM_TIP = ("<p>The amplitude spectrum (flat-top window, averaged): each tone reads its amplitude. Dotted: the "
                "mesh's harmonics.</p>")
MAP_TIP = ("<p>The run-up's spectrum, each block's line against the speed then. Top: against frequency (the orders "
           "fan out). Bottom: against order, from the record resampled at equal angles (they stand still).</p>")
ORDERS_TIP = "<p>Each order's amplitude tracked through the run-up; dashed: its steady state at each speed.</p>"
SIDEBAND_TIP = "<p>The spectrum round the mesh; dots: the lines at the mesh ± k × the shaft's speed.</p>"
ENVELOPE_TIP = ("<p>Top: the record band-passed round the resonance, and its envelope. Bottom: the envelope's "
                "spectrum; dotted: the defect frequency and its harmonics, dots: its sidebands.</p>")
CEPSTRUM_TIP = ("<p>The real cepstrum: the inverse FFT of the log power spectrum. Dotted: 1/the shaft's speed and "
                "its multiples; dashed: 1/the mesh.</p>")

NOTES_HTML = """
<!--BEFORE-->
<h3><a name="machines"></a>What a machine's vibration says</h3>
<p>Every part of a running machine leaves its own frequencies in the casing's vibration: the shafts their
speeds and harmonics (unbalance at 1X, misalignment at 2X), the gears their tooth-mesh frequency (teeth × the
shaft's speed), the rolling bearings their defect frequencies. Condition monitoring reads the spectrum against
that list. A fault changes a line, adds new ones, or adds a pattern: sidebands, impacts, families of
harmonics. This page synthesizes a gearbox's casing acceleration and looks at it four ways.</p>

<h3><a name="orders"></a>Run-ups and order tracking</h3>
<p>While the speed changes, a frequency analysis smears: a fixed-length block sees the 1X sweep across several
lines. Each force, though, stays at a fixed <b>order</b> (a multiple of the speed), so <b>order tracking</b>
analyses in angle instead of time: the record is resampled at equal steps of the shaft's angle (from a
tachometer's pulses) and each block of whole revolutions is transformed. An order is then one line, whatever the
speed did. Against frequency, the orders are rays from the origin and a structural mode a horizontal band they
light up as they cross; against order the orders stand still and the mode's band slants. Each order's amplitude
along the run-up is its <b>order track</b>, the Bode plot of the Jeffcott rotor page measured.</p>
<p>A run-up that is too fast misreads a resonance: the response needs about Q cycles to build up, so a sweep that
crosses the half-power band in fewer peaks lower and later, and rings after.</p>

<h3><a name="gears"></a>Gear-mesh sidebands</h3>
<p>The mesh's force repeats at every tooth, at f<sub>m</sub> = z f<sub>shaft</sub>, with harmonics. A fault on
one gear modulates it once a turn of that gear, which puts <b>sidebands</b> at f<sub>m</sub> ± k f<sub>shaft</sub>.
How many tells what kind of fault: a smooth modulation (an eccentric or bent gear) gives one or two pairs; a
short event once a turn (a cracked or broken tooth) gives a long, flat family, since a narrow pulse's spectrum is
broad. Amplitude and phase modulation together make the pairs unequal. The spacing says which shaft the faulty
gear is on.</p>

<h3><a name="envelope"></a>Bearings and the envelope</h3>
<p>A rolling bearing with a spall gives a strike each time a ball meets it. Where depends on the geometry
(n balls of diameter d on a pitch circle D, contact angle α, the inner race turning at f<sub>r</sub>):</p>
<p align="center">FTF = f<sub>r</sub>/2 (1 − d/D cos α), &nbsp; BPFO = n FTF, &nbsp;
BPFI = n f<sub>r</sub>/2 (1 + d/D cos α), &nbsp; BSF = D f<sub>r</sub>/2d (1 − (d/D cos α)²).</p>
<p>Each strike is a short impulse that rings the structure's resonances, in the kilohertz. The strikes' rate
carries little energy at its own frequency, so the plain spectrum barely shows it under the shafts and the mesh.
<b>Envelope analysis</b> band-passes the record round a resonance the strikes ring (clear of the gears), takes the
envelope (the magnitude of the analytic signal, by the Hilbert transform), and transforms that: the strikes'
rate, and its harmonics, stand out. The kurtosis of the band (3 for Gaussian noise) climbs with impacts and helps
pick the band. Inner-race and ball defects pass through the load zone, so their strikes swell and fade: sidebands
at ± 1X (inner race) or ± FTF (ball). The balls slip a little, so the strikes' spacing scatters by a percent or
two: the envelope's lines stay sharp at low harmonics, while the raw spectrum's high ones smear.</p>

<h3><a name="cepstrum"></a>The cepstrum</h3>
<p>A family of lines spaced Δf is a periodic ripple in the log spectrum. The <b>cepstrum</b>, the inverse Fourier
transform of the log power spectrum, turns that ripple into one peak at the <b>quefrency</b> 1/Δf (and its
<i>rahmonics</i>). A gear's sidebands, however many and however unequal, become one peak at 1/f<sub>shaft</sub>;
the mesh's own harmonics another at 1/f<sub>m</sub>. The log also separates a source from the path it passes
through, which multiply in the spectrum and add in the cepstrum.</p>

<h3><a name="try-it"></a>Things to try</h3>
<p>Each of these sets up the machine (Things to try, top left).</p>
<!--TRIALS-->
<!--TAKEAWAYS-->
"""
NOTES_HTML = NOTES_HTML.replace("<!--BEFORE-->", before(
    ("signals_page", "spectrogram", "Time and frequency"),
    ("torsional_page", "gears", "Gears and the tooth-pass frequency"),
    ("rotor_page", "waterfall", "Waterfall and cascade plots"),
)).replace("<!--TRIALS-->", trials_html(TRIALS)).replace("<!--TAKEAWAYS-->", takeaways(
    "Each part of a machine has its own frequencies; a diagnosis reads the spectrum against them.",
    "Under a changing speed, track orders: resample at equal angles, and each order stays on one line.",
    "Sidebands at the mesh ± the shaft's speed point to a gear; a long flat family to a local fault.",
    "Bearing strikes hide in the plain spectrum; band-pass round a resonance and look at the envelope's.",
    "The cepstrum turns a family of equally spaced lines into one peak at 1/spacing.",
))


class DiagnosticsPage(QtWidgets.QWidget):
    """Controls on the left, the record and its spectrum in the centre, the analyses on the right."""

    follows_chain = False  # MainWindow: this page has its own model, not the chain

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.trials = TrialBox([t.name for t in TRIALS],
                               "<p>Each sets up the machine to show one idea from the Theory tab.</p>")
        self.trials.picked.connect(self.load_trial)

        machine = Form("Machine")
        self.rpm = machine.number("Shaft speed:", 300.0, 6000.0, 1797.0, 0, " rpm", TIPS["rpm"])
        self.teeth = machine.number("Pinion teeth:", 8, 80, 27, 0, "", TIPS["teeth"])
        self.gear = combo([g.value for g in GEAR_FAULTS], TIPS["gear"])
        machine.add("Gear fault:", self.gear, TIPS["gear"])
        self.severity = machine.number("Fault depth:", 0.0, 100.0, 30.0, 0, " %", TIPS["severity"])
        self.bearing_fault = combo([b.value for b in BEARING_FAULTS], TIPS["bearing"])
        machine.add("Bearing defect:", self.bearing_fault, TIPS["bearing"])
        self.impact = machine.number("Strike:", 0.0, 50.0, 3.0, 1, " m/s²", TIPS["impact"])
        self.resonance = machine.number("Resonance:", 500.0, 8000.0, 3000.0, 0, " Hz", TIPS["resonance"])
        self.noise = machine.number("Noise:", 0.0, 5.0, 0.2, 2, " m/s²", TIPS["noise"])

        geometry = Form("Bearing")
        self.balls = geometry.number("Balls:", 5, 30, 9, 0, "", TIPS["balls"])
        self.ball = geometry.number("Ball diameter:", 1.0, 50.0, 7.94, 2, " mm", TIPS["ball"])
        self.pitch = geometry.number("Pitch diameter:", 5.0, 300.0, 39.04, 2, " mm", TIPS["pitch"])
        self.contact = geometry.number("Contact angle:", 0.0, 40.0, 0.0, 0, "°", TIPS["contact"])
        self.low = geometry.number("Envelope from:", 100.0, 9000.0, 2550.0, 0, " Hz", TIPS["band"])
        self.high = geometry.number("to:", 200.0, 10000.0, 3450.0, 0, " Hz", TIPS["band"])
        self.bearing_info = note_label()
        geometry.row(self.bearing_info)

        run = Form("Run-up")
        self.run_from = run.number("From:", 60.0, 6000.0, 600.0, 0, " rpm", TIPS["run_from"])
        self.run_to = run.number("To:", 120.0, 9000.0, 4200.0, 0, " rpm", TIPS["run_to"])
        self.run_s = run.number("Over:", 1.0, 60.0, 20.0, 1, " s", TIPS["run_s"])
        self.mode = run.number("Casing mode:", 5.0, 200.0, 40.0, 1, " Hz", TIPS["mode"])

        connect_all((self.rpm, self.teeth, self.gear, self.severity, self.bearing_fault, self.impact,
                     self.resonance, self.noise, self.balls, self.ball, self.pitch, self.contact, self.low,
                     self.high), self.redraw)
        connect_all((self.run_from, self.run_to, self.run_s, self.mode), self._on_run)

        # --- centre
        self.time_plot = plot("Acceleration", "m/s²", "Time", "s", height=130)
        self.time_plot.setToolTip(TIME_TIP)
        self.time_item = pg.PlotDataItem()
        self.time_plot.addItem(self.time_item)
        self.spec_plot = plot("Amplitude", "m/s²", "Frequency", "Hz", log_y=True, height=170)
        self.spec_plot.setToolTip(SPECTRUM_TIP)
        self.mesh_lines = [pg.InfiniteLine(angle=90) for _ in range(3)]
        self.spec_item = pg.PlotDataItem()
        for item in (*self.mesh_lines, self.spec_item):
            self.spec_plot.addItem(item)
        self.readout = readout_label()

        # --- right: the run-up's maps
        self.freq_map = plot("Frequency", "Hz", "Shaft speed (rpm)", None, height=170)
        self.order_map_plot = plot("Order", None, "Shaft speed (rpm)", None, height=170)
        self.order_map_plot.setXLink(self.freq_map)
        self.freq_image = pg.ImageItem(axisOrder="row-major")
        self.order_image = pg.ImageItem(axisOrder="row-major")
        for p, image in ((self.freq_map, self.freq_image), (self.order_map_plot, self.order_image)):
            p.setToolTip(MAP_TIP)
            p.setMouseEnabled(x=False, y=False)
            p.addItem(image)
        self.mode_line = pg.InfiniteLine(angle=0)
        self.freq_map.addItem(self.mode_line)
        self.map_note = note_label()

        # --- right: order tracks
        self.orders_plot = plot("Amplitude", "m/s²", "Shaft speed (rpm)", None, height=230)
        self.orders_plot.setToolTip(ORDERS_TIP)
        add_legend(self.orders_plot.getPlotItem(), offset=(-10, 10))
        self.track_items = [pg.PlotDataItem(name="1X tracked"), pg.PlotDataItem(name="2X tracked")]
        self.steady_items = [pg.PlotDataItem(name="1X steady"), pg.PlotDataItem(name="2X steady")]
        for item in (*self.steady_items, *self.track_items):
            self.orders_plot.addItem(item)
        self.orders_note = note_label()

        # --- right: sidebands
        self.side_plot = plot("Amplitude", "m/s²", "Frequency", "Hz", log_y=True, height=230)
        self.side_plot.setToolTip(SIDEBAND_TIP)
        self.side_item = pg.PlotDataItem()
        self.side_dots = marker(9)
        for item in (self.side_item, self.side_dots):
            self.side_plot.addItem(item)
        self.side_note = note_label()

        # --- right: envelope
        self.band_plot = plot("Band-passed", "m/s²", "Time", "s", height=120)
        self.band_item, self.env_item = pg.PlotDataItem(), pg.PlotDataItem()
        for item in (self.band_item, self.env_item):
            self.band_plot.addItem(item)
        self.env_plot = plot("Envelope spectrum", "m/s²", "Frequency", "Hz", height=170)
        self.defect_lines = [pg.InfiniteLine(angle=90) for _ in range(4)]
        self.env_spec = pg.PlotDataItem()
        self.env_dots = marker(9)
        for item in (*self.defect_lines, self.env_spec, self.env_dots):
            self.env_plot.addItem(item)
        for p in (self.band_plot, self.env_plot):
            p.setToolTip(ENVELOPE_TIP)
        self.env_note = note_label()

        # --- right: cepstrum
        self.cep_plot = plot("Cepstrum", None, "Quefrency", "s", height=230)
        self.cep_plot.setToolTip(CEPSTRUM_TIP)
        self.rahmonic_lines = [pg.InfiniteLine(angle=90) for _ in range(3)]
        self.cep_mesh_line = pg.InfiniteLine(angle=90)
        self.cep_item = pg.PlotDataItem()
        for item in (*self.rahmonic_lines, self.cep_mesh_line, self.cep_item):
            self.cep_plot.addItem(item)
        self.cep_note = note_label()

        self.notes = theory(NOTES_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(column(self.freq_map, self.order_map_plot, self.map_note, stretch=False), "Run-up")
        self.tabs.addTab(column(self.orders_plot, self.orders_note, stretch=False), "Orders")
        self.tabs.addTab(column(self.side_plot, self.side_note, stretch=False), "Sidebands")
        self.tabs.addTab(column(self.band_plot, self.env_plot, self.env_note, stretch=False), "Envelope")
        self.tabs.addTab(column(self.cep_plot, self.cep_note, stretch=False), "Cepstrum")
        self.tabs.addTab(self.notes, "Theory")
        self.tabs.currentChanged.connect(lambda _: (self._draw_tab(), settle_plots(self)))

        three_panes(self, column(self.trials, machine, geometry, run),
                    stack((self.time_plot, 1), (self.spec_plot, 2), (self.readout, 0)), self.tabs)
        self._run_key = None
        self.apply_theme()

    # ------------------------------------------------------------- inputs
    def load_trial(self, index: int) -> None:
        t = TRIALS[index]
        for box, value in ((self.rpm, t.rpm), (self.teeth, 27), (self.gear, GEAR_FAULTS.index(t.gear)),
                           (self.severity, 30.0), (self.bearing_fault, BEARING_FAULTS.index(t.bearing)),
                           (self.impact, 3.0), (self.resonance, 3000.0), (self.noise, t.noise), (self.balls, 9),
                           (self.ball, 7.94), (self.pitch, 39.04), (self.contact, 0.0), (self.low, 2550.0),
                           (self.high, 3450.0), (self.run_from, t.run_from), (self.run_to, t.run_to),
                           (self.run_s, t.run_s), (self.mode, 40.0)):
            set_quietly(box, value)
        self.trials.show_note(t.note)
        self.tabs.setCurrentIndex(t.tab)
        self.redraw()

    @property
    def machine(self) -> Machine:
        bearing = Bearing(int(self.balls.value()), self.ball.value() / 1e3, self.pitch.value() / 1e3,
                          self.contact.value())
        return Machine(self.rpm.value() / 60, int(self.teeth.value()), GEAR_FAULTS[self.gear.currentIndex()],
                       self.severity.value() / 100, BEARING_FAULTS[self.bearing_fault.currentIndex()],
                       self.impact.value(), bearing, self.resonance.value(), noise=self.noise.value())

    @property
    def run(self) -> RunUp:
        lo, hi = sorted((self.run_from.value(), self.run_to.value()))
        return RunUp(lo / 60, max(hi, lo + 60) / 60, self.run_s.value(), self.mode.value())

    @property
    def band(self) -> tuple[float, float]:
        lo, hi = sorted((self.low.value(), self.high.value()))
        nyquist = 0.45 * self.rec.fs
        return min(lo, nyquist - 100), min(max(hi, lo + 50), nyquist)

    def _on_run(self) -> None:
        self._draw_tab()
        settle_plots(self)

    # ------------------------------------------------------------- drawing
    def apply_theme(self) -> None:
        c = colors
        dotted = QtCore.Qt.PenStyle.DotLine
        self.time_item.setPen(pg.mkPen(c.mass[0], width=1))
        self.spec_item.setPen(pg.mkPen(c.mass[0], width=1.2))
        for line in (*self.mesh_lines, *self.defect_lines, *self.rahmonic_lines):
            line.setPen(pg.mkPen(c.fit, width=1, style=dotted))
        self.cep_mesh_line.setPen(pg.mkPen(c.grey, width=1, style=QtCore.Qt.PenStyle.DashLine))
        self.mode_line.setPen(pg.mkPen(c.strong, width=1, style=dotted))
        lut = pg.ColorMap([0.0, 0.5, 1.0], [pg.mkColor(c.background), pg.mkColor(c.mode[1]),
                                            pg.mkColor(c.force)]).getLookupTable(nPts=256)
        for image in (self.freq_image, self.order_image):
            image.setLookupTable(lut)
        for item, color in zip(self.track_items, (c.mass[0], c.mass[1])):
            item.setPen(pg.mkPen(color, width=2))
        for item, color in zip(self.steady_items, (c.mass[0], c.mass[1])):
            item.setPen(pg.mkPen(color, width=1.5, style=QtCore.Qt.PenStyle.DashLine))
        self.side_item.setPen(pg.mkPen(c.mass[0], width=1.2))
        mark(self.side_dots, c.force)
        self.band_item.setPen(pg.mkPen(c.grey, width=1))
        self.env_item.setPen(pg.mkPen(c.mass[0], width=1.5))
        self.env_spec.setPen(pg.mkPen(c.mass[0], width=1.2))
        mark(self.env_dots, c.force)
        self.cep_item.setPen(pg.mkPen(c.mass[0], width=1.2))
        self.redraw()

    def redraw(self) -> None:
        m = self.machine
        self.rec = rec = record(m)
        f = m.bearing.frequencies(m.speed)
        self.bearing_info.setText(
            f"FTF {f['FTF']:.4g} Hz, BPFO {f['BPFO']:.4g} Hz, BPFI {f['BPFI']:.4g} Hz, BSF {f['BSF']:.4g} Hz "
            f"({f['BPFO'] / m.speed:.4g}, {f['BPFI'] / m.speed:.4g} and {f['BSF'] / m.speed:.4g} × the shaft).")
        n = int(SHOW_S * rec.fs)
        self.time_item.setData(rec.t[:n], rec.x[:n])
        self.time_plot.setXRange(0.0, SHOW_S, padding=0)
        self.spec = spec = amplitude_spectrum(rec.x, rec.fs)
        keep = (spec.freqs > 0) & (spec.freqs <= TOP_HZ)
        self.spec_item.setData(spec.freqs[keep], np.maximum(spec.amplitude[keep], 1e-6))
        for h, line in enumerate(self.mesh_lines, start=1):
            line.setPos(h * m.mesh_hz)
        self.spec_plot.setXRange(0.0, TOP_HZ, padding=0)
        top = float(spec.amplitude[keep].max())
        self.spec_plot.setYRange(math.log10(top) - 3.5, math.log10(top) + 0.2, padding=0)
        low, high = self.band
        self.readout.setText(
            f"Shaft {m.speed:.4g} Hz, mesh <b>{m.mesh_hz:.5g} Hz</b>. Kurtosis: the record "
            f"<b>{kurtosis(rec.x):.2f}</b>, the band {low:.0f}–{high:.0f} Hz "
            f"<b>{kurtosis(band_pass(rec.x, rec.fs, (low, high))):.2f}</b> (3 for Gaussian noise). RMS "
            f"{float(np.std(rec.x)):.3g} m/s².")
        self._draw_tab()
        settle_plots(self)

    def _draw_tab(self) -> None:
        i = self.tabs.currentIndex()
        if i in (TAB_RUNUP, TAB_ORDERS):
            self._draw_run(i)
        elif i == TAB_SIDEBANDS:
            self._draw_sidebands()
        elif i == TAB_ENVELOPE:
            self._draw_envelope()
        elif i == TAB_CEPSTRUM:
            self._draw_cepstrum()

    def _run_maps(self):
        run = self.run
        if self._run_key != run:
            rec = run_up(run)
            self._run = (run, rec, frequency_map(rec), order_map(rec))
            self._run_key = run
        return self._run

    def _draw_run(self, tab: int) -> None:
        run, _rec, fm, om = self._run_maps()
        rpm = 60 * om.speed
        if tab == TAB_RUNUP:
            for image, m, top in ((self.freq_image, fm, MAP_TOP_HZ), (self.order_image, om, MAP_TOP_ORDER)):
                cols = m.axis <= top
                # the order blocks are equal in angle, not in time: take the block nearest each of equally
                # spaced speeds, so the image's columns are
                rows = np.rint(np.interp(np.linspace(m.speed[0], m.speed[-1], m.speed.size), m.speed,
                                         np.arange(m.speed.size))).astype(int)
                level = 20 * np.log10(np.maximum(m.amplitude[rows][:, cols].T, 1e-9))
                peak = float(level.max())
                image.setImage(level, autoLevels=False, levels=(peak - MAP_DB, peak))
                s0, s1 = 60 * float(m.speed[0]), 60 * float(m.speed[-1])
                da = m.axis[1] - m.axis[0]
                image.setRect(QtCore.QRectF(s0, -da / 2, max(s1 - s0, 1.0), da * cols.sum()))
            self.mode_line.setPos(run.mode)
            self.freq_map.setXRange(60 * run.start, 60 * run.end, padding=0)
            self.freq_map.setYRange(0.0, MAP_TOP_HZ, padding=0)
            self.order_map_plot.setYRange(0.0, MAP_TOP_ORDER, padding=0)
            self.map_note.setText(
                f"{60 * run.start:.0f} to {60 * run.end:.0f} rpm in {run.duration:.3g} s; the casing's mode at "
                f"{run.mode:.3g} Hz (dotted). Colour: dB, {MAP_DB:.0f} dB below the peak to the peak.")
            return
        fine = np.linspace(run.start, run.end, 2000)
        tracks = [order_track(om, o) for o in (1, 2)]
        for o, (item, steady) in enumerate(zip(self.track_items, self.steady_items), start=1):
            item.setData(rpm, tracks[o - 1])
            steady.setData(60 * fine, run.steady(o, fine))
        self.orders_plot.setXRange(60 * run.start, 60 * run.end, padding=0)
        i = int(np.argmax(tracks[0]))
        s = run.steady(1, fine)
        j = int(np.argmax(s))
        k = int(np.argmax(tracks[1]))
        self.orders_note.setText(
            f"The 1X peaks at <b>{tracks[0][i]:.3g} m/s²</b> at {rpm[i]:.0f} rpm; its steady state at "
            f"{s[j]:.3g} m/s² at {60 * fine[j]:.0f} rpm. The 2X peaks at {rpm[k]:.0f} rpm. Order blocks of "
            f"8 revolutions, half overlapping.")

    def _draw_sidebands(self) -> None:
        m, spec = self.machine, self.spec
        lo, hi = m.mesh_hz - (SIDEBANDS + 1.5) * m.speed, m.mesh_hz + (SIDEBANDS + 1.5) * m.speed
        keep = (spec.freqs >= lo) & (spec.freqs <= hi)
        self.side_item.setData(spec.freqs[keep], np.maximum(spec.amplitude[keep], 1e-6))
        k, a = sidebands(spec.freqs, spec.amplitude, m.mesh_hz, m.speed, SIDEBANDS)
        self.side_dots.setData(m.mesh_hz + k * m.speed, np.maximum(a, 1e-6))
        self.side_plot.setXRange(lo, hi, padding=0)
        top = float(spec.amplitude[keep].max())
        self.side_plot.setYRange(math.log10(top) - 3.0, math.log10(top) + 0.2, padding=0)
        floor = float(np.median(spec.amplitude[keep]))
        pairs = ", ".join(f"±{j}: {a[SIDEBANDS - j]:.2g}/{a[SIDEBANDS + j]:.2g}" for j in (1, 2, 3))
        self.side_note.setText(
            f"The mesh at {m.mesh_hz:.5g} Hz reads {a[SIDEBANDS]:.3g} m/s²; its sidebands {m.speed:.4g} Hz apart "
            f"(lower/upper, m/s²) {pairs}; the floor here is about {floor:.2g}.")

    def _draw_envelope(self) -> None:
        m, rec = self.machine, self.rec
        low, high = self.band
        n = int(SHOW_S * rec.fs)
        y = band_pass(rec.x, rec.fs, (low, high))
        e = envelope(rec.x, rec.fs, (low, high))
        self.band_item.setData(rec.t[:n], y[:n])
        self.env_item.setData(rec.t[:n], e[:n] + float(np.mean(np.abs(scipy.signal.hilbert(y)))))
        self.band_plot.setXRange(0.0, SHOW_S, padding=0)
        es = envelope_spectrum(rec.x, rec.fs, (low, high))
        keep = es.freqs <= ENVELOPE_TOP_HZ
        self.env_spec.setData(es.freqs[keep], es.amplitude[keep])
        self.env_plot.setXRange(0.0, ENVELOPE_TOP_HZ, padding=0)
        d, fm = m.defect_hz(), m.modulation_hz()
        for h, line in enumerate(self.defect_lines, start=1):
            line.setVisible(d > 0)
            line.setPos(h * d)
        if d > 0 and fm > 0:
            x = np.array([d - fm, d + fm])
            self.env_dots.setData(x, [line_amplitude(es.freqs, es.amplitude, v) for v in x])
        else:
            self.env_dots.setData([], [])
        if d > 0:
            parts = [f"{line_amplitude(es.freqs, es.amplitude, h * d):.2g}" for h in (1, 2, 3)]
            text = (f"The strikes come at {d:.4g} Hz; the envelope spectrum reads " + ", ".join(parts)
                    + f" m/s² at its first three harmonics, the plain spectrum "
                    f"{line_amplitude(self.spec.freqs, self.spec.amplitude, d):.2g} at the first.")
            if fm > 0:
                text += f" Sidebands (dots) at ± {fm:.3g} Hz."
        else:
            text = "No bearing defect: the envelope is the band's noise, and the mesh harmonic's in the band."
        self.env_note.setText(text + f" Band {low:.0f}–{high:.0f} Hz.")

    def _draw_cepstrum(self) -> None:
        m, rec = self.machine, self.rec
        q, c = cepstrum(rec.x, rec.fs)
        keep = (q > 0.0015) & (q <= CEPSTRUM_TOP_S)
        self.cep_item.setData(q[keep], c[keep])
        for h, line in enumerate(self.rahmonic_lines, start=1):
            line.setPos(h / m.speed)
        self.cep_mesh_line.setPos(1 / m.mesh_hz)
        self.cep_plot.setXRange(0.0, CEPSTRUM_TOP_S, padding=0)
        i = round(rec.fs / m.speed)
        peak = float(c[max(i - 2, 0):i + 3].max())
        self.cep_note.setText(
            f"At 1/{m.speed:.4g} Hz = {1e3 / m.speed:.3g} ms the cepstrum reaches <b>{peak:.2g}</b>; the rest "
            f"of the range is about {float(np.median(np.abs(c[keep]))):.1g}. The mesh's harmonics sit at "
            f"{1e3 / m.mesh_hz:.3g} ms (dashed).")
