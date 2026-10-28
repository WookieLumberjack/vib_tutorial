"""Quarter-car page: a suspension on a rough road, and the conflict between comfort, grip and travel.

Left: the road (ISO 8608 class and speed) and the car (body and wheel masses, spring, damper, tyre). Centre: the
body's acceleration and the tyre's dynamic load as PSDs, with their RMS values. Right: the comfort–grip trade-off
over springs and dampers, each output against the damper, a drive over a synthesized road, and the Theory. The
page has its own model (core/quarter_car.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.quarter_car import (
    CLASS_NAMES,
    CLASSES,
    Car,
    Output,
    Road,
    drive,
    response,
    tradeoff,
)
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

TAB_TRADEOFF, TAB_DAMPER, TAB_DRIVE = range(3)
DAMPERS = np.geomspace(150.0, 10e3, 48)  # N·s/m
SPRING_FACTORS = (0.5, 1.0, 2.0, 4.0)  # the trade-off's springs, × the car's
DRIVE_S = 60.0  # s of road driven; the plot shows the first WINDOW_S
WINDOW_S = 8.0


@dataclass(frozen=True)
class CarTrial:
    name: str
    note: str
    grade: str = "C"
    speed_kmh: float = 72.0
    sprung: float = 300.0
    unsprung: float = 40.0
    spring: float = 20.0  # kN/m
    damper: float = 1500.0  # N·s/m
    tyre: float = 200.0  # kN/m
    tab: int = TAB_TRADEOFF


TRIALS = (
    CarTrial(
        "A car on an average road",
        "A quarter of a family car (300 kg of body on 20 kN/m and 1.5 kN·s/m, a 40 kg wheel on a 200 kN/m tyre) "
        "at 72 km/h on a class C road. The body bounces at 1.24 Hz and the wheel hops at 11.8 Hz. The body's "
        "acceleration is 1.48 m/s² RMS, the suspension travels 10.7 mm RMS, and the tyre's load swings by 24 % "
        "of its static load."),
    CarTrial(
        "Rougher and faster",
        "Each ISO class is four times the last's PSD, so twice its RMS, and the road's velocity is white with a "
        "PSD ∝ v, so the RMS grows as √v. Class D at 108 km/h: 3.63 m/s², and the tyre's load swings by 59 % of "
        "its static load RMS: it leaves the road 4.4 % of the time.",
        grade="D", speed_kmh=108.0),
    CarTrial(
        "Soft damper: comfort",
        "The damper at 830 N·s/m, the best for comfort on this spring (ζ = 0.17 on the body): 1.36 m/s², 8 % "
        "better. But the wheel hop is now lightly damped: the tyre's load swings by 29 %, and the suspension "
        "travels 14.4 mm.",
        damper=830.0, tab=TAB_DAMPER),
    CarTrial(
        "Firm damper: grip",
        "The damper at 2.3 kN·s/m, the best for the tyre's grip (ζ = 0.48): its load swings by 23 %, while the "
        "body's acceleration rises to 1.72 m/s². No damper is best for both: the classic conflict.",
        damper=2340.0, tab=TAB_DAMPER),
    CarTrial(
        "A softer spring",
        "The spring halved to 10 kN/m: the body bounces at 0.90 Hz and its acceleration falls to 1.35 m/s², "
        "9 % lower at the same damper (the Trade-off tab's curves move left as the spring softens). The price is "
        "sag: the body sits 294 mm down on its spring, twice as far, so a soft car needs room to move and "
        "springs that level themselves, as luxury cars' air springs do.",
        spring=10.0),
    CarTrial(
        "A heavier wheel",
        "The unsprung mass up from 40 to 60 kg (heavier wheels, a solid axle): the wheel hops at 9.6 Hz, and "
        "the tyre's load swings by 26 % instead of 24 %. Unsprung mass costs grip, which is why racing cars "
        "chase light wheels and inboard brakes.",
        unsprung=60.0),
    CarTrial(
        "A drive to check the spectra",
        "A minute of class C road at 72 km/h, synthesized from the PSD, and the car stepped over it exactly "
        "(Drive tab). The record's RMS values come within a percent or two of the PSDs' areas: the frequency "
        "domain's answer is what the car feels.",
        tab=TAB_DRIVE),
)

TIPS = {
    "grade": "<p>ISO 8608's road class: A (very good) to H (very poor), each with four times the last's "
             "displacement PSD.</p>",
    "speed": "<p>The car's speed: the road's spatial PSD becomes a PSD in time.</p>",
    "sprung": "<p>The body's share on this wheel.</p>",
    "unsprung": "<p>The wheel, tyre, brake and part of the axle and suspension.</p>",
    "spring": "<p>The suspension spring (at the wheel).</p>",
    "damper": "<p>The suspension damper (at the wheel).</p>",
    "tyre": "<p>The tyre's vertical stiffness.</p>",
}
PSD_TIP = ("<p>The output's PSD over the road's: |H(f)|² G<sub>z</sub>(f). Dotted: the body-bounce and wheel-hop "
           "frequencies.</p>")
TRADE_TIP = ("<p>The RMS body acceleration (comfort) against the tyre load's RMS over its static load (grip): "
             "a curve per spring, the damper rising along it. Lower left is better.</p>")
DAMPER_TIP = "<p>Each output's RMS against the damper, over its value at the car's damper.</p>"
DRIVE_TIP = "<p>The road's height (top) and the body's acceleration (bottom) on the synthesized road.</p>"

NOTES_HTML = """
<!--BEFORE-->
<h3><a name="road"></a>A road as a random input</h3>
<p>A road's roughness is a random process in space: its height's PSD G<sub>d</sub>(n), with n in cycles per
metre. ISO 8608 fits measured roads to G<sub>d</sub>(n) = G<sub>d</sub>(n₀)(n/n₀)<sup>−2</sup> with n₀ = 0.1
cycles/m, and grades them by G<sub>d</sub>(n₀): class A (a new motorway) 16 × 10<sup>−6</sup> m³, then four times
more per class, to H. Driven at speed v, a wavelength 1/n passes at f = vn, so the height's PSD in time is
G<sub>z</sub>(f) = G<sub>d</sub>(f/v)/v ∝ v/f²: the road's vertical velocity is white noise, of intensity ∝ v.</p>

<h3><a name="quarter-car"></a>The quarter-car</h3>
<p>One corner of a car: the body m<sub>s</sub> on the suspension (k<sub>s</sub>, c<sub>s</sub>), on the wheel
m<sub>u</sub>, on the tyre k<sub>t</sub>, on the road. It has two modes: <b>body bounce</b>, near
√(k<sub>s</sub>k<sub>t</sub>/(k<sub>s</sub> + k<sub>t</sub>)/m<sub>s</sub>)/2π (1 to 1.5 Hz in a car), and
<b>wheel hop</b>, near √((k<sub>s</sub> + k<sub>t</sub>)/m<sub>u</sub>)/2π (10 to 15 Hz). Each output's PSD is
|H(f)|² G<sub>z</sub>(f), its mean square the area under it: the Random vibration page's H G Hᴴ with one input.
Three outputs matter:</p>
<ul>
<li><b>Body acceleration</b>: ride comfort (ISO 2631 weights it to how people feel it; the page leaves it
unweighted).</li>
<li><b>Dynamic tyre load</b> over the static load: road holding. A tyre can only push, so it leaves the road when
the load swings below −1 × static; for a Gaussian load that happens a fraction Φ(−W/σ) of the time.</li>
<li><b>Suspension travel</b>: the room the suspension needs (three times the RMS to stay off the bump
stops).</li>
</ul>

<h3><a name="conflict"></a>The conflict</h3>
<p>A soft damper isolates the body above its bounce but lets the wheel hop; a firm one controls the wheel but
passes the road to the body. So the best damper for comfort (ζ ≈ 0.2 on the body) is lighter than the best for
grip (ζ ≈ 0.5), and no passive setting is best for both. A softer spring helps comfort at every damper but needs
more travel; a heavier wheel costs grip. Plotting comfort against grip for every setting draws the
<b>conflict diagram</b>: passive suspensions live on its curves, and active and semi-active ones (the Active
control page's skyhook) are how to get below them.</p>

<h3><a name="try-it"></a>Things to try</h3>
<p>Each of these sets up the road and the car (Things to try, top left).</p>
<!--TRIALS-->
<!--TAKEAWAYS-->
"""
NOTES_HTML = NOTES_HTML.replace("<!--BEFORE-->", before(
    ("random_page", "psd", "The PSD and H G Hᴴ"),
    ("isolation_page", "damping-trade-off", "Isolation's damping trade-off"),
)).replace("<!--TRIALS-->", trials_html(TRIALS)).replace("<!--TAKEAWAYS-->", takeaways(
    "A road is a PSD in space; at speed v its vertical velocity is white, of intensity ∝ v.",
    "A quarter-car has body bounce (about 1 Hz) and wheel hop (about 10 Hz); each output's PSD is |H|² G<sub>z</sub>.",
    "The best damper for comfort is lighter than the best for grip: passive suspensions trade one for the other.",
    "Soft springs buy comfort with travel; unsprung mass costs grip.",
))


class QuarterCarPage(QtWidgets.QWidget):
    """Controls on the left, the PSDs in the centre, the trade-off, damper sweep and drive on the right."""

    follows_chain = False  # MainWindow: this page has its own model, not the chain

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.trials = TrialBox([t.name for t in TRIALS],
                               "<p>Each sets up the road and the car to show one idea from the Theory tab.</p>")
        self.trials.picked.connect(self.load_trial)

        road = Form("Road")
        self.grade = combo([f"{c}: G_d(n₀) = {CLASSES[c] * 1e6:g}×10⁻⁶ m³" for c in CLASS_NAMES], TIPS["grade"])
        self.grade.setCurrentIndex(CLASS_NAMES.index("C"))
        road.add("ISO 8608 class:", self.grade, TIPS["grade"])
        self.speed = road.number("Speed:", 5.0, 250.0, 72.0, 0, " km/h", TIPS["speed"])

        car = Form("Car (one corner)")
        self.sprung = car.number("Body mₛ:", 50.0, 2000.0, 300.0, 0, " kg", TIPS["sprung"])
        self.unsprung = car.number("Wheel mᵤ:", 5.0, 300.0, 40.0, 0, " kg", TIPS["unsprung"])
        self.spring = car.number("Spring kₛ:", 2.0, 500.0, 20.0, 1, " kN/m", TIPS["spring"])
        self.damper = car.number("Damper cₛ:", 0.0, 20000.0, 1500.0, 0, " N·s/m", TIPS["damper"])
        self.tyre = car.number("Tyre kₜ:", 20.0, 2000.0, 200.0, 0, " kN/m", TIPS["tyre"])
        self.car_info = note_label()
        car.row(self.car_info)

        connect_all((self.grade, self.speed, self.sprung, self.unsprung, self.spring, self.damper, self.tyre),
                    self.redraw)

        # --- centre
        self.body_plot = plot("Body acceleration PSD", "(m/s²)²/Hz", "Frequency", "Hz", log_x=True, log_y=True,
                              height=150)
        self.tyre_plot = plot("Tyre load PSD", "N²/Hz", "Frequency", "Hz", log_x=True, log_y=True, height=150)
        self.tyre_plot.setXLink(self.body_plot)
        self.body_item, self.tyre_item = pg.PlotDataItem(), pg.PlotDataItem()
        self.mode_lines = [[pg.InfiniteLine(angle=90), pg.InfiniteLine(angle=90)] for _ in range(2)]
        for p, item, lines in ((self.body_plot, self.body_item, self.mode_lines[0]),
                               (self.tyre_plot, self.tyre_item, self.mode_lines[1])):
            p.setToolTip(PSD_TIP)
            for line in lines:
                p.addItem(line)
            p.addItem(item)
        self.readout = readout_label()

        # --- right
        self.trade_plot = plot("Tyre load RMS / static load", None, "Body acceleration RMS", "m/s²", height=200)
        self.trade_plot.setToolTip(TRADE_TIP)
        add_legend(self.trade_plot.getPlotItem(), offset=(-10, 10))
        self.trade_items = [pg.PlotDataItem(name=f"spring × {s:g}") for s in SPRING_FACTORS]
        self.trade_now = marker(13)
        self.trade_best = (marker(10, "t"), marker(10, "s"))
        for item in (*self.trade_items, *self.trade_best, self.trade_now):
            self.trade_plot.addItem(item)
        self.trade_note = note_label()

        self.damper_plot = plot("RMS / at the car's damper", None, "Damper", "N·s/m", log_x=True, height=200)
        self.damper_plot.setToolTip(DAMPER_TIP)
        add_legend(self.damper_plot.getPlotItem(), offset=(-10, 10))
        self.damper_items = {o: pg.PlotDataItem(name=o.value) for o in Output}
        self.damper_now = pg.InfiniteLine(angle=90)
        self.damper_mins = {o: marker(10) for o in (Output.BODY, Output.TYRE)}
        for item in (self.damper_now, *self.damper_items.values(), *self.damper_mins.values()):
            self.damper_plot.addItem(item)
        self.damper_note = note_label()

        self.road_plot = plot("Road height", "m", "Time", "s", height=110)
        self.acc_plot = plot("Body acceleration", "m/s²", "Time", "s", height=150)
        self.acc_plot.setXLink(self.road_plot)
        self.road_item, self.acc_item = pg.PlotDataItem(), pg.PlotDataItem()
        self.road_plot.addItem(self.road_item)
        self.acc_plot.addItem(self.acc_item)
        for p in (self.road_plot, self.acc_plot):
            p.setToolTip(DRIVE_TIP)
        self.drive_note = note_label()

        self.notes = theory(NOTES_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(column(self.trade_plot, self.trade_note, stretch=False), "Trade-off")
        self.tabs.addTab(column(self.damper_plot, self.damper_note, stretch=False), "Damper")
        self.tabs.addTab(column(self.road_plot, self.acc_plot, self.drive_note, stretch=False), "Drive")
        self.tabs.addTab(self.notes, "Theory")
        self.tabs.currentChanged.connect(lambda _: self._draw_tab())

        three_panes(self, column(self.trials, road, car),
                    stack((self.body_plot, 1), (self.tyre_plot, 1), (self.readout, 0)), self.tabs)
        self.apply_theme()

    # ------------------------------------------------------------- inputs
    def load_trial(self, index: int) -> None:
        t = TRIALS[index]
        for box, value in ((self.grade, CLASS_NAMES.index(t.grade)), (self.speed, t.speed_kmh),
                           (self.sprung, t.sprung), (self.unsprung, t.unsprung), (self.spring, t.spring),
                           (self.damper, t.damper), (self.tyre, t.tyre)):
            set_quietly(box, value)
        self.trials.show_note(t.note)
        self.tabs.setCurrentIndex(t.tab)
        self.redraw()

    @property
    def car(self) -> Car:
        return Car(self.sprung.value(), self.unsprung.value(), 1e3 * self.spring.value(), self.damper.value(),
                   1e3 * self.tyre.value())

    @property
    def road(self) -> Road:
        return Road(CLASS_NAMES[self.grade.currentIndex()], self.speed.value() / 3.6)

    # ------------------------------------------------------------- drawing
    def apply_theme(self) -> None:
        c = colors
        self.body_item.setPen(pg.mkPen(c.mass[0], width=2))
        self.tyre_item.setPen(pg.mkPen(c.mass[1], width=2))
        for lines in self.mode_lines:
            for line in lines:
                line.setPen(pg.mkPen(c.fit, width=1, style=QtCore.Qt.PenStyle.DotLine))
        for item, color in zip(self.trade_items, (c.mode[1], c.mass[0], c.mode[2], c.mode[3])):
            item.setPen(pg.mkPen(color, width=2))
        mark(self.trade_now, c.force, filled=True)
        mark(self.trade_best[0], c.mass[0])
        mark(self.trade_best[1], c.mass[1])
        for item, color in zip(self.damper_items.values(), (c.mass[0], c.mode[2], c.mass[1])):
            item.setPen(pg.mkPen(color, width=2))
        mark(self.damper_mins[Output.BODY], c.mass[0], filled=True)
        mark(self.damper_mins[Output.TYRE], c.mass[1], filled=True)
        self.damper_now.setPen(pg.mkPen(c.force, width=1.5, style=QtCore.Qt.PenStyle.DotLine))
        self.road_item.setPen(pg.mkPen(c.grey, width=1.5))
        self.acc_item.setPen(pg.mkPen(c.mass[0], width=1))
        self.redraw()

    def redraw(self) -> None:
        car, road = self.car, self.road
        fb, fw = car.modes()
        self.car_info.setText(f"Body bounce {fb:.3g} Hz, wheel hop {fw:.3g} Hz; the body's ζ on the suspension "
                              f"{car.body_zeta():.2g}.")
        self.result = r = response(car, road)
        self.body_item.setData(r.f, r.psd[Output.BODY])
        self.tyre_item.setData(r.f, r.psd[Output.TYRE])
        for lines in self.mode_lines:
            for line, fn in zip(lines, (fb, fw)):
                line.setPos(math.log10(fn))
        self.body_plot.setXRange(math.log10(r.f[0]), math.log10(r.f[-1]), padding=0)
        for p, out in ((self.body_plot, Output.BODY), (self.tyre_plot, Output.TYRE)):
            top = float(np.max(r.psd[out]))
            p.setYRange(math.log10(top) - 5, math.log10(top) + 0.3, padding=0)
        lift = r.lift_probability
        self.readout.setText(
            f"RMS: body acceleration <b>{r.rms[Output.BODY]:.3g} m/s²</b>, suspension travel "
            f"<b>{fmt(r.rms[Output.TRAVEL], 'm')}</b>, tyre load <b>{fmt(r.rms[Output.TYRE], 'N')}</b> "
            f"({100 * r.load_ratio:.3g} % of the static {fmt(r.weight, 'N')}). The tyre leaves the road "
            + (f"{100 * lift:.2g} % of the time." if lift > 1e-4 else "almost never.")
            + f" Road: class {road.grade} at {road.speed * 3.6:.3g} km/h.")
        self._draw_tab()
        settle_plots(self)

    def _draw_tab(self) -> None:
        i = self.tabs.currentIndex()
        if i == TAB_TRADEOFF:
            self._draw_tradeoff()
        elif i == TAB_DAMPER:
            self._draw_damper()
        elif i == TAB_DRIVE:
            self._draw_drive()

    def _draw_tradeoff(self) -> None:
        car, road, r = self.car, self.road, self.result
        t = tradeoff(car, road, car.spring * np.array(SPRING_FACTORS), DAMPERS)
        for i, item in enumerate(self.trade_items):
            item.setData(t.body[i], t.tyre[i])
        self.trade_now.setData([r.rms[Output.BODY]], [r.load_ratio])
        row = SPRING_FACTORS.index(1.0)
        jb, jt = int(np.argmin(t.body[row])), int(np.argmin(t.tyre[row]))
        self.trade_best[0].setData([t.body[row, jb]], [t.tyre[row, jb]])
        self.trade_best[1].setData([t.body[row, jt]], [t.tyre[row, jt]])
        x = t.body[np.isfinite(t.body)]
        y = t.tyre[np.isfinite(t.tyre)]
        self.trade_plot.setXRange(0.0, min(float(np.percentile(x, 95)), 3 * r.rms[Output.BODY]), padding=0.02)
        self.trade_plot.setYRange(0.0, min(float(np.percentile(y, 95)), 3 * r.load_ratio), padding=0.02)
        self.trade_note.setText(
            f"Dot: this car. On its spring, the most comfortable damper (triangle) is "
            f"{DAMPERS[jb]:.0f} N·s/m and the best for grip (square) {DAMPERS[jt]:.0f} N·s/m. Each curve runs "
            "from a soft damper (top) to a firm one (right).")

    def _draw_damper(self) -> None:
        car, road, r = self.car, self.road, self.result
        t = tradeoff(car, road, np.array([car.spring]), DAMPERS)
        values = {Output.BODY: t.body[0], Output.TRAVEL: t.travel[0], Output.TYRE: t.tyre[0]}
        now = {Output.BODY: r.rms[Output.BODY], Output.TRAVEL: r.rms[Output.TRAVEL], Output.TYRE: r.load_ratio}
        for o, item in self.damper_items.items():
            item.setData(DAMPERS, values[o] / now[o])
        for o, dot in self.damper_mins.items():
            j = int(np.argmin(values[o]))
            dot.setData([DAMPERS[j]], [values[o][j] / now[o]])
        self.damper_now.setPos(math.log10(max(car.damper, DAMPERS[0])))
        self.damper_plot.setXRange(math.log10(DAMPERS[0]), math.log10(DAMPERS[-1]), padding=0)
        self.damper_plot.setYRange(0.0, 3.0, padding=0)
        jb, jt = int(np.argmin(values[Output.BODY])), int(np.argmin(values[Output.TYRE]))
        crit = 2 * math.sqrt(car.spring * car.sprung)
        self.damper_note.setText(
            f"Comfort is best at {DAMPERS[jb]:.0f} N·s/m (ζ = {DAMPERS[jb] / crit:.2g}), grip at {DAMPERS[jt]:.0f} "
            f"N·s/m (ζ = {DAMPERS[jt] / crit:.2g}); travel only falls as the damper firms. Dotted: this car's.")

    def _draw_drive(self) -> None:
        car, road, r = self.car, self.road, self.result
        d = drive(car, road, DRIVE_S)
        n = int(WINDOW_S / (d.t[1] - d.t[0]))
        self.road_item.setData(d.t[:n], d.road[:n])
        self.acc_item.setData(d.t[:n], d.body[:n])
        self.road_plot.setXRange(0.0, WINDOW_S, padding=0)
        skip = int(2.0 / (d.t[1] - d.t[0]))  # let the start settle
        rms = {Output.BODY: float(np.std(d.body[skip:])), Output.TRAVEL: float(np.std(d.travel[skip:])),
               Output.TYRE: float(np.std(d.tyre[skip:]))}
        parts = [f"{o.value.lower()} {100 * (rms[o] / r.rms[o] - 1):+.2g} %" for o in Output]
        self.drive_note.setText(f"{DRIVE_S:.0f} s of road (the first {WINDOW_S:.0f} s drawn). The record's RMS "
                                "against the PSDs': " + ", ".join(parts) + ".")
