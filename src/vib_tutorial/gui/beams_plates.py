"""Beams & plates page: when a beam is too thick for Euler–Bernoulli, and the modes of a plate.

Left: what to look at (a thick beam, a rectangular plate or a circular one), its size and material, and the mode.
Centre: for the beam, each theory's frequencies against Euler–Bernoulli's and the bending wave's speed against its
wavelength; for a plate, the mode's shape with its nodal lines. Right: the frequencies, the number of modes
below f against its asymptote, and the Theory. The page has its own model (core/beams_plates.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.beams_plates import (
    THEORIES,
    Beam,
    Material,
    Plate,
    Theory,
    beam_asymptote,
    beam_frequencies,
    beam_omega,
    circular_modes,
    circular_shape,
    counting,
    plate_asymptote,
    rect_modes,
    rect_shape,
    wave_speeds,
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
    theory,
    three_panes,
    trials_html,
)
from .links import before, takeaways
from .style import colors
from .theming import add_legend

BEAM, RECT, CIRCLE = range(3)
STRUCTURES = ("Thick beam (simply supported)", "Rectangular plate (simply supported)", "Circular plate (clamped)")
TAB_TABLE, TAB_DENSITY = range(2)
MODES = 30  # modes the beam's ratio plot and the mode box span
DENSITY_MODES = 400  # modes counted for the density plot
GRID = 161  # image points across a plate
MIXES = ("alone", "+ its twin", "− its twin")


@dataclass(frozen=True)
class BeamPlateTrial:
    name: str
    note: str
    structure: int = BEAM
    length: float = 1.0  # beam L, plate a, circle radius
    depth: float = 0.1  # beam h
    side: float = 0.4  # plate b
    thickness: float = 3.0  # plate h, mm
    mode: int = 1
    mix: int = 0
    tab: int = TAB_TABLE


TRIALS = (
    BeamPlateTrial(
        "A stocky beam",
        "A steel beam 1 m long and 100 mm deep (L/h = 10), simply supported. Euler–Bernoulli's first mode, 235 Hz, "
        "is 1.7 % too high; by the fifth mode, where a half wave is only twice the depth, it is 33 % too high "
        "(Timoshenko: 0.75 of it). Shear deflection costs more than the sections' rotary inertia (0.78 against "
        "0.91 at mode 5), and the two together cost most.",
        mode=5),
    BeamPlateTrial(
        "A slender beam",
        "The same beam 20 mm deep (L/h = 50): Timoshenko's first mode is 0.07 % below Euler–Bernoulli's, its "
        "tenth 6.0 %. A beam is 'thin' only for wavelengths long against its depth: every beam gets thick at "
        "high enough modes.",
        depth=0.02, mode=10),
    BeamPlateTrial(
        "Short waves stop speeding up",
        "Euler–Bernoulli's bending wave travels at c = √ω (EI/ρA)<sup>¼</sup>, faster without limit as the "
        "wavelength shortens: impossible. With shear and rotary inertia (Timoshenko) the speed levels off below "
        "the shear wave's √(κG/ρ) = 2.9 km/s (centre, bottom). Timoshenko's second branch, thickness shear, "
        "starts near 16 kHz on this beam (√(κGA/ρI)/2π: a cut-off, below which it doesn't propagate).",
        mode=20),
    BeamPlateTrial(
        "A plate's modes",
        "A steel plate 600 × 400 mm, 3 mm thick, simply supported. Its modes are sin(mπx/a) sin(nπy/b), at "
        "ω = π²(m²/a² + n²/b²)√(D/ρh): 66.6 Hz for (1, 1), then 128 Hz (2, 1), 205 Hz (1, 2) and 230 Hz (3, 1). Mode "
        "(3, 2), the sixth, has 2 nodal lines across and 1 along.",
        structure=RECT, length=0.6, side=0.4, mode=6),
    BeamPlateTrial(
        "Twin modes and Chladni's figures",
        "A square plate (400 × 400 mm): (1, 3) and (3, 1) have the same frequency, so any mix of them is a mode "
        "too. Their sum and difference have nodal lines that are neither straight across nor along: the "
        "diagonals and rings of Chladni's sand figures, which he made by bowing a plate's edge.",
        structure=RECT, length=0.4, side=0.4, mode=5, mix=1),
    BeamPlateTrial(
        "A drum-like plate",
        "A clamped circular plate, radius 200 mm, 3 mm thick: λ² = 10.22 for the fundamental (191 Hz), then 21.26 "
        "with one nodal diameter (397 Hz), 34.88 with two (652 Hz) and 39.77 with one nodal circle (743 Hz). A "
        "mode with nodal diameters is a pair (cos nθ and sin nθ) at one frequency.",
        structure=CIRCLE, length=0.2, mode=4),
    BeamPlateTrial(
        "How many modes per hertz",
        "Count the modes below f (Modal density tab). A plate's count grows in proportion to f, so it has as many "
        "modes in every band of a given width: S/2 √(ρh/D) = 0.026 per Hz on the 600 × 400 mm plate, a mode "
        "every 39 Hz. A beam's count grows only as √f, so its modes spread apart. This is why plates and "
        "panels are treated statistically at high frequency.",
        structure=RECT, length=0.6, side=0.4, mode=1, tab=TAB_DENSITY),
)

TIPS = {
    "structure": "<p>A simply supported beam of rectangular section (with shear and rotary inertia), a simply "
                 "supported rectangular plate, or a clamped circular one.</p>",
    "length": "<p>The beam's length, the plate's side a, or the circular plate's radius.</p>",
    "depth": "<p>The beam's depth h (in the plane it bends).</p>",
    "side": "<p>The plate's other side b.</p>",
    "thickness": "<p>The plate's thickness.</p>",
    "E": "<p>Young's modulus.</p>",
    "rho": "<p>Density.</p>",
    "nu": "<p>Poisson's ratio (the shear modulus is E/2(1 + ν)).</p>",
    "mode": "<p>Which mode, counted up from the lowest.</p>",
    "mix": "<p>On a square plate, mode (m, n) and its twin (n, m) have the same frequency: show their sum or "
           "difference instead.</p>",
}
RATIO_TIP = ("<p>Each theory's frequency over Euler–Bernoulli's, mode by mode: rotary inertia (Rayleigh), shear "
             "deflection, and both (Timoshenko).</p>")
SPEED_TIP = ("<p>The bending wave's phase speed against its wavelength over the depth: Euler–Bernoulli's grows "
             "without bound as the wave shortens; Timoshenko's levels off. Dotted: √(E/ρ) and √(κG/ρ).</p>")
SHAPE_TIP = "<p>The mode's shape (colour: up and down) and its nodal lines, where it doesn't move.</p>"
DENSITY_TIP = "<p>The number of modes at or below f (steps) against its asymptote (dashed).</p>"

NOTES_HTML = """
<!--BEFORE-->
<h3><a name="thick-beams"></a>When a beam is thick</h3>
<p>Euler–Bernoulli's beam assumes plane sections stay plane <i>and normal to the axis</i>, and that only the
axis's sideways motion has inertia. Both fail once a wavelength is no longer long against the depth:</p>
<ul>
<li><b>Rotary inertia</b> (Rayleigh): each section also turns, with inertia ρI per length. It adds to the mass,
so it lowers the frequencies: ω² = EIk⁴ / (ρA + ρIk²).</li>
<li><b>Shear deflection</b>: the sections also slide on each other, an extra flexibility κGA (κ = 5/6 for a
rectangle: shear isn't uniform over the section). It lowers the frequencies more, since E/κG ≈ 3.</li>
</ul>
<p><b>Timoshenko's beam</b> keeps both, with the section's rotation ψ a second unknown beside the deflection w:</p>
<p align="center">ρA w<sub>tt</sub> = κGA (w<sub>x</sub> − ψ)<sub>x</sub>,&nbsp;&nbsp;
ρI ψ<sub>tt</sub> = EI ψ<sub>xx</sub> + κGA (w<sub>x</sub> − ψ).</p>
<p>On a simply supported beam the modes are still sin(nπx/L), and each gives a 2 × 2 determinant, a quadratic in
ω²: its lower root is the bending mode, its upper one a <b>thickness-shear</b> mode at very high frequency. The
correction depends on the wavelength over the depth: under 2 % for the first mode of a beam with L/h = 10, a
quarter by the fifth.</p>

<h3><a name="dispersion-limit"></a>Waves that can't outrun sound</h3>
<p>Euler–Bernoulli's bending wave has ω = √(EI/ρA) k², so its speed ω/k grows without limit as k does. A real
beam can't carry a disturbance faster than its material's waves, and Timoshenko's doesn't: its bending branch
levels off near the shear wave's speed √(κG/ρ). Wherever a beam is driven with wavelengths near its depth (impact,
high-frequency machinery, ultrasonics), Timoshenko is the model to use.</p>

<h3><a name="plates"></a>Plates</h3>
<p>A plate bends in two directions. With D = Eh³/12(1 − ν²) its bending stiffness and ρh its mass per area,</p>
<p align="center">D ∇⁴w + ρh w<sub>tt</sub> = 0.</p>
<p>A simply supported rectangle a × b has the modes sin(mπx/a) sin(nπy/b), at
ω<sub>mn</sub> = π²(m²/a² + n²/b²)√(D/ρh); mode (m, n) has m − 1 nodal lines across a and n − 1 across b. A
clamped circle of radius a has J<sub>n</sub>(λr/a) − (J<sub>n</sub>(λ)/I<sub>n</sub>(λ)) I<sub>n</sub>(λr/a)
times cos nθ, at ω = λ²√(D/ρh)/a², with n nodal diameters and a nodal circle for each root past the first.
When two modes share a frequency (a square plate's (m, n) and (n, m), a circle's cos nθ and sin nθ), every mix
of them is a mode too: Chladni's sand figures are such mixes on free plates.</p>

<h3><a name="modal-density"></a>Modal density</h3>
<p>Count the modes below f. A plate's count grows as N ≈ (S/4π) ω √(ρh/D), in proportion to f, so it has a
constant modal density, S/2 √(ρh/D) modes per hertz, whatever its shape. A beam's grows as √f (its modes spread
apart), a room's or a solid's as f² or f³ (they crowd together). At high frequencies, with many modes in every
band, mode-by-mode analysis gives way to energy and statistics (statistical energy analysis).</p>

<h3><a name="try-it"></a>Things to try</h3>
<p>Each of these sets up the structure and the plots (Things to try, top left).</p>
<!--TRIALS-->
<!--TAKEAWAYS-->
"""
NOTES_HTML = NOTES_HTML.replace("<!--BEFORE-->", before(
    ("continuum_page", "continuum-waves", "The continuum's waves"),
    ("energy_page", "rayleigh", "Rayleigh's quotient"),
)).replace("<!--TRIALS-->", trials_html(TRIALS)).replace("<!--TAKEAWAYS-->", takeaways(
    "Shear deflection and rotary inertia lower a beam's frequencies once a wavelength nears the depth; "
    "Timoshenko's beam keeps both.",
    "Euler–Bernoulli's bending waves speed up without limit; Timoshenko's level off below the shear wave's speed.",
    "A plate's modes have nodal lines in two directions; modes sharing a frequency mix into Chladni's figures.",
    "A plate has a constant modal density, S/2 √(ρh/D) per hertz; a beam's thins out as 1/√f.",
))


class BeamsPlatesPage(QtWidgets.QWidget):
    """Controls on the left, the beam's corrections or the plate's mode in the centre, the frequencies and the
    modal density on the right."""

    follows_chain = False  # MainWindow: this page has its own model, not the chain

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.trials = TrialBox([t.name for t in TRIALS],
                               "<p>Each sets up the structure to show one idea from the Theory tab.</p>")
        self.trials.picked.connect(self.load_trial)

        shape = Form("Structure")
        self.structure = combo(STRUCTURES, TIPS["structure"])
        shape.add("Structure:", self.structure, TIPS["structure"])
        self.length = shape.number("Length / side a:", 0.01, 100.0, 1.0, 3, " m", TIPS["length"])
        self.depth = shape.number("Beam depth h:", 0.001, 10.0, 0.1, 3, " m", TIPS["depth"])
        self.side = shape.number("Plate side b:", 0.01, 100.0, 0.4, 3, " m", TIPS["side"])
        self.thickness = shape.number("Plate thickness:", 0.1, 1000.0, 3.0, 2, " mm", TIPS["thickness"])
        self.info = note_label()
        shape.row(self.info)

        material = Form("Material (steel)")
        self.E = material.number("Young's modulus E:", 0.01, 2000.0, 210.0, 2, " GPa", TIPS["E"])
        self.rho = material.number("Density ρ:", 1.0, 30000.0, 7850.0, 0, " kg/m³", TIPS["rho"])
        self.nu = material.number("Poisson's ratio ν:", 0.0, 0.49, 0.3, 2, "", TIPS["nu"])

        pick = Form("Mode")
        self.mode = pick.number("Mode:", 1, MODES, 1, 0, "", TIPS["mode"])
        self.mix = combo(MIXES, TIPS["mix"])
        pick.add("Square plate:", self.mix, TIPS["mix"])

        connect_all((self.structure, self.length, self.depth, self.side, self.thickness, self.E, self.rho, self.nu,
                     self.mode, self.mix), self._on_inputs)

        # --- centre: the beam
        self.ratio_plot = plot("f / f<sub>Euler–Bernoulli</sub>", None, "Mode n", None, height=150)
        self.ratio_plot.setToolTip(RATIO_TIP)
        add_legend(self.ratio_plot.getPlotItem(), offset=(-10, 10))
        self.ratio_items = {th: pg.PlotDataItem(name=th.value, symbol="o", symbolSize=5)
                            for th in THEORIES[1:]}
        self.ratio_now = pg.InfiniteLine(angle=90)
        self.ratio_plot.addItem(self.ratio_now)
        for item in self.ratio_items.values():
            self.ratio_plot.addItem(item)
        self.speed_plot = plot("Phase speed", "m/s", "Wavelength / depth", None, log_x=True, height=150)
        self.speed_plot.setToolTip(SPEED_TIP)
        add_legend(self.speed_plot.getPlotItem(), offset=(-10, 10))
        self.speed_items = {th: pg.PlotDataItem(name=th.value) for th in (Theory.EULER, Theory.TIMOSHENKO)}
        self.speed_lines = [pg.InfiniteLine(angle=0), pg.InfiniteLine(angle=0)]
        self.speed_now = marker(11)
        for item in (*self.speed_lines, *self.speed_items.values(), self.speed_now):
            self.speed_plot.addItem(item)
        beam_view = column(self.ratio_plot, self.speed_plot, stretch=False)
        beam_view.layout().setContentsMargins(0, 0, 0, 0)

        # --- centre: a plate
        self.shape_plot = plot("y", "m", "x", "m", height=250)
        self.shape_plot.setToolTip(SHAPE_TIP)
        self.shape_plot.setAspectLocked(True)
        self.shape_plot.setMouseEnabled(x=False, y=False)
        self.shape_image = pg.ImageItem(axisOrder="row-major")
        self.nodal = pg.IsocurveItem(level=0.0)
        self.nodal.setParentItem(self.shape_image)
        self.outline = pg.PlotDataItem()
        for item in (self.shape_image, self.outline):
            self.shape_plot.addItem(item)

        self.centre = QtWidgets.QStackedWidget()
        self.centre.addWidget(beam_view)
        self.centre.addWidget(self.shape_plot)
        self.readout = readout_label()
        centre = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(centre)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.addWidget(self.centre, 1)
        cv.addWidget(self.readout)

        # --- right
        self.table = QtWidgets.QTextBrowser()
        self.density_plot = plot("Modes at or below f", None, "Frequency", "Hz", height=200)
        self.density_plot.setToolTip(DENSITY_TIP)
        add_legend(self.density_plot.getPlotItem(), offset=(10, 10))
        self.count_item = pg.PlotDataItem(name="counted", stepMode="right")
        self.asym_item = pg.PlotDataItem(name="asymptote")
        for item in (self.asym_item, self.count_item):
            self.density_plot.addItem(item)
        self.density_note = note_label()

        self.notes = theory(NOTES_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.table, "Frequencies")
        self.tabs.addTab(column(self.density_plot, self.density_note, stretch=False), "Modal density")
        self.tabs.addTab(self.notes, "Theory")
        self.tabs.currentChanged.connect(lambda _: self._draw_tab())

        three_panes(self, column(self.trials, shape, material, pick), centre, self.tabs)
        self._on_inputs()
        self.apply_theme()

    # ------------------------------------------------------------- inputs
    def load_trial(self, index: int) -> None:
        t = TRIALS[index]
        for box, value in ((self.structure, t.structure), (self.length, t.length), (self.depth, t.depth),
                           (self.side, t.side), (self.thickness, t.thickness), (self.E, 210.0),
                           (self.rho, 7850.0), (self.nu, 0.3), (self.mode, t.mode), (self.mix, t.mix)):
            set_quietly(box, value)
        self.trials.show_note(t.note)
        self.tabs.setCurrentIndex(t.tab)
        self._on_inputs()

    @property
    def material(self) -> Material:
        return Material(self.E.value() * 1e9, self.nu.value(), self.rho.value())

    def _on_inputs(self) -> None:
        s = self.structure.currentIndex()
        self.depth.setEnabled(s == BEAM)
        self.side.setEnabled(s == RECT)
        self.thickness.setEnabled(s != BEAM)
        self.beam = Beam(self.length.value(), self.depth.value(), 0.05, self.material)
        self.plate = Plate(self.length.value(), self.side.value() if s == RECT else self.length.value(),
                           self.thickness.value() / 1000, self.material, circular=s == CIRCLE)
        self.centre.setCurrentIndex(0 if s == BEAM else 1)
        if s == BEAM:
            b = self.beam
            self.info.setText(f"L/h = {b.length / b.depth:.3g}; slenderness L/r = {b.slenderness:.3g}.")
            self.mix.setEnabled(False)
        else:
            p = self.plate
            self.info.setText(f"D = {p.rigidity:.4g} N·m, ρh = {p.surface_mass:.4g} kg/m².")
            mode = self._plate_modes()[int(self.mode.value()) - 1]
            square = s == RECT and math.isclose(p.a, p.b, rel_tol=1e-9) and mode.m != mode.n
            self.mix.setEnabled(square)
        self.redraw()

    def _plate_modes(self, count: int = MODES):
        return circular_modes(self.plate, count) if self.plate.circular else rect_modes(self.plate, count)

    # ------------------------------------------------------------- drawing
    def apply_theme(self) -> None:
        c = colors
        for item, color in zip(self.ratio_items.values(), (c.mode[1], c.mode[2], c.mass[0])):
            item.setPen(pg.mkPen(color, width=2))
            item.setSymbolBrush(pg.mkBrush(color))
            item.setSymbolPen(None)
        self.ratio_now.setPen(pg.mkPen(c.fit, width=1, style=QtCore.Qt.PenStyle.DotLine))
        self.speed_items[Theory.EULER].setPen(pg.mkPen(c.grey, width=2, style=QtCore.Qt.PenStyle.DashLine))
        self.speed_items[Theory.TIMOSHENKO].setPen(pg.mkPen(c.mass[0], width=2))
        for line in self.speed_lines:
            line.setPen(pg.mkPen(c.faint, width=1, style=QtCore.Qt.PenStyle.DotLine))
        mark(self.speed_now, c.strong, filled=True)
        self.shape_image.setLookupTable(pg.ColorMap([0.0, 0.5, 1.0], [
            pg.mkColor(c.mass[0]), pg.mkColor(c.background), pg.mkColor(c.force)]).getLookupTable(nPts=256))
        self.nodal.setPen(pg.mkPen(c.strong, width=2))
        self.outline.setPen(pg.mkPen(c.structure, width=2))
        self.count_item.setPen(pg.mkPen(c.mass[0], width=2))
        self.asym_item.setPen(pg.mkPen(c.force, width=2, style=QtCore.Qt.PenStyle.DashLine))
        self.redraw()

    def redraw(self) -> None:
        if self.structure.currentIndex() == BEAM:
            self._draw_beam()
        else:
            self._draw_plate()
        self._draw_tab()
        settle_plots(self)

    def _draw_beam(self) -> None:
        b, n = self.beam, int(self.mode.value())
        modes = np.arange(1, MODES + 1)
        euler = beam_frequencies(b, MODES, Theory.EULER)
        ratios = {th: beam_frequencies(b, MODES, th) / euler for th in THEORIES[1:]}
        for th, item in self.ratio_items.items():
            item.setData(modes, ratios[th])
        self.ratio_now.setPos(n)
        self.ratio_plot.setXRange(0.5, MODES + 0.5, padding=0)
        self.ratio_plot.setYRange(min(0.95 * float(ratios[Theory.TIMOSHENKO].min()), 0.95), 1.01, padding=0)

        wl = np.geomspace(0.1, 1000.0, 400)  # wavelength over depth
        k = 2 * math.pi / (wl * b.depth)
        for th, item in self.speed_items.items():
            item.setData(wl, beam_omega(b, k, th) / k)
        bar, shear = wave_speeds(b)
        for line, v in zip(self.speed_lines, (bar, shear)):
            line.setPos(v)
        kn = n * math.pi / b.length
        self.speed_now.setData([2 * b.length / n / b.depth], [float(beam_omega(b, kn)) / kn])
        self.speed_plot.setXRange(-1, 3, padding=0)
        self.speed_plot.setYRange(0.0, 1.15 * bar, padding=0)

        fe, ft = float(euler[n - 1]), float(beam_frequencies(b, n)[n - 1])
        thick = float(beam_omega(b, kn, branch=1)) / (2 * math.pi)
        self.readout.setText(
            f"Mode {n}: half wavelength {fmt(b.length / n, 'm')}, {b.length / n / b.depth:.3g} depths. "
            f"Euler–Bernoulli {fmt(fe, 'Hz')}, Timoshenko {fmt(ft, 'Hz')} ({100 * (ft / fe - 1):+.3g} %); "
            f"Timoshenko's thickness-shear branch at this n: {fmt(thick, 'Hz')}.")

    def _draw_plate(self) -> None:
        p = self.plate
        modes = self._plate_modes()
        mode = modes[int(self.mode.value()) - 1]
        if p.circular:
            x = np.linspace(-1.0, 1.0, GRID)
            w = circular_shape(mode, x, x)
            rect = QtCore.QRectF(-p.a, -p.a, 2 * p.a, 2 * p.a)
            th = np.linspace(0, 2 * math.pi, 300)
            self.outline.setData(p.a * np.cos(th), p.a * np.sin(th))
            what = (f"({mode.m}, {mode.n}): {mode.m} nodal diameter{'s' if mode.m != 1 else ''}, {mode.n - 1} "
                    f"nodal circle{'s' if mode.n != 2 else ''}, λ² = {mode.root ** 2:.4g}"
                    + ("; a pair (cos nθ, sin nθ)" if mode.m else ""))
        else:
            x = np.linspace(0, p.a, GRID)
            y = np.linspace(0, p.b, max(3, int(GRID * p.b / p.a)))
            mix = (0, 1, -1)[self.mix.currentIndex()] if self.mix.isEnabled() else 0
            w = rect_shape(mode, p, x, y, mix)
            rect = QtCore.QRectF(0, 0, p.a, p.b)
            self.outline.setData([0, p.a, p.a, 0, 0], [0, 0, p.b, p.b, 0])
            what = f"({mode.m}, {mode.n}): {mode.m - 1} nodal line{'s' if mode.m != 2 else ''} across a, " \
                   f"{mode.n - 1} across b"
            if mix:
                what += f", {'+' if mix > 0 else '−'} its twin ({mode.n}, {mode.m})"
        top = float(np.nanmax(np.abs(w))) or 1.0
        img = np.nan_to_num(w / top, nan=0.0)
        self.shape_image.setImage(img, levels=(-1, 1), autoLevels=False)
        self.shape_image.setRect(rect)
        self.nodal.setData(circular_shape(mode, x, x, edge=0.97) if p.circular else w)
        self.shape_plot.setXRange(rect.left(), rect.right(), padding=0.04)
        self.shape_plot.setYRange(rect.top(), rect.bottom(), padding=0.04)
        self.readout.setText(f"Mode {int(self.mode.value())}, {fmt(mode.freq_hz, 'Hz')}: {what}.")

    def _draw_tab(self) -> None:
        i = self.tabs.currentIndex()
        if i == TAB_TABLE:
            self._draw_table()
        elif i == TAB_DENSITY:
            self._draw_density()

    def _draw_table(self) -> None:
        n = int(self.mode.value())
        rows = []
        if self.structure.currentIndex() == BEAM:
            b = self.beam
            cols = {th: beam_frequencies(b, MODES, th) for th in THEORIES}
            head = "<tr><th>n</th>" + "".join(f"<th>{th.value}</th>" for th in THEORIES) + "</tr>"
            for i in range(MODES):
                cells = "".join(f"<td align='right'>{cols[th][i]:.5g}</td>" for th in THEORIES)
                style = " style='font-weight:bold'" if i + 1 == n else ""
                rows.append(f"<tr{style}><td>{i + 1}</td>{cells}</tr>")
            title = "Frequencies (Hz), simply supported"
        else:
            modes = self._plate_modes()
            labels = ("(n, s)", "nodal diameters, circles") if self.plate.circular else ("(m, n)", "nodal lines")
            head = f"<tr><th>#</th><th>{labels[0]}</th><th>f (Hz)</th><th>{labels[1]}</th></tr>"
            for i, m in enumerate(modes):
                lines = f"{m.m}, {m.n - 1}" if self.plate.circular else f"{m.m - 1} × {m.n - 1}"
                style = " style='font-weight:bold'" if i + 1 == n else ""
                rows.append(f"<tr{style}><td>{i + 1}</td><td>({m.m}, {m.n})</td><td align='right'>"
                            f"{m.freq_hz:.5g}</td><td>{lines}</td></tr>")
            title = "Frequencies, lowest first"
        self.table.setHtml(f"<p><b>{title}</b></p><table cellspacing='4'>{head}{''.join(rows)}</table>")

    def _draw_density(self) -> None:
        if self.structure.currentIndex() == BEAM:
            b = self.beam
            freqs = beam_frequencies(b, DENSITY_MODES, Theory.EULER)
            top = float(freqs[60])
            f = np.linspace(0.0, top, 600)
            self.count_item.setData(f, counting(freqs, f))
            self.asym_item.setData(f, beam_asymptote(b, f))
            self.density_note.setText(f"Euler–Bernoulli's count grows as √f: N ≈ (L/π)(ρAω²/EI)<sup>¼</sup>. "
                                      f"The gap between modes widens: {fmt(float(freqs[1] - freqs[0]), 'Hz')} between "
                                      f"the first two, {fmt(float(freqs[60] - freqs[59]), 'Hz')} by the sixtieth.")
        else:
            p = self.plate
            modes = self._plate_modes(DENSITY_MODES)
            freqs = np.array([m.freq_hz for m in modes])
            weights = np.array([2.0 if (p.circular and m.m) else 1.0 for m in modes])
            top = float(freqs[int(0.6 * len(freqs))])
            f = np.linspace(0.0, top, 600)
            self.count_item.setData(f, counting(freqs, f, weights))
            self.asym_item.setData(f, plate_asymptote(p, f))
            per_hz = p.area / (2 * p.scale)
            self.density_note.setText(f"A plate's count grows in proportion to f: S/2 √(ρh/D) = {per_hz:.3g} modes "
                                      f"per Hz, a mode every {fmt(1 / per_hz, 'Hz')} on average"
                                      + (" (each pair counted twice)" if p.circular else "")
                                      + ". The boundary takes a few modes off the leading term.")
        self.density_plot.setXRange(0.0, top, padding=0)
        self.density_plot.enableAutoRange(axis="y")
