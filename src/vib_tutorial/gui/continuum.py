"""Continuous systems page: a string, a bar or a beam against the meshes of masses and springs that approximate it.

Left: the preset, the medium (its kind, size, material and ends), the mesh (how many
elements, which ways of sharing out the mass to show, the mode to look at) and the waves
(the pulse, the mesh's mass, a dashpot on the far end, a step in the section). Centre,
three tabs: the dispersion curves, the continuum's and each mesh's, with the modes on them;
the frequency errors, against the mode number and, for the chosen mode, against the number
of elements; and a pulse running along the continuum and the mesh. Right: the chosen mode's
shape, the frequencies in a table, and the notes. The page has its own model
(core/continuum.py and core/continuum_waves.py), not the chain.
"""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.continuum import (
    CONVERGENCE_COUNTS,
    ENDS,
    MAX_ELEMENTS,
    PRESETS,
    SCHEMES,
    TWO_PI,
    Continuum,
    End,
    FEModel,
    Kind,
    Scheme,
    convergence,
    dispersion,
    flexible_hz,
    order,
    string_dispersion,
)
from ..core.continuum_waves import (
    PLAYBACK,
    WAVE_PRESETS,
    WavePreset,
    WaveRun,
    Waves,
    end_reflection,
    step_coefficients,
)
from . import frame_stats, settle_plots
from .axes import log_axes
from .layouts import ColumnScroll, FlowLayout, group, plot_height
from .links import before, takeaways
from .panels import spin
from .style import colors
from .theming import add_legend, mute

CUSTOM = "Custom"
ALL_PRESETS = (*PRESETS, *WAVE_PRESETS)
FRAME_MS = 16
NODES_DRAWN = 80  # the mesh's nodes are dotted up to this many elements
KAPPA_POINTS = 241  # samples of each mesh's dispersion curve
SHAPE_POINTS = 801
PAST_CUTOFF = 1.25  # the dispersion plot runs this far past the mesh's cut-off, π/h

TIPS = {
    "kind": "<p><b>String</b>: transverse waves under tension, c = √(T/μ). <b>Bar</b>: axial (lengthwise) "
            "waves, c = √(E/ρ). Both obey the wave equation, so every wave travels at c. <b>Beam</b>: bending "
            "(Euler–Bernoulli), ω = √(EI/ρA) κ²: short waves travel faster than long ones.</p>",
    "length": "<p>The length between the ends.</p>",
    "tension": "<p>The string's tension T.</p>",
    "line_density": "<p>The string's mass per unit length μ.</p>",
    "modulus": "<p>Young's modulus E: steel 210 GPa, aluminium 69, wood about 10 along the grain.</p>",
    "density": "<p>The material's density ρ: steel 7850 kg/m³, aluminium 2700.</p>",
    "section": "<p>The rectangular section, width × depth. A beam bends across its depth (I = b d³/12), so its "
               "frequencies go as the depth and don't depend on the width; nor do a bar's on either.</p>",
    "ends": "<p>How each end is held. String and bar: <b>fixed</b> (held still) or <b>free</b> (a string's end "
            "on a ring that slides on a rod). Beam: <b>clamped</b> (no deflection, no slope), <b>pinned</b> "
            "(no deflection, free to turn) or <b>free</b>.</p>",
    "elements": "<p>The number of equal elements the length is cut into. Lumped, a string is then beads on a "
                "massless string, and a bar the chain of the first page.</p>",
    "schemes": "<p>How each element's mass is shared out. <b>Lumped</b>: half to each end node (a beam's "
               "slopes get none). <b>Consistent</b>: from the same shape functions as the stiffness: an "
               "upper bound on every frequency. <b>Mixed</b>: their average.</p>",
    "pulse": "<p>The pulse: a raised cosine, its full width and its centre as fractions of the length. The fewer "
             "elements across it, the more of it is short waves the mesh gets wrong.</p>",
    "moving": "<p>Start the pulse travelling right (d'Alembert's right-going wave alone), or at rest, when it "
              "splits into two halves going opposite ways. A beam's starts at rest.</p>",
    "wave_scheme": "<p>How the mesh the pulse runs on shares out its mass.</p>",
    "dashpot": "<p>Put a dashpot on the far end in place of the medium's own end, as a multiple of the "
               "impedance Z = √(Tμ) (string) or ρcA (bar). At 1 it is matched: a wave runs into it and is "
               "gone, as if the medium went on for ever. Reflected: (1 − r)/(1 + r) of the wave.</p>",
    "step": "<p>A step in the medium at this point: past it the impedance is Z₂/Z₁ times as much (a string "
            "that much heavier squared, so its waves go Z₁/Z₂ as fast; a bar of that much more section, at the "
            "same speed). A wave reflects (Z₁ − Z₂)/(Z₁ + Z₂) of itself and passes 2Z₁/(Z₁ + Z₂). On the "
            "mesh, at the nearest node.</p>",
    "playback": "<p>How fast the pulse plays: at 1× it crosses the length in about four seconds.</p>",
    "mode": "<p>The flexible mode to look at (the rigid ones, at 0 Hz, are left out): its shape, where it sits on "
            "the dispersion curves, and how its error falls with the number of elements. Or click a mode on "
            "the error plot.</p>",
}
DISPERSION_TIP = (
    "<p>Frequency against wavenumber κ = 2π/λ. The continuum's (black) is a straight line for a string or "
    "bar (every wave at the same speed) and a parabola for a beam. Each mesh's bends away from it and stops at "
    "κ = π/h, two elements a wavelength: the shortest wave the nodes can carry. The dots are the mesh's "
    "modes, each at its exact mode's wavenumber. A beam's consistent or mixed mesh has a second branch "
    "(dashed), where the slopes swing against the deflections, drawn from π/h on to 2π/h: the mesh's modes past "
    "π/h are on it, its nodes seeing a shorter wave's deflections as a longer one's.</p>")
ERROR_TIP = (
    "<p>Each mode's frequency on the mesh against the exact one, (f<sub>mesh</sub> − f) / f. The low modes "
    "span many elements and come out nearly right; the high ones are wrong by tens of per cent. Click a "
    "mode to choose it.</p>")
CONVERGENCE_TIP = (
    "<p>The chosen mode's error against the number of elements, both logarithmic: a straight line of slope "
    "−p is an error going as N<sup>−p</sup>. The dotted line marks the mesh shown. Points stop where the "
    "error reaches the eigensolver's round-off.</p>")
WAVES_TIP = (
    "<p>The pulse on the continuum (black) and on the mesh: its nodes, and the elements' own shape functions "
    "between them. The continuum's is exact: d'Alembert's two waves, carried along and scattered at the ends "
    "and the step in closed form (a string or bar), or the sum of its exact modes (a beam).</p>")
SHAPE_TIP = (
    "<p>The chosen mode's shape: the exact one (black) and each mesh's, the nodes as dots and the elements' own "
    "shape functions between them (straight for a string or bar, cubic for a beam), largest +1.</p>")

NOTES_HTML = """
<!--BEFORE-->
<h3>From a chain to a continuum</h3>
<p>Cut a string, a bar or a beam into N equal elements of length h and it becomes a chain of masses and
springs, with N or so modes where the continuum has infinitely many. How good are they? A string becomes a
string of beads and a bar becomes the first page's chain (k = EA/h, m = ρAh), and the clearest way to compare
them with the continuum is through the waves each can carry.</p>

<h3>The continuum's waves</h3>
<p>A string under tension T (mass μ per length) and a bar (EA, ρA) obey one equation, the wave equation:</p>
<p align="center">μ w<sub>tt</sub> = T w<sub>xx</sub>,&nbsp;&nbsp; ρA u<sub>tt</sub> = EA u<sub>xx</sub>.</p>
<p>A wave e<sup>i(κx − ωt)</sup> solves it when ω = cκ, with c = √(T/μ) or √(E/ρ): every wavelength travels at
the same speed (the medium is <b>non-dispersive</b>), so a pulse keeps its shape. A beam's bending,
ρA w<sub>tt</sub> = −EI w<sub>xxxx</sub>, gives ω = aκ² with a = √(EI/ρA): short waves outrun long ones
(<b>dispersive</b>), and a pulse spreads into ripples.</p>
<p>A mode is a standing wave that fits the ends. Fixed at both ends, a whole number of half wavelengths fits:
κ<sub>n</sub> = nπ/L and f<sub>n</sub> = n c / 2L, every overtone a whole multiple (a guitar string). Fixed at
one end and free at the other, an odd number of quarters: 1 : 3 : 5. A beam's ends give a transcendental
equation, cos βL cosh βL = −1 for a cantilever, so its overtones are no multiples at all (1 : 6.27 : 17.5).</p>

<h3><a name="dispersion"></a>The chain's waves</h3>
<p>On an endless chain of masses m = μh joined by springs k = T/h, try the same wave sampled at the nodes,
x<sub>j</sub> = e<sup>i(κjh − ωt)</sup>. One mass's equation gives</p>
<p align="center">ω = (2c/h) sin(κh/2).</p>
<p>For long waves (κh ≪ 1) the sine is its argument and ω = cκ: the chain is the continuum. Shorter waves go
slower, and at κh = π, two elements a wavelength, the curve tops out at 2c/h: the chain's <b>cut-off</b>.
Nothing faster can travel on it, and its group velocity dω/dκ = c cos(κh/2), the speed a pulse's energy
travels at, falls to zero there. A wave beyond κ = π/h looks, at the nodes, exactly like a longer one
(aliasing): the chain can't tell them apart.</p>
<p>A chain with ends has the same modes as the continuum in shape: the sampled standing waves, at the same
κ<sub>n</sub>. Its frequencies are the dispersion curve's at those κ<sub>n</sub>, so the low modes come out
nearly right, and the high ones, crowding towards κ = π/h, far too low: the lumped string's error is
−(κh)²/24 to begin with, and −36 % at the cut-off.</p>

<h3>Lumped and consistent masses</h3>
<p>Lumping is one way to share an element's mass out: half to each end. The finite element method's
<b>consistent</b> mass takes it from the same straight-line shape functions as the stiffness, ρAh/6 [2 1; 1 2]:
the mesh is then a Rayleigh–Ritz approximation, and every frequency is an upper bound. Its dispersion is</p>
<p align="center">ω² = (c/h)² · 2(1 − cos κh) / (1 − (1 − cos κh)/3),</p>
<p>which is high by (κh)²/24 for long waves: the same as lumping's error, the other way. Their average, the
<b>mixed</b> mass, cancels it, and its error goes as (κh)⁴. Halve the elements' length and the lumped or
consistent error falls 4 times (N<sup>−2</sup>), the mixed 16 times (N<sup>−4</sup>): the <i>Errors</i>
tab's convergence plot shows the slopes.</p>
<p>A beam's elements are cubics in its deflection, with a deflection and a slope at each node. Its consistent
mass is far better than its lumped one (N<sup>−4</sup> against N<sup>−2</sup>), and here averaging only
halves the lumped error. With two unknowns a node the endless mesh has two waves for each κ: the bending wave,
and a second branch far above it in which the slopes swing against the deflections, which the continuum
doesn't have. Lumped, the slopes carry no mass, and are condensed out exactly.</p>

<h3>Waves and reflections</h3>
<p>The <i>Waves</i> tab runs a pulse along the medium and the mesh side by side. On a string or bar the
continuum's pulse keeps its shape, u = r(x − ct) + l(x + ct), and comes back from a fixed end upside down
(the end must stay still, so the reflection cancels it there) and from a free end upright (doubling there).
On the mesh the pulse is made of waves the mesh carries at their own speeds: lumped, its short waves go
slow, and the pulse leaves ripples behind it; consistent, they go fast and the ripples lead. Nothing
shorter than two elements travels at all. Few elements across the pulse and it falls apart.</p>
<p>What a wave meets reflects some of it, by the impedance Z = √(Tμ) (string) or ρcA (bar), the force a
wave needs per unit of velocity. A <b>dashpot</b> of c<sub>d</sub> at an end returns (Z − c<sub>d</sub>)/(Z +
c<sub>d</sub>) of the wave: c<sub>d</sub> = Z is a <b>matched</b>, non-reflecting end, which absorbs the wave as
an endless medium would carry it off. A <b>step</b> from Z₁ to Z₂ returns (Z₁ − Z₂)/(Z₁ + Z₂), upside down
into something heavier, and passes 2Z₁/(Z₁ + Z₂). The energy is shared R² : T²Z₂/Z₁.</p>
<p>The mesh's impedance is the continuum's only for long waves: a dashpot matched to the continuum
reflects the mesh's short ones. That is why an absorbing boundary on a model is never quite silent, and
why the mesh's high modes, the short waves, are the ones not to trust.</p>
<p>A beam's bending waves are dispersive on the continuum too, ω = aκ²: a bump released on it spreads
into ripples, the short ones running ahead at twice the speed of waves twice as long.</p>

<!--TAKEAWAYS-->
"""
NOTES_HTML = NOTES_HTML.replace("<!--BEFORE-->", before(
    ("sim_page", "undamped-normal-modes-real", "Undamped normal modes"),
    ("sim_page", "rayleigh-quotient", "Estimating a frequency from a guessed shape"),
)).replace("<!--TAKEAWAYS-->", takeaways(
    "A mesh is a low-pass filter on the continuum: it carries the long waves correctly, its short ones too "
    "slowly (lumped) or too fast (consistent), and nothing shorter than two elements.",
    "Its low modes are good and its top modes are artefacts of the mesh: ten elements a wavelength (κh ≈ 0.63) "
    "keeps a string's lumped error under 2 %, and the rule of thumb is 6 to 10.",
    "A consistent mass makes the mesh Rayleigh–Ritz (upper bounds); mixing lumped and consistent cancels the "
    "leading error.",
    "Waves reflect at every change of impedance; a matched dashpot absorbs them.",
))



def _plot(left: str, left_units: str | None, bottom: str, bottom_units: str | None,
          log: bool = False) -> pg.PlotWidget:
    p = pg.PlotWidget(axisItems=log_axes()) if log else pg.PlotWidget()
    p.setMenuEnabled(False)
    p.hideButtons()
    p.showGrid(x=True, y=True, alpha=0.15)
    p.setLabel("left", left, units=left_units)
    p.setLabel("bottom", bottom, units=bottom_units)
    return p


def _form_row(form: QtWidgets.QFormLayout, text: str, box: QtWidgets.QWidget, tip: str) -> None:
    box.setToolTip(tip)
    label = QtWidgets.QLabel(text)
    label.setToolTip(tip)
    form.addRow(label, box)


def fmt_hz(f: float) -> str:
    return f"{f / 1e3:.4g} kHz" if f >= 1e4 else f"{f:.4g} Hz"


def scheme_color(scheme: Scheme) -> str:
    return {Scheme.LUMPED: colors.mode[0], Scheme.CONSISTENT: colors.mode[1], Scheme.MIXED: colors.mode[3]}[scheme]


class ContinuumPage(QtWidgets.QWidget):
    """Controls on the left, dispersion and errors in the centre, the mode's shape, the table and the notes on the right."""

    follows_chain = False  # MainWindow: this page has its own model, not the chain

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._errors_stale = False  # the Errors tab wants drawing when it next shows
        self.continuum: Continuum = PRESETS[0].continuum
        self.models: dict[Scheme, FEModel] = {}
        self._loading = False

        # --- left: the preset, the medium, the mesh
        self.preset = QtWidgets.QComboBox()
        self.preset.addItems([p.name for p in ALL_PRESETS] + [CUSTOM])
        self.preset.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.preset_note = QtWidgets.QLabel()
        self.preset_note.setWordWrap(True)
        mute(self.preset_note)

        medium = QtWidgets.QGroupBox("Medium")
        mf = QtWidgets.QFormLayout(medium)
        self.kind = QtWidgets.QComboBox()
        self.kind.addItems([k.value for k in Kind])
        self.length = spin(0.01, 100.0, 1.0, 3, " m")
        _form_row(mf, "Kind:", self.kind, TIPS["kind"])
        _form_row(mf, "Length:", self.length, TIPS["length"])
        self.material = QtWidgets.QStackedWidget()
        string_page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(string_page)
        form.setContentsMargins(0, 0, 0, 0)
        self.tension = spin(0.01, 1e6, 100.0, 2, " N")
        self.line_density = spin(0.001, 1e5, 10.0, 3, " g/m")
        _form_row(form, "Tension T:", self.tension, TIPS["tension"])
        _form_row(form, "Mass per length μ:", self.line_density, TIPS["line_density"])
        self.material.addWidget(string_page)
        solid_page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(solid_page)
        form.setContentsMargins(0, 0, 0, 0)
        self.modulus = spin(0.001, 1000.0, 210.0, 3, " GPa")
        self.density = spin(1.0, 30000.0, 7850.0, 0, " kg/m³")
        self.width = spin(0.01, 1000.0, 20.0, 2, " mm")
        self.depth = spin(0.01, 1000.0, 10.0, 2, " mm")
        _form_row(form, "Modulus E:", self.modulus, TIPS["modulus"])
        _form_row(form, "Density ρ:", self.density, TIPS["density"])
        _form_row(form, "Section:", group(self.width, "×", self.depth), TIPS["section"])
        self.material.addWidget(solid_page)
        mf.addRow(self.material)
        self.left_end = QtWidgets.QComboBox()
        self.right_end = QtWidgets.QComboBox()
        _form_row(mf, "Ends:", group(self.left_end, "–", self.right_end), TIPS["ends"])
        self.medium_info = QtWidgets.QLabel()
        self.medium_info.setWordWrap(True)
        mute(self.medium_info)
        mf.addRow(self.medium_info)

        mesh = QtWidgets.QGroupBox("Mesh")
        sf = QtWidgets.QFormLayout(mesh)
        self.elements = QtWidgets.QSpinBox()
        self.elements.setRange(1, MAX_ELEMENTS)
        self.elements.setKeyboardTracking(False)
        _form_row(sf, "Elements N:", self.elements, TIPS["elements"])
        self.shown: dict[Scheme, QtWidgets.QCheckBox] = {}
        row = FlowLayout()
        for scheme in SCHEMES:
            box = QtWidgets.QCheckBox(scheme.value)
            box.setChecked(True)
            box.setToolTip(TIPS["schemes"])
            self.shown[scheme] = box
            row.add(box)
        label = QtWidgets.QLabel("Mass:")
        label.setToolTip(TIPS["schemes"])
        sf.addRow(label, _holder(row))
        self.mode = QtWidgets.QSpinBox()
        self.mode.setRange(1, 1)
        self.mode.setKeyboardTracking(False)
        _form_row(sf, "Mode:", self.mode, TIPS["mode"])
        self.mesh_info = QtWidgets.QLabel()
        self.mesh_info.setWordWrap(True)
        mute(self.mesh_info)
        sf.addRow(self.mesh_info)

        waves = QtWidgets.QGroupBox("Waves")
        wf = QtWidgets.QFormLayout(waves)
        self.pulse_width = spin(1.0, 100.0, 12.5, 1, " %")
        self.pulse_at = spin(0.0, 100.0, 25.0, 1, " %")
        _form_row(wf, "Pulse:", group(self.pulse_width, "wide at", self.pulse_at), TIPS["pulse"])
        self.moving = QtWidgets.QCheckBox("Travelling right")
        self.moving.setChecked(True)
        self.moving.setToolTip(TIPS["moving"])
        wf.addRow(self.moving)
        self.wave_scheme = QtWidgets.QComboBox()
        self.wave_scheme.addItems([sc.value for sc in SCHEMES])
        _form_row(wf, "Mesh's mass:", self.wave_scheme, TIPS["wave_scheme"])
        self.dashpot_on = QtWidgets.QCheckBox("Dashpot on the far end")
        self.dashpot_on.setToolTip(TIPS["dashpot"])
        self.dashpot = spin(0.0, 100.0, 1.0, 2, " Z")
        wf.addRow(group(self.dashpot_on, self.dashpot))
        self.dashpot.setToolTip(TIPS["dashpot"])
        self.step_on = QtWidgets.QCheckBox("Step at")
        self.step_on.setToolTip(TIPS["step"])
        self.step_at = spin(1.0, 99.0, 50.0, 1, " %")
        self.step_ratio = spin(0.05, 20.0, 2.0, 2, "")
        self.step_ratio.setPrefix("Z₂/Z₁ ")
        for box in (self.step_at, self.step_ratio):
            box.setToolTip(TIPS["step"])
        wf.addRow(group(self.step_on, self.step_at, self.step_ratio))
        self.wave_info = QtWidgets.QLabel()
        self.wave_info.setWordWrap(True)
        mute(self.wave_info)
        wf.addRow(self.wave_info)

        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        presets = QtWidgets.QGroupBox("Things to try")
        preset_layout = QtWidgets.QVBoxLayout(presets)
        preset_layout.addWidget(self.preset)
        preset_layout.addWidget(self.preset_note)
        lv.addWidget(presets)
        lv.addWidget(medium)
        lv.addWidget(mesh)
        lv.addWidget(waves)
        lv.addStretch(1)
        left_scroll = ColumnScroll(left)

        # --- centre: dispersion, errors
        self.dispersion = _plot("Frequency", "Hz", "Wavenumber κ", "rad/m")
        self.dispersion.setToolTip(DISPERSION_TIP)
        plot_height(self.dispersion, 260)
        self.dispersion_legend = add_legend(self.dispersion.getPlotItem(), offset=(10, 10))  # top left
        self.readout = QtWidgets.QLabel()
        self.readout.setWordWrap(True)
        self.readout.setTextFormat(QtCore.Qt.TextFormat.RichText)
        dispersion_tab = QtWidgets.QWidget()
        dv = QtWidgets.QVBoxLayout(dispersion_tab)
        dv.addWidget(self.dispersion, 1)
        dv.addWidget(self.readout)

        self.errors = _plot("Frequency error", "%", "Mode", None)
        self.errors.setToolTip(ERROR_TIP)
        plot_height(self.errors, 180)
        self.errors_legend = add_legend(self.errors.getPlotItem(), offset=(10, 10))  # top left
        self.errors.scene().sigMouseClicked.connect(self._on_errors_click)
        self.convergence = _plot("|Frequency error|", None, "Elements N", None, log=True)
        self.convergence.setLogMode(x=True, y=True)
        self.convergence.setToolTip(CONVERGENCE_TIP)
        plot_height(self.convergence, 180)
        self.convergence_legend = add_legend(self.convergence.getPlotItem(), offset=(-10, 10))  # top right
        self.convergence_note = QtWidgets.QLabel()
        self.convergence_note.setWordWrap(True)
        self.convergence_note.setTextFormat(QtCore.Qt.TextFormat.RichText)
        errors_tab = QtWidgets.QWidget()
        ev = QtWidgets.QVBoxLayout(errors_tab)
        ev.addWidget(self.errors, 1)
        ev.addWidget(self.convergence, 1)
        ev.addWidget(self.convergence_note)
        self.views = QtWidgets.QTabWidget()
        self.dispersion_tab = dispersion_tab
        self.errors_tab = errors_tab
        self.views.addTab(dispersion_tab, "Dispersion")
        self.views.addTab(errors_tab, "Errors")

        # the pulse on the continuum and on the mesh
        self.waves_plot = _plot("Displacement", None, "Position x", "m")
        self.waves_plot.setMouseEnabled(x=False, y=False)
        self.waves_plot.setToolTip(WAVES_TIP)
        plot_height(self.waves_plot, 240)
        self.waves_legend = add_legend(self.waves_plot.getPlotItem(), offset=(-10, 10))
        self.waves_zero = pg.InfiniteLine(0.0, angle=0)
        self.waves_plot.addItem(self.waves_zero)
        self.step_region = pg.LinearRegionItem(movable=False)
        self.step_region.setZValue(-10)
        self.waves_plot.addItem(self.step_region)
        self.end_line = pg.InfiniteLine(0.0, angle=90, label="dashpot", labelOpts={"position": 0.9})
        self.waves_plot.addItem(self.end_line)
        self.exact_curve = self.waves_plot.plot([], [])
        self.mesh_curve = self.waves_plot.plot([], [])
        self.mesh_nodes = self.waves_plot.plot([], [], pen=None, symbol="o", symbolSize=6)
        self.waves_legend.addItem(self.exact_curve, "Continuum")
        self.waves_legend.addItem(self.mesh_curve, "Mesh")
        self._style_waves()
        self.play_button = QtWidgets.QPushButton("Pause")
        self.play_button.setToolTip("<p>Pause or run the pulse.</p>")
        self.restart_button = QtWidgets.QPushButton("Restart")
        self.restart_button.setToolTip("<p>Start the pulse again from where it began.</p>")
        self.playback = QtWidgets.QComboBox()
        for x in PLAYBACK:
            self.playback.addItem(f"{x:g}×", x)
        self.playback.setCurrentIndex(PLAYBACK.index(1.0))
        self.playback.setToolTip(TIPS["playback"])
        self.wave_status = QtWidgets.QLabel()
        self.wave_status.setWordWrap(True)
        self.wave_status.setTextFormat(QtCore.Qt.TextFormat.RichText)
        waves_tab = QtWidgets.QWidget()
        wv = QtWidgets.QVBoxLayout(waves_tab)
        wv.addWidget(self.waves_plot, 1)
        row = FlowLayout()
        row.add(self.play_button, self.restart_button, group("Speed:", self.playback))
        wv.addLayout(row)
        wv.addWidget(self.wave_status)
        self.waves_tab = waves_tab
        self.views.addTab(waves_tab, "Waves")
        self.run: WaveRun | None = None  # built when the Waves tab first shows after a change
        self.running = True
        self._reach = 1.0  # the plot's half height: the largest displacement seen yet

        # --- right: the mode's shape, the table, the notes
        self.shape = _plot("Deflection", None, "Position x", "m")
        self.shape.setMouseEnabled(x=False, y=False)
        self.shape.setToolTip(SHAPE_TIP)
        plot_height(self.shape, 200)
        self.shape_legend = add_legend(self.shape.getPlotItem(), offset=(10, 5), colCount=2)  # along the top
        self.shape_note = QtWidgets.QLabel()
        self.shape_note.setWordWrap(True)
        self.shape_note.setTextFormat(QtCore.Qt.TextFormat.RichText)
        shape_tab = QtWidgets.QWidget()
        sv = QtWidgets.QVBoxLayout(shape_tab)
        sv.addWidget(self.shape, 1)
        sv.addWidget(self.shape_note)

        self.table = QtWidgets.QTableWidget()
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.setToolTip("<p>Each flexible mode's exact frequency and each mesh's, with its error. "
                              "Pick a row to choose the mode.</p>")
        self.table.itemSelectionChanged.connect(self._on_table_selection)

        self.notes = QtWidgets.QTextBrowser()
        self.notes.setHtml(NOTES_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.shape_tab = shape_tab
        self.tabs.addTab(shape_tab, "Mode shape")
        self.tabs.addTab(self.table, "Frequencies")
        self.tabs.addTab(self.notes, "Theory")

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(left_scroll)
        splitter.addWidget(self.views)
        splitter.addWidget(self.tabs)
        splitter.setSizes([340, 810, 550])
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        # --- wiring
        self.preset.currentIndexChanged.connect(self._on_preset)
        self.kind.currentIndexChanged.connect(self._on_kind)
        for box in (self.length, self.tension, self.line_density, self.modulus, self.density, self.width,
                    self.depth):
            box.valueChanged.connect(self._on_edit)
        self.left_end.currentIndexChanged.connect(self._on_edit)
        self.right_end.currentIndexChanged.connect(self._on_edit)
        self.elements.valueChanged.connect(self._on_edit)
        for box in self.shown.values():
            box.toggled.connect(lambda _: self.redraw())
        self.mode.valueChanged.connect(self._on_mode)
        for box in (self.pulse_width, self.pulse_at, self.dashpot, self.step_at, self.step_ratio):
            box.valueChanged.connect(self._on_waves)
        for box in (self.moving, self.dashpot_on, self.step_on):
            box.toggled.connect(self._on_waves)
        self.wave_scheme.currentIndexChanged.connect(self._on_waves)
        self.play_button.clicked.connect(self._on_play)
        self.restart_button.clicked.connect(self.restart)
        self.playback.currentIndexChanged.connect(self._on_playback)
        self.views.currentChanged.connect(self._on_view)

        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._frame)
        self._clock = QtCore.QElapsedTimer()

        self.load_preset(PRESETS[0].name)

    # ------------------------------------------------------------- inputs
    def load_preset(self, name: str) -> None:
        """A preset's medium and mesh; a wave preset's pulse too, on the Waves tab, running."""
        p = next(p for p in ALL_PRESETS if p.name == name)
        c = p.continuum
        self._loading = True
        self.preset.setCurrentText(name)
        self.preset_note.setText(p.note)
        self.kind.setCurrentText(c.kind.value)
        self._fill_ends(c.kind)
        self.length.setValue(c.length)
        self.tension.setValue(c.tension)
        self.line_density.setValue(c.line_density * 1e3)
        self.modulus.setValue(c.modulus / 1e9)
        self.density.setValue(c.density)
        self.width.setValue(c.width * 1e3)
        self.depth.setValue(c.depth * 1e3)
        self.left_end.setCurrentText(c.left.value)
        self.right_end.setCurrentText(c.right.value)
        self.elements.setValue(p.elements)
        self._set_waves(p.waves if isinstance(p, WavePreset) else Waves(), c.kind)
        self._loading = False
        self.mode.blockSignals(True)
        self.mode.setValue(1)
        self.mode.blockSignals(False)
        self.set_continuum(c, p.elements)
        if isinstance(p, WavePreset):
            self.running = True
            self.views.setCurrentWidget(self.waves_tab)

    def frame_context(self) -> str:
        """For the frame log (frame_stats): the preset showing."""
        return self.preset.currentText()

    def _on_preset(self, index: int) -> None:
        if not self._loading and index < len(ALL_PRESETS):
            self.load_preset(ALL_PRESETS[index].name)

    def _fill_ends(self, kind: Kind) -> None:
        for box in (self.left_end, self.right_end):
            old = box.currentText()
            box.blockSignals(True)
            box.clear()
            box.addItems([e.value for e in ENDS[kind]])
            if old in [e.value for e in ENDS[kind]]:
                box.setCurrentText(old)
            box.blockSignals(False)
        self.material.setCurrentIndex(0 if kind is Kind.STRING else 1)
        for i in range(self.material.count()):  # only the page showing sets the height
            page = self.material.widget(i)
            policy = QtWidgets.QSizePolicy.Policy.Preferred if i == self.material.currentIndex() else \
                QtWidgets.QSizePolicy.Policy.Ignored
            page.setSizePolicy(policy, policy)

    def _on_kind(self, _index: int) -> None:
        if self._loading:
            return
        kind = Kind(self.kind.currentText())
        self._fill_ends(kind)
        self._loading = True
        self._set_waves(self._waves(), kind)  # a beam's pulse starts at rest, on its own ends
        self._loading = False
        self._on_edit()

    def _read(self) -> Continuum:
        return Continuum(Kind(self.kind.currentText()), self.length.value(), End(self.left_end.currentText()),
                         End(self.right_end.currentText()), tension=self.tension.value(),
                         line_density=self.line_density.value() / 1e3, modulus=self.modulus.value() * 1e9,
                         density=self.density.value(), width=self.width.value() / 1e3,
                         depth=self.depth.value() / 1e3)

    def _on_edit(self, *_args) -> None:
        if self._loading:
            return
        self.preset.blockSignals(True)
        self.preset.setCurrentText(CUSTOM)
        self.preset.blockSignals(False)
        self.preset_note.setText("")
        self.set_continuum(self._read(), self.elements.value())

    def set_continuum(self, c: Continuum, elements: int) -> None:
        """Mesh the medium every way and redraw."""
        self.continuum = c
        self.models = {s: FEModel(c, elements, s) for s in SCHEMES}
        count = max(flexible_hz(m).size for m in self.models.values())
        self._exact_kappa = c.exact_wavenumbers(count) if count else np.zeros(0)
        self.mode.blockSignals(True)
        self.mode.setRange(1, max(count, 1))
        self.mode.setEnabled(count > 0)
        self.mode.blockSignals(False)
        label = "Displacement (along the bar)" if c.kind is Kind.BAR else "Deflection"
        self.shape.setLabel("left", label)
        self.waves_plot.setLabel("left", label)
        self._show_info()
        self._fill_table()
        self.redraw()
        self._new_run()

    def _shown(self) -> list[Scheme]:
        return [s for s in SCHEMES if self.shown[s].isChecked()]

    def _show_info(self) -> None:
        c = self.continuum
        h = c.length / self.elements.value()
        if c.kind is Kind.BEAM:
            speed = f"a = √(EI/ρA) = {c.speed:.4g} m²/s (ω = aκ²)"
        else:
            speed = f"wave speed c = {c.speed:.4g} m/s"
        rigid = {0: "", 1: "; one rigid-body mode", 2: "; two rigid-body modes"}[c.rigid_modes]
        self.medium_info.setText(f"{speed}; mass {c.mass_per_length * c.length:.4g} kg{rigid}.")
        cut = math.pi / h
        tops = []
        for s in SCHEMES:
            top = self._mesh_omega(s, np.array([cut]))[0, 0] / TWO_PI
            tops.append(f"{s.value.lower()} {fmt_hz(top)}")
        self.mesh_info.setText(f"h = {h * 1e3:.4g} mm. Cut-off at κ = π/h = {cut:.4g} rad/m (λ = 2h), where the "
                               f"continuum is at {fmt_hz(float(c.omega(cut)) / TWO_PI)}; the meshes top out "
                               f"at {', '.join(tops)}.")

    def _mesh_omega(self, scheme: Scheme, kappa: np.ndarray) -> np.ndarray:
        """A mesh's dispersion (rad/s), a column per branch: the string's and bar's in closed form."""
        c, h = self.continuum, self.continuum.length / self.elements.value()
        if c.kind is Kind.BEAM:
            return dispersion(c, h, scheme, kappa)
        return string_dispersion(c, h, scheme, kappa)[:, None]

    def _fill_table(self) -> None:
        exact_count = self._exact_kappa.size
        exact = self.continuum.omega(self._exact_kappa) / TWO_PI
        self.table.blockSignals(True)
        self.table.clear()
        self.table.setColumnCount(2 + 2 * len(SCHEMES))
        self.table.setHorizontalHeaderLabels(
            ["Mode", "Exact"] + [x for s in SCHEMES for x in (s.value, "error")])
        self.table.setRowCount(exact_count)
        for i in range(exact_count):
            self.table.setItem(i, 0, _cell(f"{i + 1}"))
            self.table.setItem(i, 1, _cell(fmt_hz(exact[i])))
        for j, s in enumerate(SCHEMES):
            f = flexible_hz(self.models[s])
            for i, fi in enumerate(f):
                self.table.setItem(i, 2 + 2 * j, _cell(fmt_hz(fi)))
                self.table.setItem(i, 3 + 2 * j, _cell(f"{100 * (fi / exact[i] - 1):+.3g} %"))
        self.table.resizeColumnsToContents()
        self.table.blockSignals(False)

    def _on_mode(self, n: int) -> None:
        self.redraw()
        self.table.blockSignals(True)
        self.table.selectRow(n - 1)
        self.table.blockSignals(False)

    def _on_table_selection(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if rows:
            self.mode.setValue(rows[0].row() + 1)

    def _on_errors_click(self, event) -> None:
        vb = self.errors.getPlotItem().vb
        if not self.mode.isEnabled() or not vb.sceneBoundingRect().contains(event.scenePos()):
            return
        x = vb.mapSceneToView(event.scenePos()).x()
        self.mode.setValue(int(np.clip(round(x), 1, self.mode.maximum())))

    # ------------------------------------------------------------- waves
    def _set_waves(self, waves: Waves, kind: Kind) -> None:
        """Fill the wave controls (signals held), a beam's without the dashpot, the step or a moving start."""
        beam = kind is Kind.BEAM
        boxes = (self.pulse_width, self.pulse_at, self.moving, self.wave_scheme, self.dashpot_on, self.dashpot,
                 self.step_on, self.step_at, self.step_ratio)
        for box in boxes:
            box.blockSignals(True)
        self.pulse_width.setValue(100 * waves.width)
        self.pulse_at.setValue(100 * waves.at)
        self.moving.setChecked(waves.moving and not beam)
        self.wave_scheme.setCurrentText(waves.scheme.value)
        self.dashpot_on.setChecked(waves.dashpot is not None and not beam)
        if waves.dashpot is not None:
            self.dashpot.setValue(waves.dashpot)
        self.step_on.setChecked(waves.step_at is not None and not beam)
        if waves.step_at is not None:
            self.step_at.setValue(100 * waves.step_at)
        self.step_ratio.setValue(waves.step_ratio)
        for box in boxes:
            box.blockSignals(False)
        for box in (self.moving, self.dashpot_on, self.step_on):
            box.setEnabled(not beam)
        self._enable_waves()

    def _enable_waves(self) -> None:
        self.dashpot.setEnabled(self.dashpot_on.isChecked() and self.dashpot_on.isEnabled())
        for box in (self.step_at, self.step_ratio):
            box.setEnabled(self.step_on.isChecked() and self.step_on.isEnabled())

    def _waves(self) -> Waves:
        beam = Kind(self.kind.currentText()) is Kind.BEAM
        return Waves(width=self.pulse_width.value() / 100, at=self.pulse_at.value() / 100,
                     moving=self.moving.isChecked() and not beam, scheme=Scheme(self.wave_scheme.currentText()),
                     dashpot=self.dashpot.value() if self.dashpot_on.isChecked() and not beam else None,
                     step_at=self.step_at.value() / 100 if self.step_on.isChecked() and not beam else None,
                     step_ratio=self.step_ratio.value())

    def _on_waves(self, *_args) -> None:
        if self._loading:
            return
        self._enable_waves()
        self._new_run()

    def _new_run(self) -> None:
        """Start the pulse afresh: now if the Waves tab shows, else when it next does."""
        self.run = None
        self._show_wave_info()
        if self.views.currentWidget() is self.waves_tab:
            self.restart()

    def restart(self) -> None:
        self.run = WaveRun(self.continuum, self.elements.value(), self._waves(), float(self.playback.currentData()))
        self._reach = 1.0
        c = self.continuum
        self.waves_plot.setXRange(0.0, c.length, padding=0.02)
        self.waves_plot.setYRange(-1.15, 1.15, padding=0.0)
        step = self.run.step_x
        self.step_region.setVisible(step is not None)
        if step is not None:
            self.step_region.setRegion((step, c.length))
        self.end_line.setVisible(self.run.waves.dashpot is not None)
        self.end_line.setPos(c.length)
        self._draw_waves()
        self._update_play()
        self._start_timer()

    def _show_wave_info(self) -> None:
        c, w = self.continuum, self._waves()
        h = c.length / self.elements.value()
        parts = [f"{w.width * c.length / h:.3g} elements across the pulse."]
        if c.kind is not Kind.BEAM:
            z = math.sqrt(c.rigidity * c.mass_per_length)
            parts.append(f"Z = {z:.4g} N·s/m.")
            if w.step_at is not None:
                r, t = step_coefficients(w.step_ratio)
                parts.append(f"The step reflects {r:+.3g} and passes {t:.3g} of a wave from the left "
                             f"(energy {r * r:.3g} : {t * t * w.step_ratio:.3g}).")
            if w.dashpot is not None:
                parts.append(f"The dashpot ({w.dashpot:.3g} Z = {w.dashpot * z:.4g} N·s/m) reflects "
                             f"{end_reflection(c.right, w.dashpot):+.3g} of a wave.")
        self.wave_info.setText(" ".join(parts))

    def _on_play(self) -> None:
        self.running = not self.running
        self._update_play()
        if self.running:
            self._start_timer()
        else:
            self._timer.stop()

    def _update_play(self) -> None:
        self.play_button.setText("Pause" if self.running else "Run")

    def _on_playback(self, _index: int) -> None:
        if self.run is not None:
            self.run.set_playback(float(self.playback.currentData()))

    def _on_view(self, _index: int) -> None:
        if self.views.currentWidget() is self.errors_tab and self._errors_stale:
            self._draw_errors()
        if self.views.currentWidget() is self.waves_tab:
            if self.run is None:
                self.restart()
            else:
                self._start_timer()
        else:
            self._timer.stop()

    def showEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().showEvent(event)
        self._start_timer()

    def hideEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().hideEvent(event)
        self._timer.stop()

    def _start_timer(self) -> None:
        if (self.isVisible() and self.running and self.run is not None
                and self.views.currentWidget() is self.waves_tab and not self._timer.isActive()):
            self._clock.restart()
            self._timer.start(FRAME_MS)

    def _frame(self) -> None:
        began = self._clock.elapsed()
        with frame_stats.frame("Continuous systems", self._sim_t, FRAME_MS):
            self.step()
            settle_plots(self)
        if self.isVisible() and self.running and self.views.currentWidget() is self.waves_tab:
            # the next frame FRAME_MS after this one began, not after it ended
            self._timer.start(max(1, FRAME_MS - (self._clock.elapsed() - began)))

    def _sim_t(self) -> float | None:
        return self.run.t if self.run is not None else None

    def step(self, frames: int = 1) -> None:
        """Run the pulse on by `frames` frames and redraw."""
        if self.run is None:
            self.restart()
        self.run.advance(frames)
        self._draw_waves()

    def _draw_waves(self) -> None:
        run = self.run
        x, u = run.exact()
        self.exact_curve.setData(x, u)
        dense = np.linspace(0.0, self.continuum.length, SHAPE_POINTS)
        shape = run.mesh_shape()
        field = run.model.field(shape, dense)
        self.mesh_curve.setData(dense, field)
        nodes, w = run.mesh()
        if self.elements.value() <= NODES_DRAWN:
            self.mesh_nodes.setData(nodes, w)
        else:
            self.mesh_nodes.setData([], [])
        reach = max(float(np.max(np.abs(u))), float(np.max(np.abs(field))))
        if reach > self._reach:  # grow, never shrink, till the pulse starts again
            self._reach = reach
            self.waves_plot.setYRange(-1.15 * self._reach, 1.15 * self._reach, padding=0.0)
        crossings = run.t / run.transit
        self.wave_status.setText(
            f"t = {crossings:.2f} crossings ({_fmt_time(run.t)}; one crossing {_fmt_time(run.transit)}). "
            f"The mesh ({run.waves.scheme.value.lower()}) keeps <b>{100 * run.energy_left:.3g} %</b> of its "
            "energy.")

    def _style_waves(self) -> None:
        self.waves_zero.setPen(pg.mkPen(colors.faint, width=1))
        self.exact_curve.setPen(pg.mkPen(colors.strong, width=2))
        mesh = colors.mode[0]
        self.mesh_curve.setPen(pg.mkPen(mesh, width=1.5))
        self.mesh_nodes.setSymbolBrush(mesh)
        self.mesh_nodes.setSymbolPen(pg.mkPen(colors.background))
        shade = pg.mkColor(colors.faint)
        shade.setAlpha(70)
        self.step_region.setBrush(pg.mkBrush(shade))
        for line in self.step_region.lines:
            line.setPen(pg.mkPen(colors.muted, width=1))
        self.end_line.setPen(pg.mkPen(colors.force, width=3))
        self.end_line.label.setColor(colors.force)

    # ------------------------------------------------------------- drawing
    def apply_theme(self) -> None:
        self._style_waves()
        self.redraw()

    def redraw(self) -> None:
        self._draw_dispersion()
        self._draw_errors()
        self._draw_mode()

    def _draw_dispersion(self) -> None:
        p, c = self.dispersion, self.continuum
        p.clear()
        self.dispersion_legend.clear()
        h = c.length / self.elements.value()
        cut = math.pi / h
        kappa = np.linspace(0.0, cut, KAPPA_POINTS)
        shown = self._shown()
        count = max((flexible_hz(self.models[s]).size for s in shown), default=0)
        right = max(PAST_CUTOFF * cut, 1.02 * self._exact_kappa[count - 1] if count else 0.0)
        far = np.linspace(0.0, right, KAPPA_POINTS)
        exact = p.plot(far, c.omega(far) / TWO_PI, pen=pg.mkPen(colors.strong, width=2))
        self.dispersion_legend.addItem(exact, "Continuum")
        top = float(c.omega(cut)) / TWO_PI
        for s in shown:
            color = scheme_color(s)
            w = self._mesh_omega(s, kappa) / TWO_PI
            curve = p.plot(kappa, w[:, 0], pen=pg.mkPen(color, width=2))
            self.dispersion_legend.addItem(curve, s.value)
            for b in range(1, w.shape[1]):  # the slopes' branch, in the next zone: from π/h on to 2π/h
                p.plot(2 * cut - kappa, w[:, b], pen=pg.mkPen(color, width=1.5, style=QtCore.Qt.PenStyle.DashLine))
            f = flexible_hz(self.models[s])
            if f.size:
                p.plot(self._exact_kappa[:f.size], f, pen=None, symbol="o", symbolSize=6,
                       symbolBrush=color, symbolPen=pg.mkPen(colors.background))
            top = max(top, float(w.max()))
        line = pg.InfiniteLine(cut, angle=90, pen=pg.mkPen(colors.muted, width=1, style=QtCore.Qt.PenStyle.DashLine),
                               label="π/h", labelOpts={"position": 0.95, "color": colors.muted})
        p.addItem(line)
        n = self.mode.value() if self.mode.isEnabled() else 0
        if n:
            k = self._exact_kappa[n - 1]
            p.addItem(pg.InfiniteLine(k, angle=90, pen=pg.mkPen(colors.force, width=1,
                                                                style=QtCore.Qt.PenStyle.DotLine)))
        p.setXRange(0.0, right, padding=0.02)
        p.setYRange(0.0, 1.1 * top, padding=0.0)
        self._show_readout()

    def _show_readout(self) -> None:
        c = self.continuum
        if not self.mode.isEnabled():
            self.readout.setText("No flexible mode: one element between two held ends has nothing free to move.")
            return
        n = self.mode.value()
        h = c.length / self.elements.value()
        k = self._exact_kappa[n - 1]
        per = 2 * math.pi / (k * h)
        exact = float(c.omega(k)) / TWO_PI
        parts = []
        for s in self._shown():
            f = flexible_hz(self.models[s])
            if f.size >= n:
                parts.append(f"{s.value.lower()} {fmt_hz(f[n - 1])} ({100 * (f[n - 1] / exact - 1):+.3g} %)")
        found = "; ".join(parts) if parts else "no mesh shown"
        self.readout.setText(
            f"<b>Mode {n}</b>: κ = {k:.4g} rad/m, a wavelength of {2 * math.pi / k:.4g} m, "
            f"<b>{per:.3g} elements</b> (κh = {k * h:.3g}). Exact {fmt_hz(exact)}; {found}.")

    def _draw_errors(self) -> None:
        # Drawn only while its tab shows: the convergence study is ~2 s of eigensolves for a new
        # kind or set of ends (cached after), which made the page's first opening hang.
        self._errors_stale = self.views.currentWidget() is not self.errors_tab
        if self._errors_stale:
            return
        p = self.errors
        p.clear()
        self.errors_legend.clear()
        p.addItem(pg.InfiniteLine(0.0, angle=0, pen=pg.mkPen(colors.faint, width=1)))
        exact = self.continuum.omega(self._exact_kappa) / TWO_PI
        for s in self._shown():
            f = flexible_hz(self.models[s])
            if not f.size:
                continue
            err = 100 * (f / exact[:f.size] - 1)
            curve = p.plot(np.arange(1, f.size + 1), err, pen=pg.mkPen(scheme_color(s), width=1.5), symbol="o",
                           symbolSize=5, symbolBrush=scheme_color(s), symbolPen=None)
            self.errors_legend.addItem(curve, s.value)
        n = self.mode.value() if self.mode.isEnabled() else 0
        if n:
            p.addItem(pg.InfiniteLine(n, angle=90, pen=pg.mkPen(colors.force, width=1,
                                                                style=QtCore.Qt.PenStyle.DotLine)))
        self._draw_convergence()

    def _draw_convergence(self) -> None:
        p = self.convergence
        p.clear()
        self.convergence_legend.clear()
        if not self.mode.isEnabled():
            self.convergence_note.setText("")
            return
        n = self.mode.value()
        rates = []
        for s in self._shown():
            counts, errs = convergence(self.continuum, s, n)
            if not counts.size:
                continue
            curve = p.plot(counts, np.abs(errs), pen=pg.mkPen(scheme_color(s), width=1.5), symbol="o", symbolSize=5,
                           symbolBrush=scheme_color(s), symbolPen=None)
            self.convergence_legend.addItem(curve, s.value)
            rate = order(counts, errs)
            if rate is not None:
                rates.append(f"{s.value.lower()} N<sup>−{rate:.2g}</sup>")
        p.addItem(pg.InfiniteLine(math.log10(self.elements.value()), angle=90,
                                  pen=pg.mkPen(colors.muted, width=1, style=QtCore.Qt.PenStyle.DotLine)))
        p.setXRange(0.0, math.log10(CONVERGENCE_COUNTS[-1]), padding=0.03)
        self.convergence_note.setText(
            f"Mode {n}'s error falls as " + ", ".join(rates) + " (the three finest meshes' slopes)." if rates else
            f"No mesh shown has a mode {n} yet.")

    def _draw_mode(self) -> None:
        p, c = self.shape, self.continuum
        p.clear()
        self.shape_legend.clear()
        if not self.mode.isEnabled():
            self.shape_note.setText("")
            return
        n = self.mode.value()
        x = np.linspace(0.0, c.length, SHAPE_POINTS)
        k = self._exact_kappa[n - 1]
        exact = c.exact_shape(k, x)
        p.addItem(pg.InfiniteLine(0.0, angle=0, pen=pg.mkPen(colors.faint, width=1)))
        curve = p.plot(x, exact, pen=pg.mkPen(colors.strong, width=2))
        self.shape_legend.addItem(curve, "Exact")
        notes = []
        reach = 1.0
        for s in self._shown():
            model = self.models[s]
            hz, shapes = model.modes()
            i = n - 1 + c.rigid_modes
            if i >= hz.size:
                continue
            shape = np.array(shapes[:, i])
            nodes = model.deflections(shape)
            sign = np.sign(nodes @ c.exact_shape(k, model.nodes)) or 1.0
            color = scheme_color(s)
            field = sign * model.field(shape, x)
            reach = max(reach, float(np.max(np.abs(field))))
            line = p.plot(x, field, pen=pg.mkPen(color, width=1.5))
            p.plot(model.nodes, sign * nodes, pen=None, symbol="o", symbolSize=7, symbolBrush=color,
                   symbolPen=pg.mkPen(colors.background))
            self.shape_legend.addItem(line, s.value)
            at = c.exact_shape(k, model.nodes)
            mac = (nodes @ at)**2 / ((nodes @ nodes) * (at @ at))
            notes.append(f"{s.value.lower()} {mac:.6f}")
        p.setXRange(0.0, c.length, padding=0.02)
        p.setYRange(-1.1 * reach, 1.65 * reach, padding=0.0)  # room for the legend at the top
        per = 2 * math.pi / (k * c.length / self.elements.value())
        self.shape_note.setText(
            f"Mode {n}, {per:.3g} elements a wavelength. The MAC of each mesh's nodes against the exact shape "
            "there (1: the same shape): " + ", ".join(notes) + "." if notes else f"Mode {n}: no mesh shown has it.")


def _fmt_time(t: float) -> str:
    return f"{t * 1e3:.3g} ms" if t < 1.0 else f"{t:.3g} s"


def _cell(text: str) -> QtWidgets.QTableWidgetItem:
    item = QtWidgets.QTableWidgetItem(text)
    item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
    return item


def _holder(layout: QtWidgets.QLayout) -> QtWidgets.QWidget:
    w = QtWidgets.QWidget()
    w.setLayout(layout)
    return w
