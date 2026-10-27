"""Torsional vibration page: a drive train of discs, shafts and gear meshes, its torsional criticals and forced
response, and a motor starting it.

Left: the preset, the train (pick a disc or a link to edit it, or click it in the drawing),
the excitation, the speeds and the start-up's motor and load. Centre, two tabs: the
interference (Campbell) diagram, natural frequencies against the orders that excite them,
with the steady dynamic torque in a link against the motor's speed; and the start-up, the
shafts' torque and the motor's speed against time as it runs. Right: the train drawn, its
mode shapes, and the notes. The page has its own model (core/torsional.py and
core/torsional_startup.py), not the chain.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ..core.torsional import (
    PRESETS,
    Disc,
    DriveTrain,
    Excitation,
    Mesh,
    Shaft,
    criticals,
    forced_response,
    modes,
    natural_hz,
    order_lines,
    response_speeds,
    separation,
    sources,
)
from ..core.torsional_startup import (
    FAN_START,
    STARTUPS,
    Motor,
    MotorKind,
    StartUp,
    StartUpSimulator,
    crossings,
    running_torques,
    steady_speed,
)
from . import frame_stats, settle_plots
from .axes import log_axes
from .layouts import ColumnScroll, FlowLayout, group, plot_height
from .links import before, takeaways
from .panels import spin
from .style import colors
from .theming import add_legend, mute

CUSTOM = "Custom"
MAX_DISCS = 12
SHAPES_SHOWN = 6  # modes drawn on the shape plot
MARGIN = 10.0  # % either side of the running speed a critical should stay out of (API 617's 10 %)
FRAME_MS = 16
FRAME_BUDGET = 0.008  # s of computing per frame: past it the start-up plays slower than asked
PLAYBACK = (0.05, 0.1, 0.25, 0.5, 1.0)  # × real time
MOTORS = (None, MotorKind.INDUCTION, MotorKind.SYNCHRONOUS)  # the motor combo's items

TIPS = {
    "inertia": "<p>The disc's polar moment of inertia J about its shaft.</p>",
    "disc_damping": "<p>A damper from the disc to ground: a fan's or pump's load, which rises with speed "
                    "(its slope dT/dΩ), or windage. It damps every mode the disc swings in.</p>",
    "stiffness": "<p>The shaft's (or coupling's) torsional stiffness: torque per radian of twist, GJ/L for a "
                 "plain shaft.</p>",
    "shaft_damping": "<p>The shaft's damping: torque per rad/s of twist rate. A rubber coupling has a lot; "
                     "steel very little.</p>",
    "teeth": "<p>Teeth on the gear on the motor's side (in) and on the far side (out). The far shaft turns "
             "at z<sub>in</sub>/z<sub>out</sub> times the near one's speed: more teeth out slows it down.</p>",
    "base_radius": "<p>The base radius of the gear on the motor's side; the far gear's is r z<sub>out</sub> / "
                   "z<sub>in</sub>.</p>",
    "mesh_stiffness": "<p>The mesh's stiffness along the line of action: the teeth bending and the contact "
                      "flattening, about 14 N/µm per mm of face width for steel spur gears.</p>",
    "mesh_damping": "<p>The mesh's damping along the line of action.</p>",
    "error": "<p>The transmission error: how far the far gear lags or leads where perfect involutes would put "
             "it, here one harmonic at the tooth-pass frequency (teeth × speed). It drives the mesh as an "
             "imposed stretch.</p>",
    "excite": "<p>A pulsating torque on one disc, each harmonic an order of that disc's own speed: an "
              "engine's firing (a four-stroke six's majors are 3× and 6×), a reciprocating compressor's "
              "strokes, a misaligned coupling's 2×.</p>",
    "operating": "<p>The motor's running speed: the interference diagram and the response mark a band of "
                 "±10 % round it, where no torsional critical should be.</p>",
    "top": "<p>The highest motor speed the diagram and the response show.</p>",
    "motor": "<p>The motor on the first disc that starts the train from rest: an induction motor direct on "
             "line, or a synchronous motor started on its damper cage. None: no start-up.</p>",
    "supply": "<p>The line frequency and the motor's pole pairs: it runs at (or, an induction motor, just "
              "below) 60 f / p rpm.</p>",
    "breakdown": "<p>The cage's speed–torque curve (Kloss's): its peak torque, and the slip where it peaks. "
                 "It starts the train, and an induction motor runs on it, where it meets the load.</p>",
    "transient": "<p>Switching an induction motor on gives a torque at the line frequency on top of its "
                 "curve's, dying away with this time constant: the stator's flux starts with a DC part.</p>",
    "pulsating": "<p>A salient-pole synchronous motor's torque pulses at twice the slip frequency as it starts "
                 "(T<sub>p</sub> sin 2δ): from twice the line frequency at rest to nothing at synchronism, "
                 "sweeping through the modes below.</p>",
    "field": "<p>The field's pull-out torque (T<sub>f</sub> sin δ), switched on below the slip given, which "
             "pulls the rotor into step.</p>",
    "load": "<p>The load on the last disc, rising as its speed squared (a fan's, a centrifugal "
            "compressor's) to this, on its own shaft, with the motor at its synchronous speed.</p>",
    "duration": "<p>How long the start-up runs, in simulated time.</p>",
}
INTERFERENCE_TIP = (
    "<p>The interference (Campbell) diagram. The flat lines are the natural frequencies: they don't change "
    "with speed. Each line through the origin is an order: an excitation at that multiple of the motor's "
    "speed (solid: the train's own excitation and tooth-pass; dashed: 1× and 2× of each shaft's speed, which "
    "a standard asks to check). Where an order crosses a mode is a torsional critical speed; filled red "
    "inside the band round the running speed.</p>")
STARTUP_TIP = (
    "<p>The start-up: the chosen link's torque (a mesh: its force times the near gear's base radius) and the "
    "motor's air-gap torque against time, the dashed line what the link carries running steadily; below, "
    "the motor's speed, with the speeds where a synchronous motor's 2× slip pulsation meets each mode.</p>")
RESPONSE_TIP = (
    "<p>The steady dynamic torque in the chosen link (a mesh: its force times the near gear's base radius), "
    "the amplitude of each order on its own and their sum (the peak when they line up), against the motor's "
    "speed. Each peaks where its order meets a mode the link twists in.</p>")

NOTES_HTML = """
<!--BEFORE-->
<h3>Torsional vibration</h3>
<p>A drive train turns as a whole, but its discs also twist back and forth against each other on the shafts
between them. Each disc i has an angle θ<sub>i</sub> on top of the steady turning, and</p>
<p align="center">M θ″ + C θ′ + K θ = T(t),</p>
<p>with M the discs' inertias J on the diagonal, K each shaft's stiffness k across the pair it joins, and T
the pulsating torques. It is the chain of masses and springs again, turned on its side: the train is free at
both ends, so its first mode is the whole train turning (0 Hz), and the rest twist it. Each mode's
<b>twist</b> says which links carry its strain energy: those are the shafts that see its torque.</p>
<p>Two discs on one shaft have one mode, f = √(k (J₁ + J₂) / J₁J₂) / 2π, with a node (a point that doesn't
swing) nearer the heavier disc.</p>

<h3><a name="gears"></a>Gears</h3>
<p>A gear pair turns the far shaft at z<sub>in</sub>/z<sub>out</sub> times the near one's speed. Each gear keeps
its own inertia, and the teeth are a spring k<sub>m</sub> along the line of action: its stretch is
r<sub>in</sub>θ<sub>in</sub> − r<sub>out</sub>θ<sub>out</sub>, with r the base radii. Referred to the motor's
shaft, θ<sub>i</sub> = s<sub>i</sub>ψ<sub>i</sub> with s<sub>i</sub> the speed ratio, every inertia and
stiffness is multiplied by s², and the mesh becomes a shaft of k<sub>m</sub>(r s)²: the geared train is a
plain chain, with the same frequencies. That is the textbook <b>n² reflection</b>.</p>
<p>The mesh is far stiffer than the couplings, so its own mode, the two gears bouncing on their teeth, is
usually kilohertz. Treat the mesh as rigid and the two gears merge into one inertia: the lower modes barely
move, but that mode is gone (the <i>Mode shapes</i> tab compares). It matters because the teeth excite it:
no gear is perfect, and the <b>transmission error</b>, the far gear's lag behind where perfect teeth would put
it, repeats at every tooth, at z × the shaft's speed (the tooth-pass frequency).</p>

<h3><a name="interference"></a>The interference diagram</h3>
<p>Nothing in the train depends on its speed, so its natural frequencies are flat lines. What changes with
speed is the excitation: an engine fires at fixed orders of its speed (a four-stroke engine's at half-orders,
a six-cylinder's majors at 3× and 6×), a reciprocating compressor at 1× and 2× its own, a misaligned coupling
at 2×, the teeth at the tooth-pass order. Each is a line f = order × speed. Where it crosses a natural
frequency is a <b>torsional critical speed</b>: the order there drives the mode at its resonance.</p>
<p>A design keeps every critical out of a band round the running speed: API 617 asks for 10 % from 1× and 2× of
each shaft's speed. A start-up still passes the criticals below, briefly. Not every crossing matters: an
order at a disc the mode barely moves (near its node) hardly excites it, and the response tells which do.</p>

<h3>The steady response</h3>
<p>At each speed each order is a harmonic torque at ω = order × Ω, and the steady response is one complex
solve, (K − ω²M + iωC) Θ = T. A shaft's dynamic torque is (k + iωc) times its twist. The transmission error
enters as an imposed stretch e on the mesh: a force (k<sub>m</sub> + iωc<sub>m</sub>) e on the gears, equal
and opposite. The orders are at different frequencies, so their peaks line up only now and then: the sum of
the amplitudes is the worst the torque can be.</p>
<p>The damping ratios come from φᵀCφ / 2ωφᵀMφ, each mode's share of the dampers: exact when the damping is
light and spread over the train.</p>

<h3>Starting up</h3>
<p>The <i>Start-up</i> tab starts the train from rest with a motor on its first disc and a load on its last that
rises as its speed squared. The angles are then the whole turn, and the train's free turning (the 0 Hz mode)
carries it up to speed while the shafts twist on top. The motor's torque depends on the motor's angle and speed
and the load's on its speed, so each step solves for them at its end by Newton, on the exact FOH update of the
linear train.</p>
<p>An <b>induction motor</b> follows its speed–torque curve, here Kloss's, T = 2T<sub>b</sub> s s<sub>b</sub> /
(s² + s<sub>b</sub>²), from the slip s = 1 − Ω/Ω<sub>s</sub>. Direct on line, it also gives a torque at the line
frequency for the first few cycles (the stator's flux starts with a DC part the rotor's currents pull against):
a mode near the line frequency is struck by it at once, while the train has barely moved, and the shafts can
see many times their running torque.</p>
<p>A <b>synchronous motor</b> starts as an induction motor on its damper cage, and its salient poles add a
torque at twice the slip frequency, 2 s f. That is twice the line frequency at rest and nothing at synchronism,
so it sweeps down through every mode below 2f on the way up: each is a resonance passed at the speed where
2 s f = f<sub>n</sub>. Near synchronism the field is switched on and pulls the rotor into step. A fast start
passes the resonances before they build up; a slow one, against a heavy load, lets them grow: that is why such
drives get a torsional analysis of the start, and couplings sized for its peaks.</p>
<p>The tab compares each link's peak torque with the torque it carries in steady running: the power drawn by
the load and the dampers beyond it, divided by the link's speed.</p>
<!--TAKEAWAYS-->
"""
NOTES_HTML = NOTES_HTML.replace("<!--BEFORE-->", before(
    ("sim_page", "equations-of-motion", "Equations of motion"),
    ("sim_page", "undamped-normal-modes-real", "Undamped normal modes"),
    ("frf_page", "receptance-matrix", "The FRF (receptance) matrix"),
)).replace("<!--TAKEAWAYS-->", takeaways(
    "A drive train in torsion is the chain again, with inertias and shaft stiffnesses, free at both ends so "
    "its first mode is rigid turning.",
    "Gears refer to one shaft by the square of the speed ratio; a stiff mesh has its own mode, excited at "
    "tooth-pass.",
    "Criticals are where an excitation order's line crosses a natural frequency; keep them out of a band "
    "round the running speed.",
    "A start-up sweeps the excitations through the modes below running speed: a slow start lets them build up.",
))



def fmt_torque(t: float) -> str:
    return f"{t / 1e3:.3g} kN·m" if abs(t) >= 1e3 else f"{t:.3g} N·m"


def _plot(left: str, left_units: str | None, bottom: str, bottom_units: str | None) -> pg.PlotWidget:
    p = pg.PlotWidget(axisItems=log_axes())
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


class TorsionalPage(QtWidgets.QWidget):
    """Controls on the left, the interference diagram and the response in the centre, drawing and shapes on the right."""

    follows_chain = False  # MainWindow: this page has its own model, not the chain

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.train: DriveTrain = PRESETS[0].train
        self.selected = 0  # into elements: 2i a disc, 2i + 1 the link after it
        self._loading = False
        self._modes = []
        self._unreached = 0  # modes above every order in the speed range, left off the interference diagram
        self.startup: StartUp = FAN_START
        self.motor_on = True
        self.sim: StartUpSimulator | None = None
        self.running = False
        self._startup_note = ""  # the preset's, until something is edited

        # --- left: the preset, the train, the excitation, the speeds
        self.preset = QtWidgets.QComboBox()
        self.preset.addItems([p.name for p in PRESETS] + [CUSTOM])
        self.preset.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.preset_note = QtWidgets.QLabel()
        self.preset_note.setWordWrap(True)
        mute(self.preset_note)

        train_box = QtWidgets.QGroupBox("Drive train")
        tv = QtWidgets.QVBoxLayout(train_box)
        self.element = QtWidgets.QComboBox()
        self.element.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.element.setToolTip("<p>The disc or link to edit: or click it in the <i>Drive train</i> drawing.</p>")
        tv.addWidget(self.element)
        self.editor = QtWidgets.QStackedWidget()
        tv.addWidget(self.editor)

        disc_page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(disc_page)
        form.setContentsMargins(0, 0, 0, 0)
        self.disc_name = QtWidgets.QLineEdit()
        self.inertia = spin(1e-5, 1e5, 1.0, 4, " kg·m²")
        self.disc_damping = spin(0.0, 1e6, 0.0, 2, " N·m·s/rad")
        _form_row(form, "Name:", self.disc_name, "<p>What the disc is called.</p>")
        _form_row(form, "Inertia J:", self.inertia, TIPS["inertia"])
        _form_row(form, "Damping to ground:", self.disc_damping, TIPS["disc_damping"])
        self.editor.addWidget(disc_page)

        shaft_page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(shaft_page)
        form.setContentsMargins(0, 0, 0, 0)
        self.shaft_name = QtWidgets.QLineEdit()
        self.shaft_stiffness = spin(1e-3, 1e7, 100.0, 3, " kN·m/rad")
        self.shaft_damping = spin(0.0, 1e6, 0.0, 2, " N·m·s/rad")
        _form_row(form, "Name:", self.shaft_name, "<p>What the shaft or coupling is called.</p>")
        _form_row(form, "Stiffness k:", self.shaft_stiffness, TIPS["stiffness"])
        _form_row(form, "Damping c:", self.shaft_damping, TIPS["shaft_damping"])
        self.to_mesh = QtWidgets.QPushButton("Make it a gear mesh")
        self.to_mesh.setToolTip("<p>Replace this shaft by a gear pair: the discs either side become the gears.</p>")
        form.addRow(self.to_mesh)
        self.editor.addWidget(shaft_page)

        mesh_page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(mesh_page)
        form.setContentsMargins(0, 0, 0, 0)
        self.mesh_name = QtWidgets.QLineEdit()
        self.teeth_in = QtWidgets.QSpinBox()
        self.teeth_out = QtWidgets.QSpinBox()
        for box in (self.teeth_in, self.teeth_out):
            box.setRange(5, 1000)
            box.setKeyboardTracking(False)
        self.base_radius = spin(1.0, 5000.0, 100.0, 1, " mm")
        self.mesh_stiffness = spin(1.0, 1e5, 1000.0, 1, " N/µm")
        self.mesh_damping = spin(0.0, 1e7, 0.0, 0, " N·s/m")
        self.error = spin(0.0, 1000.0, 0.0, 2, " µm")
        _form_row(form, "Name:", self.mesh_name, "<p>What the mesh is called.</p>")
        _form_row(form, "Teeth:", group(self.teeth_in, "in,", self.teeth_out, "out"), TIPS["teeth"])
        _form_row(form, "Base radius (in):", self.base_radius, TIPS["base_radius"])
        _form_row(form, "Mesh stiffness:", self.mesh_stiffness, TIPS["mesh_stiffness"])
        _form_row(form, "Mesh damping:", self.mesh_damping, TIPS["mesh_damping"])
        _form_row(form, "Transmission error:", self.error, TIPS["error"])
        self.to_shaft = QtWidgets.QPushButton("Make it a shaft")
        self.to_shaft.setToolTip("<p>Replace the gear pair by a shaft of the mesh's referred stiffness "
                                 "(the far side then turns at the near side's speed).</p>")
        form.addRow(self.to_shaft)
        self.editor.addWidget(mesh_page)

        row = FlowLayout()
        self.add_button = QtWidgets.QPushButton("Add a disc after")
        self.add_button.setToolTip("<p>A copy of the chosen disc (or of the disc before the chosen link) after it, "
                                   "on a copy of its shaft.</p>")
        self.remove_button = QtWidgets.QPushButton("Remove disc")
        self.remove_button.setToolTip("<p>Take the chosen disc off the train, with the link after it.</p>")
        row.add(self.add_button, self.remove_button)
        tv.addLayout(row)
        self.train_info = QtWidgets.QLabel()
        self.train_info.setWordWrap(True)
        mute(self.train_info)
        tv.addWidget(self.train_info)

        excite = QtWidgets.QGroupBox("Excitation")
        excite.setToolTip(TIPS["excite"])
        ef = QtWidgets.QFormLayout(excite)
        self.excite_disc = QtWidgets.QComboBox()
        self.excite_disc.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.excite_name = QtWidgets.QLineEdit()
        self.orders = [spin(0.0, 500.0, 0.0, 2, "×") for _ in range(2)]
        self.amplitudes = [spin(0.0, 1e7, 0.0, 1, " N·m") for _ in range(2)]
        _form_row(ef, "Torque on:", self.excite_disc, TIPS["excite"])
        _form_row(ef, "Called:", self.excite_name, "<p>What the excitation is (its orders' label).</p>")
        for k in range(2):
            _form_row(ef, f"Harmonic {k + 1}:", group(self.orders[k], self.amplitudes[k]),
                      "<p>An order of the disc's own speed and its torque's amplitude (0: off).</p>")

        speeds = QtWidgets.QGroupBox("Speeds")
        sf = QtWidgets.QFormLayout(speeds)
        self.operating = spin(1.0, 1e5, 1480.0, 0, " rpm")
        self.top = spin(10.0, 1e5, 3600.0, 0, " rpm")
        _form_row(sf, "Running speed:", self.operating, TIPS["operating"])
        _form_row(sf, "Show up to:", self.top, TIPS["top"])
        self.log_scales = QtWidgets.QCheckBox("Log scales on the interference diagram")
        self.log_scales.setToolTip("<p>Both axes logarithmic: every order is then a parallel line, and modes "
                                   "decades apart (a mesh's and the couplings') all show. On by itself when "
                                   "a train loads whose modes span more than a decade.</p>")
        sf.addRow(self.log_scales)

        start_box = QtWidgets.QGroupBox("Start-up")
        stf = QtWidgets.QFormLayout(start_box)
        self.motor_kind = QtWidgets.QComboBox()
        self.motor_kind.addItems(["None"] + [k.value for k in MOTORS[1:]])
        self.line_hz = spin(1.0, 400.0, 50.0, 1, " Hz")
        self.pole_pairs = QtWidgets.QSpinBox()
        self.pole_pairs.setRange(1, 50)
        self.pole_pairs.setKeyboardTracking(False)
        self.breakdown = spin(1.0, 1e7, 1000.0, 0, " N·m")
        self.breakdown_slip = spin(0.5, 100.0, 20.0, 1, " %")
        _form_row(stf, "Motor:", self.motor_kind, TIPS["motor"])
        _form_row(stf, "Supply:", group(self.line_hz, self.pole_pairs, "pole pairs"), TIPS["supply"])
        _form_row(stf, "Breakdown:", group(self.breakdown, "at slip", self.breakdown_slip), TIPS["breakdown"])
        self.motor_stack = QtWidgets.QStackedWidget()
        induction_page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(induction_page)
        form.setContentsMargins(0, 0, 0, 0)
        self.transient = spin(0.0, 1e7, 0.0, 0, " N·m")
        self.time_constant = spin(1.0, 5000.0, 50.0, 0, " ms")
        _form_row(form, "Switch-on torque:", group(self.transient, "decay", self.time_constant), TIPS["transient"])
        self.motor_stack.addWidget(induction_page)
        synchronous_page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(synchronous_page)
        form.setContentsMargins(0, 0, 0, 0)
        self.pulsating = spin(0.0, 1e7, 0.0, 0, " N·m")
        self.field = spin(0.0, 1e7, 0.0, 0, " N·m")
        self.field_slip = spin(0.1, 100.0, 5.0, 1, " %")
        _form_row(form, "Pulsating (2× slip):", self.pulsating, TIPS["pulsating"])
        _form_row(form, "Field:", group(self.field, "below slip", self.field_slip), TIPS["field"])
        self.motor_stack.addWidget(synchronous_page)
        stf.addRow(self.motor_stack)
        self.load_torque = spin(0.0, 1e7, 300.0, 0, " N·m")
        self.duration = spin(0.1, 60.0, 4.0, 1, " s")
        _form_row(stf, "Load at full speed:", self.load_torque, TIPS["load"])
        _form_row(stf, "Run for:", self.duration, TIPS["duration"])
        row = FlowLayout()
        self.start_button = QtWidgets.QPushButton("Start")
        self.start_button.setToolTip("<p>Start the train from rest (again).</p>")
        self.pause_button = QtWidgets.QPushButton("Pause")
        self.playback = QtWidgets.QComboBox()
        for x in PLAYBACK:
            self.playback.addItem(f"{x:g}× real time", x)
        self.playback.setCurrentIndex(PLAYBACK.index(0.25))
        self.playback.setToolTip("<p>How fast the start-up plays: its first tenth of a second is what an "
                                 "induction motor's switch-on rings in.</p>")
        row.add(self.start_button, self.pause_button, self.playback)
        stf.addRow(row)
        self.startup_info = QtWidgets.QLabel()
        self.startup_info.setWordWrap(True)
        mute(self.startup_info)
        stf.addRow(self.startup_info)

        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        presets = QtWidgets.QGroupBox("Things to try")
        preset_layout = QtWidgets.QVBoxLayout(presets)
        preset_layout.addWidget(self.preset)
        preset_layout.addWidget(self.preset_note)
        lv.addWidget(presets)
        for box in (train_box, excite, speeds, start_box):
            lv.addWidget(box)
        lv.addStretch(1)
        left_scroll = ColumnScroll(left)

        # --- centre: the interference diagram, the readout, the response
        self.interference = _plot("Frequency", "Hz", "Motor speed", "rpm")
        self.interference.setToolTip(INTERFERENCE_TIP)
        plot_height(self.interference, 220)
        self.readout = QtWidgets.QLabel()
        self.readout.setWordWrap(True)
        self.readout.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.response_link = QtWidgets.QComboBox()
        self.response_link.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.response_link.setMinimumContentsLength(18)
        self.response_link.setToolTip(RESPONSE_TIP)
        self.response = _plot("Dynamic torque", "N·m", "Motor speed", "rpm")
        self.response.setLogMode(x=False, y=True)
        self.response.setToolTip(RESPONSE_TIP)
        plot_height(self.response, 180)
        self.response_legend = add_legend(self.response.getPlotItem(), offset=(-10, 10))  # top right
        self.response_note = QtWidgets.QLabel()
        self.response_note.setWordWrap(True)
        self.response_note.setTextFormat(QtCore.Qt.TextFormat.RichText)
        mute(self.response_note)
        center = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(center)
        cv.addWidget(self.interference, 3)
        cv.addWidget(self.readout)
        row = FlowLayout()
        row.add(group("Dynamic torque in:", self.response_link))
        cv.addLayout(row)
        cv.addWidget(self.response, 2)
        cv.addWidget(self.response_note)

        # the start-up: a link's torque and the motor's speed against time
        self.startup_link = QtWidgets.QComboBox()
        self.startup_link.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.startup_link.setMinimumContentsLength(18)
        self.startup_link.setToolTip(STARTUP_TIP)
        self.startup_torque = _plot("Torque", "N·m", "Time", "s")
        self.startup_torque.setToolTip(STARTUP_TIP)
        plot_height(self.startup_torque, 200)
        self.startup_legend = add_legend(self.startup_torque.getPlotItem(), offset=(-10, 10))
        self.startup_speed = _plot("Motor speed", "rpm", "Time", "s")
        self.startup_speed.setToolTip(STARTUP_TIP)
        self.startup_speed.setXLink(self.startup_torque)
        plot_height(self.startup_speed, 130)
        for p in (self.startup_torque, self.startup_speed):
            p.setClipToView(True)
            p.setDownsampling(auto=True, mode="peak")
        self.motor_curve = self.startup_torque.plot([], [])
        self.link_curve = self.startup_torque.plot([], [])
        self.startup_legend.addItem(self.link_curve, "Link")
        self.startup_legend.addItem(self.motor_curve, "Motor's air-gap torque")
        self.running_line = pg.InfiniteLine(0.0, angle=0)
        self.startup_torque.addItem(self.running_line)
        self.speed_curve = self.startup_speed.plot([], [])
        self._marks: list[tuple[pg.PlotWidget, pg.InfiniteLine]] = []  # the crossings' speeds, when the run passed them
        self._style_startup()
        self.startup_status = QtWidgets.QLabel()
        self.startup_status.setWordWrap(True)
        mute(self.startup_status)
        self.startup_note = QtWidgets.QLabel()
        self.startup_note.setWordWrap(True)
        self.startup_note.setTextFormat(QtCore.Qt.TextFormat.RichText)
        startup_tab = QtWidgets.QWidget()
        uv = QtWidgets.QVBoxLayout(startup_tab)
        row = FlowLayout()
        row.add(group("Torque in:", self.startup_link))
        uv.addLayout(row)
        uv.addWidget(self.startup_torque, 3)
        uv.addWidget(self.startup_speed, 2)
        uv.addWidget(self.startup_status)
        uv.addWidget(self.startup_note)
        self.views = QtWidgets.QTabWidget()
        self.criticals_tab = center
        self.startup_tab = startup_tab
        self.views.addTab(center, "Criticals")
        self.views.addTab(startup_tab, "Start-up")

        # --- right: the train drawn, the mode shapes, the notes
        self.drawing = pg.PlotWidget()
        self.drawing.setMenuEnabled(False)
        self.drawing.hideButtons()
        self.drawing.hideAxis("left")
        self.drawing.hideAxis("bottom")
        self.drawing.setMouseEnabled(x=False, y=False)
        self.drawing.setAspectLocked(True)
        self.drawing.scene().sigMouseClicked.connect(self._on_drawing_click)
        self.drawing.setToolTip("<p>The train, motor on the left: discs (taller the more inertia), shafts, and "
                                "gear pairs, the far shaft stepped down. Click a disc or a link to edit it (the list above the editor "
                                "names the links).</p>")
        self.drawing_note = QtWidgets.QLabel()
        self.drawing_note.setWordWrap(True)
        mute(self.drawing_note)
        drawing_tab = QtWidgets.QWidget()
        dv = QtWidgets.QVBoxLayout(drawing_tab)
        dv.addWidget(self.drawing, 1)
        dv.addWidget(self.drawing_note)

        self.shapes = _plot("Swing (referred to the motor's shaft)", None, "Disc", None)
        self.shapes.setMouseEnabled(x=False, y=False)
        self.shapes.setToolTip("<p>Each mode's swing at each disc, referred to the motor's shaft (θ / s, so a "
                               "rigid turn of the whole train would be flat), largest +1. A link the curve "
                               "crosses zero over holds a node, and twists.</p>")
        self.shapes_legend = add_legend(self.shapes.getPlotItem(), offset=(10, 10))
        self.mode_list = QtWidgets.QLabel()
        self.mode_list.setWordWrap(True)
        self.mode_list.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.check = QtWidgets.QLabel()
        self.check.setWordWrap(True)
        self.check.setTextFormat(QtCore.Qt.TextFormat.RichText)
        mute(self.check)
        shapes_tab = QtWidgets.QWidget()
        sv = QtWidgets.QVBoxLayout(shapes_tab)
        sv.addWidget(self.shapes, 1)
        sv.addWidget(self.mode_list)
        sv.addWidget(self.check)

        self.notes = QtWidgets.QTextBrowser()
        self.notes.setHtml(NOTES_HTML)
        self.tabs = QtWidgets.QTabWidget()
        self.drawing_tab = drawing_tab
        self.shapes_tab = shapes_tab
        self.tabs.addTab(drawing_tab, "Drive train")
        self.tabs.addTab(shapes_tab, "Mode shapes")
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
        self.element.currentIndexChanged.connect(self._on_element)
        for box in (self.inertia, self.disc_damping, self.shaft_stiffness, self.shaft_damping, self.base_radius,
                    self.mesh_stiffness, self.mesh_damping, self.error, self.teeth_in, self.teeth_out):
            box.valueChanged.connect(self._on_edit)
        for line in (self.disc_name, self.shaft_name, self.mesh_name):
            line.editingFinished.connect(self._on_edit)
        self.to_mesh.clicked.connect(self._make_mesh)
        self.to_shaft.clicked.connect(self._make_shaft)
        self.add_button.clicked.connect(self._add_disc)
        self.remove_button.clicked.connect(self._remove_disc)
        self.excite_disc.currentIndexChanged.connect(self._on_excitation)
        self.excite_name.editingFinished.connect(self._on_excitation)
        for box in (*self.orders, *self.amplitudes):
            box.valueChanged.connect(self._on_excitation)
        self.operating.valueChanged.connect(self._on_speeds)
        self.top.valueChanged.connect(self._on_speeds)
        self.log_scales.toggled.connect(lambda _: self._draw_interference())
        self.response_link.currentIndexChanged.connect(lambda _: self._draw_response())
        self.motor_kind.currentIndexChanged.connect(self._on_startup)
        for box in (self.line_hz, self.pole_pairs, self.breakdown, self.breakdown_slip, self.transient,
                    self.time_constant, self.pulsating, self.field, self.field_slip, self.load_torque,
                    self.duration):
            box.valueChanged.connect(self._on_startup)
        self.start_button.clicked.connect(self.start)
        self.pause_button.clicked.connect(self._on_pause)
        self.startup_link.currentIndexChanged.connect(lambda _: self._draw_startup(full=True))

        # --- the start-up's loop
        self._clock = QtCore.QElapsedTimer()
        self._last = 0.0
        self._cost = 1e-5  # s of computing per simulated s (see step)
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._frame)

        self.load_preset(PRESETS[0].name)

    # ------------------------------------------------------------- inputs
    def load_preset(self, name: str) -> None:
        p = next(p for p in PRESETS if p.name == name)
        self._loading = True
        self.operating.setValue(p.operating_rpm)
        self.top.setValue(p.top_rpm)
        self.preset.blockSignals(True)
        self.preset.setCurrentText(name)
        self.preset.blockSignals(False)
        self.preset_note.setText(p.note)
        self._loading = False
        self.selected = 0
        self.response_link.blockSignals(True)
        self.response_link.setCurrentIndex(-1)  # set_train picks the slowest mode's link
        self.response_link.blockSignals(False)
        f = natural_hz(p.train)
        self.log_scales.blockSignals(True)
        self.log_scales.setChecked(bool(f.size and f[-1] > 10 * f[0]))
        self.log_scales.blockSignals(False)
        start = STARTUPS.get(name)
        self.motor_on = start is not None
        if start is not None:
            self.startup = start.startup
        self._startup_note = (start.note if start is not None else
                              "The engine drives this train: no motor starts it. Pick one to try anyway.")
        self.running = False
        self.startup_link.blockSignals(True)
        self.startup_link.setCurrentIndex(-1)  # _new_run picks the link carrying the most
        self.startup_link.blockSignals(False)
        self._fill_startup()
        self.set_train(p.train, custom=False)

    def frame_context(self) -> str:
        """For the frame log (frame_stats): the preset showing."""
        return self.preset.currentText()

    def _on_preset(self, index: int) -> None:
        if index < len(PRESETS):
            self.load_preset(PRESETS[index].name)

    def _to_custom(self) -> None:
        self.preset.blockSignals(True)
        self.preset.setCurrentText(CUSTOM)
        self.preset.blockSignals(False)
        self.preset_note.setText("")
        if self._startup_note:
            self._startup_note = ""
            self._show_startup_info()

    def set_train(self, train: DriveTrain, custom: bool = True) -> None:
        """Show a new train: every view redrawn, the editors refilled."""
        self.train = train
        self._modes = modes(train)
        self.selected = min(self.selected, 2 * train.n - 2)
        if custom:
            self._to_custom()
        self._fill_lists()
        self._fill_editor()
        self._show_train_info()
        self.redraw()
        self._new_run()

    def _try(self, make) -> None:
        """Apply the train `make()` builds, or say why it can't be built and keep the last one."""
        try:
            train = make()
        except ValueError as e:
            self.train_info.setText(f"<b>{e}</b>: the page keeps the last train.")
            return
        self.set_train(train)

    def _fill_lists(self) -> None:
        self._loading = True
        t = self.train
        self.element.clear()
        for i, d in enumerate(t.discs):
            self.element.addItem(f"{i + 1}. {d.name}")
            if i < len(t.links):
                link = t.links[i]
                kind = (f"mesh {link.teeth_in}:{link.teeth_out}" if isinstance(link, Mesh) else "shaft")
                self.element.addItem(f"    {link.name} ({kind})")
        self.element.setCurrentIndex(self.selected)
        link = self.response_link.currentIndex()
        self.response_link.clear()
        self.response_link.addItems([link.name for link in t.links])
        if not 0 <= link < len(t.links):
            # the link the slowest mode twists most, by default
            link = self._modes[0].link if self._modes else 0
        self.response_link.setCurrentIndex(link)
        self.excite_disc.clear()
        self.excite_disc.addItem("None")
        self.excite_disc.addItems([d.name for d in t.discs])
        exc = t.excitation
        self.excite_disc.setCurrentIndex(0 if exc is None else exc.disc + 1)
        harmonics = list(exc.harmonics) if exc is not None else []
        harmonics += [(0.0, 0.0)] * (2 - len(harmonics))
        for k in range(2):
            self.orders[k].setValue(harmonics[k][0])
            self.amplitudes[k].setValue(harmonics[k][1])
        self.excite_name.setText(exc.name if exc is not None else "Torque")
        for w in (self.excite_name, *self.orders, *self.amplitudes):
            w.setEnabled(exc is not None)
        self.add_button.setEnabled(t.n < MAX_DISCS)
        self.remove_button.setEnabled(t.n > 2)
        self._loading = False

    def _fill_editor(self) -> None:
        self._loading = True
        t = self.train
        i, is_link = divmod(self.selected, 2)
        page = 0 if not is_link else 1 if isinstance(t.links[i], Shaft) else 2
        for k in range(self.editor.count()):  # the stack as tall as the page showing, not the tallest
            policy = QtWidgets.QSizePolicy.Policy.Preferred if k == page else QtWidgets.QSizePolicy.Policy.Ignored
            self.editor.widget(k).setSizePolicy(policy, policy)
        self.editor.setCurrentIndex(page)
        if not is_link:
            d = t.discs[i]
            self.disc_name.setText(d.name)
            self.inertia.setValue(d.inertia)
            self.disc_damping.setValue(d.damping)
        elif isinstance(t.links[i], Shaft):
            s = t.links[i]
            self.shaft_name.setText(s.name)
            self.shaft_stiffness.setValue(s.stiffness / 1e3)
            self.shaft_damping.setValue(s.damping)
        else:
            m = t.links[i]
            self.mesh_name.setText(m.name)
            self.teeth_in.setValue(m.teeth_in)
            self.teeth_out.setValue(m.teeth_out)
            self.base_radius.setValue(m.base_radius * 1e3)
            self.mesh_stiffness.setValue(m.stiffness / 1e6)
            self.mesh_damping.setValue(m.damping)
            self.error.setValue(m.error * 1e6)
        self._loading = False

    def select(self, element: int) -> None:
        """Edit element `element`: 2i the i-th disc, 2i + 1 the link after it."""
        self.selected = element
        self.element.blockSignals(True)
        self.element.setCurrentIndex(element)
        self.element.blockSignals(False)
        self._fill_editor()
        self._draw_train()

    def _on_element(self, index: int) -> None:
        if not self._loading and index >= 0:
            self.select(index)

    def _on_edit(self) -> None:
        if self._loading:
            return
        t = self.train
        i, is_link = divmod(self.selected, 2)
        if not is_link:
            disc = Disc(self.disc_name.text().strip() or t.discs[i].name, self.inertia.value(),
                        self.disc_damping.value())
            self._try(lambda: t.with_disc(i, disc))
        elif isinstance(t.links[i], Shaft):
            shaft = Shaft(self.shaft_stiffness.value() * 1e3, self.shaft_damping.value(),
                          self.shaft_name.text().strip() or t.links[i].name)
            self._try(lambda: t.with_link(i, shaft))
        else:
            mesh = Mesh(self.teeth_in.value(), self.teeth_out.value(), self.base_radius.value() / 1e3,
                        self.mesh_stiffness.value() * 1e6, self.mesh_damping.value(), self.error.value() / 1e6,
                        self.mesh_name.text().strip() or t.links[i].name)
            self._try(lambda: t.with_link(i, mesh))

    def _make_mesh(self) -> None:
        i = self.selected // 2
        shaft = self.train.links[i]
        # a 1:1 pair of 100 mm base radius whose referred stiffness is the shaft's
        r = 0.1
        mesh = Mesh(40, 40, r, shaft.stiffness / (r * r), shaft.damping / (r * r), 0.0, "Gear mesh")
        self._try(lambda: self.train.with_link(i, mesh))

    def _make_shaft(self) -> None:
        i = self.selected // 2
        mesh = self.train.links[i]
        r = mesh.base_radius
        self._try(lambda: self.train.with_link(i, Shaft(mesh.stiffness * r * r, mesh.damping * r * r, "Shaft")))

    def _add_disc(self) -> None:
        i = self.selected // 2
        self.selected = 2 * (i + 1)
        self._try(lambda: self.train.inserted(i))

    def _remove_disc(self) -> None:
        i = self.selected // 2
        self.selected = 2 * max(i - 1, 0)
        self._try(lambda: self.train.removed(i))

    def _on_excitation(self) -> None:
        if self._loading:
            return
        disc = self.excite_disc.currentIndex() - 1
        if disc < 0:
            self._try(lambda: self.train.with_(excitation=None))
            return
        harmonics = tuple((self.orders[k].value(), self.amplitudes[k].value()) for k in range(2))
        if self.train.excitation is None and not any(o > 0 and a > 0 for o, a in harmonics):
            harmonics = ((1.0, 100.0),)  # a fresh excitation: something to see
        name = self.excite_name.text().strip() or "Torque"
        self._try(lambda: self.train.with_(excitation=Excitation(disc, harmonics, name)))

    def _on_speeds(self) -> None:
        if self._loading:
            return
        self._to_custom()
        self.redraw()

    # ------------------------------------------------------------- start-up
    def _fill_startup(self) -> None:
        loading, self._loading = self._loading, True
        m = self.startup.motor
        self.motor_kind.setCurrentIndex(MOTORS.index(m.kind) if self.motor_on else 0)
        for box, value in ((self.line_hz, m.line_hz), (self.pole_pairs, m.pole_pairs), (self.breakdown, m.breakdown),
                           (self.breakdown_slip, 100.0 * m.breakdown_slip), (self.transient, m.transient),
                           (self.time_constant, 1e3 * m.time_constant), (self.pulsating, m.pulsating),
                           (self.field, m.field), (self.field_slip, 100.0 * m.field_slip),
                           (self.load_torque, self.startup.load), (self.duration, self.startup.duration)):
            box.setValue(value)
        page = 0 if m.kind is MotorKind.INDUCTION else 1
        for k in range(self.motor_stack.count()):  # as tall as the page showing
            policy = QtWidgets.QSizePolicy.Policy.Preferred if k == page else QtWidgets.QSizePolicy.Policy.Ignored
            self.motor_stack.widget(k).setSizePolicy(policy, policy)
        self.motor_stack.setCurrentIndex(page)
        for w in (self.line_hz, self.pole_pairs, self.breakdown, self.breakdown_slip, self.motor_stack,
                  self.load_torque, self.duration):
            w.setEnabled(self.motor_on)
        self._loading = loading

    def _on_startup(self) -> None:
        if self._loading:
            return
        kind = MOTORS[self.motor_kind.currentIndex()]
        try:
            motor = Motor(kind or self.startup.motor.kind, self.line_hz.value(), self.pole_pairs.value(),
                          self.breakdown.value(), self.breakdown_slip.value() / 100.0, self.transient.value(),
                          self.time_constant.value() / 1e3, self.pulsating.value(), self.field.value(),
                          self.field_slip.value() / 100.0)
            startup = StartUp(motor, self.load_torque.value(), self.duration.value())
        except ValueError as e:
            self.startup_info.setText(f"<b>{e}</b>: the start-up keeps the last motor.")
            return
        self.startup, self.motor_on = startup, kind is not None
        self._to_custom()
        self._fill_startup()
        self._draw_interference()
        self._new_run()

    def _new_run(self) -> None:
        """A fresh start-up of the train as it is now, from rest (still playing if it was)."""
        t, su = self.train, self.startup
        self.sim = StartUpSimulator(t, su) if self.motor_on else None
        cap = round(su.duration / self.sim.h) + 1 if self.sim is not None else 0
        self._k = 0
        self._t = np.empty(cap)
        self._rpm = np.empty(cap)
        self._motor = np.empty(cap)
        self._links = np.empty((t.n - 1, cap))
        self._running = running_torques(t, su) if self.motor_on else np.full(t.n - 1, math.nan)
        self._crossings = crossings(t, su.motor) if self.motor_on else []
        self._passed: set[int] = set()
        chosen = self.startup_link.currentIndex()
        self.startup_link.blockSignals(True)
        self.startup_link.clear()
        self.startup_link.addItems([link.name for link in t.links])
        if not 0 <= chosen < len(t.links):
            chosen = int(np.argmax(np.nan_to_num(self._running)))  # the link carrying the most
        self.startup_link.setCurrentIndex(chosen)
        self.startup_link.blockSignals(False)
        for plot, mark in self._marks:
            plot.removeItem(mark)
        self._marks = []
        for c in self._crossings:
            self._add_mark(self.startup_speed, c.rpm, 0, f"Mode {c.mode + 1}, {c.freq_hz:.3g} Hz")
        self.startup_torque.setXRange(0.0, su.duration, padding=0)
        self.startup_torque.enableAutoRange(axis="y")
        self.startup_speed.setYRange(0.0, 1.1 * su.motor.sync_rpm, padding=0)
        if self.sim is None:
            self.running = False
        self._draw_startup(full=True)
        self._show_startup_info()
        self._show_status()
        self._update_buttons()
        self._start_timer()

    def _add_mark(self, plot: pg.PlotWidget, pos: float, angle: int, label: str) -> None:
        line = pg.InfiniteLine(pos, angle=angle, label=label,
                               labelOpts={"position": 0.02 if angle == 0 else 0.95, "anchors": [(0, 1), (0, 1)]})
        plot.addItem(line)
        self._marks.append((plot, line))
        self._style_mark(line)

    def _style_mark(self, line: pg.InfiniteLine) -> None:
        line.setPen(pg.mkPen(colors.grey, width=1, style=QtCore.Qt.PenStyle.DotLine))
        line.label.setColor(colors.muted)

    def start(self) -> None:
        """Start the train from rest, and show it."""
        if not self.motor_on:
            return
        self.running = True
        self.views.setCurrentWidget(self.startup_tab)
        self._new_run()

    def _on_pause(self) -> None:
        if self.sim is None or self.sim.done:
            return
        self.running = not self.running
        self._update_buttons()
        self._start_timer()

    def _update_buttons(self) -> None:
        live = self.sim is not None and not self.sim.done
        self.start_button.setEnabled(self.motor_on)
        self.pause_button.setEnabled(live and (self.running or self._k > 0))
        self.pause_button.setText("Pause" if self.running else "Run")

    def showEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().showEvent(event)
        self._start_timer()

    def hideEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().hideEvent(event)
        self._timer.stop()

    def _start_timer(self) -> None:
        if self.isVisible() and self.running and not self._timer.isActive():
            self._clock.restart()
            self._last = 0.0
            self._timer.start(FRAME_MS)

    def _frame(self) -> None:
        began = self._clock.elapsed()
        now = began / 1000.0
        dt = min(now - self._last, 0.1)
        self._last = now
        with frame_stats.frame("Torsional vibration", self._sim_t, FRAME_MS):
            self.step(dt * float(self.playback.currentData()), FRAME_BUDGET)
            settle_plots(self)
        if self.isVisible() and self.running:  # the next frame FRAME_MS after this one began, not after it ended
            self._timer.start(max(1, FRAME_MS - (self._clock.elapsed() - began)))

    def _sim_t(self) -> float | None:
        return self.sim.t if self.sim is not None else None

    def step(self, duration: float, budget: float | None = None) -> None:
        """Run the start-up on by `duration` s of simulated time and redraw.

        With a `budget` (s of computing), no further than it allows at the cost seen so far, so a fast
        playback falls behind rather than stalling the window.
        """
        sim = self.sim
        if sim is None:
            return
        ahead = duration if budget is None else min(duration, budget / self._cost)
        if ahead >= 0.5 * sim.h and not sim.done:
            start, began = time.perf_counter(), sim.t
            while ahead >= 0.5 * sim.h and not sim.done:
                s = sim.advance(ahead)
                m = s.t.size
                if not m:
                    break
                k = self._k
                self._t[k:k + m] = s.t
                self._rpm[k:k + m] = s.rpm
                self._motor[k:k + m] = s.motor
                self._links[:, k:k + m] = s.links
                self._k += m
                ahead -= m * sim.h
            if sim.t > began:
                self._cost = 0.8 * self._cost + 0.2 * (time.perf_counter() - start) / (sim.t - began)
            self._draw_startup()
            self._show_status()
            if sim.done:
                self.running = False
                self._update_buttons()

    def _draw_startup(self, full: bool = False) -> None:
        k = self._k
        j = self.startup_link.currentIndex()
        if full:
            self.startup_legend.clear()
            name = self.train.links[j].name if 0 <= j < len(self.train.links) else "Link"
            self.startup_legend.addItem(self.link_curve, name)
            self.startup_legend.addItem(self.motor_curve, "Motor's air-gap torque")
            run = self._running[j] if 0 <= j < self._running.size else math.nan
            self.running_line.setVisible(bool(math.isfinite(run)))
            if math.isfinite(run):
                self.running_line.setPos(run)
        t = self._t[:k]
        self.link_curve.setData(t, self._links[j, :k] if 0 <= j < self._links.shape[0] else t * 0.0)
        self.motor_curve.setData(t, self._motor[:k])
        self.speed_curve.setData(t, self._rpm[:k])
        for i, c in enumerate(self._crossings):
            if i not in self._passed and k and self._rpm[k - 1] >= c.rpm:
                self._passed.add(i)
                at = self._t[int(np.argmax(self._rpm[:k] >= c.rpm))]
                self._add_mark(self.startup_torque, at, 90, f"Mode {c.mode + 1}")
        self._show_peaks()

    def _show_status(self) -> None:
        sim = self.sim
        if sim is None:
            self.startup_status.setText("No motor on the train: pick one under Start-up to run it up.")
            return
        if self._k == 0:
            self.startup_status.setText("At rest: Start runs the train up.")
            return
        m = sim.startup.motor
        rpm = self._rpm[self._k - 1]
        slip = sim.slip
        text = f"t = {sim.t:.3f} s: the motor at {rpm:.0f} rpm, slip {100 * slip:.1f} %"
        if m.kind is MotorKind.SYNCHRONOUS:
            if sim.field_on and abs(slip) < 1e-3:
                text += ", in step"
            else:
                text += f", its poles pulsing at {m.pulsation_hz(abs(slip)):.3g} Hz"
            text += ", the field on" if sim.field_on else ""
        self.startup_status.setText(text + (". Done." if sim.done else "."))

    def _show_peaks(self) -> None:
        sim = self.sim
        if sim is None or self._k == 0:
            self.startup_note.setText("")
            return
        rows = []
        ratios = np.where(self._running > 0, sim.peaks / np.where(self._running > 0, self._running, 1.0), 0.0)
        worst = int(np.argmax(ratios)) if np.isfinite(ratios).all() else -1
        for i, link in enumerate(self.train.links):
            at = int(np.searchsorted(self._t[:self._k], sim.peak_t[i]))
            rpm = self._rpm[min(at, self._k - 1)]
            text = f"{link.name}: {fmt_torque(sim.peaks[i])} at {sim.peak_t[i]:.3f} s ({rpm:.0f} rpm)"
            if math.isfinite(self._running[i]) and self._running[i] > 0:
                text += f", {ratios[i]:.1f}× its running {fmt_torque(self._running[i])}"
            rows.append(f"<b>{text}</b>" if i == worst else text)
        head = "Peak torques" if sim.done else "Peak torques so far"
        self.startup_note.setText(f"{head}: " + "; ".join(rows) + ".")

    def _show_startup_info(self) -> None:
        if not self.motor_on:
            self.startup_info.setText(self._startup_note or "No motor: pick one to start the train from rest.")
            return
        t, su = self.train, self.startup
        m = su.motor
        w = steady_speed(t, su)
        parts = [self._startup_note] if self._startup_note else []
        if math.isnan(w):
            parts.append(f"Synchronous speed {m.sync_rpm:.0f} rpm: the motor can't carry the load there, and "
                         f"stalls on the way up.")
        elif m.kind is MotorKind.INDUCTION:
            parts.append(f"Synchronous speed {m.sync_rpm:.0f} rpm; it runs at {60 * w / (2 * math.pi):.0f} rpm "
                         f"(slip {100 * (1 - w / m.sync_speed):.1f} %).")
            near = min(self._modes, key=lambda md: abs(md.freq_hz - m.line_hz), default=None)
            if near is not None:
                j = self._modes.index(near)
                parts.append(f"The switch-on rings at {m.line_hz:g} Hz; mode {j + 1} ({near.freq_hz:.3g} Hz) is "
                             f"{100 * abs(near.freq_hz - m.line_hz) / m.line_hz:.0f} % from it.")
        else:
            parts.append(f"It runs at {m.sync_rpm:.0f} rpm.")
            cs = crossings(t, m)
            if cs:
                parts.append("The poles' 2× slip pulsation meets " + ", ".join(
                    f"mode {c.mode + 1} ({c.freq_hz:.3g} Hz) at {c.rpm:.0f} rpm" for c in cs) + ".")
            else:
                parts.append(f"No mode below {2 * m.line_hz:g} Hz for the poles' pulsation to meet.")
        self.startup_info.setText(" ".join(parts))

    # ------------------------------------------------------------- drawing
    def apply_theme(self) -> None:
        self.redraw()
        self._style_startup()

    def _style_startup(self) -> None:
        self.link_curve.setPen(pg.mkPen(colors.strong, width=1.5))
        self.motor_curve.setPen(pg.mkPen(colors.force, width=1))
        self.speed_curve.setPen(pg.mkPen(colors.mass[0], width=2))
        self.running_line.setPen(pg.mkPen(colors.muted, width=1, style=QtCore.Qt.PenStyle.DashLine))
        for _, mark in self._marks:
            self._style_mark(mark)

    def redraw(self) -> None:
        self._draw_interference()
        self._draw_response()
        self._draw_train()
        self._draw_shapes()

    def _band(self) -> tuple[float, float]:
        op = self.operating.value()
        return op * (1 - MARGIN / 100), op * (1 + MARGIN / 100)

    def _band_item(self) -> pg.LinearRegionItem:
        lo, hi = self._band()
        band = pg.LinearRegionItem((lo, hi), movable=False, brush=pg.mkBrush(*colors.band_shade))
        for line in band.lines:
            line.setPen(pg.mkPen(None))
        return band

    def _draw_interference(self) -> None:
        p = self.interference
        p.clear()
        t, top = self.train, self.top.value()
        log = self.log_scales.isChecked()
        if log != p.getPlotItem().ctrl.logXCheck.isChecked():
            # through a range that reads the same both ways: 0..3600 read as log10 overflows
            p.getPlotItem().getViewBox().setRange(xRange=(0, 1), yRange=(0, 1), padding=0)
            p.setLogMode(x=log, y=log)
        lines = order_lines(t)
        crits = criticals(t, top, lines)
        # up to the fastest mode an order reaches in the speed range (or the slowest, if none does)
        reached = max((c.mode for c in crits), default=0)
        freqs = [md.freq_hz for md in self._modes[:reached + 1]]
        y_top = (2.0 if log else 1.2) * max(freqs) if freqs else 100.0
        y_lo = 0.5 * min(freqs) if log and freqs else 0.0
        x_lo = top / 100.0 if log else 0.0

        def at(x: float, y: float) -> tuple[float, float]:
            """A point in the view's coordinates (log10 of each on log scales), for the items pyqtgraph
            doesn't map itself (text, lines)."""
            return (math.log10(x), math.log10(y)) if log else (x, y)

        lo, hi = self._band()
        band = pg.LinearRegionItem((at(lo, 1)[0], at(hi, 1)[0]), movable=False, brush=pg.mkBrush(*colors.band_shade))
        for line in band.lines:
            line.setPen(pg.mkPen(None))
        p.addItem(band)
        p.addItem(pg.InfiniteLine(at(self.operating.value(), 1)[0], angle=90,
                                  pen=pg.mkPen(colors.muted, width=1, style=QtCore.Qt.PenStyle.DashLine)))
        n_sources = len(sources(t))  # order_lines puts them first
        for j, (label, order) in enumerate(lines):
            own = j < n_sources
            color = colors.mass[j % len(colors.mass)] if own else colors.grey
            pen = pg.mkPen(color, width=1.5 if own else 1,
                           style=QtCore.Qt.PenStyle.SolidLine if own else QtCore.Qt.PenStyle.DashLine)
            # from the bottom (or left) of the plot to its top, or the speed range's end
            start = max(x_lo, 60.0 * y_lo / order)
            end = min(top, 60.0 * y_top / order)
            if end <= start:
                continue
            p.plot([start, end], [order * start / 60.0, order * end / 60.0], pen=pen)
            text = pg.TextItem(label, color=color, anchor=(1, 1) if end < top else (1, 0.5))
            text.setPos(*at(end, order * end / 60.0))
            p.addItem(text)
        motor = self.startup.motor
        if self.motor_on and motor.kind is MotorKind.SYNCHRONOUS:
            # the poles' 2 s f pulsation, falling from 2f at rest to nothing at synchronism
            xs = np.linspace(max(x_lo, 1e-9), min(top, motor.sync_rpm), 200)
            ys = 2.0 * motor.line_hz * (1.0 - xs / motor.sync_rpm)
            keep = ys > max(y_lo, 1e-9)
            p.plot(xs[keep], ys[keep], pen=pg.mkPen(colors.force, width=1.5, style=QtCore.Qt.PenStyle.DashLine))
            x = 0.45 * min(top, motor.sync_rpm)  # the label ends on the line, below it (it falls to the right)
            text = pg.TextItem("2× slip (start-up)", color=colors.force, anchor=(1, 0))
            text.setPos(*at(x, 2.0 * motor.line_hz * (1.0 - x / motor.sync_rpm)))
            p.addItem(text)
            cs = [c for c in crossings(t, motor) if c.mode < len(freqs) and c.rpm >= x_lo]
            if cs:
                p.plot([c.rpm for c in cs], [c.freq_hz for c in cs], pen=None, symbol="d", symbolSize=10,
                       symbolPen=pg.mkPen(colors.force, width=1.5), symbolBrush=pg.mkBrush(colors.background))
        for j, f in enumerate(freqs):
            p.plot([max(x_lo, 1e-9), top], [f, f], pen=pg.mkPen(colors.strong, width=1.5))
            text = pg.TextItem(f"Mode {j + 1}, {f:.4g} Hz", color=colors.strong, anchor=(0, 1))
            text.setPos(*at(max(x_lo, 1e-9), f))
            p.addItem(text)
        inside = [c for c in crits if lo <= c.rpm <= hi]
        outside = [c for c in crits if not lo <= c.rpm <= hi and c.rpm >= x_lo]
        if outside:
            p.plot([c.rpm for c in outside], [c.freq_hz for c in outside], pen=None, symbol="o", symbolSize=8,
                   symbolPen=pg.mkPen(colors.strong, width=1.5), symbolBrush=pg.mkBrush(colors.background))
        if inside:
            p.plot([c.rpm for c in inside], [c.freq_hz for c in inside], pen=None, symbol="o", symbolSize=10,
                   symbolPen=pg.mkPen(colors.force, width=1.5), symbolBrush=pg.mkBrush(colors.force))
        p.getPlotItem().getViewBox().setRange(xRange=(at(x_lo or 1, 1)[0] if log else 0.0, at(top, 1)[0]),
                                              yRange=(at(1, y_lo)[1] if log else 0.0, at(1, y_top)[1]),
                                              padding=0)
        self._unreached = len(self._modes) - len(freqs)
        self._show_readout(crits, inside)

    def _show_readout(self, crits, inside) -> None:
        op = self.operating.value()
        lines = []
        if not self._modes:
            self.readout.setText("No flexible modes.")
            return
        if inside:
            listed = "; ".join(f"<b>{c.label} meets mode {c.mode + 1} ({c.freq_hz:.4g} Hz) at {c.rpm:.0f} rpm</b>, "
                               f"{100 * separation(c.rpm, op):.1f} % {'above' if c.rpm > op else 'below'}"
                               for c in inside)
            lines.append(f"Inside ±{MARGIN:g} % of {op:.0f} rpm: {listed}.")
        else:
            near = min(crits, key=lambda c: separation(c.rpm, op), default=None)
            if near is None:
                lines.append(f"No order meets a mode below {self.top.value():.0f} rpm.")
            else:
                lines.append(f"No torsional critical within ±{MARGIN:g} % of {op:.0f} rpm: the nearest, {near.label} "
                             f"on mode {near.mode + 1} ({near.freq_hz:.4g} Hz), is at {near.rpm:.0f} rpm, "
                             f"{100 * separation(near.rpm, op):.1f} % {'above' if near.rpm > op else 'below'}.")
        if self._unreached:
            lines.append(f"{self._unreached} faster mode{'s' if self._unreached > 1 else ''} no order reaches "
                         f"below {self.top.value():.0f} rpm (not drawn).")
        below = [c for c in crits if c.rpm < op * (1 - MARGIN / 100)]
        if below:
            lines.append(f"A start-up passes {len(below)} below it, the last {below[-1].label} on mode "
                         f"{below[-1].mode + 1} at {below[-1].rpm:.0f} rpm.")
        self.readout.setText(" ".join(lines))

    def _draw_response(self) -> None:
        p = self.response
        p.clear()
        self.response_legend.clear()
        t, top = self.train, self.top.value()
        link = self.response_link.currentIndex()
        if not 0 <= link < len(t.links):
            return
        rpm = response_speeds(t, top)
        fr = forced_response(t, rpm)
        if not fr.sources:
            self.response_note.setText("No excitation: pick a disc to put a torque on, or give a mesh a "
                                       "transmission error.")
            p.setXRange(0.0, top, padding=0)
            return
        p.addItem(self._band_item())
        legend = self.response_legend
        curves = [(src.label, fr.torque[a, link], colors.mass[a % len(colors.mass)], 1.5)
                  for a, src in enumerate(fr.sources)]
        total = fr.total()[link]
        if len(fr.sources) > 1:
            curves.append(("Sum", total, colors.strong, 2.0))
        peak = float(total.max())
        floor = peak * 1e-4 if peak > 0 else 1e-9
        for label, y, color, width in curves:
            item = p.plot(rpm, np.maximum(y, floor), pen=pg.mkPen(color, width=width))
            legend.addItem(item, label)
        p.addItem(pg.InfiniteLine(self.operating.value(), angle=90,
                                  pen=pg.mkPen(colors.muted, width=1, style=QtCore.Qt.PenStyle.DashLine)))
        p.setXRange(0.0, top, padding=0)
        y_top = peak * 3.0 if peak > 0 else 1.0
        p.setYRange(math.log10(floor), math.log10(y_top), padding=0)
        op = self.operating.value()
        at_op = float(np.interp(op, rpm, total))
        lo, hi = self._band()
        band = (rpm >= lo) & (rpm <= hi)
        worst = float(total[band].max()) if band.any() else at_op
        i = int(np.argmax(total))
        name = t.links[link].name
        self.response_note.setText(
            f"{name}: ±{fmt_torque(at_op)} at {op:.0f} rpm, up to ±{fmt_torque(worst)} within ±{MARGIN:g} %; "
            f"the most, ±{fmt_torque(peak)}, at {rpm[i]:.0f} rpm.")

    def _layout(self) -> tuple[list[tuple[float, float, float]], float]:
        """Each disc's (x, axis height, half height) in the drawing, and the width of a disc."""
        t = self.train
        width = 0.5
        jmax = max(d.inertia for d in t.discs)
        h = [0.25 + 0.75 * (d.inertia / jmax) ** 0.25 for d in t.discs]
        out = []
        x, y = 0.0, 0.0
        for i, d in enumerate(t.discs):
            if i:
                link = t.links[i - 1]
                if isinstance(link, Mesh):
                    # the gears on one line across: sized by their teeth, touching
                    near = out[-1]
                    r_in = near[2]
                    r_out = r_in * link.teeth_out / link.teeth_in
                    h[i] = r_out
                    y = near[1] - r_in - r_out
                    x = near[0]
                else:
                    x += 1.6
            out.append((x, y, h[i]))
        return out, width

    def _draw_train(self) -> None:
        p = self.drawing
        p.clear()
        t = self.train
        spots, width = self._layout()
        structure = colors.structure
        for i in range(t.n - 1):
            (x0, y0, h0), (x1, y1, h1) = spots[i], spots[i + 1]
            chosen = self.selected == 2 * i + 1
            color = colors.force if chosen else structure
            if isinstance(t.links[i], Mesh):
                p.plot([x0, x0], [y0 - h0, y1 + h1], pen=pg.mkPen(color, width=3 if chosen else 1.5),
                       symbol="o", symbolSize=6, symbolBrush=pg.mkBrush(color), symbolPen=None)
            else:
                p.plot([x0 + width / 2, x1 - width / 2], [y0, y1], pen=pg.mkPen(color, width=4 if chosen else 2.5))
        for i, (x, y, h) in enumerate(spots):
            chosen = self.selected == 2 * i
            rect = QtWidgets.QGraphicsRectItem(x - width / 2, y - h, width, 2 * h)
            rect.setPen(pg.mkPen(colors.force if chosen else structure, width=3 if chosen else 1.5))
            rect.setBrush(pg.mkBrush(colors.mass[i % len(colors.mass)]))
            p.addItem(rect)
            label = pg.TextItem(t.discs[i].name, color=colors.foreground, anchor=(0.5, 1))
            label.setPos(x, y + h + 0.05)
            label.setFont(_small_font())
            p.addItem(label)
        if t.excitation is not None:
            x, y, h = spots[t.excitation.disc]
            arrow = pg.ArrowItem(angle=90, tipAngle=40, headLen=14, tailLen=12, tailWidth=3,
                                 pen=None, brush=pg.mkBrush(colors.force))
            arrow.setPos(x, y - h)
            p.addItem(arrow)
        xs = [x for x, _, _ in spots]
        ys = [y - h for _, y, h in spots] + [y + h for _, y, h in spots]
        p.getPlotItem().getViewBox().setRange(xRange=(min(xs) - 1.0, max(xs) + 1.0),
                                              yRange=(min(ys) - 0.6, max(ys) + 0.4), padding=0)
        s = t.speeds
        if t.meshes:
            speeds = "Each shaft's speed over the motor's: " + ", ".join(
                f"{d.name} {s[i]:.3g}×" for i, d in enumerate(t.discs) if i and not math.isclose(s[i], s[i - 1])) + "."
        else:
            speeds = "One shaft: every disc turns at the motor's speed."
        if t.excitation is not None:
            speeds += f" The arrow is the excitation, on the {t.discs[t.excitation.disc].name.lower()}."
        self.drawing_note.setText(speeds)

    def _on_drawing_click(self, event) -> None:
        vb = self.drawing.getPlotItem().getViewBox()
        pos = vb.mapSceneToView(event.scenePos())
        spots, width = self._layout()
        best, dist = None, math.inf
        for i, (x, y, h) in enumerate(spots):
            if abs(pos.x() - x) <= width / 2 and abs(pos.y() - y) <= h:
                best, dist = 2 * i, 0.0
                break
        if best is None:
            for i in range(len(spots) - 1):
                (x0, y0, _), (x1, y1, _) = spots[i], spots[i + 1]
                d = math.hypot(pos.x() - 0.5 * (x0 + x1), pos.y() - 0.5 * (y0 + y1))
                if d < dist:
                    best, dist = 2 * i + 1, d
            if dist > 1.0:
                return
        self.select(best)

    def _draw_shapes(self) -> None:
        p = self.shapes
        p.clear()
        self.shapes_legend.clear()
        t = self.train
        x = np.arange(t.n)
        p.addItem(pg.InfiniteLine(0.0, angle=0, pen=pg.mkPen(colors.faint, width=1)))
        for j, md in enumerate(self._modes[:SHAPES_SHOWN]):
            color = colors.mode[j % len(colors.mode)]
            item = p.plot(x, md.shape, pen=pg.mkPen(color, width=2), symbol="o", symbolSize=6,
                          symbolBrush=pg.mkBrush(color), symbolPen=None)
            self.shapes_legend.addItem(item, f"Mode {j + 1}, {md.freq_hz:.4g} Hz")
        p.getAxis("bottom").setTicks([[(i, d.name) for i, d in enumerate(t.discs)]])
        p.setXRange(-0.3, t.n - 0.7, padding=0)
        p.setYRange(-1.1, 1.1, padding=0)
        rows = []
        for j, md in enumerate(self._modes):
            share = md.twist[md.link]
            rows.append(f"Mode {j + 1}: {md.freq_hz:.4g} Hz, ζ = {100 * md.zeta:.2g} %, twisting "
                        f"{t.links[md.link].name.lower()} ({100 * share:.0f} %)")
        self.mode_list.setText("<br>".join(rows) if rows else "No flexible modes.")
        self._show_check()

    def _show_check(self) -> None:
        t = self.train
        f = natural_hz(t)
        ref = natural_hz(t.referred())
        text = ("Referred to the motor's shaft (inertias and stiffnesses × s², each mesh a shaft of "
                "k<sub>m</sub>(r s)²): " + ", ".join(f"{v:.4g}" for v in ref) + " Hz, the same.")
        if t.meshes:
            try:
                rigid = natural_hz(t.rigid_gears())
            except ValueError:
                rigid = np.empty(0)
            lost = f.size - rigid.size
            text += (" With the gears rigid (the n² reflection): "
                     + (", ".join(f"{v:.4g}" for v in rigid) + " Hz" if rigid.size else "no modes")
                     + f", {lost} mode{'s' if lost != 1 else ''} lost.")
        self.check.setText(text)

    def _show_train_info(self) -> None:
        t = self.train
        ref = t.referred()
        j = sum(d.inertia for d in ref.discs)
        self.train_info.setText(f"{t.n} discs, {len(t.meshes)} gear mesh{'es' if len(t.meshes) != 1 else ''}; "
                                f"{j:.4g} kg·m² referred to the motor's shaft.")


def _small_font() -> QtGui.QFont:
    font = QtGui.QFont()
    font.setPointSizeF(font.pointSizeF() * 0.85)
    return font
