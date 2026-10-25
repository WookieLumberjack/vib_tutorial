"""Torsional vibration page: a drive train of discs, shafts and gear meshes, its torsional criticals and forced response.

Left: the preset, the train (pick a disc or a link to edit it, or click it in the drawing),
the excitation and the speeds. Centre: the interference (Campbell) diagram, natural
frequencies against the orders that excite them, and the steady dynamic torque in a link
against the motor's speed. Right: the train drawn, its mode shapes, and the notes. The page
has its own model (core/torsional.py), not the chain.
"""

from __future__ import annotations

import math

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
from .axes import log_axes
from .layouts import ColumnScroll, FlowLayout, group, plot_height
from .panels import spin
from .style import colors
from .theming import add_legend, mute

CUSTOM = "Custom"
MAX_DISCS = 12
SHAPES_SHOWN = 6  # modes drawn on the shape plot
MARGIN = 10.0  # % either side of the running speed a critical should stay out of (API 617's 10 %)

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
}
INTERFERENCE_TIP = (
    "<p>The interference (Campbell) diagram. The flat lines are the natural frequencies: they don't change "
    "with speed. Each line through the origin is an order: an excitation at that multiple of the motor's "
    "speed (solid: the train's own excitation and tooth-pass; dashed: 1× and 2× of each shaft's speed, which "
    "a standard asks to check). Where an order crosses a mode is a torsional critical speed; filled red "
    "inside the band round the running speed.</p>")
RESPONSE_TIP = (
    "<p>The steady dynamic torque in the chosen link (a mesh: its force times the near gear's base radius), "
    "the amplitude of each order on its own and their sum (the peak when they line up), against the motor's "
    "speed. Each peaks where its order meets a mode the link twists in.</p>")

NOTES_HTML = """
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

<h3>Gears</h3>
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

<h3>The interference diagram</h3>
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
"""


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

        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.addWidget(self.preset)
        lv.addWidget(self.preset_note)
        for box in (train_box, excite, speeds):
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
        cv.setContentsMargins(0, 0, 0, 0)
        cv.addWidget(self.interference, 3)
        cv.addWidget(self.readout)
        row = FlowLayout()
        row.add(group("Dynamic torque in:", self.response_link))
        cv.addLayout(row)
        cv.addWidget(self.response, 2)
        cv.addWidget(self.response_note)

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
        self.tabs.addTab(self.notes, "Notes")

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(left_scroll)
        splitter.addWidget(center)
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

    # ------------------------------------------------------------- drawing
    def apply_theme(self) -> None:
        self.redraw()

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
