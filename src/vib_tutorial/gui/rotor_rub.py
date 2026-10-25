"""Rub tab of the Jeffcott rotor page: a stator ring round the disc, which the disc rubs once it reaches it.

The ring's clearance, offset, mount and friction (core/rotor.py's `Stator`), what
they mean for the rotor at rest (the gap left under its sag, the disc's frequency
on the ring), and the ring's normal force on the disc over the strip chart's
window. The rub is in the time response only: the Campbell diagram, the
stability map and the steady-state curves are the rotor's without the ring.
"""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.rotor import XD, YD, RotorSystem, Stator
from .layouts import plot_height
from .panels import spin
from .rotor_bearing import _si
from .style import colors
from .theming import mute

NOTE = (
    "Once the disc's whirl reaches the ring, the ring pushes it back and its friction drags the disc's rim "
    "against its slip. Brushing the ring on one side (a <i>partial rub</i>, as under the disc's sag) gives "
    "the orbit harmonics, and above twice the critical speed sub-harmonics: ½X, ⅓X. Held against it all "
    "round (a <i>full rub</i>), friction pulls the rim backward, and enough of it flips the whirl backward at "
    "the disc's frequency on the ring, far above the speed: the rim then rolls round the ring (dry whirl). "
    "The ring is in the time response only: the Campbell diagram, the stability map and the steady-state "
    "curves leave it out. Watch the full spectrum (Stability tab) and the waterfall."
)
INPUT_TIPS = {
    "clearance": "<p>The ring's radial clearance: how far the disc's centre can move off the ring's centre "
                 "before its rim touches.</p>",
    "offset": "<p>The ring's centre, off the bearings' line, and the direction it is offset in (fixed axes, "
              "from +x toward +y: −90° is straight down). An offset ring is reached on one side first.</p>",
    "stiffness": "<p>The ring's mount: how hard the ring pushes back as the disc pushes it out past its "
                 "clearance. The rim and the ring are rigid; the mount gives.</p>",
    "damping": "<p>Damping in the ring's mount, while the disc pushes on it.</p>",
    "friction": "<p>The friction coefficient μ between the rim and the ring: the drag is μ times the ring's "
                "push, against the rim's slip over the ring (Coulomb's, smoothed over 1 cm/s).</p>",
}


class RubPanel(QtWidgets.QWidget):
    """The stator ring's inputs, what they mean at rest, and its force on the disc."""

    changed = QtCore.Signal()  # the user changed the ring

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        s = Stator()
        self.enabled = QtWidgets.QCheckBox("A stator ring round the disc")
        self.enabled.setToolTip("<p>A seal or casing ring round the disc, which its rim rubs on once it "
                                "reaches it.</p>")
        self.clearance = spin(1.0, 10_000.0, s.clearance * 1e6, 1, " µm")
        self.offset = spin(0.0, 10_000.0, s.offset * 1e6, 1, " µm")
        self.offset_angle = spin(-180.0, 180.0, math.degrees(s.offset_angle), 0, "°")
        self.stiffness = spin(0.01, 1000.0, s.stiffness / 1e6, 2, " MN/m")
        self.damping = spin(0.0, 1e5, s.damping, 0, " N·s/m")
        self.friction = spin(0.0, 2.0, s.friction, 2)
        self.friction.setSingleStep(0.05)
        form = QtWidgets.QGridLayout()
        rows = (("Clearance:", (self.clearance,), "clearance"),
                ("Offset:", (self.offset, "at", self.offset_angle), "offset"),
                ("Mount stiffness:", (self.stiffness,), "stiffness"),
                ("Mount damping:", (self.damping,), "damping"),
                ("Friction μ:", (self.friction,), "friction"))
        for r, (text, boxes, key) in enumerate(rows):
            label = QtWidgets.QLabel(text)
            label.setToolTip(INPUT_TIPS[key])
            form.addWidget(label, r, 0)
            for c, box in enumerate(boxes):
                if isinstance(box, str):
                    form.addWidget(QtWidgets.QLabel(box), r, 1 + c)
                    continue
                box.setToolTip(INPUT_TIPS[key])
                box.valueChanged.connect(self._on_params)
                form.addWidget(box, r, 1 + c, 1, 1 if len(boxes) > 1 else 3)
        form.setColumnStretch(1, 1)
        form.setColumnStretch(3, 1)
        self.enabled.toggled.connect(self._on_params)

        self.readout = QtWidgets.QLabel()
        self.readout.setWordWrap(True)
        self.readout.setTextFormat(QtCore.Qt.TextFormat.RichText)

        self.plot = pg.PlotWidget()
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.hideButtons()
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.plot.setLabel("left", "Push on the disc", units="N")
        self.plot.setLabel("bottom", "Time", units="s")
        self.plot.setToolTip("<p>The ring's push on the disc, normal to the ring, over the strip chart's window "
                             "(each point the largest over the samples it stands for).</p>")
        self.curve = pg.PlotDataItem()
        self.plot.addItem(self.curve)
        plot_height(self.plot, 160)
        self.contact = QtWidgets.QLabel()
        self.contact.setWordWrap(True)
        self.note = QtWidgets.QLabel(NOTE)
        self.note.setWordWrap(True)
        mute(self.note)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.enabled)
        layout.addLayout(form)
        layout.addWidget(self.readout)
        layout.addWidget(self.plot, 1)
        layout.addWidget(self.contact)
        layout.addWidget(self.note)
        self.apply_theme()
        self._on_params(notify=False)

    def stator(self) -> Stator | None:
        """The ring the inputs describe, or None if there is none (or it is not allowed)."""
        if not (self.enabled.isChecked() and self.enabled.isEnabled()):
            return None
        return Stator(clearance=self.clearance.value() / 1e6, stiffness=self.stiffness.value() * 1e6,
                      damping=self.damping.value(), friction=self.friction.value(),
                      offset=self.offset.value() / 1e6, offset_angle=math.radians(self.offset_angle.value()))

    def set_stator(self, stator: Stator | None) -> None:
        """Show `stator` (None: no ring, the inputs kept), without telling the page."""
        boxes = (self.enabled, self.clearance, self.offset, self.offset_angle, self.stiffness, self.damping,
                 self.friction)
        for w in boxes:
            w.blockSignals(True)
        self.enabled.setChecked(stator is not None)
        if stator is not None:
            self.clearance.setValue(stator.clearance * 1e6)
            self.offset.setValue(stator.offset * 1e6)
            self.offset_angle.setValue(math.degrees(stator.offset_angle))
            self.stiffness.setValue(stator.stiffness / 1e6)
            self.damping.setValue(stator.damping)
            self.friction.setValue(stator.friction)
        for w in boxes:
            w.blockSignals(False)
        self._on_params(notify=False)

    def set_allowed(self, allowed: bool) -> None:
        """A turning (asymmetric or cracked) shaft, or the oil films' whole force, can't have a ring."""
        self.enabled.setEnabled(allowed)
        self.enabled.setToolTip(
            "<p>A seal or casing ring round the disc, which its rim rubs on once it reaches it.</p>"
            + ("" if allowed else "<p><b>Not with</b> an asymmetric or cracked shaft, or with <i>Nonlinear "
                                  "film</i> ticked: their time steps don't take the ring.</p>"))
        self._on_params(notify=False)

    def _on_params(self, *_, notify: bool = True) -> None:
        on = self.enabled.isChecked() and self.enabled.isEnabled()
        for w in (self.clearance, self.offset, self.offset_angle, self.stiffness, self.damping, self.friction):
            w.setEnabled(on)
        self.offset_angle.setEnabled(on and self.offset.value() > 0.0)
        if notify:
            self.changed.emit()

    def update_info(self, system: RotorSystem) -> None:
        """What the ring means for `system` at rest: the gap left, and the disc's frequency on the ring."""
        st = system.stator
        if st is None:
            self.readout.setText("" if self.enabled.isEnabled() else
                                 "A ring needs a round shaft, with the oil films linearized.")
            self.contact.setText("")
            return
        rest = system.static_deflection(0.0)[[XD, YD]] if system.gravity else np.zeros(2)
        cx, cy = st.centre
        gap = st.clearance - math.hypot(rest[0] - cx, rest[1] - cy)
        parts = []
        if system.gravity:
            parts.append(f"The disc sags {_fmt_um(-rest[1])} under its weight")
        if gap >= 0.0:
            where = "" if not (system.gravity or st.offset) else " on its nearer side"
            parts.append(f"{'leaving' if parts else 'At rest it has'} {_fmt_um(gap)} to the ring{where}.")
        else:
            parts.append(f"{'and' if parts else 'At rest it'} presses {_fmt_um(-gap)} into the ring "
                         f"({_si(-gap * st.stiffness, 'N')}).")
        text = ", ".join(parts[:2]) if len(parts) == 2 else parts[0]
        try:
            k = 1.0 / np.linalg.inv(system.stiffness(0.0))[XD, XD]  # the rotor's stiffness at the disc
        except np.linalg.LinAlgError:
            k = math.nan
        if k > 0.0:
            f = math.sqrt((k + st.stiffness) / system.disc_mass) / (2.0 * math.pi)
            text += (f" On the ring the disc's whirl is stiffened to about {f:.3g} Hz "
                     f"(free, {math.sqrt(k / system.disc_mass) / (2.0 * math.pi):.3g} Hz): a dry whirl's "
                     f"frequency. Disc radius R = {system.disc_radius * 1e3:.3g} mm.")
        self.readout.setText(text)

    def update_contact(self, t: np.ndarray, normal: np.ndarray, window: float) -> None:
        """The ring's push over the last `window` s (`t` and `normal` from the page's history)."""
        if t.size:
            self.curve.setData(t, normal)
            end = float(t[-1])
            self.plot.setXRange(max(end - window, 0.0), max(end, window), padding=0)
            touching = normal > 0.0
            if touching.any():
                self.contact.setText(f"In contact {touching.mean():.0%} of the time; push up to "
                                     f"{_si(float(normal.max()), 'N')}.")
            else:
                self.contact.setText("Not touching the ring.")
        else:
            self.curve.setData([], [])
            self.contact.setText("")

    def apply_theme(self) -> None:
        self.curve.setPen(pg.mkPen(colors.force, width=1.5))


def _fmt_um(meters: float) -> str:
    return f"{meters * 1e3:.3g} mm" if meters >= 1e-3 else f"{meters * 1e6:.3g} µm"
