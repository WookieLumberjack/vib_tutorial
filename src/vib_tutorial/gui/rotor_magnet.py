"""Magnetic bearing tab of the Jeffcott rotor page: an active magnetic bearing under PID control.

From the actuator's air gap, bias current, pole area and turns
(core/magnetic_bearing.py): its negative stiffness, current gain and load
capacity; from the controller's gains, filter and amplifier: the bearing's
equivalent stiffness and damping against frequency, shaded where the damping is
negative (the controller's lag feeding a mode). With *Magnetic bearings* chosen
under the rotor's bearing supports, the rotor floats in this bearing at both
ends, and its modes at the speed now are marked on the plots.
"""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.magnetic_bearing import MagneticBearing
from ..core.rotor import RPM, RotorSystem, whirl_modes
from ..core.rotor_stability import Onset
from .layouts import plot_height
from .panels import spin
from .rotor_bearing import _si
from .style import colors
from .tab_groups import tucked_away
from .theming import mute

DASH = QtCore.Qt.PenStyle.DashLine
POINTS = 400  # frequencies on the curves
LOWEST = 1.0  # Hz, the plots' left end
AXIS_WIDTH = 56  # px, the plots' left axes
MODE_SPEED = 25.0  # rpm: the rotor's modes are worked out afresh once the speed has moved this much

EQUIVALENT_TIP = (
    "<p>The bearing pushes back on its journal with −K<sub>b</sub>(s) x, K<sub>b</sub>(s) = "
    "k<sub>i</sub> G(s) − k<sub>m</sub>. Shaken at a frequency f, that push is the same as a spring "
    "k<sub>b</sub>(f) (its real part) and a damper c<sub>b</sub>(f) (its imaginary part over ω): unlike "
    "a real spring and damper, both change with the frequency.</p>"
    "<p>The derivative gives the damping. Up the frequency the derivative's filter and the amplifier lag "
    "the current behind the motion, and past some frequency c<sub>b</sub> turns negative (shaded): the "
    "bearing then feeds a mode there instead of damping it. Down at the bottom the integral's lag does the "
    "same, below the frequency where it fades. The lines are the rotor's modes at the speed now, solid "
    "forward, dashed backward: a lightly damped one in a shaded band goes unstable.</p>"
)
INPUT_TIPS = {
    "gap": "<p>Air gap s<sub>0</sub> between each magnet's poles and the journal, with the journal centred. "
           "Half of it is the backup (touchdown) bearing's clearance: the journal lands there first.</p>",
    "bias": "<p>Bias current i<sub>0</sub> in every coil. The control current is added to one magnet of a "
            "pair and taken from the other. A larger bias gives more force per ampere (k<sub>i</sub> ∝ i<sub>0</sub>) "
            "and more load capacity, but a stronger negative stiffness (k<sub>m</sub> ∝ i<sub>0</sub>²).</p>",
    "area": "<p>Area A of each of a magnet's two pole faces.</p>",
    "turns": "<p>Turns N of each magnet's coil. The magnet pulls with k i²/s², k = μ<sub>0</sub>N²A/4.</p>",
    "kp": "<p>Proportional gain k<sub>p</sub>: amperes per metre the journal is off centre. "
          "k<sub>i</sub>k<sub>p</sub> has to beat the magnets' pull k<sub>m</sub>, and what is left over is "
          "the bearing's stiffness.</p>",
    "ki": "<p>Integral gain k<sub>I</sub>: a steady load cannot hold the journal off centre, since the "
          "current keeps growing until it is back. Its lag makes the damping negative at low frequency.</p>",
    "kd": "<p>Derivative gain k<sub>d</sub>: amperes per metre per second. k<sub>i</sub>k<sub>d</sub> is "
          "the damping, below the filter and the amplifier.</p>",
    "filter": "<p>The derivative's low-pass filter, f<sub>d</sub>: a pure derivative would amplify the "
              "sensor's noise without limit. Above it the derivative's phase lead fades.</p>",
    "amplifier": "<p>The power amplifier's bandwidth f<sub>a</sub>: the coils' inductance keeps the current "
                 "from following the controller at once. Its lag eats the derivative's lead.</p>",
}


class MagnetPanel(QtWidgets.QWidget):
    """Inputs, the equivalent stiffness and damping against frequency, and the readings."""

    changed = QtCore.Signal()  # the user changed the bearing

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.top_rpm = 10_000.0
        self.system: RotorSystem | None = None  # the rotor, when it runs on this bearing
        self.onset: Onset | None = None  # where the rotor running on it goes unstable
        self.rpm = 0.0  # the rotor's speed now
        self._mode_rpm = math.nan  # the speed the mode lines were drawn for
        self._stale = False  # changed while hidden: drawn when the tab shows
        m = MagneticBearing()
        self.gap = spin(0.05, 10.0, m.gap * 1e3, 3, " mm")
        self.bias = spin(0.01, 100.0, m.bias, 2, " A")
        self.area = spin(1.0, 1e5, m.area * 1e6, 0, " mm²")
        self.turns = spin(1.0, 10_000.0, m.turns, 0)
        self.kp = spin(0.0, 1e8, m.kp, 0, " A/m")
        self.ki = spin(0.0, 1e10, m.ki, 0, " A/(m·s)")
        self.kd = spin(0.0, 1e5, m.kd, 2, " A·s/m")
        self.filter = spin(1.0, 1e5, m.filter_hz, 0, " Hz")
        self.amplifier = spin(1.0, 1e5, m.amplifier_hz, 0, " Hz")
        form = QtWidgets.QGridLayout()
        boxes = (("Air gap s<sub>0</sub>:", self.gap, "gap"), ("Bias i<sub>0</sub>:", self.bias, "bias"),
                 ("Pole area A:", self.area, "area"), ("Turns N:", self.turns, "turns"),
                 ("k<sub>p</sub>:", self.kp, "kp"), ("k<sub>I</sub>:", self.ki, "ki"),
                 ("k<sub>d</sub>:", self.kd, "kd"), ("Filter f<sub>d</sub>:", self.filter, "filter"),
                 ("Amplifier f<sub>a</sub>:", self.amplifier, "amplifier"))
        heads = ("<b>Actuator</b> (a magnet pair on each axis)", "<b>Controller</b> (on each axis)")
        for col, text in enumerate(heads):
            head = QtWidgets.QLabel(text)
            head.setWordWrap(True)  # no wider than its two columns
            form.addWidget(head, 0, 2 * col, 1, 2)
        for i, (text, box, key) in enumerate(boxes):
            label = QtWidgets.QLabel(text)
            for w in (label, box):
                w.setToolTip(INPUT_TIPS[key])
            col = 0 if i < 4 else 1
            row = 1 + (i if i < 4 else i - 4)
            form.addWidget(label, row, 2 * col)
            form.addWidget(box, row, 2 * col + 1)
            box.valueChanged.connect(self._on_params)
        form.setColumnStretch(1, 1)
        form.setColumnStretch(3, 1)

        self.readout = QtWidgets.QLabel()
        self.readout.setWordWrap(True)
        self.readout.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.readout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        self.plots = pg.GraphicsLayoutWidget()
        self.plots.setToolTip(EQUIVALENT_TIP)
        self.k_plot = self.plots.addPlot(row=0, col=0)
        self.c_plot = self.plots.addPlot(row=1, col=0)
        self.c_plot.setXLink(self.k_plot)
        self.k_curve = pg.PlotDataItem()
        self.c_curve = pg.PlotDataItem()
        self.bands: list[pg.LinearRegionItem] = []  # where c_b < 0, on both plots
        self.zero_lines: list[pg.InfiniteLine] = []
        self.mode_lines: list[list[pg.InfiniteLine]] = [[], []]  # per plot
        for p, curve, (label, unit) in ((self.k_plot, self.k_curve, ("Stiffness k<sub>b</sub>", "N/m")),
                                        (self.c_plot, self.c_curve, ("Damping c<sub>b</sub>", "N·s/m"))):
            p.setMenuEnabled(False)
            p.setMouseEnabled(x=False, y=False)
            p.hideButtons()
            p.showGrid(x=True, y=True, alpha=0.15)
            p.setLogMode(x=True, y=False)
            p.getAxis("left").setWidth(AXIS_WIDTH)
            p.setLabel("left", label, units=unit)
            p.addItem(curve)
            zero = pg.InfiniteLine(pos=0.0, angle=0)
            p.addItem(zero)
            self.zero_lines.append(zero)
        self.c_plot.setLabel("bottom", "Frequency (Hz)")
        self.note = QtWidgets.QLabel()
        self.note.setWordWrap(True)
        mute(self.note)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.readout)
        plot_height(self.plots, 300)  # the two plots together
        layout.addWidget(self.plots, 1)
        layout.addWidget(self.note)
        self.apply_theme()
        self._redraw()

    # ------------------------------------------------------------- inputs
    def bearing(self) -> MagneticBearing:
        """The bearing described by the inputs (SI)."""
        return MagneticBearing(gap=self.gap.value() / 1e3, bias=self.bias.value(), area=self.area.value() / 1e6,
                               turns=round(self.turns.value()), kp=self.kp.value(), ki=self.ki.value(),
                               kd=self.kd.value(), filter_hz=self.filter.value(),
                               amplifier_hz=self.amplifier.value())

    def set_bearing(self, m: MagneticBearing, notify: bool = True) -> None:
        """Show and use `m` (for scripts and tests); `notify` tells the rotor, if it runs on it."""
        for box, value in ((self.gap, m.gap * 1e3), (self.bias, m.bias), (self.area, m.area * 1e6),
                           (self.turns, m.turns), (self.kp, m.kp), (self.ki, m.ki), (self.kd, m.kd),
                           (self.filter, m.filter_hz), (self.amplifier, m.amplifier_hz)):
            box.blockSignals(True)
            box.setValue(value)
            box.blockSignals(False)
        self._redraw()
        if notify:
            self.changed.emit()

    def set_rotor(self, system: RotorSystem | None) -> None:
        """The rotor runs on this bearing (None: it does not): its modes are marked, its loads read out."""
        self.system = system
        self._mode_rpm = math.nan
        self._redraw()

    def set_onset(self, onset: Onset | None) -> None:
        """Where the rotor, running on this bearing, goes unstable (None: it does not, or it does not run on it)."""
        self.onset = onset
        self._write()

    def set_top_rpm(self, rpm: float) -> None:
        self.top_rpm = rpm
        self._write()

    def set_speed(self, rpm: float) -> None:
        """The rotor's speed now: its modes there are marked (call while the tab shows)."""
        self.rpm = rpm
        if self.system is not None and not abs(rpm - self._mode_rpm) < MODE_SPEED:
            self._draw_modes()

    def _on_params(self) -> None:
        self._redraw()
        self.changed.emit()

    # ------------------------------------------------------------- drawing
    def showEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().showEvent(event)
        self.catch_up()

    def catch_up(self) -> None:
        """Draw what changed while the tab was hidden (its drawing waits till it shows)."""
        if self._stale:
            self._redraw()

    def apply_theme(self) -> None:
        self.k_curve.setPen(pg.mkPen(colors.mode[0], width=2))
        self.c_curve.setPen(pg.mkPen(colors.mode[1], width=2))
        for line in self.zero_lines:
            line.setPen(pg.mkPen(colors.faint, width=1))
        self._shade_bands()
        self._mode_rpm = math.nan
        if self.system is not None:
            self._draw_modes()

    def _span(self, m: MagneticBearing) -> tuple[float, float]:
        """The plots' frequency range (Hz): to a few times the amplifier's bandwidth or the band's top."""
        hi = m.damped_band()[1]
        top = max(3.0 * m.amplifier_hz, 3.0 * hi if math.isfinite(hi) else 0.0, 100.0)
        return LOWEST, 10.0 ** math.ceil(math.log10(top))

    def _redraw(self) -> None:
        self._stale = tucked_away(self)  # a tab not showing: draw when it does
        if self._stale:
            return
        m = self.bearing()
        lo, hi = self._span(m)
        f = np.geomspace(lo, hi, POINTS)
        k, c = m.equivalent(f)
        self.k_curve.setData(f, k)
        self.c_curve.setData(f, c)
        # Scale to what the bearing does where it damps, not to the integral's lag at the bottom.
        lo_band, hi_band = m.damped_band()
        inside = (f >= lo_band) & (f <= hi_band) if math.isfinite(lo_band) else np.ones(f.size, bool)
        top_c = float(c[inside].max()) if inside.any() and c[inside].max() > 0 else float(np.abs(c).max())
        self.c_plot.setRange(xRange=(math.log10(lo), math.log10(hi)), yRange=(-0.6 * top_c, 1.2 * top_c),
                             padding=0)
        k_lo, k_hi = min(float(k.min()), 0.0), max(float(k.max()), 0.0)
        self.k_plot.setRange(yRange=(k_lo - 0.05 * (k_hi - k_lo), k_hi + 0.05 * (k_hi - k_lo)), padding=0)
        self._draw_bands(m, lo, hi)
        self._mode_rpm = math.nan
        if self.system is not None:
            self._draw_modes()
        else:
            self._clear_modes()
        self._write()

    def _draw_bands(self, m: MagneticBearing, lo: float, hi: float) -> None:
        f_lo, f_hi = m.damped_band()
        spans = [(lo, hi)] if not math.isfinite(f_lo) else (
            ([(lo, f_lo)] if f_lo > lo else []) + ([(f_hi, hi)] if f_hi < hi else []))
        while len(self.bands) < 2 * len(spans):
            band = pg.LinearRegionItem(movable=False)
            band.setZValue(-10)
            (self.k_plot if len(self.bands) % 2 == 0 else self.c_plot).addItem(band)
            self.bands.append(band)
        for i, band in enumerate(self.bands):
            if i // 2 < len(spans):
                a, b = spans[i // 2]
                band.setRegion((math.log10(a), math.log10(b)))
                band.setVisible(True)
            else:
                band.setVisible(False)
        self._shade_bands()

    def _shade_bands(self) -> None:
        shade = pg.mkColor(colors.poor)
        shade.setAlpha(40)
        for band in self.bands:
            band.setBrush(pg.mkBrush(shade))
            for line in band.lines:
                line.setPen(pg.mkPen(None))

    def _clear_modes(self) -> None:
        for p, lines in zip((self.k_plot, self.c_plot), self.mode_lines):
            for line in lines:
                p.removeItem(line)
            lines.clear()

    def _draw_modes(self) -> None:
        """Lines at the rotor's natural frequencies at the speed now, solid forward, dashed backward."""
        self._mode_rpm = self.rpm
        self._clear_modes()
        modes = whirl_modes(self.system, self.rpm / RPM)
        lo, hi = self._span(self.bearing())
        for f, whirl in zip(modes.freq_hz, modes.whirl):
            if not lo <= f <= hi:
                continue
            pen = pg.mkPen(colors.strong, width=1.2, style=QtCore.Qt.PenStyle.SolidLine if whirl >= 0 else DASH)
            for p, lines in zip((self.k_plot, self.c_plot), self.mode_lines):
                line = pg.InfiniteLine(pos=math.log10(f), angle=90, pen=pen)
                p.addItem(line)
                lines.append(line)

    def _write(self) -> None:
        if tucked_away(self):
            self._stale = True
            return
        m = self.bearing()
        ks, ki = m.negative_stiffness, m.current_gain
        net = ki * m.kp - ks
        f_lo, f_hi = m.damped_band()
        parts = [(
            f"<p><b>Actuator</b>: k = μ<sub>0</sub>N²A/4 = {m.force_constant:.3g} N·m²/A²; negative stiffness "
            f"k<sub>m</sub> = 4ki<sub>0</sub>²/s<sub>0</sub>³ = {_si(ks, 'N/m')}; current gain k<sub>i</sub> = "
            f"4ki<sub>0</sub>/s<sub>0</sub>² = {ki:.3g} N/A; load capacity k<sub>i</sub>i<sub>0</sub> = "
            f"{m.capacity:.3g} N; backup bearing {_si(m.touchdown, 'm')} off centre.</p>")]
        if net <= 0.0:
            parts.append(
                f"<p><span style='color:{colors.force}'><b>k<sub>i</sub>k<sub>p</sub> − k<sub>m</sub> = "
                f"{_si(net, 'N/m')}</b>: the magnets outpull the controller. The journal falls toward a magnet"
                + (", the integral pulling it back too late, so that it swings ever wider" if m.integral else "")
                + f". Raise k<sub>p</sub> above {ks / ki:,.0f} A/m.</span></p>")
        else:
            parts.append(
                f"<p><b>Controller</b>: k<sub>i</sub>k<sub>p</sub> − k<sub>m</sub> = {_si(net, 'N/m')} of "
                "stiffness" + (" (infinite at 0 Hz with the integral: a steady load cannot move the journal)"
                               if m.integral else "")
                + f", k<sub>i</sub>k<sub>d</sub> = {ki * m.kd:.3g} N·s/m of damping before the lags. ")
            if math.isfinite(f_lo):
                parts[-1] += (f"It damps from {f_lo:.3g} to {f_hi:,.4g} Hz" if f_lo > 0.0 else
                              f"It damps up to {f_hi:,.4g} Hz") + "; outside, it feeds a mode (shaded).</p>"
            else:
                parts[-1] += "It damps nowhere: every mode is fed.</p>"
        parts.append(self._rotor_html(m))
        self.readout.setText("".join(parts))
        self.note.setText(
            "Linearized about the centred journal and the bias, with the sensor at the journal (collocated). "
            "Current saturation and the backup bearings' contact are left out: the page pauses when a journal "
            "reaches its backup bearing." if self.system is not None else
            "Choose <i>Magnetic bearings</i> under the rotor's bearing supports to run the rotor on this bearing.")

    def _rotor_html(self, m: MagneticBearing) -> str:
        if self.system is None:
            return ""
        loads = self.system.bearing_loads
        out = ""
        if any(loads):
            share = max(abs(w) for w in loads) / m.capacity
            out += ("<p>The weight takes a steady " + " and ".join(
                f"{abs(w) / m.current_gain:.3g} A at {n}" for n, w in zip("AB", loads))
                + f" of control current ({share:.0%} of the bias)"
                + (f": <span style='color:{colors.force}'>past the bias one magnet's current would run out, "
                   "and the linear model no longer holds</span>." if share > 1.0 else ".") + "</p>")
        if self.onset is None:
            return out + f"<p>The rotor on it stays stable up to {self.top_rpm:,.0f} rpm.</p>"
        if self.onset.omega == 0.0:
            return out + ("<p>The rotor on it is unstable even at rest: its journals fall onto the magnets.</p>"
                          if self.onset.freq_hz == 0.0 else
                          f"<p>The rotor on it is unstable even at rest, a whirl at {self.onset.freq_hz:,.4g} Hz "
                          "growing where the bearing's damping is negative.</p>")
        whirl = "forward" if self.onset.whirl > 0 else "backward"
        return out + (f"<p>The rotor on it goes unstable at <b>{self.onset.omega * RPM:,.0f} rpm</b>, a {whirl} "
                      f"whirl at {self.onset.freq_hz:,.4g} Hz ({self.onset.order:.2f}×) reaching where the "
                      "bearing's damping turns negative (the <i>Stability</i> tab).</p>")

