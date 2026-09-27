"""Live energy bars: stored energy, the energy ledger, and the energy in each mode."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .style import colors

# Below this the motion has decayed to float noise, so shares are meaningless
# (a 400 N/m spring stretched 1e-7 m holds 2e-12 J).
MIN_SHARE_ENERGY = 1e-12  # J

BY_TYPE_TIP = (
    "<p><b>Stored now:</b> kinetic energy T = ½ẋ<sup>T</sup>Mẋ and potential energy "
    "V = ½x<sup>T</sup>Kx (the springs). As a mode vibrates, energy swings between T and "
    "V twice per cycle while their sum only decays.</p>"
    "<p><b>Since reset:</b> where the energy came from and where it went. <i>In</i>: "
    "energy given by releasing a mode or editing a parameter (a stiffer spring holds more "
    "energy at the same stretch), and the work done by the force, ∫F·ẋ dt, or by the moving "
    "ground, −∫T<sub>1</sub>ẋ<sub>g</sub> dt (T<sub>1</sub>: tension in element 1). <i>Out</i>: "
    "the energy stored now, and the energy the dampers turned into heat, "
    "∫ẋ<sup>T</sup>Cẋ dt. Both are integrated exactly, so the two columns always match. "
    "A force that pushes against the motion takes energy out; it then shows in the "
    "<i>Out</i> column.</p>"
    "<p>The stored-energy bars are scaled to the largest stored energy in the plot window "
    "(frozen while <i>Auto-scale</i> is off); the balance columns to their total.</p>"
)
BY_MODE_TIP = (
    "<p>Each bar is one undamped (classical) mode's share of the stored energy T + V now, "
    "so it shows where the energy is even while the total decays. With "
    "mass-normalized modal coordinates q = Φ<sup>T</sup>Mx,</p>"
    "<p>&nbsp;&nbsp;T + V = Σ ½(q̇<sub>r</sub>² + ω<sub>r</sub>²q<sub>r</sub>²)</p>"
    "<p>exactly, for any damping, because Φ<sup>T</sup>MΦ = I and "
    "Φ<sup>T</sup>KΦ = diag(ω<sub>r</sub>²).</p>"
    "<p>With <b>proportional</b> damping each mode's energy only decays (or is fed by the "
    "force). With <b>non-proportional</b> damping the dampers also move energy between "
    "modes: release mode 3 with c<sub>1</sub> = 15 and watch the energy move to mode 1.</p>"
    "<p>The split uses the classical modes whichever <i>Method</i> is selected: the complex "
    "modes are not orthogonal in energy, so their energies do not add up to the total.</p>"
    "<p>While the ground moves (base excitation) the modes split the motion relative to the "
    "ground, x − x<sub>g</sub>, whose kinetic energy differs from T.</p>"
)


def fmt_energy(joules: float) -> str:
    a = abs(joules)
    for scale, unit in ((1.0, "J"), (1e-3, "mJ"), (1e-6, "µJ"), (1e-9, "nJ")):
        if a >= scale:
            return f"{joules / scale:.3g} {unit}"
    return "0 J" if a < 1e-15 else f"{joules * 1e12:.3g} pJ"


@dataclass
class EnergyState:
    kinetic: float = 0.0
    potential: float = 0.0
    modal: np.ndarray = field(default_factory=lambda: np.zeros(0))
    added: float = 0.0
    work: float = 0.0
    dissipated: float = 0.0
    relative: bool = False  # modal holds the energy of the motion relative to a moving ground

    @property
    def stored(self) -> float:
        return self.kinetic + self.potential


class EnergyBars(QtWidgets.QWidget):
    """Bar chart painted directly with QPainter: a handful of rectangles per frame."""

    MARGIN = 8
    CAPTION = 18  # px for a group's caption above the bars
    LABEL = 30  # px for the labels below the bars

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(250, 180)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.by_mode = False
        self.state = EnergyState()
        self._scale = 0.0

    def set_state(self, state: EnergyState, scale: float) -> None:
        """Show `state`; `scale` (J) is the stored energy drawn at full bar height."""
        self.state = state
        self._scale = max(scale, state.stored)
        self.update()

    # ------------------------------------------------------------------ paint
    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(colors.background))
        font = p.font()
        font.setPointSizeF(font.pointSizeF() * 0.85)
        p.setFont(font)
        area = QtCore.QRectF(self.rect()).adjusted(self.MARGIN, self.MARGIN, -self.MARGIN, -self.MARGIN)
        if self.by_mode:
            self._paint_modes(p, area)
        else:
            left, right = _split(area, 0.55)
            self._paint_stored(p, left)
            p.setPen(QtGui.QPen(QtGui.QColor(colors.faint), 1))
            p.drawLine(QtCore.QLineF(right.left() - 4, area.top(), right.left() - 4, area.bottom()))
            self._paint_ledger(p, right.adjusted(4, 0, 0, 0))
        p.end()

    def _bars_rect(self, group: QtCore.QRectF) -> QtCore.QRectF:
        return group.adjusted(0, self.CAPTION, 0, -self.LABEL)

    def _caption(self, p: QtGui.QPainter, group: QtCore.QRectF, text: str) -> None:
        p.setPen(QtGui.QColor(colors.foreground))
        rect = QtCore.QRectF(group.left(), group.top(), group.width(), self.CAPTION)
        p.drawText(rect, QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop, text)

    def _label(self, p: QtGui.QPainter, x: float, w: float, bars: QtCore.QRectF, lines: str) -> None:
        p.setPen(QtGui.QColor(colors.foreground))
        rect = QtCore.QRectF(x - 20, bars.bottom() + 2, w + 40, self.LABEL)
        p.drawText(rect, QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignTop, lines)

    def _paint_stored(self, p: QtGui.QPainter, group: QtCore.QRectF) -> None:
        s = self.state
        self._caption(p, group, f"Stored now: {fmt_energy(s.stored)}")
        bars = self._bars_rect(group)
        scale = self._scale
        c = colors.energy_colors
        cols = [
            ("T", [(s.kinetic, c["kinetic"])], s.kinetic),
            ("V", [(s.potential, c["potential"])], s.potential),
            ("T + V", [(s.kinetic, c["kinetic"]), (s.potential, c["potential"])], s.stored),
        ]
        for (name, parts, total), (x, w) in zip(cols, _slots(bars, len(cols)), strict=True):
            _stack(p, bars, x, w, parts, scale)
            self._label(p, x, w, bars, f"{name}\n{fmt_energy(total)}")
        _axis(p, bars)

    def _paint_ledger(self, p: QtGui.QPainter, group: QtCore.QRectF) -> None:
        s = self.state
        self._caption(p, group, "Since reset")
        bars = self._bars_rect(group)
        # Energy in = energy out. A negative term (a force taking energy out, or a
        # softer spring after an edit) goes to the other column as a positive one.
        c = colors.energy_colors
        terms = [
            (s.added, c["added"], True),
            (s.work, c["work"], True),
            (s.stored, c["stored"], False),
            (s.dissipated, c["dissipated"], False),
        ]
        cols: dict[bool, list[tuple[float, str]]] = {True: [], False: []}
        for value, color, is_in in terms:
            cols[is_in if value >= 0 else not is_in].append((abs(value), color))
        totals = {k: sum(v for v, _ in parts) for k, parts in cols.items()}
        scale = max(totals.values())
        for (name, is_in), (x, w) in zip((("In", True), ("Out", False)), _slots(bars, 2), strict=True):
            _stack(p, bars, x, w, cols[is_in], scale)
            self._label(p, x, w, bars, f"{name}\n{fmt_energy(totals[is_in])}")
        _axis(p, bars)

    def _paint_modes(self, p: QtGui.QPainter, group: QtCore.QRectF) -> None:
        s = self.state
        if s.relative:  # the modes split the motion relative to the ground, not T + V
            total = float(s.modal.sum())
            self._caption(p, group, f"Share of the energy relative to the ground ({fmt_energy(total)}) in each mode")
        else:
            total = s.stored
            self._caption(p, group, f"Share of the stored energy ({fmt_energy(total)}) in each mode")
        bars = self._bars_rect(group)
        p.setPen(QtGui.QPen(QtGui.QColor(colors.faint), 1, QtCore.Qt.PenStyle.DotLine))
        for frac in (0.25, 0.5, 0.75, 1.0):
            y = bars.bottom() - frac * bars.height()
            p.drawLine(QtCore.QLineF(bars.left(), y, bars.right(), y))
        show = total > MIN_SHARE_ENERGY
        for r, (x, w) in enumerate(_slots(bars, s.modal.size)):
            share = float(s.modal[r]) / total if show else 0.0
            _stack(p, bars, x, w, [(share, colors.mode[r % len(colors.mode)])], 1.0)
            self._label(p, x, w, bars, f"Mode {r + 1}\n{100 * share:.0f}%" if show else f"Mode {r + 1}")
        _axis(p, bars)


def _split(area: QtCore.QRectF, frac: float) -> tuple[QtCore.QRectF, QtCore.QRectF]:
    w = area.width() * frac
    return (
        QtCore.QRectF(area.left(), area.top(), w - 4, area.height()),
        QtCore.QRectF(area.left() + w + 4, area.top(), area.width() - w - 4, area.height()),
    )


def _slots(bars: QtCore.QRectF, k: int) -> list[tuple[float, float]]:
    """(left, width) of k evenly spaced bars across the rectangle."""
    pitch = bars.width() / max(k, 1)
    w = min(0.6 * pitch, 60.0)
    return [(bars.left() + (i + 0.5) * pitch - w / 2, w) for i in range(k)]


def _stack(p: QtGui.QPainter, bars: QtCore.QRectF, x: float, w: float, parts, scale: float) -> None:
    """Stack (value, color) parts upward from the baseline, full height = scale."""
    y = bars.bottom()
    p.setPen(QtCore.Qt.PenStyle.NoPen)
    for value, color in parts:
        h = bars.height() * min(value / scale, 1.0) if scale > 0 else 0.0
        p.setBrush(QtGui.QColor(color))
        p.drawRect(QtCore.QRectF(x, y - h, w, h))
        y -= h


def _axis(p: QtGui.QPainter, bars: QtCore.QRectF) -> None:
    p.setPen(QtGui.QPen(QtGui.QColor(colors.grey), 1))
    p.drawLine(QtCore.QLineF(bars.left(), bars.bottom(), bars.right(), bars.bottom()))


class EnergyPanel(QtWidgets.QWidget):
    """The bars with a selector between the energy by type and the energy by mode."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.view = QtWidgets.QComboBox()
        self.view.addItem("Energy: stored and balance", False)
        self.view.addItem("Energy: by mode", True)
        self.view.currentIndexChanged.connect(self._on_view)
        self.bars = EnergyBars()
        self.legend = QtWidgets.QLabel()
        self.legend.setWordWrap(True)
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(self.view)
        v.addWidget(self.bars, 1)
        v.addWidget(self.legend)
        self._on_view()

    @property
    def by_mode(self) -> bool:
        return bool(self.view.currentData())

    def _on_view(self) -> None:
        self.bars.by_mode = self.by_mode
        tip = BY_MODE_TIP if self.by_mode else BY_TYPE_TIP
        for w in (self.view, self.bars, self.legend):
            w.setToolTip(tip)
        self.legend.setVisible(not self.by_mode)
        self.apply_theme()

    def apply_theme(self) -> None:
        """The legend's swatches; the bars take the current colours when painted."""
        e = colors.energy_colors
        self.legend.setText(
            " ".join(
                f"<span style='color:{c}'>■</span>&nbsp;{name.replace(' ', '&nbsp;')}"
                for name, c in (
                    ("kinetic T", e["kinetic"]),
                    ("potential V", e["potential"]),
                    ("stored", e["stored"]),
                    ("release or edit", e["added"]),
                    ("work by force or ground", e["work"]),
                    ("dissipated", e["dissipated"]),
                )
            )
        )
        self.bars.update()
