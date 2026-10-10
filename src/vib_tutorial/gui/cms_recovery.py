"""Back expansion tab of the Substructuring page: recovering the interior DOFs after the coupled solve.

The coupled (reduced) model only solves for r = [q, x_b]. This tab runs it beside the
full chain, as the Time response tab does, and then recovers the interior masses from r
three ways: by the reduced model's own T (x_i = T_ib x_b + T_iq q), from the boundary
motion alone (static), and by re-solving each substructure's interior with x_b(t)
imposed (enhanced). It plots one interior mass (split into what x_b drags along and
what the modes add), the error of each recovery, and the force in one spring.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core import CMSModel
from ..core.back_expansion import RECOVERIES, RECOVERY_NAMES, RecoveryResponse
from .animation import ChainView
from .cms_time import MIN_SPAN, CMSRunView, SampleBuffer, _fmt
from .style import colors
from .theming import add_legend, mute

DASH, DOT, DASH_DOT = (QtCore.Qt.PenStyle.DashLine, QtCore.Qt.PenStyle.DotLine,
                       QtCore.Qt.PenStyle.DashDotLine)
SOLID = QtCore.Qt.PenStyle.SolidLine
CAPACITY = 100_000  # samples kept for the plots
HEADROOM = 1.6  # y range above the data, relative to below it: room for the legend
TIP = (
    "<p>The system-level (coupled) analysis solves only for the reduced coordinates "
    "r = [q, x<sub>b</sub>]: the kept component modes and the boundary DOFs. The interior masses "
    "are not in it. Afterwards each substructure <b>back-expands</b> them from r, and so recovers "
    "their motion and the forces in their springs (data recovery).</p>"
    "<p><b>Coupled solution:</b> the reduced model's own rows of T, "
    "x<sub>i</sub> = T<sub>ib</sub> x<sub>b</sub> + T<sub>iq</sub> q. For Craig–Bampton "
    "T<sub>ib</sub> = Ψ (the static shape the boundary drags the interior into) and "
    "T<sub>iq</sub> = Φ<sub>k</sub> (the kept fixed-interface modes).</p>"
    "<p><b>Boundary only:</b> the static shape alone, x<sub>i</sub> = −K<sub>ii</sub><sup>−1</sup>"
    "K<sub>ib</sub> x<sub>b</sub>, as when q was not kept from the system run. It misses all the "
    "interior's own vibration.</p>"
    "<p><b>Enhanced:</b> each substructure's interior re-solved on its own, with every one of its "
    "DOFs, and x<sub>b</sub>(t) from the coupled solve imposed. It recovers the modes the reduction "
    "truncated; what is left is the error in x<sub>b</sub> itself.</p>"
    "<p>Hollow, dashed masses in the lower chain are recovered; solid ones (the boundary) come "
    "straight from the coupled solve.</p>"
)


class BackExpansionView(CMSRunView):
    """Controls, the true and the back-expanded chain, and one interior mass and spring in time."""

    tip = TIP

    def _build(self) -> None:
        self.history = SampleBuffer(1, CAPACITY)
        self._n = self._nr = 0
        self._interior = np.zeros(0, dtype=int)

        pick = QtWidgets.QHBoxLayout()
        self.mass_box = QtWidgets.QComboBox()
        self.mass_box.setToolTip("The interior mass whose recovered motion is plotted")
        self.spring_box = QtWidgets.QComboBox()
        self.spring_box.setToolTip("A spring of the same substructure, whose force is recovered")
        self.chain_box = QtWidgets.QComboBox()
        for key in RECOVERIES:
            self.chain_box.addItem(RECOVERY_NAMES[key], key)
        self.chain_box.setToolTip("Which recovery the lower chain shows")
        for w in (QtWidgets.QLabel("Interior mass:"), self.mass_box, QtWidgets.QLabel("Spring:"),
                  self.spring_box):
            pick.addWidget(w)
        pick.addStretch(1)
        pick.addWidget(QtWidgets.QLabel("Lower chain:"))
        pick.addWidget(self.chain_box)

        self.full_view = ChainView(draggable=False)
        self.rec_view = ChainView(draggable=False)
        for view in (self.full_view, self.rec_view):
            view.setMinimumHeight(135)
            view.setToolTip(TIP)

        self.plots = pg.GraphicsLayoutWidget()
        self.x_plot = self.plots.addPlot(row=0, col=0)
        self.x_plot.setLabel("left", "Displacement", units="m")
        self.e_plot = self.plots.addPlot(row=1, col=0)
        self.e_plot.setLabel("left", "Error", units="m")
        self.f_plot = self.plots.addPlot(row=2, col=0)
        self.f_plot.setLabel("left", "Force", units="N")
        self.f_plot.setLabel("bottom", "Time", units="s")
        # No grid and 1 px pens: these redraw every frame (see CMSTimeView._build).
        for p in (self.x_plot, self.e_plot, self.f_plot):
            add_legend(p, offset=(5, 2), colCount=4)  # one row along the top, over the headroom
            p.setClipToView(True)
            p.setDownsampling(auto=True, mode="peak")
            p.setMouseEnabled(x=False, y=False)
            p.hideButtons()
            if p is not self.x_plot:
                p.setXLink(self.x_plot)
        # Displacement: true, coupled recovery, and its two parts. Errors and force: one per recovery.
        self.x_curves = {k: self.x_plot.plot() for k in ("true", "coupled", "from_boundary", "from_modes")}
        self.e_curves = {k: self.e_plot.plot() for k in RECOVERIES}
        self.f_curves = {k: self.f_plot.plot() for k in ("true",) + RECOVERIES}
        # Legend entries are added once and relabelled in place: clearing and re-adding them on
        # every change leaves orphaned labels that can crash Qt's teardown at exit.
        for plot, curves in ((self.x_plot, self.x_curves), (self.e_plot, self.e_curves), (self.f_plot, self.f_curves)):
            for key, curve in curves.items():
                plot.legend.addItem(curve, key)
        self.error_note = QtWidgets.QLabel()
        self.error_note.setWordWrap(True)
        mute(self.error_note)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.header)
        layout.addLayout(self.controls)
        layout.addLayout(pick)
        layout.addWidget(self.full_view, 2)
        layout.addWidget(self.rec_view, 2)
        layout.addWidget(self.plots, 7)
        layout.addWidget(self.error_note)

        self.mass_box.currentIndexChanged.connect(self._on_mass)
        self.spring_box.currentIndexChanged.connect(self._on_spring)
        self.chain_box.currentIndexChanged.connect(self._on_chain)

    # ------------------------------------------------------------- setup
    def apply_theme(self) -> None:
        for view in (self.full_view, self.rec_view):
            view.apply_theme()
        for p in (self.x_plot, self.e_plot, self.f_plot):
            p.legend.setBrush(colors.legend_brush())
        self._set_pens()

    def _styles(self) -> dict[str, tuple[str, QtCore.Qt.PenStyle]]:
        d = self.mass_dof
        e = self.spring
        return {
            "true": (colors.mass[d] if d is not None else colors.strong, SOLID),
            "true_force": (colors.mass[e] if e is not None else colors.strong, SOLID),
            "coupled": (colors.strong, DASH),
            "from_boundary": (colors.muted, DOT),
            "from_modes": (colors.mode[4], DASH_DOT),
            "boundary": (colors.fair, SOLID),
            "enhanced": (colors.fit, SOLID),
        }

    def _set_pens(self) -> None:
        styles = self._styles()
        model = self.model
        cb = model is not None and not model.free
        d, e = self.mass_dof, self.spring
        m = f"x{d + 1}" if d is not None else "x"
        parts = ("Ψ x<sub>b</sub>", "Φ<sub>k</sub> q") if cb else ("R x<sub>b</sub>", "(Φ<sub>ik</sub> − RΦ<sub>bk</sub>) q")
        names = {
            "true": f"{m} true",
            "coupled": f"{m} recovered = sum of:",
            "from_boundary": f"{parts[0]}" + (" (= boundary only)" if cb else ""),
            "from_modes": parts[1],
        }
        k = f"k{e + 1}" if e is not None else "k"
        names.update({f"e_{key}": RECOVERY_NAMES[key] for key in RECOVERIES})
        names.update({f"f_{key}": RECOVERY_NAMES[key] for key in RECOVERIES})
        names["f_true"] = f"{k} true"
        styles["f_true"] = styles["true_force"]
        for prefix, plot, curves in (("", self.x_plot, self.x_curves), ("e_", self.e_plot, self.e_curves),
                                     ("f_", self.f_plot, self.f_curves)):
            for key, curve in curves.items():
                color, style = styles.get(prefix + key, styles.get(key))
                curve.setPen(pg.mkPen(color, width=1, style=style))
                plot.legend.getLabel(curve).setText(names[prefix + key], color=colors.foreground)
        self.x_plot.setTitle(f"Interior mass m{d + 1}" if d is not None else None, size="10pt")
        self.f_plot.setTitle(self.spring_box.currentText() or None, size="10pt")

    def _make_sim(self, model: CMSModel) -> RecoveryResponse:
        return RecoveryResponse(model, self.force)

    def _on_model(self, model: CMSModel) -> None:
        n = model.system.n
        rec = self.sim.recovery
        self._n, self._nr, self._interior = n, model.n_red, rec.interior
        self.history = SampleBuffer(n + model.n_red + rec.interior.size, CAPACITY)
        for view in (self.full_view, self.rec_view):
            view.set_masses(model.system.masses)
            view.gain = 10.0
        self.rec_view.set_recovered(rec.interior)
        self.full_view.setTitle(f"Full model: {n} DOFs, the truth", size="10pt")

        previous = self.mass_dof
        self.mass_box.blockSignals(True)
        self.mass_box.clear()
        for i in rec.interior:
            self.mass_box.addItem(f"m{i + 1} (in {model.substructures[rec.owner(i)].name})", int(i))
        if previous in rec.interior:
            self.mass_box.setCurrentIndex(int(np.searchsorted(rec.interior, previous)))
        self.mass_box.blockSignals(False)
        self._fill_springs(keep=True)

        sub_word = "each substructure's" if len(model.substructures) > 1 else "the substructure's"
        if model.free:
            rows = "x<sub>i</sub> = R x<sub>b</sub> + (Φ<sub>ik</sub> − RΦ<sub>bk</sub>) q"
        else:
            rows = "x<sub>i</sub> = Ψ x<sub>b</sub> + Φ<sub>k</sub> q"
        if rec.interior.size:
            self.header.setText(
                f"<b>Back expansion: the interior, recovered after the coupled solve.</b> The {model.name} "
                f"model solves only for its {model.n_red} reduced coordinates [q, x<sub>b</sub>]; the "
                f"{rec.interior.size} interior mass{'es' if rec.interior.size != 1 else ''} (hollow) are "
                f"recovered from them with {sub_word} own rows of T, {rows}, or from x<sub>b</sub> alone, or "
                "by re-solving the interior with x<sub>b</sub>(t) imposed."
            )
        else:
            self.header.setText(
                "<b>Back expansion.</b> Every mass is a boundary DOF of this model, so the coupled solve "
                "already gives them all: there is no interior to recover. Remove a cut on the left."
            )
        self._on_chain()
        self._set_pens()

    def _fill_springs(self, keep: bool = False) -> None:
        """The springs of the chosen mass's substructure: the one on the mass's left, or the same as before."""
        d = self.mass_dof
        self.spring_box.blockSignals(True)
        previous = self.spring
        self.spring_box.clear()
        if d is not None and self.model is not None:
            sub = self.model.substructures[self.sim.recovery.owner(d)]
            for e in sub.elements:
                left = "ground" if e == 0 else f"m{e}"
                self.spring_box.addItem(f"Spring k{e + 1} ({left}–m{e + 1})", int(e))
            els = [int(e) for e in sub.elements]
            self.spring_box.setCurrentIndex(els.index(previous) if keep and previous in els else els.index(d))
        self.spring_box.blockSignals(False)

    @property
    def mass_dof(self) -> int | None:
        return self.mass_box.currentData()

    @property
    def spring(self) -> int | None:
        return self.spring_box.currentData()

    def select(self, mass: int, spring: int | None = None) -> None:
        """Plot interior mass `mass` (0-based) and, if given, spring `spring` (0-based element)."""
        self.mass_box.setCurrentIndex(self.mass_box.findData(mass))
        if spring is not None:
            self.spring_box.setCurrentIndex(self.spring_box.findData(spring))

    def _on_mass(self) -> None:
        self._fill_springs()
        self._set_pens()
        self._draw()

    def _on_spring(self) -> None:
        self._set_pens()
        self._draw()

    def _on_chain(self) -> None:
        if self.model is not None:
            key = self.chain_box.currentData()
            self.rec_view.setTitle(
                f"Back-expanded from the {self.model.short_name} coupled solve: boundary solved, "
                f"interior recovered — {RECOVERY_NAMES[key].lower()}", size="10pt")
        self._draw()

    # ------------------------------------------------------------- loop
    def _clear_display(self) -> None:
        self.history.clear()

    def _advance(self, duration: float) -> None:
        s = self.sim.record(duration)
        self.history.extend(s.t, np.hstack([s.x_full, s.r, s.x_enhanced[:, self._interior]]))

    def _recovered(self, r: np.ndarray, x_enh_interior: np.ndarray) -> dict[str, np.ndarray | None]:
        """Each recovery's displacements (k, n) from the reduced coordinates."""
        rec = self.sim.recovery
        coupled = rec.coupled(r)
        enhanced = coupled.copy()
        enhanced[:, self._interior] = x_enh_interior
        return {"coupled": coupled, "boundary": rec.boundary_only(r), "enhanced": enhanced}

    def _draw(self) -> None:
        sim = self.sim
        d, e = self.mass_dof, self.spring
        if sim is None:
            for c in (*self.x_curves.values(), *self.e_curves.values(), *self.f_curves.values()):
                c.setData([], [])
            self.error_note.setText("")
            return
        n, nr = self._n, self._nr
        rec = sim.recovery
        # The chains, now.
        r_now = sim.r[None, :]
        now = self._recovered(r_now, sim.x_enhanced[None, self._interior])
        key = self.chain_box.currentData()
        x_rec = now[key][0] if now[key] is not None else sim.x_reduced
        x_true = sim.x_full
        t, y = self.history.window(self.window.value())
        peak = max(float(np.abs(x_true).max()), float(np.abs(x_rec).max()),
                   float(np.abs(y[:, :n]).max()) if y.size else 0.0)
        f = self.force.value()
        tip, amp = n - 1, abs(self.force.settings.amplitude)
        self.full_view.update_state(x_true, peak, f, tip, amp)
        self.rec_view.update_state(x_rec, peak, f, tip, amp)
        if d is None or not t.size:
            for c in (*self.x_curves.values(), *self.e_curves.values(), *self.f_curves.values()):
                c.setData([], [])
            self.error_note.setText("" if d is not None else "No interior masses to recover.")
            return

        xf, r, xe = y[:, :n], y[:, n : n + nr], y[:, n + nr :]
        recs = self._recovered(r, xe)
        true_x = xf[:, d]
        shown = {"true": true_x, "coupled": recs["coupled"][:, d],
                 "from_boundary": rec.from_boundary(r)[:, d], "from_modes": rec.from_modes(r)[:, d]}
        for k, v in shown.items():
            self.x_curves[k].setData(t, v)
        true_f = rec.spring_forces(xf)[:, e]
        self.f_curves["true"].setData(t, true_f)
        errs, ferrs = {}, {}
        for k in RECOVERIES:
            x = recs[k]
            if x is None:
                self.e_curves[k].setData([], [])
                self.f_curves[k].setData([], [])
                continue
            errs[k] = x[:, d] - true_x
            force = rec.spring_forces(x)[:, e]
            ferrs[k] = force - true_f
            self.e_curves[k].setData(t, errs[k])
            self.f_curves[k].setData(t, force)

        w = self.window.value()
        x0 = max(t[-1] - w, 0.0) if t[-1] > w else 0.0
        self.x_plot.setXRange(x0, x0 + w, padding=0)
        span = max(MIN_SPAN, *(float(np.abs(v).max()) for v in shown.values()))
        self.x_plot.setYRange(-span, HEADROOM * span, padding=0.05)
        espan = max(max(float(np.abs(v).max()) for v in errs.values()), MIN_SPAN * 1e-3)
        self.e_plot.setYRange(-espan, HEADROOM * espan, padding=0.05)
        fspan = max(float(np.abs(true_f).max()), *(float(np.abs(v + true_f).max()) for v in ferrs.values()), 1e-9)
        self.f_plot.setYRange(-fspan, HEADROOM * fspan, padding=0.05)

        def rel(err: np.ndarray, ref: np.ndarray) -> str:
            rms = float(np.sqrt(np.mean(ref**2)))
            if rms <= 0 or rms < 1e-12 * max(1.0, float(np.abs(ref).max())):
                return "–"
            return f"{100 * float(np.sqrt(np.mean(err**2))) / rms:.3g}%"

        parts = [f"{RECOVERY_NAMES[k].split(' (')[0].lower()} <b>{rel(errs[k], true_x)}</b> / "
                 f"<b>{rel(ferrs[k], true_f)}</b>" for k in RECOVERIES if k in errs]
        b = rec.boundary
        xb_err = recs["coupled"][:, b] - xf[:, b]
        xb_rel = float(np.sqrt(np.mean(xb_err**2))) / max(float(np.sqrt(np.mean(xf[:, b] ** 2))), 1e-300)
        missing = "" if "boundary" in errs else " (no boundary-only recovery: K<sub>ii</sub> is singular)"
        self.error_note.setText(
            f"RMS error over the last {min(w, t[-1] - t[0]):.3g} s, of x{d + 1} / of the force in k{e + 1}: "
            + ", ".join(parts) + missing
            + f". Boundary motion x<sub>b</sub> from the coupled solve: {100 * xb_rel:.3g}% "
            f"(largest |error| {_fmt(float(np.abs(xb_err).max()))}): the only error left in the enhanced recovery."
            f"   t = {sim.t:.2f} s"
        )
