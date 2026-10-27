"""Main window: wires the panels and plots to the simulator on a real-time timer."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import replace

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..core import (
    ChainSystem,
    ForceController,
    ForceKind,
    ForceSettings,
    Simulator,
    element_forces,
    friction_zeta,
    kinetic_energy,
    modal_analysis,
    modal_coordinate_map,
    modal_energies,
    drag_shape,
    potential_energy,
)
from ..core.presets import Preset, without_absorber
from .animation import ChainView
from .background import COUPLING_TIP, make_background_view
from .energy import EnergyPanel, EnergyState
from .history import History
from .layouts import FlowLayout, ColumnScroll, group
from .modes import Method, mode_entries
from .panels import ForcePanel, ParameterPanel, SimControls, spin
from .plots import FrfPlot, ModalTable, ModeShapePlot, PhasorPanel, TimeHistoryPlot
from .sim_trials import TRIALS as SIM_TRIALS
from .tab_groups import PageGroups
from . import links
from . import frame_stats, settle_plots, theming
from .style import SYSTEM, colors

FRAME_MS = 16  # ~60 fps
FRAME_BUDGET = 0.012  # s of computing per frame: past it (friction on a stiff chain) the simulation runs slower
PROBE_STEPS = 200  # steps in a frame while the cost of a step isn't known yet (a new system)
# The history keeps samples at least STORE_DT apart, or STORE_PER_PERIOD to the fastest mode's or the
# forcing's period (the steps take 40-80) where that's shorter, or a tenth of a pulse: a fast step
# (a stiff chain) then doesn't fill it in a few seconds, or cost the recording of every step.
STORE_DT = 1e-3  # s
STORE_PER_PERIOD = 16

# The pages in reading order, in groups: (group title, ((attribute, tab title), ...)), from links.COURSE.
# All but the Simulation page are imported and built the first time they're opened (MainWindow._page).
PAGE_GROUPS = tuple((group, tuple((name, title) for name, title, _ in pages)) for group, pages in links.COURSE)
# The pages built on first use, in reading order: (attribute, tab title).
LAZY_PAGES = tuple(page for _, pages in PAGE_GROUPS for page in pages if page[0] != "sim_page")


def _follows_chain(page: QtWidgets.QWidget) -> bool:
    """Whether a page shows the chain (and is given it); the rotor, Duffing & Mathieu, torsional and continuous
    systems pages have models of their own."""
    return getattr(page, "follows_chain", True)


def _page_class(name: str) -> type[QtWidgets.QWidget]:
    # Plain imports, not importlib, so PyInstaller still finds these modules.
    if name == "sdof_page":
        from .sdof import SdofPage
        return SdofPage
    if name == "damping_page":
        from .damping_models import DampingModelsPage
        return DampingModelsPage
    if name == "transient_page":
        from .transient import TransientPage
        return TransientPage
    if name == "signals_page":
        from .signals import SignalsPage
        return SignalsPage
    if name == "isolation_page":
        from .isolation import IsolationPage
        return IsolationPage
    if name == "frf_page":
        from .frf_matrix import FrfMatrixPage
        return FrfMatrixPage
    if name == "shock_page":
        from .shock import ShockPage
        return ShockPage
    if name == "updating_page":
        from .model_updating import ModelUpdatingPage
        return ModelUpdatingPage
    if name == "complex_page":
        from .complex_modes import ComplexModesPage
        return ComplexModesPage
    if name == "coupling_page":
        from .modal_coupling import ModalCouplingPage
        return ModalCouplingPage
    if name == "cms_page":
        from .substructuring import SubstructuringPage
        return SubstructuringPage
    if name == "duffing_page":
        from .duffing import DuffingPage
        return DuffingPage
    if name == "torsional_page":
        from .torsional import TorsionalPage
        return TorsionalPage
    if name == "continuum_page":
        from .continuum import ContinuumPage
        return ContinuumPage
    if name == "rotor_page":
        from .rotor import RotorPage
        return RotorPage
    if name == "oma_page":
        from .oma import OmaPage
        return OmaPage
    if name == "random_page":
        from .random_vibration import RandomVibrationPage
        return RandomVibrationPage
    if name == "periodic_page":
        from .periodic import PeriodicPage
        return PeriodicPage
    if name == "energy_page":
        from .energy_methods import EnergyMethodsPage
        return EnergyMethodsPage
    if name == "integration_page":
        from .integration import IntegrationPage
        return IntegrationPage
    if name == "fatigue_page":
        from .fatigue import FatiguePage
        return FatiguePage
    if name == "self_excited_page":
        from .self_excited import SelfExcitedPage
        return SelfExcitedPage
    if name == "active_control_page":
        from .active_control import ActiveControlPage
        return ActiveControlPage
    if name == "pendulum_page":
        from .pendulum import PendulumPage
        return PendulumPage
    if name == "beams_page":
        from .beams_plates import BeamsPlatesPage
        return BeamsPlatesPage
    if name == "seismic_page":
        from .seismic import SeismicPage
        return SeismicPage
    if name == "quarter_car_page":
        from .quarter_car import QuarterCarPage
        return QuarterCarPage
    if name == "variability_page":
        from .variability import VariabilityPage
        return VariabilityPage
    from .modal_test import ModalTestPage
    return ModalTestPage
MODE_ANIMATION_HZ = 0.5  # visual rate for the animated mode-shape plot

METHOD_TIP = (
    "<p><b>Classical:</b> solve the undamped problem Kφ = ω²Mφ for N real mode shapes, "
    "then estimate each mode's damping from φᵀCφ. Exact only for proportional damping.</p>"
    "<p><b>State-space:</b> rewrite the N second-order equations as 2N first-order ones, "
    "ż = Az with z = [x, ẋ], and solve the full damped eigenproblem Aψ = λψ. This gives "
    "2N eigenvalues: complex-conjugate pairs λ, λ* for oscillatory modes (and real ones for "
    "overdamped motion), with complex mode shapes. Exact for any damping.</p>"
)
RELEASE_TIP_COMMON = (
    "<p><b>Free vibration from a mode shape.</b></p>"
    "<p>Switches off the applied force, puts the masses into the mode selected in the table "
    "above (the first row if none is selected), then lets go.</p>"
)
RELEASE_TIP = {
    Method.CLASSICAL: RELEASE_TIP_COMMON
    + "<p>The masses start at rest in the real (undamped) shape.</p>"
    "<p>With <b>proportional</b> damping only that mode responds: every mass oscillates "
    "at the mode's damped frequency f_d, keeping the same shape while the motion "
    "decays at the rate set by ζ.</p>"
    "<p>With <b>non-proportional</b> damping the real shape is not an exact "
    "mode, so other modes are excited too and the shape drifts as it decays.</p>",
    Method.STATE_SPACE: RELEASE_TIP_COMMON
    + "<p>The motion of one eigenvalue pair is x(t) = Re(ψ e<sup>λt</sup>), so the masses "
    "start at displacement Re(ψ) <i>with velocity</i> Re(λψ). That excites only this "
    "eigenvalue pair, even with non-proportional damping: the masses keep their relative "
    "amplitudes and phase lags while the motion decays.</p>"
    "<p>λ and its conjugate λ* give the same real motion, so releasing either one does "
    "the same thing. A real eigenvalue releases a non-oscillatory decay.</p>",
}
COORDS_TIP_COMMON = (
    "<p><b>Physical coordinates:</b> the displacement x<sub>i</sub> of each mass.</p>"
    "<p><b>Modal coordinates:</b> the same motion split into one curve per mode, "
    "in mode colors. Each curve is that mode's share of the displacement of the mass it "
    "moves most (where its shape is 1), so its size compares directly with x. "
    "The past motion is projected onto the modes of the current parameters. While the "
    "ground moves, the modes split the motion relative to it, x − x<sub>g</sub>.</p>"
)
COORDS_TIP = {
    Method.CLASSICAL: COORDS_TIP_COMMON
    + "<p><b>Classical:</b> q = Φ<sup>T</sup>Mx with the real mode shapes, and "
    "x = Σ φ<sub>r</sub> q<sub>r</sub> exactly. With <b>proportional</b> damping each "
    "q<sub>r</sub> is an independent damped oscillator: release a mode and only its curve "
    "moves. With <b>non-proportional</b> damping the damping couples them, so the other "
    "curves pick up motion too.</p>",
    Method.STATE_SPACE: COORDS_TIP_COMMON
    + "<p><b>State-space:</b> η = V<sup>−1</sup>z with the complex eigenvectors "
    "V = [ψ; λψ], using velocities as well as displacements. A conjugate pair adds "
    "2 Re(ψη) to x, so its curve is 2 Re(η); a real root's curve is η. These decouple "
    "for <i>any</i> damping: release a complex mode and only its curve moves.</p>",
}
ENERGY_COORDS_TIP = (
    "<p><b>Energy:</b> the energy bars beside the animation as time histories. "
    "Kinetic energy T, potential energy V and the stored energy T + V; the energy given "
    "by a release or a parameter edit E<sub>0</sub>, the work done by the force (or the moving "
    "ground) W and "
    "the energy dissipated by the dampers D and by friction D<sub>f</sub>, all since the last "
    "reset. At every instant</p>"
    "<p>&nbsp;&nbsp;E<sub>0</sub> + W = T + V + D + D<sub>f</sub>.</p>"
    "<p>Release a mode: T and V swap twice per cycle while T + V decays and D rises to "
    "E<sub>0</sub>. Drive at resonance: W and D climb together once T + V has built up.</p>"
)
# Energy curves: (legend, key of Theme.energy_colors), in the order of energy_curves().
ENERGY_CURVES = [
    ("T", "kinetic"),
    ("V", "potential"),
    ("T + V", "stored"),
    ("E₀", "added"),
    ("W", "work"),
    ("D", "dissipated"),
    ("D<sub>f</sub>", "friction"),
]
MIN_ENERGY_SPAN = 1e-9  # J; smaller energies are the float noise of decayed motion
FORCES_COORDS_TIP = (
    "<p><b>Element forces:</b> the tension in each spring and damper, in the colors of "
    "their rows under <i>System parameters</i>. Element i joins mass i−1 to mass i "
    "(element 1 joins mass 1 to the ground).</p>"
    "<p>&nbsp;&nbsp;spring: k<sub>i</sub>(x<sub>i</sub> − x<sub>i−1</sub>), "
    "damper: c<sub>i</sub>(ẋ<sub>i</sub> − ẋ<sub>i−1</sub>)</p>"
    "<p>Positive is tension: the element is stretched (or stretching) and pulls its two "
    "masses together. <i>Spring + damper</i> is the total force the element carries; "
    "for element 1 that is the force on the ground. With ground motion, element 1 stretches "
    "by x<sub>1</sub> − x<sub>g</sub>.</p>"
    "<p>Apply a step force at the last mass: once the motion settles, every spring carries "
    "the full force and the dampers carry none. While it vibrates, each element also carries "
    "the inertia force of the masses beyond it. The damper force leads the spring force "
    "by a quarter cycle.</p>"
)
# Which element forces to plot: (combo text, key, legend format).
ELEMENT_FORCES = [
    ("Springs", "spring", "k{}"),
    ("Dampers", "damper", "c{}"),
    ("Spring + damper", "total", "k{0} + c{0}"),
]
MIN_FORCE_SPAN = 1e-4  # N
THEME_TIP = (
    "<p>Colours of the whole app: plots, animation and controls.</p>"
    "<p><b>System</b> follows the desktop's light or dark setting, and switches with it.</p>"
)
THEORY_TIP = (
    "<p>Show this page's <i>Theory</i> tab: what the page shows, the ideas behind it, things to try, "
    "and its symbols. (F1)</p>"
    "<p>Ctrl+1 to Ctrl+6 switch between the groups of pages.</p>"
)
ABOUT_HTML = """<h3>Vibration Tutorial {version}</h3>
<p>An interactive teaching tool for mechanical vibration: a chain of masses, springs and dampers, and the
systems built from it. Start with <i>Foundations</i> and read each page's <i>Theory</i> tab (F1); the pages
are in the order they are meant to be read.</p>
<p><b>For educational use only.</b> This program was written with an AI coding assistant (Claude Code).
Although it has an automated test suite, its results are <b>not guaranteed to be technically correct</b>:
the simulations, analyses and theory notes may contain errors. Check any calculation independently before
relying on it, and do not use it for design, certification, safety-related or other engineering
decisions.</p>
<p>Provided "as is", without warranty of any kind. Released under the MIT License.
<a href="https://github.com/WookieLumberjack/vib_tutorial">github.com/WookieLumberjack/vib_tutorial</a></p>
"""
COMPARE_TIP = (
    "<p>Also draws, thin and dashed, the frequency response of the chain with its last mass "
    "removed. When the last mass is a vibration absorber or tuned mass damper (see "
    "<i>System parameters → Preset</i>), this is the structure before it was added.</p>"
    "<p>An undamped absorber puts an antiresonance at its own frequency √(k/m), where the "
    "mass it hangs on stands still, and splits the old resonance into two peaks. A tuned "
    "mass damper's damper flattens those two peaks.</p>"
)
FRICTION_FRF_TIP = (
    "<p>Friction is nonlinear, so the chain has no single frequency response: its force stays "
    "the same as the motion grows, so it matters most for small motions.</p>"
    "<p>The dash-dot curves keep only the first harmonic of each friction force, a square wave "
    "against the velocity of size F<sub>f</sub>: its fundamental is a damper "
    "c<sub>eq</sub> = 4F<sub>f</sub>/πωX that weakens as the amplitude X grows. The chain is "
    "solved at each frequency with these dampers, at the amplitude set on the Excitation panel "
    "(a force, or a ground motion). A mass that friction holds still through the whole cycle "
    "leaves a gap.</p>"
    "<p>Raise the amplitude and the curves approach the linear ones; lower it and friction "
    "flattens the peaks. It's an approximation: good while every mass slides through the cycle, "
    "rougher near sticking, where the real motion stick-slips. Compare a harmonic drive on the "
    "simulation, or the stepped sine on the Virtual modal test.</p>"
)
RELEASE_FIT_TIP = (
    "<p>While <i>Auto-scale animation and plots</i> is on, the plot window is also "
    "fitted to this mode (Simulation \u2192 Fit window cycles).</p>"
)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Vibration Tutorial — lumped mass-spring-damper chain")

        system = ChainSystem.uniform(4)
        self.force = ForceController(ForceSettings(target=system.n - 1))
        self.sim = Simulator(system, self.force)
        self.history = History(system.n)
        self.modal = modal_analysis(system)
        self.method = Method.CLASSICAL
        self.entries = mode_entries(self.modal, self.method)
        self.running = True
        self._energy_scale = 0.0  # J at full bar height; follows the plot window while auto-scaling
        self._ground_velocity = 0.0  # m/s over the last step, for the motion relative to the ground
        self._force_active = False  # to notice a chirp or pulse ending by itself

        # --- left: inputs
        self.params = ParameterPanel(system)
        self.force_panel = ForcePanel(self.force)
        self.controls = SimControls()
        trials = QtWidgets.QGroupBox("Things to try")
        tv = QtWidgets.QVBoxLayout(trials)
        self.trial = QtWidgets.QComboBox()
        self.trial.addItem("Pick one…")
        self.trial.addItems([t.name for t in SIM_TRIALS])
        self.trial.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.trial.setToolTip("<p>Sets up the chain and the controls to show one idea, and says what to look "
                              "for. The Theory tab (F1) lists them all.</p>")
        self.trial.activated.connect(self._on_trial)
        tv.addWidget(self.trial)
        self.trial_note = QtWidgets.QLabel(
            "New here? Press <b>Space</b> to switch the force on, or pick one of these. Then read the "
            "<i>Theory</i> tab (F1).")
        self.trial_note.setWordWrap(True)
        theming.mute(self.trial_note)
        tv.addWidget(self.trial_note)
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        for w in (trials, self.params, self.force_panel, self.controls):
            lv.addWidget(w)
        lv.addStretch(1)
        left_scroll = ColumnScroll(left)

        # --- center: animation + time histories
        self.chain = ChainView()
        self.energy = EnergyPanel()
        self.time_plot = TimeHistoryPlot()
        self.coords_combo = QtWidgets.QComboBox()
        self.coords_combo.addItem("Physical: mass displacements x", "physical")
        self.coords_combo.addItem("Modal: one curve per mode", "modal")
        self.coords_combo.addItem("Energy: T, V, work and dissipation", "energy")
        self.coords_combo.addItem("Element forces: springs and dampers", "forces")
        self.element_combo = QtWidgets.QComboBox()
        for text, key, _ in ELEMENT_FORCES:
            self.element_combo.addItem(text, key)
        self.element_combo.setToolTip(FORCES_COORDS_TIP)
        self.element_combo.setVisible(False)
        coords_label = QtWidgets.QLabel("Plot coordinates:")
        self._modal_map = np.zeros((0, 0))  # states -> modal coordinates, set with the modes
        plot_box = QtWidgets.QWidget()
        pv = QtWidgets.QVBoxLayout(plot_box)
        pv.setContentsMargins(0, 0, 0, 0)
        coords_row = QtWidgets.QHBoxLayout()
        coords_row.setContentsMargins(6, 0, 0, 0)
        coords_row.addWidget(coords_label)
        coords_row.addWidget(self.coords_combo)
        coords_row.addWidget(self.element_combo)
        coords_row.addStretch(1)
        pv.addLayout(coords_row)
        pv.addWidget(self.time_plot, 1)
        self._coords_widgets = (coords_label, self.coords_combo)
        top = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        top.addWidget(self.chain)
        top.addWidget(self.energy)
        top.setStretchFactor(0, 1)
        top.setSizes([480, 270])
        center = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        center.addWidget(top)
        center.addWidget(plot_box)
        center.setSizes([300, 450])

        # --- right: modal reference
        self.method_combo = QtWidgets.QComboBox()
        for m in Method:
            self.method_combo.addItem(m.value, m)
        self.method_combo.setToolTip(METHOD_TIP)
        # Cut short with "…" when the pane is narrow, rather than set its width (the list shows it all).
        self.method_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.method_combo.setMinimumContentsLength(16)
        method_label = QtWidgets.QLabel("Method:")
        method_label.setToolTip(METHOD_TIP)
        self.table = ModalTable()
        self.mode_plot = ModeShapePlot()
        self.phasor_plot = PhasorPanel()
        self.phasor_plot.setToolTip(
            "<p>The selected mode shape ψ in the complex plane: one arrow per mass "
            "(mass colors), length = amplitude, angle = phase.</p>"
            "<p>With <i>Animate mode shapes</i> on, the arrows rotate as ψe<sup>iω_d t</sup> "
            "(anticlockwise for Im λ &gt; 0, clockwise for its conjugate λ*). The circles on "
            "the real axis are the real parts, i.e. the masses' displacements.</p>"
            "<p>All arrows on one line (0° or 180° apart): a real mode, a standing wave. "
            "Arrows at other angles: the masses peak at different times.</p>"
        )
        self.phasor_plot.setMinimumWidth(180)
        self.modal_note = QtWidgets.QLabel()
        self.modal_note.setWordWrap(True)
        self.modal_note.setToolTip(COUPLING_TIP)
        self.modal_note.linkActivated.connect(lambda _: self.tabs.setCurrentWidget(self.background))
        self.animate_modes = QtWidgets.QCheckBox("Animate mode shapes")
        self.release_amp = spin(0.001, 1000.0, 20.0, 3, " mm")
        amp_tip = (
            "<p>Starting displacement of the mass that moves most when a mode is released. "
            "The other masses are scaled by the mode shape.</p>"
            "<p>Without friction the system is linear, so this changes the size of the motion but "
            "not how fast it decays; the decay rate is set by the mode's damping ratio ζ. To watch "
            "a well-damped mode, slow the simulation down (Simulation → Speed).</p>"
            "<p>With friction it does matter: friction takes a fixed amplitude out per cycle, so a "
            "small release dies out sooner. The table's <i>ζ friction</i> column is worked out at "
            "this amplitude.</p>"
        )
        self.release_amp.setToolTip(amp_tip)
        self.release_amp.valueChanged.connect(self._refresh_table)
        amp_label = QtWidgets.QLabel("Initial displacement:")
        amp_label.setToolTip(amp_tip)
        self.release_button = release = QtWidgets.QPushButton("Release selected mode")
        release.clicked.connect(self._release_mode)
        modal_tab = QtWidgets.QWidget()
        mv = QtWidgets.QVBoxLayout(modal_tab)
        method_row = QtWidgets.QHBoxLayout()
        method_row.addWidget(method_label)
        method_row.addWidget(self.method_combo, 1)
        mv.addLayout(method_row)
        mv.addWidget(self.table)
        mv.addWidget(self.modal_note)
        plots_row = QtWidgets.QHBoxLayout()
        plots_row.addWidget(self.mode_plot, 3)
        plots_row.addWidget(self.phasor_plot, 2)
        mv.addLayout(plots_row, 1)
        row = FlowLayout()
        row.add(self.animate_modes, group(amp_label, self.release_amp, release))
        mv.addLayout(row)

        self.frf_plot = FrfPlot()
        self.frf_compare = QtWidgets.QCheckBox("Compare with the chain without its last mass (dashed)")
        self.frf_compare.setToolTip(COMPARE_TIP)
        frf_tab = QtWidgets.QWidget()
        fv = QtWidgets.QVBoxLayout(frf_tab)
        fv.setContentsMargins(0, 4, 0, 0)
        fv.addWidget(self.frf_compare)
        self.frf_friction_note = QtWidgets.QLabel(
            "With friction the response depends on the input's size. Dash-dot: an approximate response "
            "at the Excitation panel's amplitude (describing function); a gap marks a mass that sticks."
        )
        self.frf_friction_note.setWordWrap(True)
        self.frf_friction_note.setToolTip(FRICTION_FRF_TIP)
        theming.mute(self.frf_friction_note)
        fv.addWidget(self.frf_friction_note)
        fv.addWidget(self.frf_plot, 1)
        self.background = make_background_view()
        tabs = self.tabs = QtWidgets.QTabWidget()
        tabs.addTab(modal_tab, "Modal analysis")
        self.frf_tab = frf_tab
        tabs.addTab(frf_tab, "Frequency response")
        tabs.addTab(self.background, "Theory")

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(left_scroll)
        splitter.addWidget(center)
        splitter.addWidget(tabs)
        splitter.setSizes([400, 750, 550])
        self.sim_page = splitter

        # --- the other pages: same chain, each its own layout. An empty placeholder
        # holds each tab until the page is first opened (see _page).
        self.pages = PageGroups()
        self._built: dict[str, QtWidgets.QWidget] = {}
        self._placeholders: dict[str, QtWidgets.QWidget] = {}
        for group_title, pages in PAGE_GROUPS:
            tabs_in_group = []
            for name, title in pages:
                if name == "sim_page":
                    tabs_in_group.append((self.sim_page, title))
                    continue
                self._placeholders[name] = QtWidgets.QWidget()
                tabs_in_group.append((self._placeholders[name], title))
            self.pages.add_group(group_title, *tabs_in_group)
        # The window opens on the Simulation page, the app's home, though the course starts a group earlier.
        self.pages.setCurrentWidget(self.sim_page)
        self.pages.currentChanged.connect(self._on_page_changed)
        self._wire_links(self.sim_page)
        self.theme_combo = QtWidgets.QComboBox()
        self.theme_combo.addItems(theming.names())
        self.theme_combo.setCurrentText(theming.current_name())
        self.theme_combo.setToolTip(THEME_TIP)
        theme_label = QtWidgets.QLabel("Theme:")
        theme_label.setToolTip(THEME_TIP)
        corner = QtWidgets.QWidget()
        cl = QtWidgets.QHBoxLayout(corner)
        cl.setContentsMargins(6, 2, 6, 2)
        self.theory_button = QtWidgets.QToolButton()
        self.theory_button.setText("Theory")
        self.theory_button.setToolTip(THEORY_TIP)
        self.theory_button.clicked.connect(self.show_theory)
        about = QtWidgets.QToolButton()
        about.setText("About")
        about.setToolTip("Version, disclaimer and licence.")
        about.clicked.connect(self._about)
        cl.addWidget(self.theory_button)
        cl.addWidget(about)
        cl.addSpacing(8)
        cl.addWidget(theme_label)
        cl.addWidget(self.theme_combo)
        self.pages.setCornerWidget(corner, QtCore.Qt.Corner.TopRightCorner)
        self.setCentralWidget(self.pages)
        self.resize(1700, 900)
        QtGui.QShortcut(QtGui.QKeySequence("F1"), self, self.show_theory)
        for i in range(len(PAGE_GROUPS)):  # Ctrl+1 … Ctrl+6: a group, at the page it last showed
            QtGui.QShortcut(QtGui.QKeySequence(f"Ctrl+{i + 1}"), self,
                            lambda i=i: self.pages.groups.setCurrentIndex(i))

        # --- wiring
        self.params.changed.connect(self._on_system_changed)
        self.params.preset_chosen.connect(self._on_preset)
        self.frf_compare.toggled.connect(self._on_force_changed)
        self.force_panel.settings_changed.connect(self._on_force_changed)
        self.controls.run_toggled.connect(self._on_run_toggled)
        self.controls.reset_clicked.connect(self._reset)
        self.controls.auto_scale.toggled.connect(self._on_auto_scale)
        self.animate_modes.toggled.connect(lambda on: on or self._animate_modes(0.0))
        self.table.itemSelectionChanged.connect(self._on_mode_selected)
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        self.coords_combo.currentIndexChanged.connect(self._set_plot_curves)
        self.element_combo.currentIndexChanged.connect(self._set_plot_curves)
        self.chain.mass_grabbed.connect(self._on_mass_grabbed)
        self.chain.mass_dragged.connect(self._on_mass_dragged)
        self.chain.mass_released.connect(self._on_mass_released)
        self.theme_combo.textActivated.connect(self._on_theme_picked)
        app = QtWidgets.QApplication.instance()
        app.styleHints().colorSchemeChanged.connect(self._on_desktop_scheme)

        self._apply_dof(system.n)
        self._refresh_modal()

        self._clock = QtCore.QElapsedTimer()
        self._clock.start()
        self._last_wall = 0.0
        self._sim_target = 0.0
        self._step_cost = 0.0  # s of computing per simulator step (with the history it feeds); 0 = not known
        self._started: tuple[float, bool] | None = None  # (when, ground motion?) of the advance in flight
        self._carry = 0  # steps since the last sample stored in the history
        # Single-shot timer re-armed after each frame: if a frame runs long, the
        # event loop still gets to handle input before the next one starts.
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._frame)
        self._timer.start(FRAME_MS)
        self.frame_overlay = frame_stats.install(self)  # only with VIB_FRAME_STATS set

    # --------------------------------------------------------------- pages
    sdof_page = property(lambda self: self._page("sdof_page"))
    transient_page = property(lambda self: self._page("transient_page"))
    damping_page = property(lambda self: self._page("damping_page"))
    isolation_page = property(lambda self: self._page("isolation_page"))
    signals_page = property(lambda self: self._page("signals_page"))
    frf_page = property(lambda self: self._page("frf_page"))
    coupling_page = property(lambda self: self._page("coupling_page"))
    cms_page = property(lambda self: self._page("cms_page"))
    test_page = property(lambda self: self._page("test_page"))
    oma_page = property(lambda self: self._page("oma_page"))
    rotor_page = property(lambda self: self._page("rotor_page"))
    duffing_page = property(lambda self: self._page("duffing_page"))
    torsional_page = property(lambda self: self._page("torsional_page"))
    continuum_page = property(lambda self: self._page("continuum_page"))
    random_page = property(lambda self: self._page("random_page"))
    complex_page = property(lambda self: self._page("complex_page"))
    updating_page = property(lambda self: self._page("updating_page"))
    shock_page = property(lambda self: self._page("shock_page"))
    variability_page = property(lambda self: self._page("variability_page"))
    quarter_car_page = property(lambda self: self._page("quarter_car_page"))
    seismic_page = property(lambda self: self._page("seismic_page"))
    beams_page = property(lambda self: self._page("beams_page"))
    pendulum_page = property(lambda self: self._page("pendulum_page"))
    active_control_page = property(lambda self: self._page("active_control_page"))
    self_excited_page = property(lambda self: self._page("self_excited_page"))
    fatigue_page = property(lambda self: self._page("fatigue_page"))
    integration_page = property(lambda self: self._page("integration_page"))
    energy_page = property(lambda self: self._page("energy_page"))
    periodic_page = property(lambda self: self._page("periodic_page"))

    def _page(self, name: str) -> QtWidgets.QWidget:
        """The page `name` from LAZY_PAGES, built and put in its tab on first use."""
        page = self._built.get(name)
        if page is not None:
            return page
        page = self._built[name] = _page_class(name)()
        if _follows_chain(page):
            page.dof_requested.connect(self.params.dof.setValue)
            page.edit_parameters.connect(lambda: self.pages.setCurrentWidget(self.sim_page))
            if hasattr(page, "system_requested"):
                page.system_requested.connect(self.params.set_system)
            page.set_system(self.sim.system, self.modal)
        # Put the page in its placeholder's tab (showing, if that was): only the real page is ever shown.
        holder = self._placeholders.pop(name)
        self.pages.replace(holder, page)
        self._wire_links(page)
        holder.deleteLater()
        return page

    def show_theory(self) -> QtWidgets.QTextBrowser | None:
        """Bring the page showing's Theory tab to the front (and any group of tabs it sits in); its browser."""
        page = self.pages.currentWidget()
        browser = None
        for tabs in page.findChildren(QtWidgets.QTabWidget):
            for i in range(tabs.count()):
                if tabs.tabText(i) == "Theory":
                    tabs.setCurrentIndex(i)
                    found = tabs.widget(i)
                    if not isinstance(found, QtWidgets.QTextBrowser):
                        found = found.findChild(QtWidgets.QTextBrowser)
                    browser = found or browser
        return browser

    def go_to(self, name: str, anchor: str = "") -> None:
        """Show page `name` (an attribute in PAGE_GROUPS) at its Theory, scrolled to `anchor`."""
        page = self.sim_page if name == "sim_page" else self._page(name)
        self.pages.setCurrentWidget(page)
        browser = self.show_theory()
        if browser is not None and anchor:
            QtCore.QTimer.singleShot(0, lambda: browser.scrollToAnchor(anchor))  # once it's laid out

    def _wire_links(self, page: QtWidgets.QWidget) -> None:
        for browser in page.findChildren(QtWidgets.QTextBrowser):
            links.wire(browser, self.go_to)

    def _about(self) -> None:
        from importlib.metadata import PackageNotFoundError, version

        try:
            v = version("vib-tutorial")
        except PackageNotFoundError:
            v = ""
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("About Vibration Tutorial")
        box.setTextFormat(QtCore.Qt.TextFormat.RichText)
        box.setText(ABOUT_HTML.format(version=v))
        box.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextBrowserInteraction)
        box.exec()

    def _on_page_changed(self, index: int) -> None:
        widget = self.pages.widget(index)
        for name, holder in list(self._placeholders.items()):
            if holder is widget:
                # Building a page and working out what it first shows takes up to a second (the rotor's
                # analysis, a first record): say so with the cursor until the event loop is back.
                QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
                QtCore.QTimer.singleShot(0, QtWidgets.QApplication.restoreOverrideCursor)
                self._page(name)

    # --------------------------------------------------------------- theme
    def set_theme(self, name: str) -> None:
        """Switch every page to a theme (a name from theming.names()) and redraw."""
        theming.apply(name)
        self.theme_combo.setCurrentText(name)
        self._restyle()

    def _on_theme_picked(self, name: str) -> None:
        theming.save_name(name)
        self.set_theme(name)

    def _on_desktop_scheme(self) -> None:
        # Checked once the change (which theming.apply also causes) has settled.
        QtCore.QTimer.singleShot(0, self._follow_desktop)

    def _follow_desktop(self) -> None:
        # Only while following the desktop: an explicit theme sets the scheme itself.
        if theming.current_name() == SYSTEM and colors.name != theming.system_theme().name:
            self.set_theme(SYSTEM)

    def _restyle(self) -> None:
        theming.restyle(self)
        for w in (self.chain, self.time_plot, self.mode_plot, self.phasor_plot.plot, self.frf_plot,
                  self.params, self.energy, *self._built.values()):
            w.apply_theme()
        self.force_panel.refresh()
        self._refresh_modal_views()  # mode colours: table, shapes, time and FRF curves

    # --------------------------------------------------------------- events
    def _on_system_changed(self, system: ChainSystem) -> None:
        self._collect()  # the steps in flight belong to the old system
        n_changed = system.n != self.sim.system.n
        self.sim.set_system(system)
        self._step_cost = 0.0  # friction, or a new size, changes it: measure it afresh
        # The modes first: a new size redraws the FRF (through the force panel), which needs this system's.
        modal = modal_analysis(system)
        self.modal = modal
        if n_changed:
            self._apply_dof(system.n)
        else:
            self.chain.set_masses(system.masses)
        self._refresh_modal(modal)

    def _on_preset(self, preset: Preset) -> None:
        """The preset's system is loaded; set its excitation and start from rest."""
        self.force_panel.load(preset.force)
        self._reset()
        self.frf_compare.setChecked(preset.absorber)
        if preset.absorber:
            self.tabs.setCurrentWidget(self.frf_tab)

    def _on_trial(self, index: int) -> None:
        """Set up one of the Things to try (index 0 is the prompt)."""
        if index <= 0:
            return
        t = SIM_TRIALS[index - 1]
        self.trial_note.setText(t.note)
        self.force.switch_off()
        if t.preset:
            self.params.load_preset(next(p for p in self.params.presets if p.name.startswith(t.preset)))
        else:
            system = ChainSystem.uniform(t.n)
            for i, c in t.damping.items():
                system.damping[i] = c
            system.friction[:] = t.friction
            self.params.set_system(system)
            settings = ForceSettings(target=t.n - 1)
            if t.kind is not None:
                settings = replace(settings, kind=t.kind)
            if t.sweep is not None:
                settings = replace(settings, sweep_start_hz=t.sweep[0], sweep_end_hz=t.sweep[1],
                                   sweep_time=t.sweep[2])
            if t.base is not None:
                settings = replace(settings, base=True, base_amplitude=t.base[0] * 1e-3, freq_hz=t.base[1])
            self.force_panel.load(settings)
            self.frf_compare.setChecked(False)
            self.tabs.setCurrentWidget(self.frf_tab if t.frf_tab else self.tabs.widget(0))
        if t.tune:
            self.force_panel.tune.activated.emit(t.tune)
        self.method_combo.setCurrentIndex(1 if t.state_space else 0)
        self.coords_combo.setCurrentIndex(self.coords_combo.findText(t.coords, QtCore.Qt.MatchFlag.MatchStartsWith))
        self.energy.view.setCurrentIndex(1 if t.by_mode else 0)
        self.controls.speed.setCurrentIndex(self.controls.SPEEDS.index(t.speed))
        self.controls.window.setValue(t.window)
        self.animate_modes.setChecked(t.animate)
        self.release_amp.setValue(t.amplitude)
        self._reset()
        if t.release is not None:
            self.table.selectRow(t.release)
            self._release_mode()
        elif t.apply:
            self.force_panel.button.click()
        if not self.running:
            self.controls.run.click()

    def _apply_dof(self, n: int) -> None:
        self.history.reset(n)
        self._carry = 0
        self._sim_target = self.sim.t
        self.chain.set_masses(self.sim.system.masses)
        self.force_panel.set_dof(n)

    def _refresh_modal(self, modal=None) -> None:
        self.modal = modal if modal is not None else modal_analysis(self.sim.system)
        for page in self._built.values():  # the rest are given the system when built
            if _follows_chain(page):
                page.set_system(self.sim.system, self.modal)
        self._refresh_modal_views()

    def _on_method_changed(self) -> None:
        self.method = self.method_combo.currentData()
        self._refresh_modal_views()

    def _refresh_modal_views(self) -> None:
        """Show the current modal result using the selected method."""
        self.entries = mode_entries(self.modal, self.method)
        state_space = self.method is Method.STATE_SPACE
        self._refresh_table()
        self.mode_plot.set_entries(self.entries, show_envelope=state_space)
        self.phasor_plot.setVisible(state_space)
        self.phasor_plot.set_entries(self.entries)
        self._on_mode_selected()
        self.force_panel.set_modes(self.entries)
        self.controls.set_modes(self.entries)
        self.release_button.setToolTip(RELEASE_TIP[self.method] + RELEASE_FIT_TIP)
        for w in self._coords_widgets:
            w.setToolTip(COORDS_TIP[self.method] + ENERGY_COORDS_TIP + FORCES_COORDS_TIP)
        self._modal_map = modal_coordinate_map(self.sim.system, self.modal, complex_modes=state_space)
        self._set_plot_curves()
        self.modal_note.setText(self._modal_note())
        self._on_force_changed()

    def _refresh_table(self) -> None:
        system = self.sim.system
        zf = friction_zeta(system, self.modal, self.release_amp.value() * 1e-3) if system.friction.any() else None
        self.table.set_result(self.modal, self.method, self.entries, zf)

    def _modal_note(self) -> str:
        notes = []
        n = self.sim.system.n
        if self.method is Method.STATE_SPACE:
            notes.append(
                f"State-space form: N = {n} second-order equations → 2N = {2 * n} first-order "
                "ones, so 2N eigenvalues. Oscillatory ones come in conjugate pairs λ, λ* that "
                "together make one real motion x(t) = 2 Re(ψ e<sup>λt</sup>)."
            )
            if self.modal.is_proportional:
                notes.append("Damping is <b>proportional</b>, so every ψ is real (phases 0° or 180°).")
            else:
                notes.append(
                    "Damping is <b>non-proportional</b>, so the ψ are complex: the masses peak "
                    "at different times (select a row to see the phases)."
                )
        elif self.modal.is_proportional:
            notes.append(
                "Damping is <b>proportional</b>: it does not couple the undamped modes, so the "
                "modes are real and ζ modal = ζ exact."
            )
        else:
            notes.append(
                "Damping is <b>non-proportional</b>: it couples the undamped modes "
                f"(coupling index {self.modal.coupling:.2f}; 0 = none, 1 = strong). The exact damped "
                "modes are complex (masses peak at different times) and ζ modal is an approximation."
            )
        if self.method is Method.CLASSICAL and self.sim.system.friction.any():
            notes.append(
                "Friction is nonlinear, so the modes leave it out; <i>ζ friction</i> is its "
                "equivalent damping at the initial displacement below, larger for smaller motions."
            )
        notes.append("<a href='#background'>Why?</a>")
        if self.method is Method.CLASSICAL and self.modal.overdamped_roots:
            roots = ", ".join(f"{r:.3g}" for r in self.modal.overdamped_roots)
            notes.append(f"Non-oscillatory (overdamped) roots λ = {roots} 1/s.")
        return " ".join(notes)

    def _on_mode_selected(self) -> None:
        r = self.table.currentRow() if self.table.selectedItems() else None
        self.mode_plot.set_highlight(r)
        self.phasor_plot.set_highlight(r)

    def _animate_modes(self, theta: float) -> None:
        self.mode_plot.animate(theta)
        if self.phasor_plot.isVisible():
            self.phasor_plot.animate(theta)

    @property
    def modal_view(self) -> bool:
        return self.coords_combo.currentData() == "modal"

    @property
    def energy_view(self) -> bool:
        return self.coords_combo.currentData() == "energy"

    @property
    def forces_view(self) -> bool:
        return self.coords_combo.currentData() == "forces"

    def _set_plot_curves(self) -> None:
        """Mass displacements, modal coordinates of the selected method, energies, or element forces."""
        self.element_combo.setVisible(self.forces_view)
        if self.forces_view:
            label, _, fmt = ELEMENT_FORCES[self.element_combo.currentIndex()]
            curves = [(fmt.format(i + 1), colors.mass[i]) for i in range(self.sim.system.n)]
            self.time_plot.set_curves(f"Tension: {label.lower()}", curves, units="N", min_span=MIN_FORCE_SPAN)
            return
        if self.energy_view:
            curves = [(name, colors.energy_colors[key]) for name, key in ENERGY_CURVES]
            self.time_plot.set_curves("Energy", curves, units="J", min_span=MIN_ENERGY_SPAN)
            return
        if not self.modal_view:
            self.time_plot.set_dof(self.sim.system.n)
            return
        # One coordinate per classical mode, conjugate pair or real root: the
        # entries with a legend, in the column order of modal_coordinate_map.
        # Legend text is e.g. "Mode 2: 3.1 Hz" or "λ3,4: 3.0 Hz"; keep the name.
        curves = [(e.legend.split(":")[0], e.color) for e in self.entries if e.legend is not None]
        self.time_plot.set_curves("Modal coordinate", curves)

    def _element_curves(self, s_el: np.ndarray) -> np.ndarray:
        """History element forces (n springs, n dampers) -> the selected element forces."""
        n = s_el.shape[1] // 2
        key = self.element_combo.currentData()
        if key == "spring":
            return s_el[:, :n]
        if key == "damper":
            return s_el[:, n:]
        return s_el[:, :n] + s_el[:, n:]

    def _plot_values(
        self, x: np.ndarray, v: np.ndarray, f: np.ndarray, e: np.ndarray, s_el: np.ndarray, g: np.ndarray
    ) -> Callable[[slice | np.ndarray], np.ndarray]:
        """The strip chart's rows at an index into the window: the selected curves, then f.

        The plot derives only the rows it needs (new samples and the decimated ones),
        so a long window isn't projected onto the modes, or summed, every frame.
        """
        view = self.coords_combo.currentData()
        n, m = self.sim.system.n, self._modal_map

        def rows(i: slice | np.ndarray) -> np.ndarray:
            if view == "forces":
                y = self._element_curves(s_el[i])
            elif view == "energy":
                y = energy_curves(e[i])
            elif view == "modal":
                y = (x[i] - g[i, :1]) @ m[:n] + (v[i] - g[i, 1:]) @ m[n:]
            else:
                y = x[i]
            return np.column_stack([y, f[i]])

        return rows

    def _on_force_changed(self) -> None:
        s = self.force.settings
        system = self.sim.system
        self.frf_compare.setEnabled(system.n > 1)
        reference = without_absorber(system) if self.frf_compare.isChecked() else None
        amplitude = s.base_amplitude if s.base else s.amplitude
        self.frf_plot.set_system(
            system, self.modal, s.target, base=s.base, reference=reference, friction_amplitude=amplitude
        )
        self.frf_friction_note.setVisible(bool(system.friction.any()))
        self._set_drive()
        self.time_plot.set_input(s.base)

    def _set_drive(self) -> None:
        """Mark the drive frequency: the harmonic's, or where a running chirp has swept to."""
        s = self.force.settings
        chirping = s.kind is ForceKind.CHIRP and self.force.on
        self.frf_plot.set_drive(self.force.frequency() if s.kind is ForceKind.HARMONIC or chirping else None)

    def _on_auto_scale(self, on: bool) -> None:
        self.chain.auto_scale = on
        self.time_plot.set_auto_range(on)

    def _on_run_toggled(self, running: bool) -> None:
        self.running = running
        self._sim_target = self.sim.t

    def _reset(self) -> None:
        self.sim.reset()  # drops the steps in flight
        self._started = None
        self.history.reset(self.sim.system.n)
        self._carry = 0
        self._sim_target = 0.0
        self._ground_velocity = 0.0
        self.force_panel.refresh()

    def _release_mode(self) -> None:
        r = self.table.currentRow()
        if not (0 <= r < len(self.entries)) or not self.table.selectedItems():
            r = 0
            self.table.selectRow(0)
        self.force.switch_off()
        self.force_panel.refresh()
        entry = self.entries[r]
        n = self.sim.system.n
        z0 = entry.state0 * self.release_amp.value() * 1e-3
        self.sim.set_state(z0[:n], z0[n:])
        if self.controls.auto_scale.isChecked() and entry.freq_hz > 0:
            self.controls.fit_to_mode(entry.key)

    def _on_mass_grabbed(self, _: int) -> None:
        self.force.switch_off()
        self.force_panel.refresh()

    def _on_mass_dragged(self, i: int, x: float) -> None:
        """Hold mass i at x (m) with the rest of the chain in static balance and at rest.

        With friction the other masses stay where friction holds them, so the shape
        depends on the path the mass was dragged along.
        """
        xg = self.sim.ground  # a ground motion stops, but only once time runs again
        now = self.sim.displacement - xg
        self.sim.set_displacement(xg + drag_shape(self.sim.system, now, i, x - xg))

    def _on_mass_released(self, _: int) -> None:
        self._sim_target = self.sim.t  # time stood still while the mass was held

    def _update_energy(self, es: np.ndarray) -> None:
        """Show the energy now; the stored energy in recent samples es sets the bar scale."""
        system, x, v = self.sim.system, self.sim.displacement, self.sim.velocity
        xr, vr = x - self.sim.ground, v - self._ground_velocity  # relative to the ground
        if self.controls.auto_scale.isChecked() and es.size:
            self._energy_scale = float((es[:, 0] + es[:, 1]).max())
        self.energy.bars.set_state(
            EnergyState(
                kinetic=float(kinetic_energy(system, v).sum()),
                potential=float(potential_energy(system, xr).sum()),
                modal=modal_energies(system, self.modal, xr, vr),
                added=self.sim.energy_added,
                work=self.sim.work,
                dissipated=self.sim.dissipated,
                friction=self.sim.friction_loss,
                relative=bool(self.sim.ground or self._ground_velocity),
            ),
            self._energy_scale,
        )

    # ------------------------------------------------------------ main loop
    def event(self, event: QtCore.QEvent) -> bool:
        # With frame stats on, time each repaint of the window (every widget in it).
        if frame_stats.STATS is None or event.type() != QtCore.QEvent.Type.UpdateRequest:
            return super().event(event)
        start = time.perf_counter()
        try:
            return super().event(event)
        finally:
            frame_stats.STATS.paint(time.perf_counter() - start)

    def _frame(self) -> None:
        start = self._clock.elapsed()
        try:
            if self.pages.currentWidget() is self.sim_page:
                with frame_stats.frame("Simulation", lambda: self.sim.t, FRAME_MS):
                    self._tick()
                    settle_plots(self.sim_page)
            else:
                self._tick()
        finally:
            spent = self._clock.elapsed() - start
            self._timer.start(max(1, FRAME_MS - spent))

    def _tick(self) -> None:
        now = self._clock.elapsed() / 1000.0
        dt = min(now - self._last_wall, 0.1)  # don't jump after a stall
        self._last_wall = now
        # Like the Substructuring tabs, the simulation runs only while its page shows.
        if self.pages.currentWidget() is not self.sim_page:
            return

        # Steps that run in Rust (friction) start at the end of a frame, beside its drawing, and are
        # recorded at the next; if they aren't done by then, it doesn't wait. The others run here.
        if self._started is not None and self.sim.ready:
            self._collect()
        held = self.chain.held
        steps = self.running and held is None
        if steps:
            self._sim_target += dt * self.controls.speed_factor
            if self._started is None and not self.sim.steps_aside:
                self._start()
                self._collect()
        if self._started is not None:
            return  # nothing new to show (and the state isn't there yet)
        if self.force.active != self._force_active:  # e.g. a chirp reached its end
            self._force_active = self.force.active
            self.force_panel.refresh()

        window = self.controls.window.value()
        t, x, v, f, e, s_el, g = self.history.window(window)
        peak = float(np.abs(x[-min(len(x), 20_000) :]).max()) if x.size else 0.0
        if held is None:
            s = self.force.settings
            force = 0.0 if s.base else self.force.value()  # a moving ground shows as the wall
            self.chain.update_state(
                self.sim.displacement, peak, force, s.target, abs(s.amplitude), ground=self.sim.ground
            )
        else:  # the arrow shows the force holding the mass: the springs' pull on it, reversed
            hold = float(self.sim.system.matrices()[2][held] @ (self.sim.displacement - self.sim.ground))
            self.chain.update_state(self.sim.displacement, peak, hold, held, abs(hold), ground=self.sim.ground)
        start = self.history.offset + self.history.size - t.size
        self.time_plot.update_data(t, self._plot_values(x, v, f, e, s_el, g), window, start)
        if self.energy.isVisible():
            self._update_energy(e[-min(len(e), 20_000) :])
        if self.force.settings.kind is ForceKind.CHIRP:
            self._set_drive()
        if self.animate_modes.isChecked():
            self._animate_modes(2 * math.pi * MODE_ANIMATION_HZ * now)
        self.controls.time_label.setText(f"{self.sim.t:8.3f} s   (step {self.sim.step_size() * 1e3:.3g} ms)")
        if steps and self._started is None:
            self._start()  # last, so that its steps run while this frame paints

    def _start(self) -> None:
        """Start the simulator's steps towards `_sim_target`; the next frame's `_collect` records them."""
        speed = self.controls.speed_factor
        # If the simulator can't keep up (very stiff system), drop the backlog.
        self._sim_target = min(self._sim_target, self.sim.t + 0.25 * speed + 0.05)
        # And take no more steps than FRAME_BUDGET allows at the cost seen so far: a slow
        # simulation then falls behind real time (the overlay's sim speed) instead of stalling the frames.
        steps = FRAME_BUDGET / self._step_cost if self._step_cost else PROBE_STEPS
        self._sim_target = min(self._sim_target, self.sim.t + max(1.0, steps) * self.sim.step_size())
        if self._sim_target - self.sim.t < 0.5 * self.sim.step_size():
            return  # not a step's worth yet
        began = time.perf_counter()
        base = self.force.settings.base  # what the samples' input is, whatever the panel does meanwhile
        self.sim.start(self._sim_target - self.sim.t)
        self._started = (time.perf_counter() - began, base)

    def _store_stride(self) -> int:
        """Steps between the samples the history keeps (STORE_DT, STORE_PER_PERIOD)."""
        spacing = STORE_DT
        f = self.sim.fastest_hz()
        if f > 0:
            spacing = min(spacing, 1.0 / (STORE_PER_PERIOD * f))
        s = self.force.settings
        if s.kind is ForceKind.PULSE:
            spacing = min(spacing, s.pulse_duration / 10.0)
        return max(1, int(spacing / self.sim.step_size()))

    def _collect(self) -> None:
        """Record the steps `_start` began in the history (waiting for them if they aren't done)."""
        if self._started is None:
            return
        spent, base = self._started
        self._started = None
        waits = not self.sim.ready
        began = time.perf_counter()
        ts, xs, vs, fs = self.sim.collect()
        steps = ts.size
        vg = self.sim.ground_velocity
        if vg.size:
            self._ground_velocity = float(vg[-1])
        # Keep every stride-th sample, continuing the stride across advances.
        stride = self._store_stride()
        pick = np.arange((stride - self._carry - 1) % stride, steps, stride)
        self._carry = (steps - 1 - pick[-1]) if pick.size else self._carry + steps
        ts, xs, vs, fs, vg, ledger = ts[pick], xs[pick], vs[pick], fs[pick], vg[pick], self.sim.ledger[pick]
        system = self.sim.system
        # With ground motion fs is x_g; the springs and dampers see the motion relative to it.
        xg = fs if base else np.zeros_like(fs)
        xr, vr = xs - xg[:, None], vs - vg[:, None]
        kinetic = kinetic_energy(system, vs).sum(axis=1)
        potential = potential_energy(system, xr).sum(axis=1)
        energy = np.column_stack([kinetic, potential, ledger])
        forces = np.hstack(element_forces(system, xr, vr))
        self.history.extend(ts, xs, vs, fs, energy, forces, np.column_stack([xg, vg]))
        if steps and not waits:  # a dearer step counts at once, a cheaper one gradually
            # This thread's work and the steps' (where they run beside it): the slower sets the pace.
            cost = max(spent + time.perf_counter() - began, self.sim.steps_time) / steps
            self._step_cost = cost if cost > self._step_cost else 0.8 * self._step_cost + 0.2 * cost


def energy_curves(e: np.ndarray) -> np.ndarray:
    """History energy columns (T, V, E0, W, D, D_f) -> the ENERGY_CURVES (T, V, T + V, E0, W, D, D_f)."""
    return np.column_stack([e[:, :2], e[:, 0] + e[:, 1], e[:, 2:]])
