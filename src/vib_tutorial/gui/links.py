"""Links in the Theory notes, within a page and between pages.

A note links to another page's Theory with `page:<attribute>#<anchor>`, the attribute as in
`main_window.PAGE_GROUPS` (`page:frf_page#three-forms`); `#anchor` alone stays on its own page,
and http(s) links open in the browser. `before(...)` writes a page's "Before this page" line.
The main window calls `wire` on every Theory browser and handles the `page:` links.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6 import QtCore, QtGui, QtWidgets


def page_link(page: str, anchor: str, text: str) -> str:
    return f'<a href="page:{page}#{anchor}">{text}</a>'


def before(*refs: tuple[str, str, str]) -> str:
    """The "Before this page" line: (page attribute, anchor, section title) for each section to read first."""
    items = [page_link(page, anchor, title) + f" ({PAGE_TITLES[page]})" for page, anchor, title in refs]
    text = items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]
    return f"<p><i>Before this page: {text}.</i></p>"


def takeaways(*points: str) -> str:
    """A page's closing list of what to remember."""
    return "<h3><a name=\"takeaways\"></a>What to take away</h3><ul>" + "".join(
        f"<li>{p}</li>" for p in points) + "</ul>"


# The pages in the order they're meant to be read, in groups: (group, ((attribute, tab title, what it's
# about), ...)). Each page uses only ideas from the pages before it. The main window's tabs and the
# Simulation page's course map are both made from this.
COURSE = (
    ("One mass", (
        ("sdof_page", "Single oscillator",
         "one mass on a spring and a damper: free decay, damping, resonance, half-power, beats, poles"),
        ("pendulum_page", "Pendulums",
         "gravity as the spring, large swings that soften, and the rotational form of every one-mass formula"),
        ("transient_page", "Transient response",
         "an impulse, a step or a pulse: the impulse response, convolution, and how long a pulse has to be"),
        ("periodic_page", "Periodic forcing",
         "a repeating force as a Fourier series: harmonics, Gibbs, resonance through a harmonic"),
        ("damping_page", "Damping models",
         "viscous, hysteretic, friction and viscoelastic damping: their loops, losses and decays"),
        ("isolation_page", "Vibration isolation",
         "a machine on mounts: transmissibility, static deflection, damping, two stages, start-up"),
    )),
    ("Foundations", (
        ("sim_page", "Simulation && modal analysis",
         "one mass, then a chain of them: modes, damping, energy, forcing, absorbers, friction"),
        ("energy_page", "Energy methods",
         "Rayleigh, Dunkerley, Rayleigh–Ritz and Holzer: frequencies from the energies, and how good they are"),
        ("frf_page", "FRF matrix", "every receptance H<sub>jk</sub>, and how the modes add up to it"),
        ("complex_page", "Complex modes", "when damping makes the modes complex, and what the usual approximation misses"),
        ("continuum_page", "Continuous systems",
         "a string, a bar or a beam cut into a chain: which of its modes to trust"),
        ("beams_page", "Beams && plates",
         "Timoshenko's thick beam, a plate's modes and nodal lines, and modal density"),
        ("integration_page", "Numerical integration",
         "central difference, Newmark, HHT-α and Runge–Kutta against the exact update: stability, period error, numerical damping"),
    )),
    ("Responses", (
        ("random_page", "Random vibration", "a random force or base: the response's spectrum, RMS and peaks"),
        ("fatigue_page", "Fatigue",
         "a random stress counted by rainflow: S–N curves, Miner's rule, narrow-band and Dirlik estimates"),
        ("quarter_car_page", "Quarter-car",
         "a suspension on an ISO 8608 road: comfort, grip and travel from the road's PSD"),
        ("shock_page", "Shock spectrum", "a short pulse on a bank of oscillators, and on the chain"),
        ("seismic_page", "Seismic response",
         "the chain as a building: a design spectrum, effective modal mass, and ABS, SRSS and CQC against records"),
    )),
    ("Building models", (
        ("coupling_page", "Modal coupling", "two parts joined, one mode each: when their modes split"),
        ("cms_page", "Substructuring (CMS)", "a chain reduced to a few modes per part, and back"),
    )),
    ("Testing", (
        ("signals_page", "Signals && spectra",
         "sampling, aliasing, leakage, windows and averaging, and the transducers that make the signals"),
        ("test_page", "Virtual modal test", "measuring the FRF from force and response signals, and fitting modes"),
        ("oma_page", "Operational modal analysis", "the modes from the responses alone, the forces unknown"),
        ("updating_page", "Model updating", "comparing a model with a test, and correcting it"),
        ("variability_page", "Variability && localization",
         "scatter in the chain's properties by Monte Carlo, and modes localized in a mistuned ring of blades"),
    )),
    ("Beyond the chain", (
        ("torsional_page", "Torsional vibration", "a drive train of discs, shafts and gears, and its start-up"),
        ("duffing_page", "Duffing && Mathieu", "a nonlinear spring: jumps, chaos, and parametric resonance"),
        ("self_excited_page", "Self-excited vibration",
         "negative damping and limit cycles: van der Pol, galloping, vortex-induced vibration and flutter"),
        ("active_control_page", "Active control",
         "velocity feedback, collocation, spillover and actuator lag, against a tuned mass damper"),
        ("rotor_page", "Jeffcott rotor", "a spinning shaft: critical speeds, balancing, bearings, stability, rub"),
        ("diagnostics_page", "Machinery diagnostics",
         "order tracking under a run-up, gear-mesh sidebands, bearing defects by the envelope, and the cepstrum"),
    )),
)


def course_map() -> str:
    """The pages in order, as linked lists by group."""
    html, start = [], 1
    for group, pages in COURSE:
        items = "".join(f'<li><a href="page:{name}">{title.replace("&&", "&amp;")}</a>: {about}</li>'
                        for name, title, about in pages)
        html.append(f'<p><b>{group}</b></p><ol start="{start}">{items}</ol>')
        start += len(pages)
    return "".join(html)


# Titles as the links name them (the tab titles without Qt's && escape).
PAGE_TITLES = {
    "sdof_page": "Single oscillator page",
    "pendulum_page": "Pendulums page",
    "transient_page": "Transient response page",
    "periodic_page": "Periodic forcing page",
    "damping_page": "Damping models page",
    "isolation_page": "Vibration isolation page",
    "sim_page": "Simulation page",
    "energy_page": "Energy methods page",
    "frf_page": "FRF matrix page",
    "complex_page": "Complex modes page",
    "continuum_page": "Continuous systems page",
    "beams_page": "Beams & plates page",
    "integration_page": "Numerical integration page",
    "random_page": "Random vibration page",
    "fatigue_page": "Fatigue page",
    "quarter_car_page": "Quarter-car page",
    "shock_page": "Shock spectrum page",
    "seismic_page": "Seismic response page",
    "coupling_page": "Modal coupling page",
    "cms_page": "Substructuring page",
    "signals_page": "Signals & spectra page",
    "test_page": "Virtual modal test page",
    "oma_page": "Operational modal analysis page",
    "updating_page": "Model updating page",
    "variability_page": "Variability & localization page",
    "torsional_page": "Torsional vibration page",
    "duffing_page": "Duffing & Mathieu page",
    "self_excited_page": "Self-excited vibration page",
    "active_control_page": "Active control page",
    "rotor_page": "Jeffcott rotor page",
    "diagnostics_page": "Machinery diagnostics page",
}


def wire(browser: QtWidgets.QTextBrowser, go: Callable[[str, str], None]) -> None:
    """Have `browser` follow its own #anchors, send page: links to `go(page, anchor)`, open the web outside."""
    if browser.property("links_wired"):
        return
    browser.setProperty("links_wired", True)
    browser.setOpenLinks(False)

    def follow(url: QtCore.QUrl) -> None:
        if url.scheme() == "page":
            go(url.path(), url.fragment())
        elif url.scheme() in ("http", "https"):
            QtGui.QDesktopServices.openUrl(url)
        elif url.fragment():
            browser.scrollToAnchor(url.fragment())

    browser.anchorClicked.connect(follow)
