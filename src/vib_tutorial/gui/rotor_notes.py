"""Text for the Jeffcott rotor page: the theory notes and the live equations and matrices."""

from __future__ import annotations

import math

import numpy as np

from ..core.rotor import DOF_NAMES, RPM, RotorSystem, whirl_modes
from .cms_notes import fmt
from .style import colors

LABELS = ["x<sub>A</sub>", "x<sub>D</sub>", "θ<sub>x</sub>", "x<sub>B</sub>",
          "y<sub>A</sub>", "y<sub>D</sub>", "θ<sub>y</sub>", "y<sub>B</sub>"]
assert len(LABELS) == len(DOF_NAMES)
GROUPS = ["i"] * 4 + ["b"] * 4  # x plane, y plane: tinted as the CMS page's interior and boundary blocks


def whirl_name(ratio: float) -> str:
    """Forward, backward or straight-line, from a whirl ratio (+1 circular forward, -1 backward)."""
    if ratio > 0.1:
        return "forward"
    if ratio < -0.1:
        return "backward"
    return "straight-line"


def _legend() -> str:
    tints = colors.block_tints
    return (
        "<p><small>Cell colours: "
        f"<span style='background:{tints['i', 'i']}'>&nbsp;x plane (horizontal)&nbsp;</span> "
        f"<span style='background:{tints['b', 'b']}'>&nbsp;y plane (vertical)&nbsp;</span> "
        f"<span style='background:{tints['b', 'i']}'>&nbsp;coupling between the planes&nbsp;</span>"
        "</small></p>"
    )


def _num(v: float, scale: float) -> str:
    """Four significant figures without exponents where they fit: 0.000625, 7363, 56,820."""
    if abs(v) <= 1e-10 * scale:
        return "0"
    if abs(v) >= 1e7 or abs(v) < 1e-4:
        return f"{v:.3e}"
    if abs(v) >= 1e4:
        return f"{float(f'{v:.4g}'):,.0f}"
    return f"{v:.4g}"


def _matrix(A: np.ndarray, labels: list[str], groups: list[str], title: str) -> str:
    """An HTML table of A with labelled rows and columns, its cells tinted by block (see _legend)."""
    scale = float(np.abs(A).max()) if A.size else 1.0
    head = "".join(f"<th>{c}</th>" for c in labels)
    out = [f"<table border='1' cellspacing='0' cellpadding='3'><tr><th>{title}</th>{head}</tr>"]
    for r, (label, rg) in enumerate(zip(labels, groups)):
        cells = []
        for c, cg in enumerate(groups):
            text = _num(A[r, c], scale)
            color = f" style='color:{colors.grey}'" if text == "0" else ""
            tint = colors.block_tints[tuple(sorted((rg, cg)))]
            cells.append(f"<td align='right' bgcolor='{tint}'{color}>{text}</td>")
        out.append(f"<tr><th>{label}</th>{''.join(cells)}</tr>")
    out.append("</table>")
    return "".join(out)


def _mat(A: np.ndarray, title: str) -> str:
    return _matrix(A, LABELS, GROUPS, title)


def matrices_html(system: RotorSystem, omega: float) -> str:
    """The equations of motion, every matrix with the current numbers, and the modes at speed Ω."""
    M, C, G, K = system.matrices()
    s = system
    ks = s.shaft_stiffness()
    one = ["w<sub>A</sub>", "w<sub>D</sub>", "θ<sub>D</sub>", "w<sub>B</sub>"]
    shaft = _matrix(ks, one, ["i"] * 4, "K<sub>shaft</sub>")
    modes = whirl_modes(system, omega)
    rows = []
    for r, (f, z, w) in enumerate(zip(modes.freq_hz, modes.zeta, modes.whirl)):
        lam = complex(-z * 2 * math.pi * f / math.sqrt(max(1 - z * z, 1e-12)), 2 * math.pi * f)
        rows.append(
            f"<tr><td align='center'>{r + 1}</td><td align='right'>{lam.real:.4g} + {lam.imag:.5g}i</td>"
            f"<td align='right'>{f:.4g}</td><td align='right'>{f * 60:,.0f}</td><td align='right'>{z:.3g}</td>"
            f"<td align='right'>{w:+.2f}</td><td>{whirl_name(w)}</td></tr>"
        )
    rpm = omega * RPM
    return f"""
<h3>Equations of motion at {rpm:,.0f} rpm (Ω = {omega:.4g} rad/s)</h3>
<p style='font-size:large' align='center'>M q̈ + (C + Ω G) q̇ + K q = f(t)</p>
<p>q = [x<sub>A</sub>, x<sub>D</sub>, θ<sub>x</sub>, x<sub>B</sub> | y<sub>A</sub>, y<sub>D</sub>,
θ<sub>y</sub>, y<sub>B</sub>]: the journals' displacements at A and B, the disc's displacement, and the
shaft's slope at the disc (θ<sub>x</sub> = dx/dz, θ<sub>y</sub> = dy/dz), first in the horizontal x–z
plane, then in the vertical y–z plane. Units: m and rad; forces in N, moments in N·m.</p>
{_legend()}

<h4>Mass M and stiffness K</h4>
<p>M is diagonal: the journal masses, the disc's mass m on its displacements and its diametral
inertia I<sub>d</sub> on its slopes. K is the shaft's bending stiffness plus each support spring on its
journal. Nothing joins the two planes: with isotropic supports K has two identical blocks.</p>
{_mat(M, "M")}<br>{_mat(K, "K")}

<h4>The shaft (one plane)</h4>
<p>Two Euler–Bernoulli beam elements, A to the disc (a = {s.a:.4g} m) and the disc to B
(b = {s.b:.4g} m), with EI = {s.ei:.4g} N·m². The bearings carry no moment, so the slopes at A and B are
condensed out (they follow the four DOFs kept). Moving the whole shaft sideways, or tilting it rigidly,
takes no force: K<sub>shaft</sub>[1, 1, 0, 1]<sup>T</sup> = 0 and K<sub>shaft</sub>[0, a, 1, L]<sup>T</sup> = 0.
The supports hold it.</p>
{shaft}

<h4>Damping C and the gyroscopic matrix Ω G</h4>
<p>C has only the support dampers. G has just two entries, ±I<sub>p</sub> = ±{s.ip:.4g} kg·m², joining
the disc's two slopes, one plane to the other. It is <b>skew-symmetric</b> (G<sup>T</sup> = −G), so
q̇<sup>T</sup>(ΩG)q̇ = 0: unlike C, it takes no power out. It only turns the motion from one plane into
the other, and it grows with the speed.</p>
{_mat(C, "C")}<br>{_mat(omega * G, "Ω G")}

<h4>State space</h4>
<p>With z = [q, q̇] the equations become z' = A(Ω) z + B u, u = [f<sub>x</sub>, f<sub>y</sub>] on the
disc:</p>
<p align='center'>A(Ω) = [ 0 &nbsp; I ; −M<sup>−1</sup>K &nbsp; −M<sup>−1</sup>(C + ΩG) ]</p>
<p>A is 16 × 16 and not symmetric, and its eigenvalues λ = −ζω<sub>n</sub> ± iω<sub>d</sub> come in
conjugate pairs. The modes at this speed, lowest first:</p>
<table border='1' cellspacing='0' cellpadding='3'>
<tr><th>Mode</th><th>λ [1/s]</th><th>f<sub>d</sub> [Hz]</th><th>f<sub>d</sub> [rpm]</th><th>ζ</th>
<th>Whirl ratio</th><th>Whirl</th></tr>
{''.join(rows)}
</table>
<p><small>Whirl ratio: +1 for a circular forward whirl (the same way as the spin, x toward y), −1 for
circular backward whirl, 0 for a straight line; in between, an ellipse. The stations are weighted by
their masses (and the slopes by I<sub>d</sub>).</small></p>

<h4>The unbalance force</h4>
<p>The unbalance U = m e = {s.unbalance * 1e6:.4g} g·mm turns with the shaft at angle φ (φ' = Ω).
Keeping the disc's centre of mass on its path takes</p>
<p align='center'>f<sub>x</sub> = U(Ω² cos φ + Ω' sin φ), &nbsp; f<sub>y</sub> = U(Ω² sin φ − Ω' cos φ)</p>
<p>on the disc's x<sub>D</sub> and y<sub>D</sub>: here UΩ² = {fmt(s.unbalance * omega**2, 1.0)} N
turning once per revolution. Ω' is the ramp's angular acceleration, while the speed changes.</p>
"""


THEORY_HTML = """
<h2>The Jeffcott rotor</h2>
<p>A disc on a flexible, massless shaft between two bearings is the simplest model of a rotating
machine: H. H. Jeffcott's 1919 paper (and A. Föppl's of 1895, hence also the <i>Laval</i> or
<i>Föppl–Jeffcott</i> rotor) explained why a turbine can run smoothly above the speed at which its shaft
whirls most violently. Here the bearings stand on flexible supports, springs and dampers in the
horizontal (x) and vertical (y) directions, and the disc can tilt as well as move sideways.</p>

<h3>Degrees of freedom</h3>
<p>The shaft spins about its axis z. In each plane, x–z and y–z, four numbers describe its bending: the
journal displacements at bearings A and B, the disc's displacement, and the shaft's slope at the disc.
That makes 8 DOFs and, in state space, 16 states. The midspan's motion (and the shaft's shape anywhere)
follows from these: between the stations a massless beam bends as a cubic.</p>

<h3>The classic case: disc at midspan</h3>
<p>With the disc at midspan on identical bearings the problem is symmetric about the disc: its
translation and its tilt are uncoupled. The translation sees the shaft's midspan stiffness
k<sub>s</sub> = 48EI/L³ in series with the two supports in parallel, 2k<sub>b</sub>, so (with light
journals)</p>
<p align='center'>ω<sub>c</sub> = √(k<sub>eq</sub>/m), &nbsp; 1/k<sub>eq</sub> = 1/k<sub>s</sub> + 1/(2k<sub>b</sub>)</p>
<p>With the defaults (d = 10 mm, L = 0.5 m, steel) k<sub>s</sub> = 37,700 N/m and
k<sub>b</sub> = 50,000 N/m, so ω<sub>c</sub> = 165.5 rad/s, <b>1580 rpm</b>. The full model, with its
0.1 kg journals, puts it at 1584 rpm. Flexible supports lower the critical speed: on rigid bearings it
would be 1854 rpm.</p>

<h3>Unbalance response: heavy spot and high spot</h3>
<p>The disc's centre of mass sits a distance e from the shaft's axis (U = m e). Spinning at Ω it
pulls the shaft round with a force meΩ² that turns with it, so the disc whirls once per revolution
(synchronous, 1X), forward, in a circle if the rotor is isotropic. For the classic case, with
r = Ω/ω<sub>c</sub>,</p>
<p align='center'>|x| = e r² / √((1 − r²)² + (2ζr)²)</p>
<p>and the <b>heavy spot</b> (where the unbalance is) leads the <b>high spot</b> (where the shaft is
displaced most) by a phase lag that goes from 0° well below the critical speed, through 90° at it, to
180° well above it. Above the critical speed the shaft deflects away from the heavy spot, and as
r → ∞ the disc spins about its own centre of mass: the shaft's centre whirls on a circle of radius e,
the <b>self-centring</b> that lets machines run supercritically. With the defaults (e = 20 µm):
2.2 µm at 500 rpm (lag 2°), 217 µm at the critical speed (lag 88°), 28 µm at 3000 rpm (lag 176°).</p>

<h3>The gyroscopic effect</h3>
<p>A spinning disc has angular momentum I<sub>p</sub>Ω along its axis. When the disc tilts as it
whirls, that axis turns, and turning it takes a moment: in the slopes' equations</p>
<p align='center'>I<sub>d</sub>θ̈<sub>x</sub> + ΩI<sub>p</sub>θ̇<sub>y</sub> = M<sub>x</sub>, &nbsp;
I<sub>d</sub>θ̈<sub>y</sub> − ΩI<sub>p</sub>θ̇<sub>x</sub> = M<sub>y</sub></p>
<p>So the gyroscopic term multiplies the velocities and sits beside the damping:
M q̈ + (C + ΩG) q̇ + K q = f. But G is <b>skew-symmetric</b>: the power it absorbs,
q̇<sup>T</sup>ΩGq̇, is zero. It does no work, so it cannot damp anything. What it does is couple the
two planes' tilts and make the natural frequencies depend on the speed.</p>
<p>Each whirl mode splits into a <b>forward</b> whirl (the same way as the spin) and a <b>backward</b>
whirl. In a forward whirl the gyroscopic moment resists the tilting, as if the shaft were stiffer, and
the frequency rises with speed. In a backward whirl it does the opposite. For the disc's tilt on its
own (a rigid support, stiffness k<sub>θ</sub>):</p>
<p align='center'>I<sub>d</sub>ω² ∓ I<sub>p</sub>Ωω − k<sub>θ</sub> = 0</p>
<p>For a thin disc I<sub>p</sub> = 2I<sub>d</sub>, and the forward tilting frequency tends to 2Ω: it
never meets the 1X line, so an unbalance can never make the disc's tilting mode resonate forward.</p>
<p><b>Why the classic case hides it:</b> at midspan, on identical bearings, the disc whirls without
tilting, so the gyroscopic moment never enters the first mode; only the disc's own tilting (conical)
mode, high above, splits. Move the disc off centre, or make the bearings different, and the
translation and tilt couple: the first mode splits too, and its critical speed moves.</p>

<h3>The Campbell diagram and critical speeds</h3>
<p>Plotting each damped natural frequency against the spin speed gives the Campbell diagram. The
unbalance turns once per revolution, so it drives at f = Ω/2π: the 1X line. Where a whirl frequency
crosses the 1X line, the unbalance drives that mode at its own natural frequency: a <b>critical
speed</b>. In an isotropic rotor the unbalance is a purely forward rotating force, so it drives only
the forward branches; the backward crossings are critical only for an anisotropic rotor (or for other
excitations).</p>

<h3>Anisotropic supports and backward whirl</h3>
<p>Supports stiffer vertically than horizontally (common: gravity, foundations, oil-film bearings) split
each mode into a horizontal one and a vertical one, with two critical speeds. The orbits become ellipses.
Between the two criticals the horizontal response already lags by more than 90° while the vertical
one does not yet, and the shaft whirls <b>backward</b>, against its spin, while the unbalance still
turns forward. A shaft whirling backward is bent back and forth once per revolution as it spins:
fatigue loading a forward synchronous whirl does not cause.</p>

<h3>Why state space</h3>
<p>Undamped modes and modal superposition need symmetric M, C and K, and real modes that do not change.
Here ΩG is skew-symmetric and changes with the speed, and support damping is not proportional, so the
modes are complex and speed-dependent. The first-order form z' = A(Ω)z + Bu handles all of it: the
eigenvalues of A give each mode's frequency, damping and whirl direction, and the time response is
stepped exactly. As on the other pages, each step is the exact first-order-hold matrix-exponential
update, z<sub>k+1</sub> = Φz<sub>k</sub> + Γ<sub>0</sub>u<sub>k</sub> + Γ<sub>1</sub>u<sub>k+1</sub>,
Φ = e<sup>Ah</sup>, stable however stiff the shaft. While the speed ramps, A(Ω) is updated every 2 ms of
simulated time and the unbalance force follows the exact angle φ, including the Ω' term.</p>

<h3>Things to try</h3>
<ol>
<li>Ramp from 1000 to 3000 rpm at 500 rpm/s and watch the disc's orbit grow through 1584 rpm and shrink
again to about the 20 µm eccentricity, while the heavy spot swings round to the opposite side of the
orbit. Then try 5000 rpm/s: the whirl has no time to build, and the peak is much lower.</li>
<li>Set the target to 1584 rpm and read the phase: the heavy spot leads the high spot by about 90°.</li>
<li>Untick <i>Isotropic</i> and set k<sub>x</sub> = 20,000 N/m and k<sub>y</sub> = 80,000 N/m on A
(B follows). The critical splits into 1365 rpm (horizontal) and 1668 rpm (vertical); run between them,
say at 1500 rpm, and the orbit is a backward ellipse (backward from about 1413 to 1632 rpm).</li>
<li>Bearing damping has an optimum. With the defaults the first mode has ζ = 0.046 at
c = 100 N·s/m and 0.089 at 400 N·s/m, but only 0.005 at 10,000 N·s/m: dampers that stiff lock the
journals, so the supports stop moving and nothing is left to damp. The critical speed then climbs
toward the rigid-bearing value.</li>
<li>Move the disc to a/L = 0.3: the forward and backward criticals separate (1737 and 1731 rpm) and the
disc's tilting mode splits strongly with speed (from 360 Hz at rest to 465 Hz forward and 289 Hz
backward at 6000 rpm). Tap the disc at speed and the orbit becomes a rosette.</li>
<li>Pause near the critical speed and turn the 3D view: the shaft bows in one plane, which turns with
the spin, rather than flapping back and forth.</li>
</ol>

<h3>Notation</h3>
<table border="1" cellspacing="0" cellpadding="4">
<tr><th>Symbol</th><th>Meaning</th><th>Units</th></tr>
<tr><td>L, a, b</td><td>bearing span; disc's distance from A and from B (a + b = L)</td><td>m</td></tr>
<tr><td>d, E, EI</td><td>shaft diameter, Young's modulus, bending stiffness E π d⁴/64</td><td>m, Pa, N·m²</td></tr>
<tr><td>m, I<sub>p</sub>, I<sub>d</sub></td><td>disc mass; its polar and diametral moments of inertia</td><td>kg, kg·m²</td></tr>
<tr><td>k<sub>x</sub>, k<sub>y</sub>, c<sub>x</sub>, c<sub>y</sub></td><td>a support's stiffness and damping, horizontal and vertical</td><td>N/m, N·s/m</td></tr>
<tr><td>x, y</td><td>horizontal and vertical displacement of the shaft's centre</td><td>m</td></tr>
<tr><td>θ<sub>x</sub>, θ<sub>y</sub></td><td>shaft's slope at the disc, dx/dz and dy/dz</td><td>rad</td></tr>
<tr><td>Ω, φ</td><td>spin speed; angle of the heavy spot from +x (φ' = Ω)</td><td>rad/s (rpm), rad</td></tr>
<tr><td>U = m e</td><td>unbalance: disc mass times its centre of mass's offset e</td><td>kg·m (g·mm)</td></tr>
<tr><td>M, C, G, K</td><td>mass, damping, gyroscopic and stiffness matrices (8 × 8)</td><td>—</td></tr>
<tr><td>A(Ω), B</td><td>state matrix (16 × 16) and input matrix</td><td>—</td></tr>
<tr><td>ω<sub>c</sub></td><td>critical speed</td><td>rad/s (rpm)</td></tr>
<tr><td>ζ</td><td>damping ratio of a mode, −Re λ / |λ|</td><td>—</td></tr>
</table>
"""
