"""Text for the Modal coupling page: the theory notes, the notation, and the live step-by-step matrices."""

from __future__ import annotations

import numpy as np

from ..core import ModalResult
from ..core.modal_coupling import CoupledModel
from .cms_notes import _side_by_side, fmt, matrix_html
from .style import colors

NOTATION_HTML = """
<table border="1" cellspacing="0" cellpadding="4">
<tr><th>Symbol</th><th>Meaning</th><th>Units</th></tr>
<tr><td>N, j</td><td>number of masses in the chain; the split is after mass j (A is m<sub>1</sub>…m<sub>j</sub>)</td><td>—</td></tr>
<tr><td>m<sub>i</sub>, k<sub>i</sub>, c<sub>i</sub></td><td>mass i; spring and damper i (joining mass i−1, or the ground, to mass i)</td><td>kg, N/m, N·s/m</td></tr>
<tr><td>M, C, K</td><td>mass, damping and stiffness matrices: of the chain, or of a subsystem, M<sub>A</sub>, K<sub>B</sub>, … (each subsystem's own, with only its masses)</td><td>kg, N·s/m, N/m</td></tr>
<tr><td>A, B</td><td>the subsystems: A grounded through k<sub>1</sub>, tip m<sub>j</sub> free; B with k<sub>j+1</sub>, c<sub>j+1</sub> tied to ground (its <i>base</i>)</td><td>—</td></tr>
<tr><td>m<sub>A</sub>, m<sub>B</sub></td><td>total mass of A, of B</td><td>kg</td></tr>
<tr><td>ω<sub>r</sub>, f<sub>r</sub></td><td>natural frequency of mode r of a subsystem on its own, f = ω/2π</td><td>rad/s, Hz</td></tr>
<tr><td>ψ</td><td>a mode shape at any scale; here scaled to 1 at a chosen DOF (A's tip, or B's largest entry), so dimensionless</td><td>m/m</td></tr>
<tr><td>ψ<sup>T</sup>Mψ</td><td>generalized (modal) mass of the shape ψ: Σ m<sub>i</sub>ψ<sub>i</sub>²</td><td>kg</td></tr>
<tr><td>φ</td><td>the mode shape mass-normalized, φ = ψ/√(ψ<sup>T</sup>Mψ), so φ<sup>T</sup>Mφ = 1</td><td>1/√kg</td></tr>
<tr><td>φ<sub>tip</sub></td><td>φ's entry at A's tip m<sub>j</sub>, the interface</td><td>1/√kg</td></tr>
<tr><td>m<sub>tip</sub> = 1/φ<sub>tip</sub>²</td><td>modal mass of a mode of A at the interface: its generalized mass when scaled to 1 at the tip</td><td>kg</td></tr>
<tr><td><b>r</b></td><td>influence (direction) vector: each DOF's displacement for a unit base motion in the excitation direction; <b>r</b> = <b>1</b> for the chain</td><td>m/m</td></tr>
<tr><td><b>e</b><sub>1</sub></td><td>unit vector picking B's first DOF</td><td>—</td></tr>
<tr><td>Γ</td><td>participation factor φ<sup>T</sup>M<b>r</b>: how strongly a base motion drives the mode</td><td>√kg</td></tr>
<tr><td>m<sub>eff</sub> = Γ²</td><td>effective mass along x: the mass of the mode that the base feels</td><td>kg</td></tr>
<tr><td>m<sub>res</sub></td><td>residual mass m<sub>B</sub> − m<sub>eff</sub>: B's mass in its other modes, which moves with the base</td><td>kg</td></tr>
<tr><td>q, y</td><td>modal coordinate of B's mode (x = <b>r</b>u + φq); y = q/Γ, the oscillator's motion relative to its base</td><td>m·√kg, m</td></tr>
<tr><td>u</td><td>displacement of B's base (A's tip)</td><td>m</td></tr>
<tr><td>φ<sup>T</sup>Cφ</td><td>modal damping of a mass-normalized mode, = 2ζω</td><td>1/s</td></tr>
<tr><td>m<sub>a</sub>, k<sub>a</sub>, c<sub>a</sub></td><td>A's oscillator: its mass (m<sub>tip</sub> or Γ<sub>A</sub>², plus m<sub>res</sub> if added), k<sub>a</sub> = m<sub>tip</sub>ω², c<sub>a</sub> = m<sub>tip</sub>φ<sup>T</sup>Cφ</td><td>kg, N/m, N·s/m</td></tr>
<tr><td>m<sub>b</sub>, k<sub>b</sub>, c<sub>b</sub></td><td>B's oscillator: m<sub>b</sub> = Γ², k<sub>b</sub> = m<sub>b</sub>ω², c<sub>b</sub> = m<sub>b</sub>φ<sup>T</sup>Cφ</td><td>kg, N/m, N·s/m</td></tr>
<tr><td>u<sub>a</sub>, u<sub>b</sub></td><td>displacements of the two oscillator masses (u<sub>a</sub>: A's tip)</td><td>m</td></tr>
<tr><td>μ</td><td>mass ratio m<sub>b</sub>/m<sub>a</sub></td><td>—</td></tr>
<tr><td>T</td><td>Rayleigh–Ritz basis: the chain's displacements x = T[u<sub>a</sub>; u<sub>b</sub>]</td><td>m/m</td></tr>
<tr><td>ζ</td><td>damping ratio: modal φ<sup>T</sup>Cφ/2ω for the oscillators, exact −Re λ/|λ| for the 2-DOF model and the chain</td><td>—</td></tr>
<tr><td>MAC</td><td>modal assurance criterion (φ<sup>T</sup>x)²/(φ<sup>T</sup>φ · x<sup>T</sup>x): shape similarity, 0 to 1</td><td>—</td></tr>
</table>
"""

THEORY_HTML = """
<p><i>Symbols are defined in the <a href="#notation">Notation</a> table at the end. The
<i>Matrices (step by step)</i> tab works every formula below through with the current
numbers.</i></p>
<h3>Two systems, joined</h3>
<p>A machine on a floor, a pipe on a support, an instrument on a panel: each part has its
own natural frequencies, measured or computed on its own. What happens to them when the
parts are joined? If the parts' frequencies are far apart, not much: each keeps its
mode, shifted a little. If two of them are close, and the masses involved are not too
different, the two modes <b>split</b>: neither frequency survives, and two new modes
appear either side, each a mix of both parts. A vibration absorber does this on purpose.
Most of the time it is not the intent, and it is worth being able to see it coming.</p>
<p>This page takes the chain on the Simulation page and splits it into two
<i>subsystems</i>:</p>
<ul>
<li><b>A</b>: masses 1 to j, on the ground through k<sub>1</sub>, c<sub>1</sub> as before,
its tip m<sub>j</sub> free.</li>
<li><b>B</b>: the rest, with the spring and damper that joined it to A,
k<sub>j+1</sub> and c<sub>j+1</sub>, tied to the ground instead. B's "ground" is where
A's tip will be.</li>
</ul>
<p>Each subsystem is solved on its own. Then one mode of each is replaced by a single-DOF
oscillator, and the two oscillators are joined the way A and B are, B's on A's:</p>
<p>&nbsp;&nbsp;ground — k<sub>a</sub>, c<sub>a</sub> — m<sub>a</sub> — k<sub>b</sub>,
c<sub>b</sub> — m<sub>b</sub></p>
<p>That 2-DOF system is solved and its two modes are compared with the chain's.</p>

<h3>The oscillator for one mode</h3>
<p>An oscillator keeps its mode's natural frequency ω and modal damping ratio ζ, so once
its mass m is chosen,</p>
<p>&nbsp;&nbsp;k = m ω²,&nbsp;&nbsp; c = m φ<sup>T</sup>Cφ = 2ζωm&nbsp;&nbsp;
(φ mass-normalized, φ<sup>T</sup>Mφ = 1).</p>
<p>The mass is the question: a mode has no one mass. It has as many as there are ways to
push on it, and the right one is the mass the <i>joint</i> feels (<a href="#masses">below</a>).</p>

<h3><a name="masses"></a>Which mass? The mass the joint feels</h3>
<p>Both oscillator masses are "the mass of one mode", but measured at different places, so
the formula and the value differ. A and B touch the joint in different ways:</p>
<ul>
<li>A's mode <b>moves</b> the interface, so its oscillator mass comes from how far the
interface moves in the mode: 1/φ<sub>tip</sub>².</li>
<li>B's mode <b>holds</b> the interface still and pushes on it, so its oscillator mass
comes from how hard it pushes back when the interface moves: Γ².</li>
</ul>

<h3>A: its modal mass at the interface</h3>
<p>A is grounded on the left, and B is joined to its <b>free tip</b>. In A's mode the tip
moves, so the natural coordinate for A's oscillator is the tip displacement u<sub>a</sub>.
Take the mass-normalized mode φ (φ<sup>T</sup>Mφ = 1) and scale it so the tip moves exactly
1: x = (φ/φ<sub>tip</sub>) u<sub>a</sub>. Then</p>
<p>&nbsp;&nbsp;kinetic energy&nbsp; ½ (φ/φ<sub>tip</sub>)<sup>T</sup>M(φ/φ<sub>tip</sub>)
u̇<sub>a</sub>² = ½ (1/φ<sub>tip</sub>²) u̇<sub>a</sub>²,<br>
&nbsp;&nbsp;strain energy&nbsp; ½ (ω²/φ<sub>tip</sub>²) u<sub>a</sub>².</p>
<p>An oscillator with the same energies at the same tip displacement has</p>
<p>&nbsp;&nbsp;m<sub>a</sub> = 1/φ<sub>tip</sub>²&nbsp;&nbsp;(the <b>modal mass at the
interface</b>),&nbsp;&nbsp; k<sub>a</sub> = ω²m<sub>a</sub>.</p>
<p>The driving-point FRF says the same thing: near resonance the tip receptance is
H<sub>tip,tip</sub> ≈ φ<sub>tip</sub>²/(ω<sub>r</sub>² − ω²) = 1/[m<sub>a</sub>(ω<sub>r</sub>² −
ω²)], so to a force at its tip A responds like an oscillator of mass 1/φ<sub>tip</sub>².
The tip moves more than the rest of A, so per unit of tip motion the mode looks light:
m<sub>a</sub> is less than A's total mass. This is the same modal mass the vibration
absorber presets use for their primary.</p>
<p><b>Why one entry of φ, squared, is a mass.</b> A mode shape by itself has no size: the
eigenproblem fixes only its <i>shape</i>. Scaled to 1 at the tip, ψ = φ/φ<sub>tip</sub> is
dimensionless (metres per metre of tip motion), and its <i>generalized mass</i>
ψ<sup>T</sup>Mψ = Σ m<sub>i</sub>ψ<sub>i</sub>² is in kg. Mass-normalizing divides the shape by
the square root of that mass, φ = ψ/√(ψ<sup>T</sup>Mψ), so every entry of φ has units of
1/√kg, and the tip entry is φ<sub>tip</sub> = 1/√(ψ<sup>T</sup>Mψ). Squaring and inverting it
only undoes the normalization:</p>
<p>&nbsp;&nbsp;1/φ<sub>tip</sub>² = ψ<sup>T</sup>Mψ = Σ m<sub>i</sub>(φ<sub>i</sub>/φ<sub>tip</sub>)².</p>
<p>The one entry stands for the whole sum, because the normalization put the whole mode's mass
into it.</p>
<p><b>Why only one entry.</b> Compare B's participation Γ = φ<sup>T</sup>M<b>r</b>, the mode
projected on its load, which is spread over every mass. B pushes on A only at A's tip, so A's
load is a <i>point load</i>, F<b>e</b><sub>tip</sub>, with <b>e</b><sub>tip</sub> = [0, …, 0,
1]<sup>T</sup>. Projecting A's mode on it, φ<sup>T</sup><b>e</b><sub>tip</sub> = φ<sub>tip</sub>, keeps
one entry only. The <i>Matrices (step by step)</i> tab puts the two projections side by side.</p>

<h3>B: its effective mass</h3>
<p>B's own modes are computed with its base held (its first spring and damper tied to
ground). In them <b>the interface does not move</b>: φ<sub>base</sub> = 0, so a
"1/φ<sub>base</sub>²" would be infinite. B cannot couple to the joint through its mode's
displacement there. It couples through <b>the force its mode puts on the base when the base
moves</b>.</p>
<p>When the base moves by u, the masses of B move by <b>r</b>u, plus the mode's motion
relative to the base, φq. <b>r</b> is the <b>influence vector</b> (or direction vector): how far
each DOF moves when the base moves by 1 in the direction of excitation, found from the static
response K<sub>B</sub><b>r</b> = k<sub>j+1</sub><b>e</b><sub>1</sub> (the base pulls on B's first
mass through its first spring only). Here every mass moves by 1, so <b>r</b> = <b>1</b> = [1, 1,
…, 1]<sup>T</sup>. In a 3-D model each node has x, y and z DOFs, and <b>r</b> for a shake along x
has 1 at the x DOFs and 0 at the others (plus lever arms for rotations). The chain has only
x DOFs, so there is only one direction, and <b>r</b> is all ones. Then</p>
<p>&nbsp;&nbsp;q̈ + ω²q = −Γ ü,&nbsp;&nbsp; with the <b>participation factor</b>
Γ = φ<sup>T</sup>M<b>r</b> = Σ m<sub>i</sub>φ<sub>i</sub> (M is B's own mass matrix, the diagonal of
its masses),<br>
&nbsp;&nbsp;force on the base&nbsp; F = m<sub>B</sub> ü + Γ q̈.</p>
<p>With y = q/Γ the base sees exactly two things:</p>
<ul>
<li>a mass <b>Γ²</b> on a spring of frequency ω, moving relative to it: the oscillator,
m<sub>b</sub> = Γ² (the <b>effective mass</b> along x), k<sub>b</sub> = ω²m<sub>b</sub>;</li>
<li>the remaining <b>m<sub>B</sub> − Γ²</b> moving rigidly with the base: the <b>residual
mass</b>. It belongs to B's other modes, which are stiff well below their frequencies. B's
base is A's tip, so it rides on A's oscillator: <i>Add B's residual mass to A</i> puts it
there.</li>
</ul>
<p>Units: φ is in 1/√kg and M<b>r</b> in kg, so Γ is in √kg and Γ² in kg. Like the tip modal
mass, it does not depend on how the mode is scaled: for any scaling ψ,
Γ² = (ψ<sup>T</sup>M<b>r</b>)²/(ψ<sup>T</sup>Mψ). The effective masses of all of B's modes add up to
B's total mass, <b>r</b><sup>T</sup>M<b>r</b> = m<sub>B</sub>. The lowest mode of a chain usually
carries most of it; the table lists each mode's share.</p>

<h3>Side by side</h3>
<table border="1" cellspacing="0" cellpadding="4">
<tr><th></th><th>A: modal mass at the interface</th><th>B: effective mass</th></tr>
<tr><td>Joined</td><td>at its free tip</td><td>at its base</td></tr>
<tr><td>Its modes</td><td>interface free: it moves</td><td>interface held: it does not move</td></tr>
<tr><td>Couples through</td><td>the displacement at the interface</td><td>the force on the base (inertia)</td></tr>
<tr><td>Formula</td><td>1/φ<sub>tip</sub>² = ψ<sup>T</sup>Mψ, ψ = φ/φ<sub>tip</sub></td>
<td>(φ<sup>T</sup>M<b>r</b>)², <b>r</b> = <b>1</b> here</td></tr>
<tr><td>Answers</td><td>"a force at the tip: how heavy does the mode feel?"</td>
<td>"the base shakes: how much mass swings on the spring?"</td></tr>
</table>
<p><b>Example.</b> A = m1, m2 (1 kg each, k = 400 N/m each), mode 1 at 1.967 Hz. The
mass-normalized φ = [0.526, 0.851], so</p>
<p>&nbsp;&nbsp;1/φ<sub>tip</sub>² = 1/0.851² = <b>1.38 kg</b>,&nbsp;&nbsp;
Γ = 0.526 + 0.851 = 1.376,&nbsp; Γ² = <b>1.89 kg</b>.</p>

<h3>Why not A's effective mass?</h3>
<p>Γ<sub>A</sub>² answers the base question for A: how hard shaking A's <i>ground</i> would
drive its mode. But A's ground does not move here, and B pushes on A's tip. For a uniform
chain Γ² is well above 1/φ<sub>tip</sub>² (1.89 against 1.38 kg above), because the tip
moves more than the average mass. Switch <i>Mass of A's oscillator</i> to the effective
mass to see it: the errors grow (in the tuned example of <i>Try it</i> from about 0.2% to
about 3%), and the model is no longer Rayleigh–Ritz, so its errors can have either sign.</p>

<h3>Why this works: it is a Rayleigh–Ritz model</h3>
<p>With both choices above, the 2-DOF model is exactly the chain restricted to two shapes:</p>
<ul>
<li>A in its mode φ<sub>A</sub>, scaled to u<sub>a</sub> at the tip, with all of B carried
rigidly along;</li>
<li>B in its mode φ<sub>B</sub> relative to its base: Γφ<sub>B</sub>(u<sub>b</sub> −
u<sub>a</sub>).</li>
</ul>
<p>Writing the kinetic and strain energies in u<sub>a</sub> and u<sub>b</sub> gives the
2-DOF masses and springs above. For B this is <b>Craig–Bampton</b> with one fixed-interface
mode kept (see the Substructuring page): its reduced mass matrix [[1, Γ], [Γ, m<sub>B</sub>]]
is the effective mass plus the residual mass in other coordinates. For A it is one
free-interface mode with no residual flexibility.</p>
<p>Being Rayleigh–Ritz, the 2-DOF frequencies are <b>upper bounds</b>: its mode 1 is never
below the chain's mode 1, nor its mode 2 below the chain's mode 2. The model is stiffer than
the chain because it allows fewer shapes. (Couple higher modes, and a 2-DOF mode can pair by
shape with a higher chain mode and fall below it: the bound is by mode number.) Leave out the residual
mass, or use A's effective mass, and that guarantee is gone.</p>

<h3>The 2-DOF system and mode splitting</h3>
<p>With μ = m<sub>b</sub>/m<sub>a</sub>, ω<sub>a</sub>² = k<sub>a</sub>/m<sub>a</sub> and
ω<sub>b</sub>² = k<sub>b</sub>/m<sub>b</sub>, the undamped frequencies solve</p>
<p>&nbsp;&nbsp;ω⁴ − ω²[ω<sub>a</sub>² + (1 + μ)ω<sub>b</sub>²] + ω<sub>a</sub>²ω<sub>b</sub>² = 0.</p>
<p>(With the residual mass, m<sub>a</sub> includes it and ω<sub>a</sub> is a little lower
than A's own.) Two limits:</p>
<ul>
<li><b>Far apart</b> (|f<sub>B</sub>/f<sub>A</sub> − 1| ≫ √μ): the roots are close to
ω<sub>a</sub> and ω<sub>b</sub>. The lower mode is mostly one subsystem, the upper mostly
the other. Coupling pushes them apart a little: the lower goes down, the upper up.</li>
<li><b>Tuned</b> (ω<sub>a</sub> = ω<sub>b</sub>): ω²/ω<sub>a</sub>² = 1 + μ/2 ± √(μ + μ²/4).
The product of the two roots is ω<sub>a</sub>⁴ and their sum (2 + μ)ω<sub>a</sub>², so
(ω<sub>2</sub> − ω<sub>1</sub>)² = μω<sub>a</sub>²: the two modes are exactly
<b>√μ ω<sub>a</sub></b> apart, for any μ, one either side. In the lower mode m<sub>a</sub>
and m<sub>b</sub> move in phase, in the upper in opposite phase. A mass ratio of only 1%
already splits them by 10%.</li>
</ul>
<p>In between, the <i>Veering &amp; splitting</i> tab shows the two frequencies as B is made
stiffer. They approach each other but never cross: they <b>veer</b> apart, and the mode
shapes swap over through the veering. The heavier B (the larger μ), the wider the
veering: the two curves stay apart over a band of f<sub>B</sub>/f<sub>A</sub> roughly √μ
wide. The chain's own frequencies do the same, and so does every other pair
of modes of A and B.</p>

<h3>Damping</h3>
<p>Each oscillator keeps its mode's modal damping ratio, and the 2-DOF damping ratios are
exact for the 2-DOF model. Where the modes split, each coupled mode is a mix of both, and
its damping is roughly the energy-weighted mix of the two. A lightly damped part joined to
a well-damped one at the same frequency shares its damping: that is how a tuned mass damper
works. Far apart, each keeps its own.</p>

<h3>What is left out</h3>
<ul>
<li><b>The other modes of A and B.</b> Each coupled chain mode near the two chosen ones is
well matched if the other modes are far off in frequency. A MAC well below 1 means they
take part too. Couple A's mode 2 with B's mode 1, say, to follow a different pair.</li>
<li><b>A's residual flexibility.</b> A's other modes add static flexibility at its tip,
which the 2-DOF model leaves out (MacNeal's and Rubin's methods, on the Substructuring
page, put it back). That is why the model is too stiff.</li>
<li><b>Non-proportional damping</b>: the modal damping ratios are the diagonal of
Φ<sup>T</sup>CΦ, as on the Simulation page's classical method.</li>
</ul>

<h3>Try it</h3>
<ol>
<li>Load <i>Vibration absorber (undamped)</i>, split after m1. A is the machine, B the
absorber. f<sub>B</sub>/f<sub>A</sub> = 1 and μ = 0.1: the 2-DOF model is the chain, so
the error is 0. The modes are √0.1 ≈ 32% of f<sub>A</sub> apart.</li>
<li>Uniform chain, 4 masses, split after m2. Both halves are the same, so f<sub>B</sub> =
f<sub>A</sub>, but μ is large: the modes are pushed far apart, and the upper one has a
lower MAC because the second modes of A and B are not far off.</li>
<li>Now make m3 and m4 light (0.1 kg) and k3, k4 small (40 N/m): B is tuned to A with μ ≈
0.14. The 2-DOF model is within a fraction of a percent. Sweep the mass ratio to see the
splitting shrink as B gets lighter.</li>
<li>Switch <i>Mass of A's oscillator</i> to the effective mass: the errors grow and can
change sign.</li>
<li>With 8 masses, split after m6 and couple A's mode 2 with B's mode 1: the pairing moves
to the chain's modes 2 and 3, but the lower one is 14% too low and its MAC is poor. A's mode 1
moves the tip too, and couples to B1 just as much, but it is left out. The Rayleigh–Ritz bound
still holds, by mode number: the 2-DOF mode 1 is above the chain's mode 1.</li>
</ol>
<h3><a name="notation"></a>Notation</h3>
""" + NOTATION_HTML


# ------------------------------------------------------------------ step by step (live numbers)

def _vec_table(headers: list[str], columns: list[np.ndarray], rows: list[str], total: list[str] | None = None,
               bold_row: int | None = None) -> str:
    """Term-by-term table: one row per DOF, one column per quantity, and an optional totals row."""
    scale = [float(np.abs(c).max()) if np.size(c) else 1.0 for c in columns]
    out = ["<table border='1' cellspacing='0' cellpadding='3'><tr>"
           + "".join(f"<th>{h}</th>" for h in headers) + "</tr>"]
    for r, label in enumerate(rows):
        style = f" bgcolor='{colors.block_tints['b', 'q']}'" if r == bold_row else ""
        cells = "".join(f"<td align='right'{style}>{fmt(c[r], s)}</td>" for c, s in zip(columns, scale))
        out.append(f"<tr><th>{label}</th>{cells}</tr>")
    if total is not None:
        out.append("<tr><th>Σ</th>" + "".join(f"<td align='right'><b>{t}</b></td>" for t in total) + "</tr>")
    out.append("</table>")
    return "".join(out)


def _mat(A: np.ndarray, labels: list[str], group: str, title: str) -> str:
    groups = [group] * len(labels)
    return matrix_html(A, labels, labels, groups, groups, title)


def _g(v: float) -> str:
    return f"{v:.4g}"


def matrices_html(model: CoupledModel, full: ModalResult) -> str:
    """Every step from the chain to the 2-DOF model, with the current numbers."""
    system, j, n = model.system, model.split, model.system.n
    A, B = model.A, model.B
    ra, rb = model.mode_a, model.mode_b
    M, C, K = system.matrices()
    labels = [f"x{i + 1}" for i in range(n)]
    la, lb = labels[:j], labels[j:]
    tints = colors.block_tints
    parts = [
        "<p><i>Every number below follows the current parameters and choices. Symbols are defined in "
        "the <a href='#notation'>Notation</a> table at the end.</i></p>",
        f"<p><small>Cell colours: <span style='background:{tints['i', 'i']}'>&nbsp;A&nbsp;</span> "
        f"<span style='background:{tints['b', 'b']}'>&nbsp;B&nbsp;</span> "
        f"<span style='background:{tints['b', 'i']}'>&nbsp;A–B coupling&nbsp;</span> "
        f"<span style='background:{tints['q', 'q']}'>&nbsp;2-DOF model&nbsp;</span></small></p>",
    ]

    # 1. The chain and the split
    groups = ["i"] * j + ["b"] * (n - j)
    kj, cj = system.stiffness[j], system.damping[j]
    parts.append("<h3>1. The chain, split after m" f"{j}</h3>")
    parts.append(
        f"<p>K [N/m], C [N·s/m] and M [kg] of the whole chain. A is {', '.join(la)}; B is {', '.join(lb)}. "
        f"They are joined by k<sub>{j + 1}</sub> = {_g(kj)} N/m and c<sub>{j + 1}</sub> = {_g(cj)} N·s/m, "
        "the off-diagonal (coupling) entries.</p>")
    parts.append(_side_by_side(*(matrix_html(X, labels, labels, groups, groups, t)
                                 for X, t in ((K, "K"), (C, "C"), (M, "M")))))
    parts.append(
        f"<p><b>A on its own</b> is the top-left block with k<sub>{j + 1}</sub> taken off its tip entry: "
        f"K<sub>A,{j}{j}</sub> = {_g(K[j - 1, j - 1])} − {_g(kj)} = {_g(A.K[-1, -1])}. "
        f"<b>B on its own</b>, with k<sub>{j + 1}</sub>, c<sub>{j + 1}</sub> tied to ground, is exactly the "
        f"bottom-right block: its first entry k<sub>{j + 1}</sub>"
        + (f" + k<sub>{j + 2}</sub>" if n - j > 1 else "")
        + " is unchanged, only the spring's other end is now the ground (B's <i>base</i>). "
        "M<sub>A</sub> and M<sub>B</sub> are the diagonals of each side's own masses.</p>")
    parts.append(_side_by_side(_mat(A.K, la, "i", "K<sub>A</sub>"), _mat(A.M, la, "i", "M<sub>A</sub>"),
                               _mat(B.K, lb, "b", "K<sub>B</sub>"), _mat(B.M, lb, "b", "M<sub>B</sub>")))

    # 2. Modes of each
    parts.append("<h3>2. The modes of each, on its own</h3>")
    parts.append("<p>Kψ = ω²Mψ for each subsystem. The eigenproblem fixes each mode's <i>shape</i> "
                 "but not its size: any multiple of ψ is the same mode.</p>")
    for sub, r in ((A, ra), (B, rb)):
        freqs = ", ".join((f"<b>{sub.name}{k + 1}: {_g(f)} Hz</b>" if k == r else f"{sub.name}{k + 1}: {_g(f)} Hz")
                          for k, f in enumerate(sub.fn_hz))
        parts.append(f"<p>{sub.name}: {freqs}. Coupled here: {sub.name}{r + 1} "
                     f"(ω = {_g(sub.omegas[r])} rad/s).</p>")

    # 3. Scaling: generalized mass and mass normalization
    phi_a, phi_b = A.Phi[:, ra], B.Phi[:, rb]
    ma_diag, mb_diag = np.diag(A.M), np.diag(B.M)
    tip = phi_a[-1]
    psi_a = phi_a / tip if abs(tip) > 1e-12 else phi_a / np.abs(phi_a).max()
    gm_a = float(psi_a @ A.M @ psi_a)
    peak = int(np.argmax(np.abs(phi_b)))
    psi_b = phi_b / phi_b[peak]
    gm_b = float(psi_b @ B.M @ psi_b)
    parts.append("<h3>3. Giving each mode a size: generalized mass and mass normalization</h3>")
    parts.append(
        "<p>Pick a scale, ψ, dimensionless. Its <b>generalized mass</b> ψ<sup>T</sup>Mψ = Σ m<sub>i</sub>ψ<sub>i</sub>² "
        "is in kg, and depends on the scale. The <b>mass-normalized</b> mode divides the shape by its square root, "
        "φ = ψ/√(ψ<sup>T</sup>Mψ), so φ<sup>T</sup>Mφ = 1 whatever ψ was, and φ is in <b>1/√kg</b>.</p>")
    parts.append(f"<p><b>A{ra + 1}</b>, scaled to 1 at its tip x{j} (the interface):</p>")
    parts.append(_vec_table(["", "m<sub>i</sub> [kg]", "ψ<sub>i</sub>", "m<sub>i</sub>ψ<sub>i</sub>² [kg]",
                             "φ<sub>i</sub> = ψ<sub>i</sub>/√(ψ<sup>T</sup>Mψ) [1/√kg]"],
                            [ma_diag, psi_a, ma_diag * psi_a**2, phi_a], la,
                            ["", "", f"ψ<sup>T</sup>Mψ = {_g(gm_a)}", ""], bold_row=j - 1))
    parts.append(f"<p><b>B{rb + 1}</b>, scaled to 1 at its largest entry, {lb[peak]}:</p>")
    parts.append(_vec_table(["", "m<sub>i</sub> [kg]", "ψ<sub>i</sub>", "m<sub>i</sub>ψ<sub>i</sub>² [kg]",
                             "φ<sub>i</sub> [1/√kg]"],
                            [mb_diag, psi_b, mb_diag * psi_b**2, phi_b], lb,
                            ["", "", f"ψ<sup>T</sup>Mψ = {_g(gm_b)}", ""]))

    # 4. How the joint loads each subsystem
    parts.append("<h3>4. How the joint loads each subsystem</h3>")
    parts.append(
        "<p>A mode's mass depends on <i>where and how</i> it is pushed. Write the load on the subsystem as a "
        "vector times one number, and project the mode on that vector: that is the mode's "
        "<b>participation</b>. The two sides of this joint are loaded in different ways.</p>")
    e_tip = np.zeros(j)
    e_tip[-1] = 1.0
    parts.append(
        f"<p><b>A is loaded at one point.</b> B pushes on A with the interface force F, on A's tip x{j} only. "
        f"The load is F·<b>e</b><sub>tip</sub>, with <b>e</b><sub>tip</sub> = [{', '.join('1' if v else '0' for v in e_tip)}]"
        "<sup>T</sup>: a <i>point load vector</i>, 1 at the loaded DOF and 0 elsewhere.</p>")
    rhs = np.zeros(n - j)
    rhs[0] = kj
    if kj > 0:
        r_vec = np.linalg.solve(B.K, rhs)
        r_text = (f"K<sub>B</sub><b>r</b> = k<sub>{j + 1}</sub><b>e</b><sub>1</sub> = "
                  f"[{', '.join(_g(v) for v in rhs)}]<sup>T</sup> gives <b>r</b> = "
                  f"[{', '.join(_g(v) for v in r_vec)}]<sup>T</sup>")
    else:
        r_vec = np.ones(n - j)
        r_text = (f"k<sub>{j + 1}</sub> = 0, so B floats and any base motion leaves it where it is; its rigid "
                  "motion is still <b>r</b> = <b>1</b>")
    parts.append(
        "<p><b>B is loaded by moving its base.</b> When the base moves by u, B's masses follow it by <b>r</b>u "
        "plus the mode's own motion. <b>r</b> is the <b>influence (direction) vector</b>: each DOF's static "
        "displacement for a unit base motion in the direction of excitation. The base pulls on B only through "
        f"k<sub>{j + 1}</sub> on its first mass, so {r_text}: every mass moves with the base, as a rigid body. "
        "In a 3-D model each node would have x, y and z DOFs, and <b>r</b> for a shake along x would have 1 "
        "at the x DOFs and 0 at the y and z DOFs (and lever arms for rotations). The chain has only x DOFs, "
        "so there is only the one direction and <b>r</b> = <b>1</b>. "
        "Moving the base accelerates every mass, so the load on B is inertial, −M<sub>B</sub><b>r</b>ü: "
        "spread over <i>every</i> mass, in proportion to its mass.</p>")

    # 5. Participation
    part_b = mb_diag * r_vec * phi_b
    gamma = float(part_b.sum())
    parts.append("<h3>5. Participation: project the mode on the load</h3>")
    parts.append(_side_by_side(
        "<p><b>A{0}</b>: φ<sup>T</sup><b>e</b><sub>tip</sub></p>".format(ra + 1)
        + _vec_table(["", "e<sub>tip,i</sub>", "φ<sub>i</sub>", "e<sub>tip,i</sub>φ<sub>i</sub>"],
                     [e_tip, phi_a, e_tip * phi_a], la, ["", "", f"φ<sub>tip</sub> = {_g(tip)}"], bold_row=j - 1),
        "<p><b>B{0}</b>: Γ = φ<sup>T</sup>M<sub>B</sub><b>r</b></p>".format(rb + 1)
        + _vec_table(["", "m<sub>i</sub>", "r<sub>i</sub>", "φ<sub>i</sub>", "m<sub>i</sub>r<sub>i</sub>φ<sub>i</sub>"],
                     [mb_diag, r_vec, phi_b, part_b], lb, ["", "", "", f"Γ = {_g(gamma)}"]),
    ))
    parts.append(
        "<p>This is why only <b>one entry</b> of φ counts for A: the point load vector is zero at every DOF but "
        "the tip, so the projection keeps φ<sub>tip</sub> alone. For B the load is at every mass, so every "
        "entry counts, weighted by its mass: Γ = Σ m<sub>i</sub>r<sub>i</sub>φ<sub>i</sub>. Units: "
        "φ<sub>tip</sub> in 1/√kg; Γ in (kg)(1)(1/√kg) = √kg.</p>")

    # 6. From participation to the oscillator's mass
    m_tip = float(A.tip_masses()[ra])
    parts.append("<h3>6. From participation to the oscillator's mass</h3>")
    parts.append(
        f"<p><b>A: m<sub>a</sub> = 1/φ<sub>tip</sub>²</b>. With the modal coordinate q (x = φq), the mode obeys "
        f"q̈ + ω²q = φ<sub>tip</sub>F, and the joint sees the tip move by u<sub>a</sub> = φ<sub>tip</sub>q. "
        f"Multiply by φ<sub>tip</sub>: ü<sub>a</sub> + ω²u<sub>a</sub> = φ<sub>tip</sub>²F, which is an oscillator "
        f"of mass 1/φ<sub>tip</sub>². The participation appears <i>twice</i> (in how F drives q, and in how q moves "
        f"the tip) and <i>divides</i>: a large tip motion per unit q makes the mode look light.</p>"
        f"<p>&nbsp;&nbsp;m<sub>tip</sub> = 1/φ<sub>tip</sub>² = 1/{_g(tip)}² = <b>{_g(m_tip)} kg</b> = "
        f"ψ<sup>T</sup>Mψ of step 3 ({_g(gm_a)} kg): the tip-scaled shape's generalized mass. Normalizing put the "
        "whole mode's mass into the tip entry, and squaring and inverting it gets that mass back.</p>")
    parts.append(
        f"<p><b>B: m<sub>b</sub> = Γ²</b>. The base motion drives the mode, q̈ + ω²q = −Γü, and the mode pushes "
        f"back on the base with Γq̈ (on top of m<sub>B</sub>ü for B moving rigidly). With y = q/Γ this is "
        f"ÿ + ω²y = −ü and a base force Γ²ÿ + m<sub>B</sub>ü: an oscillator of mass Γ² on the base. The "
        f"participation again appears twice (driving and pushing back) and here <i>multiplies</i>.</p>"
        f"<p>&nbsp;&nbsp;m<sub>eff</sub> = Γ² = {_g(gamma)}² = <b>{_g(gamma ** 2)} kg</b>; scale-free, "
        f"(ψ<sup>T</sup>M<b>r</b>)²/(ψ<sup>T</sup>Mψ) = {_g(float(psi_b @ B.M @ r_vec))}²/{_g(gm_b)} = "
        f"{_g(float(psi_b @ B.M @ r_vec) ** 2 / gm_b)} kg.</p>")
    eff = B.effective_masses
    parts.append("<p>Every mode of B, with its share. They add up to <b>r</b><sup>T</sup>M<sub>B</sub><b>r</b> = "
                 f"m<sub>B</sub> = {_g(B.total_mass)} kg:</p>")
    parts.append(_vec_table(["Mode", "f [Hz]", "Γ [√kg]", "Γ² [kg]", "share"],
                            [B.fn_hz, B.participation, eff, eff / B.total_mass],
                            [f"B{k + 1}" for k in range(eff.size)],
                            ["", "", f"{_g(eff.sum())}", f"{eff.sum() / B.total_mass:.3g}"], bold_row=rb))
    parts.append(
        f"<p>Residual mass: m<sub>B</sub> − Γ² = {_g(B.total_mass)} − {_g(gamma ** 2)} = "
        f"<b>{_g(model.m_residual)} kg</b>, B's other modes moving with the base (A's tip)"
        + (", added to m<sub>a</sub>." if model.residual else "; not added here (the check box is off).")
        + "</p>")
    gamma_a = float(A.participation[ra])
    parts.append(
        f"<p><b>For comparison, A's effective mass</b> uses A's own base motion, <b>r</b><sub>A</sub> = <b>1</b>: "
        f"Γ<sub>A</sub> = Σ m<sub>i</sub>φ<sub>i</sub> = {_g(gamma_a)} √kg, Γ<sub>A</sub>² = <b>{_g(gamma_a ** 2)} kg</b>, "
        f"against m<sub>tip</sub> = {_g(m_tip)} kg. It projects A's mode on a load at every mass, but the joint "
        "loads A at its tip only"
        + (" (and it is the mass in use now: <i>Mass of A's oscillator</i> is set to the effective mass)."
           if model.a_mass == "effective" else ".") + "</p>")

    # 7. The oscillators
    a, b = model.osc_a, model.osc_b
    ca, cb = A.modal_damping[ra], B.modal_damping[rb]
    parts.append("<h3>7. The two oscillators</h3>")
    parts.append(
        "<p>Each keeps its mode's frequency and modal damping: k = mω², c = m·φ<sup>T</sup>Cφ (= 2ζωm).</p>"
        f"<p>&nbsp;&nbsp;A{ra + 1}: m = {_g(a.m)} kg, k<sub>a</sub> = {_g(a.m)} × {_g(A.omegas[ra])}² = "
        f"<b>{_g(a.k)} N/m</b>, φ<sup>T</sup>C<sub>A</sub>φ = {_g(ca)} 1/s, c<sub>a</sub> = <b>{_g(a.c)} N·s/m</b>"
        + (f"; m<sub>a</sub> = {_g(a.m)} + {_g(model.m_residual)} (residual) = <b>{_g(model.m_a)} kg</b>"
           if model.residual else f"; m<sub>a</sub> = <b>{_g(model.m_a)} kg</b>")
        + f"<br>&nbsp;&nbsp;B{rb + 1}: m<sub>b</sub> = <b>{_g(b.m)} kg</b>, k<sub>b</sub> = {_g(b.m)} × "
        f"{_g(B.omegas[rb])}² = <b>{_g(b.k)} N/m</b>, φ<sup>T</sup>C<sub>B</sub>φ = {_g(cb)} 1/s, "
        f"c<sub>b</sub> = <b>{_g(b.c)} N·s/m</b></p>")

    # 8. The 2-DOF model
    l2 = ["u<sub>a</sub>", "u<sub>b</sub>"]
    parts.append("<h3>8. The 2-DOF model</h3>")
    parts.append("<p>ground — k<sub>a</sub>, c<sub>a</sub> — m<sub>a</sub> — k<sub>b</sub>, c<sub>b</sub> — "
                 "m<sub>b</sub>, assembled like a 2-mass chain:</p>")
    parts.append(_side_by_side(_mat(model.K, l2, "q", "K"), _mat(model.C, l2, "q", "C"), _mat(model.M, l2, "q", "M")))
    zs = ", ".join("overdamped" if z is None else f"{z:.4f}" for z in model.zetas)
    parts.append(
        f"<p>K η = ω² M η gives <b>{_g(model.fn_hz[0])}</b> and <b>{_g(model.fn_hz[1])} Hz</b> (ζ = {zs}, exact, "
        f"from the damped eigenvalues). The chain's lowest two: {_g(full.modes[0].fn_hz)}"
        + (f" and {_g(full.modes[1].fn_hz)}" if len(full.modes) > 1 else "") + " Hz. μ = m<sub>b</sub>/m<sub>a</sub> "
        f"= {_g(model.mass_ratio)}.</p>")

    # 9. Rayleigh-Ritz check
    T = np.zeros((n, 2))
    T[:j, 0] = psi_a if abs(tip) > 1e-12 else 0.0
    rel = B.participation[rb] * phi_b
    T[j:, 0] = r_vec - rel
    T[j:, 1] = rel
    parts.append("<h3>9. Check: the 2-DOF model is a Rayleigh–Ritz model of the chain</h3>")
    parts.append(
        "<p>The chain restricted to two shapes, x = T[u<sub>a</sub>; u<sub>b</sub>]: column 1 is A's mode scaled "
        "to 1 at the tip with B carried rigidly (<b>r</b>) minus B's mode, column 2 is B's mode relative to its "
        "base, Γφ<sub>B</sub>. Projecting the chain's matrices, T<sup>T</sup>MT and T<sup>T</sup>KT:</p>")
    TM, TK = T.T @ M @ T, T.T @ K @ T
    parts.append(_side_by_side(
        matrix_html(T, labels, l2, groups, ["q", "q"], "T"),
        _mat(TK, l2, "q", "T<sup>T</sup>KT"), _mat(TM, l2, "q", "T<sup>T</sup>MT")))
    same = np.allclose(TM, model.M, atol=1e-9 * max(1.0, M.max())) and np.allclose(TK, model.K, rtol=1e-9,
                                                                                     atol=1e-9 * K.max())
    if same:
        parts.append("<p>These are the 2-DOF K and M of step 8 exactly: the oscillator model <i>is</i> this "
                     "Rayleigh–Ritz model, so its frequencies are upper bounds of the chain's lowest two. (T<sup>T</sup>MT "
                     "is diagonal: B's mode, relative to its base, does not share inertia with the rigid motion, since "
                     "Γ·Γ − Γ²·φ<sup>T</sup>Mφ = 0.)</p>")
    else:
        parts.append(f"<p style='color:{colors.poor}'>These differ from the 2-DOF matrices of step 8: with "
                     + ("no residual mass" if not model.residual else "A's effective mass")
                     + " the oscillator model is not this Rayleigh–Ritz model, so it gives no bound.</p>")
    parts.append("<h3><a name='notation'></a>Notation</h3>" + NOTATION_HTML)
    return "".join(parts)
