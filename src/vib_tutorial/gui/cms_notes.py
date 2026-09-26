"""Text for the Substructuring page: the theory notes and the live matrix walkthrough."""

from __future__ import annotations

import numpy as np

from ..core import (
    CMSModel,
    ModalResult,
    Substructure,
    compare_modes,
    substructure_damping,
)
from ..core.modal import TWO_PI

# Cell tint for each partition block, keyed by the sorted (row, column) group pair.
BLOCK_TINTS = {
    ("i", "i"): "#dbe8f5",
    ("b", "b"): "#dcefd8",
    ("b", "i"): "#fbe7d3",
    ("q", "q"): "#ebe2f5",
    ("b", "q"): "#f6efcc",
    ("i", "q"): "#efe6f7",
}
LEGEND_HTML = (
    "<p><small>Cell colors: "
    f"<span style='background:{BLOCK_TINTS['i', 'i']}'>&nbsp;interior–interior (ii)&nbsp;</span> "
    f"<span style='background:{BLOCK_TINTS['b', 'i']}'>&nbsp;interior–boundary (ib, bi)&nbsp;</span> "
    f"<span style='background:{BLOCK_TINTS['b', 'b']}'>&nbsp;boundary–boundary (bb)&nbsp;</span> "
    f"<span style='background:{BLOCK_TINTS['q', 'q']}'>&nbsp;modal–modal (qq)&nbsp;</span> "
    f"<span style='background:{BLOCK_TINTS['b', 'q']}'>&nbsp;modal–boundary (qb)&nbsp;</span>"
    "</small></p>"
)

THEORY_HTML = """
<p><i>Symbols are defined in the <a href="#notation">Notation</a> table at the end.</i></p>
<h3>Why substructure?</h3>
<p>Real structures are built from components: an engine on a frame, a wing on a fuselage,
a payload on a launcher. <b>Dynamic substructuring</b> models each component on its own,
reduces it to a handful of coordinates, and then joins the reduced components at their
interfaces. The assembled model is much smaller than the full one, and a component can
be changed or re-analysed without touching the others. <b>Component mode synthesis
(CMS)</b> does the reduction with each component's own mode shapes.</p>
<p>On this page the chain is cut at one <i>interface</i> mass into two substructures,
<b>A</b> (grounded) and <b>B</b> (free end). The interface mass belongs to both: A owns its
mass and the springs to its left, B owns the springs to its right. (The free-interface methods
split the interface mass half and half instead, so that each substructure has mass wherever it
can move.)</p>
<p>The <i>Method</i> selector chooses how each substructure is reduced: <b>Craig–Bampton</b>
(fixed interface, the next few sections) or <b>Rubin</b> and <b>MacNeal</b> (free interface,
from <a href="#free">Free-interface methods</a> on). The <i>Compare methods</i> tab runs all
three on the same cut.</p>

<h3>Boundary (master) and interior DOFs</h3>
<ul>
<li><b>Boundary DOFs x<sub>b</sub></b> are kept as physical displacements. Here they are
the interface mass, shared by A and B, and the last mass of the chain, where the force is
applied.</li>
<li><b>Interior DOFs x<sub>i</sub></b> are every other mass. Each belongs to exactly one
substructure and is replaced by a few modal coordinates q.</li>
</ul>
<p><b>Why is the force limited to the last mass?</b> A force on a boundary DOF enters the
reduced model unchanged (f̂ = T<sup>T</sup>f = [0; f<sub>b</sub>]) and the response there
is read directly, so any difference from the full model is caused by the reduction and
not by how the load was projected. A force on an interior DOF is allowed in Craig–Bampton,
but it only reaches the model through the kept modes and the constraint modes. In practice,
the DOFs that are loaded, measured or connected to other components are made masters.</p>

<h3>What is a basis? (the key idea)</h3>
<p>Every method on this page, and modal analysis itself, rests on one idea: <b>describing the
motion with a different set of shapes</b>. It is worth getting this clear before looking at
any Craig–Bampton matrix.</p>

<h4>Coordinates are amplitudes of shapes</h4>
<p>A displacement of the chain is a vector x of N numbers. Any such vector can be built as a
sum of N independent <i>basis vectors</i> (shapes) v<sub>1</sub> … v<sub>N</sub>, each
multiplied by an amplitude c<sub>j</sub>:</p>
<p>&nbsp;&nbsp;x = v<sub>1</sub>c<sub>1</sub> + v<sub>2</sub>c<sub>2</sub> + … = V c,
&nbsp;&nbsp; V = [v<sub>1</sub> v<sub>2</sub> …]</p>
<p>The amplitudes c are the <b>coordinates</b> in that basis. The physical DOFs are just
one choice: V = I, where shape j is "move mass j by 1 m, hold all the others". Nothing makes
that choice special. It is convenient for building M and K, but usually a poor way to
<i>describe</i> the motion, because every mass moves in every mode.</p>
<p>An everyday analogy: a point on a map can be given as (east, north) or as (distance,
bearing) along a road. Same point, different coordinates. And a vibrating guitar string is
naturally described by the amplitudes of its sine-shaped harmonics, not by the displacement
of each point.</p>

<h4>What a change of basis does to the equations</h4>
<p>Substitute x = Vc into Mẍ + Cẋ + Kx = f and premultiply by V<sup>T</sup>:</p>
<p>&nbsp;&nbsp;(V<sup>T</sup>MV) c̈ + (V<sup>T</sup>CV) ċ + (V<sup>T</sup>KV) c = V<sup>T</sup>f</p>
<p>Why V<sup>T</sup>? Energy. The kinetic energy is ½ẋ<sup>T</sup>Mẋ =
½ċ<sup>T</sup>(V<sup>T</sup>MV)ċ and the strain energy is ½c<sup>T</sup>(V<sup>T</sup>KV)c, so
the new matrices describe the <i>same</i> energies in the new coordinates. V<sup>T</sup>f is
the work the force does through each shape (a generalized force). This is also the principle
of virtual work, or a Galerkin projection.</p>
<p>That gives a practical way to <b>read</b> any transformed matrix:</p>
<ul>
<li>Diagonal entry (V<sup>T</sup>KV)<sub>jj</sub> = v<sub>j</sub><sup>T</sup>Kv<sub>j</sub>: twice
the strain energy of shape j at unit amplitude, a "generalized stiffness".</li>
<li>Off-diagonal entry v<sub>j</sub><sup>T</sup>Kv<sub>k</sub>: how much shapes j and k interact
elastically. Zero means they are <i>stiffness-orthogonal</i>: they can move independently
without exchanging strain energy. The same holds for M (inertia coupling) and C.</li>
</ul>

<h4>Modal analysis is already a change of basis</h4>
<p>Choosing V = Φ, the system's own mass-normalized mode shapes, is the best-known example.
Those shapes happen to be orthogonal through both M and K, so V<sup>T</sup>MV = I and
V<sup>T</sup>KV = diag(ω²): every off-diagonal term vanishes and the equations uncouple.
That is <i>why</i> modal coordinates are so useful, not a coincidence of the solver.</p>

<h4>A worked example (2 masses, m = 1 kg, k = 400 N/m)</h4>
<p>K = [ 800 −400 ; −400 400 ], M = I. True frequencies: 1.967 and 5.150 Hz.</p>
<ol>
<li><b>A full, different basis.</b> Take v<sub>1</sub> = [1, 1] ("move together") and
v<sub>2</sub> = [1, −1] ("move apart"). Then V<sup>T</sup>KV = [ 400 400 ; 400 2000 ] and
V<sup>T</sup>MV = [ 2 0 ; 0 2 ]. The matrices look nothing like K and M, and the off-diagonal
400 says the two shapes are coupled, yet solving gives exactly 1.967 and 5.150 Hz. A
<b>square</b> (complete) basis only rewrites the problem; it cannot change the answer.</li>
<li><b>Truncate it.</b> Keep only v<sub>1</sub>: one equation, ω² = 400 / 2, f = 2.25 Hz,
14% high. The masses are now forced to move together, which is a constraint, so the structure
is stiffer and the frequency can only go up (the Rayleigh quotient,
f² ∝ v<sup>T</sup>Kv / v<sup>T</sup>Mv).</li>
<li><b>Choose a better shape.</b> Use the static deflection under a tip load instead,
v = [0.5, 1]: ω² = 200 / 1.25, f = 2.01 Hz, only 2.3% high with a single coordinate. (The true
mode is [0.618, 1].) A static shape captures most of the low-frequency motion. That is exactly
why Craig–Bampton uses static constraint modes.</li>
</ol>

<h4>Reduction = a tall basis</h4>
<p>When V has fewer columns than rows (N × m, m &lt; N), x = Vc <i>restricts</i> the motion to
combinations of m shapes: the model can only move along those "rails". Everything then depends
on whether the true motion lies (nearly) inside that set of shapes. The MAC in the comparison
table measures exactly that for each mode. Adding shapes can only enlarge the set, which is why
the frequencies converge from above.</p>

<h4>The Craig–Bampton basis, and why it is chosen this way</h4>
<p>Inside a component, any motion can be split into two parts:</p>
<ul>
<li><b>Motion forced by the boundary.</b> If the boundary moves slowly, the interior simply
follows statically. The constraint modes Ψ describe this exactly: column j is the static shape
when boundary DOF j moves by 1 and the others are held.</li>
<li><b>The component's own vibration relative to that.</b> With the boundary held, what is left
is a free vibration of the clamped component, so it is described by its fixed-interface modes
Φ, lowest first, since low-frequency loading excites them most.</li>
</ul>
<p>Together they are complete: keep every fixed-interface mode and T is square and invertible,
so nothing is lost. The design has three further consequences:</p>
<ul>
<li><b>The boundary coordinates stay physical.</b> Every fixed-interface mode is zero on the
boundary, and each constraint mode is 1 on one boundary DOF and 0 on the others. So the
coordinate multiplying constraint mode j <i>is</i> the displacement x<sub>b,j</sub>. This is what
makes assembly trivial: two components are joined simply by giving their interface the same
x<sub>b</sub>. Modal coordinates of two separate components cannot be joined like that.</li>
<li><b>K̂ is block diagonal.</b> The constraint modes are static, so they do no elastic work
against the fixed-interface modes: φ<sup>T</sup>(K<sub>ii</sub>Ψ + K<sub>ib</sub>) = 0. The two
families are stiffness-orthogonal.</li>
<li><b>M̂ is not.</b> They are not mass-orthogonal: when the boundary accelerates, the inertia
of the interior drives the component's modes. M̂<sub>qb</sub> is that inertial coupling, and it
is how the components talk to each other dynamically. (For C, see <i>Damping</i> under <i>Properties worth knowing</i>.)</li>
</ul>
<p>The <i>Basis</i> tab plots every column of the current T as a shape along the chain. For the
default 8-mass chain cut at m4 with one mode each, there are four: A's first clamped mode (over
m1–m3, zero elsewhere), B's first clamped mode (over m5–m7), a "tent" for x<sub>4</sub> (rising
linearly from the ground to m4, falling to m8, the static shape of both components when the
interface moves) and a ramp for x<sub>8</sub> (zero up to m4, rising to the tip). Every motion
the reduced model can make is a combination of those four shapes.</p>

<h3>Original vs substructured formulation</h3>
<table border="1" cellspacing="0" cellpadding="4">
<tr><th></th><th>Full (original) model</th><th>Craig–Bampton substructured model</th></tr>
<tr><td><b>Unknowns</b></td><td>x: all N physical displacements</td>
<td>[q<sub>A</sub>, q<sub>B</sub>, x<sub>b</sub>]: modal amplitudes of each component plus
the physical boundary displacements</td></tr>
<tr><td><b>Assembly</b></td><td>every element into one M, K</td>
<td>each component separately into M<sup>(s)</sup>, K<sup>(s)</sup>; the full matrices are
their sum, K = Σ L<sub>s</sub><sup>T</sup>K<sup>(s)</sup>L<sub>s</sub></td></tr>
<tr><td><b>Ordering</b></td><td>masses 1…N along the chain</td>
<td>per component: [interior | boundary]</td></tr>
<tr><td><b>Basis</b></td><td>x = I x</td>
<td>x<sup>(s)</sup> = T<sup>(s)</sup>[q; x<sub>b</sub>], &nbsp;
T = [ Φ<sub>k</sub> &nbsp;Ψ ; 0 &nbsp;I ]</td></tr>
<tr><td><b>Stiffness</b></td><td>tridiagonal K</td>
<td>K̂ = [ Λ<sub>k</sub> &nbsp;0 ; 0 &nbsp;K̂<sub>bb</sub> ]: block diagonal, modes and
boundary <i>not</i> coupled by stiffness</td></tr>
<tr><td><b>Mass</b></td><td>diagonal M</td>
<td>M̂ = [ I &nbsp;M̂<sub>qb</sub> ; M̂<sub>bq</sub> &nbsp;M̂<sub>bb</sub> ]: modes couple
to the boundary through inertia only</td></tr>
<tr><td><b>Damping</b></td><td>tridiagonal C, assembled like K</td><td>Ĉ = T<sup>T</sup>CT on the
undamped basis: not block diagonal unless C<sup>(s)</sup> ∝ K<sup>(s)</sup></td></tr>
<tr><td><b>Force</b></td><td>f</td><td>f̂ = T<sup>T</sup>f</td></tr>
<tr><td><b>Size</b></td><td>N</td><td>Σ k<sub>s</sub> + n<sub>b</sub></td></tr>
<tr><td><b>Accuracy</b></td><td>exact</td>
<td>exact if every fixed-interface mode is kept; otherwise each frequency is an upper
bound on the true one, converging as more modes are kept</td></tr>
</table>

<h3>Step by step</h3>
<p>The <i>Matrices</i> tab shows every one of these steps with the current numbers.</p>
<ol>
<li><b>Partition each component</b> into interior and boundary DOFs:
<br>&nbsp;&nbsp;[ M<sub>ii</sub> M<sub>ib</sub> ; M<sub>bi</sub> M<sub>bb</sub> ]
[ẍ<sub>i</sub>; ẍ<sub>b</sub>] + [ K<sub>ii</sub> K<sub>ib</sub> ; K<sub>bi</sub>
K<sub>bb</sub> ] [x<sub>i</sub>; x<sub>b</sub>] = [f<sub>i</sub>; f<sub>b</sub>]
<br>(for lumped masses M<sub>ib</sub> = 0).</li>
<li><b>Fixed-interface normal modes.</b> Clamp the boundary (x<sub>b</sub> = 0) and solve
K<sub>ii</sub>φ = ω²M<sub>ii</sub>φ. Keep the k lowest, mass-normalized, as
Φ<sub>k</sub>. These capture the component's own dynamics.</li>
<li><b>Constraint modes.</b> Move one boundary DOF by 1 m, hold the others, and let the
interior find its static equilibrium: Ψ = −K<sub>ii</sub><sup>−1</sup>K<sub>ib</sub>. One
column per boundary DOF. They capture how the component deforms when its boundary moves,
including rigid-body motion.</li>
<li><b>Transform.</b> x<sub>i</sub> = Φ<sub>k</sub>q + Ψx<sub>b</sub>, so
[x<sub>i</sub>; x<sub>b</sub>] = T[q; x<sub>b</sub>], and M̂ = T<sup>T</sup>MT,
K̂ = T<sup>T</sup>KT. Because Ψ is a static solution, the stiffness decouples:
<br>&nbsp;&nbsp;K̂<sub>qq</sub> = Λ<sub>k</sub> = diag(ω<sub>r</sub>²),
&nbsp; K̂<sub>qb</sub> = 0, &nbsp; K̂<sub>bb</sub> = K<sub>bb</sub> − K<sub>bi</sub>
K<sub>ii</sub><sup>−1</sup>K<sub>ib</sub>
<br>K̂<sub>bb</sub> is the component's static stiffness seen from its boundary (a Schur
complement, the same as Guyan condensation). The mass keeps a coupling term:
<br>&nbsp;&nbsp;M̂<sub>qq</sub> = I, &nbsp; M̂<sub>qb</sub> =
Φ<sub>k</sub><sup>T</sup>(M<sub>ii</sub>Ψ + M<sub>ib</sub>), &nbsp;
M̂<sub>bb</sub> = M<sub>bb</sub> + M<sub>bi</sub>Ψ + Ψ<sup>T</sup>M<sub>ib</sub> +
Ψ<sup>T</sup>M<sub>ii</sub>Ψ</li>
<li><b>Assemble.</b> The reduced components share their boundary DOFs: the interface x
is the <i>same</i> coordinate in A and B, so their boundary entries add, exactly as
element matrices add in the original assembly. Compatibility (A and B move together at the
interface) is automatic. The modal coordinates stay private to their component.</li>
<li><b>Solve and recover.</b> Solve K̂η = ω²M̂η for the reduced model, then recover the
physical shape x = Tη for comparison with the full model's modes (frequency error and
MAC).</li>
</ol>

<h3>Hurty and Craig–Bampton</h3>
<p><b>Hurty (1965)</b> introduced component mode synthesis. He described each component's
motion with rigid-body modes, "redundant" constraint modes and fixed-constraint normal
modes, treating the statically determinate boundary DOFs differently from the redundant
ones. <b>Craig and Bampton (1968)</b> simplified this: treat <i>every</i> boundary DOF the
same way, with one constraint mode each. Rigid-body motion is already contained in the
constraint modes, so no separate treatment is needed. The result is the method used here,
often called Hurty/Craig–Bampton, and still the industry standard (for example for
coupled-loads analysis of spacecraft on launchers).</p>
<p>Other CMS families use <i>free-interface</i> component modes instead (MacNeal, Rubin,
Craig–Chang), which are closer to what a modal test measures but need residual-flexibility
corrections. They are next.</p>

<h3><a name="free"></a>Free-interface methods: MacNeal and Rubin</h3>
<h4>Why a free interface?</h4>
<p>Craig–Bampton clamps each component's boundary. That is easy in a computer and hard in a
laboratory: nobody can hold a satellite's interface ring perfectly still while shaking it.
What a <b>modal test</b> measures is a component hanging freely (on soft bungees), so its
<i>free-interface</i> modes. They also belong to the component alone, not to how it will be
held, so a supplier can deliver them without knowing the rest of the structure.</p>

<h4>Free modes alone converge badly</h4>
<p>Solve Kφ = ω²Mφ for the whole substructure, boundary free, and describe its motion with the
lowest few free modes (Hou's and Goldman's methods, 1969). The trouble is at the interface. In
the assembled structure the neighbour <i>pushes</i> on the interface, and the local static
deflection under that push is spread over <i>all</i> the free modes, mostly the high ones that
were thrown away. A truncated set of free modes is therefore far too stiff at exactly the place
the components meet.</p>

<h4>The fix: residual flexibility</h4>
<p>The discarded modes are high in frequency, so under the interface forces f<sub>b</sub> they
respond almost statically. Their static response is kept even though the modes are not:</p>
<p>&nbsp;&nbsp;x = Φ<sub>k</sub>q + G<sub>d</sub> E<sub>b</sub> f<sub>b</sub>, &nbsp;&nbsp;
G<sub>d</sub> = Σ<sub>discarded</sub> φ<sub>r</sub>φ<sub>r</sub><sup>T</sup> / ω<sub>r</sub>²
= K<sup>−1</sup> − Φ<sub>k</sub>Λ<sub>k</sub><sup>−1</sup>Φ<sub>k</sub><sup>T</sup></p>
<p>G<sub>d</sub> is the <b>residual flexibility</b>: the flexibility of the component minus the
part the kept modes already describe. The second form needs no discarded modes at all, which is
how it is done in practice (from K<sup>−1</sup>, or from a static test). Its columns at the
boundary, G<sub>d</sub>E<sub>b</sub>, are <b>residual attachment modes</b>: the static
deflection under a unit force at one boundary DOF, less what the kept modes carry. They are
stiffness-orthogonal to the kept modes: Φ<sub>k</sub><sup>T</sup>KG<sub>d</sub> = 0.</p>

<h4>Making the boundary physical again</h4>
<p>Forces are awkward coordinates for joining components. The boundary rows of the equation above
give x<sub>b</sub> = Φ<sub>bk</sub>q + G<sub>bb</sub>f<sub>b</sub>, so
f<sub>b</sub> = G<sub>bb</sub><sup>−1</sup>(x<sub>b</sub> − Φ<sub>bk</sub>q). Substituting back:</p>
<p>&nbsp;&nbsp;[x<sub>i</sub>; x<sub>b</sub>] = T [q; x<sub>b</sub>], &nbsp;&nbsp;
T = [ Φ<sub>ik</sub> − RΦ<sub>bk</sub> &nbsp; R ; 0 &nbsp; I ], &nbsp;&nbsp;
R = G<sub>ib</sub>G<sub>bb</sub><sup>−1</sup></p>
<p>This has <i>exactly</i> the shape of the Craig–Bampton T: modal columns that vanish on the
boundary, and boundary columns that are 1 at their own DOF and 0 at the others. So the same
assembly works: shared x<sub>b</sub>, entries added. (This form is due to Craig and Chang, 1977.)
The stiffness is no longer block diagonal:</p>
<p>&nbsp;&nbsp;K̂ = [ Λ<sub>k</sub> + Φ<sub>bk</sub><sup>T</sup>G<sub>bb</sub><sup>−1</sup>Φ<sub>bk</sub>
&nbsp; −Φ<sub>bk</sub><sup>T</sup>G<sub>bb</sub><sup>−1</sup> ; −G<sub>bb</sub><sup>−1</sup>Φ<sub>bk</sub>
&nbsp; G<sub>bb</sub><sup>−1</sup> ]</p>
<p>G<sub>bb</sub><sup>−1</sup> is the residual stiffness seen from the boundary: a spring from each
boundary DOF to the modal motion.</p>

<h4>MacNeal (1971) and Rubin (1975)</h4>
<ul>
<li><b>MacNeal</b> treats the residual flexibility as a <i>massless spring</i>: only the kept modes
carry inertia, M̂ = [ I 0 ; 0 0 ]. The boundary DOFs then have no mass: they follow
the modes statically and are condensed out, so the model has only as many modes as modes kept.
Dropping mass raises frequencies and adding the residual flexibility lowers them, so there is no
bound, and the method is not exact even with every mode there is room for.</li>
<li><b>Rubin</b> adds the <i>residual mass</i>: the inertia of the residual attachment modes. Here
this is done as a full Rayleigh–Ritz projection, M̂ = T<sup>T</sup>MT (the Craig–Chang form of
Rubin's method). It has all of Craig–Bampton's guarantees: upper bounds that
fall as modes are added, and exact with every mode kept.</li>
</ul>
<p>Damping is not part of either original method. Here both project it like the stiffness,
Ĉ = T<sup>T</sup>CT, so the two differ only in the residual mass.</p>

<h4>Rigid-body modes</h4>
<p>B is joined to the chain only through the interface, so on its own it floats: its first free
mode is a <b>rigid-body mode</b> at 0 Hz, moving every mass by the same amount. It has infinite
flexibility (K is singular, K<sup>−1</sup> does not exist), so it cannot be part of G<sub>d</sub>
and must always be kept, which is why B's spin box starts at 1. In practice G<sub>d</sub> of a
floating component comes from <i>inertia relief</i>: the flexibility of the component held
statically determinate, with the rigid-body part projected out. On this page G<sub>d</sub> is
built directly from the discarded elastic modes, which gives the same matrix.</p>

<h4>Three methods side by side</h4>
<table border="1" cellspacing="0" cellpadding="4">
<tr><th></th><th>Craig–Bampton</th><th>Rubin</th><th>MacNeal</th></tr>
<tr><td><b>Component modes</b></td><td>boundary held (fixed interface)</td>
<td colspan="2">boundary free (free interface), rigid-body modes always kept</td></tr>
<tr><td><b>Static shapes</b></td><td>constraint modes Ψ = −K<sub>ii</sub><sup>−1</sup>K<sub>ib</sub></td>
<td colspan="2">residual attachment modes R = G<sub>ib</sub>G<sub>bb</sub><sup>−1</sup></td></tr>
<tr><td><b>Coordinates</b></td><td colspan="3">[q, x<sub>b</sub>]: the same size for the same number
of modes kept, joined the same way</td></tr>
<tr><td><b>Modes coupled to the boundary by</b></td><td>mass only (K̂ block diagonal)</td>
<td>mass and stiffness</td><td>stiffness only (boundary massless)</td></tr>
<tr><td><b>Number of modes</b></td><td colspan="2">n<sub>red</sub></td><td>modes kept only</td></tr>
<tr><td><b>Accuracy</b></td><td colspan="2">upper bounds; exact with all modes</td>
<td>no bound; never exact</td></tr>
<tr><td><b>Component modes from a test?</b></td><td>hard (the boundary must be clamped)</td>
<td colspan="2">natural (free-free test), plus a residual flexibility measurement</td></tr>
</table>
<p>With the same number of coordinates, Rubin is usually more accurate for the lowest modes and
Craig–Bampton for the higher ones: the residual flexibility is a <i>static</i> correction, so it
helps most well below the discarded frequencies. Also, B's first kept free mode is its rigid-body
mode, which Craig–Bampton gets for free inside its constraint modes.</p>

<h3>Properties worth knowing</h3>
<ul>
<li><b>No modes kept = Guyan reduction.</b> Only the constraint modes remain, so the model
is exact statically (at 0 Hz) and good for low modes when little mass is "hidden" in the
interior.</li>
<li><b>All modes kept = exact.</b> T is then square and invertible: nothing is thrown
away, the problem is only rewritten in new coordinates.</li>
<li><b>Upper bounds.</b> CB is a Rayleigh–Ritz method: it restricts the motion to a
subspace, which can only make the structure stiffer. Every CB frequency is ≥ the true one,
and the error shrinks monotonically as modes are added.</li>
<li><b>Rule of thumb.</b> Keep fixed-interface modes up to about 1.5–2× the highest
frequency of interest. The Matrices tab lists each component's fixed-interface frequencies,
kept and discarded.</li>
<li><b>Damping.</b> The component modes come from K and M only; the damping is then
projected onto them, Ĉ = T<sup>T</sup>CT, and assembled like K̂. K̂ is block diagonal
because Ψ is a static solution for K. That is not true for C, so Ĉ in general couples the
kept modes to each other and to the boundary. Only damping proportional to stiffness inside a
component (C = βK) keeps Ĉ block diagonal. In practice the modal block is often replaced by
measured or assumed modal damping, Ĉ<sub>qq</sub> = diag(2ζ<sub>r</sub>ω<sub>r</sub>),
with joint damping on the boundary. Step 9 of the Matrices tab compares the reduced
model's exact damping ratios with the true ones.</li>
<li><b>Interface size.</b> Every boundary DOF stays in the model. On this chain an
interface is one DOF; on a 3D finite-element model it can be thousands, which is why
interface reduction methods exist.</li>
</ul>

<h3>Try it</h3>
<ol>
<li>Set <i>Number of masses</i> to 8, the interface at m4, and keep 1 mode in each
substructure. The reduced model has 4 DOF (q<sub>A1</sub>, q<sub>B1</sub>, x<sub>4</sub>,
x<sub>8</sub>) instead of 8. Mode 1 is within 0.2%, mode 4 is 7% high with MAC 0.84, and
modes 5 to 8 are not in the reduced model at all.</li>
<li>Scroll down to <i>Substructures on their own</i>. With the interface at m4, A and B are
identical three-mass pieces, so each has a clamped mode at 2.44 Hz. Coupled, the two split
into modes 3 (2.84 Hz) and 4 (3.84 Hz), each carrying about half its energy: two equal
oscillators joined together always split this way. Move the interface to m7: A's three lowest
modes each become mostly one coupled mode (shares of 74–84%), while the higher ones mix more.</li>
<li>Open the <i>Basis</i> tab and change the modes kept: each kept mode adds one shape (one
column of T), and the constraint-mode shapes stay the same. Every result on this page is a
combination of the shapes shown there.</li>
<li>Click <i>Guyan (0 modes)</i>. Only the two boundary DOFs are left: mode 1 is still
within 2%, mode 2 is 11% high. On the <i>Modes &amp; FRF</i> tab the reduced FRF matches at
low frequency and drifts above mode 1.</li>
<li>Click <i>All modes (exact)</i>. Every error drops to zero, although the matrices look
nothing like the original M and K: same system, different coordinates.</li>
<li>Keep 2 modes in each substructure (6 DOFs). Every CB frequency stays above the true
one. The kept fixed-interface modes reach 4.5 Hz (Matrices tab, step 3), and modes 1 to 3
(up to 2.8 Hz) are now within 0.03%; mode 6 is still 2.7% high.</li>
<li>Move the interface to m7. B shrinks to the last spring and has no interior, so all
the reduction happens in A. With 1 mode kept in A the model has 3 DOFs and mode 3 is 50%
too high (MAC 0.27). Keep 3 modes in A (up to 3.97 Hz) and modes 1 to 4 fall within
0.2%: the rule of thumb in action.</li>
</ol>
<p><b>Free interface</b> (back to 8 masses, the interface at m4, 1 mode each):</p>
<ol start="8">
<li>Choose <i>Rubin</i>. The model is the same size (4 DOFs), but B's one kept mode is now its
rigid-body mode. Mode 1 is within 0.003% (Craig–Bampton: 0.02%) and mode 2 within 0.9%, but mode
4 is 10% high against Craig–Bampton's 7%. Open <i>Substructures on their own</i>: B1 is at 0 Hz and
carries 83% of coupled mode 1's kinetic energy.</li>
<li>Open the <i>Basis</i> tab. The boundary columns are no longer the straight-line "tent" and
ramp: they are residual attachment modes, the static deflection of the discarded modes, which
already bend like the higher modes.</li>
<li>Choose <i>MacNeal</i>. Still 4 coordinates, but the boundary DOFs are massless, so there are
only 2 modes: mode 1 is 0.6% high and mode 2 22%. Click <i>All modes</i>: 6 modes now, still not
exact (mode 6 is 66% high), because the mass of B's discarded modes is gone.</li>
<li>Open <i>Compare methods</i> and select mode 3 in the table on the left. Every curve falls as
modes are added. Craig–Bampton and Rubin reach the exact answer with all modes (drawn at the
floor); MacNeal is still 0.8% high. For mode 3 Craig–Bampton and Rubin take turns in the lead; select
mode 1 and Rubin is ahead at every size.</li>
<li>Set k<sub>1</sub> = k<sub>2</sub> = 0 on the Simulation page, with the interface at m4.
Craig–Bampton fails (m1 and m2 float when the boundary is held), but the free-interface methods
simply find more rigid-body modes in A.</li>
</ol>

<h3><a name="notation"></a>Notation</h3>
<table border="1" cellspacing="0" cellpadding="4">
<tr><th>Symbol</th><th>Meaning</th><th>Units</th></tr>
<tr><td>N</td><td>number of masses (physical DOFs) in the full chain</td><td>—</td></tr>
<tr><td>m<sub>j</sub>, k<sub>j</sub>, c<sub>j</sub></td><td>mass j; spring and damper j (joining mass j−1, or the ground, to mass j)</td><td>kg, N/m, N·s/m</td></tr>
<tr><td>x, ẋ, ẍ</td><td>physical displacements, velocities, accelerations</td><td>m, m/s, m/s²</td></tr>
<tr><td>f</td><td>applied force vector (here only at the tip)</td><td>N</td></tr>
<tr><td>M, C, K</td><td>mass, damping and stiffness matrices of the full chain</td><td>kg, N·s/m, N/m</td></tr>
<tr><td>A, B; (s)</td><td>the substructures; superscript (s) marks a quantity of substructure s, e.g. K<sup>(A)</sup></td><td>—</td></tr>
<tr><td>L<sub>s</sub></td><td>Boolean localization matrix: picks substructure s's DOFs out of the global x</td><td>—</td></tr>
<tr><td>i, b</td><td>interior DOFs (inside one substructure) and boundary / master DOFs (interface and tip)</td><td>—</td></tr>
<tr><td>x<sub>i</sub>, x<sub>b</sub></td><td>interior and boundary displacements</td><td>m</td></tr>
<tr><td>K<sub>ii</sub>, K<sub>ib</sub>, K<sub>bi</sub>, K<sub>bb</sub></td><td>partitions (blocks) of a substructure matrix; same for M and C</td><td>as K</td></tr>
<tr><td>n<sub>i</sub>, n<sub>b</sub></td><td>number of interior / boundary DOFs</td><td>—</td></tr>
<tr><td>V, v<sub>j</sub>, c</td><td>a general basis (matrix of shapes), one shape, and the coordinates (amplitudes) in that basis</td><td>—</td></tr>
<tr><td>φ<sub>r</sub>, Φ</td><td>fixed-interface normal mode r of a substructure (boundary held) and the matrix of all of them, mass-normalized: Φ<sup>T</sup>M<sub>ii</sub>Φ = I</td><td>1/√kg</td></tr>
<tr><td>Φ<sub>k</sub>, k</td><td>the kept (lowest) fixed-interface modes, and how many are kept</td><td>1/√kg, —</td></tr>
<tr><td>ω<sub>r</sub>, Λ<sub>k</sub></td><td>natural frequency of fixed-interface mode r; Λ<sub>k</sub> = diag(ω<sub>r</sub>²) of the kept ones</td><td>rad/s, rad²/s²</td></tr>
<tr><td>Ψ</td><td>constraint modes: Ψ = −K<sub>ii</sub><sup>−1</sup>K<sub>ib</sub>, static interior shape for a unit displacement of each boundary DOF</td><td>m/m (dimensionless)</td></tr>
<tr><td>q</td><td>modal coordinates: amplitudes of the kept fixed-interface modes</td><td>m·√kg</td></tr>
<tr><td>T</td><td>Craig–Bampton transformation (the basis): [x<sub>i</sub>; x<sub>b</sub>] = T [q; x<sub>b</sub>], T = [ Φ<sub>k</sub> Ψ ; 0 I ]</td><td>mixed</td></tr>
<tr><td>M̂, Ĉ, K̂, f̂</td><td>reduced matrices and force: T<sup>T</sup>MT, T<sup>T</sup>CT, T<sup>T</sup>KT, T<sup>T</sup>f</td><td>mixed</td></tr>
<tr><td>K̂<sub>bb</sub></td><td>boundary stiffness K<sub>bb</sub> − K<sub>bi</sub>K<sub>ii</sub><sup>−1</sup>K<sub>ib</sub> (Schur complement = Guyan stiffness)</td><td>N/m</td></tr>
<tr><td>M̂<sub>qb</sub></td><td>inertial coupling between the kept modes and the boundary motion</td><td>√kg</td></tr>
<tr><td>η</td><td>mode shape of the reduced model in coordinates [q, x<sub>b</sub>]; x = Tη recovers it physically</td><td>mixed</td></tr>
<tr><td>f<sub>n</sub>, f<sub>d</sub></td><td>undamped natural frequency ω<sub>n</sub>/2π, damped frequency ω<sub>d</sub>/2π</td><td>Hz</td></tr>
<tr><td>λ</td><td>damped eigenvalue −ζω<sub>n</sub> ± iω<sub>d</sub></td><td>1/s</td></tr>
<tr><td>ζ</td><td>damping ratio: −Re λ / |λ| ("alone": of a substructure with its boundary held, or free for Rubin and MacNeal)</td><td>—</td></tr>
<tr><td>MAC</td><td>modal assurance criterion (φ<sup>T</sup>x)² / (φ<sup>T</sup>φ · x<sup>T</sup>x): shape similarity, 0 to 1</td><td>—</td></tr>
<tr><td>Share</td><td>fraction of a coupled mode's strain energy carried by one fixed-interface mode (Craig–Bampton), or of its kinetic energy carried by one free-interface mode (Rubin, MacNeal)</td><td>—</td></tr>
<tr><td>H(ω)</td><td>receptance (FRF) X/F = (K − ω²M + iωC)<sup>−1</sup></td><td>m/N</td></tr>
<tr><td>Φ (free), Φ<sub>ik</sub>, Φ<sub>bk</sub></td><td>free-interface modes of a whole substructure (boundary free), mass-normalized: Φ<sup>T</sup>MΦ = I; the interior and boundary rows of the kept ones</td><td>1/√kg</td></tr>
<tr><td>Φ<sub>d</sub>, Λ<sub>d</sub></td><td>the discarded free-interface modes and their ω²</td><td>1/√kg, rad²/s²</td></tr>
<tr><td>G<sub>d</sub>, G<sub>bb</sub>, G<sub>ib</sub></td><td>residual flexibility Φ<sub>d</sub>Λ<sub>d</sub><sup>−1</sup>Φ<sub>d</sub><sup>T</sup> and its partitions</td><td>m/N</td></tr>
<tr><td>f<sub>b</sub></td><td>interface (connection) forces on a substructure's boundary DOFs</td><td>N</td></tr>
<tr><td>R</td><td>residual attachment modes G<sub>ib</sub>G<sub>bb</sub><sup>−1</sup>: interior shape for a unit displacement of each boundary DOF (free-interface methods)</td><td>m/m</td></tr>
<tr><td>E<sub>b</sub></td><td>Boolean matrix picking the boundary columns</td><td>—</td></tr>
<tr><td>CB</td><td>Craig–Bampton (the reduced model); "true" = the full N-DOF model</td><td>—</td></tr>
</table>
"""


def fmt(v: float, scale: float) -> str:
    """Four significant figures; entries negligible relative to the matrix are shown as 0."""
    if abs(v) <= 1e-10 * scale:
        return "0"
    return f"{v:.4g}"


def matrix_html(
    A: np.ndarray,
    rows: list[str],
    cols: list[str],
    row_groups: list[str],
    col_groups: list[str],
    title: str = "",
    dim_cols: int | None = None,
) -> str:
    """An HTML table with labeled rows/columns and cells tinted by partition block.

    Columns from dim_cols on are greyed out (e.g. discarded modes).
    """
    scale = float(np.abs(A).max()) if A.size else 1.0
    head = "".join(f"<th>{c}</th>" for c in cols)
    out = [f"<table border='1' cellspacing='0' cellpadding='3'><tr><th>{title}</th>{head}</tr>"]
    for r, (label, rg) in enumerate(zip(rows, row_groups)):
        cells = []
        for c, cg in enumerate(col_groups):
            dim = dim_cols is not None and c >= dim_cols
            tint = "#f4f4f4" if dim else BLOCK_TINTS[tuple(sorted((rg, cg)))]
            text = fmt(A[r, c], scale)
            color = " style='color:#aaa'" if text == "0" or dim else ""
            cells.append(f"<td align='right' bgcolor='{tint}'{color}>{text}</td>")
        out.append(f"<tr><th>{label}</th>{''.join(cells)}</tr>")
    out.append("</table>")
    return "".join(out)


def _dof_labels(dofs: np.ndarray) -> list[str]:
    return [f"x{d + 1}" for d in dofs]


def _local(sub: Substructure, mat: np.ndarray, title: str) -> str:
    groups = ["i"] * sub.ni + ["b"] * sub.nb
    labels = _dof_labels(sub.dofs)
    return matrix_html(mat, labels, labels, groups, groups, title)


def _reduced_labels(sub: Substructure) -> tuple[list[str], list[str]]:
    labels = [f"q<sub>{sub.name}{k + 1}</sub>" for k in range(sub.n_kept)] + _dof_labels(sub.boundary)
    return labels, ["q"] * sub.n_kept + ["b"] * sub.nb


def _side_by_side(*tables: str) -> str:
    cells = "".join(f"<td valign='top' style='padding-right:14px'>{t}</td>" for t in tables)
    return f"<table cellspacing='0' cellpadding='0'><tr>{cells}</tr></table>"


def _join(items: list[str]) -> str:
    return ", ".join(items) if items else "none"


def matrices_html(model: CMSModel, full: ModalResult | None = None) -> str:
    """Every step of the reduction, with the current numbers."""
    system = model.system
    n = system.n
    M, C, K = system.matrices()
    subs = model.substructures
    bnd = {int(b) for b in model.boundary}
    groups = ["b" if d in bnd else "i" for d in range(n)]
    labels = _dof_labels(np.arange(n))
    interfaces = [int(b) for b in model.boundary[:-1]]
    parts = [LEGEND_HTML]

    parts.append("<h3>1. The full model</h3>")
    parts.append(
        f"<p>N = {n} physical DOFs. Boundary (master) DOFs: <b>{_join(_dof_labels(model.boundary))}</b> "
        f"(the interface and the loaded tip). Interior: {_join([l for l, g in zip(labels, groups) if g == 'i'])}. "
        "K [N/m], C [N·s/m] and M [kg]. C is assembled from the dampers exactly like K from the "
        "springs, so it has the same tridiagonal pattern.</p>"
    )
    parts.append(_side_by_side(
        matrix_html(K, labels, labels, groups, groups, "K"),
        matrix_html(C, labels, labels, groups, groups, "C"),
        matrix_html(M, labels, labels, groups, groups, "M"),
    ))

    parts.append("<h3>2. Cut into substructures</h3>")
    for sub in subs:
        springs = ", ".join(f"k<sub>{e + 1}</sub>, c<sub>{e + 1}</sub>" for e in sub.elements)
        parts.append(
            f"<p><b>Substructure {sub.name}</b>: springs and dampers {springs}; interior "
            f"{_join(_dof_labels(sub.interior))}; boundary {_join(_dof_labels(sub.boundary))}. "
            "Reordered [interior | boundary]:</p>"
        )
        parts.append(_side_by_side(
            _local(sub, sub.K, f"K<sup>({sub.name})</sup>"),
            _local(sub, sub.C, f"C<sup>({sub.name})</sup>"),
            _local(sub, sub.M, f"M<sup>({sub.name})</sup>"),
        ))
    for j in interfaces:
        left = next(s for s in subs if j in s.boundary and j + 1 not in s.dofs)
        right = next(s for s in subs if j in s.boundary and s is not left)
        parts.append(
            f"<p>The interface entry of the full model, K<sub>{j + 1},{j + 1}</sub> = k<sub>{j + 1}</sub> + "
            f"k<sub>{j + 2}</sub> = {fmt(K[j, j], 1)}, is split: {fmt(system.stiffness[j], 1)} goes to "
            f"{left.name} and {fmt(system.stiffness[j + 1], 1)} to {right.name}. C<sub>{j + 1},{j + 1}</sub> "
            f"= c<sub>{j + 1}</sub> + c<sub>{j + 2}</sub> = {fmt(C[j, j], 1)} splits the same way "
            f"({fmt(system.damping[j], 1)} + {fmt(system.damping[j + 1], 1)}). The interface mass "
            f"m<sub>{j + 1}</sub> = {fmt(system.masses[j], 1)} "
            + (f"is split half and half, {fmt(system.masses[j] / 2, 1)} to each, so that both have mass "
               "at every DOF their free modes can move (any split works: assembly adds them back). "
               if model.free else
               f"goes to {left.name} (any split works: assembly adds them back). ")
            + "Every other entry belongs to one substructure only, so "
            "K = Σ L<sub>s</sub><sup>T</sup>K<sup>(s)</sup>L<sub>s</sub> exactly, and the same for C and M.</p>"
        )
    if model.free:
        parts += _free_steps(model)
    else:
        parts += _fixed_steps(model)
    parts += _assembly_steps(model, full)
    return "".join(parts)


def _fixed_steps(model: CMSModel) -> list[str]:
    """Craig-Bampton steps 3 to 6: fixed-interface modes, constraint modes, T, reduced matrices."""
    subs = model.substructures
    parts = []

    parts.append("<h3>3. Fixed-interface normal modes</h3>")
    parts.append("<p>Boundary clamped (x<sub>b</sub> = 0): K<sub>ii</sub>φ = ω²M<sub>ii</sub>φ, "
                 "mass-normalized so Φ<sup>T</sup>M<sub>ii</sub>Φ = I. These are <i>undamped</i> modes: "
                 "C plays no part in choosing the basis (steps 3 to 5). It is only projected onto that "
                 "basis in step 6.</p>")
    for sub in subs:
        if not sub.ni:
            parts.append(f"<p><b>{sub.name}</b> has no interior DOFs, so it has no fixed-interface modes: "
                         "it is represented by its boundary DOFs alone.</p>")
            continue
        freqs = ", ".join(
            (f"<b>{f:.4g}</b>" if r < sub.n_kept else f"<span style='color:#999'>{f:.4g}</span>")
            for r, f in enumerate(sub.omegas / TWO_PI)
        )
        parts.append(
            f"<p><b>{sub.name}</b>: f = {freqs} Hz (<b>bold</b>: kept, {sub.n_kept} of {sub.ni}; "
            "grey: discarded).</p>"
        )
        cols = [f"φ<sub>{r + 1}</sub>" for r in range(sub.ni)]
        parts.append(matrix_html(sub.Phi, _dof_labels(sub.interior), cols, ["i"] * sub.ni, ["q"] * sub.ni,
                                 f"Φ<sup>({sub.name})</sup>", dim_cols=sub.n_kept))

    parts.append("<h3>4. Constraint modes</h3>")
    parts.append("<p>Ψ = −K<sub>ii</sub><sup>−1</sup>K<sub>ib</sub>: column j is the static shape of the "
                 "interior when boundary DOF j moves 1 m and the other boundary DOFs are held. Along a "
                 "chain the interior simply interpolates between its boundary masses, weighted by the "
                 "spring flexibilities.</p>")
    tables = []
    for sub in subs:
        if sub.ni:
            tables.append(matrix_html(sub.Psi, _dof_labels(sub.interior), _dof_labels(sub.boundary),
                                      ["i"] * sub.ni, ["b"] * sub.nb, f"Ψ<sup>({sub.name})</sup>"))
    parts.append(_side_by_side(*tables) if tables else "<p>No interior DOFs.</p>")

    parts.append("<h3>5. Transformation</h3>")
    parts.append("<p>[x<sub>i</sub>; x<sub>b</sub>] = T [q; x<sub>b</sub>] with "
                 "T = [ Φ<sub>k</sub> Ψ ; 0 I ]. Only the kept modes appear, so T is tall: that is "
                 "the reduction.</p>")
    tables = []
    for sub in subs:
        red_labels, red_groups = _reduced_labels(sub)
        tables.append(matrix_html(sub.T, _dof_labels(sub.dofs), red_labels, ["i"] * sub.ni + ["b"] * sub.nb,
                                  red_groups, f"T<sup>({sub.name})</sup>"))
    parts.append(_side_by_side(*tables))

    parts.append("<h3>6. Reduced substructure matrices</h3>")
    parts.append("<p>K̂ = T<sup>T</sup>KT is block diagonal: the kept ω² (in rad²/s²) for q, and the "
                 "boundary stiffness K̂<sub>bb</sub> = K<sub>bb</sub> − K<sub>bi</sub>K<sub>ii</sub><sup>−1</sup>"
                 "K<sub>ib</sub>. M̂ = T<sup>T</sup>MT has I for q and the coupling M̂<sub>qb</sub>. "
                 "The damping is projected onto the same basis, Ĉ = T<sup>T</sup>CT:</p>"
                 "<p>&nbsp;&nbsp;Ĉ<sub>qq</sub> = Φ<sub>k</sub><sup>T</sup>C<sub>ii</sub>Φ<sub>k</sub>, &nbsp; "
                 "Ĉ<sub>qb</sub> = Φ<sub>k</sub><sup>T</sup>(C<sub>ii</sub>Ψ + C<sub>ib</sub>), &nbsp; "
                 "Ĉ<sub>bb</sub> = C<sub>bb</sub> + C<sub>bi</sub>Ψ + Ψ<sup>T</sup>C<sub>ib</sub> + "
                 "Ψ<sup>T</sup>C<sub>ii</sub>Ψ</p>"
                 "<p>Unlike K̂, Ĉ is <b>not</b> block diagonal in general. Ψ is a static solution for K, "
                 "not for C, so Ĉ<sub>qb</sub> ≠ 0, and the undamped Φ<sub>k</sub> only diagonalize "
                 "Ĉ<sub>qq</sub> when the damping is proportional. The exception is damping proportional "
                 "to stiffness inside the substructure (C<sup>(s)</sup> = βK<sup>(s)</sup>, e.g. every "
                 "c<sub>i</sub>/k<sub>i</sub> equal): then Ĉ = βK̂ and it inherits K̂'s block-diagonal form. "
                 "The diagonal of Ĉ<sub>qq</sub> gives each kept fixed-interface mode a damping ratio "
                 "ζ<sub>r</sub> = Ĉ<sub>rr</sub> / 2ω<sub>r</sub>.</p>")
    for sub in subs:
        red_labels, red_groups = _reduced_labels(sub)
        parts.append(_side_by_side(
            matrix_html(sub.K_red, red_labels, red_labels, red_groups, red_groups, f"K̂<sup>({sub.name})</sup>"),
            matrix_html(sub.C_red, red_labels, red_labels, red_groups, red_groups, f"Ĉ<sup>({sub.name})</sup>"),
            matrix_html(sub.M_red, red_labels, red_labels, red_groups, red_groups, f"M̂<sup>({sub.name})</sup>"),
        ))
        parts.append(_damping_note(sub))
    return parts


def _free_steps(model: CMSModel) -> list[str]:
    """Rubin / MacNeal steps 3 to 6: free-interface modes, residual flexibility, T, reduced matrices."""
    subs = model.substructures
    parts = ["<h3>3. Free-interface normal modes</h3>",
             "<p>Boundary free: Kφ = ω²Mφ over <i>all</i> the substructure's DOFs, mass-normalized so "
             "Φ<sup>T</sup>MΦ = I. Every mode moves the boundary too. Rigid-body modes (ω = 0) come first "
             "and are always kept. These are <i>undamped</i> modes: C plays no part in choosing the "
             "basis.</p>"]
    for sub in subs:
        if not sub.ni:
            parts.append(f"<p><b>{sub.name}</b> has no interior DOFs, so there is nothing to reduce: it is "
                         "kept as its physical boundary DOFs.</p>")
            continue
        freqs = ", ".join(
            (f"<b>{f:.4g}</b>" if r < sub.n_kept else f"<span style='color:#999'>{f:.4g}</span>")
            for r, f in enumerate(np.where(sub.omegas > 1e-9, sub.omegas, 0.0) / TWO_PI)
        )
        rigid = (f" {sub.n_rigid} rigid-body mode{'s' if sub.n_rigid > 1 else ''} at 0 Hz." if sub.n_rigid
                 else "")
        parts.append(f"<p><b>{sub.name}</b>: f = {freqs} Hz (<b>bold</b>: kept, {sub.n_kept} of "
                     f"{sub.omegas.size}; grey: discarded).{rigid}</p>")
        cols = [f"φ<sub>{r + 1}</sub>" for r in range(sub.omegas.size)]
        parts.append(matrix_html(sub.Phi, _dof_labels(sub.dofs), cols, ["i"] * sub.ni + ["b"] * sub.nb,
                                 ["q"] * sub.omegas.size, f"Φ<sup>({sub.name})</sup>", dim_cols=sub.n_kept))

    parts.append("<h3>4. Residual flexibility</h3>")
    parts.append("<p>G<sub>d</sub> = Φ<sub>d</sub>Λ<sub>d</sub><sup>−1</sup>Φ<sub>d</sub><sup>T</sup> "
                 "[m/N]: the static flexibility of the <i>discarded</i> modes, which respond almost "
                 "statically to the interface forces because they are high in frequency. Without "
                 "rigid-body modes it equals K<sup>−1</sup> − Φ<sub>k</sub>Λ<sub>k</sub><sup>−1</sup>"
                 "Φ<sub>k</sub><sup>T</sup>, so in practice the discarded modes are never computed. "
                 "Only its boundary columns are used: G<sub>d</sub>E<sub>b</sub> are the residual "
                 "attachment modes. Scaled to a unit boundary displacement, "
                 "R = G<sub>ib</sub>G<sub>bb</sub><sup>−1</sup>.</p>")
    for sub in subs:
        if not sub.ni:
            continue
        groups = ["i"] * sub.ni + ["b"] * sub.nb
        labels = _dof_labels(sub.dofs)
        parts.append(_side_by_side(
            matrix_html(sub.G, labels, labels, groups, groups, f"G<sub>d</sub><sup>({sub.name})</sup>"),
            matrix_html(sub.Psi, _dof_labels(sub.interior), _dof_labels(sub.boundary), ["i"] * sub.ni,
                        ["b"] * sub.nb, f"R<sup>({sub.name})</sup>"),
        ))

    parts.append("<h3>5. Transformation</h3>")
    parts.append("<p>x = Φ<sub>k</sub>q + G<sub>d</sub>E<sub>b</sub>f<sub>b</sub>. Its boundary rows "
                 "give f<sub>b</sub> = G<sub>bb</sub><sup>−1</sup>(x<sub>b</sub> − Φ<sub>bk</sub>q), and "
                 "substituting: [x<sub>i</sub>; x<sub>b</sub>] = T [q; x<sub>b</sub>] with "
                 "T = [ Φ<sub>ik</sub> − RΦ<sub>bk</sub> &nbsp;R ; 0 &nbsp;I ]. The same shape as "
                 "Craig–Bampton's T, so x<sub>b</sub> is physical and assembly is unchanged. The modal "
                 "columns are the kept free modes with their boundary motion taken out.</p>")
    tables = []
    for sub in subs:
        red_labels, red_groups = _reduced_labels(sub)
        tables.append(matrix_html(sub.T, _dof_labels(sub.dofs), red_labels, ["i"] * sub.ni + ["b"] * sub.nb,
                                  red_groups, f"T<sup>({sub.name})</sup>"))
    parts.append(_side_by_side(*tables))

    parts.append("<h3>6. Reduced substructure matrices</h3>")
    parts.append("<p>K̂ = T<sup>T</sup>KT = [ Λ<sub>k</sub> + Φ<sub>bk</sub><sup>T</sup>G<sub>bb</sub><sup>−1</sup>"
                 "Φ<sub>bk</sub> &nbsp;−Φ<sub>bk</sub><sup>T</sup>G<sub>bb</sub><sup>−1</sup> ; "
                 "−G<sub>bb</sub><sup>−1</sup>Φ<sub>bk</sub> &nbsp;G<sub>bb</sub><sup>−1</sup> ]: unlike "
                 "Craig–Bampton, the modes are coupled to the boundary by stiffness. G<sub>bb</sub><sup>−1</sup> "
                 "is the residual stiffness at the boundary.</p>")
    if model.method == "rubin":
        parts.append("<p><b>Rubin</b>: M̂ = T<sup>T</sup>MT, a Rayleigh–Ritz projection. The residual "
                     "attachment modes carry mass (the <i>residual mass</i>), so the modes are coupled to "
                     "the boundary by mass as well. The damping is projected the same way, "
                     "Ĉ = T<sup>T</sup>CT.</p>")
    else:
        parts.append("<p><b>MacNeal</b>: the residual flexibility is a massless spring. Only the kept "
                     "modes carry inertia: M̂ = [ I 0 ; 0 0 ], so the boundary rows of M̂ are zero. The "
                     "damping is projected like the stiffness, Ĉ = T<sup>T</sup>CT (damping was not part "
                     "of MacNeal's method; this way it differs from Rubin's only in the mass).</p>")
    for sub in subs:
        red_labels, red_groups = _reduced_labels(sub)
        parts.append(_side_by_side(
            matrix_html(sub.K_red, red_labels, red_labels, red_groups, red_groups, f"K̂<sup>({sub.name})</sup>"),
            matrix_html(sub.C_red, red_labels, red_labels, red_groups, red_groups, f"Ĉ<sup>({sub.name})</sup>"),
            matrix_html(sub.M_red, red_labels, red_labels, red_groups, red_groups, f"M̂<sup>({sub.name})</sup>"),
        ))
        if sub.ni and sub.n_kept:
            Pk = sub.Phi[:, : sub.n_kept]
            modal = np.diag(Pk.T @ sub.C @ Pk)
            zetas = ", ".join("rigid" if w < 1e-9 else f"{c / (2 * w):.4f}"
                              for c, w in zip(modal, sub.omegas[: sub.n_kept]))
            parts.append(f"<p><b>{sub.name}</b>: kept free-interface ζ = φ<sub>r</sub><sup>T</sup>Cφ<sub>r</sub>"
                         f" / 2ω<sub>r</sub> = {zetas}.</p>")
    return parts


def _assembly_steps(model: CMSModel, full: ModalResult | None) -> list[str]:
    """Steps 7 to 9, the same for every method: assemble, solve and recover, damping."""
    system = model.system
    n = system.n
    subs = model.substructures
    bnd = {int(b) for b in model.boundary}
    groups = ["b" if d in bnd else "i" for d in range(n)]
    labels = _dof_labels(np.arange(n))
    interfaces = [int(b) for b in model.boundary[:-1]]
    parts = []
    parts.append("<h3>7. Assemble the reduced model</h3>")
    shared = _join(_dof_labels(np.array(interfaces)))
    parts.append(
        f"<p>Coordinates [q, x<sub>b</sub>]: {model.n_red} DOFs instead of {n} "
        f"({model.n_modal} modal + {model.boundary.size} boundary). The interface DOF {shared} is shared, "
        "so the A and B entries on its row and column add, just like element matrices in step 1. "
        "The q of different substructures never touch.</p>"
    )
    glabels = [
        f"q<sub>{l[2:]}</sub>" if l.startswith("q_") else l for l in model.labels
    ]
    ggroups = ["q" if l.startswith("q_") else "b" for l in model.labels]
    for j in interfaces:
        g = model.labels.index(f"x{j + 1}")
        pieces = []
        for attr, sym, total_mat in (("K_red", "K̂", model.K), ("C_red", "Ĉ", model.C), ("M_red", "M̂", model.M)):
            terms = []
            for sub in subs:
                if j in sub.boundary:
                    loc = sub.n_kept + int(np.searchsorted(sub.boundary, j))
                    terms.append(fmt(getattr(sub, attr)[loc, loc], 1) + f" ({sub.name})")
            total = total_mat[g, g]
            pieces.append(f"{sym}<sub>x{j + 1},x{j + 1}</sub> = {' + '.join(terms)} = {fmt(total, 1)}")
        parts.append(f"<p>At the interface: {'; '.join(pieces)}.</p>")
    parts.append(_side_by_side(
        matrix_html(model.K, glabels, glabels, ggroups, ggroups, "K̂"),
        matrix_html(model.C, glabels, glabels, ggroups, ggroups, "Ĉ"),
        matrix_html(model.M, glabels, glabels, ggroups, ggroups, "M̂"),
    ))

    parts.append("<h3>8. Solve and recover</h3>")
    parts.append("<p>K̂η = ω²M̂η gives the reduced model's modes; the physical shapes are x = Tη, with "
                 "the global T below (rows: physical DOFs, columns: reduced coordinates). The comparison "
                 "table and the Modes &amp; FRF tab compare them with the full model.</p>")
    if model.method == "macneal":
        parts.append("<p>MacNeal's M̂ is zero on the boundary rows, so the boundary DOFs have no inertia: "
                     "they follow the modes statically. They are condensed out first, "
                     "K̂<sub>c</sub> = K̂<sub>qq</sub> − K̂<sub>qb</sub>K̂<sub>bb</sub><sup>−1</sup>K̂<sub>bq</sub>, "
                     f"which leaves {model.omegas.size} modes, and recovered as "
                     "x<sub>b</sub> = −K̂<sub>bb</sub><sup>−1</sup>K̂<sub>bq</sub>q.</p>")
    parts.append(matrix_html(model.T, labels, glabels, groups, ggroups, "T"))
    freqs = ", ".join(f"{f:.4g}" for f in model.fn_hz)
    parts.append(f"<p>Reduced-model natural frequencies: {freqs} Hz.</p>")
    parts.append(_damped_comparison(model, full))
    return parts


def _damping_note(sub: Substructure) -> str:
    if not sub.n_kept:
        return (f"<p><b>{sub.name}</b> keeps no modes, so Ĉ<sup>({sub.name})</sup> is just the damping "
                "seen from the boundary (Ĉ<sub>bb</sub>).</p>")
    modal, boundary = substructure_damping(sub)
    zetas = ", ".join(
        f"{sub.C_red[r, r] / (2 * w):.4f}" for r, w in enumerate(sub.omegas[: sub.n_kept])
    )
    if max(modal, boundary) < 1e-6:
        verdict = ("Ĉ has the same block-diagonal form as K̂: the damping in this substructure is "
                   "proportional to its stiffness, so it couples nothing.")
    else:
        found = [f"{name} {v:.2f}" for name, v in (("between kept modes", modal), ("mode–boundary", boundary))
                 if v >= 1e-6]
        verdict = (f"Ĉ couples what K̂ keeps apart. Largest relative coupling {', '.join(found)} "
                   "(0 = none, 1 = as large as the diagonal terms).")
    return f"<p><b>{sub.name}</b>: fixed-interface ζ = {zetas}. {verdict}</p>"


def _damped_comparison(model: CMSModel, full: ModalResult | None) -> str:
    """Step 9: exact damping ratios of the reduced and the full model, mode by mode."""
    out = ["<h3>9. Damping in the reduced model</h3>",
           "<p>The frequencies above ignore damping. The reduced model does include it: Ĉ is used "
           "for the FRF on the Modes &amp; FRF tab, and the damped eigenvalues of "
           "M̂η̈ + Ĉη̇ + K̂η = 0 give its damping ratios. Both columns are exact (state-space) "
           "values, so any difference comes from the reduction alone. The comparison table on the left shows the same two columns.</p>"]
    if full is None:
        return "".join(out)
    rows = ["<table border='1' cellspacing='0' cellpadding='3'>"
            f"<tr><th>Mode</th><th>ζ true</th><th>ζ {model.short_name}</th><th>ζ error</th></tr>"]
    for c in compare_modes(model, full):
        cells = [str(c.index), "overdamped" if c.zeta_true is None else f"{c.zeta_true:.4f}"]
        if c.fn_red is None:
            cells += ["<span style='color:#999'>not in model</span>", "—"]
        elif c.zeta_red is None:
            cells += ["overdamped", "—"]
        else:
            err = c.zeta_error
            color = "#000" if err is None else "#2a7d2a" if abs(err) < 1e-3 else "#b07000" if abs(err) < 0.05 else "#c1121f"
            cells += [f"{c.zeta_red:.4f}", "—" if err is None else f"<span style='color:{color}'>{100 * err:+.3g}%</span>"]
        rows.append("<tr>" + "".join(f"<td align='right'>{x}</td>" for x in cells) + "</tr>")
    rows.append("</table>")
    out.append("".join(rows))
    out.append("<p>With stiffness-proportional damping (the default), ζ<sub>r</sub> = βω<sub>r</sub>/2, "
               "so ζ CB errs exactly as much as the frequency does. With non-proportional damping "
               "(try c<sub>1</sub> = 15 on the Simulation page), the coupling terms of Ĉ matter, and "
               "truncating modes also loses the damping they carried: the higher modes' ζ can be off "
               "by much more than their frequency.</p>")
    return "".join(out)
