"""Background notes on modal analysis and damping, shown in their own tab."""

from __future__ import annotations

from PySide6 import QtWidgets

COUPLING_TIP = (
    "<p><b>Coupling index</b> = the largest off-diagonal term of the modal damping "
    "matrix C<sub>m</sub> = Φ<sup>T</sup>CΦ, relative to its diagonal:</p>"
    "<p>&nbsp;&nbsp;max<sub>r≠s</sub> |C<sub>m,rs</sub>| / √(C<sub>m,rr</sub> C<sub>m,ss</sub>)</p>"
    "<p>0 means the damping does not couple the undamped modes (proportional damping). "
    "Values near 1 mean the damping force from one mode's motion drives another mode "
    "about as strongly as it damps its own. How much this matters also depends on how "
    "close the modes' frequencies are. See the Background tab.</p>"
)

BACKGROUND_HTML = """
<h3>Equations of motion</h3>
<p>The chain obeys <b>M ẍ + C ẋ + K x = f(t)</b>: M is diagonal (the masses);
K and C are tridiagonal (each spring and damper links two neighbouring masses, or a mass
and the ground).</p>

<h3>Undamped normal modes (real)</h3>
<p>Ignoring C, free vibration x = φ cos(ωt) requires
<b>Kφ = ω²Mφ</b>. The N solutions are the natural frequencies
ω<sub>r</sub> and <b>real</b> mode shapes φ<sub>r</sub>: every mass moves exactly
in phase or exactly out of phase with the others. They all pass through zero at the same
instant, like a standing wave.</p>
<p>Collecting the shapes (mass-normalized) into Φ and writing x = Φq turns the
equations into <i>modal coordinates</i> q:</p>
<p>&nbsp;&nbsp;Φ<sup>T</sup>MΦ = I, &nbsp;&nbsp; Φ<sup>T</sup>KΦ =
diag(ω<sub>r</sub>²), &nbsp;&nbsp; Φ<sup>T</sup>CΦ = C<sub>m</sub></p>
<p>M and K always become diagonal. <b>C<sub>m</sub> in general does not.</b></p>

<h3>Proportional damping</h3>
<p>If C<sub>m</sub> <i>is</i> diagonal, every mode obeys its own independent single-DOF
equation</p>
<p>&nbsp;&nbsp;q̈<sub>r</sub> + 2ζ<sub>r</sub>ω<sub>r</sub>q̇<sub>r</sub>
+ ω<sub>r</sub>²q<sub>r</sub> = φ<sub>r</sub><sup>T</sup>f, &nbsp;&nbsp; with
ζ<sub>r</sub> = C<sub>m,rr</sub> / 2ω<sub>r</sub></p>
<p>This happens for Rayleigh damping C = αM + βK. In this app, the simplest case
is every damper being the same multiple of its spring (all c<sub>i</sub>/k<sub>i</sub>
equal), as in the default chain. The damped modes then have the same real shapes as the
undamped ones, and <b>ζ modal is exact</b>.</p>

<h3>Non-proportional damping and "coupling"</h3>
<p>Otherwise C<sub>m</sub> has off-diagonal terms: damping forces produced by motion in
mode r push on mode s, so the modal equations are <b>coupled</b>. The <b>coupling
index</b> in the note is the largest off-diagonal term relative to the diagonal,
max |C<sub>m,rs</sub>| / √(C<sub>m,rr</sub>C<sub>m,ss</sub>). It is 0 for
proportional damping and approaches 1 when cross-coupling is as strong as each mode's own
damping.</p>
<p>The <b>ζ modal</b> column keeps only the diagonal of C<sub>m</sub> and ignores the
rest. That is the standard "classical damping" approximation used throughout practice. It
works well when coupling is modest or the modes are well separated in frequency, and
degrades as either assumption fails. For example, with the default chain and only
c<sub>1</sub> changed:</p>
<table border="1" cellspacing="0" cellpadding="3">
<tr><th>c<sub>1</sub></th><th>coupling</th><th>worst ζ modal vs ζ exact</th></tr>
<tr><td>2 (default)</td><td>0</td><td>identical</td></tr>
<tr><td>5</td><td>0.36</td><td>0.0977 vs 0.0979 (0.2%)</td></tr>
<tr><td>15</td><td>0.71</td><td>0.168 vs 0.180 (7%)</td></tr>
<tr><td>50</td><td>0.90</td><td>0.45 vs 0.98 (qualitatively wrong)</td></tr>
</table>

<h3>Exact damped modes are complex</h3>
<p>To solve the damped problem exactly, write it in first-order (state-space) form with
z = [x, ẋ]:</p>
<p>&nbsp;&nbsp;ż = A z, &nbsp;&nbsp; A = [ 0 &nbsp;I ; −M<sup>−1</sup>K
&nbsp;−M<sup>−1</sup>C ]</p>
<p>Its eigenvalues come in conjugate pairs
<b>λ = −ζω<sub>n</sub> ± iω<sub>d</sub></b>, which give the
<b>ζ exact</b> and <b>f<sub>d</sub></b> columns. The eigenvectors ψ are complex:
each mass gets an amplitude <i>and a phase</i>. The physical motion is
x(t) = Re(ψ e<sup>λt</sup>), so with non-proportional damping the masses reach
their peaks at <i>different times</i>. The nodes drift and the motion looks partly like a
wave travelling along the chain, not a pure standing wave. With proportional damping all
phases are 0° or 180° and ψ reduces to the real φ.</p>

<h3>Is this an artifact of the solver?</h3>
<p><b>No, it is physics.</b> Both calculations are exact for the problem they solve:</p>
<ul>
<li><b>Undamped modes:</b> symmetric generalized eigenproblem Kφ = ω²Mφ
(scipy <code>eigh</code>). Real by construction.</li>
<li><b>Damped poles and shapes:</b> general eigenproblem of the 2N×2N state matrix
A (scipy <code>eig</code>). Complex when damping is non-proportional.</li>
</ul>
<p>The only approximation in the table is the <b>ζ modal</b> column, which describes
the damped system using the <i>undamped</i> mode shapes. When damping is proportional the
two methods agree to machine precision, as the table shows. The time simulation uses the
full M, C, K, so it always shows the true behaviour.</p>

<h3>Try it</h3>
<ol>
<li>Set c<sub>1</sub> = 15 and leave the other dampers at 2. The note switches to
non-proportional, and ζ modal and ζ exact separate.</li>
<li>Select mode 3 and click <i>Release selected mode</i> (use 0.25× speed). The
masses start in the real mode shape, but because that is not an exact mode of the damped
system, other modes are excited too and the shape drifts as it decays.</li>
<li>Set c<sub>1</sub> back to 2, so every c<sub>i</sub>/k<sub>i</sub> is equal again
(stiffness-proportional damping), and repeat. The coupling returns to 0 and the release
stays in one clean shape.</li>
</ol>
"""


def make_background_view() -> QtWidgets.QTextBrowser:
    view = QtWidgets.QTextBrowser()
    view.setHtml(BACKGROUND_HTML)
    view.setOpenExternalLinks(False)
    return view
