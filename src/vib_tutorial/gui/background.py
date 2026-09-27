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

<h3>The state-space method: 2N eigenvalues</h3>
<p>Choose <i>Method → State-space</i> on the Modal analysis tab to see the full damped
solution directly. Writing the N second-order equations as 2N first-order ones doubles
the size of the eigenproblem, so there are <b>2N eigenvalues</b> and 2N eigenvectors:</p>
<ul>
<li><b>Conjugate pairs.</b> A is real, so complex eigenvalues come in pairs
λ = σ + iω<sub>d</sub> and λ* = σ − iω<sub>d</sub>, with conjugate shapes ψ and ψ*.
Neither one alone is a real motion; together they are:
x(t) = ψe<sup>λt</sup> + ψ*e<sup>λ*t</sup> = 2 Re(ψ e<sup>λt</sup>). In the
complex-plane plot the arrows of λ turn anticlockwise and those of λ* clockwise; their
real parts, the displacements, are identical. Each oscillatory mode of the classical
table corresponds to one pair.</li>
<li><b>Real eigenvalues.</b> A heavily damped (overdamped) mode gives two real
eigenvalues instead of a pair: pure exponential decays with time constant
τ = −1/λ and no oscillation. A chain that is free to slide (k<sub>1</sub> = 0)
has λ = 0: rigid-body motion.</li>
<li><b>The eigenvector includes velocities.</b> The state eigenvector is
[ψ ; λψ]. To start the chain in exactly one mode you must set the displacements
<i>and</i> the velocities: x(0) = Re(ψ), ẋ(0) = Re(λψ). That is what <i>Release
selected mode</i> does in this method. It is why a complex mode can be released cleanly
with non-proportional damping, while the classical release (real shape, at rest) excites
other modes too.</li>
<li><b>Orthogonality.</b> A is not symmetric, so the ψ are not orthogonal in the
usual sense. Decoupling the equations needs the <i>left</i> eigenvectors of A as well
(or the symmetric "Duncan" form of the state equations). This is why the classical
method, with its simple Φ<sup>T</sup>MΦ = I, is preferred whenever damping is light
or nearly proportional.</li>
</ul>
<p>The <b>phase</b> column next to the complex-plane plot gives each mass's phase
relative to the largest one. With proportional damping every phase is 0° or 180°
(a real mode); otherwise the masses reach their peaks at different times.</p>

<h3>Watching the modal coordinates</h3>
<p>Set <i>Plot coordinates</i> (above the time histories) to <i>Modal</i> to plot the
motion in modal coordinates instead of mass displacements: one curve per mode, in the mode
colors. Each curve is scaled to metres: it is that mode's share of the displacement of
the mass the mode moves most. It follows the <i>Method</i> selector:</p>
<ul>
<li><b>Classical:</b> q = Φ<sup>T</sup>Mx. Since x = Φq exactly, the curves hold all
of the motion. With proportional damping each q<sub>r</sub> is its own damped oscillator.
With non-proportional damping the off-diagonal terms of C<sub>m</sub> make each
q<sub>r</sub> push on the others, and you see motion move from one curve to another.</li>
<li><b>State-space:</b> η = V<sup>−1</sup>z, where V holds the eigenvectors [ψ; λψ]
as columns. It needs velocities as well as displacements. The coordinates of λ and λ*
are conjugates, so each pair is one curve, 2 Re(η). These are independent for
<i>any</i> damping: each one is a pure decaying oscillation (or decay) unless the force
drives it.</li>
</ul>
<p>Under a harmonic force tuned to a natural frequency, the modal view shows which mode
takes up the energy. The physical view shows the same motion as a mix of all of them.</p>

<h3>Energy</h3>
<p>The panel beside the animation tracks the energy. The stored energy is kinetic plus
potential, T + V = ½ẋ<sup>T</sup>Mẋ + ½x<sup>T</sup>Kx. Multiplying the equations of
motion by ẋ<sup>T</sup> gives the power balance</p>
<p>&nbsp;&nbsp;d(T + V)/dt = f<sup>T</sup>ẋ − ẋ<sup>T</sup>Cẋ</p>
<p>The force adds power f<sup>T</sup>ẋ, which can be negative when it pushes against the
motion, and the dampers always remove ẋ<sup>T</sup>Cẋ ≥ 0. Integrated since the last
reset: <b>energy given + work by the force = stored + dissipated</b>. The simulator
integrates both terms exactly over every step, so the <i>In</i> and <i>Out</i> columns
match to rounding error. "Energy given" counts the jumps when a mode is released or a
parameter is edited.</p>
<p><b>By mode.</b> Because Φ<sup>T</sup>MΦ = I and Φ<sup>T</sup>KΦ = diag(ω<sub>r</sub>²),
the stored energy splits exactly into one term per undamped mode,
T + V = Σ ½(q̇<sub>r</sub>² + ω<sub>r</sub>²q<sub>r</sub>²), whatever the damping. Each mode's
energy changes at the rate</p>
<p>&nbsp;&nbsp;d/dt ½(q̇<sub>r</sub>² + ω<sub>r</sub>²q<sub>r</sub>²) = q̇<sub>r</sub>φ<sub>r</sub><sup>T</sup>f
− q̇<sub>r</sub> Σ<sub>s</sub> C<sub>m,rs</sub> q̇<sub>s</sub></p>
<p>With proportional damping only the s = r term is left, so each mode loses its own energy
and nothing else. The off-diagonal terms of C<sub>m</sub> pass energy from one mode to
another. The complex modes do not split the energy this way: they are not orthogonal
with respect to M and K, so their energies have cross terms. The panel always uses the
classical modes.</p>

<h3>Sweeps and ground motion</h3>
<p>A <b>chirp</b> is a sine whose frequency sweeps from a start to an end frequency. Each
mode swells as the sweep passes its natural frequency, so a slow sweep traces the
frequency response out in time. A sweep that is fast compared with a mode's decay
(time constant 1/(ζω<sub>n</sub>)) leaves it no time to build up: the peak comes late and
low, and the mode rings on after the sweep has moved on. A log sweep spends equal time
in every octave.</p>
<p>With <b>ground motion</b> the wall moves as x<sub>g</sub>(t) instead of a force pushing
a mass. Element 1 then stretches by x<sub>1</sub> − x<sub>g</sub>, so the ground reaches
the chain only through k<sub>1</sub> and c<sub>1</sub>:</p>
<p>&nbsp;&nbsp;Mẍ + Cẋ + Kx = (k<sub>1</sub>x<sub>g</sub> + c<sub>1</sub>ẋ<sub>g</sub>) e<sub>1</sub></p>
<p>For a harmonic x<sub>g</sub> the <i>transmissibility</i> X<sub>i</sub>/X<sub>g</sub> is
receptance column 1 times k<sub>1</sub> + iωc<sub>1</sub>. It is 1 at low frequency (the
chain moves with the ground), peaks at every mode, and falls away above the highest one:
isolation. The damper c<sub>1</sub> passes on the ground's velocity, so at high frequency
x<sub>1</sub> falls only as 1/ω rather than 1/ω². The work the ground does is the tension
in element 1 times the ground's velocity, with a minus sign: −∫T<sub>1</sub>ẋ<sub>g</sub> dt.
While the ground moves, the modal coordinates and the energy by mode use the motion
relative to it, x − x<sub>g</sub>.</p>

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
<li>Switch <i>Plot coordinates</i> to <i>Modal</i> and repeat steps 1 to 3. With
c<sub>1</sub> = 15 the released mode 3 feeds modes 1 and 2, and mode 1 keeps ringing
after mode 3 has died away. With c<sub>1</sub> = 2 only the mode 3 curve moves.</li>
<li>Set the energy panel to <i>By mode</i> and repeat. With c<sub>1</sub> = 15 mode 3
starts with all the energy, and within a second most of what is left is in mode 1.
With c<sub>1</sub> = 2 mode 3 keeps 100% while the total decays.</li>
<li>Set c<sub>1</sub> = 15 again and switch <i>Method</i> to <i>State-space</i>. The
table now has 2N rows in conjugate pairs. Select a row and tick <i>Animate mode
shapes</i>: the arrows fan out (phases other than 0°/180°), and the shape never passes
through zero everywhere at once.</li>
<li>Select λ and then λ* and release each. The motion is identical, and it now stays
in one shape as it decays: the complex mode is an exact mode of the damped system.
In the modal view only its own curve moves.</li>
<li>Raise c<sub>1</sub> to 200. One pair turns into two real eigenvalues (overdamped),
one slow and one very fast, listed with their time constants. There are still 2N in
total.</li>
</ol>
"""


def make_background_view() -> QtWidgets.QTextBrowser:
    view = QtWidgets.QTextBrowser()
    view.setHtml(BACKGROUND_HTML)
    view.setOpenExternalLinks(False)
    return view
