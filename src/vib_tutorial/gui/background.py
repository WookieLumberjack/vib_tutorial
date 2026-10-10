"""Background notes on modal analysis and damping, shown in their own tab."""

from __future__ import annotations

from PySide6 import QtWidgets

COUPLING_TIP = (
    "<p><b>Coupling index</b> = the largest off-diagonal term of the modal damping "
    "matrix C<sub>m</sub> = Φ<sup>T</sup>CΦ, relative to its diagonal:</p>"
    "<p>&nbsp;&nbsp;max<sub>r≠s</sub> |C<sub>m,rs</sub>| / √(C<sub>m,rr</sub> C<sub>m,ss</sub>)</p>"
    "<p>0 means the damping does not couple the undamped modes (proportional damping). "
    "It can never exceed 1, because C<sub>m</sub> is positive semidefinite. "
    "Values near 1 mean the damping force from one mode's motion drives another mode "
    "about as strongly as it damps its own. How much this matters also depends on how "
    "close the modes' frequencies are. See the Background tab.</p>"
)

BACKGROUND_HTML = """
<p><i>Contents:</i> <a href="#start-with-one-mass">Start with one mass</a> · <a href="#equations-of-motion">Equations of motion</a> · <a href="#undamped-normal-modes-real">Undamped normal modes (real)</a> · <a href="#proportional-damping">Proportional damping</a> · <a href="#non-proportional-damping-and-coupling">Non-proportional damping and "coupling"</a> · <a href="#exact-damped-modes-are-complex">Exact damped modes are complex</a> · <a href="#is-this-an-artifact-of-the-solver">Is this an artifact of the solver?</a> · <a href="#the-state-space-method-2n-eigenvalues">The state-space method: 2N eigenvalues</a> · <a href="#watching-the-modal-coordinates">Watching the modal coordinates</a> · <a href="#energy">Energy</a> · <a href="#sweeps-and-ground-motion">Sweeps and ground motion</a> · <a href="#vibration-absorbers-and-tuned-mass-dampers">Vibration absorbers and tuned mass dampers</a> · <a href="#try-it">Try it</a> · <a href="#notation">Notation</a></p>
<h3><a name="start-with-one-mass"></a>Start with one mass</h3>
<p>Everything on these pages is built from the single-degree-of-freedom (SDOF) oscillator:
one mass m on a spring k and a damper c, <b>m ẍ + c ẋ + k x = f(t)</b>. Set <i>Number of
masses</i> to 1 to see it on its own (with the defaults m = 1 kg, k = 400 N/m,
c = 2 N·s/m, the numbers in brackets below).</p>
<ul>
<li><b>Natural frequency</b> ω<sub>n</sub> = √(k/m), f<sub>n</sub> = ω<sub>n</sub>/2π
(20 rad/s, 3.18 Hz): how fast it vibrates if nothing damps it.</li>
<li><b>Damping ratio</b> ζ = c / 2√(km) = c / 2mω<sub>n</sub> (0.05): the damping as a
fraction of the <i>critical</i> damping 2√(km), the least that stops it oscillating.
ζ &lt; 1 is underdamped, which nearly every structure is (typically 0.5% to 5%).</li>
<li><b>Free vibration</b> decays as x(t) = X e<sup>−ζω<sub>n</sub>t</sup> cos(ω<sub>d</sub>t
− θ), at the <b>damped natural frequency</b> ω<sub>d</sub> = ω<sub>n</sub>√(1 − ζ²), barely
lower than ω<sub>n</sub> for light damping. The envelope falls by e in the <b>time
constant</b> 1/(ζω<sub>n</sub>) (1 s), and by the <b>logarithmic decrement</b>
δ = ln(x<sub>k</sub>/x<sub>k+1</sub>) = 2πζ/√(1 − ζ²) ≈ 2πζ per cycle (0.31), which is
how damping is measured from a decay record.</li>
<li><b>Forced response.</b> For f = F e<sup>iωt</sup> the steady response is X = H(ω)F with
the <b>receptance</b> H(ω) = 1/(k − ω²m + iωc). Well below ω<sub>n</sub> the spring
carries the force (X ≈ F/k, in phase); well above it the mass does (X ≈ −F/ω²m, 180°
behind); near ω<sub>n</sub> only the damper limits the motion. At ω = ω<sub>n</sub> the
response lags the force by exactly 90° and |X| = (F/k)/2ζ: the <b>dynamic
amplification</b> or <b>quality factor</b> Q ≈ 1/2ζ (10). The displacement peak itself is
slightly lower, at ω<sub>n</sub>√(1 − 2ζ²).</li>
<li><b>Half-power bandwidth.</b> The peak is about 2ζf<sub>n</sub> wide between the points
where |X| falls to 1/√2 of its peak (0.32 Hz), so ζ ≈ (f<sub>2</sub> − f<sub>1</sub>)/2f<sub>n</sub>.
A lightly damped peak is tall and narrow; the measurement pages depend on this.</li>
</ul>
<p>The rest of this tab shows that a chain of N masses is N of these oscillators, one per
mode, as long as the damping is proportional.</p>

<h3><a name="equations-of-motion"></a>Equations of motion</h3>
<p>The chain obeys <b>M ẍ + C ẋ + K x = f(t)</b>: M is diagonal (the masses);
K and C are tridiagonal (each spring and damper links two neighbouring masses, or a mass
and the ground).</p>

<h3><a name="undamped-normal-modes-real"></a>Undamped normal modes (real)</h3>
<p>Ignoring C, free vibration x = φ cos(ωt) requires
<b>Kφ = ω²Mφ</b>. The N solutions are the natural frequencies
ω<sub>r</sub> and <b>real</b> mode shapes φ<sub>r</sub>: every mass moves exactly
in phase or exactly out of phase with the others. They all pass through zero at the same
instant, like a standing wave.</p>
<p>The eigenproblem fixes each mode's <i>shape</i>, not its size: any multiple of
φ<sub>r</sub> is the same mode. A convenient size is the <b>mass-normalized</b> one, scaled
so that φ<sub>r</sub><sup>T</sup>Mφ<sub>r</sub> = 1 (its entries are then in 1/√kg). The
modes are also <b>orthogonal</b> through M and K: φ<sub>r</sub><sup>T</sup>Mφ<sub>s</sub> = 0
and φ<sub>r</sub><sup>T</sup>Kφ<sub>s</sub> = 0 for r ≠ s, which follows from Kφ = ω²Mφ and
the symmetry of M and K. (The <i>Modal coupling</i> page's Theory tab explains why one entry
of a mass-normalized mode, squared, is a mass.)</p>
<p>Collecting the mass-normalized shapes into Φ and writing x = Φq turns the
equations into <i>modal coordinates</i> q:</p>
<p>&nbsp;&nbsp;Φ<sup>T</sup>MΦ = I, &nbsp;&nbsp; Φ<sup>T</sup>KΦ =
diag(ω<sub>r</sub>²), &nbsp;&nbsp; Φ<sup>T</sup>CΦ = C<sub>m</sub></p>
<p>M and K always become diagonal. <b>C<sub>m</sub> in general does not.</b></p>

<h3><a name="proportional-damping"></a>Proportional damping</h3>
<p>If C<sub>m</sub> <i>is</i> diagonal, every mode obeys its own independent single-DOF
equation</p>
<p>&nbsp;&nbsp;q̈<sub>r</sub> + 2ζ<sub>r</sub>ω<sub>r</sub>q̇<sub>r</sub>
+ ω<sub>r</sub>²q<sub>r</sub> = φ<sub>r</sub><sup>T</sup>f, &nbsp;&nbsp; with
ζ<sub>r</sub> = C<sub>m,rr</sub> / 2ω<sub>r</sub></p>
<p>This happens for <b>Rayleigh damping</b> C = αM + βK. Then
Φ<sup>T</sup>CΦ = αI + β diag(ω<sub>r</sub>²), so</p>
<p>&nbsp;&nbsp;<b>ζ<sub>r</sub> = α/(2ω<sub>r</sub>) + βω<sub>r</sub>/2</b></p>
<p>The mass-proportional part damps the low modes most, the stiffness-proportional part the
high ones. In this app the simplest case is every damper being the same multiple of its
spring (all c<sub>i</sub>/k<sub>i</sub> = β equal), as in the default chain: C = βK with
β = 2/400 = 0.005 s, so ζ<sub>r</sub> = βω<sub>r</sub>/2 grows in proportion to frequency.
That is why the default chain's ζ column climbs from 0.017 for mode 1 to 0.094 for mode 4.
(Mass-proportional damping cannot be built from this chain: it would need a damper from
every mass to the ground.) The damped modes then have the same real shapes as the undamped
ones, and <b>ζ modal is exact</b>. The general condition, Caughey's, is that
CM<sup>−1</sup>K = KM<sup>−1</sup>C; Rayleigh damping is the most-used case of it.</p>

<h3><a name="non-proportional-damping-and-coupling"></a>Non-proportional damping and "coupling"</h3>
<p>Otherwise C<sub>m</sub> has off-diagonal terms: damping forces produced by motion in
mode r push on mode s, so the modal equations are <b>coupled</b>. The <b>coupling
index</b> in the note under the modal table is the largest off-diagonal term relative to
the diagonal, max |C<sub>m,rs</sub>| / √(C<sub>m,rr</sub>C<sub>m,ss</sub>). It is 0 for
proportional damping and approaches 1 when cross-coupling is as strong as each mode's own
damping. It cannot exceed 1: dampers only dissipate energy, so C<sub>m</sub> is positive
semidefinite, and then |C<sub>m,rs</sub>| ≤ √(C<sub>m,rr</sub>C<sub>m,ss</sub>)
(Cauchy–Schwarz).</p>
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

<h3><a name="exact-damped-modes-are-complex"></a>Exact damped modes are complex</h3>
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

<h3><a name="is-this-an-artifact-of-the-solver"></a>Is this an artifact of the solver?</h3>
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

<h3><a name="the-state-space-method-2n-eigenvalues"></a>The state-space method: 2N eigenvalues</h3>
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
τ = −1/λ and no oscillation. A chain with no spring to the ground (k<sub>1</sub> = 0)
has λ = 0: rigid-body motion, a position it keeps. With the damper c<sub>1</sub> still
there, a second real root goes with it, close to −c<sub>1</sub>/(m<sub>1</sub> + … + m<sub>N</sub>)
(−0.50 s<sup>−1</sup> for the default chain): a push slides the whole chain to a new position
as its velocity dies away through c<sub>1</sub>. With
c<sub>1</sub> = 0 too, λ = 0 is a repeated root with only one eigenvector (defective): the
chain can drift at constant velocity, which no pair of exponentials describes.</li>
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

<h3><a name="watching-the-modal-coordinates"></a>Watching the modal coordinates</h3>
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
<p>This is also how any free vibration is worked out by hand: project the starting state
onto the modes, q<sub>r</sub>(0) = φ<sub>r</sub><sup>T</sup>Mx(0) and
q̇<sub>r</sub>(0) = φ<sub>r</sub><sup>T</sup>Mẋ(0), let each q<sub>r</sub> decay as its own
single-mass oscillator, and add them back, x(t) = Σ φ<sub>r</sub>q<sub>r</sub>(t) (exact for
proportional damping). A mode release starts with one q<sub>r</sub> only; a pluck starts
with all of them.</p>
<p>Under a harmonic force tuned to a natural frequency, the modal view shows which mode
takes up the energy. The physical view shows the same motion as a mix of all of them.</p>

<h3><a name="energy"></a>Energy</h3>
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
<p><b>Friction.</b> A mass given friction F<sub>f</sub> (under <i>System parameters</i>)
feels a force of that size against its velocity while it slides, and none of its energy
comes back: friction takes out F<sub>f</sub>|ẋ<sub>i</sub>|, F<sub>f</sub> times the distance
slid. A stuck mass does not move, so its friction does no work. The balance becomes
<b>energy given + work = stored + dissipated by the dampers + by friction</b>. Friction is
not linear: it does not grow with the motion, so small vibrations die out sooner, relative
to their size, than large ones, and a mass comes to rest wherever its springs can no longer
pull it free, not at the rest position. The simulator steps the chain exactly between the
moments a mass stops or breaks free; the modes and frequency responses leave friction
out.</p>
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

<h3><a name="sweeps-and-ground-motion"></a>Sweeps and ground motion</h3>
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
isolation. For a single mass on k and c, |X/X<sub>g</sub>| is below 1 only above
<b>√2 f<sub>n</sub></b>, whatever the damping. Below √2 f<sub>n</sub> more damping lowers the
resonant peak; above it more damping makes the isolation <i>worse</i>, which is the
trade-off in choosing an isolator. In the chain each mass has its own crossover, and some
masses fall below 1 well before the highest mode (in the default chain, m3 above 1.65 Hz). The damper c<sub>1</sub> passes on the ground's velocity, so at high frequency
x<sub>1</sub> falls only as 1/ω rather than 1/ω². The work the ground does is the tension
N<sub>1</sub> in element 1 times the ground's velocity, with a minus sign:
−∫N<sub>1</sub>ẋ<sub>g</sub> dt.
While the ground moves, the modal coordinates and the energy by mode use the motion
relative to it, x − x<sub>g</sub>.</p>

<h3><a name="vibration-absorbers-and-tuned-mass-dampers"></a>Vibration absorbers and tuned mass dampers</h3>
<p>A machine or structure that resonates near its operating frequency can be fixed by
hanging a small mass m<sub>a</sub> on it with a spring k<sub>a</sub> (and perhaps a damper
c<sub>a</sub>). In this app that is one more mass on the end of the chain
(<i>System parameters → Preset</i>).</p>
<p><b>Undamped absorber.</b> For a primary mass m<sub>1</sub> on k<sub>1</sub> driven by
F sin ωt, the primary's receptance is</p>
<p>&nbsp;&nbsp;X<sub>1</sub>/F = (k<sub>a</sub> − ω²m<sub>a</sub>) /
[(k<sub>1</sub> + k<sub>a</sub> − ω²m<sub>1</sub>)(k<sub>a</sub> − ω²m<sub>a</sub>) − k<sub>a</sub>²]</p>
<p>The numerator vanishes at ω<sub>a</sub> = √(k<sub>a</sub>/m<sub>a</sub>): an
<i>antiresonance</i>. At that frequency m<sub>1</sub> stands still and the absorber moves
just enough that its spring pushes back with −F. Tuned to the primary's own frequency it
cancels the resonance, but the one peak becomes two, one either side, and they are
further apart the heavier the absorber (mass ratio μ = m<sub>a</sub>/m<sub>1</sub>). It only
works at one frequency: a machine whose speed drifts, or starts up through the peaks,
needs damping.</p>
<p><b>Tuned mass damper.</b> Add a damper c<sub>a</sub> and the peaks come down, but too
much locks the absorber to m<sub>1</sub> and the old resonance comes back. If the primary
itself is undamped, every curve of |X<sub>1</sub>/F|, whatever c<sub>a</sub>, passes through
the same two <i>fixed points</i>. Den Hartog's optimum tunes the absorber so the fixed points
are equally high, then picks the damping that puts the peaks on them (strictly, Brock's
average of the two values that each put one peak on its fixed point):</p>
<p>&nbsp;&nbsp;f<sub>a</sub>/f<sub>1</sub> = 1/(1 + μ), &nbsp;&nbsp;
ζ<sub>a</sub> = √(3μ / 8(1 + μ)³), &nbsp;&nbsp; peak |X<sub>1</sub>|k<sub>1</sub>/F =
√(1 + 2/μ)</p>
<p>Here ζ<sub>a</sub> = c<sub>a</sub> / 2m<sub>a</sub>ω<sub>1</sub>, measured against the
<i>primary's</i> frequency ω<sub>1</sub> = √(k<sub>1</sub>/m<sub>1</sub>), as in Den Hartog's
derivation. (Some textbooks use the absorber's own ω<sub>a</sub> instead; the two differ by
the factor 1 + μ, so check which one a formula assumes.) With μ = 5% and an undamped primary
both peaks are about 6.4 times the static deflection. Damping in the primary lowers them
further, and the fixed points are then only approximate.</p>
<p><b>On a chain</b>, such as a building, a damper on the roof is tuned to one mode. Near
that mode the structure behaves like a single mass on a spring whose <i>modal mass</i> at
the roof is 1/φ<sub>roof</sub>² (φ mass-normalized), so μ and the tuning use that mass
and the mode's frequency. The other modes are hardly changed.</p>
<p>Tick <i>Compare with the chain without its last mass</i> on the <i>Frequency
response</i> tab to see the structure before the absorber was added, dashed.</p>

<h3><a name="try-it"></a>Try it</h3>
<ol>
<li>Set c<sub>1</sub> = 15 and leave the other dampers at 2. The note under the modal table switches to
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
<li>Load <i>Preset → Vibration absorber (undamped)</i> and apply the force. After the
start-up transient m1 stands still and only the absorber m2 moves. Plot <i>Element
forces</i>: spring k2 carries the applied force. Change the drive frequency a little and
m1 starts moving again.</li>
<li>Load <i>Tuned mass damper (Den Hartog)</i>. On the <i>Frequency response</i> tab the
dashed primary alone peaks at 40 times its static deflection, the damped system at under
6 (its primary is lightly damped, so lower than the 6.4 of an undamped one). Set c2 to 0:
two sharp peaks. Set it to 5: the damper locks the absorber to m1 and one tall peak comes
back. Every curve passes close to the same two fixed points.</li>
</ol>

<h3><a name="notation"></a>Notation</h3>
<table border="1" cellspacing="0" cellpadding="3">
<tr><th>Symbol</th><th>Meaning</th><th>Units</th></tr>
<tr><td>m, k, c</td><td>mass, spring stiffness, damper coefficient (one element)</td><td>kg, N/m, N·s/m</td></tr>
<tr><td>M, C, K</td><td>mass, damping and stiffness matrices of the chain</td><td>kg, N·s/m, N/m</td></tr>
<tr><td>x, ẋ, ẍ; x<sub>g</sub></td><td>mass displacements, velocities, accelerations; ground displacement</td><td>m, m/s, m/s²</td></tr>
<tr><td>f, F</td><td>applied force (vector), its amplitude</td><td>N</td></tr>
<tr><td>ω<sub>n</sub>, f<sub>n</sub>; ω<sub>d</sub>, f<sub>d</sub></td><td>undamped and damped natural frequency</td><td>rad/s, Hz</td></tr>
<tr><td>ζ</td><td>damping ratio: <i>modal</i> φ<sup>T</sup>Cφ/2ω, or <i>exact</i> −Re λ/|λ|</td><td>—</td></tr>
<tr><td>φ<sub>r</sub>, Φ</td><td>undamped mode shape r, mass-normalized (φ<sup>T</sup>Mφ = 1); all of them as columns</td><td>1/√kg</td></tr>
<tr><td>q<sub>r</sub></td><td>modal coordinate of mode r, x = Φq</td><td>m·√kg</td></tr>
<tr><td>C<sub>m</sub></td><td>modal damping matrix Φ<sup>T</sup>CΦ</td><td>1/s</td></tr>
<tr><td>A, z</td><td>state matrix and state vector [x; ẋ] of the first-order form ż = Az</td><td>—</td></tr>
<tr><td>λ, λ*</td><td>eigenvalue of A and its complex conjugate</td><td>1/s</td></tr>
<tr><td>ψ</td><td>complex (damped) mode shape, the displacement part of A's eigenvector</td><td>—</td></tr>
<tr><td>V, η</td><td>eigenvectors [ψ; λψ] of A as columns; state-space modal coordinates η = V<sup>−1</sup>z</td><td>—</td></tr>
<tr><td>H(ω)</td><td>receptance X/F</td><td>m/N</td></tr>
<tr><td>T, V; E<sub>0</sub>, W, D</td><td>kinetic and potential energy; energy given, work done, energy dissipated</td><td>J</td></tr>
<tr><td>N<sub>1</sub></td><td>tension in element 1 (spring plus damper)</td><td>N</td></tr>
<tr><td>τ</td><td>time constant of a decay: 1/(ζω<sub>n</sub>), or −1/λ for a real root. The Virtual test page uses τ<sub>p</sub> for the hammer pulse and τ<sub>w</sub> for the exponential window</td><td>s</td></tr>
<tr><td>μ</td><td>mass ratio of an absorber, m<sub>a</sub>/m<sub>1</sub> (or to the modal mass)</td><td>—</td></tr>
<tr><td>α, β</td><td>Rayleigh damping coefficients, C = αM + βK</td><td>1/s, s</td></tr>
</table>
"""


def make_background_view() -> QtWidgets.QTextBrowser:
    view = QtWidgets.QTextBrowser()
    view.setHtml(BACKGROUND_HTML)
    view.setOpenExternalLinks(False)
    return view
