"""Theory notes for the Modal coupling page."""

THEORY_HTML = """
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
push on it, and the right one is the mass the <i>joint</i> feels.</p>

<h3>B: its effective mass</h3>
<p>B is joined to A at its base. When its base moves by u, every mass of B moves with it:
the influence vector is <b>1</b>. The base motion drives mode r with the <b>participation
factor</b></p>
<p>&nbsp;&nbsp;Γ<sub>r</sub> = φ<sub>r</sub><sup>T</sup>M<b>1</b>,</p>
<p>and the force the mode puts back on the base is that of a mass of</p>
<p>&nbsp;&nbsp;m<sub>eff,r</sub> = Γ<sub>r</sub>²&nbsp;&nbsp;(the <b>effective mass</b>
along x)</p>
<p>on a spring of frequency ω<sub>r</sub>. The effective masses of all the modes add up
to B's total mass. The lowest mode of a chain usually carries most of it; the table
lists each mode's share.</p>
<p>The rest of B's mass, m<sub>B</sub> − m<sub>eff</sub>, belongs to B's other modes. Well
below their frequencies they are stiff, so that mass just moves with the base: it is the
<b>residual mass</b>, and it rides on A's tip. <i>Add B's residual mass to A</i> puts it
there.</p>

<h3>A: its modal mass at the interface</h3>
<p>A is joined to B at its tip, not at its base, so what matters is the mass of A's mode
as its tip feels it. Scale the mode so the tip moves 1, φ/φ<sub>tip</sub>: its kinetic
energy is then ½ (1/φ<sub>tip</sub>²) u̇² for a tip velocity u̇, so</p>
<p>&nbsp;&nbsp;m<sub>a</sub> = 1/φ<sub>tip</sub>²&nbsp;&nbsp;(the <b>modal mass at the
interface</b>).</p>
<p>This is the same modal mass the vibration absorber presets use for their primary. A's
effective mass Γ² would be right if B were joined at A's base; <i>Mass of A's
oscillator</i> can be switched to it to see how much worse that is. For a uniform chain
the effective mass of mode 1 is well above its tip modal mass: the tip moves most, so
per unit of tip motion the mode looks light.</p>

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
shapes swap over through the veering. The closer the frequencies and the heavier B, the
wider the veering. The chain's own frequencies do the same, and so does every other pair
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
"""
