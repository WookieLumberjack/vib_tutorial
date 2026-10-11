"""Text for the Jeffcott rotor page: the theory notes and the live equations and matrices."""

from __future__ import annotations

import cmath
import math
from collections.abc import Collection

import numpy as np

from ..core.rotor import DOF_NAMES, MODE_LIMIT, RPM, XD, YD, RotorSystem, gravity_response, whirl_modes
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
    # Unwrapped: a wide matrix (a shaft with mass) scrolls sideways rather than breaking its numbers.
    nowrap = "white-space:nowrap"
    head = "".join(f"<th style='{nowrap}'>{c}</th>" for c in labels)
    out = [f"<table border='1' cellspacing='0' cellpadding='3'><tr><th style='{nowrap}'>{title}</th>{head}</tr>"]
    for r, (label, rg) in enumerate(zip(labels, groups)):
        cells = []
        for c, cg in enumerate(groups):
            text = _num(A[r, c], scale)
            style = nowrap + (f"; color:{colors.grey}" if text == "0" else "")
            tint = colors.block_tints[tuple(sorted((rg, cg)))]
            cells.append(f"<td align='right' style='{style}' bgcolor='{tint}'>{text}</td>")
        out.append(f"<tr><th style='{nowrap}'>{label}</th>{''.join(cells)}</tr>")
    out.append("</table>")
    return "".join(out)


def _labels(system: RotorSystem) -> tuple[list[str], list[str]]:
    """q's labels and blocks: the stations, then each plane's shaft modal coordinates ξ."""
    r = range(1, system.modes_kept + 1)
    return (LABELS + [f"ξ<sub>x{k}</sub>" for k in r] + [f"ξ<sub>y{k}</sub>" for k in r],
            GROUPS + ["i"] * len(r) + ["b"] * len(r))


def matrices_html(system: RotorSystem, omega: float, hidden: Collection[str] = ()) -> str:
    """The equations of motion, every matrix with the current numbers, and the modes at speed Ω.

    `hidden` names the RotorSystem fields a balancing exercise keeps out of sight: the
    unbalance's size and angle, and perhaps the skew's.
    """
    M, C, G, K = system.matrices()
    H = system.circulatory()
    s = system
    labels, groups = _labels(s)

    def _mat(A: np.ndarray, title: str) -> str:
        return _matrix(A, labels, groups, title)

    ks = s.shaft_stiffness()
    kept = s.modes_kept
    one = ["w<sub>A</sub>", "w<sub>D</sub>", "θ<sub>D</sub>", "w<sub>B</sub>"]
    one += [f"ξ<sub>{k}</sub>" for k in range(1, kept + 1)]
    shaft = _matrix(ks, one, ["i"] * len(one), "K<sub>shaft</sub>")
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
    if s.asymmetry:
        Kc, _ = s.deviatoric
        asymmetric = (
            f"<h4>The asymmetric shaft ΔK(φ)</h4><p>The shaft is stiffer bending toward its stiff axis, at "
            f"θ = φ + α from x (α = {math.degrees(s.asymmetry_angle):.0f}° from the mark), than across it: "
            f"EI(1 ± Δ) with Δ = {s.asymmetry:.3g}. Its stiffness less the mean (K<sub>shaft</sub>, in K) is, "
            "in each plane pair,</p><p align='center'>ΔK(φ) = Δ [cos 2θ K<sub>shaft</sub> &nbsp; sin 2θ "
            "K<sub>shaft</sub> ; sin 2θ K<sub>shaft</sub> &nbsp; −cos 2θ K<sub>shaft</sub>]</p><p>or, as "
            "f<sub>x</sub> + i f<sub>y</sub>, −ΔK<sub>shaft</sub>e<sup>2iθ</sup>(x − iy). It turns with the "
            "shaft, twice per revolution as seen from the bearings, so the equations have a periodic "
            "stiffness and no fixed modes. Here it is with the stiff axis along x (θ = 0):</p>"
            + _matrix(Kc, labels, groups, "ΔK(θ = 0)")
            + "<p>In coordinates turning with the shaft, q = R(φ)r, it is constant, and so is everything "
            "else if the supports are isotropic: M r̈ + (C + ΩG + 2ΩMJ) ṙ + (K + ΩH + ΔK(0) − Ω²M + ΩCJ + "
            "Ω²GJ) r = R<sup>T</sup>f. The −Ω²M is the centrifugal force: between the two critical speeds it "
            "outweighs the soft axis's stiffness but not the stiff one's, and the whirl grows, locked to 1X. "
            + ("Here the supports are isotropic, so the modes below come from that constant system."
               if s.isotropic_supports else
               "Here the supports are not isotropic, so the modes below come from Hill's method instead: the "
               "periodic system's Floquet exponents, from its harmonics up to ±3 × 2Ω.")
            + " The time stepping takes each step as half a step of the rest (the exact update), the kick "
            "of ΔK at mid-step on the velocities, and the other half step.</p>"
        )
    else:
        asymmetric = ""
    if s.overhung:
        near, far = ("B", "A") if s.a > s.length else ("A", "B")
        elements = (f"Two Euler–Bernoulli beam elements: the span between the bearings (L = {s.length:.4g} m) and "
                    f"the overhang from bearing {near} to the disc (c = {s.overhang:.4g} m), with EI = {s.ei:.4g} N·m². "
                    f"The disc sits at a = {s.a:.4g} m from A, outside the bearings.")
        disc_k = f"3EI/(c²(L + c)) = {s.disc_stiffness:,.0f} N/m for the overhung disc"
    else:
        elements = (f"Two Euler–Bernoulli beam elements, A to the disc (a = {s.a:.4g} m) and the disc to B "
                    f"(b = {s.b:.4g} m), with EI = {s.ei:.4g} N·m².")
        disc_k = f"3EIL/(a²b²) = {s.disc_stiffness:,.0f} N/m"
    n_el = s.nodes.size - 1
    if s.density:
        freqs = ", ".join(f"{f:,.0f}" for f in s.fixed_interface_freqs) or "none"
        elements = (
            f"The shaft has mass, {s.shaft_mass:.3g} kg ({s.density:,.0f} kg/m³): it is {n_el} Euler–Bernoulli "
            f"beam elements, with {'lumped' if s.lumped else 'consistent'} mass and their sections' rotary "
            f"inertia, a displacement and a slope at every node. That model is reduced (Craig–Bampton): the "
            f"stations' DOFs are kept, and the rest of the shaft moves as the static shape they give it plus "
            f"its {kept} lowest mode{'s' if kept != 1 else ''} with the stations held, up to {MODE_LIMIT:,.0f} Hz "
            f"({freqs} Hz). Each mode's mass-normalized amplitude ξ is a DOF of its own, in both planes. "
            + elements.replace("Two Euler–Bernoulli beam elements", "Without the extra nodes, the stretches are")
        )
        mass_text = (f"M is no longer diagonal: the shaft's mass couples the stations and its modes. The disc's "
                     f"mass m and I<sub>d</sub>, and the journals', are added on their own DOFs.")
        gyro_text = (f"G has ±I<sub>p</sub> = ±{s.ip:.4g} kg·m² on the disc's slopes and the shaft's own sections' "
                     f"(twice their rotary inertia), joining each plane to the other.")
        condensed = ("The stations' block of K<sub>shaft</sub> is the shaft's exact static stiffness, as for the "
                     "massless shaft; each mode adds its ω² alone on the diagonal (mass normalized).")
    else:
        mass_text = ("M is diagonal: the journal masses, the disc's mass m on its displacements and its diametral "
                     "inertia I<sub>d</sub> on its slopes.")
        gyro_text = (f"G has just two entries, ±I<sub>p</sub> = ±{s.ip:.4g} kg·m², joining the disc's two slopes, "
                     f"one plane to the other.")
        condensed = ("The bearings carry no moment, so the slopes at A and B are condensed out (they follow the "
                     "four DOFs kept).")
    kxy_a, kxy_b = s.cross_coupling(omega)
    if s.bearing_a.swirl or s.bearing_b.swirl:
        swirl = (f" With fluid swirl the cross-coupling grows with the speed, k<sub>xy</sub> + λcΩ, and that part "
                 f"sits in ΩH (below): at this speed k<sub>xy</sub> is {fmt(kxy_a, 1.0)} N/m at A and "
                 f"{fmt(kxy_b, 1.0)} N/m at B in all.")
    else:
        swirl = ""
    fb = ks @ s.bow_shape() * s.bow
    bow_journals = f"{fmt(-fb[0], 1.0)} N at A and {fmt(-fb[3], 1.0)} N at B"
    if s.gravity and s.asymmetry:
        qs = s.static_deflection(omega)
        q2 = gravity_response(s, [omega])[1][0]
        gravity = (
            f"<h4>Gravity</h4><p>On a horizontal rotor f also holds the weights, constant: "
            f"−m<sub>A</sub>g, −mg and −m<sub>B</sub>g on y<sub>A</sub>, y<sub>D</sub> and y<sub>B</sub> "
            f"(here {fmt(s.disc_mass * s.gravity, 1.0)} N on the disc)"
            + (f", and the shaft's own weight, {fmt(s.shaft_mass * s.gravity, 1.0)} N" if s.density else "")
            + ". On the asymmetric shaft a constant force meets a stiffness that turns, so the rotor never "
            "settles: its steady state is periodic, q = Q<sub>0</sub> + Re(Q<sub>2</sub>e<sup>2iφ</sup>)"
            + ("" if s.isotropic_supports else " + (4X, ...)") + ", found by harmonic balance (each "
            "harmonic k of the response is tied to k ± 2 by ΔK). The mean sag Q<sub>0</sub> puts the disc at "
            f"({qs[XD] * 1e6:.4g}, {qs[YD] * 1e6:.4g}) µm; about it the disc whirls at 2X, "
            f"{abs(q2[XD]) * 1e6:.4g} µm in x and {abs(q2[YD]) * 1e6:.4g} µm in y at this speed, most where 2Ω "
            "meets a natural frequency: the gravity critical.</p>"
        )
    elif s.gravity:
        qs = s.static_deflection(omega)
        gravity = (
            f"<h4>Gravity</h4><p>On a horizontal rotor f also holds the weights, constant: "
            f"−m<sub>A</sub>g, −mg and −m<sub>B</sub>g on y<sub>A</sub>, y<sub>D</sub> and y<sub>B</sub> "
            f"(here {fmt(s.disc_mass * s.gravity, 1.0)} N on the disc)"
            + (f", and the shaft's own weight, {fmt(s.shaft_mass * s.gravity, 1.0)} N, spread over the stations "
               f"and its modes" if s.density else "") + f". They hold the rotor at the static "
            f"deflection q<sub>s</sub> = (K + ΩH)<sup>−1</sup>f<sub>g</sub>, here the disc at "
            f"({qs[XD] * 1e6:.4g}, {qs[YD] * 1e6:.4g}) µm, about which it vibrates as without gravity.</p>"
        )
    else:
        gravity = ""
    modal = (f", then the shaft's modal coordinates ξ, {kept} in each plane (in √kg·m: the modes are mass "
             f"normalized, so a mode moving alone has kinetic energy ½ξ̇²)" if kept else "")
    rigid = " (ξ = 0: a rigid motion is a static shape)" if kept else ""
    net = s.mass_unbalance
    if hidden:
        unbalance, force = "The unbalance (hidden: find it on the <i>Balancing</i> tab)", "hidden"
    else:
        unbalance = (f"The unbalance U = m e = {s.unbalance * 1e6:.4g} g·mm at ψ = "
                     f"{math.degrees(s.unbalance_angle):.0f}°")
        on_disc = len(s.weights) + sum(s.on_disc(p) for p in s.plane_weights)
        if on_disc:
            unbalance += (f", with {on_disc} balance weight{'s' if on_disc > 1 else ''}"
                          f" (net {abs(net) * 1e6:.4g} g·mm at {math.degrees(math.atan2(net.imag, net.real)):.0f}°),")
        force = fmt(abs(net) * omega**2, 1.0)
    on_shaft = [p for p in s.plane_weights if not s.on_disc(p)]
    if on_shaft:
        listed = "; ".join(f"{abs(p.weight) * 1e6:.4g} g·mm at {math.degrees(cmath.phase(p.weight)) % 360:.0f}°, "
                           f"z = {p.z * 1e3:.0f} mm" for p in on_shaft)
        shaft_weights = (f" Balance weights on the shaft ({listed}) are point forces of the same form, shared over "
                         f"the DOFs as a load at z is, S(z)<sup>T</sup>.")
    else:
        shaft_weights = ""
    if "skew" in hidden:
        skew = "The disc's skew τ (hidden)"
        moment = "hidden"
    else:
        skew = (f"The disc's skew τ = {math.degrees(s.skew):.4g}° (toward {math.degrees(s.skew_angle):.0f}° from "
                f"the mark)")
        moment = fmt(abs(s.couple_unbalance) * omega**2, 1.0) + " N·m"
    faces = (" Balance weights on the disc's faces, a distance h from its centre plane, add wh to "
             "(I<sub>d</sub> − I<sub>p</sub>)τe<sup>iγ</sup>." if any(p.disc for p in s.plane_weights) else "")
    return f"""
<h3>Equations of motion at {rpm:,.0f} rpm (Ω = {omega:.4g} rad/s)</h3>
<p style='font-size:large' align='center'>M q̈ + (C + Ω G) q̇ + (K + Ω H{" + ΔK(φ)" if s.asymmetry else ""}) q = f(t)</p>
<p>q = [x<sub>A</sub>, x<sub>D</sub>, θ<sub>x</sub>, x<sub>B</sub> | y<sub>A</sub>, y<sub>D</sub>,
θ<sub>y</sub>, y<sub>B</sub>]: the journals' displacements at A and B, the disc's displacement, and the
shaft's slope at the disc (θ<sub>x</sub> = dx/dz, θ<sub>y</sub> = dy/dz), first in the horizontal x–z
plane, then in the vertical y–z plane{modal}. Units: m and rad; forces in N, moments in N·m.</p>
{_legend()}

<h4>Mass M and stiffness K</h4>
<p>{mass_text} K is the shaft's bending stiffness plus each support spring on its
journal. Only the bearings' cross-coupled stiffness k<sub>xy</sub> joins the two planes, as
K[x, y] = k<sub>xy</sub> and K[y, x] = −k<sub>xy</sub> at each journal (here
{fmt(s.bearing_a.kxy, 1.0)} N/m at A and {fmt(s.bearing_b.kxy, 1.0)} N/m at B). That part is
skew-symmetric, so K is symmetric only without it; with isotropic supports and no cross-coupling K has
two identical blocks.{swirl}</p>
{_mat(M, "M")}<br>{_mat(K, "K")}

<h4>The shaft (one plane)</h4>
<p>{elements} {condensed} Moving the whole shaft sideways, or tilting it rigidly,
takes no force: K<sub>shaft</sub>[1, 1, 0, 1]<sup>T</sup> = 0 and K<sub>shaft</sub>[0, a, 1, L]<sup>T</sup> = 0{rigid}.
The supports hold it.</p>
{shaft}

<h4>Damping C and the gyroscopic matrix Ω G</h4>
<p>C has the support dampers, and the shaft's internal damping η K<sub>shaft</sub> in each plane
(η = c<sub>i</sub>/k<sub>disc</sub> = {s.loss_time:.4g} s, so that it is c<sub>i</sub> =
{s.internal_damping:g} N·s/m on the disc's own stiffness k<sub>disc</sub> = {disc_k}). {gyro_text}
It is <b>skew-symmetric</b> (G<sup>T</sup> = −G), so
q̇<sup>T</sup>(ΩG)q̇ = 0: unlike C, it takes no power out. It only turns the motion from one plane into
the other, and it grows with the speed.</p>
{_mat(C, "C")}<br>{_mat(omega * G, "Ω G")}

<h4>The circulatory matrix Ω H</h4>
<p>The internal damping acts on the bending rate the spinning shaft sees, q̇ − ΩJq in fixed
coordinates (J turns x toward y). The −ΩJq part becomes a stiffness that grows with the speed,
ΩH = Ωη[0 K<sub>shaft</sub>; −K<sub>shaft</sub> 0], skew-symmetric like the cross-coupling in K. A
fluid-film bearing's swirl adds its λc at the journal, where K has k<sub>xy</sub> (here λ =
{s.bearing_a.swirl:g} at A and {s.bearing_b.swirl:g} at B). A skew stiffness pushes a forward whirl
along its orbit: per cycle it puts in energy where C takes it out. Damping falls off as the speed rises,
and past the onset the whirl grows.</p>
{_mat(omega * H, "Ω H")}
{asymmetric}

<h4>State space</h4>
<p>With z = [q, q̇] the equations become z' = A(Ω) z + B u, u = f the force (or moment) on each
DOF, B = [0 ; M<sup>−1</sup>]:</p>
<p align='center'>A(Ω) = [ 0 &nbsp; I ; −M<sup>−1</sup>(K + ΩH) &nbsp; −M<sup>−1</sup>(C + ΩG) ]</p>
<p>A is {2 * s.n_dof} × {2 * s.n_dof} and not symmetric, and its eigenvalues λ = −ζω<sub>n</sub> ± iω<sub>d</sub> come in
conjugate pairs. The modes at this speed, lowest first:</p>
<table border='1' cellspacing='0' cellpadding='3'>
<tr><th>Mode</th><th>λ [1/s]</th><th>f<sub>d</sub> [Hz]</th><th>f<sub>d</sub> [rpm]</th><th>ζ</th>
<th>Whirl ratio</th><th>Whirl</th></tr>
{''.join(rows)}
</table>
<p><small>Whirl ratio: +1 for a circular forward whirl (the same way as the spin, x toward y), −1 for
circular backward whirl, 0 for a straight line; in between, an ellipse. The stations are weighted by
their masses (and the slopes by I<sub>d</sub>). A negative ζ is a mode that grows: the rotor is unstable
at this speed.{" For the asymmetric shaft these are Floquet modes, e<sup>λt</sup> times a motion periodic in φ: each is listed by its harmonic with the most kinetic energy, and a mode locked to 1X sits at exactly the speed." if s.asymmetry else ""}</small></p>

<h4>The 1X forces</h4>
<p>{unbalance} turns with the shaft, the keyphasor mark at angle φ (φ' = Ω). For the heavy spot at the
mark, keeping the disc's centre of mass on its path takes</p>
<p align='center'>f<sub>x</sub> = U(Ω² cos φ + Ω' sin φ), &nbsp; f<sub>y</sub> = U(Ω² sin φ − Ω' cos φ)</p>
<p>on the disc's x<sub>D</sub> and y<sub>D</sub>, or in general f<sub>x</sub> + i f<sub>y</sub> =
Ue<sup>iψ</sup>(Ω² − iΩ')e<sup>iφ</sup>, with ψ the heavy spot's angle from the mark. Balance weights add
to Ue<sup>iψ</sup>. Here the force is {force} N turning once per revolution. Ω' is the ramp's angular
acceleration, while the speed changes.{shaft_weights}</p>
<p>{skew} gives the same form on its slopes, M<sub>x</sub> + i M<sub>y</sub> =
(I<sub>d</sub> − I<sub>p</sub>)τ(Ω² − iΩ')e<sup>i(φ+γ)</sup>: here {moment}.{faces} The shaft's bow δ<sub>b</sub> =
{s.bow * 1e6:.4g} µm (toward {math.degrees(s.bow_angle):.0f}°) gives K<sub>shaft</sub>q<sub>bow</sub>
turning with the shaft whatever the speed: {fmt(s.disc_stiffness * s.bow, 1.0)} N on the disc, and the
bearings' share of it back on the journals, {bow_journals}.</p>
{gravity}
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
follows from these: between the stations a massless beam bends as a cubic. (A shaft with mass needs more:
see <i>The shaft's own mass</i> below.)</p>

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

<h3>The overhung rotor</h3>
<p>Fans, pump impellers and many small turbines carry their disc outside the bearings, on an overhang
(a/L &gt; 1 past bearing B, or below 0 past A). The shaft is then a span between the bearings and a
cantilever beyond, and at the disc it is stiffness 3EI/(c²(L + c)) for an overhang c. Pushed sideways
the overhung disc tilts as much as it moves: in the first mode its slope is about 1/c of its
displacement. So the gyroscopic moment, small for a disc between the bearings, acts on the first mode
in full, and the forward and backward criticals separate strongly.</p>
<p>The <i>Overhung rotor</i> preset puts a 3 kg disc of radius 150 mm 100 mm past bearing B, on
supports of 200,000 N/m. At rest its first mode is 14.5 Hz (870 rpm). Spinning, the forward whirl
stiffens and the backward one softens: 21.4 and 8.6 Hz at 1284 rpm, 28.9 and 3.1 Hz at 5000 rpm. The
backward branch meets the 1X line at 664 rpm and the forward one at <b>1284 rpm</b>, 48% above the
critical speed the same rotor would have without the gyroscopic effect (870 rpm, I<sub>p</sub> = 0), and
above the textbook estimate that ignores the tilt (1046 rpm). The unbalance drives only the forward
crossing, so that is the one critical speed it shows. The disc's conical mode (53 Hz at rest) has a
forward branch rising faster than 1X (I<sub>p</sub> = 2I<sub>d</sub>): it is never critical.</p>

<h3>The Campbell diagram and critical speeds</h3>
<p>Plotting each damped natural frequency against the spin speed gives the Campbell diagram. The
unbalance turns once per revolution, so it drives at f = Ω/2π: the 1X line. Where a whirl frequency
crosses the 1X line, the unbalance drives that mode at its own natural frequency: a <b>critical
speed</b>. In an isotropic rotor the unbalance is a purely forward rotating force, so it drives only
the forward branches; the backward crossings are critical only for an anisotropic rotor (or for other
excitations).</p>
<p>Other excitations come at other multiples of the speed, and the Campbell diagram can draw their
lines (<i>Also draw</i>), as far up in speed as <i>Speeds up to</i> is set. <b>2X</b>: a shaft stiffer
one way than the other (a keyway, a crack) flexes twice per revolution under its own weight, and
misalignment pulls twice per revolution; where a forward branch meets 2X, at about half its critical
speed, such a rotor has a <i>secondary critical</i>. <b>½X</b>: the oil in a plain journal bearing
circulates at about half the shaft's speed, so its instability begins near ½X (below). The model has no
2X source and no ½X source; the lines show where a real machine would have trouble.</p>

<h3>Anisotropic supports and backward whirl</h3>
<p>Supports stiffer vertically than horizontally (common: gravity, foundations, oil-film bearings) split
each mode into a horizontal one and a vertical one, with two critical speeds. The orbits become ellipses.
Between the two criticals the horizontal response already lags by more than 90° while the vertical
one does not yet, and the shaft whirls <b>backward</b>, against its spin, while the unbalance still
turns forward. A shaft whirling backward is bent back and forth once per revolution as it spins:
fatigue loading a forward synchronous whirl does not cause.</p>

<h3>Run-up and coast-down: Bode and polar plots</h3>
<p>On a real machine the response is measured as the speed changes. A <b>keyphasor</b> (a once-per-turn
mark on the shaft passing +x; the heavy spot is at the mark unless you move it) starts each revolution, and a <b>tracking
filter</b> takes the 1X component: over one revolution,</p>
<p align='center'>V = (1/π) ∫ x(φ) e<sup>−iφ</sup> dφ, &nbsp; x ≈ |V| cos(φ − lag)</p>
<p>|V| is the 1X amplitude and the <b>phase lag</b> is the angle the shaft turns from the keyphasor to the
probe's positive peak. The <b>Bode plot</b> draws both against speed; the <b>polar plot</b> draws V as a
point, its angle the lag (clockwise, against the rotation). In steady state, through a lightly damped
critical speed V traces a circle, lowest (lagging 90°) at the critical. A probe in y reads 90° more lag
than one in x on a circular orbit, since the shaft turns 90° more before the high spot reaches it.</p>
<p>The <i>Run up and coast down</i> button ramps to the first speed, waits for the start-up transient to
die away, then runs up to the second speed and back at the ramp rate, tracking each revolution. Held
side by side, sweeps at different rates show:</p>
<ul>
<li><b>The transient peak is lower than the steady-state one, and lower the faster the ramp.</b> The
whirl builds up with the time constant 1/(ζω<sub>c</sub>) of the mode; a ramp at α (rad/s²) crosses
the half-power band 2ζω<sub>c</sub> in 2ζω<sub>c</sub>/α. So the ramp is slow, and reaches the
steady-state peak, only when α/ω<sub>c</sub>² is well below 2ζ². With the defaults (ζ = 0.046,
2ζ² = 0.0043, the peak 217 µm at 1590 rpm, sweeping 800 to 2400 rpm), a run-up at 100 rpm/s
(α/ω<sub>c</sub>² = 0.0004) peaks at 100%, at 500 rpm/s (0.0019) at 95%, at 2000 rpm/s at 78% and at
5000 rpm/s (0.019) at about 60%.</li>
<li><b>The peak comes after the critical speed on a run-up and before it on a coast-down.</b> The
response lags the sweep: at 500 rpm/s the run-up peaks at 1672 rpm (82 rpm late) and the coast-down
at 1498 rpm (92 rpm early); at 5000 rpm/s, some 350 rpm late and 250 to 350 rpm early. (That fast, the
speed changes by 160 rpm in one revolution near the critical, so the tracked curve is coarse: one point
per revolution.) The coast-down peaks lower, 87% of the steady-state peak at 500 rpm/s and about 50% at
5000 rpm/s: its peak comes at a lower speed, where the unbalance force UΩ² is smaller.</li>
<li><b>Beating just past the critical.</b> The whirl built up at the critical speed rings on at the
natural frequency ω<sub>c</sub> as free vibration, while the unbalance keeps driving at Ω. The two beat
at Ω − ω<sub>c</sub>, faster as the speed moves on: the 1X amplitude wobbles after the peak, and the
largest displacement over each revolution (the dots) wobbles more, since it includes the free vibration
in full. On the polar plot the tracked vector loops around the steady-state circle.</li>
<li><b>The parameter that matters is α/ω<sub>c</sub>², against ζ.</b> Measure time in periods of the
critical speed (τ = ω<sub>c</sub>t) and a single mode accelerated through resonance has only two
parameters left: ζ and the dimensionless acceleration α/ω<sub>c</sub>². So the ratio of the transient
peak to the steady-state one, and how far the peak is shifted, depend on these alone. F. M. Lewis
(1932, <i>Vibration during acceleration through a critical speed</i>) first worked this out, and much
later work on passage through resonance builds on it.</li>
</ul>
<p>The summary under the Bode plot gives each sweep's peak against the steady-state one, and how many
rpm away from it the peak came.</p>

<h3>Other 1X sources: a bent shaft and a skewed disc</h3>
<p>Unbalance is not the only thing that turns with the shaft. Anything fixed to the rotor and off its
axis drives it once per revolution, and two more are common.</p>
<p><b>A bent shaft (initial bow).</b> A shaft can keep a bend: from a thermal bow after a hot
shutdown, sagging while it stood, or a bent repair. Say it is δ<sub>b</sub> at the disc, in the
shape a load at the disc would give. Its elastic force is K<sub>shaft</sub>(q − q<sub>bow</sub>), so
the bow acts as a force K<sub>shaft</sub>q<sub>bow</sub> = kδ<sub>b</sub> turning with the shaft. Unlike
the unbalance's meΩ², it does not grow with speed. For the classic Jeffcott rotor,</p>
<p align='center'>(k − mΩ² + icΩ) X = meΩ² + kδ<sub>b</sub>e<sup>iβ</sup></p>
<p>with β the bow's angle from the heavy spot. So:</p>
<ul>
<li><b>At low speed the probes read the bow itself.</b> The shaft just carries its bend round: the
<i>slow-roll runout</i>. The bow's forces on the disc and the journals balance, so the bearings carry
nothing. The 1X response starts at δ<sub>b</sub>, not at zero, and on the polar plot the curve starts
at the bow vector, not at the origin.</li>
<li><b>Well above the critical the disc's whirl dies away.</b> The disc stays on the bearings' axis and
the shaft bends round it to cancel the bow. Contrast the unbalance, whose response tends to e.</li>
<li><b>With an unbalance, the angle between them decides.</b> In phase they add (with the defaults and
δ<sub>b</sub> = 20 µm, the peak doubles to 435 µm). Opposite (β = 180°), the bow's response falls with
speed while the unbalance's grows, and they cancel at one speed, where meΩ² = kδ<sub>b</sub> in the
classic case. Choose δ<sub>b</sub> = e opposite and the disc's centre of mass sits exactly on the
bearings' axis: nothing whirls at any speed, and the probe reads a steady 20 µm, the bow.</li>
</ul>
<p>Measured from its bent shape, the shaft's deflection y = x − δ<sub>b</sub> obeys
(k − mΩ² + icΩ) y = (me + mδ<sub>b</sub>e<sup>iβ</sup>)Ω²: relative to its bow, the shaft sees the
disc's mass carried round on the bow, an unbalance mδ<sub>b</sub>, and the probe adds the bow itself as
runout. So one weight can take out every force the bow makes, at every speed, but not the runout.
That is why the <b>slow-roll vector is subtracted</b> from the 1X readings before balancing (see
<i>Single-plane balancing</i> below).</p>
<p><b>A skewed disc (couple unbalance).</b> A disc mounted out of square has its principal axis tilted
by τ from the shaft's. Its inertia acts on its own axis, so spinning it takes a moment
(I<sub>d</sub> − I<sub>p</sub>)τΩ² that turns with the shaft, the same form as the unbalance force with
U replaced by (I<sub>d</sub> − I<sub>p</sub>)τ. A thin disc (I<sub>p</sub> &gt; I<sub>d</sub>) is pushed
toward square to the spin axis: as Ω → ∞ the shaft bends at the disc by −τ, and the disc spins flat. A
long rotor (I<sub>d</sub> &gt; I<sub>p</sub>) is pushed the other way, and its forward conical mode can
then meet the 1X line. At midspan on identical bearings the skew only tilts the disc; move the disc
off centre and the tilt and the translation couple, so the skew moves the disc too. A <i>static</i>
unbalance (the mass off the axis) can be corrected by one weight in one plane; a <i>couple</i>
unbalance needs two weights in two planes, equal and opposite.</p>

<h3>Gravity on a horizontal rotor</h3>
<p>Laid horizontal, the rotor carries its own weight: the disc's mg bends the shaft, and the bearings'
share of it, with the journals' own weight, compresses the supports. With the defaults the disc sags
378 µm and the journals 118 µm, nineteen times the unbalance's 20 µm eccentricity. The rotor whirls
about that sagged position, so its orbits sit below the bearings' centre: a DC-coupled proximity probe,
which reads the gap, sees the orbit offset by the sag; an AC-coupled one sees only the vibration (the
<i>Static sag</i> box switches the drawing between the two; the tracking, the spectrum and the phase
are always of the vibration).</p>
<p>The model is linear, so the sag is a static deflection q<sub>s</sub> = (K + ΩH)<sup>−1</sup>f<sub>g</sub>
and the vibration about it is exactly what it would be without gravity. In particular gravity alone
makes no whirl, however different the supports' stiffness vertically and horizontally: a constant force
on a linear rotor only moves its equilibrium. (A shaft stiffer one way than the other is the exception:
it bends more under its weight twice per turn, a 2X excitation; see <i>The asymmetric shaft</i> below.
And a fluid-film bearing's stiffness depends on the load it carries.)</p>
<p>The sag need not be straight down. Where the stiffness is not symmetric the weight also pushes
sideways. With cross-coupled bearings, k<sub>xy</sub> = 15,000 N/m, the journals settle 17° from the
vertical in the direction of rotation, much as a fluid-film journal sits at an <i>attitude angle</i>.
With internal damping the shaft turns through its own sag, bent the same way at every angle, and the
damping resists that: with c<sub>i</sub> = 20 N·s/m the disc moves sideways by 14 µm at 1000 rpm,
42 µm at 3000 rpm and 78 µm at 6000 rpm.</p>

<h3>Single-plane balancing</h3>
<p>Balancing puts a weight on the rotor to cancel its unbalance. Where the heavy spot is cannot be seen,
and the phase lag between it and the high spot depends on the speed, so it is found from measurements,
by <b>influence coefficients</b> (the <i>Balancing</i> tab):</p>
<ol>
<li>Turn slowly and read the 1X vector: the <b>slow roll</b> S, runout that no unbalance causes (a bow,
or on a real machine a scratch or a magnetic spot on the probe's track).</li>
<li>Run at the balancing speed and read the vector <b>as found</b>, V<sub>0</sub>.</li>
<li>Stop, add a known <b>trial weight</b> T (mass × radius, at an angle from the keyphasor mark), run
again and read V<sub>1</sub>. The rotor is linear, so the weight's effect is V<sub>1</sub> − V<sub>0</sub>,
and the <b>influence coefficient</b> is α = (V<sub>1</sub> − V<sub>0</sub>)/T: microns, and degrees of
lag, per g·mm of weight.</li>
<li>Take the trial off and put on the <b>correction</b> W = −(V<sub>0</sub> − S)/α, then run and
check.</li>
</ol>
<p>All quantities are vectors (complex numbers): a reading's angle is its phase lag from the keyphasor,
a weight's its angle on the rotor. A trial weight should change the reading by a good fraction (30% or
more) without making the rotor unsafe to run. For an unbalance at the disc alone, the one weight balances
the rotor at every speed, since unbalance and weight both make forces growing as Ω² at the same place.</p>
<p><b>Why the slow roll is subtracted.</b> Take the defaults with a bow of 20 µm at 90°, read at the
disc, balancing at 1200 rpm. The disc reads 20 µm at 100 rpm, 52 µm at 1200 rpm and 293 µm at the
critical. Subtracting the slow roll, the correction is 1.41 g at 20 mm, 225°: it cancels both the
unbalance and the bow's mδ<sub>b</sub>, and from then on the disc reads 20 µm at every speed, the runout
alone, with no force behind it. Balanced on the raw reading instead, the weight (1.95 g at 243°) zeroes
the reading at 1200 rpm by leaving the rotor out of balance: 162 µm at the critical, 41 µm at
3000 rpm. A bowed rotor cannot be made to read zero at every speed with one weight. Taking the slow roll
too fast spoils it a little: at 300 rpm it already holds some unbalance response, and 16 µm is left at
the critical instead of 19.4.</p>
<p><b>What one plane cannot do.</b> A weight in the disc's plane makes a force; a skewed disc makes a
moment. Off centre, both move the disc, so a single weight can cancel the reading at the balancing speed
but not at every speed: with a/L = 0.3 and a skew of 0.5°, balanced at 1200 rpm, 72 µm at the critical
falls only to 5.7 µm, and 9 µm is left at 6000 rpm. A couple unbalance needs <b>two-plane
balancing</b>, with weights in two planes, the influence coefficients forming a 2 × 2 matrix.</p>

<h3>Two-plane balancing</h3>
<p>A rigid body's unbalance is a force and a couple, so cancelling it takes two weights in two planes,
and two probes to see it. The procedure is the single-plane one with vectors (the <i>Balancing</i>
tab, <i>Two planes</i>): read the slow roll S and the readings as found, V<sub>0</sub> (one per probe);
put the trial weight T in plane 1 alone and read V<sub>1</sub>; move it to plane 2 and read
V<sub>2</sub>. Each trial run gives a column of the <b>influence matrix</b>,</p>
<p align='center'>α<sub>ij</sub> = (V<sub>j</sub> − V<sub>0</sub>)<sub>i</sub> / T, &nbsp; and the
correction &nbsp; W = −α<sup>−1</sup>(V<sub>0</sub> − S),</p>
<p>a weight for each plane, with both trials taken off. With more probes than planes the same formula
gives the least-squares weights.</p>
<p><b>Where the planes are matters.</b> The same rotor as above (a/L = 0.3, skew 0.5°, the default
20 g·mm unbalance), probes at the two bearings, balanced at 1200 rpm, read at the disc:</p>
<ul>
<li><b>On the disc's faces</b>, 40 mm apart: 7.32 g at 180° and 6.32 g at 0°, nearly equal and
opposite: mostly a couple. Together they make a force and a moment on the disc, the same form as the
unbalance and the skew, so they cancel both exactly, and the rotor is balanced at <i>every</i> speed:
72 µm at the critical falls to nothing, as does 13.6 µm at 6000 rpm.</li>
<li><b>On the shaft</b>, 50 mm either side of the disc: 3.23 g and 2.22 g, almost what faces 100 mm
apart would need (3.23 g and 2.23 g). But the shaft between them and the disc bends, so they are not
quite a force and a couple on the disc. Both probes read zero
at 1200 rpm, and 2.7 µm is left at the critical, 1.2 µm at 6000 rpm.</li>
<li><b>On the shaft near the bearings</b> (20 and 480 mm): small weights (1.25 g and 0.33 g), since
the planes are far apart, but the forces are now far from the unbalance. The probes read zero at
1200 rpm; at the critical the disc still reads 12.8 µm. A flexible rotor balanced away from its
unbalance is balanced only at the balancing speed (or, with more planes, at more speeds:
<i>modal balancing</i> puts a set of weights against each mode in turn).</li>
</ul>
<p><b>Planes close together are hard to tell apart.</b> A weight in one face of the disc moves the
probes almost as one in the other does, so the columns of α are nearly parallel. Their difference,
which is what the couple acts through, is small, and dividing by it inflates the weights and any error
in the readings. The <b>condition number</b> of α measures this: 27.6 for faces 40 mm apart, 11 at
100 mm, 1.5 for planes near the bearings. With 1% random error in every reading (the simulation's
readings have none), the disc's faces 40 mm apart leave a median of about 18 µm at the critical
instead of 0; 100 mm apart, about 8 µm. Real balancing machines put the planes as far apart as the rotor allows.</p>

<h3>Stability: cross-coupling and internal damping</h3>
<p>Everything so far is forced vibration: the unbalance drives the rotor, and however hard it shakes at
a critical speed, every free whirl dies away. Two effects can instead make a free whirl grow by itself.
Both are stiffnesses that are not symmetric.</p>
<p><b>Cross-coupled stiffness.</b> In a fluid-film bearing (or a seal, or an impeller's clearance) the
fluid is dragged round by the shaft, so pushing the journal in x raises the pressure ahead of it and
pushes it in y too: f<sub>x</sub> = −k<sub>xy</sub>y, f<sub>y</sub> = +k<sub>xy</sub>x. On a circular
forward orbit that force always points along the orbit, the way the journal moves: it does work on the
whirl, 2πk<sub>xy</sub>r² per cycle, where a damper c takes out 2πcωr². So the forward modes lose
damping and the backward ones gain it, and once k<sub>xy</sub> passes cω the forward mode grows. With the
defaults the threshold is k<sub>xy</sub> = 16,400 N/m at both bearings, close to cω = 16,600 N/m for
their 100 N·s/m at the 26.4 Hz whirl (ζ of the forward mode
0.046 → 0.033 → 0.019 → 0.004 at 0, 5,000, 10,000 and 15,000 N/m).</p>
<p><b>Oil whirl and oil whip.</b> In a real fluid-film bearing k<sub>xy</sub> is not constant: the oil
film is dragged round at a fraction λ of the shaft's speed (λ ≈ 0.5 in a plain journal bearing, the
fluid sticking to the turning journal and to the still bearing), and the cross-coupling grows with it,
k<sub>xy</sub> = λcΩ for a bearing of damping c. The energy balance above then reads λcΩ against cω:
the forward whirl at ω grows once <b>λΩ &gt; ω</b>, the oil's swirl outrunning the whirl, whatever the
damping. With λ = 0.5 the rotor goes unstable at <b>twice its first natural frequency</b>. Set λ in the
bearing table, or load the <i>Fluid-film bearings</i> preset: the onset is 3136 rpm, twice the forward
whirl's 26.1 Hz, and just there the whirl is at exactly ½X. Faster, it stays locked at the natural
frequency (26.3 Hz, 0.32×, at 5000 rpm, growing by e every 0.22 s): <i>oil whip</i>, the ratio to the
speed falling as the speed rises. Raising the damping does not help, since it raises the cross-coupling
with it; real machines use bearings that break up the swirl (tilting pads, pressure dams) or lower λ
with anti-swirl devices. A real oil film also stiffens as the journal moves off-centre, so the
whirl settles to a limit cycle of about the clearance; the linear model grows for ever.</p>
<p><b>Internal (rotating) damping.</b> Damping in the shaft itself (material hysteresis, friction in
shrink fits, splines and couplings) resists the bending rate that the spinning shaft sees, not the one
seen from the bearings. In a forward whirl at ω the shaft is bent toward the orbit's centre and that bend
turns at ω while the shaft turns at Ω; from the shaft, the bend moves at ω − Ω. Below the critical
speed (ω &gt; Ω) internal damping damps the whirl like any other damping. Above it the shaft turns faster
than it whirls, the bend moves backward through the shaft, and the internal damping force points
forward along the orbit: it drives the whirl. In fixed coordinates the rotating damper c<sub>i</sub>
gives c<sub>i</sub>q̇ (ordinary damping) plus Ωc<sub>i</sub> times a skew stiffness, ΩH: a cross-coupling
that grows with the speed. For the classic Jeffcott rotor on rigid bearings, with external damping
c<sub>e</sub> at the disc,</p>
<p align='center'>Ω<sub>onset</sub> = ω<sub>n</sub>(1 + c<sub>e</sub>/c<sub>i</sub>)</p>
<p>always above the critical speed, and with no external damping exactly at it. (On flexible supports
only the shaft's share of the motion is damped internally, so the onset rises further.) This was the
cause of the supercritical whirl of early built-up rotors, studied by Kimball and Newkirk in the 1920s.
With the defaults and c<sub>i</sub> = 20 N·s/m the rotor is stable up to 3,910 rpm; 10 N·s/m moves the
onset up to 6,220 rpm, 40 N·s/m down to 2,760 rpm. Below the critical, internal damping helps: at rest the
first mode's ζ rises from 0.046 to 0.078 with c<sub>i</sub> = 20 N·s/m.</p>
<p><b>The stability map</b> (the <i>Stability</i> tab) draws the damping ratio, or the log decrement
δ = 2πζ/√(1 − ζ²), of each mode against speed. The <b>onset speed</b> is where the first forward mode
crosses zero; it is marked on the Campbell diagram too, and the growing modes there are crossed. Industry
practice (API 684) asks for δ ≥ 0.1 at the running speed; the defaults' first mode has δ = 0.29.</p>
<p><b>Beyond the onset</b> a tap, or just the unbalance's start-up transient, sets off a whirl that grows
exponentially at the mode's own natural frequency, not at the speed: a <b>subsynchronous</b> whirl, at
26.9 Hz (0.32×) at 5,000 rpm with c<sub>i</sub> = 20 N·s/m, growing by e every 0.28 s, while the
unbalance still turns at 83 Hz. No balancing removes it. The <b>full spectrum</b> under the stability map
transforms x + iy of the orbit: forward whirl at positive frequencies, backward at negative. The
unbalance shows at +1X, and the instability as a growing line below it on the forward side. The model is
linear, so the whirl would grow for ever; in a machine it grows until something rubs or the bearing's
nonlinearity holds it to a limit cycle. The page pauses when it reaches 2% of the span.</p>

<h3>The fluid-film bearing</h3>
<p>A plain journal bearing has no balls or rollers: the journal turns in a sleeve a little larger than
itself, the gap (the radial clearance c, a thousandth or two of the radius) full of oil. Turning, the
journal drags the oil round with it into the narrowing side of the gap, and the oil, squeezed, rises in
pressure and holds the journal off the sleeve. Reynolds' equation gives that pressure; in a bearing short
beside its diameter (L/D up to about ½) the oil escapes mostly out of the ends, and the pressure is a
parabola along the bearing, <b>short-bearing</b> (Ocvirk) theory. Where it would go below zero the oil
cavitates instead, so only the half of the film that narrows carries load (the π-film).</p>
<p>At rest the journal lies on the bottom of its clearance. As the speed rises the film lifts it, and,
since the film's pressure peaks ahead of the thinnest point, it pushes the journal <b>sideways</b>, in
the direction of spin, as well as up. The journal settles off the load line by the <b>attitude angle</b>
at an <b>eccentricity</b> ε = e/c that depends only on the Sommerfeld number
S = (μNLD/W)(R/c)², N in rev/s: the more speed and viscosity, or the less load, the larger S, and the
closer the journal to the centre, at nearly 90° to the load. On the <i>Fluid-film bearing</i> tab, the
default bearing (D = 10 mm, L = 5 mm, c = 15 µm, oil of 30 mPa·s, 5 N) has ε = 0.80 at 100 rpm, 0.44 at
1,000 rpm and 0.22 at 3,000 rpm, its thinnest film 11.7 µm there. Every short bearing travels the same
path, the <b>locus</b>, only at a different pace.</p>
<p>For small motions about that point the film acts as springs and dampers, eight coefficients:
f = −KΔq − CΔq̇, with K = (W/c)·a(ε) and C = (W/cΩ)·b(ε). The film's sideways push makes K far from
symmetric: at 3,000 rpm, k<sub>xy</sub> = 1.36 MN/m but k<sub>yx</sub> = −1.81 MN/m. Its skew part is
the cross-coupling of the <i>Stability</i> section, and for the short bearing it is exactly
k<sub>xy</sub> − k<sub>yx</sub> = ½Ω(c<sub>xx</sub> + c<sub>yy</sub>), at every load: the oil swirls
at half the shaft's speed, the swirl ratio λ = ½ of the bearing table. As the speed rises the
cross-coupling grows with it while the direct stiffness and damping level off.</p>
<p>So a rotor on such bearings has a <b>threshold speed</b>. A rigid rotor of mass m per bearing
(Lund's formula) whirls at ν with mν² = K<sub>eq</sub> at the threshold, where
K<sub>eq</sub> = (k<sub>xx</sub>c<sub>yy</sub> + k<sub>yy</sub>c<sub>xx</sub> − k<sub>xy</sub>c<sub>yx</sub>
− k<sub>yx</sub>c<sub>xy</sub>)/(c<sub>xx</sub> + c<sub>yy</sub>). Lightly loaded, ν is half the speed:
<b>oil whirl</b>. With the default bearing carrying its load as weight (0.51 kg), the rotor goes unstable
at 21,324 rpm and whirls at 0.50×. Heavily loaded bearings (ε above 0.756) are stable at any speed,
which is one reason a lightly loaded bearing is the one that whirls. A flexible rotor goes unstable much
sooner, near twice its first critical speed, which is the oil whip of the <i>Stability</i> section.</p>
<p>The tab draws the locus, with the page's speed range solid and the speed now as a dot, and the
coefficients against the speed, or made dimensionless against S on log scales (dashed where negative),
as on a bearing designer's chart. For now the rotor's supports are still the bearing table's springs and
dampers; putting the film on the rotor, its coefficients changing with speed, is the next step.</p>

<h3>The asymmetric shaft: a two-pole rotor</h3>
<p>The rotor of a two-pole generator has slots for its winding cut along two sides, so it is stiffer
bending one way than the other; a keyway or a flat does the same on a smaller scale (and a crack, that
opens and closes, is a cousin). Set the <i>Asymmetry</i> ±Δ in the <i>Shaft and disc</i> box: bending
toward its stiff axis the shaft has EI(1 + Δ), across it EI(1 − Δ). The angle is the stiff axis's from
the keyphasor mark, drawn as the dashed diameter across the disc in the 3D view. That stiffness turns with
the shaft, so seen from the bearings it changes twice per revolution:
K(φ) = K + ΔK(2(φ + α)). The equations have a periodic coefficient, and three things follow.</p>
<p><b>Two critical speeds, and an unstable band between them.</b> Seen from axes turning with the shaft
the stiffness stands still (and so does everything else, on isotropic supports), but the centrifugal
force −mΩ² and the Coriolis force join in. To hold a deflection along an axis the shaft needs more
stiffness there than mΩ². Below the soft axis's critical speed it has it both ways. Between the two
axes' critical speeds the centrifugal force beats the soft axis but not the stiff one, so a bend along
the soft axis grows. It turns with the shaft, so from the bearings it is a forward whirl locked at
exactly 1X, growing. Undamped, that band runs exactly between the critical speeds of round shafts as
stiff as each axis: with Δ = 0.15 and undamped supports, 1479 to 1645 rpm. Damping narrows it, and
below a threshold asymmetry there is none: Δ = 0.123 with the defaults' 100 N·s/m supports, 0.178 with
400 N·s/m. Unlike the instabilities above, it ends: past the stiff axis's critical speed the rotor is
stable again. Choose <i>Asymmetry chart</i> on the <i>Stability</i> tab to map it against Δ and the
speed. It is an Ince–Strutt chart, like a Mathieu equation's, with one tongue between the two dashed
critical speeds, its tip lifted off Δ = 0 by the damping. Load the <i>Two-pole rotor</i> preset
(Δ = 0.15): its critical speeds are 1491 and 1665 rpm, and it is unstable from 1527 to 1631 rpm. Tap it
at 1580 rpm and the whirl grows ×e every 0.56 s at 26.3 Hz: the speed, not a natural frequency.</p>
<p><b>The gravity critical.</b> Laid horizontal, the shaft's weight is a constant force on a stiffness
that turns, so the rotor never settles. Its deflection swings twice per revolution about a mean sag:
a <b>2X</b> whirl, largest where 2Ω meets a natural frequency, near <b>half the critical speed</b>. The
preset's disc whirls 447 µm at 2X at 785 rpm, more than its mean sag of 384 µm. On the Bode plot, choose
<i>2X</i> beside the probe: run up and coast down from 500 to 1200 rpm at 200 rpm/s and they reach 93%
and 94% of the steady-state peak. A 2X peak at half the first critical speed on a run-up is the classic
signature of a two-pole generator (or a cracked shaft). Balancing does nothing to it.</p>
<p><b>The unbalance angle.</b> With an asymmetric shaft the 1X response depends on where the heavy spot
sits from the stiff axis, not only on its size. With Δ = 0.10, the supports' damping holding it stable,
and the 20 g·mm unbalance: along the stiff axis the disc's orbit peaks at 827 µm (1589 rpm), along the
soft axis at 848 µm (1576 rpm), but at 45° between them it has two lower peaks, 201 µm at 1547 rpm and
195 µm at 1624 rpm. A round shaft peaks at 217 µm. So the response is no longer a complex number times
the unbalance: turn the heavy spot by 90° and the response does not turn by 90°. Influence coefficients
(see <i>Balancing</i>) then hold only approximately.</p>
<p><b>How it is worked out.</b> In axes turning with the shaft on isotropic supports the rotor is
time-invariant, so its modes are exact: each is seen from the bearings as a forward and a backward
circle, at μ ± iΩ, and is listed at the one with more kinetic energy. On anisotropic supports the
modes come from Hill's method, the Floquet exponents of the periodic system from its harmonics up to
±3 × 2Ω. The steady states (the unbalance's 1X, gravity's mean sag and 2X) come from harmonic balance:
the turning stiffness ties each harmonic k to k ± 2, which closes at once on isotropic supports (1X stays
1X; the sag gives a forward 2X) and is cut at ±6X otherwise. The time stepping splits each step: half a
step of the rest (the exact update), the kick of the turning stiffness at mid-step on the velocities,
and the other half step. It agrees with the harmonic balance to about 10<sup>−5</sup>.</p>

<h3>Waterfall and cascade plots</h3>
<p>A spectrum analyser on a machine takes the spectrum of a block of signal, then of the next, and so
on. The <i>Waterfall</i> tab does the same with the full spectrum of the orbit at a station: blocks of
0.5 s by default, each starting halfway through the last. Stacked against time the spectra make a
<b>waterfall</b> plot; against the speed during a run-up or coast-down, a <b>cascade</b>. The
unbalance makes a ridge along +1X that moves with the speed and peaks at the critical. Whatever sets
the rotor ringing (the passage through the critical, a tap, an instability) shows at the free whirl
frequencies, which stay put, or move only as the gyroscopic effect moves them, while the 1X ridge
sweeps past. That is how a natural frequency is told from a forced response on a running machine.</p>
<p><b>A measured Campbell diagram.</b> Tap the disc every half second through a run-up and each tap
rings both whirls of each mode at the speed it came at, so the peaks trace the Campbell diagram's
branches. On the <i>Overhung rotor</i> preset, from rest to 2,500 rpm, the first mode's backward whirl
falls from 14.5 to 5.7 Hz and its forward whirl rises to 25.5 Hz. The 1X ridge grows where it crosses
the forward branch, at the critical speed (1284 rpm), and peaks just after it, as on the Bode plot. On a full spectrum the backward branch lies on the
other side of zero, so the +1X ridge never meets it: the backward crossing on the Campbell diagram (664
rpm) is not a critical for an isotropic rotor.</p>
<p><b>Block length.</b> A block of T seconds resolves frequencies 1/T apart: 2 Hz for 0.5 s. A longer
block resolves more finely, but while the speed ramps at α it changes by αT over the block, smearing
the 1X ridge: 100 rpm (1.7 Hz) at 200 rpm/s, but 2500 rpm (42 Hz) at 5000 rpm/s, where the ridge is a
smear. Analysers get round this by <i>order tracking</i>: sampling at fixed shaft angles, from the
keyphasor, so 1X stays in one bin whatever the speed. The Bode plot's once-per-revolution 1X vector is
that kind of measurement.</p>

<h3>The shaft's own mass: beam finite elements</h3>
<p>Jeffcott's shaft has no mass: it only holds the disc. That keeps the model small, because a massless
shaft bends as a cubic between the stations, so four DOFs in each plane describe it exactly. Tick
<i>Shaft mass</i> and the shaft gets its mass, spread along it, and its sections' rotary inertia (and
their gyroscopic moment). Its shape is then no longer set by the stations alone, so it is cut into beam
<b>finite elements</b>: each bends as a cubic between two nodes, and each node has a displacement and a
slope in each plane.</p>
<p>A light shaft changes little. The defaults' steel shaft weighs 0.31 kg beside the 1 kg disc, and
takes the first critical speed from 1584 to 1458 rpm. The old rule adds 17/35 of a simply supported
shaft's mass to the disc (Rayleigh's method, with the shaft's static deflected shape), which gives 1477
rpm. But a shaft has bending modes of its own, and a heavier one brings them down into the speed range.
In the <i>Long shaft</i> preset (a 2 kg disc on a 1 m, 20 mm shaft of 2.47 kg) the critical speeds
are 1589 and 6582 rpm. The massless model puts the first at 2146 rpm, 35% high, and has no second:
its next modes are the journals bouncing on their supports, heavily damped. The Campbell diagram draws
the massless model's branches dashed beside the real ones. Gravity shows it too: the disc sags 375 µm,
not the massless model's 200 µm.</p>
<p><b>Consistent or lumped mass.</b> The consistent mass matrix spreads each element's mass with the
same cubic shapes as its stiffness. It is a Rayleigh–Ritz model, so at rest and undamped its frequencies
are never too low, and they come down as elements are added: on the <i>Long shaft</i> the second
critical comes out at 7393, 6591 and 6582 rpm with 2, 4 and 8 elements. A lumped mass matrix, half of
each element's mass at each end, is simpler but has no such bound. Here it gives 8187, 6733 and 6620 rpm:
with few elements much of the mass sits on the bearings, where it hardly moves.</p>
<p><b>Keeping it fast.</b> Every element adds modes. The highest are at tens of kHz, far above
anything the rotor does at these speeds, but the time stepping would still have to resolve them. So the
model is reduced the way the <i>Substructuring</i> page reduces a substructure (Craig–Bampton). The
stations' DOFs are kept as they are. The rest of the shaft moves in the static shape they give it, which
is all a massless shaft ever does, plus its lowest modes with the stations held, up to 1000 Hz. Those
modes' amplitudes ξ join q, after the stations. The reduced model's modes are within 0.2% of the full
element model's below 500 Hz, and within 1% up to 1000 Hz (60,000 rpm, the top of the speed range).
With a massless shaft there are no such modes, and the reduction is exactly the Jeffcott model.</p>

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
<li>Run up and coast down from 800 to 2400 rpm at 500 rpm/s, then again at 5000 rpm/s with <i>Hold
earlier sweeps</i> ticked, and compare them with the steady-state curve on the Bode and polar plots.
Then raise the support damping to 400 N·s/m (ζ = 0.089, and the peak moves up to 1758 rpm): at
500 rpm/s the sweeps now come within a few percent of the steady-state peak (101% up, 96% down), since
2ζ² has grown fourfold.</li>
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
<li>Set the internal damping c<sub>i</sub> to 20 N·s/m and look at the <i>Stability</i> tab: the forward
first mode's damping falls with speed and crosses zero at 3,910 rpm, while the backward mode's rises. Ramp
to 5,000 rpm and tap the disc: the whirl grows at 26.9 Hz, a line at 0.32× on the full spectrum, until
the page pauses it. Lower the target to 3,000 rpm and Reset: below the onset the tap dies away.</li>
<li>Set c<sub>i</sub> back to 0 and give both bearings k<sub>xy</sub> = 15,000 N/m: the forward mode's
ζ drops to 0.004 at every speed and the backward one's rises to 0.075. Above 16,400 N/m the rotor is
unstable even at rest. More support damping raises the threshold.</li>
<li>Set the unbalance to 0 and the shaft bow to 20 µm. Turning slowly the disc's orbit is 20 µm
round, the bow; it peaks at 218 µm at the critical and falls to 8 µm at 3000 rpm (towards zero, not
towards e). Run up and coast down and look at the polar plot: it starts at the bow vector.</li>
<li>Put the 20 g·mm unbalance back and set the bow's angle to 180°: the disc's orbit stays at 20 µm at
every speed. Now set the bow to 10 µm: the orbit dips to about 1 µm at 1116 rpm, where the bow and the
unbalance cancel, and still peaks at 110 µm at the critical.</li>
<li>Set the unbalance to 0 and skew the disc by 0.1°. At midspan only the disc tilts (watch it wobble in
the 3D view); move it to a/L = 0.3 and the disc whirls too, peaking at 14 µm at 1752 rpm.</li>
<li>On the <i>Balancing</i> tab, press <i>Mystery unbalance</i> and balance the rotor in four steps:
slow roll, as found, trial run, correct and check. The check should read well under 1% of the
reading as found; <i>Reveal</i> then shows the unbalance you cancelled.</li>
<li>Set a bow of 20 µm at 90°, put the probe on the disc, and balance at 1200 rpm with and without
<i>Subtract slow roll</i>: compare the curves as found and now under the vectors.</li>
<li>Move the disc to a/L = 0.3, choose <i>Two planes: the disc's faces</i> on the <i>Balancing</i> tab
and press <i>Mystery unbalance</i>: now the disc is skewed too, by a hidden angle. Balance it in five
steps; the curve <i>Now</i> stays flat at every speed. Then try <i>Two planes on the shaft</i>, near
the bearings, and see the critical come back.</li>
<li>Tick <i>Gravity</i>: the rotor sags, 378 µm at the disc, and whirls about the sagged position,
the orbits sitting below the bearings' centre. Untick <i>Static sag</i> to look at the vibration alone:
it is the same as without gravity. Make the supports anisotropic with no unbalance: the rotor sags,
spins up and stays still.</li>
<li>Load the <i>Overhung rotor</i> preset and look at the Campbell diagram: the first mode's forward and
backward branches fan out from 14.5 Hz. Run up and coast down from 600 to 2000 rpm: one critical, at
1284 rpm. Then set the disc's polar inertia I<sub>p</sub> to a tiny value: with the gyroscopic moment
gone the critical drops to 870 rpm, the natural frequency at rest.</li>
<li>Load the <i>Fluid-film bearings</i> preset (λ = 0.5) and tick ½X on the Campbell diagram. Run at
2500 rpm and tap the disc: the tap dies away. Set the target to 4000 rpm (past the onset, 3136 rpm,
where ½X crosses the first mode) and tap again: the whirl grows at 26 Hz, 0.39× on the full spectrum, not
at ½X. Double the support damping: the onset does not move.</li>
<li>Load the <i>Overhung rotor</i> preset, set the ramp rate to 200 rpm/s and tick <i>Tap the disc
every</i> 0.5 s, then set the target to 2,500 rpm and watch the <i>Waterfall</i> tab: the taps' peaks
follow the dots of the forward and backward branches (backward from 14.5 down to 5.7 Hz), and the 1X ridge
grows where it crosses the forward branch. Switch to <i>Time</i> and set a steady speed: the taps' peaks
then stay at one frequency.</li>
<li>Load the <i>Long shaft</i> preset and run up from 800 to 8000 rpm at 1000 rpm/s. Past 6582 rpm the
shaft bends in its second mode, the one the dashed massless model doesn't have. Then set <i>Beam
elements</i> to 2 and to 4, consistent and lumped, and watch the second critical move on the Campbell
diagram.</li>
<li>Load the <i>Two-pole rotor</i> preset, choose <i>Orbit</i> and <i>2X</i> beside the probe, and run up
and coast down from 500 to 1200 rpm at 200 rpm/s: the 2X peaks at 785 rpm, half the critical speeds.
On the <i>Stability</i> tab choose <i>Asymmetry chart</i>, set the target to 1580 rpm and tap the disc:
the dot sits inside the tongue and the whirl grows at exactly 1X. Raise the support damping to 400 N·s/m
(both directions) and the band is gone: the threshold rises to Δ = 0.178.</li>
<li>Untick <i>Horizontal rotor</i> on the <i>Two-pole rotor</i> preset, set the asymmetry to 10% and the
unbalance to 20 g·mm, and look at the disc's orbit on the Bode plot with the heavy spot at 0°, 45° and
90°: one tall peak, two short ones, one tall one.</li>
<li>Pause near the critical speed and turn the 3D view: the shaft bows in one plane, which turns with
the spin, rather than flapping back and forth.</li>
</ol>

<h3>Notation</h3>
<table border="1" cellspacing="0" cellpadding="4">
<tr><th>Symbol</th><th>Meaning</th><th>Units</th></tr>
<tr><td>L, a, b, c</td><td>bearing span; disc's distance from A and from B (a + b = L); an overhung disc's distance past its bearing</td><td>m</td></tr>
<tr><td>d, E, EI</td><td>shaft diameter, Young's modulus, bending stiffness E π d⁴/64</td><td>m, Pa, N·m²</td></tr>
<tr><td>m, I<sub>p</sub>, I<sub>d</sub></td><td>disc mass; its polar and diametral moments of inertia</td><td>kg, kg·m²</td></tr>
<tr><td>k<sub>x</sub>, k<sub>y</sub>, c<sub>x</sub>, c<sub>y</sub></td><td>a support's stiffness and damping, horizontal and vertical</td><td>N/m, N·s/m</td></tr>
<tr><td>x, y</td><td>horizontal and vertical displacement of the shaft's centre</td><td>m</td></tr>
<tr><td>θ<sub>x</sub>, θ<sub>y</sub></td><td>shaft's slope at the disc, dx/dz and dy/dz</td><td>rad</td></tr>
<tr><td>Ω, φ</td><td>spin speed; angle of the keyphasor mark on the shaft from +x (φ' = Ω)</td><td>rad/s (rpm), rad</td></tr>
<tr><td>U = m e, ψ</td><td>unbalance: disc mass times its centre of mass's offset e; the heavy spot's angle from the mark</td><td>kg·m (g·mm), rad</td></tr>
<tr><td>M, C, G, K</td><td>mass, damping, gyroscopic and stiffness matrices (8 × 8 for a massless shaft)</td><td>—</td></tr>
<tr><td>ρ, ξ</td><td>the shaft's density; the amplitudes of its modes with the stations held (mass normalized)</td><td>kg/m³, √kg·m</td></tr>
<tr><td>g, f<sub>g</sub>, q<sub>s</sub></td><td>gravity; the weights on the journals and the disc; the static deflection under them</td><td>m/s², N, m</td></tr>
<tr><td>A(Ω), B</td><td>state matrix (16 × 16 for a massless shaft) and input matrix</td><td>—</td></tr>
<tr><td>ω<sub>c</sub></td><td>critical speed</td><td>rad/s (rpm)</td></tr>
<tr><td>ζ</td><td>damping ratio of a mode, −Re λ / |λ|</td><td>—</td></tr>
<tr><td>α</td><td>angular acceleration of a ramp, Ω'</td><td>rad/s² (rpm/s)</td></tr>
<tr><td>V</td><td>1X vector: amplitude |V|, phase lag −arg V</td><td>m</td></tr>
<tr><td>k<sub>xy</sub></td><td>a bearing's cross-coupled stiffness (k<sub>yx</sub> = −k<sub>xy</sub>)</td><td>N/m</td></tr>
<tr><td>λ</td><td>a fluid-film bearing's swirl ratio: k<sub>xy</sub> grows by λcΩ, c the mean of c<sub>x</sub> and c<sub>y</sub></td><td>—</td></tr>
<tr><td>D, L, c, μ, W</td><td>on the <i>Fluid-film bearing</i> tab: journal diameter, bearing length, radial clearance, oil viscosity, load</td><td>m, m, m (µm), Pa·s, N</td></tr>
<tr><td>ε, S</td><td>the journal's eccentricity (its offset over c); the Sommerfeld number (μNLD/W)(R/c)², N in rev/s</td><td>—</td></tr>
<tr><td>c<sub>i</sub>, η</td><td>internal (rotating) damping of the shaft, as a damper at the disc; η = c<sub>i</sub>/k<sub>disc</sub></td><td>N·s/m, s</td></tr>
<tr><td>H</td><td>circulatory matrix: the skew stiffness per unit speed, from internal damping and fluid swirl</td><td>N·s/m</td></tr>
<tr><td>δ</td><td>log decrement, 2πζ/√(1 − ζ²)</td><td>—</td></tr>
<tr><td>δ<sub>b</sub>, β</td><td>shaft bow at the disc, and its angle from the keyphasor mark</td><td>m (µm), rad</td></tr>
<tr><td>T, W, α</td><td>trial and correction weights (mass × radius at an angle); influence coefficient (V<sub>1</sub> − V<sub>0</sub>)/T</td><td>kg·m (g·mm); m per kg·m</td></tr>
<tr><td>τ, γ</td><td>disc skew (its axis's tilt from the shaft's), and its angle from the keyphasor mark</td><td>rad (°)</td></tr>
<tr><td>Δ, α</td><td>shaft asymmetry (EI(1 ± Δ) about its stiff and soft axes), and the stiff axis's angle from the keyphasor mark</td><td>—, rad (°)</td></tr>
<tr><td>ΔK(φ)</td><td>the asymmetric shaft's stiffness less its mean, turning with it</td><td>N/m</td></tr>
</table>
"""
