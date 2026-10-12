"""Text for the Jeffcott rotor page: the theory notes and the live equations and matrices."""

from __future__ import annotations

import cmath
import math
from collections.abc import Collection

import numpy as np

from ..core.rotor import DOF_NAMES, MODE_LIMIT, RPM, XD, YD, RotorSystem, periodic_steady, whirl_modes
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


def _coupling(s: RotorSystem) -> str:
    """Each bearing's constant k_xy, or that its coupling is an oil film's."""
    return " and ".join("the oil film's (below) at " + n if b.film is not None else f"{fmt(b.kxy, 1.0)} N/m at {n}"
                        for n, b in (("A", s.bearing_a), ("B", s.bearing_b)))


def _films_html(s: RotorSystem, omega: float) -> str:
    """The oil films' paragraph: their loads, equilibria and coefficients at speed Ω (empty without films)."""
    if not s.films:
        return ""
    parts = []
    for name, b, w, st in zip("AB", (s.bearing_a, s.bearing_b), s.bearing_loads, s.bearing_states(omega)):
        if st is None:
            continue
        k, c = st.k, st.c
        parts.append(
            f"at {name}, under {fmt(w, 1.0)} N, ε = {st.eccentricity:.3f} at {math.degrees(st.attitude):.1f}° from the "
            f"load line: k<sub>xx</sub>, k<sub>xy</sub>, k<sub>yx</sub>, k<sub>yy</sub> = {fmt(k[0, 0], 1.0)}, "
            f"{fmt(k[0, 1], 1.0)}, {fmt(k[1, 0], 1.0)}, {fmt(k[1, 1], 1.0)} N/m and c<sub>xx</sub>, "
            f"c<sub>xy</sub> = c<sub>yx</sub>, c<sub>yy</sub> = {fmt(c[0, 0], 1.0)}, {fmt(c[0, 1], 1.0)}, "
            f"{fmt(c[1, 1], 1.0)} N·s/m")
    held = max(abs(omega), min(b.film.with_load(abs(w)).slowest() for b, w in
                               zip((s.bearing_a, s.bearing_b), s.bearing_loads) if b.film is not None))
    slow = (f" Below {held * RPM:,.0f} rpm the journal is nearly on its bottom, and the film is taken as it is "
            "there (ε = 0.9)." if held > abs(omega) else "")
    return (
        "<h4>The oil films</h4><p>The journals run in plain journal bearings (short-bearing theory; see the "
        "<i>Fluid-film bearing</i> tab) in place of the springs and dampers. Each is linearized about where its "
        "journal settles under its share of the weight, f = W − K<sub>f</sub>(q − q<sub>J</sub>) − "
        "C<sub>f</sub>q̇: K<sub>f</sub> and C<sub>f</sub> are in K and C below, and the constant "
        "p = W + K<sub>f</sub>q<sub>J</sub> joins f, so the rotor rests with its journals at q<sub>J</sub>. "
        "Both change with the speed. K<sub>f</sub> is not symmetric, and its skew part is the oil's swirl at "
        "half the speed, k<sub>xy</sub> − k<sub>yx</sub> = ½Ω(c<sub>xx</sub> + c<sub>yy</sub>). At this speed "
        + "; ".join(parts) + "." + slow
        + (" The time response takes the films' whole force instead, the remainder r<sub>f</sub> beyond this "
           "linearization stepped as a load on the journals (<i>Theory</i>, Numerics)." if s.whole_films else "")
        + "</p>"
    )


def matrices_html(system: RotorSystem, omega: float, hidden: Collection[str] = ()) -> str:
    """The equations of motion, every matrix with the current numbers, and the modes at speed Ω.

    `hidden` names the RotorSystem fields a balancing exercise keeps out of sight: the
    unbalance's size and angle, and perhaps the skew's.
    """
    M, C, G, K = system.matrices()
    H = system.circulatory()
    s = system
    labels, groups = _labels(s)
    films = _films_html(s, omega)
    if s.films:
        Kf, Cf, _ = s._film_terms(omega)
        K, C = K + Kf, C + Cf

    def _mat(A: np.ndarray, title: str) -> str:
        return _matrix(A, labels, groups, title)

    ks = s.shaft_stiffness()
    kept = s.modes_kept
    one = ["x<sub>A</sub>", "x<sub>D</sub>", "θ<sub>x</sub>", "x<sub>B</sub>"]
    one += [f"ξ<sub>{k}</sub>" for k in range(1, kept + 1)]
    shaft = _matrix(ks, one, ["i"] * len(one), "K<sub>shaft</sub>")
    modes = whirl_modes(system, omega)
    rows = []
    for r, (f, z, w) in enumerate(zip(modes.freq_hz, modes.zeta, modes.whirl)):
        eig = complex(-z * 2 * math.pi * f / math.sqrt(max(1 - z * z, 1e-12)), 2 * math.pi * f)
        rows.append(
            f"<tr><td align='center'>{r + 1}</td><td align='right'>{eig.real:.4g} + {eig.imag:.5g}i</td>"
            f"<td align='right'>{f:.4g}</td><td align='right'>{f * 60:,.0f}</td><td align='right'>{z:.3g}</td>"
            f"<td align='right'>{w:+.2f}</td><td>{whirl_name(w)}</td></tr>"
        )
    rpm = omega * RPM
    if s.asymmetry:
        Kc, _ = s.deviatoric
        asymmetric = (
            f"<h4>The asymmetric shaft ΔK(φ)</h4><p>The shaft is stiffer bending toward its stiff axis, at "
            f"φ<sub>s</sub> = φ + β<sub>s</sub> from x (β<sub>s</sub> = {math.degrees(s.asymmetry_angle):.0f}° from the mark), than "
            f"across it: "
            f"EI(1 ± Δ) with Δ = {s.asymmetry:.3g}. Its stiffness less the mean (K<sub>shaft</sub>, in K) is, "
            "in each plane pair,</p><p align='center'>ΔK(φ) = Δ [cos 2φ<sub>s</sub> K<sub>shaft</sub> &nbsp; "
            "sin 2φ<sub>s</sub> K<sub>shaft</sub> ; sin 2φ<sub>s</sub> K<sub>shaft</sub> &nbsp; −cos 2φ<sub>s</sub> "
            "K<sub>shaft</sub>]</p><p>or, as f<sub>x</sub> + i f<sub>y</sub>, "
            "−ΔK<sub>shaft</sub>e<sup>2iφ<sub>s</sub></sup>(x − iy). It turns with the "
            "shaft, twice per revolution as seen from the bearings, so the equations have a periodic "
            "stiffness and no fixed modes. Here it is with the stiff axis along x (φ<sub>s</sub> = 0):</p>"
            + _matrix(Kc, labels, groups, "ΔK(φ<sub>s</sub> = 0)")
            + "<p>In coordinates turning with the shaft, q = R(φ)q<sub>r</sub>, it is constant, and so is "
            "everything else if the supports are isotropic: M q̈<sub>r</sub> + (C + ΩG + 2ΩMJ) q̇<sub>r</sub> + "
            "(K + ΩH + ΔK(0) − Ω²M + ΩCJ + Ω²GJ) q<sub>r</sub> = R<sup>T</sup>f. The −Ω²M is the centrifugal force: between the two critical speeds it "
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
    cracked = _crack_note(s) if s.crack is not None else ""
    if s.overhung:
        near, far = ("B", "A") if s.a > s.length else ("A", "B")
        elements = (f"Two Euler–Bernoulli beam elements: the span between the bearings (L = {s.length:.4g} m) and "
                    f"the overhang from bearing {near} to the disc (ℓ = {s.overhang:.4g} m), with EI = {s.ei:.4g} N·m². "
                    f"The disc sits at a = {s.a:.4g} m from A, outside the bearings.")
        disc_k = f"3EI/(ℓ²(L + ℓ)) = {s.disc_stiffness:,.0f} N/m for the overhung disc"
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
        swirl = (f" With fluid swirl the cross-coupling grows with the speed, k<sub>xy</sub> + λcΩ (c the mean of "
                 f"c<sub>x</sub> and c<sub>y</sub>), and that part "
                 f"sits in ΩH (below): at this speed k<sub>xy</sub> is {fmt(kxy_a, 1.0)} N/m at A and "
                 f"{fmt(kxy_b, 1.0)} N/m at B in all.")
    else:
        swirl = ""
    fb = ks @ s.bow_shape() * s.bow
    bow_journals = f"{fmt(-fb[0], 1.0)} N at A and {fmt(-fb[3], 1.0)} N at B"
    if s.gravity and s.periodic:
        qs, qk = (x[0] for x in periodic_steady(s, [omega], gravity=True, rotating=False, top=3))
        breathing = s.spacing == 1
        gravity = (
            f"<h4>Gravity</h4><p>On a horizontal rotor f also holds the weights, constant: "
            f"−m<sub>A</sub>g, −mg and −m<sub>B</sub>g on y<sub>A</sub>, y<sub>D</sub> and y<sub>B</sub> "
            f"(here {fmt(s.disc_mass * s.gravity, 1.0)} N on the disc)"
            + (f", and the shaft's own weight, {fmt(s.shaft_mass * s.gravity, 1.0)} N" if s.density else "")
            + ". On the turning shaft a constant force meets a stiffness that turns, so the rotor never "
            "settles: its steady state is periodic, "
            + ("q = Q<sub>0</sub> + Re(Q<sub>1</sub>e<sup>iφ</sup> + Q<sub>2</sub>e<sup>2iφ</sup> + "
               "Q<sub>3</sub>e<sup>3iφ</sup> + ...), found by harmonic balance (each harmonic n of the response "
               "is tied to n ± 1, n ± 2 and n ± 3 by the breathing crack's ΔK, up to ±6X)" if breathing else
               "q = Q<sub>0</sub> + Re(Q<sub>2</sub>e<sup>2iφ</sup>)"
               + ("" if s.rotating_frame else " + (4X, ...)") + ", found by harmonic balance (each "
               "harmonic n of the response is tied to n ± 2 by ΔK)")
            + f". The mean sag Q<sub>0</sub> puts the disc at ({qs[XD].real * 1e6:.4g}, {qs[YD].real * 1e6:.4g}) "
            "µm; about it the disc whirls at "
            + (", ".join(f"{k + 1}X {abs(qk[k, XD]) * 1e6:.3g} µm in x and {abs(qk[k, YD]) * 1e6:.3g} µm in y"
                         for k in range(3)) + " at this speed (on top of the unbalance's 1X), 2X most where 2Ω "
               "meets a natural frequency and 3X where 3Ω does." if breathing else
               f"2X, {abs(qk[1, XD]) * 1e6:.4g} µm in x and {abs(qk[1, YD]) * 1e6:.4g} µm in y at this speed, "
               "most where 2Ω meets a natural frequency: the gravity critical.") + "</p>"
        )
    elif s.gravity:
        qs = s.static_deflection(omega)
        gravity = (
            f"<h4>Gravity</h4><p>On a horizontal rotor f also holds the weights, constant: "
            f"−m<sub>A</sub>g, −mg and −m<sub>B</sub>g on y<sub>A</sub>, y<sub>D</sub> and y<sub>B</sub> "
            f"(here {fmt(s.disc_mass * s.gravity, 1.0)} N on the disc)"
            + (f", and the shaft's own weight, {fmt(s.shaft_mass * s.gravity, 1.0)} N, spread over the stations "
               f"and its modes" if s.density else "") + f". They hold the rotor at the static "
            + ("deflection q<sub>s</sub> = (K + ΩH)<sup>−1</sup>(f<sub>g</sub> + p), p the films' load (above), "
               if s.films else "deflection q<sub>s</sub> = (K + ΩH)<sup>−1</sup>f<sub>g</sub>, ") + f"here the disc at "
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
                         f"the DOFs as a load at z is, Ψ(z)<sup>T</sup>.")
    else:
        shaft_weights = ""
    if "skew" in hidden:
        skew = "The disc's skew τ (hidden)"
        moment = "hidden"
    else:
        skew = (f"The disc's skew τ = {math.degrees(s.skew):.4g}° (toward {math.degrees(s.skew_angle):.0f}° from "
                f"the mark)")
        moment = fmt(abs(s.couple_unbalance) * omega**2, 1.0) + " N·m"
    faces = (" A balance weight U<sub>w</sub> on one of the disc's faces, Δz from its centre plane, adds U<sub>w</sub>Δz to "
             "(I<sub>d</sub> − I<sub>p</sub>)τe<sup>iγ</sup>." if any(p.disc for p in s.plane_weights) else "")
    return f"""
<h3>Equations of motion at {rpm:,.0f} rpm (Ω = {omega:.4g} rad/s)</h3>
<p style='font-size:large' align='center'>M q̈ + (C + Ω G) q̇ + (K + Ω H{" + ΔK(φ)" if s.periodic else ""}) q = f(t)</p>
<p>q = [x<sub>A</sub>, x<sub>D</sub>, θ<sub>x</sub>, x<sub>B</sub> | y<sub>A</sub>, y<sub>D</sub>,
θ<sub>y</sub>, y<sub>B</sub>]: the journals' displacements at A and B, the disc's displacement, and the
shaft's slope at the disc (θ<sub>x</sub> = dx/dz, θ<sub>y</sub> = dy/dz), first in the horizontal x–z
plane, then in the vertical y–z plane{modal}. Units: m and rad; forces in N, moments in N·m.</p>
{_legend()}

<h4>Mass M and stiffness K</h4>
<p>{mass_text} K is the shaft's bending stiffness plus each support spring on its
journal. Only the bearings' cross-coupled stiffness k<sub>xy</sub> joins the two planes, as
K[x, y] = k<sub>xy</sub> and K[y, x] = k<sub>yx</sub> (−k<sub>xy</sub> unless set otherwise) at each
journal (here {_coupling(s)}). That part is
skew-symmetric, so K is symmetric only without it; with isotropic supports and no cross-coupling K has
two identical blocks.{swirl}</p>
{films}
{_mat(M, "M")}<br>{_mat(K, "K")}

<h4>The shaft (one plane; y is the same)</h4>
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
{cracked}

<h4>State space</h4>
<p>With Z = [q, q̇] the equations become Z' = A(Ω) Z + B u, u = f the force (or moment) on each
DOF, B = [0 ; M<sup>−1</sup>]:</p>
<p align='center'>A(Ω) = [ 0 &nbsp; I ; −M<sup>−1</sup>(K + ΩH) &nbsp; −M<sup>−1</sup>(C + ΩG) ]</p>
<p>A is {2 * s.n_dof} × {2 * s.n_dof} and not symmetric, and its eigenvalues s = −ζω<sub>n</sub> ± iω<sub>d</sub> come in
conjugate pairs. The modes at this speed, lowest first:</p>
<table border='1' cellspacing='0' cellpadding='3'>
<tr><th>Mode</th><th>s [1/s]</th><th>f<sub>d</sub> [Hz]</th><th>f<sub>d</sub> [rpm]</th><th>ζ</th>
<th>Whirl ratio</th><th>Whirl</th></tr>
{''.join(rows)}
</table>
<p><small>Whirl ratio: +1 for a circular forward whirl (the same way as the spin, x toward y), −1 for
circular backward whirl, 0 for a straight line; in between, an ellipse. The stations are weighted by
their masses (and the slopes by I<sub>d</sub>). A negative ζ is a mode that grows: the rotor is unstable
at this speed.{" For the turning shaft these are Floquet modes, e<sup>st</sup> times a motion periodic in φ: each is listed by its harmonic with the most kinetic energy, and a mode locked to 1X sits at exactly the speed." if s.periodic else ""}</small></p>

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
machine. H. H. Jeffcott's 1919 paper (and A. Föppl's of 1895, hence also the <i>Laval</i> or
<i>Föppl–Jeffcott</i> rotor) used it to explain why a turbine can run smoothly above the speed at which
its shaft whirls most violently. Here the bearings stand on flexible supports, springs and dampers in
the horizontal (x) and vertical (y) directions, and the disc can tilt as well as move sideways.</p>
<p>The notes come in six parts: the fundamentals; how the response is measured; what drives the rotor
once per revolution, and balancing; gravity, and shafts whose stiffness turns with them; stability; and
the numerics. Each section gives the idea, its equation and where that comes from, then the numbers on
the defaults, and ends with where to find it on the page. The <i>Things to try</i> follow the same six
parts, and the notation is listed last.</p>

<h2>1. Fundamentals</h2>

<h3>Degrees of freedom</h3>
<p>The shaft spins about its axis z. In each plane, x–z and y–z, four numbers describe its bending: the
journal displacements at bearings A and B, the disc's displacement, and the shaft's slope at the disc.
That makes 8 degrees of freedom (DOFs) and, in state space, 16 states. The shaft's shape anywhere follows
from them, since between the stations a massless beam bends as a cubic. (A shaft with mass needs more
DOFs: see <i>The shaft's own mass</i> below.)</p>

<h3>Whirl in complex coordinates</h3>
<p>Each point of the shaft moves in its own plane, so one complex number holds both of its
displacements: w = x + iy. The spin turns x toward y, so w = r<sub>w</sub>e<sup>iωt</sup> (ω &gt; 0) is a
circle traced the way the shaft spins: a <b>forward</b> whirl. w = r<sub>w</sub>e<sup>−iωt</sup> is a
<b>backward</b> one. A force that turns with the shaft is written the same way, f<sub>x</sub> +
if<sub>y</sub>, a constant times e<sup>iφ</sup>, with φ the angle the shaft has turned.</p>
<p>Any orbit at one frequency is an ellipse, and any ellipse is a forward circle plus a backward one:</p>
<p align='center'>w = w<sub>f</sub>e<sup>iωt</sup> + w<sub>b</sub>e<sup>−iωt</sup></p>
<p>Its semi-axes are |w<sub>f</sub>| + |w<sub>b</sub>| and ||w<sub>f</sub>| − |w<sub>b</sub>||, and it
is traced forward when |w<sub>f</sub>| &gt; |w<sub>b</sub>|. The <b>whirl ratio</b>
(|w<sub>f</sub>|² − |w<sub>b</sub>|²)/(|w<sub>f</sub>|² + |w<sub>b</sub>|²) is +1 for a forward circle,
−1 for a backward one and 0 for a straight line. A probe in x alone cannot tell forward from backward,
since x = Re w is the same for w and its mirror image. Two probes at right angles can: their orbit, or the
<i>full spectrum</i> of w, which puts forward whirl at positive frequencies and backward at negative.</p>
<p><b>On the page.</b> The modes table on the <i>Equations &amp; matrices</i> tab lists each mode's whirl ratio, with each station's motion weighted
by its mass. <i>Orbit</i> beside the probe reads the 1X ellipse's major semi-axis,
|w<sub>f</sub>| + |w<sub>b</sub>|.</p>

<h3>The classic case: disc at midspan</h3>
<p>With the disc at midspan on identical bearings, the problem is symmetric about the disc, so the
disc's translation and its tilt are uncoupled. The translation sees the shaft's midspan stiffness
k<sub>s</sub> = 48EI/L³ in series with the two supports in parallel, 2k<sub>b</sub>. With light
journals,</p>
<p align='center'>ω<sub>c</sub> = √(k<sub>eq</sub>/m), &nbsp; 1/k<sub>eq</sub> = 1/k<sub>s</sub> + 1/(2k<sub>b</sub>)</p>
<p>With the defaults (d = 10 mm, L = 0.5 m, steel), k<sub>s</sub> = 37,700 N/m and
k<sub>b</sub> = 50,000 N/m, so ω<sub>c</sub> = 165.5 rad/s, <b>1580 rpm</b>. The full model, with its
0.1 kg journals, puts it at 1584 rpm. Flexible supports lower the critical speed: on rigid bearings it
would be 1854 rpm.</p>

<h3>Unbalance response: heavy spot and high spot</h3>
<p>The disc's centre of mass sits a distance e from the shaft's axis: an unbalance U = me. Spinning at
Ω, it pulls the shaft round with a force meΩ² that turns with it. So the disc whirls once per revolution
(synchronous, 1X), forward, and in a circle if the rotor is isotropic.</p>
<p>Take the classic case as one mass m on a spring k with a damper c, the heavy spot at the keyphasor
mark, so that φ = Ωt. In w = x + iy the two planes' equations become one:</p>
<p align='center'>m ẅ + c ẇ + k w = meΩ²e<sup>iΩt</sup></p>
<p>Its steady state is a forward circle, w = Xe<sup>iΩt</sup>. With r = Ω/ω<sub>c</sub> and the damping
ratio ζ = c/(2mω<sub>c</sub>) = c/(2√(km)),</p>
<p align='center'>X = meΩ² / (k − mΩ² + icΩ) = e r² / (1 − r² + 2iζr)</p>
<p>So the whirl's radius is |X| = e r²/√((1 − r²)² + (2ζr)²), and the phase lag, −arg X, has
tan(lag) = 2ζr/(1 − r²).</p>
<p>The <b>heavy spot</b> (where the unbalance is) leads the <b>high spot</b> (where the shaft is
displaced most) by a phase lag. The lag is 0° well below the critical speed, about 90° near it, and 180°
well above it. Above the critical speed the shaft deflects away from the heavy spot. As r → ∞ the disc
spins about its own centre of mass, and the shaft's centre whirls on a circle of radius e. This
<b>self-centring</b> is what lets machines run supercritically.</p>
<table align='center' width='440' cellspacing='0' cellpadding='0'>
<tr><td align='center'><img src="fig:spots" width="440" height="330"></td></tr>
<tr><td><i>The disc seen end on, from beyond bearing B (z toward you), whirling about the bearings' centre O. The keyphasor mark is at φ from +x, with the heavy spot at it here (in general at ψ from it). The high spot, the rim's point farthest from O, lags the heavy spot: here by 60°, below the critical speed.</i></td></tr>
</table>
<p>With the defaults (e = 20 µm) the disc whirls 2.2 µm at 500 rpm, with 2° of lag. At the 1584 rpm
critical the lag is 88°, and the peak, 217 µm, comes at 1589 rpm. At 3000 rpm the whirl is 28 µm, with
176° of lag. At r = 1 the equation gives X = e/(2iζ): a lag of 90° and a whirl of e/(2ζ). The first mode's
ζ is 0.046, so e/(2ζ) = 215 µm, within 1% of the peak.</p>
<p>The lag reaches 90° at 1586 rpm, not quite at the critical, and the peak comes later still. The peak
is late even for one mass on a spring: an unbalance response peaks at r = 1/√(1 − 2ζ²), just above the
critical. The lag is late because the damping is at the supports, in series with the shaft, not at the
disc. As the speed rises the dampers stiffen the supports a little, so the rotor is not quite the
one-mass oscillator above.</p>

<h3>The gyroscopic effect</h3>
<p>A spinning disc has angular momentum I<sub>p</sub>Ω along its axis. Tilted by the slopes, the axis
points along (θ<sub>x</sub>, θ<sub>y</sub>, 1), so the angular momentum has a sideways part
I<sub>p</sub>Ω(θ<sub>x</sub>, θ<sub>y</sub>). As the disc whirls and its tilt changes, that part changes
at I<sub>p</sub>Ω(θ̇<sub>x</sub>, θ̇<sub>y</sub>), and changing an angular momentum takes a moment, about
an axis at right angles to the change. So tilting the disc in one plane takes a moment in the other. With
the tilt's own inertia, I<sub>d</sub>θ̈, the slopes' equations are:</p>
<p align='center'>I<sub>d</sub>θ̈<sub>x</sub> + ΩI<sub>p</sub>θ̇<sub>y</sub> = M<sub>x</sub>, &nbsp;
I<sub>d</sub>θ̈<sub>y</sub> − ΩI<sub>p</sub>θ̇<sub>x</sub> = M<sub>y</sub></p>
<p>So the gyroscopic term multiplies the velocities and sits beside the damping:
M q̈ + (C + ΩG) q̇ + K q = f. But G is <b>skew-symmetric</b>, so the power it absorbs,
q̇<sup>T</sup>ΩGq̇, is zero. It does no work, and so it cannot damp anything. Instead it couples the two
planes' tilts and makes the natural frequencies depend on the speed.</p>
<p>Each whirl mode splits into a <b>forward</b> whirl (the same way as the spin) and a <b>backward</b>
whirl. In a forward whirl the gyroscopic moment resists the tilting, as if the shaft were stiffer, so the
frequency rises with speed. In a backward whirl it does the opposite. For the disc's tilt on its own (on a
rigid support of stiffness k<sub>θ</sub>):</p>
<p align='center'>I<sub>d</sub>ω² ∓ I<sub>p</sub>Ωω − k<sub>θ</sub> = 0</p>
<p>The forward root meets the 1X line, ω = Ω, where (I<sub>d</sub> − I<sub>p</sub>)Ω² = k<sub>θ</sub>.
That has no solution whenever I<sub>p</sub> ≥ I<sub>d</sub>: for a disc, or a solid cylinder shorter than
√3 times its radius. Then an unbalance can never make the tilting mode resonate forward. For a thin disc
I<sub>p</sub> = 2I<sub>d</sub>, and the forward tilting frequency tends to 2Ω. A long rotor
(I<sub>d</sub> &gt; I<sub>p</sub>) does have a forward conical critical speed.</p>
<p><b>Why the classic case hides it.</b> At midspan, on identical bearings, the disc whirls without
tilting, so the gyroscopic moment never enters the first mode. Only the disc's own tilting (conical)
mode, high above, splits. Move the disc off centre, or make the bearings different, and the translation
and tilt couple. Then the first mode splits too, and its critical speed moves.</p>

<h3>The Campbell diagram and critical speeds</h3>
<p>Plotting each damped natural frequency against the spin speed gives the Campbell diagram. The
unbalance turns once per revolution, so it drives at f = Ω/2π: the 1X line. Where a whirl frequency
crosses the 1X line, the unbalance drives that mode at its own natural frequency: a <b>critical
speed</b>. In an isotropic rotor the unbalance is a purely forward rotating force, so it drives only the
forward branches. The backward crossings are critical only for an anisotropic rotor, or for other
excitations.</p>
<table align='center' width='540' cellspacing='0' cellpadding='0'>
<tr><td align='center'><img src="fig:campbell" width="540" height="330"></td></tr>
<tr><td><i>A schematic Campbell diagram. Spinning, a mode splits into a forward branch and a backward one. The unbalance drives the forward crossing of the 1X line, the critical speed. The backward crossing matters only on an anisotropic rotor. <i>Also draw</i> adds the 2X, 3X and ½X lines.</i></td></tr>
</table>
<p>Other excitations come at other multiples of the speed:</p>
<ul>
<li><b>2X</b>: a shaft stiffer one way than the other (a keyway, a crack) flexes twice per revolution
under its own weight, and misalignment pulls twice per revolution. Where a forward branch meets 2X, at
about half its critical speed, such a rotor has a <i>secondary critical</i> (see <i>The asymmetric
shaft</i>).</li>
<li><b>3X</b>: a breathing crack's stiffness changes with 3X in it too, so its weight drives 3X near a
third of the critical speed (see <i>The cracked shaft</i>).</li>
<li><b>½X</b>: the oil in a plain journal bearing circulates at about half the shaft's speed, so its
instability begins near ½X (see <i>Stability</i>).</li>
</ul>
<p><b>On the page.</b> <i>Also draw</i> adds these lines to the Campbell diagram, up to the speed set in
<i>Speeds up to</i>. Set an asymmetry or a crack on a horizontal rotor to see a secondary critical.</p>

<h3>The overhung rotor</h3>
<p>Fans, pump impellers and many small turbines carry their disc outside the bearings, on an overhang ℓ
(a/L &gt; 1 past bearing B, or below 0 past A). The shaft is then a span between the bearings and a
cantilever beyond, and at the disc its stiffness is 3EI/(ℓ²(L + ℓ)). Pushed sideways, the overhung disc
tilts as much as it moves: in the first mode its slope is about 1/ℓ of its displacement. So the
gyroscopic moment, small for a disc between the bearings, acts on the first mode in full, and the forward
and backward criticals separate strongly.</p>
<p>The <i>Overhung rotor</i> preset puts a 3 kg disc of radius 150 mm 100 mm past bearing B, on
supports of 200,000 N/m. At rest its first mode is 14.5 Hz (870 rpm). Spinning, the forward whirl
stiffens and the backward one softens: 21.4 and 8.6 Hz at 1284 rpm, 28.9 and 3.1 Hz at 5000 rpm. The
backward branch meets the 1X line at 664 rpm, and the forward one at <b>1284 rpm</b>. That is 48% above
the critical speed the same rotor would have without the gyroscopic effect (870 rpm, with
I<sub>p</sub> = 0), and above the textbook estimate that ignores the tilt (1046 rpm). The unbalance
drives only the forward crossing, so that is the one critical speed it shows. The disc's conical mode
(53 Hz at rest) has a forward branch rising faster than 1X (I<sub>p</sub> = 2I<sub>d</sub>), so it is
never critical.</p>

<h3>Anisotropic supports and backward whirl</h3>
<p>Supports stiffer vertically than horizontally are common: gravity, foundations and oil-film bearings
all make them so. They split each mode into a horizontal one and a vertical one, with two critical
speeds, and the orbits become ellipses. Between the two criticals the horizontal response already lags by
more than 90° while the vertical one does not yet. There the shaft whirls <b>backward</b>, against its
spin, while the unbalance still turns forward. A shaft whirling backward is bent back and forth once per
revolution as it spins: a fatigue loading that a forward synchronous whirl does not cause.</p>

<h3>The shaft's own mass: beam finite elements</h3>
<p>Jeffcott's shaft has no mass: it only holds the disc. That keeps the model small, because a massless
shaft bends as a cubic between the stations, and four DOFs in each plane describe it exactly. A shaft
with mass, spread along it, also has its sections' rotary inertia and their gyroscopic moment. Its shape
is then no longer set by the stations alone, so it is cut into beam <b>finite elements</b>. Each bends
as a cubic between two nodes, and each node has a displacement and a slope in each plane.</p>
<p>A light shaft changes little. The defaults' steel shaft weighs 0.31 kg beside the 1 kg disc, and it
takes the first critical speed from 1584 to 1458 rpm. The old rule adds 17/35 of a simply supported
shaft's mass to the disc, which gives 1477 rpm. (The rule is Rayleigh's method, taking the shaft's shape
as its static deflection under a load at the disc.)</p>
<p>But a shaft has bending modes of its own, and a heavier one brings them down into the speed range.
In the <i>Long shaft</i> preset (a 2 kg disc on a 1 m, 20 mm shaft of 2.47 kg) the critical speeds are
1589 and 6582 rpm. The massless model puts the first at 2146 rpm, 35% high, and it has no second: its
next modes are the journals bouncing on their supports, heavily damped. Gravity shows the difference
too: the disc sags 375 µm, not the massless model's 200 µm.</p>
<p><b>Consistent or lumped mass.</b> The consistent mass matrix spreads each element's mass with the
same cubic shapes as its stiffness. It is a Rayleigh–Ritz model, so at rest and undamped its frequencies
are never too low, and they come down as elements are added. On the <i>Long shaft</i> the second critical
comes out at 7393, 6591 and 6582 rpm with 2, 4 and 8 elements. A lumped mass matrix, half of each
element's mass at each end, is simpler but has no such bound. Here it gives 8187, 6733 and 6620 rpm,
because with few elements much of the mass sits on the bearings, where it hardly moves.</p>
<p><b>On the page.</b> Tick <i>Shaft mass</i> to give the shaft its mass, and set the number of
<i>Beam elements</i> and the kind of mass beside it. The Campbell diagram draws the massless model's
branches dashed beside the real ones. How the model is kept fast is under <i>Numerics</i>.</p>

<h2>2. Measuring the response</h2>

<h3>Run-up and coast-down: Bode and polar plots</h3>
<p>On a real machine the response is measured as the speed changes. A <b>keyphasor</b>, a once-per-turn
mark on the shaft, starts each revolution as it passes +x. (The heavy spot is at the mark unless you move
it.) A <b>tracking filter</b> takes the 1X component over one revolution:</p>
<p align='center'>V = (1/π) ∫ x(φ) e<sup>−iφ</sup> dφ, &nbsp; x ≈ |V| cos(φ − lag)</p>
<p>|V| is the 1X amplitude. The <b>phase lag</b> is the angle the shaft turns from the keyphasor to the
probe's positive peak. The <b>Bode plot</b> draws both against speed. The <b>polar plot</b> draws V as a
point, its angle the lag (clockwise, against the rotation). In steady state, through a lightly damped
critical speed, V traces very nearly a circle, lowest (lagging 90°) near the critical. A probe in y reads
90° more lag than one in x on a circular orbit, since the shaft turns 90° more before the high spot
reaches it.</p>
<p>Swept through a critical at a finite rate, the response differs from the steady state in four
ways:</p>
<ul>
<li><b>The transient peak is lower than the steady-state one, and lower the faster the ramp.</b> The
whirl builds up with the mode's time constant, 1/(ζω<sub>c</sub>). A ramp at Ω' (rad/s²) crosses the
half-power band 2ζω<sub>c</sub> in 2ζω<sub>c</sub>/Ω'. So the ramp is slow enough to reach the
steady-state peak only when Ω'/ω<sub>c</sub>² is well below 2ζ². With the defaults (ζ = 0.046,
2ζ² = 0.0043, the peak 217 µm at 1589 rpm), sweeping 800 to 2400 rpm, a run-up at 100 rpm/s
(Ω'/ω<sub>c</sub>² = 0.0004) peaks at 100% of it. At 500 rpm/s (0.0019) it peaks at 95%, at
2000 rpm/s at 78%, and at 5000 rpm/s (0.019) at about 60%.</li>
<li><b>The peak comes after the critical speed on a run-up, and before it on a coast-down.</b> The
response lags the sweep. At 500 rpm/s the run-up peaks at 1672 rpm (82 rpm late) and the coast-down at
1498 rpm (92 rpm early). At 5000 rpm/s they peak some 350 rpm late and 250 to 350 rpm early. (That fast,
the speed changes by 160 rpm in one revolution near the critical, so the tracked curve is coarse: one
point per revolution.) The coast-down peaks lower, at 87% of the steady-state peak at 500 rpm/s and about
50% at 5000 rpm/s, because its peak comes at a lower speed, where the unbalance force UΩ² is smaller.</li>
<li><b>Beating just past the critical.</b> The whirl built up at the critical speed rings on as free
vibration at the natural frequency ω<sub>c</sub>, while the unbalance keeps driving at Ω. The two beat
at Ω − ω<sub>c</sub>, faster as the speed moves on. So the 1X amplitude wobbles after the peak. The
largest displacement over each revolution (the dots) wobbles more, since it includes the free vibration
in full. On the polar plot the tracked vector loops around the steady-state circle.</li>
<li><b>The parameter that matters is Ω'/ω<sub>c</sub>², against ζ.</b> Measure time in units of
1/ω<sub>c</sub>, and a single mode accelerated through resonance has only two parameters left: ζ and the
dimensionless acceleration Ω'/ω<sub>c</sub>². So the ratio of the transient peak to the steady-state
one, and how far the peak is shifted, depend on these alone. F. M. Lewis first worked this out (1932,
<i>Vibration during acceleration through a critical speed</i>), and much later work on passage through
resonance builds on it.</li>
</ul>
<p><b>On the page.</b> <i>Run up and coast down</i> ramps to the first speed and waits for the
start-up transient to die away. It then runs up to the second speed and back at the ramp rate, tracking
each revolution. Tick <i>Hold earlier sweeps</i> to compare ramp rates. The summary under the Bode plot
gives each sweep's peak against the steady-state one, and how many rpm away from it the peak came.</p>

<h3>Waterfall and cascade plots</h3>
<p>A spectrum analyser on a machine takes the spectrum of a block of signal, then of the next, and so
on. Stacked against time the spectra make a <b>waterfall</b> plot; against the speed during a run-up or
coast-down, a <b>cascade</b>. The unbalance makes a ridge along +1X that moves with the speed and peaks
at the critical. Whatever sets the rotor ringing (the passage through the critical, a tap, an
instability) shows at the free whirl frequencies. These stay put, or move only as the gyroscopic effect
moves them, while the 1X ridge sweeps past. That is how a natural frequency is told from a forced
response on a running machine.</p>
<p><b>A measured Campbell diagram.</b> Tap the disc every half second through a run-up, and each tap
rings both whirls of each mode at the speed it came at. So the peaks trace the Campbell diagram's
branches. On the <i>Overhung rotor</i> preset, from rest to 2,500 rpm, the first mode's backward whirl
falls from 14.5 to 5.7 Hz and its forward whirl rises to 25.5 Hz. The 1X ridge grows where it crosses
the forward branch, at the critical speed (1284 rpm), and peaks just after it, as on the Bode plot. On a
full spectrum the backward branch lies on the other side of zero, so the +1X ridge never meets it. The
backward crossing on the Campbell diagram (664 rpm) is not a critical for an isotropic rotor.</p>
<p><b>Block length.</b> A block of T<sub>b</sub> seconds resolves frequencies 1/T<sub>b</sub> apart:
2 Hz for 0.5 s. A longer block resolves more finely, but while the speed ramps at Ω' it changes by
Ω'T<sub>b</sub> over the block, smearing the 1X ridge. That is 100 rpm (1.7 Hz) at 200 rpm/s, but
2500 rpm (42 Hz) at 5000 rpm/s, where the ridge is a smear. Analysers get round this by <i>order
tracking</i>: sampling at fixed shaft angles, from the keyphasor, so that 1X stays in one bin whatever the
speed. The Bode plot's once-per-revolution 1X vector is that kind of measurement.</p>
<p><b>On the page.</b> The <i>Waterfall</i> tab takes the full spectrum of the orbit at a station, in
blocks of 0.5 s by default, each starting halfway through the last. <i>Tap the disc every</i> taps it as
it runs.</p>

<h2>3. Once-per-revolution forces and balancing</h2>

<h3>Other 1X sources: a bent shaft and a skewed disc</h3>
<p>Unbalance is not the only thing that turns with the shaft. Anything fixed to the rotor and off its
axis drives it once per revolution, and two more are common.</p>
<p><b>A bent shaft (initial bow).</b> A shaft can keep a bend: a thermal bow after a hot shutdown, a sag
left after standing, or a bent repair. Say it is δ<sub>b</sub> at the disc, in the shape a load at the
disc would give. Its elastic force is K<sub>shaft</sub>(q − q<sub>bow</sub>), so the bow acts as a force
K<sub>shaft</sub>q<sub>bow</sub> turning with the shaft, kδ<sub>b</sub> at the disc. Unlike the
unbalance's meΩ², it does not grow with speed. For the classic Jeffcott rotor, with the heavy spot at the
mark,</p>
<p align='center'>(k − mΩ² + icΩ) X = meΩ² + kδ<sub>b</sub>e<sup>iβ</sup></p>
<p>with β the bow's angle from the mark. So:</p>
<ul>
<li><b>At low speed the probes read the bow itself.</b> The shaft just carries its bend round: the
<i>slow-roll runout</i>. The bow's forces on the disc and the journals balance, so the bearings carry
nothing. The 1X response starts at δ<sub>b</sub>, not at zero, and on the polar plot the curve starts at
the bow vector, not at the origin.</li>
<li><b>Well above the critical the disc's whirl dies away.</b> The disc stays on the bearings' axis, and
the shaft bends round it to cancel the bow. The unbalance's response, by contrast, tends to e.</li>
<li><b>With an unbalance, the angle between them decides.</b> In phase they add: with the defaults and
δ<sub>b</sub> = 20 µm, the peak doubles to 435 µm. Opposite (β = 180°), the bow's response falls with
speed while the unbalance's grows, and they cancel at one speed, where meΩ² = kδ<sub>b</sub> in the
classic case. Choose δ<sub>b</sub> = e, opposite, and the disc's centre of mass sits exactly on the
bearings' axis. Then nothing whirls at any speed, and the probe reads a steady 20 µm, the bow.</li>
</ul>
<p>Measured from its bent shape, the shaft's deflection y = x − δ<sub>b</sub> obeys
(k − mΩ² + icΩ) y = (me + mδ<sub>b</sub>e<sup>iβ</sup>)Ω². Relative to its bow, the shaft sees the
disc's mass carried round on the bow: an unbalance mδ<sub>b</sub>. The probe adds the bow itself, as
runout. With the disc at midspan the bow is flat there, so that is all of it, and one weight can take out
every force the bow makes, at every speed, but not the runout. That is why the <b>slow-roll vector is
subtracted</b> from the 1X readings before balancing (see <i>Single-plane balancing</i> below).</p>
<p>Off centre, the bow also tilts the disc, by the bowed shaft's slope θ<sub>b</sub> there. The disc's
inertia carried round on that tilt is a skew (see <i>A skewed disc</i> below), a moment
(I<sub>d</sub> − I<sub>p</sub>)θ<sub>b</sub>Ω², and one weight cannot cancel a moment at every speed. On
the defaults' light disc it is tiny: at a/L = 0.3, a 20 µm bow balanced at 1200 rpm leaves at most
0.08 µm beyond the runout up to 6000 rpm, and none at all with I<sub>d</sub> = I<sub>p</sub>.</p>
<p><b>A skewed disc (couple unbalance).</b> A disc mounted out of square has its principal axis tilted by
τ from the shaft's. Its inertia acts on its own axis, so spinning it takes a moment
(I<sub>d</sub> − I<sub>p</sub>)τΩ² that turns with the shaft. That is the unbalance force's form, with U
replaced by (I<sub>d</sub> − I<sub>p</sub>)τ. A thin disc (I<sub>p</sub> &gt; I<sub>d</sub>) is pushed
toward square to the spin axis: as Ω → ∞ the shaft bends at the disc by −τ, and the disc spins flat. A
long rotor (I<sub>d</sub> &gt; I<sub>p</sub>) is pushed the other way, and its forward conical mode can
then meet the 1X line. At midspan on identical bearings the skew only tilts the disc. Move the disc off
centre and the tilt and the translation couple, so the skew moves the disc too. A <i>static</i>
unbalance (the mass off the axis) can be corrected by one weight in one plane. A <i>couple</i>
unbalance needs two weights in two planes, equal and opposite.</p>

<h3>Single-plane balancing</h3>
<p>Balancing puts a weight on the rotor to cancel its unbalance. Where the heavy spot is cannot be seen,
and the phase lag between it and the high spot depends on the speed. So it is found from measurements, by
<b>influence coefficients</b>:</p>
<ol>
<li>Turn slowly and read the 1X vector: the <b>slow roll</b> V<sub>s</sub>, runout that no unbalance
causes (a bow, or on a real machine a scratch or a magnetic spot on the probe's track).</li>
<li>Run at the balancing speed and read the vector <b>as found</b>, V<sub>0</sub>.</li>
<li>Stop, add a known <b>trial weight</b> T (mass × radius, at an angle from the keyphasor mark), run
again and read V<sub>1</sub>. The rotor is linear, so the weight's effect is V<sub>1</sub> − V<sub>0</sub>,
and the <b>influence coefficient</b> is α = (V<sub>1</sub> − V<sub>0</sub>)/T: microns, and degrees of
lag, per g·mm of weight.</li>
<li>Take the trial off, put on the <b>correction</b> W<sub>c</sub> = −(V<sub>0</sub> − V<sub>s</sub>)/α,
then run and check.</li>
</ol>
<p>All the quantities are vectors (complex numbers). A reading's angle is its phase lag from the
keyphasor, and a weight's is its angle on the rotor. A trial weight should change the reading by a good
fraction (30% or more) without making the rotor unsafe to run. For an unbalance at the disc alone, the
one weight balances the rotor at every speed, since unbalance and weight both make forces growing as Ω²
at the same place.</p>
<p><b>Why the slow roll is subtracted.</b> Take the defaults with a bow of 20 µm at 90°, read at the
disc, balancing at 1200 rpm. The disc reads 20 µm at 100 rpm, 52 µm at 1200 rpm and 293 µm at the
critical. With the slow roll subtracted, the correction is 1.41 g at 20 mm, 225°. It cancels both the
unbalance and the bow's mδ<sub>b</sub>, and from then on the disc reads about 20 µm at every speed: the
runout alone, with no force behind it. Balanced on the raw reading instead, the weight (1.95 g at 243°)
zeroes the reading at 1200 rpm by leaving the rotor out of balance: 162 µm at the critical, 41 µm at
3000 rpm. A bowed rotor cannot be made to read zero at every speed with one weight.</p>
<p>Taking the slow roll too fast spoils the balance. At 300 rpm the reading already holds some
unbalance response, so the weight comes out a little light (1.37 g instead of 1.41 g). At the critical
the disc then moves 8.6 µm beyond its runout, against 0.9 µm with the slow roll taken at 100 rpm. The
total readings there hide this: 16.6 µm looks better than 19.5 µm only because the leftover unbalance's
response happens to partly cancel the runout. Judge a balance by the motion beyond the slow roll,
|V − V<sub>s</sub>|, not by the total.</p>
<p><b>What one plane cannot do.</b> A weight in the disc's plane makes a force, and a skewed disc makes
a moment. Off centre, both move the disc. So a single weight can cancel the reading at the balancing
speed, but not at every speed. With a/L = 0.3 and a skew of 0.5°, balanced at 1200 rpm, the 72 µm at
the critical falls only to 5.7 µm, and 9 µm is left at 6000 rpm. A couple unbalance needs <b>two-plane
balancing</b>, with weights in two planes and the influence coefficients forming a 2 × 2 matrix.</p>
<p><b>On the page.</b> The <i>Balancing</i> tab takes the readings in order. <i>Subtract slow roll</i>
chooses whether the correction uses V<sub>0</sub> − V<sub>s</sub> or V<sub>0</sub>, and <i>Mystery
unbalance</i> hides the heavy spot for you to find.</p>

<h3>Two-plane balancing</h3>
<p>A rigid body's unbalance is a force and a couple, so cancelling it takes two weights in two planes,
and two probes to see it. The procedure is the single-plane one with vectors. Read the slow roll
V<sub>s</sub> and the readings as found, V<sub>0</sub> (one per probe). Put the trial weight T in plane 1
alone and read V<sub>1</sub>, then move it to plane 2 and read V<sub>2</sub>. Each trial run gives a
column of the <b>influence matrix</b>:</p>
<p align='center'>α<sub>ij</sub> = (V<sub>j</sub> − V<sub>0</sub>)<sub>i</sub> / T, &nbsp; and the
correction &nbsp; W<sub>c</sub> = −α<sup>−1</sup>(V<sub>0</sub> − V<sub>s</sub>)</p>
<p>That is a weight for each plane, with both trials taken off. With more probes than planes, the same
formula gives the least-squares weights.</p>
<p><b>Where the planes are matters.</b> Take the same rotor as above (a/L = 0.3, skew 0.5°, the default
20 g·mm unbalance), with probes at the two bearings, balanced at 1200 rpm and read at the disc. Every
weight below is a mass at a 20 mm radius.</p>
<ul>
<li><b>On the disc's faces</b>, 40 mm apart: 7.32 g at 180° and 6.32 g at 0°. They are nearly equal and
opposite, so mostly a couple. Together they make a force and a moment on the disc, the same form as the
unbalance and the skew, so they cancel both exactly. The rotor is then balanced at <i>every</i> speed:
the 72 µm at the critical falls to nothing, as does the 13.6 µm at 6000 rpm.</li>
<li><b>On the shaft</b>, 50 mm either side of the disc: 3.23 g and 2.22 g, almost what faces 100 mm apart
would need (3.23 g and 2.23 g). But the shaft between them and the disc bends, so they are not quite a
force and a couple on the disc. Both probes read zero at 1200 rpm, and 2.7 µm is left at the critical,
1.2 µm at 6000 rpm.</li>
<li><b>On the shaft near the bearings</b> (20 and 480 mm): small weights, 1.25 g and 0.33 g, since the
planes are far apart. But the forces are now far from the unbalance. The probes read zero at 1200 rpm,
yet at the critical the disc still reads 12.8 µm. A flexible rotor balanced away from its unbalance is
balanced only at the balancing speed. (With more planes it can be balanced at more speeds: <i>modal
balancing</i> puts a set of weights against each mode in turn.)</li>
</ul>
<p><b>Planes close together are hard to tell apart.</b> A weight in one face of the disc moves the
probes almost as one in the other face does, so the columns of α are nearly parallel. Their difference,
which is what the couple acts through, is small, and dividing by it inflates the weights and any error in
the readings. The <b>condition number</b> of α measures this: 27.6 for faces 40 mm apart, 11 at 100 mm,
1.5 for planes near the bearings. With 1% random error in every reading (the simulation's readings have
none), faces 40 mm apart leave a median of about 18 µm at the critical instead of 0, and faces 100 mm
apart about 8 µm. Real balancing machines put the planes as far apart as the rotor allows.</p>
<p><b>On the page.</b> Choose <i>Two planes</i> on the <i>Balancing</i> tab, on the disc's faces or on
the shaft.</p>

<h2>4. Gravity, and stiffness that turns</h2>

<h3>Gravity on a horizontal rotor</h3>
<p>Laid horizontal, the rotor carries its own weight. The disc's mg bends the shaft, and the bearings'
share of it, with the journals' own weight, compresses the supports. With the defaults the disc sags
378 µm and the journals 118 µm, nineteen times the unbalance's 20 µm eccentricity. The rotor whirls about
that sagged position, so its orbits sit below the bearings' centre. A DC-coupled proximity probe, which
reads the gap, sees the orbit offset by the sag. An AC-coupled one sees only the vibration.</p>
<p>The model is linear, so the sag is a static deflection, q<sub>s</sub> = (K + ΩH)<sup>−1</sup>f<sub>g</sub>,
and the vibration about it is exactly what it would be without gravity. In particular gravity alone makes
no whirl, however different the supports' stiffness vertically and horizontally: a constant force on a
linear rotor only moves its equilibrium. There are three exceptions, each below. A shaft stiffer one way
than the other bends more under its weight twice per turn, a 2X excitation (<i>The asymmetric
shaft</i>). A crack is opened and shut by the weight as the shaft turns (<i>The cracked shaft</i>). And a
fluid-film bearing's stiffness depends on the load it carries (<i>The fluid-film bearing</i>).</p>
<p>The sag need not be straight down. Where the stiffness is not symmetric, the weight also pushes
sideways. With cross-coupled bearings, k<sub>xy</sub> = 15,000 N/m, the journals settle 17° from the
vertical in the direction of rotation, much as a fluid-film journal sits at an <i>attitude angle</i>.
With internal damping the shaft turns through its own sag, bent the same way at every angle, and the
damping resists that. With c<sub>i</sub> = 20 N·s/m the disc moves sideways by 14 µm at 1000 rpm,
42 µm at 3000 rpm and 78 µm at 6000 rpm.</p>
<p><b>On the page.</b> Tick <i>Horizontal rotor</i>. The <i>Static sag</i> box switches the drawing
between the two probes' views. The tracking, the spectrum and the phase are always of the vibration.</p>

<h3>The asymmetric shaft: a two-pole rotor</h3>
<p>The rotor of a two-pole generator has slots for its winding cut along two sides, so it is stiffer
bending one way than the other. A keyway or a flat does the same on a smaller scale, and a crack that
opens and closes is a cousin. Bending toward its stiff axis the shaft has EI(1 + Δ), and across it
EI(1 − Δ). The stiff axis is at β<sub>s</sub> from the keyphasor mark, and it turns with the shaft. So,
seen from the bearings, the stiffness changes twice per revolution: K(φ) = K + ΔK(2(φ + β<sub>s</sub>)).
The equations have a periodic coefficient, and three things follow.</p>
<p><b>Two critical speeds, and an unstable band between them.</b> Seen from axes turning with the
shaft, the stiffness stands still (and so does everything else, on isotropic supports). But the
centrifugal force −mΩ² and the Coriolis force join in. To hold a deflection along an axis, the shaft needs
more stiffness there than mΩ². Below the soft axis's critical speed it has it both ways. Between the two
axes' critical speeds the centrifugal force beats the soft axis but not the stiff one, so a bend along the
soft axis grows. It turns with the shaft, so from the bearings it is a growing forward whirl locked at
exactly 1X.</p>
<p>For the classic Jeffcott rotor on rigid bearings, with x<sub>r</sub> and y<sub>r</sub> along the
stiff and soft axes, undamped:</p>
<p align='center'>m(ẍ<sub>r</sub> − 2Ωẏ<sub>r</sub> − Ω²x<sub>r</sub>) + k(1 + Δ)x<sub>r</sub> = 0,
&nbsp; m(ÿ<sub>r</sub> + 2Ωẋ<sub>r</sub> − Ω²y<sub>r</sub>) + k(1 − Δ)y<sub>r</sub> = 0</p>
<p>The −Ω² terms are the centrifugal force and the 2Ω terms the Coriolis force. Try e<sup>st</sup>: the
determinant is a quadratic in s², and its constant term is (k(1 + Δ) − mΩ²)(k(1 − Δ) − mΩ²). Between the
axes' critical speeds, k(1 − Δ) &lt; mΩ² &lt; k(1 + Δ), that is negative, so one root s² is positive. Its
s is real and positive: a bend that grows without turning in these axes. Outside the band both roots s²
are negative, and the undamped rotor neither grows nor decays.</p>
<p>Undamped, that band runs exactly between the critical speeds of round shafts as stiff as each axis:
with Δ = 0.15 and undamped supports, 1479 to 1645 rpm. Damping narrows it, and below a threshold
asymmetry there is none: Δ = 0.123 with the defaults' 100 N·s/m supports, 0.178 with 400 N·s/m. Unlike
the instabilities of <i>Stability</i>, it ends: past the stiff axis's critical speed the rotor is stable
again. Mapped against Δ and the speed, it is an Ince–Strutt chart, like a Mathieu equation's. It has one
tongue between the two critical speeds, its tip lifted off Δ = 0 by the damping. The <i>Two-pole
rotor</i> preset (Δ = 0.15) has critical speeds of 1491 and 1665 rpm, and it is unstable from 1527 to
1631 rpm. Tapped at 1580 rpm, its whirl grows ×e every 0.56 s at 26.3 Hz: the speed, not a natural
frequency.</p>
<p><b>The gravity critical.</b> Laid horizontal, the shaft's weight is a constant force on a stiffness
that turns, so the rotor never settles. Its deflection swings twice per revolution about a mean sag: a
<b>2X</b> whirl, largest where 2Ω meets a natural frequency, near <b>half the critical speed</b>. The
preset's disc whirls 447 µm at 2X at 785 rpm, more than its mean sag of 384 µm. Run up and coast down
from 500 to 1200 rpm at 200 rpm/s, and the 2X reaches 93% and 94% of the steady-state peak. A 2X peak at
half the first critical speed on a run-up is the classic signature of a two-pole generator (or of a
cracked shaft). Balancing does nothing to it.</p>
<p><b>The unbalance angle.</b> With an asymmetric shaft, the 1X response depends on where the heavy
spot sits from the stiff axis, not only on its size. Take Δ = 0.10, the supports' damping holding it
stable, and the 20 g·mm unbalance. Along the stiff axis the disc's orbit peaks at 827 µm (1589 rpm), and
along the soft axis at 848 µm (1576 rpm). At 45° between them it has two lower peaks, 201 µm at 1547 rpm
and 195 µm at 1624 rpm. A round shaft peaks at 217 µm. So the response is no longer a complex number
times the unbalance: turn the heavy spot by 90° and the response does not turn by 90°.</p>
<p><b>Balancing it.</b> The response is still linear, but in the unbalance's two parts separately. A
reading is</p>
<p align='center'>V = α<sub>+</sub>U + α<sub>−</sub>Ū</p>
<p>with Ū the complex conjugate of U, and α<sub>−</sub> = 0 on a round shaft. One trial weight cannot
find two coefficients, so the single influence coefficient of <i>Single-plane balancing</i> is wrong.
Two trials at right angles, T and iT, reading V<sub>1</sub> and V<sub>2</sub>, find both:</p>
<p align='center'>α<sub>+</sub> = ((V<sub>1</sub> − V<sub>0</sub>) − i(V<sub>2</sub> − V<sub>0</sub>))/(2T),
&nbsp; α<sub>−</sub> = ((V<sub>1</sub> − V<sub>0</sub>) + i(V<sub>2</sub> − V<sub>0</sub>))/(2T̄)</p>
<p>The correction then solves α<sub>+</sub>W<sub>c</sub> + α<sub>−</sub>W̄<sub>c</sub> = −R, with
R = V<sub>0</sub> − V<sub>s</sub>. With its conjugate that is two equations for W<sub>c</sub> and
W̄<sub>c</sub>:</p>
<p align='center'>W<sub>c</sub> = (α<sub>−</sub>R̄ − ᾱ<sub>+</sub>R) / (|α<sub>+</sub>|² − |α<sub>−</sub>|²)</p>
<p>Take the rotor above (Δ = 0.10) with the heavy spot at 45°, read at the disc and balanced at
1200 rpm with a 0.5 g trial at 20 mm. One trial, at 0°, gives 1.18 g at 237°. The check run still reads
9 µm, and through the critical the orbit peaks at 283 µm, worse than the 201 µm before. Two trials, at 0°
and 90°, give 1.00 g at 225°, exactly opposite the unbalance. The orbit then peaks at 0.7 µm, left by the
slow roll's own small response at 100 rpm. (One trial does work when it happens to lie in line with the
heavy spot, at the same angle or opposite: then W<sub>c</sub> = −U exactly.) The <i>Balancing</i> tab
takes one trial per plane, so the two-trial method is for working by hand.</p>
<p><b>How it is worked out.</b> In axes turning with the shaft, on isotropic supports, the rotor is
time-invariant, so its modes are exact. Each is seen from the bearings as a forward and a backward
circle, at s ± iΩ, and is listed at the one with more kinetic energy. On anisotropic supports the modes
come from Hill's method: the Floquet exponents of the periodic system, from its harmonics up to
±3 × 2Ω. The steady states (the unbalance's 1X, gravity's mean sag and 2X) come from harmonic balance.
The turning stiffness ties each harmonic n to n ± 2. That closes at once on isotropic supports (1X stays
1X, and the sag gives a forward 2X), and is cut at ±6X otherwise. The time stepping splits each step:
half a step of the rest (the exact update), the kick of the turning stiffness on the velocities at
mid-step, and the other half step. It agrees with the harmonic balance to about 10<sup>−5</sup>.</p>
<p>A Floquet mode's frequency is defined only up to a multiple of the stiffness's harmonic spacing
Ω<sub>s</sub> (2Ω here, Ω for a breathing crack). The same mode can be listed at s + inΩ<sub>s</sub>
for any whole n, and its ζ changes with the n chosen, but its growth rate, Re s, does not. So on the
stability map of a shaft whose stiffness turns, the onset (ζ = 0) holds whichever harmonic is listed.
Read the size of ζ away from zero as a growth rate.</p>
<p><b>On the page.</b> Set the <i>Asymmetry</i> ±Δ and its angle in the <i>Shaft and disc</i> box; the
3D view draws the stiff axis as the dashed diameter across the disc. Choose <i>2X</i> beside the probe
on the Bode plot for the gravity critical, and <i>Asymmetry chart</i> on the <i>Stability</i> tab for
the unstable band.</p>

<h3>The cracked shaft</h3>
<p>A fatigue crack grows across a shaft from its surface, often where the bending stress is highest, at
a shoulder or a shrink fit. Its depth from the surface is a<sub>c</sub>, up to half the shaft's diameter
d.</p>
<p><b>A hinge.</b> Where it is open, the crack lets the shaft's slope jump across it: Δθ = κM under a
bending moment M. Its compliance κ comes from fracture mechanics (Dimarogonas and Papadopoulos). Cut the
crack into strips across its front, each an edge crack in a strip as tall as the shaft there, and add up
the energy each releases as the crack grows, (1 − ν²)K<sub>I</sub>²/E. Bending that pulls the mouth open
(the moment along the crack's front) loads every strip. Bending across it loads one half in tension and
the other in compression. So the crack gives far more one way. For a<sub>c</sub>/d = 0.4 in the 10 mm
shaft, κ<sub>o</sub> = 1.45 × 10<sup>−4</sup> rad/(N·m) opening it, and κ<sub>a</sub> =
0.60 × 10<sup>−4</sup> across.</p>
<p>The figure across counts the strips on the compressed half too, as though open. That holds only while
the bending that opens the crack (on the rotor, the weight's) keeps the whole front in tension, the
vibration's smaller bending only adding to it on one side and taking from it on the other.</p>
<p>On the shaft the hinge is condensed out exactly onto the stations. With the shaft whole, the
stations' displacements q make a bending moment b<sub>h</sub><sup>T</sup>q at the crack. Let the slope jump
by Δθ there, and the shaft, resisting the jump with a stiffness k<sub>h</sub>, carries
b<sub>h</sub><sup>T</sup>q − k<sub>h</sub>Δθ. The hinge carries the same moment, in series, so
Δθ = κ(b<sub>h</sub><sup>T</sup>q − k<sub>h</sub>Δθ). Δθ has no mass of its own, so it follows q at once:
Δθ = κb<sub>h</sub><sup>T</sup>q/(1 + κk<sub>h</sub>). Put back, it takes</p>
<p align='center'>b<sub>h</sub>b<sub>h</sub><sup>T</sup>κ / (1 + κk<sub>h</sub>)</p>
<p>off the stations' stiffness. A crack with no compliance (κ = 0) takes nothing, and a free hinge
(κ → ∞) takes b<sub>h</sub>b<sub>h</sub><sup>T</sup>/k<sub>h</sub>. Open, the crack makes an asymmetric
shaft, its two axes along and across the mouth, softened.</p>
<p><b>Breathing.</b> A real crack is shut most of the time, its faces pressed together, unless the
bending stretches its side of the shaft. On a horizontal rotor the shaft's weight bends it far more than
its vibration does (<i>weight dominance</i>, as in Mayes and Davies' model), so the weight sets it. It is
fully open as its mouth turns to the side the sag stretches (the bottom, between the bearings), and shut
half a turn later. With its mouth at φ + β<sub>c</sub> from +x, it is open by</p>
<p align='center'>σ(φ) = (1 − sin(φ + β<sub>c</sub>))/2</p>
<p>That is 1 with the mouth pointing down (−90°), and 0 pointing up. Where the weight bends the shaft
the other way, stretching its top (near and past the bearing of an overhung rotor), the sign flips:
σ = (1 + sin(φ + β<sub>c</sub>))/2. That opening
changes once per revolution, and it multiplies a stiffness that already changes twice per revolution. So
the stiffness has 1X, 2X and 3X in it, not only the asymmetric shaft's 2X.</p>
<p>It changes even in axes turning with the shaft, because the weight does not turn. So the modes come
from Hill's method, and the steady states from harmonic balance with every harmonic tied to its
neighbours ±1, ±2 and ±3 (cut at ±6X). The time stepping kicks the velocities with the whole turning
stiffness at each mid-step. On a vertical rotor nothing opens it: a breathing crack stays shut,
invisible.</p>
<p><b>What it does.</b> The <i>Cracked shaft</i> preset has a crack 40% through the shaft at the disc,
laid horizontal. On average over a turn it takes 3.6% off the disc's stiffness along its mouth and 1.7%
across, so the critical speed splits a little, to 1562 and 1574 rpm. The weight then drives three whirls
the round shaft doesn't have:</p>
<ul>
<li>A <b>2X</b>, 37.6 µm at 783 rpm, half the critical speed, as the asymmetric shaft's.</li>
<li>A <b>3X</b>, 18.6 µm at 522 rpm, a third of it. The asymmetric shaft has none, so a 3X peak at a
third of the critical on a run-up points to a crack rather than to slots or a keyway.</li>
<li>A <b>1X</b>: with no unbalance at all, the disc whirls 107 µm at the critical. It adds to the
unbalance's, so the 1X vector changes as the crack grows. (On a machine, a slowly drifting 1X vector is
the other warning sign.) With the heavy spot at the crack's mouth the disc whirls 164 µm at the
critical, against 54 µm uncracked; with the mouth opposite it, 52 µm.</li>
</ul>
<p>Held open, the crack has no 3X and twice the 2X, 75.3 µm at 775 rpm: it is then an asymmetric shaft.
The 2X and 3X grow fast with the depth: at a<sub>c</sub>/d = 0.2, 0.3, 0.4 and 0.5 the 2X peaks at 9.5,
21.8, 37.6 and 51.5 µm.</p>
<p><b>On the page.</b> Set the <i>Crack depth</i> a<sub>c</sub>/d in the <i>Shaft and disc</i> box,
with its mouth's angle from the keyphasor mark and its place along the shaft. Untick <i>Breathing</i> to
hold it open, as a sawn notch. The 3D view draws it as a dark tick on the shaft, pointing at its mouth,
thick while it is open.</p>

<h2>5. Stability</h2>

<h3>Cross-coupling and internal damping</h3>
<p>Everything so far is forced vibration. The unbalance drives the rotor, and however hard it shakes at a
critical speed, every free whirl dies away. Two effects can instead make a free whirl grow by itself, and
both are stiffnesses that are not symmetric.</p>
<p><b>Cross-coupled stiffness.</b> In a fluid-film bearing (or a seal, or an impeller's clearance) the
fluid is dragged round by the shaft. So pushing the journal in x raises the pressure ahead of it and
pushes it in y too: f<sub>x</sub> = −k<sub>xy</sub>y, f<sub>y</sub> = +k<sub>xy</sub>x. On a circular
forward orbit of radius r<sub>w</sub>, that force always points along the orbit, the way the journal
moves. It does work on the whirl, 2πk<sub>xy</sub>r<sub>w</sub>² per cycle, where a damper c takes out
2πcωr<sub>w</sub>². So the forward modes lose damping and the backward ones gain it, and once
k<sub>xy</sub> passes cω the forward mode grows. When all the damping is at the bearings, as here, and
the orbits are circles, this balance is exact, with ω the whirl frequency at the onset.</p>
<p>As k<sub>xy</sub> rises, the forward mode's ζ falls, 0.046 → 0.033 → 0.019 → 0.004 at 0, 5,000,
10,000 and 15,000 N/m, and its frequency slips from 26.4 Hz to 26.13 Hz. The threshold with the defaults
is k<sub>xy</sub> = 16,419 N/m at both bearings, exactly cω for their 100 N·s/m at 26.13 Hz.</p>
<p><b>Oil whirl and oil whip.</b> In a real fluid-film bearing k<sub>xy</sub> is not constant. The oil
film is dragged round at a fraction λ of the shaft's speed, so the bearing's damping c acts on the
journal's motion relative to the swirling oil. Oil carried round at λΩ moves at λΩJq, with J turning x
toward y, so</p>
<p align='center'>f = −c(q̇ − λΩJq) = −cq̇ + λcΩJq</p>
<p>The second term is f<sub>x</sub> = −λcΩy, f<sub>y</sub> = +λcΩx: a cross-coupling
k<sub>xy</sub> = λcΩ that grows with the speed. The oil sticks to the turning journal and to the still
bearing, so λ is about ½: exactly ½ in short-bearing theory, and just under it (about 0.42 to 0.48) in
real plain bearings. If a bearing's dampers differ in x and y, the same algebra with the damping matrix
C gives the cross-coupled stiffness −λΩCJ. The page takes k<sub>xy</sub> = λcΩ with c the mean of the
two, which is the same for equal dampers.</p>
<p>The energy balance above then reads λcΩ against cω. The forward whirl at ω grows once <b>λΩ &gt;
ω</b>, the oil's swirl outrunning the whirl, whatever the damping. With λ = ½ the rotor goes unstable at
<b>twice its first natural frequency</b>. The <i>Oil swirl λ = ½</i> preset's onset is 3136 rpm, twice
the forward whirl's 26.1 Hz, and just there the whirl is at exactly ½X. Faster, it stays locked at the
natural frequency (26.3 Hz, 0.32×, at 5000 rpm, growing by e every 0.22 s): <i>oil whip</i>, its ratio
to the speed falling as the speed rises.</p>
<p>With a constant λ the whirl turns straight into whip at the onset. On real machines <i>oil
whirl</i> often comes first: a limit cycle near 0.45X that follows the speed up, and locks into whip once
it reaches the natural frequency. Raising the damping does not help, since it raises the cross-coupling
with it. Real machines use bearings that break up the swirl (tilting pads, pressure dams), or lower λ
with anti-swirl devices. A real oil film also stiffens as the journal moves off-centre, so the whirl
settles to a limit cycle about the size of the clearance, where a linear model grows for ever. The
page's journal bearings show both (<i>The fluid-film bearing</i>, below).</p>
<p><b>Internal (rotating) damping.</b> Damping in the shaft itself (material hysteresis, friction in
shrink fits, splines and couplings) resists the bending rate that the spinning shaft sees, not the one
seen from the bearings. In a forward whirl at ω the shaft is bent toward the orbit's centre, and that
bend turns at ω while the shaft turns at Ω. So, seen from the shaft, the bend moves at ω − Ω. Below the
critical speed (ω &gt; Ω) internal damping damps the whirl like any other damping. Above it the shaft
turns faster than it whirls, and the bend moves backward through the shaft. The internal damping force
then points forward along the orbit, and it drives the whirl.</p>
<p>That is the swirl's algebra again, with the whole shaft turning (λ = 1): the damper acts on
q̇ − ΩJq, the bending rate seen from the shaft. In fixed coordinates it gives ordinary damping
c<sub>i</sub>q̇ plus a cross-coupling k<sub>xy</sub> = c<sub>i</sub>Ω, the ΩH of the equations. On a
forward circle at ω, an external damper c<sub>e</sub> and the c<sub>i</sub>q̇ take out
2π(c<sub>e</sub> + c<sub>i</sub>)ωr<sub>w</sub>² per cycle, and the cross-coupling puts in
2πc<sub>i</sub>Ωr<sub>w</sub>². The whirl grows once c<sub>e</sub>ω + c<sub>i</sub>(ω − Ω) &lt; 0. For
the classic Jeffcott rotor on rigid bearings, with c<sub>e</sub> at the disc, the whirl is at
ω<sub>n</sub>, so</p>
<p align='center'>Ω<sub>onset</sub> = ω<sub>n</sub>(1 + c<sub>e</sub>/c<sub>i</sub>)</p>
<p>It is always above the critical speed, and with no external damping exactly at it. (On flexible
supports only the shaft's share of the motion is damped internally, so the onset rises further.) This was
the cause of the supercritical whirl of early built-up rotors, studied by Kimball and Newkirk in the
1920s. The model makes the internal damping viscous, a force in proportion to the bending rate. Real
material damping is closer to hysteretic, a fixed loss per cycle whatever the rate, and the onset speed
depends on which is assumed.</p>
<p>With the defaults and c<sub>i</sub> = 20 N·s/m the rotor is stable up to 3,910 rpm. 10 N·s/m moves
the onset up to 6,220 rpm, and 40 N·s/m down to 2,760 rpm. Below the critical, internal damping helps: at
rest the first mode's ζ rises from 0.046 to 0.078 with c<sub>i</sub> = 20 N·s/m.</p>
<p><b>The stability map</b> draws the damping ratio of each mode against speed, or its log decrement
δ = 2πζ/√(1 − ζ²). The <b>onset speed</b> is where the first forward mode crosses zero. Industry practice
(API 684) asks for δ ≥ 0.1 at the running speed; the defaults' first mode has δ = 0.29.</p>
<p><b>Beyond the onset</b> a tap, or just the unbalance's start-up transient, sets off a whirl that
grows exponentially at the mode's own natural frequency, not at the speed: a <b>subsynchronous</b> whirl.
With c<sub>i</sub> = 20 N·s/m at 5,000 rpm it is at 26.9 Hz (0.32×), growing by e every 0.28 s, while the
unbalance still turns at 83 Hz. No balancing removes it. The <b>full spectrum</b> transforms w = x + iy of
the orbit, forward whirl at positive frequencies and backward at negative. The unbalance shows at +1X,
and the instability as a growing line below it on the forward side. The model is linear, so the whirl
would grow for ever. In a machine it grows until something rubs, or the bearing's nonlinearity holds it to
a limit cycle.</p>
<p><b>On the page.</b> Set k<sub>xy</sub> and λ in the bearing table, and c<sub>i</sub> under
<i>Internal damping</i>. The <i>Stability</i> tab draws the stability map, with the full spectrum under
it. The onset is marked on the Campbell diagram too, and the growing modes there are crossed. The page
pauses when a growing whirl reaches 2% of the span.</p>

<h3>The fluid-film bearing</h3>
<p>A plain journal bearing has no balls or rollers. The journal turns in a sleeve a little larger than
itself, and the gap, the radial clearance c<sub>r</sub> (a thousandth or two of the radius), is full of
oil. Turning, the journal drags the oil round with it into the narrowing side of the gap. The oil,
squeezed, rises in pressure and holds the journal off the sleeve. Reynolds' equation gives that pressure.
In a bearing short beside its diameter (L<sub>b</sub>/D<sub>b</sub> up to about ½) the oil escapes mostly
out of the ends, and the pressure is a parabola along the bearing: <b>short-bearing</b> (Ocvirk) theory.
Where the pressure would go below zero the oil cavitates instead, so only the half of the film that
narrows carries load (the π-film).</p>
<p>At rest the journal lies on the bottom of its clearance. As the speed rises the film lifts it. Since
the film's pressure peaks ahead of the thinnest point, it also pushes the journal <b>sideways</b>, in the
direction of spin. The journal settles off the load line by the <b>attitude angle</b> χ, at an
<b>eccentricity</b> ε, its offset over c<sub>r</sub>. The Sommerfeld number
S = (μNL<sub>b</sub>D<sub>b</sub>/W)(R<sub>b</sub>/c<sub>r</sub>)², N in rev/s, and the bearing's
proportions set ε. In short-bearing theory ε depends only on S(L<sub>b</sub>/D<sub>b</sub>)², the modified
Sommerfeld number: at S = 1, halving L<sub>b</sub> moves ε from 0.32 to 0.60. The more speed and
viscosity, or the less load, the larger S, and the closer the journal to the centre, at nearly 90° to
the load. Every short bearing travels the same path, the <b>locus</b>, only at a different pace.</p>
<table align='center' width='560' cellspacing='0' cellpadding='0'>
<tr><td align='center'><img src="fig:attitude" width="560" height="330"></td></tr>
<tr><td><i>A plain journal bearing, its clearance much exaggerated. The load W pushes the journal down and the film pushes it sideways, the way it spins. It settles on the locus at the eccentricity ε, at the attitude angle χ from the load line. The film is thinnest along the line of centres.</i></td></tr>
</table>
<p>The default bearing (D<sub>b</sub> = 10 mm, L<sub>b</sub> = 5 mm, c<sub>r</sub> = 15 µm, oil of
30 mPa·s, 5 N) has ε = 0.80 at 100 rpm, 0.44 at 1,000 rpm and 0.22 at 3,000 rpm, its thinnest film
11.7 µm there.</p>
<p>For small motions about that point the film acts as springs and dampers, eight coefficients:
f = −KΔq − CΔq̇, with K = (W/c<sub>r</sub>)K̄(ε) and C = (W/(c<sub>r</sub>Ω))C̄(ε), K̄ and C̄
dimensionless. The film's sideways push makes K far from symmetric: at 3,000 rpm,
k<sub>xy</sub> = 1.36 MN/m but k<sub>yx</sub> = −1.81 MN/m. Its skew part is the cross-coupling above.
For the short bearing it is exactly k<sub>xy</sub> − k<sub>yx</sub> = ½Ω(c<sub>xx</sub> + c<sub>yy</sub>),
at every load: the oil swirls at half the shaft's speed, the swirl ratio λ = ½. As the speed rises the
cross-coupling grows with it, while the direct stiffness and damping level off.</p>
<p>So a rotor on such bearings has a <b>threshold speed</b>. Lund worked it out for a rigid rotor of
mass m<sub>b</sub> per bearing. At the threshold it whirls at ω<sub>w</sub>, neither growing nor
decaying, so s = iω<sub>w</sub> solves its characteristic equation. The real and imaginary parts give</p>
<p align='center'>m<sub>b</sub>ω<sub>w</sub>² = K<sub>eq</sub> = (k<sub>xx</sub>c<sub>yy</sub> +
k<sub>yy</sub>c<sub>xx</sub> − k<sub>xy</sub>c<sub>yx</sub> − k<sub>yx</sub>c<sub>xy</sub>) /
(c<sub>xx</sub> + c<sub>yy</sub>)</p>
<p align='center'>ω<sub>w</sub>² = ((K<sub>eq</sub> − k<sub>xx</sub>)(K<sub>eq</sub> − k<sub>yy</sub>) −
k<sub>xy</sub>k<sub>yx</sub>) / (c<sub>xx</sub>c<sub>yy</sub> − c<sub>xy</sub>c<sub>yx</sub>)</p>
<p>A rotor heavier than K<sub>eq</sub>/ω<sub>w</sub>² is unstable. But the coefficients change with the
speed, through ε, so the threshold has to be searched for. At each speed, find ε, the coefficients,
K<sub>eq</sub> and ω<sub>w</sub>, and compare K<sub>eq</sub>/ω<sub>w</sub>² with m<sub>b</sub>; the
<i>Fluid-film bearing</i> tab steps up the speed until m<sub>b</sub> exceeds it, then bisects. K scales as W/c<sub>r</sub> and C as
W/(c<sub>r</sub>Ω), so ω<sub>w</sub>/Ω depends on ε alone. For the default bearing,
K<sub>eq</sub>/ω<sub>w</sub>² is 195 kg at 1,000 rpm (ε = 0.44, ω<sub>w</sub> = 0.52Ω) and 9.0 kg at
5,000 rpm (ε = 0.14, 0.51Ω). Lightly loaded, ω<sub>w</sub> tends to half the speed: <b>oil whirl</b>. With the default bearing carrying its load as weight (0.51 kg), the rotor
goes unstable at 21,324 rpm and whirls at 0.50×. Heavily loaded bearings (ε above 0.756) are stable at
any speed. That is a result of short-bearing theory with the π-film; other bearings and film models put
the limit elsewhere. It is one reason a lightly loaded bearing is the one that whirls. A flexible rotor
goes unstable much sooner, near twice its first critical speed: the oil whip above.</p>
<p><b>On the rotor.</b> Each film carries its share of the rotor's weight. On a midspan disc that is
half of it and its own journal's; on an overhung one the far bearing is pulled up instead. So the two may
sit at different points of the locus. Each film is linearized about where its journal settles,
f = W − K<sub>f</sub>(q − q<sub>J</sub>) − C<sub>f</sub>q̇. K<sub>f</sub>(Ω) and C<sub>f</sub>(Ω) join
the rotor's stiffness and damping, and the constant p = W + K<sub>f</sub>q<sub>J</sub> joins its loads,
so the rotor rests with its journals on the locus. Below the speed where ε = 0.9 the journal is nearly on
its bottom (mixed lubrication), and the film is taken as it is there.</p>
<p>The <i>Journal bearings</i> preset is the default rotor laid horizontal on the default bearing. The
film, 1 MN/m or so, is far stiffer than the shaft (38 kN/m at the disc), so the critical speed is close
to the rigid-bearing one: 1845 rpm, against 1854 rpm on rigid bearings. The film damps it little
(ζ = 0.007). The film is not isotropic, so the backward whirl crosses 1X too, at 1853 rpm, even more
lightly damped (ζ = 0.002). The unbalance excites both, and through the pair the disc's orbit is an
ellipse.</p>
<p>Linearized, nothing holds the response inside the clearance. At the critical the journals' steady
1X orbits reach about 28 µm (their half-length), almost twice the 15 µm clearance. A real journal would
touch the bearing long before that, so read them as the linear model's prediction only.</p>
<p>A little below twice the critical, at 3576 rpm, the forward whirl grows at 30.5 Hz (0.51×). This is
oil whip from the bearing itself, not from a chosen λ. It is locked at 30.7 Hz at 5000 rpm (0.37×),
growing by e every 0.85 s. Long before it reaches the size of the disc's whirl, it fills the 15 µm
clearance. A vertical rotor's bearings carry no load, and an unloaded plain bearing has no direct
stiffness, only the swirl's cross-coupling. On them the rotor whirls at any speed, which is why vertical
machines use tilting-pad or preloaded bearings.</p>
<p><b>The whole film.</b> The linearization holds for small motions about where the journal rests. Off
there the film's force is far from linear: at ε → 1 the film between journal and wall goes to nothing,
and its stiffness and damping grow without bound, so a journal never reaches the wall. The short
bearing's whole force has a closed form too. The film carries pressure over half the bearing: the half
where the gap, the oil dragged into it at half the speed and squeezed by the journal, closes. That half
turns with the journal's velocity, set by its speed along the line of centres and how fast it whirls
across it against Ω/2. Over it, with ϑ the angle from the line of centres, the force is integrals of
sin²ϑ, sin ϑ cos ϑ and cos²ϑ over (1 − ε cos ϑ)³. In the angle ϑ<sub>e</sub> with
tan(ϑ<sub>e</sub>/2) = √((1 + ε)/(1 − ε)) tan(ϑ/2), the eccentric anomaly of an orbit, each is a short
trigonometric polynomial.</p>
<p>A journal circling the centre at exactly half the speed leaves the film nothing to squeeze or drag,
and it carries no load at all. That is where oil whirl lives. The film
pushes a slower whirl along and holds a faster one back, so past twice its natural frequency, where the
rotor's whirl is slower than half the speed, the film feeds it. Tick <i>Nonlinear film</i> under the bearing supports and the time
response takes the films' whole force (<i>Numerics</i>), while the Campbell diagram, stability and the
steady-state curves stay those of the linearized films.</p>
<p>On the <i>Journal bearings</i> preset, run up through the critical at 500 rpm/s: linearized, the
journals come within 1% of the wall (ε = 0.99); whole, the films hold them to ε = 0.63, while the disc's
peak is much the same, 0.66 mm. Past the onset the films keep the journals in their clearance, but the
shaft, far softer, bends on: its whirl grows ever more slowly, but does not stop, until it would rub. On
the <i>Oil whirl</i> preset, upright, the whirl at 3500 rpm (below twice the first natural frequency) is
at half the speed, 29.2 Hz. It settles to a limit cycle with the journals circling past ε = 0.9 and the
disc about 0.16 mm off centre. At 5000 rpm it is locked at 30.8 Hz instead: oil whip, which grows on.</p>
<p><b>On the page.</b> The <i>Fluid-film bearing</i> tab draws the locus, with the page's speed range
solid and the speed now as a dot. It plots the coefficients against the speed, or dimensionless against
S(L<sub>b</sub>/D<sub>b</sub>)² on log scales (dashed where negative), as on a bearing designer's chart.
Tick <i>Oil-film journal bearings</i> under the bearing supports to run both journals in the tab's
bearing, on rigid housings, in place of the springs and dampers, and <i>Nonlinear film</i> beside it to
step their whole force. With <i>Static sag</i> ticked, the bearings' orbits rise along the locus as the
speed rises.</p>

<h2>6. Numerics</h2>

<h3>Why state space</h3>
<p>Real modes and modal superposition need symmetric M and K, damping in proportion to them, and modes
that do not change with the speed. Here ΩG is skew-symmetric and changes with the speed, and the support
damping is not proportional. So the modes are complex and depend on the speed. The first-order form
Z' = A(Ω)Z + Bu handles all of it. The eigenvalues of A give each mode's frequency, damping and whirl
direction, and the time response is stepped exactly.</p>
<p>As on the other pages, each step is the exact first-order-hold matrix-exponential update,
Z<sub>k+1</sub> = ΦZ<sub>k</sub> + Γ<sub>0</sub>u<sub>k</sub> + Γ<sub>1</sub>u<sub>k+1</sub>, with
Φ = e<sup>Ah</sup> for a step h. It is stable however stiff the shaft. While the speed ramps, A(Ω) is
updated every 2 ms of simulated time, and the unbalance force follows the exact angle φ, including the
Ω' term. The ramp's gyroscopic term I<sub>p</sub>Ω'θ is left out: at 5000 rpm/s it is about 0.03% of the
disc's tilt stiffness.</p>

<h3>Stepping the whole film</h3>
<p>The films' whole force is not linear, so no one Φ steps it. Each step is still the exact update, of
the films linearized, with the rest of their force, the remainder r<sub>f</sub>, as a load on the
journals, linear over the step as the loads are:</p>
<p align='center'>Z<sub>k+1</sub> = ΦZ<sub>k</sub> + Γ<sub>0</sub>(u<sub>k</sub> + r<sub>f,k</sub>) +
Γ<sub>1</sub>(u<sub>k+1</sub> + r<sub>f,k+1</sub>)</p>
<p>r<sub>f,k+1</sub> depends on Z<sub>k+1</sub>, so it is found by Newton's method. It has to be: a film damps
its journal in microseconds, far faster than the step. That speed is also the trap. If the exact update
carried the films' damping at the journals' rest, the remainder near the wall, many times stiffer, would
make the update ring from step to step and throw the journal through the wall. So the update carries
C<sub>b</sub> instead: the damping of the whole film, uncavitated, where the journal is. The film's
damping is a sum of positive terms over the half carrying pressure, so it is never more than
C<sub>b</sub>, whatever the velocity. The remainder is then never stiffer than the update, and the step
stays second order in h. As the journal moves round, C<sub>b</sub> is taken afresh, and Φ with it.</p>
<p>The film's force is also minus the gradient, in the journal's velocity, of its dissipation Ψ, which
is convex. So each step's journal velocities are where a convex function is least, and a Newton step
that does not lower it is cut back. That keeps Newton right where the film's force bends sharply: at the
half-speed whirl, which oil whirl circles. The step need only resolve the spin and the force on the
journals, so it is coarser than the linear films' (10 steps a period of the fastest mode), but each costs
a hundred times more. Past what a frame can compute, the page runs slower than the speed set.</p>

<h3>Keeping a shaft with mass fast</h3>
<p>Every beam element adds modes. The highest are at tens of kHz, far above anything the rotor does at
these speeds, but the time stepping would still have to resolve them. So the model is reduced the way
the <i>Substructuring</i> page reduces a substructure (Craig–Bampton). The stations' DOFs are kept as
they are. The rest of the shaft moves in the static shape they give it, which is all a massless shaft
ever does, plus its lowest modes with the stations held, up to 1000 Hz. Those modes' amplitudes ξ join q,
after the stations. The reduced model's modes are within 0.2% of the full element model's below 500 Hz,
and within 1% up to 1000 Hz (60,000 rpm, the top of the speed range). With a massless shaft there are no
such modes, and the reduction is exactly the Jeffcott model.</p>

<h2>Things to try</h2>
<h3>1. Fundamentals</h3>
<ol>
<li>Ramp from 1000 to 3000 rpm at 500 rpm/s and watch the disc's orbit grow through 1584 rpm and shrink
again toward the 20 µm eccentricity (28 µm at 3000 rpm), while the heavy spot swings round to the
opposite side of the orbit. Then try 5000 rpm/s: the whirl has no time to build, and the peak is much
lower.</li>
<li>Set the target to 1584 rpm and read the phase: the heavy spot leads the high spot by about 90°.</li>
<li>Pause near the critical speed and turn the 3D view: the shaft bows in one plane, which turns with the
spin, rather than flapping back and forth.</li>
<li>Bearing damping has an optimum. With the defaults the first mode has ζ = 0.046 at c = 100 N·s/m and
0.089 at 400 N·s/m, but only 0.005 at 10,000 N·s/m. Dampers that stiff lock the journals, so the supports
stop moving and nothing is left to damp. The critical speed then climbs toward the rigid-bearing
value.</li>
<li>Move the disc to a/L = 0.3. The first critical splits slightly, into 1731 rpm backward and 1737 rpm
forward, and the disc's tilting mode splits strongly with speed (from 360 Hz at rest to 465 Hz forward and
289 Hz backward at 6000 rpm). Tap the disc at speed and the orbit becomes a rosette.</li>
<li>Load the <i>Overhung rotor</i> preset and look at the Campbell diagram: the first mode's forward and
backward branches fan out from 14.5 Hz. Run up and coast down from 600 to 2000 rpm: one critical, at
1284 rpm. Then set the disc's polar inertia I<sub>p</sub> to a tiny value. With the gyroscopic moment gone,
the critical drops to 870 rpm, the natural frequency at rest.</li>
<li>Untick <i>Isotropic</i> and set k<sub>x</sub> = 20,000 N/m and k<sub>y</sub> = 80,000 N/m on A
(B follows). The critical splits into 1365 rpm (horizontal) and 1668 rpm (vertical). Run between them,
say at 1500 rpm, and the orbit is a backward ellipse (backward from about 1413 to 1632 rpm).</li>
<li>Load the <i>Long shaft</i> preset and run up from 800 to 8000 rpm at 1000 rpm/s. Past 6582 rpm the
shaft bends in its second mode, the one the dashed massless model doesn't have. Then set <i>Beam
elements</i> to 2 and to 4, consistent and lumped, and watch the second critical move on the Campbell
diagram.</li>
</ol>
<h3>2. Measuring the response</h3>
<ol>
<li>Run up and coast down from 800 to 2400 rpm at 500 rpm/s, then again at 5000 rpm/s with <i>Hold
earlier sweeps</i> ticked, and compare them with the steady-state curve on the Bode and polar plots. Then
raise the support damping to 400 N·s/m (ζ = 0.089, and the peak moves up to 1758 rpm). At 500 rpm/s the
sweeps now come within a few percent of the steady-state peak (101% up, 96% down), since 2ζ² has grown
fourfold.</li>
<li>Load the <i>Overhung rotor</i> preset, set the ramp rate to 200 rpm/s and tick <i>Tap the disc
every</i> 0.5 s. Then set the target to 2,500 rpm and watch the <i>Waterfall</i> tab: the taps' peaks
follow the dots of the forward and backward branches (backward from 14.5 down to 5.7 Hz), and the 1X ridge
grows where it crosses the forward branch. Switch to <i>Time</i> and set a steady speed: the taps' peaks
then stay at one frequency.</li>
</ol>
<h3>3. Once-per-revolution forces and balancing</h3>
<ol>
<li>Set the unbalance to 0 and the shaft bow to 20 µm. Turning slowly, the disc's orbit is 20 µm round,
the bow. It peaks at 218 µm at the critical and falls to 8 µm at 3000 rpm (towards zero, not towards e).
Run up and coast down and look at the polar plot: it starts at the bow vector.</li>
<li>Put the 20 g·mm unbalance back and set the bow's angle to 180°: the disc's orbit stays at 20 µm at
every speed. Now set the bow to 10 µm: the orbit dips to about 1 µm at 1116 rpm, where the bow and the
unbalance cancel, and still peaks at 110 µm at the critical.</li>
<li>Set the unbalance to 0 and skew the disc by 0.1°. At midspan only the disc tilts (watch it wobble in
the 3D view). Move it to a/L = 0.3 and the disc whirls too, peaking at 14 µm at 1752 rpm.</li>
<li>On the <i>Balancing</i> tab, press <i>Mystery unbalance</i> and balance the rotor in four steps: slow
roll, as found, trial run, correct and check. The check should read well under 1% of the reading as
found; <i>Reveal</i> then shows the unbalance you cancelled.</li>
<li>Set a bow of 20 µm at 90°, put the probe on the disc, and balance at 1200 rpm with and without
<i>Subtract slow roll</i>. Compare the curves as found and now under the vectors.</li>
<li>Move the disc to a/L = 0.3, choose <i>Two planes: the disc's faces</i> on the <i>Balancing</i> tab
and press <i>Mystery unbalance</i>: now the disc is skewed too, by a hidden angle. Balance it in five
steps, and the curve <i>Now</i> stays flat at every speed. Then try <i>Two planes on the shaft</i>, near
the bearings, and see the critical come back.</li>
</ol>
<h3>4. Gravity, and stiffness that turns</h3>
<ol>
<li>Tick <i>Gravity</i>: the rotor sags, 378 µm at the disc, and whirls about the sagged position, the
orbits sitting below the bearings' centre. Untick <i>Static sag</i> to look at the vibration alone: it is
the same as without gravity. Make the supports anisotropic with no unbalance: the rotor sags, spins up and
stays still.</li>
<li>Load the <i>Two-pole rotor</i> preset, choose <i>Orbit</i> and <i>2X</i> beside the probe, and run up
and coast down from 500 to 1200 rpm at 200 rpm/s: the 2X peaks at 785 rpm, half the critical speeds. On
the <i>Stability</i> tab choose <i>Asymmetry chart</i>, set the target to 1580 rpm and tap the disc: the
dot sits inside the tongue, and the whirl grows at exactly 1X. Raise the support damping to 400 N·s/m
(both directions) and the band is gone: the threshold rises to Δ = 0.178.</li>
<li>Untick <i>Horizontal rotor</i> on the <i>Two-pole rotor</i> preset, set the asymmetry to 10% and the
unbalance to 20 g·mm, and look at the disc's orbit on the Bode plot with the heavy spot at 0°, 45° and
90°: one tall peak, two short ones, one tall one.</li>
<li>Keep that rotor with the heavy spot at 45° and balance it on the <i>Balancing</i> tab, reading at the
disc, with the trial at 0°. The correction comes out near 1.18 g at 237°, not the 1 g at 225° that would
cancel the unbalance: the check run still reads about 9 µm, and the orbit now peaks near 283 µm at the
critical, against 201 µm before. One trial cannot balance an asymmetric shaft (see <i>Balancing it</i>
under <i>The asymmetric shaft</i>).</li>
<li>Load the <i>Cracked shaft</i> preset, choose <i>Orbit</i> and <i>3X</i> beside the probe, set the
ramp rate to 100 rpm/s and run up and coast down from 400 to 1000 rpm. The 3X peaks at 547 rpm going up
and 499 rpm coming down, either side of the steady state's 522 rpm (a third of the critical speed), and
the 2X (choose <i>2X</i>) at 804 and 757 rpm about 783 rpm. The 3X lags most, since it sweeps through its
resonance three times as fast as the speed. Untick <i>Breathing</i>: the 3X is gone and the 2X doubles.
Tick it again and turn the crack's angle to 180°, opposite the heavy spot: the 1X at the critical speed
falls from 164 to 52 µm.</li>
</ol>
<h3>5. Stability</h3>
<ol>
<li>Set the internal damping c<sub>i</sub> to 20 N·s/m and look at the <i>Stability</i> tab. The forward
first mode's damping falls with speed and crosses zero at 3,910 rpm, while the backward mode's rises.
Ramp to 5,000 rpm and tap the disc: the whirl grows at 26.9 Hz, a line at 0.32× on the full spectrum,
until the page pauses it. Lower the target to 3,000 rpm and Reset: below the onset the tap dies away.</li>
<li>Set c<sub>i</sub> back to 0 and give both bearings k<sub>xy</sub> = 15,000 N/m. The forward mode's ζ
drops to 0.004 at every speed, and the backward one's rises to 0.075. Above 16,419 N/m the rotor is
unstable even at rest. More support damping raises the threshold.</li>
<li>Load the <i>Oil swirl λ = ½</i> preset and tick ½X on the Campbell diagram. Run at 2500 rpm and tap
the disc: the tap dies away. Set the target to 4000 rpm (past the onset, 3136 rpm, where ½X crosses the
first mode) and tap again: the whirl grows at 26 Hz, 0.39× on the full spectrum, not at ½X. Double the
support damping: the onset does not move.</li>
<li>Load the <i>Journal bearings</i> preset, tick <i>Static sag</i> and run up to 3000 rpm. The bearings'
orbits climb from where the journals rest, (4.8, −12.6) µm, toward the centre along the locus on the
<i>Fluid-film bearing</i> tab. Then set the target to 5000 rpm and tap the disc: the whirl grows at
30.7 Hz, the journals staying in their clearance while the disc's whirl grows on. Untick <i>Nonlinear
film</i>, Reset and run again: the linearized films let a journal reach its clearance, and the page
pauses. Then make the clearance 30 µm and the oil 10 mPa·s (hotter). The journals sink lower (ε = 0.71
at 3000 rpm, against 0.25), the bearings more heavily loaded, and the onset rises from 3576 to 4358 rpm.
A heavily loaded plain bearing resists whirl.</li>
<li>Load the <i>Oil whirl</i> preset and watch the bearings' orbits: past 3000 rpm a whirl at half the
speed sets in and grows until the journals circle close to the wall, where it holds. The full spectrum
shows it at +½X. Set the target to 5000 rpm: the whirl stays at the natural frequency, 30.8 Hz, and
grows on, oil whip. Untick <i>Nonlinear film</i> to see the linear films' whirl at 3500 rpm grow until a
journal reaches its clearance.</li>
</ol>

<h2>Notation</h2>
<table border="1" cellspacing="0" cellpadding="4">
<tr><th>Symbol</th><th>Meaning</th><th>Units</th></tr>
<tr><td colspan="3"><b>The rotor</b></td></tr>
<tr><td>L, a, b, ℓ</td><td>bearing span; the disc's distance from A and from B (a + b = L); an overhung disc's distance past its bearing</td><td>m</td></tr>
<tr><td>z</td><td>position along the shaft, from bearing A</td><td>m</td></tr>
<tr><td>d, E, ν, EI</td><td>shaft diameter, Young's modulus, Poisson's ratio, bending stiffness E π d⁴/64</td><td>m, Pa, —, N·m²</td></tr>
<tr><td>m, I<sub>p</sub>, I<sub>d</sub></td><td>disc mass; its polar and diametral moments of inertia</td><td>kg, kg·m²</td></tr>
<tr><td>k<sub>x</sub>, k<sub>y</sub>, c<sub>x</sub>, c<sub>y</sub></td><td>a support's stiffness and damping, horizontal and vertical</td><td>N/m, N·s/m</td></tr>
<tr><td>k, c</td><td>stiffness and damping in a one-mass model</td><td>N/m, N·s/m</td></tr>
<tr><td>k<sub>s</sub>, k<sub>b</sub>, k<sub>eq</sub>, k<sub>θ</sub></td><td>the shaft's midspan stiffness; a support's; the two in series; a disc's tilt stiffness</td><td>N/m, N·m/rad</td></tr>
<tr><td>ρ, ξ</td><td>the shaft's density; the amplitudes of its modes with the stations held (mass normalized)</td><td>kg/m³, √kg·m</td></tr>
<tr><td colspan="3"><b>Motion and speed</b></td></tr>
<tr><td>x, y</td><td>horizontal and vertical displacement of the shaft's centre</td><td>m</td></tr>
<tr><td>w, w<sub>f</sub>, w<sub>b</sub></td><td>the complex displacement x + iy; the forward and backward circles of an orbit at one frequency, w = w<sub>f</sub>e<sup>iωt</sup> + w<sub>b</sub>e<sup>−iωt</sup></td><td>m</td></tr>
<tr><td>θ<sub>x</sub>, θ<sub>y</sub></td><td>shaft's slope at the disc, dx/dz and dy/dz</td><td>rad</td></tr>
<tr><td>q</td><td>the DOFs: the stations' displacements and slopes (then ξ)</td><td>m, rad</td></tr>
<tr><td>Ω, Ω', φ</td><td>spin speed; its rate of change on a ramp; angle of the keyphasor mark on the shaft from +x (φ' = Ω)</td><td>rad/s (rpm), rad/s² (rpm/s), rad</td></tr>
<tr><td>ω, ω<sub>n</sub>, ω<sub>c</sub></td><td>a whirl frequency; a natural frequency; a critical speed</td><td>rad/s (Hz, rpm)</td></tr>
<tr><td>r</td><td>speed ratio Ω/ω<sub>c</sub></td><td>—</td></tr>
<tr><td>r<sub>w</sub>, X</td><td>radius of a circular whirl orbit; a one-mass model's complex amplitude, w = Xe<sup>iΩt</sup></td><td>m</td></tr>
<tr><td>ζ, δ</td><td>damping ratio of a mode, −Re s / |s|; log decrement, 2πζ/√(1 − ζ²)</td><td>—</td></tr>
<tr><td>s</td><td>eigenvalue of a mode (for a turning stiffness, a Floquet exponent)</td><td>1/s</td></tr>
<tr><td colspan="3"><b>Equations</b></td></tr>
<tr><td>M, C, G, K, H</td><td>mass, damping, gyroscopic, stiffness and circulatory matrices (8 × 8 for a massless shaft); H is the skew stiffness per unit speed, from internal damping and fluid swirl</td><td>—</td></tr>
<tr><td>J</td><td>the matrix that turns x toward y, in each pair of planes</td><td>—</td></tr>
<tr><td>Z, u, A(Ω), B</td><td>state vector [q, q̇]; the loads; the state and input matrices (16 × 16 and 16 × 8 for a massless shaft)</td><td>—</td></tr>
<tr><td>h, Φ, Γ<sub>0</sub>, Γ<sub>1</sub></td><td>the time step; the exact update's matrices</td><td>s, —</td></tr>
<tr><td>r<sub>f</sub>, C<sub>b</sub></td><td>stepping the whole film: the remainder of its force on the journals; the whole uncavitated film's damping, which bounds the film's</td><td>N, N·s/m</td></tr>
<tr><td>R(φ), q<sub>r</sub></td><td>the rotation to axes turning with the shaft; q in those axes, q = R(φ)q<sub>r</sub></td><td>—, m</td></tr>
<tr><td>Q<sub>0</sub>, Q<sub>n</sub></td><td>a periodic steady state's mean and its nX harmonic</td><td>m</td></tr>
<tr><td>Ω<sub>s</sub></td><td>the harmonic spacing of a turning stiffness (2Ω, or Ω for a breathing crack)</td><td>rad/s</td></tr>
<tr><td colspan="3"><b>Once-per-revolution forces and balancing</b></td></tr>
<tr><td>U = m e, ψ</td><td>unbalance: disc mass times its centre of mass's offset e; the heavy spot's angle from the mark</td><td>kg·m (g·mm), rad</td></tr>
<tr><td>δ<sub>b</sub>, β, θ<sub>b</sub></td><td>shaft bow at the disc, its angle from the keyphasor mark, and the bowed shaft's slope at the disc</td><td>m (µm), rad, rad</td></tr>
<tr><td>τ, γ</td><td>disc skew (its axis's tilt from the shaft's), and its angle from the keyphasor mark</td><td>rad (°)</td></tr>
<tr><td>V</td><td>1X vector: amplitude |V|, phase lag −arg V</td><td>m</td></tr>
<tr><td>V<sub>s</sub>, V<sub>0</sub>, V<sub>j</sub></td><td>the readings: slow roll, as found, with the trial weight in plane j</td><td>m</td></tr>
<tr><td>T, W<sub>c</sub>, U<sub>w</sub></td><td>trial and correction weights, and any other balance weight (mass × radius at an angle)</td><td>kg·m (g·mm)</td></tr>
<tr><td>α; α<sub>+</sub>, α<sub>−</sub></td><td>influence coefficient (V<sub>1</sub> − V<sub>0</sub>)/T, or matrix; on an asymmetric shaft, the coefficients of U and Ū, V = α<sub>+</sub>U + α<sub>−</sub>Ū</td><td>m per kg·m</td></tr>
<tr><td>T<sub>b</sub></td><td>a spectrum's block length</td><td>s</td></tr>
<tr><td colspan="3"><b>Gravity, the asymmetric shaft and the crack</b></td></tr>
<tr><td>g, f<sub>g</sub>, q<sub>s</sub></td><td>gravity; the weights on the journals and the disc; the static deflection under them</td><td>m/s², N, m</td></tr>
<tr><td>x<sub>r</sub>, y<sub>r</sub></td><td>a Jeffcott rotor's displacement in axes turning with the shaft, along its stiff and soft axes</td><td>m</td></tr>
<tr><td>Δ, β<sub>s</sub></td><td>shaft asymmetry (EI(1 ± Δ) about its stiff and soft axes), and the stiff axis's angle from the keyphasor mark</td><td>—, rad (°)</td></tr>
<tr><td>ΔK(φ)</td><td>the asymmetric or cracked shaft's stiffness less its mean, turning with it</td><td>N/m</td></tr>
<tr><td>a<sub>c</sub>, β<sub>c</sub></td><td>a crack's depth from the shaft's surface (a<sub>c</sub>/d of the diameter); its mouth's angle from the keyphasor mark</td><td>m, rad (°)</td></tr>
<tr><td>κ, κ<sub>o</sub>, κ<sub>a</sub>, Δθ</td><td>a crack's compliance (the jump in slope across it per unit bending moment), opening it and across; the jump itself</td><td>rad/(N·m), rad</td></tr>
<tr><td>b<sub>h</sub>, k<sub>h</sub></td><td>the moment at a crack per unit of each station's displacement; the shaft's stiffness against a jump in slope there</td><td>N, N·m/rad</td></tr>
<tr><td>σ(φ)</td><td>how open a breathing crack is, 0 (shut) to 1</td><td>—</td></tr>
<tr><td>P, D<sub>o</sub>, D<sub>a</sub>, E<sub>n</sub></td><td>the crack's loss of stiffness on the stations: its mean; held open, along and across the mouth; its nX harmonic</td><td>N/m</td></tr>
<tr><td colspan="3"><b>Stability and the fluid-film bearing</b></td></tr>
<tr><td>k<sub>xy</sub>, k<sub>yx</sub>; c<sub>xy</sub>, c<sub>yx</sub></td><td>a bearing's cross-coupled stiffness (k<sub>yx</sub> = −k<sub>xy</sub> unless given) and damping: f<sub>x</sub> = −k<sub>xy</sub>y − c<sub>xy</sub>ẏ</td><td>N/m; N·s/m</td></tr>
<tr><td>c<sub>i</sub>, c<sub>e</sub>, η</td><td>internal (rotating) damping of the shaft, as a damper at the disc; external damping at the disc; η = c<sub>i</sub>/k<sub>disc</sub></td><td>N·s/m, s</td></tr>
<tr><td>λ</td><td>a fluid-film bearing's swirl ratio: k<sub>xy</sub> grows by λcΩ, c the mean of c<sub>x</sub> and c<sub>y</sub></td><td>—</td></tr>
<tr><td>D<sub>b</sub>, R<sub>b</sub>, L<sub>b</sub>, c<sub>r</sub>, μ, W</td><td>on the <i>Fluid-film bearing</i> tab: journal diameter and radius, bearing length, radial clearance, oil viscosity, load</td><td>m, m, m, m (µm), Pa·s, N</td></tr>
<tr><td>ε, χ</td><td>the journal's eccentricity (its offset over c<sub>r</sub>); its attitude angle from the load line</td><td>—, rad (°)</td></tr>
<tr><td>S, N</td><td>the Sommerfeld number (μNL<sub>b</sub>D<sub>b</sub>/W)(R<sub>b</sub>/c<sub>r</sub>)²; the speed in rev/s</td><td>—, 1/s</td></tr>
<tr><td>K̄, C̄</td><td>a bearing's coefficients made dimensionless: K c<sub>r</sub>/W and C c<sub>r</sub>Ω/W</td><td>—</td></tr>
<tr><td>K<sub>f</sub>, C<sub>f</sub>, q<sub>J</sub>, p</td><td>on the rotor, the films' stiffness and damping; the journals' equilibrium; the films' constant load W + K<sub>f</sub>q<sub>J</sub></td><td>N/m, N·s/m, m, N</td></tr>
<tr><td>ϑ, ϑ<sub>e</sub>, Ψ</td><td>the whole film: the angle round the bearing from the line of centres, and its eccentric anomaly; the film's dissipation</td><td>rad, rad, W</td></tr>
<tr><td>m<sub>b</sub>, ω<sub>w</sub>, K<sub>eq</sub></td><td>Lund's rigid rotor: its mass per bearing; its whirl frequency at the threshold; the film's equivalent stiffness</td><td>kg, rad/s, N/m</td></tr>
</table>
"""


def _crack_note(s: RotorSystem) -> str:
    """The live notes' section on the crack: its compliance, its loss of stiffness and how it breathes."""
    crack = s.crack
    c_open, c_across = s.crack_compliances
    if not s.crack_opening:
        state = ("It is a breathing crack, but nothing opens it: with no gravity its faces stay pressed together, "
                 "and the shaft is as if round. Untick <i>Breathing</i> to hold it open.")
    elif crack.breathing:
        down = s.crack_sign > 0
        state = ("It breathes: the shaft's weight opens it as its mouth turns "
                 + ("down (the shaft sags there, stretching its bottom)" if down else
                    "up (the shaft hogs there, stretching its top)")
                 + ", and shuts it half a turn later, open by σ(φ) = (1 "
                 + ("−" if down else "+") + " sin(φ + β<sub>c</sub>))/2. The loss is σ(φ) times the open crack's, "
                 "so it has the mean ½P in K and harmonics 1X, 2X and 3X: ΔK(φ) = Σ E<sub>n</sub>e<sup>inφ</sup> + "
                 "conj. Even on isotropic supports it changes in axes turning with the shaft (the weight does "
                 "not turn), so the modes below come from Hill's method, from harmonics up to ±3Ω.")
    else:
        state = ("It is held open: the loss is P + [the deviatoric part], an asymmetric shaft's, its soft axis along "
                 "the mouth, turning with the shaft.")
    return (
        f"<h4>The crack</h4><p>A crack {crack.depth:.0%} of the shaft's diameter deep, at z = {s.crack_z * 1e3:.4g} mm, "
        f"its mouth at β<sub>c</sub> = {math.degrees(crack.angle):.0f}° from the mark. Open, it is a hinge whose "
        f"compliance is κ<sub>o</sub> = {c_open:.3g} rad/(N·m) to bending that opens it and κ<sub>a</sub> = "
        f"{c_across:.3g} across. Condensed onto the stations it takes D = b<sub>h</sub>b<sub>h</sub><sup>T</sup>κ/"
        "(1 + κk<sub>h</sub>) off one plane's stiffness, b<sub>h</sub> the bending moment at the crack per unit of "
        "each station's displacement and k<sub>h</sub> the "
        "shaft's stiffness against a jump in its slope there: D<sub>o</sub> along the mouth and D<sub>a</sub> "
        "across, so in fixed axes P = (D<sub>o</sub> + D<sub>a</sub>)/2 on both planes and the deviatoric part of "
        f"(D<sub>o</sub> − D<sub>a</sub>)/2 turning at 2X, as the asymmetric shaft's. {state}</p>"
    )
