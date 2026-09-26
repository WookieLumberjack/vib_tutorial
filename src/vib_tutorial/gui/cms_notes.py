"""Text for the Substructuring page: the theory notes and the live matrix walkthrough."""

from __future__ import annotations

import numpy as np

from ..core import (
    CraigBamptonModel,
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
<h3>Why substructure?</h3>
<p>Real structures are built from components: an engine on a frame, a wing on a fuselage,
a payload on a launcher. <b>Dynamic substructuring</b> models each component on its own,
reduces it to a handful of coordinates, and then joins the reduced components at their
interfaces. The assembled model is much smaller than the full one, and a component can
be changed or re-analysed without touching the others. <b>Component mode synthesis
(CMS)</b> does the reduction with each component's own mode shapes.</p>
<p>On this page the chain is cut at one <i>interface</i> mass into two substructures,
<b>A</b> (grounded) and <b>B</b> (free end). The interface mass belongs to both: A owns its
mass and the springs to its left, B owns the springs to its right.</p>

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
corrections.</p>

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


def matrices_html(model: CraigBamptonModel, full: ModalResult | None = None) -> str:
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
            f"m<sub>{j + 1}</sub> = {fmt(system.masses[j], 1)} goes to {left.name} (any split works: "
            "assembly adds them back). Every other entry belongs to one substructure only, so "
            "K = Σ L<sub>s</sub><sup>T</sup>K<sup>(s)</sup>L<sub>s</sub> exactly, and the same for C and M.</p>"
        )

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
            for r, f in enumerate(sub.fixed_omegas / TWO_PI)
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
    parts.append(matrix_html(model.T, labels, glabels, groups, ggroups, "T"))
    freqs = ", ".join(f"{f:.4g}" for f in model.fn_hz)
    parts.append(f"<p>Reduced-model natural frequencies: {freqs} Hz.</p>")
    parts.append(_damped_comparison(model, full))
    return "".join(parts)


def _damping_note(sub: Substructure) -> str:
    if not sub.n_kept:
        return (f"<p><b>{sub.name}</b> keeps no modes, so Ĉ<sup>({sub.name})</sup> is just the damping "
                "seen from the boundary (Ĉ<sub>bb</sub>).</p>")
    modal, boundary = substructure_damping(sub)
    zetas = ", ".join(
        f"{sub.C_red[r, r] / (2 * w):.4f}" for r, w in enumerate(sub.fixed_omegas[: sub.n_kept])
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


def _damped_comparison(model: CraigBamptonModel, full: ModalResult | None) -> str:
    """Step 9: exact damping ratios of the reduced and the full model, mode by mode."""
    out = ["<h3>9. Damping in the reduced model</h3>",
           "<p>The frequencies above ignore damping. The reduced model does include it: Ĉ is used "
           "for the FRF on the Modes &amp; FRF tab, and the damped eigenvalues of "
           "M̂η̈ + Ĉη̇ + K̂η = 0 give its damping ratios. Both columns are exact (state-space) "
           "values, so any difference comes from the reduction alone. The comparison table on the left shows the same two columns.</p>"]
    if full is None:
        return "".join(out)
    rows = ["<table border='1' cellspacing='0' cellpadding='3'>"
            "<tr><th>Mode</th><th>ζ true</th><th>ζ CB</th><th>ζ error</th></tr>"]
    for c in compare_modes(model, full):
        cells = [str(c.index), "overdamped" if c.zeta_true is None else f"{c.zeta_true:.4f}"]
        if c.fn_cb is None:
            cells += ["<span style='color:#999'>not in model</span>", "—"]
        elif c.zeta_cb is None:
            cells += ["overdamped", "—"]
        else:
            err = c.zeta_error
            color = "#000" if err is None else "#2a7d2a" if abs(err) < 1e-3 else "#b07000" if abs(err) < 0.05 else "#c1121f"
            cells += [f"{c.zeta_cb:.4f}", "—" if err is None else f"<span style='color:{color}'>{100 * err:+.3g}%</span>"]
        rows.append("<tr>" + "".join(f"<td align='right'>{x}</td>" for x in cells) + "</tr>")
    rows.append("</table>")
    out.append("".join(rows))
    out.append("<p>With stiffness-proportional damping (the default), ζ<sub>r</sub> = βω<sub>r</sub>/2, "
               "so ζ CB errs exactly as much as the frequency does. With non-proportional damping "
               "(try c<sub>1</sub> = 15 on the Simulation page), the coupling terms of Ĉ matter, and "
               "truncating modes also loses the damping they carried: the higher modes' ζ can be off "
               "by much more than their frequency.</p>")
    return "".join(out)
