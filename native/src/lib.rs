//! Rust kernels for vib_tutorial's hot loops, imported as `vib_tutorial._native`.
//!
//! Each kernel mirrors a Python function that stays the reference implementation
//! (`vib_tutorial.core.native` picks between them), and the tests compare the two.

use std::f64::consts::PI;

use pyo3::prelude::*;

/// The whole film's force is worked out no nearer the wall than this
/// (`fluid_bearing.ECCENTRICITY_MAX`; a test checks they agree).
const ECCENTRICITY_MAX: f64 = 1.0 - 1e-6;

/// (f_x, f_y, k_xx, k_xy, k_yx, k_yy, c_xx, c_xy, c_yy, Ψ), a tuple in Python as `pi_film` returns.
type Film = (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64);

/// The π-film's whole force on the journal, its derivatives and its dissipation:
/// `fluid_bearing.pi_film`, line for line (its docstring has the derivation).
#[pyfunction]
fn pi_film(k: f64, c: f64, x: f64, y: f64, vx: f64, vy: f64, omega: f64) -> Film {
    let mut e = x.hypot(y);
    let (cp, sp);
    if e < 1e-9 * c {
        // centred: the line of centres is any direction
        (cp, sp) = (1.0, 0.0);
        e = 1e-9 * c;
    } else {
        (cp, sp) = (x / e, y / e);
    }
    let mut eps = e / c;
    if eps > ECCENTRICITY_MAX {
        eps = ECCENTRICITY_MAX;
        e = eps * c;
    }
    let vr = vx * cp + vy * sp; // e'
    let vt = vy * cp - vx * sp; // e ψ'
    let a = omega * e - 2.0 * vt;
    let b = -2.0 * vr;
    // The film carries pressure for φ from p1 to p1 + π; u is φ's eccentric anomaly.
    let p1 = PI - b.atan2(a);
    let s2 = 1.0 - eps * eps;
    let s = s2.sqrt();
    let beta = eps / (1.0 + s);
    let (sn, cs) = p1.sin_cos();
    let u1 = p1 + 2.0 * (beta * sn / (1.0 - beta * cs)).atan();
    let u2 = p1 + PI - 2.0 * (beta * sn / (1.0 + beta * cs)).atan();
    let (s1, c1) = u1.sin_cos();
    let (s2u, c2u) = u2.sin_cos();
    let (du, ds, dc) = (u2 - u1, s2u - s1, c2u - c1);
    let dsc = s2u * c2u - s1 * c1;
    let ds3 = s2u.powi(3) - s1.powi(3);
    let s4 = s2 * s2;
    // ∫ (sin φ cos φ, cos²φ, sin²φ) / (1 - ε cos φ)³ dφ over the arc, and their ε-derivatives.
    let isc = (0.5 * (s2u * s2u - s1 * s1) - eps * dc) / s4;
    let icc = ((0.5 + eps * eps) * du + 0.5 * dsc + 2.0 * eps * ds) / (s4 * s);
    let iss = 0.5 * (du - dsc) / (s2 * s);
    let disc = -((c2u + eps).powi(3) - (c1 + eps).powi(3)) / (s4 * s2);
    let dicc = 3.0 * (ds - ds3 / 3.0 + 1.5 * eps * (du + dsc) + 3.0 * eps * eps * ds + eps.powi(3) * du)
        / (s4 * s2 * s);
    let diss = 3.0 * (ds3 / 3.0 + 0.5 * eps * (du - dsc)) / (s4 * s);
    let fr = k * (a * isc + b * icc); // along the line of centres
    let ft = k * (a * iss + b * isc); // across it, the way ψ turns
    let psi = 0.25 * k * (a * a * iss + 2.0 * a * b * isc + b * b * icc);
    // Derivatives in e and in ψ (the latter turning the frame too).
    let fr_e = k * (omega * isc + (a * disc + b * dicc) / c);
    let ft_e = k * (omega * iss + (a * diss + b * disc) / c);
    let am = a - omega * e;
    let fr_p = k * (am * icc - b * isc) - ft;
    let ft_p = k * (am * isc - b * iss) + fr;
    let (fx, fy) = (fr * cp - ft * sp, fr * sp + ft * cp);
    let (gx_e, gy_e) = (fr_e * cp - ft_e * sp, fr_e * sp + ft_e * cp);
    let (gx_p, gy_p) = ((fr_p * cp - ft_p * sp) / e, (fr_p * sp + ft_p * cp) / e);
    let (crr, crt, ctt) = (2.0 * k * icc, 2.0 * k * isc, 2.0 * k * iss);
    (
        fx,
        fy,
        gx_p * sp - gx_e * cp,
        -gx_e * sp - gx_p * cp,
        gy_p * sp - gy_e * cp,
        -gy_e * sp - gy_p * cp,
        crr * cp * cp - 2.0 * crt * cp * sp + ctt * sp * sp,
        (crr - ctt) * cp * sp + crt * (cp * cp - sp * sp),
        crr * sp * sp + 2.0 * crt * cp * sp + ctt * cp * cp,
        psi,
    )
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("ECCENTRICITY_MAX", ECCENTRICITY_MAX)?;
    m.add_function(wrap_pyfunction!(pi_film, m)?)?;
    Ok(())
}
