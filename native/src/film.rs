//! The oil films' whole force (`fluid_bearing.pi_film`, `full_film_damping`) and the rotor's
//! nonlinear film step (`rotor._Films`, `RotorSimulator._film_steps_reference`).

use std::f64::consts::PI;

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// The whole film's force is worked out no nearer the wall than this
/// (`fluid_bearing.ECCENTRICITY_MAX`; a test checks they agree).
pub const ECCENTRICITY_MAX: f64 = 1.0 - 1e-6;

/// (f_x, f_y, k_xx, k_xy, k_yx, k_yy, c_xx, c_xy, c_yy, Ψ), a tuple in Python as `pi_film` returns.
pub type Film = (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64);

/// The π-film's whole force on the journal, its derivatives and its dissipation:
/// `fluid_bearing.pi_film`, line for line (its docstring has the derivation).
#[pyfunction]
pub fn pi_film(k: f64, c: f64, x: f64, y: f64, vx: f64, vy: f64, omega: f64) -> Film {
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

/// (c_xx, c_xy, c_yy): the whole uncavitated film's damping at (x, y): `fluid_bearing.full_film_damping`.
#[pyfunction]
pub fn full_film_damping(k: f64, c: f64, x: f64, y: f64) -> (f64, f64, f64) {
    let e = x.hypot(y);
    let (cp, sp) = if e > 0.0 { (x / e, y / e) } else { (1.0, 0.0) };
    let eps = (e / c).min(ECCENTRICITY_MAX);
    let s2 = 1.0 - eps * eps;
    let crr = 2.0 * PI * k * (1.0 + 2.0 * eps * eps) / s2.powf(2.5);
    let ctt = 2.0 * PI * k / s2.powf(1.5);
    (crr * cp * cp + ctt * sp * sp, (crr - ctt) * cp * sp, crr * sp * sp + ctt * cp * cp)
}

/// A dense row-major matrix, copied from NumPy.
struct Mat {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Mat {
    fn from(a: &PyReadonlyArray2<f64>) -> Self {
        let a = a.as_array();
        let (rows, cols) = a.dim();
        Mat { rows, cols, data: a.iter().copied().collect() }
    }

    fn at(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.cols + j]
    }

    /// out += self @ x
    fn mul_add(&self, x: &[f64], out: &mut [f64]) {
        for (i, o) in out.iter_mut().enumerate().take(self.rows) {
            let row = &self.data[i * self.cols..(i + 1) * self.cols];
            *o += row.iter().zip(x).map(|(a, b)| a * b).sum::<f64>();
        }
    }

    fn check(&self, name: &str, rows: usize, cols: usize) -> PyResult<()> {
        if self.rows == rows && self.cols == cols {
            Ok(())
        } else {
            Err(PyValueError::new_err(format!("{name} is {}x{}, not {rows}x{cols}", self.rows, self.cols)))
        }
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn max_abs(a: &[f64]) -> f64 {
    a.iter().fold(0.0, |m, x| m.max(x.abs()))
}

/// x with a x = b, a (n × n, row-major) by Gaussian elimination with partial pivoting; a and b are spent.
fn solve(a: &mut [f64], b: &mut [f64], n: usize) -> Vec<f64> {
    for col in 0..n {
        let pivot = (col..n).max_by(|&i, &j| a[i * n + col].abs().total_cmp(&a[j * n + col].abs())).unwrap();
        if pivot != col {
            for j in 0..n {
                a.swap(col * n + j, pivot * n + j);
            }
            b.swap(col, pivot);
        }
        let d = a[col * n + col];
        for i in col + 1..n {
            let f = a[i * n + col] / d;
            if f != 0.0 {
                for j in col..n {
                    a[i * n + j] -= f * a[col * n + j];
                }
                b[i] -= f * b[col];
            }
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let s: f64 = (i + 1..n).map(|j| a[i * n + j] * x[j]).sum();
        x[i] = (b[i] - s) / a[i * n + i];
    }
    x
}

/// One damping bound's discretization of the rotor with nonlinear films, and the steps on it:
/// what `_Films.take` sets up, stepped as `_film_steps_reference` does until a journal drifts.
#[pyclass(module = "vib_tutorial._native")]
pub struct FilmSteps {
    phi: Mat,
    g0: Mat,
    g1: Mat,
    r0: Mat,
    r1: Mat,
    e: Mat,   // (2m × m): the journals' [x, x'] at the step's end, per remainder force
    es: Mat,  // (m × m): sym(E_v)
    ev: Mat,  // (m × m)
    lin: Mat, // (m × 2m): [K_f C_b], the films the update carries
    kf: Mat,
    cb: Mat,
    p: Vec<f64>,
    idx: Vec<usize>,              // the journals' x then x' in the state: dofs + rates
    films: Vec<(f64, f64, f64)>,  // (k, c, the speed below which it's held), per film
    taken: Vec<(f64, f64, f64)>,  // each film's damping bound where it was taken
    limit: f64,                   // a bound term drifting this far takes it afresh
    tol: f64,
    correction: f64,
    iterations: usize,
}

/// What `at(r)` gives in `_Films.solve`: the residual, the films' [K C] and Q.
struct At {
    res: Vec<f64>,
    d: Vec<f64>, // (m × 2m), row-major
    q: f64,
}

impl FilmSteps {
    fn m(&self) -> usize {
        self.p.len()
    }

    /// (f, D, Ψ) with the journals at y = [x, x']: `_Films.forces`.
    fn forces(&self, y: &[f64], omega: f64, f: &mut [f64], d: &mut [f64]) -> f64 {
        let m = self.m();
        let mut psi = 0.0;
        d.fill(0.0);
        for (i, &(k, c, slowest)) in self.films.iter().enumerate() {
            let j = 2 * i;
            let o = pi_film(k, c, y[j], y[j + 1], y[m + j], y[m + j + 1], omega.abs().max(slowest));
            f[j] = o.0;
            f[j + 1] = o.1;
            let w = 2 * m;
            d[j * w + j] = o.2;
            d[j * w + j + 1] = o.3;
            d[(j + 1) * w + j] = o.4;
            d[(j + 1) * w + j + 1] = o.5;
            d[j * w + m + j] = o.6;
            d[j * w + m + j + 1] = o.7;
            d[(j + 1) * w + m + j] = o.7;
            d[(j + 1) * w + m + j + 1] = o.8;
            psi += o.9;
        }
        psi
    }

    fn at(&self, yb: &[f64], r: &[f64], omega: f64) -> At {
        let m = self.m();
        let mut y = yb.to_vec();
        self.e.mul_add(r, &mut y);
        let mut f = vec![0.0; m];
        let mut d = vec![0.0; 2 * m * m];
        let psi = self.forces(&y, omega, &mut f, &mut d);
        let (x, v) = y.split_at(m);
        let mut es_r = vec![0.0; m];
        self.es.mul_add(r, &mut es_r);
        let mut cb_v = vec![0.0; m];
        self.cb.mul_add(v, &mut cb_v);
        let mut load = self.p.clone(); // p - K_f x
        let mut kf_x = vec![0.0; m];
        self.kf.mul_add(x, &mut kf_x);
        load.iter_mut().zip(&kf_x).for_each(|(l, k)| *l -= k);
        let q = 0.5 * dot(r, &es_r) + psi - 0.5 * dot(v, &cb_v) + dot(&load, v);
        let mut lin_y = vec![0.0; m];
        self.lin.mul_add(&y, &mut lin_y);
        let res = (0..m).map(|i| r[i] - f[i] + self.p[i] - lin_y[i]).collect();
        At { res, d, q }
    }

    /// r at the step's end, where z = zb + R1 r: Newton from the guess r (`_Films.solve`).
    fn solve(&self, zb: &[f64], mut r: Vec<f64>, omega: f64) -> Vec<f64> {
        let m = self.m();
        let yb: Vec<f64> = self.idx.iter().map(|&i| zb[i]).collect();
        let mut now = self.at(&yb, &r, omega);
        for _ in 0..self.iterations {
            let size = max_abs(&now.res);
            if size <= self.tol {
                break;
            }
            // J = I + (D - lin) E
            let mut jac = vec![0.0; m * m];
            for i in 0..m {
                for j in 0..m {
                    let s: f64 = (0..2 * m).map(|l| (now.d[i * 2 * m + l] - self.lin.at(i, l)) * self.e.at(l, j)).sum();
                    jac[i * m + j] = s + if i == j { 1.0 } else { 0.0 };
                }
            }
            let mut rhs: Vec<f64> = now.res.iter().map(|x| -x).collect();
            let dr = solve(&mut jac, &mut rhs, m);
            if max_abs(&dr) <= self.correction {
                // what is left is far smaller still
                return r.iter().zip(&dr).map(|(a, b)| a + b).collect();
            }
            let mut ev_dr = vec![0.0; m];
            self.ev.mul_add(&dr, &mut ev_dr);
            let slope = dot(&now.res, &ev_dr);
            let mut t = 1.0;
            let next = loop {
                let rt: Vec<f64> = r.iter().zip(&dr).map(|(a, b)| a + t * b).collect();
                let trial = self.at(&yb, &rt, omega);
                if trial.q <= now.q + 1e-4 * t * slope || max_abs(&trial.res) <= 0.5 * size || t < 1e-3 {
                    break (rt, trial);
                }
                t *= 0.5;
            };
            (r, now) = next;
        }
        r
    }

    /// True once a journal has moved far enough that the damping bound should be taken afresh.
    fn drifted(&self, z: &[f64]) -> bool {
        self.films.iter().zip(&self.taken).enumerate().any(|(i, (&(k, c, _), old))| {
            let new = full_film_damping(k, c, z[self.idx[2 * i]], z[self.idx[2 * i + 1]]);
            [new.0 - old.0, new.1 - old.1, new.2 - old.2].iter().any(|d| d.abs() > self.limit)
        })
    }
}

#[pymethods]
impl FilmSteps {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (phi, g0, g1, r0, r1, e, es, ev, lin, kf, cb, p, idx, films, taken, limit, tol, correction, iterations))]
    fn new(
        phi: PyReadonlyArray2<f64>,
        g0: PyReadonlyArray2<f64>,
        g1: PyReadonlyArray2<f64>,
        r0: PyReadonlyArray2<f64>,
        r1: PyReadonlyArray2<f64>,
        e: PyReadonlyArray2<f64>,
        es: PyReadonlyArray2<f64>,
        ev: PyReadonlyArray2<f64>,
        lin: PyReadonlyArray2<f64>,
        kf: PyReadonlyArray2<f64>,
        cb: PyReadonlyArray2<f64>,
        p: PyReadonlyArray1<f64>,
        idx: Vec<usize>,
        films: Vec<(f64, f64, f64)>,
        taken: Vec<(f64, f64, f64)>,
        limit: f64,
        tol: f64,
        correction: f64,
        iterations: usize,
    ) -> PyResult<Self> {
        let s = FilmSteps {
            phi: Mat::from(&phi),
            g0: Mat::from(&g0),
            g1: Mat::from(&g1),
            r0: Mat::from(&r0),
            r1: Mat::from(&r1),
            e: Mat::from(&e),
            es: Mat::from(&es),
            ev: Mat::from(&ev),
            lin: Mat::from(&lin),
            kf: Mat::from(&kf),
            cb: Mat::from(&cb),
            p: p.as_array().to_vec(),
            idx,
            films,
            taken,
            limit,
            tol,
            correction,
            iterations,
        };
        let (ns, nu, m) = (s.phi.rows, s.g0.cols, s.p.len());
        s.phi.check("Phi", ns, ns)?;
        s.g1.check("G1", ns, nu)?;
        s.g0.check("G0", ns, nu)?;
        s.r0.check("R0", ns, m)?;
        s.r1.check("R1", ns, m)?;
        s.e.check("E", 2 * m, m)?;
        s.es.check("Es", m, m)?;
        s.ev.check("Ev", m, m)?;
        s.lin.check("lin", m, 2 * m)?;
        s.kf.check("Kf", m, m)?;
        s.cb.check("Cb", m, m)?;
        if s.idx.len() != 2 * m || s.idx.iter().any(|&i| i >= ns) || 2 * s.films.len() != m || s.taken.len() != s.films.len() {
            return Err(PyValueError::new_err("the journals' indices, films and bounds don't match the matrices"));
        }
        Ok(s)
    }

    /// Steps `start`, `start + 1`, … of `_film_steps_reference`'s loop, writing q into `out`'s rows.
    ///
    /// z is the state, r, last and older the remainder at it and the two steps before; u0 the load
    /// before step 0 and u (n × n_dof) after each step, w1 the speed after each, omega0 before step 0.
    /// Returns (j, z, r, last, older): j = n when every step is done, or the step at which a journal
    /// drifted, before it is taken; take the bound afresh there and call again from j.
    #[allow(clippy::too_many_arguments)]
    fn steps(
        &self,
        z: Vec<f64>,
        r: Vec<f64>,
        last: Vec<f64>,
        older: Vec<f64>,
        u0: PyReadonlyArray1<f64>,
        u: PyReadonlyArray2<f64>,
        w1: PyReadonlyArray1<f64>,
        start: usize,
        mut out: PyReadwriteArray2<f64>,
    ) -> PyResult<(usize, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
        let (u0, u, w1) = (u0.as_array(), u.as_array(), w1.as_array());
        let mut out = out.as_array_mut();
        let ns = self.phi.rows;
        let (n, nu) = u.dim();
        let m = self.m();
        if z.len() != ns || u0.len() != nu || nu != self.g0.cols || w1.len() != n || out.nrows() < n || out.ncols() > ns
            || [&r, &last, &older].iter().any(|v| v.len() != m)
        {
            return Err(PyValueError::new_err("the state, loads or output don't match the matrices"));
        }
        let (mut z, mut r, mut last, mut older) = (z, r, last, older);
        let n_out = out.ncols();
        let mut zb = vec![0.0; ns];
        for j in start..n {
            if j > start && self.drifted(&z) {
                return Ok((j, z, r, last, older));
            }
            zb.fill(0.0);
            self.phi.mul_add(&z, &mut zb);
            let before: Vec<f64> = if j > 0 { u.row(j - 1).to_vec() } else { u0.to_vec() };
            self.g0.mul_add(&before, &mut zb);
            self.g1.mul_add(&u.row(j).to_vec(), &mut zb);
            self.r0.mul_add(&r, &mut zb);
            // r on the parabola through the last three steps
            let guess: Vec<f64> = (0..m).map(|i| 3.0 * (r[i] - last[i]) + older[i]).collect();
            let new = self.solve(&zb, guess, w1[j]);
            (older, last, r) = (last, r, new);
            z.copy_from_slice(&zb);
            self.r1.mul_add(&r, &mut z);
            for (i, v) in z.iter().take(n_out).enumerate() {
                out[[j, i]] = *v;
            }
        }
        Ok((n, z, r, last, older))
    }
}
