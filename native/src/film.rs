//! The oil films' whole force (`fluid_bearing.pi_film`, `full_film_damping`) and the rotor's
//! nonlinear film step (`rotor._Films`, `RotorSimulator._film_steps_reference`).

use std::f64::consts::PI;

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray2};

use crate::linalg::{Mat, dot, expm_foh, max_abs, solve};
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
    let dicc =
        3.0 * (ds - ds3 / 3.0 + 1.5 * eps * (du + dsc) + 3.0 * eps * eps * ds + eps.powi(3) * du) / (s4 * s2 * s);
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
    (
        crr * cp * cp + ctt * sp * sp,
        (crr - ctt) * cp * sp,
        crr * sp * sp + ctt * cp * cp,
    )
}

/// The rotor's nonlinear film steps at one held speed and step length: `_Films` (`take`, `remainder`,
/// `solve`) and `RotorSimulator._film_steps_reference`'s loop, which these mirror step for step.
#[pyclass(module = "vib_tutorial._native")]
pub struct FilmSteps {
    // At the held speed, from Python:
    a: Mat,    // the state matrix A(Ω)
    b: Mat,    // the input matrix
    minv: Mat, // (n_dof × m): M⁻¹'s columns at the journals
    cf: Mat,   // (m × m): the films' linearized damping, which A carries
    kf: Mat,   // (m × m): their stiffness
    p: Vec<f64>,
    h: f64,
    idx: Vec<usize>,             // the journals' x then x' in the state: dofs + rates
    films: Vec<(f64, f64, f64)>, // (k, c, the speed below which it's held), per film
    drift: f64,                  // FILM_DRIFT
    tol: f64,
    correction: f64,
    iterations: usize,
    // Taken with the damping bound (`take`):
    bound: Option<Taken>,
    takes: usize, // how many times it has been taken
}

/// The update with the films' damping bound C_b where it was taken.
struct Taken {
    taken: Vec<(f64, f64, f64)>, // each film's bound
    limit: f64,                  // a bound term drifting this far takes it afresh
    cb: Mat,
    phi: Mat,
    g0: Mat,
    g1: Mat,
    r0: Mat,
    r1: Mat,
    e: Mat,  // (2m × m): the journals' [x, x'] at the step's end, per remainder force
    es: Mat, // sym(E_v)
    ev: Mat,
    lin: Mat, // (m × 2m): [K_f C_b], the films the update carries
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

    fn bounds(&self, z: &[f64]) -> Vec<(f64, f64, f64)> {
        self.films
            .iter()
            .enumerate()
            .map(|(i, &(k, c, _))| full_film_damping(k, c, z[self.idx[2 * i]], z[self.idx[2 * i + 1]]))
            .collect()
    }

    /// Bound the damping where the journals are in z, and discretize over h (`_Films.take`).
    fn take(&mut self, z: &[f64]) {
        let (m, ns) = (self.m(), self.a.rows);
        let n = self.minv.rows;
        let taken = self.bounds(z);
        let size = taken
            .iter()
            .flat_map(|&(a, b, c)| [a.abs(), b.abs(), c.abs()])
            .fold(0.0, f64::max);
        let mut cb = Mat::zeros(m, m);
        for (i, &(cxx, cxy, cyy)) in taken.iter().enumerate() {
            let j = 2 * i;
            cb.set(j, j, cxx);
            cb.set(j, j + 1, cxy);
            cb.set(j + 1, j, cxy);
            cb.set(j + 1, j + 1, cyy);
        }
        // A with C_b in place of C_f at the journals: A[n:2n, rates] -= M⁻¹[:, d] (C_b - C_f).
        let extra = self.minv.matmul(&cb.combine(1.0, &self.cf, -1.0));
        let rates = &self.idx[m..];
        // foh_discretize, on A with C_b
        let mut a = self.a.clone();
        for i in 0..n {
            for (k, &r) in rates.iter().enumerate() {
                a.set(n + i, r, a.at(n + i, r) - extra.at(i, k));
            }
        }
        let (phi, g1_full, g1) = expm_foh(&a, &self.b, self.h);
        let g0 = g1_full.combine(1.0, &g1, -1.0);
        let all: Vec<usize> = (0..ns).collect();
        let dofs = &self.idx[..m];
        let r0 = g0.select(&all, dofs);
        let r1 = g1.select(&all, dofs);
        let e = r1.select(&self.idx, &(0..m).collect::<Vec<_>>());
        let ev = r1.select(rates, &(0..m).collect::<Vec<_>>());
        let mut es = Mat::zeros(m, m);
        for i in 0..m {
            for j in 0..m {
                es.set(i, j, 0.5 * (ev.at(i, j) + ev.at(j, i)));
            }
        }
        let mut lin = Mat::zeros(m, 2 * m);
        for i in 0..m {
            for j in 0..m {
                lin.set(i, j, self.kf.at(i, j));
                lin.set(i, m + j, cb.at(i, j));
            }
        }
        self.takes += 1;
        self.bound = Some(Taken {
            taken,
            limit: self.drift * size,
            cb,
            phi,
            g0,
            g1,
            r0,
            r1,
            e,
            es,
            ev,
            lin,
        });
    }

    /// True once a journal has moved far enough that the damping bound should be taken afresh.
    fn drifted(&self, z: &[f64]) -> bool {
        let t = self.bound.as_ref().unwrap();
        self.bounds(z).iter().zip(&t.taken).any(|(new, old)| {
            [new.0 - old.0, new.1 - old.1, new.2 - old.2]
                .iter()
                .any(|d| d.abs() > t.limit)
        })
    }

    /// (f, D, Ψ) with the journals at y = [x, x']: `_Films.forces`.
    fn forces(&self, y: &[f64], omega: f64, f: &mut [f64], d: &mut [f64]) -> f64 {
        let m = self.m();
        let w = 2 * m;
        let mut psi = 0.0;
        d.fill(0.0);
        for (i, &(k, c, slowest)) in self.films.iter().enumerate() {
            let j = 2 * i;
            let o = pi_film(k, c, y[j], y[j + 1], y[m + j], y[m + j + 1], omega.abs().max(slowest));
            f[j] = o.0;
            f[j + 1] = o.1;
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

    /// r at the state z: the films' force less its part in the exact update (`_Films.remainder`).
    fn remainder(&self, z: &[f64], omega: f64) -> Vec<f64> {
        let m = self.m();
        let t = self.bound.as_ref().unwrap();
        let y: Vec<f64> = self.idx.iter().map(|&i| z[i]).collect();
        let mut f = vec![0.0; m];
        let mut d = vec![0.0; 2 * m * m];
        self.forces(&y, omega, &mut f, &mut d);
        let mut r: Vec<f64> = f.iter().zip(&self.p).map(|(f, p)| f - p).collect();
        t.lin.mul_add(&y, &mut r);
        r
    }

    fn at(&self, yb: &[f64], r: &[f64], omega: f64) -> At {
        let m = self.m();
        let t = self.bound.as_ref().unwrap();
        let mut y = yb.to_vec();
        t.e.mul_add(r, &mut y);
        let mut f = vec![0.0; m];
        let mut d = vec![0.0; 2 * m * m];
        let psi = self.forces(&y, omega, &mut f, &mut d);
        let (x, v) = y.split_at(m);
        let mut es_r = vec![0.0; m];
        t.es.mul_add(r, &mut es_r);
        let mut cb_v = vec![0.0; m];
        t.cb.mul_add(v, &mut cb_v);
        let mut kf_x = vec![0.0; m];
        self.kf.mul_add(x, &mut kf_x);
        let load: Vec<f64> = self.p.iter().zip(&kf_x).map(|(p, k)| p - k).collect(); // p - K_f x
        let q = 0.5 * dot(r, &es_r) + psi - 0.5 * dot(v, &cb_v) + dot(&load, v);
        let mut lin_y = vec![0.0; m];
        t.lin.mul_add(&y, &mut lin_y);
        let res = (0..m).map(|i| r[i] - f[i] + self.p[i] - lin_y[i]).collect();
        At { res, d, q }
    }

    /// r at the step's end, where z = zb + R1 r: Newton from the guess r (`_Films.solve`).
    fn solve(&self, zb: &[f64], mut r: Vec<f64>, omega: f64) -> Vec<f64> {
        let m = self.m();
        let t = self.bound.as_ref().unwrap();
        let yb: Vec<f64> = self.idx.iter().map(|&i| zb[i]).collect();
        let mut now = self.at(&yb, &r, omega);
        for _ in 0..self.iterations {
            let size = max_abs(&now.res);
            if size <= self.tol {
                break;
            }
            // dr = -(I + (D - lin) E)⁻¹ R
            let mut jac = Mat::eye(m);
            for i in 0..m {
                for j in 0..m {
                    let s: f64 = (0..2 * m)
                        .map(|l| (now.d[i * 2 * m + l] - t.lin.at(i, l)) * t.e.at(l, j))
                        .sum();
                    jac.data[i * m + j] += s;
                }
            }
            let rhs = Mat {
                rows: m,
                cols: 1,
                data: now.res.iter().map(|x| -x).collect(),
            };
            let dr = solve(jac, rhs).data;
            if max_abs(&dr) <= self.correction {
                // what is left is far smaller still
                return r.iter().zip(&dr).map(|(a, b)| a + b).collect();
            }
            let mut ev_dr = vec![0.0; m];
            t.ev.mul_add(&dr, &mut ev_dr);
            let slope = dot(&now.res, &ev_dr);
            let mut step = 1.0;
            let next = loop {
                let rt: Vec<f64> = r.iter().zip(&dr).map(|(a, b)| a + step * b).collect();
                let trial = self.at(&yb, &rt, omega);
                if trial.q <= now.q + 1e-4 * step * slope || max_abs(&trial.res) <= 0.5 * size || step < 1e-3 {
                    break (rt, trial);
                }
                step *= 0.5;
            };
            (r, now) = next;
        }
        r
    }
}

#[pymethods]
impl FilmSteps {
    /// The film steps at a held speed: its A, B and the films' terms there (`_Films.hold`), over steps of h.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (a, b, minv, cf, kf, p, h, idx, films, drift, tol, correction, iterations))]
    fn new(
        a: PyReadonlyArray2<f64>,
        b: PyReadonlyArray2<f64>,
        minv: PyReadonlyArray2<f64>,
        cf: PyReadonlyArray2<f64>,
        kf: PyReadonlyArray2<f64>,
        p: PyReadonlyArray1<f64>,
        h: f64,
        idx: Vec<usize>,
        films: Vec<(f64, f64, f64)>,
        drift: f64,
        tol: f64,
        correction: f64,
        iterations: usize,
    ) -> PyResult<Self> {
        let s = FilmSteps {
            a: Mat::from_numpy(&a),
            b: Mat::from_numpy(&b),
            minv: Mat::from_numpy(&minv),
            cf: Mat::from_numpy(&cf),
            kf: Mat::from_numpy(&kf),
            p: p.as_array().to_vec(),
            h,
            idx,
            films,
            drift,
            tol,
            correction,
            iterations,
            bound: None,
            takes: 0,
        };
        let (ns, m, n) = (s.a.rows, s.p.len(), s.minv.rows);
        s.a.check("A", ns, ns)?;
        s.b.check("B", ns, s.b.cols)?;
        s.minv.check("minv", n, m)?;
        s.cf.check("Cf", m, m)?;
        s.kf.check("Kf", m, m)?;
        if 2 * n > ns
            || s.idx.len() != 2 * m
            || s.idx.iter().any(|&i| i >= ns)
            || 2 * s.films.len() != m
            || s.idx[m..].iter().any(|&i| i < n || i >= 2 * n)
        {
            return Err(PyValueError::new_err(
                "the journals' indices and films don't match the matrices",
            ));
        }
        Ok(s)
    }

    /// `_film_steps_reference`'s loop: from the state z, steps to the speeds w1 (Ω was omega0 at z) under
    /// the loads u0 (before the first step) and u (n × n_dof, after each), writing q into `out`'s rows.
    /// Takes the bound where it first steps and afresh as the journals drift; returns the last state.
    #[allow(clippy::too_many_arguments)]
    fn steps(
        &mut self,
        z: Vec<f64>,
        u0: PyReadonlyArray1<f64>,
        u: PyReadonlyArray2<f64>,
        w1: PyReadonlyArray1<f64>,
        omega0: f64,
        mut out: PyReadwriteArray2<f64>,
    ) -> PyResult<Vec<f64>> {
        let (u0, u, w1) = (u0.as_array(), u.as_array(), w1.as_array());
        let mut out = out.as_array_mut();
        let ns = self.a.rows;
        let (n, nu) = u.dim();
        if z.len() != ns || u0.len() != nu || nu != self.b.cols || w1.len() != n || out.nrows() != n || out.ncols() > ns
        {
            return Err(PyValueError::new_err(
                "the state, loads or output don't match the matrices",
            ));
        }
        let mut z = z;
        if self.bound.is_none() {
            self.take(&z);
        }
        let m = self.m();
        let mut r = self.remainder(&z, omega0);
        let (mut last, mut older) = (r.clone(), r.clone());
        let n_out = out.ncols();
        let mut zb = vec![0.0; ns];
        let mut before = u0.to_vec();
        for j in 0..n {
            if self.drifted(&z) {
                self.take(&z);
                r = self.remainder(&z, if j > 0 { w1[j - 1] } else { omega0 });
                (last, older) = (r.clone(), r.clone());
            }
            let t = self.bound.as_ref().unwrap();
            let after = u.row(j).to_vec();
            zb.fill(0.0);
            t.phi.mul_add(&z, &mut zb);
            t.g0.mul_add(&before, &mut zb);
            t.g1.mul_add(&after, &mut zb);
            t.r0.mul_add(&r, &mut zb);
            // r on the parabola through the last three steps
            let guess: Vec<f64> = (0..m).map(|i| 3.0 * (r[i] - last[i]) + older[i]).collect();
            let new = self.solve(&zb, guess, w1[j]);
            (older, last, r) = (last, r, new);
            let t = self.bound.as_ref().unwrap();
            z.copy_from_slice(&zb);
            t.r1.mul_add(&r, &mut z);
            for (i, v) in z.iter().take(n_out).enumerate() {
                out[[j, i]] = *v;
            }
            before = after;
        }
        Ok(z)
    }

    /// How many times the damping bound has been taken (once where it first steps, then as the journals drift).
    #[getter]
    fn takes(&self) -> usize {
        self.takes
    }

    /// Each film's damping bound (c_xx, c_xy, c_yy) as last taken, or None before the first step.
    #[getter]
    fn taken(&self) -> Option<Vec<(f64, f64, f64)>> {
        self.bound.as_ref().map(|t| t.taken.clone())
    }
}
