//! The disc rubbing on a stator ring (`rotor.stator_force`, `_rub_solve`) and the rotor's rub step
//! (`RotorSimulator._rub_steps_reference`).

use numpy::ndarray::Array2;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray2};

use crate::linalg::{Mat, max_abs};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// The ring: (clearance, stiffness, damping, friction μ, centre x, centre y), as `rotor.Stator` gives them.
type Ring = (f64, f64, f64, f64, f64, f64);

/// (f_x, f_y, N, J): `rotor.stator_force`, line for line (its docstring has the derivation).
fn stator_force(ring: &Ring, slip0: f64, radius: f64, y: &[f64; 4], omega: f64) -> (f64, f64, f64, [f64; 8]) {
    let &(clearance, k, c, mu, cx, cy) = ring;
    let (dx, dy) = (y[0] - cx, y[1] - cy);
    let rho = dx.hypot(dy);
    let pen = rho - clearance;
    if pen <= 0.0 || rho == 0.0 {
        return (0.0, 0.0, 0.0, [0.0; 8]);
    }
    let (nx, ny) = (dx / rho, dy / rho); // out from the ring's centre
    let (tx, ty) = (-ny, nx); // along the ring, in the direction of spin
    let (vn, vt) = (nx * y[2] + ny * y[3], tx * y[2] + ty * y[3]);
    let normal = k * pen + c * vn;
    if normal <= 0.0 {
        // pulling away faster than the ring springs back: no contact
        return (0.0, 0.0, 0.0, [0.0; 8]);
    }
    let s = radius * omega + vt;
    let root = (s * s + slip0 * slip0).sqrt();
    let (sat, dsat) = (s / root, slip0 * slip0 / (root * root * root));
    let (gx, gy) = (nx + mu * sat * tx, ny + mu * sat * ty); // f = -N g
    let dn = [k * nx + c * vt / rho * tx, k * ny + c * vt / rho * ty, c * nx, c * ny];
    let ds = [-vn / rho * tx, -vn / rho * ty, tx, ty];
    let mut jac = [0.0; 8];
    for (row, (g, n, t)) in [(gx, nx, tx), (gy, ny, ty)].into_iter().enumerate() {
        let turn = (t - mu * sat * n) / rho;
        let dg = [
            turn * tx + mu * t * dsat * ds[0],
            turn * ty + mu * t * dsat * ds[1],
            mu * t * dsat * ds[2],
            mu * t * dsat * ds[3],
        ];
        for i in 0..4 {
            jac[4 * row + i] = -(g * dn[i] + normal * dg[i]);
        }
    }
    (-normal * gx, -normal * gy, normal, jac)
}

#[pyclass]
pub struct RubSteps {
    phi: Mat,
    r0: Mat,          // (n_states × 2): Γ0's columns at the disc
    r1: Mat,          // and Γ1's
    rows: [usize; 4], // the disc's x, y, x', y' in the state
    e: [f64; 8],      // (4 × 2): R1's rows there
    ring: Ring,
    slip: f64,
    radius: f64,
    tol: f64,
    iterations: usize,
}

impl RubSteps {
    /// (r, N): `_rub_solve`, Newton from the guess r, each step cut back until it shrinks the residual.
    fn solve(&self, yb: &[f64; 4], mut r: [f64; 2], omega: f64) -> ([f64; 2], f64) {
        let at = |r: &[f64; 2]| {
            let mut y = *yb;
            for (i, v) in y.iter_mut().enumerate() {
                *v += self.e[2 * i] * r[0] + self.e[2 * i + 1] * r[1];
            }
            let (fx, fy, normal, jac) = stator_force(&self.ring, self.slip, self.radius, &y, omega);
            ([r[0] - fx, r[1] - fy], jac, normal)
        };
        let (mut res, mut jac, mut normal) = at(&r);
        for _ in 0..self.iterations {
            let size = max_abs(&res);
            if size <= self.tol {
                break;
            }
            // (I - J E) dr = -R
            let mut m = [1.0, 0.0, 0.0, 1.0];
            for i in 0..2 {
                for j in 0..2 {
                    m[2 * i + j] -= (0..4).map(|k| jac[4 * i + k] * self.e[2 * k + j]).sum::<f64>();
                }
            }
            let det = m[0] * m[3] - m[1] * m[2];
            let dr = [
                (-res[0] * m[3] + res[1] * m[1]) / det,
                (-res[1] * m[0] + res[0] * m[2]) / det,
            ];
            let mut t = 1.0;
            loop {
                let trial = [r[0] + t * dr[0], r[1] + t * dr[1]];
                let (rt, jt, nt) = at(&trial);
                if max_abs(&rt) < size || t < 1e-3 {
                    (r, res, jac, normal) = (trial, rt, jt, nt);
                    break;
                }
                t *= 0.5;
            }
        }
        (r, normal)
    }
}

#[pymethods]
impl RubSteps {
    /// The rub steps over the FOH update Φ, Γ0, Γ1 at a held speed, the disc at `xd`, `yd` of q (n_dof long).
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (phi, g0, g1, n_dof, xd, yd, ring, slip, radius, tol, iterations))]
    fn new(
        phi: PyReadonlyArray2<f64>,
        g0: PyReadonlyArray2<f64>,
        g1: PyReadonlyArray2<f64>,
        n_dof: usize,
        xd: usize,
        yd: usize,
        ring: Ring,
        slip: f64,
        radius: f64,
        tol: f64,
        iterations: usize,
    ) -> PyResult<Self> {
        let phi = Mat::from_numpy(&phi);
        let (g0, g1) = (Mat::from_numpy(&g0), Mat::from_numpy(&g1));
        let ns = phi.rows;
        phi.check("Phi", ns, ns)?;
        g0.check("G0", ns, n_dof)?;
        g1.check("G1", ns, n_dof)?;
        if 2 * n_dof > ns || xd >= n_dof || yd >= n_dof {
            return Err(PyValueError::new_err("the disc's DOFs don't match the matrices"));
        }
        let all: Vec<usize> = (0..ns).collect();
        let (r0, r1) = (g0.select(&all, &[xd, yd]), g1.select(&all, &[xd, yd]));
        let rows = [xd, yd, n_dof + xd, n_dof + yd];
        let mut e = [0.0; 8];
        for (i, &row) in rows.iter().enumerate() {
            e[2 * i] = r1.at(row, 0);
            e[2 * i + 1] = r1.at(row, 1);
        }
        Ok(RubSteps {
            phi,
            r0,
            r1,
            rows,
            e,
            ring,
            slip,
            radius,
            tol,
            iterations,
        })
    }

    /// `_rub_steps_reference`'s loop: from the state z (Ω was omega0 there), steps to the speeds w1 under
    /// the loads' part of the update, `drive` (n_states × n), writing q into `out`'s rows and the ring's
    /// force, push and heat into `rubs`' (n × 4). Returns the last state. Steps without the GIL.
    #[allow(clippy::too_many_arguments)]
    fn steps(
        &self,
        py: Python<'_>,
        z: Vec<f64>,
        drive: PyReadonlyArray2<f64>,
        w1: PyReadonlyArray1<f64>,
        omega0: f64,
        mut out: PyReadwriteArray2<f64>,
        mut rubs: PyReadwriteArray2<f64>,
    ) -> PyResult<Vec<f64>> {
        let (drive, w1) = (drive.as_array().to_owned(), w1.as_array().to_vec());
        let mut out = out.as_array_mut();
        let mut rubs = rubs.as_array_mut();
        let ns = self.phi.rows;
        let n = w1.len();
        if z.len() != ns || drive.dim() != (ns, n) || out.nrows() != n || out.ncols() > ns || rubs.dim() != (n, 4) {
            return Err(PyValueError::new_err(
                "the state, loads or outputs don't match the matrices",
            ));
        }
        let n_out = out.ncols();
        let (z, qs, rs) = py.detach(|| {
            let mut qs = Array2::zeros((n, n_out));
            let mut rs = Array2::zeros((n, 4));
            let mut z = z;
            let pick = |z: &[f64]| [z[self.rows[0]], z[self.rows[1]], z[self.rows[2]], z[self.rows[3]]];
            let (fx, fy, _, _) = stator_force(&self.ring, self.slip, self.radius, &pick(&z), omega0);
            let mut r = [fx, fy];
            let mut zb = vec![0.0; ns];
            let (mu, cx, cy) = (self.ring.3, self.ring.4, self.ring.5);
            for j in 0..n {
                for (i, v) in zb.iter_mut().enumerate() {
                    *v = drive[[i, j]];
                }
                self.phi.mul_add(&z, &mut zb);
                self.r0.mul_add(&r, &mut zb);
                let (new, normal) = self.solve(&pick(&zb), r, w1[j]);
                r = new;
                z.copy_from_slice(&zb);
                self.r1.mul_add(&r, &mut z);
                for (i, v) in z.iter().take(n_out).enumerate() {
                    qs[[j, i]] = *v;
                }
                let mut heat = 0.0;
                if normal != 0.0 {
                    // the drag's work on the slip, μ N sat(s) s
                    let [x, y, vx, vy] = pick(&z);
                    let rho = (x - cx).hypot(y - cy);
                    let slip = self.radius * w1[j] + ((x - cx) * vy - (y - cy) * vx) / rho;
                    heat = mu * normal * slip * slip / (slip * slip + self.slip * self.slip).sqrt();
                }
                for (i, v) in [r[0], r[1], normal, heat].into_iter().enumerate() {
                    rs[[j, i]] = v;
                }
            }
            (z, qs, rs)
        });
        out.assign(&qs);
        rubs.assign(&rs);
        Ok(z)
    }
}
