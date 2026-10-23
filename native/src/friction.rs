//! The chain's Coulomb-friction stepper: `Simulator._friction_step` and its loops (`_advance_friction`,
//! `drive`), which these mirror, with `_slip`, `_first_event` and `first_root`.
//!
//! One difference in how the energy ledger is found, not in what it is: the Python integrates each
//! (sub-)step's damper loss and work as quadratic forms (Van Loan, two exponentials of twice the size).
//! Here the step's matrix carries one more state, q = ∫ x_k dt (x_k the mass the input works on), from
//! which the input's work follows exactly (u is linear over the step); and the dampers' loss from the
//! exact energy balance, work - friction loss - ΔE. So each event's sub-step costs one exponential.

use std::collections::HashMap;

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::linalg::{Mat, expm_balanced};

/// NumPy's sign: 0 for ±0.
fn sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// `model.held`: static friction holds a mass at rest against `pull`, to within rounding.
fn held(pull: f64, static_: f64) -> bool {
    pull.abs() <= static_ * (1.0 + 1e-9)
}

/// p(τ) = c[0] τ³ + c[1] τ² + c[2] τ + c[3]
fn cubic(c: &[f64; 4], t: f64) -> f64 {
    ((c[0] * t + c[1]) * t + c[2]) * t + c[3]
}

/// The smallest real root of the cubic in (1e-12, 1 + 1e-9], as `simulator.first_root` takes it from
/// `np.roots`: a root where p changes sign (bisected to rounding), or one it touches (a double root,
/// which NumPy reports as real while its imaginary part is under 1e-9).
fn smallest_root(c: [f64; 4]) -> Option<f64> {
    let (lo, hi) = (1e-12, 1.0 + 1e-9);
    // np.roots drops leading zeros: the degree is what's left.
    let lead = c.iter().position(|&x| x != 0.0)?;
    if lead == 3 {
        return None; // a constant
    }
    // Where p' = 0 inside the interval: p is monotone between these.
    let (a, b, cc) = (3.0 * c[0], 2.0 * c[1], c[2]);
    let mut crit: Vec<f64> = Vec::new();
    if a != 0.0 {
        let disc = b * b - 4.0 * a * cc;
        if disc >= 0.0 {
            let q = -0.5 * (b + sign_or_one(b) * disc.sqrt());
            crit.push(q / a);
            if q != 0.0 {
                crit.push(cc / q);
            }
        }
    } else if b != 0.0 {
        crit.push(-cc / b);
    }
    crit.retain(|&t| t > lo && t < hi);
    crit.sort_by(f64::total_cmp);
    let mut edges = vec![lo];
    edges.extend(&crit);
    edges.push(hi);
    for w in edges.windows(2) {
        let (x0, x1) = (w[0], w[1]);
        let (p0, p1) = (cubic(&c, x0), cubic(&c, x1));
        if x0 > lo {
            // x0 is a critical point: a touch counts as a root if it's within NumPy's 1e-9 of real
            let curv = (6.0 * c[0] * x0 + 2.0 * c[1]).abs();
            if p0 == 0.0 || p0.abs() <= 0.5 * curv * 1e-18 {
                return Some(x0);
            }
        }
        if p1 == 0.0 {
            return Some(x1);
        }
        if (p0 < 0.0) != (p1 < 0.0) && p0 != 0.0 {
            let (mut a, mut b, neg_a) = (x0, x1, p0 < 0.0);
            for _ in 0..200 {
                let mid = 0.5 * (a + b);
                if mid <= a || mid >= b {
                    break;
                }
                let pm = cubic(&c, mid);
                if pm == 0.0 {
                    return Some(mid);
                }
                if (pm < 0.0) == neg_a {
                    a = mid;
                } else {
                    b = mid;
                }
            }
            return Some(if cubic(&c, a).abs() <= cubic(&c, b).abs() { a } else { b });
        }
    }
    None
}

fn sign_or_one(x: f64) -> f64 {
    if x < 0.0 { -1.0 } else { 1.0 }
}

/// `simulator.first_root`: the first time in (0, h] where the cubic with g(0) = g0, g'(0) = d0,
/// g(h) = g1, g'(h) = d1 is zero.
fn first_root(g0: f64, d0: f64, g1: f64, d1: f64, h: f64) -> f64 {
    let (a, b) = (h * d0, h * d1);
    let c = [2.0 * g0 + a - 2.0 * g1 + b, -3.0 * g0 - 2.0 * a + 3.0 * g1 - b, a, g0];
    match smallest_root(c) {
        Some(t) => h * t.min(1.0),
        None => h * if g0 > 0.0 { g0 / (g0 - g1) } else { 1.0 },
    }
}

/// The chain with friction, for one input (a force on a mass, or the ground's motion).
#[pyclass(module = "vib_tutorial._native")]
pub struct FrictionSteps {
    n: usize,
    aw: Mat, // (3n+3)²: w = [x, v, u, du, e, q] with no mass stuck; q' = x_k
    p: Mat,  // (n × 2n+2): the springs', dampers' and input's pull on each mass
    masses: Vec<f64>,
    stiffness: Vec<f64>,
    friction: Vec<f64>,
    static_: Vec<f64>,
    base: bool,
    k: usize,     // the mass the input works on (q integrates its x)
    c1: f64,      // damper 1 (the ground's work)
    damped: bool, // any damper at all: without, nothing is dissipated (exactly, as the Python's Q_d = 0 gives)
    tries: usize,
    systems: HashMap<u64, Mat>,      // Aw per stuck pattern
    steps: HashMap<(u64, u64), Mat>, // e^{Aw h} per stuck pattern and h
}

/// One (sub-)step's (dissipated, work, friction loss), or zeros without the ledger.
type Losses = [f64; 3];

impl FrictionSteps {
    fn m(&self) -> usize {
        3 * self.n + 2
    }

    fn mask(stuck: &[bool]) -> u64 {
        stuck
            .iter()
            .enumerate()
            .fold(0, |m, (i, &s)| if s { m | (1 << i) } else { m })
    }

    /// Aw with the stuck masses held still (their acceleration rows zeroed): `_friction_system`.
    fn system(&mut self, stuck: &[bool]) -> u64 {
        let key = Self::mask(stuck);
        if !self.systems.contains_key(&key) {
            if self.systems.len() > 4096 {
                self.systems.clear();
                self.steps.clear();
            }
            let mut aw = self.aw.clone();
            let cols = aw.cols;
            for (i, _) in stuck.iter().enumerate().filter(|(_, s)| **s) {
                aw.data[(self.n + i) * cols..(self.n + i + 1) * cols].fill(0.0);
            }
            self.systems.insert(key, aw);
        }
        key
    }

    /// e^{Aw s}: cached for whole steps (s = h), as `_friction_step_maps` caches them.
    fn transition(&mut self, key: u64, s: f64, h: f64) -> Mat {
        if s == h
            && let Some(phi) = self.steps.get(&(key, h.to_bits()))
        {
            return phi.clone();
        }
        let aw = &self.systems[&key];
        let phi = expm_balanced(&aw.combine(s, aw, 0.0));
        if s == h {
            self.steps.insert((key, h.to_bits()), phi.clone());
        }
        phi
    }

    fn pull(&self, i: usize, w: &[f64]) -> f64 {
        self.p.row(i).iter().zip(w).map(|(a, b)| a * b).sum()
    }

    /// (stuck, e): which masses stick, and the friction force on each sliding one (`_slip`).
    fn slip(&self, w: &[f64], forced: Option<(usize, f64)>) -> (Vec<bool>, Vec<f64>) {
        let n = self.n;
        let v = &w[n..2 * n];
        let mut way: Vec<f64> = v.iter().map(|&x| sign(x)).collect();
        let rest: Vec<bool> = (0..n).map(|i| v[i] == 0.0 && self.friction[i] > 0.0).collect();
        let mut stuck = vec![false; n];
        if forced.is_some() || rest.iter().any(|&r| r) {
            for i in 0..n {
                if rest[i] {
                    let pull = self.pull(i, w);
                    way[i] = sign(pull);
                    stuck[i] = held(pull, self.static_[i]);
                }
            }
        }
        if let Some((i, d)) = forced {
            stuck[i] = false;
            way[i] = d;
        }
        let e = (0..n)
            .map(|i| if stuck[i] { 0.0 } else { -self.friction[i] * way[i] })
            .collect();
        (stuck, e)
    }

    /// (time, mass, way) of the first stop (way 0) or break-free (way ±1) within the step w0 -> w1.
    #[allow(clippy::too_many_arguments)]
    fn first_event(
        &self,
        w0: &[f64],
        w1: &[f64],
        aw: &Mat,
        stuck: &[bool],
        e: &[f64],
        s: f64,
    ) -> Option<(f64, usize, f64)> {
        let n = self.n;
        let reversed: Vec<usize> = (0..n).filter(|&i| e[i] * w1[n + i] > 0.0).collect();
        let mut frees: Vec<(usize, f64)> = Vec::new();
        if stuck.iter().any(|&s| s) {
            for i in (0..n).filter(|&i| stuck[i]) {
                let p1 = self.pull(i, w1);
                if !held(p1, self.static_[i]) {
                    frees.push((i, p1));
                }
            }
        } else if reversed.is_empty() {
            return None;
        }
        if reversed.is_empty() && frees.is_empty() {
            return None;
        }
        let mut d0 = vec![0.0; aw.rows];
        let mut d1 = vec![0.0; aw.rows];
        aw.mul_add(w0, &mut d0);
        aw.mul_add(w1, &mut d1);
        let mut best: Option<(f64, usize, f64)> = None;
        for &i in &reversed {
            let way = -sign(e[i]);
            let t = first_root(way * w0[n + i], way * d0[n + i], way * w1[n + i], way * d1[n + i], s);
            if best.is_none_or(|b| t < b.0) {
                best = Some((t, i, 0.0));
            }
        }
        for &(i, p) in &frees {
            let sg = sign(p);
            let fs = self.static_[i];
            let t = first_root(
                fs - sg * self.pull(i, w0),
                -sg * self.pull(i, &d0),
                fs - sg * p,
                -sg * self.pull(i, &d1),
                s,
            );
            if best.is_none_or(|b| t < b.0) {
                best = Some((t, i, sg));
            }
        }
        best
    }

    /// Kinetic plus potential energy with the ground where the input puts it (`Simulator._stored_w`).
    fn stored(&self, w: &[f64]) -> f64 {
        let n = self.n;
        let ground = if self.base { w[2 * n] } else { 0.0 };
        let mut prev = ground;
        let mut energy = 0.0;
        for i in 0..n {
            let v = w[n + i];
            let stretch = w[i] - prev;
            energy += 0.5 * self.masses[i] * v * v + 0.5 * self.stiffness[i] * stretch * stretch;
            prev = w[i];
        }
        energy
    }

    /// The input's work over a sub-step of length s from w0 to w1, from q = ∫ x_k (`Simulator._energy_quadratics`'
    /// Q_work, integrated): a force u on mass k does ∫ u v_k; the ground, -∫ (k1 (x1 - u) + c1 (v1 - u')) u'.
    fn work(&self, w0: &[f64], w1: &[f64], s: f64) -> f64 {
        let n = self.n;
        let (u0, du, k) = (w0[2 * n], w0[2 * n + 1], self.k);
        let q = w1[self.m()];
        if self.base {
            let integral_u = u0 * s + 0.5 * du * s * s;
            -du * (self.stiffness[0] * (q - integral_u) + self.c1 * (w1[0] - w0[0]) - self.c1 * du * s)
        } else {
            u0 * (w1[k] - w0[k]) + du * (s * w1[k] - q)
        }
    }

    /// Step w = [x, v, u, du, e] by h in place (`_friction_step`); the step's (dissipated, work, friction loss).
    fn step(&mut self, w: &mut Vec<f64>, h: f64, ledger: bool) -> Losses {
        let (n, m) = (self.n, self.m());
        let mut totals = [0.0; 3];
        let mut left = h;
        let mut forced: Option<(usize, f64)> = None;
        w.resize(m + 1, 0.0);
        for attempt in 0..self.tries {
            let (stuck, e) = self.slip(w, forced.take());
            w[2 * n + 2..m].copy_from_slice(&e);
            w[m] = 0.0; // q from this sub-step's start
            let key = self.system(&stuck);
            let mut phi = self.transition(key, left, h);
            let mut w1 = vec![0.0; m + 1];
            phi.mul_add(w, &mut w1);
            let event = if attempt < self.tries - 1 {
                self.first_event(w, &w1, &self.systems[&key], &stuck, &e, left)
            } else {
                None
            };
            let mut s = left;
            if let Some((t, _, _)) = event
                && t < left
            {
                s = t;
                phi = self.transition(key, s, h);
                w1.fill(0.0);
                phi.mul_add(w, &mut w1);
            }
            if ledger {
                let rubbed: f64 = -(0..n).map(|i| e[i] * (w1[i] - w[i])).sum::<f64>();
                let work = self.work(w, &w1, s);
                let change = self.stored(&w1) - self.stored(w);
                if self.damped {
                    totals[0] += work - rubbed - change;
                }
                totals[1] += work;
                totals[2] += rubbed;
            }
            // Hold the stuck masses exactly, and stop the one that came to rest; the (tiny) energy this takes
            // out is friction's.
            if event.is_some() || stuck.iter().any(|&s| s) {
                let before = if ledger { self.stored(&w1) } else { 0.0 };
                for i in (0..n).filter(|&i| stuck[i]) {
                    w1[i] = w[i];
                    w1[n + i] = 0.0;
                }
                if let Some((_, i, way)) = event {
                    if way == 0.0 {
                        w1[n + i] = 0.0;
                    } else {
                        forced = Some((i, way));
                    }
                }
                if ledger {
                    totals[2] += before - self.stored(&w1);
                }
            }
            w.copy_from_slice(&w1);
            left -= s;
            if event.is_none() || left <= 0.0 {
                break;
            }
        }
        w.truncate(m);
        totals
    }
}

#[pymethods]
impl FrictionSteps {
    /// The chain's friction steps for one input: `aw` is `input_system(A, b, b_du)` ((2n+2)²), `p` the pull rows
    /// (`_pull_rows`), `k` the mass the input works on (the force's target, or 0 for the ground).
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (aw, p, masses, stiffness, damping, friction, static_friction, base, k, events_per_mass))]
    fn new(
        aw: PyReadonlyArray2<f64>,
        p: PyReadonlyArray2<f64>,
        masses: Vec<f64>,
        stiffness: Vec<f64>,
        damping: Vec<f64>,
        friction: Vec<f64>,
        static_friction: Vec<f64>,
        base: bool,
        k: usize,
        events_per_mass: usize,
    ) -> PyResult<Self> {
        let n = masses.len();
        let input = Mat::from_numpy(&aw);
        let p = Mat::from_numpy(&p);
        input.check("aw", 2 * n + 2, 2 * n + 2)?;
        p.check("p", n, 2 * n + 2)?;
        if n == 0
            || n > 64
            || [&stiffness, &damping, &friction, &static_friction]
                .iter()
                .any(|v| v.len() != n)
            || k >= n
        {
            return Err(PyValueError::new_err("the chain's arrays don't match (1 to 64 masses)"));
        }
        // w = [x, v, u, du, e, q]: the friction forces act on the masses, and q' = x_k.
        let m = 3 * n + 2;
        let mut big = Mat::zeros(m + 1, m + 1);
        for i in 0..2 * n + 2 {
            for j in 0..2 * n + 2 {
                big.set(i, j, input.at(i, j));
            }
        }
        for i in 0..n {
            big.set(n + i, 2 * n + 2 + i, 1.0 / masses[i]);
        }
        big.set(m, k, 1.0);
        Ok(FrictionSteps {
            n,
            aw: big,
            p,
            masses,
            stiffness,
            friction,
            static_: static_friction,
            base,
            k,
            c1: damping[0],
            damped: damping.iter().any(|&c| c > 0.0),
            tries: events_per_mass * n + 1,
            systems: HashMap::new(),
            steps: HashMap::new(),
        })
    }

    /// `_advance_friction`'s steps from the state z with the input u0 = `u0`, to the inputs `u` (one per step,
    /// at its end) over steps of h: each step's state into `out` (steps × 2n) and its (dissipated, work,
    /// friction loss) into `losses` (steps × 3).
    fn advance(
        &mut self,
        z: Vec<f64>,
        u0: f64,
        u: PyReadonlyArray1<f64>,
        h: f64,
        mut out: PyReadwriteArray2<f64>,
        mut losses: PyReadwriteArray2<f64>,
    ) -> PyResult<()> {
        let n = self.n;
        let u = u.as_array();
        let (mut out, mut losses) = (out.as_array_mut(), losses.as_array_mut());
        if z.len() != 2 * n || out.dim() != (u.len(), 2 * n) || losses.dim() != (u.len(), 3) {
            return Err(PyValueError::new_err(
                "the state, inputs or outputs don't match the chain",
            ));
        }
        let mut w = vec![0.0; 3 * n + 2];
        w[..2 * n].copy_from_slice(&z);
        let mut f0 = u0;
        for (j, &f1) in u.iter().enumerate() {
            w[2 * n] = f0;
            w[2 * n + 1] = (f1 - f0) / h;
            let totals = self.step(&mut w, h, true);
            for i in 0..2 * n {
                out[[j, i]] = w[i];
            }
            for i in 0..3 {
                losses[[j, i]] = totals[i];
            }
            f0 = f1;
        }
        Ok(())
    }

    /// `drive`'s steps, without the ledger: from z with the input u0, through the inputs `u` (at each step's
    /// end), each state into `out`. While every mass is at rest, held by its friction, and the input doesn't
    /// change, nothing moves until it does.
    fn drive(
        &mut self,
        z: Vec<f64>,
        u0: f64,
        u: PyReadonlyArray1<f64>,
        h: f64,
        mut out: PyReadwriteArray2<f64>,
    ) -> PyResult<()> {
        let n = self.n;
        let u = u.as_array();
        let mut out = out.as_array_mut();
        if z.len() != 2 * n || out.dim() != (u.len(), 2 * n) {
            return Err(PyValueError::new_err(
                "the state, inputs or output don't match the chain",
            ));
        }
        let mut w = vec![0.0; 3 * n + 2];
        w[..2 * n].copy_from_slice(&z);
        let mut f0 = u0;
        let mut k = 0;
        while k < u.len() {
            let f1 = u[k];
            if f1 == f0 && w[n..2 * n].iter().all(|&v| v == 0.0) {
                w[2 * n] = f0;
                w[2 * n + 1] = 0.0;
                if (0..n).all(|i| held(self.pull(i, &w), self.static_[i])) {
                    let stop = (k..u.len()).find(|&j| u[j] != f0).unwrap_or(u.len());
                    for j in k..stop {
                        for i in 0..2 * n {
                            out[[j, i]] = w[i];
                        }
                    }
                    k = stop;
                    continue;
                }
            }
            w[2 * n] = f0;
            w[2 * n + 1] = (f1 - f0) / h;
            self.step(&mut w, h, false);
            for i in 0..2 * n {
                out[[k, i]] = w[i];
            }
            f0 = f1;
            k += 1;
        }
        Ok(())
    }
}
