//! Small dense matrices: products, solves and the matrix exponential, for the kernels' few-dozen-state systems.

use numpy::{PyArrayMethods, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// A dense row-major matrix.
#[derive(Clone)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Mat {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Mat {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn eye(n: usize) -> Self {
        let mut m = Mat::zeros(n, n);
        for i in 0..n {
            m.data[i * n + i] = 1.0;
        }
        m
    }

    pub fn from_numpy(a: &PyReadonlyArray2<f64>) -> Self {
        let a = a.as_array();
        let (rows, cols) = a.dim();
        Mat {
            rows,
            cols,
            data: a.iter().copied().collect(),
        }
    }

    pub fn at(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.cols + j]
    }

    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        self.data[i * self.cols + j] = v;
    }

    pub fn row(&self, i: usize) -> &[f64] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }

    /// out += self @ x
    pub fn mul_add(&self, x: &[f64], out: &mut [f64]) {
        for (i, o) in out.iter_mut().enumerate().take(self.rows) {
            *o += self.row(i).iter().zip(x).map(|(a, b)| a * b).sum::<f64>();
        }
    }

    pub fn matmul(&self, other: &Mat) -> Mat {
        assert_eq!(self.cols, other.rows);
        let (m, k, n) = (self.rows, self.cols, other.cols);
        let mut out = Mat::zeros(m, n);
        // SAFETY: each pointer spans its matrix, row-major with the strides given, and out is its own.
        unsafe {
            matrixmultiply::dgemm(
                m,
                k,
                n,
                1.0,
                self.data.as_ptr(),
                k as isize,
                1,
                other.data.as_ptr(),
                n as isize,
                1,
                0.0,
                out.data.as_mut_ptr(),
                n as isize,
                1,
            );
        }
        out
    }

    /// self * a + other * b, elementwise
    pub fn combine(&self, a: f64, other: &Mat, b: f64) -> Mat {
        let data = self.data.iter().zip(&other.data).map(|(x, y)| a * x + b * y).collect();
        Mat {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    /// The rows `rows` and columns `cols` of self.
    pub fn select(&self, rows: &[usize], cols: &[usize]) -> Mat {
        let mut out = Mat::zeros(rows.len(), cols.len());
        for (i, &r) in rows.iter().enumerate() {
            for (j, &c) in cols.iter().enumerate() {
                out.set(i, j, self.at(r, c));
            }
        }
        out
    }

    /// The maximum absolute column sum.
    pub fn norm1(&self) -> f64 {
        (0..self.cols)
            .map(|j| (0..self.rows).map(|i| self.at(i, j).abs()).sum::<f64>())
            .fold(0.0, f64::max)
    }

    pub fn check(&self, name: &str, rows: usize, cols: usize) -> PyResult<()> {
        if self.rows == rows && self.cols == cols {
            Ok(())
        } else {
            Err(PyValueError::new_err(format!(
                "{name} is {}x{}, not {rows}x{cols}",
                self.rows, self.cols
            )))
        }
    }
}

pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub fn max_abs(a: &[f64]) -> f64 {
    a.iter().fold(0.0, |m, x| m.max(x.abs()))
}

/// X with A X = B, by LU with partial pivoting; A (n × n) and B (n × k) are spent.
pub fn solve(mut a: Mat, mut b: Mat) -> Mat {
    let n = a.rows;
    let k = b.cols;
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&i, &j| a.at(i, col).abs().total_cmp(&a.at(j, col).abs()))
            .unwrap();
        if pivot != col {
            for j in 0..n {
                a.data.swap(col * n + j, pivot * n + j);
            }
            for j in 0..k {
                b.data.swap(col * k + j, pivot * k + j);
            }
        }
        let d = a.at(col, col);
        let (top, rest) = a.data.split_at_mut((col + 1) * n);
        let prow = &top[col * n + col..col * n + n];
        let (btop, brest) = b.data.split_at_mut((col + 1) * k);
        let brow = &btop[col * k..];
        for (arow, bi) in rest.chunks_exact_mut(n).zip(brest.chunks_exact_mut(k)) {
            let f = arow[col] / d;
            if f != 0.0 {
                arow[col..].iter_mut().zip(prow).for_each(|(x, p)| *x -= f * p);
                bi.iter_mut().zip(brow).for_each(|(x, p)| *x -= f * p);
            }
        }
    }
    // Back substitution, a row of X at a time: X_i = (B_i - Σ_{l>i} a_il X_l) / a_ii.
    for i in (0..n).rev() {
        let (head, tail) = b.data.split_at_mut((i + 1) * k);
        let bi = &mut head[i * k..];
        for (l, xl) in tail.chunks_exact(k).enumerate() {
            let c = a.at(i, i + 1 + l);
            if c != 0.0 {
                bi.iter_mut().zip(xl).for_each(|(x, y)| *x -= c * y);
            }
        }
        let d = a.at(i, i);
        bi.iter_mut().for_each(|x| *x /= d);
    }
    b
}

/// The Padé approximants' coefficients, degrees 3, 5, 7, 9 and 13 (Higham 2005).
const B3: [f64; 4] = [120.0, 60.0, 12.0, 1.0];
const B5: [f64; 6] = [30240.0, 15120.0, 3360.0, 420.0, 30.0, 1.0];
const B7: [f64; 8] = [17297280.0, 8648640.0, 1995840.0, 277200.0, 25200.0, 1512.0, 56.0, 1.0];
const B9: [f64; 10] = [
    17643225600.0,
    8821612800.0,
    2075673600.0,
    302702400.0,
    30270240.0,
    2162160.0,
    110880.0,
    3960.0,
    90.0,
    1.0,
];
const B13: [f64; 14] = [
    64764752532480000.0,
    32382376266240000.0,
    7771770303897600.0,
    1187353796428800.0,
    129060195264000.0,
    10559470521600.0,
    670442572800.0,
    33522128640.0,
    1323241920.0,
    40840800.0,
    960960.0,
    16380.0,
    182.0,
    1.0,
];
/// The largest ‖A^k‖^{1/k} each degree takes to double precision (Al-Mohy & Higham 2009, Table 3.1).
#[allow(clippy::excessive_precision)] // as the paper gives them
const THETA: [f64; 5] = [
    1.495585217958292e-2,
    2.539398330063230e-1,
    9.504178996162932e-1,
    2.097847961257068e0,
    4.25,
];
/// 1 / |c_{2m+1}|, the leading coefficient of each degree's backward error (for `ell`).
const C_RECIP: [f64; 5] = [
    100800.0,
    10059033600.0,
    4487938430976000.0,
    5914384781877411840000.0,
    113250775606021113483283660800000000.0,
];

/// A matrix algebra the matrix exponential works in: dense matrices, or the FOH's structured ones.
pub trait Algebra: Sized {
    fn mul(&self, other: &Self) -> Self;
    /// self * a + other * b
    fn lin(&self, a: f64, other: &Self, b: f64) -> Self;
    fn identity(&self) -> Self;
    fn norm1(&self) -> f64;
    /// ‖|A|^p‖₁, by powering a vector of ones through |A|ᵀ.
    fn abs_power_norm(&self, p: usize) -> f64;
    /// X with den X = num.
    fn divide(num: Self, den: Self) -> Self;
}

impl Algebra for Mat {
    fn mul(&self, other: &Self) -> Self {
        self.matmul(other)
    }

    fn lin(&self, a: f64, other: &Self, b: f64) -> Self {
        self.combine(a, other, b)
    }

    fn identity(&self) -> Self {
        Mat::eye(self.rows)
    }

    fn norm1(&self) -> f64 {
        Mat::norm1(self)
    }

    fn abs_power_norm(&self, p: usize) -> f64 {
        let n = self.rows;
        let abs_t = abs_transpose(self);
        let mut v = vec![1.0; n];
        for _ in 0..p {
            let mut w = vec![0.0; n];
            abs_t.mul_add(&v, &mut w);
            v = w;
        }
        v.iter().fold(0.0, |x: f64, &y| x.max(y))
    }

    fn divide(num: Self, den: Self) -> Self {
        solve(den, num)
    }
}

fn abs_transpose(a: &Mat) -> Mat {
    let mut t = Mat::zeros(a.cols, a.rows);
    for i in 0..a.rows {
        for j in 0..a.cols {
            t.set(j, i, a.at(i, j).abs());
        }
    }
    t
}

/// The first-order hold's augmented matrices, and every polynomial and ratio of them:
///
///     [[T, R], [0, s I + s' N]],   N = [[0, I], [0, 0]] (2nu × 2nu, N² = 0),
///
/// T (ns × ns), R (ns × 2nu). `foh_discretize`'s [[A h, B h, 0], [0, 0, I], [0, 0, 0]] is one, with
/// T = A h, R = [B h, 0], s = 0, s' = 1; so is its exponential, [[Φ, Γ1, Γ2], [0, I, I], [0, 0, I]].
/// A product costs ns³ + 2 ns² nu, against (ns + 2nu)³ dense.
#[derive(Clone)]
pub struct Foh {
    t: Mat,
    r: Mat,
    s: f64,
    s1: f64,
}

impl Foh {
    fn nu(&self) -> usize {
        self.r.cols / 2
    }

    /// R (s I + s' N): R N moves R's first half into its second.
    fn r_times_s(&self, r: &Mat, s: f64, s1: f64) -> Mat {
        let nu = self.nu();
        let mut out = r.combine(s, r, 0.0);
        if s1 != 0.0 {
            for i in 0..r.rows {
                for j in 0..nu {
                    out.data[i * r.cols + nu + j] += s1 * r.at(i, j);
                }
            }
        }
        out
    }
}

impl Algebra for Foh {
    fn mul(&self, o: &Self) -> Self {
        let r = self
            .t
            .matmul(&o.r)
            .combine(1.0, &self.r_times_s(&self.r, o.s, o.s1), 1.0);
        Foh {
            t: self.t.matmul(&o.t),
            r,
            s: self.s * o.s,
            s1: self.s * o.s1 + self.s1 * o.s,
        }
    }

    fn lin(&self, a: f64, o: &Self, b: f64) -> Self {
        Foh {
            t: self.t.combine(a, &o.t, b),
            r: self.r.combine(a, &o.r, b),
            s: a * self.s + b * o.s,
            s1: a * self.s1 + b * o.s1,
        }
    }

    fn identity(&self) -> Self {
        Foh {
            t: Mat::eye(self.t.rows),
            r: Mat::zeros(self.r.rows, self.r.cols),
            s: 1.0,
            s1: 0.0,
        }
    }

    fn norm1(&self) -> f64 {
        // The largest column sum: T's columns, then R's over s I + s' N's (|s| in the first nu, |s| + |s'| after).
        let nu = self.nu();
        let t = self.t.norm1();
        let r = (0..self.r.cols)
            .map(|j| {
                let col: f64 = (0..self.r.rows).map(|i| self.r.at(i, j).abs()).sum();
                col + self.s.abs() + if j >= nu { self.s1.abs() } else { 0.0 }
            })
            .fold(0.0, f64::max);
        t.max(r)
    }

    fn abs_power_norm(&self, p: usize) -> f64 {
        // |X|ᵀ [v, w] = [|T|ᵀ v, |R|ᵀ v + (|s| I + |s'| Nᵀ) w]
        let (ns, nu) = (self.t.rows, self.nu());
        let (t_abs, r_abs) = (abs_transpose(&self.t), abs_transpose(&self.r));
        let mut v = vec![1.0; ns];
        let mut w = vec![1.0; 2 * nu];
        for _ in 0..p {
            let mut v2 = vec![0.0; ns];
            t_abs.mul_add(&v, &mut v2);
            let mut w2: Vec<f64> = (0..2 * nu)
                .map(|j| self.s.abs() * w[j] + if j >= nu { self.s1.abs() * w[j - nu] } else { 0.0 })
                .collect();
            r_abs.mul_add(&v, &mut w2);
            (v, w) = (v2, w2);
        }
        v.iter().chain(&w).fold(0.0, |x: f64, &y| x.max(y))
    }

    fn divide(num: Self, den: Self) -> Self {
        // den X = num, block by block: X_s = den_s⁻¹ num_s, then den_T [X_T, X_R] = [num_T, num_R - den_R X_s].
        let (a, b) = (den.s, den.s1);
        let (s, s1) = (num.s / a, num.s1 / a - b * num.s / (a * a)); // (a I + b N)⁻¹ = I/a - b N/a²
        let rhs_r = num.r.combine(1.0, &num.r_times_s(&den.r, s, s1), -1.0);
        let (ns, k) = (num.t.rows, num.r.cols);
        let mut rhs = Mat::zeros(ns, ns + k);
        for i in 0..ns {
            rhs.data[i * (ns + k)..i * (ns + k) + ns].copy_from_slice(num.t.row(i));
            rhs.data[i * (ns + k) + ns..(i + 1) * (ns + k)].copy_from_slice(rhs_r.row(i));
        }
        let x = solve(den.t, rhs);
        let all: Vec<usize> = (0..ns).collect();
        Foh {
            t: x.select(&all, &all),
            r: x.select(&all, &(ns..ns + k).collect::<Vec<_>>()),
            s,
            s1,
        }
    }
}

/// How many more squarings degree `DEGREES[i]` needs for A's backward error, beyond what the norms say
/// (Al-Mohy & Higham 2009, eq. 3.13, as SciPy's `_ell`).
fn ell<M: Algebra>(a: &M, i: usize) -> i32 {
    const DEGREES: [usize; 5] = [3, 5, 7, 9, 13];
    let m = DEGREES[i];
    let abs_norm = a.abs_power_norm(2 * m + 1);
    if abs_norm == 0.0 {
        return 0;
    }
    let alpha = abs_norm / (a.norm1() * C_RECIP[i]);
    let u = 2f64.powi(-53);
    ((alpha / u).log2() / (2 * m) as f64).ceil().max(0.0) as i32
}

/// e^A by scaling and squaring with a Padé approximant, the degree and scaling picked from the norms of
/// A's powers (Al-Mohy & Higham, SIAM J. Matrix Anal. Appl. 31, 2009, Algorithm 5.1; as SciPy's expm).
/// A matrix far smaller in its powers than in itself (fast modes over a short step) takes few squarings.
pub fn expm<M: Algebra>(a: &M) -> M {
    let eye = a.identity();
    let a2 = a.mul(a);
    let a4 = a2.mul(&a2);
    let a6 = a4.mul(&a2);
    let d4 = a4.norm1().powf(0.25);
    let d6 = a6.norm1().powf(1.0 / 6.0);
    // (U, V): the odd and even parts of the approximant's numerator, so r = (V - U)⁻¹ (V + U).
    let pade = |(u, v): (M, M)| M::divide(v.lin(1.0, &u, 1.0), v.lin(1.0, &u, -1.0));
    let low = |b: &[f64]| -> (M, M) {
        let a8 = if b.len() > 8 { Some(a4.mul(&a4)) } else { None }; // degree 9's only
        let powers = [Some(&eye), Some(&a2), Some(&a4), Some(&a6), a8.as_ref()]; // I, A², A⁴, …
        let mut u = eye.lin(0.0, &eye, 0.0);
        let mut v = eye.lin(0.0, &eye, 0.0);
        for k in 0..b.len() / 2 {
            let p = powers[k].unwrap();
            u = u.lin(1.0, p, b[2 * k + 1]);
            v = v.lin(1.0, p, b[2 * k]);
        }
        (a.mul(&u), v)
    };
    let eta1 = d4.max(d6);
    if eta1 <= THETA[0] && ell(a, 0) == 0 {
        return pade(low(&B3));
    }
    if eta1 <= THETA[1] && ell(a, 1) == 0 {
        return pade(low(&B5));
    }
    let a8 = a4.mul(&a4);
    let d8 = a8.norm1().powf(0.125);
    let eta3 = d6.max(d8);
    if eta3 <= THETA[2] && ell(a, 2) == 0 {
        return pade(low(&B7));
    }
    if eta3 <= THETA[3] && ell(a, 3) == 0 {
        return pade(low(&B9));
    }
    let d10 = a4.mul(&a6).norm1().powf(0.1);
    let eta5 = eta3.min(d8.max(d10));
    let mut s = if eta5 == 0.0 {
        0
    } else {
        ((eta5 / THETA[4]).log2().ceil() as i32).max(0)
    };
    s += ell(&a.lin(0.5f64.powi(s), a, 0.0), 4);
    let scale = |m: &M, k: i32| m.lin(0.5f64.powi(k * s), m, 0.0);
    let (a, a2, a4, a6) = (scale(a, 1), scale(&a2, 2), scale(&a4, 4), scale(&a6, 6));
    let b = &B13;
    let inner_u = a6.lin(b[13], &a4, b[11]).lin(1.0, &a2, b[9]);
    let outer_u = a6.lin(b[7], &a4, b[5]).lin(1.0, &a2, b[3]).lin(1.0, &eye, b[1]);
    let u = a.mul(&a6.mul(&inner_u).lin(1.0, &outer_u, 1.0));
    let inner_v = a6.lin(b[12], &a4, b[10]).lin(1.0, &a2, b[8]);
    let outer_v = a6.lin(b[6], &a4, b[4]).lin(1.0, &a2, b[2]).lin(1.0, &eye, b[0]);
    let v = a6.mul(&inner_v).lin(1.0, &outer_v, 1.0);
    let mut r = pade((u, v));
    for _ in 0..s {
        r = r.mul(&r);
    }
    r
}

/// Powers of 2 d (one per state) that balance [T R] as EISPACK's `balanc` does (Parlett & Reinsch 1969):
/// D⁻¹ T D and D⁻¹ R with each state's row and column of similar size. The rotor's A h runs from h (x' = v)
/// to K/M h; balanced, its norm is a few hundred times smaller, and its exponential far more accurate.
#[allow(clippy::needless_range_loop)] // i indexes t and r too
fn balance(t: &mut Mat, r: &mut Mat) -> Vec<f64> {
    let ns = t.rows;
    let mut d = vec![1.0; ns];
    loop {
        let mut done = true;
        for i in 0..ns {
            let c: f64 = (0..ns).filter(|&j| j != i).map(|j| t.at(j, i).abs()).sum();
            let row: f64 = (0..ns).filter(|&j| j != i).map(|j| t.at(i, j).abs()).sum::<f64>()
                + r.row(i).iter().map(|x| x.abs()).sum::<f64>();
            if c == 0.0 || row == 0.0 {
                continue;
            }
            let (mut f, mut cs, s) = (1.0, c, c + row);
            while cs < row / 2.0 {
                f *= 2.0;
                cs *= 4.0;
            }
            while cs >= row * 2.0 {
                f /= 2.0;
                cs /= 4.0;
            }
            if (cs + row) / f < 0.95 * s {
                done = false;
                d[i] *= f;
                for j in 0..ns {
                    t.data[i * ns + j] /= f;
                    t.data[j * ns + i] *= f;
                }
                r.data[i * r.cols..(i + 1) * r.cols].iter_mut().for_each(|x| *x /= f);
            }
        }
        if done {
            return d;
        }
    }
}

/// (Φ, Γ1, Γ2) with e^{[[A h, B h, 0], [0, 0, I], [0, 0, 0]]} = [[Φ, Γ1, Γ2], [0, I, I], [0, 0, I]]:
/// `foh_discretize`'s exponential (its Γ0 is Γ1 - Γ2), worked out in `Foh`'s blocks, balanced.
pub fn expm_foh(a: &Mat, b: &Mat, h: f64) -> (Mat, Mat, Mat) {
    let (ns, nu) = (a.rows, b.cols);
    let mut t = a.combine(h, a, 0.0);
    let mut r = Mat::zeros(ns, 2 * nu);
    for i in 0..ns {
        for j in 0..nu {
            r.set(i, j, b.at(i, j) * h);
        }
    }
    let d = balance(&mut t, &mut r);
    let mut e = expm(&Foh { t, r, s: 0.0, s1: 1.0 });
    // Undo the balancing: e^X = D e^{X'} D⁻¹, the inputs unscaled.
    for (i, di) in d.iter().enumerate() {
        for (j, dj) in d.iter().enumerate() {
            e.t.data[i * ns + j] *= di / dj;
        }
        e.r.data[i * 2 * nu..(i + 1) * 2 * nu].iter_mut().for_each(|x| *x *= di);
    }
    let all: Vec<usize> = (0..ns).collect();
    (
        e.t,
        e.r.select(&all, &(0..nu).collect::<Vec<_>>()),
        e.r.select(&all, &(nu..2 * nu).collect::<Vec<_>>()),
    )
}

/// e^A, balanced first (`balance`): for matrices as badly scaled as a chain's or a rotor's A h.
pub fn expm_balanced(a: &Mat) -> Mat {
    let n = a.rows;
    let mut t = a.clone();
    let d = balance(&mut t, &mut Mat::zeros(n, 0));
    let mut e = expm(&t);
    for (i, di) in d.iter().enumerate() {
        for (j, dj) in d.iter().enumerate() {
            e.data[i * n + j] *= di / dj;
        }
    }
    e
}

type Array<'py> = Bound<'py, numpy::PyArray2<f64>>;

fn to_numpy<'py>(py: Python<'py>, m: Mat) -> PyResult<Array<'py>> {
    numpy::PyArray1::from_vec(py, m.data).reshape([m.rows, m.cols])
}

/// e^A, for the tests to check against SciPy's.
#[pyfunction(name = "expm")]
pub fn expm_py<'py>(py: Python<'py>, a: PyReadonlyArray2<'py, f64>) -> PyResult<Array<'py>> {
    let m = Mat::from_numpy(&a);
    if m.rows != m.cols {
        return Err(PyValueError::new_err("expm needs a square matrix"));
    }
    to_numpy(py, expm(&m))
}

/// (Φ, Γ0, Γ1) as `simulator.foh_discretize(A, B, h)` returns them, for the tests to check.
#[pyfunction(name = "foh_discretize")]
pub fn foh_discretize_py<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray2<'py, f64>,
    h: f64,
) -> PyResult<(Array<'py>, Array<'py>, Array<'py>)> {
    let (a, b) = (Mat::from_numpy(&a), Mat::from_numpy(&b));
    if a.rows != a.cols || b.rows != a.rows {
        return Err(PyValueError::new_err("A must be square, and B have its rows"));
    }
    let (phi, g1, g2) = expm_foh(&a, &b, h);
    Ok((
        to_numpy(py, phi)?,
        to_numpy(py, g1.combine(1.0, &g2, -1.0))?,
        to_numpy(py, g2)?,
    ))
}
