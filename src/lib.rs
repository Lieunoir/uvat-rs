//! Partial reimplementation in Rust of the paper
//! ["Joint optimization of distortion and cut location for mesh parameterization using an Ambrosio-Tortorelli functional"](https://perso.liris.cnrs.fr/david.coeurjolly/publication/uv-at/uv-at.pdf)
//! with better performances.
//!
//! Only the variational part is provided: the final cutting part is missing, as well as the
//! initialization part (which should be provided using Tutte's method after an initial cut).
//!
//! An interactive demo can be run using `cargo run -r --example demo`. Obj files to use can be
//! found [here](https://github.com/Lieunoir/UV-AT/tree/main/input).
//!
//! Can be used the following way (assuming $v$ holds vertices values and $f$ faces indices):
//! ```
//! use uvat_rs::utils::compute_tutte_parameterization;
//! use uvat_rs::utils::{build_edge_map, get_boundary_loop};
//!
//! let mut e = build_edge_map(&f, v.len());
//! let mut b = get_boundary_loop(&f, &e);
//! // If no boundary is found, we assume genus 0 surface and apply a simple cut.
//! // Higher genus surfaces are not handled
//! if b.len() == 0 {
//!     let v0 = v[f[0][1] as usize].to_owned();
//!     v.push(v0);
//!     f[0][1] = v.len() as u32 - 1;
//!     e = build_edge_map(&f, v.len());
//!     b = get_boundary_loop(&f, &e);
//! }
//!
//! let mut p = compute_tutte_parameterization(&v, &f, e, &b[0]);
//! let mut uvat_solver = UVAT::new(&v, &f, &mut p, UVAToptions::default());
//! let v_values = uvat_solver.solve(&f, &mut p);
//! // Now `p` holds the resulting parameterization, and `v_values` the values
//! // of $v$ in UVAT
//! ```
use core::f64::consts::SQRT_2;
use faer::linalg::solvers::Solve;
use faer::reborrow::Reborrow;
use faer::sparse::linalg::solvers::SymbolicLlt;
use faer::sparse::{Pair, SparseColMat, SymbolicSparseColMat, Triplet};
use faer::{Col, ColMut, Index};
use pulp::Arch;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelIterator,
};
use rayon::slice::ParallelSlice;
use utils::{apply_constraints, build_edge_map, flip_avoiding_line_search, MyIndex};
pub mod utils;

// v1 + t v1' (x1 + t x1', y1 + t y1')
//
//
// v2 + t v2' (x2 + t x2', y2 + t y2')
//
// x1 y2 + t ((x1' y2) + (x1 y2') + t2 x1' y2'
// - y1 x2 + t ((y1' x2) + (y1 x2') + t2 y1' x2'
//
// x1 y2 - y1 x2
// + t (x1' y2 + x1 y2' - y1' x2 - y1 x2')
// + t^2 (x1' y2' - y1' x2')

fn flip(u: &[f64; 4], v: &[f64; 4]) -> [f64; 4] {
    [
        //Actually the two last equal the two firsts
        (u[1] * v[0] + u[0] * v[2]) / SQRT_2,
        (u[1] * v[1] + u[0] * v[3]) / SQRT_2,
        (u[3] * v[0] + u[2] * v[2]) / SQRT_2,
        (u[3] * v[1] + u[2] * v[3]) / SQRT_2,
    ]
}

fn twist(u: &[f64; 4], v: &[f64; 4]) -> [f64; 4] {
    [
        //Actually the last two equal the first two
        (u[1] * v[0] - u[0] * v[2]) / SQRT_2,
        (u[1] * v[1] - u[0] * v[3]) / SQRT_2,
        (u[3] * v[0] - u[2] * v[2]) / SQRT_2,
        (u[3] * v[1] - u[2] * v[3]) / SQRT_2,
    ]
}
fn recover(u: &[f64; 4], v: &[f64; 4], s: &[f64; 2]) -> [f64; 4] {
    [
        (u[0] * v[0] * s[0] + u[1] * v[2] * s[1]),
        (u[0] * v[1] * s[0] + u[1] * v[3] * s[1]),
        (u[2] * v[0] * s[0] + u[3] * v[2] * s[1]),
        (u[2] * v[1] * s[0] + u[3] * v[3] * s[1]),
    ]
}

fn g(u: &[f64; 4], v: &[f64; 4], s: &[f64; 2]) -> [f64; 4] {
    [
        (u[0] * v[0] * s[1] + u[1] * v[2] * s[0]),
        (u[0] * v[1] * s[1] + u[1] * v[3] * s[0]),
        (u[2] * v[0] * s[1] + u[3] * v[2] * s[0]),
        (u[2] * v[1] * s[1] + u[3] * v[3] * s[0]),
    ]
}
fn g2(f: &[f64; 4]) -> [f64; 4] {
    if f[0] * f[3] - f[1] * f[2] > 0. {
        [f[3], -f[2], -f[1], f[0]]
    } else {
        [-f[3], f[2], f[1], -f[0]]
    }
}

struct SVD2d {
    u: [f64; 4],
    v: [f64; 4],
    s: [f64; 2],
}

fn x_scale(u: &[f64; 4], v: &[f64; 4]) -> [f64; 4] {
    [u[0] * v[0], u[0] * v[1], u[2] * v[0], u[2] * v[1]]
}

fn y_scale(u: &[f64; 4], v: &[f64; 4]) -> [f64; 4] {
    [u[1] * v[2], u[1] * v[3], u[3] * v[2], u[3] * v[3]]
}

fn prod_transpose(v: &[f64; 4], mul: f64) -> [f64; 16] {
    [
        v[0] * v[0] * mul,
        v[0] * v[1] * mul,
        v[0] * v[2] * mul,
        v[0] * v[3] * mul,
        v[1] * v[0] * mul,
        v[1] * v[1] * mul,
        v[1] * v[2] * mul,
        v[1] * v[3] * mul,
        v[2] * v[0] * mul,
        v[2] * v[1] * mul,
        v[2] * v[2] * mul,
        v[2] * v[3] * mul,
        v[3] * v[0] * mul,
        v[3] * v[1] * mul,
        v[3] * v[2] * mul,
        v[3] * v[3] * mul,
    ]
}

fn add(v1: &mut [f64; 16], v2: [f64; 16]) {
    for (x1, x2) in v1.iter_mut().zip(v2) {
        *x1 += x2;
    }
}

// from https://scicomp.stackexchange.com/questions/8899/robust-algorithm-for-2-times-2-svd
fn rq2x2_helper(f: &[f64; 4]) -> [f64; 5] {
    let a = f[0];
    let b = f[1];
    let c = f[2];
    let d = f[3];

    if c == 0. {
        return [a, b, d, 1., 0.];
    }
    let maxden = f64::max(c.abs(), d.abs());

    let rcmaxden = 1. / maxden;
    let c = c * rcmaxden;
    let d = d * rcmaxden;

    let den = 1. / (c * c + d * d).sqrt();

    let numx = -b * c + a * d;
    let numy = a * c + b * d;
    [numx * den, numy * den, maxden / den, d * den, -c * den]
}

fn svd_mat_2d(m: &[f64; 4]) -> SVD2d {
    // Calculate RQ decomposition of A
    let rq = rq2x2_helper(m);
    let x = rq[0];
    let y = rq[1];
    let z = rq[2];
    let c2 = rq[3];
    let s2 = rq[4];

    // Calculate tangent of rotation on R[x,y;0,z] to diagonalize R^T*R
    let scaler = 1. / f64::max(x.abs(), y.abs());
    let x_ = x * scaler;
    let y_ = y * scaler;
    let z_ = z * scaler;
    let numer = ((z_ - x_) * (z_ + x_)) + y_ * y_;
    let gamma = if numer == 0. { 0. } else { x_ * y_ };
    let zeta = numer / gamma;

    let t = 2. * zeta.signum() / (zeta.abs() + (zeta * zeta + 4.).sqrt());

    // Calculate sines and cosines
    let c1 = 1. / (1. + t * t).sqrt();
    let s1 = c1 * t;

    // Calculate U*S = R*R(c1,s1)
    let usa = c1 * x - s1 * y;
    let usb = s1 * x + c1 * y;
    let usc = -s1 * z;
    let usd = c1 * z;

    // Update V = R(c1,s1)^T*Q
    let t = c1 * c2 + s1 * s2;
    let s2 = c2 * s1 - c1 * s2;
    let c2 = t;

    // Separate U and S
    let d1 = f64::hypot(usa, usc);
    let d2 = f64::hypot(usb, usd);
    let dmax = f64::max(d1, d2);
    let (usmax1, usmax2) = if d2 > d1 { (usd, usb) } else { (usa, -usc) };

    //let signd1 = (x * z).signum();
    let signd1num = x * z;
    //let dmax = if d2 > d1 { signd1 * dmax } else { dmax };
    let dmax = if d2 > d1 {
        dmax.copysign(signd1num)
    } else {
        dmax
    };
    //let d2 = d2 * signd1;
    let d2 = d2.copysign(signd1num);
    let rcpdmax = 1. / dmax;

    let c1 = if dmax != 0. { usmax1 * rcpdmax } else { 1. };
    let s1 = if dmax != 0. { usmax2 * rcpdmax } else { 0. };
    let s = [d1, d2];
    let u = [c1, s1, -s1, c1];
    let v = [c2, -s2, s2, c2];
    SVD2d { u, v, s }
}

fn svd_mat_2d_old(m: &[f64; 4]) -> SVD2d {
    let e = 0.5 * (m[0] + m[3]);
    let f = 0.5 * (m[0] - m[3]);
    let g = 0.5 * (m[2] + m[1]);
    let h = 0.5 * (m[2] - m[1]);
    let q = (e.powi(2) + h.powi(2)).sqrt();
    let r = (f.powi(2) + g.powi(2)).sqrt();
    let sx = q + r;
    let sy = q - r;
    let a1 = g.atan2(f);
    let a2 = h.atan2(e);
    let theta = 0.5 * (a2 - a1);
    let (sin_t, cos_t) = theta.sin_cos();
    let phi = 0.5 * (a2 + a1);
    let (sin_p, cos_p) = phi.sin_cos();
    let s = [sx, sy.abs()];
    let u = [cos_p, -sin_p, sin_p, cos_p];
    let v = if sy >= 0. {
        [cos_t, -sin_t, sin_t, cos_t]
    } else {
        [cos_t, -sin_t, -sin_t, -cos_t]
    };
    SVD2d { s, u, v }
}

fn inv_mat_2d(mat: &[f64; 4]) -> [f64; 4] {
    let det = mat[0] * mat[3] - mat[1] * mat[2];
    [mat[3] / det, -mat[1] / det, -mat[2] / det, mat[0] / det]
}

fn compute_deformation_gradient_half(v1t: [f64; 3], v2t: [f64; 3], v3t: [f64; 3]) -> [f64; 4] {
    let vm1 = [v2t[0] - v1t[0], v2t[1] - v1t[1], v2t[2] - v1t[2]];
    let vm2 = [v3t[0] - v1t[0], v3t[1] - v1t[1], v3t[2] - v1t[2]];
    let l1 = (vm1[0].powi(2) + vm1[1].powi(2) + vm1[2].powi(2)).sqrt();
    let l2 = (vm2[0].powi(2) + vm2[1].powi(2) + vm2[2].powi(2)).sqrt();
    let cos = (vm1[0] * vm2[0] + vm1[1] * vm2[1] + vm1[2] * vm2[2]) / l1 / l2;
    let sin = (1. - cos.powi(2)).sqrt();
    let v1r = [0., 0.];
    let v2r = [l1, 0.];
    let v3r = [l2 * cos, l2 * sin];
    let m2 = [
        v2r[0] - v1r[0],
        v3r[0] - v1r[0],
        v2r[1] - v1r[1],
        v3r[1] - v1r[1],
    ];
    inv_mat_2d(&m2)
}

fn compute_deformation_gradient(m1: &[f64; 4], m2: &[f64; 4]) -> [f64; 4] {
    [
        m1[0] * m2[0] + m1[1] * m2[2],
        m1[0] * m2[1] + m1[1] * m2[3],
        m1[2] * m2[0] + m1[3] * m2[2],
        m1[2] * m2[1] + m1[3] * m2[3],
    ]
}

fn dfdx(m2: [f64; 4]) -> [f64; 24] {
    [
        -(m2[0] + m2[2]),
        0.,
        m2[0],
        0.,
        m2[2],
        0.,
        -(m2[1] + m2[3]),
        0.,
        m2[1],
        0.,
        m2[3],
        0.,
        0.,
        -(m2[0] + m2[2]),
        0.,
        m2[0],
        0.,
        m2[2],
        0.,
        -(m2[1] + m2[3]),
        0.,
        m2[1],
        0.,
        m2[3],
        // -(m2[0] + m2[2]), 0., m2[0], 0., m2[2], 0.,
        // -(m2[1] + m2[3]), 0., m2[1], 0., m2[3], 0.,
        // 0., -(m2[0] + m2[2]), 0., m2[0], 0., m2[2],
        // 0., -(m2[1] + m2[3]), 0., m2[1], 0., m2[3],
    ]
}

// const dfdx: [f64; 24] = [
// -2., 0., 1., 0., 1., 0.,
// -2., 0., 1., 0., 1., 0.,
// 0., -2., 0., 1., 0., 1.,
// 0., -2., 0., 1., 0., 1.,
// ];

fn finalize_hess(dfdx: &[f64; 24], h: &[f64; 16]) -> [f64; 36] {
    let mut acc = [0.; 24];
    for i in 0..4 {
        for j in 0..6 {
            //for i2 in (i * 4)..(i * 4 + 4) {
            for j2 in 0..4 {
                //let j2 = j + 6 * j2;
                let i2 = j2 + 4 * i;
                let j2 = j + 6 * j2;
                acc[i * 6 + j] += h[i2] * dfdx[j2];
            }
            // }
        }
    }
    let mut res = [0.; 36];
    for i in 0..6 {
        for j in 0..6 {
            for i2 in 0..4 {
                let j2 = j + 6 * i2;
                let i2 = i + 6 * i2;
                //for j2 in 0..4 {
                //let j2 = j + 6 * j2;
                res[i * 6 + j] += acc[j2] * dfdx[i2];
                //}
            }
        }
    }
    res
}

fn finalize_grad(dfdx: &[f64; 24], g: &[f64; 4]) -> [f64; 6] {
    let mut res = [0.; 6];
    for i in 0..6 {
        for j in 0..4 {
            res[i] += dfdx[i + j * 6] * g[j];
        }
    }
    res
}

fn compute_projected_deformation_gradient(
    v1: [f64; 2],
    v2: [f64; 2],
    v3: [f64; 2],
    m2: &[f64; 4],
) -> ([f64; 6], [f64; 36]) {
    let m1 = [v2[0] - v1[0], v3[0] - v1[0], v2[1] - v1[1], v3[1] - v1[1]];
    let f = compute_deformation_gradient(&m1, m2);

    let SVD2d { u, s, v } = svd_mat_2d_old(&f);
    //let s = bound_s(s, k);
    let f = recover(&u, &v, &s);

    let m1inv = inv_mat_2d(&m1);
    let m2 = [
        m1inv[0] * f[0] + m1inv[1] * f[2],
        m1inv[0] * f[1] + m1inv[1] * f[3],
        m1inv[2] * f[0] + m1inv[3] * f[2],
        m1inv[2] * f[1] + m1inv[3] * f[3],
    ];
    let dfdx = dfdx(m2);
    //let old_svd = svd_mat_2d_old(&f);
    let SVD2d { u, s, v } = svd_mat_2d_old(&f);
    //let SVD2d { u, s, v } = svd_mat_2d(&f);
    let d1 = x_scale(&u, &v);
    let d2 = y_scale(&u, &v);
    let l = flip(&u, &v);
    let t = twist(&u, &v);
    let g = g(&u, &v, &s);
    let lamb_1 = 1. + 3. / s[0].powi(4);
    let lamb_2 = 1. + 3. / s[1].powi(4);
    let i3 = s[0] * s[1];
    let i2 = s[0].powi(2) + s[1].powi(2);
    let lamb_3 = 1. + 1. / i3.powi(2) + i2 / i3.powi(3);
    let lamb_4 = 1. + 1. / i3.powi(2) - i2 / i3.powi(3);
    let mut grad = f;
    for (val, val_g) in grad.iter_mut().zip(g) {
        *val = *val * (1. + 1. / i3.powi(2)) - i2 / i3.powi(3) * val_g;
    }
    //Optimization messes something up
    let mut res = std::hint::black_box(prod_transpose(&d1, lamb_1));
    add(&mut res, prod_transpose(&d2, lamb_2));
    if lamb_3 > 0. {
        add(&mut res, prod_transpose(&l, lamb_3));
    }
    if lamb_4 > 0. {
        add(&mut res, prod_transpose(&t, lamb_4));
    }
    (finalize_grad(&dfdx, &grad), finalize_hess(&dfdx, &res))
}

fn build_hess_grad<I: Index>(
    f: &[[I; 3]],
    deformation_halves: &[[f64; 4]],
    p: &[[f64; 2]],
    order: &[I],
    offsets: &[I],
    weights: &[f64],
    values: &mut [f64],
    grad_values: &mut [f64],
) {
    let arch = Arch::new();
    arch.dispatch(|| {
        let mut hessians = Vec::new();
        let mut grads = Vec::new();

        f.par_iter()
            .zip(deformation_halves)
            .map(|(f, m2)| {
                let v1 = p[f[0].zx()];
                let v2 = p[f[1].zx()];
                let v3 = p[f[2].zx()];
                let res = compute_projected_deformation_gradient(v1, v2, v3, m2);
                res
            })
            .unzip_into_vecs(&mut grads, &mut hessians);
        for v in grad_values.iter_mut() {
            *v = 0.;
        }
        for ((grad, f), w) in grads.into_iter().zip(f).zip(weights) {
            for i in 0..3 {
                grad_values[2 * f[i].zx()] += -grad[2 * i] * w;
                grad_values[2 * f[i].zx() + 1] += -grad[2 * i + 1] * w;
            }
        }
        let hessians = hessians.into_flattened();
        values
            .par_iter_mut()
            .zip(offsets.par_windows(2))
            .for_each(|(v, offsets)| {
                let start = offsets[0];
                let end = offsets[1];
                let indices = &order[start.zx()..end.zx()];
                *v = 0.;
                for i in indices {
                    assert!(hessians[i.zx()].is_finite());
                    assert!(weights[i.zx() / 36].is_normal());
                    *v += hessians[i.zx()] * weights[i.zx() / 36];
                }
            });
    });
}

fn build_symbolic<I: Index>(f: &[[I; 3]]) -> (Vec<Pair<I, I>>, Vec<I>, Vec<I>) {
    let mut res_t = Vec::with_capacity(36 * f.len());
    let mut permut = [0; 36];
    let mut k = 0;
    for i in 0..3 {
        for j in 0..3 {
            permut[k] = 2 * i + 6 * 2 * j;
            permut[k + 1] = 2 * i + 1 + 6 * 2 * j;
            permut[k + 2] = 2 * i + 6 + 6 * 2 * j;
            permut[k + 3] = 2 * i + 7 + 6 * 2 * j;
            k += 4;
        }
    }
    for (index, f) in f.iter().enumerate() {
        for i in 0..3 {
            for j in 0..3 {
                let index_off = 36 * index;
                res_t.push((
                    (f[i] + f[i], f[j] + f[j]),
                    index_off + permut[i * 4 * 3 + j * 4],
                ));
                res_t.push((
                    (f[i] + f[i] + I::truncate(1), f[j] + f[j]),
                    index_off + permut[i * 4 * 3 + j * 4 + 1],
                ));
                res_t.push((
                    (f[i] + f[i], f[j] + f[j] + I::truncate(1)),
                    index_off + permut[i * 4 * 3 + j * 4 + 2],
                ));
                res_t.push((
                    (f[i] + f[i] + I::truncate(1), f[j] + f[j] + I::truncate(1)),
                    index_off + permut[i * 4 * 3 + j * 4 + 3],
                ));
            }
        }
    }
    res_t.sort_unstable_by_key(|t| t.0);
    let mut res = Vec::new();
    let mut offsets = vec![I::truncate(0)];
    let mut indices = Vec::new();
    let mut j = 0;
    let mut cur = res_t[0].0;
    for t in res_t {
        indices.push(Pair::new(t.0 .0, t.0 .1));
        res.push(I::truncate(t.1));
        if t.0 != cur {
            cur = t.0;
            offsets.push(I::truncate(j));
        }
        j += 1;
    }
    offsets.push(I::truncate(j));
    (indices, res, offsets)
}

/// User values for the UVAT solvers
#[derive(Clone)]
pub struct UVATOptions {
    /// Coefficient penalizing seam length over distorsion (lower means more seams allowed)
    pub lambda: f64,
    /// Upper bound of steps used to optimize UV
    pub steps: usize,
    /// Diffusion coefficient at the start of the optimization
    pub epsilon_start: f64,
    /// Diffusion coefficient at the end of the optimization
    pub epsilon_end: f64,
    /// Stabilization term for v
    pub mu: f64,
    /// Number of UV-then-AT steps ran for a fixed value of epsilon
    pub epsilon_steps: usize,
}

impl Default for UVATOptions {
    fn default() -> Self {
        UVATOptions {
            lambda: 1.,
            steps: 200,
            epsilon_start: 0.1,
            epsilon_end: 0.01,
            mu: 10e-9,
            epsilon_steps: 3,
        }
    }
}

/// UVAT solver. Simultaneously optimize distorsion and cuts using an Ambrosio-Tortorelli inspired method.
///
/// Cuts are indicated by a vector `v` holding a value for each face. Values close to `0` indicate the cut should
/// go along the face. The solver does not implement the actual cutting method, only the variational part.
pub struct UVAT<I: MyIndex> {
    options: UVATOptions,
    //Precomputed stuff
    deformation_halves: Vec<[f64; 4]>,
    order: Vec<I>,
    offsets: Vec<I>,
    init_weights: Vec<f64>,
    l: f64,
    //Matrices, vectors and solvers
    symbolic_uv: SymbolicLlt<I>,
    symbolic_at: SymbolicLlt<I>,
    hess: SparseColMat<I, f64>,
    grad_values: Vec<f64>,
    face_mass: SparseColMat<I, f64>,
    face_lap: SparseColMat<I, f64>,
    weights: Vec<f64>,
    //State
    current_epsilon: f64,
    current_epsilon_step: usize,
    total_steps: usize,
}

impl<I: MyIndex> UVAT<I> {
    /// Initializes the solver and precompute what can be precomputed once.
    ///
    /// Arguments :
    /// * `v` : 3d coordinates of the surface vertices
    /// * `f` : faces indices of the surface
    /// * `p` : a valid initial parameterization (such as one provided by the Tutte method)
    /// * `options` : user settings
    pub fn new(v: &[[f64; 3]], f: &[[I; 3]], p: &mut [[f64; 2]], options: UVATOptions) -> Self {
        //Center the anchor vertex
        let off = p[f[200][0].zx()];
        for val in p.iter_mut() {
            val[0] -= off[0];
            val[1] -= off[1];
        }
        let deformation_halves: Vec<_> = f
            .iter()
            .map(|f| compute_deformation_gradient_half(v[f[0].zx()], v[f[1].zx()], v[f[2].zx()]))
            .collect();
        let (indices, order, offsets) = build_symbolic(f);
        let sym = SymbolicSparseColMat::try_new_from_indices(2 * v.len(), 2 * v.len(), &indices)
            .unwrap()
            .0;
        let symbolic_uv = SymbolicLlt::try_new(sym.rb(), faer::Side::Lower).unwrap();
        let empty_vals = vec![0.; sym.compute_nnz()];
        let hess = SparseColMat::<I, f64>::new(sym, empty_vals);
        let grad_values = vec![0.; 2 * v.len()];
        let areas = areas(f, v);
        let l = areas
            .iter()
            .fold(0., |acc, v| acc + v.sqrt() / (areas.len() as f64));
        let face_mass = face_mass_matrix(&areas);
        let face_lap = face_laplacian(f, v.len());
        let symbolic_at =
            SymbolicLlt::try_new(face_lap.symbolic().rb(), faer::Side::Lower).unwrap();
        let mut weights = areas.clone();
        let tot_area: f64 = areas.iter().sum();
        for w in weights.iter_mut() {
            *w = *w / tot_area;
        }
        let current_epsilon = options.epsilon_start;
        Self {
            options,
            deformation_halves,
            symbolic_uv,
            order,
            offsets,
            symbolic_at,
            hess,
            grad_values,
            init_weights: weights.clone(),
            weights,
            l,
            face_mass,
            face_lap,
            current_epsilon,
            current_epsilon_step: 0,
            total_steps: 0,
        }
    }

    /// Optimize UV and AT.
    ///
    /// Arguments :
    /// * `f` : faces indices of the surface
    /// * `p` : a valid initial parameterization (such as one provided by the Tutte method)
    pub fn solve(&mut self, f: &[[I; 3]], p: &mut [[f64; 2]]) -> Vec<f64> {
        let mut v = vec![1.; p.len()];
        while !self.single_step(&f, p, &mut v) {}
        v
    }

    /// Single step of UV and then AT optimization. Returns wether the solver has ended or not. Can be used to dsplay intermediate states of the optimization.
    ///
    /// Arguments :
    /// * `f` : faces indices of the surface
    /// * `p` : a valid parameterization
    /// * `v` : current value for v, should be initialized as `vec![1; nf]` before the first iteration.
    pub fn single_step(&mut self, f: &[[I; 3]], p: &mut [[f64; 2]], v: &mut [f64]) -> bool {
        for cur_step in 0..self.options.steps {
            build_hess_grad(
                f,
                &self.deformation_halves,
                p,
                &self.order,
                &self.offsets,
                &self.weights,
                self.hess.val_mut(),
                &mut self.grad_values,
            );
            let mut grad = ColMut::from_slice_mut(&mut self.grad_values);
            //let norm = grad.norm_max() / f.len() as f64;
            let norm = grad.norm_max();
            if norm < 10e-4 {
                self.total_steps += cur_step;
                break;
            }
            apply_constraints(
                &mut self.hess,
                grad.as_mut(),
                &[2 * f[200][0].zx(), 2 * f[200][0].zx() + 1],
                &[0., 0.],
            );
            let llt = faer::sparse::linalg::solvers::Llt::try_new_with_symbolic(
                self.symbolic_uv.clone(),
                self.hess.rb(),
                faer::Side::Lower,
            )
            .unwrap();
            llt.solve_in_place(grad.as_mat_mut());
            //TODO avoid allocation here
            let res: Vec<_> = self
                .grad_values
                .chunks(2)
                .into_iter()
                .map(|c| [c[0], c[1]])
                .collect();
            let e = |p: &[[f64; 2]]| {
                let sym_dir_vec = sym_dir_vec(&f, &self.deformation_halves, p);
                sym_dir_vec
                    .into_iter()
                    .zip(self.weights.iter())
                    .map(|(e, w)| e * w)
                    .sum::<f64>()
            };
            flip_avoiding_line_search(p, &res, f, e);
        }
        let sym_dir_vec = sym_dir_vec(f, &self.deformation_halves, p);
        solve_v_system(
            self.options.lambda * self.l,
            self.current_epsilon * self.l,
            &self.face_mass,
            &self.face_lap,
            sym_dir_vec,
            v,
        );
        for ((&v, &init_w), w) in v
            .iter()
            .zip(self.init_weights.iter())
            .zip(self.weights.iter_mut())
        {
            *w = (v.powi(2) + self.options.mu) * init_w;
        }
        self.current_epsilon_step += 1;
        if self.current_epsilon_step == self.options.epsilon_steps {
            self.current_epsilon_step = 0;
            self.current_epsilon *= 0.5;
            if self.current_epsilon < self.options.epsilon_end {
                true
            } else {
                false
            }
        } else {
            false
        }
    }
}

fn areas<I: Index>(f: &[[I; 3]], v: &[[f64; 3]]) -> Vec<f64> {
    let arch = Arch::new();
    let lengths = arch.dispatch(|| {
        let mut edges_t = Vec::new();
        f.into_par_iter()
            .map(|row| {
                let v1 = &v[row[0].zx()];
                let v2 = &v[row[1].zx()];
                let v3 = &v[row[2].zx()];
                let edge1 = [v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]];
                let edge2 = [v3[0] - v2[0], v3[1] - v2[1], v3[2] - v2[2]];
                let edge3 = [v1[0] - v3[0], v1[1] - v3[1], v1[2] - v3[2]];
                [
                    edge1[0], edge1[1], edge1[2], edge2[0], edge2[1], edge2[2], edge3[0], edge3[1],
                    edge3[2],
                ]
            })
            .collect_into_vec(&mut edges_t);
        let edges_t = edges_t.into_flattened();
        pulp::as_arrays::<9, f64>(&edges_t)
            .0
            .into_iter()
            //.chunks_exact(9)
            .map(|row| {
                [
                    f64::sqrt(row[0] * row[0] + row[1] * row[1] + row[2] * row[2]),
                    f64::sqrt(row[3] * row[3] + row[4] * row[4] + row[5] * row[5]),
                    f64::sqrt(row[6] * row[6] + row[7] * row[7] + row[8] * row[8]),
                ]
            })
            .collect::<Vec<_>>()
    });
    let mut areas = Vec::new();
    lengths
        .into_par_iter()
        .map(|l| {
            let s = 0.5 * (l[0] + l[1] + l[2]);
            (s * (s - l[0]) * (s - l[1]) * (s - l[2])).sqrt()
        })
        .collect_into_vec(&mut areas);
    areas
}

fn face_mass_matrix<I: Index>(areas: &[f64]) -> SparseColMat<I, f64> {
    let triplets: Vec<_> = areas
        .into_iter()
        .enumerate()
        .map(|(i, &area)| Triplet::new(I::truncate(i), I::truncate(i), area))
        .collect();
    SparseColMat::try_new_from_triplets(areas.len(), areas.len(), &triplets).unwrap()
}

fn sym_dir_vec<I: Index>(
    f: &[[I; 3]],
    deformation_halves: &[[f64; 4]],
    p: &[[f64; 2]],
) -> Vec<f64> {
    let mut res = Vec::new();
    f.par_iter()
        .zip(deformation_halves)
        .map(|(f, m2)| {
            let v1 = p[f[0].zx()];
            let v2 = p[f[1].zx()];
            let v3 = p[f[2].zx()];
            let m1 = [v2[0] - v1[0], v3[0] - v1[0], v2[1] - v1[1], v3[1] - v1[1]];
            let f = compute_deformation_gradient(&m1, m2);
            let f_norm = f[0] * f[0] + f[1] * f[1] + f[2] * f[2] + f[3] * f[3];
            let f_inv_norm = f_norm / (f[0] * f[3] - f[1] * f[2]).powi(2);
            0.25 * (f_norm + f_inv_norm) - 1.
        })
        .collect_into_vec(&mut res);
    res
}

fn build_v_system<I: MyIndex>(
    lambda: f64,
    epsilon: f64,
    mass: &SparseColMat<I, f64>,
    lap: &SparseColMat<I, f64>,
    sym_dir_vec: Vec<f64>,
) -> SparseColMat<I, f64> {
    let mut m_i = mass.clone();
    m_i.val_mut()
        .par_iter_mut()
        .zip(sym_dir_vec.par_iter())
        .for_each(|(m, s)| {
            *m = *m * (s + lambda / (4. * epsilon));
        });
    let res = faer::Scale(epsilon * lambda) * lap.clone() + m_i;
    res
}

fn solve_v_system<I: MyIndex>(
    lambda: f64,
    epsilon: f64,
    mass: &SparseColMat<I, f64>,
    lap: &SparseColMat<I, f64>,
    sym_dir_vec: Vec<f64>,
    v: &mut [f64],
) {
    let hess = build_v_system(lambda, epsilon, mass, lap, sym_dir_vec);
    let rhs = Col::<f64>::ones(v.len());
    let mut rhs = faer::Scale(lambda / (4. * epsilon)) * mass * rhs;
    let llt = hess.sp_cholesky(faer::Side::Lower).unwrap();
    llt.solve_in_place(rhs.as_mat_mut());
    for (v, &v_rhs) in v.iter_mut().zip(rhs.iter()) {
        *v = v_rhs;
    }
}

fn face_laplacian<I: MyIndex>(f: &[[I; 3]], nv: usize) -> SparseColMat<I, f64> {
    let e = build_edge_map(&f, nv);
    let nf = f.len();
    let mut triplets = Vec::new();
    for (i, f) in e.into_iter().enumerate() {
        for j in 0..3 {
            if f.1[j] >= 0 && I::truncate(i) < f.0[j] {
                triplets.push(Triplet::new(I::truncate(i), f.0[j], -1.));
                triplets.push(Triplet::new(f.0[j], I::truncate(i), -1.));
                triplets.push(Triplet::new(I::truncate(i), I::truncate(i), 1.));
                triplets.push(Triplet::new(f.0[j], f.0[j], 1.));
            }
        }
    }
    SparseColMat::try_new_from_triplets(nf, nf, &triplets).unwrap()
}
