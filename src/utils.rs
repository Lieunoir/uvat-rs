use faer::linalg::solvers::Solve;
use faer::reborrow::ReborrowMut;
use faer::sparse::{SparseColMat, SymbolicSparseColMat, Triplet};
use faer::{ColMut, Index, Mat, MatMut, Row};
use faer_traits::Index as tIndex;
use pulp::Arch;
use rayon::prelude::*;
use std::collections::VecDeque;
use std::f64::consts::PI;
use std::ops::{BitAnd, Mul, Shl, Shr};
use std::sync::atomic::{AtomicI8, AtomicU32, AtomicU64, Ordering};

/// Index trait encompassing operations from `faer`'s Index trait and some more.
pub trait MyIndex:
    Index
    + Shl<usize, Output = Self>
    + Shr<usize, Output = Self>
    + BitAnd<Output = Self>
    + Mul<Output = Self>
{
}

impl MyIndex for u32 {}

#[cfg(not(target_arch = "wasm32"))]
impl MyIndex for u64 {}

impl MyIndex for usize {}

/// Apply constraints on matrix `l` and right hand side column vector `rhs`.
/// Constraints are given by a list of indices `c_indices` and the values `c_values` :
/// in the result, the values at index `c_index[i]` will equal `c_value[i]`.
pub fn apply_constraints<I: Index>(
    l: &mut SparseColMat<I, f64>,
    mut rhs: ColMut<f64>,
    c_indices: &[usize],
    c_values: &[f64],
) {
    {
        let (sym, v) = l.parts_mut();
        let indices = sym.row_idx();
        indices
            .par_iter()
            .zip(v.par_iter_mut())
            .for_each(|(i, value)| {
                if c_indices.contains(&i.zx()) {
                    *value = 0.;
                }
            });
    }
    for (index, &c_value) in c_indices.iter().zip(c_values) {
        let (sym, v) = l.parts_mut();
        let indices = sym.row_idx_of_col_raw(*index);
        let range = sym.col_range(*index);
        for (value, i) in v[range].iter_mut().zip(indices) {
            if i.zx() != *index {
                rhs[i.zx()] -= *value * c_value;
                *value = 0.;
            } else {
                rhs[i.zx()] = c_value;
                *value = 1.;
            }
        }
    }
}

/// Same as [`apply_constraints`] but with a matrix for rhs and rows for constraints values.
pub fn apply_constraints_mat<I: Index>(
    l: &mut SparseColMat<I, f64>,
    mut rhs: MatMut<f64>,
    c_indices: &[I],
    c_values: &[Row<f64>],
) {
    let (sym, v) = l.parts_mut();
    let indices = sym.row_idx();
    indices
        .par_iter()
        .zip(v.par_iter_mut())
        .for_each(|(i, value)| {
            if c_indices.contains(&i) {
                *value = 0.;
            }
        });
    for (index, c_value) in c_indices.iter().zip(c_values) {
        let (sym, v) = l.parts_mut();
        let indices = sym.row_idx_of_col_raw(index.zx());
        let range = sym.col_range(index.zx());
        for (value, i) in v[range].iter_mut().zip(indices) {
            if *i != *index {
                rhs.rb_mut()
                    .row_mut(i.zx())
                    .iter_mut()
                    .zip(c_value.iter())
                    .for_each(|(v, c)| *v -= *value * c);
                *value = 0.;
            } else {
                rhs.rb_mut()
                    .row_mut(i.zx())
                    .iter_mut()
                    .zip(c_value.iter())
                    .for_each(|(v, c)| *v = *c);
                *value = 1.;
            }
        }
    }
}

/// Move the vertices from `src` by at most `delta_dst` while avoiding triangle flips and also minimizing the energy evaluted by `e`
pub fn flip_avoiding_line_search<I: Index, F: Fn(&[[f64; 2]]) -> f64>(
    src: &mut [[f64; 2]],
    delta_dst: &[[f64; 2]],
    f: &[[I; 3]],
    e: F,
) -> f64 {
    let mut t = 1.;
    let start_e = e(src);
    //TODO parallelize
    for face in f {
        let v1 = src[face[0].zx()];
        let v1p = delta_dst[face[0].zx()];
        let v2 = src[face[1].zx()];
        let v2p = delta_dst[face[1].zx()];
        let v3 = src[face[2].zx()];
        let v3p = delta_dst[face[2].zx()];
        let p1 = [v2[0] - v1[0], v2[1] - v1[1]];
        let p1p = [v2p[0] - v1p[0], v2p[1] - v1p[1]];
        let p2 = [v3[0] - v1[0], v3[1] - v1[1]];
        let p2p = [v3p[0] - v1p[0], v3p[1] - v1p[1]];
        let x1 = p1[0];
        let y1 = p1[1];
        let x1p = p1p[0];
        let y1p = p1p[1];
        let x2 = p2[0];
        let y2 = p2[1];
        let x2p = p2p[0];
        let y2p = p2p[1];
        let c = x1 * y2 - y1 * x2;
        let b = x1p * y2 + x1 * y2p - y1p * x2 - y1 * x2p;
        let a = x1p * y2p - y1p * x2p;
        let delta = b.powi(2) - 4. * a * c;
        if delta < 0. {
            continue;
        }
        let t1 = (-b - delta.sqrt()) / (2. * a);
        let t2 = (-b + delta.sqrt()) / (2. * a);
        if t1 > 0. && t1 < t {
            t = t1 * 0.9;
        }
        if t2 > 0. && t2 < t {
            t = t2 * 0.9;
        }
    }
    for _ in 0..10 {
        let mut temp_res = src.to_owned();
        for (src_row, t_row) in temp_res.iter_mut().zip(delta_dst) {
            src_row[0] += t * t_row[0];
            src_row[1] += t * t_row[1];
        }
        let cur_e = e(&temp_res);
        if cur_e < start_e {
            src.copy_from_slice(&temp_res);
            return cur_e;
        }
        t *= 0.5;
    }
    start_e
}

/// Build an edge map from faces `f` and total the number of vertices `nv`.
///
/// The result is given as a list of elements corresponding to each face, where each element has 3 indices values corresponding
/// to the adjacent face for the edge, and an number indicating the index in which the edge is also found in the opposite face
/// (negative values are used to indicate border edges).
///
/// The first value corresponds to the edge given by vertices `0` and `1`, the second by vertices `1` and `2` et the third `2` and `0`.
// Relies on atomics, however only load and stores are used, no memory lock should happen.
pub fn build_edge_map<
    I: Index
        + std::ops::Shl<usize, Output = I>
        + std::ops::Shr<usize, Output = I>
        + std::ops::BitAnd<Output = I>,
>(
    f: &[[I; 3]],
    nv: usize,
) -> Vec<([I; 3], [i8; 3])> {
    let arch = Arch::new();
    arch.dispatch(|| {
        let mut faces_deg = vec![0_u8; nv + 1];
        let mut faces_offsets = vec![0_u8; 3 * f.len()];
        for (row, off) in f.into_iter().zip(faces_offsets.chunks_exact_mut(3)) {
            if row[0] < row[1] {
                off[0] = faces_deg[row[0].zx()] as u8;
                faces_deg[row[0].zx()] += 1;
            } else {
                off[0] = faces_deg[row[1].zx()] as u8;
                faces_deg[row[1].zx()] += 1;
            }
            if row[1] < row[2] {
                off[1] = faces_deg[row[1].zx()] as u8;
                faces_deg[row[1].zx()] += 1;
            } else {
                off[1] = faces_deg[row[2].zx()] as u8;
                faces_deg[row[2].zx()] += 1;
            }
            if row[0] < row[2] {
                off[2] = faces_deg[row[0].zx()] as u8;
                faces_deg[row[0].zx()] += 1;
            } else {
                off[2] = faces_deg[row[2].zx()] as u8;
                faces_deg[row[2].zx()] += 1;
            }
        }

        let mut offset = I::truncate(0);
        let faces_deg: Vec<_> = faces_deg
            .into_iter()
            .map(|v| {
                let value = I::truncate(v as usize);
                offset += value;
                offset - value
            })
            .collect();

        let edges_to_faces: Vec<_> =
            std::iter::repeat_with(|| (AtomicU32::new(0), AtomicU32::new(0)))
                .take(offset.zx())
                .collect();
        let cur_offset = faces_deg;

        f.into_par_iter()
            .enumerate()
            .zip(faces_offsets.par_chunks_exact(3))
            .for_each(|((i, row), f_off)| {
                if row[0] < row[1] {
                    edges_to_faces[cur_offset[row[0].zx()].zx() + f_off[0] as usize]
                        .0
                        .store(
                            (row[1].zx() << 1) as u32,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                    edges_to_faces[cur_offset[row[0].zx()].zx() + f_off[0] as usize]
                        .1
                        .store((i << 1) as u32, std::sync::atomic::Ordering::Relaxed);
                } else {
                    edges_to_faces[cur_offset[row[1].zx()].zx() + f_off[0] as usize]
                        .0
                        .store(
                            (row[0].zx() << 1) as u32,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                    edges_to_faces[cur_offset[row[1].zx()].zx() + f_off[0] as usize]
                        .1
                        .store((i << 1) as u32, std::sync::atomic::Ordering::Relaxed);
                }
                if row[1] < row[2] {
                    edges_to_faces[cur_offset[row[1].zx()].zx() + f_off[1] as usize]
                        .0
                        .store(
                            (row[2].zx() << 1) as u32,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                    edges_to_faces[cur_offset[row[1].zx()].zx() + f_off[1] as usize]
                        .1
                        .store(((i << 1) + 1) as u32, std::sync::atomic::Ordering::Relaxed);
                } else {
                    edges_to_faces[cur_offset[row[2].zx()].zx() + f_off[1] as usize]
                        .0
                        .store(
                            (row[1].zx() << 1) as u32,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                    edges_to_faces[cur_offset[row[2].zx()].zx() + f_off[1] as usize]
                        .1
                        .store(((i << 1) + 1) as u32, std::sync::atomic::Ordering::Relaxed);
                }
                if row[0] < row[2] {
                    edges_to_faces[cur_offset[row[0].zx()].zx() + f_off[2] as usize]
                        .0
                        .store(
                            ((row[2].zx() << 1) + 1) as u32,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                    edges_to_faces[cur_offset[row[0].zx()].zx() + f_off[2] as usize]
                        .1
                        .store((i << 1) as u32, std::sync::atomic::Ordering::Relaxed);
                } else {
                    edges_to_faces[cur_offset[row[2].zx()].zx() + f_off[2] as usize]
                        .0
                        .store(
                            ((row[0].zx() << 1) + 1) as u32,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                    edges_to_faces[cur_offset[row[2].zx()].zx() + f_off[2] as usize]
                        .1
                        .store((i << 1) as u32, std::sync::atomic::Ordering::Relaxed);
                }
            });
        let edges: Vec<_> = vec![([0; 3], [-1; 3]); f.len()]
            .into_iter()
            .map(|(v1, v2)| {
                (
                    [
                        AtomicU32::from(v1[0]),
                        AtomicU32::from(v1[1]),
                        AtomicU32::from(v1[2]),
                    ],
                    [
                        AtomicI8::from(v2[0]),
                        AtomicI8::from(v2[1]),
                        AtomicI8::from(v2[2]),
                    ],
                )
            })
            .collect();

        let mut edges_to_faces: Vec<_> = edges_to_faces
            .into_iter()
            .map(|(e1, e2)| (e1.into_inner(), e2.into_inner()))
            .collect();

        rayon::iter::split(
            SubSlices {
                idx: &cur_offset,
                data: &mut edges_to_faces,
            },
            SubSlices::splitter,
        )
        .for_each(|s| {
            let r = s.idx[0];
            for offs in s.idx.windows(2) {
                let slice = &mut s.data[(offs[0] - r).zx()..(offs[1] - r).zx()];
                slice.sort_unstable_by_key(|item| item.0);
                let mut i = 0;
                while slice.len() > 0 && i < (slice.len() - 1) {
                    let item1 = slice[i];
                    let item2 = slice[i + 1];
                    if item1.0 >> 1 == item2.0 >> 1 {
                        let f1 = item1.1 >> 1;
                        let f2 = item2.1 >> 1;
                        let e_i_1 = 2 * (item1.0 & 1) + (item1.1 & 1);
                        let e_i_2 = 2 * (item2.0 & 1) + (item2.1 & 1);
                        unsafe {
                            edges.get_unchecked(f1.zx()).0[e_i_1.zx() as usize]
                                .store(f2.zx() as u32, std::sync::atomic::Ordering::Relaxed);
                            edges.get_unchecked(f1.zx()).1[e_i_1.zx() as usize]
                                .store(e_i_2.zx() as i8, std::sync::atomic::Ordering::Relaxed);
                            edges.get_unchecked(f2.zx()).0[e_i_2.zx() as usize]
                                .store(f1.zx() as u32, std::sync::atomic::Ordering::Relaxed);
                            edges.get_unchecked(f2.zx()).1[e_i_2.zx() as usize]
                                .store(e_i_1.zx() as i8, std::sync::atomic::Ordering::Relaxed);
                        }
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
            }
        });
        edges
            .into_iter()
            .map(|(e, e_i)| {
                let [e0, e1, e2] = e;
                let [e_i0, e_i1, e_i2] = e_i;
                (
                    [
                        I::truncate(e0.into_inner() as usize),
                        I::truncate(e1.into_inner() as usize),
                        I::truncate(e2.into_inner() as usize),
                    ],
                    [e_i0.into_inner(), e_i1.into_inner(), e_i2.into_inner()],
                )
            })
            .collect()
    })
}

struct SubSlices<'a, 'b, T, I: Index> {
    idx: &'a [I],
    data: &'b mut [T],
}

impl<'a, 'b, T, I: Index> SubSlices<'a, 'b, T, I> {
    fn splitter(self) -> (Self, Option<Self>) {
        if self.idx.len() <= 2 {
            return (self, None);
        }
        let mid = self.idx.len() / 2;
        let (idx_r, idx_l) = (&self.idx[0..mid + 1], &self.idx[mid..]);
        let (data_r, data_l) = self.data.split_at_mut(idx_l[0].zx() - idx_r[0].zx());
        (
            Self {
                idx: idx_r,
                data: data_r,
            },
            Some(Self {
                idx: idx_l,
                data: data_l,
            }),
        )
    }
}

fn get_next_boundary_vertex<I: Index>(
    index: &mut I,
    vertex_index: &mut u8,
    f: &[[I; 3]],
    e: &[([I; 3], [i8; 3])],
    marked: &mut [bool],
) -> I {
    if e[index.zx()].1[*vertex_index as usize] < 0 {
        *vertex_index = (*vertex_index + 1) % 3;
        f[index.zx()][*vertex_index as usize]
    } else {
        loop {
            let tmp_vert = (e[index.zx()].1[*vertex_index as usize] + 1) % 3;
            let tmp_f = e[index.zx()].0[*vertex_index as usize];
            *vertex_index = tmp_vert as u8;
            *index = tmp_f;
            marked[index.zx()] = true;
            if e[index.zx()].1[*vertex_index as usize] < 0 {
                *vertex_index = (*vertex_index + 1) % 3;
                return f[index.zx()][*vertex_index as usize];
            }
        }
    }
}

/// Find all ordered boundary loops from face indices `f` and edge map `e`.
pub fn get_boundary_loop<I: Index>(f: &[[I; 3]], e: &[([I; 3], [i8; 3])]) -> Vec<Vec<I>> {
    let mut res = Vec::new();
    let mut marked = vec![false; e.len()];
    for (i, (face, (_, adj))) in f.into_iter().zip(e.iter()).enumerate() {
        if !marked[i] && (adj[0] < 0 || adj[1] < 0 || adj[2] < 0) {
            marked[i] = true;
            let (start_vertex, mut cur_index) = if adj[0] < 0 {
                (face[0], 1)
            } else if adj[1] < 0 {
                (face[1], 2)
            } else {
                (face[2], 0)
            };
            let mut cur_face = I::truncate(i);
            let mut temp_res = vec![start_vertex, face[cur_index as usize]];
            loop {
                let new_vertex: I =
                    get_next_boundary_vertex(&mut cur_face, &mut cur_index, &f, &e, &mut marked);
                if new_vertex == start_vertex {
                    break;
                }
                temp_res.push(new_vertex);
            }
            res.push(temp_res);
        } else {
            marked[i] = true;
        }
    }
    res
}

/// Compute `len` evenly space points on a circle
pub fn map_to_circle(len: usize) -> Vec<[f64; 2]> {
    (0..len)
        .map(|i| {
            let angle = 2. * PI * i as f64 / len as f64;
            [angle.cos(), angle.sin()]
        })
        .collect()
}

/// Add length to all edge lengths if the triangle inequality isn't satisfied with enough margin
/// for stability. From <https://www.cs.cmu.edu/~kmcrane/Projects/NonmanifoldLaplace/NonmanifoldLaplace.pdf>
pub fn mollify(l: &mut [[f64; 3]], delta: f64) {
    let arch = Arch::new();
    arch.dispatch(|| {
        let max: f64 = l
            .par_iter()
            .map(|row| {
                let mut max: f64 = 0.;
                for i in 0..3 {
                    let l1 = row[i];
                    let l2 = row[(i + 1) % 3];
                    let l3 = row[(i + 2) % 3];
                    max = max.max(delta - l1 - l2 + l3);
                }
                max
            })
            .reduce_with(|a, b| a.max(b))
            .unwrap();
        let epsilon = max.max(0.);

        l.par_iter_mut().for_each(|l| {
            l[0] += epsilon;
            l[1] += epsilon;
            l[2] += epsilon;
        });
    })
}

/// Returns wether the triangle pair with shaired edge `l_s`, shaire by triangle `1` ( `l_s`,
/// `l11`, `l12`) and triangle `2` (`l_s`, `l21`, `l22`) is Delaunay.
pub fn is_delaunay(l_s: f64, l11: f64, l12: f64, l21: f64, l22: f64) -> bool {
    let tan_1 = f64::sqrt(
        ((l_s - l11 + l12) * (l_s - l12 + l11)) / ((l_s + l11 + l12) * (-l_s + l11 + l12)),
    );
    let tan_2 = f64::sqrt(
        ((l_s - l21 + l22) * (l_s - l22 + l21)) / ((l_s + l21 + l22) * (-l_s + l21 + l22)),
    );
    let cot_1 = (1. - tan_1 * tan_1) / (2. * tan_1);
    let cot_2 = (1. - tan_2 * tan_2) / (2. * tan_2);
    cot_1 + cot_2 >= 0.
}

/// Compute lengths, faces and edge map from the intrinsic Delaunay triangulations of the surface.
///
/// Arguments :
/// * `v` : vertices of the surface
/// * `f` : faces indices of the surface
/// * `e` : edge map of the surface, as computed by [`build_edge_map`]
pub fn build_intrinsic_delaunay<
    I: Index
        + std::ops::Shl<usize, Output = I>
        + std::ops::Shr<usize, Output = I>
        + std::ops::BitAnd<Output = I>,
>(
    v: &[[f64; 3]],
    mut f: Vec<[I; 3]>,
    mut e: Vec<([I; 3], [i8; 3])>,
) -> (Vec<[f64; 3]>, Vec<[I; 3]>, Vec<([I; 3], [i8; 3])>) {
    struct Face<I> {
        v: [I; 3],
        adj_f: [I; 3],
        adj_f_i: [i8; 3],
        l: [f64; 3],
    }

    let arch = Arch::new();
    let mut lengths = arch.dispatch(|| {
        let mut edges_t = Vec::new();
        f.par_iter()
            .map(|row| {
                let v1 = &v[row[0].zx()];
                let v2 = &v[row[1].zx()];
                let v3 = &v[row[2].zx()];
                [
                    v1[0] - v2[0],
                    v1[1] - v2[1],
                    v1[2] - v2[2],
                    v3[0] - v2[0],
                    v3[1] - v2[1],
                    v3[2] - v2[2],
                    v1[0] - v3[0],
                    v1[1] - v3[1],
                    v1[2] - v3[2],
                ]
            })
            .collect_into_vec(&mut edges_t);
        let edges_t = edges_t.into_flattened();
        pulp::as_arrays::<9, f64>(&edges_t)
            .0
            .into_iter()
            .map(|row| {
                [
                    f64::sqrt(row[0] * row[0] + row[1] * row[1] + row[2] * row[2]),
                    f64::sqrt(row[3] * row[3] + row[4] * row[4] + row[5] * row[5]),
                    f64::sqrt(row[6] * row[6] + row[7] * row[7] + row[8] * row[8]),
                ]
            })
            .collect::<Vec<_>>()
    });

    mollify(&mut lengths, 10e-8);

    let mut faces = Vec::new();
    f.par_iter()
        .zip(lengths.par_iter())
        .zip(e.par_iter())
        .map(|((f, l), e)| Face {
            v: [f[0], f[1], f[2]],
            adj_f: e.0,
            adj_f_i: e.1,
            l: *l,
        })
        .collect_into_vec(&mut faces);

    let mut edges_queue = VecDeque::with_capacity(faces.len());
    let mut edges_marked = vec![false; 3 * faces.len()];

    for (face_index, face) in faces.iter().enumerate() {
        for i in 0..3 {
            if face.adj_f_i[i] >= 0 && face_index < face.adj_f[i].zx() {
                edges_queue.push_back((I::truncate(face_index), i as u8));
                edges_marked[3 * face_index + i] = true;
            }
        }
    }

    while let Some((face_index, edge_index)) = edges_queue.pop_back() {
        edges_marked[3 * face_index.zx() + edge_index as usize] = false;
        let f1 = &faces[face_index.zx()];
        let opp_f = f1.adj_f[edge_index as usize];
        let opp_f_i = f1.adj_f_i[edge_index as usize];
        if opp_f_i >= 0 && face_index != opp_f {
            let e_i_1 = edge_index;
            let e_i_2 = if edge_index + 1 < 3 {
                edge_index + 1
            } else {
                edge_index - 2
            };
            let e_i_3 = if edge_index + 2 < 3 {
                edge_index + 2
            } else {
                edge_index - 1
            };
            let l_s = f1.l[e_i_1 as usize];
            let l11 = f1.l[e_i_2 as usize];
            let l12 = f1.l[e_i_3 as usize];
            let f2 = &faces[opp_f.zx()];
            let oe_i_1 = opp_f_i as usize;
            let oe_i_2 = if opp_f_i + 1 < 3 {
                oe_i_1 + 1
            } else {
                oe_i_1 - 2
            };
            let oe_i_3 = if opp_f_i + 2 < 3 {
                oe_i_1 + 2
            } else {
                oe_i_1 - 1
            };
            let l21 = f2.l[oe_i_2];
            let l22 = f2.l[oe_i_3];
            if !is_delaunay(l_s, l11, l12, l21, l22) {
                let f11 = f1.adj_f[e_i_2 as usize];
                let f12 = f1.adj_f[e_i_3 as usize];
                let f21 = f2.adj_f[oe_i_2 as usize];
                let f22 = f2.adj_f[oe_i_3 as usize];
                let i11 = f1.adj_f_i[e_i_2 as usize];
                let i12 = f1.adj_f_i[e_i_3 as usize];
                let i21 = f2.adj_f_i[oe_i_2 as usize];
                let i22 = f2.adj_f_i[oe_i_3 as usize];
                let v11 = f1.v[e_i_2 as usize];
                let v12 = f1.v[e_i_3 as usize];
                let v21 = f2.v[oe_i_2 as usize];
                let v22 = f2.v[oe_i_3 as usize];
                let tan_a_2 = f64::sqrt(
                    ((l12 - l11 + l_s) * (l12 - l_s + l11))
                        / ((l12 + l11 + l_s) * (l_s + l11 - l12)),
                );
                let tan_d_2 = f64::sqrt(
                    ((l21 - l22 + l_s) * (l21 - l_s + l22))
                        / ((l21 + l22 + l_s) * (l_s + l22 - l21)),
                );
                let tan_s = (tan_a_2 + tan_d_2) / (1. - tan_a_2 * tan_d_2);
                let cos = (1. - tan_s * tan_s) / (1. + tan_s * tan_s);
                let new_l = f64::sqrt(l11 * l11 + l22 * l22 - 2. * l11 * l22 * cos);
                let mut new_f_1 = match e_i_1 {
                    2 => Face {
                        v: [v11, v12, v22],
                        l: [l11, new_l, l22],
                        adj_f: [f11, opp_f, f22],
                        adj_f_i: [i11, oe_i_3 as i8, i22],
                    },
                    1 => Face {
                        v: [v12, v22, v11],
                        l: [new_l, l22, l11],
                        adj_f: [opp_f, f22, f11],
                        adj_f_i: [oe_i_3 as i8, i22, i11],
                    },
                    _ => Face {
                        v: [v22, v11, v12],
                        l: [l22, l11, new_l],
                        adj_f: [f22, f11, opp_f],
                        adj_f_i: [i22, i11, oe_i_3 as i8],
                    },
                };
                let mut new_f_2 = match oe_i_1 {
                    2 => Face {
                        v: [v21, v22, v12],
                        l: [l21, new_l, l12],
                        adj_f: [f21, face_index, f12],
                        adj_f_i: [i21, e_i_3 as i8, i12],
                    },
                    1 => Face {
                        v: [v22, v12, v21],
                        l: [new_l, l12, l21],
                        adj_f: [face_index, f12, f21],
                        adj_f_i: [e_i_3 as i8, i12, i21],
                    },
                    _ => Face {
                        v: [v12, v21, v22],
                        l: [l12, l21, new_l],
                        adj_f: [f12, f21, face_index],
                        adj_f_i: [i12, i21, e_i_3 as i8],
                    },
                };
                if f22 == face_index {
                    new_f_1.adj_f_i[e_i_1 as usize] = e_i_1 as i8;
                } else if f22 == opp_f {
                    new_f_1.adj_f_i[e_i_1 as usize] = e_i_1 as i8;
                    new_f_1.adj_f[e_i_1 as usize] = face_index;
                }
                if f11 == face_index {
                    new_f_1.adj_f_i[e_i_2 as usize] = e_i_2 as i8;
                } else if f11 == opp_f {
                    new_f_1.adj_f_i[e_i_2 as usize] = e_i_2 as i8;
                    new_f_1.adj_f[e_i_2 as usize] = face_index;
                }
                if f12 == opp_f {
                    new_f_2.adj_f_i[oe_i_1 as usize] = oe_i_1 as i8;
                } else if f12 == face_index {
                    new_f_2.adj_f_i[oe_i_1 as usize] = oe_i_1 as i8;
                    new_f_2.adj_f[oe_i_1 as usize] = opp_f;
                }
                if f21 == opp_f {
                    new_f_2.adj_f_i[oe_i_2 as usize] = oe_i_2 as i8;
                } else if f21 == face_index {
                    new_f_2.adj_f_i[oe_i_2 as usize] = oe_i_2 as i8;
                    new_f_2.adj_f[oe_i_2] = opp_f;
                }

                faces[face_index.zx()] = new_f_1;
                faces[opp_f.zx()] = new_f_2;

                if i11 >= 0 && f11 != face_index && f11 != opp_f {
                    let edge = if f11 < face_index {
                        (f11, i11 as u8)
                    } else {
                        (face_index, e_i_2)
                    };
                    if !edges_marked[3 * edge.0.zx() + edge.1 as usize] {
                        edges_marked[3 * edge.0.zx() + edge.1 as usize] = true;
                        edges_queue.push_back(edge);
                    }
                }
                if i22 >= 0 && f22 != face_index && f22 != opp_f {
                    faces[f22.zx()].adj_f[i22 as usize] = face_index;
                    faces[f22.zx()].adj_f_i[i22 as usize] = e_i_1 as i8;
                    let edge = if f22 < face_index {
                        (f22, i22 as u8)
                    } else {
                        (face_index, e_i_1)
                    };
                    if !edges_marked[3 * edge.0.zx() + edge.1 as usize] {
                        edges_marked[3 * edge.0.zx() + edge.1 as usize] = true;
                        edges_queue.push_back(edge);
                    }
                }
                if i12 >= 0 && f12 != face_index && f12 != opp_f {
                    faces[f12.zx()].adj_f[i12 as usize] = opp_f;
                    faces[f12.zx()].adj_f_i[i12 as usize] = oe_i_1 as i8;
                    let edge = if f12 < opp_f {
                        (f12, i12 as u8)
                    } else {
                        (opp_f, oe_i_1 as u8)
                    };
                    if !edges_marked[3 * edge.0.zx() + edge.1 as usize] {
                        edges_marked[3 * edge.0.zx() + edge.1 as usize] = true;
                        edges_queue.push_back(edge);
                    }
                }
                if i21 >= 0 && f21 != face_index && f21 != opp_f {
                    let edge = if f21 < opp_f {
                        (f21, i21 as u8)
                    } else {
                        (opp_f, oe_i_2 as u8)
                    };
                    if !edges_marked[3 * edge.0.zx() + edge.1 as usize] {
                        edges_marked[3 * edge.0.zx() + edge.1 as usize] = true;
                        edges_queue.push_back(edge);
                    }
                }
            }
        }
    }

    faces
        .into_par_iter()
        .zip(lengths.par_iter_mut())
        .zip(f.par_iter_mut())
        .zip(e.par_iter_mut())
        .for_each(|(((face, l), f), e)| {
            *l = face.l;
            f[0] = face.v[0];
            f[1] = face.v[1];
            f[2] = face.v[2];
            e.0 = face.adj_f;
            e.1 = face.adj_f_i;
        });
    (lengths, f, e)
}

/// Build mass and stiffness matrices for the intrinsic Laplacian, from the intrinsic Delaunay triangulation
/// (as precomputed by [`build_intrinsic_delaunay`]).
///
/// Slightly overkill for Tutte parameterization.
pub fn build_mass_cotan_laplacian_intrinsic<I: Index>(
    lengths: &[[f64; 3]],
    f: &[[I; 3]],
    e: &[([I; 3], [i8; 3])],
    nv: usize,
) -> (SparseColMat<I, f64>, SparseColMat<I, f64>) {
    let mut cots = Vec::new();
    lengths
        .par_iter()
        .map(|l| {
            let l1 = l[1];
            let l2 = l[2];
            let l3 = l[0];
            let l12 = l1 * l1;
            let l22 = l2 * l2;
            let l32 = l3 * l3;
            let s = (l1 + l2 + l3) / 2.;
            let a = f64::sqrt(s * (s - l1) * (s - l2) * (s - l3));
            let cotan1 = (l22 + l32 - l12) / (8. * a);
            let cotan2 = (l12 + l32 - l22) / (8. * a);
            let cotan3 = (l22 + l12 - l32) / (8. * a);
            [cotan1, cotan2, cotan3]
        })
        .collect_into_vec(&mut cots);

    let mut sum_cots = Vec::new();
    cots.par_iter()
        .zip(e.into_par_iter())
        .enumerate()
        .map(|(i, (cot, e))| {
            let mut cotan0 = cot[0];
            let mut cotan1 = cot[1];
            let mut cotan2 = cot[2];
            if e.1[1] >= 0 && e.0[1].zx() > i {
                cotan0 += cots[e.0[1].zx()][(e.1[1] as usize + 2) % 3];
            }
            if e.1[2] >= 0 && e.0[2].zx() > i {
                cotan1 += cots[e.0[2].zx()][(e.1[2] as usize + 2) % 3];
            }
            if e.1[0] >= 0 && e.0[0].zx() > i {
                cotan2 += cots[e.0[0].zx()][(e.1[0] as usize + 2) % 3];
            }
            [cotan0, cotan1, cotan2]
        })
        .collect_into_vec(&mut sum_cots);

    let mut deg = vec![0_u8; nv + 1];
    let mut faces_offsets = vec![0_u8; 6 * f.len()];
    for (i, ((face, edge), off)) in f
        .into_iter()
        .zip(e)
        .zip(faces_offsets.chunks_exact_mut(6))
        .enumerate()
    {
        if edge.1[0] < 0 || edge.0[0].zx() > i {
            deg[face[0].zx()] += 1;
            deg[face[1].zx()] += 1;
            off[0] = deg[face[0].zx()];
            off[1] = deg[face[1].zx()];
        }
        if edge.1[1] < 0 || edge.0[1].zx() > i {
            deg[face[1].zx()] += 1;
            deg[face[2].zx()] += 1;
            off[2] = deg[face[1].zx()];
            off[3] = deg[face[2].zx()];
        }
        if edge.1[2] < 0 || edge.0[2].zx() > i {
            deg[face[2].zx()] += 1;
            deg[face[0].zx()] += 1;
            off[4] = deg[face[2].zx()];
            off[5] = deg[face[0].zx()];
        }
    }
    let mut offset = 0_usize;
    let offsets: Vec<I> = deg
        .iter()
        .map(|deg| {
            offset += *deg as usize + 1;
            I::truncate(offset - (*deg + 1) as usize)
        })
        .collect();

    let indices: Vec<_> = vec![(0_u32, 0_u64); offset]
        .into_iter()
        .map(|(v1, v2)| (AtomicU32::from(v1 as u32), AtomicU64::from(v2)))
        .collect();

    f.into_par_iter()
        .zip(sum_cots.into_par_iter())
        .zip(faces_offsets.par_chunks_exact(6))
        .for_each(|((face, cots), off)| {
            let cotan0 = cots[0];
            let cotan1 = cots[1];
            let cotan2 = cots[2];
            if off[0] > 0 {
                indices[offsets[face[0].zx()].zx() + off[0] as usize]
                    .0
                    .store(face[1].zx() as u32, Ordering::Relaxed);
                indices[offsets[face[0].zx()].zx() + off[0] as usize]
                    .1
                    .store((-cotan2).to_bits(), Ordering::Relaxed);

                indices[offsets[face[1].zx()].zx() + off[1] as usize]
                    .0
                    .store(face[0].zx() as u32, Ordering::Relaxed);
                indices[offsets[face[1].zx()].zx() + off[1] as usize]
                    .1
                    .store((-cotan2).to_bits(), Ordering::Relaxed);
            }
            if off[2] > 0 {
                indices[offsets[face[1].zx()].zx() + off[2] as usize]
                    .0
                    .store(face[2].zx() as u32, Ordering::Relaxed);
                indices[offsets[face[1].zx()].zx() + off[2] as usize]
                    .1
                    .store((-cotan0).to_bits(), Ordering::Relaxed);

                indices[offsets[face[2].zx()].zx() + off[3] as usize]
                    .0
                    .store(face[1].zx() as u32, Ordering::Relaxed);
                indices[offsets[face[2].zx()].zx() + off[3] as usize]
                    .1
                    .store((-cotan0).to_bits(), Ordering::Relaxed);
            }
            if off[4] > 0 {
                indices[offsets[face[2].zx()].zx() + off[4] as usize]
                    .0
                    .store(face[0].zx() as u32, Ordering::Relaxed);
                indices[offsets[face[2].zx()].zx() + off[4] as usize]
                    .1
                    .store((-cotan1).to_bits(), Ordering::Relaxed);

                indices[offsets[face[0].zx()].zx() + off[5] as usize]
                    .0
                    .store(face[2].zx() as u32, Ordering::Relaxed);
                indices[offsets[face[0].zx()].zx() + off[5] as usize]
                    .1
                    .store((-cotan1).to_bits(), Ordering::Relaxed);
            }
        });

    offsets[..(offsets.len() - 1)]
        .par_iter()
        .enumerate()
        .for_each(|(i, off)| {
            indices[off.zx()].0.store(i as u32, Ordering::Relaxed);
        });
    let mut indices: Vec<_> = indices
        .into_iter()
        .map(|(i, v)| {
            (
                I::truncate(i.into_inner() as usize),
                f64::from_bits(v.into_inner()),
            )
        })
        .collect();

    rayon::iter::split(
        SubSlices {
            idx: &offsets,
            data: &mut indices,
        },
        SubSlices::splitter,
    )
    .for_each(|s| {
        let r = s.idx[0];
        for offs in s.idx.windows(2) {
            let slice = &mut s.data[(offs[0] - r).zx()..(offs[1] - r).zx()];
            slice[0].1 = -slice[1..].iter().fold(0., |acc, x| acc + x.1);
            slice.sort_unstable_by_key(|item| item.0);
        }
    });
    let (indices, coeffs): (Vec<_>, Vec<_>) = indices.into_iter().unzip();

    let mut coeffs_m = vec![0.; nv];
    for ((row, l), cot) in f.into_iter().zip(lengths.iter()).zip(cots) {
        let i1 = row[0];
        let i2 = row[1];
        let i3 = row[2];
        let l1 = l[1];
        let l2 = l[2];
        let l3 = l[0];
        let l12 = l1 * l1;
        let l22 = l2 * l2;
        let l32 = l3 * l3;
        let cotan1 = cot[0];
        let cotan2 = cot[1];
        let cotan3 = cot[2];
        coeffs_m[i1.zx()] += (l22 * cotan2 + l32 * cotan3) / 4.;
        coeffs_m[i2.zx()] += (l12 * cotan1 + l32 * cotan3) / 4.;
        coeffs_m[i3.zx()] += (l12 * cotan1 + l22 * cotan2) / 4.;
    }
    let triplets_m: Vec<_> = coeffs_m
        .into_iter()
        .enumerate()
        .map(|(i, value)| Triplet::new(I::truncate(i), I::truncate(i), value))
        .collect();

    let symbolic = unsafe { SymbolicSparseColMat::new_unchecked(nv, nv, offsets, None, indices) };
    (
        SparseColMat::<I, f64>::try_new_from_triplets(nv, nv, &triplets_m).unwrap(),
        SparseColMat::<I, f64>::new(symbolic, coeffs),
    )
}

/// Compute Tutte parameterization with border `b` mapped to a circle.
///
/// Uses intrinsic Laplacian, which is overkill
///
/// Arguments :
/// * `v` : surface vertices
/// * `f` : surface face indices
/// * `e` : surface edge map as obtained from [`build_edge_map`]
/// * `b` : choosen boundary to map on the circle (can be obtained from [`get_boundary_loop`])
pub fn compute_tutte_parameterization<I: MyIndex>(
    v: &[[f64; 3]],
    f: &[[I; 3]],
    e: Vec<([I; 3], [i8; 3])>,
    b: &[I],
) -> Vec<[f64; 2]> {
    let (l, f2, e) = build_intrinsic_delaunay(&v, f.to_owned(), e);
    let tot_area = l.iter().fold(0., |acc, row| {
        let l1 = row[0];
        let l2 = row[1];
        let l3 = row[2];
        let s = (l1 + l2 + l3) * 0.5;
        acc + (s * (s - l1) * (s - l2) * (s - l3)).sqrt()
    });

    let (_mass, mut lap) = build_mass_cotan_laplacian_intrinsic(&l, &f2, &e, v.len());
    let mut rhs = Mat::<f64>::zeros(v.len(), 2);

    let c = map_to_circle(b.len());
    let rows: Vec<_> = c
        .into_iter()
        .map(|row| Row::from_fn(2, |i| row[i] * tot_area.sqrt() / PI.sqrt()))
        .collect();

    apply_constraints_mat(&mut lap, rhs.as_mut(), &b, &rows);
    let llt = lap.sp_cholesky(faer::Side::Lower).unwrap();
    llt.solve_in_place(rhs.as_mut());

    rhs.row_iter()
        .map(|row| [row[0] as f64, row[1] as f64])
        .collect()
}
