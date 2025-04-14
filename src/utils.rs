use faer::sparse::SparseColMat;
use faer::{ColMut, Index};
use faer_traits::Index as tIndex;
use pulp::Arch;
use rayon::prelude::*;
use std::ops::{BitAnd, Mul, Shl, Shr};
use std::sync::atomic::{AtomicI8, AtomicU32};

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
