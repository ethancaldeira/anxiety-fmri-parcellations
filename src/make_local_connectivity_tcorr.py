# make_local_connectivity_tcorr.py
#
# Adapted from pyClusterROI, Copyright (C) 2010 R. Cameron Craddock,
# https://github.com/ccraddock/cluster_roi; modified by Ethan Caldeira, 2026.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
Spatially constrained local connectivity via the temporal correlation (tcorr)
of each voxel with its 26 neighbours (Craddock et al., 2012, Human Brain
Mapping 33, 1914-1928).

Edges with r below the threshold are dropped. Every mask voxel is a node of
the graph: a voxel with no neighbour at or above the threshold (or with a
constant timeseries) keeps only a self-loop of weight 1.
"""

import nibabel as nb
import numpy as np


def indx_1dto3d(idx, sz):
    x = np.floor_divide(idx, np.prod(sz[1:3]))
    y = np.floor_divide(idx - x * np.prod(sz[1:3]), sz[2])
    z = idx - x * np.prod(sz[1:3]) - y * sz[2]
    return np.vstack((x, y, z)).T


def indx_3dto1d(idx, sz):
    idx = np.asarray(idx, dtype=int)
    if idx.ndim == 1:
        return idx[0]*np.prod(sz[1:3]) + idx[1]*sz[2] + idx[2]
    else:
        return idx[:, 0]*np.prod(sz[1:3]) + idx[:, 1]*sz[2] + idx[:, 2]


def make_local_connectivity_tcorr(infile, maskfile, outfile, thresh):
    """
    Compute local connectivity matrix using temporal correlation of
    neighboring voxels within a 26-neighborhood.

    Parameters:
        infile:   Path to 4D fMRI NIfTI file
        maskfile: Path to 3D binary brain mask NIfTI
        outfile:  Path to save sparse connectivity as .npy
        thresh:   Correlation threshold (edges below this are zeroed)

    The output is a flat vector [i..., j..., w...] of length 3 * n_edges with
    node indices in mask order (0-based).
    """
    neighbors = np.array([[x, y, z] for x in [-1, 0, 1]
                                     for y in [-1, 0, 1]
                                     for z in [-1, 0, 1]])

    msk = nb.load(maskfile)
    msz = np.shape(msk.get_fdata())
    mskdat = np.reshape(msk.get_fdata(), np.prod(msz))
    iv = np.nonzero(mskdat)[0]
    m = len(iv)
    print(m, '# of non-zero voxels in the mask')

    nim = nb.load(infile)
    sz = nim.shape
    imdat = np.reshape(nim.get_fdata(), (np.prod(sz[:3]), sz[3]))

    msk_index = np.zeros(np.prod(sz[:-1]), dtype=int)
    msk_index[iv] = np.arange(1, m + 1)

    sparse_i = []
    sparse_j = []
    sparse_w = []

    for i in range(m):
        if i % 1000 == 0:
            print('voxel #', i)

        seed_idx = iv[i]
        ndx3d = indx_1dto3d(seed_idx, sz[:-1]) + neighbors
        ndx3d = np.clip(ndx3d, a_min=0, a_max=np.array(sz[:-1]) - 1)
        ndx1d = indx_3dto1d(ndx3d, sz[:-1]).astype(int)

        ondx1d = msk_index[ndx1d]
        valid_mask = ondx1d > 0
        ndx1d = ndx1d[valid_mask]
        ondx1d = ondx1d[valid_mask]

        if seed_idx not in ndx1d:
            node_id = msk_index[seed_idx] - 1
            sparse_i.append(np.array([node_id]))
            sparse_j.append(np.array([node_id]))
            sparse_w.append(np.array([1.0]))
            continue

        nndx = np.where(ndx1d == seed_idx)[0]
        if len(nndx) == 0:
            node_id = msk_index[seed_idx] - 1
            sparse_i.append(np.array([node_id]))
            sparse_j.append(np.array([node_id]))
            sparse_w.append(np.array([1.0]))
            continue

        tc = imdat[ndx1d, :]

        if np.var(tc[nndx, :]) == 0:
            node_id = ondx1d[nndx[0]] - 1
            sparse_i.append(np.array([node_id]))
            sparse_j.append(np.array([node_id]))
            sparse_w.append(np.array([1.0]))
            continue

        R = np.corrcoef(tc)
        if R.ndim == 0:
            R = np.reshape(R, (1, 1))
        R = R[nndx[0], :].flatten()
        R = np.nan_to_num(R, nan=0.0)

        R[R < thresh] = 0
        nzndx = np.nonzero(R)[0]

        if len(nzndx) > 0:
            sparse_i.append(ondx1d[nzndx] - 1)
            sparse_j.append(np.full(len(nzndx), ondx1d[nndx[0]] - 1))
            sparse_w.append(R[nzndx])
        else:
            node_id = ondx1d[nndx[0]] - 1
            sparse_i.append(np.array([node_id]))
            sparse_j.append(np.array([node_id]))
            sparse_w.append(np.array([1.0]))

    if sparse_i:
        out_i = np.concatenate(sparse_i)
        out_j = np.concatenate(sparse_j)
        out_w = np.concatenate(sparse_w)
        outlist = np.concatenate([out_i, out_j, out_w])
        np.save(outfile, outlist)
        print(f'finished {infile}, output saved to {outfile}')
        n_nodes = np.unique(np.concatenate([out_i, out_j])).size
        print(f'Graph includes {n_nodes} / {m} mask voxels as nodes')
    else:
        print("No connections met the threshold. Output not saved.")
