# make_local_connectivity_scorr.py
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
Spatially constrained functional connectivity via spatial correlation (scorr).

For each voxel, computes whole-brain functional connectivity maps for the voxel
and its 26 spatial neighbours, then correlates those FC maps. This means two
neighbouring voxels are considered similar if they connect to the rest of the
brain in the same way — a functionally meaningful similarity measure.

Every mask voxel is a node of the graph: voxels with a constant timeseries
(and any voxel left without an edge) keep only a self-loop of weight 1, so
the graph always has the full mask size.

Based on Craddock et al. (2012), Human Brain Mapping 33, 1914-1928.
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
        return idx[0] * np.prod(sz[1:3]) + idx[1] * sz[2] + idx[2]
    else:
        return idx[:, 0] * np.prod(sz[1:3]) + idx[:, 1] * sz[2] + idx[:, 2]


def make_local_connectivity_scorr(infile, maskfile, outfile, thresh):
    """
    Compute local connectivity using spatial correlation of whole-brain
    functional connectivity maps.

    For each voxel in the mask, this:
      1. Extracts time series for the voxel and its 26 neighbours
      2. Computes each voxel's whole-brain FC map (correlation with every
         other in-mask voxel)
      3. Correlates those FC maps (spatial correlation)
      4. Thresholds and stores the result as a sparse edge list

    Parameters:
        infile:   Path to 4D fMRI NIfTI file
        maskfile: Path to 3D binary brain mask NIfTI
        outfile:  Path to save sparse connectivity as .npy
        thresh:   Spatial correlation threshold (edges below this are zeroed)

    The output is a flat vector [i..., j..., w...] of length 3 * n_edges with
    node indices in mask order (0-based).
    """
    # 26-neighbourhood offsets
    neighbors = np.array([[x, y, z] for x in [-1, 0, 1]
                                     for y in [-1, 0, 1]
                                     for z in [-1, 0, 1]])

    # Load mask
    msk = nb.load(maskfile)
    msz = msk.shape
    mskdat = np.reshape(msk.get_fdata(), np.prod(msz))
    iv = np.nonzero(mskdat)[0]

    # Load fMRI data
    nim = nb.load(infile)
    sz = nim.shape
    print(f"{sz} dimensions of the 4D fMRI data")

    # Reshape to (n_voxels, n_timepoints) and mask
    imdat = np.reshape(nim.get_fdata(), (np.prod(sz[:3]), sz[3]))
    imdat = imdat[iv, :]
    imdat_sz = imdat.shape

    # Z-score time courses (makes correlation = dot product / (T-1))
    imdat_std = np.tile(np.std(imdat, axis=1), (imdat_sz[1], 1)).T
    imdat_std[imdat_std == 0] = 1e6  # avoid div by zero
    imdat_mean = np.tile(np.mean(imdat, axis=1), (imdat_sz[1], 1)).T
    imdat = (imdat - imdat_mean) / imdat_std

    # Zero out voxels that had no variance
    imdat[imdat_std == 1e6] = 0
    imdat[np.isnan(imdat)] = 0

    # Remove zero-variance voxels from the neighbourhood search and the FC maps.
    # Node indices in the output stay in full mask order.
    n_mask = len(iv)
    vndx = np.nonzero(np.var(imdat, axis=1) != 0)[0]
    iv = iv[vndx]
    imdat = imdat[vndx, :]

    m = len(iv)
    print(f"{m} # of non-zero variance voxels in the mask")

    # Lookup: global 1-D voxel index -> row of imdat + 1 (0 = not a graph voxel)
    msk_index = np.zeros(np.prod(msz), dtype=int)
    msk_index[iv] = np.arange(1, m + 1)

    sparse_i = []
    sparse_j = []
    sparse_w = []

    for i in range(m):
        if i % 1000 == 0:
            print(f"voxel # {i}")

        # Convert to 3D, find neighbours, convert back to 1D
        ndx3d = indx_1dto3d(iv[i], sz[:-1]) + neighbors
        ndx3d = np.clip(ndx3d, a_min=0, a_max=np.array(sz[:-1]) - 1)
        ndx1d = indx_3dto1d(ndx3d, sz[:-1]).astype(int)

        # Map to rows of imdat
        rows = msk_index[ndx1d]
        valid = rows > 0
        ndx1d = ndx1d[valid]
        rows = rows[valid] - 1

        # Find the seed voxel position within the neighbourhood
        nndx = np.nonzero(ndx1d == iv[i])[0]
        if len(nndx) == 0:
            continue

        # Extract time courses for neighbourhood voxels
        tc = imdat[rows, :]

        # Compute whole-brain FC maps for each neighbourhood voxel
        # fc[v, :] = correlation of voxel v with every in-mask voxel
        fc = np.dot(tc, imdat.T) / (sz[3] - 1)

        # Spatial correlation: how similar are these FC maps?
        R = np.corrcoef(fc)
        if R.ndim == 0:
            R = np.reshape(R, (1, 1))

        # Clean up
        R[np.isnan(R)] = 0
        R[R < thresh] = 0

        # Store edges from seed to all valid neighbours (full mask indices)
        nodes = vndx[rows]
        sparse_i.extend(nodes.tolist())
        sparse_j.extend([nodes[nndx[0]]] * len(nodes))
        sparse_w.extend(R[nndx[0], :].flatten().tolist())

    # Build output
    sparse_i = np.array(sparse_i, dtype=int)
    sparse_j = np.array(sparse_j, dtype=int)
    sparse_w = np.array(sparse_w, dtype=float)

    # Remove zero-weight edges
    nz = sparse_w > 0
    sparse_i = sparse_i[nz]
    sparse_j = sparse_j[nz]
    sparse_w = sparse_w[nz]

    # Self-loops for mask voxels without any edge, so that the graph has
    # one node per mask voxel
    has_edge = np.zeros(n_mask, dtype=bool)
    has_edge[sparse_i] = True
    has_edge[sparse_j] = True
    isolated = np.nonzero(~has_edge)[0]
    if len(isolated) > 0:
        print(f"{len(isolated)} mask voxels without edges get a self-loop")
        sparse_i = np.concatenate([sparse_i, isolated])
        sparse_j = np.concatenate([sparse_j, isolated])
        sparse_w = np.concatenate([sparse_w, np.ones(len(isolated))])

    outlist = np.concatenate([sparse_i, sparse_j, sparse_w])
    np.save(outfile, outlist)
    n_nodes = np.unique(np.concatenate([sparse_i, sparse_j])).size
    print(f"finished {infile}, output saved to {outfile}")
    print(f"Graph includes {n_nodes} / {n_mask} mask voxels as nodes")
