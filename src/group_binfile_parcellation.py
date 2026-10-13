# group_binfile_parcellation.py
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
Group-level two-level normalised-cut clustering. Individual clusterings are
converted to coincidence matrices, averaged, and re-clustered. Referred to as
2-level clustering in Craddock et al. (2012), Human Brain Mapping 33, 1914-1928.
"""

import numpy as np
from scipy.sparse import csc_matrix, spdiags
from .python_ncut_lib import ncut, discretisation, seed_clustering


def group_binfile_parcellate(infiles, outfile, K, n_voxels, seed=None):
    """
    2-level group parcellation: builds coincidence matrices from individual
    cluster assignments, averages them, then performs normalized cut clustering.

    Parameters:
        infiles:  list of .npy/.bin files containing per-subject cluster labels
                  (1D vectors where value = cluster assignment 1..K for each voxel)
        outfile:  output path for group parcellation (.npy)
        K:        number of clusters (must match the K used for individual clustering)
        n_voxels: number of voxels in the brain mask
        seed:     seed for the random initialisation of the clustering
                  (eigsh start vector and discretisation); None leaves it random

    Returns:
        group_img: n_voxels x 1 array of group cluster assignments (1..K)
    """
    print(f"Reading {len(infiles)} individual parcellations")
    for i in range(0, len(infiles)):
        if infiles[i].endswith(".npy"):
            imdat = np.load(infiles[i])
        else:
            with open(infiles[i], 'rb') as fid:
                imdat = np.fromfile(fid)

        sparse_i = []
        sparse_j = []
        for j in range(1, K + 1):
            grp_ndx = np.nonzero(imdat == j)[0]
            ff = np.tile(grp_ndx, (len(grp_ndx), 1))
            sparse_i = np.append(sparse_i, np.reshape(ff, np.prod(np.shape(ff))))
            sparse_j = np.append(sparse_j, np.reshape(ff.transpose(), np.prod(np.shape(ff))))

        if i == 0:
            W = csc_matrix((np.ones(len(sparse_i)), (sparse_i, sparse_j)),
                           (n_voxels, n_voxels), dtype=float)
        else:
            W = W + csc_matrix((np.ones(len(sparse_i)), (sparse_i, sparse_j)),
                               (n_voxels, n_voxels), dtype=float)

    W = W - spdiags(W.diagonal(), [0], n_voxels, n_voxels, "csc")

    W = W / len(infiles)
    print("Finished building the mean coincidence matrix")

    v0 = seed_clustering(seed, n_voxels)
    eigenval, eigenvec = ncut(W, K, v0=v0)
    eigenvec_discrete = discretisation(eigenvec)

    group_img = eigenvec_discrete[:, 0]

    for i in range(1, K):
        group_img = group_img + (i + 1) * eigenvec_discrete[:, i]

    np.save(outfile, group_img.todense())
    print("Finished group parcellation")

    return np.array(group_img.todense())
