# group_mean_binfile_parcellation.py
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
Group-level normalised-cut clustering of the mean of the individual
connectivity graphs. Referred to as group-mean clustering in
Craddock et al. (2012), Human Brain Mapping 33, 1914-1928.
"""

import time
import numpy as np
from scipy.sparse import csc_matrix
from .python_ncut_lib import ncut, discretisation, seed_clustering


def group_mean_binfile_parcellate(infiles, outfile, K, n_voxels, seed=None):
    """
    Group-mean parcellation: averages individual connectivity matrices,
    then performs normalized cut clustering on the mean.

    Parameters:
        infiles:  list of .npy/.bin files containing sparse connectivity
                  matrices (3*N vectors: [i_indices, j_indices, weights])
        outfile:  output prefix (suffixed with _K.npy)
        K:        list of cluster counts to generate
        n_voxels: number of voxels in the brain mask
        seed:     seed for the random initialisation of the clustering
                  (eigsh start vector and discretisation); None leaves it random

    Writes ``<outfile>_<k>.npy`` for each k: n_voxels x 1 labels (1..k).
    """
    if not infiles or not outfile or not K or not n_voxels or K == 0 or n_voxels == 0:
        raise ValueError("group_mean_binfile_parcellate: infiles, outfile, K and "
                         "n_voxels must all be non-empty / non-zero")

    start = time.time()

    print(f"Reading {len(infiles)} connectivity graphs")
    for i in range(0, len(infiles)):
        if infiles[i].endswith(".npy"):
            a = np.load(infiles[i])
        else:
            with open(infiles[i], 'rb') as fileobj:
                a = np.fromfile(fileobj)

        n = len(a) // 3
        a = np.reshape(a, (3, n))

        if i == 0:
            W = csc_matrix((a[2, :], (a[0, :], a[1, :])), shape=(n_voxels, n_voxels))
        else:
            W = W + csc_matrix((a[2, :], (a[0, :], a[1, :])), shape=(n_voxels, n_voxels))

    W = W / len(infiles)

    print(f"Finished building the mean connectivity graph after {time.time() - start:.1f}s")

    Kmax = max(K)
    v0 = seed_clustering(seed, n_voxels)
    eigenval, eigenvec = ncut(W, Kmax, v0=v0)

    print(f"Finished computing eigenvectors after {time.time() - start:.1f}s")

    for k in K:
        eigk = eigenvec[:, :k]
        eigenvec_discrete = discretisation(eigk)

        group_img = eigenvec_discrete[:, 0]
        for i in range(1, k):
            group_img = group_img + (i + 1) * eigenvec_discrete[:, i]

        outname = outfile + '_' + str(k) + '.npy'
        np.save(outname, group_img.todense())

        print(f"Finished K = {k} after {time.time() - start:.1f}s")
