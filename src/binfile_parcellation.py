# binfile_parcellation.py
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
Subject-level normalised-cut clustering of a sparse connectivity graph
(Craddock et al., 2012, Human Brain Mapping 33, 1914-1928).
"""

import numpy as np
import time
from scipy.sparse import csc_matrix
from .python_ncut_lib import ncut, discretisation, seed_clustering


def binfile_parcellate(infile, outfile_prefix, K, seed=None):
    """
    Performs subject-level normalized cut clustering on a sparse connectivity matrix.

    Parameters:
        infile (str): Path to a .npy or .bin file containing a 3*N vector with [i, j, w_ij].
        outfile_prefix (str): Output filename prefix (e.g., "subj01_parc").
        K (int or list[int]): Number of clusters to compute. Can be a single int or list.
        seed (int or None): Seed for the random initialisation of the clustering
            (eigsh start vector and discretisation). None leaves it random.

    Writes ``<outfile_prefix>_<k>.npy`` for each k: 0-based uint16 labels, one per node.
    """
    start = time.time()
    print(f"Reading connectivity data from {infile}")

    if infile.endswith(".npy"):
        data = np.load(infile)
    else:
        with open(infile, 'rb') as f:
            data = np.fromfile(f)

    data = np.reshape(data, (3, -1))

    i_indices = data[0, :].astype(int)
    j_indices = data[1, :].astype(int)
    weights = data[2, :]

    m = max(i_indices.max(), j_indices.max()) + 1
    W = csc_matrix((weights, (i_indices, j_indices)), shape=(m, m))

    print(f"Constructed sparse matrix W with shape {W.shape} after {time.time() - start:.2f}s")

    if isinstance(K, int):
        K_list = [K]
    else:
        K_list = sorted(K)

    Kmax = max(K_list)

    v0 = seed_clustering(seed, m)

    print(f"Computing n-cut eigenvectors (max K = {Kmax})")
    eigenval, eigenvec = ncut(W, Kmax, v0=v0)
    print(f"Finished eigendecomposition in {time.time() - start:.2f}s")

    for k in K_list:
        print(f"Clustering into {k} parcels")
        eig_k = eigenvec[:, :k]
        discrete_vecs = discretisation(eig_k).toarray()

        cluster_labels = np.argmax(discrete_vecs, axis=1).astype(np.uint16)

        outpath = f"{outfile_prefix}_{k}.npy"
        np.save(outpath, cluster_labels)
        print(f"Saved: {outpath} (shape: {cluster_labels.shape})")

    print(f"Done. Total time: {time.time() - start:.2f}s")
