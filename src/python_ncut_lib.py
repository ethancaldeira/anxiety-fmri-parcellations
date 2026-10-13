# python_ncut_lib.py
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
Normalised-cut spectral clustering and SVD-based discretisation
(Shi & Malik, 2000; Yu & Shi, 2003), as used by Craddock et al. (2012),
Human Brain Mapping 33, 1914-1928.

Both steps have a random initialisation: ``eigsh`` starts from a random
vector unless ``v0`` is given, and ``discretisation`` picks its first
rotation vector with NumPy's global random generator. ``seed_clustering``
fixes both for reproducible runs.
"""

import sys
import numpy as np
from scipy.sparse import csc_matrix, spdiags
from scipy.sparse.linalg import eigsh
from scipy.linalg import norm, svd, LinAlgError


class SVDError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


def seed_clustering(seed, n):
    """Seed NumPy's global random generator (used by ``discretisation``) and
    return a fixed start vector of length ``n`` for ``eigsh``.

    Returns None, and leaves the generator untouched, when ``seed`` is None.
    """
    if seed is None:
        return None
    np.random.seed(seed)
    return np.random.RandomState(seed).uniform(-1.0, 1.0, n)


def ncut(W, nbEigenValues, v0=None):
    """Leading ``nbEigenValues`` eigenvectors of the normalised affinity
    matrix of ``W``. ``v0`` is the optional ``eigsh`` start vector."""
    offset = 0.5
    maxiterations = 100
    eigsErrorTolerence = 1e-6
    eps = 2.2204e-16

    m = W.shape[1]

    d = np.abs(W).sum(0)
    dr = 0.5 * (d - W.sum(0))
    d = d + offset * 2
    dr = dr + offset

    W = W + spdiags(dr, [0], m, m, format="csc")
    Dinvsqrt = spdiags((1.0 / np.sqrt(d + eps)), [0], m, m, format="csc")
    P = Dinvsqrt @ (W @ Dinvsqrt)

    nbEigenValues = min(nbEigenValues, m - 1)
    eigen_val, eigen_vec = eigsh(P, nbEigenValues, maxiter=maxiterations,
                                  tol=eigsErrorTolerence, which='LA', v0=v0)

    i = np.argsort(-eigen_val)
    eigen_val = eigen_val[i]
    eigen_vec = eigen_vec[:, i]

    eigen_vec = Dinvsqrt @ eigen_vec
    norm_ones = norm(np.ones((m, 1)))

    for i in range(eigen_vec.shape[1]):
        eigen_vec[:, i] = (eigen_vec[:, i] / norm(eigen_vec[:, i])) * norm_ones
        if eigen_vec[0, i] != 0:
            eigen_vec[:, i] = -1 * eigen_vec[:, i] * np.sign(eigen_vec[0, i])

    return eigen_val, eigen_vec


def discretisation(eigen_vec):
    """Discretise continuous eigenvectors into a one-hot cluster assignment
    (n x k sparse matrix)."""
    eps = 2.2204e-16

    n, k = eigen_vec.shape
    vm = np.sqrt((eigen_vec ** 2).sum(1))[:, np.newaxis]
    eigen_vec = np.divide(eigen_vec, vm)

    svd_restarts = 0
    exitLoop = False

    while svd_restarts < 30 and not exitLoop:
        c = np.zeros((n, 1))
        R = np.zeros((k, k))
        R[:, 0] = eigen_vec[np.random.randint(0, n), :].T

        for j in range(1, k):
            c = c + np.abs(eigen_vec @ R[:, j - 1].reshape(-1, 1))
            R[:, j] = eigen_vec[np.argmin(c), :].T

        lastObjectiveValue = 0
        nbIterationsDiscretisation = 0
        nbIterationsDiscretisationMax = 20

        while True:
            nbIterationsDiscretisation += 1
            tDiscrete = eigen_vec @ R
            j = np.asarray(tDiscrete.argmax(axis=1)).reshape(-1)

            eigenvec_discrete = csc_matrix((np.ones(len(j)), (np.arange(n), j)), shape=(n, k))
            tSVD = eigenvec_discrete.T @ eigen_vec

            try:
                U, S, Vh = svd(tSVD)
            except LinAlgError:
                print("SVD did not converge, randomizing and trying again", file=sys.stderr)
                break

            NcutValue = 2 * (n - S.sum())
            if (abs(NcutValue - lastObjectiveValue) < eps or
                nbIterationsDiscretisation > nbIterationsDiscretisationMax):
                exitLoop = True
                break
            else:
                lastObjectiveValue = NcutValue
                R = Vh.T @ U.T

        svd_restarts += 1

    if not exitLoop:
        raise SVDError("SVD did not converge after 30 retries")

    return eigenvec_discrete
