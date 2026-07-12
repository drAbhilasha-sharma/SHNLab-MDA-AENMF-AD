"""
Similarity Network Fusion (SNF)
================================
Fuses multiple disease similarity matrices into a single consensus network.
Implements Wang et al. (2014) Nature Methods.
"""

import numpy as np
from typing import List


def _knn_affinity(W: np.ndarray, k: int) -> np.ndarray:
    """Row-normalised kNN affinity matrix from similarity matrix W."""
    n = W.shape[0]
    P = np.zeros_like(W)
    for i in range(n):
        row   = W[i].copy(); row[i] = 0
        top_k = np.argsort(row)[::-1][:k]
        denom = row[top_k].sum()
        if denom > 0:
            P[i, top_k] = row[top_k] / denom
    return P


def _status_matrix(W: np.ndarray, k: int) -> np.ndarray:
    """Full kernel normalised by row-sum (for diffusion)."""
    n    = W.shape[0]
    D    = W.sum(axis=1, keepdims=True)
    D    = np.where(D == 0, 1, D)
    S    = W / D
    P    = _knn_affinity(W, k)
    return P, S


def snf_fuse(matrices: List[np.ndarray], k: int = 20,
             t: int = 20) -> np.ndarray:
    """
    Fuse a list of disease similarity matrices using SNF.

    Parameters
    ----------
    matrices : list of np.ndarray, each (n, n)
    k        : number of nearest neighbours
    t        : number of diffusion iterations

    Returns
    -------
    np.ndarray : (n, n) consensus similarity matrix
    """
    n  = matrices[0].shape[0]
    Ps = []
    Ss = []
    # Normalise and build affinity matrices
    for W in matrices:
        W_norm = (W + W.T) / 2
        np.fill_diagonal(W_norm, 0)
        row_max = W_norm.max(axis=1, keepdims=True)
        row_max = np.where(row_max == 0, 1, row_max)
        W_norm  = W_norm / row_max
        P, S    = _status_matrix(W_norm, k)
        Ps.append(P); Ss.append(S)

    # Iterative diffusion
    for _ in range(t):
        new_Ps = []
        for i, Pi in enumerate(Ps):
            agg = np.zeros((n, n))
            for j, Pj in enumerate(Ps):
                if j != i:
                    agg += Pj
            agg /= (len(Ps) - 1)
            new_Ps.append(Pi @ agg @ Pi.T)
        # Symmetrise
        Ps = [(P + P.T) / 2 for P in new_Ps]

    # Average final networks
    W_fused = np.mean(Ps, axis=0)
    # Normalise to [0,1]
    W_fused = (W_fused + W_fused.T) / 2
    np.fill_diagonal(W_fused, 1)
    mn, mx  = W_fused.min(), W_fused.max()
    if mx > mn:
        W_fused = (W_fused - mn) / (mx - mn)
    return W_fused
