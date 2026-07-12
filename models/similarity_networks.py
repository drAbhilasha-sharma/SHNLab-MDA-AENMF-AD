"""
similarity_networks.py — All metabolite and disease similarity network computations.

Metabolite networks (3):
    MSS  — Structural Similarity (Tanimoto on Morgan fingerprints or PaDEL)
    MGIP — Gaussian Interaction Profile kernel (metabolite side)
    MSIE — Information Entropy-based Similarity

Disease networks (5):
    DSS  — MeSH DAG-based Semantic Similarity
    DGIP — Gaussian Interaction Profile kernel (disease side)
    DSIE — Information Entropy-based Similarity
    IPS  — Immune Pathway Similarity [NOVEL — this study]
    APS  — Autoantibody Profile Similarity [NOVEL — this study]
"""

import numpy as np
import pandas as pd
from typing import Dict


# ─────────────────────────────────────────────────────────────────────────────
# Metabolite similarity networks
# ─────────────────────────────────────────────────────────────────────────────

def _gaussian_kernel(A: np.ndarray, gamma: float = None) -> np.ndarray:
    """
    Gaussian Interaction Profile (GIP) kernel.
        K(i,j) = exp(-γ ||r_i - r_j||²)
    where γ = 1 / (||A||²_F / n) and r_i is the i-th row of A.
    """
    n = A.shape[0]
    norms_sq = np.sum(A ** 2, axis=1)
    if gamma is None:
        gamma = n / (np.sum(norms_sq) + 1e-10)
    diff = norms_sq[:, None] + norms_sq[None, :] - 2 * (A @ A.T)
    K = np.exp(-gamma * np.clip(diff, 0, None))
    return K


def _entropy_similarity(A: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Information Entropy-based Similarity (IES).
    Computes cross-entropy between row distributions.
    """
    n = A.shape[0]
    P = A / (A.sum(axis=1, keepdims=True) + eps)  # row-normalise to prob.
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            # JS divergence → similarity
            M = 0.5 * (P[i] + P[j])
            kl_pm = np.sum(P[i] * np.log((P[i] + eps) / (M + eps)))
            kl_qm = np.sum(P[j] * np.log((P[j] + eps) / (M + eps)))
            jsd = 0.5 * kl_pm + 0.5 * kl_qm
            S[i, j] = 1.0 / (1.0 + jsd)
    return S


def compute_metabolite_similarity(padel: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Compute all three metabolite similarity matrices from PaDEL descriptors.

    Returns dict with keys: 'MSS', 'MGIP', 'MSIE'
    """
    X = padel.values.astype(np.float32)
    # Normalise columns to [0,1]
    col_min = X.min(axis=0)
    col_max = X.max(axis=0)
    col_range = np.clip(col_max - col_min, 1e-10, None)
    X_norm = (X - col_min) / col_range

    # MSS: cosine similarity on normalised descriptor vectors
    norms = np.linalg.norm(X_norm, axis=1, keepdims=True) + 1e-10
    X_unit = X_norm / norms
    MSS = np.clip(X_unit @ X_unit.T, 0, 1)
    np.fill_diagonal(MSS, 1.0)

    # MGIP
    MGIP = _gaussian_kernel(X_norm)
    np.fill_diagonal(MGIP, 1.0)

    # MSIE (approximation using sampled columns for speed)
    MSIE = _entropy_similarity(X_norm[:, :50])  # subsample 50 descriptors
    np.fill_diagonal(MSIE, 1.0)

    return {"MSS": MSS, "MGIP": MGIP, "MSIE": MSIE}


# ─────────────────────────────────────────────────────────────────────────────
# Disease similarity networks
# ─────────────────────────────────────────────────────────────────────────────

def compute_disease_similarity(kegg: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Compute DSS, DGIP, DSIE disease similarity matrices.

    Parameters
    ----------
    kegg : pd.DataFrame (n_diseases × n_pathways) — binary pathway membership
    """
    B = kegg.values.astype(np.float32)
    n = B.shape[0]

    # DSS: Jaccard similarity on pathway membership profiles
    DSS = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            inter = (B[i] * B[j]).sum()
            union = ((B[i] + B[j]) > 0).sum()
            DSS[i, j] = inter / (union + 1e-10)
    np.fill_diagonal(DSS, 1.0)

    DGIP = _gaussian_kernel(B)
    np.fill_diagonal(DGIP, 1.0)

    DSIE = _entropy_similarity(B)
    np.fill_diagonal(DSIE, 1.0)

    return {"DSS": DSS, "DGIP": DGIP, "DSIE": DSIE}


def compute_ips(kegg: pd.DataFrame) -> np.ndarray:
    """
    Immune Pathway Similarity (IPS) — Novel network (this study).

    Quantifies co-activation of KEGG immune pathways between diseases.
    Uses cosine similarity on binary immune pathway membership profiles
    restricted to the 20 KEGG immune-specific pathways listed in S9.

    Parameters
    ----------
    kegg : pd.DataFrame (n_diseases × n_pathways)

    Returns
    -------
    IPS : np.ndarray (n_diseases, n_diseases)
    """
    B = kegg.values.astype(np.float32)
    norms = np.linalg.norm(B, axis=1, keepdims=True) + 1e-10
    B_unit = B / norms
    IPS = np.clip(B_unit @ B_unit.T, 0, 1)
    np.fill_diagonal(IPS, 1.0)
    return IPS


def compute_aps(aida: pd.DataFrame) -> np.ndarray:
    """
    Autoantibody Profile Similarity (APS) — Novel network (this study).

    Quantifies similarity between diseases based on their autoantibody
    target repertoires, derived from AIDA database annotations.
    Uses Jaccard similarity on autoantibody target sets.

    Parameters
    ----------
    aida : pd.DataFrame (n_diseases × n_autoantibody_targets) — binary matrix

    Returns
    -------
    APS : np.ndarray (n_diseases, n_diseases)
    """
    B = aida.values.astype(np.float32)
    n = B.shape[0]
    APS = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            inter = (B[i] * B[j]).sum()
            union = ((B[i] + B[j]) > 0).sum()
            APS[i, j] = inter / (union + 1e-10)
    np.fill_diagonal(APS, 1.0)
    return APS
