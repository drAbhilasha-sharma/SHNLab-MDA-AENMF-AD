"""
Disease Similarity Network Construction
=========================================
Builds IPS, APS, DGIP, DSS, DSIE matrices from raw data sources.
All matrices returned as np.ndarray (n_diseases, n_diseases), values in [0,1].
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from typing import List, Dict


# ── IPS: Immune Pathway Similarity ───────────────────────────────────────────
def build_ips(pathway_file: str, diseases: List[str]) -> np.ndarray:
    """
    Compute Jaccard similarity over shared KEGG immune pathway membership.

    Parameters
    ----------
    pathway_file : CSV with columns [disease, pathway_id, in_pathway (0/1)]
    diseases     : ordered list of disease names

    Returns
    -------
    np.ndarray (n_diseases, n_diseases)
    """
    df = pd.read_csv(pathway_file)
    n  = len(diseases)
    W  = np.zeros((n, n))

    for i, d1 in enumerate(diseases):
        set1 = set(df[df['disease'] == d1][df['in_pathway'] == 1]['pathway_id'])
        for j, d2 in enumerate(diseases):
            set2 = set(df[df['disease'] == d2][df['in_pathway'] == 1]['pathway_id'])
            if not set1 or not set2:
                continue
            W[i, j] = len(set1 & set2) / len(set1 | set2)

    np.fill_diagonal(W, 1.0)
    return W


# ── APS: Autoantibody Profile Similarity ──────────────────────────────────────
def build_aps(aida_file: str, diseases: List[str]) -> np.ndarray:
    """
    Compute cosine similarity over AIDA v2.0 autoantibody binary profiles.
    """
    df = pd.read_csv(aida_file)
    n  = len(diseases)
    W  = np.zeros((n, n))

    profiles: Dict[str, np.ndarray] = {}
    for d in diseases:
        row = df[df['disease'] == d]
        if len(row) == 0:
            profiles[d] = np.zeros(df.shape[1] - 1)
        else:
            profiles[d] = row.drop('disease', axis=1).values.flatten().astype(float)

    for i, d1 in enumerate(diseases):
        for j, d2 in enumerate(diseases):
            v1, v2 = profiles[d1], profiles[d2]
            if v1.sum() == 0 or v2.sum() == 0:
                W[i, j] = 0.0
            else:
                W[i, j] = 1 - cosine(v1, v2)

    np.fill_diagonal(W, 1.0)
    return W


# ── DGIP: Gaussian Interaction Profile Kernel ─────────────────────────────────
def build_dgip(assoc_matrix: np.ndarray) -> np.ndarray:
    """
    Compute Gaussian kernel over disease association profile vectors.

    Parameters
    ----------
    assoc_matrix : (n_diseases, n_metabolites) binary association matrix

    Returns
    -------
    np.ndarray (n_diseases, n_diseases)
    """
    n    = assoc_matrix.shape[0]
    diffs = []
    for i in range(n):
        for j in range(n):
            diffs.append(np.linalg.norm(assoc_matrix[i] - assoc_matrix[j]) ** 2)
    gamma = 1.0 / np.mean(diffs)

    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            diff   = np.linalg.norm(assoc_matrix[i] - assoc_matrix[j]) ** 2
            W[i, j] = np.exp(-gamma * diff)
    return W


# ── DSS: Disease Semantic Similarity (MeSH) ───────────────────────────────────
def build_dss(mesh_file: str, diseases: List[str]) -> np.ndarray:
    """
    Load precomputed MeSH Lin semantic similarity matrix.
    Expects CSV with diseases as row/column headers.
    """
    df = pd.read_csv(mesh_file, index_col=0)
    n  = len(diseases)
    W  = np.zeros((n, n))
    for i, d1 in enumerate(diseases):
        for j, d2 in enumerate(diseases):
            if d1 in df.index and d2 in df.columns:
                W[i, j] = float(df.loc[d1, d2])
    np.fill_diagonal(W, 1.0)
    return W


# ── DSIE: Disease Symptom Information Entropy ─────────────────────────────────
def build_dsie(symptom_file: str, diseases: List[str]) -> np.ndarray:
    """
    Compute Jensen-Shannon divergence-based similarity over symptom profiles.
    """
    from scipy.special import rel_entr
    df = pd.read_csv(symptom_file, index_col=0)
    n  = len(diseases)
    W  = np.zeros((n, n))

    def js_similarity(p: np.ndarray, q: np.ndarray) -> float:
        p = p / (p.sum() + 1e-12)
        q = q / (q.sum() + 1e-12)
        m = 0.5 * (p + q)
        jsd = 0.5 * rel_entr(p, m + 1e-15).sum() + \
              0.5 * rel_entr(q, m + 1e-15).sum()
        return float(np.exp(-jsd))  # convert divergence to similarity

    for i, d1 in enumerate(diseases):
        p = df.loc[d1].values if d1 in df.index else np.zeros(df.shape[1])
        for j, d2 in enumerate(diseases):
            q = df.loc[d2].values if d2 in df.index else np.zeros(df.shape[1])
            W[i, j] = js_similarity(p, q)

    np.fill_diagonal(W, 1.0)
    return W


# ── Build all five networks ────────────────────────────────────────────────────
def build_all_similarity_networks(config: dict, diseases: List[str],
                                   assoc_matrix: np.ndarray) -> dict:
    """
    Convenience wrapper — returns dict of all five similarity matrices.
    """
    return {
        'IPS':  build_ips( config['data']['kegg_path'],  diseases),
        'APS':  build_aps( config['data']['aida_path'],  diseases),
        'DGIP': build_dgip(assoc_matrix),
        'DSS':  build_dss( config['data']['mesh_path'],  diseases),
        'DSIE': build_dsie(config['data']['dsie_path'],  diseases),
    }
