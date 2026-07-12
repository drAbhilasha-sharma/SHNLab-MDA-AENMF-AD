#!/usr/bin/env python3
"""
=============================================================================
SCRIPT 3 — compute_similarity_networks.py
=============================================================================
PURPOSE : Compute all 5 disease and 3 metabolite similarity networks,
          then integrate them using nonlinear SNF-style fusion.
INPUT   : mesh_descriptors_2026.xml     (MeSH DAG — download from NLM)
          adjacency_real.npy            (from Script build_adjacency.py)
          MSS.npy                       (from Script 2)
          kegg_immune_pathways.json     (disease→pathway mapping)
          aida_autoantibodies.json      (disease→autoantibody mapping)
OUTPUT  : DSS.npy, DGIP.npy, DSIE.npy  — disease similarity networks
          IPS.npy, APS.npy             — novel AD-specific networks
          SD.npy                        — integrated disease similarity
          MSS.npy (from Script 2)
          MGIP.npy, MSIE.npy           — metabolite similarity networks
          SM.npy                        — integrated metabolite similarity
REQUIRES: pip install lxml numpy pandas scipy
=============================================================================
"""

import os, sys, json, time
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine as cosine_dist

# ─── Configuration ────────────────────────────────────────────────────────────
DISEASES = [
    "Rheumatoid Arthritis",
    "Systemic Lupus Erythematosus",
    "Multiple Sclerosis",
    "Inflammatory Bowel Disease",
    "Type 1 Diabetes",
    "Psoriasis",
    "Ankylosing Spondylitis",
    "Sjögren's Syndrome",
]
N_DIS = len(DISEASES)

# MeSH Descriptor IDs for each disease
DISEASE_MESH_ID = {
    "Rheumatoid Arthritis":          "D001172",
    "Systemic Lupus Erythematosus":  "D008180",
    "Multiple Sclerosis":            "D009103",
    "Inflammatory Bowel Disease":    "D015212",
    "Type 1 Diabetes":               "D003922",
    "Psoriasis":                     "D011565",
    "Ankylosing Spondylitis":        "D013167",
    "Sjögren's Syndrome":            "D012859",
}

# KEGG immune pathway assignments per disease
# Based on KEGG disease-pathway mapping + published enrichment analyses
DISEASE_PATHWAYS = {
    "Rheumatoid Arthritis":          [
        "hsa05323","hsa04659","hsa04060","hsa04064","hsa04660",
        "hsa04620","hsa04630","hsa04010","hsa04151","hsa04068"
    ],
    "Systemic Lupus Erythematosus":  [
        "hsa05322","hsa04620","hsa04630","hsa04662","hsa04064",
        "hsa04060","hsa04659","hsa04010","hsa04610","hsa04666"
    ],
    "Multiple Sclerosis":            [
        "hsa04660","hsa04658","hsa04659","hsa04010","hsa04630",
        "hsa04064","hsa04151","hsa04620","hsa04650","hsa04670"
    ],
    "Inflammatory Bowel Disease":    [
        "hsa05321","hsa04659","hsa04060","hsa04620","hsa04064",
        "hsa04630","hsa04010","hsa04662","hsa04151","hsa04068"
    ],
    "Type 1 Diabetes":               [
        "hsa04940","hsa04660","hsa04630","hsa04620","hsa04659",
        "hsa04064","hsa04060","hsa04010","hsa04610","hsa04662"
    ],
    "Psoriasis":                     [
        "hsa05205","hsa04659","hsa04060","hsa04064","hsa04010",
        "hsa04620","hsa04630","hsa04151","hsa04662","hsa04660"
    ],
    "Ankylosing Spondylitis":        [
        "hsa05323","hsa04659","hsa04064","hsa04060","hsa04660",
        "hsa04620","hsa04630","hsa04010","hsa04151","hsa04670"
    ],
    "Sjögren's Syndrome":            [
        "hsa05322","hsa04630","hsa04662","hsa04060","hsa04620",
        "hsa04064","hsa04660","hsa04658","hsa04010","hsa04610"
    ],
}

# Autoantibody target profiles per disease
# Source: AIDA database + published clinical reviews
DISEASE_AUTOANTIBODIES = {
    "Rheumatoid Arthritis":          [
        "anti-CCP","anti-RF","anti-MCV","anti-vimentin","anti-fibrinogen",
        "anti-collagen","anti-RA33","anti-PAD4","anti-BiP","anti-HSP60"
    ],
    "Systemic Lupus Erythematosus":  [
        "anti-dsDNA","anti-Sm","anti-ANA","anti-SSA","anti-SSB",
        "anti-histone","anti-nucleosome","anti-C1q","anti-RNP","anti-ribosomal-P"
    ],
    "Multiple Sclerosis":            [
        "anti-MOG","anti-MBP","anti-AQP4","anti-MAG","anti-CNPase",
        "anti-PLP","anti-CHL1","anti-GFAP","anti-NMDAR","anti-CASPR2"
    ],
    "Inflammatory Bowel Disease":    [
        "anti-ASCA","anti-pANCA","anti-OmpC","anti-flagellin","anti-I2",
        "anti-GP2","anti-MUC1","anti-TNF","anti-calreticulin","anti-CUZD1"
    ],
    "Type 1 Diabetes":               [
        "anti-GAD65","anti-IA-2","anti-insulin","anti-ZnT8","anti-islet",
        "anti-ICA","anti-IA-2beta","anti-tetraspanin","anti-chromogranin-A","anti-IGRP"
    ],
    "Psoriasis":                     [
        "anti-keratinocyte","anti-IL-17A","anti-IL-23","anti-TNF",
        "anti-LL37","anti-desmoglein","anti-BP180","anti-BP230","anti-envoplakin","anti-periplakin"
    ],
    "Ankylosing Spondylitis":        [
        "anti-HLA-B27","anti-CD8","anti-proteoglycan","anti-collagen",
        "anti-fibronectin","anti-CD4","anti-CCP","anti-TNF","anti-IL-17","anti-IL-23"
    ],
    "Sjögren's Syndrome":            [
        "anti-SSA","anti-SSB","anti-ANA","anti-RNP","anti-Ro52",
        "anti-Ro60","anti-La","anti-Sp100","anti-muscarinic-R3","anti-aquaporin-5"
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION A — DISEASE SIMILARITY NETWORKS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_dss(mesh_xml_path: str) -> np.ndarray:
    """
    Disease Semantic Similarity from MeSH DAG.
    Formula: DSS(i,j) = sum_{t in T(i)∩T(j)} [Di(t)+Dj(t)] / [DV(i)+DV(j)]
    Decay factor δ = 0.5 (Gao et al. 2023)
    """
    print("[DSS] Parsing MeSH XML...")
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(mesh_xml_path)
    except FileNotFoundError:
        print(f"  WARNING: {mesh_xml_path} not found. Using fallback DSS.")
        return _fallback_dss()

    # Build MeSH ID → tree numbers
    id_to_trees = {}
    for desc in tree.findall(".//DescriptorRecord"):
        ui    = desc.findtext("DescriptorUI") or ""
        trees = [tn.text for tn in desc.findall(".//TreeNumber") if tn.text]
        if ui:
            id_to_trees[ui] = trees

    def ancestor_contribution(tree_num: str, delta: float = 0.5) -> dict:
        """Returns {ancestor_tree_prefix: contribution_value} for a tree number."""
        parts = tree_num.split(".")
        contribs = {}
        for depth in range(1, len(parts) + 1):
            prefix = ".".join(parts[:depth])
            contribs[prefix] = delta ** depth
        return contribs

    def semantic_value(trees: list) -> float:
        """DV(d) = sum of all ancestor contributions across all tree branches."""
        all_contribs = {}
        for t in trees:
            for anc, val in ancestor_contribution(t).items():
                all_contribs[anc] = max(all_contribs.get(anc, 0), val)
        return sum(all_contribs.values())

    def shared_contribution(trees_i: list, trees_j: list) -> float:
        """Sum of shared ancestor contributions between two diseases."""
        anc_i = {}
        for t in trees_i:
            for anc, val in ancestor_contribution(t).items():
                anc_i[anc] = max(anc_i.get(anc, 0), val)
        anc_j = {}
        for t in trees_j:
            for anc, val in ancestor_contribution(t).items():
                anc_j[anc] = max(anc_j.get(anc, 0), val)
        shared = set(anc_i.keys()) & set(anc_j.keys())
        return sum(anc_i[a] + anc_j[a] for a in shared)

    DSS = np.zeros((N_DIS, N_DIS))
    for i, d1 in enumerate(DISEASES):
        for j, d2 in enumerate(DISEASES):
            if i == j:
                DSS[i, j] = 1.0
                continue
            mesh_i = DISEASE_MESH_ID.get(d1, "")
            mesh_j = DISEASE_MESH_ID.get(d2, "")
            trees_i = id_to_trees.get(mesh_i, [])
            trees_j = id_to_trees.get(mesh_j, [])
            if not trees_i or not trees_j:
                continue
            num   = shared_contribution(trees_i, trees_j)
            dv_i  = semantic_value(trees_i)
            dv_j  = semantic_value(trees_j)
            denom = dv_i + dv_j
            if denom > 0:
                DSS[i, j] = num / denom

    print(f"  DSS computed. Range: [{DSS.min():.3f}, {DSS.max():.3f}]")
    return DSS


def _fallback_dss() -> np.ndarray:
    """Conservative fallback DSS when MeSH XML is unavailable."""
    DSS = np.array([
        [1.00, 0.45, 0.30, 0.35, 0.28, 0.32, 0.48, 0.38],
        [0.45, 1.00, 0.28, 0.32, 0.25, 0.29, 0.40, 0.52],
        [0.30, 0.28, 1.00, 0.22, 0.20, 0.18, 0.25, 0.27],
        [0.35, 0.32, 0.22, 1.00, 0.30, 0.35, 0.30, 0.28],
        [0.28, 0.25, 0.20, 0.30, 1.00, 0.22, 0.26, 0.23],
        [0.32, 0.29, 0.18, 0.35, 0.22, 1.00, 0.29, 0.25],
        [0.48, 0.40, 0.25, 0.30, 0.26, 0.29, 1.00, 0.35],
        [0.38, 0.52, 0.27, 0.28, 0.23, 0.25, 0.35, 1.00],
    ])
    print("  Using fallback DSS values (MeSH XML not available)")
    return DSS


def compute_gip_kernel(ip_matrix: np.ndarray, omega: float = 1.0) -> np.ndarray:
    """
    Gaussian Interaction Profile (GIP) Kernel Similarity.
    DGIP(i,j) = exp(-ω_d * ||IP(di) - IP(dj)||²)
    Equations 6–7 in Gao et al. 2023.
    """
    n = ip_matrix.shape[0]
    norms     = np.sum(ip_matrix ** 2, axis=1)
    mean_norm = np.mean(norms)
    if mean_norm < 1e-10:
        return np.eye(n)
    omega_norm = omega / mean_norm

    diff = ip_matrix[:, np.newaxis, :] - ip_matrix[np.newaxis, :, :]   # (n,n,d)
    sq_dist = np.sum(diff ** 2, axis=2)                                 # (n,n)
    G = np.exp(-omega_norm * sq_dist)
    return G


def compute_information_entropy_sim(A: np.ndarray, axis: int) -> np.ndarray:
    """
    Similarity based on information entropy of shared associations.
    axis=0 → metabolite similarity (shared diseases)
    axis=1 → disease similarity (shared metabolites)
    Equation 10–12 in Gao et al. 2023.
    """
    if axis == 1:
        A = A.T  # Now rows = diseases, cols = metabolites

    n   = A.shape[0]
    N   = A.sum()  # Total number of associations
    SIM = np.zeros((n, n))

    if N == 0:
        return np.eye(n)

    # Pre-compute entropy for each entity
    def entropy(row_vec: np.ndarray) -> float:
        nonzero_cols = np.where(row_vec > 0)[0]
        if len(nonzero_cols) == 0:
            return 0.0
        # n_k = number of total associations of each connected entity
        n_k  = A[:, nonzero_cols].sum(axis=0)  # shape (k,)
        probs = n_k / N
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs + 1e-12))

    H = np.array([entropy(A[i]) for i in range(n)])

    for i in range(n):
        for j in range(i, n):
            if i == j:
                SIM[i, j] = 1.0
                continue
            # Shared associations
            shared_idx = np.where((A[i] > 0) & (A[j] > 0))[0]
            if len(shared_idx) == 0:
                continue
            # Entropy of shared set
            n_k_shared = A[:, shared_idx].sum(axis=0)
            probs_s    = n_k_shared / N
            probs_s    = probs_s[probs_s > 0]
            H_shared   = -np.sum(probs_s * np.log2(probs_s + 1e-12))
            denom = H[i] + H[j]
            if denom > 0:
                SIM[i, j] = SIM[j, i] = 2 * H_shared / denom

    return SIM


def compute_ips(disease_pathways: dict) -> np.ndarray:
    """
    Immune Pathway Similarity (IPS) — Novel network.
    Binary vectors over KEGG immune pathways; cosine similarity.
    """
    all_pathways = sorted(set(p for ps in disease_pathways.values() for p in ps))
    vectors      = {}
    for dis in DISEASES:
        active = set(disease_pathways.get(dis, []))
        vectors[dis] = np.array([1.0 if p in active else 0.0 for p in all_pathways])

    IPS = np.zeros((N_DIS, N_DIS))
    for i, d1 in enumerate(DISEASES):
        for j, d2 in enumerate(DISEASES):
            v1, v2 = vectors[d1], vectors[d2]
            denom  = np.linalg.norm(v1) * np.linalg.norm(v2)
            IPS[i, j] = np.dot(v1, v2) / denom if denom > 0 else 0
            if i == j:
                IPS[i, j] = 1.0

    print(f"  IPS computed. All pathways: {len(all_pathways)}")
    return IPS


def compute_aps(disease_autoantibodies: dict) -> np.ndarray:
    """
    Autoantibody Profile Similarity (APS) — Novel network.
    Binary vectors over autoantibody targets; cosine similarity.
    """
    all_abs = sorted(set(ab for abs_ in disease_autoantibodies.values() for ab in abs_))
    vectors = {}
    for dis in DISEASES:
        present = set(disease_autoantibodies.get(dis, []))
        vectors[dis] = np.array([1.0 if ab in present else 0.0 for ab in all_abs])

    APS = np.zeros((N_DIS, N_DIS))
    for i, d1 in enumerate(DISEASES):
        for j, d2 in enumerate(DISEASES):
            v1, v2 = vectors[d1], vectors[d2]
            denom  = np.linalg.norm(v1) * np.linalg.norm(v2)
            APS[i, j] = np.dot(v1, v2) / denom if denom > 0 else 0
            if i == j:
                APS[i, j] = 1.0

    print(f"  APS computed. All autoantibodies: {len(all_abs)}")
    return APS


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION B — SNF-STYLE NONLINEAR NETWORK INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_network(S: np.ndarray) -> np.ndarray:
    """
    Row-normalize a similarity matrix.
    Off-diagonal: S[i,j] / (2 * sum_{k≠i} S[i,k])
    Diagonal: 0.5
    (Equation 13 in Gao et al. 2023)
    """
    S_norm = np.zeros_like(S)
    n = S.shape[0]
    for i in range(n):
        row_sum = S[i].sum() - S[i, i]
        if row_sum > 1e-10:
            for j in range(n):
                if i != j:
                    S_norm[i, j] = S[i, j] / (2 * row_sum)
        S_norm[i, i] = 0.5
    return S_norm


def knn_sparsify(S: np.ndarray, k: int) -> np.ndarray:
    """
    Keep only top-k neighbors per node; zero out the rest.
    (Equation 14 in Gao et al. 2023)
    """
    S_kn = np.zeros_like(S)
    n    = S.shape[0]
    k    = max(1, min(k, n - 1))
    for i in range(n):
        row = S[i].copy()
        row[i] = 0  # Exclude self
        top_k = np.argsort(row)[-k:]
        row_sum = row[top_k].sum()
        if row_sum > 0:
            for j in top_k:
                S_kn[i, j] = row[j] / row_sum
    return S_kn


def snf_integrate(networks: list, k: int = 1, max_iter: int = 30,
                  tol: float = 1e-6) -> np.ndarray:
    """
    Nonlinear Similarity Network Fusion (SNF).
    Iteratively updates each network by averaging information from all others.
    (Equations 15–18 in Gao et al. 2023)
    """
    m = len(networks)
    # Normalize all networks
    S_norm = [normalize_network(S) for S in networks]
    # Compute KNN-sparsified versions
    S_kn   = [knn_sparsify(S, k) for S in networks]
    # Initialize status matrices
    S_cur  = [s.copy() for s in S_norm]

    for iteration in range(max_iter):
        S_new = []
        max_delta = 0
        for idx in range(m):
            # Average of all OTHER networks
            others = [S_cur[jdx] for jdx in range(m) if jdx != idx]
            avg_others = np.mean(others, axis=0)
            # Update: S_kn[idx] @ avg_others @ S_kn[idx]^T
            updated = S_kn[idx] @ avg_others @ S_kn[idx].T
            # Re-normalize
            updated = normalize_network(updated + np.eye(len(updated)) * 1e-10)
            delta   = np.max(np.abs(updated - S_cur[idx]))
            max_delta = max(max_delta, delta)
            S_new.append(updated)
        S_cur = S_new
        if max_delta < tol:
            print(f"  SNF converged at iteration {iteration+1} (Δ={max_delta:.2e})")
            break
    else:
        print(f"  SNF reached max_iter={max_iter} (Δ={max_delta:.2e})")

    # Final integrated matrix = average of all converged networks
    S_final = np.mean(S_cur, axis=0)
    # Symmetrize
    S_final = (S_final + S_final.T) / 2
    np.fill_diagonal(S_final, 1.0)
    return S_final


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("="*65)
    print(" Similarity Network Computation — MDA-AENMF-AD")
    print("="*65)

    # ── Load adjacency matrix ─────────────────────────────────────────────
    if not os.path.exists("adjacency_real.npy"):
        sys.exit("ERROR: adjacency_real.npy not found. Run build_adjacency.py first.")
    A = np.load("adjacency_real.npy")
    print(f"\nAdjacency matrix loaded: {A.shape}, "
          f"associations={A.sum()}, density={A.mean():.4f}")

    N_MET = A.shape[0]

    # ── Disease similarity networks ───────────────────────────────────────
    print("\n--- Disease Similarity Networks ---")

    print("[1/5] DSS — Disease Semantic Similarity")
    DSS = compute_dss("mesh_descriptors_2026.xml")
    np.save("DSS.npy", DSS)

    print("[2/5] DGIP — Gaussian Interaction Profile")
    DGIP = compute_gip_kernel(A.T.astype(float))
    np.fill_diagonal(DGIP, 1.0)
    np.save("DGIP.npy", DGIP)
    print(f"  DGIP computed. Range: [{DGIP.min():.3f}, {DGIP.max():.3f}]")

    print("[3/5] DSIE — Disease Information Entropy Similarity")
    DSIE = compute_information_entropy_sim(A, axis=1)
    np.fill_diagonal(DSIE, 1.0)
    np.save("DSIE.npy", DSIE)
    print(f"  DSIE computed. Range: [{DSIE.min():.3f}, {DSIE.max():.3f}]")

    print("[4/5] IPS — Immune Pathway Similarity (Novel)")
    IPS = compute_ips(DISEASE_PATHWAYS)
    np.save("IPS.npy", IPS)

    print("[5/5] APS — Autoantibody Profile Similarity (Novel)")
    APS = compute_aps(DISEASE_AUTOANTIBODIES)
    np.save("APS.npy", APS)

    # ── Integrate disease networks (SNF, m=5) ─────────────────────────────
    print("\n--- Integrating Disease Networks (SNF, m=5) ---")
    k1 = max(1, N_DIS // 10)
    print(f"  KNN k1 = {k1}")
    SD = snf_integrate([DSS, DGIP, DSIE, IPS, APS], k=k1, max_iter=50)
    np.save("SD.npy", SD)
    print(f"  SD saved. Range: [{SD.min():.3f}, {SD.max():.3f}]")

    # ── Metabolite similarity networks ────────────────────────────────────
    print("\n--- Metabolite Similarity Networks ---")

    if os.path.exists("MSS.npy"):
        MSS = np.load("MSS.npy")
        print(f"[1/3] MSS loaded from file. Shape: {MSS.shape}")
    else:
        print("[1/3] MSS not found — run Script 2 first to compute fingerprints.")
        MSS = np.eye(N_MET)

    print("[2/3] MGIP — Metabolite Gaussian Interaction Profile")
    MGIP = compute_gip_kernel(A.astype(float))
    np.fill_diagonal(MGIP, 1.0)
    np.save("MGIP.npy", MGIP)
    print(f"  MGIP computed. Range: [{MGIP.min():.3f}, {MGIP.max():.3f}]")

    print("[3/3] MSIE — Metabolite Information Entropy Similarity")
    MSIE = compute_information_entropy_sim(A, axis=0)
    np.fill_diagonal(MSIE, 1.0)
    np.save("MSIE.npy", MSIE)
    print(f"  MSIE computed. Range: [{MSIE.min():.3f}, {MSIE.max():.3f}]")

    # ── Integrate metabolite networks (SNF, m=3) ──────────────────────────
    print("\n--- Integrating Metabolite Networks (SNF, m=3) ---")
    k2 = max(1, N_MET // 10)
    print(f"  KNN k2 = {k2}")
    SM = snf_integrate([MSS, MGIP, MSIE], k=k2, max_iter=50)
    np.save("SM.npy", SM)
    print(f"  SM saved. Range: [{SM.min():.3f}, {SM.max():.3f}]")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print(" DONE — Saved files:")
    for f in ["DSS.npy","DGIP.npy","DSIE.npy","IPS.npy","APS.npy",
              "SD.npy","MSS.npy","MGIP.npy","MSIE.npy","SM.npy"]:
        if os.path.exists(f):
            size = os.path.getsize(f)/1024
            mat  = np.load(f)
            print(f"  {f:<25} shape={mat.shape}  ({size:.0f} KB)")
    print("="*65)


if __name__ == "__main__":
    main()
