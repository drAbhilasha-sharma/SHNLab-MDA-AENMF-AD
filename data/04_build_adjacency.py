#!/usr/bin/env python3
"""
=============================================================================
SCRIPT 4 — build_adjacency.py
=============================================================================
PURPOSE : Convert HMDB parsed associations CSV into the binary adjacency
          matrix A required by MDA-AENMF-AD. Drops fingerprint-failed items.
INPUT   : hmdb_autoimmune_associations.csv  (from Script 1)
          metabolite_fingerprints_metadata.csv (from Script 2 checkpoint)
OUTPUT  : adjacency_real.npy               (N_met × 8 binary matrix)
          metabolite_list_real.csv          (hmdb_id, metabolite, index)
          disease_list.csv                  (disease, index, n_metabolites)
          adjacency_stats.txt               (summary statistics)
REQUIRES: pip install pandas numpy
=============================================================================
"""

import os, sys
import numpy as np
import pandas as pd

# ─── Configuration ────────────────────────────────────────────────────────────
IN_ASSOC    = "hmdb_autoimmune_associations.csv"
OUT_ADJ     = "adjacency_real.npy"
OUT_MET     = "metabolite_list_real.csv"
OUT_DIS     = "disease_list.csv"
OUT_STATS   = "adjacency_stats.txt"

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

# Full disease name → variants that may appear in HMDB
DISEASE_ALIASES = {
    "Rheumatoid Arthritis": [
        "rheumatoid arthritis","arthritis rheumatoid","arthritis, rheumatoid",
    ],
    "Systemic Lupus Erythematosus": [
        "systemic lupus erythematosus","lupus erythematosus systemic",
        "lupus erythematosus, systemic","sle","lupus nephritis",
    ],
    "Multiple Sclerosis": [
        "multiple sclerosis","sclerosis multiple","sclerosis, multiple",
    ],
    "Inflammatory Bowel Disease": [
        "inflammatory bowel disease","crohn disease","crohn's disease",
        "crohns disease","ulcerative colitis","colitis ulcerative",
        "colitis, ulcerative","ibd",
    ],
    "Type 1 Diabetes": [
        "type 1 diabetes","diabetes mellitus type 1","diabetes mellitus, type 1",
        "insulin-dependent diabetes mellitus","iddm","juvenile diabetes",
        "diabetes, type 1",
    ],
    "Psoriasis": [
        "psoriasis","psoriasis vulgaris","plaque psoriasis",
    ],
    "Ankylosing Spondylitis": [
        "ankylosing spondylitis","spondylitis ankylosing","spondylitis, ankylosing",
        "bechterew disease","marie-struempell disease",
    ],
    "Sjögren's Syndrome": [
        "sjogren syndrome","sjögren syndrome","sjogren's syndrome",
        "sjögrens syndrome","sicca syndrome","sjogren-larsson syndrome",
    ],
}

# Build reverse lookup: alias → standard name
ALIAS_TO_STD = {}
for std, aliases in DISEASE_ALIASES.items():
    for alias in aliases:
        ALIAS_TO_STD[alias.lower()] = std


def standardize_disease(raw_name: str):
    low = raw_name.lower().strip()
    # Exact match
    if low in ALIAS_TO_STD:
        return ALIAS_TO_STD[low]
    # Substring match
    for alias, std in ALIAS_TO_STD.items():
        if alias in low:
            return std
    return None


def build_adjacency():
    print("="*65)
    print(" Build Adjacency Matrix — MDA-AENMF-AD")
    print("="*65)

    if not os.path.exists(IN_ASSOC):
        sys.exit(f"ERROR: {IN_ASSOC} not found. Run Script 1 (parse_hmdb.py) first.")

    # ── Load associations ─────────────────────────────────────────────────
    df = pd.read_csv(IN_ASSOC)
    print(f"\nLoaded {len(df):,} raw associations")

    # ── Standardize disease names ─────────────────────────────────────────
    if "disease_std" not in df.columns:
        df["disease_std"] = df["disease"].apply(standardize_disease)

    # Keep only recognized diseases
    df_filt = df.dropna(subset=["disease_std"]).copy()
    df_filt = df_filt[df_filt["disease_std"].isin(DISEASES)]

    # ── Filter out failed fingerprints to keep dimensions aligned ─────────
    if os.path.exists("metabolite_fingerprints_metadata.csv"):
        meta_fp = pd.read_csv("metabolite_fingerprints_metadata.csv")
        valid_ids = meta_fp[meta_fp["status"] == "ok"]["hmdb_id"].tolist()
        df_filt = df_filt[df_filt["hmdb_id"].isin(valid_ids)]
        print(f"After fingerprint status validation filtering: {len(df_filt):,} associations remain")
    else:
        print("WARNING: metabolite_fingerprints_metadata.csv not found. Running baseline setup.")

    if "hmdb_id" not in df_filt.columns:
        sys.exit("ERROR: 'hmdb_id' column missing from input file.")

    # ── Build metabolite index ────────────────────────────────────────────
    met_counts = df_filt.groupby("hmdb_id")["disease_std"].nunique().sort_values(ascending=False)
    metabolites = met_counts.index.tolist()
    
    met_idx = {m: i for i, m in enumerate(metabolites)}
    dis_idx = {d: j for j, d in enumerate(DISEASES)}

    N_MET = len(metabolites)
    N_DIS = len(DISEASES)
    print(f"Unique metabolites : {N_MET}")
    print(f"Target diseases    : {N_DIS}")

    # ── Build adjacency matrix ────────────────────────────────────────────
    A = np.zeros((N_MET, N_DIS), dtype=np.int8)
    for _, row in df_filt.iterrows():
        i = met_idx.get(row["hmdb_id"])
        j = dis_idx.get(row["disease_std"])
        if i is not None and j is not None:
            A[i, j] = 1

    print(f"\nAdjacency matrix   : {A.shape}")
    print(f"Known associations : {A.sum()}")
    print(f"Density            : {A.mean():.4f} ({A.mean()*100:.2f}%)")

    # ── Per-disease counts ────────────────────────────────────────────────
    print("\nAssociations per disease:")
    dis_stats = []
    for j, dis in enumerate(DISEASES):
        n_met = A[:, j].sum()
        print(f"  {dis:<45} {n_met:>4} metabolites")
        dis_stats.append({"disease": dis, "index": j, "n_metabolites": int(n_met)})

    # ── Save outputs ──────────────────────────────────────────────────────
    np.save(OUT_ADJ, A)
    print(f"\nSaved: {OUT_ADJ}")

    # Metabolite list with metadata
    met_meta = (df_filt[["hmdb_id", "metabolite", "smiles"]]
                .drop_duplicates("hmdb_id")
                .set_index("hmdb_id")
                .reindex(metabolites)
                .reset_index())
    met_meta.insert(0, "index", range(N_MET))
    met_meta["n_diseases"] = A.sum(axis=1)
    met_meta.to_csv(OUT_MET, index=False)
    print(f"Saved: {OUT_MET}")

    pd.DataFrame(dis_stats).to_csv(OUT_DIS, index=False)
    print(f"Saved: {OUT_DIS}")

    # Statistics file
    stats_lines = [
        "MDA-AENMF-AD Adjacency Matrix Statistics",
        "=" * 50,
        f"Source file   : {IN_ASSOC}",
        f"N metabolites : {N_MET}",
        f"N diseases    : {N_DIS}",
        f"Associations  : {int(A.sum())}",
        f"Density       : {A.mean():.4f}",
        "",
        "Per-disease metabolite counts:",
    ] + [f"  {DISEASES[j]}: {int(A[:,j].sum())}" for j in range(N_DIS)]

    with open(OUT_STATS, "w") as f:
        f.write("\n".join(stats_lines))
    print(f"Saved: {OUT_STATS}")

    print("\n" + "="*65)
    print(" DONE")
    print("="*65)
    return A, metabolites, DISEASES


if __name__ == "__main__":
    build_adjacency()
    