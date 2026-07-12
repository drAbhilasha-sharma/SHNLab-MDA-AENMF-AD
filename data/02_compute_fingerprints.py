#!/usr/bin/env python3
"""
=============================================================================
SCRIPT 2 — compute_fingerprints.py
=============================================================================
PURPOSE : Compute 2209-dimensional molecular descriptor + fingerprint vectors
          for all metabolites using PaDELPy (wraps PaDEL-Descriptor v2.21).
          Produces the Metabolite Structural Similarity (MSS) matrix.
INPUT   : hmdb_metabolite_smiles.csv  (from Script 1)
OUTPUT  : metabolite_fingerprints_raw.csv      (N × 2209 + ID columns)
          metabolite_fingerprints_normalized.npy (N × D, Z-score normalized)
          metabolite_fingerprints_metadata.csv  (hmdb_id, metabolite, status)
          MSS.npy                               (N × N cosine similarity)
RUNTIME : ~2–4 sec per metabolite. 300 metabolites ≈ 15–20 min total.
REQUIRES: pip install padelpy pandas numpy scikit-learn tqdm
=============================================================================
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from padelpy import from_smiles
except ImportError:
    sys.exit("Install padelpy:  pip install padelpy")

# ─── Configuration ────────────────────────────────────────────────────────────
IN_SMILES   = "hmdb_metabolite_smiles.csv"
OUT_RAW     = "metabolite_fingerprints_raw.csv"
OUT_NORM    = "metabolite_fingerprints_normalized.npy"
OUT_META    = "metabolite_fingerprints_metadata.csv"
OUT_MSS     = "MSS.npy"
CACHE_FILE  = "fingerprint_cache.json"   # Resume interrupted runs
TIMEOUT_SEC = 90                          # Per-molecule timeout

# ─── Load cache (enables resuming interrupted runs) ───────────────────────────
def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)

# ─── Compute fingerprints ─────────────────────────────────────────────────────
def compute_fingerprints(smiles_df: pd.DataFrame):
    cache    = load_cache()
    results  = []
    failed   = []
    metadata = []

    print(f"Computing fingerprints for {len(smiles_df)} metabolites")
    print(f"Cache has {len(cache)} pre-computed entries")
    print(f"Estimated remaining time: "
          f"~{(len(smiles_df)-len(cache))*3/60:.0f} min\n")

    for _, row in tqdm(smiles_df.iterrows(), total=len(smiles_df), desc="PaDEL"):
        hmdb_id   = row["hmdb_id"]
        met_name  = row["metabolite"]
        smiles    = row["smiles"]

        # Use cache if available
        if hmdb_id in cache:
            desc = cache[hmdb_id]
            desc["hmdb_id"]    = hmdb_id
            desc["metabolite"] = met_name
            results.append(desc)
            metadata.append({"hmdb_id": hmdb_id, "metabolite": met_name,
                              "smiles": smiles, "status": "cached"})
            continue

        # Compute descriptors
        try:
            desc = from_smiles(
                smiles,
                fingerprints=True,    # PubChem fingerprints (881 bits)
                descriptors=True,     # 1D+2D descriptors (1328 features)
                timeout=TIMEOUT_SEC,
            )
            # Validate: should have ~2209 keys
            if len(desc) < 100:
                raise ValueError(f"Too few descriptors returned: {len(desc)}")

            # Store in cache (without ID columns)
            cache[hmdb_id] = {k: v for k, v in desc.items()}
            save_cache(cache)

            desc["hmdb_id"]    = hmdb_id
            desc["metabolite"] = met_name
            results.append(desc)
            metadata.append({"hmdb_id": hmdb_id, "metabolite": met_name,
                              "smiles": smiles, "status": "ok"})

        except Exception as e:
            tqdm.write(f"  FAILED: {met_name} ({hmdb_id}) — {e}")
            failed.append({"hmdb_id": hmdb_id, "metabolite": met_name,
                           "error": str(e)})
            metadata.append({"hmdb_id": hmdb_id, "metabolite": met_name,
                              "smiles": smiles, "status": f"failed: {e}"})

        time.sleep(0.05)  # Brief pause to prevent CPU overload

    print(f"\nComputed: {len(results)}, Failed: {len(failed)}")
    if failed:
        print("Failed metabolites:")
        for f in failed[:10]:
            print(f"  {f['hmdb_id']} — {f['metabolite']}: {f['error']}")
        if len(failed) > 10:
            print(f"  ... and {len(failed)-10} more")

    return results, failed, metadata


# ─── Post-processing ──────────────────────────────────────────────────────────
def process_and_normalize(results, metadata):
    raw_df = pd.DataFrame(results)
    meta_df = pd.DataFrame(metadata)

    # Identify descriptor columns (not ID columns)
    id_cols   = ["hmdb_id", "metabolite"]
    feat_cols = [c for c in raw_df.columns if c not in id_cols]

    print(f"\nRaw feature matrix: {len(raw_df)} metabolites × {len(feat_cols)} features")

    # Convert to float, replace NaN/inf with 0
    X = raw_df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Remove constant columns (zero variance — uninformative)
    col_std  = X.std(axis=0)
    keep_idx = col_std > 1e-9
    X_filt   = X[:, keep_idx]
    kept_cols = [feat_cols[i] for i, k in enumerate(keep_idx) if k]
    print(f"After removing constant columns: {X_filt.shape[1]} features remain")

    # Z-score normalization (as in MDA-AENMF paper, Eq. 4)
    col_mean = X_filt.mean(axis=0)
    col_std2 = X_filt.std(axis=0) + 1e-9
    X_norm   = (X_filt - col_mean) / col_std2
    print(f"Z-score normalized. Shape: {X_norm.shape}")

    # Save normalized matrix
    np.save(OUT_NORM, X_norm)
    print(f"Saved: {OUT_NORM}")

    # Save raw CSV (with headers for inspection)
    raw_df[id_cols + feat_cols].to_csv(OUT_RAW, index=False)
    print(f"Saved: {OUT_RAW}")

    # Save metadata
    meta_df.to_csv(OUT_META, index=False)
    print(f"Saved: {OUT_META}")

    return X_norm, kept_cols, raw_df[id_cols]


# ─── Compute MSS (Metabolite Structural Similarity) ───────────────────────────
def compute_mss(X_norm: np.ndarray) -> np.ndarray:
    """
    Metabolite Structural Similarity using cosine similarity of
    normalized 2209-dim descriptor vectors (Eq. 5 in Gao et al. 2023).

    MSS(i,j) = (sum_k x_ik * x_jk) /
               sqrt(sum_k x_ik^2 * sum_k x_jk^2)
    """
    print("\nComputing MSS cosine similarity matrix...")
    n = X_norm.shape[0]

    # Compute norms
    norms = np.linalg.norm(X_norm, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9  # Avoid division by zero
    X_unit = X_norm / norms

    # Cosine similarity = dot product of unit vectors
    MSS = X_unit @ X_unit.T
    np.fill_diagonal(MSS, 1.0)
    MSS = np.clip(MSS, 0, 1)  # Cosine can be negative; clip for similarity network

    print(f"MSS matrix: {MSS.shape}, "
          f"mean={MSS.mean():.3f}, "
          f"off-diag mean={MSS[~np.eye(n, dtype=bool)].mean():.3f}")

    np.save(OUT_MSS, MSS)
    print(f"Saved: {OUT_MSS}")
    return MSS


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*60)
    print(" Metabolite Fingerprint Computation (PaDELPy)")
    print("="*60)

    if not os.path.exists(IN_SMILES):
        sys.exit(f"ERROR: {IN_SMILES} not found. Run Script 1 first.")

    smiles_df = pd.read_csv(IN_SMILES)
    print(f"Loaded {len(smiles_df)} metabolites from {IN_SMILES}")

    # Remove duplicates
    smiles_df = smiles_df.drop_duplicates("hmdb_id").reset_index(drop=True)

    # Compute
    results, failed, metadata = compute_fingerprints(smiles_df)

    if not results:
        sys.exit("ERROR: No fingerprints computed. Check SMILES strings.")

    X_norm, kept_cols, id_df = process_and_normalize(results, metadata)
    MSS = compute_mss(X_norm)

    print("\n" + "="*60)
    print(f" DONE")
    print(f"  Metabolites processed : {len(results)}")
    print(f"  Features (normalized) : {X_norm.shape[1]}")
    print(f"  MSS matrix            : {MSS.shape}")
    print(f"  Failed metabolites    : {len(failed)}")
    print("="*60)
