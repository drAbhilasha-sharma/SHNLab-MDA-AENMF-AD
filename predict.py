#!/usr/bin/env python3
"""
predict.py — Generate novel metabolite–disease association predictions.

Usage:
    python predict.py \
        --model_path results/best_model.pt \
        --metabolites data/candidate_metabolites.csv \
        --diseases RA MS IBD \
        --top_k 20 \
        --filter_endogenous \
        --output results/novel_predictions.csv
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch

from models.mda_aenmf_ad import MDAAENMFAD
from models.similarity_networks import (
    compute_metabolite_similarity, compute_disease_similarity,
    compute_ips, compute_aps
)
from models.snf import snf_fuse

# Known non-endogenous HMDB accessions to exclude from predictions
# (environmental contaminants, heavy metals — artefacts of PaDEL descriptor overlap)
NON_ENDOGENOUS_HMDB = {
    "HMDB0031518",  # Cyclohexane (industrial solvent)
    "HMDB0015551",  # Thallium (heavy metal toxicant)
    "HMDB0015530",  # Mercury (heavy metal toxicant)
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict novel metabolite-disease associations")
    p.add_argument("--model_path", required=True, help="Path to saved model checkpoint (.pt)")
    p.add_argument("--metabolites", required=True,
                   help="CSV with candidate metabolites: [hmdb_id, name, padel_features...]")
    p.add_argument("--padel_features", default=None,
                   help="Precomputed PaDEL CSV (if separate from --metabolites)")
    p.add_argument("--kegg_pathways", required=True)
    p.add_argument("--aida_autoantibodies", required=True)
    p.add_argument("--known_associations", default=None,
                   help="CSV of known associations to exclude from novel predictions")
    p.add_argument("--diseases", nargs="+", default=["RA", "MS", "IBD"],
                   help="Disease IDs to predict for")
    p.add_argument("--top_k", type=int, default=20,
                   help="Number of top predictions to return per disease")
    p.add_argument("--filter_endogenous", action="store_true", default=True,
                   help="Exclude known non-endogenous HMDB entries (default: True)")
    p.add_argument("--threshold", type=float, default=0.0,
                   help="Minimum prediction score threshold (0 = return top_k regardless)")
    p.add_argument("--output", default="results/novel_predictions.csv")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_model(model_path: str, device: torch.device) -> MDAAENMFAD:
    checkpoint = torch.load(model_path, map_location=device)
    model = MDAAENMFAD(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def filter_known(pairs_df: pd.DataFrame, known_path: str) -> pd.DataFrame:
    known = pd.read_csv(known_path)
    known_set = set(zip(known.metabolite_id, known.disease_id))
    mask = ~pairs_df.apply(
        lambda r: (r.metabolite_id, r.disease_id) in known_set, axis=1
    )
    return pairs_df[mask].reset_index(drop=True)


def main():
    args  = parse_args()
    device = torch.device(args.device)
    print(f"Device: {device}")

    # Load inputs
    mets  = pd.read_csv(args.metabolites)
    kegg  = pd.read_csv(args.kegg_pathways,  index_col=0)
    aida  = pd.read_csv(args.aida_autoantibodies, index_col=0)

    if args.padel_features:
        padel = pd.read_csv(args.padel_features, index_col=0)
    else:
        padel_cols = [c for c in mets.columns if c not in ("hmdb_id", "name")]
        padel = mets.set_index("hmdb_id")[padel_cols]

    # Optionally remove known non-endogenous entries
    if args.filter_endogenous:
        before = len(padel)
        padel = padel[~padel.index.isin(NON_ENDOGENOUS_HMDB)]
        removed = before - len(padel)
        if removed:
            print(f"  Excluded {removed} non-endogenous HMDB entries "
                  f"({list(NON_ENDOGENOUS_HMDB & set(padel.index))})")

    # Build similarity matrices
    print("Building similarity matrices...")
    M_sim = compute_metabolite_similarity(padel)
    D_std = compute_disease_similarity(kegg)
    D_ips = compute_ips(kegg)
    D_aps = compute_aps(aida)
    SM = snf_fuse(list(M_sim.values()), k=37)
    SD = snf_fuse([*D_std.values(), D_ips, D_aps], k=1)

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path, device)

    # Build all candidate pairs
    all_mets = padel.index.tolist()
    all_pairs = pd.DataFrame(
        [(m, d) for m in all_mets for d in args.diseases],
        columns=["metabolite_id", "disease_id"]
    )

    # Remove known associations if provided
    if args.known_associations:
        all_pairs = filter_known(all_pairs, args.known_associations)
        print(f"  {len(all_pairs)} candidate pairs after excluding known associations.")

    # Score all pairs
    print(f"Scoring {len(all_pairs):,} candidate pairs...")
    with torch.no_grad():
        logits, _ = model(all_pairs, SM, SD, padel)
        scores = torch.sigmoid(logits).cpu().numpy().squeeze()

    all_pairs["score"] = scores

    # Apply threshold
    if args.threshold > 0:
        all_pairs = all_pairs[all_pairs.score >= args.threshold]

    # Top-k per disease
    results = []
    for disease in args.diseases:
        subset = all_pairs[all_pairs.disease_id == disease].copy()
        subset = subset.sort_values("score", ascending=False).head(args.top_k)
        subset["rank"] = range(1, len(subset) + 1)
        results.append(subset)

    out_df = pd.concat(results, ignore_index=True)

    # Add metabolite name if available
    if "name" in mets.columns:
        name_map = mets.set_index("hmdb_id")["name"].to_dict()
        out_df["metabolite_name"] = out_df.metabolite_id.map(name_map)

    col_order = [c for c in ["rank","disease_id","metabolite_id","metabolite_name","score"]
                 if c in out_df.columns]
    out_df = out_df[col_order]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"\nTop-{args.top_k} novel predictions per disease saved to {args.output}")
    print(out_df.groupby("disease_id").head(3).to_string(index=False))


if __name__ == "__main__":
    main()
