#!/usr/bin/env python3
"""
train.py — MDA-AENMF-AD training script
Five-fold stratified cross-validation with 1:1 positive:negative sampling.

Usage:
    python train.py --associations data/associations.csv \
                    --padel_features data/padel_2d.csv \
                    --kegg_pathways data/kegg_immune_pathways.csv \
                    --aida_autoantibodies data/aida_annotations.csv \
                    --folds 5 --seed 42 --output results/
"""

import argparse
import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression

from models.mda_aenmf_ad import MDAAENMFAD
from models.similarity_networks import (
    compute_metabolite_similarity, compute_disease_similarity,
    compute_ips, compute_aps
)
from models.snf import snf_fuse
from utils.evaluation import compute_metrics, print_cv_summary


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MDA-AENMF-AD")
    p.add_argument("--associations", required=True,
                   help="CSV: columns [metabolite_id, disease_id, label]")
    p.add_argument("--padel_features", required=True,
                   help="CSV: PaDEL 2D descriptors, rows=metabolites")
    p.add_argument("--kegg_pathways", required=True,
                   help="CSV: KEGG immune pathway membership matrix")
    p.add_argument("--aida_autoantibodies", required=True,
                   help="CSV: AIDA autoantibody target annotations")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--l2", type=float, default=0.01)
    p.add_argument("--nmf_rank", type=int, default=4)
    p.add_argument("--nmf_iters", type=int, default=1000)
    p.add_argument("--ae_epochs", type=int, default=20)
    p.add_argument("--mlp_epochs", type=int, default=50)
    p.add_argument("--output", default="results/")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--sample_mode", action="store_true",
                   help="Use sample data for quick pipeline test")
    return p.parse_args()


def load_data(args: argparse.Namespace):
    assoc = pd.read_csv(args.associations)
    padel = pd.read_csv(args.padel_features, index_col=0)
    kegg  = pd.read_csv(args.kegg_pathways, index_col=0)
    aida  = pd.read_csv(args.aida_autoantibodies, index_col=0)
    return assoc, padel, kegg, aida


def build_similarity_matrices(padel, kegg, aida):
    """Construct and fuse all metabolite and disease similarity networks."""
    print("  Building metabolite similarity networks...")
    M_sim = compute_metabolite_similarity(padel)          # dict of 3 matrices

    print("  Building disease similarity networks...")
    D_std = compute_disease_similarity(kegg)               # DSS, DGIP, DSIE
    D_ips = compute_ips(kegg)                              # IPS (novel)
    D_aps = compute_aps(aida)                              # APS (novel)

    print("  Running SNF fusion...")
    SM = snf_fuse(list(M_sim.values()), k=37)              # metabolite fused (k=37)
    SD = snf_fuse([*D_std.values(), D_ips, D_aps], k=1)   # disease fused  (k=1)

    return SM, SD


def sample_negatives(pos_pairs, all_metabolites, all_diseases, n_neg, rng):
    """Sample n_neg negative pairs not present in pos_pairs."""
    pos_set = set(map(tuple, pos_pairs[["metabolite_id", "disease_id"]].values))
    neg = []
    while len(neg) < n_neg:
        m = rng.choice(all_metabolites)
        d = rng.choice(all_diseases)
        if (m, d) not in pos_set:
            neg.append((m, d, 0))
            pos_set.add((m, d))
    return pd.DataFrame(neg, columns=["metabolite_id", "disease_id", "label"])


def train_one_fold(
    train_df, test_df, SM, SD, padel, args, fold_idx, device
):
    """Train and evaluate on one CV fold. Returns dict of metrics."""
    rng = np.random.default_rng(args.seed + fold_idx)
    all_metabolites = padel.index.tolist()
    all_diseases    = SD.index.tolist() if hasattr(SD, 'index') else list(range(SD.shape[0]))

    # 1:1 negative sampling per fold
    pos_train = train_df[train_df.label == 1]
    neg_train = sample_negatives(pos_train, all_metabolites, all_diseases, len(pos_train), rng)
    train_balanced = pd.concat([pos_train, neg_train]).sample(frac=1, random_state=args.seed + fold_idx)

    pos_test  = test_df[test_df.label == 1]
    neg_test  = sample_negatives(pos_test,  all_metabolites, all_diseases, len(pos_test),  rng)
    test_balanced  = pd.concat([pos_test, neg_test]).sample(frac=1, random_state=args.seed + fold_idx + 100)

    print(f"    Fold {fold_idx+1}: train={len(train_balanced)}, test={len(test_balanced)}")

    # Initialise model
    model = MDAAENMFAD(
        n_metabolites=len(all_metabolites),
        n_diseases=len(all_diseases),
        padel_dim=padel.shape[1],
        nmf_rank=args.nmf_rank,
        nmf_iters=args.nmf_iters,
        ae_epochs=args.ae_epochs,
        device=device,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    criterion = nn.BCEWithLogitsLoss()

    # Feature pre-computation (AE + NMF) on training fold
    model.fit_unsupervised(SM, SD, padel, train_balanced)

    # MLP training
    model.train()
    for epoch in range(args.mlp_epochs):
        optimizer.zero_grad()
        logits, _ = model(train_balanced, SM, SD, padel)
        labels = torch.tensor(train_balanced.label.values, dtype=torch.float32, device=device)
        loss = criterion(logits.squeeze(), labels)
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        logits_test, features_test = model(test_balanced, SM, SD, padel)
        probs = torch.sigmoid(logits_test).cpu().numpy().squeeze()

    labels_test = test_balanced.label.values
    auc  = roc_auc_score(labels_test, probs)
    aupr = average_precision_score(labels_test, probs)

    # Logistic regression baseline on raw 136-dim features
    lr_baseline = LogisticRegression(max_iter=1000, random_state=args.seed)
    train_feats_np = features_test.cpu().numpy()  # reuse test features shape
    # (In practice train features would be computed separately)
    baseline_auc = 0.0  # placeholder if features unavailable at baseline step

    return {"fold": fold_idx + 1, "auc": auc, "aupr": aupr,
            "n_train": len(train_balanced), "n_test": len(test_balanced),
            "baseline_auc": baseline_auc}


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output, exist_ok=True)
    device = torch.device(args.device)
    print(f"Device: {device}")

    print("Loading data...")
    assoc, padel, kegg, aida = load_data(args)

    print("Building similarity matrices...")
    SM, SD = build_similarity_matrices(padel, kegg, aida)

    # Stratified K-fold on positive associations
    pos = assoc[assoc.label == 1].reset_index(drop=True)
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    # Stratify by disease to ensure representation
    fold_results = []
    for fold_i, (train_idx, test_idx) in enumerate(skf.split(pos, pos.disease_id)):
        print(f"\n--- Fold {fold_i+1}/{args.folds} ---")
        train_pos = pos.iloc[train_idx]
        test_pos  = pos.iloc[test_idx]
        result = train_one_fold(train_pos, test_pos, SM, SD, padel, args, fold_i, device)
        fold_results.append(result)
        print(f"    AUC={result['auc']:.4f}  AUPR={result['aupr']:.4f}")

    print_cv_summary(fold_results)

    # Save results
    results_df = pd.DataFrame(fold_results)
    out_path = os.path.join(args.output, "cv_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    # Save run config
    config_path = os.path.join(args.output, "run_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Config saved to {config_path}")


if __name__ == "__main__":
    main()
