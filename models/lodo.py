"""
Leave-One-Disease-Out (LODO) Evaluation
=========================================
Trains the full MDA-AENMF-AD model on all diseases except the target,
then evaluates predictions on the held-out disease.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from ..models.mda_aenmf_ad import MDAAENMFAD
from .metrics import precision_at_k


def run_lodo(config: dict,
             all_pairs: pd.DataFrame,
             snf_matrix: np.ndarray,
             ecfp4: np.ndarray,
             edge_index: np.ndarray,
             disease_names: List[str],
             metabolite_names: List[str]) -> pd.DataFrame:
    """
    Run LODO evaluation for all diseases listed in config.

    Parameters
    ----------
    config        : experiment config dict
    all_pairs     : DataFrame with columns [disease, metabolite, label]
    snf_matrix    : (n_diseases, n_diseases) SNF consensus matrix
    ecfp4         : (n_metabolites, 1024) fingerprint matrix
    edge_index    : (2, n_edges) metabolite graph edges
    disease_names : list of disease names in order matching snf_matrix rows
    metabolite_names : list of metabolite names in order matching ecfp4 rows

    Returns
    -------
    pd.DataFrame with LODO metrics per disease
    """
    results = []
    target_diseases = config['evaluation']['lodo_diseases']

    for target in target_diseases:
        print(f'  LODO: holding out {target} ...')

        # Split
        test_mask  = all_pairs['disease'] == target
        train_data = all_pairs[~test_mask].reset_index(drop=True)
        test_data  = all_pairs[test_mask].reset_index(drop=True)

        if len(test_data) == 0:
            print(f'    Skipped: no test associations for {target}')
            continue

        # Build reduced disease index (exclude target)
        train_diseases = [d for d in disease_names if d != target]
        train_d_idx    = [disease_names.index(d) for d in train_diseases]
        train_snf      = snf_matrix[np.ix_(train_d_idx, train_d_idx)]

        # Re-index training pairs
        d2i = {d: i for i, d in enumerate(train_diseases)}
        m2i = {m: i for i, m in enumerate(metabolite_names)}
        train_data = train_data[train_data['disease'].isin(train_diseases)]

        # Instantiate and train model
        model = MDAAENMFAD(config, len(train_diseases), len(metabolite_names))
        model.disease_names    = train_diseases
        model.metabolite_names = metabolite_names
        model.pretrain(train_snf, ecfp4, edge_index)
        model.build_embeddings(train_snf, ecfp4, edge_index)

        # Build training pair features
        X_train, y_train = _build_pair_features(
            train_data, d2i, m2i, model.nmf)
        pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        model.train_mlp(X_train, y_train, pos_weight=float(pos_weight))

        # Predict on test disease using cross-disease transfer
        # Use target disease similarity to train diseases via SNF row
        target_idx = disease_names.index(target)
        sim_to_train = snf_matrix[target_idx, train_d_idx]
        proxy_emb    = (sim_to_train[:, None] *
                        model.nmf.W_disease).sum(axis=0, keepdims=True)
        proxy_emb    = np.abs(proxy_emb)

        test_m_idx  = [m2i[m] for m in test_data['metabolite']
                       if m in m2i]
        test_scores = []
        for mi in test_m_idx:
            feat = np.concatenate([proxy_emb[0], model.nmf.W_metabolite[mi]])
            import torch
            s = model.mlp.predict_proba(
                torch.FloatTensor(feat).unsqueeze(0).to(model.device))
            test_scores.append(s.item())

        y_true = test_data[test_data['metabolite'].isin(m2i)]['label'].values
        y_score = np.array(test_scores)

        if len(np.unique(y_true)) < 2:
            continue

        auc  = roc_auc_score(y_true, y_score)
        aupr = average_precision_score(y_true, y_score)
        f1   = f1_score(y_true, (y_score >= 0.5).astype(int))
        p10  = precision_at_k(y_true, y_score, k=10)
        p20  = precision_at_k(y_true, y_score, k=20)

        results.append({
            'disease':     target,
            'n_pos':       int((y_true == 1).sum()),
            'n_neg':       int((y_true == 0).sum()),
            'lodo_auc':    round(auc,  4),
            'lodo_aupr':   round(aupr, 4),
            'lodo_f1':     round(f1,   4),
            'precision_at_10': round(p10, 4),
            'precision_at_20': round(p20, 4),
        })
        print(f'    AUC={auc:.4f}  AUPR={aupr:.4f}  F1={f1:.4f}')

    return pd.DataFrame(results)


def _build_pair_features(pairs: pd.DataFrame, d2i: dict, m2i: dict,
                          nmf) -> tuple:
    X, y = [], []
    for _, row in pairs.iterrows():
        if row['disease'] not in d2i or row['metabolite'] not in m2i:
            continue
        di = d2i[row['disease']]; mi = m2i[row['metabolite']]
        feat = np.concatenate([nmf.W_disease[di], nmf.W_metabolite[mi]])
        X.append(feat); y.append(row['label'])
    return np.array(X), np.array(y)
