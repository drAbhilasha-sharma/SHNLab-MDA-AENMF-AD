"""
Y-Randomisation (Permutation) Test
====================================
Verifies that model performance reflects real learned signal.
"""

import numpy as np
import pandas as pd
from typing import List
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

from ..models.mda_aenmf_ad import MDAAENMFAD


def run_permutation_test(config: dict, pairs: pd.DataFrame,
                          snf_matrix: np.ndarray, ecfp4: np.ndarray,
                          edge_index: np.ndarray,
                          disease_names: List[str],
                          metabolite_names: List[str],
                          n_runs: int = 3) -> pd.DataFrame:
    """
    Run n_runs independent Y-randomisation experiments.
    In each run, association labels are shuffled while all features unchanged.

    Returns
    -------
    pd.DataFrame with AUC / AUPR per run
    """
    rng     = np.random.default_rng(42)
    results = []

    for run in range(1, n_runs + 1):
        print(f'  Y-rand run {run}/{n_runs} ...')
        shuffled        = pairs.copy()
        shuffled_labels = shuffled['label'].values.copy()
        rng.shuffle(shuffled_labels)
        shuffled['label'] = shuffled_labels

        model = MDAAENMFAD(config, len(disease_names), len(metabolite_names))
        model.disease_names    = disease_names
        model.metabolite_names = metabolite_names
        model.pretrain(snf_matrix, ecfp4, edge_index)
        model.build_embeddings(snf_matrix, ecfp4, edge_index)

        d2i = {d: i for i, d in enumerate(disease_names)}
        m2i = {m: i for i, m in enumerate(metabolite_names)}
        X, y = _build_features(shuffled, d2i, m2i, model.nmf)

        # 5-fold CV on permuted labels
        skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=run)
        aucs, auprs = [], []
        for train_idx, test_idx in skf.split(X, y):
            model.train_mlp(X[train_idx], y[train_idx])
            import torch
            with torch.no_grad():
                preds = model.mlp.predict_proba(
                    torch.FloatTensor(X[test_idx]).to(model.device)
                ).cpu().numpy()
            if len(np.unique(y[test_idx])) < 2:
                continue
            aucs.append(roc_auc_score(y[test_idx], preds))
            auprs.append(average_precision_score(y[test_idx], preds))

        results.append({
            'run':       run,
            'perm_auc':  round(float(np.mean(aucs)), 4),
            'perm_aupr': round(float(np.mean(auprs)), 4),
        })
        print(f'    AUC={results[-1]["perm_auc"]:.4f}')

    return pd.DataFrame(results)


def _build_features(pairs, d2i, m2i, nmf):
    X, y = [], []
    for _, row in pairs.iterrows():
        if row['disease'] not in d2i or row['metabolite'] not in m2i:
            continue
        di = d2i[row['disease']]; mi = m2i[row['metabolite']]
        X.append(np.concatenate([nmf.W_disease[di], nmf.W_metabolite[mi]]))
        y.append(row['label'])
    return np.array(X), np.array(y)
