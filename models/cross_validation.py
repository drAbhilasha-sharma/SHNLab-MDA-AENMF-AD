"""
5-Fold Stratified Cross-Validation
=====================================
Stratified within each disease stratum to prevent IBD dominance.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import List

from ..models.mda_aenmf_ad import MDAAENMFAD
from ..utils.snf import snf_fuse
from ..data.similarity import build_all_similarity_networks
from .metrics import compute_all_metrics


def run_cv(config: dict,
           pairs: pd.DataFrame,
           snf_matrix: np.ndarray,
           ecfp4: np.ndarray,
           edge_index: np.ndarray,
           disease_names: List[str],
           metabolite_names: List[str]) -> pd.DataFrame:
    """
    Run 5-fold stratified CV.  Returns per-fold metrics DataFrame.
    """
    n_folds = config['evaluation']['n_folds']
    skf     = StratifiedKFold(n_splits=n_folds, shuffle=True,
                               random_state=config['data']['random_seed'])
    d2i = {d: i for i, d in enumerate(disease_names)}
    m2i = {m: i for i, m in enumerate(metabolite_names)}

    # Build strata labels: combine disease index + label for stratification
    strata = (pairs['disease'].map(lambda d: d2i.get(d, 0)) * 2 +
              pairs['label']).values

    results = []

    for fold, (train_idx, test_idx) in enumerate(
            skf.split(np.zeros(len(pairs)), strata), 1):

        print(f'  Fold {fold}/{n_folds} ...')
        train_df = pairs.iloc[train_idx].reset_index(drop=True)
        test_df  = pairs.iloc[test_idx].reset_index(drop=True)

        model = MDAAENMFAD(config, len(disease_names), len(metabolite_names))
        model.disease_names    = disease_names
        model.metabolite_names = metabolite_names
        model.pretrain(snf_matrix, ecfp4, edge_index)
        model.build_embeddings(snf_matrix, ecfp4, edge_index)

        # Build features for training split
        X_tr, y_tr = _features(train_df, d2i, m2i, model.nmf)
        X_te, y_te = _features(test_df,  d2i, m2i, model.nmf)

        pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
        model.train_mlp(X_tr, y_tr, pos_weight=pos_weight)

        preds = model.mlp.predict_proba(
            torch.FloatTensor(X_te).to(model.device)).cpu().numpy()

        if len(np.unique(y_te)) < 2:
            continue

        m = compute_all_metrics(y_te, preds,
                                 threshold=config['evaluation']['threshold'],
                                 k_list=config['evaluation']['top_k'])
        m['fold']     = fold
        m['n_train']  = len(train_idx)
        m['n_test']   = len(test_idx)

        # Logistic regression baseline
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_tr, y_tr)
        base_score = lr.predict_proba(X_te)[:, 1]
        m['baseline_auc'] = round(roc_auc_score(y_te, base_score), 6)

        results.append(m)
        print(f'    AUC={m["auc"]:.4f}  AUPR={m["aupr"]:.4f}')

    return pd.DataFrame(results)


def _features(df, d2i, m2i, nmf):
    X, y = [], []
    for _, row in df.iterrows():
        if row['disease'] not in d2i or row['metabolite'] not in m2i:
            continue
        di = d2i[row['disease']]; mi = m2i[row['metabolite']]
        X.append(np.concatenate([nmf.W_disease[di], nmf.W_metabolite[mi]]))
        y.append(row['label'])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
