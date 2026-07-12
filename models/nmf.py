"""
Joint Non-Negative Matrix Factorisation (NMF)
=============================================
Co-embeds disease and metabolite latent vectors in a shared rank-4 space.
"""

import numpy as np
from sklearn.decomposition import NMF
from typing import Tuple


class JointNMF:
    """
    Jointly factorise disease (64-d) and metabolite (64-d) latent vectors
    into a shared rank-r representation.

    Usage:
        nmf = JointNMF(rank=4)
        nmf.fit(disease_embeddings, metabolite_embeddings)
        W_d, W_m = nmf.transform(disease_embeddings, metabolite_embeddings)
        pair_features = nmf.pair_features(disease_idx, metabolite_idx)
    """

    def __init__(self, rank: int = 4, max_iter: int = 500,
                 tol: float = 1e-6, random_state: int = 42):
        self.rank         = rank
        self.max_iter     = max_iter
        self.tol          = tol
        self.random_state = random_state
        self._nmf         = NMF(n_components=rank, max_iter=max_iter,
                                 tol=tol, random_state=random_state,
                                 init='nndsvda')
        self.W_disease    = None
        self.W_metabolite = None

    def fit(self, disease_emb: np.ndarray,
            metabolite_emb: np.ndarray) -> 'JointNMF':
        """Fit NMF on vertically stacked embeddings."""
        combined = np.vstack([
            np.abs(disease_emb),
            np.abs(metabolite_emb),
        ])
        W = self._nmf.fit_transform(combined)
        nd = disease_emb.shape[0]
        self.W_disease    = W[:nd]
        self.W_metabolite = W[nd:]
        return self

    def transform(self, disease_emb: np.ndarray,
                  metabolite_emb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Transform new embeddings into the learned NMF space."""
        combined = np.vstack([np.abs(disease_emb), np.abs(metabolite_emb)])
        W        = self._nmf.transform(combined)
        nd       = disease_emb.shape[0]
        return W[:nd], W[nd:]

    def pair_features(self, disease_idx: int,
                      metabolite_idx: int) -> np.ndarray:
        """Return 8-d concatenated NMF factors for a metabolite-disease pair."""
        return np.concatenate([
            self.W_disease[disease_idx],
            self.W_metabolite[metabolite_idx]
        ])

    def all_pair_features(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns X (n_pairs, 2*rank), d_idx, m_idx for all disease-metabolite pairs.
        """
        nd = self.W_disease.shape[0]
        nm = self.W_metabolite.shape[0]
        d_idx, m_idx = np.meshgrid(np.arange(nd), np.arange(nm), indexing='ij')
        d_idx = d_idx.ravel(); m_idx = m_idx.ravel()
        X = np.hstack([self.W_disease[d_idx], self.W_metabolite[m_idx]])
        return X, d_idx, m_idx
