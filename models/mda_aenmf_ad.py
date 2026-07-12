"""
MDA-AENMF-AD  —  Full Integrated Model
========================================
Orchestrates the DAE, GAE, NMF, and MLP into a complete
metabolite-disease association prediction pipeline.
"""

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional

from .dae import DiseaseAutoencoder, pretrain_dae
from .gae import MetaboliteGAE, pretrain_gae
from .nmf import JointNMF
from .mlp import MLPPredictor


class MDAAENMFAD:
    """
    Full MDA-AENMF-AD pipeline.

    Parameters
    ----------
    config : dict
        Loaded from configs/default.yaml
    n_diseases : int
        Number of diseases in the dataset
    n_metabolites : int
        Number of unique metabolites
    """

    def __init__(self, config: dict, n_diseases: int, n_metabolites: int):
        self.cfg           = config
        self.n_diseases    = n_diseases
        self.n_metabolites = n_metabolites
        self.device        = config['training']['device'] \
                             if torch.cuda.is_available() else 'cpu'

        mc = config['model']
        self.dae = DiseaseAutoencoder(
            n_diseases   = n_diseases,
            hidden_dim   = mc['dae_hidden_dim'],
            latent_dim   = mc['dae_latent_dim'],
            dropout      = mc['dae_dropout'],
            batch_norm   = mc['dae_batch_norm'],
        )
        self.gae = MetaboliteGAE(
            in_channels     = mc.get('ecfp_nbits', 1024),
            hidden_channels = mc['gae_hidden_dim'],
            latent_dim      = mc['gae_latent_dim'],
            dropout         = mc['gae_dropout'],
        )
        self.nmf = JointNMF(
            rank         = mc['nmf_rank'],
            max_iter     = mc['nmf_max_iter'],
            tol          = mc['nmf_tol'],
            random_state = config['data']['random_seed'],
        )
        self.mlp = MLPPredictor(
            input_dim = mc['nmf_rank'] * 2,
            hidden1   = mc['mlp_hidden1'],
            hidden2   = mc['mlp_hidden2'],
            dropout   = mc['mlp_dropout'],
        )

        self.disease_names   : List[str] = []
        self.metabolite_names: List[str] = []

    @classmethod
    def from_config(cls, config_path: str,
                    n_diseases: int, n_metabolites: int) -> 'MDAAENMFAD':
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return cls(cfg, n_diseases, n_metabolites)

    # ── Phase 1: Pre-train individual encoders ────────────────────────────────
    def pretrain(self, snf_matrix: np.ndarray, ecfp4: np.ndarray,
                 edge_index: np.ndarray) -> None:
        tc = self.cfg['training']
        snf_t = torch.FloatTensor(snf_matrix)
        pretrain_dae(self.dae, snf_t, lr=tc['lr'],
                     epochs=tc['pretrain_dae_epochs'],
                     patience=tc['patience'],
                     val_split=tc['val_split'],
                     device=self.device)

        x_t = torch.FloatTensor(ecfp4)
        ei_t = torch.LongTensor(edge_index)
        pretrain_gae(self.gae, x_t, ei_t, lr=tc['lr'],
                     epochs=tc['pretrain_gae_epochs'],
                     patience=tc['patience'],
                     device=self.device)

    # ── Phase 2: Build NMF co-embedding ──────────────────────────────────────
    def build_embeddings(self, snf_matrix: np.ndarray,
                         ecfp4: np.ndarray,
                         edge_index: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.dae.eval(); self.gae.eval()
        with torch.no_grad():
            snf_t  = torch.FloatTensor(snf_matrix).to(self.device)
            d_emb  = self.dae.encode(snf_t).cpu().numpy()

            x_t   = torch.FloatTensor(ecfp4).to(self.device)
            ei_t  = torch.LongTensor(edge_index).to(self.device)
            m_emb = self.gae.encode(x_t, ei_t).cpu().numpy()

        self.nmf.fit(d_emb, m_emb)
        return d_emb, m_emb

    # ── Phase 3: Train MLP predictor ─────────────────────────────────────────
    def train_mlp(self, pairs: np.ndarray, labels: np.ndarray,
                  pos_weight: Optional[float] = None) -> List[float]:
        tc  = self.cfg['training']
        X   = torch.FloatTensor(pairs)
        y   = torch.FloatTensor(labels)

        n_val = max(1, int(len(X) * tc['val_split']))
        idx   = torch.randperm(len(X))
        val_X, train_X = X[idx[:n_val]], X[idx[n_val:]]
        val_y, train_y = y[idx[:n_val]], y[idx[n_val:]]

        loader    = DataLoader(TensorDataset(train_X, train_y),
                               batch_size=tc['batch_size'], shuffle=True)
        optimizer = torch.optim.Adam(self.mlp.parameters(), lr=tc['lr'],
                                     weight_decay=tc['weight_decay'])
        pw        = torch.tensor([pos_weight]) if pos_weight else None
        criterion = nn.BCELoss()

        self.mlp = self.mlp.to(self.device)
        best_val  = float('inf')
        no_improv = 0
        losses    = []

        for epoch in range(tc['max_epochs']):
            self.mlp.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                pred = self.mlp(xb)
                loss = criterion(pred, yb)
                loss.backward(); optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss)

            self.mlp.eval()
            with torch.no_grad():
                val_pred = self.mlp(val_X.to(self.device))
                val_loss = criterion(val_pred, val_y.to(self.device)).item()
            if val_loss < best_val:
                best_val = val_loss; no_improv = 0
            else:
                no_improv += 1
                if no_improv >= tc['patience']:
                    break

        return losses

    # ── Prediction ────────────────────────────────────────────────────────────
    def predict_all(self) -> np.ndarray:
        """Return full n_diseases × n_metabolites score matrix."""
        X, d_idx, m_idx = self.nmf.all_pair_features()
        scores = self.mlp.predict_proba(
            torch.FloatTensor(X).to(self.device)).cpu().numpy()
        mat = np.zeros((self.n_diseases, self.n_metabolites))
        mat[d_idx, m_idx] = scores
        return mat

    def predict_disease(self, disease_name: str, top_k: int = 20) -> Dict:
        """Return top-k metabolite predictions for a single disease."""
        d_idx  = self.disease_names.index(disease_name)
        scores = self.predict_all()[d_idx]
        top    = np.argsort(scores)[::-1][:top_k]
        return {
            'disease':  disease_name,
            'rank':     list(range(1, top_k + 1)),
            'metabolite': [self.metabolite_names[i] for i in top],
            'score':    scores[top].tolist(),
        }

    # ── Serialisation ─────────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        torch.save({
            'dae': self.dae.state_dict(),
            'gae': self.gae.state_dict(),
            'mlp': self.mlp.state_dict(),
            'nmf_W_disease':    self.nmf.W_disease,
            'nmf_W_metabolite': self.nmf.W_metabolite,
        }, path)

    def load_weights(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.dae.load_state_dict(ckpt['dae'])
        self.gae.load_state_dict(ckpt['gae'])
        self.mlp.load_state_dict(ckpt['mlp'])
        self.nmf.W_disease    = ckpt['nmf_W_disease']
        self.nmf.W_metabolite = ckpt['nmf_W_metabolite']
