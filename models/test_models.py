"""Unit tests for MDA-AENMF-AD model components."""

import numpy as np
import torch
import pytest

from src.models.dae import DiseaseAutoencoder
from src.models.gae import MetaboliteGAE
from src.models.nmf import JointNMF
from src.models.mlp import MLPPredictor


class TestDAE:
    def test_forward_shape(self):
        model = DiseaseAutoencoder(n_diseases=4, hidden_dim=16, latent_dim=8)
        x     = torch.rand(4, 4)
        recon, z = model(x)
        assert recon.shape == (4, 4), "Reconstruction shape mismatch"
        assert z.shape    == (4, 8), "Latent shape mismatch"

    def test_reconstruction_loss_positive(self):
        model = DiseaseAutoencoder(n_diseases=4, hidden_dim=16, latent_dim=8)
        x     = torch.rand(4, 4)
        loss  = model.reconstruction_loss(x)
        assert loss.item() >= 0, "Loss must be non-negative"

    def test_encode_consistency(self):
        model = DiseaseAutoencoder(n_diseases=4, hidden_dim=16, latent_dim=8)
        model.eval()
        x  = torch.rand(4, 4)
        z1 = model.encode(x)
        _, z2 = model(x)
        assert torch.allclose(z1, z2), "Encode and forward z must match"


class TestGAE:
    def test_forward_shape(self):
        model      = MetaboliteGAE(in_channels=16, hidden_channels=8, latent_dim=4)
        x          = torch.rand(10, 16)
        edge_index = torch.randint(0, 10, (2, 20))
        recon, z   = model(x, edge_index)
        assert z.shape == (10, 4), "GAE latent shape mismatch"

    def test_recon_loss_positive(self):
        model      = MetaboliteGAE(in_channels=16, hidden_channels=8, latent_dim=4)
        x          = torch.rand(10, 16)
        edge_index = torch.randint(0, 10, (2, 20))
        z          = model.encode(x, edge_index)
        loss       = model.recon_loss(z, edge_index)
        assert loss.item() >= 0


class TestJointNMF:
    def test_fit_transform_shapes(self):
        nmf  = JointNMF(rank=4)
        d_emb = np.abs(np.random.randn(4, 64))
        m_emb = np.abs(np.random.randn(20, 64))
        nmf.fit(d_emb, m_emb)
        assert nmf.W_disease.shape    == (4, 4),  "Disease NMF shape"
        assert nmf.W_metabolite.shape == (20, 4), "Metabolite NMF shape"

    def test_pair_features_shape(self):
        nmf  = JointNMF(rank=4)
        d_emb = np.abs(np.random.randn(4, 64))
        m_emb = np.abs(np.random.randn(20, 64))
        nmf.fit(d_emb, m_emb)
        feat = nmf.pair_features(0, 5)
        assert feat.shape == (8,), "Pair feature vector should be 8-d"

    def test_all_pair_features(self):
        nmf  = JointNMF(rank=4)
        d_emb = np.abs(np.random.randn(4, 64))
        m_emb = np.abs(np.random.randn(20, 64))
        nmf.fit(d_emb, m_emb)
        X, d_idx, m_idx = nmf.all_pair_features()
        assert X.shape == (80, 8), "All pairs: 4*20=80 rows, 8 cols"


class TestMLPPredictor:
    def test_output_range(self):
        mlp = MLPPredictor(input_dim=8)
        x   = torch.rand(32, 8)
        out = mlp(x)
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0
        assert out.shape == (32,)

    def test_predict_proba_no_grad(self):
        mlp = MLPPredictor(input_dim=8)
        x   = torch.rand(10, 8)
        out = mlp.predict_proba(x)
        assert out.shape == (10,)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
