"""Unit tests for evaluation utilities."""

import numpy as np
import pytest
from src.evaluation.metrics import precision_at_k, compute_all_metrics
from src.utils.snf import snf_fuse


class TestMetrics:
    def test_precision_at_k_perfect(self):
        y     = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        score = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1, 0.05, 0.04, 0.03, 0.01])
        assert precision_at_k(y, score, k=3) == 1.0

    def test_precision_at_k_random(self):
        y     = np.zeros(20); y[:5] = 1
        score = np.random.rand(20)
        p10   = precision_at_k(y, score, k=10)
        assert 0 <= p10 <= 1.0

    def test_compute_all_metrics_keys(self):
        np.random.seed(0)
        y     = np.array([1]*20 + [0]*20)
        score = np.random.rand(40)
        m     = compute_all_metrics(y, score)
        assert 'auc'  in m
        assert 'aupr' in m
        assert 'f1'   in m
        assert 'precision_at_10' in m
        assert 'precision_at_20' in m

    def test_auc_range(self):
        np.random.seed(1)
        y     = np.random.randint(0, 2, 100)
        score = np.random.rand(100)
        m     = compute_all_metrics(y, score)
        assert 0.0 <= m['auc']  <= 1.0
        assert 0.0 <= m['aupr'] <= 1.0


class TestSNF:
    def test_output_shape(self):
        mats = [np.random.rand(4, 4) for _ in range(5)]
        for m in mats:
            np.fill_diagonal(m, 1.0)
            m[:] = (m + m.T) / 2
        W = snf_fuse(mats, k=2, t=5)
        assert W.shape == (4, 4)

    def test_output_range(self):
        mats = [np.abs(np.random.rand(4, 4)) for _ in range(3)]
        for m in mats:
            np.fill_diagonal(m, 1.0)
            m[:] = (m + m.T) / 2
        W = snf_fuse(mats, k=2, t=5)
        assert W.min() >= 0.0 - 1e-6
        assert W.max() <= 1.0 + 1e-6

    def test_symmetry(self):
        mats = [np.abs(np.random.rand(4, 4)) for _ in range(3)]
        for m in mats:
            np.fill_diagonal(m, 1.0)
            m[:] = (m + m.T) / 2
        W = snf_fuse(mats, k=2, t=5)
        assert np.allclose(W, W.T, atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
