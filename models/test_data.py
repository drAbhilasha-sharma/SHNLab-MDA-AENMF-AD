"""Unit tests for data utilities."""
import numpy as np
import pytest
from src.utils.snf import snf_fuse
from src.data.features import compute_ecfp4


class TestSNF:
    def _sym_mats(self, n=4, k=5):
        mats = []
        np.random.seed(0)
        for _ in range(k):
            M = np.abs(np.random.rand(n, n))
            M = (M + M.T) / 2
            np.fill_diagonal(M, 1.0)
            mats.append(M)
        return mats

    def test_shape(self):
        W = snf_fuse(self._sym_mats(4, 5), k=2, t=3)
        assert W.shape == (4, 4)

    def test_values_in_range(self):
        W = snf_fuse(self._sym_mats(4, 5), k=2, t=3)
        assert W.min() >= -1e-9
        assert W.max() <= 1.0 + 1e-9

    def test_symmetric(self):
        W = snf_fuse(self._sym_mats(4, 5), k=2, t=3)
        assert np.allclose(W, W.T, atol=1e-5)

    def test_single_matrix(self):
        mats = self._sym_mats(4, 1)
        W = snf_fuse(mats, k=2, t=3)
        assert W.shape == (4, 4)


class TestECFP4:
    def test_zero_for_empty(self):
        fps = compute_ecfp4(['', None, ''])
        assert fps.shape == (3, 1024)
        assert fps.sum() == 0.0

    def test_shape(self):
        smiles = ['C', 'CC', 'CCC']
        fps = compute_ecfp4(smiles, n_bits=512)
        assert fps.shape == (3, 512)

    def test_dtype(self):
        fps = compute_ecfp4(['C'], n_bits=128)
        assert fps.dtype == np.float32


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
