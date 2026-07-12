"""
disease_ae.py — Disease Auto-encoder module.

Input : n_diseases × n_diseases SNF-integrated similarity matrix (SD).
Output: n_diseases × latent_dim encoded disease feature matrix.

Architecture (scaled for 8-disease input):
    Encoder: 8 → intermediate → 64-dim latent
    Decoder: 64-dim latent → intermediate → 8 (reconstruction)
Loss: MSE reconstruction loss, optimised with Adam.
"""

import torch
import torch.nn as nn
from torch.optim import Adam


class DiseaseAutoEncoder(nn.Module):
    """
    Auto-encoder for the SNF-integrated disease similarity matrix.

    For the published 8-disease dataset the input dimensionality is 8,
    and the model maps to a 64-dimensional latent space consistent with
    the original MDA-AENMF framework (Gao et al. 2023).

    Note: The architecture notation '8→6→5→4→4→3' in Supplementary Table S10
    describes the dimensionality reduction sequence applied to the raw
    similarity matrix prior to AE input, not the AE hidden layer neuron counts.
    See manuscript Supplementary Table S10 footnote †.
    """

    def __init__(
        self,
        in_dim: int = 8,
        latent_dim: int = 64,
        ae_epochs: int = 20,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.in_dim     = in_dim
        self.latent_dim = latent_dim
        self.ae_epochs  = ae_epochs
        self.lr         = lr

        # For small in_dim (≤ latent_dim) a single linear layer is used.
        # For larger inputs intermediate layers are inserted automatically.
        if in_dim <= latent_dim:
            # Direct: in_dim → latent_dim (e.g., 8 → 64)
            self.encoder = nn.Sequential(
                nn.Linear(in_dim, latent_dim),
                nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, in_dim),
                nn.Sigmoid(),
            )
        else:
            # Multi-layer: in_dim → hidden → latent_dim
            hidden = max(in_dim // 2, latent_dim * 2)
            self.encoder = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, latent_dim), nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, in_dim), nn.Sigmoid(),
            )

    def fit(self, SD: torch.Tensor) -> None:
        """
        Pre-train the auto-encoder on the disease similarity matrix.

        Parameters
        ----------
        SD : Tensor (n_diseases, n_diseases) — SNF-fused similarity matrix.
        """
        optimizer = Adam(self.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        self.train()
        for epoch in range(self.ae_epochs):
            optimizer.zero_grad()
            recon = self.decoder(self.encoder(SD))
            loss  = criterion(recon, SD)
            loss.backward()
            optimizer.step()

    def encode(self, SD: torch.Tensor) -> torch.Tensor:
        """Return latent disease feature matrix (n_diseases, latent_dim)."""
        self.eval()
        with torch.no_grad():
            return self.encoder(SD)

    def forward(self, SD: torch.Tensor):
        z    = self.encoder(SD)
        recon = self.decoder(z)
        return recon, z
