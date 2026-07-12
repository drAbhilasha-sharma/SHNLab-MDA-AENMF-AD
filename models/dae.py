"""
Disease Autoencoder (DAE)
=========================
Learns a compact latent representation of the SNF-fused disease similarity
matrix.  Encoder: [n_diseases x 128 x 64], decoder mirrored.
"""

import torch
import torch.nn as nn
from typing import Tuple


class DiseaseAutoencoder(nn.Module):
    """Symmetric encoder-decoder for disease similarity reconstruction."""

    def __init__(self, n_diseases: int, hidden_dim: int = 128,
                 latent_dim: int = 64, dropout: float = 0.3,
                 batch_norm: bool = True):
        super().__init__()
        self.n_diseases  = n_diseases
        self.latent_dim  = latent_dim

        # Encoder
        enc_layers = [nn.Linear(n_diseases, hidden_dim)]
        if batch_norm:
            enc_layers.append(nn.BatchNorm1d(hidden_dim))
        enc_layers += [nn.ReLU(), nn.Dropout(dropout),
                       nn.Linear(hidden_dim, latent_dim)]
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_layers = [nn.Linear(latent_dim, hidden_dim)]
        if batch_norm:
            dec_layers.append(nn.BatchNorm1d(hidden_dim))
        dec_layers += [nn.ReLU(), nn.Dropout(dropout),
                       nn.Linear(hidden_dim, n_diseases),
                       nn.Sigmoid()]
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return 64-d latent embedding for each disease row."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z

    def reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        recon, _ = self.forward(x)
        return nn.functional.mse_loss(recon, x)


def pretrain_dae(model: DiseaseAutoencoder, snf_matrix: torch.Tensor,
                 lr: float = 1e-3, epochs: int = 100,
                 patience: int = 20, val_split: float = 0.1,
                 device: str = 'cpu') -> DiseaseAutoencoder:
    """Pre-train DAE on the SNF consensus similarity matrix."""
    model = model.to(device)
    snf_matrix = snf_matrix.to(device)

    n = snf_matrix.size(0)
    n_val = max(1, int(n * val_split))
    idx = torch.randperm(n)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val  = float('inf')
    no_improv = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = model.reconstruction_loss(snf_matrix[train_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = model.reconstruction_loss(snf_matrix[val_idx]).item()

        if val_loss < best_val:
            best_val  = val_loss
            no_improv = 0
        else:
            no_improv += 1
            if no_improv >= patience:
                break

    return model
