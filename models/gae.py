"""
Metabolite Graph Autoencoder (GAE)
===================================
Two-layer GCN encoder over the metabolite interaction graph.
ECFP4 fingerprints (1024-bit) as node features.
Dot-product decoder for adjacency reconstruction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
from typing import Tuple


class GCNEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, dropout: float = 0.3):
        super().__init__()
        self.conv1   = GCNConv(in_channels, hidden_channels)
        self.conv2   = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


class DotProductDecoder(nn.Module):
    """Reconstructs adjacency via sigmoid(z_i · z_j)."""
    def forward(self, z: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        return torch.sigmoid((z[src] * z[dst]).sum(dim=-1))


class MetaboliteGAE(nn.Module):
    def __init__(self, in_channels: int = 1024, hidden_channels: int = 256,
                 latent_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.encoder = GCNEncoder(in_channels, hidden_channels,
                                   latent_dim, dropout)
        self.decoder = DotProductDecoder()
        self.latent_dim = latent_dim

    def encode(self, x: torch.Tensor,
               edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def decode(self, z: torch.Tensor,
               edge_index: torch.Tensor) -> torch.Tensor:
        return self.decoder(z, edge_index)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z    = self.encode(x, edge_index)
        recon = self.decode(z, edge_index)
        return recon, z

    def recon_loss(self, z: torch.Tensor,
                   pos_edge_index: torch.Tensor,
                   neg_edge_index: torch.Tensor = None) -> torch.Tensor:
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(
                pos_edge_index, z.size(0),
                num_neg_samples=pos_edge_index.size(1))

        pos_loss = -torch.log(self.decode(z, pos_edge_index) + 1e-15).mean()
        neg_loss = -torch.log(1 - self.decode(z, neg_edge_index) + 1e-15).mean()
        return pos_loss + neg_loss


def pretrain_gae(model: MetaboliteGAE, x: torch.Tensor,
                 edge_index: torch.Tensor, lr: float = 1e-3,
                 epochs: int = 150, patience: int = 30,
                 device: str = 'cpu') -> MetaboliteGAE:
    """Pre-train GAE on metabolite interaction graph."""
    model      = model.to(device)
    x          = x.to(device)
    edge_index = edge_index.to(device)
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr,
                                   weight_decay=1e-5)
    best_loss  = float('inf')
    no_improv  = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        z    = model.encode(x, edge_index)
        loss = model.recon_loss(z, edge_index)
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            no_improv = 0
        else:
            no_improv += 1
            if no_improv >= patience:
                break

    return model
