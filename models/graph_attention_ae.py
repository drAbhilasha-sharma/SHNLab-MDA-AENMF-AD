"""
graph_attention_ae.py — Graph Attention Auto-encoder (GAE) for metabolite features.

Encodes PaDEL molecular descriptors through a graph-structured encoder
informed by the metabolite structural similarity network (SM).

Architecture:
    Encoder: PaDEL (2019-dim) → 512 → 256 → 128 → 64-dim latent
    Decoder: 64 → 128 → 256 → 512 → 2019-dim (reconstruction)
    Attention: graph attention weights from SM used to aggregate
               neighbourhood information at each encoder layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """Single graph attention layer with SM-weighted neighbourhood aggregation."""

    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert out_dim % n_heads == 0
        self.n_heads    = n_heads
        self.head_dim   = out_dim // n_heads
        self.W          = nn.Linear(in_dim, out_dim, bias=False)
        self.a          = nn.Linear(2 * self.head_dim, 1, bias=False)
        self.dropout    = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        X : (n, in_dim)
        A : (n, n) — similarity adjacency (SM or SD)
        """
        H   = self.W(X)                                  # (n, out_dim)
        H_h = H.view(H.size(0), self.n_heads, self.head_dim)

        # Attention: use A as soft adjacency mask
        # Compute pairwise attention scores for each head
        attn = torch.bmm(H_h.permute(1, 0, 2),
                         H_h.permute(1, 2, 0))           # (heads, n, n)
        attn = attn / (self.head_dim ** 0.5)

        # Mask with SM: zero out non-neighbours (below median similarity)
        threshold = A.median()
        mask = (A < threshold).unsqueeze(0).expand_as(attn)
        attn = attn.masked_fill(mask, -1e9)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Aggregate
        out = torch.bmm(attn, H_h.permute(1, 0, 2))     # (heads, n, head_dim)
        out = out.permute(1, 0, 2).contiguous().view(H.size(0), -1)  # (n, out_dim)
        return self.layer_norm(F.relu(out) + H)          # residual


class GraphAttentionAutoEncoder(nn.Module):
    """
    GAE for metabolite feature extraction.

    Encoder uses graph-attentional layers informed by structural similarity SM.
    Decoder is a standard MLP (no graph structure needed for reconstruction).
    """

    def __init__(
        self,
        in_dim: int = 2019,
        hidden_dims: tuple = (512, 256, 128),
        latent_dim: int = 64,
        n_heads: int = 4,
        ae_epochs: int = 30,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.ae_epochs = ae_epochs
        self.lr        = lr

        # Encoder: graph-attentional layers
        enc_layers = []
        prev = in_dim
        for h in hidden_dims:
            enc_layers.append(GraphAttentionLayer(prev, h, n_heads=n_heads))
            prev = h
        enc_layers.append(nn.Linear(prev, latent_dim))
        enc_layers.append(nn.ReLU())
        self.encoder_layers = nn.ModuleList(enc_layers[:-2])  # GAT layers
        self.enc_final      = nn.Sequential(*enc_layers[-2:])  # linear + relu

        # Decoder: standard MLP
        dec_dims = list(reversed(hidden_dims)) + [in_dim]
        dec_layers = []
        prev = latent_dim
        for h in dec_dims:
            dec_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        dec_layers[-1] = nn.Sigmoid()  # replace last ReLU with sigmoid
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, X: torch.Tensor, SM: torch.Tensor) -> torch.Tensor:
        """Encode metabolite features through graph-attentional encoder."""
        h = X
        for layer in self.encoder_layers:
            h = layer(h, SM)
        return self.enc_final(h)

    def forward(self, X: torch.Tensor, SM: torch.Tensor):
        z    = self.encode(X, SM)
        recon = self.decoder(z)
        return recon, z
