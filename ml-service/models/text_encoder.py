"""
Textual Encoder Module for Multimodal Two-Tower News Recommender.

Architecture:
- Backbone: Token embedding layer (d_emb = 256 or projected to d_model).
- Token-level Multi-Head Self-Attention (heads = 4, d_k = 64 => d_model = 256).
- Additive Attention pooling layer (NRMS-style) to aggregate token representations
  into a single title representation t_i in R^{d_model}.
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """
    Token-level Multi-Head Self-Attention.
    heads = 4, d_k = 64, total dim = 256.
    """

    def __init__(self, d_model: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input token representations [B, seq_len, d_model]
            mask: Optional boolean padding mask [B, seq_len] where True indicates valid tokens.

        Returns:
            Contextual token representations [B, seq_len, d_model]
        """
        residual = x
        b, seq_len, _ = x.size()

        # Project and reshape into heads: [B, num_heads, seq_len, d_k]
        q = self.q_proj(x).view(b, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(b, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(b, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores: [B, num_heads, seq_len, seq_len]

        if mask is not None:
            # mask: [B, seq_len] -> [B, 1, 1, seq_len]
            attn_mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~attn_mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        # Handle cases where entire row was masked
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)  # [B, num_heads, seq_len, d_k]
        context = context.transpose(1, 2).contiguous().view(b, seq_len, self.d_model)

        out = self.out_proj(context)
        out = self.layer_norm(residual + self.dropout(out))
        return out


class AdditiveAttention(nn.Module):
    """
    Additive Attention Pooling layer.
    Computes learnable importance weights for tokens/articles:
        a_i = q^T tanh(W_a * h_i + b_a)
        alpha_i = exp(a_i) / sum_j exp(a_j)
        pooled = sum_i alpha_i * h_i
    """

    def __init__(self, d_model: int = 256, hidden_dim: int = 200):
        super().__init__()
        self.proj = nn.Linear(d_model, hidden_dim)
        self.query = nn.Parameter(torch.randn(hidden_dim))
        # Kaiming uniform init for query
        nn.init.uniform_(self.query, -0.1, 0.1)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, d_model]
            mask: Optional boolean mask [B, seq_len] (True for valid items)

        Returns:
            Pooled representation [B, d_model]
        """
        # [B, seq_len, hidden_dim]
        h = torch.tanh(self.proj(x))

        # Dot product with learnable query vector: [B, seq_len]
        scores = torch.matmul(h, self.query)

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)

        # Weighted sum: [B, 1, seq_len] x [B, seq_len, d_model] -> [B, d_model]
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        return pooled


class TextEncoder(nn.Module):
    """
    Full Textual Tower for News Titles.
    Token Embedding -> Multi-Head Self-Attention -> Additive Attention Pooling.
    """

    def __init__(
        self,
        vocab_size: int = 50000,
        d_emb: int = 256,
        d_model: int = 256,
        num_heads: int = 4,
        additive_hidden_dim: int = 200,
        dropout: float = 0.2,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.padding_idx = padding_idx

        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings,
                freeze=False,
                padding_idx=padding_idx,
            )
            d_emb = pretrained_embeddings.size(1)
        else:
            self.embedding = nn.Embedding(
                vocab_size, d_emb, padding_idx=padding_idx
            )

        # Projection if embedding dim != model dim
        if d_emb != d_model:
            self.input_proj = nn.Linear(d_emb, d_model)
        else:
            self.input_proj = nn.Identity()

        self.self_attention = MultiHeadSelfAttention(
            d_model=d_model, num_heads=num_heads, dropout=dropout
        )
        self.additive_attention = AdditiveAttention(
            d_model=d_model, hidden_dim=additive_hidden_dim
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, token_ids: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            token_ids: Tensor of token indices [B, seq_len]
            mask: Optional boolean mask [B, seq_len] where True indicates real tokens.
                  If None, inferred from token_ids != padding_idx.

        Returns:
            t_i: Title representation vector [B, d_model]
        """
        if mask is None:
            mask = token_ids != self.padding_idx

        # Token embedding: [B, seq_len, d_emb]
        embedded = self.embedding(token_ids)
        x = self.input_proj(embedded)  # [B, seq_len, d_model]
        x = self.dropout(x)

        # Multi-head self-attention: [B, seq_len, d_model]
        attn_out = self.self_attention(x, mask=mask)

        # Additive attention pooling: [B, d_model]
        t_i = self.additive_attention(attn_out, mask=mask)
        t_i = self.layer_norm(t_i)
        t_i = self.dropout(t_i)
        return t_i
