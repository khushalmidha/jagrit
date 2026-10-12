"""
Multimodal Article Fusion Module for Multimodal Two-Tower News Recommender.

Mathematical Specifications:
1. Gated Multimodal Fusion (Default):
   g = sigma(W_g * [v_i, t_i] + b_g)
   e_i = g * (W_v * v_i) + (1 - g) * (W_t * t_i)
   e_i = LayerNorm(e_i)

2. Linear Concat Baseline:
   e_i = LayerNorm(Linear(concat([v_i, t_i]) -> d_model))
"""

import torch
import torch.nn as nn


class GatedMultimodalFusion(nn.Module):
    """
    Gated Cross-Modal Fusion module combining visual and textual representations.
    """

    def __init__(
        self,
        d_model: int = 256,
        dropout: float = 0.2,
        use_gating: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_gating = use_gating

        if use_gating:
            # Gating network: W_g * [v_i, t_i] + b_g -> d_model
            self.w_gate = nn.Linear(2 * d_model, d_model)
            # Modal-specific projections
            self.w_v = nn.Linear(d_model, d_model)
            self.w_t = nn.Linear(d_model, d_model)
        else:
            # Direct concatenation projection
            self.concat_proj = nn.Linear(2 * d_model, d_model)

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, v_i: torch.Tensor, t_i: torch.Tensor) -> torch.Tensor:
        """
        Args:
            v_i: Visual embedding tensor [B, d_model]
            t_i: Textual embedding tensor [B, d_model]

        Returns:
            e_i: Fused article embedding [B, d_model]
        """
        assert v_i.shape == t_i.shape, (
            f"Dimension mismatch between visual ({v_i.shape}) and textual ({t_i.shape}) embeddings."
        )

        concat_repr = torch.cat([v_i, t_i], dim=-1)  # [B, 2 * d_model]

        if self.use_gating:
            # Learnable gate: g in (0, 1) [B, d_model]
            g = torch.sigmoid(self.w_gate(concat_repr))

            # Transformed visual and textual features
            h_v = self.w_v(v_i)
            h_t = self.w_t(t_i)

            # Gated combination
            fused = g * h_v + (1.0 - g) * h_t
        else:
            fused = self.concat_proj(concat_repr)

        # LayerNorm and Dropout
        e_i = self.layer_norm(fused)
        e_i = self.dropout(e_i)
        return e_i
