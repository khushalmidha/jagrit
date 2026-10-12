"""
Two-Tower Multimodal News Recommendation Model.

Architecture & Mathematical Specifications:
1. Article Tower (ArticleEncoder):
   - Visual Tower (ResNet-50) + Textual Tower (MHA + Additive Pooling)
   - Gated Multimodal Fusion: e_i = g * W_v(v_i) + (1-g) * W_t(t_i)
   - L2 normalized: e_i = e_i / ||e_i||_2

2. User Tower (UserEncoder):
   - Sequence of last K clicked article embeddings {e_1, ..., e_K}
   - Multi-Head Self-Attention over history sequence (heads=4, d_k=64)
   - Additive Attention pooling layer to aggregate into user vector u_u
   - L2 normalized: u_u = u_u / ||u_u||_2

3. Stage 1 Scoring & Loss:
   - Relevance score: s(u, i) = cos(u_u, e_i) = u_u . e_i
   - InfoNCE Loss with tau = 0.07:
     L = - log( exp(s(u, pos)/tau) / [exp(s(u, pos)/tau) + sum(exp(s(u, neg)/tau))] )
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .visual_encoder import VisualEncoder
from .text_encoder import TextEncoder, MultiHeadSelfAttention, AdditiveAttention
from .multimodal_fusion import GatedMultimodalFusion


class UserEncoder(nn.Module):
    """
    User Modeling Tower.
    Aggregates user's historical clicked article embeddings using
    Multi-Head Self-Attention and Additive Attention pooling.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 4,
        additive_hidden_dim: int = 200,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.d_model = d_model
        self.self_attention = MultiHeadSelfAttention(
            d_model=d_model, num_heads=num_heads, dropout=dropout
        )
        self.additive_attention = AdditiveAttention(
            d_model=d_model, hidden_dim=additive_hidden_dim
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        history_embeddings: torch.Tensor,
        history_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            history_embeddings: [B, K, d_model] embeddings of last K clicked articles.
            history_mask: Optional boolean mask [B, K] (True for valid clicks).

        Returns:
            u_u: User representation vector [B, d_model].
        """
        # Multi-head self-attention over history articles: [B, K, d_model]
        attn_out = self.self_attention(history_embeddings, mask=history_mask)

        # Additive attention pooling across history items: [B, d_model]
        user_vector = self.additive_attention(attn_out, mask=history_mask)
        user_vector = self.layer_norm(user_vector)
        user_vector = self.dropout(user_vector)
        return user_vector


class ArticleEncoder(nn.Module):
    """
    Multimodal Article Tower.
    Combines VisualEncoder + TextEncoder via GatedMultimodalFusion.
    """

    def __init__(
        self,
        d_model: int = 256,
        vocab_size: int = 50000,
        d_emb: int = 256,
        num_heads: int = 4,
        additive_hidden_dim: int = 200,
        dropout: float = 0.2,
        pretrained_vision: bool = True,
        freeze_vision_blocks: bool = True,
        use_gating: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.visual_encoder = VisualEncoder(
            d_model=d_model,
            dropout=dropout,
            pretrained=pretrained_vision,
            freeze_lower_blocks=freeze_vision_blocks,
        )
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            d_emb=d_emb,
            d_model=d_model,
            num_heads=num_heads,
            additive_hidden_dim=additive_hidden_dim,
            dropout=dropout,
        )
        self.fusion = GatedMultimodalFusion(
            d_model=d_model,
            dropout=dropout,
            use_gating=use_gating,
        )

    def forward(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            images: [N, 3, 224, 224] or precomputed [N, 2048]
            token_ids: [N, seq_len]
            text_mask: Optional [N, seq_len]

        Returns:
            e_i: Article embedding [N, d_model]
        """
        v_i = self.visual_encoder(images)
        t_i = self.text_encoder(token_ids, mask=text_mask)
        e_i = self.fusion(v_i, t_i)
        return e_i


class TwoTowerNewsRecommender(nn.Module):
    """
    End-to-End Multimodal Two-Tower News Recommendation Network.
    """

    def __init__(
        self,
        d_model: int = 256,
        vocab_size: int = 50000,
        d_emb: int = 256,
        num_heads: int = 4,
        additive_hidden_dim: int = 200,
        dropout: float = 0.2,
        temperature: float = 0.07,
        pretrained_vision: bool = True,
        freeze_vision_blocks: bool = True,
        use_gating: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature

        self.article_encoder = ArticleEncoder(
            d_model=d_model,
            vocab_size=vocab_size,
            d_emb=d_emb,
            num_heads=num_heads,
            additive_hidden_dim=additive_hidden_dim,
            dropout=dropout,
            pretrained_vision=pretrained_vision,
            freeze_vision_blocks=freeze_vision_blocks,
            use_gating=use_gating,
        )
        self.user_encoder = UserEncoder(
            d_model=d_model,
            num_heads=num_heads,
            additive_hidden_dim=additive_hidden_dim,
            dropout=dropout,
        )

    def encode_articles(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Generates multimodal article embeddings e_i in R^{d_model}.
        """
        e_i = self.article_encoder(images, token_ids, text_mask=text_mask)
        if normalize:
            e_i = F.normalize(e_i, p=2, dim=-1)
        return e_i

    def encode_user(
        self,
        history_embeddings: torch.Tensor,
        history_mask: Optional[torch.Tensor] = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Generates user embedding u_u in R^{d_model} from sequence of clicked article embeddings.
        """
        u_u = self.user_encoder(history_embeddings, history_mask=history_mask)
        if normalize:
            u_u = F.normalize(u_u, p=2, dim=-1)
        return u_u

    def forward(
        self,
        history_images: torch.Tensor,
        history_tokens: torch.Tensor,
        candidate_images: torch.Tensor,
        candidate_tokens: torch.Tensor,
        history_mask: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Training forward pass.
        Args:
            history_images: [B, K, 3, 224, 224] or precomputed [B, K, 2048]
            history_tokens: [B, K, seq_len]
            candidate_images: [B, C, 3, 224, 224] or precomputed [B, C, 2048] (C = 1 pos + N neg)
            candidate_tokens: [B, C, seq_len]
            history_mask: [B, K] boolean mask (True for valid clicks)
            candidate_mask: Optional [B, C] mask

        Returns:
            logits: Scaled cosine similarity scores [B, C] for InfoNCE loss: s / tau.
            user_embeddings: L2-normalized user vectors [B, d_model].
            candidate_embeddings: L2-normalized candidate vectors [B, C, d_model].
        """
        b, k = history_tokens.size(0), history_tokens.size(1)
        c = candidate_tokens.size(1)
        seq_len = history_tokens.size(2)

        # 1. Encode user history articles
        # Reshape to flatten [B * K, ...]
        if history_images.dim() == 5:
            hist_img_flat = history_images.view(b * k, 3, 224, 224)
        else:
            hist_img_flat = history_images.view(b * k, -1)
        hist_tok_flat = history_tokens.view(b * k, seq_len)

        hist_embeds_flat = self.article_encoder(hist_img_flat, hist_tok_flat)
        hist_embeds = hist_embeds_flat.view(b, k, self.d_model)

        # 2. Encode user vector u_u: [B, d_model]
        user_emb = self.user_encoder(hist_embeds, history_mask=history_mask)
        user_emb_norm = F.normalize(user_emb, p=2, dim=-1)

        # 3. Encode candidate articles: [B * C, d_model]
        cand_seq_len = candidate_tokens.size(2)
        if candidate_images.dim() == 5:
            cand_img_flat = candidate_images.view(b * c, 3, 224, 224)
        else:
            cand_img_flat = candidate_images.view(b * c, -1)
        cand_tok_flat = candidate_tokens.view(b * c, cand_seq_len)

        cand_embeds_flat = self.article_encoder(cand_img_flat, cand_tok_flat)
        cand_embeds = cand_embeds_flat.view(b, c, self.d_model)
        cand_embeds_norm = F.normalize(cand_embeds, p=2, dim=-1)

        # 4. Cosine similarity: s(u, i) = u_u . e_i
        # user_emb_norm: [B, d_model, 1]
        # cand_embeds_norm: [B, C, d_model]
        # bmm([B, C, d_model], [B, d_model, 1]) -> [B, C, 1] -> squeeze to [B, C]
        cosine_sim = torch.bmm(cand_embeds_norm, user_emb_norm.unsqueeze(-1)).squeeze(-1)

        # 5. Logits scaled by temperature tau = 0.07 for InfoNCE
        logits = cosine_sim / self.temperature

        return logits, user_emb_norm, cand_embeds_norm
