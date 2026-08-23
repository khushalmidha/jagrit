import torch
import torch.nn as nn
from .attention import MultiHeadSelfAttention, AdditiveAttention

class UserEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super(UserEncoder, self).__init__()
        
        self.self_attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.additive_attention = AdditiveAttention(embed_dim, hidden_dim)
        
    def forward(self, news_vectors):
        # news_vectors shape: [batch_size, history_len, embed_dim]
        
        # 1. Multi-Head Self-Attention over history
        # [batch_size, history_len, embed_dim]
        attn_out = self.self_attention(news_vectors)
        
        # 2. Additive Attention (Pooling)
        # [batch_size, embed_dim]
        user_vector = self.additive_attention(attn_out)
        
        return user_vector
