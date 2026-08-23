import torch
import torch.nn as nn
from .attention import MultiHeadSelfAttention, AdditiveAttention

class NewsEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, word_embeddings=None, vocab_size=None):
        super(NewsEncoder, self).__init__()
        
        if word_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(word_embeddings), freeze=False, padding_idx=0
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            
        self.self_attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.additive_attention = AdditiveAttention(embed_dim, hidden_dim)
        
    def forward(self, x):
        # x shape: [batch_size, title_len]
        
        # 1. Word Embeddings
        # [batch_size, title_len, embed_dim]
        embedded = self.embedding(x)
        
        # 2. Multi-Head Self-Attention
        # [batch_size, title_len, embed_dim]
        attn_out = self.self_attention(embedded)
        
        # 3. Additive Attention (Pooling)
        # [batch_size, embed_dim]
        news_vector = self.additive_attention(attn_out)
        
        return news_vector
