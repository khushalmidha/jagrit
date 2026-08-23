import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, embed_dim]
        batch_size, seq_len, embed_dim = x.size()
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # scores shape: [batch_size, num_heads, seq_len, seq_len]
        attn = F.softmax(scores, dim=-1)
        
        # weighted sum
        out = torch.matmul(attn, v) # [batch_size, num_heads, seq_len, head_dim]
        
        # reshape back
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        return out

class AdditiveAttention(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(AdditiveAttention, self).__init__()
        self.linear = nn.Linear(embed_dim, hidden_dim)
        self.query = nn.Parameter(torch.randn(hidden_dim))
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, embed_dim]
        # pass through linear + tanh
        h = torch.tanh(self.linear(x)) # [batch_size, seq_len, hidden_dim]
        
        # dot product with query vector
        # query shape: [hidden_dim] -> [batch_size, seq_len]
        alpha = torch.matmul(h, self.query)
        
        # softmax over sequence length
        alpha = F.softmax(alpha, dim=-1) # [batch_size, seq_len]
        
        # weighted sum over sequence length
        out = torch.bmm(alpha.unsqueeze(1), x).squeeze(1) # [batch_size, embed_dim]
        
        return out
