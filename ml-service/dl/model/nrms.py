import torch
import torch.nn as nn
from .news_encoder import NewsEncoder
from .user_encoder import UserEncoder

class NRMS(nn.Module):
    def __init__(self, embed_dim=300, num_heads=15, hidden_dim=200, word_embeddings=None, vocab_size=None):
        super(NRMS, self).__init__()
        
        self.news_encoder = NewsEncoder(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            hidden_dim=hidden_dim, 
            word_embeddings=word_embeddings, 
            vocab_size=vocab_size
        )
        
        self.user_encoder = UserEncoder(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            hidden_dim=hidden_dim
        )
        
    def forward(self, history, candidates):
        """
        history: [batch_size, history_len, title_len]
        candidates: [batch_size, num_candidates, title_len]
        """
        batch_size, history_len, title_len = history.size()
        num_candidates = candidates.size(1)
        
        # Flatten history for news encoder
        # [batch_size * history_len, title_len]
        history_flat = history.view(-1, title_len)
        history_enc = self.news_encoder(history_flat)
        # [batch_size, history_len, embed_dim]
        history_enc = history_enc.view(batch_size, history_len, -1)
        
        # Encode User
        # [batch_size, embed_dim]
        user_vector = self.user_encoder(history_enc)
        
        # Flatten candidates for news encoder
        # [batch_size * num_candidates, title_len]
        candidates_flat = candidates.view(-1, title_len)
        candidates_enc = self.news_encoder(candidates_flat)
        # [batch_size, num_candidates, embed_dim]
        candidates_enc = candidates_enc.view(batch_size, num_candidates, -1)
        
        # Dot product scoring
        # user_vector: [batch_size, embed_dim] -> [batch_size, embed_dim, 1]
        user_vector = user_vector.unsqueeze(-1)
        
        # bmm( [batch_size, num_candidates, embed_dim], [batch_size, embed_dim, 1] )
        # -> [batch_size, num_candidates, 1]
        scores = torch.bmm(candidates_enc, user_vector).squeeze(-1)
        
        return scores
