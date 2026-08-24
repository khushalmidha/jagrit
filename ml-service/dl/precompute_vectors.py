import os
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from dl.model.nrms import NRMS

def precompute_vectors():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'dl', 'checkpoints', 'nrms_best.pt')
    vocab_path = os.path.join(base_dir, 'dl', 'data', 'word2idx.pkl')
    embed_npy = os.path.join(base_dir, 'dl', 'data', 'embedding_matrix_300d.npy')
    
    news_path = os.path.join(base_dir, 'data', 'raw', 'MINDsmall_train', 'news.tsv')
    behaviors_path = os.path.join(base_dir, 'data', 'raw', 'MINDsmall_train', 'behaviors.tsv')
    
    if not os.path.exists(model_path):
        print("Model checkpoint not found. Run train_nrms.py first.")
        return
        
    print("Loading vocab and embeddings...")
    with open(vocab_path, 'rb') as f:
        word2idx = pickle.load(f)
    word_embeddings = np.load(embed_npy)
    vocab_size = len(word2idx)
    
    print("Loading model...")
    embed_dim = 300
    num_heads = 15
    hidden_dim = 200
    max_title_len = 20
    max_history_len = 30
    
    model = NRMS(
        embed_dim=embed_dim, num_heads=num_heads, hidden_dim=hidden_dim, 
        word_embeddings=word_embeddings, vocab_size=vocab_size
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    def tokenize(text):
        words = text.lower().split()
        words = [w.strip('.,!?"\'():;') for w in words]
        words = [w for w in words if w]
        tokens = [word2idx.get(w, word2idx['<UNK>']) for w in words]
        if len(tokens) > max_title_len:
            return tokens[:max_title_len]
        return tokens + [word2idx['<PAD>']] * (max_title_len - len(tokens))

    print("Precomputing News Vectors...")
    news_cols = ['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']
    df_news = pd.read_csv(news_path, sep='\t', header=None, names=news_cols)
    
    news_vectors = {}
    with torch.no_grad():
        for _, row in tqdm(df_news.iterrows(), total=len(df_news)):
            news_id = row['news_id']
            title = str(row['title'])
            tokens = torch.tensor([tokenize(title)])
            vec = model.news_encoder(tokens).squeeze(0).numpy()
            news_vectors[news_id] = vec

    print("Precomputing User Vectors (from history)...")
    behavior_cols = ['impression_id', 'user_id', 'time', 'history', 'impressions']
    df_beh = pd.read_csv(behaviors_path, sep='\t', header=None, names=behavior_cols)
    
    user_vectors = {}
    pad_title = [word2idx['<PAD>']] * max_title_len
    with torch.no_grad():
        for _, row in tqdm(df_beh.iterrows(), total=len(df_beh)):
            user_id = row['user_id']
            if user_id in user_vectors:
                continue
                
            hist_str = str(row['history'])
            hist_ids = hist_str.split() if hist_str != 'nan' else []
            
            # Reconstruct word ids
            hist_tokens = []
            for h in hist_ids[-max_history_len:]:
                title = df_news.loc[df_news['news_id'] == h, 'title']
                if not title.empty:
                    hist_tokens.append(tokenize(str(title.values[0])))
                else:
                    hist_tokens.append(pad_title)
                    
            while len(hist_tokens) < max_history_len:
                hist_tokens.insert(0, pad_title)
                
            hist_tensor = torch.tensor([hist_tokens])
            user_vec = model.user_encoder(hist_tensor).squeeze(0).numpy()
            user_vectors[user_id] = user_vec

    print("Saving precomputed vectors...")
    with open(os.path.join(base_dir, 'dl', 'data', 'nrms_news_vectors.pkl'), 'wb') as f:
        pickle.dump(news_vectors, f)
        
    with open(os.path.join(base_dir, 'dl', 'data', 'nrms_user_vectors.pkl'), 'wb') as f:
        pickle.dump(user_vectors, f)
        
    print("Done!")

if __name__ == '__main__':
    precompute_vectors()
