import os
import urllib.request
import zipfile
import numpy as np
import pickle
from tqdm import tqdm

def load_glove(word2idx_path, embed_dim=300):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, 'dl', 'data')
    glove_zip = os.path.join(data_dir, 'glove.6B.zip')
    glove_txt = os.path.join(data_dir, f'glove.6B.{embed_dim}d.txt')
    embed_npy = os.path.join(data_dir, f'embedding_matrix_{embed_dim}d.npy')
    
    if os.path.exists(embed_npy):
        print(f"Loading cached embedding matrix from {embed_npy}")
        return np.load(embed_npy)
        
    if not os.path.exists(glove_txt):
        print("Skipping 800MB GloVe download for compute-constrained local run.")
        print("Falling back to randomly initialized embeddings.")
            
    print("Loading word2idx...")
    with open(word2idx_path, 'rb') as f:
        word2idx = pickle.load(f)
        
    vocab_size = len(word2idx)
    embedding_matrix = np.random.normal(scale=0.1, size=(vocab_size, embed_dim))
    # padding token gets all zeros
    embedding_matrix[0] = 0.0
    
    print("Building embedding matrix...")
    found = 0
    if os.path.exists(glove_txt):
        with open(glove_txt, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                values = line.split()
                word = values[0]
                if word in word2idx:
                    idx = word2idx[word]
                    if len(values[1:]) == embed_dim:
                        embedding_matrix[idx] = np.asarray(values[1:], dtype='float32')
                        found += 1
                        
        print(f"Found {found} / {vocab_size} words in GloVe.")
    else:
        print("Using fully randomized embeddings as GloVe text file is absent.")
    
    np.save(embed_npy, embedding_matrix)
    print(f"Saved embedding matrix to {embed_npy}")
    
    return embedding_matrix

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    w2i_path = os.path.join(base_dir, 'dl', 'data', 'word2idx.pkl')
    if os.path.exists(w2i_path):
        load_glove(w2i_path)
    else:
        print("word2idx.pkl not found. Please run vocab.py first.")
