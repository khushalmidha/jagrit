import os
import pandas as pd
from collections import Counter
import pickle

def build_vocab(max_vocab_size=50000, min_freq=2):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raw_dir = os.path.join(base_dir, 'data', 'raw')
    
    train_news_path = os.path.join(raw_dir, 'MINDsmall_train', 'news.tsv')
    dev_news_path = os.path.join(raw_dir, 'MINDsmall_dev', 'news.tsv')
    
    # We might not have these files if they haven't finished downloading.
    # We will assume they will exist when this is run.
    
    news_cols = ['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']
    
    word_freq = Counter()
    
    print("Building vocabulary from news titles...")
    for path in [train_news_path, dev_news_path]:
        if os.path.exists(path):
            df = pd.read_csv(path, sep='\t', header=None, names=news_cols)
            for title in df['title'].dropna():
                # basic tokenization: split on whitespace + strip punctuation (using a simple approach)
                words = title.lower().split()
                # further strip basic punctuation
                words = [w.strip('.,!?"\'():;') for w in words]
                words = [w for w in words if w]
                word_freq.update(words)
        else:
            print(f"Warning: {path} not found. Skipping.")
            
    # Filter by frequency
    valid_words = [w for w, c in word_freq.items() if c >= min_freq]
    
    # Sort by frequency
    valid_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)
    
    # Cap size
    valid_words = valid_words[:max_vocab_size-2] # leaving room for PAD and UNK
    
    word2idx = {'<PAD>': 0, '<UNK>': 1}
    for idx, word in enumerate(valid_words, start=2):
        word2idx[word] = idx
        
    print(f"Vocabulary size: {len(word2idx)}")
    
    vocab_path = os.path.join(base_dir, 'dl', 'data', 'word2idx.pkl')
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
    with open(vocab_path, 'wb') as f:
        pickle.dump(word2idx, f)
        
    return word2idx

if __name__ == '__main__':
    build_vocab()
