import os
import torch
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from dl.data.nrms_dataset import NRMSDataset
from dl.model.nrms import NRMS
from src.train_ranker import dcg_score, ndcg_score, mrr_score

def evaluate():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    data_dir = os.path.join(base_dir, 'data', 'raw')
    dl_data_dir = os.path.join(base_dir, 'dl', 'data')
    checkpoints_dir = os.path.join(base_dir, 'dl', 'checkpoints')
    results_path = os.path.join(base_dir, 'dl', 'nrms_results.md')
    
    # We evaluate on Dev set (or train set if dev missing)
    dev_behaviors = os.path.join(data_dir, 'MINDsmall_dev', 'behaviors.tsv')
    dev_news = os.path.join(data_dir, 'MINDsmall_dev', 'news.tsv')
    
    if not os.path.exists(dev_behaviors):
        print("Dev behaviors not found, evaluating on train set...")
        dev_behaviors = os.path.join(data_dir, 'MINDsmall_train', 'behaviors.tsv')
        dev_news = os.path.join(data_dir, 'MINDsmall_train', 'news.tsv')
        
    vocab_path = os.path.join(dl_data_dir, 'word2idx.pkl')
    embed_npy = os.path.join(dl_data_dir, 'embedding_matrix_300d.npy')
    best_model_path = os.path.join(checkpoints_dir, 'nrms_best.pt')
    
    with open(vocab_path, 'rb') as f:
        word2idx = pickle.load(f)
    vocab_size = len(word2idx)
    word_embeddings = np.load(embed_npy)
    
    # Init Model
    embed_dim = 300
    num_heads = 15
    hidden_dim = 200
    
    model = NRMS(
        embed_dim=embed_dim, num_heads=num_heads, hidden_dim=hidden_dim,
        word_embeddings=word_embeddings, vocab_size=vocab_size
    )
    
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
        print("Loaded best model checkpoint.")
    else:
        print("No checkpoint found. Please run train_nrms.py first.")
        return
        
    model.eval()
    
    # Load dataset
    max_title_len = 20
    max_history_len = 30
    
    dev_dataset = NRMSDataset(
        dev_behaviors, dev_news, word2idx,
        max_title_len=max_title_len, max_history_len=max_history_len,
        is_train=False
    )
    
    # Batch size 1 because each impression has a variable number of candidates
    dev_loader = DataLoader(dev_dataset, batch_size=1, shuffle=False)
    
    aucs = []
    mrrs = []
    ndcg5s = []
    ndcg10s = []
    
    print("Evaluating on Full Impressions (Apples-to-Apples with XGBoost)...")
    
    with torch.no_grad():
        for i, batch in enumerate(dev_loader):
            history = batch['history']
            candidates = batch['candidates']
            labels = batch['labels'].squeeze(0).numpy()
            
            if len(np.unique(labels)) < 2:
                # skip impression if it only contains positives or only negatives
                continue
                
            scores = model(history, candidates).squeeze(0).numpy()
            
            auc = roc_auc_score(labels, scores)
            mrr = mrr_score(labels, scores)
            ndcg5 = ndcg_score(labels, scores, k=5)
            ndcg10 = ndcg_score(labels, scores, k=10)
            
            aucs.append(auc)
            mrrs.append(mrr)
            ndcg5s.append(ndcg5)
            ndcg10s.append(ndcg10)
            
            if (i+1) % 500 == 0:
                print(f"Evaluated {i+1} impressions...")
                
    final_auc = np.mean(aucs) * 100
    final_mrr = np.mean(mrrs) * 100
    final_ndcg5 = np.mean(ndcg5s) * 100
    final_ndcg10 = np.mean(ndcg10s) * 100
    
    print("\n--- NRMS Evaluation Results ---")
    print(f"AUC:     {final_auc:.2f}")
    print(f"MRR:     {final_mrr:.2f}")
    print(f"nDCG@5:  {final_ndcg5:.2f}")
    print(f"nDCG@10: {final_ndcg10:.2f}")
    
    with open(results_path, 'w') as f:
        f.write("# NRMS Evaluation Results\n\n")
        f.write("| Metric  | Value |\n")
        f.write("|---------|-------|\n")
        f.write(f"| AUC     | {final_auc:.2f} |\n")
        f.write(f"| MRR     | {final_mrr:.2f} |\n")
        f.write(f"| nDCG@5  | {final_ndcg5:.2f} |\n")
        f.write(f"| nDCG@10 | {final_ndcg10:.2f} |\n")
        
    print(f"Results saved to {results_path}")

if __name__ == '__main__':
    evaluate()
