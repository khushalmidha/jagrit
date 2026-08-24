import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import mlflow
import pickle
import numpy as np

from dl.data.nrms_dataset import NRMSDataset
from dl.model.nrms import NRMS
from sklearn.metrics import roc_auc_score

def train():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Paths
    data_dir = os.path.join(base_dir, 'data', 'raw')
    dl_data_dir = os.path.join(base_dir, 'dl', 'data')
    mlruns_dir = os.path.join(base_dir, 'mlruns')
    checkpoints_dir = os.path.join(base_dir, 'dl', 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    train_behaviors = os.path.join(data_dir, 'MINDsmall_train', 'behaviors.tsv')
    train_news = os.path.join(data_dir, 'MINDsmall_train', 'news.tsv')
    
    # The Kaggle dataset might not have MINDsmall_dev extracted properly. 
    # If not found, we will split the train set for dev.
    dev_behaviors = os.path.join(data_dir, 'MINDsmall_dev', 'behaviors.tsv')
    dev_news = os.path.join(data_dir, 'MINDsmall_dev', 'news.tsv')
    
    if not os.path.exists(dev_behaviors):
        # fallback to train as dev if dev is truly missing
        print("Dev behaviors not found, falling back to train for dev (not ideal but avoids crash)")
        dev_behaviors = train_behaviors
        dev_news = train_news
        
    vocab_path = os.path.join(dl_data_dir, 'word2idx.pkl')
    embed_npy = os.path.join(dl_data_dir, 'embedding_matrix_300d.npy')
    
    # Load vocab and embeddings
    print("Loading vocab and embeddings...")
    with open(vocab_path, 'rb') as f:
        word2idx = pickle.load(f)
    
    vocab_size = len(word2idx)
    
    if os.path.exists(embed_npy):
        word_embeddings = np.load(embed_npy)
    else:
        print("Embedding matrix not found. Please run glove_loader.py first.")
        return
        
    # Hyperparams (Compute Constrained)
    embed_dim = 300
    num_heads = 15
    hidden_dim = 200
    batch_size = 64
    epochs = 1
    lr = 3e-4
    max_title_len = 20
    max_history_len = 30
    num_negatives = 2
    
    # MLflow Setup
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri(f"file://{mlruns_dir}")
    mlflow.set_experiment("nrms-deep-learning")
    
    with mlflow.start_run(run_name="NRMS_Training"):
        mlflow.log_params({
            'embed_dim': embed_dim, 'num_heads': num_heads, 'hidden_dim': hidden_dim,
            'batch_size': batch_size, 'epochs': epochs, 'lr': lr,
            'max_title_len': max_title_len, 'max_history_len': max_history_len,
            'num_negatives': num_negatives
        })
        
        print("Loading datasets...")
        train_dataset = NRMSDataset(
            train_behaviors, train_news, word2idx, 
            max_title_len=max_title_len, max_history_len=max_history_len, 
            num_negatives=num_negatives, is_train=True
        )
        
        # We will use a smaller subset for dev during training just to monitor AUC
        # We'll reuse train dataset but with is_train=False to get validation AUC
        dev_dataset = NRMSDataset(
            dev_behaviors, dev_news, word2idx,
            max_title_len=max_title_len, max_history_len=max_history_len,
            is_train=False
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # using batch size 1 for dev since each impression has variable candidates
        dev_loader = DataLoader(dev_dataset, batch_size=1, shuffle=False)
        
        model = NRMS(
            embed_dim=embed_dim, num_heads=num_heads, hidden_dim=hidden_dim, 
            word_embeddings=word_embeddings, vocab_size=vocab_size
        )
        
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        best_auc = 0.0
        
        print("Starting training loop...")
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            for i, (history, candidates, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                
                # Forward
                scores = model(history, candidates)
                
                loss = criterion(scores, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                if (i+1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/100 (Constrained)], Loss: {loss.item():.4f}")
                
                if i >= 100:
                    print("Reached compute-constrained limit of 100 steps. Finishing epoch.")
                    break
                
                if (i+1) % 50 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
            
            avg_train_loss = total_loss / (i + 1)
            print(f"Epoch {epoch+1} Average Training Loss: {avg_train_loss:.4f}")
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            
            # Dev Evaluation
            model.eval()
            all_aucs = []
            
            # evaluate on a subset (e.g. 1000 impressions) to save time during epoch eval
            # the full rigorous evaluation happens in evaluate_nrms.py
            with torch.no_grad():
                for i, batch in enumerate(dev_loader):
                    if i >= 1000:
                        break
                    
                    history = batch['history']
                    candidates = batch['candidates']
                    labels = batch['labels'].squeeze(0).numpy()
                    
                    if len(np.unique(labels)) < 2:
                        continue # skip all positive or all negative
                        
                    scores = model(history, candidates).squeeze(0).numpy()
                    
                    try:
                        auc = roc_auc_score(labels, scores)
                        all_aucs.append(auc)
                    except ValueError:
                        pass
                        
            avg_dev_auc = np.mean(all_aucs) if all_aucs else 0.0
            print(f"Epoch {epoch+1} Dev Subset AUC: {avg_dev_auc:.4f}")
            mlflow.log_metric("dev_auc_subset", avg_dev_auc, step=epoch)
            
            if avg_dev_auc > best_auc:
                best_auc = avg_dev_auc
                best_model_path = os.path.join(checkpoints_dir, 'nrms_best.pt')
                torch.save(model.state_dict(), best_model_path)
                print(f"Saved best model with AUC: {best_auc:.4f}")
                
        print("Training complete.")
        
if __name__ == '__main__':
    train()
