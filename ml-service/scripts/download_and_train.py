import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pickle
import time
import random

def generate_synthetic_data(num_samples=10000):
    print(f"Generating {num_samples} synthetic interactions for training...")
    categories = ['sports', 'news', 'finance', 'technology', 'entertainment', 'politics']
    
    features = []
    labels = []
    
    for _ in range(num_samples):
        # Synthetic user profile
        hist_len = random.randint(1, 50)
        user_fav_cat = random.choice(categories)
        cat_diversity = random.uniform(0.1, 2.5)
        
        # Synthetic candidate article
        cand_cat = random.choice(categories)
        
        # Calculate features
        cat_match = 1.0 if cand_cat == user_fav_cat else random.uniform(0.0, 0.2)
        article_popularity = random.uniform(0.01, 0.15)
        
        # Logic: High cat match + High popularity = High click probability
        prob_click = (cat_match * 0.6) + (article_popularity * 3) + random.uniform(-0.1, 0.1)
        clicked = 1 if prob_click > 0.4 else 0
        
        features.append([cat_match, hist_len, cat_diversity, article_popularity])
        labels.append(clicked)
        
    return np.array(features), np.array(labels)

def train_model():
    X, y = generate_synthetic_data(num_samples=20000)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training highly optimized XGBoost on {len(X_train)} samples...")
    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    
    print(f"Model Training Complete! Validation AUC: {auc:.4f}")
    
    # Save Model
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "ranker.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        
    print(f"High-Accuracy Model saved to {model_path}")

if __name__ == "__main__":
    start = time.time()
    train_model()
    print(f"Total time: {time.time() - start:.2f}s")
