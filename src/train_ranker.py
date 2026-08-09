import os
import pandas as pd
import numpy as np
import mlflow
import pickle
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best if best != 0 else 0.0

def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true) if np.sum(y_true) > 0 else 0.0

def evaluate_impression_level(df, model, feature_cols, k_list=[5, 10]):
    """
    Evaluates ranking model per impression.
    """
    preds = model.predict_proba(df[feature_cols])[:, 1]
    df_eval = df[['impression_id', 'clicked']].copy()
    df_eval['score'] = preds
    
    aucs, mrrs = [], []
    ndcgs = {k: [] for k in k_list}
    
    # Group by impression to calculate ranking metrics
    for imp_id, group in df_eval.groupby('impression_id'):
        y_true = group['clicked'].values
        y_score = group['score'].values
        
        # Only evaluate if there's at least one positive and one negative
        if len(np.unique(y_true)) > 1:
            aucs.append(roc_auc_score(y_true, y_score))
            mrrs.append(mrr_score(y_true, y_score))
            for k in k_list:
                ndcgs[k].append(ndcg_score(y_true, y_score, k))
                
    results = {
        'AUC': np.mean(aucs) if aucs else 0,
        'MRR': np.mean(mrrs) if mrrs else 0,
    }
    for k in k_list:
        results[f'nDCG@{k}'] = np.mean(ndcgs[k]) if ndcgs[k] else 0
        
    return results

def train_models():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    features_path = os.path.join(base_dir, "data", "processed", "features.parquet")
    models_dir = os.path.join(base_dir, "models")
    mlruns_dir = os.path.join(base_dir, "mlruns")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(mlruns_dir, exist_ok=True)
    
    df = pd.read_parquet(features_path)
    
    # Time-based split: MINDsmall_train -> train, MINDsmall_dev -> test
    train_df = df[df['split'] == 'MINDsmall_train']
    test_df = df[df['split'] == 'MINDsmall_dev']
    
    feature_cols = [
        'category_match_score', 'article_popularity', 'article_recency_hours',
        'title_length', 'abstract_length', 'user_total_clicks_in_history', 'user_category_diversity'
    ]
    
    X_train, y_train = train_df[feature_cols], train_df['clicked']
    X_test, y_test = test_df[feature_cols], test_df['clicked']
    
    # MLflow Setup
    mlflow.set_tracking_uri(f"file://{mlruns_dir}")
    mlflow.set_experiment("content-ranking-mind")
    
    print("Training Logistic Regression Baseline...")
    with mlflow.start_run(run_name="Logistic_Regression"):
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train, y_train)
        
        lr_metrics = evaluate_impression_level(test_df, lr, feature_cols)
        mlflow.log_metrics(lr_metrics)
        mlflow.sklearn.log_model(lr, "model")
        
    print("Training XGBoost Classifier...")
    with mlflow.start_run(run_name="XGBoost"):
        xgb_model = xgb.XGBClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=5, 
            use_label_encoder=False, eval_metric='logloss'
        )
        xgb_model.fit(X_train, y_train)
        
        xgb_metrics = evaluate_impression_level(test_df, xgb_model, feature_cols)
        mlflow.log_metrics(xgb_metrics)
        mlflow.xgboost.log_model(xgb_model, "model")
        
        # Save feature importance
        import matplotlib.pyplot.subplots
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        xgb.plot_importance(xgb_model, ax=ax)
        fig.savefig(os.path.join(models_dir, "feature_importance.png"))
        mlflow.log_artifact(os.path.join(models_dir, "feature_importance.png"))
        
        # Save best model locally
        with open(os.path.join(models_dir, "ranker.pkl"), "wb") as f:
            pickle.dump(xgb_model, f)
            
    print("\n--- MODEL COMPARISON ---")
    print(f"{'Metric':<10} | {'LR':<10} | {'XGBoost':<10} | {'LibFM (Pub)':<10} | {'DeepFM (Pub)':<10}")
    print("-" * 65)
    
    # Published Baselines (MIND paper, Wu et al. 2020)
    libfm = {'AUC': 59.74, 'MRR': 26.33, 'nDCG@5': 27.95, 'nDCG@10': 34.29}
    deepfm = {'AUC': 59.89, 'MRR': 26.21, 'nDCG@5': 27.74, 'nDCG@10': 34.06}
    
    for metric in ['AUC', 'MRR', 'nDCG@5', 'nDCG@10']:
        lr_val = lr_metrics[metric] * 100
        xgb_val = xgb_metrics[metric] * 100
        print(f"{metric:<10} | {lr_val:<10.2f} | {xgb_val:<10.2f} | {libfm[metric]:<10.2f} | {deepfm[metric]:<10.2f}")
        
    print("\nNote: Published metrics are * 100 (e.g. 59.74% AUC). Our metrics are scaled here to match.")
    print(f"Saved best model to {os.path.join(models_dir, 'ranker.pkl')}")

if __name__ == "__main__":
    train_models()
