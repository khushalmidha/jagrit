import os
import pickle
import pandas as pd
from src.features import FeatureEngineer
from experiments.replay_evaluator import ReplayEvaluator
from experiments.stats import two_proportion_z_test, required_sample_size

def run_experiment():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    features_path = os.path.join(base_dir, "data", "processed", "features.parquet")
    models_dir = os.path.join(base_dir, "models")
    mlruns_dir = os.path.join(base_dir, "mlruns")
    
    if not os.path.exists(features_path):
        print("Features data not found. Please run feature engineering first.")
        return
        
    df = pd.read_parquet(features_path)
    # Evaluate on the dev set
    test_df = df[df['split'] == 'MINDsmall_dev'].copy()
    
    feature_cols = [
        'category_match_score', 'article_popularity', 'article_recency_hours',
        'title_length', 'abstract_length', 'user_total_clicks_in_history', 'user_category_diversity'
    ]
    
    # In a real scenario, we'd load both Model A (LR) and Model B (XGBoost)
    # Here, we'll train a quick LR to act as Model A, and load XGBoost as Model B
    from sklearn.linear_model import LogisticRegression
    
    train_df = df[df['split'] == 'MINDsmall_train']
    
    print("Training Model A (Logistic Regression)...")
    lr = LogisticRegression(max_iter=1000)
    lr.fit(train_df[feature_cols], train_df['clicked'])
    
    print("Loading Model B (XGBoost)...")
    xgb_path = os.path.join(models_dir, "ranker.pkl")
    with open(xgb_path, "rb") as f:
        xgb_model = pickle.load(f)
        
    print("\nRunning Unbiased Replay Evaluation (Li et al. 2011)...")
    eval_A = ReplayEvaluator(test_df.copy(), lr, feature_cols)
    matches_A, clicks_A = eval_A.evaluate()
    
    eval_B = ReplayEvaluator(test_df.copy(), xgb_model, feature_cols)
    matches_B, clicks_B = eval_B.evaluate()
    
    results = two_proportion_z_test(clicks_A, matches_A, clicks_B, matches_B)
    
    print("\n--- A/B TEST RESULTS (REPLAY METHOD) ---")
    print(f"{'Metric':<25} | {'Model A (LR)':<15} | {'Model B (XGBoost)':<15} | {'Lift':<10} | {'p-value':<10}")
    print("-" * 85)
    
    ctr_A_pct = results['ctr_A'] * 100
    ctr_B_pct = results['ctr_B'] * 100
    lift_pct = results['lift'] * 100
    p_val = results['p_value']
    
    print(f"{'Replay CTR':<25} | {ctr_A_pct:<14.2f}% | {ctr_B_pct:<14.2f}% | {lift_pct:>+8.2f}% | {p_val:<10.4f}")
    print(f"{'Matched impressions':<25} | {matches_A:<15} | {matches_B:<15} | {'-':<10} | {'-':<10}")
    
    # Power analysis
    mde = 0.05 # 5% relative lift
    req_n = required_sample_size(results['ctr_A'], mde)
    
    print(f"\nPower Analysis:")
    print(f"To detect a {mde*100}% relative lift with 80% power at alpha=0.05, we would need")
    print(f"{req_n:,} matched impressions per variant.")
    
    # Write to results.md
    res_path = os.path.join(base_dir, "experiments", "results.md")
    with open(res_path, "w") as f:
        f.write("# Offline A/B Test Results (Replay Method)\n\n")
        f.write("## Methodology\n")
        f.write("Evaluation performed using the Replay Method (Li et al. 2011) to ensure unbiased estimation of CTR on logged data. Impressions where the model's top recommendation did not match the logged action were discarded.\n\n")
        f.write("## Results\n")
        f.write("| Metric | Model A (LR) | Model B (XGBoost) | Lift | p-value |\n")
        f.write("|--------|--------------|-------------------|------|---------|\n")
        f.write(f"| Replay CTR | {ctr_A_pct:.2f}% | {ctr_B_pct:.2f}% | {lift_pct:+.2f}% | {p_val:.4f} |\n")
        f.write(f"| Matched Impressions | {matches_A} | {matches_B} | - | - |\n\n")
        f.write("## Conclusion\n")
        f.write(f"Model {'B' if ctr_B_pct > ctr_A_pct else 'A'} performed better. ")
        if p_val < 0.05:
            f.write("The difference is statistically significant (p < 0.05).\n\n")
        else:
            f.write("However, the difference is NOT statistically significant given the sample size.\n\n")
        f.write("## Caveats\n")
        f.write("The Replay Method is data-inefficient because it discards any impression where the model's action doesn't match the logging policy. This reduces the effective sample size significantly, which can make it hard to reach statistical significance without a massive log dataset.\n")

if __name__ == "__main__":
    run_experiment()
