"""
Temporal Splitting and Feature Prior Statistics for MIND Dataset.

Enforces strict, leakage-free chronological dataset partitioning:
- 80% Chronological Train
- 10% Chronological Validation (for alpha tuning & threshold optimization)
- 10% Chronological Test (untouched final evaluation)

Guarantees:
1. max(T_train) <= min(T_val) <= max(T_val) <= min(T_test)
2. Historical priors (Bayesian CTR, popularity, catalog first-seen timestamps)
   are derived STRICTLY from the Train split. Zero future leakage.
"""

import os
import math
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd

logger = logging.getLogger("TemporalSplit")


def load_temporal_splits(
    behaviors_path: str,
    train_val_test_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    train_samples: Optional[int] = 3000,
    val_samples: Optional[int] = 500,
    test_samples: Optional[int] = 500,
    cache_dir: Optional[str] = "data/processed",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads MIND behaviors.tsv, parses event timestamps, sorts strictly ascending,
    and returns chronologically isolated (train, val, test) DataFrames.

    If subsample sizes are provided, extracts chronological slices from each split
    preserving boundary separation: max(T_train) <= min(T_val) <= max(T_val) <= min(T_test).
    """
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cached_sorted_csv = os.path.join(cache_dir, "behaviors_sorted.feather")
    else:
        cached_sorted_csv = None

    cols = ["impression_id", "user_id", "time", "history", "impressions"]

    if cached_sorted_csv and os.path.exists(cached_sorted_csv):
        logger.info(f"Loading cached sorted behaviors from {cached_sorted_csv}...")
        df = pd.read_feather(cached_sorted_csv)
    else:
        logger.info(f"Loading and sorting behaviors chronologically from {behaviors_path}...")
        df = pd.read_csv(behaviors_path, sep="\t", header=None, names=cols)
        df["datetime"] = pd.to_datetime(df["time"], format="%m/%d/%Y %I:%M:%S %p")
        df = df.sort_values("datetime").reset_index(drop=True)
        if cached_sorted_csv:
            try:
                df.to_feather(cached_sorted_csv)
                logger.info(f"Cached sorted behaviors to {cached_sorted_csv}")
            except Exception as e:
                logger.warning(f"Could not cache to feather: {e}")

    total_rows = len(df)
    r_tr, r_va, r_te = train_val_test_ratio
    n_train = int(total_rows * r_tr)
    n_val = int(total_rows * r_va)

    full_train_df = df.iloc[:n_train].reset_index(drop=True)
    full_val_df = df.iloc[n_train : n_train + n_val].reset_index(drop=True)
    full_test_df = df.iloc[n_train + n_val :].reset_index(drop=True)

    logger.info(
        f"Temporal Partitioning ({total_rows} total sessions):\n"
        f"  - Train Partition: {len(full_train_df)} sessions [{full_train_df['time'].iloc[0]} -> {full_train_df['time'].iloc[-1]}]\n"
        f"  - Val Partition:   {len(full_val_df)} sessions [{full_val_df['time'].iloc[0]} -> {full_val_df['time'].iloc[-1]}]\n"
        f"  - Test Partition:  {len(full_test_df)} sessions [{full_test_df['time'].iloc[0]} -> {full_test_df['time'].iloc[-1]}]"
    )

    # Apply subsamples if specified for fast reproducible benchmarks
    train_df = full_train_df.iloc[:train_samples] if train_samples and train_samples > 0 else full_train_df
    val_df = full_val_df.iloc[:val_samples] if val_samples and val_samples > 0 else full_val_df
    test_df = full_test_df.iloc[:test_samples] if test_samples and test_samples > 0 else full_test_df

    logger.info(
        f"Active Experiment Slices:\n"
        f"  - Train: {len(train_df)} sessions\n"
        f"  - Val:   {len(val_df)} sessions\n"
        f"  - Test:  {len(test_df)} sessions"
    )
    return train_df, val_df, test_df


def compute_train_prior_statistics(
    train_df: pd.DataFrame,
    output_path: Optional[str] = "checkpoints/train_feature_priors.json",
) -> Dict[str, Any]:
    """
    Computes catalog statistics STRICTLY from the training partition:
    - Article click counts and impression counts
    - Global baseline CTR prior p
    - First appearance epoch timestamp per article
    """
    article_impressions: Dict[str, int] = {}
    article_clicks: Dict[str, int] = {}
    article_first_seen: Dict[str, float] = {}

    total_clicks = 0
    total_impressions = 0

    for _, row in train_df.iterrows():
        dt_val = row["datetime"] if "datetime" in row else pd.to_datetime(row["time"], format="%m/%d/%Y %I:%M:%S %p")
        t_epoch = dt_val.timestamp()

        # Update first seen from user browsing history
        hist_str = str(row["history"])
        if hist_str and hist_str != "nan":
            for nid in hist_str.split():
                if nid not in article_first_seen or t_epoch < article_first_seen[nid]:
                    article_first_seen[nid] = t_epoch

        # Update impressions, clicks, and first seen from session candidate impressions
        imp_str = str(row["impressions"])
        if imp_str and imp_str != "nan":
            for imp in imp_str.split():
                parts = imp.split("-")
                if len(parts) == 2:
                    nid, lbl_str = parts
                    lbl = int(lbl_str)
                    article_impressions[nid] = article_impressions.get(nid, 0) + 1
                    article_clicks[nid] = article_clicks.get(nid, 0) + lbl
                    total_impressions += 1
                    total_clicks += lbl
                    if nid not in article_first_seen or t_epoch < article_first_seen[nid]:
                        article_first_seen[nid] = t_epoch

    global_ctr_prior = float(total_clicks / max(1, total_impressions))
    min_train_t = float(train_df["datetime"].iloc[0].timestamp()) if "datetime" in train_df else 0.0

    stats = {
        "global_ctr_prior": global_ctr_prior,
        "total_train_impressions": total_impressions,
        "total_train_clicks": total_clicks,
        "article_impressions": article_impressions,
        "article_clicks": article_clicks,
        "article_first_seen": article_first_seen,
        "min_train_timestamp": min_train_t,
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(stats, f)
        logger.info(f"Saved leak-free train feature priors (global CTR: {global_ctr_prior:.4f}) to {output_path}")

    return stats


def extract_ranking_features(
    sim: float,
    cat_affinity: float,
    subcat_affinity: float,
    cat_entropy: float,
    session_timestamp: float,
    article_id: str,
    priors: Dict[str, Any],
    smoothing_pseudo_counts: float = 10.0,
) -> List[float]:
    """
    Extracts deterministic ranking features for Stage 2 LambdaMART reranking
    without any random variables or magic hardcoded numbers.

    Features (8):
    1. Cosine similarity s(u, i)
    2. User-category affinity
    3. User-subcategory affinity
    4. User category entropy
    5. Non-linear recency decay exp(-delta_hours / 24.0)
    6. Bayesian smoothed historical CTR
    7. Historical popularity log1p(impressions)
    8. Interaction cross-term sim * cat_affinity
    """
    # 1. Recency Decay
    first_seen = priors.get("article_first_seen", {}).get(article_id)
    if first_seen is None:
        # Article first appeared in this impression (delta = 0)
        delta_hours = 0.0
    else:
        delta_seconds = max(0.0, session_timestamp - float(first_seen))
        delta_hours = delta_seconds / 3600.0
    exp_decay = math.exp(-delta_hours / 24.0)

    # 2. Bayesian Smoothed CTR Prior
    global_p = priors.get("global_ctr_prior", 0.04)
    item_imps = priors.get("article_impressions", {}).get(article_id, 0)
    item_clicks = priors.get("article_clicks", {}).get(article_id, 0)
    bayesian_ctr = (item_clicks + smoothing_pseudo_counts * global_p) / float(item_imps + smoothing_pseudo_counts)

    # 3. Popularity Prior
    popularity = math.log1p(float(item_imps))

    # 4. Cross term
    cross_term = sim * cat_affinity

    return [
        float(sim),
        float(cat_affinity),
        float(subcat_affinity),
        float(cat_entropy),
        float(exp_decay),
        float(bayesian_ctr),
        float(popularity),
        float(cross_term),
    ]
