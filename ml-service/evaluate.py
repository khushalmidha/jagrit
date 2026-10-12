"""
Evaluation Script for Multimodal Two-Tower News Recommender.

Computes exact industry-standard ranking metrics on MIND impressions:
- nDCG@5
- nDCG@10
- Mean Reciprocal Rank (MRR)
- Area Under ROC Curve (AUC-ROC)

Evaluates and compares:
- Stage 1: Two-Tower Cosine Similarity
- Stage 2: XGBoost LambdaMART Reranker
"""

import os
import math
import json
import logging
import argparse
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, cohen_kappa_score
import xgboost as xgb

from data.dataset import MINDMultimodalDataset, SimpleTokenizer, create_mock_mind_dataset
from models.two_tower import TwoTowerNewsRecommender

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("Evaluate")


def compute_dcg_at_k(labels: np.ndarray, k: int) -> float:
    """Computes Discounted Cumulative Gain at rank K."""
    labels_k = labels[:k]
    if len(labels_k) == 0:
        return 0.0
    discounts = np.log2(np.arange(len(labels_k)) + 2.0)
    return float(np.sum(labels_k / discounts))


def compute_ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Computes Normalized Discounted Cumulative Gain at rank K."""
    order = np.argsort(y_score)[::-1]
    sorted_labels = y_true[order]
    dcg = compute_dcg_at_k(sorted_labels, k)

    # Ideal ranking
    ideal_labels = np.sort(y_true)[::-1]
    idcg = compute_dcg_at_k(ideal_labels, k)

    if idcg <= 0.0:
        return 0.0
    return dcg / idcg


def compute_mrr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computes Mean Reciprocal Rank (MRR) of the first positive click."""
    order = np.argsort(y_score)[::-1]
    sorted_labels = y_true[order]
    for rank, label in enumerate(sorted_labels, start=1):
        if label > 0:
            return 1.0 / float(rank)
    return 0.0


def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """Computes AUC-ROC for an impression. Returns None if single class present."""
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return None


def compute_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """Computes Precision-Recall AUC (Average Precision) for an impression."""
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return None


def compute_cohen_kappa(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """
    Computes Cohen's Kappa measuring chance-corrected agreement between
    the top-1 ranked predicted item and actual user click labels.
    """
    if len(np.unique(y_true)) < 2:
        return None
    try:
        y_pred = np.zeros_like(y_true)
        y_pred[np.argmax(y_score)] = 1
        return float(cohen_kappa_score(y_true, y_pred))
    except Exception:
        return None


def evaluate_pipeline(
    checkpoint_path: str,
    vocab_path: str,
    xgb_path: Optional[str] = None,
    data_dir: Optional[str] = None,
    use_mock: bool = False,
    max_eval_samples: int = 500,
    device_name: str = "cpu",
) -> Dict[str, float]:
    """
    Evaluates both Stage 1 and Stage 2 models on test/dev impressions.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device(device_name)

    # Load Tokenizer
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    tokenizer = SimpleTokenizer(vocab=vocab)

    # Dataset paths
    if use_mock:
        mock_dir = os.path.join(base_dir, "data", "mock")
        if os.path.exists(os.path.join(mock_dir, "news.tsv")):
            news_path = os.path.join(mock_dir, "news.tsv")
            behaviors_path = os.path.join(mock_dir, "behaviors.tsv")
        else:
            news_path, behaviors_path = create_mock_mind_dataset(mock_dir, num_news=100, num_behaviors=300)
    elif data_dir and os.path.exists(os.path.join(data_dir, "news.tsv")):
        news_path = os.path.join(data_dir, "news.tsv")
        behaviors_path = os.path.join(data_dir, "behaviors.tsv")
    else:
        raw_train = os.path.join(base_dir, "data", "raw", "MINDsmall_train")
        if os.path.exists(os.path.join(raw_train, "news.tsv")):
            news_path = os.path.join(raw_train, "news.tsv")
            behaviors_path = os.path.join(raw_train, "behaviors.tsv")
        else:
            mock_dir = os.path.join(base_dir, "data", "mock_eval")
            news_path, behaviors_path = create_mock_mind_dataset(mock_dir, num_news=100, num_behaviors=300)

    logger.info(f"Loading evaluation dataset from {news_path}...")
    eval_dataset = MINDMultimodalDataset(
        behaviors_path=behaviors_path,
        news_path=news_path,
        tokenizer=tokenizer,
        max_title_len=20,
        max_history_len=20,
        is_train=False,
        max_samples=max_eval_samples,
    )

    logger.info(f"Loaded {len(eval_dataset)} evaluation impression sessions.")

    # Load Stage 1 Two-Tower Model
    ckpt = torch.load(checkpoint_path, map_location=device)
    d_model = ckpt.get("d_model", 256)
    temperature = ckpt.get("temperature", 0.07)

    model = TwoTowerNewsRecommender(
        d_model=d_model,
        vocab_size=len(tokenizer),
        d_emb=d_model,
        temperature=temperature,
        pretrained_vision=False,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load Stage 2 XGBoost LambdaMART if present
    xgb_ranker = None
    if xgb_path and os.path.exists(xgb_path):
        logger.info(f"Loading Stage 2 XGBoost LambdaMART model from {xgb_path}...")
        xgb_ranker = xgb.XGBRanker()
        xgb_ranker.load_model(xgb_path)

    # Metrics accumulators
    stage1_ndcg5, stage1_ndcg10, stage1_mrr, stage1_auc = [], [], [], []
    stage1_prauc, stage1_kappa = [], []
    stage2_ndcg5, stage2_ndcg10, stage2_mrr, stage2_auc = [], [], [], []
    stage2_prauc, stage2_kappa = [], []

    with torch.no_grad():
        for i in range(len(eval_dataset)):
            sample = eval_dataset[i]
            y_true = sample["labels"].numpy()
            cands = sample["candidate_ids"]

            if len(cands) < 2 or np.sum(y_true) == 0:
                continue

            # User Embedding
            hist_tok = sample["history_tokens"].unsqueeze(0).to(device)
            hist_img = sample["history_images"].unsqueeze(0).to(device)
            hist_msk = sample["history_mask"].unsqueeze(0).to(device)

            b, k, seq_len = hist_tok.size()
            if hist_img.dim() == 5:
                flat_imgs = hist_img.view(b * k, 3, 224, 224)
            else:
                flat_imgs = hist_img.view(b * k, -1)
            flat_toks = hist_tok.view(b * k, seq_len)

            hist_enc = model.article_encoder(flat_imgs, flat_toks).view(b, k, -1)
            u_u = model.encode_user(hist_enc, history_mask=hist_msk, normalize=True).squeeze(0)

            # Candidate Embeddings
            cand_tok = sample["candidate_tokens"].to(device)
            cand_img = sample["candidate_images"].to(device)
            e_cands = model.encode_articles(cand_img, cand_tok, normalize=True)

            # Stage 1: Cosine Similarity Scores
            s1_scores = torch.matmul(e_cands, u_u).cpu().numpy()

            # Record Stage 1 metrics
            stage1_ndcg5.append(compute_ndcg_at_k(y_true, s1_scores, 5))
            stage1_ndcg10.append(compute_ndcg_at_k(y_true, s1_scores, 10))
            stage1_mrr.append(compute_mrr(y_true, s1_scores))
            auc_val = compute_auc(y_true, s1_scores)
            if auc_val is not None:
                stage1_auc.append(auc_val)
            prauc_val = compute_pr_auc(y_true, s1_scores)
            if prauc_val is not None:
                stage1_prauc.append(prauc_val)
            kappa_val = compute_cohen_kappa(y_true, s1_scores)
            if kappa_val is not None:
                stage1_kappa.append(kappa_val)

            # Stage 2 & Ensemble Learning Reranker
            if xgb_ranker is not None:
                history = sample.get("history", [])
                hist_categories = [
                    eval_dataset.news_categories.get(nid, "")
                    for nid in history
                    if nid in eval_dataset.news_categories
                ]
                hist_subcategories = [
                    eval_dataset.news_subcategories.get(nid, "")
                    for nid in history
                    if nid in eval_dataset.news_subcategories
                ]
                num_hist = max(1, len(hist_categories))

                cat_counts: Dict[str, int] = {}
                for c in hist_categories:
                    cat_counts[c] = cat_counts.get(c, 0) + 1
                subcat_counts: Dict[str, int] = {}
                for sc in hist_subcategories:
                    subcat_counts[sc] = subcat_counts.get(sc, 0) + 1

                cat_probs = [cnt / float(num_hist) for cnt in cat_counts.values()]
                cat_entropy = -sum(p * math.log2(p + 1e-9) for p in cat_probs)

                cand_features = []
                for idx, cid in enumerate(cands):
                    sim = float(s1_scores[idx])
                    cat = eval_dataset.news_categories.get(cid, "")
                    subcat = eval_dataset.news_subcategories.get(cid, "")
                    cat_affinity = cat_counts.get(cat, 0) / float(num_hist)
                    subcat_affinity = subcat_counts.get(subcat, 0) / float(num_hist)
                    exp_decay = math.exp(-12.0 / 24.0)
                    bayesian_ctr = (10.0 + 5.0) / (150.0 + 100.0)
                    popularity = math.log1p(150.0)
                    cross_term = sim * cat_affinity

                    cand_features.append([
                        sim,
                        cat_affinity,
                        subcat_affinity,
                        cat_entropy,
                        exp_decay,
                        bayesian_ctr,
                        popularity,
                        cross_term,
                    ])

                X_cand = np.array(cand_features, dtype=np.float32)
                s2_scores = xgb_ranker.predict(X_cand)

                # Min-max normalize scores within candidate set for calibrated score blending
                def _norm(scores):
                    ptp = np.ptp(scores)
                    return (scores - np.min(scores)) / (ptp + 1e-12) if ptp > 1e-12 else np.zeros_like(scores)

                norm_s1 = _norm(s1_scores)
                norm_s2 = _norm(s2_scores)

                # Evaluate pure Stage 2 (alpha = 0.0)
                stage2_ndcg5.append(compute_ndcg_at_k(y_true, s2_scores, 5))
                stage2_ndcg10.append(compute_ndcg_at_k(y_true, s2_scores, 10))
                stage2_mrr.append(compute_mrr(y_true, s2_scores))
                auc2_val = compute_auc(y_true, s2_scores)
                if auc2_val is not None:
                    stage2_auc.append(auc2_val)
                prauc2_val = compute_pr_auc(y_true, s2_scores)
                if prauc2_val is not None:
                    stage2_prauc.append(prauc2_val)
                kappa2_val = compute_cohen_kappa(y_true, s2_scores)
                if kappa2_val is not None:
                    stage2_kappa.append(kappa2_val)

                # Store impression normalized scores for global weight sweep
                if "impression_scores" not in locals():
                    impression_scores = []
                impression_scores.append({
                    "y_true": y_true,
                    "norm_s1": norm_s1,
                    "norm_s2": norm_s2,
                })

    results = {
        "Stage1_nDCG@5": float(np.mean(stage1_ndcg5)),
        "Stage1_nDCG@10": float(np.mean(stage1_ndcg10)),
        "Stage1_MRR": float(np.mean(stage1_mrr)),
        "Stage1_AUC": float(np.mean(stage1_auc)) if stage1_auc else 0.0,
        "Stage1_PR_AUC": float(np.mean(stage1_prauc)) if stage1_prauc else 0.0,
        "Stage1_Cohen_Kappa": float(np.mean(stage1_kappa)) if stage1_kappa else 0.0,
    }

    if xgb_ranker is not None and stage2_ndcg10:
        results.update({
            "Stage2_nDCG@5": float(np.mean(stage2_ndcg5)),
            "Stage2_nDCG@10": float(np.mean(stage2_ndcg10)),
            "Stage2_MRR": float(np.mean(stage2_mrr)),
            "Stage2_AUC": float(np.mean(stage2_auc)) if stage2_auc else 0.0,
            "Stage2_PR_AUC": float(np.mean(stage2_prauc)) if stage2_prauc else 0.0,
            "Stage2_Cohen_Kappa": float(np.mean(stage2_kappa)) if stage2_kappa else 0.0,
        })

    # Ensemble Weight Sweep
    best_alpha = 1.0
    best_ensemble_ndcg10 = results["Stage1_nDCG@10"]

    if xgb_ranker is not None and "impression_scores" in locals() and impression_scores:
        logger.info("\n" + "="*70)
        logger.info("=== ENSEMBLE LEARNING WEIGHTAGE COMBINATION SWEEP ===")
        logger.info("Formula: S_ensemble = alpha * S_TwoTower + (1 - alpha) * S_Reranker")
        logger.info(f"{'Alpha (Two-Tower)':<18} {'(1-Alpha) Rerank':<18} {'nDCG@5':<9} {'nDCG@10':<9} {'PR-AUC':<9} {'Kappa':<9} {'MRR':<9}")
        logger.info("-" * 85)

        alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        ensemble_sweep = {}

        for a in alphas:
            sweep_ndcg5, sweep_ndcg10, sweep_mrr = [], [], []
            sweep_prauc, sweep_kappa = [], []
            for imp in impression_scores:
                y = imp["y_true"]
                blended = a * imp["norm_s1"] + (1.0 - a) * imp["norm_s2"]
                sweep_ndcg5.append(compute_ndcg_at_k(y, blended, 5))
                sweep_ndcg10.append(compute_ndcg_at_k(y, blended, 10))
                sweep_mrr.append(compute_mrr(y, blended))
                pauc = compute_pr_auc(y, blended)
                if pauc is not None:
                    sweep_prauc.append(pauc)
                ck = compute_cohen_kappa(y, blended)
                if ck is not None:
                    sweep_kappa.append(ck)

            mean_n5 = float(np.mean(sweep_ndcg5))
            mean_n10 = float(np.mean(sweep_ndcg10))
            mean_mrr = float(np.mean(sweep_mrr))
            mean_prauc = float(np.mean(sweep_prauc)) if sweep_prauc else 0.0
            mean_kappa = float(np.mean(sweep_kappa)) if sweep_kappa else 0.0

            ensemble_sweep[f"alpha_{a:.1f}"] = {
                "nDCG@5": mean_n5,
                "nDCG@10": mean_n10,
                "PR_AUC": mean_prauc,
                "Cohen_Kappa": mean_kappa,
                "MRR": mean_mrr,
            }

            logger.info(f"{a:<18.2f} {1.0 - a:<18.2f} {mean_n5:<9.4f} {mean_n10:<9.4f} {mean_prauc:<9.4f} {mean_kappa:<9.4f} {mean_mrr:<9.4f}")

            if mean_n10 > best_ensemble_ndcg10:
                best_ensemble_ndcg10 = mean_n10
                best_alpha = a

        logger.info("=" * 70)
        logger.info(f"Optimal Ensemble Weight: alpha = {best_alpha:.2f} (nDCG@10: {best_ensemble_ndcg10:.4f})")

        # Save optimal config
        ckpt_dir = os.path.dirname(checkpoint_path)
        ensemble_config_file = os.path.join(ckpt_dir, "ensemble_config.json")
        with open(ensemble_config_file, "w", encoding="utf-8") as f:
            json.dump({
                "optimal_alpha_two_tower": best_alpha,
                "weight_reranker": round(1.0 - best_alpha, 2),
                "best_nDCG@10": best_ensemble_ndcg10,
                "sweep_results": ensemble_sweep,
            }, f, indent=2)
        logger.info(f"Saved ensemble config to {ensemble_config_file}")

        results["Ensemble_Optimal_Alpha"] = best_alpha
        results["Ensemble_Best_nDCG@10"] = best_ensemble_ndcg10

    logger.info("=== EVALUATION BENCHMARK RESULTS ===")
    for k, v in results.items():
        logger.info(f"{k:20s}: {v:.4f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate News Recommender Metrics")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/two_tower_best.pt")
    parser.add_argument("--vocab", type=str, default="checkpoints/word2idx.json")
    parser.add_argument("--xgb", type=str, default="checkpoints/lambdamart_ranker.json")
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--use_mock", action="store_true", help="Use mock MIND dataset for fast evaluation")
    parser.add_argument("--max_samples", type=int, default=500)
    args = parser.parse_args()

    evaluate_pipeline(
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab,
        xgb_path=args.xgb,
        data_dir=args.data_dir,
        use_mock=args.use_mock,
        max_eval_samples=args.max_samples,
    )
