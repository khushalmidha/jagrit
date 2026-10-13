"""
Evaluation Pipeline for Multimodal Two-Stage News Recommendation System.

Protocol:
- Evaluates on the UNTOUCHED Chronological Test Partition (Zero Test Leakage).
- Uses locked ensemble weights from checkpoints/ensemble_config.json.
- Uses train feature priors from checkpoints/train_feature_priors.json.
- Computes comprehensive ranking, classification concordance, and diversity metrics:
  * nDCG@5, nDCG@10
  * Mean Reciprocal Rank (MRR)
  * Area Under ROC Curve (ROC-AUC)
  * Precision-Recall AUC (PR-AUC)
  * Cohen's Kappa (chance-corrected agreement)
  * Intra-List Diversity (ILD)
  * P95 Inference Latency (ms)
- Emits results/test_results.json.
"""

import os
import math
import time
import json
import logging
import argparse
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import xgboost as xgb

from data.dataset import MINDMultimodalDataset, SimpleTokenizer, create_mock_mind_dataset
from data.split import load_temporal_splits, extract_ranking_features
from models.two_tower import TwoTowerNewsRecommender
from models.metrics import (
    compute_ndcg_at_k,
    compute_mrr,
    compute_auc,
    compute_pr_auc,
    compute_cohen_kappa,
    compute_intra_list_diversity,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("Evaluate")


def run_mmr_rerank(
    candidate_ids: List[str],
    candidate_scores: np.ndarray,
    news_categories: Dict[str, str],
    diversity_lambda: float = 0.75,
    top_k: int = 10,
) -> Tuple[List[str], np.ndarray]:
    """
    Maximal Marginal Relevance (MMR) greedy diversity reranking:
    MMR = argmax_{i in R \\ S} [ lambda * score_i - (1 - lambda) * max_{j in S} Sim(i, j) ]
    """
    if len(candidate_ids) <= top_k:
        top_k = len(candidate_ids)

    selected_indices: List[int] = []
    unselected = list(range(len(candidate_ids)))

    norm_scores = candidate_scores.copy()
    ptp = np.ptp(norm_scores)
    if ptp > 1e-12:
        norm_scores = (norm_scores - np.min(norm_scores)) / ptp
    else:
        norm_scores = np.zeros_like(norm_scores)

    # Pick top item by pure relevance
    first_idx = int(np.argmax(norm_scores))
    selected_indices.append(first_idx)
    unselected.remove(first_idx)

    while len(selected_indices) < top_k and unselected:
        best_mmr = -float("inf")
        best_cand = unselected[0]

        for cand_idx in unselected:
            rel = float(norm_scores[cand_idx])
            cand_cat = news_categories.get(candidate_ids[cand_idx], "")

            # Category overlap penalty with already selected items
            max_sim = 0.0
            for sel_idx in selected_indices:
                sel_cat = news_categories.get(candidate_ids[sel_idx], "")
                if cand_cat and cand_cat == sel_cat:
                    max_sim = 1.0
                    break

            mmr_score = diversity_lambda * rel - (1.0 - diversity_lambda) * max_sim
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_cand = cand_idx

        selected_indices.append(best_cand)
        unselected.remove(best_cand)

    # Append remaining items in relative order
    all_ordered = selected_indices + unselected
    reranked_ids = [candidate_ids[idx] for idx in all_ordered]
    reranked_scores = np.array([norm_scores[idx] for idx in all_ordered], dtype=np.float32)
    return reranked_ids, reranked_scores


def evaluate_pipeline(
    checkpoint_path: str,
    vocab_path: str,
    xgb_path: str,
    ensemble_cfg_path: str,
    priors_path: str,
    data_dir: Optional[str] = None,
    use_mock: bool = False,
    max_eval_samples: int = 500,
    device_name: str = "cpu",
    results_output_dir: str = "results",
) -> Dict[str, Any]:
    """
    Evaluates Stage 1, Stage 2, Locked Ensemble, and MMR on the untouched Test partition.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device(device_name)
    os.makedirs(results_output_dir, exist_ok=True)

    # Load Tokenizer
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    tokenizer = SimpleTokenizer(vocab=vocab)

    # Load Locked Ensemble Config
    locked_alpha = 0.50
    if os.path.exists(ensemble_cfg_path):
        with open(ensemble_cfg_path, "r", encoding="utf-8") as f:
            ens_cfg = json.load(f)
        locked_alpha = ens_cfg.get("optimal_alpha_two_tower", 0.50)
        logger.info(f"Loaded LOCKED Ensemble alpha = {locked_alpha:.2f} from {ensemble_cfg_path}")
    else:
        logger.warning(f"Ensemble config not found at {ensemble_cfg_path}, falling back to alpha=0.50")

    # Load Train Priors
    train_priors = {}
    if os.path.exists(priors_path):
        with open(priors_path, "r", encoding="utf-8") as f:
            train_priors = json.load(f)
        logger.info(f"Loaded leak-free train priors (global CTR: {train_priors.get('global_ctr_prior', 0.04):.4f}) from {priors_path}")
    else:
        logger.warning(f"Train priors not found at {priors_path}, using default heuristic priors.")

    # Load Chronological Test Partition
    raw_dir = data_dir if data_dir else os.path.join(base_dir, "data", "raw", "MINDsmall_train")
    if not use_mock and os.path.exists(os.path.join(raw_dir, "news.tsv")) and os.path.exists(os.path.join(raw_dir, "behaviors.tsv")):
        news_path = os.path.join(raw_dir, "news.tsv")
        behaviors_path = os.path.join(raw_dir, "behaviors.tsv")
        _, _, test_df = load_temporal_splits(
            behaviors_path=behaviors_path,
            train_val_test_ratio=(0.8, 0.1, 0.1),
            test_samples=max_eval_samples,
            cache_dir=os.path.join(base_dir, "data", "processed"),
        )
    else:
        mock_dir = os.path.join(base_dir, "data", "mock_eval")
        news_path, behaviors_path = create_mock_mind_dataset(mock_dir, num_news=100, num_behaviors=300)
        _, _, test_df = load_temporal_splits(
            behaviors_path=behaviors_path,
            train_val_test_ratio=(0.8, 0.1, 0.1),
            test_samples=max_eval_samples,
            cache_dir=None,
        )

    logger.info(f"Initializing test dataset on {len(test_df)} chronologically isolated sessions...")
    test_dataset = MINDMultimodalDataset(
        behaviors_df=test_df,
        news_path=news_path,
        tokenizer=tokenizer,
        max_title_len=20,
        max_history_len=20,
        is_train=False,
    )
    logger.info(f"Loaded {len(test_dataset)} test evaluation impression sessions.")

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

    # Load Stage 2 XGBoost LambdaMART
    xgb_ranker = None
    if os.path.exists(xgb_path):
        logger.info(f"Loading Stage 2 XGBoost LambdaMART model from {xgb_path}...")
        xgb_ranker = xgb.XGBRanker()
        xgb_ranker.load_model(xgb_path)

    # Metric Accumulators
    s1_metrics = {"ndcg5": [], "ndcg10": [], "mrr": [], "auc": [], "prauc": [], "kappa": [], "latencies": []}
    s2_metrics = {"ndcg5": [], "ndcg10": [], "mrr": [], "auc": [], "prauc": [], "kappa": [], "latencies": []}
    ens_metrics = {"ndcg5": [], "ndcg10": [], "mrr": [], "auc": [], "prauc": [], "kappa": [], "latencies": []}
    mmr_metrics = {"ndcg5": [], "ndcg10": [], "mrr": [], "auc": [], "prauc": [], "kappa": [], "ild": [], "latencies": []}

    def _norm(scores):
        ptp = np.ptp(scores)
        return (scores - np.min(scores)) / (ptp + 1e-12) if ptp > 1e-12 else np.zeros_like(scores)

    with torch.no_grad():
        for i in range(len(test_dataset)):
            sample = test_dataset[i]
            y_true = sample["labels"].numpy()
            cands = sample["candidate_ids"]

            if len(cands) < 2 or np.sum(y_true) == 0:
                continue

            # Stage 1: Two-Tower Inference
            t0_s1 = time.perf_counter()
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

            cand_tok = sample["candidate_tokens"].to(device)
            cand_img = sample["candidate_images"].to(device)
            e_cands = model.encode_articles(cand_img, cand_tok, normalize=True)

            s1_scores = torch.matmul(e_cands, u_u).cpu().numpy()
            lat_s1 = (time.perf_counter() - t0_s1) * 1000.0
            s1_metrics["latencies"].append(lat_s1)

            # Record Stage 1
            s1_metrics["ndcg5"].append(compute_ndcg_at_k(y_true, s1_scores, 5))
            s1_metrics["ndcg10"].append(compute_ndcg_at_k(y_true, s1_scores, 10))
            s1_metrics["mrr"].append(compute_mrr(y_true, s1_scores))
            a_val = compute_auc(y_true, s1_scores)
            if a_val is not None:
                s1_metrics["auc"].append(a_val)
            p_val = compute_pr_auc(y_true, s1_scores)
            if p_val is not None:
                s1_metrics["prauc"].append(p_val)
            k_val = compute_cohen_kappa(y_true, s1_scores)
            if k_val is not None:
                s1_metrics["kappa"].append(k_val)

            # Stage 2: LambdaMART Inference
            if xgb_ranker is not None:
                t0_s2 = time.perf_counter()
                history = sample.get("history", [])
                hist_categories = [test_dataset.news_categories.get(nid, "") for nid in history if nid in test_dataset.news_categories]
                hist_subcategories = [test_dataset.news_subcategories.get(nid, "") for nid in history if nid in test_dataset.news_subcategories]
                num_hist = max(1, len(hist_categories))

                cat_counts: Dict[str, int] = {}
                subcat_counts: Dict[str, int] = {}
                for c in hist_categories:
                    cat_counts[c] = cat_counts.get(c, 0) + 1
                for sc in hist_subcategories:
                    subcat_counts[sc] = subcat_counts.get(sc, 0) + 1

                cat_probs = [cnt / float(num_hist) for cnt in cat_counts.values()]
                cat_entropy = -sum(p * math.log2(p + 1e-9) for p in cat_probs)
                session_t = float(sample.get("timestamp", train_priors.get("min_train_timestamp", 0.0)))

                cand_features = []
                for idx, cid in enumerate(cands):
                    sim = float(s1_scores[idx])
                    cat = test_dataset.news_categories.get(cid, "")
                    subcat = test_dataset.news_subcategories.get(cid, "")
                    cat_affinity = cat_counts.get(cat, 0) / float(num_hist)
                    subcat_affinity = subcat_counts.get(subcat, 0) / float(num_hist)

                    feats = extract_ranking_features(
                        sim=sim,
                        cat_affinity=cat_affinity,
                        subcat_affinity=subcat_affinity,
                        cat_entropy=cat_entropy,
                        session_timestamp=session_t,
                        article_id=cid,
                        priors=train_priors,
                    )
                    cand_features.append(feats)

                X_cand = np.array(cand_features, dtype=np.float32)
                s2_scores = xgb_ranker.predict(X_cand)
                lat_s2 = (time.perf_counter() - t0_s2) * 1000.0
                s2_metrics["latencies"].append(lat_s2)

                s2_metrics["ndcg5"].append(compute_ndcg_at_k(y_true, s2_scores, 5))
                s2_metrics["ndcg10"].append(compute_ndcg_at_k(y_true, s2_scores, 10))
                s2_metrics["mrr"].append(compute_mrr(y_true, s2_scores))
                a2_val = compute_auc(y_true, s2_scores)
                if a2_val is not None:
                    s2_metrics["auc"].append(a2_val)
                p2_val = compute_pr_auc(y_true, s2_scores)
                if p2_val is not None:
                    s2_metrics["prauc"].append(p2_val)
                k2_val = compute_cohen_kappa(y_true, s2_scores)
                if k2_val is not None:
                    s2_metrics["kappa"].append(k2_val)

                # Locked Ensemble Score Fusion
                t0_ens = time.perf_counter()
                norm_s1 = _norm(s1_scores)
                norm_s2 = _norm(s2_scores)
                blended_scores = locked_alpha * norm_s1 + (1.0 - locked_alpha) * norm_s2
                lat_ens = (time.perf_counter() - t0_ens) * 1000.0 + lat_s1 + lat_s2
                ens_metrics["latencies"].append(lat_ens)

                ens_metrics["ndcg5"].append(compute_ndcg_at_k(y_true, blended_scores, 5))
                ens_metrics["ndcg10"].append(compute_ndcg_at_k(y_true, blended_scores, 10))
                ens_metrics["mrr"].append(compute_mrr(y_true, blended_scores))
                ae_val = compute_auc(y_true, blended_scores)
                if ae_val is not None:
                    ens_metrics["auc"].append(ae_val)
                pe_val = compute_pr_auc(y_true, blended_scores)
                if pe_val is not None:
                    ens_metrics["prauc"].append(pe_val)
                ke_val = compute_cohen_kappa(y_true, blended_scores)
                if ke_val is not None:
                    ens_metrics["kappa"].append(ke_val)

                # MMR Diversity Layer (lambda = 0.75)
                t0_mmr = time.perf_counter()
                reranked_cands, reranked_scores = run_mmr_rerank(
                    candidate_ids=cands,
                    candidate_scores=blended_scores,
                    news_categories=test_dataset.news_categories,
                    diversity_lambda=0.75,
                    top_k=10,
                )
                lat_mmr = (time.perf_counter() - t0_mmr) * 1000.0 + lat_ens
                mmr_metrics["latencies"].append(lat_mmr)

                # Map original labels to reranked order
                cand_to_lbl = {cands[idx]: y_true[idx] for idx in range(len(cands))}
                reranked_y_true = np.array([cand_to_lbl[cid] for cid in reranked_cands], dtype=np.float32)

                mmr_metrics["ndcg5"].append(compute_ndcg_at_k(reranked_y_true, reranked_scores, 5))
                mmr_metrics["ndcg10"].append(compute_ndcg_at_k(reranked_y_true, reranked_scores, 10))
                mmr_metrics["mrr"].append(compute_mrr(reranked_y_true, reranked_scores))
                am_val = compute_auc(reranked_y_true, reranked_scores)
                if am_val is not None:
                    mmr_metrics["auc"].append(am_val)
                pm_val = compute_pr_auc(reranked_y_true, reranked_scores)
                if pm_val is not None:
                    mmr_metrics["prauc"].append(pm_val)
                km_val = compute_cohen_kappa(reranked_y_true, reranked_scores)
                if km_val is not None:
                    mmr_metrics["kappa"].append(km_val)

                # Calculate intra-list diversity across top 10
                top_cats = [test_dataset.news_categories.get(cid, "") for cid in reranked_cands[:10]]
                mmr_metrics["ild"].append(compute_intra_list_diversity(top_cats, top_k=10))

    def _agg(m: Dict[str, list]) -> Dict[str, float]:
        return {
            "nDCG@5": float(np.mean(m["ndcg5"])),
            "nDCG@10": float(np.mean(m["ndcg10"])),
            "MRR": float(np.mean(m["mrr"])),
            "ROC-AUC": float(np.mean(m["auc"])) if m["auc"] else 0.0,
            "PR-AUC": float(np.mean(m["prauc"])) if m["prauc"] else 0.0,
            "Cohen_Kappa": float(np.mean(m["kappa"])) if m["kappa"] else 0.0,
            "P95_Latency_ms": float(np.percentile(m["latencies"], 95)) if m["latencies"] else 0.0,
        }

    results = {
        "evaluation_protocol": "Chronological Test Split (Zero Test-Set Tuning)",
        "test_sessions_evaluated": len(s1_metrics["ndcg10"]),
        "locked_alpha": locked_alpha,
        "Stage1_TwoTower": _agg(s1_metrics),
        "Stage2_LambdaMART": _agg(s2_metrics) if xgb_ranker else {},
        "Ensemble_Locked": _agg(ens_metrics) if xgb_ranker else {},
        "Ensemble_MMR": {
            **_agg(mmr_metrics),
            "Intra_List_Diversity": float(np.mean(mmr_metrics["ild"])) if mmr_metrics["ild"] else 0.0,
        } if xgb_ranker else {},
    }

    # Save to results/test_results.json
    out_file = os.path.join(results_output_dir, "test_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("\n" + "=" * 90)
    logger.info("=== RESEARCH-GRADE OFFLINE TEST BENCHMARK RESULTS (TEST SPLIT) ===")
    logger.info(f"{'Model / Architecture':<30} {'nDCG@5':<9} {'nDCG@10':<9} {'PR-AUC':<9} {'ROC-AUC':<9} {'Kappa':<9} {'MRR':<9} {'P95 (ms)':<9}")
    logger.info("-" * 90)

    for name, r in [
        ("Stage 1: Two-Tower (PyTorch)", results["Stage1_TwoTower"]),
        ("Stage 2: LambdaMART (XGBoost)", results["Stage2_LambdaMART"]),
        (f"Ensemble (Locked α={locked_alpha:.2f})", results["Ensemble_Locked"]),
        ("Ensemble + MMR (λ=0.75)", results["Ensemble_MMR"]),
    ]:
        if r:
            logger.info(
                f"{name:<30} {r['nDCG@5']:<9.4f} {r['nDCG@10']:<9.4f} {r['PR-AUC']:<9.4f} "
                f"{r['ROC-AUC']:<9.4f} {r['Cohen_Kappa']:<9.4f} {r['MRR']:<9.4f} {r['P95_Latency_ms']:<9.2f}"
            )
    logger.info("=" * 90)
    logger.info(f"Saved verified test benchmark to {out_file}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate News Recommender Metrics on Test Partition")
    parser.add_argument("--config", type=str, default="configs/mind.yaml", help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/two_tower_best.pt")
    parser.add_argument("--vocab", type=str, default="checkpoints/word2idx.json")
    parser.add_argument("--xgb", type=str, default="checkpoints/lambdamart_ranker.json")
    parser.add_argument("--ensemble_cfg", type=str, default="checkpoints/ensemble_config.json")
    parser.add_argument("--priors", type=str, default="checkpoints/train_feature_priors.json")
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--use_mock", action="store_true", help="Use mock MIND dataset")
    parser.add_argument("--max_samples", type=int, default=500)
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    if args.config:
        cfg_path = args.config if os.path.isabs(args.config) else os.path.join(base_dir, args.config)
        if os.path.exists(cfg_path):
            import yaml
            logger.info(f"Loading evaluation parameters from {cfg_path}...")
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            if "checkpoints" in cfg:
                ckpt_sec = cfg["checkpoints"]
                args.checkpoint = os.path.join(base_dir, ckpt_sec.get("two_tower_weights", args.checkpoint))
                args.vocab = os.path.join(base_dir, ckpt_sec.get("vocabulary", args.vocab))
                args.xgb = os.path.join(base_dir, ckpt_sec.get("lambdamart_model", args.xgb))
                args.ensemble_cfg = os.path.join(base_dir, ckpt_sec.get("ensemble_config", args.ensemble_cfg))
            if "data" in cfg:
                args.max_samples = cfg["data"].get("test_samples", args.max_samples)
                if cfg["data"].get("raw_dir"):
                    args.data_dir = os.path.join(base_dir, cfg["data"]["raw_dir"])

    evaluate_pipeline(
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab,
        xgb_path=args.xgb,
        ensemble_cfg_path=args.ensemble_cfg,
        priors_path=args.priors,
        data_dir=args.data_dir,
        use_mock=args.use_mock,
        max_eval_samples=args.max_samples,
    )
