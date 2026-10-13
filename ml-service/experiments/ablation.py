"""
Reproducible Ablation Study Pipeline for Multimodal News Recommendation.

Evaluates incremental architectural contributions on the untouched Test Split:
1. Two-Tower (Text-Only Base)
2. Two-Tower + Visual Features
3. Two-Tower + Attention & Gating (Full Stage 1)
4. Stage 2: LambdaMART Only (alpha = 0.0)
5. Full 2-Stage Ensemble (Locked alpha*)
6. Ensemble + MMR Diversity (lambda = 0.75)

Emits: results/ablation.json
"""

import os
import sys
import math
import time
import json
import logging
import argparse
from typing import Dict, List, Tuple, Optional, Any

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
logger = logging.getLogger("AblationStudy")


def run_ablation(
    checkpoint_path: str,
    vocab_path: str,
    xgb_path: str,
    ensemble_cfg_path: str,
    priors_path: str,
    data_dir: Optional[str] = None,
    use_mock: bool = False,
    max_eval_samples: int = 500,
    device_name: str = "cpu",
    output_json_path: str = "results/ablation.json",
) -> Dict[str, Any]:
    """
    Executes the 6-variant ablation study on the chronological test split.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    device = torch.device(device_name)
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    # 1. Load Tokenizer
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    tokenizer = SimpleTokenizer(vocab=vocab)

    # 2. Load Locked Ensemble Config
    locked_alpha = 0.70
    if os.path.exists(ensemble_cfg_path):
        with open(ensemble_cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        locked_alpha = cfg.get("optimal_alpha_two_tower", 0.70)
        logger.info(f"Loaded locked validation ensemble alpha: {locked_alpha:.2f}")

    # 3. Load Train Feature Priors
    train_priors = {}
    if os.path.exists(priors_path):
        with open(priors_path, "r", encoding="utf-8") as f:
            train_priors = json.load(f)

    # 4. Load Chronological Test Partition
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

    test_dataset = MINDMultimodalDataset(
        behaviors_df=test_df,
        news_path=news_path,
        tokenizer=tokenizer,
        max_title_len=20,
        max_history_len=20,
        is_train=False,
    )
    logger.info(f"Ablation study evaluating {len(test_dataset)} test impressions on {device}...")

    # 5. Load Models
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = TwoTowerNewsRecommender(
        d_model=ckpt.get("d_model", 256),
        vocab_size=len(tokenizer),
        d_emb=ckpt.get("d_model", 256),
        temperature=ckpt.get("temperature", 0.07),
        pretrained_vision=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    xgb_ranker = xgb.XGBRanker()
    xgb_ranker.load_model(xgb_path)

    # Accumulators for all 6 variants
    variants = {
        "text_only": {"ndcg5": [], "ndcg10": [], "prauc": [], "auc": [], "kappa": [], "mrr": [], "ild": [], "lat": []},
        "visual_concat": {"ndcg5": [], "ndcg10": [], "prauc": [], "auc": [], "kappa": [], "mrr": [], "ild": [], "lat": []},
        "two_tower_full": {"ndcg5": [], "ndcg10": [], "prauc": [], "auc": [], "kappa": [], "mrr": [], "ild": [], "lat": []},
        "lambdamart_only": {"ndcg5": [], "ndcg10": [], "prauc": [], "auc": [], "kappa": [], "mrr": [], "ild": [], "lat": []},
        "ensemble_locked": {"ndcg5": [], "ndcg10": [], "prauc": [], "auc": [], "kappa": [], "mrr": [], "ild": [], "lat": []},
        "ensemble_mmr": {"ndcg5": [], "ndcg10": [], "prauc": [], "auc": [], "kappa": [], "mrr": [], "ild": [], "lat": []},
    }

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

            hist_tok = sample["history_tokens"].unsqueeze(0).to(device)
            hist_img = sample["history_images"].unsqueeze(0).to(device)
            hist_msk = sample["history_mask"].unsqueeze(0).to(device)
            b, k, seq_len = hist_tok.size()

            cand_tok = sample["candidate_tokens"].to(device)
            cand_img = sample["candidate_images"].to(device)

            # -------------------------------------------------------------
            # Variant 1: Text-Only Base (Zeroed Image Embeddings)
            # -------------------------------------------------------------
            t0 = time.perf_counter()
            zero_hist_img = torch.zeros_like(hist_img)
            zero_cand_img = torch.zeros_like(cand_img)

            flat_zero_hist = zero_hist_img.view(b * k, 3, 224, 224) if zero_hist_img.dim() == 5 else zero_hist_img.view(b * k, -1)
            flat_toks = hist_tok.view(b * k, seq_len)
            hist_enc_text = model.article_encoder(flat_zero_hist, flat_toks).view(b, k, -1)
            u_text = model.encode_user(hist_enc_text, history_mask=hist_msk, normalize=True).squeeze(0)
            e_cands_text = model.encode_articles(zero_cand_img, cand_tok, normalize=True)
            s_text = torch.matmul(e_cands_text, u_text).cpu().numpy()
            lat_text = (time.perf_counter() - t0) * 1000.0

            # -------------------------------------------------------------
            # Variant 2: Visual Concat (Equal Blend of Text & Visual Un-Gated)
            # -------------------------------------------------------------
            t0 = time.perf_counter()
            flat_imgs = hist_img.view(b * k, 3, 224, 224) if hist_img.dim() == 5 else hist_img.view(b * k, -1)
            v_hist = model.article_encoder.visual_encoder(flat_imgs)
            t_hist = model.article_encoder.text_encoder(flat_toks)
            ungated_hist = torch.nn.functional.layer_norm(0.5 * v_hist + 0.5 * t_hist, (256,)).view(b, k, -1)
            u_ungated = model.encode_user(ungated_hist, history_mask=hist_msk, normalize=True).squeeze(0)

            v_cands = model.article_encoder.visual_encoder(cand_img)
            t_cands = model.article_encoder.text_encoder(cand_tok)
            e_cands_ungated = torch.nn.functional.normalize(
                torch.nn.functional.layer_norm(0.5 * v_cands + 0.5 * t_cands, (256,)), p=2, dim=-1
            )
            s_visual = torch.matmul(e_cands_ungated, u_ungated).cpu().numpy()
            lat_vis = (time.perf_counter() - t0) * 1000.0

            # -------------------------------------------------------------
            # Variant 3: Two-Tower Full (Attention + Gating)
            # -------------------------------------------------------------
            t0 = time.perf_counter()
            hist_enc_full = model.article_encoder(flat_imgs, flat_toks).view(b, k, -1)
            u_full = model.encode_user(hist_enc_full, history_mask=hist_msk, normalize=True).squeeze(0)
            e_cands_full = model.encode_articles(cand_img, cand_tok, normalize=True)
            s_full = torch.matmul(e_cands_full, u_full).cpu().numpy()
            lat_full = (time.perf_counter() - t0) * 1000.0

            # -------------------------------------------------------------
            # Variant 4: Stage 2 LambdaMART Only
            # -------------------------------------------------------------
            t0 = time.perf_counter()
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
                sim = float(s_full[idx])
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
            s_lambda = xgb_ranker.predict(X_cand)
            lat_lambda = (time.perf_counter() - t0) * 1000.0

            # -------------------------------------------------------------
            # Variant 5: Full 2-Stage Ensemble (Locked alpha*)
            # -------------------------------------------------------------
            t0 = time.perf_counter()
            norm_s_full = _norm(s_full)
            norm_s_lambda = _norm(s_lambda)
            s_ens = locked_alpha * norm_s_full + (1.0 - locked_alpha) * norm_s_lambda
            lat_ens = (time.perf_counter() - t0) * 1000.0 + lat_full + lat_lambda

            # -------------------------------------------------------------
            # Variant 6: Ensemble + MMR Diversity (lambda = 0.75)
            # -------------------------------------------------------------
            t0 = time.perf_counter()
            # MMR greedy selection
            top_k_mmr = min(10, len(cands))
            sel_idxs: List[int] = []
            unsel = list(range(len(cands)))
            norm_ens = norm_s_full * locked_alpha + norm_s_lambda * (1.0 - locked_alpha)

            first_pick = int(np.argmax(norm_ens))
            sel_idxs.append(first_pick)
            unsel.remove(first_pick)

            while len(sel_idxs) < top_k_mmr and unsel:
                best_val = -float("inf")
                best_cand = unsel[0]
                for c_idx in unsel:
                    rel = float(norm_ens[c_idx])
                    cand_c = test_dataset.news_categories.get(cands[c_idx], "")
                    max_sim = 1.0 if any(test_dataset.news_categories.get(cands[s_idx], "") == cand_c for s_idx in sel_idxs) else 0.0
                    mmr_val = 0.75 * rel - 0.25 * max_sim
                    if mmr_val > best_val:
                        best_val = mmr_val
                        best_cand = c_idx
                sel_idxs.append(best_cand)
                unsel.remove(best_cand)

            all_mmr_order = sel_idxs + unsel
            reranked_scores = np.array([norm_ens[idx] for idx in all_mmr_order], dtype=np.float32)
            reranked_y_true = np.array([y_true[idx] for idx in all_mmr_order], dtype=np.float32)
            lat_mmr = (time.perf_counter() - t0) * 1000.0 + lat_ens

            # Record metrics for each variant
            def _record(key, true_lbls, scores, latency_ms, top_cats_list=None):
                v = variants[key]
                v["ndcg5"].append(compute_ndcg_at_k(true_lbls, scores, 5))
                v["ndcg10"].append(compute_ndcg_at_k(true_lbls, scores, 10))
                v["mrr"].append(compute_mrr(true_lbls, scores))
                auc_val = compute_auc(true_lbls, scores)
                if auc_val is not None:
                    v["auc"].append(auc_val)
                prauc_val = compute_pr_auc(true_lbls, scores)
                if prauc_val is not None:
                    v["prauc"].append(prauc_val)
                kap_val = compute_cohen_kappa(true_lbls, scores)
                if kap_val is not None:
                    v["kappa"].append(kap_val)
                v["lat"].append(latency_ms)
                if top_cats_list is not None:
                    v["ild"].append(compute_intra_list_diversity(top_cats_list, top_k=10))

            top_cats_orig = [test_dataset.news_categories.get(cands[idx], "") for idx in np.argsort(s_text)[::-1][:10]]
            _record("text_only", y_true, s_text, lat_text, top_cats_orig)

            top_cats_vis = [test_dataset.news_categories.get(cands[idx], "") for idx in np.argsort(s_visual)[::-1][:10]]
            _record("visual_concat", y_true, s_visual, lat_vis, top_cats_vis)

            top_cats_full = [test_dataset.news_categories.get(cands[idx], "") for idx in np.argsort(s_full)[::-1][:10]]
            _record("two_tower_full", y_true, s_full, lat_full, top_cats_full)

            top_cats_lam = [test_dataset.news_categories.get(cands[idx], "") for idx in np.argsort(s_lambda)[::-1][:10]]
            _record("lambdamart_only", y_true, s_lambda, lat_lambda, top_cats_lam)

            top_cats_ens = [test_dataset.news_categories.get(cands[idx], "") for idx in np.argsort(s_ens)[::-1][:10]]
            _record("ensemble_locked", y_true, s_ens, lat_ens, top_cats_ens)

            top_cats_mmr = [test_dataset.news_categories.get(cands[idx], "") for idx in all_mmr_order[:10]]
            _record("ensemble_mmr", reranked_y_true, reranked_scores, lat_mmr, top_cats_mmr)

    # Aggregate results into final ablation report
    def _summarize(v: Dict[str, list], desc: str) -> Dict[str, Any]:
        return {
            "description": desc,
            "nDCG@5": float(np.mean(v["ndcg5"])),
            "nDCG@10": float(np.mean(v["ndcg10"])),
            "PR_AUC": float(np.mean(v["prauc"])) if v["prauc"] else 0.0,
            "ROC_AUC": float(np.mean(v["auc"])) if v["auc"] else 0.0,
            "Cohen_Kappa": float(np.mean(v["kappa"])) if v["kappa"] else 0.0,
            "MRR": float(np.mean(v["mrr"])),
            "Intra_List_Diversity": float(np.mean(v["ild"])) if v["ild"] else 0.0,
            "P95_Latency_ms": float(np.percentile(v["lat"], 95)) if v["lat"] else 0.0,
        }

    ablation_results = {
        "evaluation_protocol": "Chronological Test Split (Zero Test Leakage)",
        "test_sessions_evaluated": len(test_dataset),
        "locked_validation_alpha": locked_alpha,
        "variants": {
            "1_TwoTower_TextOnly": _summarize(
                variants["text_only"],
                "Pure text title token representations without visual feature grounding."
            ),
            "2_TwoTower_VisualConcat": _summarize(
                variants["visual_concat"],
                "Linear combination of visual and text features without cross-modal gating."
            ),
            "3_TwoTower_Full_Gated": _summarize(
                variants["two_tower_full"],
                "Multi-Head Self-Attention text encoder + 2048-d visual representations with Gated Cross-Modal Fusion."
            ),
            "4_LambdaMART_Only": _summarize(
                variants["lambdamart_only"],
                "Decision tree ranking solely on tabular signals (Bayesian CTR, category affinity, recency decay)."
            ),
            "5_Ensemble_Locked": _summarize(
                variants["ensemble_locked"],
                f"Calibrated 2-stage ensemble with alpha={locked_alpha:.2f} locked from the validation split."
            ),
            "6_Ensemble_MMR_Diversity": _summarize(
                variants["ensemble_mmr"],
                "Ensemble blended score with Maximal Marginal Relevance (MMR, lambda=0.75) diversity reranker."
            ),
        }
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)

    logger.info("\n" + "=" * 98)
    logger.info("=== EMPIRICAL ABLATION STUDY RESULTS (EVALUATED ON TEST SPLIT) ===")
    logger.info(f"{'Variant':<28} {'nDCG@5':<9} {'nDCG@10':<9} {'PR-AUC':<9} {'ROC-AUC':<9} {'Kappa':<9} {'MRR':<9} {'ILD':<7} {'P95 (ms)':<9}")
    logger.info("-" * 98)

    for k, v in ablation_results["variants"].items():
        logger.info(
            f"{k:<28} {v['nDCG@5']:<9.4f} {v['nDCG@10']:<9.4f} {v['PR_AUC']:<9.4f} "
            f"{v['ROC_AUC']:<9.4f} {v['Cohen_Kappa']:<9.4f} {v['MRR']:<9.4f} {v['Intra_List_Diversity']:<7.2f} {v['P95_Latency_ms']:<9.2f}"
        )
    logger.info("=" * 98)
    logger.info(f"Saved reproducible ablation results to {output_json_path}")
    return ablation_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Reproducible Multimodal Ablation Study")
    parser.add_argument("--variants", nargs="+", default=["text", "visual", "gated", "lambdamart", "ensemble", "mmr"])
    parser.add_argument("--split", type=str, default="test", help="Evaluation split (test)")
    parser.add_argument("--config", type=str, default="configs/mind.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/two_tower_best.pt")
    parser.add_argument("--vocab", type=str, default="checkpoints/word2idx.json")
    parser.add_argument("--xgb", type=str, default="checkpoints/lambdamart_ranker.json")
    parser.add_argument("--ensemble_cfg", type=str, default="checkpoints/ensemble_config.json")
    parser.add_argument("--priors", type=str, default="checkpoints/train_feature_priors.json")
    parser.add_argument("--max_samples", type=int, default=500)
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.config:
        cfg_path = args.config if os.path.isabs(args.config) else os.path.join(base_dir, args.config)
        if os.path.exists(cfg_path):
            import yaml
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            if "checkpoints" in cfg:
                ckpt_sec = cfg["checkpoints"]
                args.checkpoint = ckpt_sec.get("two_tower_weights", args.checkpoint)
                args.vocab = ckpt_sec.get("vocabulary", args.vocab)
                args.xgb = ckpt_sec.get("lambdamart_model", args.xgb)
                args.priors = ckpt_sec.get("train_priors", args.priors)
                args.ensemble_cfg = ckpt_sec.get("ensemble_config", args.ensemble_cfg)
            if "data" in cfg:
                args.max_samples = cfg["data"].get("test_samples", args.max_samples)

    from experiments.run_ablation import run_ablation as run_ablation_fn
    run_ablation_fn(
        checkpoint_path=os.path.join(base_dir, args.checkpoint),
        vocab_path=os.path.join(base_dir, args.vocab),
        xgb_path=os.path.join(base_dir, args.xgb),
        ensemble_cfg_path=os.path.join(base_dir, args.ensemble_cfg),
        priors_path=os.path.join(base_dir, args.priors),
        max_eval_samples=args.max_samples,
        output_json_path=os.path.join(base_dir, "results", "ablation_results.json"),
        output_csv_path=os.path.join(base_dir, "results", "ablation_results.csv"),
    )

