"""
Training Pipeline for Multimodal Two-Tower News Recommender & Stage 2 XGBoost Reranker.

Pipeline Methodology:
1. Strict Temporal Splitting (Zero Future Leakage):
   - 80% Train | 10% Validation | 10% Test
   - Prior statistics (Bayesian CTR, catalog first-seen timestamps) computed
     STRICTLY from the Train partition.

2. Stage 1: Multimodal Two-Tower Training (PyTorch):
   - Multi-Head Self-Attention text encoder (heads=4, d_k=64, additive word pooling)
   - Visual Tower feature projection (2048-d -> 256-d)
   - Gated Cross-Modal Fusion: g = sigma(W_g [v_i, t_i] + b_g)
   - User History Multi-Head Attention Tower
   - InfoNCE contrastive loss with tau = 0.07 and 50/50 negative curriculum
   - AdamW with Linear Warmup + Cosine Annealing Decay
   - Precomputes multimodal embeddings & builds FAISS IndexFlatIP

3. Stage 2: XGBoost LambdaMART Reranker:
   - Trained on Train partition using exact MIND-derived cross-signals:
     * Cosine similarity s(u, i)
     * User-category match affinity
     * User-subcategory match affinity
     * User reading category entropy
     * Non-linear exponential recency decay exp(-delta_hours / 24.0)
     * Bayesian smoothed historical CTR
     * Historical popularity prior log(1 + impressions)
     * Cross-term sim * cat_affinity
   - Fits XGBRanker with objective='rank:ndcg', eval_metric='ndcg@10'

4. Validation Tuning & Alpha Blending Lock:
   - Evaluates on the isolated Validation partition
   - Sweeps alpha in [0.0, 1.0] to maximize validation nDCG@10
   - Locks optimal alpha* in checkpoints/ensemble_config.json
   - Zero test-set tuning
"""

import os
import math
import time
import json
import logging
import argparse
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import xgboost as xgb
import faiss

from data.dataset import (
    MINDMultimodalDataset,
    train_collate_fn,
    create_mock_mind_dataset,
    SimpleTokenizer,
)
from data.split import (
    load_temporal_splits,
    compute_train_prior_statistics,
    extract_ranking_features,
)
from models.two_tower import TwoTowerNewsRecommender
from models.metrics import compute_ndcg_at_k, compute_mrr, compute_auc, compute_pr_auc, compute_cohen_kappa

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("TrainPipeline")


class LinearWarmupCosineDecayLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Linear warmup followed by cosine annealing decay learning rate schedule.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(warmup_steps + 1, total_steps)
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            alpha = float(step) / float(self.warmup_steps)
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            progress = float(step - self.warmup_steps) / float(
                max(1, self.total_steps - self.warmup_steps)
            )
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * factor
                for base_lr in self.base_lrs
            ]


def train_stage1_twotower(
    model: TwoTowerNewsRecommender,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: LinearWarmupCosineDecayLR,
    device: torch.device,
    epochs: int = 2,
    checkpoint_dir: str = "checkpoints",
) -> str:
    """
    Trains Stage 1 Multimodal Two-Tower Recommender using InfoNCE loss.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.train()

    device_type = "cuda" if device.type == "cuda" else "cpu"
    use_cuda = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    loss_fn = nn.CrossEntropyLoss()
    best_loss = float("inf")
    best_checkpoint_path = os.path.join(checkpoint_dir, "two_tower_best.pt")

    logger.info(f"Starting Stage 1 Training: {epochs} epochs, {len(train_loader)} batches/epoch on {device}")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        batch_count = 0

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            hist_tokens = batch["history_tokens"].to(device)
            hist_images = batch["history_images"].to(device)
            hist_mask = batch["history_mask"].to(device)
            cand_tokens = batch["candidate_tokens"].to(device)
            cand_images = batch["candidate_images"].to(device)
            labels = batch["labels"].to(device)  # Always 0 (positive item)

            with torch.amp.autocast(device_type=device_type, enabled=use_cuda):
                logits, _, _ = model(
                    history_images=hist_images,
                    history_tokens=hist_tokens,
                    candidate_images=cand_images,
                    candidate_tokens=cand_tokens,
                    history_mask=hist_mask,
                )
                loss = loss_fn(logits, labels)

            if use_cuda:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()
            epoch_loss += loss.item()
            batch_count += 1

            if batch_idx % 20 == 0 or batch_idx == len(train_loader) - 1:
                current_lr = scheduler.get_lr()[0]
                logger.info(
                    f"Epoch [{epoch}/{epochs}] Batch [{batch_idx}/{len(train_loader)}] "
                    f"InfoNCE Loss: {loss.item():.4f} | LR: {current_lr:.6f}"
                )

        avg_loss = epoch_loss / max(1, batch_count)
        logger.info(f"Epoch {epoch} Completed | Average InfoNCE Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                    "d_model": model.d_model,
                    "temperature": model.temperature,
                },
                best_checkpoint_path,
            )
            logger.info(f"Saved new best checkpoint to {best_checkpoint_path}")

    elapsed = time.time() - start_time
    logger.info(f"Stage 1 Training finished in {elapsed:.2f}s.")
    return best_checkpoint_path


def build_faiss_index(
    model: TwoTowerNewsRecommender,
    dataset: MINDMultimodalDataset,
    device: torch.device,
    index_path: str = "checkpoints/faiss_index.bin",
    id_map_path: str = "checkpoints/article_ids.json",
) -> Tuple[faiss.Index, List[str]]:
    """
    Precomputes multimodal embeddings for all articles and indexes into FAISS IndexFlatIP.
    """
    model.eval()
    logger.info("Precomputing multimodal embeddings for FAISS candidate generation...")

    news_ids = [nid for nid in dataset.all_news_ids if nid != "<PAD>"]
    all_embeddings = []
    batch_size = 128

    with torch.no_grad():
        for i in range(0, len(news_ids), batch_size):
            batch_ids = news_ids[i : i + batch_size]
            tokens = torch.stack([dataset.news_titles[nid] for nid in batch_ids], dim=0).to(device)
            images = torch.stack([dataset.news_images[nid] for nid in batch_ids], dim=0).to(device)

            embeds = model.encode_articles(images, tokens, normalize=True)
            all_embeddings.append(embeds.cpu().numpy().astype(np.float32))

    emb_matrix = np.vstack(all_embeddings)
    d_model = emb_matrix.shape[1]

    index = faiss.IndexFlatIP(d_model)
    index.add(emb_matrix)

    faiss.write_index(index, index_path)
    with open(id_map_path, "w", encoding="utf-8") as f:
        json.dump(news_ids, f)

    logger.info(f"Indexed {len(news_ids)} articles into FAISS IndexFlatIP (dim={d_model}). Saved to {index_path}")
    return index, news_ids


def train_stage2_xgboost_ranker(
    model: TwoTowerNewsRecommender,
    dataset: MINDMultimodalDataset,
    priors: Dict[str, Any],
    device: torch.device,
    output_path: str = "checkpoints/lambdamart_ranker.json",
    num_samples: int = 1500,
) -> xgb.XGBRanker:
    """
    Extracts deterministic cross-modal ranking features derived from real MIND data
    and trains Stage 2 XGBoost LambdaMART ranker.
    """
    logger.info("Extracting features for Stage 2 XGBoost LambdaMART reranker using real MIND priors...")
    model.eval()

    feature_rows = []
    labels_list = []
    groups = []
    samples_to_use = dataset.samples[:num_samples]

    with torch.no_grad():
        for sample in samples_to_use:
            history = sample["history"]
            hist_tokens, hist_images, hist_mask = dataset._get_history_tensors(history)

            hist_tokens = hist_tokens.unsqueeze(0).to(device)
            hist_images = hist_images.unsqueeze(0).to(device)
            hist_mask = hist_mask.unsqueeze(0).to(device)

            b, k, seq_len = hist_tokens.size()
            if hist_images.dim() == 5:
                flat_imgs = hist_images.view(b * k, 3, 224, 224)
            else:
                flat_imgs = hist_images.view(b * k, -1)
            flat_toks = hist_tokens.view(b * k, seq_len)

            hist_enc = model.article_encoder(flat_imgs, flat_toks).view(b, k, -1)
            u_u = model.encode_user(hist_enc, history_mask=hist_mask, normalize=True).squeeze(0)

            hist_categories = [dataset.news_categories.get(nid, "") for nid in history if nid in dataset.news_categories]
            hist_subcategories = [dataset.news_subcategories.get(nid, "") for nid in history if nid in dataset.news_subcategories]
            num_hist = max(1, len(hist_categories))

            cat_counts: Dict[str, int] = {}
            subcat_counts: Dict[str, int] = {}
            for c in hist_categories:
                cat_counts[c] = cat_counts.get(c, 0) + 1
            for sc in hist_subcategories:
                subcat_counts[sc] = subcat_counts.get(sc, 0) + 1

            cat_probs = [cnt / float(num_hist) for cnt in cat_counts.values()]
            cat_entropy = -sum(p * math.log2(p + 1e-9) for p in cat_probs)

            if "candidates" in sample:
                cands = sample["candidates"]
                lbls = sample["labels"]
            else:
                cands = [sample["positive"]] + sample["negatives"]
                lbls = [1] + [0] * len(sample["negatives"])

            cand_tokens = torch.stack([dataset.news_titles[c] for c in cands], dim=0).to(device)
            cand_images = torch.stack([dataset.news_images[c] for c in cands], dim=0).to(device)
            e_cands = model.encode_articles(cand_images, cand_tokens, normalize=True)

            sims = torch.matmul(e_cands, u_u).cpu().numpy()
            session_t = float(sample.get("timestamp", priors.get("min_train_timestamp", 0.0)))

            group_size = 0
            for idx, cid in enumerate(cands):
                sim = float(sims[idx])
                cat = dataset.news_categories.get(cid, "")
                subcat = dataset.news_subcategories.get(cid, "")
                cat_affinity = cat_counts.get(cat, 0) / float(num_hist)
                subcat_affinity = subcat_counts.get(subcat, 0) / float(num_hist)

                feats = extract_ranking_features(
                    sim=sim,
                    cat_affinity=cat_affinity,
                    subcat_affinity=subcat_affinity,
                    cat_entropy=cat_entropy,
                    session_timestamp=session_t,
                    article_id=cid,
                    priors=priors,
                )
                feature_rows.append(feats)
                labels_list.append(int(lbls[idx]))
                group_size += 1

            if group_size > 0:
                groups.append(group_size)

    X = np.array(feature_rows, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int32)
    group = np.array(groups, dtype=np.int32)

    logger.info(f"Training XGBoost LambdaMART on {len(X)} instances across {len(groups)} query groups...")
    ranker = xgb.XGBRanker(
        objective="rank:ndcg",
        eval_metric="ndcg@10",
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    ranker.fit(X, y, group=group)
    ranker.save_model(output_path)
    logger.info(f"Saved Stage 2 XGBoost LambdaMART model to {output_path}")
    return ranker


def tune_ensemble_alpha_on_validation(
    model: TwoTowerNewsRecommender,
    val_dataset: MINDMultimodalDataset,
    xgb_ranker: xgb.XGBRanker,
    priors: Dict[str, Any],
    device: torch.device,
    output_path: str = "checkpoints/ensemble_config.json",
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Evaluates Candidate Rankings on the Chronologically Isolated Validation Split.
    Sweeps alpha in [0.0, 1.0] and locks the optimal alpha* before test evaluation.
    Formula: S_ensemble = alpha * S_TwoTower + (1 - alpha) * S_Reranker
    """
    logger.info("\n" + "=" * 75)
    logger.info("=== VALIDATION HYPERPARAMETER OPTIMIZATION: ENSEMBLE ALPHA SWEEP ===")
    logger.info("Protocol: Alpha is selected strictly on the chronological Validation split.")
    logger.info("=" * 75)

    model.eval()
    val_impression_scores = []

    with torch.no_grad():
        for i in range(len(val_dataset)):
            sample = val_dataset[i]
            y_true = sample["labels"].numpy()
            cands = sample["candidate_ids"]

            if len(cands) < 2 or np.sum(y_true) == 0:
                continue

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

            history = sample.get("history", [])
            hist_categories = [val_dataset.news_categories.get(nid, "") for nid in history if nid in val_dataset.news_categories]
            hist_subcategories = [val_dataset.news_subcategories.get(nid, "") for nid in history if nid in val_dataset.news_subcategories]
            num_hist = max(1, len(hist_categories))

            cat_counts: Dict[str, int] = {}
            subcat_counts: Dict[str, int] = {}
            for c in hist_categories:
                cat_counts[c] = cat_counts.get(c, 0) + 1
            for sc in hist_subcategories:
                subcat_counts[sc] = subcat_counts.get(sc, 0) + 1

            cat_probs = [cnt / float(num_hist) for cnt in cat_counts.values()]
            cat_entropy = -sum(p * math.log2(p + 1e-9) for p in cat_probs)

            session_t = float(sample.get("timestamp", priors.get("min_train_timestamp", 0.0)))

            cand_features = []
            for idx, cid in enumerate(cands):
                sim = float(s1_scores[idx])
                cat = val_dataset.news_categories.get(cid, "")
                subcat = val_dataset.news_subcategories.get(cid, "")
                cat_affinity = cat_counts.get(cat, 0) / float(num_hist)
                subcat_affinity = subcat_counts.get(subcat, 0) / float(num_hist)

                feats = extract_ranking_features(
                    sim=sim,
                    cat_affinity=cat_affinity,
                    subcat_affinity=subcat_affinity,
                    cat_entropy=cat_entropy,
                    session_timestamp=session_t,
                    article_id=cid,
                    priors=priors,
                )
                cand_features.append(feats)

            X_cand = np.array(cand_features, dtype=np.float32)
            s2_scores = xgb_ranker.predict(X_cand)

            def _norm(scores):
                ptp = np.ptp(scores)
                return (scores - np.min(scores)) / (ptp + 1e-12) if ptp > 1e-12 else np.zeros_like(scores)

            val_impression_scores.append({
                "y_true": y_true,
                "norm_s1": _norm(s1_scores),
                "norm_s2": _norm(s2_scores),
            })

    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    sweep_results = {}
    best_alpha = 1.0
    best_val_ndcg10 = -1.0

    logger.info(f"{'Alpha (TT)':<12} {'(1-Alpha) Rerank':<18} {'nDCG@5':<10} {'nDCG@10':<10} {'PR-AUC':<10} {'MRR':<10}")
    logger.info("-" * 75)

    for a in alphas:
        sw_n5, sw_n10, sw_mrr, sw_prauc = [], [], [], []
        for imp in val_impression_scores:
            y = imp["y_true"]
            blended = a * imp["norm_s1"] + (1.0 - a) * imp["norm_s2"]
            sw_n5.append(compute_ndcg_at_k(y, blended, 5))
            sw_n10.append(compute_ndcg_at_k(y, blended, 10))
            sw_mrr.append(compute_mrr(y, blended))
            pauc = compute_pr_auc(y, blended)
            if pauc is not None:
                sw_prauc.append(pauc)

        mean_n5 = float(np.mean(sw_n5))
        mean_n10 = float(np.mean(sw_n10))
        mean_mrr = float(np.mean(sw_mrr))
        mean_prauc = float(np.mean(sw_prauc)) if sw_prauc else 0.0

        sweep_results[f"alpha_{a:.1f}"] = {
            "nDCG@5": mean_n5,
            "nDCG@10": mean_n10,
            "PR_AUC": mean_prauc,
            "MRR": mean_mrr,
        }

        logger.info(f"{a:<12.2f} {1.0 - a:<18.2f} {mean_n5:<10.4f} {mean_n10:<10.4f} {mean_prauc:<10.4f} {mean_mrr:<10.4f}")

        if mean_n10 > best_val_ndcg10:
            best_val_ndcg10 = mean_n10
            best_alpha = a

    logger.info("=" * 75)
    logger.info(f"LOCKED OPTIMAL ENSEMBLE ALPHA: alpha* = {best_alpha:.2f} (Validation nDCG@10: {best_val_ndcg10:.4f})")
    logger.info("Ensemble config is now frozen. Test split evaluation will use this locked alpha.")

    config_payload = {
        "optimal_alpha_two_tower": best_alpha,
        "weight_reranker": round(1.0 - best_alpha, 2),
        "validation_best_nDCG@10": best_val_ndcg10,
        "selection_protocol": "Chronological validation partition (zero test-set tuning)",
        "sweep_results": sweep_results,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)

    logger.info(f"Saved frozen ensemble configuration to {output_path}")
    return best_alpha, best_val_ndcg10, config_payload


def run_pipeline():
    parser = argparse.ArgumentParser(description="Multimodal Two-Tower + XGBoost News Recommender Training Pipeline")
    parser.add_argument("--config", type=str, default="configs/mind.yaml", help="Path to YAML config file")
    parser.add_argument("--epochs", type=int, default=1, help="Stage 1 training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--d_model", type=int, default=256, help="Latent embedding dimension")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--data_dir", type=str, default="", help="Path to MIND dataset folder")
    parser.add_argument("--use_mock", action="store_true", help="Use mock dataset")
    parser.add_argument("--train_samples", type=int, default=3000, help="Train behaviors count")
    parser.add_argument("--val_samples", type=int, default=500, help="Validation behaviors count")
    parser.add_argument("--test_samples", type=int, default=500, help="Test behaviors count")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoints_dir = os.path.join(base_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Load from YAML config if provided
    if args.config:
        cfg_path = args.config if os.path.isabs(args.config) else os.path.join(base_dir, args.config)
        if os.path.exists(cfg_path):
            import yaml
            logger.info(f"Loading configuration from {cfg_path}...")
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            if "training" in cfg:
                args.epochs = cfg["training"].get("epochs", args.epochs)
                args.batch_size = cfg["training"].get("batch_size", args.batch_size)
                args.lr = float(cfg["training"].get("learning_rate", args.lr))
            if "model" in cfg:
                args.d_model = cfg["model"].get("d_model", args.d_model)
                args.num_heads = cfg["model"].get("num_heads", args.num_heads)
            if "data" in cfg:
                data_sec = cfg["data"]
                if data_sec.get("train_samples"):
                    args.train_samples = data_sec["train_samples"]
                if data_sec.get("val_samples"):
                    args.val_samples = data_sec["val_samples"]
                if data_sec.get("test_samples"):
                    args.test_samples = data_sec["test_samples"]
                if data_sec.get("raw_dir"):
                    args.data_dir = os.path.join(base_dir, data_sec["raw_dir"])

    # Determine data paths
    raw_dir = args.data_dir if args.data_dir else os.path.join(base_dir, "data", "raw", "MINDsmall_train")
    if not args.use_mock and os.path.exists(os.path.join(raw_dir, "news.tsv")) and os.path.exists(os.path.join(raw_dir, "behaviors.tsv")):
        news_path = os.path.join(raw_dir, "news.tsv")
        behaviors_path = os.path.join(raw_dir, "behaviors.tsv")
        logger.info(f"Using existing MIND dataset at {raw_dir}")
        train_df, val_df, test_df = load_temporal_splits(
            behaviors_path=behaviors_path,
            train_val_test_ratio=(0.8, 0.1, 0.1),
            train_samples=args.train_samples,
            val_samples=args.val_samples,
            test_samples=args.test_samples,
            cache_dir=os.path.join(base_dir, "data", "processed"),
        )
    else:
        logger.info("Creating mock MIND dataset for training verification...")
        mock_dir = os.path.join(base_dir, "data", "mock")
        news_path, behaviors_path = create_mock_mind_dataset(mock_dir, num_news=150, num_behaviors=400)
        train_df, val_df, test_df = load_temporal_splits(
            behaviors_path=behaviors_path,
            train_val_test_ratio=(0.8, 0.1, 0.1),
            train_samples=250,
            val_samples=75,
            test_samples=75,
            cache_dir=None,
        )

    # 1. Compute training prior statistics (zero future leakage)
    priors_path = os.path.join(checkpoints_dir, "train_feature_priors.json")
    train_priors = compute_train_prior_statistics(train_df, output_path=priors_path)

    # 2. Initialize Train Dataset and DataLoader
    logger.info("Initializing MINDMultimodalDataset on Train partition...")
    train_dataset = MINDMultimodalDataset(
        behaviors_df=train_df,
        news_path=news_path,
        max_title_len=20,
        max_history_len=20,
        num_negatives=4,
        is_train=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_collate_fn,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Initializing TwoTowerNewsRecommender on {device}...")

    model = TwoTowerNewsRecommender(
        d_model=args.d_model,
        vocab_size=len(train_dataset.tokenizer),
        d_emb=args.d_model,
        num_heads=args.num_heads,
        temperature=0.07,
        pretrained_vision=False,
        freeze_vision_blocks=True,
        use_gating=True,
    ).to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = LinearWarmupCosineDecayLR(
        optimizer, warmup_steps=warmup_steps, total_steps=total_steps, min_lr=1e-6
    )

    # Save Tokenizer
    vocab_file = os.path.join(checkpoints_dir, "word2idx.json")
    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump(train_dataset.tokenizer.word2idx, f)
    logger.info(f"Saved vocabulary to {vocab_file}")

    # 3. Train Stage 1 Two-Tower
    train_stage1_twotower(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        checkpoint_dir=checkpoints_dir,
    )

    # 4. Build FAISS Index
    faiss_idx_path = os.path.join(checkpoints_dir, "faiss_index.bin")
    id_map_path = os.path.join(checkpoints_dir, "article_ids.json")
    build_faiss_index(model, train_dataset, device, faiss_idx_path, id_map_path)

    # 5. Train Stage 2 XGBoost LambdaMART
    xgb_model_path = os.path.join(checkpoints_dir, "lambdamart_ranker.json")
    ranker = train_stage2_xgboost_ranker(
        model=model,
        dataset=train_dataset,
        priors=train_priors,
        device=device,
        output_path=xgb_model_path,
        num_samples=min(1200, len(train_dataset)),
    )

    # 6. Initialize Validation Dataset & Tune Alpha
    logger.info("Initializing Validation dataset for hyperparameter tuning...")
    val_dataset = MINDMultimodalDataset(
        behaviors_df=val_df,
        news_path=news_path,
        tokenizer=train_dataset.tokenizer,
        max_title_len=20,
        max_history_len=20,
        is_train=False,
    )

    ensemble_cfg_path = os.path.join(checkpoints_dir, "ensemble_config.json")
    tune_ensemble_alpha_on_validation(
        model=model,
        val_dataset=val_dataset,
        xgb_ranker=ranker,
        priors=train_priors,
        device=device,
        output_path=ensemble_cfg_path,
    )

    logger.info("\n=== TRAINING & VALIDATION PIPELINE SUCCESSFULLY COMPLETED ===")
    logger.info("Stage 1 Two-Tower + Stage 2 LambdaMART trained on Train split.")
    logger.info("Ensemble alpha locked on Validation split. Ready for evaluate.py on Test split.")


if __name__ == "__main__":
    run_pipeline()
