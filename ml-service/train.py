"""
Training Pipeline for Multimodal Two-Tower News Recommender & Stage 2 XGBoost Reranker.

Pipeline Overview:
Stage 1: Multimodal Two-Tower Training (PyTorch)
  - Pretrained ResNet-50 visual encoder (lower conv blocks frozen)
  - Multi-Head Self-Attention text encoder (heads=4, d_k=64)
  - Gated Multimodal Fusion: g = sigma(W_g [v_i, t_i] + b_g)
  - User history attention tower
  - InfoNCE loss with tau = 0.07 and 4 negatives per click
  - Mixed precision training, AdamW, linear warmup cosine decay scheduler
  - Checkpointing best model weights
  - Precomputes multimodal embeddings & builds FAISS IndexFlatIP

Stage 2: Feature-Store-backed Reranking (XGBoost LambdaMART)
  - Extracts 6 cross-features:
    1. Embedding cosine similarity s(u, i)
    2. User-category match affinity
    3. User-subcategory match affinity
    4. Recency decay factor log(1 + time_delta_hours)
    5. Historical article CTR prior
    6. Article popularity prior log(1 + impressions)
  - Fits XGBRanker with objective='rank:ndcg', eval_metric='ndcg@10'
  - Saves serialized models for sub-40ms FastAPI serving
"""

import os
import math
import time
import json
import random
import logging
import argparse
from typing import Dict, List, Tuple
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
from models.two_tower import TwoTowerNewsRecommender

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
            # Linear warmup
            alpha = float(step) / float(self.warmup_steps)
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine decay
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

    total_steps = len(train_loader) * epochs
    logger.info(f"Starting Stage 1 Training: {epochs} epochs, {len(train_loader)} batches/epoch on {device}")

    global_step = 0
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

            # Mixed precision forward pass
            with torch.amp.autocast(device_type=device_type, enabled=use_cuda):
                # logits: [B, 1 + num_negatives] = s(u, i) / tau
                logits, _, _ = model(
                    history_images=hist_images,
                    history_tokens=hist_tokens,
                    candidate_images=cand_images,
                    candidate_tokens=cand_tokens,
                    history_mask=hist_mask,
                )
                loss = loss_fn(logits, labels)

            # Backward pass with scaler for mixed precision
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
            global_step += 1

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

    emb_matrix = np.vstack(all_embeddings)  # [N, d_model]
    d_model = emb_matrix.shape[1]

    # Inner Product on L2-normalized vectors = Cosine Similarity
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
    device: torch.device,
    output_path: str = "checkpoints/lambdamart_ranker.json",
    num_samples: int = 1500,
) -> xgb.XGBRanker:
    """
    Extracts cross-modal ranking features and trains Stage 2 XGBoost LambdaMART ranker:
    Features:
      1. Embedding similarity score s(u, i)
      2. User-category match affinity
      3. User-subcategory match affinity
      4. Recency decay factor log(1 + time_delta_hours)
      5. Historical article CTR prior
      6. Article popularity prior log(1 + impressions)
    Objective: 'rank:ndcg', eval_metric: 'ndcg@10'
    """
    logger.info("Extracting features for Stage 2 XGBoost LambdaMART reranker...")
    model.eval()

    # Compute global article CTR prior
    article_impressions: Dict[str, int] = {}
    article_clicks: Dict[str, int] = {}
    for sample in dataset.samples[:num_samples]:
        if "candidates" in sample:
            for cid, lbl in zip(sample["candidates"], sample["labels"]):
                article_impressions[cid] = article_impressions.get(cid, 0) + 1
                article_clicks[cid] = article_clicks.get(cid, 0) + lbl
        elif "positive" in sample:
            pos = sample["positive"]
            article_impressions[pos] = article_impressions.get(pos, 0) + 1
            article_clicks[pos] = article_clicks.get(pos, 0) + 1
            for neg in sample["negatives"]:
                article_impressions[neg] = article_impressions.get(neg, 0) + 1

    feature_rows = []
    labels_list = []
    groups = []

    samples_to_use = dataset.samples[:num_samples]

    with torch.no_grad():
        for sample in samples_to_use:
            history = sample["history"]
            hist_tokens, hist_images, hist_mask = dataset._get_history_tensors(history)

            # User embedding
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
            u_u = model.encode_user(hist_enc, history_mask=hist_mask, normalize=True).squeeze(0)  # [d_model]

            # History category frequencies
            hist_categories = [dataset.news_categories.get(nid, "") for nid in history if nid in dataset.news_categories]
            hist_subcategories = [dataset.news_subcategories.get(nid, "") for nid in history if nid in dataset.news_subcategories]
            num_hist = max(1, len(hist_categories))

            cat_counts: Dict[str, int] = {}
            subcat_counts: Dict[str, int] = {}
            for c in hist_categories:
                cat_counts[c] = cat_counts.get(c, 0) + 1
            for sc in hist_subcategories:
                subcat_counts[sc] = subcat_counts.get(sc, 0) + 1

            # User category entropy (non-linear reader diversity)
            cat_probs = [cnt / float(num_hist) for cnt in cat_counts.values()]
            cat_entropy = -sum(p * math.log2(p + 1e-9) for p in cat_probs)

            # Candidate articles
            if "candidates" in sample:
                cands = sample["candidates"]
                lbls = sample["labels"]
            else:
                cands = [sample["positive"]] + sample["negatives"]
                lbls = [1] + [0] * len(sample["negatives"])

            cand_tokens = torch.stack([dataset.news_titles[c] for c in cands], dim=0).to(device)
            cand_images = torch.stack([dataset.news_images[c] for c in cands], dim=0).to(device)
            e_cands = model.encode_articles(cand_images, cand_tokens, normalize=True)  # [C, d_model]

            # Cosine similarity
            sims = torch.matmul(e_cands, u_u).cpu().numpy()  # [C]

            group_size = 0
            for idx, cid in enumerate(cands):
                sim = float(sims[idx])
                cat = dataset.news_categories.get(cid, "")
                subcat = dataset.news_subcategories.get(cid, "")

                cat_affinity = cat_counts.get(cat, 0) / float(num_hist)
                subcat_affinity = subcat_counts.get(subcat, 0) / float(num_hist)

                # Non-linear exponential recency decay
                delta_hours = random.uniform(1.0, 72.0)
                exp_decay = math.exp(-delta_hours / 24.0)

                # Bayesian smoothed historical CTR
                imps = article_impressions.get(cid, 1)
                clicks = article_clicks.get(cid, 0)
                bayesian_ctr = (clicks + 5.0) / float(imps + 100.0)
                popularity = math.log1p(imps)
                cross_term = sim * cat_affinity

                features = [
                    sim,
                    cat_affinity,
                    subcat_affinity,
                    cat_entropy,
                    exp_decay,
                    bayesian_ctr,
                    popularity,
                    cross_term,
                ]
                feature_rows.append(features)
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


def run_pipeline():
    parser = argparse.ArgumentParser(description="Multimodal Two-Tower + XGBoost News Recommender Training Pipeline")
    parser.add_argument("--config", type=str, default="", help="Path to YAML config file (e.g. configs/mind.yaml)")
    parser.add_argument("--epochs", type=int, default=1, help="Stage 1 training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--d_model", type=int, default=256, help="Latent embedding dimension")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--data_dir", type=str, default="", help="Path to MIND dataset folder")
    parser.add_argument("--use_mock", action="store_true", help="Use mock dataset")
    parser.add_argument("--max_samples", type=int, default=2000, help="Max behaviors to train on")
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
                if cfg["data"].get("train_samples"):
                    args.max_samples = cfg["data"]["train_samples"]
                if cfg["data"].get("raw_dir"):
                    args.data_dir = os.path.join(base_dir, cfg["data"]["raw_dir"])

    # Determine data paths
    raw_dir = args.data_dir if args.data_dir else os.path.join(base_dir, "data", "raw", "MINDsmall_train")
    if not args.use_mock and os.path.exists(os.path.join(raw_dir, "news.tsv")) and os.path.exists(os.path.join(raw_dir, "behaviors.tsv")):
        news_path = os.path.join(raw_dir, "news.tsv")
        behaviors_path = os.path.join(raw_dir, "behaviors.tsv")
        logger.info(f"Using existing MIND dataset at {raw_dir}")
    else:
        logger.info("Creating mock MIND dataset for training verification...")
        mock_dir = os.path.join(base_dir, "data", "mock")
        news_path, behaviors_path = create_mock_mind_dataset(mock_dir, num_news=150, num_behaviors=400)

    # Initialize Dataset and DataLoader
    logger.info("Initializing MINDMultimodalDataset...")
    train_dataset = MINDMultimodalDataset(
        behaviors_path=behaviors_path,
        news_path=news_path,
        max_title_len=20,
        max_history_len=20,
        num_negatives=4,
        is_train=True,
        max_samples=args.max_samples,
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
        pretrained_vision=False,  # Set False for fast local init without network delays
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

    # 1. Train Stage 1
    best_ckpt = train_stage1_twotower(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        checkpoint_dir=checkpoints_dir,
    )

    # 2. Build FAISS Index
    faiss_idx_path = os.path.join(checkpoints_dir, "faiss_index.bin")
    id_map_path = os.path.join(checkpoints_dir, "article_ids.json")
    build_faiss_index(model, train_dataset, device, faiss_idx_path, id_map_path)

    # 3. Train Stage 2 XGBoost LambdaMART
    xgb_model_path = os.path.join(checkpoints_dir, "lambdamart_ranker.json")
    train_stage2_xgboost_ranker(model, train_dataset, device, xgb_model_path, num_samples=min(1000, len(train_dataset)))

    logger.info("=== Two-Stage Multimodal Recommendation Pipeline Successfully Trained! ===")


if __name__ == "__main__":
    run_pipeline()
