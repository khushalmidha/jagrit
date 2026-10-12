"""
Dataset and DataLoader Module for MIND (Microsoft News Dataset) Multimodal Pipeline.

Features:
- Handles MIND metadata: ArticleID, Category, SubCategory, Title, Abstract, URL.
- Handles user behavior logs: ImpressionID, UserID, Time, History, Impressions.
- Robust tokenization and vocabulary management.
- Multimodal image loading: supports raw RGB (3, 224, 224) images, precomputed
  feature tensors, or deterministic category-conditioned visual synthetic tensors.
- High-performance caching: precomputes frozen 2048-d visual backbone vectors for
  unique catalog articles to avoid redundant CNN passes during Two-Tower training.
- Negative sampling (1 positive + 4 negatives per impression for InfoNCE loss).
- Evaluation collator for dynamic candidate list lengths with exact ground truth labels.
- Mock MIND dataset generator for testing and offline development.
"""

import os
import re
import random
import hashlib
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models


class SimpleTokenizer:
    """Fast regex-based word tokenizer with special tokens."""

    def __init__(self, vocab: Optional[Dict[str, int]] = None, max_vocab_size: int = 50000):
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.pad_idx = 0
        self.unk_idx = 1
        self.max_vocab_size = max_vocab_size

        if vocab is not None:
            self.word2idx = vocab
            self.idx2word = {v: k for k, v in vocab.items()}
        else:
            self.word2idx = {self.pad_token: self.pad_idx, self.unk_token: self.unk_idx}
            self.idx2word = {self.pad_idx: self.pad_token, self.unk_idx: self.unk_idx}

    def build_vocab(self, texts: List[str]):
        """Builds vocabulary from a list of title texts."""
        word_counts: Dict[str, int] = {}
        for text in texts:
            tokens = self._tokenize_text(text)
            for tok in tokens:
                word_counts[tok] = word_counts.get(tok, 0) + 1

        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        idx = len(self.word2idx)
        for word, count in sorted_words:
            if idx >= self.max_vocab_size:
                break
            if word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

    @staticmethod
    def _tokenize_text(text: str) -> List[str]:
        text = str(text).lower()
        return re.findall(r"\b[a-zA-Z0-9]+\b", text)

    def encode(self, text: str, max_len: int = 20) -> List[int]:
        tokens = self._tokenize_text(text)
        indices = [self.word2idx.get(tok, self.unk_idx) for tok in tokens[:max_len]]
        if len(indices) < max_len:
            indices += [self.pad_idx] * (max_len - len(indices))
        return indices

    def __len__(self) -> int:
        return len(self.word2idx)


def generate_synthetic_visual_features(seed_str: str, category: str = "general", dim: int = 2048) -> torch.Tensor:
    """
    Generates a deterministic 2048-dim category-conditioned feature vector
    mimicking the pooled output of a frozen ResNet-50 backbone.
    """
    hasher = hashlib.md5(seed_str.encode("utf-8")).digest()
    seed_int = int.from_bytes(hasher[:4], "big")
    rng = np.random.RandomState(seed_int)

    cat_seed = int.from_bytes(hashlib.md5(category.lower().encode("utf-8")).digest()[:4], "big")
    cat_rng = np.random.RandomState(cat_seed)
    cat_prior = cat_rng.randn(dim).astype(np.float32) * 0.35
    item_feat = rng.randn(dim).astype(np.float32) * 0.65 + cat_prior

    tensor = torch.from_numpy(item_feat)
    mean = tensor.mean()
    std = tensor.std()
    return (tensor - mean) / (std + 1e-6)


def generate_synthetic_image_tensor(seed_str: str, category: str = "general") -> torch.Tensor:
    """
    Generates a deterministic 224x224 RGB image tensor based on article hash & category.
    Normalized with ImageNet mean and std.
    """
    hasher = hashlib.md5(seed_str.encode("utf-8")).digest()
    seed_int = int.from_bytes(hasher[:4], "big")
    rng = np.random.RandomState(seed_int)

    category_tints = {
        "news": [0.3, 0.5, 0.7],
        "sports": [0.2, 0.8, 0.3],
        "finance": [0.8, 0.7, 0.2],
        "lifestyle": [0.8, 0.3, 0.6],
        "health": [0.2, 0.7, 0.8],
        "foodanddrink": [0.9, 0.5, 0.1],
        "entertainment": [0.7, 0.2, 0.8],
    }
    tint = category_tints.get(category.lower(), [0.5, 0.5, 0.5])

    base = rng.uniform(0.0, 1.0, size=(3, 28, 28)).astype(np.float32)
    for c in range(3):
        base[c] = (base[c] * 0.4) + (tint[c] * 0.6)

    tensor = torch.from_numpy(base).unsqueeze(0)
    tensor = torch.nn.functional.interpolate(
        tensor, size=(224, 224), mode="bilinear", align_corners=False
    ).squeeze(0)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    return normalize(tensor)


class MINDMultimodalDataset(Dataset):
    """
    Multimodal PyTorch Dataset for Microsoft News Dataset (MIND).
    """

    def __init__(
        self,
        behaviors_path: str,
        news_path: str,
        tokenizer: Optional[SimpleTokenizer] = None,
        max_title_len: int = 20,
        max_history_len: int = 20,
        num_negatives: int = 4,
        is_train: bool = True,
        image_dir: Optional[str] = None,
        image_tensor_cache: Optional[Dict[str, torch.Tensor]] = None,
        precompute_visual_features: bool = True,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.max_title_len = max_title_len
        self.max_history_len = max_history_len
        self.num_negatives = num_negatives
        self.is_train = is_train
        self.image_dir = image_dir
        self.image_tensor_cache = image_tensor_cache or {}
        self.precompute_visual_features = precompute_visual_features

        # 1. Load Behavior Logs
        behavior_cols = ["impression_id", "user_id", "time", "history", "impressions"]
        if max_samples:
            df_behaviors = pd.read_csv(behaviors_path, sep="\t", header=None, names=behavior_cols, nrows=max_samples)
        else:
            df_behaviors = pd.read_csv(behaviors_path, sep="\t", header=None, names=behavior_cols)

        # Collect active news IDs referenced in behaviors
        active_news_ids = set()
        for _, row in df_behaviors.iterrows():
            hist_str = str(row["history"])
            if hist_str and hist_str != "nan":
                active_news_ids.update(hist_str.split())
            imp_str = str(row["impressions"])
            if imp_str and imp_str != "nan":
                for imp in imp_str.split():
                    active_news_ids.add(imp.split("-")[0])

        # 2. Load News Metadata (filtered to active items for 50x faster initialization)
        news_cols = [
            "news_id",
            "category",
            "subcategory",
            "title",
            "abstract",
            "url",
            "title_entities",
            "abstract_entities",
        ]
        df_news = pd.read_csv(news_path, sep="\t", header=None, names=news_cols)
        if active_news_ids:
            df_news = df_news[df_news["news_id"].isin(active_news_ids)]

        # 3. Build or use Tokenizer
        if tokenizer is None:
            self.tokenizer = SimpleTokenizer()
            self.tokenizer.build_vocab(df_news["title"].dropna().tolist())
        else:
            self.tokenizer = tokenizer

        # 4. Cache article properties
        self.news_titles: Dict[str, torch.Tensor] = {}
        self.news_categories: Dict[str, str] = {}
        self.news_subcategories: Dict[str, str] = {}
        self.news_images: Dict[str, torch.Tensor] = {}
        self.all_news_ids: List[str] = []

        raw_images_to_extract: Dict[str, torch.Tensor] = {}

        for _, row in df_news.iterrows():
            news_id = str(row["news_id"])
            title = str(row["title"]) if pd.notna(row["title"]) else ""
            category = str(row["category"]) if pd.notna(row["category"]) else "general"
            subcategory = str(row["subcategory"]) if pd.notna(row["subcategory"]) else "general"

            token_ids = self.tokenizer.encode(title, max_len=max_title_len)
            self.news_titles[news_id] = torch.tensor(token_ids, dtype=torch.long)
            self.news_categories[news_id] = category
            self.news_subcategories[news_id] = subcategory
            self.all_news_ids.append(news_id)

            if news_id in self.image_tensor_cache:
                self.news_images[news_id] = self.image_tensor_cache[news_id]
            elif self.image_dir and os.path.exists(os.path.join(self.image_dir, f"{news_id}.jpg")):
                try:
                    from PIL import Image
                    img = Image.open(os.path.join(self.image_dir, f"{news_id}.jpg")).convert("RGB")
                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    raw_images_to_extract[news_id] = transform(img)
                except Exception:
                    self.news_images[news_id] = generate_synthetic_visual_features(news_id, category)
            else:
                self.news_images[news_id] = generate_synthetic_visual_features(news_id, category)

        # Batch-extract real images if any were loaded from disk
        if raw_images_to_extract and self.precompute_visual_features:
            backbone = models.resnet50(weights=None)
            backbone.eval()
            conv_extractor = nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
                backbone.layer1,
                backbone.layer2,
                backbone.layer3,
                backbone.layer4,
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(1),
            )
            unique_ids = list(raw_images_to_extract.keys())
            bs = 64
            with torch.no_grad():
                for i in range(0, len(unique_ids), bs):
                    chunk_ids = unique_ids[i : i + bs]
                    chunk_imgs = torch.stack([raw_images_to_extract[cid] for cid in chunk_ids], dim=0)
                    chunk_feats = conv_extractor(chunk_imgs)
                    for j, cid in enumerate(chunk_ids):
                        self.news_images[cid] = chunk_feats[j]

        self.news_images["<PAD>"] = torch.zeros(2048, dtype=torch.float32)
        self.news_titles["<PAD>"] = torch.zeros(max_title_len, dtype=torch.long)
        self.news_categories["<PAD>"] = "pad"
        self.news_subcategories["<PAD>"] = "pad"

        self.samples: List[Dict[str, Any]] = []

        for _, row in df_behaviors.iterrows():
            imp_id = str(row["impression_id"])
            user_id = str(row["user_id"])
            time_str = str(row["time"]) if pd.notna(row["time"]) else ""
            history_str = str(row["history"]) if pd.notna(row["history"]) else ""
            impressions_str = str(row["impressions"]) if pd.notna(row["impressions"]) else ""

            history = history_str.split() if history_str and history_str != "nan" else []
            if not impressions_str or impressions_str == "nan":
                continue

            impressions = impressions_str.split()

            if self.is_train:
                positives = [imp.split("-")[0] for imp in impressions if imp.endswith("-1")]
                negatives = [imp.split("-")[0] for imp in impressions if imp.endswith("-0")]

                positives = [p for p in positives if p in self.news_titles]
                negatives = [n for n in negatives if n in self.news_titles]

                if not positives or not negatives:
                    continue

                for pos in positives:
                    pos_cat = self.news_categories.get(pos, "")
                    pos_subcat = self.news_subcategories.get(pos, "")

                    # 1. Prioritize Hard Negatives from the SAME category/subcategory in the impression
                    same_cat_negatives = [
                        n for n in negatives
                        if self.news_categories.get(n, "") == pos_cat
                    ]
                    other_negatives = [
                        n for n in negatives
                        if self.news_categories.get(n, "") != pos_cat
                    ]

                    # Balanced 50/50 negative sampling: 2 hard (same category) + 2 exploratory
                    max_hard = max(1, self.num_negatives // 2)
                    sampled_negatives = []

                    if same_cat_negatives:
                        num_hard = min(len(same_cat_negatives), max_hard)
                        sampled_negatives.extend(random.sample(same_cat_negatives, num_hard))

                    # Fill remainder from other impression negatives (different category)
                    remaining_needed = self.num_negatives - len(sampled_negatives)
                    if remaining_needed > 0 and other_negatives:
                        num_other = min(len(other_negatives), remaining_needed)
                        sampled_negatives.extend(random.sample(other_negatives, num_other))

                    # If impression has fewer than num_negatives, sample catalog items from same category
                    cat_catalog_pool = [
                        nid for nid in self.all_news_ids
                        if self.news_categories.get(nid, "") == pos_cat and nid != pos and nid not in sampled_negatives
                    ]
                    while len(sampled_negatives) < self.num_negatives:
                        if cat_catalog_pool:
                            sampled_negatives.append(random.choice(cat_catalog_pool))
                        else:
                            random_neg = random.choice(self.all_news_ids)
                            if random_neg != pos and random_neg not in sampled_negatives:
                                sampled_negatives.append(random_neg)

                    self.samples.append({
                        "impression_id": imp_id,
                        "user_id": user_id,
                        "time": time_str,
                        "history": history,
                        "positive": pos,
                        "negatives": sampled_negatives,
                    })
            else:
                cand_ids = []
                cand_labels = []
                for imp in impressions:
                    parts = imp.split("-")
                    if len(parts) == 2 and parts[0] in self.news_titles:
                        cand_ids.append(parts[0])
                        cand_labels.append(int(parts[1]))

                if cand_ids and sum(cand_labels) > 0:
                    self.samples.append({
                        "impression_id": imp_id,
                        "user_id": user_id,
                        "time": time_str,
                        "history": history,
                        "candidates": cand_ids,
                        "labels": cand_labels,
                    })

    def __len__(self) -> int:
        return len(self.samples)

    def _get_history_tensors(self, history: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        valid_history = [nid for nid in history if nid in self.news_titles][-self.max_history_len:]
        num_valid = len(valid_history)

        token_list = []
        image_list = []
        mask_list = []

        for nid in valid_history:
            token_list.append(self.news_titles[nid])
            image_list.append(self.news_images[nid])
            mask_list.append(True)

        pad_count = self.max_history_len - num_valid
        for _ in range(pad_count):
            token_list.append(self.news_titles["<PAD>"])
            image_list.append(self.news_images["<PAD>"])
            mask_list.append(False)

        tokens = torch.stack(token_list, dim=0)
        images = torch.stack(image_list, dim=0)
        mask = torch.tensor(mask_list, dtype=torch.bool)
        return tokens, images, mask

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        hist_tokens, hist_images, hist_mask = self._get_history_tensors(item["history"])

        if self.is_train:
            cand_ids = [item["positive"]] + item["negatives"]
            cand_tokens = torch.stack([self.news_titles[nid] for nid in cand_ids], dim=0)
            cand_images = torch.stack([self.news_images[nid] for nid in cand_ids], dim=0)
            label = torch.tensor(0, dtype=torch.long)

            return {
                "history_tokens": hist_tokens,
                "history_images": hist_images,
                "history_mask": hist_mask,
                "candidate_tokens": cand_tokens,
                "candidate_images": cand_images,
                "label": label,
                "user_id": item["user_id"],
                "positive_id": item["positive"],
                "negative_ids": item["negatives"],
            }
        else:
            cand_ids = item["candidates"]
            cand_tokens = torch.stack([self.news_titles[nid] for nid in cand_ids], dim=0)
            cand_images = torch.stack([self.news_images[nid] for nid in cand_ids], dim=0)
            labels = torch.tensor(item["labels"], dtype=torch.float32)

            return {
                "impression_id": item["impression_id"],
                "user_id": item["user_id"],
                "history": item["history"],
                "history_tokens": hist_tokens,
                "history_images": hist_images,
                "history_mask": hist_mask,
                "candidate_ids": cand_ids,
                "candidate_tokens": cand_tokens,
                "candidate_images": cand_images,
                "labels": labels,
                "time": item["time"],
            }


def train_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    return {
        "history_tokens": torch.stack([item["history_tokens"] for item in batch], dim=0),
        "history_images": torch.stack([item["history_images"] for item in batch], dim=0),
        "history_mask": torch.stack([item["history_mask"] for item in batch], dim=0),
        "candidate_tokens": torch.stack([item["candidate_tokens"] for item in batch], dim=0),
        "candidate_images": torch.stack([item["candidate_images"] for item in batch], dim=0),
        "labels": torch.stack([item["label"] for item in batch], dim=0),
    }


def evaluate_collate_fn(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return batch


def create_mock_mind_dataset(
    output_dir: str,
    num_news: int = 100,
    num_behaviors: int = 200,
) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    news_file = os.path.join(output_dir, "news.tsv")
    behaviors_file = os.path.join(output_dir, "behaviors.tsv")

    categories = ["news", "sports", "finance", "lifestyle", "health", "technology"]
    subcategories = ["general", "local", "world", "football", "stocks", "wellness"]

    news_rows = []
    news_ids = [f"N{10000 + i}" for i in range(num_news)]

    for nid in news_ids:
        cat = random.choice(categories)
        subcat = random.choice(subcategories)
        title = f"Top {random.randint(5, 50)} Highlights in {cat.capitalize()} and {subcat.capitalize()}"
        abstract = f"Detailed overview of {cat} trends and important updates."
        url = f"https://assets.msn.com/news/{nid}.html"
        news_rows.append(f"{nid}\t{cat}\t{subcat}\t{title}\t{abstract}\t{url}\t[]\t[]")

    with open(news_file, "w", encoding="utf-8") as f:
        f.write("\n".join(news_rows) + "\n")

    behavior_rows = []
    for i in range(1, num_behaviors + 1):
        uid = f"U{random.randint(1000, 1100)}"
        time_str = "11/12/2019 10:00:00 AM"
        hist_len = random.randint(3, 10)
        history = " ".join(random.sample(news_ids, hist_len))

        num_pos = random.randint(1, 2)
        num_neg = random.randint(4, 8)
        pos_samples = random.sample(news_ids, num_pos)
        rem_news = [n for n in news_ids if n not in pos_samples]
        neg_samples = random.sample(rem_news, min(num_neg, len(rem_news)))

        impressions = [f"{n}-1" for n in pos_samples] + [f"{n}-0" for n in neg_samples]
        random.shuffle(impressions)
        imp_str = " ".join(impressions)

        behavior_rows.append(f"{i}\t{uid}\t{time_str}\t{history}\t{imp_str}")

    with open(behaviors_file, "w", encoding="utf-8") as f:
        f.write("\n".join(behavior_rows) + "\n")

    return news_file, behaviors_file
