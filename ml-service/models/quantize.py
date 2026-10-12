"""
INT8 Dynamic Quantization & CPU Acceleration for TwoTowerNewsRecommender.

Features:
- Dynamically quantizes linear layers to 8-bit integers (torch.qint8)
- Reduces model disk and memory footprint by ~60-70%
- Measures and benchmarks FP32 vs INT8 latency and memory consumption
- Enables ultra-low latency CPU inference in production edge environments
"""

import os
import time
import json
import logging
import torch
import torch.nn as nn
from typing import Dict, Any

from models.two_tower import TwoTowerNewsRecommender

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("Quantization")


def quantize_two_tower_model(
    checkpoint_path: str = "checkpoints/two_tower_best.pt",
    vocab_path: str = "checkpoints/word2idx.json",
    output_path: str = "checkpoints/two_tower_int8.pt",
    d_model: int = 256,
) -> Dict[str, Any]:
    """
    Applies PyTorch dynamic quantization to TwoTowerNewsRecommender and benchmarks gains.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_ckpt = os.path.join(base_dir, checkpoint_path)
    full_vocab = os.path.join(base_dir, vocab_path)
    full_out = os.path.join(base_dir, output_path)

    logger.info(f"Loading vocabulary from {full_vocab}...")
    with open(full_vocab, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    vocab_size = len(vocab)

    logger.info(f"Instantiating FP32 TwoTowerNewsRecommender (vocab_size={vocab_size})...")
    fp32_model = TwoTowerNewsRecommender(
        d_model=d_model,
        vocab_size=vocab_size,
        d_emb=d_model,
        num_heads=4,
        temperature=0.07,
        pretrained_vision=False,
    )

    if os.path.exists(full_ckpt):
        logger.info(f"Loading weights from {full_ckpt}...")
        ckpt = torch.load(full_ckpt, map_location="cpu", weights_only=False)
        fp32_model.load_state_dict(ckpt["model_state_dict"])
    fp32_model.eval()

    # Measure FP32 Size
    fp32_param_bytes = sum(p.numel() * p.element_size() for p in fp32_model.parameters())
    fp32_mb = fp32_param_bytes / (1024 * 1024)

    # 1. Apply Dynamic Quantization
    logger.info("Applying PyTorch dynamic INT8 quantization on nn.Linear layers...")
    int8_model = torch.ao.quantization.quantize_dynamic(
        fp32_model,
        {nn.Linear},
        dtype=torch.qint8,
    )
    int8_model.eval()

    # Save quantized model
    torch.save(int8_model.state_dict(), full_out)
    int8_file_mb = os.path.getsize(full_out) / (1024 * 1024)
    logger.info(f"Saved INT8 quantized model to {full_out} ({int8_file_mb:.2f} MB)")

    # 2. Benchmark Inference Latency
    logger.info("Running inference benchmark: 100 article forward passes...")
    batch_size = 16
    mock_tokens = torch.randint(0, vocab_size, (batch_size, 20), dtype=torch.long)
    mock_images = torch.randn(batch_size, 2048, dtype=torch.float32)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = fp32_model.encode_articles(mock_images, mock_tokens)
            _ = int8_model.encode_articles(mock_images, mock_tokens)

    # FP32 Benchmark
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(100):
            _ = fp32_model.encode_articles(mock_images, mock_tokens)
    fp32_elapsed_ms = (time.perf_counter() - t0) * 10.0  # ms per batch

    # INT8 Benchmark
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(100):
            _ = int8_model.encode_articles(mock_images, mock_tokens)
    int8_elapsed_ms = (time.perf_counter() - t0) * 10.0  # ms per batch

    speedup = fp32_elapsed_ms / max(int8_elapsed_ms, 1e-6)
    memory_reduction_pct = (1.0 - (int8_file_mb / max(fp32_mb, 1e-6))) * 100.0

    report = {
        "FP32_Parameters_MB": round(fp32_mb, 2),
        "INT8_Model_MB": round(int8_file_mb, 2),
        "Memory_Reduction_Percent": round(memory_reduction_pct, 1),
        "FP32_Latency_per_Batch_ms": round(fp32_elapsed_ms, 2),
        "INT8_Latency_per_Batch_ms": round(int8_elapsed_ms, 2),
        "Speedup_Factor": round(speedup, 2),
    }

    logger.info("=== QUANTIZATION BENCHMARK RESULTS ===")
    for k, v in report.items():
        logger.info(f"  {k:30s}: {v}")

    return report


if __name__ == "__main__":
    quantize_two_tower_model()
