# Reproducibility Guide: Jagrit Multimodal News Recommendation Engine

This document provides complete, deterministic instructions to reproduce all data splits, training runs, validation hyperparameter selections, test benchmarks, and ablation studies reported in this repository.

---

## 1. Environment & Hardware Specifications

- **Operating System:** Windows 11 / Linux (Ubuntu 20.04+)
- **Python Version:** 3.10+
- **Primary Libraries:**
  - `torch`: 2.1.0+ (CPU or CUDA)
  - `xgboost`: 2.0.0+
  - `faiss-cpu`: 1.7.4+
  - `pandas`: 2.0.0+
  - `numpy`: 1.24.0+
  - `pydantic`: 2.0.0+
  - `fastapi` / `uvicorn`: 0.100.0+

### Installation

```bash
cd ml-service
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux:
# source venv/bin/activate

pip install -r requirements.txt
pip install -r requirements-dl.txt
```

---

## 2. Dataset Protocol & Leakage Prevention

### Dataset Source
Microsoft News Dataset (MIND-small):
- Source Impressions: 156,965 sessions
- Catalog News Articles: 65,238 articles
- Timestamp Span: `2019-11-09 00:00:19` to `2019-11-14 23:59:13`

### Temporal Splitting Protocol
To strictly prevent future news exposure and temporal target leakage:
1. **Chronological Sorting:** All impressions are sorted strictly by impression timestamp $t_i$.
2. **Atomic Impression Partitioning:**
   - **Train Split (80%):** 125,572 sessions (`2019-11-09 00:00:19` $\rightarrow$ `2019-11-13 20:36:19`)
   - **Validation Split (10%):** 15,696 sessions (`2019-11-13 20:36:26` $\rightarrow$ `2019-11-14 10:45:09`)
   - **Test Split (10%):** 15,697 sessions (`2019-11-14 10:45:09` $\rightarrow$ `2019-11-14 23:59:13`)
3. **Mathematical Leakage Assertion:**
   $$\max(T_{\text{train}}) \le \min(T_{\text{val}}) \le \max(T_{\text{val}}) \le \min(T_{\text{test}})$$
   Enforced in `data/split.py`. If violated, pipeline execution immediately raises a fatal exception.
4. **Leak-Free Prior Statistics:**
   Global CTR $p_{\text{global}}$ and Bayesian priors $(\alpha_{\text{prior}}, \beta_{\text{prior}})$ are computed **exclusively** on the Train partition and frozen into `checkpoints/train_feature_priors.json`. Validation and Test sets NEVER leak historical click rates into feature computation.

---

## 3. End-to-End Execution Sequence

### Step 1: Model Training
Trains the Stage 1 Two-Tower neural network and Stage 2 XGBoost LambdaMART ranker on the **Train** partition:

```bash
python train.py --config configs/mind.yaml
```

**Artifacts Generated:**
- `results/split_manifest.json`: Verification manifest of partition boundaries, unique users, and article counts.
- `checkpoints/train_feature_priors.json`: Train-only Bayesian smoothed CTR priors and article first-seen timestamps.
- `checkpoints/word2idx.json`: Text vocabulary tokenizer dictionary.
- `checkpoints/two_tower_best.pt`: Two-Tower model weights.
- `checkpoints/lambdamart_ranker.json`: Trained XGBoost LambdaMART ranker.
- `checkpoints/faiss_index.bin`: Inner-product FAISS vector index of candidate news embeddings.

---

### Step 2: Validation Hyperparameter Tuning
Evaluates Stage 1 and Stage 2 on the **Validation** partition across candidate ensemble blending weights $\alpha \in [0.0, 1.0]$ and MMR diversity parameters $\lambda \in [0.25, 1.0]$. The optimal configuration maximizing validation nDCG@10 is frozen:

```bash
python evaluate.py --config configs/mind.yaml --split validation --checkpoint checkpoints/two_tower_best.pt --tune
```

**Artifacts Generated:**
- `results/best_validation_config.json`: Frozen validation configuration including $\alpha^* = 0.60$, $\lambda^* = 0.75$, validation nDCG@10, and sweep curves.
- `checkpoints/ensemble_config.json`: Mirrored config loaded automatically by online inference engines.

---

### Step 3: Single-Pass Test Evaluation
Evaluates the locked pipeline on the **Test** partition using the frozen hyperparameters. Zero test tuning is permitted:

```bash
python evaluate.py --config configs/mind.yaml --split test --checkpoint checkpoints/two_tower_best.pt --frozen-config results/best_validation_config.json
```

**Artifacts Generated:**
- `results/test_results.json`: Complete metrics breakdown (nDCG@5, nDCG@10, MRR, ROC-AUC, PR-AUC, Cohen's Kappa, ILD, P50 and P95 latency).

---

### Step 4: Full Architectural Ablation Study
Executes the component-by-component ablation benchmark across all 6 variants on the untouched Test partition:

```bash
python experiments/run_ablation.py --config configs/mind.yaml
```

**Artifacts Generated:**
- `results/ablation_results.json`: Detailed JSON record of all 6 architectural variants.
- `results/ablation_results.csv`: Tabular CSV record for reporting and spreadsheet analysis.

---

### Step 5: Automated Documentation Synchronization
Updates both `ml-service/README.md` and repository root `README.md` directly from the generated JSON artifacts:

```bash
python scripts/sync_benchmarks_to_readme.py
```

---

## 4. Serving & Live Inference Verification

Launch the FastAPI microservice:

```bash
uvicorn serve:app --host 0.0.0.0 --port 8000
```

Verify the endpoint with an HTTP request:

```bash
curl -X POST http://localhost:8000/rank \
  -H "Content-Type: application/json" \
  -d '{"user_id": "U13740", "top_k": 5}'
```
