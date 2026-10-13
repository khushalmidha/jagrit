# Jagrit — Personalized AI News Platform

Jagrit is a fully functional, personalized bilingual news platform that combines real-time news ingestion with machine learning to deliver a curated "For You" experience. Inspired by premium editorial websites (like The Hindu or New York Times), Jagrit uses a microservices architecture featuring a React frontend, a Node.js API backend, and a Python-based ML recommendation engine.

## 🚀 Key Features

* **Real-time News Ingestion**: Automatically aggregates live breaking news from **NewsAPI** and premium **RSS Feeds** (Times of India, The Hindu, BBC, CNBC), extracting real high-quality images and full summaries.
* **ML-Powered "For You" Feed**: Implements a 2-stage multimodal recommendation pipeline—combining a PyTorch **Multimodal Two-Tower** retrieval model (ResNet-50 + Multi-Head Attention NLP) with **XGBoost LambdaMART** reranking and **MMR** diversification, achieving **0.2877 nDCG@10** (Two-Tower) on a chronologically isolated MIND test split.
* **Bilingual Experience (AI Translation)**: Integrates the **Google Gemini (1.5 Flash) API** to instantly translate English news headlines and abstracts into Hindi. Translations are intelligently cached in **MongoDB** to guarantee fast load times and minimize API costs.
* **AI News Assistant (RAG Pipeline)**: Features a Retrieval-Augmented Generation pipeline using Gemini 1.5 Flash, allowing users to ask natural language questions about the latest news and get context-aware answers directly from the real-time feature store.
* **Hybrid Feed Layout**: The UI splits the feed into a premium **For You** hero section (top 6 ML-ranked articles) and a **Global News** grid for trending worldwide stories.
* **Category Filtering**: A sleek sidebar allows users to filter the entire feed by specific topics (Politics, Technology, Business, Sports, etc.).
* **Echo-Chamber Prevention (MMR Diversity)**: Employs **Maximal Marginal Relevance (MMR)** selection with a tunable tradeoff parameter $\lambda=0.75$, guaranteeing fresh topic coverage across the Top-10 personalized feed.
* **Implicit Skip & Negative Dwell-Time Ingestion**: Tracks quick bounces (<3s) and implicit skips via a real-time `/feedback` endpoint, applying dynamic category fatigue penalties to eliminate repetitive recommendations.
* **Saved News Archive**: Users can bookmark their favorite articles and access them later in a dedicated, personalized "Saved News" dashboard.
* **Zero-Downtime Microservices**: Designed to be deployed across Vercel (Frontend), Render (Backend), and a dedicated VM for the Python ML Engine + Redis.

---

## 🛠️ Technology Stack

### Frontend (Web)
* **React.js** (Vite): Fast, modern component-based UI.
* **Tailwind CSS**: Custom, responsive editorial-style design without bloated stylesheets.
* **React Router v6**: Client-side routing for seamless navigation.

### Backend (API & Ingestion)
* **Node.js & Express.js**: Handles user authentication (JWT), preferences, and live news fetching.
* **MongoDB (Mongoose)**: Primary database for user data, preferences, and translated text caching.
* **RSS Parser & Axios**: Used in scheduled ingestion cron jobs to parse live news sources.
* **Google Gemini API**: Powers the English-to-Hindi translation layer.

### ML & Recommendation Service
* **PyTorch**: Custom from-scratch Multimodal Two-Tower architecture (ResNet-50 vision, Multi-Head Self-Attention NLP, Gated Cross-Modal Fusion, User History Attention, InfoNCE loss).
* **FAISS**: In-memory dense vector indexing (`IndexFlatIP`) for candidate generation.
* **XGBoost (LambdaMART)**: Non-linear reranker scoring multi-dimensional cross signals (Bayesian CTR, category affinities, recency decay).
* **Ensemble Learning**: Calibrated score fusion ($\alpha^* = 0.60$) combining semantic and tabular features.
* **FastAPI**: Microservice serving endpoint (`/recommend`).
* **Redis**: Feature store caching recent article vectors and popularity counters.
* **Kafka**: Real-time event streaming for user clicks and impressions.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Client ["Client Layer"]
        UI["💻 React / Vite Editorial UI<br/>(Personalized Feed, Hindi Toggle, RAG Q&A)"]
    end

    subgraph Backend ["Backend & Data Ingestion (Node.js)"]
        API["⚙️ Express API Gateway<br/>(Auth, Translation Cache, Ingestion Crons)"]
        Mongo[("MongoDB Atlas<br/>(Users, Bookmarks, Translations)")]
        LiveRSS["📰 NewsAPI & RSS Feeds<br/>(Times of India, Hindu, BBC)"]
        LiveRSS -->|Hourly Cron| API
        API <--> Mongo
    end

    subgraph MLPipeline ["ML Recommendation Engine (FastAPI)"]
        FastAPIApp["🚀 FastAPI Serving Layer"]
        RedisFS[("🗄️ Redis Feature Store<br/>(ZSETs, Recency, CTR)")]
        
        subgraph Stage1Retrieval ["Stage 1: Two-Tower Candidate Retrieval"]
            TwoTower["🧠 PyTorch Two-Tower Model<br/>• ResNet-50 Visual Tower<br/>• Multi-Head Self-Attention NLP<br/>• Gated Cross-Modal Fusion"]
            FAISSIndex[("⚡ FAISS IndexFlatIP<br/>(14.8k Articles | <0.2ms)")]
            TwoTower --> FAISSIndex
        end

        subgraph Stage2Rerank ["Stage 2: Non-Linear Ensemble Reranker"]
            LambdaMART["🌲 XGBoost LambdaMART<br/>(Objective: rank:ndcg)"]
            CrossFeats["⚙️ Cross-Modal Signals<br/>(Cosine Sim, Bayesian CTR, Affinities)"]
            CrossFeats --> LambdaMART
        end

        ScoreBlend["⚖️ Calibrated Score Fusion (α=0.60)<br/>S_ensemble = α · S_TT + (1-α) · S_Rerank"]
        FAISSIndex --> CrossFeats
        LambdaMART --> ScoreBlend
        FastAPIApp --> Stage1Retrieval
        ScoreBlend --> FastAPIApp
        API -->|Fetch Features / Stream| RedisFS
        RedisFS --> CrossFeats
    end

    subgraph LLM ["Generative AI Layer"]
        GeminiFlash["🤖 Google Gemini 1.5 Flash<br/>• Real-time English-to-Hindi Translation<br/>• Live News Context RAG Q&A"]
    end

    UI -->|Read Stories / Toggle Hindi| API
    UI -->|Ask News Question| API
    API -->|Prompt with Context| GeminiFlash
    API -->|Fetch Personalized Top-K| FastAPIApp
    FastAPIApp -->|Top-10 News JSON| API
```

---

## 📊 Rigorous Evaluation & Verified Test Holdout

Evaluated on a chronological **80/10/10 split** of Microsoft News Dataset (MIND) sessions with **validation-only hyperparameter selection** and an **untouched test partition** (zero test-set tuning).

### Verified Test Holdout Summary

* **Stage 1: Multimodal Two-Tower (PyTorch)**
  * **nDCG@10:** 0.2877
  * **nDCG@5:** 0.2210
  * **MRR:** 0.2474
  * **Offline Pipeline Latency (P95, CPU)*:** 65.00 ms
* **Stage 2: LambdaMART (XGBoost alone)**
  * **nDCG@10:** 0.2616
  * **Offline Tree Ranking Latency (P95, CPU)*:** 4.09 ms
* **Frozen Ensemble ($\alpha^* = 0.60$)**
  * **Ensemble Weight:** $\alpha^* = 0.60$ (Two-Tower), $1 - \alpha^* = 0.40$ (LambdaMART)
  * **nDCG@10:** 0.2768
  * **Offline Pipeline Latency (P95, CPU)*:** ~69.00 ms
* **Maximal Marginal Relevance (MMR, $\lambda^* = 0.75$)**
  * **Diversity Parameter:** $\lambda^* = 0.75$
  * **nDCG@10:** 0.2768
  * **Intra-List Diversity (ILD):** 0.77 $\rightarrow$ 0.90 (+16.9% diversity lift while preserving the ensemble's nDCG@10)

*\*Latency scope: Offline recommendation pipeline latency measured on commodity CPU (includes neural forward pass, FAISS candidate inner product, tabular feature construction, tree ranking, and MMR diversification; excludes network transport and external database round-trips).*

### Scientific Finding: Validation / Test Generalization Gap

The validation-selected ensemble ($\alpha^* = 0.60$, which achieved validation nDCG@10 = 0.3372) did not outperform the standalone Two-Tower model on the temporally isolated test set (test nDCG@10 = 0.2768 vs. Two-Tower test nDCG@10 = 0.2877). 

This gap may reflect chronological distribution shift and degradation of historical popularity/CTR priors:
- **Chronological distribution shift:** The semantic Two-Tower representations appear to generalize better to unseen breaking news articles than tabular decision trees.
- **Non-stationary CTR priors:** Historical click signals and popularity counters decay across emerging news cohorts.
- **Limited generalization of validation-selected ensemble weight:** Blending weights optimal on validation data do not guarantee superior accuracy on future time horizons.

Rather than retuning $\alpha$ on the test split, preserving this gap provides an honest, leak-free evaluation.

> For the comprehensive 6-variant ablation breakdown, mathematical formulations of all 8 ranking features, and full metric tables (PR-AUC, ROC-AUC, Cohen's $\kappa$), see [ml-service/README.md](ml-service/README.md). For end-to-end execution commands, see [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

---

## 📐 Dataset Split Methodology & Training Specs

* **Dataset Split**: Full population: 156,966 sessions (80% Train: 125.6k, 10% Val: 15.7k, 10% Test: 15.7k). Rapid reproducible experiment slices use chronologically preserved subsets (3,000 train / 500 val / 500 test) enforcing strict temporal boundary separation ($\max(T_{\text{train}}) \le \min(T_{\text{val}}) \le \max(T_{\text{val}}) \le \min(T_{\text{test}})$).
* **Leakage Prevention**: Strict temporal splitting by timestamp (`time`). User history is strictly pre-impression clicks with zero future session leakage. Historical CTR priors and catalog first-seen timestamps are derived strictly from the train split. Optimal blending weight $\alpha^* = 0.60$ is tuned on the validation split and frozen before test evaluation.
* **50/50 Negative Curriculum**: 2 in-session unclicked items (hard negatives) + 2 exploratory catalog items (soft negatives) per positive click, preventing false-negative gradient penalty.
* **Training Hyperparameters**: AdamW (LR: $10^{-4}$ with Linear Warmup + Cosine Decay to $10^{-6}$, weight decay $10^{-4}$), $\tau = 0.07$ InfoNCE temperature, $d_{\text{model}} = 256$, heads $H = 4$, batch size 64.
* **Deterministic Reproducibility**: See [REPRODUCIBILITY.md](file:///c:/Users/khush/OneDrive/Desktop/Jagrit/content-ranking-engine/REPRODUCIBILITY.md) for full commands:
  ```bash
  cd ml-service
  python train.py --config configs/mind.yaml
  python evaluate.py --config configs/mind.yaml --split validation --checkpoint checkpoints/two_tower_best.pt --tune
  python evaluate.py --config configs/mind.yaml --split test --checkpoint checkpoints/two_tower_best.pt --frozen-config results/best_validation_config.json
  python experiments/run_ablation.py --config configs/mind.yaml
  python scripts/sync_benchmarks_to_readme.py
  ```

## ⚡ Quick Start (Local Development)

### Prerequisites
* Node.js (v18+)
* Python 3.9+
* Redis Server (running locally on port 6379)
* MongoDB (Local or Atlas URI)

### 1. Setup Backend
```bash
cd backend
npm install
# Create a .env file with MONGODB_URI, JWT_SECRET, GEMINI_API_KEY, NEWS_API_KEY, REDIS_URL
npm start
```

### 2. Setup ML Service
```bash
cd ml-service
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
# Ensure GEMINI_API_KEY is exported or present in a .env file for the RAG feature
uvicorn app.main:app --reload --port 8000
```

### 3. Setup Frontend
```bash
cd web
npm install
# Create a .env file with VITE_API_URL=http://localhost:5000
npm run dev
```

---

## 🌐 Production Deployment Guide

Since Jagrit utilizes a microservices architecture, you can deploy the frontend and backend on serverless platforms (like Vercel and Render), while the ML service needs a dedicated environment.

### 1. Product Backend (Node.js)
* **Provider**: Render (Web Service)
* **Settings**:
  * Root Directory: `backend`
  * Build Command: `npm install`
  * Start Command: `npm start`
  * Env Vars: `MONGODB_URI`, `JWT_SECRET`, `GEMINI_API_KEY`, `NEWS_API_KEY`, `REDIS_URL` (From Aiven/Upstash), and `ML_SERVICE_URL`.

### 2. Web Frontend (React)
* **Provider**: Vercel
* **Settings**: 
  * Root Directory: `web`
  * Framework Preset: Vite
  * Env Vars: `VITE_API_URL` pointing to your deployed Render backend.

### 3. ML Service (Python FastAPI)
* **Provider**: AWS EC2, DigitalOcean, or Render Docker Service.
* Ensure you connect it to the same remote `REDIS_URL` used by the Node.js backend so the ML engine can access the live news ingested by the backend.
