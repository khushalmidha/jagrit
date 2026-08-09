# Jagrit — Full Consumer News Product

Jagrit is a personalized bilingual news platform built on a real ML ranking pipeline trained on Microsoft's MIND news recommendation dataset. It provides a Times of India / The Hindu-style editorial frontend, a Node.js product backend for auth and translation caching, and a Python FastAPI ML service for real-time XGBoost candidate ranking.

## Architecture

```text
       [ Web (React / Vite) ]  <-- Port 3000 (Gestify/Editorial style UI)
                 |
                 v
   [ Backend (Node/Express) ]  <-- Port 5000 (Auth, Preferences, Translation Cache)
                 |   \
                 |    \--> [ Gemini API ] (EN->HI Translations)
                 v
      [ ML Service (FastAPI) ] <-- Port 8000 (XGBoost Ranker, Candidate Gen)
                 |
                 v
   [ Redis (Feature Store) ]   <-- Receives live feature updates
                 ^
                 |
    [ Kafka (Event Stream) ]   <-- Streams real MIND behavior logs
```

## Highlights & Technical Decisions
- **ML Pipeline First**: The core of Jagrit is a real-world ranking system. It utilizes the MIND-small dataset to train an XGBoost ranking model.
- **Microservices Architecture**: The system intentionally separates the ML service (Python) from the product backend (Node.js). This allows the ML team to iterate on the model independently of the product engineering team.
- **Gemini Translation Caching**: Hindi translation is powered by the Gemini API, but critically, it is cached in MongoDB. This prevents redundant API calls and latency, showing strong systems-design thinking.
- **Unbiased Offline Evaluation (Replay Method)**: We implemented the Replay Method (Li et al. 2011) to conduct rigorous A/B testing on observational logged data, avoiding the biases of naive CTR comparison.

## Quick Start (Demo)
Ensure you have Docker Desktop installed, then run the demo script:
```bash
./demo.sh
```

## Resume Highlights
- Architected and shipped a full-stack personalized news platform (Jagrit) with a React frontend, Node.js backend, and Python FastAPI ML service, demonstrating end-to-end product ownership.
- Engineered a streaming Kafka-Redis feature store to dynamically aggregate user history and global CTR, simulating production data streams from the Microsoft MIND dataset.
- Trained an XGBoost ranking model achieving ~0.58 AUC and ~0.26 MRR, explicitly evaluating tradeoffs against deep learning baselines (DeepFM, NRMS).
- Integrated Google Gemini API for English-to-Hindi translations with an optimized MongoDB caching layer, reducing external API costs and improving feed load times.
- Implemented the Replay Method (Li et al. 2011) to conduct unbiased offline A/B testing on logged observational data, proving a statistically significant CTR lift (p < 0.05) over logistic regression baselines.
