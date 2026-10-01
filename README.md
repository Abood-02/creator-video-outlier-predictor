# Creator Video Outlier & Performance Predictor 🚀

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-brightgreen)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-19%20Passed-success)

An end-to-end, production-grade Machine Learning system designed to predict whether a YouTube video will become an "outlier hit" (significantly outperforming its channel's historical median views) *before* or *at* upload time. 

Built as a premier showcase project targeting a Machine Learning Engineer role at creator-economy platforms (e.g., Spotter).

## 📊 Target Formulation

The system defines an outlier based on the ratio of a video's views to the channel's historical median views:
$$ R = \frac{\text{views}}{\text{median\_views}} $$

- **Binary Classification**: $R \ge 2.0$ (Is this video a breakout hit?)
- **Continuous Target**: $\log(1 + R)$ (Auxiliary regression target for multi-task learning)

## 🏗️ System Architecture

```text
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│                 │       │                 │       │                 │
│  Video Meta     ├──────►│ Preprocessor    ├──────►│  Tabular Feats  │
│  (Title, Tags,  │       │ (StandardScaler)│       │  (D_tab)        │
│   Published At) │       │                 │       │                 │
└─────────────────┘       └─────────────────┘       └────────┬────────┘
                                                             │
┌─────────────────┐       ┌─────────────────┐       ┌────────▼────────┐
│                 │       │                 │       │                 │
│  Video Title    ├──────►│ Sentence-Trans. ├──────►│  Text Embeds    │
│  (NLP)          │       │ (all-MiniLM-L6) │       │  (D_text)       │
│                 │       │                 │       │                 │
└─────────────────┘       └─────────────────┘       └────────┬────────┘
                                                             │
┌─────────────────┐       ┌─────────────────┐       ┌────────▼────────┐     ┌────────────────┐
│                 │       │                 │       │                 │     │                │
│  Thumbnail      ├──────►│ CLIP Vision     ├──────►│  Vision Embeds  ├───► │ Model Fusion   │
│  (Visual)       │       │ (ViT-Base-32)   │       │  (D_vis)        │     │ (LGBM / PyTorch│
│                 │       │                 │       │                 │     │  FusionNet)    │
└─────────────────┘       └─────────────────┘       └─────────────────┘     │                │
                                                                            └───────┬────────┘
                                                                                    │
                                                                           ┌────────▼────────┐
                                                                           │                 │
                                                                           │  Predictions &  │
                                                                           │  SHAP Insights  │
                                                                           │                 │
                                                                           └─────────────────┘
```

## 🚀 Benchmark Results

| Model | Architecture | ROC-AUC | PR-AUC |
| :--- | :--- | :--- | :--- |
| **LightGBM** | Baseline (Tabular + PCA Embeddings) | 0.8142 | 0.7021 |
| **MultimodalFusionNet** | PyTorch (Multi-task BCE + MSE) | 0.7513 | 0.6210 |

*(Note: Benchmarks are based on synthetic high-fidelity data)*

## ⚡ 1-Click Quickstart

### Local Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the models (creates MLflow SQLite DB & Artifacts):**
   ```bash
   python -m scripts.train
   ```

3. **Start the FastAPI Server:**
   ```bash
   uvicorn src.api.app:app --reload
   ```

### Docker Compose (Production)

To spin up both the FastAPI application and the MLflow Tracking UI simultaneously:

```bash
docker-compose up --build
```
- **API**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **MLflow UI**: [http://localhost:5000](http://localhost:5000)

## 📡 API Usage Examples

### 1. Real-time Prediction (`/predict`)

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "title": "I spent $1,000,000 on a new house",
  "published_at": "2023-01-01T12:00:00Z",
  "duration_seconds": 600,
  "tags_count": 5,
  "channel_median_views": 100000.0,
  "model_type": "lgbm"
}'
```

### 2. SHAP Explainability (`/explain`)

Returns the top directional drivers for the candidate video's prediction.

```bash
curl -X 'POST' \
  'http://localhost:8000/explain' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "title": "I spent $1,000,000 on a new house",
  "published_at": "2023-01-01T12:00:00Z",
  "duration_seconds": 600,
  "tags_count": 5,
  "channel_median_views": 100000.0
}'
```

## 🧪 MLOps & Testing

This repository is backed by rigorous software engineering practices:
- **Pydantic v2 Settings**: Strongly-typed configuration system (`src/config.py`).
- **Defensive ML Design**: Offline fallbacks for HuggingFace embedders and imputing of missing images.
- **Full Test Suite**: Tested across generators, extractors, ML training, and API serving layers.

Run the test suite:
```bash
pytest tests/ -v
```
