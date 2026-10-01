"""
FastAPI Application for Video Outlier Prediction.
Handles model serving, inference, and SHAP explainability.
"""

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import HealthResponse, VideoPredictionRequest, VideoPredictionResponse, FeatureContribution
from src.config import load_config
from src.data.preprocessor import VideoPreprocessor
from src.features.fusion import MultimodalFeatureAligner
from src.features.text_embedder import TextEmbedder
from src.features.vision_embedder import VisionEmbedder
from src.models.baseline_lgbm import LightGBMClassifier
from src.models.fusion_nn import MultimodalFusionNet
from src.explainability.shap_explainer import ModelExplainer

logger = logging.getLogger(__name__)

# Global singletons for loaded artifacts
ARTIFACTS: Dict[str, Any] = {}
START_TIME: float = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts on application startup."""
    global ARTIFACTS
    logger.info("Initializing ML models and extractors...")
    cfg = load_config()
    
    device = torch.device(cfg.project.get_resolved_device())
    artifact_path = Path(cfg.tracking.artifact_path)
    
    if not artifact_path.exists():
        logger.warning(f"Artifact path {artifact_path} does not exist. Run training first.")
        yield
        return

    try:
        # Load Preprocessor and Aligner
        ARTIFACTS["preprocessor"] = VideoPreprocessor.load(artifact_path / "preprocessor.joblib")
        ARTIFACTS["aligner"] = MultimodalFeatureAligner.load(artifact_path / "aligner.joblib")
        
        # Load Embedders
        ARTIFACTS["text_embedder"] = TextEmbedder(
            model_name=cfg.features.text_embedder.model_name,
            device=cfg.project.get_resolved_device(),
            offline_fallback=True
        )
        ARTIFACTS["vision_embedder"] = VisionEmbedder(
            model_name=cfg.features.vision_embedder.model_name,
            device=cfg.project.get_resolved_device(),
            offline_fallback=True
        )
        
        # Load LightGBM
        lgbm_path = artifact_path / "lgbm_model.joblib"
        if lgbm_path.exists():
            lgbm = LightGBMClassifier.load(lgbm_path)
            ARTIFACTS["lgbm"] = lgbm
            
            # Initialize Explainer
            # To get feature names, we construct them from preprocessor
            feature_names = ARTIFACTS["preprocessor"].get_feature_names_out()
            if ARTIFACTS["aligner"].apply_pca:
                text_dim = ARTIFACTS["aligner"].text_pca_dim
                vis_dim = ARTIFACTS["aligner"].vision_pca_dim
            else:
                text_dim = cfg.features.text_embedder.embedding_dim
                vis_dim = cfg.features.vision_embedder.embedding_dim
                
            text_names = [f"text_pca_{i}" for i in range(text_dim)]
            vis_names = [f"vis_pca_{i}" for i in range(vis_dim)]
            all_feature_names = feature_names + text_names + vis_names
            
            ARTIFACTS["explainer"] = ModelExplainer(model=lgbm, feature_names=all_feature_names)
            
        # Load FusionNet
        nn_path = artifact_path / "fusion_nn.pt"
        if nn_path.exists():
            # Get dimensions
            tab_dim = len(ARTIFACTS["preprocessor"].get_feature_names_out())
            nn_model = MultimodalFusionNet(
                tabular_dim=tab_dim,
                text_dim=cfg.features.text_embedder.embedding_dim,
                vision_dim=cfg.features.vision_embedder.embedding_dim,
                tabular_hidden_dim=cfg.model_fusion_nn.tabular_hidden_dim,
                text_hidden_dim=cfg.model_fusion_nn.text_hidden_dim,
                vision_hidden_dim=cfg.model_fusion_nn.vision_hidden_dim,
                fusion_hidden_dims=cfg.model_fusion_nn.fusion_hidden_dims,
                dropout=0.0  # Eval mode
            )
            nn_model.load_state_dict(torch.load(nn_path, map_location=device, weights_only=True))
            nn_model.to(device)
            nn_model.eval()
            ARTIFACTS["fusion_nn"] = nn_model
            ARTIFACTS["device"] = device
            
        logger.info("All artifacts loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading artifacts: {e}")
        
    yield
    # Cleanup on shutdown
    ARTIFACTS.clear()


app = FastAPI(
    title="Creator Video Outlier Predictor API",
    description="Real-time ML serving for multimodal video performance prediction.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Liveness and readiness probe."""
    models_loaded = []
    if "lgbm" in ARTIFACTS:
        models_loaded.append("LightGBM")
    if "fusion_nn" in ARTIFACTS:
        models_loaded.append("FusionNet")
        
    return HealthResponse(
        status="OK" if models_loaded else "DEGRADED (Models missing)",
        models_loaded=models_loaded,
        uptime_seconds=time.time() - START_TIME
    )


@app.post("/predict", response_model=VideoPredictionResponse)
async def predict_outlier(request: VideoPredictionRequest):
    """Predict whether a candidate video will be an outlier."""
    start_t = time.time()
    
    if request.model_type not in ARTIFACTS:
        raise HTTPException(status_code=400, detail=f"Model '{request.model_type}' is not loaded.")
        
    # 1. Prepare Dataframe for tabular preprocessor
    df = pd.DataFrame([{
        "video_title": request.title,
        "published_at": request.published_at,
        "duration_seconds": request.duration_seconds,
        "tags_count": request.tags_count,
        "channel_median_views": request.channel_median_views,
    }])
    
    try:
        X_tab = ARTIFACTS["preprocessor"].transform(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Tabular preprocessing failed: {e}")
        
    # 2. Extract Embeddings
    txt_emb = ARTIFACTS["text_embedder"].encode([request.title])
    
    vis_path = request.thumbnail_url_or_path
    vis_emb = ARTIFACTS["vision_embedder"].encode([vis_path])
    
    # 3. Inference
    prob = 0.0
    view_ratio = 0.0
    top_drivers = []
    
    if request.model_type == "lgbm":
        X_fused = ARTIFACTS["aligner"].transform(X_tab, txt_emb, vis_emb)
        preds = ARTIFACTS["lgbm"].predict(X_fused)
        prob = float(preds[0])
        # We don't have a LightGBM regressor in baseline, fallback to log(1 + 2) ~ 1.0 if prob is high
        view_ratio = float(np.exp(prob * 2.0)) - 1.0  # Pseudo-heuristic for API completeness
        
    elif request.model_type == "fusion_nn":
        device = ARTIFACTS["device"]
        model = ARTIFACTS["fusion_nn"]
        
        tab_t = torch.tensor(X_tab, dtype=torch.float32).to(device)
        txt_t = torch.tensor(txt_emb, dtype=torch.float32).to(device)
        vis_t = torch.tensor(vis_emb, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            logits, reg_preds = model(tab_t, txt_t, vis_t)
            prob = float(torch.sigmoid(logits)[0, 0].item())
            # reg_preds is log_view_ratio
            log_vr = reg_preds[0, 0].item()
            view_ratio = float(np.exp(log_vr) - 1.0)
            
    # Risk Tier assignment
    if prob >= 0.75:
        risk_tier = "High"
    elif prob >= 0.4:
        risk_tier = "Medium"
    else:
        risk_tier = "Low"
        
    latency = (time.time() - start_t) * 1000.0
    
    return VideoPredictionResponse(
        outlier_probability=prob,
        is_outlier_prediction=bool(prob >= 0.5),
        predicted_view_ratio=max(view_ratio, 0.0),
        risk_tier=risk_tier,
        top_drivers=top_drivers,
        latency_ms=latency
    )


@app.post("/explain", response_model=VideoPredictionResponse)
async def explain_outlier(request: VideoPredictionRequest):
    """Predict and return SHAP explanations for the LightGBM model."""
    if "lgbm" not in ARTIFACTS or "explainer" not in ARTIFACTS:
        raise HTTPException(status_code=400, detail="LightGBM model or Explainer is not loaded.")
        
    # Must use LGBM for SHAP
    request.model_type = "lgbm"
    
    # 1. Process
    start_t = time.time()
    df = pd.DataFrame([{
        "video_title": request.title,
        "published_at": request.published_at,
        "duration_seconds": request.duration_seconds,
        "tags_count": request.tags_count,
        "channel_median_views": request.channel_median_views,
    }])
    X_tab = ARTIFACTS["preprocessor"].transform(df)
    txt_emb = ARTIFACTS["text_embedder"].encode([request.title])
    vis_emb = ARTIFACTS["vision_embedder"].encode([request.thumbnail_url_or_path])
    X_fused = ARTIFACTS["aligner"].transform(X_tab, txt_emb, vis_emb)
    
    # 2. Predict
    prob = float(ARTIFACTS["lgbm"].predict(X_fused)[0])
    view_ratio = float(np.exp(prob * 2.0)) - 1.0
    
    # 3. Explain
    try:
        contributions = ARTIFACTS["explainer"].explain_single(features=X_fused[0])
        # Only take top 5 drivers to avoid bloating payload
        top_drivers = [FeatureContribution(**c) for c in contributions[:5]]
    except Exception as e:
        logger.error(f"SHAP Explainer error: {e}")
        top_drivers = []
        
    if prob >= 0.75:
        risk_tier = "High"
    elif prob >= 0.4:
        risk_tier = "Medium"
    else:
        risk_tier = "Low"
        
    latency = (time.time() - start_t) * 1000.0
    
    return VideoPredictionResponse(
        outlier_probability=prob,
        is_outlier_prediction=bool(prob >= 0.5),
        predicted_view_ratio=max(view_ratio, 0.0),
        risk_tier=risk_tier,
        top_drivers=top_drivers,
        latency_ms=latency
    )


@app.get("/metadata")
async def get_metadata():
    """Retrieve model metadata."""
    meta = {"preprocessor_features": []}
    if "preprocessor" in ARTIFACTS:
        meta["preprocessor_features"] = ARTIFACTS["preprocessor"].get_feature_names_out()
    if "explainer" in ARTIFACTS:
        meta["global_importance"] = ARTIFACTS["explainer"].global_importance()
        
    return meta
