"""
Training Pipeline and Experiment Tracking.
Reads configuration, ingests multimodal data, trains LightGBM baseline
and PyTorch Fusion models, and logs all metrics and artifacts to MLflow.
"""

import logging
from pathlib import Path
from typing import Any, Dict

import mlflow
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.config import load_config
from src.data import MultimodalVideoDataset, VideoPreprocessor, generate_synthetic_dataset, load_video_dataset
from src.features import MultimodalFeatureAligner, TextEmbedder, VisionEmbedder
from src.models import FusionNetTrainer, LightGBMClassifier, MultimodalFusionNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    cfg = load_config()
    mlflow.set_tracking_uri(cfg.tracking.tracking_uri)
    mlflow.set_experiment(cfg.tracking.experiment_name)

    logger.info(f"Starting pipeline for '{cfg.project.name}' version {cfg.project.version}")
    np.random.seed(cfg.project.seed)
    torch.manual_seed(cfg.project.seed)
    
    device = torch.device(cfg.project.get_resolved_device())
    logger.info(f"Using device: {device}")

    # 1. Data Ingestion
    data_path = Path(cfg.data.raw_data_path)
    if not data_path.is_file():
        logger.info(f"Dataset not found at {data_path}. Generating synthetic data...")
        df = generate_synthetic_dataset(
            output_path=data_path,
            num_samples=cfg.data.synthetic_num_samples,
            num_channels=cfg.data.num_channels,
            seed=cfg.project.seed,
        )
    else:
        logger.info(f"Loading dataset from {data_path}...")
        df = load_video_dataset(data_path)

    logger.info(f"Loaded {len(df)} records. Outlier rate: {df['is_outlier'].mean():.2%}")

    # 2. Train / Val / Test Split
    # Split train+val vs test
    df_train_val, df_test = train_test_split(
        df,
        test_size=cfg.data.test_size,
        random_state=cfg.project.seed,
        stratify=df["is_outlier"] if cfg.data.stratify_by_target else None,
    )
    # Split train vs val
    val_ratio = cfg.data.val_size / (1.0 - cfg.data.test_size)
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=val_ratio,
        random_state=cfg.project.seed,
        stratify=df_train_val["is_outlier"] if cfg.data.stratify_by_target else None,
    )

    # 3. Tabular Preprocessing
    logger.info("Fitting tabular preprocessor...")
    preprocessor = VideoPreprocessor(
        numerical_features=cfg.features.numerical_features,
        boolean_features=cfg.features.boolean_features,
    )
    
    X_tab_train = preprocessor.fit_transform(df_train)
    X_tab_val = preprocessor.transform(df_val)
    X_tab_test = preprocessor.transform(df_test)
    
    y_cls_train = df_train["is_outlier"].to_numpy(dtype=np.float32)
    y_cls_val = df_val["is_outlier"].to_numpy(dtype=np.float32)
    y_cls_test = df_test["is_outlier"].to_numpy(dtype=np.float32)
    
    y_reg_train = df_train["log_view_ratio"].to_numpy(dtype=np.float32)
    y_reg_val = df_val["log_view_ratio"].to_numpy(dtype=np.float32)
    y_reg_test = df_test["log_view_ratio"].to_numpy(dtype=np.float32)

    # 4. Multimodal Feature Extraction
    logger.info("Initializing Embedders...")
    text_embedder = TextEmbedder(
        model_name=cfg.features.text_embedder.model_name,
        device=cfg.project.get_resolved_device(),
        embedding_dim=cfg.features.text_embedder.embedding_dim,
    )
    vision_embedder = VisionEmbedder(
        model_name=cfg.features.vision_embedder.model_name,
        device=cfg.project.get_resolved_device(),
        embedding_dim=cfg.features.vision_embedder.embedding_dim,
    )

    logger.info("Extracting text embeddings...")
    txt_train = text_embedder.encode(df_train["video_title"].tolist(), batch_size=cfg.features.text_embedder.batch_size)
    txt_val = text_embedder.encode(df_val["video_title"].tolist(), batch_size=cfg.features.text_embedder.batch_size)
    txt_test = text_embedder.encode(df_test["video_title"].tolist(), batch_size=cfg.features.text_embedder.batch_size)

    logger.info("Extracting vision embeddings...")
    vis_train = vision_embedder.encode(df_train["thumbnail_path"].tolist(), batch_size=cfg.features.vision_embedder.batch_size)
    vis_val = vision_embedder.encode(df_val["thumbnail_path"].tolist(), batch_size=cfg.features.vision_embedder.batch_size)
    vis_test = vision_embedder.encode(df_test["thumbnail_path"].tolist(), batch_size=cfg.features.vision_embedder.batch_size)

    # 5. Fusion & Alignment for LightGBM
    logger.info("Fitting MultimodalFeatureAligner (PCA)...")
    aligner = MultimodalFeatureAligner(
        apply_pca=True,
        text_pca_dim=cfg.features.text_embedder.pca_dim,
        vision_pca_dim=cfg.features.vision_embedder.pca_dim,
        random_state=cfg.project.seed,
    )
    
    X_fused_train = aligner.fit_transform(X_tab_train, txt_train, vis_train)
    X_fused_val = aligner.transform(X_tab_val, txt_val, vis_val)
    X_fused_test = aligner.transform(X_tab_test, txt_test, vis_test)

    # Save artifacts
    artifacts_dir = Path(cfg.tracking.artifact_path)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    preprocessor_path = preprocessor.save(artifacts_dir / "preprocessor.joblib")
    aligner_path = aligner.save(artifacts_dir / "aligner.joblib")

    # MLFLOW RUN
    with mlflow.start_run(run_name="LGBM_and_FusionNN") as run:
        mlflow.log_params(cfg.model_dump())
        mlflow.log_artifact(str(preprocessor_path))
        mlflow.log_artifact(str(aligner_path))

        # 6. Train LightGBM Baseline (on train+val for CV, but we use train split for demo speed)
        logger.info("=== Training LightGBM Baseline ===")
        lgbm_config = cfg.model_lgbm.model_dump()
        lgbm = LightGBMClassifier(config=lgbm_config)
        
        # Combine train+val for robust CV
        X_cv = np.vstack([X_fused_train, X_fused_val])
        y_cv = np.concatenate([y_cls_train, y_cls_val])
        
        lgbm_metrics = lgbm.fit_cv(X_cv, y_cv)
        
        # Log CV metrics
        for k, v in lgbm_metrics.items():
            mlflow.log_metric(f"lgbm_cv_{k}", v)
            
        lgbm_path = lgbm.save(artifacts_dir / "lgbm_model.joblib")
        mlflow.log_artifact(str(lgbm_path))

        # 7. Train PyTorch FusionNet
        logger.info("=== Training PyTorch MultimodalFusionNet ===")
        
        train_dataset = MultimodalVideoDataset(
            X_tab_train, df_train["video_title"].tolist(), df_train["thumbnail_path"].tolist(),
            y_cls_train, y_reg_train, precomputed_text_embeddings=txt_train, precomputed_vision_embeddings=vis_train
        )
        val_dataset = MultimodalVideoDataset(
            X_tab_val, df_val["video_title"].tolist(), df_val["thumbnail_path"].tolist(),
            y_cls_val, y_reg_val, precomputed_text_embeddings=txt_val, precomputed_vision_embeddings=vis_val
        )
        
        train_loader = DataLoader(train_dataset, batch_size=cfg.model_fusion_nn.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=cfg.model_fusion_nn.batch_size, shuffle=False)
        
        fusion_model = MultimodalFusionNet(
            tabular_dim=X_tab_train.shape[1],
            text_dim=cfg.features.text_embedder.embedding_dim,
            vision_dim=cfg.features.vision_embedder.embedding_dim,
            tabular_hidden_dim=cfg.model_fusion_nn.tabular_hidden_dim,
            text_hidden_dim=cfg.model_fusion_nn.text_hidden_dim,
            vision_hidden_dim=cfg.model_fusion_nn.vision_hidden_dim,
            fusion_hidden_dims=cfg.model_fusion_nn.fusion_hidden_dims,
            dropout=cfg.model_fusion_nn.dropout,
        )
        
        trainer = FusionNetTrainer(
            model=fusion_model,
            device=device,
            learning_rate=cfg.model_fusion_nn.learning_rate,
            weight_decay=cfg.model_fusion_nn.weight_decay,
            loss_weights=cfg.model_fusion_nn.loss_weights,
        )
        
        nn_metrics = trainer.train_with_early_stopping(
            train_loader, val_loader, 
            epochs=cfg.model_fusion_nn.epochs, 
            patience=cfg.model_fusion_nn.early_stopping_patience
        )
        
        # Log NN metrics
        for k, v in nn_metrics.items():
            mlflow.log_metric(f"nn_val_{k}", v)
            
        torch.save(fusion_model.state_dict(), artifacts_dir / "fusion_nn.pt")
        mlflow.log_artifact(str(artifacts_dir / "fusion_nn.pt"))
        
        logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
