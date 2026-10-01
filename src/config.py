"""
Configuration management module for Creator Video Outlier Predictor.
Leverages Pydantic v2 and pydantic-settings for robust schema validation,
type safety, and environment variable overrides.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProjectConfig(BaseModel):
    """General project metadata and reproducibility settings."""
    name: str = "video-outlier-predictor"
    version: str = "0.1.0"
    seed: int = 42
    device: Literal["cuda", "cpu", "auto"] = "auto"

    def get_resolved_device(self) -> str:
        """Resolves 'auto' to 'cuda' if GPU is available, else 'cpu'."""
        if self.device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.device


class DataConfig(BaseModel):
    """Dataset and synthetic data generation settings."""
    raw_data_path: Path = Path("data/sample_dataset.json")
    processed_dir: Path = Path("data/processed")
    synthetic_num_samples: int = Field(default=2500, ge=100)
    num_channels: int = Field(default=80, ge=5)
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    val_size: float = Field(default=0.1, gt=0.0, lt=1.0)
    stratify_by_target: bool = True
    target_outlier_ratio_threshold: float = Field(
        default=2.0,
        description="Threshold R = views / channel_median_views to label a video as hit (y=1)"
    )


class TextEmbedderConfig(BaseModel):
    """Configuration for sentence-transformers NLP backbone."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    batch_size: int = 32
    pca_dim: int = 32


class VisionEmbedderConfig(BaseModel):
    """Configuration for CLIP / ViT image backbone."""
    model_name: str = "openai/clip-vit-base-patch32"
    embedding_dim: int = 512
    batch_size: int = 32
    pca_dim: int = 32
    image_size: Tuple[int, int] = (224, 224)


class FeatureConfig(BaseModel):
    """Specification of input features and embeddings."""
    numerical_features: List[str] = Field(
        default_factory=lambda: [
            "duration_seconds",
            "tags_count",
            "title_char_length",
            "title_word_count",
            "title_uppercase_ratio",
            "published_hour",
            "published_day_of_week",
            "sin_hour",
            "cos_hour",
            "sin_day",
            "cos_day",
            "log_channel_median_views",
            "viral_keywords_count",
        ]
    )
    boolean_features: List[str] = Field(
        default_factory=lambda: [
            "has_question_or_exclamation",
            "is_weekend",
            "is_short",
        ]
    )
    text_embedder: TextEmbedderConfig = Field(default_factory=TextEmbedderConfig)
    vision_embedder: VisionEmbedderConfig = Field(default_factory=VisionEmbedderConfig)

    @property
    def all_tabular_features(self) -> List[str]:
        """Returns the full ordered list of tabular features."""
        return self.numerical_features + self.boolean_features


class LGBMModelConfig(BaseModel):
    """LightGBM hyperparameter and training settings."""
    objective: str = "binary"
    metric: str = "auc"
    boosting_type: str = "gbdt"
    learning_rate: float = 0.04
    n_estimators: int = 400
    num_leaves: int = 31
    max_depth: int = 6
    subsample: float = 0.85
    colsample_bytree: float = 0.8
    min_child_samples: int = 20
    scale_pos_weight: Union[str, float] = "auto"
    early_stopping_rounds: int = 35
    n_splits: int = 5
    random_state: int = 42


class FusionNNModelConfig(BaseModel):
    """PyTorch Multimodal Fusion Network hyperparameters."""
    tabular_hidden_dim: int = 64
    text_hidden_dim: int = 128
    vision_hidden_dim: int = 128
    fusion_hidden_dims: List[int] = Field(default_factory=lambda: [128, 64])
    dropout: float = 0.3
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    batch_size: int = 64
    epochs: int = 25
    early_stopping_patience: int = 5
    loss_weights: Dict[str, float] = Field(
        default_factory=lambda: {"bce": 1.0, "mse": 0.4}
    )


class TrackingConfig(BaseModel):
    """MLflow experiment tracking and artifact storage settings."""
    experiment_name: str = "creator-video-outlier-predictor"
    tracking_uri: str = "mlruns"
    artifact_path: str = "artifacts"
    log_models: bool = True


class APIConfig(BaseModel):
    """FastAPI serving configurations."""
    title: str = "Creator Video Outlier & Performance Predictor API"
    description: str = "Production FastAPI inference service for creator video hit prediction and SHAP explainability."
    version: str = "0.1.0"
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])
    model_type: Literal["lightgbm", "fusion_nn"] = "lightgbm"
    model_artifact_path: Path = Path("artifacts/models/lgbm_model.pkl")
    preprocessor_artifact_path: Path = Path("artifacts/preprocessor/preprocessor.joblib")
    text_embedder_cache: bool = True
    vision_embedder_cache: bool = True
    enable_shap: bool = True
    shap_background_samples: int = 100
    prediction_threshold: float = 0.5


class AppConfig(BaseSettings):
    """Master application configuration supporting YAML loading and env overrides."""
    model_config = SettingsConfigDict(
        env_prefix="CREATOR_ML_",
        env_nested_delimiter="__",
        arbitrary_types_allowed=True,
        extra="ignore",
    )

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    model_lgbm: LGBMModelConfig = Field(default_factory=LGBMModelConfig)
    model_fusion_nn: FusionNNModelConfig = Field(default_factory=FusionNNModelConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    api: APIConfig = Field(default_factory=APIConfig)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "AppConfig":
        """Load configuration from a YAML file with fallback to defaults.
        
        Args:
            yaml_path: Path to the YAML configuration file.
            
        Returns:
            AppConfig instance initialized with YAML parameters.
        """
        path = Path(yaml_path)
        if not path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")

        with open(path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}

        return cls(**raw_data)

    @classmethod
    def get_default_config(cls) -> "AppConfig":
        """Returns the default configuration, attempting to load train_config.yaml if found."""
        candidate_paths = [
            Path("configs/train_config.yaml"),
            Path(__file__).resolve().parent.parent / "configs" / "train_config.yaml",
        ]
        for p in candidate_paths:
            if p.is_file():
                return cls.from_yaml(p)
        return cls()


def load_config(config_path: Optional[Union[str, Path]] = None) -> AppConfig:
    """Helper function to load the AppConfig.
    
    Args:
        config_path: Optional path to YAML config file. If None, searches standard locations.
        
    Returns:
        Validated AppConfig object.
    """
    if config_path:
        return AppConfig.from_yaml(config_path)
    return AppConfig.get_default_config()
