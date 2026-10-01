"""
API Data Contracts (Pydantic v2 schemas).
"""

from typing import Any, List, Literal, Optional, Union
from pydantic import BaseModel, Field


class VideoPredictionRequest(BaseModel):
    """Payload for real-time video outlier prediction."""
    
    title: str = Field(..., description="The video title.")
    published_at: str = Field(..., description="ISO 8601 timestamp or string datetime.")
    duration_seconds: int = Field(..., ge=1, description="Video duration in seconds.")
    tags_count: int = Field(..., ge=0, description="Number of tags on the video.")
    channel_median_views: float = Field(..., gt=0, description="Median views of the channel.")
    thumbnail_url_or_path: Optional[str] = Field(None, description="Local path or URL to the thumbnail image.")
    model_type: Literal["lgbm", "fusion_nn"] = Field("lgbm", description="Which model to use for inference.")


class FeatureContribution(BaseModel):
    """Represents the SHAP contribution of a single feature."""
    
    feature_name: str = Field(..., description="Name of the feature.")
    shap_value: float = Field(..., description="Directional SHAP impact (log-odds).")
    feature_value: Any = Field(..., description="The raw or processed feature value.")


class VideoPredictionResponse(BaseModel):
    """Response containing prediction scores and risk analysis."""
    
    outlier_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of being an outlier.")
    is_outlier_prediction: bool = Field(..., description="Binary thresholded prediction (>=0.5).")
    predicted_view_ratio: float = Field(..., description="Expected View Ratio.")
    risk_tier: str = Field(..., description="Risk tier classification: 'Low' | 'Medium' | 'High'")
    top_drivers: List[FeatureContribution] = Field(default_factory=list, description="Top SHAP drivers.")
    latency_ms: float = Field(..., description="Inference latency in milliseconds.")


class HealthResponse(BaseModel):
    """Health check response schema."""
    
    status: str = Field(..., description="API operational status.")
    models_loaded: List[str] = Field(..., description="List of loaded model architectures.")
    uptime_seconds: float = Field(..., description="Server uptime in seconds.")
