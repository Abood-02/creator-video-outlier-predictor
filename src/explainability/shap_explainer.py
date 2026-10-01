"""
SHAP-based Explainability Module for Model Interpretability.
Provides local and global feature attributions for LightGBM models.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import shap

from src.models import LightGBMClassifier

logger = logging.getLogger(__name__)


class ModelExplainer:
    """Wrapper around SHAP TreeExplainer for interpreting LightGBM models."""

    def __init__(self, model: LightGBMClassifier, feature_names: List[str]) -> None:
        """Initialize the explainer.
        
        Args:
            model: Trained LightGBMClassifier instance.
            feature_names: List of feature names matching the input array dimensions.
        """
        self.model = model
        self.feature_names = feature_names
        self.explainers = []

        if not self.model.is_fitted or not self.model.models:
            raise ValueError("Model must be fitted before initializing SHAP explainer.")

        # Initialize an explainer for each fold model in the ensemble
        for booster in self.model.models:
            self.explainers.append(shap.TreeExplainer(booster))

    def explain_single(
        self, features: np.ndarray, feature_values: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """Explain a single prediction.
        
        Args:
            features: 1D array of standardized feature values (shape: D,).
            feature_values: 1D array of raw feature values (for human readability).
            
        Returns:
            List of dictionaries with feature_name, shap_value, and feature_value.
            Sorted by absolute SHAP value (highest impact first).
        """
        if features.ndim == 1:
            features_2d = features.reshape(1, -1)
        else:
            features_2d = features

        if features_2d.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Feature dimension mismatch. Expected {len(self.feature_names)}, "
                f"got {features_2d.shape[1]}"
            )

        # Average SHAP values across fold models
        fold_shap_values = []
        for explainer in self.explainers:
            sv = explainer.shap_values(features_2d)
            if isinstance(sv, list):
                # LightGBM binary classification might return a list [sv_class0, sv_class1]
                sv = sv[1]
            fold_shap_values.append(sv)
            
        avg_shap_values = np.mean(fold_shap_values, axis=0)[0]  # Shape: (D,)

        contributions = []
        for i, name in enumerate(self.feature_names):
            raw_val = feature_values[i] if feature_values is not None else float(features_2d[0, i])
            contributions.append(
                {
                    "feature_name": name,
                    "shap_value": float(avg_shap_values[i]),
                    "feature_value": raw_val,
                }
            )

        # Sort by absolute impact descending
        contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        return contributions

    def global_importance(self) -> Dict[str, float]:
        """Calculate global feature importance based on LightGBM split/gain.
        
        Since computing global SHAP over a large dataset at runtime is expensive,
        we expose the booster's native gain importance.
        """
        importance_dict = {name: 0.0 for name in self.feature_names}
        for booster in self.model.models:
            imp = booster.feature_importance(importance_type="gain")
            for i, val in enumerate(imp):
                importance_dict[self.feature_names[i]] += float(val)
                
        # Average across folds
        num_models = len(self.model.models)
        return {k: v / num_models for k, v in importance_dict.items()}
