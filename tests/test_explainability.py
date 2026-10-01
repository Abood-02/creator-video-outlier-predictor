"""
Unit tests for the SHAP Explainer module.
"""

import numpy as np
import pytest

from src.models.baseline_lgbm import LightGBMClassifier
from src.explainability.shap_explainer import ModelExplainer


def test_explainer_initialization_and_single_explain():
    """Verify ModelExplainer can explain a single prediction."""
    # Create dummy LightGBM model and train it
    X = np.random.randn(50, 5).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int32)
    
    config = {
        "objective": "binary",
        "metric": "auc",
        "n_estimators": 5,
        "n_splits": 2,
        "verbose": -1,
    }
    
    model = LightGBMClassifier(config)
    model.fit_cv(X, y)
    
    feature_names = ["feat_1", "feat_2", "feat_3", "feat_4", "feat_5"]
    
    explainer = ModelExplainer(model, feature_names)
    
    # Test single explanation
    test_features = np.random.randn(5).astype(np.float32)
    test_feature_values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    
    contributions = explainer.explain_single(test_features, test_feature_values)
    
    assert isinstance(contributions, list)
    assert len(contributions) == 5
    
    # Check structure
    assert "feature_name" in contributions[0]
    assert "shap_value" in contributions[0]
    assert "feature_value" in contributions[0]
    
    # Check sorted by absolute impact
    for i in range(len(contributions) - 1):
        assert abs(contributions[i]["shap_value"]) >= abs(contributions[i+1]["shap_value"])

def test_global_importance():
    """Verify global importance aggregation."""
    X = np.random.randn(50, 3).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int32)
    
    config = {
        "objective": "binary",
        "metric": "auc",
        "n_estimators": 5,
        "n_splits": 2,
        "verbose": -1,
    }
    
    model = LightGBMClassifier(config)
    model.fit_cv(X, y)
    
    feature_names = ["feat_1", "feat_2", "feat_3"]
    explainer = ModelExplainer(model, feature_names)
    
    global_imp = explainer.global_importance()
    
    assert isinstance(global_imp, dict)
    assert len(global_imp) == 3
    assert "feat_1" in global_imp
