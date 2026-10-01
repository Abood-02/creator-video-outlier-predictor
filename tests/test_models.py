"""
Unit tests for Machine Learning Models (LightGBM and PyTorch FusionNet).
"""

import numpy as np
import pytest
import torch

from src.models import LightGBMClassifier, MultimodalFusionNet


def test_lightgbm_classifier():
    """Verify LightGBM training, CV evaluation, and prediction."""
    n_samples = 100
    n_features = 10
    
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    # Create somewhat separable dummy data
    y = (X[:, 0] > 0).astype(np.int32)
    
    config = {
        "objective": "binary",
        "metric": "auc",
        "n_estimators": 10,
        "n_splits": 2,
        "early_stopping_rounds": 5,
        "scale_pos_weight": "auto",
        "verbose": -1,
    }
    
    model = LightGBMClassifier(config)
    metrics = model.fit_cv(X, y)
    
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "f1" in metrics
    assert "log_loss" in metrics
    
    assert model.is_fitted
    assert len(model.models) == 2  # 2 splits
    
    preds = model.predict(X)
    assert preds.shape == (n_samples,)
    assert np.all((preds >= 0.0) & (preds <= 1.0))


def test_fusion_nn_forward_pass():
    """Verify MultimodalFusionNet forward pass and tensor shapes."""
    batch_size = 4
    tab_dim = 16
    text_dim = 384
    vis_dim = 512
    
    tab = torch.randn(batch_size, tab_dim)
    txt = torch.randn(batch_size, text_dim)
    vis = torch.randn(batch_size, vis_dim)
    
    model = MultimodalFusionNet(
        tabular_dim=tab_dim,
        text_dim=text_dim,
        vision_dim=vis_dim,
        tabular_hidden_dim=32,
        text_hidden_dim=64,
        vision_hidden_dim=64,
        fusion_hidden_dims=[64, 32],
    )
    
    logits, reg_preds = model(tab, txt, vis)
    
    assert logits.shape == (batch_size, 1)
    assert reg_preds.shape == (batch_size, 1)
    
    # Check that gradients can flow back
    loss = logits.sum() + reg_preds.sum()
    loss.backward()
    
    # Check that weights have gradients
    assert model.cls_head.weight.grad is not None
    assert model.tab_proj.linear.weight.grad is not None
