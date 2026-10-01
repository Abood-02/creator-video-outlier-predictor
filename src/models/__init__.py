"""
Machine Learning models module.
Includes baseline LightGBM pipeline and PyTorch Multimodal Fusion Network.
"""

from src.models.baseline_lgbm import LightGBMClassifier
from src.models.fusion_nn import MultimodalFusionNet, FusionNetTrainer

__all__ = [
    "LightGBMClassifier",
    "MultimodalFusionNet",
    "FusionNetTrainer",
]
