"""
Multimodal feature extraction and alignment module.
Provides text embedders (SentenceTransformers), vision embedders (CLIP/ViT),
and feature fusion layers (PCA reduction, concatenation) for downstream models.
"""

from src.features.fusion import MultimodalFeatureAligner
from src.features.text_embedder import TextEmbedder
from src.features.vision_embedder import VisionEmbedder

__all__ = [
    "TextEmbedder",
    "VisionEmbedder",
    "MultimodalFeatureAligner",
]
