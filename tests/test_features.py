"""
Unit tests for Multimodal Features Extractors.
"""

import numpy as np
import pytest

from src.features import MultimodalFeatureAligner, TextEmbedder, VisionEmbedder


def test_text_embedder():
    """Verify TextEmbedder returns correctly shaped, normalized arrays."""
    embedder = TextEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu", offline_fallback=True)
    texts = ["Hello world", "Another test title", None, ""]
    
    embeddings = embedder.encode(texts)
    
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (4, 384)
    assert embeddings.dtype == np.float32
    
    # Check L2 normalization (norm should be ~1.0 for each row)
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, np.ones_like(norms), rtol=1e-4)


def test_vision_embedder_fallback():
    """Verify VisionEmbedder fallback handles missing images correctly."""
    embedder = VisionEmbedder(model_name="openai/clip-vit-base-patch32", device="cpu", offline_fallback=True)
    
    # Using non-existent paths should trigger the zero-padding fallback (or random fallback)
    paths = ["missing_1.jpg", "missing_2.jpg", None]
    
    embeddings = embedder.encode(paths)
    
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (3, 512)
    assert embeddings.dtype == np.float32
    
    # In the current implementation, if the model fails to load, it returns zeros
    # If the model loads but images are missing, it might return zeros for those items.
    assert np.all(embeddings == 0.0)


def test_multimodal_feature_aligner():
    """Verify MultimodalFeatureAligner concatenation and PCA reduction."""
    n_samples = 10
    d_tab = 5
    d_text = 20
    d_vis = 30
    
    pca_text_dim = 4
    pca_vis_dim = 6
    
    tab = np.random.randn(n_samples, d_tab).astype(np.float32)
    txt = np.random.randn(n_samples, d_text).astype(np.float32)
    vis = np.random.randn(n_samples, d_vis).astype(np.float32)
    
    aligner = MultimodalFeatureAligner(
        apply_pca=True,
        text_pca_dim=pca_text_dim,
        vision_pca_dim=pca_vis_dim,
    )
    
    fused = aligner.fit_transform(tab, txt, vis)
    
    expected_dim = d_tab + pca_text_dim + pca_vis_dim
    assert fused.shape == (n_samples, expected_dim)
    assert fused.dtype == np.float32
    
    # Without PCA
    aligner_no_pca = MultimodalFeatureAligner(apply_pca=False)
    fused_no_pca = aligner_no_pca.fit_transform(tab, txt, vis)
    
    expected_dim_no_pca = d_tab + d_text + d_vis
    assert fused_no_pca.shape == (n_samples, expected_dim_no_pca)
