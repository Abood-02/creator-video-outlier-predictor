"""
Unit tests for SyntheticVideoDataGenerator and PyTorch MultimodalVideoDataset.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.dataset import (
    MultimodalVideoDataset,
    SyntheticVideoDataGenerator,
    load_video_dataset,
)


def test_synthetic_data_generation() -> None:
    """Validate distribution characteristics of synthetic creator records."""
    generator = SyntheticVideoDataGenerator(seed=123, num_channels=12)
    df = generator.generate_dataframe(total_samples=120)

    # Check schema columns
    required_cols = [
        "video_id", "channel_id", "channel_title", "video_title",
        "thumbnail_url", "thumbnail_path", "published_at",
        "duration_seconds", "tags_count", "tags",
        "channel_median_views", "video_views", "view_ratio",
        "is_outlier", "log_view_ratio"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"

    assert len(df) >= 100
    # Check positive values
    assert (df["duration_seconds"] > 0).all()
    assert (df["video_views"] > 0).all()
    assert (df["channel_median_views"] > 0).all()

    # Target calculation integrity: R = views / median
    expected_r = df["video_views"] / df["channel_median_views"]
    np.testing.assert_allclose(df["view_ratio"], expected_r, rtol=1e-3)

    # Outlier threshold condition: is_outlier == 1 iff R >= 2.0
    expected_is_outlier = (df["view_ratio"] >= 2.0).astype(int)
    np.testing.assert_array_equal(df["is_outlier"], expected_is_outlier)

    # Log ratio: log1p(R)
    expected_log_r = np.log1p(df["view_ratio"])
    np.testing.assert_allclose(df["log_view_ratio"], expected_log_r, rtol=1e-4)


def test_synthetic_dataset_json_io() -> None:
    """Test saving and reloading synthetic dataset to/from JSON."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        json_file = Path(tmp_dir) / "test_dataset.json"
        generator = SyntheticVideoDataGenerator(seed=42, num_channels=5)
        saved_path = generator.save_to_json(json_file, total_samples=50)

        assert saved_path.is_file()
        loaded_df = load_video_dataset(saved_path)
        assert len(loaded_df) >= 40
        assert "is_outlier" in loaded_df.columns


def test_multimodal_dataset_and_dataloader() -> None:
    """Verify PyTorch dataset indexing, tensors, and batch collation."""
    n_samples = 32
    d_tab = 16
    d_text = 384
    d_vision = 512

    tabular = np.random.randn(n_samples, d_tab).astype(np.float32)
    titles = [f"Sample Video Title {i}" for i in range(n_samples)]
    thumbnails = [f"data/thumbnails/mock_{i}.jpg" for i in range(n_samples)]
    labels_cls = (np.random.rand(n_samples) > 0.7).astype(np.float32)
    labels_reg = np.log1p(np.random.exponential(1.5, size=n_samples)).astype(np.float32)
    text_emb = np.random.randn(n_samples, d_text).astype(np.float32)
    vision_emb = np.random.randn(n_samples, d_vision).astype(np.float32)

    dataset = MultimodalVideoDataset(
        tabular_features=tabular,
        titles=titles,
        thumbnail_paths=thumbnails,
        labels_cls=labels_cls,
        labels_reg=labels_reg,
        precomputed_text_embeddings=text_emb,
        precomputed_vision_embeddings=vision_emb,
    )

    assert len(dataset) == n_samples
    item = dataset[0]
    assert item["tabular"].shape == (d_tab,)
    assert item["label_cls"].shape == (1,)
    assert item["label_reg"].shape == (1,)
    assert item["text_embedding"].shape == (d_text,)
    assert item["vision_embedding"].shape == (d_vision,)

    # Test PyTorch DataLoader integration
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    batch = next(iter(loader))

    assert batch["tabular"].shape == (8, d_tab)
    assert batch["label_cls"].shape == (8, 1)
    assert batch["label_reg"].shape == (8, 1)
    assert batch["text_embedding"].shape == (8, d_text)
    assert batch["vision_embedding"].shape == (8, d_vision)
