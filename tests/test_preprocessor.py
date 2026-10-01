"""
Unit tests for VideoPreprocessor and feature engineering transformations.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessor import VideoPreprocessor


@pytest.fixture
def sample_raw_data() -> pd.DataFrame:
    """Fixture providing representative video records."""
    return pd.DataFrame([
        {
            "video_id": "vid_001",
            "channel_id": "UC_001",
            "video_title": "I Spent 50 Hours In A Maximum Security Prison! (SHOCKING)",
            "thumbnail_url": "https://example.com/1.jpg",
            "published_at": "2026-03-15T14:30:00Z",  # Sunday (is_weekend=1), hour 14
            "duration_seconds": 950.0,
            "tags_count": 18,
            "channel_median_views": 150000.0,
            "video_views": 450000.0,
            "view_ratio": 3.0,
            "is_outlier": 1,
            "log_view_ratio": np.log1p(3.0),
        },
        {
            "video_id": "vid_002",
            "channel_id": "UC_002",
            "video_title": "casual grocery haul and cooking vlog",
            "thumbnail_url": "https://example.com/2.jpg",
            "published_at": "2026-03-17T09:15:00Z",  # Tuesday (is_weekend=0), hour 9
            "duration_seconds": 420.0,
            "tags_count": 5,
            "channel_median_views": 30000.0,
            "video_views": 25000.0,
            "view_ratio": 0.833,
            "is_outlier": 0,
            "log_view_ratio": np.log1p(0.833),
        },
        {
            "video_id": "vid_003",
            "channel_id": "UC_003",
            "video_title": "Why You MUST AVOID This $10,000 Tech Mistake?",
            "thumbnail_url": "https://example.com/3.jpg",
            "published_at": "2026-03-18T18:00:00Z",  # Wednesday, hour 18
            "duration_seconds": 45.0,  # is_short=1
            "tags_count": 12,
            "channel_median_views": 500000.0,
            "video_views": 1200000.0,
            "view_ratio": 2.4,
            "is_outlier": 1,
            "log_view_ratio": np.log1p(2.4),
        },
    ])


def test_engineer_features_structure(sample_raw_data: pd.DataFrame) -> None:
    """Verify all expected engineered columns are produced with correct types."""
    preprocessor = VideoPreprocessor()
    engineered = preprocessor.engineer_features(sample_raw_data)

    expected_cols = [
        "title_char_length",
        "title_word_count",
        "title_uppercase_ratio",
        "has_question_or_exclamation",
        "viral_keywords_count",
        "published_hour",
        "published_day_of_week",
        "is_weekend",
        "sin_hour",
        "cos_hour",
        "sin_day",
        "cos_day",
        "duration_seconds",
        "is_short",
        "log_channel_median_views",
    ]
    for col in expected_cols:
        assert col in engineered.columns, f"Missing engineered column: {col}"


def test_text_heuristics_correctness(sample_raw_data: pd.DataFrame) -> None:
    """Verify specific mathematical heuristics on text titles."""
    preprocessor = VideoPreprocessor()
    df = preprocessor.engineer_features(sample_raw_data)

    # First row has exclamation and uppercase letters
    assert df.loc[0, "has_question_or_exclamation"] == 1
    assert df.loc[0, "title_uppercase_ratio"] > 0.15
    assert df.loc[0, "viral_keywords_count"] >= 1  # "SHOCKING"

    # Second row is lowercase, no punctuation
    assert df.loc[1, "has_question_or_exclamation"] == 0
    assert df.loc[1, "title_uppercase_ratio"] == 0.0
    assert df.loc[1, "viral_keywords_count"] == 0

    # Third row has question mark and short duration
    assert df.loc[2, "has_question_or_exclamation"] == 1
    assert df.loc[2, "is_short"] == 1


def test_fit_transform_and_dimensions(sample_raw_data: pd.DataFrame) -> None:
    """Check matrix shape, finite values, and DataFrame output format."""
    preprocessor = VideoPreprocessor()
    matrix = preprocessor.fit_transform(sample_raw_data)

    assert isinstance(matrix, np.ndarray)
    assert matrix.shape[0] == len(sample_raw_data)
    assert matrix.shape[1] == len(preprocessor.feature_names)
    assert np.all(np.isfinite(matrix)), "Transformed matrix contains NaN or Inf"

    df_result = preprocessor.transform(sample_raw_data, as_dataframe=True)
    assert isinstance(df_result, pd.DataFrame)
    assert list(df_result.columns) == preprocessor.feature_names


def test_transform_single(sample_raw_data: pd.DataFrame) -> None:
    """Ensure transform_single returns matching 1D vector."""
    preprocessor = VideoPreprocessor()
    preprocessor.fit(sample_raw_data)

    single_record = sample_raw_data.iloc[0].to_dict()
    vec = preprocessor.transform_single(single_record)

    assert isinstance(vec, np.ndarray)
    assert vec.ndim == 1
    assert len(vec) == len(preprocessor.feature_names)

    # Verify identical to row 0 of full transform
    full_matrix = preprocessor.transform(sample_raw_data)
    np.testing.assert_allclose(vec, full_matrix[0], rtol=1e-5)


def test_serialization_persistence(sample_raw_data: pd.DataFrame) -> None:
    """Test joblib save and load maintains exact numerical parity."""
    preprocessor = VideoPreprocessor()
    matrix_before = preprocessor.fit_transform(sample_raw_data)

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "preprocessor.joblib"
        preprocessor.save(save_path)
        assert save_path.is_file()

        loaded_preprocessor = VideoPreprocessor.load(save_path)
        matrix_after = loaded_preprocessor.transform(sample_raw_data)

        np.testing.assert_allclose(matrix_before, matrix_after, rtol=1e-6)
        assert loaded_preprocessor.feature_names == preprocessor.feature_names


def test_missing_and_edge_case_inputs() -> None:
    """Ensure defensive behavior against missing values and empty strings."""
    preprocessor = VideoPreprocessor()
    train_data = pd.DataFrame([
        {
            "video_title": "Baseline Video",
            "published_at": "2026-01-01T00:00:00Z",
            "duration_seconds": 300.0,
            "tags_count": 5,
            "channel_median_views": 10000.0,
        },
        {
            "video_title": "Second Video",
            "published_at": "2026-01-02T12:00:00Z",
            "duration_seconds": 600.0,
            "tags_count": 10,
            "channel_median_views": 20000.0,
        },
    ])
    preprocessor.fit(train_data)

    # Test corrupted/sparse record
    edge_record = {
        "video_title": "",
        "published_at": "invalid_date_string",
        "duration_seconds": None,
        "tags_count": None,
        "channel_median_views": 0,  # Zero views edge case
    }
    vec = preprocessor.transform_single(edge_record)
    assert len(vec) == len(preprocessor.feature_names)
    assert np.all(np.isfinite(vec))
