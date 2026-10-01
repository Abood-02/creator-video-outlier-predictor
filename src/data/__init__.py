"""
Data module for Creator Video Outlier & Performance Predictor.
Includes synthetic creator data generation, tabular preprocessors, and PyTorch datasets.
"""

from src.data.dataset import (
    MultimodalVideoDataset,
    SyntheticVideoDataGenerator,
    VideoRecord,
    generate_synthetic_dataset,
    load_video_dataset,
)
from src.data.preprocessor import VideoPreprocessor

__all__ = [
    "VideoRecord",
    "SyntheticVideoDataGenerator",
    "MultimodalVideoDataset",
    "VideoPreprocessor",
    "generate_synthetic_dataset",
    "load_video_dataset",
]
