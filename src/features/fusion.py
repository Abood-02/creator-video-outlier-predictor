"""
Multimodal feature alignment and fusion.
Combines tabular, text, and visual representations into a single unified matrix.
Includes optional PCA dimensionality reduction for tree-based models.
"""

from typing import Optional, Tuple

import joblib
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA


class MultimodalFeatureAligner:
    """Aligns and concatenates heterogeneous feature modalities."""

    def __init__(
        self,
        apply_pca: bool = True,
        text_pca_dim: int = 32,
        vision_pca_dim: int = 32,
        random_state: int = 42,
    ) -> None:
        """Initialize the feature aligner.

        Args:
            apply_pca: Whether to apply PCA to dense embeddings (useful for LightGBM).
            text_pca_dim: Target dimension for text embeddings after PCA.
            vision_pca_dim: Target dimension for vision embeddings after PCA.
            random_state: Seed for PCA.
        """
        self.apply_pca = apply_pca
        self.text_pca_dim = text_pca_dim
        self.vision_pca_dim = vision_pca_dim
        self.random_state = random_state

        self.text_pca: Optional[PCA] = None
        self.vision_pca: Optional[PCA] = None
        self.is_fitted = False

    def fit(self, text_emb: np.ndarray, vision_emb: np.ndarray) -> "MultimodalFeatureAligner":
        """Fit the PCA transformers on the embedding matrices.

        Args:
            text_emb: Text embeddings array (N, D_text).
            vision_emb: Vision embeddings array (N, D_vision).
            
        Returns:
            Fitted instance.
        """
        if self.apply_pca:
            # Fit text PCA if dim is larger than target
            if text_emb.shape[1] > self.text_pca_dim:
                self.text_pca = PCA(n_components=self.text_pca_dim, random_state=self.random_state)
                self.text_pca.fit(text_emb)
            else:
                self.text_pca = None

            # Fit vision PCA if dim is larger than target
            if vision_emb.shape[1] > self.vision_pca_dim:
                self.vision_pca = PCA(n_components=self.vision_pca_dim, random_state=self.random_state)
                self.vision_pca.fit(vision_emb)
            else:
                self.vision_pca = None

        self.is_fitted = True
        return self

    def transform(
        self,
        tabular_features: np.ndarray,
        text_emb: np.ndarray,
        vision_emb: np.ndarray,
    ) -> np.ndarray:
        """Transform and concatenate modalities.

        Args:
            tabular_features: Standardized tabular features (N, D_tab).
            text_emb: Text embeddings (N, D_text).
            vision_emb: Vision embeddings (N, D_vision).

        Returns:
            Concatenated feature matrix of shape (N, D_fused).
        """
        if not self.is_fitted and self.apply_pca:
            raise RuntimeError("MultimodalFeatureAligner must be fitted before transform().")

        n_samples = tabular_features.shape[0]
        if text_emb.shape[0] != n_samples or vision_emb.shape[0] != n_samples:
            raise ValueError("All modalities must have the same number of samples.")

        t_emb = text_emb
        v_emb = vision_emb

        if self.apply_pca:
            if self.text_pca is not None:
                t_emb = self.text_pca.transform(text_emb)
            if self.vision_pca is not None:
                v_emb = self.vision_pca.transform(vision_emb)

        # Concatenate horizontally
        fused = np.hstack([tabular_features, t_emb, v_emb]).astype(np.float32)
        return fused

    def fit_transform(
        self,
        tabular_features: np.ndarray,
        text_emb: np.ndarray,
        vision_emb: np.ndarray,
    ) -> np.ndarray:
        """Fit PCA and return the fused feature matrix."""
        self.fit(text_emb, vision_emb)
        return self.transform(tabular_features, text_emb, vision_emb)

    def save(self, file_path: str | Path) -> Path:
        """Serialize fitted aligner to disk."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, file_path: str | Path) -> "MultimodalFeatureAligner":
        """Load fitted aligner from disk."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Feature aligner artifact not found at {path.resolve()}")
        aligner = joblib.load(path)
        if not isinstance(aligner, cls):
            raise TypeError(f"Loaded object is of type {type(aligner)}, expected {cls}.")
        return aligner
