"""
Text Embedding Module for extracting dense vector representations from text.
Uses SentenceTransformers with fallback capabilities for offline mode.
"""

import logging
from typing import List, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class TextEmbedder:
    """Wrapper for Sentence-Transformers for extracting text features."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto",
        embedding_dim: int = 384,
        offline_fallback: bool = True,
    ) -> None:
        """Initialize the text embedder.

        Args:
            model_name: Name of the pre-trained SentenceTransformer model.
            device: 'cuda', 'cpu', or 'auto'.
            embedding_dim: Expected output dimensionality.
            offline_fallback: If True, uses random embeddings when model loading fails.
        """
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.offline_fallback = offline_fallback

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model: Union[SentenceTransformer, None] = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Loaded text model '{self.model_name}' on {self.device}")
        except Exception as e:
            if self.offline_fallback:
                logger.warning(
                    f"Failed to load text model '{self.model_name}': {e}. "
                    "Using random embeddings (fallback mode)."
                )
                self.model = None
            else:
                raise RuntimeError(f"Could not load TextEmbedder model: {e}") from e

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode a list of texts into dense vectors with L2 normalization.

        Args:
            texts: List of input strings.
            batch_size: Batch size for model inference.

        Returns:
            Numpy array of shape (len(texts), embedding_dim).
        """
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        # Handle NaNs or non-string inputs robustly
        cleaned_texts = [str(t) if t is not None else "" for t in texts]

        if self.model is None:
            # Fallback mode: Generate random normalized vectors
            embeddings = np.random.randn(len(cleaned_texts), self.embedding_dim).astype(np.float32)
        else:
            embeddings = self.model.encode(
                cleaned_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,  # We'll normalize explicitly
            )

        # Ensure output is a numpy array (sentence-transformers sometimes returns a list of tensors if params differ)
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)

        # L2 Normalization
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1e-8, norms)
        embeddings = embeddings / norms

        return embeddings.astype(np.float32)
