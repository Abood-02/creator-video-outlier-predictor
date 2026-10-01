"""
Vision Embedding Module for extracting dense vector representations from images.
Uses pre-trained CLIP / ViT models with padding support for missing images.
"""

import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)


class VisionEmbedder:
    """Wrapper for CLIP visual backbone for extracting thumbnail features."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "auto",
        embedding_dim: int = 512,
        offline_fallback: bool = True,
    ) -> None:
        """Initialize the vision embedder.

        Args:
            model_name: HuggingFace model identifier.
            device: 'cuda', 'cpu', or 'auto'.
            embedding_dim: Expected output dimensionality (512 for ViT-B/32).
            offline_fallback: If True, uses random embeddings on failure.
        """
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.offline_fallback = offline_fallback

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model: Optional[CLIPModel] = None
        self.processor: Optional[CLIPProcessor] = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            # We only need the visual projection part of CLIP
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model.eval()
            logger.info(f"Loaded vision model '{self.model_name}' on {self.device}")
        except Exception as e:
            if self.offline_fallback:
                logger.warning(
                    f"Failed to load vision model '{self.model_name}': {e}. "
                    "Using zero/random embeddings (fallback mode)."
                )
                self.model = None
                self.processor = None
            else:
                raise RuntimeError(f"Could not load VisionEmbedder model: {e}") from e

    def _load_image(self, path_or_url: str) -> Optional[Image.Image]:
        """Load an image from a local path, returning None if missing/invalid."""
        try:
            if not path_or_url:
                return None
            p = Path(path_or_url)
            if p.is_file():
                img = Image.open(p).convert("RGB")
                # Force load to catch corruption early
                img.verify()
                img = Image.open(p).convert("RGB") # Reopen after verify
                return img
            return None
        except Exception:
            return None

    @torch.no_grad()
    def encode(self, image_paths: List[str], batch_size: int = 32) -> np.ndarray:
        """Extract visual embeddings for a batch of image paths.

        Missing or invalid images are represented as zero vectors.

        Args:
            image_paths: List of local paths to image files.
            batch_size: Inference batch size.

        Returns:
            Numpy array of shape (len(image_paths), embedding_dim).
        """
        n_samples = len(image_paths)
        if n_samples == 0:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        out_embeddings = np.zeros((n_samples, self.embedding_dim), dtype=np.float32)

        if self.model is None or self.processor is None:
            # Fallback mode
            logger.debug("VisionEmbedder running in fallback mode (zero embeddings).")
            return out_embeddings

        for i in range(0, n_samples, batch_size):
            batch_paths = image_paths[i : i + batch_size]
            valid_images = []
            valid_indices = []

            for local_idx, path in enumerate(batch_paths):
                img = self._load_image(path)
                if img is not None:
                    valid_images.append(img)
                    valid_indices.append(local_idx)

            if valid_images:
                inputs = self.processor(images=valid_images, return_tensors="pt")
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model.vision_model(pixel_values=inputs["pixel_values"])
                image_features = self.model.visual_projection(outputs.pooler_output)
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                
                image_features_np = image_features.cpu().numpy()

                # Map back to output array
                for img_feat, local_idx in zip(image_features_np, valid_indices):
                    out_embeddings[i + local_idx] = img_feat

        return out_embeddings
