"""API Module for FastAPI serving."""

from src.api.app import app
from src.api.schemas import VideoPredictionRequest, VideoPredictionResponse

__all__ = ["app", "VideoPredictionRequest", "VideoPredictionResponse"]
