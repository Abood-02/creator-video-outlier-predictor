"""
Baseline LightGBM classifier for creator video outlier prediction.
Includes cross-validation, hyperparameter management, and model serialization.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


class LightGBMClassifier:
    """Production wrapper for LightGBM model training and evaluation."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the model with configuration parameters.

        Args:
            config: Dictionary containing LightGBM hyperparameters and settings.
        """
        self.config = config.copy()
        self.n_splits = self.config.pop("n_splits", 5)
        self.random_state = self.config.get("random_state", 42)
        self.early_stopping_rounds = self.config.pop("early_stopping_rounds", 35)
        self.scale_pos_weight_config = self.config.pop("scale_pos_weight", "auto")
        
        # Will store the models from each fold
        self.models: List[lgb.Booster] = []
        self.is_fitted = False

    def fit_cv(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Train LightGBM using Stratified K-Fold cross-validation.

        Args:
            X: Feature matrix of shape (N, D).
            y: Binary target array of shape (N,).

        Returns:
            Dictionary of averaged out-of-fold metrics (PR-AUC, ROC-AUC, F1, Log-Loss).
        """
        logger.info(f"Starting Stratified {self.n_splits}-Fold CV for LightGBM...")
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        # Calculate scale_pos_weight if 'auto'
        if self.scale_pos_weight_config == "auto":
            n_pos = np.sum(y == 1)
            n_neg = np.sum(y == 0)
            scale_pos_weight = float(n_neg / max(n_pos, 1))
            self.config["scale_pos_weight"] = scale_pos_weight
            logger.info(f"Auto-computed scale_pos_weight: {scale_pos_weight:.3f}")
        else:
            self.config["scale_pos_weight"] = float(self.scale_pos_weight_config)

        oof_preds = np.zeros(len(y), dtype=np.float32)
        metrics_list = {"roc_auc": [], "pr_auc": [], "f1": [], "log_loss": []}

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            callbacks = [lgb.early_stopping(stopping_rounds=self.early_stopping_rounds, verbose=False)]

            model = lgb.train(
                params=self.config,
                train_set=train_data,
                valid_sets=[train_data, val_data],
                callbacks=callbacks,
            )

            self.models.append(model)
            
            # Out-of-fold predictions
            preds = model.predict(X_val, num_iteration=model.best_iteration)
            oof_preds[val_idx] = preds

            # Fold metrics
            fold_roc_auc = roc_auc_score(y_val, preds)
            fold_pr_auc = average_precision_score(y_val, preds)
            fold_f1 = f1_score(y_val, (preds >= 0.5).astype(int))
            fold_loss = log_loss(y_val, preds)

            metrics_list["roc_auc"].append(fold_roc_auc)
            metrics_list["pr_auc"].append(fold_pr_auc)
            metrics_list["f1"].append(fold_f1)
            metrics_list["log_loss"].append(fold_loss)

            logger.info(
                f"Fold {fold+1} | ROC-AUC: {fold_roc_auc:.4f} | "
                f"PR-AUC: {fold_pr_auc:.4f} | F1: {fold_f1:.4f}"
            )

        self.is_fitted = True

        # Calculate average metrics
        avg_metrics = {k: float(np.mean(v)) for k, v in metrics_list.items()}
        logger.info(f"CV OOF Metrics: {avg_metrics}")
        return avg_metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities by averaging over all fold models.

        Args:
            X: Feature matrix of shape (N, D).

        Returns:
            Probabilities array of shape (N,).
        """
        if not self.is_fitted or not self.models:
            raise RuntimeError("Model must be fitted before calling predict().")

        preds = np.zeros(X.shape[0], dtype=np.float32)
        for model in self.models:
            preds += model.predict(X, num_iteration=model.best_iteration)
        
        return preds / len(self.models)

    def save(self, file_path: str | Path) -> Path:
        """Serialize fitted model to disk.

        Args:
            file_path: Destination path.

        Returns:
            Path where file was saved.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")
            
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"models": self.models, "config": self.config}, path)
        return path

    @classmethod
    def load(cls, file_path: str | Path) -> "LightGBMClassifier":
        """Load fitted model from disk.

        Args:
            file_path: Path to serialized model.

        Returns:
            Loaded LightGBMClassifier instance.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Model artifact not found at {path.resolve()}")
            
        artifact = joblib.load(path)
        instance = cls(config=artifact.get("config", {}))
        instance.models = artifact["models"]
        instance.is_fitted = True
        return instance
