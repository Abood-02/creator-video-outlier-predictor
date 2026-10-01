"""
PyTorch Multimodal Fusion Network.
Fuses tabular, text, and vision modalities using projection MLPs and a joint
multi-task objective (BCE for classification, MSE for regression).
"""

import copy
import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class MLPBlock(nn.Module):
    """Standard Multi-Layer Perceptron block with LayerNorm and Dropout."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.norm(self.linear(x))))


class MultimodalFusionNet(nn.Module):
    """PyTorch model for multimodal outlier hit prediction."""

    def __init__(
        self,
        tabular_dim: int,
        text_dim: int,
        vision_dim: int,
        tabular_hidden_dim: int = 64,
        text_hidden_dim: int = 128,
        vision_hidden_dim: int = 128,
        fusion_hidden_dims: List[int] = [128, 64],
        dropout: float = 0.3,
    ):
        super().__init__()
        
        # Unimodal projections
        self.tab_proj = MLPBlock(tabular_dim, tabular_hidden_dim, dropout)
        self.text_proj = MLPBlock(text_dim, text_hidden_dim, dropout)
        self.vis_proj = MLPBlock(vision_dim, vision_hidden_dim, dropout)
        
        # Fusion
        fused_dim = tabular_hidden_dim + text_hidden_dim + vision_hidden_dim
        
        fusion_layers = []
        in_d = fused_dim
        for out_d in fusion_hidden_dims:
            fusion_layers.append(MLPBlock(in_d, out_d, dropout))
            in_d = out_d
            
        self.fusion_mlp = nn.Sequential(*fusion_layers)
        
        # Multi-task Heads
        self.cls_head = nn.Linear(in_d, 1)  # Binary classification (BCEWithLogitsLoss)
        self.reg_head = nn.Linear(in_d, 1)  # Auxiliary regression (MSELoss)

    def forward(
        self, tabular: torch.Tensor, text: torch.Tensor, vision: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Returns:
            Tuple of (classification_logits, regression_preds)
        """
        h_tab = self.tab_proj(tabular)
        h_text = self.text_proj(text)
        h_vis = self.vis_proj(vision)
        
        h_fused = torch.cat([h_tab, h_text, h_vis], dim=1)
        h_out = self.fusion_mlp(h_fused)
        
        logits = self.cls_head(h_out)
        reg_preds = self.reg_head(h_out)
        
        return logits, reg_preds


class FusionNetTrainer:
    """Handles training and evaluation of the MultimodalFusionNet."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        loss_weights: Dict[str, float] = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        
        self.loss_weights = loss_weights or {"bce": 1.0, "mse": 0.4}

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        
        for batch in loader:
            tab = batch["tabular"].to(self.device)
            txt = batch["text_embedding"].to(self.device)
            vis = batch["vision_embedding"].to(self.device)
            y_cls = batch["label_cls"].to(self.device)
            y_reg = batch["label_reg"].to(self.device)
            
            self.optimizer.zero_grad()
            logits, preds_reg = self.model(tab, txt, vis)
            
            loss_cls = self.bce_loss(logits, y_cls)
            loss_reg = self.mse_loss(preds_reg, y_reg)
            
            loss = (self.loss_weights["bce"] * loss_cls) + (self.loss_weights["mse"] * loss_reg)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * tab.size(0)
            
        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        
        all_logits = []
        all_y_cls = []
        
        for batch in loader:
            tab = batch["tabular"].to(self.device)
            txt = batch["text_embedding"].to(self.device)
            vis = batch["vision_embedding"].to(self.device)
            y_cls = batch["label_cls"].to(self.device)
            y_reg = batch["label_reg"].to(self.device)
            
            logits, preds_reg = self.model(tab, txt, vis)
            
            loss_cls = self.bce_loss(logits, y_cls)
            loss_reg = self.mse_loss(preds_reg, y_reg)
            loss = (self.loss_weights["bce"] * loss_cls) + (self.loss_weights["mse"] * loss_reg)
            total_loss += loss.item() * tab.size(0)
            
            all_logits.append(logits.cpu().numpy())
            all_y_cls.append(y_cls.cpu().numpy())
            
        avg_loss = total_loss / len(loader.dataset)
        
        logits_arr = np.concatenate(all_logits, axis=0)
        y_cls_arr = np.concatenate(all_y_cls, axis=0)
        probs = torch.sigmoid(torch.tensor(logits_arr)).numpy()
        
        roc_auc = roc_auc_score(y_cls_arr, probs)
        pr_auc = average_precision_score(y_cls_arr, probs)
        f1 = f1_score(y_cls_arr, (probs >= 0.5).astype(int))
        
        return {
            "val_loss": avg_loss,
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "f1": float(f1),
        }

    def train_with_early_stopping(
        self, train_loader: DataLoader, val_loader: DataLoader, epochs: int, patience: int
    ) -> Dict[str, float]:
        best_val_loss = float("inf")
        best_metrics = {}
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            metrics = self.evaluate(val_loader)
            val_loss = metrics["val_loss"]
            
            logger.info(
                f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f} - ROC-AUC: {metrics['roc_auc']:.4f} - "
                f"PR-AUC: {metrics['pr_auc']:.4f}"
            )
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics = metrics
                best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
                
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            
        return best_metrics
