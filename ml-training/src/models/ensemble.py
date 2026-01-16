"""
Ensemble Model
Combines multiple models for improved performance
"""

import torch
import torch.nn as nn
from typing import List


class EnsembleModel(nn.Module):
    """
    Ensemble of multiple models.
    
    Combines predictions from:
    - CNN-LSTM
    - ResNet1D
    - Transformer
    
    Using weighted averaging or voting.
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        weights: List[float] = None,
        method: str = 'weighted_average'
    ):
        """
        Args:
            models: List of trained models
            weights: Weights for each model (if None, use equal weights)
            method: 'weighted_average', 'voting', or 'stacking'
        """
        super(EnsembleModel, self).__init__()
        
        self.models = nn.ModuleList(models)
        self.method = method
        
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            assert len(weights) == len(models), "Number of weights must match number of models"
            # Normalize weights
            total = sum(weights)
            self.weights = [w / total for w in weights]
        
        # For stacking method
        if method == 'stacking':
            num_classes = models[0].num_classes if hasattr(models[0], 'num_classes') else 5
            self.meta_classifier = nn.Sequential(
                nn.Linear(len(models) * num_classes, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble.
        
        Args:
            x: Input tensor
            
        Returns:
            Ensemble predictions
        """
        # Get predictions from all models
        predictions = []
        for model in self.models:
            with torch.no_grad() if not self.training else torch.enable_grad():
                pred = model(x)
                predictions.append(pred)
        
        if self.method == 'weighted_average':
            # Weighted average of logits
            ensemble_pred = sum(w * p for w, p in zip(self.weights, predictions))
            
        elif self.method == 'voting':
            # Majority voting (hard voting)
            votes = torch.stack([torch.argmax(p, dim=1) for p in predictions], dim=1)
            ensemble_pred = torch.mode(votes, dim=1)[0]
            
        elif self.method == 'stacking':
            # Stack predictions and use meta-classifier
            stacked = torch.cat(predictions, dim=1)
            ensemble_pred = self.meta_classifier(stacked)
        
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")
        
        return ensemble_pred
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get probability predictions.
        
        Args:
            x: Input tensor
            
        Returns:
            Class probabilities
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)
