"""
Focal Loss Implementation for Class Imbalance

Implements focal loss to handle class imbalance in ECG arrhythmia classification.
Provides better performance on imbalanced datasets like MIT-BIH.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Focal Loss = -(1 - pt)^γ * log(pt)
    where pt is the model's estimated probability for the correct class.
    
    This loss function focuses training on hard, misclassified examples
    and reduces the relative loss for well-classified examples.
    """
    
    def __init__(
        self,
        alpha: Optional[Union[float, torch.Tensor]] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        label_smoothing: float = 0.0,
        eps: float = 1e-8
    ):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for rare classes (float or tensor)
            gamma: Focusing parameter (γ > 0)
            reduction: Reduction method ('none', 'mean', 'sum')
            label_smoothing: Label smoothing factor
            eps: Small constant for numerical stability
        """
        super().__init__()
        
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.eps = eps
        
        if gamma < 0:
            raise ValueError("Gamma must be non-negative")
        
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError("Reduction must be 'none', 'mean', or 'sum'")
        
        if label_smoothing < 0 or label_smoothing >= 1.0:
            raise ValueError("Label smoothing must be in [0, 1)")
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate focal loss.
        
        Args:
            inputs: Predictions (logits) of shape (batch_size, num_classes)
            targets: Ground truth labels of shape (batch_size)
            
        Returns:
            Focal loss tensor
        """
        num_classes = inputs.size(1)
        batch_size = inputs.size(0)
        
        # Apply label smoothing
        if self.label_smoothing > 0:
            targets = self._apply_label_smoothing(targets, num_classes)
        
        # Calculate cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Calculate probability of correct class
        pt = torch.exp(-ce_loss)
        
        # Apply focal term
        focal_term = (1 - pt) ** self.gamma
        
        # Apply alpha weighting
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            else:
                alpha_t = self.alpha.gather(0, targets)
            
            focal_loss = alpha_t * focal_term * ce_loss
        else:
            focal_loss = focal_term * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss
    
    def _apply_label_smoothing(self, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
        """Apply label smoothing to targets."""
        # Convert to one-hot encoding
        smooth_targets = torch.zeros_like(inputs)
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0)
        
        # Apply smoothing
        smooth_targets = smooth_targets * (1 - self.label_smoothing)
        smooth_targets += self.label_smoothing / num_classes
        
        return smooth_targets

class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Cross Entropy Loss.
    
    Implements label smoothing to improve model generalization
    and prevent overconfident predictions.
    """
    
    def __init__(self, smoothing: float = 0.1, reduction: str = 'mean'):
        """
        Initialize Label Smoothing Loss.
        
        Args:
            smoothing: Smoothing factor (0.0 to 1.0)
            reduction: Reduction method ('none', 'mean', 'sum')
        """
        super().__init__()
        
        self.smoothing = smoothing
        self.reduction = reduction
        self.confidence = 1.0 - smoothing
        
        if smoothing < 0.0 or smoothing >= 1.0:
            raise ValueError("Smoothing must be in [0.0, 1.0)")
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate label smoothed cross entropy loss.
        
        Args:
            inputs: Predictions (logits) of shape (batch_size, num_classes)
            targets: Ground truth labels of shape (batch_size)
            
        Returns:
            Label smoothed loss
        """
        num_classes = inputs.size(1)
        
        # Create smooth targets
        smooth_targets = torch.zeros_like(inputs)
        smooth_targets.fill_(self.smoothing / (num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        # Calculate log softmax
        log_probs = F.log_softmax(inputs, dim=1)
        
        # Calculate loss
        loss = -torch.sum(smooth_targets * log_probs, dim=1)
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss combining focal loss with class frequency weighting.
    
    Automatically calculates class weights based on frequency in the dataset
    and applies them to focal loss for better handling of imbalance.
    """
    
    def __init__(
        self,
        num_classes: int,
        gamma: float = 2.0,
        beta: float = 0.9999,
        reduction: str = 'mean'
    ):
        """
        Initialize Class-Balanced Loss.
        
        Args:
            num_classes: Number of classes
            gamma: Focal loss gamma parameter
            beta: Effective number weighting parameter
            reduction: Reduction method
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.gamma = gamma
        self.beta = beta
        self.reduction = reduction
        
        # Initialize class weights (will be updated with actual frequencies)
        self.register_buffer('class_weights', torch.ones(num_classes))
        self.register_buffer('class_counts', torch.zeros(num_classes))
        self.total_samples = 0
    
    def update_class_weights(self, class_counts: torch.Tensor):
        """
        Update class weights based on current class frequencies.
        
        Args:
            class_counts: Tensor with count of each class
        """
        self.class_counts = class_counts.float()
        self.total_samples = class_counts.sum().item()
        
        # Calculate effective number of samples
        effective_num = 1.0 - torch.pow(self.beta, self.class_counts / self.total_samples)
        effective_num = (effective_num - 1) / (self.beta - 1)
        
        # Calculate class weights
        class_weights = self.total_samples / (self.num_classes * effective_num)
        
        # Normalize weights
        class_weights = class_weights / class_weights.sum() * self.num_classes
        
        self.class_weights = class_weights
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate class-balanced focal loss.
        
        Args:
            inputs: Predictions (logits) of shape (batch_size, num_classes)
            targets: Ground truth labels of shape (batch_size)
            
        Returns:
            Class-balanced focal loss
        """
        # Calculate cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Calculate probability of correct class
        pt = torch.exp(-ce_loss)
        
        # Apply focal term
        focal_term = (1 - pt) ** self.gamma
        
        # Get class weights for current batch
        alpha_t = self.class_weights.gather(0, targets)
        
        # Calculate final loss
        focal_loss = alpha_t * focal_term * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss

class CombinedLoss(nn.Module):
    """
    Combined Loss function for ECG arrhythmia classification.
    
    Combines multiple loss functions to improve model performance:
    - Focal loss for class imbalance
    - Label smoothing for generalization
    - Auxiliary loss for intermediate features
    """
    
    def __init__(
        self,
        num_classes: int,
        focal_alpha: Optional[torch.Tensor] = None,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1,
        auxiliary_weight: float = 0.2,
        reduction: str = 'mean'
    ):
        """
        Initialize Combined Loss.
        
        Args:
            num_classes: Number of classes
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
            label_smoothing: Label smoothing factor
            auxiliary_weight: Weight for auxiliary loss
            reduction: Reduction method
        """
        super().__init__()
        
        self.focal_loss = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            reduction='none',
            label_smoothing=label_smoothing
        )
        
        self.label_smoothing_loss = LabelSmoothingLoss(
            smoothing=label_smoothing,
            reduction='none'
        )
        
        self.auxiliary_weight = auxiliary_weight
        self.reduction = reduction
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        auxiliary_outputs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate combined loss.
        
        Args:
            inputs: Main predictions (logits)
            targets: Ground truth labels
            auxiliary_outputs: Auxiliary predictions for intermediate supervision
            
        Returns:
            Combined loss
        """
        # Main losses
        focal_loss = self.focal_loss(inputs, targets)
        smooth_loss = self.label_smoothing_loss(inputs, targets)
        
        # Combine main losses
        main_loss = 0.7 * focal_loss + 0.3 * smooth_loss
        
        # Add auxiliary loss if available
        if auxiliary_outputs is not None:
            aux_loss = F.cross_entropy(auxiliary_outputs, targets, reduction='none')
            total_loss = main_loss + self.auxiliary_weight * aux_loss
        else:
            total_loss = main_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
        else:  # 'none'
            return total_loss

def create_loss_function(
    loss_type: str,
    num_classes: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create different loss functions.
    
    Args:
        loss_type: Type of loss function
        num_classes: Number of classes
        **kwargs: Additional parameters
        
    Returns:
        Loss function instance
    """
    if loss_type == "focal":
        return FocalLoss(num_classes=num_classes, **kwargs)
    elif loss_type == "label_smoothing":
        return LabelSmoothingLoss(**kwargs)
    elif loss_type == "class_balanced":
        return ClassBalancedLoss(num_classes=num_classes, **kwargs)
    elif loss_type == "combined":
        return CombinedLoss(num_classes=num_classes, **kwargs)
    elif loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

if __name__ == "__main__":
    # Test loss functions
    batch_size = 4
    num_classes = 5
    sequence_length = 1000
    
    # Create dummy data
    inputs = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    print("Testing Loss Functions:")
    print("=" * 50)
    
    # Test Focal Loss
    focal_loss = FocalLoss(alpha=torch.tensor([1.0, 2.0, 3.0, 2.0, 2.0]), gamma=2.0)
    focal_output = focal_loss(inputs, targets)
    print(f"Focal Loss: {focal_output.mean().item():.4f}")
    
    # Test Label Smoothing Loss
    smooth_loss = LabelSmoothingLoss(smoothing=0.1)
    smooth_output = smooth_loss(inputs, targets)
    print(f"Label Smoothing Loss: {smooth_output.mean().item():.4f}")
    
    # Test Class Balanced Loss
    balanced_loss = ClassBalancedLoss(num_classes=num_classes)
    # Simulate class frequencies (imbalanced)
    class_counts = torch.tensor([1000, 100, 50, 25, 25])  # Imbalanced
    balanced_loss.update_class_weights(class_counts)
    balanced_output = balanced_loss(inputs, targets)
    print(f"Class Balanced Loss: {balanced_output.mean().item():.4f}")
    
    # Test Combined Loss
    combined_loss = CombinedLoss(num_classes=num_classes)
    combined_output = combined_loss(inputs, targets)
    print(f"Combined Loss: {combined_output.mean().item():.4f}")
    
    # Test Cross Entropy (baseline)
    ce_loss = nn.CrossEntropyLoss()
    ce_output = ce_loss(inputs, targets)
    print(f"Cross Entropy Loss: {ce_output.mean().item():.4f}")
    
    print("=" * 50)
    print("Loss comparison for imbalanced classification:")
    print("  - Focal Loss: Focuses on hard examples")
    print("  - Label Smoothing: Prevents overconfidence")
    print("  - Class Balanced: Handles frequency imbalance")
    print("  - Combined: Multiple loss strategies")
