"""
Evaluation Metrics for ECG Classification
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
from typing import Dict, Tuple
import torch


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
    average: str = 'weighted'
) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (for AUC)
        average: Averaging method for multi-class metrics
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    # Sensitivity and Specificity (for binary or per-class)
    if len(np.unique(y_true)) == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # AUC if probabilities provided
    if y_prob is not None:
        try:
            if len(np.unique(y_true)) == 2:
                # Binary classification
                metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                # Multi-class (one-vs-rest)
                metrics['auc'] = roc_auc_score(
                    y_true, y_prob, 
                    multi_class='ovr', 
                    average=average
                )
        except ValueError:
            pass
    
    return metrics


def calculate_clinical_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    critical_classes: list = None
) -> Dict[str, float]:
    """
    Calculate clinical performance metrics.
    
    Focus on critical arrhythmias (VFib, VT, etc.)
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        critical_classes: List of critical class indices
        
    Returns:
        Clinical metrics
    """
    metrics = {}
    
    if critical_classes is None:
        # Default: VFib (5), VT (4) are critical
        critical_classes = [4, 5]
    
    # Create binary labels (critical vs non-critical)
    y_true_binary = np.isin(y_true, critical_classes).astype(int)
    y_pred_binary = np.isin(y_pred, critical_classes).astype(int)
    
    # Calculate metrics for critical events
    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
    
    metrics['critical_sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['critical_specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['critical_ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
    metrics['critical_npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
    
    # False alarm rate (important for clinical use)
    metrics['false_alarm_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # Missed critical events rate
    metrics['missed_critical_rate'] = fn / (tp + fn) if (tp + fn) > 0 else 0
    
    return metrics


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list = None
):
    """
    Print detailed classification report.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
    """
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4
    )
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Overall metrics
    metrics = calculate_metrics(y_true, y_pred)
    print("\nOverall Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Clinical metrics
    clinical_metrics = calculate_clinical_metrics(y_true, y_pred)
    print("\nClinical Metrics (Critical Events):")
    for key, value in clinical_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("="*60 + "\n")


def calculate_fda_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None
) -> Dict[str, float]:
    """
    Calculate FDA-required performance metrics.
    
    For medical device approval (510(k)).
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        
    Returns:
        FDA metrics
    """
    metrics = {}
    
    # Sensitivity (True Positive Rate) - Critical for FDA
    # Must be >95% for critical arrhythmias
    metrics['sensitivity'] = recall_score(y_true, y_pred, average='weighted')
    
    # Specificity
    cm = confusion_matrix(y_true, y_pred)
    specificity_per_class = []
    for i in range(len(cm)):
        tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
        fp = np.sum(cm[:, i]) - cm[i, i]
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity_per_class.append(spec)
    metrics['specificity'] = np.mean(specificity_per_class)
    
    # Positive Predictive Value (Precision)
    metrics['ppv'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Negative Predictive Value
    npv_per_class = []
    for i in range(len(cm)):
        tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
        fn = np.sum(cm[i, :]) - cm[i, i]
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        npv_per_class.append(npv)
    metrics['npv'] = np.mean(npv_per_class)
    
    # Accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # F1 Score
    metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
    
    # AUC (if probabilities available)
    if y_prob is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
        except ValueError:
            pass
    
    return metrics
