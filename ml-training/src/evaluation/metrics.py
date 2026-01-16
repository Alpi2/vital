"""
Evaluation Metrics for ECG Classification

Comprehensive metrics calculation and visualization
for ECG arrhythmia classification models.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (optional)
        class_names: Class names (optional)
        
    Returns:
        Dictionary of metrics
    """
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'precision_micro': precision_micro,
        'precision_weighted': precision_weighted,
        'recall_macro': recall_macro,
        'recall_micro': recall_micro,
        'recall_weighted': recall_weighted,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'f1_weighted': f1_weighted
    }
    
    # Add per-class metrics
    num_classes = len(np.unique(y_true))
    for i in range(num_classes):
        class_name = f'class_{i}' if not class_names else class_names[i]
        metrics[f'precision_{class_name}'] = precision_per_class[i] if i < len(precision_per_class) else 0.0
        metrics[f'recall_{class_name}'] = recall_per_class[i] if i < len(recall_per_class) else 0.0
        metrics[f'f1_{class_name}'] = f1_per_class[i] if i < len(f1_per_class) else 0.0
    
    # ROC-AUC if probabilities are provided
    if y_prob is not None:
        try:
            # Multi-class ROC-AUC
            if len(np.unique(y_true)) > 2:
                roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
                metrics['roc_auc_macro'] = roc_auc
                metrics['roc_auc_weighted'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
            else:
                roc_auc = roc_auc_score(y_true, y_prob[:, 1])
                metrics['roc_auc'] = roc_auc
            
            # Average precision
            if len(np.unique(y_true)) > 2:
                avg_precision = average_precision_score(y_true, y_prob, average='macro')
                metrics['avg_precision_macro'] = avg_precision
                metrics['avg_precision_weighted'] = average_precision_score(y_true, y_prob, average='weighted')
            else:
                avg_precision = average_precision_score(y_true, y_prob[:, 1])
                metrics['avg_precision'] = avg_precision
                
        except Exception as e:
            logger.warning(f"Failed to calculate ROC-AUC: {e}")
    
    return metrics

def calculate_confusion_matrix_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Calculate detailed confusion matrix metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with confusion matrix and derived metrics
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate TP, TN, FP, FN for each class
    num_classes = cm.shape[0]
    metrics = {}
    
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        # Calculate metrics for this class
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivity = recall  # Same as recall
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics[f'class_{i}_tp'] = tp
        metrics[f'class_{i}_tn'] = tn
        metrics[f'class_{i}_fp'] = fp
        metrics[f'class_{i}_fn'] = fn
        metrics[f'class_{i}_precision'] = precision
        metrics[f'class_{i}_recall'] = recall
        metrics[f'class_{i}_specificity'] = specificity
        metrics[f'class_{i}_sensitivity'] = sensitivity
        metrics[f'class_{i}_f1'] = f1
    
    # Add confusion matrix
    metrics['confusion_matrix'] = cm
    
    # Overall metrics
    metrics['total_tp'] = sum(metrics[f'class_{i}_tp'] for i in range(num_classes))
    metrics['total_fp'] = sum(metrics[f'class_{i}_fp'] for i in range(num_classes))
    metrics['total_fn'] = sum(metrics[f'class_{i}_fn'] for i in range(num_classes))
    metrics['total_tn'] = sum(metrics[f'class_{i}_tn'] for i in range(num_classes))
    
    return metrics

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    normalize: bool = True
) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Class names
        save_path: Path to save plot
        normalize: Whether to normalize the matrix
        
    Returns:
        Matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.round(cm, 2)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        ax=ax,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    
    # Labels
    if class_names:
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names)
    else:
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    
    return fig

def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot ROC curves for multi-class classification.
    
    Args:
        y_true: True labels
        y_prob: Prediction probabilities
        class_names: Class names
        save_path: Path to save plot
        
    Returns:
        Matplotlib figure
    """
    num_classes = y_prob.shape[1]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot ROC curve for each class
    for i in range(num_classes):
        # Create binary labels for this class
        y_true_binary = (y_true == i).astype(int)
        y_prob_class = y_prob[:, i]
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true_binary, y_prob_class)
        roc_auc = roc_auc_score(y_true_binary, y_prob_class)
        
        # Plot
        class_label = f'Class {i}' if not class_names else class_names[i]
        ax.plot(fpr, tpr, label=f'{class_label} (AUC = {roc_auc:.3f})', linewidth=2)
    
    # Plot diagonal line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curves saved to {save_path}")
    
    return fig

def plot_precision_recall_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot Precision-Recall curves for multi-class classification.
    
    Args:
        y_true: True labels
        y_prob: Prediction probabilities
        class_names: Class names
        save_path: Path to save plot
        
    Returns:
        Matplotlib figure
    """
    num_classes = y_prob.shape[1]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot PR curve for each class
    for i in range(num_classes):
        # Create binary labels for this class
        y_true_binary = (y_true == i).astype(int)
        y_prob_class = y_prob[:, i]
        
        # Calculate PR curve
        precision, recall, _ = precision_recall_curve(y_true_binary, y_prob_class)
        avg_precision = average_precision_score(y_true_binary, y_prob_class)
        
        # Plot
        class_label = f'Class {i}' if not class_names else class_names[i]
        ax.plot(recall, precision, label=f'{class_label} (AP = {avg_precision:.3f})', linewidth=2)
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"PR curves saved to {save_path}")
    
    return fig

def plot_class_performance(
    metrics: Dict[str, Any],
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot per-class performance metrics.
    
    Args:
        metrics: Metrics dictionary
        class_names: Class names
        save_path: Path to save plot
        
    Returns:
        Matplotlib figure
    """
    # Extract per-class metrics
    num_classes = len([k for k in metrics.keys() if k.startswith('precision_')])
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(num_classes)]
    
    precision = []
    recall = []
    f1 = []
    
    for i in range(num_classes):
        precision.append(metrics.get(f'precision_class_{i}', 0.0))
        recall.append(metrics.get(f'recall_class_{i}', 0.0))
        f1.append(metrics.get(f'f1_class_{i}', 0.0))
    
    # Create plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    x = np.arange(num_classes)
    width = 0.25
    
    # Precision
    ax1.bar(x, precision, width, label='Precision', color='skyblue', alpha=0.8)
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Precision')
    ax1.set_title('Per-Class Precision')
    ax1.set_xticks(x + width/2)
    ax1.set_xticklabels(class_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Recall
    ax2.bar(x, recall, width, label='Recall', color='lightcoral', alpha=0.8)
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Recall')
    ax2.set_title('Per-Class Recall')
    ax2.set_xticks(x + width/2)
    ax2.set_xticklabels(class_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # F1 Score
    ax3.bar(x, f1, width, label='F1-Score', color='lightgreen', alpha=0.8)
    ax3.set_xlabel('Class')
    ax3.set_ylabel('F1-Score')
    ax3.set_title('Per-Class F1-Score')
    ax3.set_xticks(x + width/2)
    ax3.set_xticklabels(class_names, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Class performance plot saved to {save_path}")
    
    return fig

def calculate_model_complexity(model) -> Dict[str, Any]:
    """
    Calculate model complexity metrics.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with complexity metrics
    """
    import torch
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate model size
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    model_size_mb = (param_size + buffer_size) / (1024 ** 2)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'model_size_mb': model_size_mb,
        'parameter_size_mb': param_size / (1024 ** 2),
        'buffer_size_mb': buffer_size / (1024 ** 2)
    }

def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> str:
    """
    Generate comprehensive classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities
        class_names: Class names
        save_path: Path to save report
        
    Returns:
        Classification report as string
    """
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_prob, class_names)
    cm_metrics = calculate_confusion_matrix_metrics(y_true, y_pred)
    
    # Generate report
    report = []
    report.append("=" * 60)
    report.append("CLASSIFICATION REPORT")
    report.append("=" * 60)
    
    # Overall metrics
    report.append("\nOVERALL METRICS:")
    report.append(f"Accuracy: {metrics['accuracy']:.4f}")
    report.append(f"Precision (Macro): {metrics['precision_macro']:.4f}")
    report.append(f"Precision (Weighted): {metrics['precision_weighted']:.4f}")
    report.append(f"Recall (Macro): {metrics['recall_macro']:.4f}")
    report.append(f"Recall (Weighted): {metrics['recall_weighted']:.4f}")
    report.append(f"F1-Score (Macro): {metrics['f1_macro']:.4f}")
    report.append(f"F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
    
    if 'roc_auc_macro' in metrics:
        report.append(f"ROC-AUC (Macro): {metrics['roc_auc_macro']:.4f}")
    
    # Per-class metrics
    num_classes = len([k for k in metrics.keys() if k.startswith('precision_')])
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(num_classes)]
    
    report.append("\nPER-CLASS METRICS:")
    report.append("-" * 40)
    
    for i in range(num_classes):
        class_name = class_names[i]
        precision = metrics.get(f'precision_class_{i}', 0.0)
        recall = metrics.get(f'recall_class_{i}', 0.0)
        f1 = metrics.get(f'f1_class_{i}', 0.0)
        tp = cm_metrics.get(f'class_{i}_tp', 0)
        tn = cm_metrics.get(f'class_{i}_tn', 0)
        fp = cm_metrics.get(f'class_{i}_fp', 0)
        fn = cm_metrics.get(f'class_{i}_fn', 0)
        
        report.append(f"\n{class_name}:")
        report.append(f"  Precision: {precision:.4f}")
        report.append(f"  Recall: {recall:.4f}")
        report.append(f"  F1-Score: {f1:.4f}")
        report.append(f"  Specificity: {tn/(tn+fp) if (tn+fp) > 0 else 0.0:.4f}")
        report.append(f"  Sensitivity: {recall:.4f}")
        report.append(f"  True Positives: {tp}")
        report.append(f"  True Negatives: {tn}")
        report.append(f"  False Positives: {fp}")
        report.append(f"  False Negatives: {fn}")
    
    # Confusion matrix
    report.append("\nCONFUSION MATRIX:")
    report.append("-" * 40)
    cm = cm_metrics['confusion_matrix']
    
    # Header
    header = "Predicted ->"
    if class_names:
        for name in class_names:
            header += f"{name:>10}"
    else:
        for i in range(num_classes):
            header += f"Class {i:>10}"
    report.append(header)
    
    # Matrix rows
    for i in range(num_classes):
        row = f"True {i:2} ->"
        for j in range(num_classes):
            row += f"{cm[i][j]:>10}"
        report.append(row)
    
    report.append("\n" + "=" * 60)
    
    report_str = "\n".join(report)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_str)
        logger.info(f"Classification report saved to {save_path}")
    
    return report_str

if __name__ == "__main__":
    # Test evaluation metrics
    np.random.seed(42)
    
    # Generate dummy data
    n_samples = 1000
    n_classes = 5
    
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = np.random.randint(0, n_classes, n_samples)
    y_prob = np.random.dirichlet(np.ones(n_classes), n_samples)
    
    class_names = ['Normal', 'AFIB', 'PVC', 'VT', 'Other']
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_prob, class_names)
    
    print("Evaluation Metrics Test:")
    print("=" * 50)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1-Score (Macro): {metrics['f1_macro']:.4f}")
    print(f"F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
    
    # Generate report
    report = generate_classification_report(y_true, y_pred, y_prob, class_names)
    print(report)
