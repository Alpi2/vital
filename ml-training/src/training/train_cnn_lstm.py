"""
CNN-LSTM Training Script with Advanced Features

Comprehensive training script for ECG arrhythmia classification
with CNN-LSTM architecture, focal loss, and advanced optimization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import argparse
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from models.cnn_lstm_attention import CNNLSTMAttention, create_cnn_lstm_model
from losses.focal_loss import FocalLoss, create_loss_function
from data.mitbih_dataset import create_data_loaders, MITBIHDataset
from evaluation.metrics import calculate_metrics, plot_confusion_matrix

logger = logging.getLogger(__name__)

class Trainer:
    """
    Advanced trainer for CNN-LSTM ECG classification.
    
    Features:
    - Mixed precision training
    - Gradient clipping
    - Learning rate scheduling
    - Early stopping
    - Model checkpointing
    - TensorBoard logging
    - Class imbalance handling
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: Dict[str, Any]
    ):
        """
        Initialize trainer.
        
        Args:
            model: CNN-LSTM model
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            config: Training configuration
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.get('use_gpu', True) else 'cpu')
        self.model.to(self.device)
        
        # Mixed precision training
        self.use_amp = config.get('use_amp', True) and self.device.type == 'cuda'
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Loss function
        self.criterion = create_loss_function(
            loss_type=config.get('loss_type', 'focal'),
            num_classes=config.get('num_classes', 5),
            alpha=config.get('focal_alpha'),
            gamma=config.get('focal_gamma', 2.0),
            label_smoothing=config.get('label_smoothing', 0.1)
        )
        self.criterion.to(self.device)
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Logging
        self.log_dir = Path(config.get('log_dir', 'logs'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # Checkpointing
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Using AMP: {self.use_amp}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        optimizer_type = self.config.get('optimizer', 'adamw')
        lr = self.config.get('learning_rate', 0.001)
        weight_decay = self.config.get('weight_decay', 1e-4)
        
        if optimizer_type == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_type == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay,
                nesterov=True
            )
        elif optimizer_type == 'rmsprop':
            return optim.RMSprop(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler."""
        scheduler_type = self.config.get('scheduler', 'cosine')
        
        if scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('num_epochs', 100),
                eta_min=self.config.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=self.config.get('scheduler_patience', 5),
                min_lr=self.config.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'step':
            return optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=self.config.get('milestones', [30, 60, 90]),
                gamma=0.1
            )
        elif scheduler_type == 'exponential':
            return optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=0.95
            )
        else:
            return optim.lr_scheduler.ConstantLR(self.optimizer)
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(self.train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(data)
                    loss = self.criterion(outputs, targets)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.get('grad_clip', 1.0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.get('grad_clip', 1.0)
                    )
                
                self.scaler.step(self.optimizer)
            else:
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                loss.backward()
                
                # Gradient clipping
                if self.config.get('grad_clip', 1.0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.get('grad_clip', 1.0)
                    )
                
                self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Log batch statistics
            if batch_idx % 100 == 0:
                self.writer.add_scalar('Train/BatchLoss', loss.item(), self.current_epoch * len(self.train_loader) + batch_idx)
                self.writer.add_scalar('Train/BatchAcc', 100. * correct / total, self.current_epoch * len(self.train_loader) + batch_idx)
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self) -> Tuple[float, float, Dict[str, Any]]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in self.val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(data)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(data)
                        loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total
        
        # Calculate detailed metrics
        metrics = calculate_metrics(all_targets, all_predictions)
        
        return avg_loss, accuracy, metrics
    
    def train(self) -> Dict[str, Any]:
        """Main training loop."""
        num_epochs = self.config.get('num_epochs', 100)
        patience = self.config.get('early_stopping_patience', 15)
        save_best_only = self.config.get('save_best_only', True)
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch()
            
            # Validation
            val_loss, val_acc, val_metrics = self.validate_epoch()
            
            # Learning rate scheduling
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_acc)
            else:
                self.scheduler.step()
            
            epoch_time = time.time() - epoch_start_time
            
            # Logging
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            self.writer.add_scalar('Train/EpochLoss', train_loss, epoch)
            self.writer.add_scalar('Train/EpochAcc', train_acc, epoch)
            self.writer.add_scalar('Val/EpochLoss', val_loss, epoch)
            self.writer.add_scalar('Val/EpochAcc', val_acc, epoch)
            self.writer.add_scalar('Train/LearningRate', self.optimizer.param_groups[0]['lr'], epoch)
            self.writer.add_scalar('Time/EpochTime', epoch_time, epoch)
            
            # Log detailed metrics
            for metric_name, metric_value in val_metrics.items():
                self.writer.add_scalar(f'Val/{metric_name}', metric_value, epoch)
            
            logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                       f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
                       f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%, "
                       f"Time={epoch_time:.2f}s")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                self._save_checkpoint(epoch, is_best=True)
                logger.info(f"New best model saved with accuracy: {val_acc:.2f}%")
            else:
                self.patience_counter += 1
            
            # Save regular checkpoints
            if not save_best_only and (epoch + 1) % self.config.get('save_interval', 10) == 0:
                self._save_checkpoint(epoch, is_best=False)
            
            # Early stopping
            if self.patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Close TensorBoard writer
        self.writer.close()
        
        # Return training history
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'total_epochs': epoch + 1
        }
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'model_info': self.model.get_model_info() if hasattr(self.model, 'get_model_info') else {}
        }
        
        if is_best:
            checkpoint_path = self.checkpoint_dir / 'best_model.pth'
            logger.info(f"Saving best model to {checkpoint_path}")
        else:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
            logger.info(f"Saving checkpoint to {checkpoint_path}")
        
        torch.save(checkpoint, checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return 0
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_acc = checkpoint['best_val_acc']
        self.best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint['epoch'] + 1
    
    def test(self) -> Dict[str, Any]:
        """Test model on test set."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, targets in self.test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(data)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(data)
                        loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        avg_loss = total_loss / len(self.test_loader)
        accuracy = 100.0 * correct / total
        
        # Calculate detailed metrics
        metrics = calculate_metrics(all_targets, all_predictions, all_probabilities)
        
        # Generate classification report
        class_names = ['Normal', 'Bundle Branch Block', 'Supraventricular Ectopic', 'Ventricular Ectopic', 'Other']
        report = classification_report(all_targets, all_predictions, target_names=class_names, output_dict=True)
        
        logger.info(f"Test Results:")
        logger.info(f"  Loss: {avg_loss:.4f}")
        logger.info(f"  Accuracy: {accuracy:.2f}%")
        logger.info(f"  F1 Score (macro): {metrics['f1_macro']:.4f}")
        logger.info(f"  F1 Score (weighted): {metrics['f1_weighted']:.4f}")
        
        return {
            'test_loss': avg_loss,
            'test_accuracy': accuracy,
            'metrics': metrics,
            'classification_report': report,
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities
        }
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.train_losses, label='Train Loss')
        axes[0, 0].plot(self.val_losses, label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(self.train_accuracies, label='Train Acc')
        axes[0, 1].plot(self.val_accuracies, label='Val Acc')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate plot
        lrs = [self.optimizer.param_groups[0]['lr'] for _ in range(len(self.train_losses))]
        axes[1, 0].plot(lrs)
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True)
        
        # Remove empty subplot
        fig.delaxes(axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        else:
            plt.show()

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train CNN-LSTM for ECG Classification')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='mitbih_data', help='Path to MIT-BIH data')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='standard', choices=['standard', 'lightweight', 'heavy', 'multiscale'])
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes')
    parser.add_argument('--sequence_length', type=int, default=1000, help='Input sequence length')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'adam', 'sgd', 'rmsprop'])
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'plateau', 'step', 'exponential'])
    parser.add_argument('--loss_type', type=str, default='focal', choices=['focal', 'cross_entropy', 'label_smoothing', 'class_balanced', 'combined'])
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping norm')
    
    # System arguments
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--use_amp', action='store_true', help='Use mixed precision training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Logging arguments
    parser.add_argument('--log_dir', type=str, default='logs', help='TensorBoard log directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        random_seed=args.seed
    )
    
    # Create model
    model = create_cnn_lstm_model(
        model_type=args.model_type,
        num_classes=args.num_classes,
        input_channels=1,
        sequence_length=args.sequence_length
    )
    
    # Training configuration
    config = {
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'optimizer': args.optimizer,
        'scheduler': args.scheduler,
        'loss_type': args.loss_type,
        'weight_decay': args.weight_decay,
        'grad_clip': args.grad_clip,
        'use_gpu': args.use_gpu,
        'use_amp': args.use_amp,
        'log_dir': args.log_dir,
        'checkpoint_dir': args.checkpoint_dir,
        'early_stopping_patience': 15,
        'save_interval': 10,
        'save_best_only': False,
        'focal_alpha': None,  # Will be calculated from data
        'focal_gamma': 2.0,
        'label_smoothing': 0.1,
        'min_lr': 1e-6,
        'milestones': [30, 60, 90]
    }
    
    # Calculate focal alpha from training data
    if args.loss_type == 'focal':
        # Get class distribution from training data
        train_dataset = train_loader.dataset.dataset
        train_labels = [train_dataset.labels[i] for i in train_dataset.indices]
        unique_labels, counts = np.unique(train_labels, return_counts=True)
        total_samples = len(train_labels)
        
        # Calculate alpha values (inverse frequency)
        alpha_values = np.ones(args.num_classes)
        for label, count in zip(unique_labels, counts):
            if label < args.num_classes:
                alpha_values[label] = total_samples / (len(unique_labels) * count)
        
        # Normalize alpha values
        alpha_values = alpha_values / np.sum(alpha_values) * args.num_classes
        config['focal_alpha'] = torch.FloatTensor(alpha_values)
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, test_loader, config)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
    
    # Train model
    training_history = trainer.train()
    
    # Test model
    test_results = trainer.test()
    
    # Save final model
    final_checkpoint_path = Path(config['checkpoint_dir']) / 'final_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'model_info': model.get_model_info(),
        'training_history': training_history,
        'test_results': test_results
    }, final_checkpoint_path)
    
    # Plot training history
    plot_path = Path(config['log_dir']) / 'training_history.png'
    trainer.plot_training_history(str(plot_path))
    
    # Save results
    results_path = Path(config['log_dir']) / 'training_results.json'
    results = {
        'config': config,
        'training_history': training_history,
        'test_results': test_results,
        'model_info': model.get_model_info()
    }
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Training completed successfully!")
    logger.info(f"Best validation accuracy: {training_history['best_val_acc']:.2f}%")
    logger.info(f"Test accuracy: {test_results['test_accuracy']:.2f}%")
    logger.info(f"Model saved to: {final_checkpoint_path}")
    logger.info(f"Results saved to: {results_path}")

if __name__ == '__main__':
    main()
