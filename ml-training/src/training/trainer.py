"""
Training Loop for ECG Models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable
import numpy as np
from tqdm import tqdm
import mlflow
from pathlib import Path


class Trainer:
    """
    Training manager for ECG classification models.
    
    Features:
    - Training and validation loops
    - Learning rate scheduling
    - Early stopping
    - Model checkpointing
    - MLflow experiment tracking
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: str = 'cuda',
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        early_stopping_patience: int = 10,
        checkpoint_dir: str = 'checkpoints',
        use_mlflow: bool = True
    ):
        """
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on
            scheduler: Learning rate scheduler
            early_stopping_patience: Patience for early stopping
            checkpoint_dir: Directory to save checkpoints
            use_mlflow: Whether to use MLflow tracking
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.use_mlflow = use_mlflow
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch} [Train]')
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def validate(self) -> Dict[str, float]:
        """
        Validate the model.
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {self.current_epoch} [Val]')
            
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                
                pbar.set_postfix({
                    'loss': total_loss / len(self.val_loader),
                    'acc': 100. * correct / total
                })
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'predictions': np.array(all_preds),
            'targets': np.array(all_targets)
        }
    
    def save_checkpoint(self, filename: str = 'checkpoint.pth'):
        """
        Save model checkpoint.
        
        Args:
            filename: Checkpoint filename
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, self.checkpoint_dir / filename)
    
    def load_checkpoint(self, filename: str = 'checkpoint.pth'):
        """
        Load model checkpoint.
        
        Args:
            filename: Checkpoint filename
        """
        checkpoint = torch.load(self.checkpoint_dir / filename)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    def train(self, num_epochs: int, experiment_name: str = 'ecg_training'):
        """
        Full training loop.
        
        Args:
            num_epochs: Number of epochs to train
            experiment_name: MLflow experiment name
        """
        if self.use_mlflow:
            mlflow.set_experiment(experiment_name)
            mlflow.start_run()
        
        try:
            for epoch in range(num_epochs):
                self.current_epoch = epoch
                
                # Train
                train_metrics = self.train_epoch()
                
                # Validate
                val_metrics = self.validate()
                
                # Update history
                self.history['train_loss'].append(train_metrics['loss'])
                self.history['train_acc'].append(train_metrics['accuracy'])
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_acc'].append(val_metrics['accuracy'])
                
                # Log to MLflow
                if self.use_mlflow:
                    mlflow.log_metrics({
                        'train_loss': train_metrics['loss'],
                        'train_acc': train_metrics['accuracy'],
                        'val_loss': val_metrics['loss'],
                        'val_acc': val_metrics['accuracy']
                    }, step=epoch)
                
                # Learning rate scheduling
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['loss'])
                    else:
                        self.scheduler.step()
                
                # Save best model
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint('best_model.pth')
                    self.patience_counter = 0
                    print(f"\nNew best model saved! Val Loss: {val_metrics['loss']:.4f}")
                else:
                    self.patience_counter += 1
                
                # Early stopping
                if self.patience_counter >= self.early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break
                
                # Save regular checkpoint
                if (epoch + 1) % 10 == 0:
                    self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
            
            # Save final model
            self.save_checkpoint('final_model.pth')
            
        finally:
            if self.use_mlflow:
                mlflow.end_run()
        
        return self.history
