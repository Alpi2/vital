"""
Transformer Model for ECG Analysis
Attention-based architecture for temporal pattern recognition
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class ECGTransformer(nn.Module):
    """
    Transformer model for ECG classification.
    
    Uses self-attention to capture long-range dependencies in ECG signals.
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        input_channels: int = 1,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 1000
    ):
        """
        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels
            d_model: Dimension of model embeddings
            nhead: Number of attention heads
            num_encoder_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
        """
        super(ECGTransformer, self).__init__()
        
        self.d_model = d_model
        
        # Input embedding
        self.input_projection = nn.Conv1d(input_channels, d_model, kernel_size=1)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, channels, seq_len)
            
        Returns:
            Output logits (batch_size, num_classes)
        """
        # Project input to d_model dimensions
        x = self.input_projection(x)  # (batch, d_model, seq_len)
        
        # Reshape for transformer (seq_len, batch, d_model)
        x = x.permute(2, 0, 1)
        
        # Add positional encoding
        x = self.pos_encoder(x * math.sqrt(self.d_model))
        
        # Transformer encoding
        x = self.transformer_encoder(x)
        
        # Global average pooling over sequence
        x = x.mean(dim=0)  # (batch, d_model)
        
        # Classification
        out = self.classifier(x)
        
        return out


class MultiScaleTransformer(nn.Module):
    """
    Multi-scale transformer for ECG analysis.
    Processes signal at multiple temporal resolutions.
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        input_channels: int = 1,
        scales: list = [1, 2, 4],
        **transformer_kwargs
    ):
        super(MultiScaleTransformer, self).__init__()
        
        self.scales = scales
        
        # Create transformer for each scale
        self.transformers = nn.ModuleList([
            ECGTransformer(num_classes, input_channels, **transformer_kwargs)
            for _ in scales
        ])
        
        # Fusion layer
        d_model = transformer_kwargs.get('d_model', 128)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * len(scales), 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = []
        
        for scale, transformer in zip(self.scales, self.transformers):
            # Downsample if scale > 1
            if scale > 1:
                x_scaled = nn.functional.avg_pool1d(x, kernel_size=scale, stride=scale)
            else:
                x_scaled = x
            
            # Extract features
            feat = transformer.extract_features(x_scaled)
            features.append(feat)
        
        # Concatenate features from all scales
        features = torch.cat(features, dim=1)
        
        # Fusion and classification
        out = self.fusion(features)
        
        return out
