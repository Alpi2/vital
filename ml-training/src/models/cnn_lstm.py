"""
CNN-LSTM Hybrid Model for ECG Classification
Combines spatial feature extraction (CNN) with temporal modeling (LSTM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CNNLSTM(nn.Module):
    """
    CNN-LSTM hybrid architecture for arrhythmia detection.
    
    Architecture:
    1. CNN layers for feature extraction
    2. LSTM layers for temporal modeling
    3. Fully connected layers for classification
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        input_channels: int = 1,
        cnn_channels: list = [64, 128, 256],
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.5,
    ):
        """
        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels (1 for single-lead, 12 for 12-lead)
            cnn_channels: List of CNN channel sizes
            lstm_hidden: LSTM hidden size
            lstm_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(CNNLSTM, self).__init__()
        
        self.num_classes = num_classes
        
        # CNN feature extractor
        self.conv_layers = nn.ModuleList()
        in_channels = input_channels
        
        for out_channels in cnn_channels:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(kernel_size=2),
                    nn.Dropout(dropout)
                )
            )
            in_channels = out_channels
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 256),  # *2 for bidirectional
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, channels, length)
            
        Returns:
            Output logits (batch_size, num_classes)
        """
        # CNN feature extraction
        for conv in self.conv_layers:
            x = conv(x)
        
        # Reshape for LSTM (batch_size, seq_len, features)
        x = x.permute(0, 2, 1)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        # Concatenate forward and backward hidden states
        h_n = torch.cat([h_n[-2], h_n[-1]], dim=1)
        
        # Classification
        out = self.fc(h_n)
        
        return out
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before classification layer.
        
        Args:
            x: Input tensor
            
        Returns:
            Feature vector
        """
        # CNN
        for conv in self.conv_layers:
            x = conv(x)
        
        # LSTM
        x = x.permute(0, 2, 1)
        lstm_out, (h_n, c_n) = self.lstm(x)
        h_n = torch.cat([h_n[-2], h_n[-1]], dim=1)
        
        return h_n


class AttentionCNNLSTM(CNNLSTM):
    """
    CNN-LSTM with attention mechanism.
    Allows model to focus on important temporal regions.
    """
    
    def __init__(self, *args, **kwargs):
        super(AttentionCNNLSTM, self).__init__(*args, **kwargs)
        
        # Attention layer
        lstm_hidden = kwargs.get('lstm_hidden', 128)
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN
        for conv in self.conv_layers:
            x = conv(x)
        
        # LSTM
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        
        # Attention weights
        attention_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Weighted sum
        context = torch.sum(lstm_out * attention_weights, dim=1)  # (batch, hidden*2)
        
        # Classification
        out = self.fc(context)
        
        return out
