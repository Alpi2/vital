"""
MIT-BIH Arrhythmia Database Loader

Provides efficient data loading, preprocessing, and augmentation
for the MIT-BIH Arrhythmia Database with proper train/val/test splits.
"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import pandas as pd
from pathlib import Path
import wfdb
from typing import Tuple, Optional, List, Dict, Any
import logging
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import random

logger = logging.getLogger(__name__)

class MITBIHDataset(Dataset):
    """
    MIT-BIH Arrhythmia Dataset for ECG signal classification.
    
    Provides:
    - Raw ECG signals with annotations
    - Preprocessing and normalization
    - Data augmentation for training
    - Class imbalance handling
    - Multiple lead support
    """
    
    # Class mapping for MIT-BIH annotations
    CLASS_MAPPING = {
        'N': 0,  # Normal
        'L': 1,  # Left bundle branch block
        'R': 2,  # Right bundle branch block
        'A': 3,  # Atrial premature beat
        'a': 3,  # Aberrated atrial premature beat
        'J': 3,  # Nodal (junctional) premature beat
        'S': 3,  # Supraventricular premature beat
        'V': 4,  # Premature ventricular contraction
        'E': 4,  # Ventricular escape beat
        'F': 5,  # Fusion of ventricular and normal beat
        '/': 5,  # Paced beat
        'f': 5,  # Fusion of paced and normal beat
        'Q': 5,  # Unclassifiable beat
        '?': 5,  # Unknown beat
    }
    
    # Simplified class mapping for 5-class classification
    SIMPLIFIED_MAPPING = {
        0: 0,  # Normal
        1: 1,  # Bundle branch block (L, R combined)
        2: 2,  # Supraventricular ectopic (A, a, J, S combined)
        3: 3,  # Ventricular ectopic (V, E combined)
        4: 4,  # Other (F, /, f, Q, ? combined)
    }
    
    CLASS_NAMES = ['Normal', 'Bundle Branch Block', 'Supraventricular Ectopic', 'Ventricular Ectopic', 'Other']
    
    def __init__(
        self,
        data_path: str,
        records: Optional[List[str]] = None,
        segment_length: int = 1000,
        overlap: float = 0.5,
        sampling_rate: int = 360,
        normalize: bool = True,
        scaler_type: str = 'standard',
        augmentation: bool = True,
        class_mapping: str = 'simplified',  # 'full' or 'simplified'
        cache_data: bool = True
    ):
        """
        Initialize MIT-BIH Dataset.
        
        Args:
            data_path: Path to MIT-BIH database
            records: List of record names (None for all records)
            segment_length: Length of each ECG segment
            overlap: Overlap between segments (0.0 to 1.0)
            sampling_rate: Target sampling rate
            normalize: Whether to normalize signals
            scaler_type: Type of scaler ('standard', 'robust', 'minmax')
            augmentation: Whether to apply data augmentation
            class_mapping: Class mapping strategy
            cache_data: Whether to cache processed data
        """
        self.data_path = Path(data_path)
        self.records = records or self._get_all_records()
        self.segment_length = segment_length
        self.overlap = overlap
        self.sampling_rate = sampling_rate
        self.normalize = normalize
        self.scaler_type = scaler_type
        self.augmentation = augmentation
        self.class_mapping = class_mapping
        self.cache_data = cache_data
        
        # Data cache
        self._data_cache = {}
        self._scaler = None
        
        # Load and process data
        self._load_data()
        
        logger.info(f"MIT-BIH Dataset initialized with {len(self.records)} records")
        logger.info(f"Total samples: {len(self)}")
    
    def _get_all_records(self) -> List[str]:
        """Get all available MIT-BIH records."""
        # Common records for training
        return [
            '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
            '111', '112', '113', '114', '115', '116', '117', '118', '119',
            '121', '122', '123', '124', '200', '201', '202', '203', '205',
            '207', '208', '209', '210', '212', '213', '214', '215', '217',
            '219', '220', '221', '222', '223', '228', '230', '231', '232',
            '233', '234'
        ]
    
    def _load_data(self):
        """Load and process MIT-BIH data."""
        all_segments = []
        all_labels = []
        all_metadata = []
        
        for record_name in self.records:
            try:
                segments, labels, metadata = self._load_record(record_name)
                all_segments.extend(segments)
                all_labels.extend(labels)
                all_metadata.extend(metadata)
                
                logger.info(f"Loaded {len(segments)} segments from record {record_name}")
                
            except Exception as e:
                logger.error(f"Failed to load record {record_name}: {e}")
                continue
        
        # Convert to numpy arrays
        self.segments = np.array(all_segments, dtype=np.float32)
        self.labels = np.array(all_labels, dtype=np.int64)
        self.metadata = all_metadata
        
        # Apply normalization
        if self.normalize:
            self._fit_scaler()
            self.segments = self._apply_normalization(self.segments)
        
        # Calculate class weights for imbalanced dataset
        self.class_weights = self._calculate_class_weights()
        
        logger.info(f"Dataset loaded: {len(self.segments)} samples")
        logger.info(f"Class distribution: {np.bincount(self.labels)}")
    
    def _load_record(self, record_name: str) -> Tuple[List[np.ndarray], List[int], List[Dict]]:
        """Load a single MIT-BIH record."""
        # Download data if not exists
        try:
            record = wfdb.rdrecord(str(self.data_path / record_name), pb_dir='atr')
            annotation = wfdb.rdann(str(self.data_path / record_name), 'atr')
        except Exception as e:
            logger.error(f"Failed to load record {record_name}: {e}")
            return [], [], []
        
        # Resample if necessary
        if record.fs != self.sampling_rate:
            # Simple resampling (would use scipy in production)
            resample_factor = self.sampling_rate / record.fs
            record.p_signal = self._resample_signal(record.p_signal[0], resample_factor)
            record.fs = self.sampling_rate
        
        # Extract segments
        signal = record.p_signal
        annotation_sample = annotation.sample
        annotation_symbol = annotation.symbol
        
        segments = []
        labels = []
        metadata = []
        
        # Create segments around each annotation
        for i, (sample_idx, symbol) in enumerate(zip(annotation_sample, annotation_symbol)):
            if symbol not in self.CLASS_MAPPING:
                continue
            
            # Get class label
            if self.class_mapping == 'full':
                class_label = self.CLASS_MAPPING[symbol]
            else:  # simplified
                class_label = self.SIMPLIFIED_MAPPING.get(self.CLASS_MAPPING[symbol], 5)
            
            # Extract segment around annotation
            start_idx = max(0, sample_idx - self.segment_length // 2)
            end_idx = min(len(signal), start_idx + self.segment_length)
            
            if end_idx - start_idx < self.segment_length:
                # Pad segment if too short
                segment = np.zeros(self.segment_length)
                segment[:end_idx - start_idx] = signal[start_idx:end_idx]
            else:
                segment = signal[start_idx:end_idx]
            
            # Apply preprocessing
            segment = self._preprocess_segment(segment)
            
            segments.append(segment)
            labels.append(class_label)
            metadata.append({
                'record_name': record_name,
                'annotation_idx': i,
                'sample_idx': sample_idx,
                'symbol': symbol,
                'start_idx': start_idx,
                'end_idx': end_idx
            })
        
        return segments, labels, metadata
    
    def _resample_signal(self, signal: np.ndarray, factor: float) -> np.ndarray:
        """Simple resampling (placeholder - would use scipy in production)."""
        # This is a simplified resampling
        # In production, use scipy.signal.resample
        from scipy import signal
        return signal.resample(signal, int(len(signal) * factor))
    
    def _preprocess_segment(self, segment: np.ndarray) -> np.ndarray:
        """Preprocess ECG segment."""
        # Remove baseline wander
        segment = self._remove_baseline_wander(segment)
        
        # Apply bandpass filter
        segment = self._bandpass_filter(segment)
        
        # Normalize amplitude
        segment = self._normalize_amplitude(segment)
        
        return segment
    
    def _remove_baseline_wander(self, signal: np.ndarray) -> np.ndarray:
        """Remove baseline wander using median filter."""
        from scipy import signal
        
        # Median filter for baseline estimation
        kernel_size = int(0.2 * self.sampling_rate)  # 200ms window
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        baseline = signal.medfilt(signal, kernel_size=kernel_size)
        return signal - baseline
    
    def _bandpass_filter(self, signal: np.ndarray) -> np.ndarray:
        """Apply bandpass filter (0.5-40 Hz)."""
        from scipy import signal
        
        # Design Butterworth bandpass filter
        nyquist = self.sampling_rate / 2
        low = 0.5 / nyquist
        high = 40.0 / nyquist
        
        b, a = signal.butter(4, [low, high], btype='band')
        
        # Apply filter
        return signal.filtfilt(b, a, signal)
    
    def _normalize_amplitude(self, signal: np.ndarray) -> np.ndarray:
        """Normalize signal amplitude."""
        # Z-score normalization
        return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    
    def _fit_scaler(self):
        """Fit normalization scaler."""
        if self.scaler_type == 'standard':
            self._scaler = StandardScaler()
        elif self.scaler_type == 'robust':
            self._scaler = RobustScaler()
        elif self.scaler_type == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            self._scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")
        
        # Fit on all data
        self._scaler.fit(self.segments.reshape(-1, 1))
    
    def _apply_normalization(self, data: np.ndarray) -> np.ndarray:
        """Apply normalization to data."""
        if self._scaler is None:
            return data
        
        original_shape = data.shape
        data_flat = data.reshape(-1, 1)
        data_normalized = self._scaler.transform(data_flat)
        return data_normalized.reshape(original_shape)
    
    def _calculate_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced dataset."""
        if len(self.labels) == 0:
            return torch.ones(5)
        
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        total_samples = len(self.labels)
        
        # Calculate inverse frequency weights
        class_weights = np.zeros(5)
        for label, count in zip(unique_labels, counts):
            if label < 5:  # Only for valid classes
                weight = total_samples / (len(unique_labels) * count)
                class_weights[label] = weight
        
        return torch.FloatTensor(class_weights)
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.segments)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item from dataset."""
        segment = self.segments[idx]
        label = self.labels[idx]
        
        # Apply augmentation if training
        if self.augmentation and self.training:
            segment = self._apply_augmentation(segment)
        
        # Convert to tensors
        segment_tensor = torch.FloatTensor(segment).unsqueeze(0)  # Add channel dimension
        label_tensor = torch.LongTensor([label])
        
        return segment_tensor, label_tensor
    
    def _apply_augmentation(self, segment: np.ndarray) -> np.ndarray:
        """Apply data augmentation to ECG segment."""
        # Random augmentation
        if random.random() < 0.3:
            # Add Gaussian noise
            noise = np.random.normal(0, 0.01, segment.shape)
            segment = segment + noise
        
        if random.random() < 0.2:
            # Random scaling
            scale = np.random.uniform(0.8, 1.2)
            segment = segment * scale
        
        if random.random() < 0.2:
            # Random time shift
            shift = np.random.randint(-50, 50)
            segment = np.roll(segment, shift)
        
        if random.random() < 0.1:
            # Random amplitude inversion
            segment = -segment
        
        return segment
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get class distribution."""
        if len(self.labels) == 0:
            return {}
        
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        distribution = {}
        
        for label, count in zip(unique_labels, counts):
            if label < len(self.CLASS_NAMES):
                class_name = self.CLASS_NAMES[label]
                distribution[class_name] = int(count)
        
        return distribution
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        return {
            'total_samples': len(self.segments),
            'num_records': len(self.records),
            'segment_length': self.segment_length,
            'sampling_rate': self.sampling_rate,
            'class_distribution': self.get_class_distribution(),
            'class_names': self.CLASS_NAMES,
            'class_weights': self.class_weights.tolist() if hasattr(self.class_weights, 'tolist') else self.class_weights
        }

def create_data_loaders(
    data_path: str,
    batch_size: int = 32,
    test_size: float = 0.2,
    val_size: float = 0.1,
    num_workers: int = 4,
    random_seed: int = 42,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        data_path: Path to MIT-BIH database
        batch_size: Batch size for data loaders
        test_size: Test set proportion
        val_size: Validation set proportion
        num_workers: Number of worker processes
        random_seed: Random seed for reproducibility
        **dataset_kwargs: Additional dataset arguments
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Set random seeds
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # Create full dataset
    full_dataset = MITBIHDataset(data_path, **dataset_kwargs)
    
    # Get indices for splitting
    indices = list(range(len(full_dataset)))
    train_indices, test_indices = train_test_split(
        indices, test_size=test_size, random_state=random_seed, stratify=full_dataset.labels
    )
    train_indices, val_indices = train_test_split(
        train_indices, test_size=val_size/(1-test_size), random_state=random_seed, 
        stratify=[full_dataset.labels[i] for i in train_indices]
    )
    
    # Create datasets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
    # Create weighted sampler for training (handles class imbalance)
    train_labels = [full_dataset.labels[i] for i in train_indices]
    class_sample_counts = np.bincount(train_labels, minlength=5)
    class_weights = 1.0 / class_sample_counts
    sample_weights = class_weights[train_labels]
    
    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def download_mitbih_data(data_path: str):
    """Download MIT-BIH database if not exists."""
    data_path = Path(data_path)
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Check if data already exists
    if any(data_path.glob('*.dat')):
        logger.info(f"MIT-BIH data already exists in {data_path}")
        return
    
    logger.info("Downloading MIT-BIH database...")
    
    # Download records (would use wfdb in production)
    records = MITBIHDataset._get_all_records()
    
    for record_name in records:
        try:
            wfdb.dl_database(record_name, dl_dir=str(data_path))
            logger.info(f"Downloaded record {record_name}")
        except Exception as e:
            logger.error(f"Failed to download {record_name}: {e}")

if __name__ == "__main__":
    # Test dataset
    data_path = "mitbih_data"
    
    # Download data (commented out for testing)
    # download_mitbih_data(data_path)
    
    # Create dataset
    dataset = MITBIHDataset(
        data_path=data_path,
        records=['100', '101', '102'],  # Test with few records
        segment_length=1000,
        augmentation=True
    )
    
    print("Dataset Information:")
    info = dataset.get_data_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test data loading
    print(f"\nTesting data loading:")
    for i in range(min(5, len(dataset))):
        segment, label = dataset[i]
        print(f"  Sample {i}: shape={segment.shape}, label={label.item()}, class={dataset.CLASS_NAMES[label.item()]}")
    
    # Test data loaders
    print(f"\nTesting data loaders:")
    try:
        train_loader, val_loader, test_loader = create_data_loaders(
            data_path=data_path,
            batch_size=4,
            records=['100', '101', '102']
        )
        
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        
        # Test one batch
        for batch_segments, batch_labels in train_loader:
            print(f"  Batch shape: {batch_segments.shape}")
            print(f"  Labels: {batch_labels}")
            break
            
    except Exception as e:
        print(f"  Error creating data loaders: {e}")
