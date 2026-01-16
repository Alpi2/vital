"""
ECG Dataset Loader
Supports PhysioNet, MIT-BIH, PTB-XL databases
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import wfdb
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import pandas as pd


class ECGDataset(Dataset):
    """
    ECG Dataset for arrhythmia detection and classification.
    
    Supports multiple databases:
    - MIT-BIH Arrhythmia Database
    - PTB-XL Database
    - PhysioNet Challenge datasets
    """
    
    # Arrhythmia class mapping (AAMI EC57 standard)
    ARRHYTHMIA_CLASSES = {
        'N': 0,   # Normal beat
        'S': 1,   # Supraventricular ectopic beat
        'V': 2,   # Ventricular ectopic beat
        'F': 3,   # Fusion beat
        'Q': 4,   # Unknown beat
    }
    
    # Extended arrhythmia types (15+ types)
    EXTENDED_CLASSES = {
        'NORM': 0,   # Normal sinus rhythm
        'AFIB': 1,   # Atrial fibrillation
        'AFL': 2,    # Atrial flutter
        'SVTA': 3,   # Supraventricular tachycardia
        'VT': 4,     # Ventricular tachycardia
        'VFIB': 5,   # Ventricular fibrillation
        'PVC': 6,    # Premature ventricular contraction
        'PAC': 7,    # Premature atrial contraction
        'LBBB': 8,   # Left bundle branch block
        'RBBB': 9,   # Right bundle branch block
        'BRADY': 10, # Bradycardia
        'TACHY': 11, # Tachycardia
        'MI': 12,    # Myocardial infarction
        'STEMI': 13, # ST-elevation MI
        'NSTEMI': 14,# Non-ST-elevation MI
    }
    
    def __init__(
        self,
        data_dir: str,
        database: str = 'mitdb',
        split: str = 'train',
        window_size: int = 360,  # 1 second at 360 Hz
        transform=None,
        target_transform=None,
        use_extended_classes: bool = False,
    ):
        """
        Args:
            data_dir: Root directory containing ECG data
            database: Database name ('mitdb', 'ptbxl', 'physionet')
            split: 'train', 'val', or 'test'
            window_size: Number of samples per window
            transform: Optional transform to apply to signals
            target_transform: Optional transform to apply to labels
            use_extended_classes: Use extended arrhythmia classification
        """
        self.data_dir = Path(data_dir)
        self.database = database
        self.split = split
        self.window_size = window_size
        self.transform = transform
        self.target_transform = target_transform
        self.use_extended_classes = use_extended_classes
        
        self.classes = self.EXTENDED_CLASSES if use_extended_classes else self.ARRHYTHMIA_CLASSES
        self.num_classes = len(self.classes)
        
        # Load data
        self.records = self._load_records()
        self.samples = self._prepare_samples()
        
    def _load_records(self) -> List[str]:
        """Load record names from database"""
        if self.database == 'mitdb':
            return self._load_mitdb_records()
        elif self.database == 'ptbxl':
            return self._load_ptbxl_records()
        else:
            raise ValueError(f"Unsupported database: {self.database}")
    
    def _load_mitdb_records(self) -> List[str]:
        """Load MIT-BIH Arrhythmia Database records"""
        # MIT-BIH has 48 records (100-234)
        all_records = [
            '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
            '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
            '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
            '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
            '222', '223', '228', '230', '231', '232', '233', '234'
        ]
        
        # Standard train/val/test split (70/15/15)
        if self.split == 'train':
            return all_records[:34]  # 70%
        elif self.split == 'val':
            return all_records[34:41]  # 15%
        else:  # test
            return all_records[41:]  # 15%
    
    def _load_ptbxl_records(self) -> List[str]:
        """Load PTB-XL Database records"""
        # PTB-XL has predefined splits
        metadata_file = self.data_dir / 'ptbxl_database.csv'
        if not metadata_file.exists():
            raise FileNotFoundError(f"PTB-XL metadata not found: {metadata_file}")
        
        df = pd.read_csv(metadata_file)
        split_map = {'train': 1, 'val': 9, 'test': 10}
        records = df[df['strat_fold'] == split_map[self.split]]['filename_hr'].tolist()
        return records
    
    def _prepare_samples(self) -> List[Dict]:
        """Prepare windowed samples from records"""
        samples = []
        
        for record_name in self.records:
            try:
                # Read WFDB record
                record_path = self.data_dir / self.database / record_name
                record = wfdb.rdrecord(str(record_path.with_suffix('')))
                annotation = wfdb.rdann(str(record_path.with_suffix('')), 'atr')
                
                # Extract signal (use lead II for single-lead, or all leads for 12-lead)
                signal = record.p_signal[:, 0]  # Lead II
                fs = record.fs  # Sampling frequency
                
                # Create windows around each annotation
                for i, (sample_idx, symbol) in enumerate(zip(annotation.sample, annotation.symbol)):
                    # Map annotation to class
                    if self.use_extended_classes:
                        label = self._map_extended_annotation(symbol)
                    else:
                        label = self._map_aami_annotation(symbol)
                    
                    if label is None:
                        continue
                    
                    # Extract window
                    start = max(0, sample_idx - self.window_size // 2)
                    end = start + self.window_size
                    
                    if end > len(signal):
                        continue
                    
                    window = signal[start:end]
                    
                    samples.append({
                        'signal': window,
                        'label': label,
                        'record': record_name,
                        'sample_idx': sample_idx,
                        'fs': fs,
                    })
                    
            except Exception as e:
                print(f"Error loading record {record_name}: {e}")
                continue
        
        return samples
    
    def _map_aami_annotation(self, symbol: str) -> Optional[int]:
        """Map beat annotation to AAMI EC57 class"""
        # AAMI EC57 standard mapping
        aami_map = {
            'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',  # Normal
            'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',  # Supraventricular
            'V': 'V', 'E': 'V',  # Ventricular
            'F': 'F',  # Fusion
            '/': 'Q', 'f': 'Q', 'Q': 'Q',  # Unknown
        }
        
        mapped = aami_map.get(symbol)
        return self.ARRHYTHMIA_CLASSES.get(mapped) if mapped else None
    
    def _map_extended_annotation(self, symbol: str) -> Optional[int]:
        """Map to extended arrhythmia classes"""
        # This would require more sophisticated mapping based on context
        # For now, use basic mapping
        basic_map = {
            'N': 'NORM',
            'V': 'PVC',
            'A': 'PAC',
            'L': 'LBBB',
            'R': 'RBBB',
        }
        
        mapped = basic_map.get(symbol)
        return self.EXTENDED_CLASSES.get(mapped) if mapped else None
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        
        signal = sample['signal'].astype(np.float32)
        label = sample['label']
        
        # Normalize signal
        signal = (signal - signal.mean()) / (signal.std() + 1e-8)
        
        # Convert to tensor
        signal = torch.from_numpy(signal).unsqueeze(0)  # Add channel dimension
        
        # Apply transforms
        if self.transform:
            signal = self.transform(signal)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return signal, label
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced dataset"""
        labels = [s['label'] for s in self.samples]
        class_counts = np.bincount(labels, minlength=self.num_classes)
        
        # Inverse frequency weighting
        weights = 1.0 / (class_counts + 1e-8)
        weights = weights / weights.sum() * self.num_classes
        
        return torch.from_numpy(weights).float()


class MultiLeadECGDataset(ECGDataset):
    """12-lead ECG dataset for comprehensive analysis"""
    
    def __init__(self, *args, num_leads: int = 12, **kwargs):
        self.num_leads = num_leads
        super().__init__(*args, **kwargs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        
        # Load all leads
        record_path = self.data_dir / self.database / sample['record']
        record = wfdb.rdrecord(str(record_path.with_suffix('')))
        
        # Extract window for all leads
        start = max(0, sample['sample_idx'] - self.window_size // 2)
        end = start + self.window_size
        
        signals = record.p_signal[start:end, :self.num_leads].T  # (num_leads, window_size)
        
        # Normalize each lead
        for i in range(signals.shape[0]):
            signals[i] = (signals[i] - signals[i].mean()) / (signals[i].std() + 1e-8)
        
        signals = torch.from_numpy(signals.astype(np.float32))
        label = sample['label']
        
        if self.transform:
            signals = self.transform(signals)
        
        return signals, label
