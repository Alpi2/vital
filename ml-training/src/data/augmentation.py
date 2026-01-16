"""
ECG Data Augmentation
Techniques for improving model generalization
"""

import numpy as np
import torch
from typing import Callable, List
import random


class ECGAugmentation:
    """
    Data augmentation for ECG signals.
    
    Techniques:
    - Amplitude scaling
    - Time warping
    - Gaussian noise addition
    - Baseline wander simulation
    - Lead dropout (for multi-lead)
    """
    
    def __init__(self, p: float = 0.5):
        """
        Args:
            p: Probability of applying each augmentation
        """
        self.p = p
    
    def amplitude_scaling(self, signal: np.ndarray, scale_range: tuple = (0.8, 1.2)) -> np.ndarray:
        """
        Randomly scale signal amplitude.
        
        Args:
            signal: Input ECG signal
            scale_range: (min_scale, max_scale)
            
        Returns:
            Scaled signal
        """
        if random.random() < self.p:
            scale = random.uniform(*scale_range)
            return signal * scale
        return signal
    
    def add_gaussian_noise(self, signal: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
        """
        Add Gaussian noise to signal.
        
        Args:
            signal: Input ECG signal
            snr_db: Signal-to-noise ratio in dB
            
        Returns:
            Noisy signal
        """
        if random.random() < self.p:
            signal_power = np.mean(signal ** 2)
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
            return signal + noise
        return signal
    
    def time_warp(self, signal: np.ndarray, warp_range: tuple = (0.9, 1.1)) -> np.ndarray:
        """
        Apply time warping (stretch/compress).
        
        Args:
            signal: Input ECG signal
            warp_range: (min_warp, max_warp)
            
        Returns:
            Warped signal
        """
        if random.random() < self.p:
            warp_factor = random.uniform(*warp_range)
            new_length = int(len(signal) * warp_factor)
            
            # Resample
            from scipy.interpolate import interp1d
            x_old = np.linspace(0, 1, len(signal))
            x_new = np.linspace(0, 1, new_length)
            f = interp1d(x_old, signal, kind='cubic')
            warped = f(x_new)
            
            # Crop or pad to original length
            if len(warped) > len(signal):
                start = (len(warped) - len(signal)) // 2
                return warped[start:start + len(signal)]
            else:
                pad = len(signal) - len(warped)
                return np.pad(warped, (pad//2, pad - pad//2), mode='edge')
        return signal
    
    def add_baseline_wander(self, signal: np.ndarray, freq: float = 0.5, amplitude: float = 0.1) -> np.ndarray:
        """
        Simulate baseline wander.
        
        Args:
            signal: Input ECG signal
            freq: Wander frequency (Hz)
            amplitude: Wander amplitude (relative to signal std)
            
        Returns:
            Signal with baseline wander
        """
        if random.random() < self.p:
            t = np.arange(len(signal))
            wander = amplitude * np.std(signal) * np.sin(2 * np.pi * freq * t / 360)
            return signal + wander
        return signal
    
    def random_shift(self, signal: np.ndarray, max_shift: int = 20) -> np.ndarray:
        """
        Randomly shift signal in time.
        
        Args:
            signal: Input ECG signal
            max_shift: Maximum shift in samples
            
        Returns:
            Shifted signal
        """
        if random.random() < self.p:
            shift = random.randint(-max_shift, max_shift)
            return np.roll(signal, shift)
        return signal
    
    def lead_dropout(self, signal: np.ndarray, dropout_prob: float = 0.1) -> np.ndarray:
        """
        Randomly dropout leads (for multi-lead ECG).
        
        Args:
            signal: Input multi-lead ECG (num_leads, length)
            dropout_prob: Probability of dropping each lead
            
        Returns:
            Signal with dropped leads
        """
        if len(signal.shape) == 1:
            return signal  # Single lead, no dropout
        
        if random.random() < self.p:
            mask = np.random.random(signal.shape[0]) > dropout_prob
            signal_copy = signal.copy()
            signal_copy[~mask] = 0
            return signal_copy
        return signal
    
    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply random augmentations.
        
        Args:
            signal: Input ECG signal
            
        Returns:
            Augmented signal
        """
        # Apply augmentations in sequence
        signal = self.amplitude_scaling(signal)
        signal = self.add_gaussian_noise(signal)
        signal = self.time_warp(signal)
        signal = self.add_baseline_wander(signal)
        signal = self.random_shift(signal)
        signal = self.lead_dropout(signal)
        
        return signal


class Compose:
    """Compose multiple transforms"""
    
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms
    
    def __call__(self, signal):
        for t in self.transforms:
            signal = t(signal)
        return signal
