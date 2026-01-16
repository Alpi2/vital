"""
ECG Signal Preprocessing
Implements signal quality assessment and filtering
"""

import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, iirnotch
import pywt
from typing import Tuple, Optional


class ECGPreprocessor:
    """
    ECG signal preprocessing pipeline.
    
    Implements:
    - Baseline wander removal
    - Powerline interference filtering (50/60 Hz)
    - High-frequency noise removal
    - Signal quality assessment
    """
    
    def __init__(
        self,
        fs: float = 360.0,
        powerline_freq: float = 60.0,
        lowcut: float = 0.5,
        highcut: float = 40.0,
    ):
        """
        Args:
            fs: Sampling frequency (Hz)
            powerline_freq: Powerline frequency (50 or 60 Hz)
            lowcut: High-pass filter cutoff (Hz)
            highcut: Low-pass filter cutoff (Hz)
        """
        self.fs = fs
        self.powerline_freq = powerline_freq
        self.lowcut = lowcut
        self.highcut = highcut
    
    def remove_baseline_wander(self, signal_data: np.ndarray) -> np.ndarray:
        """
        Remove baseline wander using wavelet decomposition.
        
        Args:
            signal_data: Input ECG signal
            
        Returns:
            Signal with baseline wander removed
        """
        # Wavelet decomposition
        coeffs = pywt.wavedec(signal_data, 'db4', level=9)
        
        # Remove approximation coefficients (low-frequency baseline)
        coeffs[0] = np.zeros_like(coeffs[0])
        
        # Reconstruct signal
        return pywt.waverec(coeffs, 'db4')
    
    def remove_powerline_interference(self, signal_data: np.ndarray) -> np.ndarray:
        """
        Remove powerline interference using notch filter.
        
        Args:
            signal_data: Input ECG signal
            
        Returns:
            Filtered signal
        """
        # Design notch filter
        Q = 30.0  # Quality factor
        b, a = iirnotch(self.powerline_freq, Q, self.fs)
        
        # Apply filter
        return filtfilt(b, a, signal_data)
    
    def bandpass_filter(self, signal_data: np.ndarray) -> np.ndarray:
        """
        Apply bandpass filter to remove high and low frequency noise.
        
        Args:
            signal_data: Input ECG signal
            
        Returns:
            Filtered signal
        """
        # Design Butterworth bandpass filter
        nyquist = 0.5 * self.fs
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        
        b, a = butter(4, [low, high], btype='band')
        
        # Apply zero-phase filter
        return filtfilt(b, a, signal_data)
    
    def calculate_sqi(self, signal_data: np.ndarray) -> float:
        """
        Calculate Signal Quality Index (SQI).
        
        Args:
            signal_data: Input ECG signal
            
        Returns:
            SQI score (0-1, higher is better)
        """
        # Multiple quality metrics
        
        # 1. Signal-to-noise ratio
        signal_power = np.mean(signal_data ** 2)
        noise_estimate = np.var(np.diff(signal_data))
        snr = 10 * np.log10(signal_power / (noise_estimate + 1e-10))
        snr_score = np.clip(snr / 20.0, 0, 1)  # Normalize to 0-1
        
        # 2. Kurtosis (should be close to 3 for good ECG)
        from scipy.stats import kurtosis
        kurt = kurtosis(signal_data)
        kurt_score = 1.0 - np.abs(kurt - 3.0) / 10.0
        kurt_score = np.clip(kurt_score, 0, 1)
        
        # 3. Baseline stability
        baseline_var = np.var(signal_data - signal.medfilt(signal_data, 71))
        baseline_score = 1.0 / (1.0 + baseline_var)
        
        # 4. Peak detection reliability
        peaks = self._detect_r_peaks(signal_data)
        if len(peaks) > 1:
            rr_intervals = np.diff(peaks)
            rr_std = np.std(rr_intervals)
            rr_mean = np.mean(rr_intervals)
            rr_score = 1.0 - np.clip(rr_std / rr_mean, 0, 1)
        else:
            rr_score = 0.0
        
        # Weighted combination
        sqi = 0.3 * snr_score + 0.2 * kurt_score + 0.2 * baseline_score + 0.3 * rr_score
        
        return float(sqi)
    
    def _detect_r_peaks(self, signal_data: np.ndarray) -> np.ndarray:
        """
        Simple R-peak detection using scipy.signal.find_peaks.
        
        Args:
            signal_data: Input ECG signal
            
        Returns:
            Array of R-peak indices
        """
        # Find peaks with minimum distance and height
        min_distance = int(0.6 * self.fs)  # Minimum 0.6s between peaks (100 bpm max)
        height_threshold = 0.5 * np.max(signal_data)
        
        peaks, _ = signal.find_peaks(
            signal_data,
            distance=min_distance,
            height=height_threshold
        )
        
        return peaks
    
    def detect_artifacts(self, signal_data: np.ndarray) -> Tuple[bool, str]:
        """
        Detect various artifacts in ECG signal.
        
        Args:
            signal_data: Input ECG signal
            
        Returns:
            (has_artifact, artifact_type)
        """
        # 1. Saturation detection
        max_val = np.max(np.abs(signal_data))
        if max_val > 0.95 * np.max(signal_data):  # Near saturation
            return True, "saturation"
        
        # 2. Flat line detection
        if np.std(signal_data) < 0.01:
            return True, "flat_line"
        
        # 3. Excessive noise
        high_freq_power = np.sum(np.abs(np.fft.fft(signal_data)[int(len(signal_data)/2):]))
        total_power = np.sum(np.abs(np.fft.fft(signal_data)))
        if high_freq_power / total_power > 0.5:
            return True, "excessive_noise"
        
        # 4. Motion artifact (sudden large changes)
        diff = np.abs(np.diff(signal_data))
        if np.max(diff) > 5 * np.median(diff):
            return True, "motion_artifact"
        
        return False, "none"
    
    def preprocess(self, signal_data: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """
        Complete preprocessing pipeline.
        
        Args:
            signal_data: Raw ECG signal
            
        Returns:
            (processed_signal, sqi_score, is_valid)
        """
        # Check for artifacts
        has_artifact, artifact_type = self.detect_artifacts(signal_data)
        
        if has_artifact:
            print(f"Warning: Artifact detected - {artifact_type}")
        
        # Apply filters
        signal_filtered = self.remove_baseline_wander(signal_data)
        signal_filtered = self.remove_powerline_interference(signal_filtered)
        signal_filtered = self.bandpass_filter(signal_filtered)
        
        # Calculate quality
        sqi = self.calculate_sqi(signal_filtered)
        
        # Determine if signal is valid (SQI > 0.7)
        is_valid = sqi > 0.7 and not has_artifact
        
        return signal_filtered, sqi, is_valid


class PanTompkinsDetector:
    """
    Pan-Tompkins algorithm for QRS detection.
    Industry standard for R-peak detection.
    """
    
    def __init__(self, fs: float = 360.0):
        self.fs = fs
    
    def detect_qrs(self, signal_data: np.ndarray) -> np.ndarray:
        """
        Detect QRS complexes using Pan-Tompkins algorithm.
        
        Args:
            signal_data: Preprocessed ECG signal
            
        Returns:
            Array of QRS peak indices
        """
        # 1. Bandpass filter (5-15 Hz)
        nyquist = 0.5 * self.fs
        b, a = butter(2, [5/nyquist, 15/nyquist], btype='band')
        filtered = filtfilt(b, a, signal_data)
        
        # 2. Derivative (emphasize QRS slope)
        derivative = np.diff(filtered)
        
        # 3. Squaring (emphasize higher frequencies)
        squared = derivative ** 2
        
        # 4. Moving window integration
        window_size = int(0.150 * self.fs)  # 150ms window
        integrated = np.convolve(squared, np.ones(window_size)/window_size, mode='same')
        
        # 5. Adaptive thresholding
        peaks = self._adaptive_threshold(integrated)
        
        return peaks
    
    def _adaptive_threshold(self, signal_data: np.ndarray) -> np.ndarray:
        """
        Adaptive thresholding for peak detection.
        """
        # Initial threshold
        threshold = 0.5 * np.max(signal_data)
        
        peaks, _ = signal.find_peaks(
            signal_data,
            height=threshold,
            distance=int(0.2 * self.fs)  # Minimum 200ms between peaks
        )
        
        return peaks
