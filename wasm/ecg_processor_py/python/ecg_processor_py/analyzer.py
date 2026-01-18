"""
ECG Analyzer with Rust FFI Integration
High-performance ECG processing with robust fallback and error handling
"""

import numpy as np
import asyncio
import time
import logging
import threading
from typing import Dict, List, Optional, Union, Any
from functools import lru_cache
from dataclasses import dataclass

from .config import ECGAnalysisConfig
from . import FFI_AVAILABLE, FFI_VERSION

# Setup logging
logger = logging.getLogger(__name__)

# Try to import Rust FFI module
if FFI_AVAILABLE:
    try:
        import ecg_processor_py as _rust_ffi
        RUST_AVAILABLE = True
        logger.info("✅ Rust FFI backend available")
    except Exception as e:
        RUST_AVAILABLE = False
        logger.error(f"💥 Failed to initialize Rust FFI: {e}")
else:
    RUST_AVAILABLE = False
    logger.warning("⚠️ Rust FFI module not available")


@dataclass
class ECGAnalysisResult:
    """ECG analysis result with comprehensive metrics"""
    
    heart_rate: float
    rr_intervals: List[float]
    qrs_peaks: List[int]
    hrv_metrics: Dict[str, float]
    anomalies: List[str]
    signal_quality: float
    processing_time_ms: float
    algorithm_version: str
    processing_backend: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "heart_rate": self.heart_rate,
            "rr_intervals": self.rr_intervals,
            "qrs_peaks": self.qrs_peaks,
            "hrv_metrics": self.hrv_metrics,
            "anomalies": self.anomalies,
            "signal_quality": self.signal_quality,
            "processing_time_ms": self.processing_time_ms,
            "algorithm_version": self.algorithm_version,
            "processing_backend": self.processing_backend,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        import json
        return json.dumps(self.to_dict(), indent=2)
    
    def summary(self) -> str:
        """Get summary string"""
        return (
            f"ECG Analysis: HR={self.heart_rate:.1f} BPM, "
            f"Quality={self.signal_quality:.2f}, "
            f"Time={self.processing_time_ms:.2f}ms, "
            f"Backend={self.processing_backend}"
        )


class ECGAnalyzerFFI:
    """
    ECG Analyzer with Rust FFI backend and Python fallback
    
    Features:
    - High-performance Rust backend (<5ms latency)
    - Robust Python fallback
    - Zero-copy NumPy integration
    - Async support
    - Batch processing
    - Memory leak protection
    - Thread safety
    """
    
    def __init__(self, use_ffi: bool = True, config: Optional[ECGAnalysisConfig] = None):
        """
        Initialize ECG analyzer
        
        Args:
            use_ffi: Whether to use Rust FFI backend
            config: Default analysis configuration
        """
        self.use_ffi = use_ffi and RUST_AVAILABLE
        self.default_config = config or ECGAnalysisConfig()
        
        # Performance monitoring
        self._stats = {
            "total_analyses": 0,
            "ffi_analyses": 0,
            "python_analyses": 0,
            "total_time_ms": 0.0,
            "errors": 0,
        }
        self._stats_lock = threading.Lock()
        
        if self.use_ffi:
            logger.info("✅ Using Rust FFI for ECG processing")
            # Test connection
            try:
                test_result = _rust_ffi.test_connection()
                logger.info(f"✅ Rust FFI test: {test_result}")
            except Exception as e:
                logger.error(f"💥 Rust FFI test failed: {e}")
                self.use_ffi = False
        else:
            logger.warning("⚠️ Using Python fallback for ECG processing")
    
    def _update_stats(self, backend: str, processing_time: float, error: bool = False):
        """Update performance statistics"""
        with self._stats_lock:
            self._stats["total_analyses"] += 1
            self._stats["total_time_ms"] += processing_time
            if error:
                self._stats["errors"] += 1
            elif backend == "rust_ffi":
                self._stats["ffi_analyses"] += 1
            else:
                self._stats["python_analyses"] += 1
    
    def _prepare_signal(self, signal: Union[np.ndarray, List[float]]) -> np.ndarray:
        """
        Prepare signal for analysis
        
        Args:
            signal: Input signal as NumPy array or list
            
        Returns:
            Prepared NumPy array (contiguous, float64)
        """
        # Convert to NumPy array if needed
        if not isinstance(signal, np.ndarray):
            signal = np.array(signal, dtype=np.float64)
        
        # Ensure correct dtype
        if signal.dtype != np.float64:
            signal = signal.astype(np.float64)
        
        # Ensure contiguous memory layout
        if not signal.flags['C_CONTIGUOUS']:
            signal = np.ascontiguousarray(signal)
        
        # Validate signal
        if len(signal) < 100:
            raise ValueError(f"Signal too short: {len(signal)} samples (minimum: 100)")
        
        if len(signal) > 100000:
            raise ValueError(f"Signal too long: {len(signal)} samples (maximum: 100000)")
        
        # Check for invalid values
        if np.any(~np.isfinite(signal)):
            raise ValueError("Signal contains invalid values (NaN or Inf)")
        
        return signal
    
    def _validate_config(self, config: Optional[ECGAnalysisConfig]) -> ECGAnalysisConfig:
        """Validate and merge configuration"""
        if config is None:
            return self.default_config
        
        if not isinstance(config, ECGAnalysisConfig):
            raise TypeError("config must be ECGAnalysisConfig instance")
        
        return config
    
    def analyze(
        self,
        signal: Union[np.ndarray, List[float]],
        config: Optional[ECGAnalysisConfig] = None,
    ) -> ECGAnalysisResult:
        """
        Analyze ECG signal (synchronous)
        
        Args:
            signal: ECG signal as NumPy array or list
            config: Analysis configuration
            
        Returns:
            ECG analysis result
        """
        config = self._validate_config(config)
        signal = self._prepare_signal(signal)
        start_time = time.time()
        
        try:
            if self.use_ffi:
                return self._analyze_with_ffi(signal, config)
            else:
                return self._analyze_with_python(signal, config)
        except Exception as e:
            logger.error(f"💥 Analysis failed: {e}")
            self._update_stats("error", (time.time() - start_time) * 1000, error=True)
            
            # Try fallback if FFI failed
            if self.use_ffi:
                logger.info("🔄 Attempting Python fallback...")
                try:
                    result = self._analyze_with_python(signal, config)
                    result.processing_backend += "_fallback"
                    return result
                except Exception as fallback_error:
                    logger.error(f"💥 Python fallback also failed: {fallback_error}")
            
            raise RuntimeError(f"ECG analysis failed: {e}")
    
    def _analyze_with_ffi(
        self,
        signal: np.ndarray,
        config: ECGAnalysisConfig,
    ) -> ECGAnalysisResult:
        """Analyze using Rust FFI (fast path)"""
        try:
            # Create Rust config
            rust_config = _rust_ffi.ECGAnalysisConfig(
                sampling_rate=config.sampling_rate,
                enable_hrv=config.enable_hrv,
                enable_anomaly_detection=config.enable_anomaly_detection,
                min_heart_rate=config.min_heart_rate,
                max_heart_rate=config.max_heart_rate,
                noise_threshold=config.noise_threshold,
            )
            
            # Call Rust function (zero-copy, GIL released automatically)
            rust_result = _rust_ffi.analyze_ecg_with_config(signal, rust_config)
            
            # Convert to Python result
            result = ECGAnalysisResult(
                heart_rate=rust_result.heart_rate,
                rr_intervals=list(rust_result.rr_intervals),
                qrs_peaks=list(rust_result.qrs_peaks),
                hrv_metrics=dict(rust_result.hrv_metrics),
                anomalies=list(rust_result.anomalies),
                signal_quality=rust_result.signal_quality,
                processing_time_ms=rust_result.processing_time_ms,
                algorithm_version=rust_result.algorithm_version,
                processing_backend=rust_result.processing_backend,
            )
            
            self._update_stats("rust_ffi", result.processing_time_ms)
            return result
            
        except Exception as e:
            logger.error(f"💥 FFI analysis failed: {e}")
            raise
    
    def _analyze_with_python(
        self,
        signal: np.ndarray,
        config: ECGAnalysisConfig,
    ) -> ECGAnalysisResult:
        """Fallback Python implementation"""
        start_time = time.time()
        
        try:
            # Simple Python implementation (placeholder)
            # In production, this would use scipy or similar
            heart_rate = self._estimate_heart_rate_python(signal, config.sampling_rate)
            qrs_peaks = self._detect_qrs_peaks_python(signal)
            rr_intervals = self._calculate_rr_intervals_python(qrs_peaks, config.sampling_rate)
            hrv_metrics = self._calculate_hrv_metrics_python(rr_intervals) if config.enable_hrv else {}
            signal_quality = self._estimate_signal_quality_python(signal)
            anomalies = self._detect_anomalies_python(heart_rate, signal_quality, config)
            
            processing_time = (time.time() - start_time) * 1000.0
            
            result = ECGAnalysisResult(
                heart_rate=heart_rate,
                rr_intervals=rr_intervals,
                qrs_peaks=qrs_peaks,
                hrv_metrics=hrv_metrics,
                anomalies=anomalies,
                signal_quality=signal_quality,
                processing_time_ms=processing_time,
                algorithm_version="1.0.0-python",
                processing_backend="python_fallback",
            )
            
            self._update_stats("python", processing_time)
            return result
            
        except Exception as e:
            logger.error(f"💥 Python analysis failed: {e}")
            raise
    
    async def analyze_async(
        self,
        signal: Union[np.ndarray, List[float]],
        config: Optional[ECGAnalysisConfig] = None,
    ) -> ECGAnalysisResult:
        """
        Analyze ECG signal (asynchronous)
        
        Args:
            signal: ECG signal as NumPy array or list
            config: Analysis configuration
            
        Returns:
            ECG analysis result
        """
        config = self._validate_config(config)
        signal = self._prepare_signal(signal)
        
        if self.use_ffi:
            return await self._analyze_async_with_ffi(signal, config)
        else:
            # Run Python version in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, self._analyze_with_python, signal, config
            )
    
    async def _analyze_async_with_ffi(
        self,
        signal: np.ndarray,
        config: ECGAnalysisConfig,
    ) -> ECGAnalysisResult:
        """Analyze using Rust FFI (async)"""
        try:
            # Create Rust config
            rust_config = _rust_ffi.ECGAnalysisConfig(
                sampling_rate=config.sampling_rate,
                enable_hrv=config.enable_hrv,
                enable_anomaly_detection=config.enable_anomaly_detection,
                min_heart_rate=config.min_heart_rate,
                max_heart_rate=config.max_heart_rate,
                noise_threshold=config.noise_threshold,
            )
            
            # Call async Rust function
            rust_result = await _rust_ffi.analyze_ecg_async(signal, config.sampling_rate)
            
            # Convert to Python result
            result = ECGAnalysisResult(
                heart_rate=rust_result.heart_rate,
                rr_intervals=list(rust_result.rr_intervals),
                qrs_peaks=list(rust_result.qrs_peaks),
                hrv_metrics=dict(rust_result.hrv_metrics),
                anomalies=list(rust_result.anomalies),
                signal_quality=rust_result.signal_quality,
                processing_time_ms=rust_result.processing_time_ms,
                algorithm_version=rust_result.algorithm_version,
                processing_backend=rust_result.processing_backend,
            )
            
            self._update_stats("rust_ffi", result.processing_time_ms)
            return result
            
        except Exception as e:
            logger.error(f"💥 Async FFI analysis failed: {e}")
            raise
    
    def analyze_batch(
        self,
        signals: List[Union[np.ndarray, List[float]]],
        config: Optional[ECGAnalysisConfig] = None,
    ) -> List[ECGAnalysisResult]:
        """
        Batch ECG analysis with parallel processing
        
        Args:
            signals: List of ECG signals
            config: Analysis configuration
            
        Returns:
            List of analysis results
        """
        if not signals:
            return []
        
        config = self._validate_config(config)
        prepared_signals = [self._prepare_signal(s) for s in signals]
        
        if self.use_ffi:
            return self._analyze_batch_with_ffi(prepared_signals, config)
        else:
            return [self._analyze_with_python(s, config) for s in prepared_signals]
    
    def _analyze_batch_with_ffi(
        self,
        signals: List[np.ndarray],
        config: ECGAnalysisConfig,
    ) -> List[ECGAnalysisResult]:
        """Batch analysis using Rust FFI"""
        try:
            # Create Rust config
            rust_config = _rust_ffi.ECGAnalysisConfig(
                sampling_rate=config.sampling_rate,
                enable_hrv=config.enable_hrv,
                enable_anomaly_detection=config.enable_anomaly_detection,
                min_heart_rate=config.min_heart_rate,
                max_heart_rate=config.max_heart_rate,
                noise_threshold=config.noise_threshold,
            )
            
            # Call Rust batch function
            rust_results = _rust_ffi.analyze_ecg_batch(signals, config.sampling_rate)
            
            # Convert to Python results
            results = []
            for rust_result in rust_results:
                result = ECGAnalysisResult(
                    heart_rate=rust_result.heart_rate,
                    rr_intervals=list(rust_result.rr_intervals),
                    qrs_peaks=list(rust_result.qrs_peaks),
                    hrv_metrics=dict(rust_result.hrv_metrics),
                    anomalies=list(rust_result.anomalies),
                    signal_quality=rust_result.signal_quality,
                    processing_time_ms=rust_result.processing_time_ms,
                    algorithm_version=rust_result.algorithm_version,
                    processing_backend=rust_result.processing_backend,
                )
                results.append(result)
            
            # Update stats (average time)
            if results:
                avg_time = sum(r.processing_time_ms for r in results) / len(results)
                self._update_stats("rust_ffi", avg_time)
            
            return results
            
        except Exception as e:
            logger.error(f"💥 Batch FFI analysis failed: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        with self._stats_lock:
            stats = self._stats.copy()
        
        if stats["total_analyses"] > 0:
            stats["average_time_ms"] = stats["total_time_ms"] / stats["total_analyses"]
            stats["error_rate"] = stats["errors"] / stats["total_analyses"]
            stats["ffi_usage_rate"] = stats["ffi_analyses"] / stats["total_analyses"]
        else:
            stats["average_time_ms"] = 0.0
            stats["error_rate"] = 0.0
            stats["ffi_usage_rate"] = 0.0
        
        return stats
    
    def reset_stats(self):
        """Reset performance statistics"""
        with self._stats_lock:
            self._stats = {
                "total_analyses": 0,
                "ffi_analyses": 0,
                "python_analyses": 0,
                "total_time_ms": 0.0,
                "errors": 0,
            }
    
    # Python fallback implementations (simplified)
    def _estimate_heart_rate_python(self, signal: np.ndarray, sampling_rate: float) -> float:
        """Simple heart rate estimation"""
        # Find peaks using simple threshold
        threshold = np.mean(np.abs(signal)) + 2 * np.std(signal)
        peaks = np.where(signal > threshold)[0]
        
        if len(peaks) < 2:
            return 72.0  # Default
        
        # Calculate RR intervals
        rr_intervals = np.diff(peaks) / sampling_rate
        avg_rr = np.mean(rr_intervals)
        
        return 60.0 / avg_rr if avg_rr > 0 else 72.0
    
    def _detect_qrs_peaks_python(self, signal: np.ndarray) -> List[int]:
        """Simple QRS peak detection"""
        threshold = np.mean(np.abs(signal)) + 2 * np.std(signal)
        peaks = []
        
        for i in range(1, len(signal) - 1):
            if (signal[i] > threshold and 
                signal[i] > signal[i-1] and 
                signal[i] > signal[i+1]):
                peaks.append(i)
        
        return peaks
    
    def _calculate_rr_intervals_python(
        self, qrs_peaks: List[int], sampling_rate: float
    ) -> List[float]:
        """Calculate RR intervals"""
        if len(qrs_peaks) < 2:
            return []
        
        return [(qrs_peaks[i+1] - qrs_peaks[i]) / sampling_rate * 1000.0 
                for i in range(len(qrs_peaks) - 1)]
    
    def _calculate_hrv_metrics_python(self, rr_intervals: List[float]) -> Dict[str, float]:
        """Calculate HRV metrics"""
        if len(rr_intervals) < 2:
            return {}
        
        rr_array = np.array(rr_intervals)
        
        metrics = {
            "mean_rr": float(np.mean(rr_array)),
            "sdnn": float(np.std(rr_array)),
            "rmssd": float(np.sqrt(np.mean(np.diff(rr_array) ** 2))),
        }
        
        return metrics
    
    def _estimate_signal_quality_python(self, signal: np.ndarray) -> float:
        """Estimate signal quality"""
        # Simple quality metric based on noise level
        signal_power = np.mean(signal ** 2)
        noise_power = np.var(np.diff(signal))
        
        if signal_power == 0:
            return 0.0
        
        snr = signal_power / noise_power
        quality = min(1.0, snr / 10.0)  # Normalize to 0-1
        
        return max(0.0, quality)
    
    def _detect_anomalies_python(
        self, heart_rate: float, signal_quality: float, config: ECGAnalysisConfig
    ) -> List[str]:
        """Detect anomalies"""
        anomalies = []
        
        if heart_rate < config.min_heart_rate:
            anomalies.append(f"Low heart rate: {heart_rate:.1f} BPM")
        
        if heart_rate > config.max_heart_rate:
            anomalies.append(f"High heart rate: {heart_rate:.1f} BPM")
        
        if signal_quality < config.noise_threshold:
            anomalies.append(f"Poor signal quality: {signal_quality:.2f}")
        
        return anomalies


# Performance monitoring decorator
def monitor_performance(func):
    """Decorator to monitor function performance"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            duration = (time.time() - start_time) * 1000.0
            logger.debug(f"{func.__name__} completed in {duration:.2f}ms")
    
    return wrapper


# Thread-safe singleton for global analyzer
_global_analyzer = None
_analyzer_lock = threading.Lock()


def get_global_analyzer() -> ECGAnalyzerFFI:
    """Get global ECG analyzer instance"""
    global _global_analyzer
    
    if _global_analyzer is None:
        with _analyzer_lock:
            if _global_analyzer is None:
                _global_analyzer = ECGAnalyzerFFI()
    
    return _global_analyzer


# Convenience functions
def analyze_ecg(signal: Union[np.ndarray, List[float]], 
               config: Optional[ECGAnalysisConfig] = None) -> ECGAnalysisResult:
    """Convenience function for ECG analysis"""
    analyzer = get_global_analyzer()
    return analyzer.analyze(signal, config)


async def analyze_ecg_async(signal: Union[np.ndarray, List[float]], 
                         config: Optional[ECGAnalysisConfig] = None) -> ECGAnalysisResult:
    """Convenience function for async ECG analysis"""
    analyzer = get_global_analyzer()
    return await analyzer.analyze_async(signal, config)
