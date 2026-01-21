"""
REST API Endpoints for ECG Feature Extraction

FastAPI endpoints for feature extraction, analysis,
and feature management services.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np
import logging
from datetime import datetime

from ..services.feature_extraction_service import get_feature_service, FeatureExtractionService
from ..ml.features.feature_scaler import FeatureScaler

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/features", tags=["features"])

# Pydantic models
class ECGSignalRequest(BaseModel):
    """Request model for ECG signal analysis."""
    signal: List[float] = Field(..., description="ECG signal samples")
    sampling_rate: int = Field(250, description="Sampling rate in Hz")
    peaks: Optional[List[int]] = Field(None, description="R-peak indices (optional)")
    rr_intervals: Optional[List[float]] = Field(None, description="RR intervals in ms (optional)")
    feature_types: Optional[List[str]] = Field(None, description="Feature types to extract")
    use_cache: bool = Field(True, description="Use feature cache")

class BatchECGRequest(BaseModel):
    """Request model for batch ECG analysis."""
    signals: List[List[float]] = Field(..., description="List of ECG signals")
    sampling_rate: int = Field(250, description="Sampling rate in Hz")
    peaks_list: Optional[List[List[int]]] = Field(None, description="List of R-peak indices")
    rr_intervals_list: Optional[List[List[float]]] = Field(None, description="List of RR intervals")
    feature_types: Optional[List[str]] = Field(None, description="Feature types to extract")
    use_cache: bool = Field(True, description="Use feature cache")

class FeatureExtractionResponse(BaseModel):
    """Response model for feature extraction."""
    features: Dict[str, float] = Field(..., description="Extracted features")
    feature_count: int = Field(..., description="Number of features extracted")
    extraction_time_ms: float = Field(..., description="Extraction time in milliseconds")
    cache_hit: bool = Field(False, description="Whether result was from cache")

class BatchFeatureResponse(BaseModel):
    """Response model for batch feature extraction."""
    results: List[FeatureExtractionResponse] = Field(..., description="Batch results")
    total_signals: int = Field(..., description="Total number of signals processed")
    total_time_ms: float = Field(..., description="Total processing time in milliseconds")
    avg_time_per_signal_ms: float = Field(..., description="Average time per signal")

class FeatureInfoResponse(BaseModel):
    """Response model for feature information."""
    total_features: int = Field(..., description="Total number of available features")
    time_domain_features: int = Field(..., description="Number of time-domain features")
    frequency_domain_features: int = Field(..., description="Number of frequency-domain features")
    wavelet_features: int = Field(..., description="Number of wavelet features")
    feature_names: Dict[str, List[str]] = Field(..., description="Feature names by category")

class PerformanceMetricsResponse(BaseModel):
    """Response model for performance metrics."""
    cache_hit_rate: float = Field(..., description="Cache hit rate (0-1)")
    cache_hits: int = Field(..., description="Number of cache hits")
    cache_misses: int = Field(..., description="Number of cache misses")
    total_requests: int = Field(..., description="Total number of requests")
    avg_extraction_time_ms: float = Field(..., description="Average extraction time in ms")
    median_extraction_time_ms: float = Field(..., description="Median extraction time in ms")
    p95_extraction_time_ms: float = Field(..., description="95th percentile extraction time in ms")
    cache_size: int = Field(..., description="Current cache size")

# Dependency to get feature service
def get_feature_extraction_service() -> FeatureExtractionService:
    """Get feature extraction service instance."""
    return get_feature_service()

@router.post("/extract", response_model=FeatureExtractionResponse)
async def extract_features(
    request: ECGSignalRequest,
    service: FeatureExtractionService = Depends(get_feature_extraction_service)
):
    """
    Extract features from ECG signal.
    
    - **signal**: ECG signal samples
    - **sampling_rate**: Sampling rate in Hz (default: 250)
    - **peaks**: Optional R-peak indices
    - **rr_intervals**: Optional RR intervals in milliseconds
    - **feature_types**: Optional list of feature types to extract
    - **use_cache**: Whether to use cache (default: true)
    
    Returns comprehensive feature set including time-domain, frequency-domain, and wavelet features.
    """
    try:
        import time
        start_time = time.time()
        
        # Convert to numpy array
        signal = np.array(request.signal)
        
        # Use provided peaks/RR intervals or detect them
        peaks = request.peaks if request.peaks is not None else []
        rr_intervals = np.array(request.rr_intervals) if request.rr_intervals is not None else np.array([])
        
        # Extract features
        features = service.extract_all_features(
            signal=signal,
            peaks=peaks,
            rr_intervals=rr_intervals,
            use_cache=request.use_cache,
            feature_types=request.feature_types
        )
        
        extraction_time = (time.time() - start_time) * 1000
        
        # Check if result was from cache
        metrics = service.get_performance_metrics()
        cache_hit = metrics['cache_hits'] > 0 and metrics['cache_misses'] == 0
        
        return FeatureExtractionResponse(
            features=features,
            feature_count=len(features),
            extraction_time_ms=extraction_time,
            cache_hit=cache_hit
        )
        
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/extract-batch", response_model=BatchFeatureResponse)
async def extract_features_batch(
    request: BatchECGRequest,
    background_tasks: BackgroundTasks,
    service: FeatureExtractionService = Depends(get_feature_extraction_service)
):
    """
    Extract features from multiple ECG signals in batch.
    
    - **signals**: List of ECG signals
    - **sampling_rate**: Sampling rate in Hz (default: 250)
    - **peaks_list**: Optional list of R-peak indices for each signal
    - **rr_intervals_list**: Optional list of RR intervals for each signal
    - **feature_types**: Optional list of feature types to extract
    - **use_cache**: Whether to use cache (default: true)
    
    Processes multiple signals efficiently with batch optimization.
    """
    try:
        import time
        start_time = time.time()
        
        # Convert to numpy arrays
        signals = [np.array(signal) for signal in request.signals]
        
        # Use provided peaks/RR intervals or empty lists
        peaks_list = request.peaks_list if request.peaks_list is not None else [[] for _ in signals]
        rr_intervals_list = request.rr_intervals_list if request.rr_intervals_list is not None else [np.array([]) for _ in signals]
        
        # Extract features in batch
        batch_results = service.extract_features_batch(
            signals=signals,
            peaks_list=peaks_list,
            rr_intervals_list=rr_intervals_list,
            use_cache=request.use_cache,
            feature_types=request.feature_types
        )
        
        total_time = (time.time() - start_time) * 1000
        avg_time_per_signal = total_time / len(signals)
        
        # Create response objects
        results = []
        for i, features in enumerate(batch_results):
            results.append(FeatureExtractionResponse(
                features=features,
                feature_count=len(features),
                extraction_time_ms=avg_time_per_signal,  # Approximate
                cache_hit=False  # Hard to determine per signal in batch
            ))
        
        return BatchFeatureResponse(
            results=results,
            total_signals=len(signals),
            total_time_ms=total_time,
            avg_time_per_signal_ms=avg_time_per_signal
        )
        
    except Exception as e:
        logger.error(f"Batch feature extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/info", response_model=FeatureInfoResponse)
async def get_feature_info(
    service: FeatureExtractionService = Depends(get_feature_extraction_service)
):
    """
    Get information about available features.
    
    Returns detailed information about:
    - Total number of features
    - Features by category (time-domain, frequency-domain, wavelet)
    - Feature names and descriptions
    """
    try:
        info = service.get_feature_info()
        
        return FeatureInfoResponse(
            total_features=info['total_features'],
            time_domain_features=info['time_domain_features'],
            frequency_domain_features=info['frequency_domain_features'],
            wavelet_features=info['wavelet_features'],
            feature_names={
                'time_domain': info['time_domain_feature_names'],
                'frequency_domain': info['frequency_domain_feature_names'],
                'wavelet': info['wavelet_feature_names']
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get feature info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance", response_model=PerformanceMetricsResponse)
async def get_performance_metrics(
    service: FeatureExtractionService = Depends(get_feature_extraction_service)
):
    """
    Get feature extraction performance metrics.
    
    Returns performance statistics including:
    - Cache hit rate and statistics
    - Extraction time metrics (average, median, 95th percentile)
    - Request counts and cache size
    """
    try:
        metrics = service.get_performance_metrics()
        
        return PerformanceMetricsResponse(
            cache_hit_rate=metrics['cache_hit_rate'],
            cache_hits=metrics['cache_hits'],
            cache_misses=metrics['cache_misses'],
            total_requests=metrics['total_requests'],
            avg_extraction_time_ms=metrics['avg_extraction_time_ms'],
            median_extraction_time_ms=metrics['median_extraction_time_ms'],
            p95_extraction_time_ms=metrics['p95_extraction_time_ms'],
            cache_size=metrics['cache_size']
        )
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cache/clear")
async def clear_cache(
    service: FeatureExtractionService = Depends(get_feature_extraction_service)
):
    """
    Clear feature extraction cache.
    
    Clears all cached features and resets performance metrics.
    """
    try:
        service.clear_cache()
        
        return {
            "message": "Feature cache cleared successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check(
    service: FeatureExtractionService = Depends(get_feature_extraction_service)
):
    """
    Health check for feature extraction service.
    
    Returns service status and basic health information.
    """
    try:
        # Test basic functionality
        test_signal = np.random.randn(1000)
        test_peaks = [100, 200, 300, 400, 500]
        test_rr = np.array([800, 820, 810, 830])
        
        test_features = service.extract_all_features(
            test_signal, test_peaks, test_rr, use_cache=False
        )
        
        metrics = service.get_performance_metrics()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service_info": {
                "sampling_rate": service.sampling_rate,
                "cache_size": service.cache_size,
                "cache_ttl": service.cache_ttl,
                "workers": service.n_workers,
                "parallel_processing": service.enable_parallel
            },
            "performance": {
                "total_requests": metrics['total_requests'],
                "cache_hit_rate": metrics['cache_hit_rate'],
                "avg_extraction_time_ms": metrics['avg_extraction_time_ms']
            },
            "test_result": {
                "features_extracted": len(test_features),
                "test_passed": len(test_features) > 50
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
        )

@router.post("/validate")
async def validate_signal(
    request: ECGSignalRequest,
    service: FeatureExtractionService = Depends(get_feature_extraction_service)
):
    """
    Validate ECG signal and extract basic statistics.
    
    Performs signal validation and returns basic statistics
    without full feature extraction.
    """
    try:
        signal = np.array(request.signal)
        
        # Basic validation
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check signal length
        if len(signal) == 0:
            validation_results["is_valid"] = False
            validation_results["errors"].append("Signal is empty")
        elif len(signal) < 100:
            validation_results["warnings"].append("Signal is very short (<100 samples)")
        
        # Check for invalid values
        if np.any(np.isnan(signal)):
            validation_results["is_valid"] = False
            validation_results["errors"].append("Signal contains NaN values")
        
        if np.any(np.isinf(signal)):
            validation_results["is_valid"] = False
            validation_results["errors"].append("Signal contains infinite values")
        
        # Basic statistics
        stats = {
            "length": len(signal),
            "sampling_rate": request.sampling_rate,
            "duration_seconds": len(signal) / request.sampling_rate,
            "mean": float(np.mean(signal)),
            "std": float(np.std(signal)),
            "min": float(np.min(signal)),
            "max": float(np.max(signal)),
            "range": float(np.ptp(signal)),
            "rms": float(np.sqrt(np.mean(signal**2)))
        }
        
        return {
            "validation": validation_results,
            "statistics": stats
        }
        
    except Exception as e:
        logger.error(f"Signal validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/types")
async def get_feature_types():
    """
    Get available feature types.
    
    Returns list of available feature extraction types
    and their descriptions.
    """
    return {
        "feature_types": {
            "time": {
                "name": "Time-Domain Features",
                "description": "RR intervals, HRV metrics, morphology features",
                "count": "~25"
            },
            "frequency": {
                "name": "Frequency-Domain Features",
                "description": "FFT coefficients, power spectral density, spectral analysis",
                "count": "~20"
            },
            "wavelet": {
                "name": "Wavelet Features",
                "description": "DWT, CWT coefficients, wavelet packets",
                "count": "~30"
            }
        },
        "total_features": "~75",
        "extraction_time_ms": {
            "time": "<5ms",
            "frequency": "<8ms",
            "wavelet": "<10ms",
            "all": "<15ms"
        }
    }
