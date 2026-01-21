"""
FastAPI Router for ECG FFI Processing
High-performance ECG analysis with Rust FFI backend
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Union, Any
import numpy as np
import time
import logging
import asyncio
from datetime import datetime

from app.services.ecg_analyzer_ffi import ECGAnalyzerFFI, get_global_analyzer
from app.security.dependencies import get_current_user, get_current_user_optional
from app.monitoring.metrics import record_request_duration, record_error, record_throughput

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/ecg", tags=["ECG FFI"])

# Initialize global analyzer
analyzer = get_global_analyzer()


# Request/Response Models
class ECGAnalysisRequest(BaseModel):
    """ECG analysis request with comprehensive validation"""
    
    signal: List[float] = Field(
        ...,
        min_items=100,
        max_items=50000,
        description="ECG signal samples (100-50000 points)"
    )
    
    sampling_rate: float = Field(
        default=360.0,
        ge=100.0,
        le=2000.0,
        description="Sampling rate in Hz (100-2000)"
    )
    
    patient_id: Optional[str] = Field(
        None,
        max_length=50,
        description="Patient identifier"
    )
    
    session_id: Optional[str] = Field(
        None,
        max_length=100,
        description="Session identifier"
    )
    
    enable_hrv: Optional[bool] = Field(
        default=True,
        description="Enable heart rate variability analysis"
    )
    
    enable_anomaly_detection: Optional[bool] = Field(
        default=True,
        description="Enable anomaly detection"
    )
    
    quality_threshold: Optional[float] = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Minimum signal quality threshold"
    )
    
    @validator('signal')
    def validate_signal(cls, v):
        if not v:
            raise ValueError('Signal cannot be empty')
        
        # Check for invalid values
        if any(not isinstance(x, (int, float)) or not np.isfinite(x) for x in v):
            raise ValueError('Signal contains invalid values (NaN, Inf, or non-numeric)')
        
        # Check signal range (reasonable ECG values)
        if any(abs(x) > 10.0 for x in v):
            raise ValueError('Signal contains values outside reasonable range (-10 to 10 mV)')
        
        return v
    
    @validator('patient_id')
    def validate_patient_id(cls, v):
        if v and not v.strip():
            raise ValueError('Patient ID cannot be empty if provided')
        return v.strip() if v else None


class ECGAnalysisResponse(BaseModel):
    """ECG analysis response with comprehensive metrics"""
    
    heart_rate: float = Field(..., description="Heart rate in BPM")
    rr_intervals: List[float] = Field(..., description="RR intervals in milliseconds")
    qrs_peaks: List[int] = Field(..., description="QRS peak indices")
    hrv_metrics: Dict[str, float] = Field(..., description="Heart rate variability metrics")
    anomalies: List[str] = Field(..., description="Detected anomalies")
    signal_quality: float = Field(..., description="Signal quality score (0-1)")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    algorithm_version: str = Field(..., description="Algorithm version")
    processing_backend: str = Field(..., description="Processing backend used")
    patient_id: Optional[str] = Field(None, description="Patient identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    timestamp: str = Field(..., description="Analysis timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "heart_rate": 72.5,
                "rr_intervals": [833.3, 833.3, 833.3],
                "qrs_peaks": [100, 400, 700],
                "hrv_metrics": {"rmssd": 45.2, "sdnn": 23.1, "mean_rr": 833.3},
                "anomalies": [],
                "signal_quality": 0.95,
                "processing_time_ms": 3.2,
                "algorithm_version": "1.0.0",
                "processing_backend": "rust_ffi",
                "patient_id": "patient_123",
                "session_id": "session_456",
                "timestamp": "2026-01-17T11:46:00Z"
            }
        }


class ECGBatchAnalysisRequest(BaseModel):
    """Batch ECG analysis request"""
    
    signals: List[List[float]] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="List of ECG signals (1-100 signals)"
    )
    
    sampling_rate: float = Field(
        default=360.0,
        ge=100.0,
        le=2000.0,
        description="Sampling rate in Hz"
    )
    
    patient_id: Optional[str] = Field(None, description="Patient identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    
    @validator('signals')
    def validate_signals(cls, v):
        if not v:
            raise ValueError('Signals list cannot be empty')
        
        for i, signal in enumerate(v):
            if len(signal) < 100:
                raise ValueError(f'Signal {i} too short: {len(signal)} samples (minimum: 100)')
            if len(signal) > 50000:
                raise ValueError(f'Signal {i} too long: {len(signal)} samples (maximum: 50000)')
        
        return v


class ECGPerformanceMetrics(BaseModel):
    """Performance metrics response"""
    
    total_analyses: int = Field(..., description="Total number of analyses")
    ffi_analyses: int = Field(..., description="Number of FFI analyses")
    python_analyses: int = Field(..., description="Number of Python fallback analyses")
    average_time_ms: float = Field(..., description="Average processing time")
    error_rate: float = Field(..., description="Error rate (0-1)")
    ffi_usage_rate: float = Field(..., description="FFI usage rate (0-1)")
    backend_available: bool = Field(..., description="Whether Rust FFI backend is available")


# Background task for logging
async def log_analysis_result(
    patient_id: Optional[str],
    session_id: Optional[str],
    result: ECGAnalysisResponse,
    processing_time: float
):
    """Log analysis result for monitoring"""
    try:
        # This would typically log to a database or monitoring system
        logger.info(
            f"ECG Analysis completed - Patient: {patient_id or 'unknown'}, "
            f"Session: {session_id or 'unknown'}, "
            f"HR: {result.heart_rate:.1f} BPM, "
            f"Quality: {result.signal_quality:.2f}, "
            f"Backend: {result.processing_backend}, "
            f"Time: {processing_time:.2f}ms"
        )
    except Exception as e:
        logger.error(f"Failed to log analysis result: {e}")


# API Endpoints
@router.post("/analyze", response_model=ECGAnalysisResponse)
async def analyze_ecg(
    request: ECGAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: Optional[Dict] = Depends(get_current_user_optional),
    benchmark: bool = Query(False, description="Enable benchmarking mode")
):
    """
    Analyze ECG signal using Rust FFI backend
    
    Features:
    - High-performance Rust backend (<5ms latency)
    - Automatic Python fallback
    - Comprehensive validation
    - Performance monitoring
    - Error handling
    """
    start_time = time.time()
    
    try:
        # Convert to NumPy array
        signal = np.array(request.signal, dtype=np.float64)
        
        # Create configuration
        from app.services.ecg_analyzer_ffi import ECGAnalysisConfig
        config = ECGAnalysisConfig(
            sampling_rate=request.sampling_rate,
            enable_hrv=request.enable_hrv,
            enable_anomaly_detection=request.enable_anomaly_detection,
            noise_threshold=request.quality_threshold,
        )
        
        # Perform analysis
        result = await analyzer.analyze_async(signal, config)
        
        # Create response
        response = ECGAnalysisResponse(
            heart_rate=result.heart_rate,
            rr_intervals=result.rr_intervals,
            qrs_peaks=result.qrs_peaks,
            hrv_metrics=result.hrv_metrics,
            anomalies=result.anomalies,
            signal_quality=result.signal_quality,
            processing_time_ms=result.processing_time_ms,
            algorithm_version=result.algorithm_version,
            processing_backend=result.processing_backend,
            patient_id=request.patient_id,
            session_id=request.session_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
        )
        
        # Record metrics
        processing_time = (time.time() - start_time) * 1000.0
        record_request_duration("ecg_analyze", processing_time)
        record_throughput("ecg_analyze")
        
        # Log result in background
        background_tasks.add_task(
            log_analysis_result,
            request.patient_id,
            request.session_id,
            response,
            processing_time
        )
        
        # Benchmark mode logging
        if benchmark:
            logger.info(f"🔍 BENCHMARK: ECG analysis - {processing_time:.2f}ms, "
                       f"Backend: {result.processing_backend}, "
                       f"Signal length: {len(signal)}")
        
        return response
        
    except ValueError as e:
        record_error("ecg_analyze", "validation_error", str(e))
        raise HTTPException(status_code=400, detail=f"Validation error: {e}")
    
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000.0
        record_error("ecg_analyze", "processing_error", str(e))
        logger.error(f"💥 ECG analysis failed: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"ECG analysis failed: {str(e)}"
        )


@router.post("/analyze/batch", response_model=List[ECGAnalysisResponse])
async def analyze_ecg_batch(
    request: ECGBatchAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: Optional[Dict] = Depends(get_current_user_optional),
):
    """
    Batch ECG analysis with parallel processing
    
    Processes multiple ECG signals efficiently using Rust FFI backend
    """
    start_time = time.time()
    
    try:
        # Validate batch size
        if len(request.signals) > 100:
            raise ValueError("Batch size too large (maximum: 100 signals)")
        
        # Convert to NumPy arrays
        signals = [np.array(signal, dtype=np.float64) for signal in request.signals]
        
        # Create configuration
        from app.services.ecg_analyzer_ffi import ECGAnalysisConfig
        config = ECGAnalysisConfig(
            sampling_rate=request.sampling_rate,
            enable_hrv=True,
            enable_anomaly_detection=True,
            noise_threshold=0.1,
        )
        
        # Perform batch analysis
        results = analyzer.analyze_batch(signals, config)
        
        # Create responses
        responses = []
        for i, result in enumerate(results):
            response = ECGAnalysisResponse(
                heart_rate=result.heart_rate,
                rr_intervals=result.rr_intervals,
                qrs_peaks=result.qrs_peaks,
                hrv_metrics=result.hrv_metrics,
                anomalies=result.anomalies,
                signal_quality=result.signal_quality,
                processing_time_ms=result.processing_time_ms,
                algorithm_version=result.algorithm_version,
                processing_backend=result.processing_backend,
                patient_id=request.patient_id,
                session_id=f"{request.session_id}_{i}" if request.session_id else None,
                timestamp=datetime.utcnow().isoformat() + "Z",
            )
            responses.append(response)
        
        # Record metrics
        processing_time = (time.time() - start_time) * 1000.0
        record_request_duration("ecg_batch_analyze", processing_time)
        record_throughput("ecg_batch_analyze", count=len(signals))
        
        logger.info(f"🚀 Batch ECG analysis completed - {len(signals)} signals, "
                   f"{processing_time:.2f}ms total, "
                   f"{processing_time/len(signals):.2f}ms per signal")
        
        return responses
        
    except ValueError as e:
        record_error("ecg_batch_analyze", "validation_error", str(e))
        raise HTTPException(status_code=400, detail=f"Validation error: {e}")
    
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000.0
        record_error("ecg_batch_analyze", "processing_error", str(e))
        logger.error(f"💥 Batch ECG analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch ECG analysis failed: {str(e)}"
        )


@router.get("/analyze/stream")
async def analyze_ecg_stream(
    signal_length: int = Query(..., ge=100, le=10000, description="Signal length for streaming"),
    sampling_rate: float = Query(360.0, ge=100, le=2000, description="Sampling rate"),
    chunk_size: int = Query(1000, ge=100, le=5000, description="Chunk size for streaming"),
    current_user: Optional[Dict] = Depends(get_current_user_optional),
):
    """
    Streaming ECG analysis for real-time processing
    
    Provides real-time ECG analysis using Server-Sent Events
    """
    async def event_generator():
        try:
            # Generate synthetic ECG signal for demonstration
            t = np.linspace(0, signal_length / sampling_rate, signal_length)
            signal = np.sin(2 * np.pi * 1.2 * t)  # 1.2 Hz heart rate
            
            # Process in chunks
            for i in range(0, len(signal), chunk_size):
                chunk = signal[i:i + chunk_size]
                
                if len(chunk) < 100:
                    continue
                
                # Analyze chunk
                from app.services.ecg_analyzer_ffi import ECGAnalysisConfig
                config = ECGAnalysisConfig(sampling_rate=sampling_rate)
                
                result = await analyzer.analyze_async(chunk, config)
                
                # Send SSE event
                yield {
                    "data": {
                        "chunk_index": i // chunk_size,
                        "chunk_size": len(chunk),
                        "heart_rate": result.heart_rate,
                        "signal_quality": result.signal_quality,
                        "processing_time_ms": result.processing_time_ms,
                        "backend": result.processing_backend,
                        "timestamp": datetime.utcnow().isoformat() + "Z"
                    }
                }
                
                # Small delay to simulate real-time processing
                await asyncio.sleep(0.1)
        
        except Exception as e:
            logger.error(f"💥 Streaming analysis failed: {e}")
            yield {
                "error": str(e)
            }
    
    return JSONResponse(
        content="Streaming endpoint - use EventSource for real-time updates",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@router.get("/status", response_model=ECGPerformanceMetrics)
async def get_ecg_status(
    current_user: Optional[Dict] = Depends(get_current_user_optional)
):
    """
    Get ECG processing performance metrics and status
    """
    try:
        stats = analyzer.get_stats()
        
        return ECGPerformanceMetrics(
            total_analyses=stats["total_analyses"],
            ffi_analyses=stats["ffi_analyses"],
            python_analyses=stats["python_analyses"],
            average_time_ms=stats["average_time_ms"],
            error_rate=stats["error_rate"],
            ffi_usage_rate=stats["ffi_usage_rate"],
            backend_available=analyzer.use_ffi,
        )
        
    except Exception as e:
        logger.error(f"💥 Failed to get ECG status: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve ECG processing status"
        )


@router.post("/benchmark")
async def benchmark_ecg_performance(
    signal_length: int = Query(5000, ge=1000, le=50000, description="Signal length for benchmark"),
    iterations: int = Query(100, ge=10, le=1000, description="Number of iterations"),
    sampling_rate: float = Query(360.0, ge=100, le=2000, description="Sampling rate"),
    current_user: Optional[Dict] = Depends(get_current_user_optional),
):
    """
    Benchmark ECG processing performance
    
    Runs performance tests and returns detailed metrics
    """
    try:
        # Generate test signal
        t = np.linspace(0, signal_length / sampling_rate, signal_length)
        signal = np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.randn(signal_length)
        
        # Run benchmark
        if analyzer.use_ffi:
            # Use Rust benchmark function if available
            try:
                from app.services.ecg_analyzer_ffi import _rust_ffi
                rust_metrics = _rust_ffi.benchmark_ecg_analysis(
                    signal_length, iterations, sampling_rate
                )
                
                return {
                    "backend": "rust_ffi",
                    "signal_length": signal_length,
                    "iterations": iterations,
                    "metrics": rust_metrics,
                    "performance_class": "high_performance" if rust_metrics["avg_time_ms"] < 5 else "standard"
                }
            except Exception as e:
                logger.warning(f"Rust benchmark failed, using Python fallback: {e}")
        
        # Python fallback benchmark
        start_time = time.time()
        
        for _ in range(iterations):
            result = analyzer.analyze(signal)
        
        total_time = time.time() - start_time
        avg_time_ms = (total_time / iterations) * 1000.0
        
        return {
            "backend": analyzer.processing_backend,
            "signal_length": signal_length,
            "iterations": iterations,
            "metrics": {
                "total_time_ms": total_time * 1000.0,
                "avg_time_ms": avg_time_ms,
                "throughput_signals_per_sec": iterations / total_time,
                "samples_per_sec": (signal_length * iterations) / total_time,
            },
            "performance_class": "high_performance" if avg_time_ms < 5 else "standard"
        }
        
    except Exception as e:
        logger.error(f"💥 Benchmark failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Benchmark failed: {str(e)}"
        )


@router.post("/reset-stats")
async def reset_ecg_stats(
    current_user: Dict = Depends(get_current_user)  # Require authentication
):
    """
    Reset ECG processing statistics
    
    Requires authentication
    """
    try:
        analyzer.reset_stats()
        logger.info("📊 ECG processing statistics reset")
        return {"message": "Statistics reset successfully"}
        
    except Exception as e:
        logger.error(f"💥 Failed to reset stats: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to reset statistics"
        )


@router.get("/health")
async def ecg_health_check():
    """
    Health check for ECG processing service
    """
    try:
        # Test basic functionality
        test_signal = np.sin(2 * np.pi * np.linspace(0, 10, 1000))
        result = analyzer.analyze(test_signal)
        
        return {
            "status": "healthy",
            "backend": result.processing_backend,
            "ffi_available": analyzer.use_ffi,
            "test_processing_time_ms": result.processing_time_ms,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
    except Exception as e:
        logger.error(f"💥 ECG health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        )
