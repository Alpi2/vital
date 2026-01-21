"""
Model Management API Endpoints

REST API endpoints for managing ML models, A/B testing,
and inference operations.
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
import logging
from pathlib import Path
import json
from datetime import datetime

from app.ml import ModelRegistry, InferenceService, ABTestManager
from app.ml.model_registry import ModelMetadata, ABTestConfig
from app.ml.inference_service import InferenceResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/models", tags=["models"])

# Global instances (would be injected by dependency injection in production)
model_registry: Optional[ModelRegistry] = None
inference_service: Optional[InferenceService] = None
ab_test_manager: Optional[ABTestManager] = None

def initialize_services(
    registry: ModelRegistry,
    inference: InferenceService,
    ab_manager: ABTestManager
):
    """Initialize API services.
    
    Args:
        registry: Model registry instance
        inference: Inference service instance
        ab_manager: A/B test manager instance
    """
    global model_registry, inference_service, ab_test_manager
    model_registry = registry
    inference_service = inference
    ab_test_manager = ab_manager

@router.get("/", response_model=Dict[str, Any])
async def list_models(
    status: Optional[str] = None,
    framework: Optional[str] = None,
    tags: Optional[str] = None
):
    """List all registered models.
    
    Args:
        status: Filter by status
        framework: Filter by framework
        tags: Filter by tags (comma-separated)
        
    Returns:
        List of model metadata
    """
    if not model_registry:
        raise HTTPException(status_code=503, detail="Model service not available")
    
    try:
        # Parse tags if provided
        tag_list = None
        if tags:
            tag_list = [tag.strip() for tag in tags.split(',')]
        
        models = model_registry.list_models(
            status=None,  # Would convert string to enum
            framework=framework,
            tags=tag_list
        )
        
        return {
            "models": [model.to_dict() for model in models],
            "total": len(models)
        }
        
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{model_id}", response_model=Dict[str, Any])
async def get_model(model_id: str):
    """Get model details.
    
    Args:
        model_id: Model identifier
        
    Returns:
        Model metadata
    """
    if not model_registry:
        raise HTTPException(status_code=503, detail="Model service not available")
    
    try:
        model = model_registry.get_model(model_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        return model.to_dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/", response_model=Dict[str, Any])
async def register_model(
    background_tasks: BackgroundTasks,
    model_id: str = Form(...),
    name: str = Form(...),
    description: str = Form(...),
    framework: str = Form(...),
    input_shape: str = Form(...),
    output_shape: str = Form(...),
    classes: str = Form(...),
    version: str = Form(default="1.0.0"),
    tags: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None),
    model_file: UploadFile = File(...)
):
    """Register a new model.
    
    Args:
        model_id: Unique model identifier
        name: Human-readable name
        description: Model description
        framework: ML framework
        input_shape: Input shape as string (e.g., "1,1,1000")
        output_shape: Output shape as string
        classes: Classes as JSON string
        version: Model version
        tags: Tags as comma-separated string
        metadata: Additional metadata as JSON string
        model_file: Model file upload
        
    Returns:
        Registration result
    """
    if not model_registry:
        raise HTTPException(status_code=503, detail="Model service not available")
    
    try:
        # Parse input/output shapes
        input_shape_tuple = tuple(map(int, input_shape.split(',')))
        output_shape_tuple = tuple(map(int, output_shape.split(',')))
        
        # Parse classes
        classes_list = json.loads(classes)
        
        # Parse tags
        tag_list = None
        if tags:
            tag_list = [tag.strip() for tag in tags.split(',')]
        
        # Parse metadata
        metadata_dict = None
        if metadata:
            metadata_dict = json.loads(metadata)
        
        # Save uploaded file
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        file_path = models_dir / f"{model_id}_{version}.{model_file.filename.split('.')[-1]}"
        
        with open(file_path, "wb") as buffer:
            content = await model_file.read()
            buffer.write(content)
        
        # Register model in background
        def register_model_bg():
            success = model_registry.register_model(
                model_id=model_id,
                name=name,
                description=description,
                model_path=file_path,
                framework=framework,
                input_shape=input_shape_tuple,
                output_shape=output_shape_tuple,
                classes=classes_list,
                version=version,
                tags=tag_list,
                metadata=metadata_dict
            )
            
            if not success:
                # Clean up file if registration failed
                file_path.unlink(missing_ok=True)
        
        background_tasks.add_task(register_model_bg)
        
        return {
            "message": "Model registration started",
            "model_id": model_id,
            "version": version,
            "file_path": str(file_path)
        }
        
    except Exception as e:
        logger.error(f"Failed to register model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{model_id}/status", response_model=Dict[str, Any])
async def update_model_status(
    model_id: str,
    status: str = Form(...)
):
    """Update model status.
    
    Args:
        model_id: Model identifier
        status: New status
        
    Returns:
        Update result
    """
    if not model_registry:
        raise HTTPException(status_code=503, detail="Model service not available")
    
    try:
        # Convert status string to enum (simplified)
        success = model_registry.update_model_status(model_id, None)  # Would convert string
        
        if not success:
            raise HTTPException(status_code=404, detail="Model not found")
        
        return {
            "message": "Model status updated",
            "model_id": model_id,
            "new_status": status
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update model status {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{model_id}", response_model=Dict[str, Any])
async def delete_model(model_id: str):
    """Delete a model.
    
    Args:
        model_id: Model identifier
        
    Returns:
        Deletion result
    """
    if not model_registry:
        raise HTTPException(status_code=503, detail="Model service not available")
    
    try:
        success = model_registry.delete_model(model_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Model not found")
        
        return {
            "message": "Model deleted successfully",
            "model_id": model_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{model_id}/predict", response_model=Dict[str, Any])
async def predict(
    model_id: str,
    features: List[float],
    request_id: Optional[str] = None
):
    """Run inference on a model.
    
    Args:
        model_id: Model identifier
        features: Input features
        request_id: Optional request identifier
        
    Returns:
        Inference result
    """
    if not inference_service:
        raise HTTPException(status_code=503, detail="Inference service not available")
    
    try:
        import numpy as np
        features_array = np.array(features, dtype=np.float32)
        
        result = inference_service.predict_sync(
            features=features_array,
            model_id=model_id,
            request_id=request_id
        )
        
        return result.to_dict()
        
    except Exception as e:
        logger.error(f"Failed to run inference for model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{model_id}/predict-batch", response_model=Dict[str, Any])
async def predict_batch(
    model_id: str,
    features_batch: List[List[float]],
    request_ids: Optional[List[str]] = None
):
    """Run batch inference on a model.
    
    Args:
        model_id: Model identifier
        features_batch: List of feature arrays
        request_ids: Optional request identifiers
        
    Returns:
        Batch inference results
    """
    if not inference_service:
        raise HTTPException(status_code=503, detail="Inference service not available")
    
    try:
        import numpy as np
        features_arrays = [np.array(features, dtype=np.float32) for features in features_batch]
        
        results = inference_service.predict_batch(
            features_batch=features_arrays,
            model_id=model_id,
            request_ids=request_ids
        )
        
        return {
            "results": [result.to_dict() for result in results],
            "total": len(results)
        }
        
    except Exception as e:
        logger.error(f"Failed to run batch inference for model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# A/B Testing Endpoints

@router.get("/ab-tests", response_model=Dict[str, Any])
async def list_ab_tests():
    """List all A/B tests.
    
    Returns:
        List of A/B test configurations
    """
    if not ab_test_manager:
        raise HTTPException(status_code=503, detail="A/B test service not available")
    
    try:
        tests = ab_test_manager.list_tests()
        
        return {
            "tests": [test.to_dict() for test in tests],
            "total": len(tests)
        }
        
    except Exception as e:
        logger.error(f"Failed to list A/B tests: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ab-tests", response_model=Dict[str, Any])
async def create_ab_test(
    test_config: Dict[str, Any]
):
    """Create a new A/B test.
    
    Args:
        test_config: A/B test configuration
        
    Returns:
        Creation result
    """
    if not ab_test_manager:
        raise HTTPException(status_code=503, detail="A/B test service not available")
    
    try:
        config = ABTestConfig.from_dict(test_config)
        success = ab_test_manager.create_test(config)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to create A/B test")
        
        return {
            "message": "A/B test created successfully",
            "test_id": config.test_id,
            "config": config.to_dict()
        }
        
    except Exception as e:
        logger.error(f"Failed to create A/B test: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ab-tests/{test_id}", response_model=Dict[str, Any])
async def get_ab_test(test_id: str):
    """Get A/B test details.
    
    Args:
        test_id: Test identifier
        
    Returns:
        A/B test configuration and metrics
    """
    if not ab_test_manager:
        raise HTTPException(status_code=503, detail="A/B test service not available")
    
    try:
        # Get test configuration
        tests = ab_test_manager.list_tests()
        test_config = None
        
        for test in tests:
            if test.test_id == test_id:
                test_config = test
                break
        
        if not test_config:
            raise HTTPException(status_code=404, detail="A/B test not found")
        
        # Get test metrics
        metrics = ab_test_manager.get_test_metrics(test_id)
        
        return {
            "config": test_config.to_dict(),
            "metrics": {
                model_id: {
                    "total_requests": m.total_requests,
                    "success_rate": m.success_rate,
                    "average_latency_ms": m.average_latency_ms,
                    "average_accuracy": m.average_accuracy
                }
                for model_id, m in (metrics or {}).items()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get A/B test {test_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ab-tests/{test_id}/analyze", response_model=Dict[str, Any])
async def analyze_ab_test(test_id: str):
    """Analyze A/B test results.
    
    Args:
        test_id: Test identifier
        
    Returns:
        Statistical analysis results
    """
    if not ab_test_manager:
        raise HTTPException(status_code=503, detail="A/B test service not available")
    
    try:
        result = ab_test_manager.analyze_test(test_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="A/B test not found")
        
        return {
            "test_id": result.test_id,
            "winner": result.winner,
            "confidence": result.confidence,
            "p_value": result.p_value,
            "effect_size": result.effect_size,
            "statistical_significance": result.statistical_significance,
            "recommendation": result.recommendation,
            "analysis_time": result.analysis_time.isoformat(),
            "model_a_metrics": {
                "total_requests": result.model_a_metrics.total_requests,
                "success_rate": result.model_a_metrics.success_rate,
                "average_latency_ms": result.model_a_metrics.average_latency_ms
            },
            "model_b_metrics": {
                "total_requests": result.model_b_metrics.total_requests,
                "success_rate": result.model_b_metrics.success_rate,
                "average_latency_ms": result.model_b_metrics.average_latency_ms
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to analyze A/B test {test_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/ab-tests/{test_id}", response_model=Dict[str, Any])
async def stop_ab_test(test_id: str):
    """Stop an A/B test.
    
    Args:
        test_id: Test identifier
        
    Returns:
        Stop result
    """
    if not ab_test_manager:
        raise HTTPException(status_code=503, detail="A/B test service not available")
    
    try:
        success = ab_test_manager.stop_test(test_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="A/B test not found")
        
        return {
            "message": "A/B test stopped successfully",
            "test_id": test_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to stop A/B test {test_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Metrics and Monitoring Endpoints

@router.get("/metrics", response_model=Dict[str, Any])
async def get_metrics():
    """Get system and model metrics.
    
    Returns:
        Performance metrics
    """
    if not inference_service:
        raise HTTPException(status_code=503, detail="Inference service not available")
    
    try:
        metrics = inference_service.get_metrics()
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics/prometheus", response_model=str)
async def get_prometheus_metrics():
    """Get metrics in Prometheus format.
    
    Returns:
        Prometheus-formatted metrics
    """
    if not inference_service:
        raise HTTPException(status_code=503, detail="Inference service not available")
    
    try:
        # This would use the MetricsExporter
        # For now, return basic metrics
        metrics = inference_service.get_metrics()
        
        # Convert to Prometheus format (simplified)
        prometheus_metrics = []
        
        prometheus_metrics.append(f"# HELP inference_requests_total Total number of inference requests")
        prometheus_metrics.append(f"# TYPE inference_requests_total counter")
        prometheus_metrics.append(f"inference_requests_total {metrics.get('total_requests', 0)}")
        
        prometheus_metrics.append(f"# HELP inference_latency_ms Inference latency in milliseconds")
        prometheus_metrics.append(f"# TYPE inference_latency_ms histogram")
        prometheus_metrics.append(f"inference_latency_ms_sum {metrics.get('total_latency_ms', 0)}")
        prometheus_metrics.append(f"inference_latency_ms_count {metrics.get('successful_requests', 0)}")
        
        prometheus_metrics.append(f"# HELP cache_hit_rate Cache hit rate")
        prometheus_metrics.append(f"# TYPE cache_hit_rate gauge")
        prometheus_metrics.append(f"cache_hit_rate {metrics.get('cache_hit_rate', 0)}")
        
        return "\n".join(prometheus_metrics)
        
    except Exception as e:
        logger.error(f"Failed to get Prometheus metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cache/clear", response_model=Dict[str, Any])
async def clear_cache():
    """Clear inference cache.
    
    Returns:
        Cache clear result
    """
    if not inference_service:
        raise HTTPException(status_code=503, detail="Inference service not available")
    
    try:
        inference_service.clear_cache()
        
        return {
            "message": "Inference cache cleared successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health Check

@router.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Health check endpoint.
    
    Returns:
        Service health status
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {}
    }
    
    # Check model registry
    if model_registry:
        try:
            stats = model_registry.get_registry_stats()
            health_status["services"]["model_registry"] = {
                "status": "healthy",
                "total_models": stats.get("total_models", 0)
            }
        except Exception as e:
            health_status["services"]["model_registry"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["status"] = "degraded"
    else:
        health_status["services"]["model_registry"] = {
            "status": "unavailable"
        }
        health_status["status"] = "degraded"
    
    # Check inference service
    if inference_service:
        try:
            metrics = inference_service.get_metrics()
            health_status["services"]["inference_service"] = {
                "status": "healthy",
                "total_requests": metrics.get("total_requests", 0)
            }
        except Exception as e:
            health_status["services"]["inference_service"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["status"] = "degraded"
    else:
        health_status["services"]["inference_service"] = {
            "status": "unavailable"
        }
        health_status["status"] = "degraded"
    
    # Check A/B test manager
    if ab_test_manager:
        try:
            tests = ab_test_manager.list_tests()
            health_status["services"]["ab_test_manager"] = {
                "status": "healthy",
                "active_tests": len(tests)
            }
        except Exception as e:
            health_status["services"]["ab_test_manager"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["status"] = "degraded"
    else:
        health_status["services"]["ab_test_manager"] = {
            "status": "unavailable"
        }
        health_status["status"] = "degraded"
    
    return health_status
