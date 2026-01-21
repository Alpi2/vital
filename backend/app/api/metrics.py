"""
Metrics API Endpoint
Exposes Prometheus metrics for compliance monitoring
"""

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST
import logging

from app.compliance.metrics import metrics_endpoint as get_metrics

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/metrics", tags=["metrics"])

@router.get("/")
async def metrics():
    """
    Prometheus metrics endpoint
    
    Returns compliance metrics in Prometheus format for monitoring and alerting.
    This endpoint is used by Prometheus to scrape metrics from the application.
    
    Metrics include:
    - Compliance scores by category
    - Overall compliance score
    - Background task metrics
    - Breach alert metrics
    - Audit log metrics
    - System health metrics
    - API request metrics
    - Cache performance metrics
    """
    try:
        return await get_metrics()
    except Exception as e:
        logger.error(f"Error generating metrics: {e}")
        return Response(
            content="Error generating metrics",
            status_code=500,
            media_type="text/plain"
        )
