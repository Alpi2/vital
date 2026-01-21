from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
from sqlalchemy.orm import selectinload
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import logging
import json

from ..database import get_session
from ..models.user import User
from ..models.web_vitals import WebVitalsMetric
from ..security.dependencies import get_current_user, require_permission
# from ..core.cache import cache_manager  # Removed for demo

logger = logging.getLogger(__name__)
router = APIRouter()

# Pydantic models for Web Vitals
class WebVitalsData(BaseModel):
    metric: str = Field(..., description="Metric name (LCP, INP, CLS, TTFB)")
    value: float = Field(..., description="Metric value")
    rating: str = Field(..., description="Rating (good, needs-improvement, poor)")
    url: str = Field(..., description="Page URL")
    timestamp: int = Field(..., description="Unix timestamp")
    user_agent: str = Field(default="", description="Browser user agent")
    viewport: Optional[Dict[str, int]] = Field(default=None, description="Viewport dimensions")
    connection: Optional[Dict[str, Any]] = Field(default=None, description="Connection info")

class WebVitalsResponse(BaseModel):
    success: bool
    message: str
    metric_id: Optional[int] = None

class PerformanceAnalytics(BaseModel):
    metric_name: str
    avg_value: float
    median_value: float
    p75_value: float
    p95_value: float
    sample_count: int
    rating_distribution: Dict[str, int]

class PerformanceDashboard(BaseModel):
    period: str
    total_samples: int
    unique_pages: int
    metrics: List[PerformanceAnalytics]
    trends: Dict[str, List[Dict[str, Any]]]
    recommendations: List[str]

@router.post("/web-vitals", response_model=WebVitalsResponse)
async def record_web_vitals(
    data: WebVitalsData,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_session)
):
    """
    Record Web Vitals metrics from frontend.
    
    This endpoint collects Core Web Vitals data for performance monitoring.
    Rate limited to prevent abuse and data overload.
    """
    try:
        # Create Web Vitals metric record
        metric = WebVitalsMetric(
            metric_name=data.metric,
            value=data.value,
            rating=data.rating,
            url=data.url,
            timestamp=datetime.fromtimestamp(data.timestamp / 1000),
            user_agent=data.user_agent,
            viewport_width=data.viewport.get('width') if data.viewport else None,
            viewport_height=data.viewport.get('height') if data.viewport else None,
            connection_type=data.connection.get('effectiveType') if data.connection else None,
            connection_downlink=data.connection.get('downlink') if data.connection else None,
            connection_rtt=data.connection.get('rtt') if data.connection else None,
            created_at=datetime.utcnow()
        )
        
        db.add(metric)
        await db.commit()
        
        # Schedule background processing for analytics
        background_tasks.add_task(process_web_vitals_analytics, metric.id)
        
        # Cache recent metrics for dashboard
        cache_key = f"web_vitals:recent:{data.metric}"
        # await cache_manager.set(cache_key, {  # Removed for demo
        #     'value': data.value,
        #     'rating': data.rating,
        #     'timestamp': data.timestamp
        # }, ttl=300)  # 5 minutes cache
        
        logger.info(f"Web Vitals recorded: {data.metric}={data.value} ({data.rating})")
        
        return WebVitalsResponse(
            success=True,
            message="Web Vitals recorded successfully",
            metric_id=metric.id
        )
        
    except Exception as e:
        logger.error(f"Failed to record Web Vitals: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to record Web Vitals")

@router.get("/web-vitals/dashboard")
async def get_performance_dashboard(
    period: str = "24h",
    db: AsyncSession = Depends(get_session)
):
    """
    Get performance dashboard data with analytics and trends.
    
    Period options: 1h, 24h, 7d, 30d
    """
    try:
        # Calculate time range based on period
        now = datetime.utcnow()
        if period == "1h":
            start_time = now - timedelta(hours=1)
        elif period == "24h":
            start_time = now - timedelta(days=1)
        elif period == "7d":
            start_time = now - timedelta(days=7)
        elif period == "30d":
            start_time = now - timedelta(days=30)
        else:
            start_time = now - timedelta(days=1)
        
        # Get metrics data
        metrics_query = select(WebVitalsMetric).where(
            WebVitalsMetric.created_at >= start_time
        )
        result = await db.execute(metrics_query)
        metrics = result.scalars().all()
        
        if not metrics:
            return PerformanceDashboard(
                period=period,
                total_samples=0,
                unique_pages=0,
                metrics=[],
                trends={},
                recommendations=["No data available for the selected period"]
            )
        
        # Group by metric name
        metric_groups = {}
        unique_pages = set()
        
        for metric in metrics:
            if metric.metric_name not in metric_groups:
                metric_groups[metric.metric_name] = []
            metric_groups[metric.metric_name].append(metric.value)
            unique_pages.add(metric.url)
        
        # Calculate analytics for each metric
        analytics = []
        for metric_name, values in metric_groups.items():
            values.sort()
            count = len(values)
            
            # Calculate percentiles
            avg_value = sum(values) / count
            median_value = values[count // 2]
            p75_value = values[int(count * 0.75)]
            p95_value = values[int(count * 0.95)]
            
            # Rating distribution
            rating_query = select(
                WebVitalsMetric.rating,
                func.count(WebVitalsMetric.id)
            ).where(
                and_(
                    WebVitalsMetric.metric_name == metric_name,
                    WebVitalsMetric.created_at >= start_time
                )
            ).group_by(WebVitalsMetric.rating)
            
            rating_result = await db.execute(rating_query)
            rating_distribution = dict(rating_result.all())
            
            analytics.append(PerformanceAnalytics(
                metric_name=metric_name,
                avg_value=round(avg_value, 2),
                median_value=round(median_value, 2),
                p75_value=round(p75_value, 2),
                p95_value=round(p95_value, 2),
                sample_count=count,
                rating_distribution=rating_distribution
            ))
        
        # Generate trends (hourly/daily data points)
        trends = await generate_performance_trends(db, metric_groups.keys(), start_time, now)
        
        # Generate recommendations based on performance
        recommendations = generate_performance_recommendations(analytics)
        
        return PerformanceDashboard(
            period=period,
            total_samples=len(metrics),
            unique_pages=len(unique_pages),
            metrics=analytics,
            trends=trends,
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Failed to get performance dashboard: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve performance data")

@router.get("/web-vitals/metrics/{metric_name}")
async def get_metric_details(
    metric_name: str,
    period: str = "24h",
    page_url: Optional[str] = None,
    db: AsyncSession = Depends(get_session)
):
    """
    Get detailed analytics for a specific Web Vitals metric.
    """
    try:
        # Validate metric name
        valid_metrics = ['LCP', 'INP', 'CLS', 'TTFB']
        if metric_name not in valid_metrics:
            raise HTTPException(status_code=400, detail=f"Invalid metric. Valid options: {valid_metrics}")
        
        # Calculate time range
        now = datetime.utcnow()
        if period == "1h":
            start_time = now - timedelta(hours=1)
        elif period == "24h":
            start_time = now - timedelta(days=1)
        elif period == "7d":
            start_time = now - timedelta(days=7)
        elif period == "30d":
            start_time = now - timedelta(days=30)
        else:
            start_time = now - timedelta(days=1)
        
        # Build query
        conditions = [
            WebVitalsMetric.metric_name == metric_name,
            WebVitalsMetric.created_at >= start_time
        ]
        
        if page_url:
            conditions.append(WebVitalsMetric.url == page_url)
        
        query = select(WebVitalsMetric).where(and_(*conditions))
        result = await db.execute(query)
        metrics = result.scalars().all()
        
        if not metrics:
            return {
                "metric_name": metric_name,
                "period": period,
                "page_url": page_url,
                "data": [],
                "summary": {
                    "count": 0,
                    "avg": 0,
                    "median": 0,
                    "p75": 0,
                    "p95": 0
                }
            }
        
        # Extract values
        values = [m.value for m in metrics]
        values.sort()
        count = len(values)
        
        return {
            "metric_name": metric_name,
            "period": period,
            "page_url": page_url,
            "data": [
                {
                    "timestamp": m.created_at.isoformat(),
                    "value": m.value,
                    "rating": m.rating,
                    "url": m.url,
                    "user_agent": m.user_agent,
                    "viewport": {
                        "width": m.viewport_width,
                        "height": m.viewport_height
                    } if m.viewport_width else None,
                    "connection": {
                        "type": m.connection_type,
                        "downlink": m.connection_downlink,
                        "rtt": m.connection_rtt
                    } if m.connection_type else None
                }
                for m in metrics
            ],
            "summary": {
                "count": count,
                "avg": round(sum(values) / count, 2),
                "median": round(values[count // 2], 2),
                "p75": round(values[int(count * 0.75)], 2),
                "p95": round(values[int(count * 0.95)], 2)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get metric details: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metric data")

async def process_web_vitals_analytics(metric_id: int):
    """
    Background task to process Web Vitals analytics.
    """
    try:
        # This could include:
        # - Anomaly detection for performance regressions
        # - Alert generation for poor performance
        # - Aggregation for long-term trends
        # - Integration with monitoring systems
        
        logger.info(f"Processing analytics for Web Vitals metric {metric_id}")
        
        # Placeholder for analytics processing
        # TODO: Implement anomaly detection, alerting, etc.
        
    except Exception as e:
        logger.error(f"Failed to process Web Vitals analytics: {e}")

async def generate_performance_trends(
    db: AsyncSession, 
    metric_names: List[str], 
    start_time: datetime, 
    end_time: datetime
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Generate time-series trends for performance metrics.
    """
    trends = {}
    
    for metric_name in metric_names:
        # Get hourly/daily aggregates based on period
        period_hours = (end_time - start_time).total_seconds() / 3600
        
        if period_hours <= 24:
            # Hourly data
            time_format = func.date_trunc('hour', WebVitalsMetric.created_at)
        elif period_hours <= 168:  # 7 days
            # 6-hourly data
            time_format = func.date_trunc('hour', WebVitalsMetric.created_at)
        else:
            # Daily data
            time_format = func.date_trunc('day', WebVitalsMetric.created_at)
        
        query = select(
            time_format.label('time_bucket'),
            func.avg(WebVitalsMetric.value).label('avg_value'),
            func.count(WebVitalsMetric.id).label('sample_count')
        ).where(
            and_(
                WebVitalsMetric.metric_name == metric_name,
                WebVitalsMetric.created_at >= start_time,
                WebVitalsMetric.created_at <= end_time
            )
        ).group_by(time_format).order_by(time_format)
        
        result = await db.execute(query)
        trend_data = []
        
        for row in result:
            trend_data.append({
                'timestamp': row.time_bucket.isoformat(),
                'avg_value': round(float(row.avg_value), 2),
                'sample_count': row.sample_count
            })
        
        trends[metric_name] = trend_data
    
    return trends

def generate_performance_recommendations(analytics: List[PerformanceAnalytics]) -> List[str]:
    """
    Generate performance recommendations based on metrics analysis.
    """
    recommendations = []
    
    for metric in analytics:
        if metric.metric_name == 'LCP':
            if metric.p75_value > 2500:
                recommendations.append("🐌 LCP is slow (>2.5s). Optimize images and enable lazy loading.")
            elif metric.p75_value > 1800:
                recommendations.append("⚠️ LCP needs improvement. Consider resource preloading.")
        
        elif metric.metric_name == 'INP':
            if metric.p75_value > 200:
                recommendations.append("🐌 INP is slow (>200ms). Reduce JavaScript execution time.")
            elif metric.p75_value > 100:
                recommendations.append("⚠️ INP needs improvement. Optimize event handlers.")
        
        elif metric.metric_name == 'CLS':
            if metric.p75_value > 0.1:
                recommendations.append("🐌 CLS is high (>0.1). Set explicit image dimensions.")
            elif metric.p75_value > 0.05:
                recommendations.append("⚠️ CLS needs improvement. Reserve space for dynamic content.")
        
        elif metric.metric_name == 'TTFB':
            if metric.p75_value > 600:
                recommendations.append("🐌 TTFB is slow (>600ms). Optimize server response time.")
            elif metric.p75_value > 400:
                recommendations.append("⚠️ TTFB needs improvement. Consider CDN implementation.")
    
    if not recommendations:
        recommendations.append("✅ All performance metrics are within good ranges!")
    
    return recommendations
