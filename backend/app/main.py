from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
from pythonjsonlogger import jsonlogger
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from .config import settings
from .database import engine, Base, get_session
from sqlalchemy.ext.asyncio import AsyncSession
from .api import patients, anomalies, reports, auth, websocket_routes, sse_routes, dashboard, admin, localization
from .limiter import limiter
from slowapi import _rate_limit_exceeded_handler
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi.errors import RateLimitExceeded
from .middleware import (
    SecurityHeadersMiddleware,
    RequestLoggingMiddleware,
    GlobalExceptionHandler,
)
from .metrics import set_app_info


# Configure structured JSON logging
logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter('%(asctime)s %(levelname)s %(name)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

app = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure data directory exists (used by sqlite file path)
    Path("data").mkdir(parents=True, exist_ok=True)

    # Validate critical environment settings at startup
    if not getattr(settings, "secret_key", None):
        raise RuntimeError("SECRET_KEY must be set")
    if getattr(settings, "database_url", "").startswith("sqlite") and os.getenv("PRODUCTION"):
        raise RuntimeError("SQLite not allowed in production")

    # Startup: create DB tables (run sync metadata.create_all in async context)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        logging.getLogger(__name__).info("Database tables created")
    
    # Set application info for Prometheus
    set_app_info(
        version="1.0.0",
        environment=getattr(settings, 'environment', 'development')
    )
    
    yield
    # Shutdown
        logging.getLogger(__name__).info("Shutting down...")


app = FastAPI(
    title="VitalStream API",
    description="""# VitalStream API

## Overview
Real-time ECG Analysis Backend Service for monitoring patient vital signs and detecting anomalies.

## Features
- 👨‍⚕️ **Patient Management**: Create and manage patient records
- 📈 **Real-time ECG Analysis**: Process and analyze ECG data using WebAssembly
- ⚠️ **Anomaly Detection**: Detect and track cardiac anomalies
- 📊 **Reports**: Generate comprehensive PDF reports
- 🔒 **Authentication**: JWT-based authentication with refresh tokens
- 🔌 **Real-time Streaming**: WebSocket and SSE support for live data
- 📊 **Monitoring**: Prometheus metrics and Grafana dashboards

## Authentication
Most endpoints require JWT authentication. Include the token in the Authorization header:
```
Authorization: Bearer <your_token>
```

## Rate Limiting
API endpoints are rate-limited to prevent abuse:
- Default: 100 requests per minute
- Authentication endpoints: 5 requests per minute

## WebSocket & SSE
For real-time data streaming, use:
- WebSocket: `ws://localhost:8000/api/v1/ws/ecg/{patient_id}`
- SSE: `http://localhost:8000/api/v1/sse/ecg/{patient_id}`

## Support
For issues and questions, visit: https://github.com/yourusername/vitalstream
    """,
    version="1.0.0",
    openapi_tags=[
        {
            "name": "Authentication",
            "description": """JWT-based authentication endpoints.
            
**Features:**
- Login with username/password
- Access token + refresh token
- Token refresh mechanism
- Logout with token blacklist
- Get current user info

**Security:**
- Passwords hashed with bcrypt
- Tokens expire after configured time
- Refresh tokens for extended sessions
- Redis-based token blacklist
            """
        },
        {
            "name": "Patients",
            "description": """Patient record management.
            
**Operations:**
- Create new patient records
- List all patients with filtering
- Get patient details by ID
- Update patient information
- Delete patient records

**Data:**
- Patient demographics (name, age, gender)
- Medical record number
- Contact information
- Created/updated timestamps
            """
        },
        {
            "name": "Anomalies",
            "description": """Anomaly detection and tracking.
            
**Features:**
- Record detected anomalies
- Filter by patient, severity, type
- Get anomaly statistics
- Resolve anomalies
- Track anomaly history

**Anomaly Types:**
- Arrhythmia
- Tachycardia
- Bradycardia
- Irregular rhythm
- Custom types

**Severity Levels:**
- Critical
- Warning
- Info
            """
        },
        {
            "name": "Reports",
            "description": """PDF report generation.
            
**Features:**
- Generate patient session reports
- Include ECG charts and statistics
- Anomaly summaries
- Downloadable PDF format

**Report Contents:**
- Patient information
- Session details
- ECG waveform visualization
- Detected anomalies
- Statistical analysis
- Timestamps and metadata
            """
        },
        {
            "name": "WebSocket",
            "description": """Real-time WebSocket streaming.
            
**Endpoints:**
- `/ws/ecg/{patient_id}` - Patient-specific ECG stream
- `/ws/monitor` - Monitor all patients
- `/ws/alerts` - Real-time alerts

**Features:**
- Bidirectional communication
- Auto-reconnection
- Ping/pong keep-alive
- Message type routing

**Message Types:**
- `ecg_data` - Real-time ECG points
- `anomaly_detected` - Anomaly events
- `critical_alert` - Critical alerts
- `ping/pong` - Keep-alive
            """
        },
        {
            "name": "SSE",
            "description": """Server-Sent Events streaming.
            
**Endpoints:**
- `/sse/ecg/{patient_id}` - Patient ECG stream
- `/sse/monitor` - Monitor all patients
- `/sse/alerts` - Alert stream
- `/sse/stats` - Statistics

**Features:**
- Server-to-client streaming
- Automatic reconnection
- Heartbeat mechanism
- Event type filtering

**Event Types:**
- `connected` - Connection established
- `heartbeat` - Keep-alive
- `ecg_data` - ECG data points
- `anomaly_detected` - Anomalies
- `critical_alert` - Critical alerts
            """
        },
    ],
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    contact={
        "name": "VitalStream Support",
        "url": "https://github.com/yourusername/vitalstream",
        "email": "support@vitalstream.com"
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    }
)

# Initialize limiter with the FastAPI app and register handler
limiter.init_app(app)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Instrument application with Prometheus metrics
Instrumentator().instrument(app).expose(app)

# Add custom middleware
app.add_middleware(GlobalExceptionHandler)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(SecurityHeadersMiddleware)

# CORS middleware (should be last to ensure it wraps all other middleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(patients.router, prefix="/api/v1/patients", tags=["Patients"])
app.include_router(anomalies.router, prefix="/api/v1/anomalies", tags=["Anomalies"])
app.include_router(reports.router, prefix="/api/v1/reports", tags=["Reports"])
app.include_router(websocket_routes.router, prefix="/api/v1", tags=["WebSocket"])
app.include_router(sse_routes.router, prefix="/api/v1", tags=["SSE"])
app.include_router(dashboard.router, prefix="/api/v1", tags=["Dashboard"])
# Note: Dashboard endpoints converted from Flask (bi/api/dashboard_endpoints.py)
# Now using FastAPI async with SQLAlchemy async sessions
app.include_router(admin.router, prefix="/api/v1", tags=["Admin"])
app.include_router(localization.router, prefix="/api/v1", tags=["Localization"])


@app.get("/")
async def root():
    return {
        "message": "VitalStream API",
        "version": "1.0.0",
        "docs": "/api/docs",
        "health": "/api/health"
    }


@app.get("/api/health")
async def health_check(db: AsyncSession = Depends(get_session)):
    try:
        # simple lightweight check
        await db.execute("SELECT 1")
        db_status = "healthy"
    except Exception:
        db_status = "unhealthy"

    return {
        "status": "healthy" if db_status == "healthy" else "degraded",
        "timestamp": datetime.utcnow(),
        "database": db_status,
        "version": "1.0.0"
    }
