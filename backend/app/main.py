from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from datetime import datetime

from .config import settings
from .database import engine, Base
from .api import patients, anomalies, reports, auth


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create DB tables (run sync metadata.create_all in async context)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("Database tables created")
    yield
    # Shutdown
    print("Shutting down...")


app = FastAPI(
    title="VitalStream API",
    description="Real-time ECG Analysis Backend Service",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(patients.router, prefix="/api/patients", tags=["Patients"])
app.include_router(anomalies.router, prefix="/api/anomalies", tags=["Anomalies"])
app.include_router(reports.router, prefix="/api/reports", tags=["Reports"])


@app.get("/")
async def root():
    return {
        "message": "VitalStream API",
        "version": "1.0.0",
        "docs": "/api/docs",
        "health": "/api/health"
    }


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}
from fastapi import FastAPI
from .api import patients, anomalies

app = FastAPI(title="vitalstream-backend")


@app.get("/health")
async def health():
    return {"status": "ok"}


# include routers if available
try:
    app.include_router(patients.router, prefix="/api/patients")
except Exception:
    pass

try:
    app.include_router(anomalies.router, prefix="/api/anomalies")
except Exception:
    pass
from fastapi import FastAPI

app = FastAPI(title="vitalstream-backend")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/add")
async def add(a: int = 1, b: int = 2):
    return {"result": a + b}
