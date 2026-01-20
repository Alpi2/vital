"""
pytest-bdd configuration with proper fixtures and data isolation
Addresses critical issues from original plan
"""

import asyncio
import pytest
import uuid
import time
from typing import AsyncGenerator, Dict, Any
from pathlib import Path
import docker
from docker.models.containers import Container
import httpx
import grpc
import aredis
import asyncpg
from factory.alchemy import SQLAlchemyModelFactory

# Test configuration
TEST_CONFIG = {
    "database_url": "postgresql://test_user:test_password@localhost:5433/vitalstream_test",
    "redis_url": "redis://localhost:6380/0",
    "backend_url": "http://localhost:8001",
    "alarm_engine_grpc": "localhost:50052",
    "dicom_service_grpc": "localhost:50053",
    "websocket_url": "ws://localhost:8002",
    "frontend_url": "http://localhost:3001"
}

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session", autouse=True)
async def setup_test_environment():
    """Setup test environment using Docker Compose"""
    import subprocess
    
    # Start test environment
    compose_file = Path(__file__).parent / "docker-compose.test.yml"
    
    try:
        # Start services
        subprocess.run([
            "docker-compose", "-f", str(compose_file), 
            "up", "-d", "--build"
        ], check=True, timeout=300)
        
        # Wait for services to be healthy
        await wait_for_services()
        
        yield
        
    finally:
        # Cleanup
        subprocess.run([
            "docker-compose", "-f", str(compose_file), 
            "down", "-v", "--remove-orphans"
        ], check=False)

async def wait_for_services(timeout: int = 120):
    """Wait for all services to be healthy"""
    import aiohttp
    
    async with aiohttp.ClientSession() as session:
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Check backend
                async with session.get(f"{TEST_CONFIG['backend_url']}/health", timeout=5) as resp:
                    if resp.status == 200:
                        break
            except:
                pass
            
            await asyncio.sleep(2)
        else:
            raise TimeoutError("Services not ready within timeout")

@pytest.fixture
async def test_tenant_id() -> str:
    """Generate unique tenant ID for test isolation"""
    return f"test-tenant-{uuid.uuid4().hex[:8]}"

@pytest.fixture
async def http_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Create HTTP client for API testing"""
    async with httpx.AsyncClient(
        base_url=TEST_CONFIG["backend_url"],
        timeout=30.0
    ) as client:
        yield client

@pytest.fixture
async def authenticated_http_client(http_client: httpx.AsyncClient, test_tenant_id: str) -> httpx.AsyncClient:
    """Create authenticated HTTP client"""
    # Login and get token
    login_data = {
        "username": f"doctor_{test_tenant_id}",
        "password": "test_password",
        "tenant_id": test_tenant_id
    }
    
    response = await http_client.post("/api/v1/auth/login", json=login_data)
    assert response.status_code == 200
    
    token = response.json()["access_token"]
    http_client.headers["Authorization"] = f"Bearer {token}"
    http_client.headers["X-Tenant-ID"] = test_tenant_id
    
    return http_client

@pytest.fixture
async def grpc_channel(test_tenant_id: str) -> AsyncGenerator[grpc.aio.Channel, None]:
    """Create gRPC channel for testing"""
    channel = grpc.aio.insecure_channel(TEST_CONFIG["alarm_engine_grpc"])
    
    # Add tenant metadata
    metadata = [("tenant-id", test_tenant_id)]
    
    yield channel
    
    await channel.close()

@pytest.fixture
async def websocket_client(test_tenant_id: str) -> AsyncGenerator[Any, None]:
    """Create WebSocket client for testing"""
    import websockets
    
    uri = f"{TEST_CONFIG['websocket_url']}/ws/notifications?tenant_id={test_tenant_id}"
    
    async with websockets.connect(uri) as websocket:
        yield websocket

@pytest.fixture
async def redis_client() -> AsyncGenerator[aredis.Redis, None]:
    """Create Redis client for testing"""
    client = aredis.Redis.from_url(TEST_CONFIG["redis_url"])
    
    # Clear test data before each test
    await client.flushdb()
    
    yield client
    
    await client.close()

@pytest.fixture
async def database_connection(test_tenant_id: str) -> AsyncGenerator[asyncpg.Connection, None]:
    """Create database connection with tenant isolation"""
    conn = await asyncpg.connect(TEST_CONFIG["database_url"])
    
    # Set tenant context
    await conn.execute("SET app.current_tenant_id = $1", test_tenant_id)
    
    yield conn
    
    await conn.close()

@pytest.fixture
async def test_database_cleanup(database_connection: asyncpg.Connection, test_tenant_id: str):
    """Cleanup test data after each test"""
    yield
    
    # Clean up only data for this tenant
    tables = [
        "ecg_analysis_results", "patient_data", "anomaly_alerts",
        "audit_logs", "user_sessions", "test_results"
    ]
    
    for table in tables:
        await database_connection.execute(
            f"DELETE FROM {table} WHERE tenant_id = $1", test_tenant_id
        )

@pytest.fixture
async def sample_patient_data(test_tenant_id: str) -> Dict[str, Any]:
    """Generate sample patient data for testing"""
    return {
        "id": f"patient-{test_tenant_id}-{uuid.uuid4().hex[:8]}",
        "first_name": "Test",
        "last_name": f"Patient-{test_tenant_id[:8]}",
        "date_of_birth": "1980-01-01",
        "gender": "M",
        "tenant_id": test_tenant_id,
        "medical_record_number": f"MRN-{test_tenant_id[:8].upper()}"
    }

@pytest.fixture
async def sample_ecg_data() -> Dict[str, Any]:
    """Generate synthetic ECG data for testing"""
    import numpy as np
    
    # Generate 10 seconds of ECG at 360 Hz
    sampling_rate = 360
    duration = 10
    n_samples = sampling_rate * duration
    
    # Generate synthetic ECG with noise
    t = np.linspace(0, duration, n_samples)
    ecg_signal = (
        np.sin(2 * np.pi * 1.2 * t) +  # Heartbeat ~72 bpm
        0.2 * np.sin(2 * np.pi * 2.4 * t) +  # Harmonic
        0.1 * np.random.normal(0, 1, n_samples)  # Noise
    )
    
    # Add some R-peaks
    peak_indices = np.arange(0, n_samples, sampling_rate // 72)
    for idx in peak_indices:
        if idx < n_samples:
            ecg_signal[idx] += 1.5
    
    return {
        "samples": ecg_signal.tolist(),
        "sampling_rate": sampling_rate,
        "duration": duration,
        "patient_id": None,  # Will be set in test
        "quality_score": 0.85
    }

@pytest.fixture
async def sample_anomaly_data(test_tenant_id: str) -> Dict[str, Any]:
    """Generate sample anomaly data for testing"""
    return {
        "id": f"anomaly-{test_tenant_id}-{uuid.uuid4().hex[:8]}",
        "type": "arrhythmia",
        "severity": "medium",
        "confidence": 0.87,
        "timestamp": time.time(),
        "patient_id": None,  # Will be set in test
        "tenant_id": test_tenant_id,
        "metadata": {
            "heart_rate": 95,
            "rhythm": "irregular",
            "duration": 3.2
        }
    }

# Visual testing fixtures
@pytest.fixture
async def browser_context():
    """Create browser context for visual/accessibility testing"""
    from playwright.async_api import async_playwright
    
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        context = await browser.new_context(
            viewport={"width": 1280, "height": 720},
            ignore_https_errors=True
        )
        yield context
        await context.close()
        await browser.close()

@pytest.fixture
async def page(browser_context):
    """Create page for testing"""
    page = await browser_context.new_page()
    await page.goto(TEST_CONFIG["frontend_url"])
    yield page
    await page.close()

# Performance testing fixtures
@pytest.fixture
async def lighthouse_runner():
    """Create Lighthouse CI runner for performance testing"""
    import subprocess
    import json
    
    lighthouse_config = {
        "ci": {
            "collect": {
                "numberOfRuns": 3,
                "settings": {
                    "chromeFlags": "--headless"
                }
            },
            "assert": {
                "assertions": {
                    "categories:performance": ["warn", {"minScore": 0.8}],
                    "categories:accessibility": ["error", {"minScore": 0.9}],
                    "categories:best-practices": ["warn", {"minScore": 0.8}],
                    "categories:seo": ["off"]
                }
            },
            "upload": {
                "target": "temporary-public-storage"
            }
        }
    }
    
    return lighthouse_config

# gRPC testing fixtures
@pytest.fixture
async def alarm_engine_stub(grpc_channel: grpc.aio.Channel):
    """Create alarm engine gRPC stub"""
    # Import generated protobuf classes
    from alarm_engine_pb2_grpc import AlarmEngineServiceStub
    
    return AlarmEngineServiceStub(grpc_channel)

@pytest.fixture
async def dicom_service_stub(grpc_channel: grpc.aio.Channel):
    """Create DICOM service gRPC stub"""
    # Import generated protobuf classes
    from dicom_service_pb2_grpc import DICOMServiceStub
    
    return DICOMServiceStub(grpc_channel)
