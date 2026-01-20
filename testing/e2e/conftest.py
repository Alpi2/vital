"""
Simplified pytest-bdd configuration for basic testing
"""

import pytest
import uuid
import time
from typing import AsyncGenerator
from httpx import AsyncClient

# Test configuration
TEST_CONFIG = {
    "backend_url": "http://localhost:8001",
}

@pytest.fixture
async def test_tenant_id() -> str:
    """Generate unique tenant ID for test isolation"""
    return f"test-tenant-{uuid.uuid4().hex[:8]}"

@pytest.fixture
async def http_client() -> AsyncGenerator[AsyncClient, None]:
    """Create HTTP client for testing"""
    async with AsyncClient() as client:
        yield client

@pytest.fixture
async def authenticated_http_client(http_client: AsyncClient, test_tenant_id: str) -> AsyncClient:
    """Create authenticated HTTP client"""
    # Mock authentication - in real implementation, this would login
    http_client.headers["Authorization"] = "Bearer mock-token"
    http_client.headers["X-Tenant-ID"] = test_tenant_id
    return http_client

@pytest.fixture
async def sample_patient_data(test_tenant_id: str) -> dict:
    """Generate sample patient data for testing"""
    return {
        "id": f"patient-{test_tenant_id}",
        "first_name": "Test",
        "last_name": "Patient",
        "date_of_birth": "1980-01-01",
        "gender": "M",
        "medical_record_number": f"MRN-{test_tenant_id}",
        "tenant_id": test_tenant_id
    }

@pytest.fixture
async def sample_ecg_data() -> dict:
    """Generate sample ECG data for testing"""
    import numpy as np
    
    # Generate synthetic ECG data
    sampling_rate = 360
    duration = 10
    n_samples = sampling_rate * duration
    t = np.linspace(0, duration, n_samples)
    
    # Simple ECG-like signal
    ecg_signal = np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.randn(n_samples)
    
    return {
        "samples": ecg_signal.tolist(),
        "sampling_rate": sampling_rate,
        "duration": duration,
        "quality_score": 0.9
    }
