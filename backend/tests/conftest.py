...existing code...
"""Pytest configuration and fixtures."""

import pytest
import asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.database import Base, get_session
from app.models import Patient, ECGSession, AnomalyLog


# Test database URL (in-memory SQLite)
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
async def engine():
    """Create a test database engine."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()


@pytest.fixture(scope="function")
async def db_session(engine):
    """Create a test database session."""
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session


@pytest.fixture(scope="function")
async def client(db_session):
    """Create a test client with database session override."""
    
    async def override_get_session():
        yield db_session
    
    app.dependency_overrides[get_session] = override_get_session
    
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac
    
    app.dependency_overrides.clear()


# Backwards-compatible alias used in tests
@pytest.fixture(scope="function")
async def async_client(client):
    return client


# User fixtures for auth tests
@pytest.fixture
async def demo_user(db_session):
    """Create a demo user with username 'doctor'."""
    from app.models.user import User
    from datetime import datetime

    user = User(
        username="doctor",
        email="doctor@example.com",
        role="doctor",
        is_demo=True,
        is_active=True,
    )
    # set a past last_login to assert update
    user.last_login = datetime(2000, 1, 1)
    user.set_password("DemoPass123!")

    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest.fixture
async def demo_users(db_session):
    """Create several demo users (for demo-login by role)."""
    from app.models.user import User

    users = []
    for i in range(2):
        u = User(
            username=f"demo_{i}",
            email=f"demo_{i}@example.com",
            role="doctor",
            is_demo=True,
            is_active=True,
        )
        u.set_password("irrelevant")
        db_session.add(u)
        users.append(u)

    await db_session.commit()
    for u in users:
        await db_session.refresh(u)
    return users


@pytest.fixture
async def real_user(db_session):
    """Create a real non-demo user for authentication tests."""
    from app.models.user import User

    user = User(
        username="realuser",
        email="realuser@example.com",
        role="nurse",
        is_demo=False,
        is_active=True,
    )
    user.set_password("SecurePass123!")

    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest.fixture
async def inactive_user(db_session):
    """Create an inactive user."""
    from app.models.user import User

    user = User(
        username="inactive",
        email="inactive@example.com",
        role="nurse",
        is_demo=False,
        is_active=False,
    )
    user.set_password("password")

    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest.fixture
async def sample_patient(db_session):
    """Create a sample patient for testing."""
    from datetime import date
    
    patient = Patient(
        medical_id="TEST-001",
        first_name="John",
        last_name="Doe",
        date_of_birth=date(1980, 1, 1),
        gender="M",
        height=180,
        weight=75,
        blood_type="A+",
    )
    
    db_session.add(patient)
    await db_session.commit()
    await db_session.refresh(patient)
    
    return patient


@pytest.fixture
async def sample_session(db_session, sample_patient):
    """Create a sample ECG session for testing."""
    from datetime import datetime
    
    session = ECGSession(
        patient_id=sample_patient.id,
        session_id=f"session_{datetime.utcnow().timestamp()}",
        duration=300.0,
        sample_rate=360,
        data_points=108000,
        average_bpm=72.5,
        min_bpm=60.0,
        max_bpm=85.0,
        anomaly_count=0,
    )
    
    db_session.add(session)
    await db_session.commit()
    await db_session.refresh(session)
    
    return session
