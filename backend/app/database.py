from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker as sync_sessionmaker
from .config import settings


# Async engine (used for async tasks/migrations)
# Configure connection pool for PostgreSQL
engine_kwargs = {
    "echo": settings.debug,
    "future": True,
}

# Add connection pool settings for PostgreSQL
if "postgresql" in settings.database_url:
    engine_kwargs.update({
        "pool_pre_ping": True,  # Verify connections before using
        "pool_size": settings.db_pool_size,
        "max_overflow": settings.db_max_overflow,
        "pool_timeout": settings.db_pool_timeout,
        "pool_recycle": settings.db_pool_recycle,
    })

engine = create_async_engine(settings.database_url, **engine_kwargs)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# Sync engine + session (convenience for sync CRUD endpoints/tests)
# Convert async sqlite URL to sync sqlite if needed
sync_db_url = settings.database_url
if sync_db_url.startswith("sqlite+aiosqlite:///"):
    sync_db_url = sync_db_url.replace("sqlite+aiosqlite:///", "sqlite:///")
elif "+asyncpg" in sync_db_url:
    sync_db_url = sync_db_url.replace("+asyncpg", "")

sync_engine = create_engine(
    sync_db_url,
    connect_args={"check_same_thread": False} if "sqlite" in sync_db_url else {},
)
SyncSessionLocal = sync_sessionmaker(bind=sync_engine, autoflush=False, autocommit=False)

Base = declarative_base()


async def get_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session


def get_db():
    """Synchronous DB session generator for legacy sync endpoints."""
    db = SyncSessionLocal()
    try:
        yield db
    finally:
        db.close()
