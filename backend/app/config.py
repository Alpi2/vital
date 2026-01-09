# Try to use pydantic BaseSettings when available, otherwise fall back to a
# small, explicit environment-based Settings implementation so tests and
# lightweight environments don't need an extra dependency.
try:
    from pydantic_settings import BaseSettings
    from pydantic import Field

    class Settings(BaseSettings):
        database_url: str = "sqlite+aiosqlite:///./data/vital.db"
        redis_url: str = "redis://localhost:6379/0"
        secret_key: str = Field(..., env="SECRET_KEY")
        algorithm: str = "HS256"
        access_token_expire_minutes: int = 15
        refresh_token_expire_days: int = 7
        debug: bool = False
        CORS_ORIGINS: list[str] = ["http://localhost:4200"]
        sentry_dsn: str | None = None
        environment: str = "development"
        
        # Database connection pool settings
        db_pool_size: int = 10
        db_max_overflow: int = 20
        db_pool_timeout: int = 30
        db_pool_recycle: int = 3600

        class Config:
            env_file = ".env"

    settings = Settings()

except Exception:  # pragma: no cover - fallback for environments without pydantic-settings
    import os

    class Settings:
        def __init__(self):
            self.database_url = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///./data/vital.db")
            self.redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
            # Require SECRET_KEY to be explicitly set — fail-fast to avoid insecure defaults
            secret_key = os.environ.get("SECRET_KEY")
            if not secret_key:
                raise ValueError("SECRET_KEY environment variable must be set")
            self.secret_key = secret_key
            self.algorithm = "HS256"
            self.access_token_expire_minutes = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", "15"))
            self.refresh_token_expire_days = int(os.environ.get("REFRESH_TOKEN_EXPIRE_DAYS", "7"))
            self.debug = os.environ.get("DEBUG", "False").lower() in ("1", "true", "yes")
            self.CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:4200").split(",")
            self.sentry_dsn = os.environ.get("SENTRY_DSN")
            self.environment = os.environ.get("ENVIRONMENT", "development")
            
            # Database connection pool settings
            self.db_pool_size = int(os.environ.get("DB_POOL_SIZE", "10"))
            self.db_max_overflow = int(os.environ.get("DB_MAX_OVERFLOW", "20"))
            self.db_pool_timeout = int(os.environ.get("DB_POOL_TIMEOUT", "30"))
            self.db_pool_recycle = int(os.environ.get("DB_POOL_RECYCLE", "3600"))

    settings = Settings()
