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
        debug: bool = False
        CORS_ORIGINS: list[str] = ["http://localhost:4200"]

        class Config:
            env_file = ".env"

    settings = Settings()

except Exception:  # pragma: no cover - fallback for environments without pydantic-settings
    import os

    class Settings:
        def __init__(self):
            self.database_url = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///./data/vital.db")
            self.redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
            self.secret_key = os.environ.get("SECRET_KEY", "")
            self.debug = os.environ.get("DEBUG", "False").lower() in ("1", "true", "yes")
            self.CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:4200").split(",")

    settings = Settings()
