from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    # Use a local sqlite DB by default for development
    database_url: str = "sqlite+aiosqlite:///./data/vital.db"
    redis_url: str = "redis://localhost:6379/0"

    # SECRET_KEY is required and must be supplied via environment variable
    # (set SECRET_KEY in .env or your deployment environment)
    secret_key: str = Field(..., env="SECRET_KEY")

    # Default to safe production-like behavior; set DEBUG=True in .env for local dev
    debug: bool = False

    # Restrict default CORS origins to local frontend during development
    CORS_ORIGINS: list[str] = ["http://localhost:4200"]

    class Config:
        env_file = ".env"


settings = Settings()
