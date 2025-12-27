from pydantic import BaseSettings


class Settings(BaseSettings):
    database_url: str = "sqlite+aiosqlite:///./data/vital.db"
    redis_url: str = "redis://localhost:6379/0"
    secret_key: str = "please-change-me"
    debug: bool = True
    CORS_ORIGINS: list[str] = ["*"]

    class Config:
        env_file = ".env"


settings = Settings()
