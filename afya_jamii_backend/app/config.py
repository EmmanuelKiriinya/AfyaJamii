import os
import secrets
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
from dotenv import load_dotenv

# Load .env first so os.getenv can pick up values if needed
load_dotenv()

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        secrets_dir="/run/secrets"
    )

    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Afya Jamii AI"

    # Security
    SECRET_KEY: str = "change-this-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # MySQL Database Configuration (all come from .env)
    DATABASE_URL: Optional[str] = None
    DATABASE_HOST: str
    DATABASE_PORT: int
    DATABASE_NAME: str
    DATABASE_USER: str
    DB_PASSWORD: str

    # Connection Pool
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 30
    DB_POOL_RECYCLE: int = 3600
    DB_ECHO: bool = False

    # ML Model
    MODEL_PATH: str = "./data/risk_model_v1.pkl"

    # Groq LLM
    GROQ_API_KEY: str
    LLM_MODEL_NAME: str = "openai/gpt-oss-20b"
    LLM_TEMPERATURE: float = 0.0

    # App Environment
    ENVIRONMENT: str = "production"
    DEBUG: bool = True
    LOG_LEVEL: str = DEBUG

    # CORS
    CORS_ORIGINS: list[str] = [
        "https://127.0.0.1:8000",
        "http://localhost:8000"]

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = True

    # Security Headers
    CSP_DIRECTIVES: str = "default-src 'self'; script-src 'self' 'unsafe-inline'"

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    MAX_REQUESTS_PER_USER: int = 1000

    # --- Validators ---
    @field_validator("SECRET_KEY", mode="before")
    def validate_secret_key(cls, v):
        if v == "change-this-in-production":
            return secrets.token_urlsafe(32)
        return v

    @field_validator("DATABASE_URL", mode="before")
    def assemble_db_connection(cls, v, info):
        # If DATABASE_URL is missing, build it from individual pieces
        if v:
            return v
        data = info.data
        return (
            f"mysql+pymysql://{data['DATABASE_USER']}:{data['DB_PASSWORD']}@"
            f"{data['DATABASE_HOST']}:{data['DATABASE_PORT']}/{data['DATABASE_NAME']}"
        )


settings = Settings()
