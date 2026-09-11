"""Application configuration.

Every value is read from the environment (or a local ``.env`` file) so that the
same image can be promoted through development, staging, and production without
a rebuild. See ``.env.example`` for the full list with commentary.

Settings are validated once at import time. A misconfigured deployment fails
here, at start-up, rather than on the first request that happens to need the
missing value.
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import Field, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

# Placeholder values that appear in .env.example. Treat them as "unset" so an
# operator who copies the example without editing it gets a clear error.
PLACEHOLDER_VALUES = {
    "change-this-in-production",
    "your-groq-api-key-here",
    "replace-me",
}


# Query parameters that belong to the MySQL command-line client rather than to
# the driver. SQLAlchemy forwards anything it does not recognise straight to
# PyMySQL, which then raises TypeError on these.
CLIENT_ONLY_QUERY_KEYS = {"ssl-mode", "sslmode", "ssl_mode"}


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ── Application ────────────────────────────────────────────────────────
    PROJECT_NAME: str = "Afya Jamii AI"
    API_V1_STR: str = "/api/v1"
    VERSION: str = "1.0.0"
    ENVIRONMENT: Environment = Environment.PRODUCTION
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "text"  # "text" for humans, "json" for log aggregators

    # ── Server ─────────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = False

    # ── Security ───────────────────────────────────────────────────────────
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    BCRYPT_ROUNDS: int = 12

    # Hosts this API will answer to. "*" is only permitted outside production.
    ALLOWED_HOSTS: str = "*"

    # ── CORS ───────────────────────────────────────────────────────────────
    # Comma-separated list, e.g. "https://afyajamii.co.ke,http://localhost:8080"
    CORS_ORIGINS: str = "http://localhost:8080"

    # ── Database ───────────────────────────────────────────────────────────
    DATABASE_URL: Optional[str] = None
    DATABASE_HOST: str = "localhost"
    DATABASE_PORT: int = 3306
    DATABASE_NAME: str = "afya_jamii"
    DATABASE_USER: str = "afya_jamii"
    DB_PASSWORD: str = ""

    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 30
    DB_POOL_RECYCLE: int = 3600
    DB_POOL_TIMEOUT: int = 30
    DB_ECHO: bool = False

    # Transport security for the database connection.
    #
    #   verify   encrypted, certificate checked against DB_SSL_CA (strongest)
    #   require  encrypted, certificate not checked (stops passive listening)
    #   disable  plaintext — local development only
    #
    # The default is `require` rather than `disable` deliberately: a managed
    # MySQL reached over the public internet will happily accept an
    # unencrypted connection, and patient vitals must not travel that way just
    # because nobody configured TLS.
    DB_SSL_MODE: str = "require"
    DB_SSL_CA: Optional[str] = None

    # ── Machine learning model ─────────────────────────────────────────────
    MODEL_PATH: str = "./data/risk_model_v1.pkl"
    LABEL_ENCODER_PATH: str = "./data/risk_label_encoder.pkl"

    # ── Language model ─────────────────────────────────────────────────────
    GROQ_API_KEY: str
    LLM_MODEL_NAME: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    LLM_TEMPERATURE: float = Field(default=0.2, ge=0.0, le=2.0)
    LLM_MAX_TOKENS: int = Field(default=1500, gt=0)
    LLM_TIMEOUT_SECONDS: int = Field(default=45, gt=0)
    LLM_MAX_RETRIES: int = Field(default=2, ge=0)

    # ── Prompts ────────────────────────────────────────────────────────────
    # Directory holding the prompt JSON files, and the prompt to use for chat.
    PROMPT_DIR: Optional[str] = None
    PROMPT_NAME: str = "clinical_assistant"
    PROMPT_RELOAD_ON_CHANGE: bool = False

    # How many past turns to replay into the model as conversation history.
    CHAT_HISTORY_TURNS: int = Field(default=10, ge=0, le=100)

    # Total characters of replayed history allowed in a prompt, and the cap on
    # any single past reply. Without these the prompt grows with every exchange
    # — the model's answers run to several thousand characters each, so ten
    # turns reached ~40k characters and each question took longer than the one
    # before it.
    CHAT_HISTORY_CHAR_BUDGET: int = Field(default=6000, ge=0)
    CHAT_HISTORY_REPLY_CHARS: int = Field(default=700, ge=100)

    # ── Rate limiting ──────────────────────────────────────────────────────
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_DEFAULT: str = "60/minute"
    RATE_LIMIT_LOGIN: str = "5/minute"
    RATE_LIMIT_SIGNUP: str = "10/minute"
    RATE_LIMIT_INFERENCE: str = "20/minute"

    # ── Security headers ───────────────────────────────────────────────────
    CSP_DIRECTIVES: str = (
        "default-src 'self'; script-src 'self'; object-src 'none'; frame-ancestors 'none'"
    )
    HSTS_MAX_AGE: int = 63072000

    # ── Error reporting ────────────────────────────────────────────────────
    SENTRY_DSN: Optional[str] = None

    # ── Validators ─────────────────────────────────────────────────────────

    @field_validator("SECRET_KEY")
    @classmethod
    def validate_secret_key(cls, value: str) -> str:
        """Reject placeholder or weak signing keys.

        The key must be stable across restarts and across workers: generating a
        random one at boot would silently invalidate every issued token on
        deploy, and would hand different keys to each Gunicorn worker.
        """
        if value in PLACEHOLDER_VALUES:
            raise ValueError(
                "SECRET_KEY is still set to a placeholder. Generate one with:\n"
                "  python -c 'import secrets; print(secrets.token_urlsafe(48))'"
            )
        if len(value) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters long")
        return value

    @field_validator("GROQ_API_KEY")
    @classmethod
    def validate_groq_key(cls, value: str) -> str:
        if not value or value in PLACEHOLDER_VALUES:
            raise ValueError(
                "GROQ_API_KEY is not configured. Obtain a key from https://console.groq.com "
                "and set it in the environment."
            )
        return value

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        level = value.upper()
        if not isinstance(logging.getLevelName(level), int):
            raise ValueError(
                f"LOG_LEVEL must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL (got {value!r})"
            )
        return level

    @field_validator("LOG_FORMAT")
    @classmethod
    def validate_log_format(cls, value: str) -> str:
        fmt = value.lower()
        if fmt not in {"text", "json"}:
            raise ValueError("LOG_FORMAT must be either 'text' or 'json'")
        return fmt

    @field_validator("DB_SSL_MODE")
    @classmethod
    def validate_ssl_mode(cls, value: str) -> str:
        mode = value.lower().strip()
        if mode not in {"disable", "require", "verify"}:
            raise ValueError("DB_SSL_MODE must be one of: disable, require, verify")
        return mode

    @field_validator("DATABASE_URL")
    @classmethod
    def assemble_database_url(cls, value: Optional[str], info: ValidationInfo) -> str:
        """Use DATABASE_URL when given, otherwise build it from the parts.

        A provider's connection string is accepted verbatim and normalised:
        the driver is filled in (``mysql://`` means ``mysql+pymysql://`` here),
        and TLS query parameters written in the MySQL client's spelling
        (``ssl-mode=REQUIRED``) are stripped, because SQLAlchemy passes unknown
        query arguments straight to the driver, which then rejects them. Use
        DB_SSL_MODE to configure TLS instead.
        """
        if value:
            return cls._normalise_database_url(value)

        data = info.data
        required = ("DATABASE_USER", "DB_PASSWORD", "DATABASE_HOST", "DATABASE_PORT", "DATABASE_NAME")
        if any(data.get(key) in (None, "") for key in required if key != "DB_PASSWORD"):
            raise ValueError(
                "Set DATABASE_URL, or all of DATABASE_USER, DB_PASSWORD, DATABASE_HOST, "
                "DATABASE_PORT and DATABASE_NAME."
            )

        from urllib.parse import quote_plus

        return (
            f"mysql+pymysql://{quote_plus(data['DATABASE_USER'])}:"
            f"{quote_plus(data['DB_PASSWORD'])}@"
            f"{data['DATABASE_HOST']}:{data['DATABASE_PORT']}/{data['DATABASE_NAME']}"
        )

    @model_validator(mode="after")
    def enforce_production_invariants(self) -> "Settings":
        """Guard against configurations that are unsafe once deployed."""
        if self.ENVIRONMENT is not Environment.PRODUCTION:
            return self

        problems: list[str] = []

        if self.DEBUG:
            problems.append("DEBUG must be false in production")
        if self.RELOAD:
            problems.append("RELOAD must be false in production")
        if self.ALLOWED_HOSTS.strip() == "*":
            problems.append(
                "ALLOWED_HOSTS must name the API's real hostnames in production, not '*'"
            )
        if "*" in self.cors_origins:
            problems.append(
                "CORS_ORIGINS must list explicit origins in production. A wildcard cannot be "
                "combined with credentialed requests."
            )

        if problems:
            raise ValueError(
                "Invalid production configuration:\n  - " + "\n  - ".join(problems)
            )
        return self

    # ── Derived values ─────────────────────────────────────────────────────

    @classmethod
    def _normalise_database_url(cls, url: str) -> str:
        from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

        parts = urlsplit(url)

        # Only server URLs are rewritten. A SQLite URL has an empty netloc, and
        # round-tripping it through urlunsplit collapses `sqlite:///./db` into
        # the unparseable `sqlite:/./db`.
        if parts.scheme.split("+")[0] not in {"mysql", "postgres", "postgresql"}:
            return url

        scheme = parts.scheme
        if scheme == "mysql":
            scheme = "mysql+pymysql"
        elif scheme in {"postgres", "postgresql"}:
            scheme = "postgresql+psycopg"

        kept = [
            (key, val)
            for key, val in parse_qsl(parts.query, keep_blank_values=True)
            if key.lower() not in CLIENT_ONLY_QUERY_KEYS
        ]

        return urlunsplit((scheme, parts.netloc, parts.path, urlencode(kept), parts.fragment))

    @staticmethod
    def _parse_list(raw: str) -> list[str]:
        """Parse a list setting from either supported format.

        Comma-separated is the documented form. A JSON array is also accepted
        because the previous release typed these as ``list[str]``, which
        pydantic-settings required to be JSON — an existing deployment's
        ``CORS_ORIGINS=["https://example.com"]`` would otherwise be read as a
        single origin with the brackets and quotes still attached, and CORS
        would fail at runtime rather than at start-up.
        """
        raw = raw.strip()
        if not raw:
            return []

        if raw.startswith("["):
            import json

            try:
                parsed = json.loads(raw)
            except ValueError:
                pass
            else:
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]

        return [item.strip() for item in raw.split(",") if item.strip()]

    @property
    def cors_origins(self) -> list[str]:
        return self._parse_list(self.CORS_ORIGINS)

    @property
    def allowed_hosts(self) -> list[str]:
        return self._parse_list(self.ALLOWED_HOSTS)

    @property
    def prompt_directory(self) -> Path:
        return Path(self.PROMPT_DIR) if self.PROMPT_DIR else BASE_DIR / "app" / "prompts"

    @property
    def model_file(self) -> Path:
        return self._resolve(self.MODEL_PATH)

    @property
    def label_encoder_file(self) -> Path:
        return self._resolve(self.LABEL_ENCODER_PATH)

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT is Environment.PRODUCTION

    @staticmethod
    def _resolve(path: str) -> Path:
        """Resolve a possibly-relative path against the backend root.

        This keeps paths working regardless of the process's working directory,
        which differs between `uvicorn app.main:app` and systemd.
        """
        candidate = Path(path)
        return candidate if candidate.is_absolute() else (BASE_DIR / candidate).resolve()

    def summary(self) -> dict[str, Any]:
        """Non-secret configuration, safe to log at start-up."""
        return {
            "environment": self.ENVIRONMENT.value,
            "debug": self.DEBUG,
            "log_level": self.LOG_LEVEL,
            "llm_model": self.LLM_MODEL_NAME,
            "llm_temperature": self.LLM_TEMPERATURE,
            "prompt": self.PROMPT_NAME,
            "prompt_dir": str(self.prompt_directory),
            "model_path": str(self.model_file),
            "cors_origins": self.cors_origins,
            "rate_limiting": self.RATE_LIMIT_ENABLED,
        }


def _load_settings() -> Settings:
    try:
        return Settings()
    except Exception as exc:  # pragma: no cover - start-up failure path
        raise SystemExit(
            f"\nAfya Jamii failed to start: configuration is invalid.\n\n{exc}\n\n"
            "Copy .env.example to .env and fill in the required values.\n"
        ) from exc


settings = _load_settings()
