"""Shared test configuration.

Settings are validated at import time, so the environment has to be populated
before anything under `app` is imported.
"""

import os

os.environ.setdefault("SECRET_KEY", "test-secret-key-at-least-32-characters-long")
os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("DATABASE_URL", "sqlite:///./.pytest_afya.db")
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")
os.environ.setdefault("LOG_LEVEL", "WARNING")
