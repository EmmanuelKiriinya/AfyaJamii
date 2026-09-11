"""Database engine, sessions, and schema management.

The engine is created once at import time and shared by every request through
the :func:`get_session` dependency. Connection settings come from the
environment; see ``app.config``.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import QueuePool
from sqlmodel import Session, SQLModel, create_engine

from app.config import settings

logger = logging.getLogger(__name__)

# Columns that must hold more than MySQL's default TEXT capacity. Model advice
# and feature-importance documents routinely exceed 64 KB.
LONGTEXT_COLUMNS: tuple[tuple[str, str], ...] = (
    ("conversation_history", "ai_response"),
    ("vitals_records", "ml_feature_importances"),
)

_is_mysql = settings.DATABASE_URL.startswith("mysql")
_is_sqlite = settings.DATABASE_URL.startswith("sqlite")

def _mysql_ssl_context() -> Any:
    """Build the TLS context for the MySQL connection.

    Returns None when TLS is disabled. A managed database reached over the
    public internet accepts unencrypted connections unless the client asks for
    TLS, so leaving this unset would quietly send patient vitals in plaintext —
    hence `require` rather than `disable` as the default in ``app.config``.

    `verify` additionally checks the server certificate, which needs the
    provider's CA in DB_SSL_CA. Aiven and several others sign with a private
    CA that the system trust store does not know, so verification fails with
    CERTIFICATE_VERIFY_FAILED until that file is supplied.
    """
    import ssl

    mode = settings.DB_SSL_MODE
    if mode == "disable":
        logger.warning(
            "Database TLS is disabled. Only do this for a local database — "
            "traffic to a remote host would be sent in plaintext."
        )
        return None

    if mode == "verify":
        if not settings.DB_SSL_CA:
            raise RuntimeError(
                "DB_SSL_MODE=verify requires DB_SSL_CA to point at the provider's "
                "CA certificate (Aiven: service page -> CA certificate)."
            )
        ca_path = Path(settings.DB_SSL_CA)
        if not ca_path.is_file():
            raise RuntimeError(f"DB_SSL_CA file not found: {ca_path}")

        context = ssl.create_default_context(cafile=str(ca_path))
        logger.info("Database TLS: encrypted, certificate verified against %s", ca_path)
        return context

    # mode == "require": encrypt, but do not validate the certificate. This
    # defeats passive interception, not an active man-in-the-middle. Supply a
    # CA and switch to `verify` where the data warrants it.
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    logger.warning(
        "Database TLS: encrypted but the server certificate is NOT verified. "
        "Set DB_SSL_CA and DB_SSL_MODE=verify for full protection."
    )
    return context


_connect_args: dict[str, Any] = {}
if _is_mysql:
    _connect_args = {"charset": "utf8mb4", "connect_timeout": 15}
    _ssl_context = _mysql_ssl_context()
    if _ssl_context is not None:
        _connect_args["ssl"] = _ssl_context
elif _is_sqlite:
    # SQLite is only used for local development and tests. FastAPI runs sync
    # endpoints on a thread pool, so its default same-thread guard has to be
    # relaxed for connections to be reusable across requests.
    _connect_args = {"check_same_thread": False}

_engine_options: dict[str, Any] = {
    "echo": settings.DB_ECHO,
    "connect_args": _connect_args,
    # Verify a pooled connection before handing it out; MySQL closes idle
    # connections and the alternative is an intermittent 500 on the first
    # request after a quiet period.
    "pool_pre_ping": True,
}

if not _is_sqlite:
    # SQLite's default pool does not accept these, and pooling a local file
    # would gain nothing anyway.
    _engine_options.update(
        poolclass=QueuePool,
        pool_size=settings.DB_POOL_SIZE,
        max_overflow=settings.DB_MAX_OVERFLOW,
        pool_recycle=settings.DB_POOL_RECYCLE,
        pool_timeout=settings.DB_POOL_TIMEOUT,
    )

engine = create_engine(settings.DATABASE_URL, **_engine_options)


def create_db_and_tables() -> None:
    """Create any missing tables and widen the large text columns.

    Raises:
        SQLAlchemyError: if the schema cannot be created; start-up should fail.
    """
    SQLModel.metadata.create_all(engine)
    logger.info("Database schema verified")

    if not _is_mysql:
        return

    # MySQL only: SQLModel maps str to TEXT, which truncates longer advice.
    with engine.begin() as connection:
        for table, column in LONGTEXT_COLUMNS:
            try:
                connection.execute(text(f"ALTER TABLE {table} MODIFY {column} LONGTEXT"))
                logger.debug("Ensured %s.%s is LONGTEXT", table, column)
            except SQLAlchemyError as exc:
                logger.warning("Could not widen %s.%s to LONGTEXT: %s", table, column, exc)


def get_session() -> Iterator[Session]:
    """FastAPI dependency yielding a request-scoped session."""
    with Session(engine) as session:
        try:
            yield session
        except Exception:
            session.rollback()
            raise


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional session for use outside the request cycle.

    Commits on success, rolls back on error.
    """
    session = Session(engine)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        logger.exception("Database transaction rolled back")
        raise
    finally:
        session.close()


def check_connection() -> bool:
    """Return True when the database answers a trivial query."""
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except SQLAlchemyError as exc:
        logger.error("Database connection check failed: %s", exc)
        return False


def pool_stats() -> dict[str, Any]:
    """Connection-pool counters, for the health endpoint and dashboards."""
    pool = engine.pool
    try:
        return {
            "size": pool.size(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
        }
    except AttributeError:  # pragma: no cover - non-queue pools
        return {}


def dispose_engine() -> None:
    """Close every pooled connection. Called during application shutdown."""
    engine.dispose()
    logger.info("Database connection pool closed")
