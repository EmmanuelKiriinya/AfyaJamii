# Afya Jamii API container image.
#
# Built in two stages so the runtime image carries no compilers or build
# headers — xgboost and scikit-learn pull in a substantial toolchain that has
# no business being in production.

# ── Build stage ────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Dependencies are installed before the source is copied, so a code change
# does not invalidate the (slow) dependency layer.
COPY requirements.lock.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.lock.txt

# ── Runtime stage ──────────────────────────────────────────────────────────
FROM python:3.12-slim

# libgomp is required at runtime by xgboost's OpenMP kernels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 afya

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY --chown=afya:afya app ./app
COPY --chown=afya:afya data ./data

USER afya
EXPOSE 8000

# Reports "unhealthy" once the API's own /health says a required component is
# down, so the orchestrator can replace the container.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Uvicorn workers under Gunicorn: Gunicorn handles process supervision, and
# uvicorn provides the ASGI runtime. Raise --workers on larger instances.
CMD ["gunicorn", "app.main:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--workers", "2", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--graceful-timeout", "30", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
