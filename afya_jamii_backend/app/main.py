"""Afya Jamii AI — HTTP API.

A maternal health service that scores submitted vitals with a risk model and
turns that score into plain-language guidance with a language model.

Start-up policy: the database and the risk model are required, and the process
exits if either is unavailable. The language model is optional — if it cannot
be reached, vitals capture, risk scoring, authentication, and history all keep
working, and advice endpoints report that guidance is temporarily unavailable.
"""

# NOTE: `from __future__ import annotations` is deliberately not used here.
# It turns every endpoint signature into a string annotation, which FastAPI
# cannot resolve when it builds the request models, producing
# "PydanticUndefinedAnnotation: name 'UserCreate' is not defined" at import.

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Session, select

from app.auth import (
    authenticate_user,
    create_access_token,
    get_current_active_user,
    get_password_hash,
    verify_password,
)
from app.config import settings
from app.database import (
    check_connection,
    create_db_and_tables,
    dispose_engine,
    get_session,
    pool_stats,
)
from app.llm_groq import UNAVAILABLE_MESSAGE, LLMUnavailableError, afya_llm, initialize_llm_service
from app.logging_config import configure_logging, request_id_var
from app.ml_model import InvalidFeaturesError, ModelNotLoadedError, initialize_model, risk_model
from app.models import (
    AccountDeletion,
    AccountType,
    CombinedResponse,
    ConversationHistory,
    ConversationResponse,
    DeletionSummary,
    HealthResponse,
    LLMAdviceRequest,
    LLMAdviceResponse,
    MLModelOutput,
    PasswordChange,
    ProfileResponse,
    Token,
    UserCreate,
    UserDB,
    UserLogin,
    UserResponse,
    UserUpdate,
    VitalsInput,
    VitalsRecord,
    VitalsRecordResponse,
    VitalsSubmission,
    utcnow,
)

configure_logging()
logger = logging.getLogger("afya_jamii.api")

limiter = Limiter(
    key_func=get_remote_address,
    enabled=settings.RATE_LIMIT_ENABLED,
    default_limits=[settings.RATE_LIMIT_DEFAULT],
)


# ── Application lifecycle ──────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s v%s", settings.PROJECT_NAME, settings.VERSION)
    logger.info("Configuration: %s", json.dumps(settings.summary(), default=str))

    try:
        create_db_and_tables()
    except SQLAlchemyError as exc:
        logger.critical("Database is unavailable, cannot start: %s", exc)
        raise RuntimeError("Database initialisation failed") from exc

    if not initialize_model():
        logger.critical(
            "The risk model could not be loaded from %s. Refusing to start: the service "
            "would accept vitals it cannot score.",
            settings.model_file,
        )
        raise RuntimeError("Risk model initialisation failed")

    if initialize_llm_service():
        logger.info("Advice service ready")
    else:
        # Deliberately non-fatal — see the module docstring.
        logger.warning(
            "Advice service unavailable at start-up; risk scoring and history remain online"
        )

    logger.info("%s is ready", settings.PROJECT_NAME)
    yield

    logger.info("Shutting down")
    dispose_engine()


app = FastAPI(
    title=settings.PROJECT_NAME,
    description=(
        "Clinical decision support for maternal health: vitals capture, risk "
        "scoring, and guidance for Kenyan pregnant and postnatal mothers."
    ),
    version=settings.VERSION,
    lifespan=lifespan,
    # API documentation is public in development only.
    docs_url=None if settings.is_production else "/docs",
    redoc_url=None if settings.is_production else "/redoc",
    openapi_url=None if settings.is_production else "/openapi.json",
)
app.state.limiter = limiter


# ── Middleware ─────────────────────────────────────────────────────────────
# Registered outermost-first: request id wraps logging wraps the rest.

@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    """Give every request an id, echoed back in X-Request-ID."""
    incoming = request.headers.get("X-Request-ID")
    request_id = incoming if incoming and len(incoming) <= 64 else uuid.uuid4().hex[:12]
    token = request_id_var.set(request_id)
    request.state.request_id = request_id
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        request_id_var.reset(token)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        duration = (time.perf_counter() - start) * 1000
        logger.exception(
            "%s %s failed after %.1fms", request.method, request.url.path, duration
        )
        raise

    duration = (time.perf_counter() - start) * 1000
    client = request.client.host if request.client else "unknown"
    logger.info(
        "%s %s -> %d (%.1fms) from %s",
        request.method,
        request.url.path,
        response.status_code,
        duration,
        client,
    )
    response.headers["X-Response-Time"] = f"{duration:.1f}ms"
    return response


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Permissions-Policy", "geolocation=(), microphone=(), camera=()")

    if settings.is_production:
        response.headers.setdefault("Content-Security-Policy", settings.CSP_DIRECTIVES)
        response.headers.setdefault(
            "Strict-Transport-Security",
            f"max-age={settings.HSTS_MAX_AGE}; includeSubDomains",
        )
    return response


if settings.RATE_LIMIT_ENABLED:
    # Without this the @limiter.limit decorators are inert.
    app.add_middleware(SlowAPIMiddleware)

app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    # PATCH and DELETE are needed by the account settings endpoints; without
    # them the browser's preflight fails and those calls never leave the page.
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    expose_headers=["X-Request-ID"],
    max_age=600,
)


# ── Error handling ─────────────────────────────────────────────────────────

def _error(request: Request, status_code: int, detail: Any) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"detail": detail, "request_id": getattr(request.state, "request_id", None)},
    )


@app.exception_handler(RateLimitExceeded)
async def handle_rate_limit(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    logger.warning("Rate limit hit on %s", request.url.path)
    return _error(request, status.HTTP_429_TOO_MANY_REQUESTS, "Too many requests. Please slow down.")


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    # Flatten Pydantic's structure into messages a UI can show directly.
    problems = [
        {
            "field": ".".join(str(part) for part in error["loc"] if part not in ("body", "query")),
            "message": error["msg"],
        }
        for error in exc.errors()
    ]
    logger.info("Validation failed on %s: %s", request.url.path, problems)
    return _error(request, status.HTTP_422_UNPROCESSABLE_ENTITY, problems)


@app.exception_handler(HTTPException)
async def handle_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
    if exc.status_code >= 500:
        logger.error("%s on %s: %s", exc.status_code, request.url.path, exc.detail)
    response = _error(request, exc.status_code, exc.detail)
    if exc.headers:
        response.headers.update(exc.headers)
    return response


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    # The message is deliberately generic; details go to the log, keyed by the
    # request id the client receives.
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return _error(
        request,
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "Something went wrong on our side. Quote the request id if you contact support.",
    )


# ── Helpers ────────────────────────────────────────────────────────────────

def _decode_importances(raw: Optional[str]) -> dict[str, float]:
    """Parse the stored feature-importance JSON, tolerating older rows."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        logger.warning("Skipping malformed feature importances")
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(k): float(v) for k, v in parsed.items() if isinstance(v, (int, float))}


def _describe_vitals(vitals: VitalsInput, account_type: AccountType, prediction) -> str:
    """Build the context block handed to the language model."""
    ranked = sorted(prediction.feature_importances.items(), key=lambda item: item[1], reverse=True)
    top_factors = ", ".join(f"{name} ({value:.0%})" for name, value in ranked[:3]) or "not available"

    return "\n".join(
        [
            "The user has just submitted a set of vitals.",
            "",
            f"- Age: {vitals.age} years",
            f"- Blood pressure: {vitals.systolic_bp}/{vitals.diastolic_bp} mmHg",
            f"- Blood sugar: {vitals.bs} mmol/L",
            f"- Body temperature: {vitals.body_temp_celsius}°C",
            f"- Heart rate: {vitals.heart_rate} bpm",
            f"- Account type: {account_type.value}",
            f"- Reported history: {vitals.patient_history or 'none given'}",
            "",
            f"Risk assessment: {prediction.label} "
            f"(confidence {prediction.probability:.0%}).",
            f"Most influential readings: {top_factors}.",
        ]
    )


def _condense(reply: str, limit: int) -> str:
    """Shorten a past reply to ``limit`` characters on a word boundary."""
    reply = reply.strip()
    if len(reply) <= limit:
        return reply

    clipped = reply[:limit]
    boundary = clipped.rfind(" ")
    if boundary > limit // 2:
        clipped = clipped[:boundary]
    return clipped.rstrip() + " […]"


def _recent_history(session: Session, user_id: int, turns: int) -> str:
    """Return recent exchanges, oldest first, within a fixed size budget.

    Two limits apply. Each past reply is condensed to
    ``CHAT_HISTORY_REPLY_CHARS``, and turns are admitted newest-first until
    ``CHAT_HISTORY_CHAR_BUDGET`` is spent. Without them the prompt grew with
    every exchange — replies run to several thousand characters each, so a
    ten-turn history reached ~40k characters and every question took longer
    than the one before it.

    The user's own questions are kept in full: they are short, and they carry
    the thread of the conversation.
    """
    if turns <= 0:
        return ""

    # Take the newest rows, then restore chronological order for the prompt.
    # (A previous revision ordered ascending and truncated, which fed the model
    # the user's oldest turns and dropped everything recent.)
    records = session.exec(
        select(ConversationHistory)
        .where(ConversationHistory.user_id == user_id)
        .order_by(ConversationHistory.created_at.desc())
        .limit(turns)
    ).all()

    budget = settings.CHAT_HISTORY_CHAR_BUDGET
    exchanges: list[str] = []

    for record in records:  # newest first, so the oldest turns drop out
        exchange = (
            f"User: {record.user_message}\n"
            f"Afya Jamii: {_condense(record.ai_response, settings.CHAT_HISTORY_REPLY_CHARS)}"
        )
        if len(exchange) > budget:
            break
        exchanges.append(exchange)
        budget -= len(exchange)

    return "\n\n".join(reversed(exchanges))


async def _advise(values: dict[str, Any]) -> LLMAdviceResponse:
    """Ask the model for advice, degrading to a safe message on failure."""
    try:
        advice = await afya_llm.generate_advice(values)
        return LLMAdviceResponse(advice=advice, timestamp=utcnow(), generated=True)
    except LLMUnavailableError as exc:
        logger.warning("Falling back to the offline advice message: %s", exc)
        return LLMAdviceResponse(advice=UNAVAILABLE_MESSAGE, timestamp=utcnow(), generated=False)


# ── System endpoints ───────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {"service": settings.PROJECT_NAME, "version": settings.VERSION, "status": "ok"}


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check(response: Response) -> HealthResponse:
    """Report component health.

    Returns 503 when a required component is down, so orchestrators can pull
    the instance out of rotation.
    """
    database_ok = check_connection()
    model_ok = risk_model.is_loaded
    llm_status = afya_llm.status()

    if not (database_ok and model_ok):
        state = "unhealthy"
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    elif not llm_status.ready:
        # Advice is degraded but the core service still works.
        state = "degraded"
    else:
        state = "healthy"

    return HealthResponse(
        status=state,
        timestamp=utcnow(),
        version=settings.VERSION,
        environment=settings.ENVIRONMENT.value,
        services={
            "database": {"ready": database_ok, "pool": pool_stats()},
            "risk_model": {"ready": model_ok, "classes": risk_model.classes},
            "advice": llm_status.as_dict(),
        },
    )


# ── Authentication ─────────────────────────────────────────────────────────

@app.post(
    f"{settings.API_V1_STR}/auth/signup",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["auth"],
)
@limiter.limit(settings.RATE_LIMIT_SIGNUP)
async def signup(
    request: Request,
    user_data: UserCreate,
    session: Session = Depends(get_session),
) -> UserDB:
    # Compare case-insensitively: MySQL's default collation already treats
    # "Amina" and "amina" as the same username, so the check that picks the
    # message must too, or a username clash is reported as an email clash.
    existing = session.exec(
        select(UserDB).where(
            (func.lower(UserDB.username) == user_data.username.lower())
            | (func.lower(UserDB.email) == user_data.email.lower())
        )
    ).first()
    if existing is not None:
        field = "username" if existing.username.lower() == user_data.username.lower() else "email"
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"That {field} is already registered",
        )

    user = UserDB(
        **user_data.model_dump(exclude={"password"}),
        hashed_password=get_password_hash(user_data.password),
    )
    session.add(user)
    session.commit()
    session.refresh(user)

    logger.info("Account created: %s (%s)", user.username, user.account_type.value)
    return user


@app.post(f"{settings.API_V1_STR}/auth/login", response_model=Token, tags=["auth"])
@limiter.limit(settings.RATE_LIMIT_LOGIN)
async def login(
    request: Request,
    login_data: UserLogin,
    session: Session = Depends(get_session),
) -> Token:
    user = authenticate_user(session, login_data.username, login_data.password)
    if user is None:
        logger.info("Failed sign-in for %s", login_data.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    logger.info("Signed in: %s", user.username)
    return Token(
        access_token=create_access_token(data={"sub": user.username}),
        token_type="bearer",
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        username=user.username,
        account_type=user.account_type,
    )


@app.get(f"{settings.API_V1_STR}/auth/me", response_model=UserResponse, tags=["auth"])
async def read_current_user(
    current_user: UserDB = Depends(get_current_active_user),
) -> UserDB:
    """Return the signed-in user, so a client can restore a session from a token."""
    return current_user


# ── Account settings ───────────────────────────────────────────────────────

@app.get(f"{settings.API_V1_STR}/users/me", response_model=UserResponse, tags=["settings"])
async def get_profile(current_user: UserDB = Depends(get_current_active_user)) -> UserDB:
    """The signed-in user's profile."""
    return current_user


@app.patch(f"{settings.API_V1_STR}/users/me", response_model=ProfileResponse, tags=["settings"])
async def update_profile(
    request: Request,
    updates: UserUpdate,
    current_user: UserDB = Depends(get_current_active_user),
    session: Session = Depends(get_session),
) -> ProfileResponse:
    """Change the profile fields a user is allowed to edit.

    Only the fields present in the request body are touched.

    Changing the username is allowed, but the username is the subject of every
    issued token — the caller's existing token would stop validating the
    moment the row changed. A replacement is returned in ``access_token`` so
    the client can swap it and carry on, rather than being signed out in the
    middle of editing a profile.
    """
    changes = updates.model_dump(exclude_unset=True)
    previous_username = current_user.username

    # Both columns are unique; check before writing so the caller gets a clear
    # 409 rather than an integrity error surfacing as a 500.
    if "username" in changes and changes["username"] != previous_username:
        clash = session.exec(
            select(UserDB).where(
                func.lower(UserDB.username) == changes["username"].lower(),
                UserDB.id != current_user.id,
            )
        ).first()
        if clash is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="That username is already taken",
            )

    if "email" in changes:
        clash = session.exec(
            select(UserDB).where(
                func.lower(UserDB.email) == changes["email"].lower(),
                UserDB.id != current_user.id,
            )
        ).first()
        if clash is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="That email is already registered to another account",
            )

    for field, value in changes.items():
        setattr(current_user, field, value)
    current_user.updated_at = utcnow()

    session.add(current_user)
    session.commit()
    session.refresh(current_user)

    renamed = current_user.username != previous_username
    if renamed:
        logger.info("Username changed: %s -> %s", previous_username, current_user.username)
    logger.info("Profile updated for %s: %s", current_user.username, ", ".join(changes))

    return ProfileResponse(
        **current_user.model_dump(exclude={"hashed_password", "updated_at"}),
        access_token=create_access_token(data={"sub": current_user.username}) if renamed else None,
    )


@app.post(
    f"{settings.API_V1_STR}/users/me/password",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["settings"],
)
@limiter.limit(settings.RATE_LIMIT_LOGIN)
async def change_password(
    request: Request,
    payload: PasswordChange,
    current_user: UserDB = Depends(get_current_active_user),
    session: Session = Depends(get_session),
) -> Response:
    """Set a new password, confirming the current one first.

    Note that existing tokens stay valid until they expire: they are signed
    with the application secret, not with the password. Revoking them would
    need a token version or a deny-list, which this service does not yet have.
    """
    if not verify_password(payload.current_password, current_user.hashed_password):
        logger.info("Password change refused for %s: wrong current password", current_user.username)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your current password is not correct",
        )

    current_user.hashed_password = get_password_hash(payload.new_password)
    current_user.updated_at = utcnow()
    session.add(current_user)
    session.commit()

    logger.info("Password changed for %s", current_user.username)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post(
    f"{settings.API_V1_STR}/users/me/deactivate",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["settings"],
)
async def deactivate_account(
    request: Request,
    current_user: UserDB = Depends(get_current_active_user),
    session: Session = Depends(get_session),
) -> Response:
    """Disable the account without destroying anything.

    Offered alongside deletion because it is what most people actually want:
    sign-in stops working, but the health records survive, so a mother who
    returns to the service still has her history. Reactivation is a manual
    operation for an administrator.
    """
    current_user.is_active = False
    current_user.updated_at = utcnow()
    session.add(current_user)
    session.commit()

    logger.info("Account deactivated: %s", current_user.username)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.delete(
    f"{settings.API_V1_STR}/users/me",
    response_model=DeletionSummary,
    tags=["settings"],
)
@limiter.limit(settings.RATE_LIMIT_LOGIN)
async def delete_account(
    request: Request,
    confirmation: AccountDeletion,
    current_user: UserDB = Depends(get_current_active_user),
    session: Session = Depends(get_session),
) -> DeletionSummary:
    """Permanently delete the account and every record attached to it.

    This is irreversible. It requires the account password and the typed
    phrase 'DELETE MY ACCOUNT', so neither a stolen token alone nor a
    mis-click is enough to destroy someone's health history.

    Rows are removed child-first — conversations, then vitals, then the user —
    because both child tables carry a foreign key to users, and conversations
    additionally reference vitals. The whole thing runs in one transaction: a
    failure part-way through leaves the account intact rather than orphaned.
    """
    if not verify_password(confirmation.password, current_user.hashed_password):
        logger.warning("Account deletion refused for %s: wrong password", current_user.username)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your password is not correct",
        )

    username = current_user.username
    user_id = current_user.id

    try:
        conversations = session.exec(
            select(ConversationHistory).where(ConversationHistory.user_id == user_id)
        ).all()
        for conversation in conversations:
            session.delete(conversation)

        vitals = session.exec(
            select(VitalsRecord).where(VitalsRecord.user_id == user_id)
        ).all()
        for record in vitals:
            session.delete(record)

        # Flush the children before removing the parent so a foreign-key
        # violation surfaces here rather than at commit.
        session.flush()
        session.delete(current_user)
        session.commit()
    except SQLAlchemyError:
        session.rollback()
        logger.exception("Account deletion failed for %s; nothing was removed", username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not delete the account. Nothing was removed — please try again.",
        ) from None

    logger.info(
        "Account deleted: %s (%d vitals records, %d conversations)",
        username,
        len(vitals),
        len(conversations),
    )

    return DeletionSummary(
        detail="Your account and all associated health records have been permanently deleted.",
        username=username,
        vitals_records_deleted=len(vitals),
        conversations_deleted=len(conversations),
        deleted_at=utcnow(),
    )


# ── Vitals ─────────────────────────────────────────────────────────────────

@app.post(
    f"{settings.API_V1_STR}/vitals/submit",
    response_model=CombinedResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["vitals"],
)
@limiter.limit(settings.RATE_LIMIT_INFERENCE)
async def submit_vitals(
    request: Request,
    submission: VitalsSubmission,
    current_user: UserDB = Depends(get_current_active_user),
    session: Session = Depends(get_session),
) -> CombinedResponse:
    """Score a set of vitals and return guidance alongside the result."""
    vitals = submission.vitals
    account_type = submission.account_type or current_user.account_type

    features = {
        "Age": vitals.age,
        "SystolicBP": vitals.systolic_bp,
        "DiastolicBP": vitals.diastolic_bp,
        "BS": vitals.bs,
        # The model was trained in Celsius; convert before scoring.
        "BodyTemp": vitals.body_temp_celsius,
        "HeartRate": vitals.heart_rate,
    }

    try:
        prediction = risk_model.predict(features)
    except InvalidFeaturesError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except ModelNotLoadedError as exc:
        logger.error("Scoring attempted while the model is unavailable: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Risk assessment is temporarily unavailable. Please try again shortly.",
        ) from exc

    record = VitalsRecord(
        user_id=current_user.id,
        age=vitals.age,
        systolic_bp=vitals.systolic_bp,
        diastolic_bp=vitals.diastolic_bp,
        bs=vitals.bs,
        body_temp=vitals.body_temp,
        body_temp_unit=vitals.body_temp_unit.value,
        heart_rate=vitals.heart_rate,
        patient_history=vitals.patient_history,
        ml_risk_label=prediction.label,
        ml_probability=prediction.probability,
        ml_feature_importances=json.dumps(prediction.feature_importances),
    )
    session.add(record)
    session.commit()
    session.refresh(record)

    logger.info(
        "Vitals %d scored for %s: %s (%.0f%%)",
        record.id,
        current_user.username,
        prediction.label,
        prediction.probability * 100,
    )

    advice = await _advise(
        {
            "context": _describe_vitals(vitals, account_type, prediction),
            "history": _recent_history(session, current_user.id, settings.CHAT_HISTORY_TURNS),
            "question": (
                "Give an initial risk assessment and practical recommendations "
                "based on these vitals."
            ),
        }
    )

    # Only record exchanges that carry real advice, so a failed call does not
    # poison the history replayed to the model on the next turn.
    if advice.generated:
        session.add(
            ConversationHistory(
                user_id=current_user.id,
                vitals_record_id=record.id,
                user_message="Initial assessment of submitted vitals",
                ai_response=advice.advice,
            )
        )
        session.commit()

    return CombinedResponse(
        user_id=current_user.id,
        submission_id=record.id,
        timestamp=utcnow(),
        ml_output=MLModelOutput(
            risk_label=prediction.label,
            probability=prediction.probability,
            class_probabilities=prediction.class_probabilities,
            feature_importances=prediction.feature_importances,
        ),
        llm_advice=advice,
    )


# ── Chat ───────────────────────────────────────────────────────────────────

@app.post(
    f"{settings.API_V1_STR}/chat/advice",
    response_model=LLMAdviceResponse,
    tags=["chat"],
)
@limiter.limit(settings.RATE_LIMIT_INFERENCE)
async def get_advice(
    request: Request,
    advice_request: LLMAdviceRequest,
    current_user: UserDB = Depends(get_current_active_user),
    session: Session = Depends(get_session),
) -> LLMAdviceResponse:
    """Answer a follow-up question in the context of the user's history."""
    latest_vitals = session.exec(
        select(VitalsRecord)
        .where(VitalsRecord.user_id == current_user.id)
        .order_by(VitalsRecord.created_at.desc())
        .limit(1)
    ).first()

    if latest_vitals is not None:
        context = (
            f"The user is asking a follow-up question. Their most recent assessment "
            f"({latest_vitals.created_at:%d %B %Y}) was {latest_vitals.ml_risk_label}, "
            f"from a blood pressure of {latest_vitals.systolic_bp}/{latest_vitals.diastolic_bp} mmHg, "
            f"blood sugar {latest_vitals.bs} mmol/L, and heart rate {latest_vitals.heart_rate} bpm."
        )
    else:
        context = (
            "The user is asking a question and has not submitted any vitals yet. "
            f"Their account type is {current_user.account_type.value}."
        )

    advice = await _advise(
        {
            "context": context,
            "history": _recent_history(session, current_user.id, settings.CHAT_HISTORY_TURNS),
            "question": advice_request.question,
        }
    )

    if advice.generated:
        session.add(
            ConversationHistory(
                user_id=current_user.id,
                vitals_record_id=latest_vitals.id if latest_vitals else None,
                user_message=advice_request.question,
                ai_response=advice.advice,
            )
        )
        session.commit()
    else:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=UNAVAILABLE_MESSAGE,
        )

    return advice


# ── History ────────────────────────────────────────────────────────────────

@app.get(
    f"{settings.API_V1_STR}/history/vitals",
    response_model=list[VitalsRecordResponse],
    tags=["history"],
)
async def get_vitals_history(
    request: Request,
    limit: int = 10,
    current_user: UserDB = Depends(get_current_active_user),
    session: Session = Depends(get_session),
) -> list[VitalsRecordResponse]:
    limit = max(1, min(limit, 100))
    records = session.exec(
        select(VitalsRecord)
        .where(VitalsRecord.user_id == current_user.id)
        .order_by(VitalsRecord.created_at.desc())
        .limit(limit)
    ).all()

    return [
        VitalsRecordResponse(
            id=record.id,
            user_id=record.user_id,
            age=record.age,
            systolic_bp=record.systolic_bp,
            diastolic_bp=record.diastolic_bp,
            bs=record.bs,
            body_temp=record.body_temp,
            body_temp_unit=record.body_temp_unit,
            heart_rate=record.heart_rate,
            patient_history=record.patient_history,
            ml_risk_label=record.ml_risk_label,
            ml_probability=record.ml_probability,
            ml_feature_importances=_decode_importances(record.ml_feature_importances),
            created_at=record.created_at,
        )
        for record in records
    ]


@app.get(
    f"{settings.API_V1_STR}/history/conversations",
    response_model=list[ConversationResponse],
    tags=["history"],
)
async def get_conversation_history(
    request: Request,
    limit: int = 20,
    current_user: UserDB = Depends(get_current_active_user),
    session: Session = Depends(get_session),
) -> list[ConversationHistory]:
    limit = max(1, min(limit, 100))
    return session.exec(
        select(ConversationHistory)
        .where(ConversationHistory.user_id == current_user.id)
        .order_by(ConversationHistory.created_at.desc())
        .limit(limit)
    ).all()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        log_config=None,  # configure_logging() already installed handlers
    )
