"""Password hashing, token issuing, and the authenticated-user dependency."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlmodel import Session, select

from app.config import settings
from app.database import get_session
from app.models import TokenData, UserDB

logger = logging.getLogger(__name__)

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=settings.BCRYPT_ROUNDS,
)

# auto_error=False so a missing header produces our own 401 with a
# WWW-Authenticate challenge rather than FastAPI's bare 403.
security = HTTPBearer(auto_error=False)

CREDENTIALS_EXCEPTION = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)

# bcrypt silently truncates beyond 72 bytes; reject rather than accept a
# password whose tail is ignored.
BCRYPT_MAX_BYTES = 72


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except ValueError:
        # Raised for a malformed or truncated hash in the database.
        logger.warning("Password verification failed: stored hash is malformed")
        return False


def get_password_hash(password: str) -> str:
    if len(password.encode("utf-8")) > BCRYPT_MAX_BYTES:
        raise ValueError(f"Password must be at most {BCRYPT_MAX_BYTES} bytes long")
    return pwd_context.hash(password)


# Hash of an unguessable value, compared against when the username is unknown
# so that sign-in timing does not reveal whether an account exists.
_DUMMY_HASH = pwd_context.hash("afya-jamii-timing-equaliser")


def authenticate_user(session: Session, username: str, password: str) -> Optional[UserDB]:
    """Return the user when the credentials match, otherwise None."""
    user = session.exec(select(UserDB).where(UserDB.username == username)).first()

    if user is None:
        # Verify against a throwaway hash anyway, so an unknown username costs
        # the same time as a wrong password and the endpoint does not leak
        # which accounts exist.
        verify_password(password, _DUMMY_HASH)
        return None

    if not user.is_active:
        logger.info("Rejected sign-in for deactivated account: %s", username)
        return None

    if not verify_password(password, user.hashed_password):
        return None

    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc)})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    session: Session = Depends(get_session),
) -> UserDB:
    if credentials is None or not credentials.credentials:
        raise CREDENTIALS_EXCEPTION

    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
        )
    except JWTError as exc:
        # Expected for expired or tampered tokens; log at debug so a scan does
        # not fill the error log.
        logger.debug("Rejected token: %s", exc)
        raise CREDENTIALS_EXCEPTION from exc

    username = payload.get("sub")
    if not username:
        raise CREDENTIALS_EXCEPTION

    token_data = TokenData(username=username)
    user = session.exec(select(UserDB).where(UserDB.username == token_data.username)).first()

    if user is None:
        raise CREDENTIALS_EXCEPTION
    return user


async def get_current_active_user(
    current_user: UserDB = Depends(get_current_user),
) -> UserDB:
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This account has been deactivated",
        )
    return current_user
