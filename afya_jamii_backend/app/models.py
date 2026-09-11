"""API schemas and database tables.

Request and response models are Pydantic; persisted tables are SQLModel. The
two are kept separate on purpose — a table is never returned directly from an
endpoint, so a column added later (a password hash, an internal note) cannot
leak into an API response by accident.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator, model_validator
from sqlmodel import Field as SQLField
from sqlmodel import SQLModel


def utcnow() -> datetime:
    """Timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


class AccountType(str, Enum):
    PREGNANT = "pregnant"
    POSTNATAL = "postnatal"
    GENERAL = "general"


class TemperatureUnit(str, Enum):
    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"


# Accepted body-temperature range per unit. The model is trained in Celsius, so
# Fahrenheit readings are converted before scoring.
TEMPERATURE_RANGES: dict[TemperatureUnit, tuple[float, float]] = {
    TemperatureUnit.CELSIUS: (35.0, 42.0),
    TemperatureUnit.FAHRENHEIT: (95.0, 107.6),
}


# ── Users ──────────────────────────────────────────────────────────────────

class UserBase(BaseModel):
    username: str = Field(
        ...,
        min_length=3,
        max_length=50,
        pattern=r"^[a-zA-Z0-9_.-]+$",
        description="Letters, digits, underscore, dot, and hyphen only",
    )
    email: EmailStr
    account_type: AccountType
    full_name: Optional[str] = Field(None, max_length=100)


class UserCreate(UserBase):
    password: str = Field(..., min_length=8, max_length=128)

    @field_validator("password")
    @classmethod
    def password_is_reasonable(cls, value: str) -> str:
        if value.isdigit() or value.isalpha():
            raise ValueError("Password must combine letters with digits or symbols")
        return value


class UserLogin(BaseModel):
    username: str = Field(..., min_length=1, max_length=50)
    password: str = Field(..., min_length=1, max_length=128)


class UserResponse(UserBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    is_active: bool = True


# ── Account settings ───────────────────────────────────────────────────────

class UserUpdate(BaseModel):
    """A partial update of the signed-in user's profile.

    Every field is optional; only those present in the request body are
    changed. The username is deliberately not editable — it identifies the
    account and appears in issued tokens.
    """

    email: Optional[EmailStr] = None
    full_name: Optional[str] = Field(None, max_length=100)
    account_type: Optional[AccountType] = None

    @model_validator(mode="after")
    def at_least_one_field(self) -> "UserUpdate":
        if self.model_fields_set == set():
            raise ValueError("Provide at least one field to update")
        return self


class PasswordChange(BaseModel):
    current_password: str = Field(..., min_length=1, max_length=128)
    new_password: str = Field(..., min_length=8, max_length=128)

    @field_validator("new_password")
    @classmethod
    def password_is_reasonable(cls, value: str) -> str:
        if value.isdigit() or value.isalpha():
            raise ValueError("Password must combine letters with digits or symbols")
        return value

    @model_validator(mode="after")
    def must_actually_change(self) -> "PasswordChange":
        if self.current_password == self.new_password:
            raise ValueError("The new password must differ from the current one")
        return self


class AccountDeletion(BaseModel):
    """Confirmation for permanently deleting an account.

    Deletion is irreversible and removes the user's health records, so it asks
    for the password again and for the confirmation phrase to be typed out. A
    mis-click cannot satisfy both.
    """

    password: str = Field(..., min_length=1, max_length=128)
    confirmation: str = Field(
        ...,
        description="Must be exactly 'DELETE MY ACCOUNT'",
    )

    @field_validator("confirmation")
    @classmethod
    def check_confirmation(cls, value: str) -> str:
        if value.strip() != "DELETE MY ACCOUNT":
            raise ValueError("Type 'DELETE MY ACCOUNT' to confirm")
        return value.strip()


class DeletionSummary(BaseModel):
    """What was removed, so the client can show the user the outcome."""

    detail: str
    username: str
    vitals_records_deleted: int
    conversations_deleted: int
    deleted_at: datetime


# ── Vitals ─────────────────────────────────────────────────────────────────

class VitalsInput(BaseModel):
    age: int = Field(..., ge=15, le=50, description="Age in years")
    systolic_bp: int = Field(..., ge=70, le=200, description="Systolic blood pressure (mmHg)")
    diastolic_bp: int = Field(..., ge=40, le=130, description="Diastolic blood pressure (mmHg)")
    bs: float = Field(..., ge=3.0, le=30.0, description="Blood sugar (mmol/L)")
    body_temp: float = Field(..., description="Body temperature, in the unit given below")
    body_temp_unit: TemperatureUnit = TemperatureUnit.CELSIUS
    heart_rate: int = Field(..., ge=40, le=150, description="Heart rate (bpm)")
    patient_history: Optional[str] = Field(None, max_length=1000)

    @model_validator(mode="after")
    def check_temperature_range(self) -> "VitalsInput":
        low, high = TEMPERATURE_RANGES[self.body_temp_unit]
        if not low <= self.body_temp <= high:
            raise ValueError(
                f"Body temperature must be between {low:g} and {high:g} "
                f"degrees {self.body_temp_unit.value.title()}"
            )
        return self

    @model_validator(mode="after")
    def check_blood_pressure_ordering(self) -> "VitalsInput":
        if self.diastolic_bp >= self.systolic_bp:
            raise ValueError("Diastolic pressure must be lower than systolic pressure")
        return self

    @property
    def body_temp_celsius(self) -> float:
        """Temperature in Celsius, which is what the model was trained on."""
        if self.body_temp_unit is TemperatureUnit.FAHRENHEIT:
            return round((self.body_temp - 32.0) * 5.0 / 9.0, 2)
        return self.body_temp


class VitalsSubmission(BaseModel):
    vitals: VitalsInput
    # Optional: defaults to the account type on the authenticated user.
    account_type: Optional[AccountType] = None


class VitalsRecordResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    # Retained for compatibility: the previous release serialised the table
    # model directly, so consumers may still read this field.
    user_id: int
    age: int
    systolic_bp: int
    diastolic_bp: int
    bs: float
    body_temp: float
    body_temp_unit: str
    heart_rate: int
    patient_history: Optional[str] = None
    ml_risk_label: str
    ml_probability: float
    ml_feature_importances: dict[str, float] = Field(default_factory=dict)
    created_at: datetime


# ── Model output ───────────────────────────────────────────────────────────

class MLModelOutput(BaseModel):
    risk_label: str
    probability: float = Field(..., ge=0.0, le=1.0, description="Confidence in the predicted class")
    class_probabilities: dict[str, float] = Field(default_factory=dict)
    feature_importances: dict[str, float] = Field(default_factory=dict)


class LLMAdviceRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Question cannot be blank")
        return cleaned


class LLMAdviceResponse(BaseModel):
    advice: str
    timestamp: datetime
    # False when the advice service was unreachable and the text is a fallback.
    generated: bool = True


class CombinedResponse(BaseModel):
    user_id: int
    submission_id: int
    timestamp: datetime
    ml_output: MLModelOutput
    llm_advice: LLMAdviceResponse


class ConversationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    vitals_record_id: Optional[int] = None
    user_message: str
    ai_response: str
    created_at: datetime


# ── Authentication ─────────────────────────────────────────────────────────

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    # Returned so the client need not guess; the previous frontend defaulted
    # every session to "general" after login.
    username: str
    account_type: AccountType


class TokenData(BaseModel):
    username: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    environment: str
    services: dict[str, Any]


# ── Tables ─────────────────────────────────────────────────────────────────

class UserDB(SQLModel, table=True):
    __tablename__ = "users"

    id: Optional[int] = SQLField(default=None, primary_key=True)
    username: str = SQLField(unique=True, index=True, max_length=50)
    email: str = SQLField(unique=True, index=True, max_length=255)
    full_name: Optional[str] = SQLField(default=None, max_length=100)
    account_type: AccountType
    hashed_password: str = SQLField(max_length=255)
    is_active: bool = SQLField(default=True)
    created_at: datetime = SQLField(default_factory=utcnow)
    updated_at: datetime = SQLField(default_factory=utcnow)


class VitalsRecord(SQLModel, table=True):
    __tablename__ = "vitals_records"

    id: Optional[int] = SQLField(default=None, primary_key=True)
    user_id: int = SQLField(foreign_key="users.id", index=True)
    age: int
    systolic_bp: int
    diastolic_bp: int
    bs: float
    body_temp: float
    body_temp_unit: str = SQLField(max_length=16)
    heart_rate: int
    patient_history: Optional[str] = SQLField(default=None, max_length=1000)
    ml_risk_label: str = SQLField(max_length=32)
    ml_probability: float
    ml_feature_importances: Optional[str] = SQLField(default=None)  # JSON document
    created_at: datetime = SQLField(default_factory=utcnow, index=True)


class ConversationHistory(SQLModel, table=True):
    __tablename__ = "conversation_history"

    id: Optional[int] = SQLField(default=None, primary_key=True)
    user_id: int = SQLField(foreign_key="users.id", index=True)
    vitals_record_id: Optional[int] = SQLField(foreign_key="vitals_records.id", default=None)
    user_message: str = SQLField(max_length=500)
    ai_response: str
    created_at: datetime = SQLField(default_factory=utcnow, index=True)
