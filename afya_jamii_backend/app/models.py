from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum
from sqlmodel import SQLModel, Field as SQLField

class AccountType(str, Enum):
    PREGNANT = "pregnant"
    POSTNATAL = "postnatal"
    GENERAL = "general"

# Pydantic Models for API
class UserBase(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
    account_type: AccountType
    full_name: Optional[str] = Field(None, max_length=100)

class UserCreate(UserBase):
    password: str = Field(..., min_length=8)

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(UserBase):
    id: int
    created_at: datetime
    is_active: bool = True
    
    class Config:
        from_attributes = True

class VitalsInput(BaseModel):
    age: int = Field(..., ge=15, le=50, description="Age in years (15-50)")
    systolic_bp: int = Field(..., ge=70, le=200, description="Systolic BP in mmHg")
    diastolic_bp: int = Field(..., ge=40, le=130, description="Diastolic BP in mmHg")
    bs: float = Field(..., ge=3.0, le=30.0, description="Blood sugar in mmol/L")
    body_temp: float = Field(..., ge=35.0, le=42.0, description="Body temperature")
    body_temp_unit: str = Field("celsius", pattern="^(celsius|fahrenheit)$")
    heart_rate: int = Field(..., ge=40, le=150, description="Heart rate in bpm")
    patient_history: Optional[str] = Field(None, max_length=1000)
    
    @validator('body_temp')
    def validate_body_temp(cls, v, values):
        if 'body_temp_unit' in values and values['body_temp_unit'] == 'fahrenheit':
            if v < 95 or v > 107.6:  # 35°C to 42°C in Fahrenheit
                raise ValueError('Body temperature out of range for Fahrenheit')
        return v

class VitalsSubmission(BaseModel):
    vitals: VitalsInput
    account_type: AccountType

class MLModelOutput(BaseModel):
    risk_label: str
    probability: float
    feature_importances: Optional[Dict[str, float]] = None

class LLMAdviceRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)

class LLMAdviceResponse(BaseModel):
    advice: str
    timestamp: datetime

class CombinedResponse(BaseModel):
    user_id: int
    submission_id: int
    timestamp: datetime
    ml_output: MLModelOutput
    llm_advice: LLMAdviceResponse

class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int

class TokenData(BaseModel):
    username: Optional[str] = None

# Database Models
class UserDB(SQLModel, table=True):
    __tablename__ = "users"
    
    id: Optional[int] = SQLField(default=None, primary_key=True)
    username: str = SQLField(unique=True, index=True, max_length=50)
    email: str = SQLField(unique=True, index=True, max_length=255)
    full_name: Optional[str] = SQLField(default=None, max_length=100)
    account_type: AccountType
    hashed_password: str = SQLField(max_length=255)
    is_active: bool = SQLField(default=True)
    created_at: datetime = SQLField(default_factory=datetime.utcnow)
    updated_at: datetime = SQLField(default_factory=datetime.utcnow)

class VitalsRecord(SQLModel, table=True):
    __tablename__ = "vitals_records"
    
    id: Optional[int] = SQLField(default=None, primary_key=True)
    user_id: int = SQLField(foreign_key="users.id")
    age: int
    systolic_bp: int
    diastolic_bp: int
    bs: float
    body_temp: float
    body_temp_unit: str
    heart_rate: int
    patient_history: Optional[str] = SQLField(default=None, max_length=1000)
    ml_risk_label: str
    ml_probability: float
    ml_feature_importances: Optional[str] = SQLField(default=None)  # JSON string
    created_at: datetime = SQLField(default_factory=datetime.utcnow)

class ConversationHistory(SQLModel, table=True):
    __tablename__ = "conversation_history"
    
    id: Optional[int] = SQLField(default=None, primary_key=True)
    user_id: int = SQLField(foreign_key="users.id")
    vitals_record_id: Optional[int] = SQLField(foreign_key="vitals_records.id", default=None)
    user_message: str = SQLField(max_length=500)
    ai_response: str
    created_at: datetime = SQLField(default_factory=datetime.utcnow)