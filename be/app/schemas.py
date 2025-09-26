from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime

# ----- Auth Schemas -----

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    age: Optional[int] = None
    gender: Optional[str] = None
    major: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

# ----- Chat Schemas -----

class ChatResponse(BaseModel):
    answer: str

class ChatHistoryOut(BaseModel):
    question: str
    answer: str
    timestamp: datetime

class UserProfile(BaseModel):
    email: str
    age: Optional[int] = None
    gender: Optional[str] = None
    major: Optional[str] = None

    class Config:
        orm_mode = True

from pydantic import validator

ALLOWED_MODELS = ["gemini", "gpt", "gemma"]

class ChatRequest(BaseModel):
    question: str
    model_name: Optional[str] = "gemini"

    @validator("model_name")
    def validate_model(cls, v):
        if v.lower() not in ALLOWED_MODELS:
            raise ValueError(f"Invalid model. Choose from: {', '.join(ALLOWED_MODELS)}")
        return v.lower()


