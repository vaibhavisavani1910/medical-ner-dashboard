from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, field_validator


class RawRecord(BaseModel):
    id: str
    text: str


class ExtractedEntities(BaseModel):
    record_id: str
    conditions: list[str] = []
    symptoms: list[str] = []
    medications: list[str] = []
    procedures: list[str] = []
    error: Optional[str] = None

    @field_validator("conditions", "symptoms", "medications", "procedures", mode="before")
    @classmethod
    def lowercase_and_strip(cls, v: list) -> list:
        if not isinstance(v, list):
            return []
        return [str(item).lower().strip() for item in v if item and str(item).strip()]


class EntityCount(BaseModel):
    name: str
    count: int


class CodedEntity(BaseModel):
    name: str
    count: int
    code: Optional[str] = None
    code_system: str
    code_description: Optional[str] = None


class DashboardData(BaseModel):
    conditions: list[CodedEntity]
    symptoms: list[CodedEntity]
    medications: list[CodedEntity]
    procedures: list[CodedEntity]
