from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class SkillOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    text: str
    label: str
    score: float


class RolePredictionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    rank: int
    soc_code: str
    soc_title: str
    confidence: float


class JobDescriptionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    source_id: str | None
    title: str | None
    raw_text: str
    status: str
    created_at: datetime
    processed_at: datetime | None
    skills: list[SkillOut] = []
    predictions: list[RolePredictionOut] = []


class ClassifyRequest(BaseModel):
    text: str
    source_id: str | None = None
    title: str | None = None
    top_k: int = 5


class ClassifyResponse(BaseModel):
    id: uuid.UUID
    source_id: str | None
    title: str | None
    skills: list[SkillOut]
    predictions: list[RolePredictionOut]
    status: str


class EvaluationReport(BaseModel):
    total_samples: int
    precision_at_1: float
    precision_at_3: float
    mrr: float
    skill_extraction_f1: float | None
    notes: str | None = None
