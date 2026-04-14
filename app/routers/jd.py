"""
Job description classification endpoints.

POST /api/v1/jd     — submit text, run NER + classification, persist, return results
GET  /api/v1/jd     — list recent job descriptions
GET  /api/v1/jd/{id} — get a specific job description with skills + predictions
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db import get_db
from app.models.jd import ExtractedSkill, JobDescription, RolePrediction
from app.schemas.jd import ClassifyRequest, ClassifyResponse, JobDescriptionOut, SkillOut, RolePredictionOut
from app.services.classifier import classify_jd
from app.services.nlp import extract_skills

router = APIRouter(prefix="/api/v1", tags=["jd"])


async def _run_pipeline(jd: JobDescription, db: AsyncSession) -> None:
    """Run NER + classification and persist results onto the JD record."""
    try:
        # Extract skills
        raw_skills = extract_skills(jd.raw_text)
        for s in raw_skills:
            skill = ExtractedSkill(
                jd_id=jd.id,
                text=s["text"],
                label=s["label"],
                score=s.get("score", 1.0),
            )
            db.add(skill)

        # Classify role
        predictions = classify_jd(jd.raw_text, top_k=5)
        for pred in predictions:
            role = RolePrediction(
                jd_id=jd.id,
                soc_code=pred["soc_code"],
                soc_title=pred["soc_title"],
                confidence=pred["confidence"],
                rank=pred["rank"],
            )
            db.add(role)

        jd.word_count = len(jd.raw_text.split())
        jd.status = "completed"
        jd.processed_at = datetime.now(timezone.utc)
    except Exception as exc:
        jd.status = "failed"
        jd.error_message = str(exc)


@router.post("/jd", response_model=ClassifyResponse, status_code=201)
async def classify(req: ClassifyRequest, db: AsyncSession = Depends(get_db)):
    """
    Submit a job description for NER skill extraction and O*NET SOC classification.

    Returns top-5 role predictions and extracted tech skills.
    """
    jd = JobDescription(
        raw_text=req.text,
        source_id=req.source_id,
        title=req.title,
        status="processing",
    )
    db.add(jd)
    await db.flush()

    await _run_pipeline(jd, db)
    await db.commit()
    await db.refresh(jd)

    # Reload with relationships
    result = await db.execute(
        select(JobDescription)
        .options(selectinload(JobDescription.skills), selectinload(JobDescription.predictions))
        .where(JobDescription.id == jd.id)
    )
    jd = result.scalar_one()

    return ClassifyResponse(
        id=jd.id,
        source_id=jd.source_id,
        title=jd.title,
        status=jd.status,
        skills=[SkillOut(text=s.text, label=s.label, score=s.score) for s in jd.skills],
        predictions=[
            RolePredictionOut(rank=p.rank, soc_code=p.soc_code, soc_title=p.soc_title, confidence=p.confidence)
            for p in sorted(jd.predictions, key=lambda x: x.rank)
        ],
    )


@router.get("/jd/{jd_id}", response_model=JobDescriptionOut)
async def get_jd(jd_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(JobDescription)
        .options(selectinload(JobDescription.skills), selectinload(JobDescription.predictions))
        .where(JobDescription.id == jd_id)
    )
    jd = result.scalar_one_or_none()
    if not jd:
        raise HTTPException(status_code=404, detail="Job description not found")
    return jd


@router.get("/jd", response_model=list[JobDescriptionOut])
async def list_jds(
    limit: int = 20,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(JobDescription)
        .options(selectinload(JobDescription.skills), selectinload(JobDescription.predictions))
        .order_by(JobDescription.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return result.scalars().all()
