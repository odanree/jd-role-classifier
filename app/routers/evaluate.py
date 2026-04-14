from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.schemas.jd import EvaluationReport
from app.services.evaluator import compute_evaluation_report

router = APIRouter(prefix="/api/v1", tags=["evaluate"])


@router.get("/evaluate", response_model=EvaluationReport)
async def evaluate(db: AsyncSession = Depends(get_db)):
    """
    Compute aggregate evaluation metrics across all JDs with ground-truth SOC labels.

    Metrics:
    - precision@1: fraction of JDs where top prediction matches ground truth
    - precision@3: fraction of JDs where ground truth appears in top-3 predictions
    - MRR: mean reciprocal rank of the correct SOC code
    """
    report = await compute_evaluation_report(db)
    return EvaluationReport(**report)
