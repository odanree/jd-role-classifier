"""
Evaluation metrics for the JD role classifier.

Computes:
  - Precision@1 and Precision@3 (does correct SOC code appear in top-k?)
  - MRR (Mean Reciprocal Rank)
  - Skill extraction F1 (if ground-truth skills provided)

These follow the same spirit as RAGAS faithfulness/precision metrics —
measuring retrieval and generation quality against known ground-truth labels.
"""
from __future__ import annotations

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.jd import JobDescription, RolePrediction


async def compute_evaluation_report(db: AsyncSession) -> dict:
    """
    Compute aggregate evaluation metrics across all evaluated job descriptions.

    A JD is considered "evaluated" if it has at least one RolePrediction
    and a ground_truth_soc_code stored.
    """
    # Load all JDs that have ground_truth + predictions
    result = await db.execute(
        select(JobDescription).where(
            JobDescription.status == "completed",
            JobDescription.ground_truth_soc_code.is_not(None),
        )
    )
    jds = result.scalars().all()

    if not jds:
        return {
            "total_samples": 0,
            "precision_at_1": 0.0,
            "precision_at_3": 0.0,
            "mrr": 0.0,
            "skill_extraction_f1": None,
            "notes": "No evaluated samples with ground truth labels found.",
        }

    p1_hits = 0
    p3_hits = 0
    reciprocal_ranks = []

    for jd in jds:
        result = await db.execute(
            select(RolePrediction)
            .where(RolePrediction.jd_id == jd.id)
            .order_by(RolePrediction.rank)
        )
        preds = result.scalars().all()
        if not preds:
            reciprocal_ranks.append(0.0)
            continue

        codes = [p.soc_code for p in preds]
        truth = jd.ground_truth_soc_code

        if codes[0] == truth:
            p1_hits += 1
        if truth in codes[:3]:
            p3_hits += 1

        try:
            rank = codes.index(truth) + 1
            reciprocal_ranks.append(1.0 / rank)
        except ValueError:
            reciprocal_ranks.append(0.0)

    n = len(jds)
    return {
        "total_samples": n,
        "precision_at_1": round(p1_hits / n, 4),
        "precision_at_3": round(p3_hits / n, 4),
        "mrr": round(sum(reciprocal_ranks) / n, 4),
        "skill_extraction_f1": None,
        "notes": f"Evaluated {n} job descriptions with ground truth SOC labels.",
    }
