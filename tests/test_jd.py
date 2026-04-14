"""
JD classification endpoint tests using MOCK_NLP=true.
"""
import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession


SAMPLE_JD = """
We are looking for a Senior Machine Learning Engineer to join our AI platform team.

Responsibilities:
- Design and implement scalable ML pipelines using Python and PyTorch
- Deploy models to production using Docker, Kubernetes, and AWS SageMaker
- Build RAG systems with LLM APIs (OpenAI, Anthropic) and vector databases (Qdrant, Pinecone)
- Implement LoRA fine-tuning for domain-specific language models
- Collaborate with data scientists using Jupyter, pandas, and scikit-learn
- Write RESTful APIs with FastAPI and PostgreSQL backends

Requirements:
- 5+ years Python, 3+ years ML engineering
- Strong experience with PyTorch, Hugging Face transformers, PEFT/LoRA
- AWS or Azure cloud experience
- Familiarity with CI/CD pipelines (GitHub Actions, Jenkins)
"""


@pytest.mark.asyncio
async def test_classify_jd(client: AsyncClient):
    resp = await client.post(
        "/api/v1/jd",
        json={
            "text": SAMPLE_JD,
            "title": "Senior ML Engineer",
            "source_id": "test-001",
        },
    )
    assert resp.status_code == 201
    body = resp.json()
    assert "id" in body
    assert body["status"] == "completed"
    assert len(body["predictions"]) > 0
    assert body["predictions"][0]["rank"] == 1
    assert "soc_code" in body["predictions"][0]
    assert "soc_title" in body["predictions"][0]
    assert body["predictions"][0]["confidence"] > 0


@pytest.mark.asyncio
async def test_classify_extracts_skills(client: AsyncClient):
    resp = await client.post(
        "/api/v1/jd",
        json={"text": SAMPLE_JD},
    )
    assert resp.status_code == 201
    body = resp.json()
    skill_texts = [s["text"] for s in body["skills"]]
    # Mock NLP should pick up at least some well-known keywords
    assert len(body["skills"]) > 0
    for s in body["skills"]:
        assert s["label"] in {"SKILL", "TOOL", "LANGUAGE", "FRAMEWORK", "PLATFORM"}
        assert s["score"] >= 0.0


@pytest.mark.asyncio
async def test_get_jd(client: AsyncClient):
    # Create first
    create_resp = await client.post("/api/v1/jd", json={"text": SAMPLE_JD})
    assert create_resp.status_code == 201
    jd_id = create_resp.json()["id"]

    # Fetch by ID
    resp = await client.get(f"/api/v1/jd/{jd_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == jd_id
    assert body["status"] == "completed"


@pytest.mark.asyncio
async def test_list_jds(client: AsyncClient):
    # Create two
    await client.post("/api/v1/jd", json={"text": SAMPLE_JD, "title": "ML Engineer"})
    await client.post("/api/v1/jd", json={"text": "Data analyst position with SQL and Python experience.", "title": "Data Analyst"})

    resp = await client.get("/api/v1/jd?limit=10")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) >= 2


@pytest.mark.asyncio
async def test_get_jd_not_found(client: AsyncClient):
    import uuid
    resp = await client.get(f"/api/v1/jd/{uuid.uuid4()}")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_evaluate_no_ground_truth(client: AsyncClient):
    # No JDs with ground truth → returns zeros
    resp = await client.get("/api/v1/evaluate")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_samples"] == 0
    assert body["precision_at_1"] == 0.0
    assert body["mrr"] == 0.0
