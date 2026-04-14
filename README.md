# JD Role Classifier

NLP pipeline that extracts tech skills from job descriptions using spaCy NER and classifies them into O*NET SOC occupational codes using BERT + LoRA/PEFT fine-tuning.

Built to demonstrate real-world ML engineering skills on actual job market data from the [beacon job-search-pipeline](../job-search-pipeline).

## Features

- **spaCy NER** — Custom `EntityRuler` patterns for 60+ technologies (languages, frameworks, tools, platforms, skills) layered on `en_core_web_sm`
- **BERT + LoRA** — `google-bert/bert-base-uncased` with PEFT LoRA adapters for parameter-efficient fine-tuning; only ~0.5% of parameters trained
- **O*NET SOC Taxonomy** — 30 software/AI/data-specific occupational codes (e.g., `15-2053.00 Machine Learning Engineers`, `15-1299.09 Data Engineers`)
- **Evaluation Metrics** — Precision@1, Precision@3, MRR computed against ground-truth labels
- **Beacon Integration** — `seed_from_beacon.py` pulls real JDs from the beacon PostgreSQL DB for live classification
- **Mock NLP mode** — Keyword-based fallback for local dev and CI (no GPU required)

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/jd` | Submit a JD for skill extraction + role classification |
| `GET` | `/api/v1/jd` | List recent classified JDs |
| `GET` | `/api/v1/jd/{id}` | Get a specific JD with skills + predictions |
| `GET` | `/api/v1/evaluate` | Compute precision@1, precision@3, MRR metrics |
| `GET` | `/health` | Liveness check |
| `GET` | `/health/db` | Database connectivity check |

## Quick Start

```bash
# Copy and configure environment
cp .env.example .env
# Set: DATABASE_URL, MOCK_NLP=false (and MODEL_PATH if using fine-tuned model)

# Install spaCy model
python -m spacy download en_core_web_sm

# Run with Docker Compose (from portfolio-infra)
docker compose up -d jd-classifier-api

# Classify a JD
curl -X POST http://localhost:8002/api/v1/jd \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Senior ML Engineer with PyTorch, RAG, LLM fine-tuning, AWS SageMaker",
    "title": "Senior ML Engineer"
  }' | jq .

# Seed from beacon (requires beacon DB access)
python scripts/seed_from_beacon.py \
  --beacon-url postgresql://user:pass@localhost:5432/beacon \
  --limit 500
```

## Development

```bash
# Install dependencies
pip install -e ".[dev]"
python -m spacy download en_core_web_sm

# Run tests (MOCK_NLP=true, SQLite in-memory)
pytest tests/ -v --cov=app --cov-report=term-missing

# Lint
ruff check . --select=E,F,W --ignore=E501,E402,F401

# Run locally (mock NLP mode, no GPU needed)
MOCK_NLP=true LOAD_CLASSIFIER=false DATABASE_URL="sqlite+aiosqlite:///./dev.db" \
  uvicorn app.main:app --reload --port 8002
```

## Fine-tuning

The `scripts/fine_tune.py` script trains a LoRA-adapted BERT classifier on labeled JD data:

```bash
# Prepare labeled data as JSONL:
# {"text": "Senior ML Engineer with PyTorch...", "soc_code": "15-2053.00"}

# Fine-tune (requires GPU for reasonable training time)
python scripts/fine_tune.py \
  --data-path data/labeled_jds.jsonl \
  --output-dir models/bert-lora-onet \
  --epochs 3 \
  --lora-r 16 \
  --lora-alpha 32

# Enable in API
export MODEL_PATH=models/bert-lora-onet
export LOAD_CLASSIFIER=true
uvicorn app.main:app --reload
```

### LoRA Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Base model | `google-bert/bert-base-uncased` | 110M parameters |
| LoRA rank (r) | 16 | Adapter dimensionality |
| LoRA alpha | 32 | Scaling factor |
| Target modules | `query`, `value` | Attention projection matrices |
| Trainable params | ~0.5% | ~500K vs 110M base |
| Task type | `SEQ_CLS` | Sequence classification |

## Architecture

```
POST /api/v1/jd
    │
    ├─ 1. spaCy NER with EntityRuler
    │      → Extract SKILL, TOOL, LANGUAGE, FRAMEWORK, PLATFORM entities
    │
    ├─ 2. BERT + LoRA classification
    │      → Top-5 O*NET SOC code predictions with confidence scores
    │      → Fallback: keyword-overlap scoring (mock mode)
    │
    └─ 3. Persist to PostgreSQL
           → job_descriptions, extracted_skills, role_predictions tables

GET /api/v1/evaluate
    │
    └─ Compute Precision@1, Precision@3, MRR
       against ground_truth_soc_code labels
```

## O*NET SOC Codes Covered

| SOC Code | Title |
|----------|-------|
| 15-1252.00 | Software Developers |
| 15-2053.00 | Machine Learning Engineers |
| 15-1299.09 | Data Engineers |
| 15-2051.00 | Data Scientists |
| 15-2052.00 | Data Analysts |
| 15-1299.08 | DevOps Engineers |
| 15-1256.00 | Cloud Engineers |
| 15-1254.00 | Web Developers |
| 15-1212.00 | Information Security Analysts |
| 15-1299.06 | Prompt Engineers / AI Application Engineers |
| 15-1242.00 | Database Administrators |
| 15-1253.00 | Software Quality Assurance Analysts and Testers |
| ... | 30 codes total covering software/AI/data roles |
