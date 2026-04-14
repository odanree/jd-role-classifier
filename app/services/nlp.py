"""
spaCy-based NER for tech skill extraction.

Uses a standard en_core_web_sm pipeline augmented with a custom EntityRuler
that matches technology-specific patterns (languages, frameworks, tools, platforms).
"""
from __future__ import annotations

import re
from functools import lru_cache

from app.config import get_settings


TECH_PATTERNS = [
    # Programming languages
    {"label": "LANGUAGE", "pattern": "Python"},
    {"label": "LANGUAGE", "pattern": "TypeScript"},
    {"label": "LANGUAGE", "pattern": "JavaScript"},
    {"label": "LANGUAGE", "pattern": "Go"},
    {"label": "LANGUAGE", "pattern": "Golang"},
    {"label": "LANGUAGE", "pattern": "Rust"},
    {"label": "LANGUAGE", "pattern": "Java"},
    {"label": "LANGUAGE", "pattern": "Scala"},
    {"label": "LANGUAGE", "pattern": "Kotlin"},
    {"label": "LANGUAGE", "pattern": "C++"},
    {"label": "LANGUAGE", "pattern": "C#"},
    {"label": "LANGUAGE", "pattern": "Ruby"},
    {"label": "LANGUAGE", "pattern": "PHP"},
    {"label": "LANGUAGE", "pattern": "Swift"},
    {"label": "LANGUAGE", "pattern": "SQL"},
    {"label": "LANGUAGE", "pattern": "R"},
    # ML / AI frameworks
    {"label": "FRAMEWORK", "pattern": "PyTorch"},
    {"label": "FRAMEWORK", "pattern": "TensorFlow"},
    {"label": "FRAMEWORK", "pattern": "JAX"},
    {"label": "FRAMEWORK", "pattern": "Keras"},
    {"label": "FRAMEWORK", "pattern": "Hugging Face"},
    {"label": "FRAMEWORK", "pattern": [{"LOWER": "hugging"}, {"LOWER": "face"}]},
    {"label": "FRAMEWORK", "pattern": "scikit-learn"},
    {"label": "FRAMEWORK", "pattern": "LangChain"},
    {"label": "FRAMEWORK", "pattern": "LlamaIndex"},
    {"label": "FRAMEWORK", "pattern": "FastAPI"},
    {"label": "FRAMEWORK", "pattern": "Django"},
    {"label": "FRAMEWORK", "pattern": "Flask"},
    {"label": "FRAMEWORK", "pattern": "React"},
    {"label": "FRAMEWORK", "pattern": "Next.js"},
    {"label": "FRAMEWORK", "pattern": "Vue"},
    {"label": "FRAMEWORK", "pattern": "Angular"},
    {"label": "FRAMEWORK", "pattern": "Spark"},
    {"label": "FRAMEWORK", "pattern": "Apache Spark"},
    {"label": "FRAMEWORK", "pattern": [{"LOWER": "apache"}, {"LOWER": "spark"}]},
    {"label": "FRAMEWORK", "pattern": "Kafka"},
    {"label": "FRAMEWORK", "pattern": "Airflow"},
    {"label": "FRAMEWORK", "pattern": "dbt"},
    # Tools / technologies
    {"label": "TOOL", "pattern": "Docker"},
    {"label": "TOOL", "pattern": "Kubernetes"},
    {"label": "TOOL", "pattern": "Terraform"},
    {"label": "TOOL", "pattern": "Ansible"},
    {"label": "TOOL", "pattern": "Git"},
    {"label": "TOOL", "pattern": "GitHub Actions"},
    {"label": "TOOL", "pattern": [{"LOWER": "github"}, {"LOWER": "actions"}]},
    {"label": "TOOL", "pattern": "Jenkins"},
    {"label": "TOOL", "pattern": "CircleCI"},
    {"label": "TOOL", "pattern": "Grafana"},
    {"label": "TOOL", "pattern": "Prometheus"},
    {"label": "TOOL", "pattern": "Elasticsearch"},
    {"label": "TOOL", "pattern": "Redis"},
    {"label": "TOOL", "pattern": "PostgreSQL"},
    {"label": "TOOL", "pattern": "MySQL"},
    {"label": "TOOL", "pattern": "MongoDB"},
    {"label": "TOOL", "pattern": "Snowflake"},
    {"label": "TOOL", "pattern": "Databricks"},
    {"label": "TOOL", "pattern": "dbt"},
    {"label": "TOOL", "pattern": "Qdrant"},
    {"label": "TOOL", "pattern": "Pinecone"},
    {"label": "TOOL", "pattern": "Weaviate"},
    # Platforms / cloud
    {"label": "PLATFORM", "pattern": "AWS"},
    {"label": "PLATFORM", "pattern": "Azure"},
    {"label": "PLATFORM", "pattern": "GCP"},
    {"label": "PLATFORM", "pattern": "Google Cloud"},
    {"label": "PLATFORM", "pattern": [{"LOWER": "google"}, {"LOWER": "cloud"}]},
    {"label": "PLATFORM", "pattern": "Vertex AI"},
    {"label": "PLATFORM", "pattern": [{"LOWER": "vertex"}, {"LOWER": "ai"}]},
    {"label": "PLATFORM", "pattern": "SageMaker"},
    {"label": "PLATFORM", "pattern": "Bedrock"},
    {"label": "PLATFORM", "pattern": "OpenAI"},
    {"label": "PLATFORM", "pattern": "Anthropic"},
    # Skills / concepts
    {"label": "SKILL", "pattern": "RAG"},
    {"label": "SKILL", "pattern": "retrieval-augmented generation"},
    {"label": "SKILL", "pattern": [{"LOWER": "retrieval"}, {"OP": "?"}, {"LOWER": "augmented"}, {"LOWER": "generation"}]},
    {"label": "SKILL", "pattern": "LLM"},
    {"label": "SKILL", "pattern": "large language model"},
    {"label": "SKILL", "pattern": "fine-tuning"},
    {"label": "SKILL", "pattern": "LoRA"},
    {"label": "SKILL", "pattern": "RLHF"},
    {"label": "SKILL", "pattern": "vector database"},
    {"label": "SKILL", "pattern": [{"LOWER": "vector"}, {"LOWER": "database"}]},
    {"label": "SKILL", "pattern": "embeddings"},
    {"label": "SKILL", "pattern": "transformer"},
    {"label": "SKILL", "pattern": "BERT"},
    {"label": "SKILL", "pattern": "GPT"},
    {"label": "SKILL", "pattern": "CI/CD"},
    {"label": "SKILL", "pattern": "microservices"},
    {"label": "SKILL", "pattern": "REST API"},
    {"label": "SKILL", "pattern": [{"LOWER": "rest"}, {"LOWER": "api"}]},
    {"label": "SKILL", "pattern": "GraphQL"},
    {"label": "SKILL", "pattern": "data pipeline"},
    {"label": "SKILL", "pattern": [{"LOWER": "data"}, {"LOWER": "pipeline"}]},
    {"label": "SKILL", "pattern": "machine learning"},
    {"label": "SKILL", "pattern": [{"LOWER": "machine"}, {"LOWER": "learning"}]},
    {"label": "SKILL", "pattern": "deep learning"},
    {"label": "SKILL", "pattern": [{"LOWER": "deep"}, {"LOWER": "learning"}]},
    {"label": "SKILL", "pattern": "natural language processing"},
    {"label": "SKILL", "pattern": "NLP"},
    {"label": "SKILL", "pattern": "computer vision"},
    {"label": "SKILL", "pattern": [{"LOWER": "computer"}, {"LOWER": "vision"}]},
    {"label": "SKILL", "pattern": "distributed systems"},
    {"label": "SKILL", "pattern": [{"LOWER": "distributed"}, {"LOWER": "systems"}]},
    {"label": "SKILL", "pattern": "agile"},
    {"label": "SKILL", "pattern": "scrum"},
]


@lru_cache(maxsize=1)
def _load_nlp():
    settings = get_settings()
    if settings.mock_nlp:
        return None
    import spacy
    from spacy.pipeline import EntityRuler
    nlp = spacy.load(settings.spacy_model)
    ruler = nlp.add_pipe("entity_ruler", before="ner", config={"overwrite_ents": True})
    ruler.add_patterns(TECH_PATTERNS)
    return nlp


def _mock_extract(text: str) -> list[dict]:
    """Return synthetic skills by scanning for known tech keywords."""
    found = []
    text_lower = text.lower()
    keyword_map = [
        ("python", "Python", "LANGUAGE"),
        ("typescript", "TypeScript", "LANGUAGE"),
        ("pytorch", "PyTorch", "FRAMEWORK"),
        ("react", "React", "FRAMEWORK"),
        ("docker", "Docker", "TOOL"),
        ("kubernetes", "Kubernetes", "TOOL"),
        ("aws", "AWS", "PLATFORM"),
        ("azure", "Azure", "PLATFORM"),
        ("rag", "RAG", "SKILL"),
        ("llm", "LLM", "SKILL"),
        ("machine learning", "machine learning", "SKILL"),
        ("sql", "SQL", "LANGUAGE"),
        ("postgresql", "PostgreSQL", "TOOL"),
        ("fastapi", "FastAPI", "FRAMEWORK"),
    ]
    seen = set()
    for kw, display, label in keyword_map:
        if kw in text_lower and display not in seen:
            found.append({"text": display, "label": label, "score": 1.0})
            seen.add(display)
    return found


def extract_skills(text: str) -> list[dict]:
    """
    Extract tech skills from a job description text.

    Returns list of {text, label, score} dicts.
    """
    settings = get_settings()
    if settings.mock_nlp:
        return _mock_extract(text)

    nlp = _load_nlp()
    doc = nlp(text)
    seen = set()
    results = []
    for ent in doc.ents:
        if ent.label_ in {"SKILL", "TOOL", "LANGUAGE", "FRAMEWORK", "PLATFORM"}:
            key = (ent.text.lower(), ent.label_)
            if key not in seen:
                seen.add(key)
                results.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "score": 1.0,
                })
    return results
