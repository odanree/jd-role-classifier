from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database
    database_url: str = "postgresql+asyncpg://portfolio_user:secret@localhost:5432/jd_role_classifier"

    # NLP / classifier
    spacy_model: str = "en_core_web_sm"
    hf_model_name: str = "google-bert/bert-base-uncased"
    model_path: str = ""  # path to fine-tuned LoRA weights; empty = use base model
    load_classifier: bool = True  # False = skip HF model load (faster CI startup)
    mock_nlp: bool = False  # True = return synthetic extractions (overrides load_classifier)

    # Beacon DB (read-only) — for seeding from the job search pipeline
    beacon_database_url: str = "postgresql+psycopg2://jsp_user:changeme@localhost:15433/job_search"

    # App
    app_env: str = "development"


@lru_cache
def get_settings() -> Settings:
    return Settings()
