import uuid
from datetime import datetime
from sqlalchemy import BigInteger, DateTime, Float, ForeignKey, Integer, String, Text, Uuid
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Renders as INTEGER in SQLite (autoincrement support) but BIGINT in PostgreSQL
_BigIntPK = BigInteger().with_variant(Integer, "sqlite")


class Base(DeclarativeBase):
    pass


class JobDescription(Base):
    """A raw job description submitted for NLP extraction and role classification."""
    __tablename__ = "job_descriptions"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    source: Mapped[str] = mapped_column(String(50), default="manual")   # manual|beacon|api
    source_id: Mapped[str | None] = mapped_column(String(100), index=True)  # external ID (e.g. beacon job_id)
    title: Mapped[str | None] = mapped_column(String(300))
    company: Mapped[str | None] = mapped_column(String(300))
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    word_count: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(30), default="pending")  # pending|processing|completed|failed
    error_message: Mapped[str | None] = mapped_column(Text)
    ground_truth_soc_code: Mapped[str | None] = mapped_column(String(20))  # for evaluation
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    skills: Mapped[list["ExtractedSkill"]] = relationship(back_populates="jd", cascade="all, delete-orphan")
    predictions: Mapped[list["RolePrediction"]] = relationship(back_populates="jd", cascade="all, delete-orphan")


class ExtractedSkill(Base):
    """A skill, tool, or requirement extracted from a JD via spaCy NER."""
    __tablename__ = "extracted_skills"

    id: Mapped[int] = mapped_column(_BigIntPK, primary_key=True, autoincrement=True)
    jd_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("job_descriptions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    text: Mapped[str] = mapped_column(String(300), nullable=False)
    label: Mapped[str] = mapped_column(String(50), nullable=False)   # SKILL|TOOL|LANGUAGE|FRAMEWORK|PLATFORM
    start_char: Mapped[int | None] = mapped_column(Integer)
    end_char: Mapped[int | None] = mapped_column(Integer)
    score: Mapped[float] = mapped_column(Float, default=1.0)

    jd: Mapped["JobDescription"] = relationship(back_populates="skills")


class RolePrediction(Base):
    """A predicted O*NET SOC code for a job description."""
    __tablename__ = "role_predictions"

    id: Mapped[int] = mapped_column(_BigIntPK, primary_key=True, autoincrement=True)
    jd_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("job_descriptions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    soc_code: Mapped[str] = mapped_column(String(20), nullable=False)    # e.g. "15-1252.00"
    soc_title: Mapped[str] = mapped_column(String(300), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)

    jd: Mapped["JobDescription"] = relationship(back_populates="predictions")
