"""Initial schema: job_descriptions, extracted_skills, role_predictions

Revision ID: a1b2c3d4e5f6
Revises:
Create Date: 2026-04-14 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "a1b2c3d4e5f6"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "job_descriptions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source", sa.String(50), server_default="manual", nullable=False),
        sa.Column("source_id", sa.String(100), nullable=True),
        sa.Column("title", sa.String(300), nullable=True),
        sa.Column("company", sa.String(300), nullable=True),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column("word_count", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(30), server_default="pending", nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("ground_truth_soc_code", sa.String(20), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_jd_source_id", "job_descriptions", ["source_id"])
    op.create_index("ix_jd_status", "job_descriptions", ["status"])

    op.create_table(
        "extracted_skills",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("jd_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("text", sa.String(300), nullable=False),
        sa.Column("label", sa.String(50), nullable=False),
        sa.Column("start_char", sa.Integer(), nullable=True),
        sa.Column("end_char", sa.Integer(), nullable=True),
        sa.Column("score", sa.Float(), server_default="1.0", nullable=False),
        sa.ForeignKeyConstraint(["jd_id"], ["job_descriptions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_extracted_skills_jd_id", "extracted_skills", ["jd_id"])

    op.create_table(
        "role_predictions",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("jd_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("soc_code", sa.String(20), nullable=False),
        sa.Column("soc_title", sa.String(300), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["jd_id"], ["job_descriptions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_role_predictions_jd_id", "role_predictions", ["jd_id"])


def downgrade() -> None:
    op.drop_index("ix_role_predictions_jd_id", table_name="role_predictions")
    op.drop_table("role_predictions")
    op.drop_index("ix_extracted_skills_jd_id", table_name="extracted_skills")
    op.drop_table("extracted_skills")
    op.drop_index("ix_jd_status", table_name="job_descriptions")
    op.drop_index("ix_jd_source_id", table_name="job_descriptions")
    op.drop_table("job_descriptions")
