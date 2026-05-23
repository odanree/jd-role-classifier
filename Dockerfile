FROM python:3.12-slim AS builder
WORKDIR /build
COPY . .
# Install the CPU-only torch first. The default PyPI torch bundles CUDA
# (nvidia-* + triton wheels, ~5GB) which this GPU-less VPS can't use and which
# repeatedly filled the disk during image extraction. Installing it before
# `pip install .` satisfies the torch>=2.2.0 constraint so the CUDA build is
# never pulled.
RUN pip install --upgrade pip && pip install hatchling && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir . && \
    python -m spacy download en_core_web_sm

FROM python:3.12-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
RUN groupadd -r appgroup && useradd -r -g appgroup appuser
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY . .
RUN chown -R appuser:appgroup /app
USER appuser
# HuggingFace BERT model downloaded on first request (controlled by LOAD_CLASSIFIER env var).
# Set MOCK_NLP=true to skip all model loading for lightweight deployments.
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1
CMD ["/bin/sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
