# Multi-stage build for Adelon MMM


# Builder stage: install deps, run pipeline, produce artifacts
FROM python:3.12-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/
COPY config/ config/
COPY data/ data/
RUN pip install --no-cache-dir ".[bayesian,dashboard]"

# Run full pipeline: generate data -> train model -> evaluate
RUN mkdir -p traces/ artifacts/ && adelon-run

# Runtime stage: slim image with pre-built artifacts
FROM python:3.12-slim-bookworm AS runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl g++ && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local /usr/local

# Copy application code, data, and pre-built artifacts
COPY --from=builder /app/src/ src/
COPY --from=builder /app/config/ config/
COPY --from=builder /app/data/ data/
COPY --from=builder /app/traces/ traces/
COPY --from=builder /app/artifacts/ artifacts/
COPY dashboards/ dashboards/

ENV PYTENSOR_FLAGS='device=cpu,cxx=g++'
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=3s \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "dashboards/app.py", \
            "--server.port=8501", "--server.address=0.0.0.0"]
