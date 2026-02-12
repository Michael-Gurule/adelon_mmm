# Multi-stage build for Adelon MMM


# Builder stage:
FROM python:3.12-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir --prefix=/install ".[bayesian,dashboard]"

# Runtime stage: slim image
FROM python:3.12-slim-bookworm AS runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl g++ && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code and data
COPY src/ src/
COPY dashboards/ dashboards/
COPY config/ config/
COPY data/ data/

ENV PYTENSOR_FLAGS='device=cpu,floatx=float64,cxx=g++'
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Create artifact directories
RUN mkdir -p traces/ artifacts/

EXPOSE 8501

COPY docker-entrypoint.sh .

HEALTHCHECK --interval=30s --timeout=3s --start-period=300s \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["./docker-entrypoint.sh"]
