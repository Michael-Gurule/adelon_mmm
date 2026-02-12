# ============================================================
# Multi-stage build for Adelon MMM
# ============================================================

# --- Builder stage: compile native dependencies ---
FROM python:3.12-slim AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ gfortran && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

COPY pyproject.toml .
COPY src/ src/
COPY config/ config/
COPY data/ data/
COPY dashboards/ dashboards/

RUN pip install --no-cache-dir --prefix=/install -e .

# --- Runtime stage: slim image ---
FROM python:3.12-slim AS runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local
COPY --from=builder /build /app

# Create artifact directories
RUN mkdir -p traces/ artifacts/

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "dashboards/app.py", \
            "--server.port=8501", "--server.address=0.0.0.0"]
