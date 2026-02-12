#!/bin/bash
set -e

echo "============================================================"
echo "Adelon MMM — Container Startup"
echo "============================================================"

# Run the full pipeline: generate -> train -> evaluate
# Uses --skip-generate since synthetic data is baked into the image
echo "Running pipeline (train + evaluate)..."
adelon-run --skip-generate

echo "============================================================"
echo "Pipeline complete — launching dashboard"
echo "============================================================"

exec streamlit run dashboards/app.py \
    --server.port=8501 \
    --server.address=0.0.0.0
