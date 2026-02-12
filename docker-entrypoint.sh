#!/bin/bash
set -e

echo "============================================================"
echo "Adelon MMM — Container Startup"
echo "============================================================"

# Run the full pipeline: generate -> train -> evaluate
echo "Running full pipeline..."
adelon-run

echo "============================================================"
echo "Pipeline complete — launching dashboard"
echo "============================================================"

exec streamlit run dashboards/app.py \
    --server.port=8501 \
    --server.address=0.0.0.0
