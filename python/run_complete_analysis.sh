#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate || true
pip install -r python/requirements.txt
python python/analytics_engine.py --results simulation_results.csv --all
ls -1 dist_cost.png dist_service.png feat_importance.png feature_importance_top10.csv || true
