#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_experiment.sh — Personalization experiment pipeline
#
# Usage:
#   bash run_experiment.sh fast       # ~5-10 min validation run
#   bash run_experiment.sh full       # full statistical run
#
# Steps:
#   0. Download models to HF cache  (skipped if already cached)
#   1. Generate subset prompts
#   2. Run Qwen3.5-9B inference        [GPU 0]
#   3. Run Qwen3.5-4B stance eval      [GPU 1]
#   4. Analyse results + produce figures
# ─────────────────────────────────────────────────────────────────────────────

set -e

MODE=${1:-fast}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo " IssueBench Personalization Experiment  —  mode: $MODE"
echo "============================================================"

cd "$SCRIPT_DIR"

echo ""
echo "[0/4] Downloading models (skipped if already cached) ..."
python 0_download_models.py

echo ""
echo "[1/4] Generating prompts ..."
python 1_generate_subset.py --mode "$MODE"

echo ""
echo "[2/4] Running inference  (Qwen3.5-9B, GPU 0) ..."
python 2_run_inference.py --mode "$MODE"

echo ""
echo "[3/4] Running stance eval  (Qwen3.5-4B, GPU 1) ..."
python 3_run_stance_eval.py --mode "$MODE"

echo ""
echo "[4/4] Analysing results ..."
python 4_analyse.py --mode "$MODE"

echo ""
echo "============================================================"
echo " Done!"
echo " Results : $SCRIPT_DIR/results/"
echo " Figures : $SCRIPT_DIR/results/figures/"
echo "============================================================"
