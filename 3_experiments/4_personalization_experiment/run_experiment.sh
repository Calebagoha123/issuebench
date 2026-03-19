#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_experiment.sh — Personalization experiment pipeline
#
# Usage:
#   bash run_experiment.sh fast       # ~5-10 min validation run
#   bash run_experiment.sh full       # full statistical run
# ─────────────────────────────────────────────────────────────────────────────

set -e

MODE=${1:-fast}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Resolve python interpreter ────────────────────────────────────────────────
# Prefer the active venv/conda python, fall back to python3
if command -v python &>/dev/null; then
    PYTHON=python
elif command -v python3 &>/dev/null; then
    PYTHON=python3
else
    echo "ERROR: No python interpreter found."
    echo "  Run setup first:  bash setup.sh"
    echo "  Then activate:    source .venv/bin/activate"
    exit 1
fi

echo "Using: $($PYTHON --version) at $(which $PYTHON)"

# Warn if not inside the uv venv
if [ -z "$VIRTUAL_ENV" ]; then
    echo "WARNING: uv venv not active. Run: source .venv/bin/activate"
fi

echo ""
echo "============================================================"
echo " IssueBench Personalization Experiment  —  mode: $MODE"
echo "============================================================"

cd "$SCRIPT_DIR"

echo ""
echo "[0/4] Downloading models (skipped if already cached) ..."
$PYTHON 0_download_models.py

echo ""
echo "[1/4] Generating prompts ..."
$PYTHON 1_generate_subset.py --mode "$MODE"

echo ""
echo "[2/4] Running inference  (Qwen3.5-9B, GPU 0) ..."
$PYTHON 2_run_inference.py --mode "$MODE"

echo ""
echo "[3/4] Running stance eval  (Qwen3.5-4B, GPU 1) ..."
$PYTHON 3_run_stance_eval.py --mode "$MODE"

echo ""
echo "[4/4] Analysing results ..."
$PYTHON 4_analyse.py --mode "$MODE"

echo ""
echo "============================================================"
echo " Done!"
echo " Results : $SCRIPT_DIR/results/"
echo " Figures : $SCRIPT_DIR/results/figures/"
echo "============================================================"
