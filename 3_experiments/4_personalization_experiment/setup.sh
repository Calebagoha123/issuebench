#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup.sh — Create the uv virtual environment and install dependencies
#
# Run once before the experiment:
#   bash setup.sh
#
# Then activate and run:
#   source .venv/bin/activate
#   bash run_experiment.sh fast
# ─────────────────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Check uv is available ─────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "uv $(uv --version)"

# ── Create venv ───────────────────────────────────────────────────────────────
echo ""
echo "Creating virtual environment at .venv ..."
uv venv .venv --python 3.11

# ── Install PyTorch with CUDA ─────────────────────────────────────────────────
# Detect CUDA version from the system; fall back to cu121
CUDA_VERSION=$(nvcc --version 2>/dev/null | grep -oP "release \K[0-9]+\.[0-9]+" | tr -d '.' | head -1)

if [ -z "$CUDA_VERSION" ]; then
    echo "nvcc not found — defaulting to CUDA 12.1 torch build."
    CUDA_VERSION="121"
fi

TORCH_INDEX="https://download.pytorch.org/whl/cu${CUDA_VERSION}"
echo ""
echo "Installing PyTorch for CUDA cu${CUDA_VERSION} ..."
uv pip install torch torchvision --index-url "$TORCH_INDEX" --python .venv/bin/python

# ── Install remaining dependencies ────────────────────────────────────────────
echo ""
echo "Installing remaining dependencies from requirements.txt ..."
uv pip install -r requirements.txt --python .venv/bin/python

echo ""
echo "============================================================"
echo " Setup complete!"
echo ""
echo " Activate the environment:"
echo "   source .venv/bin/activate"
echo ""
echo " Then run the experiment:"
echo "   bash run_experiment.sh fast"
echo "============================================================"
