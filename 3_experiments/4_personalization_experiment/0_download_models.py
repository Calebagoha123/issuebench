"""
Step 0: Download Qwen3.5-9B and Qwen3.5-4B into the shared HuggingFace cache.

Run this once before the experiment. Downloads ~19 GB + ~8 GB of model weights.
Subsequent runs are instant (files already cached).

Usage:
  python 0_download_models.py
  python 0_download_models.py --model gen      # only download generation model
  python 0_download_models.py --model eval     # only download eval model
"""

import os
import sys
import argparse
from huggingface_hub import snapshot_download, HfApi

sys.path.insert(0, os.path.dirname(__file__))
from config import HF_CACHE_DIR, GEN_MODEL_ID, EVAL_MODEL_ID


def check_already_downloaded(model_id: str, cache_dir: str) -> bool:
    """Check if the model snapshot is already fully cached."""
    # HF cache dir format: models--{org}--{model}/snapshots/
    folder = "models--" + model_id.replace("/", "--")
    snapshots_dir = os.path.join(cache_dir, folder, "snapshots")
    if not os.path.isdir(snapshots_dir):
        return False
    # At least one snapshot hash folder should exist and be non-empty
    hashes = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
    return len(hashes) > 0


def download_model(model_id: str, cache_dir: str):
    if check_already_downloaded(model_id, cache_dir):
        print(f"  [{model_id}] Already cached at {cache_dir} — skipping download.")
        return

    print(f"  [{model_id}] Downloading to {cache_dir} ...")
    print(f"  This may take several minutes depending on network speed.")

    snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        # Prefer safetensors; skip .pt/.bin to save space if both exist
        ignore_patterns=["*.pt", "*.bin", "original/*"],
        resume_download=True,   # resume interrupted downloads safely
    )
    print(f"  [{model_id}] Download complete.")


def verify_model(model_id: str, cache_dir: str):
    """Quick sanity check: list files in the snapshot."""
    folder = "models--" + model_id.replace("/", "--")
    snapshots_dir = os.path.join(cache_dir, folder, "snapshots")
    hashes = os.listdir(snapshots_dir)
    if not hashes:
        print(f"  WARNING: No snapshot found for {model_id}")
        return
    snapshot_path = os.path.join(snapshots_dir, hashes[0])
    files = os.listdir(snapshot_path)
    safetensors = [f for f in files if f.endswith(".safetensors")]
    print(f"  [{model_id}] Verified: {len(safetensors)} safetensors shards, "
          f"{len(files)} total files in snapshot.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["gen", "eval", "both"],
        default="both",
        help="Which model to download (default: both)",
    )
    args = parser.parse_args()

    os.makedirs(HF_CACHE_DIR, exist_ok=True)
    print(f"HuggingFace cache directory: {HF_CACHE_DIR}\n")

    models_to_download = []
    if args.model in ("gen", "both"):
        models_to_download.append((GEN_MODEL_ID, "Generation model (Qwen3.5-9B)"))
    if args.model in ("eval", "both"):
        models_to_download.append((EVAL_MODEL_ID, "Evaluation model (Qwen3.5-4B)"))

    for model_id, desc in models_to_download:
        print(f"=== {desc} ===")
        download_model(model_id, HF_CACHE_DIR)
        verify_model(model_id, HF_CACHE_DIR)
        print()

    print("All models ready. You can now run the experiment pipeline.")


if __name__ == "__main__":
    main()
