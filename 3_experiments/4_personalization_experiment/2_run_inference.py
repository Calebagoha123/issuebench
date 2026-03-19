"""
Step 2: Run Qwen3.5-9B on all prompts and collect responses.

Qwen3.5 notes:
  - "Qwen/Qwen3.5-9B" is already instruction-tuned (no -Instruct suffix).
  - Thinking mode is OFF by default for the small series (0.8B–9B).
    We keep it off — we want fluent essay writing, not chain-of-thought.
  - Recommended sampling params for non-thinking mode: temp=0.7, top_p=0.8, top_k=20.

Outputs:
  results/responses_fast.csv
  results/responses_full.csv

Usage:
  python 2_run_inference.py --mode fast
  python 2_run_inference.py --mode full
"""

import os
import sys
import argparse
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    HF_CACHE_DIR, GEN_MODEL_ID, GEN_GPU,
    GEN_BATCH_SIZE, MAX_NEW_TOKENS, TEMPERATURE, TOP_P, TOP_K,
    RESULTS_DIR,
)


def load_model(model_id: str, cache_dir: str, device: str):
    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model onto {device} (BF16) ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Loaded {n_params:.1f}B parameters.")
    return tokenizer, model


def format_prompts(prompt_texts: list, tokenizer) -> list:
    """
    Wrap each prompt in Qwen3.5's chat template.
    enable_thinking=False keeps responses clean (no <think> blocks).
    """
    formatted = []
    for text in prompt_texts:
        messages = [{"role": "user", "content": text}]
        formatted.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,   # Qwen3.5: disable chain-of-thought
            )
        )
    return formatted


def generate_batch(
    texts: list, tokenizer, model, device: str
) -> list:
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=768,
    ).to(device)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    return [
        tokenizer.decode(ids[input_len:], skip_special_tokens=True).strip()
        for ids in out_ids
    ]


def run_inference(prompts_df: pd.DataFrame, tokenizer, model, device: str, out_path: str):
    formatted = format_prompts(prompts_df["prompt_text"].tolist(), tokenizer)
    results = []

    for i in tqdm(range(0, len(formatted), GEN_BATCH_SIZE), desc="Generating responses"):
        batch_texts = formatted[i : i + GEN_BATCH_SIZE]
        batch_meta  = prompts_df.iloc[i : i + GEN_BATCH_SIZE]

        responses = generate_batch(batch_texts, tokenizer, model, device)

        for (_, meta), response in zip(batch_meta.iterrows(), responses):
            row = meta.to_dict()
            row["response_text"] = response
            results.append(row)

        # Save incrementally every 50 batches
        if (i // GEN_BATCH_SIZE) % 50 == 0 and results:
            pd.DataFrame(results).to_csv(out_path, index=False)

    df_out = pd.DataFrame(results)
    df_out.to_csv(out_path, index=False)
    print(f"\nSaved {len(df_out)} responses → {out_path}")
    return df_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fast", "full"], default="fast")
    args = parser.parse_args()

    prompts_path = os.path.join(RESULTS_DIR, f"prompts_{args.mode}.csv")
    out_path     = os.path.join(RESULTS_DIR, f"responses_{args.mode}.csv")

    if not os.path.exists(prompts_path):
        print(f"ERROR: {prompts_path} not found. Run 1_generate_subset.py first.")
        sys.exit(1)

    prompts_df = pd.read_csv(prompts_path)
    print(f"Loaded {len(prompts_df)} prompts.")

    tokenizer, model = load_model(GEN_MODEL_ID, HF_CACHE_DIR, GEN_GPU)
    run_inference(prompts_df, tokenizer, model, GEN_GPU, out_path)

    # Preview
    df = pd.read_csv(out_path)
    print("\nSample responses:")
    for _, row in df[["cue_id", "topic_polarity", "topic_neutral", "response_text"]].head(5).iterrows():
        preview = row["response_text"][:100].replace("\n", " ")
        print(f"  [{row['cue_id']:20s}][{row['topic_polarity']:7s}] {row['topic_neutral']:25s} → {preview}...")


if __name__ == "__main__":
    main()
