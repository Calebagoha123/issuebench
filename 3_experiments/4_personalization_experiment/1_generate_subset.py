"""
Step 1: Generate the subset of prompts for the personalization experiment.

For each (issue, framing, template, cue) combination, constructs a prompt:
  [cue_text + "\n\n"] + template_with_topic_phrase

Three framings per topic:
  neutral → "Write an essay about capitalism."
  pro     → "Write an essay about capitalism being good."
  con     → "Write an essay about capitalism being bad."

Outputs:
  results/prompts_fast.csv  — 15 issues × 3 templates × 3 framings × 5 cues = 675 prompts
  results/prompts_full.csv  — full-scale version (run with --mode full)

Usage:
  python 1_generate_subset.py
  python 1_generate_subset.py --mode full
"""

import os
import sys
import random
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    ISSUES_CSV, RESULTS_DIR,
    FAST_N_ISSUES, FAST_ISSUE_IDS, FAST_TEMPLATES,
    FULL_N_ISSUES, FULL_N_TEMPLATES, TEMPLATES_CSV,
    TOPIC_POLARITIES, IDENTITY_CUES,
)

RANDOM_SEED = 42


def load_issues(n_issues: int, fixed_ids: list) -> pd.DataFrame:
    issues = pd.read_csv(ISSUES_CSV)
    issues = issues[
        (issues["tag_exclude"] != 1) &
        (~issues["topic_neutral"].isin(["UNASSIGNED CLUSTER", "SPAM"]))
    ].copy()

    selected_ids = [i for i in fixed_ids if i in issues["topic_id"].values]

    if len(selected_ids) < n_issues:
        remaining = issues[~issues["topic_id"].isin(selected_ids)]
        extra = remaining.sample(
            n=min(n_issues - len(selected_ids), len(remaining)),
            random_state=RANDOM_SEED,
        )
        selected_ids += extra["topic_id"].tolist()

    return issues[issues["topic_id"].isin(selected_ids[:n_issues])].reset_index(drop=True)


def load_templates(n_templates: int) -> list:
    df = pd.read_csv(TEMPLATES_CSV)
    # Use the annotator-cleaned template column (annot1_template)
    templates = (
        df["annot1_template"]
        .dropna()
        .str.strip()
        .tolist()
    )
    random.seed(RANDOM_SEED)
    return random.sample(templates, min(n_templates, len(templates)))


def build_prompt(cue_text: str | None, template: str, topic_phrase: str) -> str:
    filled = template.replace("X", topic_phrase)
    if cue_text:
        return f"{cue_text}\n\n{filled}"
    return filled


def generate_prompts(issues: pd.DataFrame, templates: list) -> pd.DataFrame:
    rows = []
    for _, issue in issues.iterrows():
        topic_phrases = {
            "neutral": issue["topic_neutral"],
            "pro":     issue["topic_pro"],
            "con":     issue["topic_con"],
        }
        for polarity in TOPIC_POLARITIES:
            topic_phrase = topic_phrases[polarity]
            for t_idx, template in enumerate(templates):
                for cue_id, cue_text in IDENTITY_CUES.items():
                    rows.append({
                        "prompt_id": f"{issue['topic_id']}.{polarity}.t{t_idx}.{cue_id}",
                        "topic_id":       issue["topic_id"],
                        "topic_neutral":  issue["topic_neutral"],
                        "topic_pro":      issue["topic_pro"],
                        "topic_con":      issue["topic_con"],
                        "topic_polarity": polarity,
                        "topic_phrase":   topic_phrase,
                        "template_idx":   t_idx,
                        "template":       template,
                        "cue_id":         cue_id,
                        "cue_text":       cue_text or "",
                        "prompt_text":    build_prompt(cue_text, template, topic_phrase),
                    })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fast", "full"], default="fast")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.mode == "fast":
        issues = load_issues(FAST_N_ISSUES, FAST_ISSUE_IDS)
        templates = FAST_TEMPLATES
    else:
        issues = load_issues(FULL_N_ISSUES, FAST_ISSUE_IDS)
        templates = load_templates(FULL_N_TEMPLATES)

    print(f"Issues ({len(issues)}): {issues['topic_neutral'].tolist()}")
    print(f"Templates ({len(templates)}): {templates[:3]} ...")

    prompts = generate_prompts(issues, templates)
    out = os.path.join(RESULTS_DIR, f"prompts_{args.mode}.csv")
    prompts.to_csv(out, index=False)

    n_issues = len(issues)
    n_t = len(templates)
    n_pol = len(TOPIC_POLARITIES)
    n_cues = len(IDENTITY_CUES)
    print(f"\nSaved {len(prompts)} prompts → {out}")
    print(f"  {n_issues} issues × {n_t} templates × {n_pol} framings × {n_cues} cues = {n_issues*n_t*n_pol*n_cues}")

    print("\nSample (first 8 rows):")
    cols = ["cue_id", "topic_polarity", "topic_neutral", "prompt_text"]
    for _, row in prompts[cols].head(8).iterrows():
        preview = row["prompt_text"][:90].replace("\n", " ↵ ")
        print(f"  [{row['cue_id']:20s}] [{row['topic_polarity']:7s}] {row['topic_neutral']:25s} | {preview}")


if __name__ == "__main__":
    main()
