"""
Step 4: Analyse results and produce the dot plot figure.

Dot plot (matching IssueBench paper style):
  - Y-axis: each topic, with three sub-rows (neutral / pro / con framing)
  - X-axis: bias score from -1.0 (100% con) to +1.0 (100% pro)
            bias = (pct_pro - pct_con) = (pct_1 + pct_2) - (pct_4 + pct_5)
  - Each dot = one identity cue condition, colored/shaped per cue

Also produces:
  - Console summary table of stance shifts vs baseline
  - Wilcoxon signed-rank tests (systematic shift per cue across topics)
  - results/analysis_{mode}.csv

Usage:
  python 4_analyse.py --mode fast
  python 4_analyse.py --mode full
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

matplotlib.rcParams["font.family"] = "DejaVu Sans"

sys.path.insert(0, os.path.dirname(__file__))
from config import RESULTS_DIR, IDENTITY_CUES, TOPIC_POLARITIES, CUE_STYLES

FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
VALID_LABELS = {"1", "2", "3", "4", "5", "refusal"}
CUE_ORDER = list(IDENTITY_CUES.keys())
POLARITY_ORDER = ["neutral", "pro", "con"]
POLARITY_LABEL = {"neutral": "neutral", "pro": "pro", "con": "con"}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(mode: str) -> pd.DataFrame:
    path = os.path.join(RESULTS_DIR, f"evaluated_{mode}.csv")
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Run 3_run_stance_eval.py first.")
        sys.exit(1)
    df = pd.read_csv(path)
    df["eval_label"] = df["eval_label"].astype(str)
    df.loc[~df["eval_label"].isin(VALID_LABELS), "eval_label"] = "PARSE_ERROR"
    return df


# ── Aggregation ───────────────────────────────────────────────────────────────

def compute_bias_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (topic_neutral, topic_polarity, cue_id): compute
      bias_score = (pct_1 + pct_2) - (pct_4 + pct_5)  ∈ [-1, +1]
      mean_label = mean of numeric labels (ignoring refusal/errors)
    """
    rows = []
    for (topic, polarity, cue_id), grp in df.groupby(
        ["topic_neutral", "topic_polarity", "cue_id"]
    ):
        n = len(grp)
        vc = grp["eval_label"].value_counts()
        pct = {lbl: vc.get(lbl, 0) / n for lbl in ["1", "2", "3", "4", "5", "refusal"]}
        bias = (pct["1"] + pct["2"]) - (pct["4"] + pct["5"])

        numeric = grp[grp["eval_label"].isin(["1","2","3","4","5"])]["eval_label"].astype(int)
        mean_lbl = numeric.mean() if len(numeric) > 0 else np.nan

        rows.append({
            "topic_neutral":  topic,
            "topic_polarity": polarity,
            "cue_id":         cue_id,
            "n":              n,
            "bias_score":     bias,
            "mean_label":     mean_lbl,
            **{f"pct_{l}": pct[l] for l in ["1","2","3","4","5","refusal"]},
        })
    return pd.DataFrame(rows)


def add_delta_vs_baseline(scores_df: pd.DataFrame) -> pd.DataFrame:
    """Add bias_delta = bias_score - baseline_bias_score for each (topic, polarity)."""
    baseline = scores_df[scores_df["cue_id"] == "baseline"][
        ["topic_neutral", "topic_polarity", "bias_score"]
    ].rename(columns={"bias_score": "baseline_bias"})
    merged = scores_df.merge(baseline, on=["topic_neutral", "topic_polarity"], how="left")
    merged["bias_delta"] = merged["bias_score"] - merged["baseline_bias"]
    return merged


# ── Dot plot ──────────────────────────────────────────────────────────────────

def make_dot_plot(scores_df: pd.DataFrame, out_path: str, mode: str):
    """
    One row per topic (neutral framing only).
    X-axis: bias score -1.0 (100% con) → +1.0 (100% pro).
    Five colored dots per row, one per identity cue.
    Topics sorted by baseline bias score (most pro at top).
    """
    # Use neutral framing only — isolates the cue effect from topic polarity
    neutral = scores_df[scores_df["topic_polarity"] == "neutral"].copy()

    # Sort topics by baseline bias score so the plot has a natural ordering
    baseline_order = (
        neutral[neutral["cue_id"] == "baseline"]
        .set_index("topic_neutral")["bias_score"]
        .sort_values(ascending=True)   # ascending = con topics at bottom, pro at top
    )
    topics = baseline_order.index.tolist()
    n_topics = len(topics)

    # One y-position per topic
    y_pos = {topic: i for i, topic in enumerate(topics)}

    fig_height = max(7, n_topics * 0.55)
    fig, ax = plt.subplots(figsize=(13, fig_height))

    # ── Dotted guide lines ──
    for i in range(n_topics):
        ax.axhline(i, color="#e0e0e0", linewidth=0.7, zorder=0)

    # ── Vertical jitter so overlapping dots don't pile on top of each other ──
    # Spread the 5 cues symmetrically around the row centre
    n_cues = len(CUE_ORDER)
    jitter_offsets = np.linspace(-0.22, 0.22, n_cues)   # vertical spread

    for cue_idx, cue_id in enumerate(CUE_ORDER):
        color, marker, label = CUE_STYLES[cue_id]
        sub = neutral[neutral["cue_id"] == cue_id]

        xs, ys = [], []
        for _, row in sub.iterrows():
            if row["topic_neutral"] in y_pos:
                xs.append(row["bias_score"])
                ys.append(y_pos[row["topic_neutral"]] + jitter_offsets[cue_idx])

        ax.scatter(
            xs, ys,
            color=color,
            marker=marker,
            s=90,
            zorder=3,
            label=label,
            edgecolors="white",
            linewidths=0.6,
            alpha=0.95,
        )

    # ── Y-axis: topic names ──
    ax.set_yticks(range(n_topics))
    ax.set_yticklabels(topics, fontsize=9)
    ax.set_ylim(-0.6, n_topics - 0.4)

    # ── X-axis ──
    ax.set_xlim(-1.18, 1.18)
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.tick_params(axis="x", labelsize=9)
    ax.axvline(0, color="#aaaaaa", linewidth=0.9, linestyle="--", zorder=1)

    # X-axis label with -1/0/+1 semantics inline
    ax.set_xlabel(
        "−1.00 = 100% con                         0.00                         100% pro = +1.00",
        fontsize=9,
    )

    # ── Legend — use actual scatter handles for correct marker shapes ──
    legend_handles = []
    for cue_id in CUE_ORDER:
        color, marker, label = CUE_STYLES[cue_id]
        legend_handles.append(
            plt.Line2D(
                [0], [0],
                marker=marker,
                color="w",
                markerfacecolor=color,
                markeredgecolor="white",
                markersize=9,
                label=label,
            )
        )
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        fontsize=8.5,
        framealpha=0.92,
        edgecolor="#cccccc",
    )

    ax.set_title(
        f"Stance by identity cue — Qwen3.5-9B on IssueBench topics ({mode} run)\n"
        "Neutral framing only · each dot = mean bias score (pct pro − pct con) across templates",
        fontsize=10,
        pad=12,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved dot plot → {out_path}")


# ── Summary statistics ────────────────────────────────────────────────────────

def print_summary(scores_df: pd.DataFrame):
    print("\n" + "=" * 72)
    print("BIAS SCORE BY CUE  (bias = pct_pro − pct_con,  range −1 to +1)")
    print("=" * 72)
    summary = (
        scores_df[scores_df["topic_polarity"] == "neutral"]
        .groupby("cue_id")["bias_score"]
        .agg(["mean", "std", "median"])
        .reindex(CUE_ORDER)
    )
    print(summary.round(3).to_string())

    non_base = scores_df[(scores_df["cue_id"] != "baseline") & (scores_df["topic_polarity"] == "neutral")]
    print("\nBIAS DELTA (vs. baseline, neutral framing):")
    delta = (
        non_base.groupby("cue_id")["bias_delta"]
        .agg(["mean", "std", "count"])
        .reindex([c for c in CUE_ORDER if c != "baseline"])
    )
    print(delta.round(3).to_string())


def wilcoxon_tests(scores_df: pd.DataFrame):
    print("\n" + "-" * 72)
    print("WILCOXON SIGNED-RANK TEST  H₀: cue produces no stance shift")
    print("(neutral framing only, delta = cue bias − baseline bias)")
    neutral = scores_df[scores_df["topic_polarity"] == "neutral"]
    for cue_id in CUE_ORDER:
        if cue_id == "baseline":
            continue
        deltas = neutral[neutral["cue_id"] == cue_id]["bias_delta"].dropna()
        if len(deltas) < 5:
            print(f"  {cue_id:22s}: too few data points ({len(deltas)})")
            continue
        stat, p = stats.wilcoxon(deltas)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(
            f"  {cue_id:22s}: W={stat:.0f}, p={p:.4f} {sig:3s}  "
            f"mean_delta={deltas.mean():+.3f}  "
            f"(n={len(deltas)})"
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fast", "full"], default="fast")
    args = parser.parse_args()

    os.makedirs(FIGURES_DIR, exist_ok=True)

    df = load_data(args.mode)
    n_errors = (df["eval_label"] == "PARSE_ERROR").sum()
    print(f"Loaded {len(df)} rows | parse errors: {n_errors} ({100*n_errors/len(df):.1f}%)")

    scores_df = compute_bias_scores(df)
    scores_df = add_delta_vs_baseline(scores_df)

    out_csv = os.path.join(RESULTS_DIR, f"analysis_{args.mode}.csv")
    scores_df.to_csv(out_csv, index=False)
    print(f"Saved analysis table → {out_csv}")

    print_summary(scores_df)
    wilcoxon_tests(scores_df)

    print("\nGenerating dot plot ...")
    make_dot_plot(
        scores_df,
        out_path=os.path.join(FIGURES_DIR, f"dot_plot_{args.mode}.png"),
        mode=args.mode,
    )
    print("Done.")


if __name__ == "__main__":
    main()
