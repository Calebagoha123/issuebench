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
    Dot plot matching IssueBench paper style:
      - Topics on left y-axis, each with 3 sub-rows (neutral / pro / con)
      - X-axis: bias score -1 (100% con) to +1 (100% pro)
      - Colored + shaped dots per identity cue
    """
    topics = sorted(scores_df["topic_neutral"].unique())
    n_topics = len(topics)

    # Build y-axis positions:
    # Each topic gets 3 sub-rows + a gap row between topics
    sub_gap = 0.28    # vertical space between framings within a topic
    topic_gap = 0.55  # vertical space between topic groups

    y_positions = {}   # (topic, polarity) → y
    y_topic_center = {}  # topic → y center for the topic label
    current_y = 0.0

    for topic in reversed(topics):   # top-to-bottom = reversed list
        ys = []
        for p_idx, polarity in enumerate(POLARITY_ORDER):
            y = current_y + p_idx * sub_gap
            y_positions[(topic, polarity)] = y
            ys.append(y)
        y_topic_center[topic] = np.mean(ys)
        current_y += len(POLARITY_ORDER) * sub_gap + topic_gap

    total_height = current_y
    fig_height = max(8, total_height * 0.55)
    fig, ax = plt.subplots(figsize=(13, fig_height))

    # ── Draw dotted guide lines ──
    for (topic, polarity), y in y_positions.items():
        ax.axhline(y, color="#dddddd", linewidth=0.6, zorder=0)

    # ── Draw dots ──
    n_cues = len(CUE_ORDER)
    jitter_step = 0.055   # horizontal jitter so overlapping dots spread apart

    for cue_idx, cue_id in enumerate(CUE_ORDER):
        color, marker, label = CUE_STYLES[cue_id]
        sub = scores_df[scores_df["cue_id"] == cue_id]

        xs, ys, topics_seen = [], [], []
        for _, row in sub.iterrows():
            key = (row["topic_neutral"], row["topic_polarity"])
            if key in y_positions:
                xs.append(row["bias_score"])
                ys.append(y_positions[key])

        ax.scatter(
            xs, ys,
            color=color,
            marker=marker,
            s=60,
            zorder=3,
            label=label,
            edgecolors="white",
            linewidths=0.4,
            alpha=0.9,
        )

    # ── Y-axis labels ──
    # Topic name on the left (centered on the 3 sub-rows)
    # Polarity sub-labels (neutral/pro/con) as small secondary labels
    ax.set_yticks([y for y in y_positions.values()])
    ax.set_yticklabels(
        [POLARITY_LABEL[pol] for (_, pol) in y_positions.keys()],
        fontsize=7,
        color="#666666",
    )

    # Topic name annotations to the left of the polarity labels
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(list(y_topic_center.values()))
    ax2.set_yticklabels(list(y_topic_center.keys()), fontsize=8.5)
    ax2.yaxis.set_ticks_position("left")
    ax2.yaxis.set_label_position("left")
    ax2.tick_params(left=False, right=False)
    # Shift topic labels further left of polarity labels
    ax2.yaxis.set_tick_params(pad=120)

    # ── X-axis ──
    ax.set_xlim(-1.15, 1.15)
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_xlabel(
        "−1.00 = 100% con                                    "
        "0.00                                    "
        "100% pro = +1.00",
        fontsize=9,
    )
    ax.axvline(0, color="#aaaaaa", linewidth=0.8, linestyle="--", zorder=1)

    # ── Legend ──
    legend_handles = [
        mpatches.Patch(color=CUE_STYLES[c][0], label=CUE_STYLES[c][2])
        for c in CUE_ORDER
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        fontsize=8,
        framealpha=0.9,
        ncol=1,
    )

    ax.set_title(
        f"Stance by identity cue — Qwen3.5-9B on IssueBench topics ({mode} run)\n"
        "Dot = mean bias score (pro−con proportion) per topic × framing × cue",
        fontsize=10,
        pad=12,
    )
    ax.set_frame_on(False)
    ax2.set_frame_on(False)
    ax.tick_params(left=True, right=False, bottom=True)

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
