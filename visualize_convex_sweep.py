"""Plot ConvexAgent sweep: expected production Gini vs lambda, overlaid by selfish opponent.

Uses Gini of expected production after initial placement (what the LP optimizes).
Example:
    python visualize_convex_sweep.py --input-dir results/convex_sweep
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ConvexAgent sweep results")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=None)
    return parser.parse_args()


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (np.nan, np.nan)
    if vals.size == 1:
        return (float(vals[0]), float(vals[0]))
    idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
    samples = vals[idx].mean(axis=1)
    return float(np.quantile(samples, alpha / 2)), float(np.quantile(samples, 1 - alpha / 2))


def main() -> None:
    args = parse_args()
    outdir = args.outdir or (args.input_dir / "plots")
    outdir.mkdir(parents=True, exist_ok=True)

    game_df = pd.read_csv(args.input_dir / "game_metrics.csv")
    player_df = pd.read_csv(args.input_dir / "player_metrics.csv")

    sns.set_theme(style="whitegrid", context="talk")

    # Use prod_gini (expected production after initial placement) - fallback to vp_gini for old data
    gini_col = "prod_gini" if "prod_gini" in game_df.columns else "vp_gini"
    range_col = "prod_range" if "prod_range" in game_df.columns else "vp_range"

    # Aggregate by (lambda, selfish_agent):
    agg = game_df.groupby(["lambda", "selfish_agent"], as_index=False).agg(
        gini_mean=(gini_col, "mean"),
        gini_std=(gini_col, "std"),
        range_mean=(range_col, "mean"),
        range_std=(range_col, "std"),
        n=(gini_col, "count"),
    )

    # Bootstrap CI for gini
    ci_rows = []
    for (lam, selfish), g in game_df.groupby(["lambda", "selfish_agent"]):
        lo, hi = bootstrap_ci(g[gini_col].to_numpy())
        ci_rows.append({"lambda": lam, "selfish_agent": selfish, "gini_ci_low": lo, "gini_ci_high": hi})
    ci_df = pd.DataFrame(ci_rows)
    agg = agg.merge(ci_df, on=["lambda", "selfish_agent"], how="left")

    selfish_agents = sorted(game_df["selfish_agent"].unique().tolist())
    palette = sns.color_palette("husl", n_colors=len(selfish_agents))
    color_map = {a: palette[i] for i, a in enumerate(selfish_agents)}

    gini_label = "Expected Production Gini" if gini_col == "prod_gini" else "VP Gini"

    # 1) Expected production Gini vs lambda, one line per selfish agent (overlaid)
    fig, ax = plt.subplots(figsize=(10, 6))
    for selfish in selfish_agents:
        sub = agg[agg["selfish_agent"] == selfish].sort_values("lambda")
        if sub.empty:
            continue
        x = sub["lambda"].to_numpy()
        y = sub["gini_mean"].to_numpy()
        lo = sub["gini_ci_low"].to_numpy()
        hi = sub["gini_ci_high"].to_numpy()
        ax.plot(x, y, marker="o", linewidth=2, label=selfish, color=color_map.get(selfish, "gray"))
        ax.fill_between(x, lo, hi, alpha=0.2, color=color_map.get(selfish, "gray"))
    ax.set_xlabel("Lambda")
    ax.set_ylabel(f"{gini_label} (lower = fairer)")
    ax.set_title("ConvexAgent Fairness vs Lambda: Expected Production After Initial Placement")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(outdir / "prod_gini_vs_lambda_by_opponent.png", dpi=200)
    plt.close(fig)

    # 2) Production range vs lambda, one line per selfish agent (overlaid)
    fig, ax = plt.subplots(figsize=(10, 6))
    for selfish in selfish_agents:
        sub = agg[agg["selfish_agent"] == selfish].sort_values("lambda")
        if sub.empty:
            continue
        x = sub["lambda"].to_numpy()
        y = sub["range_mean"].to_numpy()
        ax.plot(x, y, marker="o", linewidth=2, label=selfish, color=color_map.get(selfish, "gray"))
    ax.set_xlabel("Lambda")
    ax.set_ylabel("Expected Production Range (max - min)")
    ax.set_title("Production Range vs Lambda by Selfish Opponent")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(outdir / "prod_range_vs_lambda_by_opponent.png", dpi=200)
    plt.close(fig)

    # 3) Bar chart: Expected production Gini by selfish agent at lambda=0.5
    target_lam = 0.5
    lam_vals = game_df["lambda"].dropna().unique()
    nearest = float(lam_vals[np.argmin(np.abs(lam_vals - target_lam))])
    slice_df = game_df[np.isclose(game_df["lambda"], nearest)]
    bar_rows = []
    for selfish in slice_df["selfish_agent"].unique():
        vals = slice_df[slice_df["selfish_agent"] == selfish][gini_col].to_numpy()
        lo, hi = bootstrap_ci(vals)
        bar_rows.append({
            "selfish_agent": selfish,
            "gini_mean": float(np.mean(vals)),
            "gini_ci_low": lo,
            "gini_ci_high": hi,
        })
    bar_means = pd.DataFrame(bar_rows).sort_values("gini_mean")

    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(bar_means))
    y = bar_means["gini_mean"].to_numpy()
    yerr_lo = y - bar_means["gini_ci_low"].to_numpy()
    yerr_hi = bar_means["gini_ci_high"].to_numpy() - y
    colors = [color_map.get(a, "gray") for a in bar_means["selfish_agent"]]
    ax.bar(x, y, color=colors, alpha=0.9)
    ax.errorbar(x, y, yerr=[yerr_lo, yerr_hi], fmt="none", ecolor="black", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(bar_means["selfish_agent"])
    ax.set_ylabel(f"{gini_label} (lower = fairer)")
    ax.set_title(f"Fairness by Selfish Opponent (lambda={nearest:.2f})")
    fig.tight_layout()
    fig.savefig(outdir / "prod_gini_by_opponent_at_lambda05.png", dpi=200)
    plt.close(fig)

    # 4) Resources held: Convex vs Random vs Selfish, by opponent
    fig, ax = plt.subplots(figsize=(12, 6))
    slice_players = player_df[np.isclose(player_df["lambda"], nearest)]
    agent_palette = {"CONVEX": "#2A9D8F", "R": "#457B9D"}
    for a in selfish_agents:
        agent_palette[a] = color_map.get(a, "gray")
    sns.boxplot(
        data=slice_players,
        x="selfish_agent",
        y="total_resources_in_hand",
        hue="agent",
        ax=ax,
        palette=agent_palette,
    )
    ax.set_xlabel("Selfish Opponent")
    ax.set_ylabel("Final Resources Held")
    ax.set_title(f"Resource Equalization by Opponent (lambda={nearest:.2f})")
    ax.legend(title="Agent", loc="best")
    plt.xticks(rotation=15)
    fig.tight_layout()
    fig.savefig(outdir / "resources_by_opponent.png", dpi=200)
    plt.close(fig)

    print(f"Saved plots to {outdir}")


if __name__ == "__main__":
    main()
