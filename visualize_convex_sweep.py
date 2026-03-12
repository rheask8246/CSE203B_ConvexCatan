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

    # Backward compat: if no lineup_type, treat all as CONVEX
    if "lineup_type" not in game_df.columns:
        game_df["lineup_type"] = "CONVEX"
    if "lineup_type" not in player_df.columns:
        player_df["lineup_type"] = "CONVEX"

    convex_df = game_df[game_df["lineup_type"] == "CONVEX"]
    baseline_df = game_df[game_df["lineup_type"] == "BASELINE"] if "BASELINE" in game_df["lineup_type"].values else pd.DataFrame()

    sns.set_theme(style="whitegrid", context="talk")

    # Use prod_gini (expected production after initial placement) - fallback to vp_gini for old data
    gini_col = "prod_gini" if "prod_gini" in game_df.columns else "vp_gini"
    range_col = "prod_range" if "prod_range" in game_df.columns else "vp_range"

    # Aggregate CONVEX by (lambda, selfish_agent):
    agg = convex_df.groupby(["lambda", "selfish_agent"], as_index=False).agg(
        gini_mean=(gini_col, "mean"),
        gini_std=(gini_col, "std"),
        range_mean=(range_col, "mean"),
        range_std=(range_col, "std"),
        n=(gini_col, "count"),
    )

    # Bootstrap CI for gini (CONVEX only)
    ci_rows = []
    for (lam, selfish), g in convex_df.groupby(["lambda", "selfish_agent"]):
        lo, hi = bootstrap_ci(g[gini_col].to_numpy())
        ci_rows.append({"lambda": lam, "selfish_agent": selfish, "gini_ci_low": lo, "gini_ci_high": hi})
    ci_df = pd.DataFrame(ci_rows)
    agg = agg.merge(ci_df, on=["lambda", "selfish_agent"], how="left")

    selfish_agents = sorted(convex_df["selfish_agent"].unique().tolist())

    # Baseline aggregates (mean gini/range per selfish agent)
    baseline_agg = None
    if not baseline_df.empty:
        baseline_agg = baseline_df.groupby("selfish_agent", as_index=False).agg(
            gini_mean=(gini_col, "mean"),
            range_mean=(range_col, "mean"),
        )
    palette = sns.color_palette("husl", n_colors=len(selfish_agents))
    color_map = {a: palette[i] for i, a in enumerate(selfish_agents)}

    gini_label = "Expected Production Gini" if gini_col == "prod_gini" else "VP Gini"

    # 1) Expected production Gini vs lambda, one line per selfish agent (overlaid) + baseline
    fig, ax = plt.subplots(figsize=(10, 6))
    lam_min, lam_max = agg["lambda"].min(), agg["lambda"].max()
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
        # Baseline horizontal line (R,R,SELFISH)
        if baseline_agg is not None and not baseline_agg[baseline_agg["selfish_agent"] == selfish].empty:
            bl_mean = baseline_agg[baseline_agg["selfish_agent"] == selfish]["gini_mean"].iloc[0]
            ax.hlines(bl_mean, lam_min, lam_max, colors=color_map.get(selfish, "gray"), linestyles="--", linewidth=1.5, alpha=0.8)
    if baseline_agg is not None:
        ax.plot([], [], "k--", linewidth=1.5, label="Baseline (R,R,SELFISH)")
    ax.set_xlabel("Lambda")
    ax.set_ylabel(f"{gini_label} (lower = fairer)")
    ax.set_title("ConvexAgent Fairness vs Lambda (dashed = baseline R,R,SELFISH)")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(outdir / "prod_gini_vs_lambda_by_opponent.png", dpi=200)
    plt.close(fig)

    # 2) Production range vs lambda, one line per selfish agent (overlaid) + baseline
    fig, ax = plt.subplots(figsize=(10, 6))
    for selfish in selfish_agents:
        sub = agg[agg["selfish_agent"] == selfish].sort_values("lambda")
        if sub.empty:
            continue
        x = sub["lambda"].to_numpy()
        y = sub["range_mean"].to_numpy()
        ax.plot(x, y, marker="o", linewidth=2, label=selfish, color=color_map.get(selfish, "gray"))
        if baseline_agg is not None and not baseline_agg[baseline_agg["selfish_agent"] == selfish].empty:
            bl_mean = baseline_agg[baseline_agg["selfish_agent"] == selfish]["range_mean"].iloc[0]
            ax.hlines(bl_mean, lam_min, lam_max, colors=color_map.get(selfish, "gray"), linestyles="--", linewidth=1.5, alpha=0.8)
    if baseline_agg is not None:
        ax.plot([], [], "k--", linewidth=1.5, label="Baseline (R,R,SELFISH)")
    ax.set_xlabel("Lambda")
    ax.set_ylabel("Expected Production Range (max - min)")
    ax.set_title("Production Range vs Lambda (dashed = baseline R,R,SELFISH)")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(outdir / "prod_range_vs_lambda_by_opponent.png", dpi=200)
    plt.close(fig)

    # 3) Bar chart: Expected production Gini by selfish agent at lambda=0.5 (Convex vs Baseline)
    target_lam = 0.5
    lam_vals = convex_df["lambda"].dropna().unique()
    nearest = float(lam_vals[np.argmin(np.abs(lam_vals - target_lam))])
    slice_df = convex_df[np.isclose(convex_df["lambda"], nearest)]
    bar_rows = []
    for selfish in slice_df["selfish_agent"].unique():
        vals = slice_df[slice_df["selfish_agent"] == selfish][gini_col].to_numpy()
        lo, hi = bootstrap_ci(vals)
        bar_rows.append({
            "selfish_agent": selfish,
            "lineup": "CONVEX",
            "gini_mean": float(np.mean(vals)),
            "gini_ci_low": lo,
            "gini_ci_high": hi,
        })
    if not baseline_df.empty:
        for selfish in baseline_df["selfish_agent"].unique():
            vals = baseline_df[baseline_df["selfish_agent"] == selfish][gini_col].to_numpy()
            lo, hi = bootstrap_ci(vals)
            bar_rows.append({
                "selfish_agent": selfish,
                "lineup": "Baseline",
                "gini_mean": float(np.mean(vals)),
                "gini_ci_low": lo,
                "gini_ci_high": hi,
            })
    bar_means = pd.DataFrame(bar_rows)

    fig, ax = plt.subplots(figsize=(10, 6))
    selfish_order = sorted(bar_means["selfish_agent"].unique())
    x = np.arange(len(selfish_order))
    width = 0.35
    for i, selfish in enumerate(selfish_order):
        cv = bar_means[(bar_means["selfish_agent"] == selfish) & (bar_means["lineup"] == "CONVEX")]
        bl = bar_means[(bar_means["selfish_agent"] == selfish) & (bar_means["lineup"] == "Baseline")]
        if not cv.empty:
            y_cv = cv["gini_mean"].iloc[0]
            err_cv = [[y_cv - cv["gini_ci_low"].iloc[0]], [cv["gini_ci_high"].iloc[0] - y_cv]]
            ax.bar(x[i] - width / 2, y_cv, width, label="CONVEX" if i == 0 else "", color=color_map.get(selfish, "gray"), alpha=0.9)
            ax.errorbar(x[i] - width / 2, y_cv, yerr=err_cv, fmt="none", ecolor="black", capsize=3)
        if not bl.empty:
            y_bl = bl["gini_mean"].iloc[0]
            err_bl = [[y_bl - bl["gini_ci_low"].iloc[0]], [bl["gini_ci_high"].iloc[0] - y_bl]]
            ax.bar(x[i] + width / 2, y_bl, width, label="Baseline (R,R,SELFISH)" if i == 0 else "", color=color_map.get(selfish, "gray"), alpha=0.5, hatch="//")
            ax.errorbar(x[i] + width / 2, y_bl, yerr=err_bl, fmt="none", ecolor="black", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(selfish_order)
    ax.set_ylabel(f"{gini_label} (lower = fairer)")
    ax.set_title(f"Fairness by Opponent: Convex vs Baseline (lambda={nearest:.2f})")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(outdir / "prod_gini_by_opponent_at_lambda05.png", dpi=200)
    plt.close(fig)

    # 4) Resources held: Convex vs Random vs Selfish, by opponent (Convex lineup only)
    convex_players = player_df[(player_df["lineup_type"] == "CONVEX") & np.isclose(player_df["lambda"], nearest)]
    fig, ax = plt.subplots(figsize=(12, 6))
    agent_palette = {"CONVEX": "#2A9D8F", "R": "#457B9D"}
    for a in selfish_agents:
        agent_palette[a] = color_map.get(a, "gray")
    sns.boxplot(
        data=convex_players,
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
