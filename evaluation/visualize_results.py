"""Create key plots for comparing Catan agents.

Example:
    python visualize_results.py --input-dir results/eval_20260306_120000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    '''Parse command-line arguments for plotting Catan evaluation results.'''
    parser = argparse.ArgumentParser(description="Plot Catan evaluation outputs")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory with player_metrics.csv")
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=None,
        help="Directory with analysis CSVs (default: <input-dir>/analysis)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output plot directory (default: <input-dir>/plots)",
    )
    return parser.parse_args()


def main() -> None:
    '''Main function to visualize Catan evaluation results, creating plots for win rates, VP distributions, and more.'''
    args = parse_args()
    input_dir = args.input_dir
    analysis_dir = args.analysis_dir or (input_dir / "analysis")
    outdir = args.outdir or (input_dir / "plots")
    outdir.mkdir(parents=True, exist_ok=True)

    players = pd.read_csv(input_dir / "player_metrics.csv")
    agent_summary = pd.read_csv(analysis_dir / "agent_summary.csv")
    seat_summary = pd.read_csv(analysis_dir / "seat_summary.csv")
    pairwise = pd.read_csv(analysis_dir / "pairwise_vp_diff.csv", index_col=0)

    sns.set_theme(style="whitegrid", context="talk")

    # 1) Win-rate bars + 95% bootstrap CI
    fig, ax = plt.subplots(figsize=(9, 6))
    df = agent_summary.sort_values("win_rate", ascending=False)
    x = np.arange(len(df))
    y = df["win_rate"].to_numpy()
    yerr_low = y - df["win_rate_ci_low"].to_numpy()
    yerr_high = df["win_rate_ci_high"].to_numpy() - y

    ax.bar(x, y, color="#457B9D", alpha=0.9)
    ax.errorbar(x, y, yerr=[yerr_low, yerr_high], fmt="none", ecolor="black", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(df["agent"])
    ax.set_ylabel("Win Rate")
    ax.set_ylim(0, max(0.35, float(y.max()) * 1.2))
    ax.set_title("Win Rate by Agent (95% Bootstrap CI)")
    fig.tight_layout()
    fig.savefig(outdir / "win_rate_ci.png", dpi=200)
    plt.close(fig)

    # 2) Final VP distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=players, x="agent", y="final_vp", inner="quartile", cut=0, ax=ax, color="#A8DADC")
    ax.set_title("Final VP Distribution by Agent")
    ax.set_xlabel("Agent")
    ax.set_ylabel("Final VP")
    fig.tight_layout()
    fig.savefig(outdir / "final_vp_violin.png", dpi=200)
    plt.close(fig)

    # 3) Turns-to-end distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=players, x="agent", y="num_turns", ax=ax, color="#F4A261")
    ax.set_title("Game Length Distribution (Turns)")
    ax.set_xlabel("Agent")
    ax.set_ylabel("Turns")
    fig.tight_layout()
    fig.savefig(outdir / "turns_box.png", dpi=200)
    plt.close(fig)

    # 3b) Final resources held distribution
    if "total_resources_in_hand" in players.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=players, x="agent", y="total_resources_in_hand", ax=ax, color="#90BE6D")
        ax.set_title("Final Resources Held by Agent")
        ax.set_xlabel("Agent")
        ax.set_ylabel("Total Resources In Hand")
        fig.tight_layout()
        fig.savefig(outdir / "resources_held_box.png", dpi=200)
        plt.close(fig)
    else:
        print("Skipping resources-held plot: total_resources_in_hand not found in player_metrics.csv")

    # 3c) Standalone fairness distribution by agent
    # NOTE: Commented out - misleading because vp_gini_game is a game-level metric,
    # so all agents in the same game share the same value. Use lineup plot instead.
    # if "vp_gini_game" in players.columns:
    #     fig, ax = plt.subplots(figsize=(10, 6))
    #     sns.boxplot(data=players, x="agent", y="vp_gini_game", ax=ax, color="#CDB4DB")
    #     ax.set_title("Fairness Distribution by Agent")
    #     ax.set_xlabel("Agent")
    #     ax.set_ylabel("Game VP Gini (lower = fairer)")
    #     fig.tight_layout()
    #     fig.savefig(outdir / "fairness_gini_by_agent.png", dpi=200)
    #     plt.close(fig)
    # else:
    #     print("Skipping fairness plot: vp_gini_game not found in player_metrics.csv")

    # 3d) Fairness distribution by lineup
    if "vp_gini_game" in players.columns and "lineup" in players.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        # Sort lineups by median Gini for better visualization
        lineup_order = players.groupby("lineup")["vp_gini_game"].median().sort_values().index
        sns.boxplot(data=players, x="lineup", y="vp_gini_game", order=lineup_order, ax=ax, color="#B5838D")
        ax.set_title("Fairness Distribution by Lineup")
        ax.set_xlabel("Lineup")
        ax.set_ylabel("Game VP Gini (lower = fairer)")
        plt.xticks(rotation=45, ha='right')
        fig.tight_layout()
        fig.savefig(outdir / "fairness_gini_by_lineup.png", dpi=200)
        plt.close(fig)
    else:
        print("Skipping lineup fairness plot: vp_gini_game or lineup not found in player_metrics.csv")

    # 4) Pairwise VP differential heatmap
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(pairwise, annot=True, fmt=".2f", cmap="RdBu_r", center=0.0, linewidths=0.5, ax=ax)
    ax.set_title("Pairwise VP Differential: E[VP(row) - VP(col)]")
    ax.set_xlabel("Opponent Agent")
    ax.set_ylabel("Row Agent")
    fig.tight_layout()
    fig.savefig(outdir / "pairwise_vp_diff_heatmap.png", dpi=200)
    plt.close(fig)

    # 5) Pareto: win rate vs fairness + compute cost
    fig, ax = plt.subplots(figsize=(9, 6))
    s = 200 + 20 * agent_summary["avg_decide_ms"].to_numpy()
    sc = ax.scatter(
        agent_summary["avg_game_vp_gini"],
        agent_summary["win_rate"],
        s=s,
        c=agent_summary["avg_decide_ms"],
        cmap="viridis",
        alpha=0.9,
        edgecolor="black",
        linewidth=0.6,
    )
    for _, r in agent_summary.iterrows():
        ax.annotate(r["agent"], (r["avg_game_vp_gini"], r["win_rate"]), xytext=(6, 4), textcoords="offset points")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Avg Decide Time (ms)")
    ax.set_xlabel("Avg Game VP Gini (lower = fairer)")
    ax.set_ylabel("Win Rate")
    ax.set_title("Pareto View: Performance vs Fairness vs Compute")
    fig.tight_layout()
    fig.savefig(outdir / "pareto_fairness_winrate_compute.png", dpi=200)
    plt.close(fig)

    # 6) Turn-order sensitivity
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=seat_summary,
        x="turn_order",
        y="win_rate",
        hue="agent",
        marker="o",
        linewidth=2,
        ax=ax,
    )
    turn_orders = sorted(seat_summary["turn_order"].dropna().unique().astype(int).tolist())
    ax.set_xticks(turn_orders)
    ax.set_xlabel("Turn Order")
    ax.set_ylabel("Win Rate")
    ax.set_title("Turn-Order Effect on Win Rate")
    fig.tight_layout()
    fig.savefig(outdir / "turn_order_effect_win_rate.png", dpi=200)
    plt.close(fig)

    print(f"Saved plots to: {outdir}")


if __name__ == "__main__":
    main()
