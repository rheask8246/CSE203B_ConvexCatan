"""Create plots for lambda-sweep production fairness experiments.

Example:
    python visualize_lambda_sweep.py --input-dir results/lambda_sweep_20260311_120000
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from catanatron.models.board import get_edges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot lambda-sweep production fairness outputs")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory with lambda sweep CSV outputs")
    parser.add_argument("--target-lambda", type=float, default=0.5, help="Lambda used for baseline comparison plots")
    parser.add_argument("--outdir", type=Path, default=None, help="Output plot directory (default: <input-dir>/plots)")
    return parser.parse_args()


def _nearest_lambda(values: np.ndarray, target: float) -> float:
    values = np.asarray(values, dtype=float)
    idx = int(np.argmin(np.abs(values - target)))
    return float(values[idx])


def _board_layout_positions(n_nodes: int, edges: list[tuple[int, int]]) -> np.ndarray:
    """Build deterministic 2D node positions from graph distance rings."""
    if n_nodes <= 0:
        return np.zeros((0, 2), dtype=float)

    neighbors = [set() for _ in range(n_nodes)]
    for a, b in edges:
        if 0 <= a < n_nodes and 0 <= b < n_nodes:
            neighbors[a].add(b)
            neighbors[b].add(a)

    center = int(np.argmax([len(neigh) for neigh in neighbors]))

    dist = np.full(n_nodes, -1, dtype=int)
    q: deque[int] = deque([center])
    dist[center] = 0
    while q:
        u = q.popleft()
        for v in neighbors[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)

    # Handle disconnected leftovers robustly (should be rare).
    if np.any(dist < 0):
        max_d = int(np.max(dist)) if np.any(dist >= 0) else 0
        dist[dist < 0] = max_d + 1

    pos = np.zeros((n_nodes, 2), dtype=float)
    max_ring = int(np.max(dist))
    ring_scale = 1.0 / max(1, max_ring)

    for ring in range(max_ring + 1):
        nodes = np.where(dist == ring)[0]
        if nodes.size == 0:
            continue

        if ring == 0:
            pos[nodes[0]] = (0.0, 0.0)
            continue

        ordered = np.array(sorted(nodes.tolist()), dtype=int)
        angles = np.linspace(0.0, 2.0 * np.pi, num=ordered.size, endpoint=False)
        radius = ring * ring_scale
        pos[ordered, 0] = radius * np.cos(angles)
        pos[ordered, 1] = radius * np.sin(angles)

    return pos


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    outdir = args.outdir or (input_dir / "plots")
    outdir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(input_dir / "lambda_sweep_summary.csv")
    game_df = pd.read_csv(input_dir / "lambda_sweep_game_metrics.csv")
    player_df = pd.read_csv(input_dir / "lambda_sweep_player_metrics.csv")
    dual_df = pd.read_csv(input_dir / "lambda_sweep_duals.csv")

    sns.set_theme(style="whitegrid", context="talk")

    lp = summary[summary["method"] == "LP"].sort_values("lambda")
    baselines = summary[summary["method"] != "LP"]

    # 1) Gini vs lambda (with CI)
    fig, ax = plt.subplots(figsize=(9, 6))
    x = lp["lambda"].to_numpy(dtype=float)
    y = lp["gini_mean"].to_numpy(dtype=float)
    lo = lp["gini_ci_low"].to_numpy(dtype=float)
    hi = lp["gini_ci_high"].to_numpy(dtype=float)

    ax.plot(x, y, marker="o", linewidth=2, color="#264653", label="LP")
    ax.fill_between(x, lo, hi, color="#2A9D8F", alpha=0.2, label="95% CI")

    for _, row in baselines.groupby("method", as_index=False).mean(numeric_only=True).iterrows():
        ax.axhline(row["gini_mean"], linestyle="--", linewidth=1.5, label=f"{row['method']} mean")

    ax.set_xlabel("Lambda")
    ax.set_ylabel("Production Gini (lower = fairer)")
    ax.set_title("Production Gini vs Lambda")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(outdir / "gini_vs_lambda.png", dpi=220)
    plt.close(fig)

    # 2) Lambda vs average resource entropy
    fig, ax = plt.subplots(figsize=(9, 6))
    y = lp["mean_entropy_mean"].to_numpy(dtype=float)
    lo = lp["mean_entropy_ci_low"].to_numpy(dtype=float)
    hi = lp["mean_entropy_ci_high"].to_numpy(dtype=float)

    ax.plot(x, y, marker="o", linewidth=2, color="#1D3557", label="LP")
    ax.fill_between(x, lo, hi, color="#457B9D", alpha=0.2, label="95% CI")
    ax.set_xlabel("Lambda")
    ax.set_ylabel("Mean Shannon Entropy")
    ax.set_title("Resource Diversity (Entropy) vs Lambda")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(outdir / "entropy_vs_lambda.png", dpi=220)
    plt.close(fig)

    # 3) Lambda vs total expected production
    fig, ax = plt.subplots(figsize=(9, 6))
    y = lp["total_expected_production_mean"].to_numpy(dtype=float)
    lo = lp["total_expected_production_ci_low"].to_numpy(dtype=float)
    hi = lp["total_expected_production_ci_high"].to_numpy(dtype=float)

    ax.plot(x, y, marker="o", linewidth=2, color="#8D5524", label="LP")
    ax.fill_between(x, lo, hi, color="#D9A066", alpha=0.2, label="95% CI")
    ax.set_xlabel("Lambda")
    ax.set_ylabel("Total Expected Production")
    ax.set_title("Efficiency vs Lambda")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(outdir / "total_expected_production_vs_lambda.png", dpi=220)
    plt.close(fig)

    # 4) Pareto: total production vs gini (one point per lambda)
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(
        lp["gini_mean"],
        lp["total_expected_production_mean"],
        c=lp["lambda"],
        cmap="viridis",
        s=120,
        edgecolor="black",
        linewidth=0.6,
    )
    for _, row in lp.iterrows():
        ax.annotate(f"{row['lambda']:.2f}", (row["gini_mean"], row["total_expected_production_mean"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=9)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Lambda")
    ax.set_xlabel("Production Gini (lower = fairer)")
    ax.set_ylabel("Total Expected Production")
    ax.set_title("Pareto Curve: Efficiency vs Fairness")
    fig.tight_layout()
    fig.savefig(outdir / "pareto_total_production_vs_gini.png", dpi=220)
    plt.close(fig)

    # Prepare method-comparison slice at target lambda
    available_lambdas = lp["lambda"].dropna().to_numpy(dtype=float)
    chosen_lambda = _nearest_lambda(available_lambdas, args.target_lambda)
    comp_players = player_df[np.isclose(player_df["lambda"], chosen_lambda)].copy()
    comp_games = game_df[np.isclose(game_df["lambda"], chosen_lambda)].copy()

    # 5) Resource entropy violin by method
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=comp_players, x="method", y="entropy", inner="quartile", cut=0, ax=ax, color="#84A59D")
    ax.set_xlabel("Method")
    ax.set_ylabel("Shannon Entropy")
    ax.set_title(f"Resource Entropy by Method (lambda={chosen_lambda:.2f})")
    fig.tight_layout()
    fig.savefig(outdir / "entropy_violin_by_method.png", dpi=220)
    plt.close(fig)

    # 6) Max-min ratio bar chart by method
    fig, ax = plt.subplots(figsize=(9, 6))
    ratio_means = comp_games.groupby("method", as_index=False)["max_min_ratio"].mean().sort_values("max_min_ratio")
    sns.barplot(data=ratio_means, x="method", y="max_min_ratio", ax=ax, color="#BC6C25")
    ax.set_xlabel("Method")
    ax.set_ylabel("Max-Min Production Ratio")
    ax.set_title(f"Max-Min Ratio by Method (lambda={chosen_lambda:.2f})")
    fig.tight_layout()
    fig.savefig(outdir / "max_min_ratio_by_method.png", dpi=220)
    plt.close(fig)

    # 7) Production gap (max - min across players) distributions by method (overlapping histograms)
    if "production_gap" not in comp_games.columns:
        comp_games["production_gap"] = comp_games["strongest_production"] - comp_games["weakest_production"]
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = sorted(comp_games["method"].dropna().unique().tolist())
    palette = sns.color_palette("Set2", n_colors=max(3, len(methods)))
    for i, m in enumerate(methods):
        vals = comp_games[comp_games["method"] == m]["production_gap"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            ax.hist(vals, bins=20, alpha=0.35, density=True, label=m, color=palette[i])
    ax.set_xlabel("Production Gap (max - min across players)")
    ax.set_ylabel("Density")
    ax.set_title(f"Production Gap Distribution by Method (lambda={chosen_lambda:.2f})")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(outdir / "production_gap_hist_by_method.png", dpi=220)
    plt.close(fig)

    # 8) Dual heat map (node index x 1 strip, averaged over seeds)
    dual_slice = dual_df[np.isclose(dual_df["lambda"], chosen_lambda)].copy()
    if not dual_slice.empty:
        node_duals = (
            dual_slice.groupby("node_id", as_index=False)["capacity_dual"].mean().sort_values("node_id")
        )
        heat = np.expand_dims(node_duals["capacity_dual"].to_numpy(dtype=float), axis=0)

        fig, ax = plt.subplots(figsize=(12, 2.4))
        sns.heatmap(heat, cmap="magma", cbar=True, ax=ax)
        ax.set_xlabel("Board Node ID")
        ax.set_yticks([])
        ax.set_title(f"Dual Heat Map (Location Capacity Shadow Prices, lambda={chosen_lambda:.2f})")
        fig.tight_layout()
        fig.savefig(outdir / "dual_heatmap_capacity.png", dpi=220)
        plt.close(fig)

        # 9) Dual heat map on board-topology layout (edges + colored nodes)
        edges = get_edges()
        n_nodes = int(node_duals["node_id"].max()) + 1
        node_vals = np.zeros(n_nodes, dtype=float)
        node_vals[node_duals["node_id"].to_numpy(dtype=int)] = node_duals["capacity_dual"].to_numpy(dtype=float)
        pos = _board_layout_positions(n_nodes=n_nodes, edges=edges)

        fig, ax = plt.subplots(figsize=(8.5, 7.5))
        for a, b in edges:
            if 0 <= a < n_nodes and 0 <= b < n_nodes:
                ax.plot(
                    [pos[a, 0], pos[b, 0]],
                    [pos[a, 1], pos[b, 1]],
                    color="#9AA5B1",
                    linewidth=0.8,
                    alpha=0.5,
                    zorder=1,
                )

        sc = ax.scatter(
            pos[:, 0],
            pos[:, 1],
            c=node_vals,
            cmap="magma",
            s=110,
            edgecolors="black",
            linewidths=0.35,
            zorder=2,
        )
        cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
        cbar.set_label("Average Capacity Dual")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Board Dual Heat Map (lambda={chosen_lambda:.2f})")
        fig.tight_layout()
        fig.savefig(outdir / "dual_heatmap_board_layout.png", dpi=220)
        plt.close(fig)

    print(f"Saved lambda-sweep plots to: {outdir}")


if __name__ == "__main__":
    main()
