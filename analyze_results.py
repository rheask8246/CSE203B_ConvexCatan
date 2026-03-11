"""Analyze evaluation outputs and compute summary metrics with bootstrap CIs.

Example:
    python analyze_results.py --input-dir results/eval_20260306_120000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    '''Parse command-line arguments for analyzing Catan evaluation results.'''
    parser = argparse.ArgumentParser(description="Analyze Catan evaluation outputs")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory with player_metrics.csv")
    parser.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap draws")
    parser.add_argument("--seed", type=int, default=7, help="RNG seed for bootstrap")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory (default: <input-dir>/analysis)",
    )
    return parser.parse_args()


def bootstrap_ci(values: np.ndarray, n_boot: int, rng: np.random.Generator, alpha: float = 0.05):
    '''Compute a bootstrap confidence interval for the mean of the given values.'''
    if len(values) == 0:
        return (np.nan, np.nan)
    if len(values) == 1:
        return (float(values[0]), float(values[0]))

    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    samples = values[idx].mean(axis=1)
    lo = float(np.quantile(samples, alpha / 2))
    hi = float(np.quantile(samples, 1 - alpha / 2))
    return lo, hi


def main() -> None:
    '''Main function to analyze Catan evaluation results, computing summary metrics and saving analysis CSVs.'''
    args = parse_args()
    input_dir = args.input_dir
    outdir = args.outdir or (input_dir / "analysis")
    outdir.mkdir(parents=True, exist_ok=True)

    player_path = input_dir / "player_metrics.csv"
    game_path = input_dir / "game_metrics.csv"

    if not player_path.exists() or not game_path.exists():
        raise FileNotFoundError("Missing player_metrics.csv or game_metrics.csv in input directory")

    players = pd.read_csv(player_path)
    games = pd.read_csv(game_path)

    rng = np.random.default_rng(args.seed)

    agent_rows = []
    for agent, g in players.groupby("agent", sort=True):
        win = g["is_winner"].to_numpy(dtype=float)
        vp = g["final_vp"].to_numpy(dtype=float)
        rank = g["rank"].to_numpy(dtype=float)
        turns = g["num_turns"].to_numpy(dtype=float)
        decide = g["avg_decide_ms"].to_numpy(dtype=float)
        vp_delta = g["vp_minus_table_mean"].to_numpy(dtype=float)
        fairness = g["vp_gini_game"].to_numpy(dtype=float)

        win_lo, win_hi = bootstrap_ci(win, args.bootstrap, rng)
        vp_lo, vp_hi = bootstrap_ci(vp, args.bootstrap, rng)

        agent_rows.append(
            {
                "agent": agent,
                "n_rows": int(len(g)),
                "win_rate": float(win.mean()),
                "win_rate_ci_low": win_lo,
                "win_rate_ci_high": win_hi,
                "avg_final_vp": float(vp.mean()),
                "avg_final_vp_ci_low": vp_lo,
                "avg_final_vp_ci_high": vp_hi,
                "avg_rank": float(rank.mean()),
                "avg_turns": float(turns.mean()),
                "avg_decide_ms": float(decide.mean()),
                "avg_vp_minus_table_mean": float(vp_delta.mean()),
                "avg_game_vp_gini": float(fairness.mean()),
                "longest_road_rate": float(g["has_longest_road"].mean()),
                "largest_army_rate": float(g["has_largest_army"].mean()),
                "avg_roads_built": float(g["roads_built"].mean()),
                "avg_settlements_built": float(g["settlements_built"].mean()),
                "avg_cities_built": float(g["cities_built"].mean()),
            }
        )

    agent_summary = pd.DataFrame(agent_rows).sort_values("win_rate", ascending=False)

    seat_summary = (
        players.groupby(["agent", "turn_order"], as_index=False)
        .agg(
            n=("is_winner", "size"),
            win_rate=("is_winner", "mean"),
            avg_final_vp=("final_vp", "mean"),
            avg_rank=("rank", "mean"),
        )
        .sort_values(["agent", "turn_order"])
    )

    lineup_summary = (
        players.groupby(["lineup", "agent"], as_index=False)
        .agg(
            n=("is_winner", "size"),
            win_rate=("is_winner", "mean"),
            avg_final_vp=("final_vp", "mean"),
            avg_rank=("rank", "mean"),
            avg_turns=("num_turns", "mean"),
            avg_game_vp_gini=("vp_gini_game", "mean"),
        )
        .sort_values(["lineup", "win_rate"], ascending=[True, False])
    )

    # Pairwise VP differential matrix: E[VP_i - VP_j]
    pair_rows = []
    for _, grp in players.groupby("global_game_id"):
        recs = grp[["agent", "final_vp"]].to_records(index=False)
        for i in range(len(recs)):
            for j in range(len(recs)):
                if i == j:
                    continue
                pair_rows.append(
                    {
                        "agent_i": recs[i][0],
                        "agent_j": recs[j][0],
                        "vp_diff": float(recs[i][1] - recs[j][1]),
                    }
                )

    pair_df = pd.DataFrame(pair_rows)
    pairwise_vp_diff = (
        pair_df.groupby(["agent_i", "agent_j"], as_index=False)["vp_diff"]
        .mean()
        .pivot(index="agent_i", columns="agent_j", values="vp_diff")
        .sort_index()
    )

    game_summary = games.groupby("lineup", as_index=False).agg(
        n_games=("global_game_id", "size"),
        avg_turns=("num_turns", "mean"),
        avg_duration_s=("game_duration_s", "mean"),
        avg_vp_std=("vp_std", "mean"),
        avg_vp_gini=("vp_gini", "mean"),
        avg_vp_range=("vp_range", "mean"),
    )

    agent_summary.to_csv(outdir / "agent_summary.csv", index=False)
    seat_summary.to_csv(outdir / "seat_summary.csv", index=False)
    lineup_summary.to_csv(outdir / "lineup_summary.csv", index=False)
    pairwise_vp_diff.to_csv(outdir / "pairwise_vp_diff.csv")
    game_summary.to_csv(outdir / "game_summary.csv", index=False)

    report = {
        "input_dir": str(input_dir),
        "rows": {
            "player_metrics": int(len(players)),
            "game_metrics": int(len(games)),
        },
        "agents": agent_summary["agent"].tolist(),
        "best_win_rate_agent": agent_summary.iloc[0]["agent"] if len(agent_summary) else None,
        "best_win_rate": float(agent_summary.iloc[0]["win_rate"]) if len(agent_summary) else None,
    }
    (outdir / "analysis_report.json").write_text(json.dumps(report, indent=2))

    print(f"Saved analysis to: {outdir}")
    print(agent_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
