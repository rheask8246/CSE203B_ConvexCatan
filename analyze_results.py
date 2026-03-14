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
    parser = argparse.ArgumentParser(description="Analyze Catan evaluation outputs")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--outdir", type=Path, default=None)
    return parser.parse_args()


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
):
    """Bootstrap 95% CI for the mean. Returns (lo, hi)."""
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

    # Coerce production columns to numeric (may be empty string if convex_solver absent).
    for col in ("prod_gini_final", "prod_max_min_ratio", "turns_until_equalized"):
        if col in players.columns:
            players[col] = pd.to_numeric(players[col], errors="coerce")
        if col in games.columns:
            games[col] = pd.to_numeric(games[col], errors="coerce")

    rng = np.random.default_rng(args.seed)

    # ------------------------------------------------------------------ #
    # 1. Per-agent summary
    # ------------------------------------------------------------------ #
    agent_rows = []
    # Sort by agent name for deterministic bootstrap order.
    for agent, g in players.groupby("agent", sort=True):
        win = g["is_winner"].to_numpy(dtype=float)
        vp = g["final_vp"].to_numpy(dtype=float)
        rank = g["rank"].to_numpy(dtype=float)
        decide = g["avg_decide_ms"].to_numpy(dtype=float)
        vp_delta = g["vp_minus_table_mean"].to_numpy(dtype=float)

        # Production fairness — these are game-level metrics stored per player row.
        # To avoid double-counting, aggregate at game level first, then average.
        # We use first() per game since the value is the same for all players in a game.
        game_prod = (
            g.groupby("global_game_id")[["prod_gini_final", "prod_max_min_ratio", "turns_until_equalized"]]
            .first()
        )

        win_lo, win_hi = bootstrap_ci(win, args.bootstrap, rng)
        vp_lo, vp_hi = bootstrap_ci(vp, args.bootstrap, rng)

        agent_rows.append({
            "agent": agent,
            "n_games": int(g["global_game_id"].nunique()),
            "n_rows": int(len(g)),
            "win_rate": float(win.mean()),
            "win_rate_ci_low": win_lo,
            "win_rate_ci_high": win_hi,
            "avg_final_vp": float(vp.mean()),
            "avg_final_vp_ci_low": vp_lo,
            "avg_final_vp_ci_high": vp_hi,
            "avg_rank": float(rank.mean()),
            "avg_decide_ms": float(decide.mean()),
            "avg_vp_minus_table_mean": float(vp_delta.mean()),
            # Production fairness: averaged over games this agent appeared in.
            # These are meaningful because CONVEX appears in different lineups than R/GREEDY.
            "avg_prod_gini_final": float(game_prod["prod_gini_final"].mean()),
            "avg_prod_max_min_ratio": float(game_prod["prod_max_min_ratio"].mean()),
            "avg_turns_until_equalized": float(
                game_prod.loc[game_prod["turns_until_equalized"] >= 0, "turns_until_equalized"].mean()
            ) if (game_prod["turns_until_equalized"] >= 0).any() else float("nan"),
            # Building metrics
            "avg_roads_built": float(g["roads_built"].mean()),
            "avg_settlements_built": float(g["settlements_built"].mean()),
            "avg_cities_built": float(g["cities_built"].mean()),
            "longest_road_rate": float(g["has_longest_road"].mean()),
            "largest_army_rate": float(g["has_largest_army"].mean()),
        })

    agent_summary = pd.DataFrame(agent_rows).sort_values("win_rate", ascending=False)

    # ------------------------------------------------------------------ #
    # 2. Per-seat (turn order) breakdown
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # 3. Per-lineup breakdown
    # ------------------------------------------------------------------ #
    lineup_summary = (
        players.groupby(["lineup", "agent"], as_index=False)
        .agg(
            n=("is_winner", "size"),
            win_rate=("is_winner", "mean"),
            avg_final_vp=("final_vp", "mean"),
            avg_rank=("rank", "mean"),
            avg_turns=("num_turns", "mean"),
        )
        .sort_values(["lineup", "win_rate"], ascending=[True, False])
    )

    # Merge in game-level production fairness at lineup level.
    game_lineup_fairness = (
        games.groupby("lineup", as_index=False)
        .agg(
            avg_prod_gini=("prod_gini_final", "mean"),
            avg_prod_max_min=("prod_max_min_ratio", "mean"),
            avg_vp_gini=("vp_gini", "mean"),
            avg_vp_range=("vp_range", "mean"),
        )
    )
    lineup_summary = lineup_summary.merge(game_lineup_fairness, on="lineup", how="left")

    # ------------------------------------------------------------------ #
    # 4. Pairwise VP differential matrix
    # ------------------------------------------------------------------ #
    # Compute per-game (agent_i, agent_j) VP diff, then average.
    # Use pivot_table (handles duplicate keys via aggfunc) instead of pivot.
    pair_rows = []
    for _, grp in players.groupby("global_game_id"):
        recs = grp[["agent", "final_vp"]].to_records(index=False)
        for i in range(len(recs)):
            for j in range(len(recs)):
                if i == j:
                    continue
                pair_rows.append({
                    "agent_i": recs[i][0],
                    "agent_j": recs[j][0],
                    "vp_diff": float(recs[i][1]) - float(recs[j][1]),
                })

    pair_df = pd.DataFrame(pair_rows)
    pairwise_vp_diff = (
        pair_df.groupby(["agent_i", "agent_j"], as_index=False)["vp_diff"]
        .mean()
        .pipe(lambda df: df.pivot_table(
            index="agent_i", columns="agent_j", values="vp_diff", aggfunc="mean"
        ))
        .rename_axis(index="agent_i", columns="agent_j")
        .sort_index()
    )

    # ------------------------------------------------------------------ #
    # 5. Game-level summary
    # ------------------------------------------------------------------ #
    game_summary = games.groupby("lineup", as_index=False).agg(
        n_games=("global_game_id", "size"),
        avg_turns=("num_turns", "mean"),
        avg_duration_s=("game_duration_s", "mean"),
        avg_vp_std=("vp_std", "mean"),
        avg_vp_gini=("vp_gini", "mean"),
        avg_vp_range=("vp_range", "mean"),
        avg_prod_gini_final=("prod_gini_final", "mean"),
        avg_prod_max_min_ratio=("prod_max_min_ratio", "mean"),
        avg_turns_until_equalized=("turns_until_equalized", lambda x: x[x >= 0].mean()),
    )

    # ------------------------------------------------------------------ #
    # 6. Fairness-focused cross-lineup summary
    #    Primary evaluation table: one row per lineup, all fairness metrics.
    # ------------------------------------------------------------------ #
    fairness_summary = games.groupby("lineup", as_index=False).agg(
        n_games=("global_game_id", "size"),
        # VP fairness
        vp_gini_mean=("vp_gini", "mean"),
        vp_gini_std=("vp_gini", "std"),
        vp_range_mean=("vp_range", "mean"),
        # Production fairness
        prod_gini_mean=("prod_gini_final", "mean"),
        prod_gini_std=("prod_gini_final", "std"),
        prod_max_min_mean=("prod_max_min_ratio", "mean"),
        # Equalization speed
        turns_equalized_mean=("turns_until_equalized", lambda x: x[x >= 0].mean()),
    )

    # ------------------------------------------------------------------ #
    # Save outputs
    # ------------------------------------------------------------------ #
    agent_summary.to_csv(outdir / "agent_summary.csv", index=False)
    seat_summary.to_csv(outdir / "seat_summary.csv", index=False)
    lineup_summary.to_csv(outdir / "lineup_summary.csv", index=False)
    pairwise_vp_diff.to_csv(outdir / "pairwise_vp_diff.csv")
    game_summary.to_csv(outdir / "game_summary.csv", index=False)
    fairness_summary.to_csv(outdir / "fairness_summary.csv", index=False)

    report = {
        "input_dir": str(input_dir),
        "rows": {
            "player_metrics": int(len(players)),
            "game_metrics": int(len(games)),
        },
        "agents": agent_summary["agent"].tolist(),
        "best_win_rate_agent": agent_summary.iloc[0]["agent"] if len(agent_summary) else None,
        "best_win_rate": float(agent_summary.iloc[0]["win_rate"]) if len(agent_summary) else None,
        "fairness_by_lineup": fairness_summary.set_index("lineup")[
            ["vp_gini_mean", "prod_gini_mean", "prod_max_min_mean"]
        ].to_dict(orient="index"),
    }
    (outdir / "analysis_report.json").write_text(json.dumps(report, indent=2, default=str))

    print(f"Saved analysis to: {outdir}")
    print("\n=== Agent Summary ===")
    print(agent_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== Fairness Summary ===")
    print(fairness_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()