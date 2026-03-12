"""Run lambda-sweep experiments for initial-placement production fairness metrics.

Example:
    python lambda_sweep.py --num-seeds 300 --lambda-start 0 --lambda-end 2 --lambda-count 25 --outdir results/lambda_sweep
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from catanatron.game import Game
from catanatron.models.enums import RESOURCES
from catanatron.models.player import Color, RandomPlayer
from catanatron.models.board import get_edges

from agents.convex_solver import solve_initial_all_players, summarize_production

COLORS = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lambda sweep for initial-placement fairness metrics")
    parser.add_argument("--num-seeds", type=int, default=200, help="Number of random boards (seeds)")
    parser.add_argument("--seed-start", type=int, default=1, help="First RNG seed")
    parser.add_argument("--num-players", type=int, default=3, choices=[3, 4], help="Number of players")
    parser.add_argument("--lambda-start", type=float, default=0.0, help="Lambda sweep start")
    parser.add_argument("--lambda-end", type=float, default=2.0, help="Lambda sweep end")
    parser.add_argument("--lambda-count", type=int, default=25, help="Number of lambda points")
    parser.add_argument(
        "--no-baselines",
        action="store_true",
        help="Disable baseline placement comparisons (RANDOM, WEIGHTED_RANDOM, GREEDY)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory (default: results/lambda_sweep_<timestamp>)",
    )
    return parser.parse_args()


def _production_matrix(game: Game) -> np.ndarray:
    cmap = game.state.board.map
    node_ids = sorted(cmap.node_production.keys())
    n_nodes = (max(node_ids) + 1) if node_ids else 0
    A = np.zeros((n_nodes, len(RESOURCES)), dtype=float)
    for node_id, prod in cmap.node_production.items():
        if 0 <= node_id < n_nodes:
            for k, res in enumerate(RESOURCES):
                A[node_id, k] = float(prod.get(res, 0.0))
    return A


def _adjacency(n_nodes: int) -> list[set[int]]:
    adj = [set() for _ in range(n_nodes)]
    for a, b in get_edges():
        if 0 <= a < n_nodes and 0 <= b < n_nodes:
            adj[a].add(b)
            adj[b].add(a)
    return adj


def _feasible_nodes(blocked: set[int], n_nodes: int) -> list[int]:
    return [n for n in range(n_nodes) if n not in blocked]


def _sample_baseline_allocation(A: np.ndarray, players: int, rng: np.random.Generator, mode: str) -> np.ndarray:
    n_nodes = A.shape[0]
    adj = _adjacency(n_nodes)
    blocked: set[int] = set()
    x = np.zeros((players, n_nodes), dtype=float)

    node_score = A.sum(axis=1)

    for p in range(players):
        for _ in range(2):
            candidates = _feasible_nodes(blocked, n_nodes)
            if not candidates:
                continue

            if mode == "greedy":
                max_score = float(np.max(node_score[candidates]))
                best = [n for n in candidates if abs(node_score[n] - max_score) < 1e-12]
                choice = int(rng.choice(best))
            elif mode == "weighted":
                weights = node_score[candidates].astype(float) + 1e-6
                weights = weights / np.sum(weights)
                choice = int(rng.choice(candidates, p=weights))
            else:
                choice = int(rng.choice(candidates))

            x[p, choice] = 1.0
            blocked.add(choice)
            blocked.update(adj[choice])

    return x


def _rows_from_summary(seed: int, method: str, lambda_value: float | None, summary: dict) -> tuple[dict, list[dict]]:
    v = np.asarray(summary["production_by_player"], dtype=float)
    ent = np.asarray(summary["entropy_by_player"], dtype=float)
    gaps = np.asarray(summary["gap_from_max_by_player"], dtype=float)

    weakest = float(np.min(v)) if v.size else np.nan
    strongest = float(np.max(v)) if v.size else np.nan
    game_row = {
        "seed": seed,
        "method": method,
        "lambda": (float(lambda_value) if lambda_value is not None else np.nan),
        "gini": float(summary["gini"]),
        "max_min_ratio": float(summary["max_min_ratio"]),
        "total_expected_production": float(summary["total_expected_production"]),
        "mean_entropy": float(summary["mean_entropy"]),
        "weakest_production": weakest,
        "strongest_production": strongest,
        "production_gap": float(strongest - weakest) if (v.size and np.isfinite(strongest) and np.isfinite(weakest)) else np.nan,
    }

    player_rows = []
    for i in range(len(v)):
        player_rows.append(
            {
                "seed": seed,
                "method": method,
                "lambda": (float(lambda_value) if lambda_value is not None else np.nan),
                "player_index": i,
                "expected_production": float(v[i]),
                "entropy": float(ent[i]),
                "gap_from_max": float(gaps[i]),
            }
        )

    return game_row, player_rows


def bootstrap_ci(values: np.ndarray, n_boot: int, rng: np.random.Generator, alpha: float = 0.05) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (np.nan, np.nan)
    if values.size == 1:
        return (float(values[0]), float(values[0]))
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    samples = values[idx].mean(axis=1)
    return float(np.quantile(samples, alpha / 2)), float(np.quantile(samples, 1 - alpha / 2))


def build_summary(game_df: pd.DataFrame, n_boot: int = 2000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    keys = ["method", "lambda"]
    for (method, lam), g in game_df.groupby(keys, dropna=False):
        row = {"method": method, "lambda": lam, "n": int(len(g))}
        for metric in ["gini", "max_min_ratio", "total_expected_production", "mean_entropy"]:
            vals = g[metric].to_numpy(dtype=float)
            lo, hi = bootstrap_ci(vals, n_boot=n_boot, rng=rng)
            row[f"{metric}_mean"] = float(np.mean(vals))
            row[f"{metric}_ci_low"] = lo
            row[f"{metric}_ci_high"] = hi
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["method", "lambda"], na_position="last")


def prepare_outdir(outdir: Path | None) -> Path:
    if outdir is not None:
        out = outdir
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out = Path("results") / f"lambda_sweep_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def main() -> None:
    args = parse_args()
    outdir = prepare_outdir(args.outdir)

    lambda_values = np.linspace(args.lambda_start, args.lambda_end, args.lambda_count, dtype=float)

    game_rows: list[dict] = []
    player_rows: list[dict] = []
    dual_rows: list[dict] = []

    total_steps = args.num_seeds * len(lambda_values)
    step = 0

    for s in range(args.num_seeds):
        seed = args.seed_start + s
        players = [RandomPlayer(COLORS[i]) for i in range(args.num_players)]
        game = Game(players, seed=seed, vps_to_win=10)

        # LP sweep rows
        for lam in lambda_values:
            details = solve_initial_all_players(game, lambda_value=float(lam))
            summary = details["metrics"]
            g_row, p_rows = _rows_from_summary(seed=seed, method="LP", lambda_value=float(lam), summary=summary)
            game_rows.append(g_row)
            player_rows.extend(p_rows)

            duals = np.asarray(details.get("capacity_duals", []), dtype=float)
            for node_id, dual in enumerate(duals):
                dual_rows.append(
                    {
                        "seed": seed,
                        "method": "LP",
                        "lambda": float(lam),
                        "node_id": node_id,
                        "capacity_dual": float(dual),
                    }
                )

            step += 1
            if step % 100 == 0 or step == total_steps:
                print(f"progress: {step}/{total_steps} LP evaluations")

        if not args.no_baselines:
            A = _production_matrix(game)
            rng = np.random.default_rng(seed)
            random_x = _sample_baseline_allocation(A, args.num_players, rng, mode="random")
            weighted_x = _sample_baseline_allocation(A, args.num_players, rng, mode="weighted")
            greedy_x = _sample_baseline_allocation(A, args.num_players, rng, mode="greedy")

            baseline_payloads = [
                ("RANDOM", summarize_production(random_x @ A)),
                ("WEIGHTED_RANDOM", summarize_production(weighted_x @ A)),
                ("GREEDY", summarize_production(greedy_x @ A)),
            ]

            # Repeat baselines across lambdas to simplify overlay plots.
            for lam in lambda_values:
                for method, summary in baseline_payloads:
                    g_row, p_rows = _rows_from_summary(seed=seed, method=method, lambda_value=float(lam), summary=summary)
                    game_rows.append(g_row)
                    player_rows.extend(p_rows)

    game_df = pd.DataFrame(game_rows)
    player_df = pd.DataFrame(player_rows)
    dual_df = pd.DataFrame(dual_rows)
    summary_df = build_summary(game_df)

    game_df.to_csv(outdir / "lambda_sweep_game_metrics.csv", index=False)
    player_df.to_csv(outdir / "lambda_sweep_player_metrics.csv", index=False)
    dual_df.to_csv(outdir / "lambda_sweep_duals.csv", index=False)
    summary_df.to_csv(outdir / "lambda_sweep_summary.csv", index=False)

    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_seeds": args.num_seeds,
        "seed_start": args.seed_start,
        "num_players": args.num_players,
        "lambda_start": args.lambda_start,
        "lambda_end": args.lambda_end,
        "lambda_count": args.lambda_count,
        "with_baselines": bool(not args.no_baselines),
        "outputs": {
            "game_metrics": str(outdir / "lambda_sweep_game_metrics.csv"),
            "player_metrics": str(outdir / "lambda_sweep_player_metrics.csv"),
            "duals": str(outdir / "lambda_sweep_duals.csv"),
            "summary": str(outdir / "lambda_sweep_summary.csv"),
        },
    }
    (outdir / "metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"Saved lambda sweep outputs to: {outdir}")


if __name__ == "__main__":
    main()
