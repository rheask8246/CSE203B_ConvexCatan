"""Run full games: ConvexAgent vs Random vs SelfishAgent, sweeping lambda.
Also runs baseline: R, R, SELFISH (no ConvexAgent) for comparison.

Plays CONVEX(lambda), R, SELFISH and R, R, SELFISH for each selfish agent.
Tracks Gini of expected production after initial placement (what the LP optimizes).
Plot both on same axes to show ConvexAgent's equalizing effect.

Example:
    python evaluate_convex_sweep.py --num-games 100 --outdir results/convex_sweep
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Sequence

from catanatron.game import Game
from catanatron.models.player import Color, Player, RandomPlayer

from agents.greedy_agent import GreedyAgent
from agents.players import ConvexAgent
from agents.mcts import MCTSPlayer
from agents.minimax import AlphaBetaPlayer
from agents.value import ValueFunctionPlayer
from agents.weighted_random import WeightedRandomPlayer as CustomWeightedRandomPlayer
from agents.convex_solver import expected_production_gini


COLORS = [Color.RED, Color.BLUE, Color.ORANGE]
SELFISH_AGENTS = ["GREEDY", "AB", "MCTS", "VALUE", "WR"]


def pkey(idx: int, key: str) -> str:
    return f"P{idx}_{key}"


def gini(values: Sequence[float]) -> float:
    arr = [float(v) for v in values]
    n = len(arr)
    if n == 0:
        return float("nan")
    mean_v = sum(arr) / n
    if mean_v == 0:
        return 0.0
    diff_sum = sum(abs(i - j) for i in arr for j in arr)
    return diff_sum / (2 * n * n * mean_v)


def get_selfish_factory(code: str) -> Callable[[Color], Player]:
    code = code.strip().upper()
    factories = {
        "GREEDY": GreedyAgent,
        "AB": AlphaBetaPlayer,
        "MCTS": MCTSPlayer,
        "VALUE": ValueFunctionPlayer,
        "WR": CustomWeightedRandomPlayer,
    }
    if code not in factories:
        raise ValueError(f"Unknown selfish agent '{code}'. Supported: {', '.join(factories)}")
    return factories[code]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ConvexAgent vs Random vs SelfishAgent across lambda sweep"
    )
    parser.add_argument("--num-games", type=int, default=100, help="Games per (lambda, selfish_agent)")
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--lambda-start", type=float, default=0.0)
    parser.add_argument("--lambda-end", type=float, default=2.0)
    parser.add_argument("--lambda-count", type=int, default=10, help="Lambda points between start and end (e.g. 10 for 0..2)")
    parser.add_argument(
        "--selfish-agents",
        nargs="+",
        default=SELFISH_AGENTS,
        help=f"Selfish agents to compare (default: {' '.join(SELFISH_AGENTS)})",
    )
    parser.add_argument("--outdir", type=Path, default=None)
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    outdir = args.outdir or Path("results") / f"convex_sweep_{time.strftime('%Y%m%d_%H%M%S')}"
    outdir.mkdir(parents=True, exist_ok=True)

    lambda_values = [
        round(v, 6)
        for v in [
            args.lambda_start + i * (args.lambda_end - args.lambda_start) / (args.lambda_count - 1)
            for i in range(args.lambda_count)
        ]
    ]

    game_fields = [
        "lineup_type", "lambda", "selfish_agent", "game_idx", "seed",
        "prod_gini", "prod_min", "prod_max", "prod_range",
        "winner_agent", "num_turns",
    ]
    player_fields = [
        "lineup_type", "lambda", "selfish_agent", "game_idx", "seed", "turn_order", "agent",
        "expected_production", "final_vp", "total_resources_in_hand", "prod_gini_game",
    ]

    total = len(lambda_values) * len(args.selfish_agents) * args.num_games
    total += len(args.selfish_agents) * args.num_games  # baseline
    step = 0

    def run_game(players, agents, lam, selfish_code, game_idx, lineup_type):
        nonlocal step
        seed = args.seed_start + game_idx
        game = Game(players, seed=seed, vps_to_win=10)
        while game.state.is_initial_build_phase and game.winning_color() is None:
            game.play_tick()
        prod_gini, prod_by_player = expected_production_gini(game)
        prod_min, prod_max = float(prod_by_player.min()), float(prod_by_player.max())
        prod_range = prod_max - prod_min
        game.play()
        state = game.state
        vps = [int(state.player_state[pkey(i, "ACTUAL_VICTORY_POINTS")]) for i in range(3)]
        winner_color = game.winning_color()
        winner_agent = agents[state.colors.index(winner_color)] if winner_color else "NONE"
        return {
            "game": {"lineup_type": lineup_type, "lambda": lam, "selfish_agent": selfish_code,
                     "game_idx": game_idx, "seed": seed, "prod_gini": round(prod_gini, 6),
                     "prod_min": round(prod_min, 6), "prod_max": round(prod_max, 6),
                     "prod_range": round(prod_range, 6), "winner_agent": winner_agent,
                     "num_turns": int(state.num_turns)},
            "players": prod_by_player, "vps": vps, "agents": agents, "state": state,
        }

    with (outdir / "game_metrics.csv").open("w", newline="") as gf, \
         (outdir / "player_metrics.csv").open("w", newline="") as pf:
        g_writer = csv.DictWriter(gf, fieldnames=game_fields)
        p_writer = csv.DictWriter(pf, fieldnames=player_fields)
        g_writer.writeheader()
        p_writer.writeheader()

        for lam in lambda_values:
            for selfish_code in args.selfish_agents:
                selfish_factory = get_selfish_factory(selfish_code)
                for game_idx in range(args.num_games):
                    step += 1
                    players = [ConvexAgent(Color.RED, lambda_value=lam), RandomPlayer(Color.BLUE), selfish_factory(Color.ORANGE)]
                    agents = ["CONVEX", "R", selfish_code]
                    out = run_game(players, agents, lam, selfish_code, game_idx, "CONVEX")
                    g_writer.writerow(out["game"])
                    for i in range(3):
                        total_res = sum(int(out["state"].player_state[pkey(i, r)]) for r in ["WOOD_IN_HAND", "BRICK_IN_HAND", "SHEEP_IN_HAND", "WHEAT_IN_HAND", "ORE_IN_HAND"])
                        p_writer.writerow({
                            "lineup_type": "CONVEX", "lambda": lam, "selfish_agent": selfish_code,
                            "game_idx": game_idx, "seed": out["game"]["seed"], "turn_order": i, "agent": agents[i],
                            "expected_production": round(float(out["players"][i]), 6), "final_vp": out["vps"][i],
                            "total_resources_in_hand": total_res, "prod_gini_game": out["game"]["prod_gini"],
                        })
                    if step % 50 == 0 or step == total:
                        print(f"  progress: {step}/{total} games")

        # Baseline: R, R, SELFISH (no ConvexAgent)
        for selfish_code in args.selfish_agents:
            selfish_factory = get_selfish_factory(selfish_code)
            for game_idx in range(args.num_games):
                step += 1
                players = [RandomPlayer(Color.RED), RandomPlayer(Color.BLUE), selfish_factory(Color.ORANGE)]
                agents = ["R", "R", selfish_code]
                out = run_game(players, agents, float("nan"), selfish_code, game_idx, "BASELINE")
                g_writer.writerow(out["game"])
                for i in range(3):
                    total_res = sum(int(out["state"].player_state[pkey(i, r)]) for r in ["WOOD_IN_HAND", "BRICK_IN_HAND", "SHEEP_IN_HAND", "WHEAT_IN_HAND", "ORE_IN_HAND"])
                    p_writer.writerow({
                        "lineup_type": "BASELINE", "lambda": float("nan"), "selfish_agent": selfish_code,
                        "game_idx": game_idx, "seed": out["game"]["seed"], "turn_order": i, "agent": agents[i],
                        "expected_production": round(float(out["players"][i]), 6), "final_vp": out["vps"][i],
                        "total_resources_in_hand": total_res, "prod_gini_game": out["game"]["prod_gini"],
                    })
                if step % 50 == 0 or step == total:
                    print(f"  progress: {step}/{total} games")

    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_games_per_combo": args.num_games,
        "lambda_values": lambda_values,
        "selfish_agents": args.selfish_agents,
        "seed_start": args.seed_start,
    }
    (outdir / "metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"Saved to {outdir}")


if __name__ == "__main__":
    run()
