"""Run reproducible Catan evaluations and export per-player/per-game metrics.

Examples:
    python evaluate.py --num-games 300 --outdir results/exp1
    python evaluate.py --lineup R,R,R,R --lineup GREEDY,R,R,R --lineup CONVEX,R,R,R
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence

from catanatron.game import Game
from catanatron.models.player import Color, Player, RandomPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer

from agents.greedy_agent import GreedyAgent
from agents.players import ConvexAgent
from agents.mcts import MCTSPlayer
from agents.minimax import AlphaBetaPlayer
from agents.value import ValueFunctionPlayer
from agents.weighted_random import WeightedRandomPlayer as CustomWeightedRandomPlayer


MAX_ROADS = 15
MAX_SETTLEMENTS = 5
MAX_CITIES = 4
COLORS = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]


@dataclass
class TimedPlayer(Player):
    """Wrap a player and track total decision latency."""

    # inner player to delegate to
    inner: Player

    def __init__(self, inner: Player):
        '''Initialize the TimedPlayer with an inner player.'''
        super().__init__(inner.color, is_bot=inner.is_bot)
        self.inner = inner
        self.total_decide_s = 0.0
        self.decide_calls = 0

    def decide(self, game, playable_actions):
        '''Override the decide method to time the decision latency.'''
        t0 = time.perf_counter()
        action = self.inner.decide(game, playable_actions)
        self.total_decide_s += time.perf_counter() - t0
        self.decide_calls += 1
        return action

    def reset_state(self):
        '''Reset the player's state and timing metrics.'''
        if hasattr(self.inner, "reset_state"):
            self.inner.reset_state()
        self.total_decide_s = 0.0
        self.decide_calls = 0


def parse_args() -> argparse.Namespace:
    '''Parse command-line arguments for the evaluation script.'''
    parser = argparse.ArgumentParser(description="Evaluate Catan agents with reproducible seeds")
    parser.add_argument(
        "--lineup",
        action="append",
        default=[],
        help="One 4-player lineup, comma-separated (e.g. CONVEX,R,R,R). Repeat for multiple scenarios.",
    )
    parser.add_argument("--num-games", type=int, default=200, help="Games per lineup")
    parser.add_argument("--seed-start", type=int, default=1, help="First RNG seed")
    parser.add_argument("--vps-to-win", type=int, default=10, help="Victory points needed to win")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory (default: results/eval_<timestamp>)",
    )
    return parser.parse_args()


def default_lineups() -> List[str]:
    '''Return default lineups if none are provided via command-line arguments.'''
    return [
        "R,R,R,R",
        "GREEDY,R,R,R",
        "CONVEX,R,R,R",
    ]


def get_agent_factory(code: str) -> Callable[[Color], Player]:
    '''Map an agent code to a factory function that creates a Player instance for a given color.'''
    code = code.strip().upper()
    factories: Dict[str, Callable[[Color], Player]] = {
        "R": RandomPlayer,
        # "W": WeightedRandomPlayer,
        "GREEDY": GreedyAgent,
        "CONVEX": ConvexAgent,
        "MCTS": MCTSPlayer,
        "AB": AlphaBetaPlayer,
        "VALUE": ValueFunctionPlayer,
        "WR": CustomWeightedRandomPlayer,
    }
    if code not in factories:
        supported = ", ".join(sorted(factories))
        raise ValueError(f"Unknown agent code '{code}'. Supported: {supported}")
    return factories[code]


def parse_lineup(lineup: str) -> List[str]:
    '''Parse a lineup string into a list of agent codes, validating the format and supported agents.'''
    parts = [p.strip().upper() for p in lineup.split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError(f"Lineup must contain exactly 4 players: '{lineup}'")
    for part in parts:
        get_agent_factory(part)
    return parts


def gini(values: Sequence[float]) -> float:
    '''Compute the Gini coefficient for a list of values, which is a measure of inequality.'''
    arr = [float(v) for v in values]
    n = len(arr)
    if n == 0:
        return float("nan")
    mean_v = sum(arr) / n
    if mean_v == 0:
        return 0.0
    diff_sum = 0.0
    for i in arr:
        for j in arr:
            diff_sum += abs(i - j)
    return diff_sum / (2 * n * n * mean_v)


def competition_ranks(values: Sequence[float]) -> List[int]:
    '''Assign competition ranks to a list of values, with 1 being the highest rank.'''
    unique = sorted(set(values), reverse=True)
    rank_map = {v: i + 1 for i, v in enumerate(unique)}
    return [rank_map[v] for v in values]


def pkey(idx: int, key: str) -> str:
    '''Helper to create a player-specific key for game state lookups.'''
    return f"P{idx}_{key}"


def prepare_outdir(outdir: Path | None) -> Path:
    '''Prepare the output directory for saving results, creating it if necessary.'''
    if outdir is not None:
        out = outdir
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out = Path("results") / f"eval_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def run() -> None:
    '''Main function to run the Catan evaluation, generating player and game metrics CSVs and a metadata JSON.'''
    args = parse_args()
    lineups = args.lineup if args.lineup else default_lineups()
    parsed_lineups = [parse_lineup(x) for x in lineups]
    all_codes = {code for lineup in parsed_lineups for code in lineup}

    if "CONVEX" in all_codes:
        try:
            import cvxpy  # noqa: F401
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "CONVEX lineup requested but cvxpy is not installed. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

    outdir = prepare_outdir(args.outdir)
    player_csv = outdir / "player_metrics.csv"
    game_csv = outdir / "game_metrics.csv"
    meta_json = outdir / "metadata.json"

    player_fields = [
        "lineup_id",
        "lineup",
        "game_in_lineup",
        "global_game_id",
        "seed",
        "input_slot",
        "seat_index",
        "color",
        "agent",
        "is_winner",
        "rank",
        "final_vp",
        "public_vp",
        "roads_built",
        "settlements_built",
        "cities_built",
        "longest_road_length",
        "has_longest_road",
        "has_largest_army",
        "knights_played",
        "wood_in_hand",
        "brick_in_hand",
        "sheep_in_hand",
        "wheat_in_hand",
        "ore_in_hand",
        "total_resources_in_hand",
        "decide_calls",
        "total_decide_s",
        "avg_decide_ms",
        "num_turns",
        "game_duration_s",
        "vp_std_game",
        "vp_min_game",
        "vp_range_game",
        "vp_gini_game",
        "vp_minus_table_mean",
    ]

    game_fields = [
        "lineup_id",
        "lineup",
        "game_in_lineup",
        "global_game_id",
        "seed",
        "has_winner",
        "winner_color",
        "winner_agent",
        "num_turns",
        "game_duration_s",
        "vp_std",
        "vp_min",
        "vp_max",
        "vp_range",
        "vp_gini",
    ]

    global_game_id = 0
    total_games = len(parsed_lineups) * args.num_games

    with player_csv.open("w", newline="") as pf, game_csv.open("w", newline="") as gf:
        p_writer = csv.DictWriter(pf, fieldnames=player_fields)
        g_writer = csv.DictWriter(gf, fieldnames=game_fields)
        p_writer.writeheader()
        g_writer.writeheader()

        for lineup_id, lineup_codes in enumerate(parsed_lineups):
            lineup_name = ",".join(lineup_codes)
            print(f"[lineup {lineup_id + 1}/{len(parsed_lineups)}] {lineup_name}")

            for game_in_lineup in range(args.num_games):
                seed = args.seed_start + game_in_lineup
                global_game_id += 1

                raw_players = []
                code_by_color: Dict[Color, str] = {}
                input_slot_by_color: Dict[Color, int] = {}

                for i, code in enumerate(lineup_codes):
                    color = COLORS[i]
                    player = get_agent_factory(code)(color)
                    raw_players.append(TimedPlayer(player))
                    code_by_color[color] = code
                    input_slot_by_color[color] = i

                game = Game(raw_players, seed=seed, vps_to_win=args.vps_to_win)
                t0 = time.perf_counter()
                game.play()
                game_duration_s = time.perf_counter() - t0

                state = game.state
                winner_color = game.winning_color()
                num_turns = int(state.num_turns)

                # state.players is seating order (P0..P3)
                seat_players: List[TimedPlayer] = list(state.players)
                vps = []
                colors = []
                for seat_idx, p in enumerate(seat_players):
                    vp = int(state.player_state[pkey(seat_idx, "ACTUAL_VICTORY_POINTS")])
                    vps.append(vp)
                    colors.append(p.color)

                ranks = competition_ranks(vps)
                table_mean = statistics.mean(vps)
                vp_std = statistics.pstdev(vps)
                vp_min = min(vps)
                vp_max = max(vps)
                vp_range = vp_max - vp_min
                vp_gini = gini(vps)

                has_winner = winner_color is not None
                winner_color_name = winner_color.value if has_winner else "NONE"
                winner_agent = code_by_color[winner_color] if has_winner else "NONE"

                g_writer.writerow(
                    {
                        "lineup_id": lineup_id,
                        "lineup": lineup_name,
                        "game_in_lineup": game_in_lineup,
                        "global_game_id": global_game_id,
                        "seed": seed,
                        "has_winner": int(has_winner),
                        "winner_color": winner_color_name,
                        "winner_agent": winner_agent,
                        "num_turns": num_turns,
                        "game_duration_s": round(game_duration_s, 6),
                        "vp_std": round(vp_std, 6),
                        "vp_min": vp_min,
                        "vp_max": vp_max,
                        "vp_range": vp_range,
                        "vp_gini": round(vp_gini, 6),
                    }
                )

                for seat_idx, p in enumerate(seat_players):
                    color = p.color
                    vp = vps[seat_idx]
                    rank = ranks[seat_idx]
                    public_vp = int(state.player_state[pkey(seat_idx, "VICTORY_POINTS")])
                    roads_built = MAX_ROADS - int(state.player_state[pkey(seat_idx, "ROADS_AVAILABLE")])
                    settlements_built = MAX_SETTLEMENTS - int(
                        state.player_state[pkey(seat_idx, "SETTLEMENTS_AVAILABLE")]
                    )
                    cities_built = MAX_CITIES - int(state.player_state[pkey(seat_idx, "CITIES_AVAILABLE")])

                    has_longest_road = bool(state.player_state[pkey(seat_idx, "HAS_ROAD")])
                    has_largest_army = bool(state.player_state[pkey(seat_idx, "HAS_ARMY")])
                    longest_road_length = int(state.player_state[pkey(seat_idx, "LONGEST_ROAD_LENGTH")])
                    knights_played = int(state.player_state[pkey(seat_idx, "PLAYED_KNIGHT")])
                    wood_in_hand = int(state.player_state[pkey(seat_idx, "WOOD_IN_HAND")])
                    brick_in_hand = int(state.player_state[pkey(seat_idx, "BRICK_IN_HAND")])
                    sheep_in_hand = int(state.player_state[pkey(seat_idx, "SHEEP_IN_HAND")])
                    wheat_in_hand = int(state.player_state[pkey(seat_idx, "WHEAT_IN_HAND")])
                    ore_in_hand = int(state.player_state[pkey(seat_idx, "ORE_IN_HAND")])
                    total_resources_in_hand = (
                        wood_in_hand + brick_in_hand + sheep_in_hand + wheat_in_hand + ore_in_hand
                    )

                    avg_decide_ms = 1000.0 * p.total_decide_s / p.decide_calls if p.decide_calls else 0.0

                    p_writer.writerow(
                        {
                            "lineup_id": lineup_id,
                            "lineup": lineup_name,
                            "game_in_lineup": game_in_lineup,
                            "global_game_id": global_game_id,
                            "seed": seed,
                            "input_slot": input_slot_by_color[color],
                            "seat_index": seat_idx,
                            "color": color.value,
                            "agent": code_by_color[color],
                            "is_winner": int(color == winner_color),
                            "rank": rank,
                            "final_vp": vp,
                            "public_vp": public_vp,
                            "roads_built": roads_built,
                            "settlements_built": settlements_built,
                            "cities_built": cities_built,
                            "longest_road_length": longest_road_length,
                            "has_longest_road": int(has_longest_road),
                            "has_largest_army": int(has_largest_army),
                            "knights_played": knights_played,
                            "wood_in_hand": wood_in_hand,
                            "brick_in_hand": brick_in_hand,
                            "sheep_in_hand": sheep_in_hand,
                            "wheat_in_hand": wheat_in_hand,
                            "ore_in_hand": ore_in_hand,
                            "total_resources_in_hand": total_resources_in_hand,
                            "decide_calls": p.decide_calls,
                            "total_decide_s": round(p.total_decide_s, 6),
                            "avg_decide_ms": round(avg_decide_ms, 6),
                            "num_turns": num_turns,
                            "game_duration_s": round(game_duration_s, 6),
                            "vp_std_game": round(vp_std, 6),
                            "vp_min_game": vp_min,
                            "vp_range_game": vp_range,
                            "vp_gini_game": round(vp_gini, 6),
                            "vp_minus_table_mean": round(vp - table_mean, 6),
                        }
                    )

                if global_game_id % 25 == 0 or global_game_id == total_games:
                    print(f"  progress: {global_game_id}/{total_games} games")

    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "lineups": [",".join(l) for l in parsed_lineups],
        "num_games_per_lineup": args.num_games,
        "seed_start": args.seed_start,
        "vps_to_win": args.vps_to_win,
        "outputs": {
            "player_metrics": str(player_csv),
            "game_metrics": str(game_csv),
        },
    }
    meta_json.write_text(json.dumps(meta, indent=2))

    print(f"Saved: {player_csv}")
    print(f"Saved: {game_csv}")
    print(f"Saved: {meta_json}")


if __name__ == "__main__":
    run()
