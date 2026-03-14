"""Run reproducible Catan evaluations and export per-player/per-game metrics.

Examples:
    python evaluate.py --num-games 300 --outdir results/exp1
    python evaluate.py --lineup R,R,R --lineup GREEDY,R,R --lineup CONVEX,R,R
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
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


MAX_ROADS = 15
MAX_SETTLEMENTS = 5
MAX_CITIES = 4
COLORS = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]

# ---------------------------------------------------------------------------
# Player-state key helper
# ---------------------------------------------------------------------------

def pkey(idx: int, key: str) -> str:
    """
    Build the player-state dict key used by catanatron.
    catanatron stores player state as f"P{turn_order}_{field}".
    """
    return f"P{idx}_{key}"


def safe_int(state, key: str, default: int = 0) -> int:
    """Read an integer from state.player_state, returning default if missing."""
    try:
        return int(state.player_state[key])
    except (KeyError, TypeError, ValueError):
        return default


def safe_bool(state, key: str) -> bool:
    """Read a boolean from state.player_state, returning False if missing."""
    try:
        return bool(state.player_state[key])
    except (KeyError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Timing wrapper — NOT a dataclass to avoid conflict with Player.__init__
# ---------------------------------------------------------------------------

class TimedPlayer(Player):
    """Wrap a player and track total decision latency."""

    def __init__(self, inner: Player):
        # Delegate color and is_bot to parent; do NOT use @dataclass.
        super().__init__(inner.color, is_bot=inner.is_bot)
        self.inner = inner
        self.total_decide_s: float = 0.0
        self.decide_calls: int = 0

    def decide(self, game, playable_actions):
        t0 = time.perf_counter()
        action = self.inner.decide(game, playable_actions)
        self.total_decide_s += time.perf_counter() - t0
        self.decide_calls += 1
        return action

    def reset_state(self):
        if hasattr(self.inner, "reset_state"):
            self.inner.reset_state()
        self.total_decide_s = 0.0
        self.decide_calls = 0


# ---------------------------------------------------------------------------
# Fairness helpers
# ---------------------------------------------------------------------------

def gini(values: Sequence[float]) -> float:
    """Gini coefficient of a sequence of non-negative floats."""
    arr = [float(v) for v in values]
    n = len(arr)
    if n == 0:
        return float("nan")
    mean_v = sum(arr) / n
    if mean_v == 0:
        return 0.0
    diff_sum = sum(abs(i - j) for i in arr for j in arr)
    return diff_sum / (2 * n * n * mean_v)


def competition_ranks(values: Sequence[float]) -> List[int]:
    """Standard competition ranking: 1 = highest. Ties share the same rank."""
    unique = sorted(set(values), reverse=True)
    rank_map = {v: i + 1 for i, v in enumerate(unique)}
    return [rank_map[v] for v in values]


def expected_production_gini_from_state(game) -> float:
    """
    Gini of expected resource production across players using current buildings.
    Uses settlements (weight 1) and cities (weight 2).
    This is the primary fairness metric the LP optimises.
    """
    try:
        from agents.convex_solver import expected_production_gini
        g, _ = expected_production_gini(game)
        return float(g)
    except Exception:
        return float("nan")


def production_max_min_ratio(game) -> float:
    """max(v_p) / min(v_p) of expected production; 1.0 = perfectly fair."""
    try:
        from agents.convex_solver import expected_production_gini
        import numpy as np
        _, v = expected_production_gini(game)
        if v is None or len(v) == 0:
            return float("nan")
        vmin = float(np.min(v))
        if vmin < 1e-12:
            return float("inf")
        return float(np.max(v) / vmin)
    except Exception:
        return float("nan")


def turns_until_equalized(production_history: List[float], threshold: float = 0.10) -> int:
    """
    First turn at which Gini of expected production drops below `threshold`
    and stays there. Returns -1 if never equalized.
    """
    for t, g in enumerate(production_history):
        if g <= threshold:
            return t
    return -1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Catan agents with reproducible seeds")
    parser.add_argument(
        "--lineup",
        action="append",
        default=[],
        help="One 3-player lineup, comma-separated (e.g. CONVEX,R,R). Repeat for multiple.",
    )
    parser.add_argument("--num-games", type=int, default=200)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--vps-to-win", type=int, default=10)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument(
        "--track-production",
        action="store_true",
        default=False,
        help="Record per-turn production Gini (slower; needed for equalization metric).",
    )
    return parser.parse_args()


def default_lineups() -> List[str]:
    return ["R,R,R", "GREEDY,R,R", "CONVEX,R,R"]


def get_agent_factory(code: str) -> Callable[[Color], Player]:
    code = code.strip().upper()
    factories: Dict[str, Callable[[Color], Player]] = {
        "R": RandomPlayer,
        "GREEDY": GreedyAgent,
        "CONVEX": ConvexAgent,
        "MCTS": MCTSPlayer,
        "AB": AlphaBetaPlayer,
        "VALUE": ValueFunctionPlayer,
        "WR": CustomWeightedRandomPlayer,
    }
    if code not in factories:
        raise ValueError(f"Unknown agent code '{code}'. Supported: {', '.join(sorted(factories))}")
    return factories[code]


def parse_lineup(lineup: str) -> List[str]:
    parts = [p.strip().upper() for p in lineup.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"Lineup must contain exactly 3 players: '{lineup}'")
    for part in parts:
        get_agent_factory(part)
    return parts


def prepare_outdir(outdir: Path | None) -> Path:
    out = outdir or (Path("results") / f"eval_{time.strftime('%Y%m%d_%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# Production tracking via game callback
# ---------------------------------------------------------------------------

class ProductionTracker:
    """
    Hooks into a game to record per-turn production Gini.
    Call .attach(game) before game.play(); results in .history after.
    """

    def __init__(self):
        self.history: List[float] = []   # Gini at each turn
        self._game = None

    def attach(self, game: Game):
        self._game = game
        # Monkey-patch: wrap game.play_turn if available, else record at end.
        # Simplest reliable approach: record after each action by wrapping
        # the game's action-application method.
        original_apply = game.state.apply_action if hasattr(game.state, "apply_action") else None
        if original_apply is not None:
            tracker = self

            def patched_apply(action):
                result = original_apply(action)
                try:
                    g = expected_production_gini_from_state(tracker._game)
                    tracker.history.append(g)
                except Exception:
                    pass
                return result

            game.state.apply_action = patched_apply


# ---------------------------------------------------------------------------
# Main run loop
# ---------------------------------------------------------------------------

def run() -> None:
    args = parse_args()
    lineups = args.lineup if args.lineup else default_lineups()
    parsed_lineups = [parse_lineup(x) for x in lineups]

    if any("CONVEX" in l for l in parsed_lineups):
        try:
            import cvxpy  # noqa: F401
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "CONVEX lineup requested but cvxpy is not installed."
            ) from exc

    outdir = prepare_outdir(args.outdir)
    player_csv = outdir / "player_metrics.csv"
    game_csv = outdir / "game_metrics.csv"
    meta_json = outdir / "metadata.json"

    player_fields = [
        "lineup_id", "lineup", "game_in_lineup", "global_game_id", "seed",
        "input_slot", "turn_order", "color", "agent",
        "is_winner", "rank", "final_vp", "public_vp",
        "roads_built", "settlements_built", "cities_built",
        "longest_road_length", "has_longest_road", "has_largest_army",
        "knights_played",
        "wood_in_hand", "brick_in_hand", "sheep_in_hand", "wheat_in_hand", "ore_in_hand",
        "total_resources_in_hand",
        "decide_calls", "total_decide_s", "avg_decide_ms",
        "num_turns", "game_duration_s",
        # VP fairness (game-level, repeated per player row for convenience)
        "vp_std_game", "vp_min_game", "vp_range_game", "vp_gini_game",
        "vp_minus_table_mean",
        # Production fairness (game-level)
        "prod_gini_final",        # Gini of expected production at game end
        "prod_max_min_ratio",     # max/min production ratio at game end
        "turns_until_equalized",  # first turn prod Gini < 0.10 (-1 if never)
    ]

    game_fields = [
        "lineup_id", "lineup", "game_in_lineup", "global_game_id", "seed",
        "has_winner", "winner_color", "winner_agent",
        "num_turns", "game_duration_s",
        "vp_std", "vp_min", "vp_max", "vp_range", "vp_gini",
        "prod_gini_final", "prod_max_min_ratio", "turns_until_equalized",
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

                # Build TimedPlayer wrappers; keep external references so we
                # can read timing after the game (catanatron may reorder state.players).
                timed: List[TimedPlayer] = []
                code_by_color: Dict[Color, str] = {}
                input_slot_by_color: Dict[Color, int] = {}

                for i, code in enumerate(lineup_codes):
                    color = COLORS[i]
                    inner = get_agent_factory(code)(color)
                    tp = TimedPlayer(inner)
                    timed.append(tp)
                    code_by_color[color] = code
                    input_slot_by_color[color] = i

                # Map color -> TimedPlayer for O(1) lookup after the game.
                timed_by_color: Dict[Color, TimedPlayer] = {tp.color: tp for tp in timed}

                game = Game(timed, seed=seed, vps_to_win=args.vps_to_win)

                # Optionally attach production tracker.
                tracker = ProductionTracker()
                if args.track_production:
                    tracker.attach(game)

                t0 = time.perf_counter()
                game.play()
                game_duration_s = time.perf_counter() - t0

                state = game.state
                winner_color = game.winning_color()
                num_turns = int(state.num_turns)

                # Turn order comes from state.players (ordered by seat).
                # We identify each player by color, not by object identity,
                # to avoid issues with catanatron internally copying players.
                seat_players = list(state.players)
                color_to_turn_order: Dict[Color, int] = {
                    p.color: i for i, p in enumerate(seat_players)
                }

                # Collect VP for all players.
                vps: List[float] = []
                ordered_colors: List[Color] = []
                for turn_order, p in enumerate(seat_players):
                    vp = safe_int(state, pkey(turn_order, "ACTUAL_VICTORY_POINTS"))
                    vps.append(float(vp))
                    ordered_colors.append(p.color)

                ranks = competition_ranks(vps)
                table_mean = statistics.mean(vps)
                vp_std = statistics.pstdev(vps)
                vp_min = min(vps)
                vp_max = max(vps)
                vp_range = vp_max - vp_min
                vp_gini_val = gini(vps)

                # Production fairness metrics.
                prod_gini_val = expected_production_gini_from_state(game)
                prod_mmr = production_max_min_ratio(game)
                equalized_turn = turns_until_equalized(tracker.history) if args.track_production else -1

                has_winner = winner_color is not None
                winner_color_name = winner_color.value if has_winner else "NONE"
                winner_agent = code_by_color[winner_color] if has_winner else "NONE"

                g_writer.writerow({
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
                    "vp_min": int(vp_min),
                    "vp_max": int(vp_max),
                    "vp_range": int(vp_range),
                    "vp_gini": round(vp_gini_val, 6),
                    "prod_gini_final": round(prod_gini_val, 6) if prod_gini_val == prod_gini_val else "",
                    "prod_max_min_ratio": round(prod_mmr, 6) if prod_mmr != float("inf") and prod_mmr == prod_mmr else "",
                    "turns_until_equalized": equalized_turn,
                })

                for turn_order, color in enumerate(ordered_colors):
                    vp = int(vps[turn_order])
                    rank = ranks[turn_order]

                    public_vp = safe_int(state, pkey(turn_order, "VICTORY_POINTS"))
                    roads_built = MAX_ROADS - safe_int(state, pkey(turn_order, "ROADS_AVAILABLE"), MAX_ROADS)
                    settlements_built = MAX_SETTLEMENTS - safe_int(state, pkey(turn_order, "SETTLEMENTS_AVAILABLE"), MAX_SETTLEMENTS)
                    cities_built = MAX_CITIES - safe_int(state, pkey(turn_order, "CITIES_AVAILABLE"), MAX_CITIES)
                    longest_road_length = safe_int(state, pkey(turn_order, "LONGEST_ROAD_LENGTH"))
                    has_longest_road = safe_bool(state, pkey(turn_order, "HAS_ROAD"))
                    has_largest_army = safe_bool(state, pkey(turn_order, "HAS_ARMY"))
                    knights_played = safe_int(state, pkey(turn_order, "PLAYED_KNIGHT"))
                    wood = safe_int(state, pkey(turn_order, "WOOD_IN_HAND"))
                    brick = safe_int(state, pkey(turn_order, "BRICK_IN_HAND"))
                    sheep = safe_int(state, pkey(turn_order, "SHEEP_IN_HAND"))
                    wheat = safe_int(state, pkey(turn_order, "WHEAT_IN_HAND"))
                    ore = safe_int(state, pkey(turn_order, "ORE_IN_HAND"))
                    total_res = wood + brick + sheep + wheat + ore

                    # Read timing from our external TimedPlayer reference by color.
                    tp = timed_by_color[color]
                    avg_decide_ms = (
                        1000.0 * tp.total_decide_s / tp.decide_calls
                        if tp.decide_calls > 0 else 0.0
                    )

                    p_writer.writerow({
                        "lineup_id": lineup_id,
                        "lineup": lineup_name,
                        "game_in_lineup": game_in_lineup,
                        "global_game_id": global_game_id,
                        "seed": seed,
                        "input_slot": input_slot_by_color[color],
                        "turn_order": turn_order,
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
                        "wood_in_hand": wood,
                        "brick_in_hand": brick,
                        "sheep_in_hand": sheep,
                        "wheat_in_hand": wheat,
                        "ore_in_hand": ore,
                        "total_resources_in_hand": total_res,
                        "decide_calls": tp.decide_calls,
                        "total_decide_s": round(tp.total_decide_s, 6),
                        "avg_decide_ms": round(avg_decide_ms, 6),
                        "num_turns": num_turns,
                        "game_duration_s": round(game_duration_s, 6),
                        "vp_std_game": round(vp_std, 6),
                        "vp_min_game": int(vp_min),
                        "vp_range_game": int(vp_range),
                        "vp_gini_game": round(vp_gini_val, 6),
                        "vp_minus_table_mean": round(vp - table_mean, 6),
                        "prod_gini_final": round(prod_gini_val, 6) if prod_gini_val == prod_gini_val else "",
                        "prod_max_min_ratio": round(prod_mmr, 6) if prod_mmr != float("inf") and prod_mmr == prod_mmr else "",
                        "turns_until_equalized": equalized_turn,
                    })

                if global_game_id % 20 == 0 or global_game_id == total_games:
                    print(f"  progress: {global_game_id}/{total_games} games")

    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "lineups": [",".join(l) for l in parsed_lineups],
        "num_games_per_lineup": args.num_games,
        "seed_start": args.seed_start,
        "vps_to_win": args.vps_to_win,
        "track_production": args.track_production,
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