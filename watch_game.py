#!/usr/bin/env python3
"""
Run a single game and print board state + VP after each action.
Useful for verifying the convex agent re-solves each turn.

Usage:
    python watch_game.py
    python watch_game.py --max-turns 50
"""

import argparse
import sys

sys.path.insert(0, ".")

from catanatron import Game
from catanatron.models.player import Color
from catanatron.models.enums import ActionType
from catanatron.state_functions import get_actual_victory_points, get_player_freqdeck

from agents.players import ConvexAgent

# Use RandomPlayer from catanatron for other slots
try:
    from catanatron import RandomPlayer
except ImportError:
    from catanatron.models.player import RandomPlayer


def vp_summary(game):
    """VP per player."""
    return "  ".join(
        f"{c.value}:{get_actual_victory_points(game.state, c)}"
        for c in game.state.colors
    )


def resources_summary(game):
    """Resources per player: W B S Wh O (wood, brick, sheep, wheat, ore)."""
    lines = []
    for c in game.state.colors:
        deck = get_player_freqdeck(game.state, c)
        res_str = " ".join(f"{deck[i]}" for i in range(5))
        total = sum(deck)
        lines.append(f"  {c.value}: {res_str}  (total={total})")
    return "\n".join(lines)


def run(max_turns=500, verbose=True):
    players = [
        ConvexAgent(Color.RED),
        RandomPlayer(Color.BLUE),
        RandomPlayer(Color.ORANGE),
        RandomPlayer(Color.WHITE),
    ]
    game = Game(players)
    turn = 0

    if verbose:
        print("Starting game. Convex agent re-solves LP on each turn when it has build options.")
        print("Resources: W=wood B=brick S=sheep Wh=wheat O=ore\n")

    while game.winning_color() is None and turn < max_turns:
        state = game.state
        color = state.current_color()
        action = state.current_player().decide(game, game.playable_actions)

        game.execute(action)
        turn += 1

        if verbose:
            vps = vp_summary(game)
            action_str = str(action.action_type.value)
            if action.action_type in (ActionType.BUILD_SETTLEMENT, ActionType.BUILD_CITY):
                action_str += f" @{action.value}"
            elif action.action_type == ActionType.BUILD_ROAD:
                action_str += f" {action.value}"
            print(f"Turn {turn:4}  {color.value:6}  {action_str:30}  VP: {vps}")
            print("Resources (W B S Wh O):")
            print(resources_summary(game))

    winner = game.winning_color()
    if verbose:
        print(f"\nWinner: {winner.value if winner else 'None (max turns)'}")
    return winner


def main():
    parser = argparse.ArgumentParser(description="Watch a single game with turn-by-turn output")
    parser.add_argument("--max-turns", type=int, default=500, help="Stop after N turns")
    parser.add_argument("-q", "--quiet", action="store_true", help="Only print winner")
    args = parser.parse_args()

    run(max_turns=args.max_turns, verbose=not args.quiet)


if __name__ == "__main__":
    main()
