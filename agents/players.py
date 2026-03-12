"""
Agent registry for catanatron-play.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catanatron import Player
import os
import importlib.util

_here = os.path.dirname(__file__)
_greedy_path = os.path.join(_here, "greedy_agent.py")

_spec = importlib.util.spec_from_file_location("greedy_agent", _greedy_path)
_greedy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_greedy)

GreedyAgent = _greedy.GreedyAgent
from catanatron.models.enums import ActionType, ActionPrompt

from pathlib import Path
from agents.mcts import MCTSPlayer
from agents.minimax import AlphaBetaPlayer
from agents.value import ValueFunctionPlayer
from agents.weighted_random import WeightedRandomPlayer

try:
    from catanatron.cli import register_cli_player
except ImportError:
    register_cli_player = None


class ConvexAgent(Player):
    """Fair resource allocation via convex LP (maximin + balance penalty)."""

    def decide(self, game, playable_actions):
        actions = list(playable_actions)
        state = game.state

        if state.is_initial_build_phase:
            return self._initial(actions, game)
        if state.current_prompt != ActionPrompt.PLAY_TURN:
            # If we are being asked to move the robber, choose a fairness-aligned
            # robber action rather than defaulting arbitrarily.
            robber_actions = [
                a for a in actions if a.action_type == ActionType.MOVE_ROBBER
            ]
            if robber_actions:
                from agents.convex_solver import score_robber_hexes_fairness

                scores = score_robber_hexes_fairness(game)
                return max(
                    robber_actions,
                    key=lambda a: scores.get(a.value[0], 0.0),
                )
            return actions[0]

        build = [a for a in actions if a.action_type in (
            ActionType.BUILD_SETTLEMENT, ActionType.BUILD_CITY, ActionType.BUILD_ROAD
        )]
        if not build:
            for at in (ActionType.ROLL, ActionType.END_TURN, ActionType.BUY_DEVELOPMENT_CARD):
                for a in actions:
                    if a.action_type == at:
                        return a
            return actions[0]

        from agents.convex_solver import solve_build, score_action

        node_scores, edge_scores, edge_to_idx = solve_build(game, self.color)
        return max(build, key=lambda a: score_action(a, node_scores, edge_scores, edge_to_idx))

    def _initial(self, actions, game):
        settle = [a for a in actions if a.action_type == ActionType.BUILD_SETTLEMENT]
        road = [a for a in actions if a.action_type == ActionType.BUILD_ROAD]

        if settle:
            from agents.convex_solver import solve_initial
            scores = solve_initial(game, self.color)
            return max(settle, key=lambda a: scores[a.value])

        if road:
            from agents.convex_solver import solve_initial
            scores = solve_initial(game, self.color)
            return max(road, key=lambda a: (scores[a.value[0]] + scores[a.value[1]]) / 2)

        return actions[0]


if register_cli_player:
    register_cli_player("CONVEX", ConvexAgent)
    register_cli_player("GREEDY", GreedyAgent)
    register_cli_player("MCTS", MCTSPlayer)
    register_cli_player("AB", AlphaBetaPlayer)
    register_cli_player("VALUE", ValueFunctionPlayer)
    register_cli_player("WR", WeightedRandomPlayer)
    