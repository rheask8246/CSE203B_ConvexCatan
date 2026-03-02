"""
Agent registry for catanatron-play.

Each agent must be a Player subclass with a decide(game, playable_actions) method
that returns one of the playable_actions.
"""

from catanatron import Player
import os
import importlib.util

_here = os.path.dirname(__file__)
_greedy_path = os.path.join(_here, "greedy_agent.py")

_spec = importlib.util.spec_from_file_location("greedy_agent", _greedy_path)
_greedy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_greedy)

GreedyAgent = _greedy.GreedyAgent

try:
    from catanatron.cli import register_cli_player
except ImportError:
    register_cli_player = None  # catanatron-play not available (PyPI package)

# --- Add your agents here ---

# Example agent stub
class ConvexAgent(Player):
    """Placeholder for convex optimization agent. Implement decide()."""

    def decide(self, game, playable_actions):
        # TODO: implement convex optimization logic
        return next(iter(playable_actions))


# Register for catanatron-play (--code=agents/players.py)
if register_cli_player:
    register_cli_player("CONVEX", ConvexAgent)
    register_cli_player("GREEDY", GreedyAgent)
    