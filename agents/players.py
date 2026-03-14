"""
Agent registry for catanatron-play.

ConvexAgent is defined in agents/convex_agent.py.
This file imports it so evaluate.py can use `from agents.players import ConvexAgent`.
"""

import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import importlib.util

from catanatron import Player
from catanatron.models.enums import ActionType, ActionPrompt

_here = os.path.dirname(__file__)
_greedy_path = os.path.join(_here, "greedy_agent.py")
_spec = importlib.util.spec_from_file_location("greedy_agent", _greedy_path)
_greedy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_greedy)
GreedyAgent = _greedy.GreedyAgent

from agents.mcts import MCTSPlayer
from agents.minimax import AlphaBetaPlayer
from agents.value import ValueFunctionPlayer
from agents.weighted_random import WeightedRandomPlayer

# Single definition — do not duplicate here.
from agents.convex_agent import ConvexAgent

try:
    from catanatron.cli import register_cli_player
    register_cli_player("CONVEX", ConvexAgent)
    register_cli_player("GREEDY", GreedyAgent)
    register_cli_player("MCTS",   MCTSPlayer)
    register_cli_player("AB",     AlphaBetaPlayer)
    register_cli_player("VALUE",  ValueFunctionPlayer)
    register_cli_player("WR",     WeightedRandomPlayer)
except (ImportError, Exception):
    pass