"""
Loader for catanatron-play --code. Imports agents from agents.players so they
can be pickled when using --db (required for web GUI).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents.players import ConvexAgent
from catanatron.cli.cli_players import register_cli_player

register_cli_player("CONVEX", ConvexAgent)
