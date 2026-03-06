#!/usr/bin/env python3
"""
Run Catan simulations via catanatron-play (progress bar, stats tables, etc.).
Use --watch to visualize a single game turn-by-turn.

Usage:
    python run_simulation.py --players R,R,R,R --num 100
    python run_simulation.py --players CONVEX,R,R,R --num 100
    python run_simulation.py --watch
"""

import argparse
import subprocess
import sys

AGENTS_FILE = "agents/players.py"


def main():
    parser = argparse.ArgumentParser(description="Run Catan simulations")
    parser.add_argument("--players", default="R,R,R,R", help="Comma-separated: R, W, or custom (CONVEX, etc.)")
    parser.add_argument("--num", type=int, default=100, help="Number of games")
    parser.add_argument("--watch", action="store_true", help="Run 1 game with turn-by-turn VP output")
    args = parser.parse_args()

    if args.watch:
        from watch_game import run
        run(max_turns=500, verbose=True)
        return

    cmd = [
        "catanatron-play",
        f"--players={args.players}",
        f"--num={args.num}",
        f"--code={AGENTS_FILE}",
    ]
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
