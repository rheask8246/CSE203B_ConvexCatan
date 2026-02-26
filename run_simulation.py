#!/usr/bin/env python3
"""
Run Catan simulations via catanatron-play (progress bar, stats tables, etc.).
Evaluation metrics can be added later.

Usage:
    python run_simulation.py --players R,R,R,R --num 100
    python run_simulation.py --players CONVEX,R,R,R --num 100
"""

import argparse
import subprocess
import sys

# Path to agents file (for custom agents like CONVEX)
AGENTS_FILE = "agents/players.py"


def main():
    parser = argparse.ArgumentParser(description="Run Catan simulations")
    parser.add_argument("--players", default="R,R,R,R", help="Comma-separated: R, W, or custom (CONVEX, etc.)")
    parser.add_argument("--num", type=int, default=100, help="Number of games")
    args = parser.parse_args()

    cmd = [
        "catanatron-play",
        f"--players={args.players}",
        f"--num={args.num}",
        f"--code={AGENTS_FILE}",
    ]
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
