# CSE203B ConvexCatan

Simulation framework for evaluating convex optimization and baseline agents in Settlers of Catan. Uses [Catanatron](https://docs.catanatron.com/) for game logic.

## Setup

### Virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Note:** Install Catanatron from GitHub (not PyPI) to get `catanatron-play` with rich output. If `pip install -r requirements.txt` fails, try `pip install git+https://github.com/bcollazo/catanatron.git` directly.

### Quick test

```bash
python run_simulation.py --players R,R,R,R --num 100
```

**Use `run_simulation.py` as the main entry point.** It wraps `catanatron-play` and will be extended with evaluation metrics later.

## Running simulations

```bash
python run_simulation.py --players R,R,R,R --num 100
python run_simulation.py --players CONVEX,R,R,R --num 100
```

### catanatron-play (direct)

If you need to run the underlying CLI directly:

```bash
catanatron-play --num 100 --players R,R,R,R
catanatron-play --num 1000 --players R,R,W,W --code agents/players.py
catanatron-play --num 100 --players CONVEX,R,R,R --code agents/players.py
```

Includes:
- Progress bar
- Win distribution by player
- Last 10 games (seating, turns, VP, winner)
- Player summary (wins, avg VP, settlements, cities, roads, army, dev VP)
- Game summary (avg turns, duration)

**Built-in agents:** `R` (random), `W` (weighted random)

**Custom agents:** Add in `agents/players.py` and register with `register_cli_player`

See [Catanatron docs](https://docs.catanatron.com/advanced/editor) for game state and action types.
