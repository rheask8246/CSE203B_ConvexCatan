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

## Visualizing games

**Turn-by-turn (terminal):** Run a single game and see VP after each action:
```bash
python run_simulation.py --watch
# or
python watch_game.py
```

## Convex agent

Fair resource allocation via LP: maximin production across all players + penalty for resource imbalance. Acts to equalize every player including itself.

**Usage:**
```bash
python run_simulation.py --players CONVEX,R,R,R --num 100
```

**Behavior:**
- Re-solves LP each turn when it has build options (not just at initial placement)
- Initial: 2 settlements + 2 roads by LP
- Mid-game: picks best settlement/city/road from LP scores
- Non-build: ROLL, then END_TURN, then dev card

See [Catanatron docs](https://docs.catanatron.com/advanced/editor) for game state and action types.
