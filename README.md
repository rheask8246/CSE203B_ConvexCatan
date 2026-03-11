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
- Last 10 games (turn order, number of turns, VP, winner)
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

## Evaluation Pipeline

Use these scripts to compare agents with reproducible seeds.

### Available Agents

**Custom Agents:**

- `CONVEX` - Fair resource allocation via LP (maximin + balance penalty)
- `GREEDY` - Pip score + resource diversification heuristic
- `MCTS` - Monte Carlo Tree Search (10 simulations default)
- `AB` - Alpha-Beta minimax search (depth 2 default)
- `VALUE` - Heuristic-based value function
- `WR` - Weighted random (prefers cities > settlements > dev cards)

**Built-in Agents:**

- `R` - Pure random (baseline)
- `W` - Catanatron's weighted random

### 1) Run experiments

```bash
python evaluate.py --num-games 300 --outdir results
# custom scenarios:
python evaluate.py \
  --lineup R,R,R,R \
  --lineup GREEDY,R,R,R \
  --lineup CONVEX,R,R,R \
  --lineup MCTS,AB,VALUE,WR \
  --num-games 300 \
  --outdir results
```

Outputs:

- `player_metrics.csv` (one row per player per game)
- `game_metrics.csv` (one row per game)
- `metadata.json`

### 2) Aggregate metrics + confidence intervals

```bash
python analyze_results.py --input-dir results
```

Outputs in `results/analysis`:

- `agent_summary.csv`
- `lineup_summary.csv`
- `seat_summary.csv`
- `pairwise_vp_diff.csv`
- `game_summary.csv`

### 3) Generate plots

```bash
python visualize_results.py --input-dir results
```

Outputs in `results/plots`:

- `win_rate_ci.png`
- `final_vp_violin.png`
- `turns_box.png`
- `resources_held_box.png`
<!-- - `fairness_gini_by_agent.png` (commented out - misleading for game-level metric) -->
- `fairness_gini_by_lineup.png`
- `pairwise_vp_diff_heatmap.png`
- `pareto_fairness_winrate_compute.png`
- `turn_order_effect_win_rate.png`
