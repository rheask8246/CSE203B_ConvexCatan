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

## LP Lambda Sweep (Initial Placement Fairness)

This experiment measures LP fairness/efficiency directly on initial-placement expected production.

### 1) Run lambda sweep

```bash
python lambda_sweep.py \
  --num-seeds 300 \
  --lambda-start 0 \
  --lambda-end 2 \
  --lambda-count 25 \
  --num-players 3 \
  --outdir results/lambda_sweep
```

Outputs in `results/lambda_sweep`:

- `lambda_sweep_game_metrics.csv`
- `lambda_sweep_player_metrics.csv`
- `lambda_sweep_duals.csv`
- `lambda_sweep_summary.csv`
- `metadata.json`

Metrics include:

- Production Gini across players
- Max-min production ratio
- Total expected production
- Mean Shannon entropy (resource diversity)
- Capacity-constraint dual values (shadow prices)

### 2) Plot lambda sweep results

```bash
python visualize_lambda_sweep.py --input-dir results/lambda_sweep --target-lambda 0.5
```

Outputs in `results/lambda_sweep/plots`:

- `gini_vs_lambda.png` - Gini vs lambda (line + 95% CI)
- `entropy_vs_lambda.png` - Mean Shannon entropy vs lambda
- `total_expected_production_vs_lambda.png` - Efficiency vs lambda
- `pareto_total_production_vs_gini.png` - Pareto curve (one point per lambda)
- `entropy_violin_by_method.png` - ConvexAgent vs Random vs WeightedRandom
- `max_min_ratio_by_method.png` - Bar chart at lambda=0.5
- `production_gap_hist_by_method.png` - Distribution of (max - min) per game
- `dual_heatmap_capacity.png` - Shadow prices (node strip)
- `dual_heatmap_board_layout.png` - Shadow prices on board topology

## ConvexAgent vs Selfish Opponents (Full Games)

Plays **CONVEX(lambda), R, SELFISH** for each lambda and each selfish agent (GREEDY, AB, MCTS, VALUE, WR). Tracks **Gini of expected production after initial placement** (what the LP directly optimizes), eliminating the confound of in-game play quality.

### 1) Run evaluation

Plays CONVEX, R, and one selfish agent — 100 games per (lambda, selfish_agent). All 5 selfish agents (GREEDY, AB, MCTS, VALUE, WR) are included by default.

```bash
# Quick (5 games per combo, ~2 min)
python evaluate_convex_sweep.py --num-games 5 --lambda-count 10 --outdir results/convex_sweep_quick

# Full (100 games per combo, ~30-60 min)
python evaluate_convex_sweep.py \
  --num-games 100 \
  --lambda-start 0 \
  --lambda-end 2 \
  --lambda-count 10 \
  --outdir results/convex_sweep
```

### 2) Generate plots (overlaid by opponent)

```bash
python visualize_convex_sweep.py --input-dir results/convex_sweep
# or for quick run:
python visualize_convex_sweep.py --input-dir results/convex_sweep_quick
```

Outputs in `results/convex_sweep/plots`:

- `prod_gini_vs_lambda_by_opponent.png` - Expected production Gini vs lambda (overlaid by opponent)
- `prod_range_vs_lambda_by_opponent.png` - Production range vs lambda by opponent
- `prod_gini_by_opponent_at_lambda05.png` - Bar chart: fairness by opponent at lambda=0.5
- `resources_by_opponent.png` - Final resources held: Convex vs R vs Selfish

## Commands to test the pipeline

**Quick test (few minutes):**
```bash
source .venv/bin/activate
pip install -r requirements.txt   # if not already installed

# Lambda sweep (initial placement only): 20 seeds, 10 lambda points
python lambda_sweep.py --num-seeds 20 --lambda-count 10 --outdir results/lambda_quick
python visualize_lambda_sweep.py --input-dir results/lambda_quick --target-lambda 0.5

# Convex vs selfish (full games): 5 games per combo, 10 lambda, all 5 selfish agents
python evaluate_convex_sweep.py --num-games 5 --lambda-count 10 --outdir results/convex_sweep_quick
python visualize_convex_sweep.py --input-dir results/convex_sweep_quick
```

**Full Convex vs selfish (100 games per combo, ~30-60 min):**
```bash
python evaluate_convex_sweep.py \
  --num-games 100 \
  --lambda-start 0 \
  --lambda-end 2 \
  --lambda-count 10 \
  --outdir results/convex_sweep

python visualize_convex_sweep.py --input-dir results/convex_sweep
```

**Full lambda sweep (initial placement only, 200 seeds):**
```bash
python lambda_sweep.py \
  --num-seeds 200 \
  --lambda-start 0 \
  --lambda-end 2 \
  --lambda-count 25 \
  --num-players 3 \
  --outdir results/lambda_sweep

python visualize_lambda_sweep.py --input-dir results/lambda_sweep --target-lambda 0.5
```

**Run everything (evaluations + lambda sweep):**
```bash
bash run_all.sh
```
