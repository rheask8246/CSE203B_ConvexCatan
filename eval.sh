#!/usr/bin/env bash
set -euo pipefail

# End-to-end runner for:
# 1) Option 2 full-game evaluations (one output folder per lineup)
# 2) Lambda sweep experiments and plots (screenshot setup)
# 3) Convex-vs-selfish full-game lambda sweep + plots
#
# Usage:
#   bash eval.sh
#
# Optional overrides (examples):
#   PYTHON_BIN="conda run -p /opt/anaconda3 --no-capture-output python" bash eval.sh
#   EVAL_NUM_GAMES=150 bash eval.sh
#   LAMBDA_SEED_COUNTS="300" bash eval.sh

PYTHON_BIN=${PYTHON_BIN:-python}
EVAL_NUM_GAMES=${EVAL_NUM_GAMES:-100}
EVAL_SEED_START=${EVAL_SEED_START:-1}
EVAL_VPS_TO_WIN=${EVAL_VPS_TO_WIN:-10}

CONVEX_SWEEP_NUM_GAMES=${CONVEX_SWEEP_NUM_GAMES:-100}
CONVEX_SWEEP_SEED_START=${CONVEX_SWEEP_SEED_START:-1}
CONVEX_SWEEP_LAMBDA_START=${CONVEX_SWEEP_LAMBDA_START:-0}
CONVEX_SWEEP_LAMBDA_END=${CONVEX_SWEEP_LAMBDA_END:-2}
CONVEX_SWEEP_LAMBDA_COUNT=${CONVEX_SWEEP_LAMBDA_COUNT:-10}
CONVEX_SWEEP_SELFISH_AGENTS=${CONVEX_SWEEP_SELFISH_AGENTS:-"GREEDY AB MCTS VALUE WR"}

LAMBDA_NUM_PLAYERS=${LAMBDA_NUM_PLAYERS:-3}
LAMBDA_START=${LAMBDA_START:-0}
LAMBDA_END=${LAMBDA_END:-2}
LAMBDA_COUNT=${LAMBDA_COUNT:-25}
LAMBDA_SEED_START=${LAMBDA_SEED_START:-1}
LAMBDA_TARGET=${LAMBDA_TARGET:-0.5}

LAMBDA_SEED_COUNTS=${LAMBDA_SEED_COUNTS:-"200 500"}

STAMP=$(date +"%Y%m%d_%H%M%S")
BASE_OUTDIR="results/batch_${STAMP}"
EVAL_BASE_OUTDIR="${BASE_OUTDIR}/eval"
LAMBDA_BASE_OUTDIR="${BASE_OUTDIR}/lambda"
CONVEX_SWEEP_OUTDIR="${BASE_OUTDIR}/convex_sweep"

mkdir -p "${EVAL_BASE_OUTDIR}" "${LAMBDA_BASE_OUTDIR}" "${CONVEX_SWEEP_OUTDIR}"

echo "=== Output root: ${BASE_OUTDIR} ==="

echo "=== Running Option 2 lineup experiments ==="
LINEUPS=(
  "convex_random_greedy|CONVEX,R,GREEDY"
  "convex_random_mcts|CONVEX,R,MCTS"
  "convex_random_ab|CONVEX,R,AB"
  "convex_random_value|CONVEX,R,VALUE"
  "convex_random_wr|CONVEX,R,WR"
)

for entry in "${LINEUPS[@]}"; do
  IFS='|' read -r slug lineup <<< "${entry}"
  outdir="${EVAL_BASE_OUTDIR}/${slug}"

  echo "--- [${slug}] evaluate ---"
  ${PYTHON_BIN} -m evaluation.evaluate \
    --num-games "${EVAL_NUM_GAMES}" \
    --seed-start "${EVAL_SEED_START}" \
    --vps-to-win "${EVAL_VPS_TO_WIN}" \
    --outdir "${outdir}" \
    --lineup "${lineup}"

  echo "--- [${slug}] analyze ---"
  ${PYTHON_BIN} -m evaluation.analyze_results --input-dir "${outdir}"

  echo "--- [${slug}] visualize ---"
  ${PYTHON_BIN} -m evaluation.visualize_results --input-dir "${outdir}"

done

echo "=== Running lambda sweep experiments ==="
for num_seeds in ${LAMBDA_SEED_COUNTS}; do
  outdir="${LAMBDA_BASE_OUTDIR}/lambda_${num_seeds}"

  echo "--- [lambda_${num_seeds}] sweep ---"
  ${PYTHON_BIN} -m evaluation.lambda_sweep \
    --num-seeds "${num_seeds}" \
    --seed-start "${LAMBDA_SEED_START}" \
    --num-players "${LAMBDA_NUM_PLAYERS}" \
    --lambda-start "${LAMBDA_START}" \
    --lambda-end "${LAMBDA_END}" \
    --lambda-count "${LAMBDA_COUNT}" \
    --outdir "${outdir}"

  echo "--- [lambda_${num_seeds}] visualize ---"
  ${PYTHON_BIN} -m evaluation.visualize_lambda_sweep \
    --input-dir "${outdir}" \
    --target-lambda "${LAMBDA_TARGET}"

done

echo "=== Running convex sweep experiments ==="
${PYTHON_BIN} -m evaluation.evaluate_convex_sweep \
  --num-games "${CONVEX_SWEEP_NUM_GAMES}" \
  --seed-start "${CONVEX_SWEEP_SEED_START}" \
  --lambda-start "${CONVEX_SWEEP_LAMBDA_START}" \
  --lambda-end "${CONVEX_SWEEP_LAMBDA_END}" \
  --lambda-count "${CONVEX_SWEEP_LAMBDA_COUNT}" \
  --selfish-agents ${CONVEX_SWEEP_SELFISH_AGENTS} \
  --outdir "${CONVEX_SWEEP_OUTDIR}"

${PYTHON_BIN} -m evaluation.visualize_convex_sweep \
  --input-dir "${CONVEX_SWEEP_OUTDIR}"

echo "=== All experiments complete ==="
echo "Results saved under: ${BASE_OUTDIR}"
