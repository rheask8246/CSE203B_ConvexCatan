#!/usr/bin/env bash
set -euo pipefail

# End-to-end runner for:
# 1) Option 2 full-game evaluations (one output folder per lineup)
# 2) Lambda sweep experiments and plots (screenshot setup)
#
# Usage:
#   bash scripts/run_option2_and_lambda.sh
#
# Optional overrides (examples):
#   PYTHON_BIN="conda run -p /opt/anaconda3 --no-capture-output python" bash scripts/run_option2_and_lambda.sh
#   EVAL_NUM_GAMES=150 bash scripts/run_option2_and_lambda.sh
#   LAMBDA_SEED_COUNTS="300" bash scripts/run_option2_and_lambda.sh

PYTHON_BIN=${PYTHON_BIN:-python}
EVAL_NUM_GAMES=${EVAL_NUM_GAMES:-100}
EVAL_SEED_START=${EVAL_SEED_START:-1}
EVAL_VPS_TO_WIN=${EVAL_VPS_TO_WIN:-10}

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

mkdir -p "${EVAL_BASE_OUTDIR}" "${LAMBDA_BASE_OUTDIR}"

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
  ${PYTHON_BIN} evaluate.py \
    --num-games "${EVAL_NUM_GAMES}" \
    --seed-start "${EVAL_SEED_START}" \
    --vps-to-win "${EVAL_VPS_TO_WIN}" \
    --outdir "${outdir}" \
    --lineup "${lineup}"

  echo "--- [${slug}] analyze ---"
  ${PYTHON_BIN} analyze_results.py --input-dir "${outdir}"

  echo "--- [${slug}] visualize ---"
  ${PYTHON_BIN} visualize_results.py --input-dir "${outdir}"

done

echo "=== Running lambda sweep experiments ==="
for num_seeds in ${LAMBDA_SEED_COUNTS}; do
  outdir="${LAMBDA_BASE_OUTDIR}/lambda_${num_seeds}"

  echo "--- [lambda_${num_seeds}] sweep ---"
  ${PYTHON_BIN} lambda_sweep.py \
    --num-seeds "${num_seeds}" \
    --seed-start "${LAMBDA_SEED_START}" \
    --num-players "${LAMBDA_NUM_PLAYERS}" \
    --lambda-start "${LAMBDA_START}" \
    --lambda-end "${LAMBDA_END}" \
    --lambda-count "${LAMBDA_COUNT}" \
    --outdir "${outdir}"

  echo "--- [lambda_${num_seeds}] visualize ---"
  ${PYTHON_BIN} visualize_lambda_sweep.py \
    --input-dir "${outdir}" \
    --target-lambda "${LAMBDA_TARGET}"

done

echo "=== All experiments complete ==="
echo "Results saved under: ${BASE_OUTDIR}"
