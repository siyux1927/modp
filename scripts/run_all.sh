#!/usr/bin/env bash
# The full matrix on one GPU. Stage 1 and 2 run once; every arm reuses their output.
#
# Wall-clock on a single RTX 4090, reference config (3 domains x 2000 prompts x 4
# samples = 24k trajectories, 512 new tokens, K=64):
#     stage 1  on-policy rollout        ~50 min
#     stage 1  off-policy rollout       ~70 min   (only needed for arm A1)
#     stage 2  3 teachers x 2 corpora   ~4.5 h
#     stage 3  9 training runs          ~6 h
#     eval     10 checkpoints           ~2 h
#   total ~15 GPU-hours per seed.
set -euo pipefail
cd "$(dirname "$0")/.."

STUDENT=${STUDENT:-Qwen/Qwen3-1.7B-Base}
OFFPOLICY_GEN=${OFFPOLICY_GEN:-Qwen/Qwen3-8B}
SEED=${SEED:-0}
PER_DOMAIN=${PER_DOMAIN:-2000}

declare -A TEACHERS=(
  [math]=Qwen/Qwen3-4B
  [instruct]=Qwen/Qwen3-8B
  [general]=Qwen/Qwen2.5-Coder-7B-Instruct
)

# ---- stage 1 -----------------------------------------------------------------
mopd rollout --model "$STUDENT" --out data/onpolicy/rollout.parquet \
  --per-domain "$PER_DOMAIN" --n-samples 4 --seed "$SEED"

mopd rollout --model "$OFFPOLICY_GEN" --out data/offpolicy/rollout.parquet \
  --per-domain "$PER_DOMAIN" --n-samples 4 --seed "$SEED"

# ---- stage 2 -----------------------------------------------------------------
for corpus in onpolicy offpolicy; do
  for role in "${!TEACHERS[@]}"; do
    out="data/$corpus/teacher-$role.parquet"
    [[ -f "$out" ]] && { echo "skip $out"; continue; }
    mopd teacher --model "${TEACHERS[$role]}" --role "$role" \
      --rollout "data/$corpus/rollout.parquet" --out "$out" -k 64 --batch-size 8
  done
done

# ---- stage 3 -----------------------------------------------------------------
python configs/arms.py
for cfg in configs/a[1-5]_*.json configs/abl_*.json; do
  name=$(basename "$cfg" .json)
  echo "=== $name (seed $SEED) ==="
  mopd train --config "$cfg" --set "seed=$SEED" "out_dir=\"runs/$name-s$SEED\""
done

# ---- eval --------------------------------------------------------------------
# A0: the untrained student, for the baseline row.
mopd eval --model "$STUDENT" --out "results/a0_base"
for cfg in configs/a[1-5]_*.json configs/abl_*.json; do
  name=$(basename "$cfg" .json)
  mopd eval --model "runs/$name-s$SEED/final" --out "results/$name-s$SEED"
done

python scripts/collect_results.py results/ > results/table.md
cat results/table.md
