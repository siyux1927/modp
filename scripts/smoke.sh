#!/usr/bin/env bash
# End-to-end smoke test: ~10 minutes on one 24 GB card, ~4 GB of weights downloaded.
# Runs every stage with tiny settings so a wiring bug surfaces before you spend a
# real rollout pass on it. Uses the same 0.6B model for all three teacher roles --
# the point is that the plumbing works, not that anything is learned.
set -euo pipefail
cd "$(dirname "$0")/.."

SMALL=Qwen/Qwen3-0.6B
OUT=data/smoke
RUN=runs/smoke

mopd rollout \
  --model "$SMALL" --out "$OUT/rollout.parquet" \
  --domains math ifeval --per-domain 16 --n-samples 2 \
  --max-new-tokens 96 --temperature 1.0

for role in math instruct general; do
  mopd teacher \
    --model "$SMALL" --role "$role" \
    --rollout "$OUT/rollout.parquet" \
    --out "$OUT/teacher-$role.parquet" \
    -k 32 --batch-size 4
done

python - <<PY
import json, pathlib
cfg = {
  "student": "$SMALL",
  "rollout_path": "$OUT/rollout.parquet",
  "teacher_paths": {r: "$OUT/teacher-%s.parquet" % r for r in ["math","instruct","general"]},
  "out_dir": "$RUN",
  "router": "confidence", "fusion_mode": "arithmetic", "beta": 0.5,
  "failure_aware": True, "epochs": 1, "batch_size": 2, "grad_accum": 2,
  "max_len": 512, "log_every": 1,
}
pathlib.Path("configs/smoke.json").write_text(json.dumps(cfg, indent=2))
PY

mopd train --config configs/smoke.json

echo
echo "smoke OK -- checkpoint at $RUN/final, loss curve at $RUN/log.jsonl"
