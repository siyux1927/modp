# Next steps

Written as a handoff. Everything below is executable on a fresh machine with a GPU;
nothing depends on state that exists only on the laptop this was written on.

Status at handoff: **numerics verified, GPU path unverified.** 56 CPU tests pass and
ruff is clean, but no stage has ever talked to a real model. Task group P0 exists to
close exactly that gap, and until P0-4 is green no result from this repo means anything.

---

## 0. Device migration

### What is in git

Source, tests, configs, scripts, docs. That is the whole project — everything else is
regenerable.

### What is deliberately *not* in git

| Path | Size | How to get it back |
|---|---|---|
| `data/` | ~30 GB | Regenerate: P1-1, P1-2. Never commit; never rsync — regenerating is faster than copying and removes any doubt about which config produced it |
| `runs/` | ~7 GB/arm | Regenerate: P1-3 |
| `results/` | small | Regenerate: P1-4. **Do commit these once you have real numbers** — see P2-1 |
| `.venv/` | — | Recreate below |
| HF model cache | ~40 GB | Re-downloads on first use |

### Bootstrap on the new box

```bash
git clone <repo-url> modp && cd modp
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[pipeline,verify,eval,dev]"
pytest                      # must be 56 passed before you touch a GPU
```

Then set these once, in `~/.bashrc` on the GPU box:

```bash
export HF_HOME=/workspace/hf                 # put the cache on the big disk, not /
export HF_HUB_ENABLE_HF_TRANSFER=1           # ~3x faster model pulls
export TOKENIZERS_PARALLELISM=false          # silences a warning per dataloader worker
export WANDB_PROJECT=mopd                    # optional; JSONL logging works regardless
```

On a rented spot box, `/workspace` (or whatever the persistent volume is) needs
**~90 GB**: 40 GB models + 30 GB teacher signals + 20 GB checkpoints. This is the most
common way the first full run dies.

### Sanity check that the machine is what you think it is

```bash
python -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory/1e9)"
```

Under 22 GB usable means the reference config will OOM in stage 3 — go to P0-5.

---

## P0 — Make the GPU path real

Nothing here produces a result. All of it produces *confidence that a result would be
real.* Budget half a day and about 2 GPU-hours.

### P0-1 · Verify the three dataset loaders against live HF schemas

**Why this is first.** `load_math` and `load_ifeval` were written against documented
field names and are low-risk. `load_tool` is not: BFCL's HF repo has reorganised its
splits and column names more than once, and `question` is a nested list whose depth has
changed between v2 and v3. The loader has defensive handling but it is guessing.

```bash
python - <<'PY'
from mopd.data.registry import load_math, load_ifeval, load_tool
for name, fn in [("math", load_math), ("ifeval", load_ifeval), ("tool", load_tool)]:
    try:
        rs = fn(limit=3)
        print(f"\n=== {name}: {len(rs)} records ===")
        print(rs[0])
    except Exception as e:
        print(f"\n=== {name}: FAILED {type(e).__name__}: {e} ===")
PY
```

**Acceptance:** each domain yields records whose `messages` render sensibly and whose
reference fields are populated. For `tool`, `reference_calls` must be a non-empty list
of `{name: {arg: [values]}}` — an empty list means the column name changed and every
reward will silently be 0.

**If `load_tool` fails:** inspect the actual columns
(`load_dataset(...).column_names`) and fix `mopd/data/registry.py:load_tool`. Do not
work around it by dropping the tool domain — the cross-domain retention metric is where
this project's result is most likely to live.

**Fallback if BFCL proves too unstable to be worth the time:** substitute
`Salesforce/xlam-function-calling-60k` (same task shape, single flat `answers` JSON
column, far more stable schema) and keep the same AST verifier. Cheaper than fighting
BFCL, at the cost of leaderboard comparability. Decide within an hour, do not sink a
day into this.

---

### P0-2 · Verify the three verifiers score real model output, not just fixtures

`tests/test_verifiers.py` proves the *parsers* work. It does not prove the reward is
calibrated on text an actual model emits.

```bash
mopd rollout --model Qwen/Qwen3-0.6B --out data/probe/rollout.parquet \
  --domains math ifeval tool --per-domain 32 --n-samples 1 --max-new-tokens 256
```

Read the `pass@1` line the rollout prints per domain, then eyeball 10 completions
against their scores:

```bash
python - <<'PY'
from mopd.data.store import read_table
rows = list(read_table("data/probe/rollout.parquet").values())
for r in rows[:10]:
    print(f"\n--- {r['domain']} reward={r['reward']} ---\n{r['completion'][:400]}")
PY
```

**Acceptance:** pass@1 is in a *plausible* band for a 0.6B model — roughly 5–25% on
math, 15–40% on ifeval, 5–30% on tool. And critically, the manual read agrees with the
scores.

**The failure this catches:** a verifier that returns 0.0 for everything looks exactly
like a weak student. A domain stuck at 0.000 is a bug, not a result. A domain at 1.000
is also a bug.

---

### P0-3 · Verify teacher collection round-trips

```bash
mopd teacher --model Qwen/Qwen3-1.7B --role probe \
  --rollout data/probe/rollout.parquet --out data/probe/teacher-probe.parquet \
  -k 64 --batch-size 4
```

```bash
python - <<'PY'
import torch, numpy as np
from mopd.data.store import read_table
t = read_table("data/probe/teacher-probe.parquet")
r = read_table("data/probe/rollout.parquet")
tid = next(iter(t))
row, roll = t[tid], r[tid]
k, n = row["k"], row["n_pred"]
assert n == len(roll["input_ids"]) - roll["prompt_len"], "n_pred != completion length"
lp = np.asarray(row["topk_logprobs"], dtype=np.float32).reshape(n, k)
mass = np.exp(lp).sum(-1)
print(f"n_pred={n}  top-{k} mass: min={mass.min():.3f} mean={mass.mean():.3f}")
print(f"tail_mass mean={np.mean(row['tail_mass']):.4f}  mean_logprob={row['mean_logprob']:.3f}")
assert (mass <= 1.0001).all() and (mass > 0.3).all()
PY
```

**Acceptance:** `n_pred` equals the completion length exactly (this is the alignment
invariant `tests/test_alignment.py` pins on synthetic data, now checked on real data);
top-64 mass averages **> 0.9**; `mean_logprob` is a plausible negative number, roughly
−0.3 to −3.

**If top-64 mass is below 0.8**, K=64 is too aggressive for this student's entropy.
Raise K to 128 and re-cost the disk budget before starting P1.

---

### P0-4 · Full smoke, end to end

```bash
bash scripts/smoke.sh
```

**Acceptance, all four:**

1. It completes without exception.
2. `runs/smoke/log.jsonl` shows `loss` **decreasing** over the run. Flat loss with a
   healthy `grad_norm` means the target is wrong, not that learning is slow.
3. `route_entropy` is strictly between 0 and 1. At exactly 1 the confidence router has
   degenerated into the uniform ablation; at 0 it has degenerated into single-teacher.
   Either way arms A3/A4/A2 would be measuring the same thing and the matrix is void.
4. The `w/math`, `w/instruct`, `w/general` columns are not all identical.

**This gate is not optional.** Every downstream number is conditioned on it.

---

### P0-5 · Fit the memory budget to the actual card

Run one training step at the real config and watch peak memory:

```bash
mopd train --config configs/a4_onpolicy_confidence.json \
  --set 'rollout_path="data/probe/rollout.parquet"' \
        'teacher_paths={"probe":"data/probe/teacher-probe.parquet"}' \
        'out_dir="runs/memprobe"' 'log_every=1'
```

Escalate in this order if it OOMs — each step costs less than the one after it:

1. `grad_accum` 4 → 8 and `batch_size` 4 → 2 (free; same effective batch)
2. `optim: "adamw_8bit"` (needs `pip install bitsandbytes`; ~4 GB saved, negligible
   quality cost at this scale)
3. `max_len` 1280 → 1024 (truncates long math CoT; note it in the writeup)
4. student `Qwen3-1.7B-Base` → `Qwen3-0.6B-Base` (**last resort** — read the risk note
   in P2-3 before doing this)

Record what you settled on in `configs/arms.py:BASE` and commit it. The config that
produced the numbers has to be in git.

---

## P1 — Run the matrix

~15 GPU-hours per seed, three seeds. Budget two days wall-clock and about $25.

### P1-1 · On-policy rollout corpus

```bash
mopd rollout --model Qwen/Qwen3-1.7B-Base --out data/onpolicy/rollout.parquet \
  --per-domain 2000 --n-samples 4 --seed 0
```

**Acceptance:** ~24k trajectories; per-domain pass@1 printed and recorded. Write those
three numbers down — they are the A0 baseline row and the denominator for every claim
of improvement.

**Watch for:** a pass@1 near 0 on any domain makes `failure_aware_weights` degenerate
(everything is a failure, so the gate is a constant). If a domain lands under 3%, say so
explicitly in the writeup rather than quietly reporting the arm.

### P1-2 · Off-policy corpus + all teacher signals

```bash
mopd rollout --model Qwen/Qwen3-8B --out data/offpolicy/rollout.parquet \
  --per-domain 2000 --n-samples 4 --seed 0
# then stage 2 for 3 roles x 2 corpora
```

`scripts/run_all.sh` does both and skips any shard already on disk, so it is safe to
re-run after a spot preemption. ~4.5 h, the single longest block. Run it detached
(`tmux` / `nohup`) — a dropped SSH session should not cost you the pass.

### P1-3 · Train all arms and ablations

```bash
for s in 0 1 2; do SEED=$s bash scripts/run_all.sh; done
```

Three seeds is the minimum, not a nicety: at 1.7B the arm-to-arm gaps this project is
chasing are plausibly the same size as seed noise, and a single-seed table cannot tell
you which you have.

**Per-run acceptance:** loss decreases; `route_entropy` stays in (0.1, 0.95) for A4;
final checkpoint loads.

### P1-4 · Evaluate and collect

```bash
python scripts/collect_results.py results/ > results/table.md
```

**Commit `results/table.md` and the raw `results/**/results*.json`.** They are small,
and they are the only artifacts of a 45-GPU-hour spend.

---

## P2 — Turn runs into a result

### P2-1 · Read the 2×2 before doing anything else

H1 needs four numbers:

| | single teacher | three teachers |
|---|---|---|
| **off-policy** | (not run) | A1 |
| **on-policy** | A2 | A4 |

- **A4 > A2** — multi-teacher helps *given* on-policy.
- **A4 > A1** — on-policy helps *given* multi-teacher.
- **A4 > A3** — the routing, specifically, is what helps. This is the one most likely
  to come out null, and a null here is a publishable finding: it would say confidence
  routing is not worth its complexity, which is useful to know.
- **A5 − A4** — how much headroom a better router could buy. If A5 ≈ A4, routing is
  saturated and there is no point building a learned router (P3-2).

Report every gap with its seed spread. A 0.8-point gap on ±1.2 is not a result, and
writing it up as one is the single easiest way to waste this whole exercise.

### P2-2 · Cross-domain retention

The metric most likely to show multi-teacher's real value, and it needs one extra run
per condition rather than a new pipeline: train A2 with `single_index` pointing at the
math teacher only, then measure IFEval and BFCL.

**Hypothesis:** single-teacher distillation on a math-skewed teacher degrades the other
two domains; multi-teacher does not. If that holds, it is a cleaner story than a
half-point of GSM8K.

Add as `configs/abl_single_math_only.json` (`router: "single"`, `single_index: 0`).

### P2-3 · Decide whether the scale is honest

If after three seeds the arm differences are inside noise on every domain, the answer is
**not** "run more seeds." It is one of:

- **Scale the student up** to 4B. Costs roughly 2.5× and needs `adamw_8bit` plus
  `max_len` 1024 on a 24 GB card; still under $100.
- **Report the null.** "Multi-teacher routing does not measurably beat uniform fusion at
  1.7B on these three domains, over three seeds, with these effect sizes" is a real
  finding, and it is what the pre-registered matrix was for.

Deciding this *after* seeing the numbers and then reframing is the failure mode. Write
down now which way you will go.

### P2-4 · The ablations, in priority order

1. **`fusion_mode: geometric`** — the most likely original contribution here. Arithmetic
   pooling can put mass on a compromise token no teacher supports; log-linear cannot. If
   geometric wins, that is the paper.
2. **`beta ∈ {0, 0.5, 1}`** — expect reverse (β=1) to win for a small student. If forward
   KL wins instead, that contradicts the GKD result and needs investigating, not
   reporting.
3. **`K ∈ {8, 64, 256}`** — pure engineering, but the storage/fidelity curve is
   genuinely useful to anyone reproducing this. Needs re-running stage 2 per K; do it on
   a 4k-trajectory subset, not the full corpus.
4. **`failure_aware: false`** — cheapest ablation in the set, run it regardless.
5. **`router: confidence_token`** — only worth it if A4 > A3 in P2-1. If per-trajectory
   routing bought nothing, per-token will not either.

---

## P3 — Stage two, only after H1 resolves

Do not start any of this before P2-1. Each is a project.

- **P3-1 · Iterative on-policy.** Currently one rollout pass; MOPD's premise is a loop.
  Add `mopd loop` alternating stages 1–3 with `iteration` incrementing (the column is
  already in the rollout schema). The real question: does iteration 2 beat iteration 1
  by more than the extra compute would buy by just training longer on iteration 1?
- **P3-2 · Learned router.** A small head over trajectory features predicting which
  teacher's signal most reduces held-out loss. Only worth building if P2-1 shows
  A5 ≫ A4 — otherwise there is no headroom to capture.
- **P3-3 · `L = L_MOPD + λ·L_GRPO`.** The joint objective from `copy.md`. This is where
  migrating to `verl` finally pays for its configuration cost. Needs P3-1 first, since
  GRPO wants fresh on-policy data every step.
- **P3-4 · Agent / long-horizon.** The original motivation, and the largest jump: it
  reintroduces environments and therefore reintroduces the sandbox this design removed.
  Cheapest credible entry is a multi-turn tool-calling setting with a mock environment
  (τ-bench-style), *not* a real one.

---

## Standing risks

| Risk | Detection | Response |
|---|---|---|
| BFCL schema drift | P0-1 | Switch to xlam-function-calling-60k |
| Verifier silently returns all-zero | P0-2 pass@1 read | Fix before any training |
| Arm differences inside seed noise | P2-1 spread | P2-3 — scale up or report the null |
| Confidence router collapses | `route_entropy` in logs | Tune `router_temperature`; if it will not hold, A4 ≡ A3 and say so |
| Spot preemption mid-stage-2 | Missing shard | `run_all.sh` skips completed shards; just re-run |
| Top-64 too lossy | P0-3 mass check | K=128, re-budget disk |

## Things I would not spend time on

- Migrating to `verl` before P3-3. Its configuration cost is real and buys nothing at
  single-GPU scale.
- Multi-GPU. The whole design is built so one card is enough; distributed training adds
  a failure surface for no result.
- Adding a code-execution domain. It reintroduces the sandbox, and IFEval already gives
  a cleanly orthogonal second ability.
- Prompt engineering the teachers. Tempting, unfalsifiable at this budget, and it
  contaminates every arm comparison at once.
