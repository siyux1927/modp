# MOPD — Multi-Teacher On-Policy Distillation

A minimal reproduction and research scaffold: multiple teachers supervise a student on
trajectories the **student itself** generated, with per-state teacher routing and
verifier-grounded rewards.

Designed to run on **one 24 GB GPU** for **~$30–40 of rented time**, with **no sandbox,
no container, and no environment simulator**.

---

## The claim under test

> **H1.** Multi-teacher distillation on the student's own on-policy trajectories, with
> confidence-based teacher routing, beats (a) single-teacher on-policy distillation,
> (b) multi-teacher off-policy distillation, and (c) equal-weight multi-teacher fusion.

Four arms, one shared rollout corpus, one shared set of teacher signals. If H1's 2×2
holds, the project has a result; the Agent / long-horizon extension is stage two, not
the entry ticket.

## Two design decisions that set the cost

**Everything is Qwen.** Student and teachers share one tokenizer and one 151k vocab, so
token-level KL works directly — no vocabulary mapping, no response-level fallback. This
is the single choice that turns a two-month project into a two-week one.

**No model is ever co-resident with another.** Three stages hand off through Parquet:

```
stage 1  mopd rollout    student generates, verifier scores   -> rollout.parquet
stage 2  mopd teacher    one teacher at a time scores those   -> teacher-<role>.parquet
stage 3  mopd train      fuse, distil                         -> checkpoint
```

Teacher size is therefore bounded by *inference* memory, not by what is left over after
the optimiser state. An 8B teacher runs comfortably on the same card that trains the
student.

## No sandbox: three pure-function verifiers

Code execution is the only thing that forces a distillation project to build and
maintain a sandbox. Dropping the code domain removes that entire axis:

| Domain | Data | Verifier | Executes anything? |
|---|---|---|---|
| `math` | GSM8K | HF [`math-verify`](https://github.com/huggingface/Math-Verify) (SymPy) | no |
| `ifeval` | [google/IFEval](https://huggingface.co/datasets/google/IFEval) | Google's official constraint registry, 25 checkers | no |
| `tool` | BFCL v3 AST split | AST comparison against the reference call | no |

They probe close-to-orthogonal abilities — symbolic reasoning, constraint satisfaction,
structured API emission — which is what makes them a real test of whether multi-teacher
fusion fuses *heterogeneous* skills. The cross-domain retention metric (does distilling
math damage instruction-following?) is where multi-teacher should show its value most
clearly.

## Install

```bash
pip install -e ".[pipeline,verify,eval]"
```

CPU-only development (numerics, verifiers, tests) needs just `pip install -e ".[dev]"`.

## Run

```bash
bash scripts/smoke.sh
```

~10 minutes on one card, all four stages at toy scale, using Qwen3-0.6B for everything.
Run this before spending a real rollout pass.

```bash
bash scripts/run_all.sh
```

The full matrix. ~15 GPU-hours per seed; three seeds is the recommended minimum, since
at this model scale a one-point difference is noise.

## The experiment matrix

| Arm | On-policy | Teachers | Router | Tests |
|---|---|---|---|---|
| A0 | — | — | — | untrained student baseline |
| A1 | ✗ | 3 | uniform | classical multi-teacher KD |
| A2 | ✓ | 1 | — | single-teacher GKD |
| A3 | ✓ | 3 | uniform | does routing matter at all? |
| **A4** | ✓ | 3 | confidence | **the method** |
| A5 | ✓ | 3 | oracle | how much headroom routing has |

Ablations (each is A4 with exactly one field changed, so any delta is attributable):
`beta ∈ {0, 0.5, 1}`, `fusion_mode ∈ {arithmetic, geometric}`, `failure_aware`,
per-token vs per-trajectory routing.

Evaluation runs through EleutherAI `lm-eval` (`gsm8k`, `ifeval`) so numbers are
comparable to published leaderboards; BFCL is scored in-process with the same AST
verifier used for training rewards.

## Where the research questions actually live

**`src/mopd/router.py`** — how much each teacher gets to say. The `confidence` router
weights teachers by the mean log-probability they assign to the student's *own*
completion: a teacher that finds the student's trajectory unsurprising is one that can
model this state well. It needs no labels and no verifier, so unlike `oracle` it is
deployable. Training logs `routing_entropy`; a confidence router that collapses to 0 has
silently become A2, and one that sits at 1 has become A3 — either way the arm is not
testing what it claims to.

**`src/mopd/fusion.py`** — how the teachers' distributions combine. `copy.md` specifies
the arithmetic mean of probabilities, and that is the default, but a linear opinion pool
can place mass on a compromise token *no individual teacher supports*. Log-linear
(geometric) pooling is conjunctive and cannot. This is implemented as a one-field switch
and is probably the most promising small original contribution here.

**`src/mopd/loss.py`** — `failure_aware_weights` replaces the hand-tuned
`if tool_error: w *= 4` heuristic with two bounded, measurable gates: the student's own
normalised predictive entropy, and whether the verifier rejected the trajectory.

## Two things worth knowing before you read results

**Top-K truncation is an approximation, and its error is not negligible.** Storing full
teacher logits would need ~3.7 TB; top-64 plus a tail-mass scalar needs ~5 GB. The mass
outside the union support is carried as an explicit "other" bucket rather than silently
renormalised away, but `tests/test_loss.py::test_truncation_error_shrinks_as_k_grows`
shows the residual error at K=8 is real. That is why K is swept rather than assumed.

**`beta` runs the direction you may not expect.** Following TRL and the GKD paper,
`beta=0` is *forward* KL(teacher‖student) — mass-covering — and `beta=1` is *reverse*
KL(student‖teacher) — mode-seeking. For a 1.7B student under an 8B teacher, the reverse
end is usually the right one. `tests/test_loss.py::test_matches_trl_generalized_jsd`
pins our sparse implementation to TRL's dense one at every beta, endpoints included.

## Deliberate deviation from TRL

The training loop does not use TRL's `GKDTrainer`, despite it being the reference
implementation of on-policy distillation. `GKDTrainer` keeps the teacher resident in the
training process and generates on-policy samples with HF `generate` — both of which are
exactly what the staged design exists to avoid. The *loss* follows TRL's convention
exactly and is tested against it, so numbers stay comparable.

## Cost

| | |
|---|---|
| GPU | one RTX 4090 or L4, 24 GB. No multi-GPU, so no DeepSpeed/FSDP config |
| Spot price | $0.34–0.44/h (RunPod, Vast.ai) |
| Full matrix, 3 seeds | ~45–70 GPU-hours → **$30–40** |
| Disk | ~14 GB of teacher signals per corpus |

## Testing

```bash
pytest
```

56 tests, no GPU and no network required. The ones that matter:

- `test_fusion.py` — sparse top-K fusion is *exact* against a dense reference when
  K = vocab_size, for both pooling modes.
- `test_alignment.py` — the index arithmetic. Off-by-one here produces a plausible loss
  curve and a model that learned nothing, so the convention
  (`logits[:, t]` predicts `input_ids[:, t+1]`; supervised window is `[P-1, N-2]`) is
  pinned down explicitly.
- `test_loss.py` — endpoint identities, TRL agreement, truncation convergence.
- `test_verifiers.py` — including that the tool-call parser never evaluates what it
  parses.

## Layout

```
src/mopd/
  rollout.py        stage 1: vLLM generation + verifier scoring
  teacher.py        stage 2: teacher-forcing forward, top-K logprobs
  train.py          stage 3: fuse + distil
  router.py         teacher weighting          <- research variable
  fusion.py         sparse distribution fusion <- research variable
  loss.py           generalised JSD + failure-aware weighting
  evaluate.py       lm-eval wrapper + local BFCL scoring
  data/registry.py  GSM8K / IFEval / BFCL loaders
  data/store.py     Parquet schema, dataset, collation
  verifiers/        three pure-function verifiers
configs/arms.py     the experiment matrix, as data
scripts/            smoke.sh, run_all.sh, collect_results.py
```
