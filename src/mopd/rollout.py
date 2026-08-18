"""Stage 1 -- generate trajectories and score them.

Only one model is resident here (the generator), and it is released before stage 2
starts.  That is the whole reason the pipeline is staged: teacher size is then bounded
by inference memory, not by training memory.

`source="student"` is the on-policy arm: sequences come from the model being trained,
so the teachers end up commenting on states the student actually visits.
`source="external"` is the off-policy control: the same prompts, but sequences drawn
from a fixed stronger generator, which is what classical KD does.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

from .data.registry import load_domains
from .data.store import rollout_schema, write_table
from .verifiers import score_batch


@dataclass
class RolloutConfig:
    model: str
    out_path: str
    domains: tuple[str, ...] = ("math", "ifeval", "tool")
    per_domain: int = 2000
    n_samples: int = 4
    max_prompt_tokens: int = 768
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 0.95
    seed: int = 0
    iteration: int = 0
    gpu_memory_utilization: float = 0.85
    source: str = "student"


def run_rollout(cfg: RolloutConfig) -> Path:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    records = load_domains(list(cfg.domains), split="train", per_domain=cfg.per_domain)
    tok = AutoTokenizer.from_pretrained(cfg.model)

    prompts, keep = [], []
    for r in records:
        text = tok.apply_chat_template(
            r["messages"], tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        if len(tok(text).input_ids) <= cfg.max_prompt_tokens:
            prompts.append(text)
            keep.append(r)
    print(f"[rollout] {len(keep)} prompts ({len(records) - len(keep)} dropped as too long)")

    llm = LLM(
        model=cfg.model,
        dtype="bfloat16",
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        enable_prefix_caching=True,
        max_model_len=cfg.max_prompt_tokens + cfg.max_new_tokens + 8,
        seed=cfg.seed,
    )
    params = SamplingParams(
        n=cfg.n_samples,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_tokens=cfg.max_new_tokens,
        seed=cfg.seed,
    )
    outputs = llm.generate(prompts, params)

    rows, flat_records, flat_texts = [], [], []
    traj_id = 0
    for rec, out in zip(keep, outputs, strict=True):
        for cand in out.outputs:
            rows.append(
                {
                    "traj_id": traj_id,
                    "uid": rec["uid"],
                    "domain": rec["domain"],
                    "prompt_len": len(out.prompt_token_ids),
                    # prompt ids come straight from vLLM, so no re-tokenisation drift
                    "input_ids": list(out.prompt_token_ids) + list(cand.token_ids),
                    "completion": cand.text,
                    "reward": 0.0,
                    "iteration": cfg.iteration,
                }
            )
            flat_records.append(rec)
            flat_texts.append(cand.text)
            traj_id += 1

    del llm
    gc.collect()
    _free_cuda()

    rewards = score_batch(flat_records, flat_texts)
    for row, rew in zip(rows, rewards, strict=True):
        row["reward"] = float(rew)

    by_domain: dict[str, list[float]] = {}
    for row in rows:
        by_domain.setdefault(row["domain"], []).append(row["reward"])
    for d, rs in sorted(by_domain.items()):
        print(f"[rollout] {d:8s} n={len(rs):6d} pass@1={sum(rs) / len(rs):.3f}")

    path = write_table(cfg.out_path, rows, rollout_schema())
    print(f"[rollout] wrote {len(rows)} trajectories -> {path}")
    return path


def _free_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
