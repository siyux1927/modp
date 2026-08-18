"""Stage 2 -- collect each teacher's opinion of the student's trajectories.

Teacher-forcing forward only: no generation, no KV cache growth, no sampling.  One
teacher is loaded, run over every trajectory, and freed before the next is loaded, so
peak memory is a single model's inference footprint.  An 8B teacher in bf16 needs
~17 GB and fits on a 24 GB card alongside a batch of 8 x 512.

Alignment: `logits[:, t]` predicts `input_ids[:, t+1]`, so the supervised window for a
row with prompt length P and total length N is ``t in [P-1, N-2]``, giving ``N - P``
predictions -- exactly the completion tokens.

Only the top-K entries are stored.  What is dropped is recorded as `tail_mass`, so
downstream fusion can reconstruct a normalised distribution rather than silently
renormalising over a truncated support.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from .data.store import read_table, teacher_schema, write_table


@dataclass
class TeacherConfig:
    model: str
    role: str
    rollout_path: str
    out_path: str
    k: int = 64
    batch_size: int = 8
    temperature: float = 1.0
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    device: str = "cuda"


@torch.no_grad()
def collect_teacher(cfg: TeacherConfig) -> Path:
    from transformers import AutoModelForCausalLM

    rollouts = read_table(cfg.rollout_path)
    order = sorted(rollouts, key=lambda t: len(rollouts[t]["input_ids"]))

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model,
        torch_dtype=getattr(torch, cfg.dtype),
        attn_implementation=cfg.attn_implementation,
        device_map=cfg.device,
    ).eval()

    rows: list[dict] = []
    for start in range(0, len(order), cfg.batch_size):
        chunk = [rollouts[t] for t in order[start : start + cfg.batch_size]]
        rows.extend(_process_batch(model, chunk, cfg))
        if start % (cfg.batch_size * 50) == 0:
            done = min(start + cfg.batch_size, len(order))
            print(f"[teacher:{cfg.role}] {done}/{len(order)}", flush=True)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rows.sort(key=lambda r: r["traj_id"])
    path = write_table(cfg.out_path, rows, teacher_schema())
    print(f"[teacher:{cfg.role}] wrote {len(rows)} rows -> {path}")
    return path


def _process_batch(model, chunk: list[dict], cfg: TeacherConfig) -> list[dict]:
    device = next(model.parameters()).device
    pad_id = getattr(model.config, "pad_token_id", None) or 0
    lens = [len(c["input_ids"]) for c in chunk]
    tmax = max(lens)

    ids = torch.full((len(chunk), tmax), pad_id, dtype=torch.long)
    attn = torch.zeros((len(chunk), tmax), dtype=torch.long)
    for i, c in enumerate(chunk):
        ids[i, : lens[i]] = torch.tensor(c["input_ids"], dtype=torch.long)
        attn[i, : lens[i]] = 1
    ids, attn = ids.to(device), attn.to(device)

    logits = model(input_ids=ids, attention_mask=attn).logits[:, :-1]
    if cfg.temperature != 1.0:
        logits = logits / cfg.temperature
    logprobs = F.log_softmax(logits.float(), dim=-1)

    topk = logprobs.topk(cfg.k, dim=-1)
    tail = (1.0 - topk.values.exp().sum(-1)).clamp_min(0.0)
    own = logprobs.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)

    out = []
    for i, c in enumerate(chunk):
        p, n = int(c["prompt_len"]), lens[i]
        lo, hi = p - 1, n - 1  # prediction indices covering the completion
        n_pred = hi - lo
        if n_pred <= 0:
            continue
        out.append(
            {
                "traj_id": int(c["traj_id"]),
                "n_pred": n_pred,
                "k": cfg.k,
                "topk_ids": topk.indices[i, lo:hi].reshape(-1).to(torch.int32).tolist(),
                "topk_logprobs": topk.values[i, lo:hi].reshape(-1).cpu().tolist(),
                "tail_mass": tail[i, lo:hi].cpu().tolist(),
                # how unsurprising the student's own completion was to this teacher;
                # this is the signal the `confidence` router consumes
                "mean_logprob": float(own[i, lo:hi].mean()),
            }
        )
    return out
