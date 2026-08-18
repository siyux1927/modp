"""Prompt sources.

Each loader returns a list of plain dicts with a common shape:

    uid       str, globally unique
    domain    "math" | "ifeval" | "tool"
    messages  list[{"role", "content"}], ready for a chat template
    ...       whatever the domain's verifier needs as a reference

All three are public HF datasets with permissive licences and no execution
requirement.  Nothing here downloads model weights or spins up an environment.
"""

from __future__ import annotations

import json
from typing import Any

from ..verifiers.math_verifier import gsm8k_gold

MATH_SYSTEM = (
    "Solve the problem step by step. Put your final answer inside \\boxed{}."
)
TOOL_SYSTEM = (
    "You are a function-calling assistant. Reply with a JSON list of calls, e.g. "
    '[{"name": "fn", "arguments": {"arg": "value"}}]. Emit nothing else.'
)

DOMAINS = ("math", "ifeval", "tool")


def load_math(split: str = "train", limit: int | None = None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return [
        {
            "uid": f"gsm8k-{split}-{i}",
            "domain": "math",
            "messages": [
                {"role": "system", "content": MATH_SYSTEM},
                {"role": "user", "content": r["question"]},
            ],
            "answer": gsm8k_gold(r["answer"]),
        }
        for i, r in enumerate(ds)
    ]


def load_ifeval(split: str = "train", limit: int | None = None,
                holdout: int = 150, seed: int = 0) -> list[dict[str, Any]]:
    """IFEval ships a single 541-row split; we carve a deterministic holdout."""
    import random

    from datasets import load_dataset

    ds = load_dataset("google/IFEval", split="train")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)  # noqa: S311 -- reproducible split, not a secret
    idx = idx[holdout:] if split == "train" else idx[:holdout]
    if limit:
        idx = idx[:limit]
    out = []
    for i in idx:
        r = ds[i]
        out.append(
            {
                "uid": f"ifeval-{r['key']}",
                "domain": "ifeval",
                "messages": [{"role": "user", "content": r["prompt"]}],
                "prompt": r["prompt"],
                "instruction_id_list": list(r["instruction_id_list"]),
                "kwargs": [dict(k) for k in r["kwargs"]],
            }
        )
    return out


def load_tool(split: str = "train", limit: int | None = None,
              category: str = "simple", holdout: int = 150,
              seed: int = 0) -> list[dict[str, Any]]:
    """BFCL v3 AST categories. Reference answers only -- nothing is invoked."""
    import random

    from datasets import load_dataset

    ds = load_dataset(
        "gorilla-llm/Berkeley-Function-Calling-Leaderboard", split=category
    )
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)  # noqa: S311 -- reproducible split, not a secret
    idx = idx[holdout:] if split == "train" else idx[:holdout]
    if limit:
        idx = idx[:limit]

    out = []
    for i in idx:
        r = ds[i]
        funcs = _as_obj(r.get("function"))
        answer = _as_obj(r.get("ground_truth") or r.get("possible_answer"))
        question = _as_obj(r.get("question"))
        user = _flatten_question(question)
        if not user or not answer:
            continue
        out.append(
            {
                "uid": f"bfcl-{category}-{r.get('id', i)}",
                "domain": "tool",
                "messages": [
                    {"role": "system",
                     "content": TOOL_SYSTEM + "\n\nAvailable functions:\n"
                     + json.dumps(funcs, ensure_ascii=False)},
                    {"role": "user", "content": user},
                ],
                "function": funcs,
                "reference_calls": answer if isinstance(answer, list) else [answer],
                "test_category": category,
            }
        )
    return out


LOADERS = {"math": load_math, "ifeval": load_ifeval, "tool": load_tool}


def load_domains(domains: list[str], split: str = "train",
                 per_domain: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for d in domains:
        if d not in LOADERS:
            raise ValueError(f"unknown domain {d!r}; have {sorted(LOADERS)}")
        records.extend(LOADERS[d](split=split, limit=per_domain))
    return records


def _as_obj(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _flatten_question(question) -> str:
    """BFCL stores the user turn as a nested list of chat messages."""
    if isinstance(question, str):
        return question
    if isinstance(question, dict):
        return str(question.get("content", ""))
    if isinstance(question, list):
        parts = [_flatten_question(q) for q in question]
        return "\n".join(p for p in parts if p)
    return ""
