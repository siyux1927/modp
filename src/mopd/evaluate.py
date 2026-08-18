"""Evaluation.

GSM8K and IFEval go through EleutherAI's `lm-eval` rather than a hand-rolled loop, so
the numbers are directly comparable to published results and none of the prompt
formatting or answer-extraction decisions are ours to get wrong.

BFCL has no lm-eval task, so the tool domain is scored in-process with the same AST
verifier used for training rewards -- still no execution.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

LM_EVAL_TASKS = {"math": "gsm8k", "ifeval": "ifeval"}


def run_lm_eval(
    model_path: str,
    tasks: list[str],
    out_dir: str | Path,
    batch_size: str = "auto",
    limit: int | None = None,
    max_model_len: int = 2048,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    args = [
        sys.executable, "-m", "lm_eval",
        "--model", "vllm",
        "--model_args",
        (f"pretrained={model_path},dtype=bfloat16,gpu_memory_utilization=0.85,"
         f"max_model_len={max_model_len}"),
        "--tasks", ",".join(tasks),
        "--batch_size", batch_size,
        "--output_path", str(out_dir),
    ]
    if limit:
        args += ["--limit", str(limit)]
    print("[eval]", " ".join(args), flush=True)
    subprocess.run(args, check=True)  # noqa: S603 -- argv built here, no shell
    return out_dir


def run_tool_eval(
    model_path: str,
    out_path: str | Path,
    limit: int | None = 150,
    max_new_tokens: int = 256,
    category: str = "simple",
) -> dict:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from .data.registry import load_tool
    from .verifiers import get_verifier

    records = load_tool(split="test", limit=limit, category=category)
    tok = AutoTokenizer.from_pretrained(model_path)
    prompts = [
        tok.apply_chat_template(r["messages"], tokenize=False,
                                add_generation_prompt=True, enable_thinking=False)
        for r in records
    ]
    llm = LLM(model=model_path, dtype="bfloat16", max_model_len=4096)
    outs = llm.generate(
        prompts, SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    )
    verifier = get_verifier("tool")
    scores = [verifier.score(r, o.outputs[0].text) for r, o in zip(records, outs, strict=True)]

    result = {
        "task": f"bfcl_{category}",
        "n": len(scores),
        "accuracy": sum(scores) / max(len(scores), 1),
        "official_checker": getattr(verifier, "used_official", False),
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(result, indent=2))
    print(f"[eval] {result}")
    return result
