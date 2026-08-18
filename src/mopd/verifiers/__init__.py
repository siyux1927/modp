"""Verifiers: reward signal for a student rollout.

Every verifier here is a pure function of (reference record, generated text).  No
subprocess, no container, no network, no filesystem.  That is a deliberate design
constraint: code-execution benchmarks are what force a distillation project to build
and maintain a sandbox, and dropping them removes that entire axis of complexity
without giving up a credible evaluation story.

  math    GSM8K / MATH-500, checked with HF's `math-verify` (SymPy equivalence).
  ifeval  Google's IFEval, checked with the official `instruction_following_eval`
          constraint registry -- 25 programmatically verifiable instruction types.
  tool    BFCL v3, checked by AST comparison of the emitted call against the
          reference.  BFCL's AST checker never *invokes* the function.

The three probe close-to-orthogonal abilities (symbolic reasoning / constraint
satisfaction / structured API emission), which is what makes them useful for testing
whether multi-teacher fusion actually fuses heterogeneous skills.
"""

from __future__ import annotations

from typing import Protocol

from .bfcl_verifier import ToolVerifier
from .ifeval_verifier import IFEvalVerifier
from .math_verifier import MathVerifier


class Verifier(Protocol):
    domain: str

    def score(self, record: dict, completion: str) -> float:
        """Return 1.0 if the completion satisfies the record's reference, else 0.0."""


_REGISTRY: dict[str, type] = {
    "math": MathVerifier,
    "ifeval": IFEvalVerifier,
    "tool": ToolVerifier,
}


def get_verifier(domain: str) -> Verifier:
    try:
        return _REGISTRY[domain]()
    except KeyError:
        raise ValueError(
            f"no verifier for domain {domain!r}; have {sorted(_REGISTRY)}"
        ) from None


def score_batch(records: list[dict], completions: list[str]) -> list[float]:
    """Score a heterogeneous batch, dispatching on each record's `domain`."""
    cache: dict[str, Verifier] = {}
    out = []
    for rec, comp in zip(records, completions, strict=True):
        d = rec["domain"]
        if d not in cache:
            cache[d] = get_verifier(d)
        out.append(cache[d].score(rec, comp))
    return out


__all__ = [
    "IFEvalVerifier",
    "MathVerifier",
    "ToolVerifier",
    "Verifier",
    "get_verifier",
    "score_batch",
]
