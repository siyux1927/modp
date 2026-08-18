"""Math verification via HF `math-verify` (SymPy-based answer equivalence)."""

from __future__ import annotations

import re

_INSTALL = "pip install math-verify"


class MathVerifier:
    domain = "math"

    def __init__(self, strict: bool = False):
        try:
            from math_verify import parse, verify
        except ImportError as e:  # pragma: no cover - environment dependent
            raise ImportError(f"math domain needs math-verify: {_INSTALL}") from e
        self._parse, self._verify = parse, verify
        self.strict = strict

    def score(self, record: dict, completion: str) -> float:
        gold = record["answer"]
        try:
            gold_expr = self._parse(_wrap(gold))
            pred_expr = self._parse(completion)
            if not gold_expr or not pred_expr:
                return 0.0
            return 1.0 if self._verify(gold_expr, pred_expr) else 0.0
        except Exception:
            # math-verify raises on pathological SymPy input; an unparseable answer
            # is a wrong answer, not a crash.
            return 0.0


def _wrap(gold: str) -> str:
    """math-verify wants a LaTeX-ish string; GSM8K golds are bare numbers."""
    gold = gold.strip()
    if "\\boxed" in gold or "$" in gold:
        return gold
    return f"${gold}$"


GSM8K_ANSWER_RE = re.compile(r"####\s*(.+?)\s*$")


def gsm8k_gold(answer_field: str) -> str:
    """Strip GSM8K's chain-of-thought, keeping the value after '####'."""
    m = GSM8K_ANSWER_RE.search(answer_field.strip())
    return (m.group(1) if m else answer_field).replace(",", "").strip()
