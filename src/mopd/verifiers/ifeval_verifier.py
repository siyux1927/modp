"""IFEval verification via Google's official constraint registry.

Each IFEval record carries `instruction_id_list` (e.g. "length_constraints:number_words")
and a parallel `kwargs` list.  The official registry maps each id to a class exposing
`build_description(**kwargs)` and `check_following(text) -> bool`.  Everything is
pure Python string/regex work.

Two import paths are tried, because the PyPI mirror of this package is not reliably
maintained:
  1. `instruction_following_eval` (pip install instruction-following-eval)
  2. `lm_eval.tasks.ifeval.instructions_registry` (comes with lm-eval, which the
     project already depends on for evaluation)
"""

from __future__ import annotations


def _load_registry():
    try:
        from instruction_following_eval import instructions_registry
        return instructions_registry
    except ImportError:
        pass
    try:
        from lm_eval.tasks.ifeval import instructions_registry
        return instructions_registry
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError(
            "ifeval domain needs the official constraint registry. Install either "
            "`pip install instruction-following-eval` or `pip install lm-eval`."
        ) from e


class IFEvalVerifier:
    domain = "ifeval"

    def __init__(self, strict: bool = True):
        self._registry = _load_registry()
        # strict: the response is checked as-is.  IFEval also defines a "loose" mode
        # that strips markdown wrappers and retries; strict is the harder number and
        # the one we report.
        self.strict = strict

    def score(self, record: dict, completion: str) -> float:
        ids = record.get("instruction_id_list") or []
        kwargs_list = record.get("kwargs") or [{}] * len(ids)
        if not ids:
            return 0.0
        text = completion if self.strict else _loosen(completion)
        for iid, kw in zip(ids, kwargs_list, strict=False):
            cls = self._registry.INSTRUCTION_DICT.get(iid)
            if cls is None:
                return 0.0
            inst = cls(iid)
            inst.build_description(**{k: v for k, v in (kw or {}).items() if v is not None})
            args = inst.get_instruction_args()
            if args and "prompt" in args:
                inst.build_description(prompt=record["prompt"])
            if not text.strip() or not inst.check_following(text):
                return 0.0
        return 1.0


def _loosen(text: str) -> str:
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if not lines:
        return text
    body = "\n".join(lines)
    return body.replace("*", "").strip()
