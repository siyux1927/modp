"""Tool-call verification against BFCL references, by AST comparison.

Nothing is executed.  BFCL's "AST" evaluation category compares the *structure* of
the emitted call -- function name, argument names, argument values, against the
type/enum constraints in the function schema -- to a reference answer.

Preferred path is BFCL's own checker (`pip install bfcl-eval`), so the numbers are
comparable to the public leaderboard.  If it is not installed we fall back to a small
local matcher with the same contract; it is stricter about formatting and slightly
more pessimistic, and `used_official` records which one produced a score.
"""

from __future__ import annotations

import ast
import json
import re


class ToolVerifier:
    domain = "tool"

    def __init__(self):
        self._official = _load_official()
        self.used_official = self._official is not None

    def score(self, record: dict, completion: str) -> float:
        calls = parse_calls(completion)
        if not calls:
            return 0.0
        reference = record["reference_calls"]  # list[{name: {arg: [accepted values]}}]

        if self._official is not None:
            try:
                res = self._official(
                    func_description=record.get("function", []),
                    model_output=calls,
                    possible_answer=reference,
                    language=record.get("language", "Python"),
                    test_category=record.get("test_category", "simple"),
                    model_name="mopd-student",
                )
                return 1.0 if (res.get("valid") if isinstance(res, dict) else bool(res)) else 0.0
            except Exception:  # noqa: S110
                # bfcl-eval's checker signature drifts between releases; a failure
                # here means we could not use the official path, not that the call
                # was wrong, so fall through to the local matcher.
                pass

        return float(_local_match(calls, reference))


def _load_official():
    try:
        from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
        return ast_checker
    except ImportError:
        return None


_CALL_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*\s*\(.*?\)", re.DOTALL)


def parse_calls(text: str) -> list[dict]:
    """Extract calls from a completion as [{name: {arg: value}}].

    Accepts both the JSON tool-call form and BFCL's Python-literal form:
        [{"name": "get_weather", "arguments": {"city": "Paris"}}]
        [get_weather(city='Paris')]
    """
    text = text.strip()
    for block in _json_candidates(text):
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        items = data if isinstance(data, list) else [data]
        out = []
        for it in items:
            if isinstance(it, dict) and "name" in it:
                args = it.get("arguments", it.get("parameters", {})) or {}
                out.append({it["name"]: args})
        if out:
            return out

    out = []
    for snippet in _CALL_RE.findall(text):
        try:
            node = ast.parse(snippet.strip(), mode="eval").body
        except SyntaxError:
            continue
        if not isinstance(node, ast.Call) or node.args:
            # BFCL references are keyword-only; a positional-arg call is either a
            # malformed emission or an incidental sub-expression (`__import__('os')`),
            # and admitting it with an empty argument dict would let a call that
            # specifies nothing match a reference whose arguments are all optional.
            continue
        name = ast.unparse(node.func)
        args = {}
        ok = True
        for kw in node.keywords:
            try:
                args[kw.arg] = ast.literal_eval(kw.value)
            except (ValueError, SyntaxError):
                ok = False
                break
        if ok:
            out.append({name: args})
    return out


def _json_candidates(text: str) -> list[str]:
    cands = [text]
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fence:
        cands.insert(0, fence.group(1).strip())
    bracket = re.search(r"[\[{].*[\]}]", text, re.DOTALL)
    if bracket:
        cands.append(bracket.group(0))
    return cands


def _local_match(calls: list[dict], reference: list[dict]) -> bool:
    if len(calls) != len(reference):
        return False
    for got, want in zip(calls, reference, strict=True):
        (gname, gargs), (wname, wargs) = next(iter(got.items())), next(iter(want.items()))
        if gname.split(".")[-1] != wname.split(".")[-1]:
            return False
        for arg, raw_accepted in wargs.items():
            accepted = raw_accepted if isinstance(raw_accepted, list) else [raw_accepted]
            optional = "" in accepted or None in accepted
            if arg not in gargs:
                if not optional:
                    return False
                continue
            if not any(_loose_eq(gargs[arg], a) for a in accepted):
                return False
        if set(gargs) - set(wargs):
            return False
    return True


def _loose_eq(a, b) -> bool:
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().lower() == b.strip().lower()
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 1e-6
    return a == b
