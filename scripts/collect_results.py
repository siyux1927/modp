#!/usr/bin/env python3
"""Collapse lm-eval JSON dumps plus the local BFCL score into one markdown table.

Runs are grouped by arm name with the trailing `-s<seed>` stripped, so repeated seeds
aggregate into mean +/- std. With three seeds a difference smaller than roughly two
combined standard deviations is not a result -- the table prints the spread so that
stays visible rather than getting rounded away.
"""

from __future__ import annotations

import json
import math
import re
import statistics
import sys
from pathlib import Path

METRICS = {
    "gsm8k": ("exact_match,strict-match", "GSM8K"),
    "ifeval": ("prompt_level_strict_acc,none", "IFEval"),
}
SEED_RE = re.compile(r"-s\d+$")


def load_run(run_dir: Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    for path in run_dir.rglob("results*.json"):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        for task, (key, label) in METRICS.items():
            entry = data.get("results", {}).get(task)
            if entry and key in entry:
                scores[label] = float(entry[key]) * 100
    bfcl = run_dir / "bfcl.json"
    if bfcl.exists():
        scores["BFCL"] = json.loads(bfcl.read_text())["accuracy"] * 100
    return scores


def main(root: str = "results") -> int:
    runs: dict[str, list[dict[str, float]]] = {}
    for d in sorted(Path(root).iterdir()):
        if not d.is_dir():
            continue
        scores = load_run(d)
        if scores:
            runs.setdefault(SEED_RE.sub("", d.name), []).append(scores)

    if not runs:
        print(f"no results found under {root}/", file=sys.stderr)
        return 1

    labels = [lab for _, lab in METRICS.values()] + ["BFCL"]
    print("| arm | seeds | " + " | ".join(labels) + " | mean |")
    print("|---|---|" + "---|" * (len(labels) + 1))
    for name, seeds in sorted(runs.items()):
        cells, means = [], []
        for lab in labels:
            vals = [s[lab] for s in seeds if lab in s]
            if not vals:
                cells.append("--")
                continue
            m = statistics.mean(vals)
            means.append(m)
            sd = statistics.stdev(vals) if len(vals) > 1 else math.nan
            cells.append(f"{m:.1f}" if math.isnan(sd) else f"{m:.1f} ± {sd:.1f}")
        avg = f"{statistics.mean(means):.1f}" if means else "--"
        print(f"| {name} | {len(seeds)} | " + " | ".join(cells) + f" | {avg} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
