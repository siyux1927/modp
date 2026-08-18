"""Command line entry point: `mopd <stage> ...`.

Stages are separate commands on purpose -- each one owns the GPU alone, and a crash
in stage 2 does not cost you stage 1's rollouts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser("mopd")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("rollout", help="stage 1: generate + verify trajectories")
    r.add_argument("--model", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--domains", nargs="+", default=["math", "ifeval", "tool"])
    r.add_argument("--per-domain", type=int, default=2000)
    r.add_argument("--n-samples", type=int, default=4)
    r.add_argument("--max-new-tokens", type=int, default=512)
    r.add_argument("--temperature", type=float, default=1.0)
    r.add_argument("--iteration", type=int, default=0)
    r.add_argument("--seed", type=int, default=0)

    t = sub.add_parser("teacher", help="stage 2: collect one teacher's top-K logprobs")
    t.add_argument("--model", required=True)
    t.add_argument("--role", required=True)
    t.add_argument("--rollout", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("-k", type=int, default=64)
    t.add_argument("--batch-size", type=int, default=8)

    x = sub.add_parser("train", help="stage 3: distil into the student")
    x.add_argument("--config", required=True, help="JSON TrainConfig")
    x.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")

    e = sub.add_parser("eval", help="lm-eval on gsm8k/ifeval + local BFCL")
    e.add_argument("--model", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--tasks", nargs="+", default=["gsm8k", "ifeval"])
    e.add_argument("--limit", type=int, default=None)
    e.add_argument("--skip-tool", action="store_true")

    args = p.parse_args(argv)

    if args.cmd == "rollout":
        from .rollout import RolloutConfig, run_rollout

        run_rollout(
            RolloutConfig(
                model=args.model, out_path=args.out, domains=tuple(args.domains),
                per_domain=args.per_domain, n_samples=args.n_samples,
                max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                iteration=args.iteration, seed=args.seed,
            )
        )
    elif args.cmd == "teacher":
        from .teacher import TeacherConfig, collect_teacher

        collect_teacher(
            TeacherConfig(
                model=args.model, role=args.role, rollout_path=args.rollout,
                out_path=args.out, k=args.k, batch_size=args.batch_size,
            )
        )
    elif args.cmd == "train":
        from .train import TrainConfig, train

        cfg_dict = json.loads(Path(args.config).read_text())
        cfg_dict.update(_parse_overrides(args.set))
        train(TrainConfig(**cfg_dict))
    elif args.cmd == "eval":
        from .evaluate import run_lm_eval, run_tool_eval

        run_lm_eval(args.model, args.tasks, args.out, limit=args.limit)
        if not args.skip_tool:
            run_tool_eval(args.model, Path(args.out) / "bfcl.json", limit=args.limit or 150)

    return 0


def _parse_overrides(pairs: list[str]) -> dict:
    out: dict = {}
    for pair in pairs:
        key, _, raw = pair.partition("=")
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out


if __name__ == "__main__":
    raise SystemExit(main())
