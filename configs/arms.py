"""The experiment matrix, as data.

Run `python configs/arms.py` to emit one TrainConfig JSON per arm into `configs/`.
Every arm shares the same rollouts and the same teacher signals; they differ only in
how those signals are combined, which is what makes the comparison clean and what
makes the whole matrix cost one rollout pass plus one teacher pass.

    A0  no distillation                        student baseline (skip training)
    A1  off-policy, 3 teachers, uniform        classical multi-teacher KD
    A2  on-policy,  1 teacher                  single-teacher GKD
    A3  on-policy,  3 teachers, uniform        ablation: does routing matter?
    A4  on-policy,  3 teachers, confidence     the method under test
    A5  on-policy,  3 teachers, oracle         routing headroom
"""

from __future__ import annotations

import json
from pathlib import Path

STUDENT = "Qwen/Qwen3-1.7B-Base"

TEACHERS = {
    "math": "Qwen/Qwen3-4B",
    "instruct": "Qwen/Qwen3-8B",
    "general": "Qwen/Qwen2.5-Coder-7B-Instruct",
}

ON_POLICY = "data/onpolicy"      # rollouts from the student
OFF_POLICY = "data/offpolicy"    # rollouts from a fixed stronger generator

BASE = dict(
    student=STUDENT,
    beta=0.5,
    fusion_mode="arithmetic",
    failure_aware=True,
    lr=1e-5,
    epochs=1,
    batch_size=4,
    grad_accum=4,
    max_len=1280,
    gradient_checkpointing=True,
    seed=0,
)


def paths(root: str) -> dict[str, str]:
    return {role: f"{root}/teacher-{role}.parquet" for role in TEACHERS}


ARMS: dict[str, dict] = {
    "a1_offpolicy_uniform": dict(
        rollout_path=f"{OFF_POLICY}/rollout.parquet",
        teacher_paths=paths(OFF_POLICY),
        router="uniform",
        failure_aware=False,  # no student-visited states to be failure-aware about
    ),
    "a2_onpolicy_single": dict(
        rollout_path=f"{ON_POLICY}/rollout.parquet",
        teacher_paths=paths(ON_POLICY),
        router="single",
        single_index=1,  # the 8B generalist
    ),
    "a3_onpolicy_uniform": dict(
        rollout_path=f"{ON_POLICY}/rollout.parquet",
        teacher_paths=paths(ON_POLICY),
        router="uniform",
    ),
    "a4_onpolicy_confidence": dict(
        rollout_path=f"{ON_POLICY}/rollout.parquet",
        teacher_paths=paths(ON_POLICY),
        router="confidence",
        router_temperature=0.5,
    ),
    "a5_onpolicy_oracle": dict(
        rollout_path=f"{ON_POLICY}/rollout.parquet",
        teacher_paths=paths(ON_POLICY),
        router="oracle",
    ),
}

# Ablations off the main arm. Each is A4 with exactly one field changed, so any
# difference is attributable.
ABLATIONS: dict[str, dict] = {
    # beta follows the TRL/GKD convention: 0 = forward KL (mass-covering),
    # 1 = reverse KL (mode-seeking). See mopd/loss.py.
    "abl_beta0_forward_kl": dict(beta=0.0),
    "abl_beta1_reverse_kl": dict(beta=1.0),
    "abl_geometric_pooling": dict(fusion_mode="geometric"),
    "abl_no_failure_aware": dict(failure_aware=False),
    "abl_token_routing": dict(router="confidence_token"),
}


def build() -> dict[str, dict]:
    out = {}
    for name, over in ARMS.items():
        out[name] = {**BASE, **over, "out_dir": f"runs/{name}", "tags": [name]}
    for name, over in ABLATIONS.items():
        out[name] = {
            **out["a4_onpolicy_confidence"],
            **over,
            "out_dir": f"runs/{name}",
            "tags": [name, "ablation"],
        }
    return out


if __name__ == "__main__":
    here = Path(__file__).parent
    for name, cfg in build().items():
        (here / f"{name}.json").write_text(json.dumps(cfg, indent=2) + "\n")
        print(f"wrote configs/{name}.json")
