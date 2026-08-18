"""Teacher routing: how much each teacher gets to say at each position.

This is the module the experiment matrix actually varies.  Every router returns a
weight tensor shaped ``[B, n_teachers]`` (per trajectory) or ``[B, T, n_teachers]``
(per token), non-negative, which `mopd.fusion.fuse_teachers` renormalises.

    uniform      equal weights                        -- ablation: is routing needed?
    single       all mass on one teacher              -- single-teacher GKD baseline
    oracle       hand-specified domain -> teacher map -- upper bound on routing
    confidence   softmax over each teacher's mean log-prob of the student's own
                 tokens -- the method under test.  A teacher that finds the student's
                 trajectory unsurprising is one that can model this state well.
    confidence_token  the same signal computed per position instead of per
                 trajectory, so routing can change mid-trajectory.

`confidence` uses no labels and no verifier, so it is deployable; `oracle` needs the
domain label at training time and exists only to bound how much headroom routing has.
"""

from __future__ import annotations

import torch

# Which teacher should dominate on which domain, for the `oracle` router.
# Keys are domain names from `mopd.data.registry`, values are teacher *roles*.
ORACLE_PRIOR: dict[str, dict[str, float]] = {
    "math": {"math": 0.8, "instruct": 0.1, "general": 0.1},
    "ifeval": {"math": 0.1, "instruct": 0.8, "general": 0.1},
    "tool": {"math": 0.1, "instruct": 0.2, "general": 0.7},
}

ROUTERS = ("uniform", "single", "oracle", "confidence", "confidence_token")


def route(
    mode: str,
    roles: list[str],
    *,
    mean_logprobs: torch.Tensor | None = None,
    token_logprobs: torch.Tensor | None = None,
    domains: list[str] | None = None,
    single_index: int = 0,
    temperature: float = 0.5,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Compute teacher weights.

    roles          teacher role names, ordered as the signals are ordered.
    mean_logprobs  [B, n] mean log-prob each teacher assigns to the student's own
                   completion tokens.  Required by `confidence`.
    token_logprobs [B, T, n] the same, per position.  Required by `confidence_token`.
    domains        length-B domain labels.  Required by `oracle`.
    """
    n = len(roles)
    if mode == "uniform":
        b = _batch_size(mean_logprobs, token_logprobs)
        return torch.full((b, n), 1.0 / n, device=device)

    if mode == "single":
        b = _batch_size(mean_logprobs, token_logprobs)
        w = torch.zeros(b, n, device=device)
        w[:, single_index] = 1.0
        return w

    if mode == "oracle":
        if domains is None:
            raise ValueError("oracle routing needs domain labels")
        rows = []
        for d in domains:
            prior = ORACLE_PRIOR.get(d)
            if prior is None:
                rows.append([1.0 / n] * n)
            else:
                rows.append([prior.get(r, 0.0) for r in roles])
        return torch.tensor(rows, dtype=torch.float32, device=device)

    if mode == "confidence":
        if mean_logprobs is None:
            raise ValueError("confidence routing needs mean_logprobs")
        return torch.softmax(mean_logprobs.float().to(device) / temperature, dim=-1)

    if mode == "confidence_token":
        if token_logprobs is None:
            raise ValueError("confidence_token routing needs token_logprobs")
        return torch.softmax(token_logprobs.float().to(device) / temperature, dim=-1)

    raise ValueError(f"unknown router {mode!r}; expected one of {ROUTERS}")


def _batch_size(*candidates: torch.Tensor | None) -> int:
    for c in candidates:
        if c is not None:
            return c.shape[0]
    raise ValueError("cannot infer batch size; pass mean_logprobs or token_logprobs")


def routing_entropy(weights: torch.Tensor) -> torch.Tensor:
    """Normalised entropy of the routing distribution, in [0, 1].

    Logged during training: a confidence router that collapses to entropy 0 has become
    a single-teacher run, and one that stays at 1 has become the uniform ablation.
    Either way the arm is not testing what it claims to test.
    """
    n = weights.shape[-1]
    w = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
    ent = -(w * w.clamp_min(1e-12).log()).sum(-1)
    return ent / torch.tensor(float(n), device=w.device).log()
