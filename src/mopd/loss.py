"""Distillation objective.

Generalised Jensen-Shannon divergence, matching the convention of the GKD paper
(Agarwal et al., 2024) and TRL's `generalized_jsd_loss` exactly -- including its
choice of which end of the beta range is which, which is easy to get backwards:

    M       = (1 - beta) * Q_student + beta * P_teacher
    JSD_b   = beta * KL(P || M) + (1 - beta) * KL(Q || M)

with the endpoints special-cased the way TRL special-cases them:

    beta = 0  ->  KL(P_teacher || Q_student)   forward KL, mass-covering
    beta = 1  ->  KL(Q_student || P_teacher)   reverse KL, mode-seeking
    beta = .5 ->  symmetric JSD

For on-policy distillation into a much smaller student, the reverse end (beta near 1)
is usually the right one: forward KL forces a 1.7B student to cover every mode of an
8B teacher, which it cannot, and the result is a flattened policy.  This is the `beta`
sweep in the experiment matrix.

Note that JSD_b is discontinuous at the endpoints under this definition -- the
interior formula tends to 0 as beta -> 0, not to forward KL.  TRL has the same
discontinuity; `tests/test_loss.py` pins our values to theirs rather than to a
smoothed version, because comparability with published GKD numbers is worth more here
than tidiness.

Everything here operates on the *union support + other bucket* representation from
`mopd.fusion`, so the student's full-vocabulary softmax is never materialised.
"""

from __future__ import annotations

import math

import torch

from .fusion import FusedTarget

_NEG = -1e30


def _student_logprobs(
    student_logits: torch.Tensor, target: FusedTarget, temperature: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """log q over (union support, other bucket) without a dense [B,T,V] tensor."""
    logits = student_logits.float()
    if temperature != 1.0:
        logits = logits / temperature
    lse = torch.logsumexp(logits, dim=-1, keepdim=True)  # [B, T, 1]
    q = logits.gather(-1, target.ids) - lse  # [B, T, M]
    q = torch.where(target.valid, q, torch.full_like(q, _NEG))
    inside = torch.logsumexp(q, dim=-1).exp().clamp(max=1.0 - 1e-7)
    q_other = (1.0 - inside).clamp_min(1e-30).log()
    return q, q_other


def _kl(log_a: torch.Tensor, log_b: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """sum_v a(v) * (log a(v) - log b(v)) over the last dim."""
    a = log_a.exp()
    term = a * (log_a - log_b)
    return torch.where(valid, term, torch.zeros_like(term)).sum(dim=-1)


def jsd_loss(
    student_logits: torch.Tensor,
    target: FusedTarget,
    mask: torch.Tensor,
    beta: float = 0.5,
    temperature: float = 1.0,
    token_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weighted mean generalised JSD over supervised positions.

    student_logits [B, T, V] -- already aligned so that position t is the prediction
                                the target at position t describes.
    mask           [B, T]    -- 1 on positions to supervise.
    token_weights  [B, T]    -- optional per-token multiplier (see below).
    """
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must lie in [0, 1]")

    q_u, q_o = _student_logprobs(student_logits, target, temperature)
    p_u = torch.where(target.valid, target.logp, torch.full_like(target.logp, _NEG))
    p_o = target.other_logp

    log_q = torch.cat([q_u, q_o.unsqueeze(-1)], dim=-1)
    log_p = torch.cat([p_u, p_o.unsqueeze(-1)], dim=-1)
    valid = torch.cat([target.valid, torch.ones_like(target.valid[..., :1])], dim=-1)

    if beta <= 0.0:
        per_token = _kl(log_p, log_q, valid)      # forward KL(teacher || student)
    elif beta >= 1.0:
        per_token = _kl(log_q, log_p, valid)      # reverse KL(student || teacher)
    else:
        log_m = torch.logaddexp(
            log_q + math.log1p(-beta), log_p + math.log(beta)
        )
        per_token = beta * _kl(log_p, log_m, valid) + (1 - beta) * _kl(log_q, log_m, valid)

    w = mask.float()
    if token_weights is not None:
        w = w * token_weights
    return (per_token * w).sum() / w.sum().clamp_min(1e-8)


def failure_aware_weights(
    student_logits: torch.Tensor,
    target: FusedTarget,
    mask: torch.Tensor,
    reward: torch.Tensor,
    entropy_gate: bool = True,
    failure_scale: float = 2.0,
    max_weight: float = 4.0,
) -> torch.Tensor:
    """Per-token distillation weights that concentrate teacher signal on failures.

    Two multiplicative gates, both bounded so no single token can dominate a batch:

      entropy gate -- the student's own normalised predictive entropy at that
                      position, in [0, 1].  Where the student is already confident
                      and correct there is little to learn; where it is uncertain the
                      teacher is worth listening to.  Detached: this reweights the
                      gradient, it is not itself optimised.
      failure gate -- `failure_scale` on trajectories the verifier rejected.

    This replaces the hand-tuned `if tool_error: w *= 4` style heuristic with two
    quantities that are measurable and ablatable.  `reward` is [B], per trajectory.
    """
    w = torch.ones(mask.shape, device=mask.device, dtype=torch.float32)

    if entropy_gate:
        with torch.no_grad():
            q_u, q_o = _student_logprobs(student_logits, target, 1.0)
            log_q = torch.cat([q_u, q_o.unsqueeze(-1)], dim=-1)
            valid = torch.cat([target.valid, torch.ones_like(target.valid[..., :1])], dim=-1)
            p = log_q.exp()
            ent = -torch.where(valid, p * log_q, torch.zeros_like(p)).sum(-1)
            w = w * (1.0 + ent / math.log(log_q.shape[-1]))

    fail = (reward <= 0).float().unsqueeze(-1)
    w = w * (1.0 + fail * (failure_scale - 1.0))
    return w.clamp(max=max_weight) * mask.float()
