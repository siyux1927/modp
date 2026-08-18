"""Sparse multi-teacher distribution fusion.

Teachers are stored as top-K log-probabilities (see `mopd.teacher`).  Fusing them
naively would require materialising a dense [B, T, V] tensor per teacher, which for
V=151k is several GB.  Instead we work on the *per-position union support*: the set
of token ids that appear in at least one teacher's top-K at that position.  Its size
is at most `n_teachers * K` (192 for 3 teachers at K=64), so the whole thing is three
orders of magnitude cheaper than dense.

Mass that falls outside the union is not thrown away -- it is tracked as a single
extra "other" bucket so that downstream losses still operate on a proper probability
distribution.  See `mopd.loss`.

Two pooling rules are supported, and which one is right is an open question that this
project treats as an experimental variable:

  arithmetic:  p_M(v) = sum_i w_i p_i(v)              (linear opinion pool)
  geometric:   log p_M(v) ∝ sum_i w_i log p_i(v)      (log-linear opinion pool)

Arithmetic pooling is what most multi-teacher KD papers do, but it can place mass on
a "compromise" token that no individual teacher supports.  Geometric pooling is
conjunctive: it only keeps tokens that *all* teachers find plausible.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Log-probability assigned to a token that a teacher did not rank in its top-K and
# whose tail mass is (numerically) zero.  Keeps geometric pooling finite.
LOG_FLOOR = -30.0


@dataclass
class SparseLogprobs:
    """One teacher's next-token distribution, truncated to top-K.

    Shapes carry arbitrary leading dims (typically ``[B, T]``):
        topk_ids       [..., K]  int64
        topk_logprobs  [..., K]  float32, log p over the full vocabulary
        tail_mass      [...]     float32, 1 - sum(exp(topk_logprobs)) >= 0
    """

    topk_ids: torch.Tensor
    topk_logprobs: torch.Tensor
    tail_mass: torch.Tensor

    @property
    def k(self) -> int:
        return self.topk_ids.shape[-1]

    def to(self, *args, **kwargs) -> SparseLogprobs:
        return SparseLogprobs(
            self.topk_ids.to(*args, **kwargs),
            self.topk_logprobs.to(*args, **kwargs),
            self.tail_mass.to(*args, **kwargs),
        )


@dataclass
class FusedTarget:
    """A fused teacher distribution on a padded per-position union support.

        ids         [..., M]  int64, token ids (padding slots hold duplicates)
        logp        [..., M]  float32, log p_M restricted to those ids
        valid       [..., M]  bool, False on duplicate/padding slots
        other_logp  [...]     float32, log of the mass outside the union support
    """

    ids: torch.Tensor
    logp: torch.Tensor
    valid: torch.Tensor
    other_logp: torch.Tensor


def _lookup(signal: SparseLogprobs, query_ids: torch.Tensor, vocab_size: int,
            tail_uniform: bool) -> torch.Tensor:
    """log p_i(v) for arbitrary v, from a top-K representation.

    Tokens inside the teacher's top-K get their stored log-prob.  Tokens outside it
    get the tail mass spread uniformly over the V-K unranked tokens (the maximum
    entropy completion of what we stored), or LOG_FLOOR if `tail_uniform` is off.
    """
    sorted_ids, order = signal.topk_ids.sort(dim=-1)
    sorted_logp = signal.topk_logprobs.gather(-1, order)

    pos = torch.searchsorted(sorted_ids.contiguous(), query_ids.contiguous())
    pos = pos.clamp(max=signal.k - 1)
    hit = sorted_ids.gather(-1, pos) == query_ids
    hit_logp = sorted_logp.gather(-1, pos)

    if tail_uniform:
        n_unranked = max(vocab_size - signal.k, 1)
        miss_logp = (signal.tail_mass.clamp_min(0) / n_unranked).clamp_min(1e-30).log()
    else:
        miss_logp = torch.full_like(signal.tail_mass, LOG_FLOOR)
    miss_logp = miss_logp.unsqueeze(-1).expand_as(hit_logp).clamp_min(LOG_FLOOR)

    return torch.where(hit, hit_logp, miss_logp)


def union_support(signals: list[SparseLogprobs]) -> tuple[torch.Tensor, torch.Tensor]:
    """Padded per-position union of the teachers' top-K supports.

    Returns ``(ids, valid)`` both shaped ``[..., n_teachers * K]``.  Duplicate ids are
    kept in place (so the tensor stays rectangular) but flagged False in `valid`.
    """
    cat = torch.cat([s.topk_ids for s in signals], dim=-1)
    ids, _ = cat.sort(dim=-1)
    dup = ids[..., 1:] == ids[..., :-1]
    valid = torch.cat([torch.ones_like(dup[..., :1]), ~dup], dim=-1)
    return ids, valid


def fuse_teachers(
    signals: list[SparseLogprobs],
    weights: torch.Tensor,
    vocab_size: int,
    mode: str = "arithmetic",
    tail_uniform: bool = True,
) -> FusedTarget:
    """Combine several sparse teacher distributions into one target.

    `weights` is shaped ``[n_teachers]`` or ``[..., n_teachers]`` (per-position
    routing) and is renormalised to sum to 1.
    """
    if not signals:
        raise ValueError("need at least one teacher signal")
    n = len(signals)
    ids, valid = union_support(signals)

    w = weights.to(signals[0].topk_logprobs.dtype)
    if w.shape[-1] != n:
        raise ValueError(f"weights last dim {w.shape[-1]} != n_teachers {n}")
    w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    # -> [..., n_teachers, 1] so it broadcasts against [..., n_teachers, M].
    # Insert the missing axes *before* the teacher axis: per-trajectory weights
    # [B, n] must become [B, 1, n], not [1, B, n].
    while w.dim() < ids.dim():
        w = w.unsqueeze(-2)
    w = w.unsqueeze(-1)

    logp_each = torch.stack(
        [_lookup(s, ids, vocab_size, tail_uniform) for s in signals], dim=-2
    )  # [..., n_teachers, M]

    neg_inf = torch.finfo(logp_each.dtype).min
    if mode == "arithmetic":
        # logsumexp over teachers of (log w_i + log p_i)
        logp = torch.logsumexp(w.clamp_min(1e-30).log() + logp_each, dim=-2)
    elif mode == "geometric":
        logp = (w * logp_each).sum(dim=-2)
    else:
        raise ValueError(f"unknown fusion mode {mode!r}")

    logp = torch.where(valid, logp, torch.full_like(logp, neg_inf))
    total = torch.logsumexp(logp, dim=-1, keepdim=True)

    if mode == "arithmetic":
        # Mass genuinely outside the union: 1 - sum_U p_M(v), clamped for float error.
        inside = total.squeeze(-1).exp().clamp(max=1.0)
        other_logp = (1.0 - inside).clamp_min(1e-30).log()
    else:
        # Log-linear pooling is only defined up to a constant, so it is normalised on
        # the union support and carries no residual bucket.
        logp = logp - total
        other_logp = torch.full_like(total.squeeze(-1), LOG_FLOOR)

    return FusedTarget(ids=ids, logp=logp, valid=valid, other_logp=other_logp)


def dense_reference(signals: list[SparseLogprobs], weights: torch.Tensor,
                    vocab_size: int, mode: str = "arithmetic") -> torch.Tensor:
    """Slow dense fusion over the full vocabulary. For tests only."""
    w = weights / weights.sum(dim=-1, keepdim=True)
    dense = []
    for s in signals:
        n_unranked = max(vocab_size - s.k, 1)
        floor = (s.tail_mass.clamp_min(0) / n_unranked).clamp_min(1e-30).log()
        d = floor.unsqueeze(-1).expand(*s.topk_ids.shape[:-1], vocab_size).clone()
        d.scatter_(-1, s.topk_ids, s.topk_logprobs)
        dense.append(d.clamp_min(LOG_FLOOR))
    stacked = torch.stack(dense, dim=-2)
    wb = w.reshape(*([1] * (stacked.dim() - 2)), -1, 1) if w.dim() == 1 else w.unsqueeze(-1)
    if mode == "arithmetic":
        out = torch.logsumexp(wb.log() + stacked, dim=-2)
    else:
        out = (wb * stacked).sum(dim=-2)
    return out - torch.logsumexp(out, dim=-1, keepdim=True)
