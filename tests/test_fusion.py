import pytest
import torch

from mopd.fusion import SparseLogprobs, dense_reference, fuse_teachers, union_support


def make_signal(logits: torch.Tensor, k: int) -> SparseLogprobs:
    lp = torch.log_softmax(logits, dim=-1)
    top = lp.topk(k, dim=-1)
    tail = (1.0 - top.values.exp().sum(-1)).clamp_min(0)
    return SparseLogprobs(top.indices, top.values, tail)


def gather_valid(target, vocab):
    """Dense-ify a FusedTarget's union support for comparison.

    Duplicate slots are routed to a scratch column rather than scattered over the
    real one -- scatter_ resolves duplicate indices by last-write-wins, so writing
    -inf for the duplicates would clobber the valid entry that precedes them.
    """
    out = torch.full((*target.ids.shape[:-1], vocab + 1), -float("inf"))
    ids = torch.where(target.valid, target.ids, torch.full_like(target.ids, vocab))
    out.scatter_(-1, ids, target.logp)
    return out[..., :vocab]


@pytest.mark.parametrize("mode", ["arithmetic", "geometric"])
def test_full_k_matches_dense(mode):
    """With K = vocab_size the sparse path must be exact, not approximate."""
    torch.manual_seed(0)
    v, t = 8, 5
    logits = [torch.randn(t, v) for _ in range(3)]
    signals = [make_signal(x, k=v) for x in logits]
    w = torch.tensor([0.5, 0.3, 0.2])

    got = gather_valid(fuse_teachers(signals, w, vocab_size=v, mode=mode), v)
    want = dense_reference(signals, w, vocab_size=v, mode=mode)
    assert torch.allclose(got.exp(), want.exp(), atol=1e-5)


def test_arithmetic_equals_weighted_probability_average():
    """The linear opinion pool is what copy.md specifies; pin it down explicitly."""
    torch.manual_seed(1)
    v, t = 6, 4
    logits = [torch.randn(t, v) for _ in range(2)]
    signals = [make_signal(x, k=v) for x in logits]
    w = torch.tensor([0.7, 0.3])

    got = gather_valid(fuse_teachers(signals, w, vocab_size=v), v).exp()
    want = 0.7 * logits[0].softmax(-1) + 0.3 * logits[1].softmax(-1)
    assert torch.allclose(got, want, atol=1e-6)


def test_geometric_differs_from_arithmetic_on_disagreement():
    """Log-linear pooling is conjunctive: a token only one teacher likes is damped."""
    v = 4
    a = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    b = torch.tensor([[0.0, 10.0, 0.0, 0.0]])
    signals = [make_signal(a, v), make_signal(b, v)]
    w = torch.tensor([0.5, 0.5])

    arith = gather_valid(fuse_teachers(signals, w, v, mode="arithmetic"), v).exp()
    geom = gather_valid(fuse_teachers(signals, w, v, mode="geometric"), v).exp()

    # arithmetic keeps both peaks; geometric flattens toward the tokens both allow
    assert arith[0, 0] > 0.4 and arith[0, 1] > 0.4
    assert geom[0, 0] == pytest.approx(geom[0, 1], abs=1e-6)
    assert geom[0, 0] < arith[0, 0]


def test_union_support_deduplicates():
    a = SparseLogprobs(torch.tensor([[1, 2, 3]]), torch.zeros(1, 3), torch.zeros(1))
    b = SparseLogprobs(torch.tensor([[2, 3, 9]]), torch.zeros(1, 3), torch.zeros(1))
    ids, valid = union_support([a, b])
    assert ids.shape == (1, 6)
    assert sorted(ids[0][valid[0]].tolist()) == [1, 2, 3, 9]


def test_truncation_keeps_tail_mass_accounted():
    """Truncating to top-K must not silently renormalise the distribution."""
    torch.manual_seed(2)
    v, t, k = 64, 3, 8
    logits = [torch.randn(t, v) * 0.5 for _ in range(2)]
    signals = [make_signal(x, k=k) for x in logits]
    w = torch.tensor([0.5, 0.5])

    target = fuse_teachers(signals, w, vocab_size=v)
    inside = torch.where(target.valid, target.logp, torch.tensor(-1e30)).logsumexp(-1).exp()
    total = inside + target.other_logp.exp()
    assert torch.allclose(total, torch.ones(t), atol=1e-4)
    assert (target.other_logp.exp() > 1e-4).all(), "tail mass should be non-trivial here"


def test_per_trajectory_weights_broadcast_over_time():
    """[B, n] weights must expand along T, not collide with the batch axis."""
    torch.manual_seed(3)
    b, t, v, k = 2, 4, 32, 8
    signals = [make_signal(torch.randn(b, t, v), k) for _ in range(3)]

    per_traj = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    fused = fuse_teachers(signals, per_traj, vocab_size=v, tail_uniform=False)
    assert fused.logp.shape == (b, t, 3 * k)

    # Row 0 is routed entirely to teacher 0, so the target must be teacher 0's own
    # distribution -- even though the union support also carries teachers 1 and 2's
    # tokens, which here must come out at zero.
    got = gather_valid(fused, v).exp()
    for row, teacher in ((0, 0), (1, 2)):
        solo = gather_valid(
            fuse_teachers([signals[teacher]], torch.tensor([1.0]), v, tail_uniform=False), v
        ).exp()
        assert torch.allclose(got[row], solo[row], atol=1e-5)


def test_uniform_tail_fills_the_union_support():
    """A token another teacher ranked, but this one did not, gets its tail estimate --
    not zero. Otherwise truncation would fabricate hard zeros in the target."""
    torch.manual_seed(31)
    t, v, k = 3, 32, 4
    signals = [make_signal(torch.randn(t, v), k) for _ in range(2)]

    with_tail = fuse_teachers(signals, torch.tensor([1.0, 0.0]), v, tail_uniform=True)
    without = fuse_teachers(signals, torch.tensor([1.0, 0.0]), v, tail_uniform=False)

    extra = gather_valid(with_tail, v).exp() - gather_valid(without, v).exp()
    floor = signals[0].tail_mass / (v - k)
    assert (extra >= -1e-6).all()
    assert torch.allclose(extra.max(-1).values, floor, atol=1e-6)


def test_per_token_weights_accepted():
    torch.manual_seed(4)
    b, t, v, k = 2, 3, 32, 8
    signals = [make_signal(torch.randn(b, t, v), k) for _ in range(2)]
    w = torch.rand(b, t, 2)
    fused = fuse_teachers(signals, w, vocab_size=v)
    assert fused.logp.shape == (b, t, 2 * k)
    assert torch.isfinite(fused.logp[fused.valid]).all()
