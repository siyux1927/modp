import math

import pytest
import torch

from mopd.fusion import fuse_teachers
from mopd.loss import failure_aware_weights, jsd_loss
from tests.test_fusion import make_signal


def setup(v=16, b=2, t=3, k=None, seed=0):
    torch.manual_seed(seed)
    k = k or v
    teacher_logits = torch.randn(b, t, v)
    signals = [make_signal(teacher_logits, k)]
    target = fuse_teachers(signals, torch.tensor([1.0]), vocab_size=v)
    student_logits = torch.randn(b, t, v)
    mask = torch.ones(b, t, dtype=torch.bool)
    return teacher_logits, student_logits, target, mask, v


def dense_kl(log_p, log_q):
    return (log_p.exp() * (log_p - log_q)).sum(-1).mean()


def test_beta_zero_is_forward_kl():
    """TRL's convention: beta=0 is KL(teacher || student). Easy to get backwards."""
    tl, sl, target, mask, v = setup()
    got = jsd_loss(sl, target, mask, beta=0.0)
    want = dense_kl(tl.log_softmax(-1), sl.log_softmax(-1))
    assert got == pytest.approx(float(want), abs=1e-4)


def test_beta_one_is_reverse_kl():
    tl, sl, target, mask, v = setup()
    got = jsd_loss(sl, target, mask, beta=1.0)
    want = dense_kl(sl.log_softmax(-1), tl.log_softmax(-1))
    assert got == pytest.approx(float(want), abs=1e-4)


def test_identical_distributions_give_zero_loss():
    tl, _, target, mask, v = setup()
    for beta in (0.0, 0.3, 0.5, 1.0):
        assert jsd_loss(tl, target, mask, beta=beta) == pytest.approx(0.0, abs=1e-5)


def test_jsd_is_bounded_and_positive():
    tl, sl, target, mask, v = setup()
    loss = jsd_loss(sl, target, mask, beta=0.5)
    assert 0.0 < loss < math.log(2) + 1e-6


def test_forward_and_reverse_kl_differ_for_mismatched_supports():
    """The whole point of the beta sweep: the two directions are not interchangeable."""
    v = 8
    teacher = torch.tensor([[[5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
    student = torch.tensor([[[8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
    target = fuse_teachers([make_signal(teacher, v)], torch.tensor([1.0]), vocab_size=v)
    mask = torch.ones(1, 1, dtype=torch.bool)
    fwd = jsd_loss(student, target, mask, beta=0.0)
    rev = jsd_loss(student, target, mask, beta=1.0)
    assert fwd > rev, "mass-covering forward KL should punish the dropped mode harder"


def test_mask_excludes_padding():
    tl, sl, target, mask, v = setup()
    masked = mask.clone()
    masked[:, -1] = False
    full = jsd_loss(sl, target, mask, beta=0.5)
    part = jsd_loss(sl, target, masked, beta=0.5)
    assert full != pytest.approx(float(part), abs=1e-6)

    # a fully-masked position must not contribute at all
    sl2 = sl.clone()
    sl2[:, -1] = torch.randn(sl.shape[0], v) * 10
    assert jsd_loss(sl2, target, masked, beta=0.5) == pytest.approx(float(part), abs=1e-6)


def test_truncation_error_shrinks_as_k_grows():
    """Top-K truncation must converge to the exact loss, monotonically in K.

    This is the invariant the `K in {8, 64, 256}` ablation rests on. The absolute
    error at small K is not small -- an independent random student puts real mass on
    the teacher's unranked tail -- which is exactly why K is swept rather than
    assumed.
    """
    v = 256
    torch.manual_seed(7)
    tl = torch.randn(1, 8, v) * 2.0
    sl = torch.randn(1, 8, v) * 2.0
    mask = torch.ones(1, 8, dtype=torch.bool)
    exact = float(jsd_loss(sl, fuse_teachers([make_signal(tl, v)], torch.tensor([1.0]), v),
                           mask, beta=0.5))
    errs = []
    for k in (8, 32, 128):
        trunc = float(jsd_loss(sl, fuse_teachers([make_signal(tl, k)], torch.tensor([1.0]), v),
                               mask, beta=0.5))
        errs.append(abs(exact - trunc))
    assert errs == sorted(errs, reverse=True), f"error should shrink with K, got {errs}"
    assert errs[-1] < 0.05 * exact


def test_failure_aware_weights_upweight_failures():
    tl, sl, target, mask, v = setup(b=2)
    reward = torch.tensor([1.0, 0.0])
    w = failure_aware_weights(sl, target, mask, reward, entropy_gate=False)
    assert torch.allclose(w[0], torch.ones_like(w[0]))
    assert torch.allclose(w[1], torch.full_like(w[1], 2.0))
    assert (w <= 4.0).all()


def test_failure_aware_entropy_gate_tracks_uncertainty():
    v = 32
    confident = torch.zeros(1, 1, v)
    confident[0, 0, 0] = 20.0
    uncertain = torch.zeros(1, 1, v)
    logits = torch.cat([confident, uncertain], dim=0)
    target = fuse_teachers([make_signal(torch.randn(2, 1, v), v)],
                           torch.tensor([1.0]), vocab_size=v)
    mask = torch.ones(2, 1, dtype=torch.bool)
    w = failure_aware_weights(logits, target, mask, torch.ones(2), entropy_gate=True)
    assert w[1, 0] > w[0, 0]


def _trl_jsd():
    """TRL moved GKD to `trl.experimental` in 1.x; accept either location."""
    import os

    os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
    for mod in ("trl.experimental.gkd.gkd_trainer", "trl.trainer.gkd_trainer"):
        try:
            return __import__(mod, fromlist=["GKDTrainer"]).GKDTrainer.generalized_jsd_loss
        except (ImportError, AttributeError):
            continue
    return None


@pytest.mark.parametrize("beta", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_matches_trl_generalized_jsd(beta):
    """Cross-check every beta against TRL's reference implementation.

    This is what caught the mixture being defined with beta on the wrong side: our
    sparse path and TRL's dense path must agree token for token, endpoints included.
    """
    pytest.importorskip("trl")
    trl_jsd = _trl_jsd()
    if trl_jsd is None:
        pytest.skip("TRL present but GKDTrainer.generalized_jsd_loss not found")

    tl, sl, target, mask, v = setup(v=16, b=2, t=3)
    ours = jsd_loss(sl, target, mask, beta=beta)
    # TRL's "batchmean" divides by batch size; we average over supervised tokens.
    theirs = trl_jsd(student_logits=sl, teacher_logits=tl, beta=beta) / sl.shape[1]
    assert float(ours) == pytest.approx(float(theirs), abs=1e-5)
